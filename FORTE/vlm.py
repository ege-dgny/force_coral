"""VLM-based semantic parser for FORTE.

Queries GPT-4o with scene image + geometry to produce a multi-phase PhysicsConfig.
The VLM outputs a sequence of TaskPhases — each with its own cost weights,
contact strategy, force band, and goal. The FORTE cost structure (Eq. 5) is
unchanged; only its parameters rotate when a phase trigger fires.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from FORTE.types import (
    ContactStrategy,
    ForceBand,
    PhysicsConfig,
    TaskPhase,
    default_physics_config,
)


_SYSTEM_PROMPT_TEMPLATE = """\
You are a robotics physics expert. A Panda robot must perform a contact-rich \
manipulation task. You are given an image of the scene, the task description, \
and the 3D positions/dimensions of all objects.

## Scene geometry (meters, world frame: +x=right, +y=forward, +z=up)
{scene_geometry}

## Instructions
Break the task into ordered phases. Each phase has its own cost function \
parameters, contact strategy, and force constraints. The robot's MPPI planner \
uses these to generate trajectories. Output a single JSON object inside a \
```json block with this exact schema:

```json
{{
  "stiffness": {{
    "x": "HIGH|MEDIUM|LOW",
    "y": "HIGH|MEDIUM|LOW",
    "z": "HIGH|MEDIUM|LOW"
  }},
  "task_frame_euler": [0.0, 0.0, 0.0],
  "recovery_hints": ["re_establish_contact"],
  "phases": [
    {{
      "name": "approach",
      "trigger": "initial",
      "cost_weights": {{
        "task_height": 2.0,
        "task_contact": 8.0,
        "task_pose": 18.0,
        "energy": 0.0,
        "force_upper": 0.0,
        "force_lower": 0.0
      }},
      "contact_strategy": {{
        "approach_face_axis": 1,
        "approach_face_sign": -1.0,
        "contact_standoff": 0.03,
        "contact_vertical_offset_scale": 0.0
      }},
      "force_band": {{"lower": 0.0, "upper": 100.0}},
      "goal": {{"target_height": 0.50}}
    }},
    {{
      "name": "push_to_wall",
      "trigger": "eef_near_box:0.05",
      "cost_weights": {{...}},
      "contact_strategy": {{...}},
      "force_band": {{...}},
      "goal": {{...}}
    }},
    {{
      "name": "lift",
      "trigger": "wall_contact",
      "cost_weights": {{
        "task_height": 18.0,
        "task_contact": 14.0,
        "task_pose": 2.0,
        "energy": 0.2,
        "force_upper": 25.0,
        "force_lower": 12.0
      }},
      "contact_strategy": {{
        "approach_face_axis": 1,
        "approach_face_sign": -1.0,
        "contact_standoff": 0.02,
        "contact_vertical_offset_scale": -0.5
      }},
      "force_band": {{"lower": 15.0, "upper": 35.0}},
      "goal": {{"target_height": 0.50, "gap_target": -0.03}},
      "action_prior": [0.0, 0.6, 0.4, 0.0, 0.0, 0.0]
    }}
  ]
}}
```

## Field definitions

### Global fields
- stiffness: per-axis contact stiffness prior in the task frame.
  x=wall-normal, y=upward sliding, z=lateral.
  HIGH=rigid contact (wall), LOW=free sliding, MEDIUM=moderate resistance.
- task_frame_euler: [rx, ry, rz] degrees. Rotates world frame to task frame.
- recovery_hints: chosen from \
  ["re_establish_contact", "reduce_normal_force_if_stalled", "raise_subgoal"]

### Per-phase fields
- name: human-readable phase name.
- trigger: condition to enter this phase. Options:
  "initial" (first phase, always active),
  "eef_near_box:<distance_m>" (EEF within distance of contact point),
  "wall_contact" (box touching wall),
  "contact_force:<force_N>" (wall contact AND normal force above threshold),
  "height_above:<height_m>" (box top above height).
- cost_weights: MPPI cost function weights (Eq. 5 in FORTE).
  task_height: penalizes (target_height - box_height)^2.
  task_contact: penalizes wall_gap^2.
  task_pose: penalizes ||eef - contact_point||^2.
  energy: lambda_E for interaction energy delta^T Sigma delta.
  force_upper: rho for ReLU(F_pred - F_max)^2 barrier.
  force_lower: gamma for ReLU(F_min - F_pred)^2 contact maintenance.
  Set energy/force_upper/force_lower to 0 for pre-contact phases.
- contact_strategy: where on the object the robot should push.
  approach_face_axis: 0=x, 1=y, 2=z of the box body frame.
  approach_face_sign: -1 or +1 direction.
  contact_standoff: meters from face surface.
  contact_vertical_offset_scale: fraction of box half-height \
    (0.0=center, -1.0=bottom edge, +1.0=top edge). \
    Use negative for lifting tasks (push from below center of mass).
- force_band: [F_min, F_max] in Newtons for this phase.
  Set wide (0, 100) for pre-contact. Tighten for force-regulated phases.
- action_prior: [px, py, pz, rx, ry, rz] bias for trajectory sampling. \
  Values in [-1, 1]. Zero = no bias (uniform sampling). \
  For wall-lift: [0, +0.6, +0.4, 0, 0, 0] biases toward pushing into wall (y+) \
  and upward (z+). This helps the planner find coordinated multi-axis actions.
- goal: task goal for this phase. Keys:
  target_height: desired box top height in meters.
  gap_target: desired wall gap in meters. Negative = push INTO wall. \
    For friction-based lift, use -0.02 to -0.05 (creates sustained normal force). \
    For approach/pre-contact, use 0.0 (just reach the wall).

## Key principles
- Pre-contact phases: set energy/force_upper/force_lower to 0.
- Contact phases: task_contact should be high to maintain wall pressure.
- Lift phases: task_height should dominate. contact_vertical_offset_scale \
  should be negative (push from below COM) to create upward moment via friction.
- The force_lower (contact maintenance) term is critical — it forces the \
  planner to maintain minimum wall contact, preventing the critical failure \
  mode of losing contact entirely.
- FRICTION PHYSICS: For friction-based lift against a wall, \
  friction_force = mu * F_normal. To support a box of mass m against gravity, \
  need F_normal >= m*g / mu. With typical mu=0.3-0.5 and m=0.3-1.0 kg, \
  this means force_band lower should be 10-20N for lift phases (NOT 2-5N). \
  A low force_lower allows the planner to lose wall contact → box falls.
- force_upper should be ~2x force_lower to allow headroom (e.g., lower=15, upper=35).
"""


def _build_scene_geometry_text(scene_info: Dict[str, Any]) -> str:
    lines = []
    for key, val in scene_info.items():
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, list):
            formatted = "[" + ", ".join(f"{v:.4f}" for v in val) + "]"
            lines.append(f"- {key}: {formatted}")
        else:
            lines.append(f"- {key}: {val}")
    return "\n".join(lines)


class TaskPhysicsParser:
    """Parse a task image + prompt into a multi-phase PhysicsConfig via VLM."""

    def __init__(self, client: Any = None, model: str = "gpt-4o") -> None:
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                import openai
                self.client = openai.OpenAI()
            except Exception:
                self.client = None

    def parse_task(
        self,
        image: Any,
        text_prompt: str,
        scene_info: Optional[Dict[str, Any]] = None,
    ) -> PhysicsConfig:
        if self.client is None:
            raise RuntimeError("No OpenAI client. Set OPENAI_API_KEY or pass client=.")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        geo_text = _build_scene_geometry_text(scene_info or {})
        prompt = _SYSTEM_PROMPT_TEMPLATE.format(scene_geometry=geo_text or "Not provided.")

        content = [
            {"type": "input_text", "text": f"Task: {text_prompt}\n\n{prompt}"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return self._parse_response(resp.output_text.strip())

    @staticmethod
    def _parse_response(raw_text: str) -> PhysicsConfig:
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON block in VLM response. Raw: {raw_text[:300]}")
        data = json.loads(match.group(1))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        defaults = default_physics_config()

        # -- Global fields --
        stiffness = data.get("stiffness", {})
        if not isinstance(stiffness, dict):
            stiffness = {}
        stiffness_norm = {
            axis: str(stiffness.get(axis, "MEDIUM")).upper()
            for axis in ["x", "y", "z"]
        }

        euler = data.get("task_frame_euler", defaults.task_frame_euler)
        if not isinstance(euler, list) or len(euler) != 3:
            euler = list(defaults.task_frame_euler)
        euler = [float(v) for v in euler]

        recovery_hints = data.get("recovery_hints", defaults.recovery_hints)
        if not isinstance(recovery_hints, list):
            recovery_hints = [str(recovery_hints)]
        recovery_hints = [str(item) for item in recovery_hints]

        # -- Phases --
        raw_phases = data.get("phases", [])
        if not isinstance(raw_phases, list) or len(raw_phases) == 0:
            phases = list(defaults.phases)
        else:
            phases = [_parse_phase(p, defaults) for p in raw_phases]

        first = phases[0]
        return PhysicsConfig(
            stiffness_prior=stiffness_norm,
            task_frame_euler=euler,
            task_frame=Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
            phases=phases,
            recovery_hints=recovery_hints,
            force_band=ForceBand(lower=first.force_band.lower, upper=first.force_band.upper),
            goal=dict(first.goal),
            cost_weights=dict(first.cost_weights),
            contact_strategy=ContactStrategy(
                approach_face_axis=first.contact_strategy.approach_face_axis,
                approach_face_sign=first.contact_strategy.approach_face_sign,
                contact_standoff=first.contact_strategy.contact_standoff,
                contact_vertical_offset_scale=first.contact_strategy.contact_vertical_offset_scale,
            ),
        )


def _parse_phase(raw: Any, defaults: PhysicsConfig) -> TaskPhase:
    """Parse a single phase dict from VLM output."""
    if not isinstance(raw, dict):
        return defaults.phases[0]

    name = str(raw.get("name", "unknown"))
    trigger = str(raw.get("trigger", "initial"))

    # Cost weights — merge with defaults from first phase
    default_weights = defaults.phases[0].cost_weights
    cw = raw.get("cost_weights", {})
    if not isinstance(cw, dict):
        cw = {}
    cost_weights = {k: float(cw.get(k, v)) for k, v in default_weights.items()}

    # Contact strategy
    cs_raw = raw.get("contact_strategy", {})
    if not isinstance(cs_raw, dict):
        cs_raw = {}
    cs_def = ContactStrategy()
    contact_strategy = ContactStrategy(
        approach_face_axis=int(cs_raw.get("approach_face_axis", cs_def.approach_face_axis)),
        approach_face_sign=float(cs_raw.get("approach_face_sign", cs_def.approach_face_sign)),
        contact_standoff=float(cs_raw.get("contact_standoff", cs_def.contact_standoff)),
        contact_vertical_offset_scale=float(
            cs_raw.get("contact_vertical_offset_scale", cs_def.contact_vertical_offset_scale)
        ),
    )

    # Force band
    fb = raw.get("force_band", {})
    if not isinstance(fb, dict):
        fb = {}
    lower = max(0.0, float(fb.get("lower", 0.0)))
    upper = float(fb.get("upper", 100.0))
    if upper <= lower:
        upper = lower + 1.0

    # Goal
    goal = raw.get("goal", {"target_height": 0.50})
    if not isinstance(goal, dict):
        goal = {"target_height": 0.50}

    # Action prior: 6-DOF bias for MPPI sampling [px, py, pz, rx, ry, rz]
    action_prior_raw = raw.get("action_prior", [0.0] * 6)
    if not isinstance(action_prior_raw, list) or len(action_prior_raw) != 6:
        action_prior_raw = [0.0] * 6
    action_prior = [float(v) for v in action_prior_raw]

    return TaskPhase(
        name=name,
        trigger=trigger,
        cost_weights=cost_weights,
        contact_strategy=contact_strategy,
        force_band=ForceBand(lower=lower, upper=upper),
        goal=goal,
        action_prior=action_prior,
    )

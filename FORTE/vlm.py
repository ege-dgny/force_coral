"""VLM-based semantic parser for FORTE.

Queries GPT-4o with scene image + geometry to produce a multi-phase PhysicsConfig.
The VLM outputs a sequence of TaskPhases — each with its own cost weights,
contact strategy, force band, and goal. The FORTE cost structure (Eq. 5) is
unchanged; only its parameters rotate when a phase trigger fires.

Includes `refine_phases()` for LLM-based plan revision (CoRAL parity):
when the monitor detects failure, the LLM is re-queried with the current
config + metrics + scene image to produce a revised PhysicsConfig.
"""

from __future__ import annotations

import base64
import io
import json
import logging
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

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialization prompt — VLM generates the phase plan from scratch
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a robotics physics expert. A Panda robot must perform a contact-rich \
manipulation task in MuJoCo simulation. You are given an image of the scene, \
the task description, and the 3D positions/dimensions/physics of all objects.

## Scene geometry and physics (meters, Newtons, world frame: +x=right, +y=forward, +z=up)
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
  "task_frame_euler": [rx, ry, rz],
  "recovery_hints": ["re_establish_contact"],
  "phases": [
    {{
      "name": "phase_name",
      "trigger": "trigger_condition",
      "cost_weights": {{
        "task_height": 0.0,
        "task_contact": 0.0,
        "task_pose": 0.0,
        "task_lateral": 0.0,
        "task_tilt": 0.0,
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
      "goal": {{"target_height": 0.50}},
      "action_prior": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }}
  ]
}}
```

## Field definitions

### Global fields
- stiffness: per-axis contact stiffness prior in the task frame. \
  HIGH=rigid contact, LOW=free sliding, MEDIUM=moderate resistance. \
  The Riemannian stiffness estimator will adapt these online from force feedback.
- task_frame_euler: [rx, ry, rz] degrees. Rotates world frame to task frame \
  where x=contact normal, y=task progress direction, z=lateral.
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
- cost_weights: MPPI cost function weights (FORTE Eq. 5).
  task_height: penalizes (target_height - box_height)^2.
  task_contact: penalizes wall_gap^2.
  task_pose: penalizes ||eef - contact_point||^2.
  task_lateral: penalizes (box_x - box_x_init)^2 — lateral drift.
  task_tilt: penalizes box tilt from upright (degrees beyond 10° threshold).
  energy: lambda_E for interaction energy delta^T K delta.
  force_upper: rho for ReLU(F - F_max)^2 barrier (prevent jamming).
  force_lower: gamma for ReLU(F_min - F)^2 barrier (maintain contact).
  Set energy/force_upper/force_lower to 0 for pre-contact phases.
- contact_strategy: where on the object the robot should push.
  approach_face_axis: 0=x, 1=y, 2=z of the box body frame.
  approach_face_sign: -1 or +1 direction.
  contact_standoff: meters from face surface (robot target offset).
  contact_vertical_offset_scale: fraction of box half-height \
    (0.0=center, -1.0=bottom edge, +1.0=top edge). \
    NOTE: the Panda robot's workspace has limited downward reach. \
    Avoid values below -0.3 for objects on a table surface.
- force_band: [F_min, F_max] in Newtons for this phase.
  Set wide (0, 100) for pre-contact. Tighten for force-regulated phases.
- goal: task goal for this phase. Keys:
  target_height: desired box top height in meters.
  gap_target: desired wall gap in meters (0.0=touching, negative=pressed in).
- action_prior: [px, py, pz, rx, ry, rz] bias for trajectory sampling. \
  Values in [-1, 1]. Zero = unbiased. Helps MPPI explore coordinated \
  multi-axis actions (e.g., simultaneous push + lift).

## Physics principles
- Friction force = mu * F_normal. Direction: opposes relative sliding motion.
- For lifting an object UPWARD along a wall: friction from wall-normal force \
  OPPOSES the upward motion. Net lift = F_push_up - mg - mu * F_normal.
- Therefore: minimize wall-normal force while maintaining enough contact \
  for friction to help (not hinder). The optimal push angle is steep \
  (mostly upward, light wall pressure).
- The robot has limited maximum force per axis (see physics parameters above). \
  Allocate force budget wisely between axes.
- The stiffness estimator adapts online — initial stiffness_prior just seeds it.
"""

# ---------------------------------------------------------------------------
# Refinement prompt — LLM revises the plan given failure feedback
# ---------------------------------------------------------------------------

_REFINEMENT_PROMPT_TEMPLATE = """\
You are a robotics physics expert refining a manipulation plan that is not \
succeeding. Analyze the failure mode and output a revised phase plan.

## Scene geometry and physics
{scene_geometry}

## Current phase plan
{current_config}

## Execution state
- Current phase: {current_phase} (phase {phase_index}/{total_phases})
- Steps executed: {steps_executed}

## Monitor feedback
{monitor_feedback}

## Recent metrics
{recent_metrics}

## Stiffness estimator state
{estimator_state}

## Instructions
Analyze why the current plan is failing. Common failure modes:
- Box tilting instead of lifting → push angle too shallow, need steeper upward push
- Stall (no height progress) → insufficient upward force or too much friction
- Contact lost → force_lower too low or approach angle wrong
- Excessive lateral drift → task_lateral weight too low
- Over-force / jamming → force_upper too high or gap_target too negative

You may change ANY parameter: cost_weights, force_band, contact_strategy, \
action_prior, goal, task_frame_euler, stiffness, or add/remove/reorder phases. \
You may also change which phase should be active by setting the first phase's \
trigger to "initial".

Output a COMPLETE revised JSON config (same schema as initial planning). \
Inside a ```json block.
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
    """Parse a task image + prompt into a multi-phase PhysicsConfig via VLM.

    Also provides `refine_phases()` for LLM-based plan revision (CoRAL parity).
    """

    def __init__(self, client: Any = None, model: str = "gpt-4o") -> None:
        self.model = model
        # Store API key at init time (env may change in subprocesses)
        import os
        self._api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
        if client is not None:
            self.client = client
        else:
            try:
                import openai
                self.client = openai.OpenAI()
            except Exception:
                self.client = None
        self._scene_info: Dict[str, Any] = {}

    def parse_task(
        self,
        image: Any,
        text_prompt: str,
        scene_info: Optional[Dict[str, Any]] = None,
    ) -> PhysicsConfig:
        """Initial VLM query — generate phase plan from scene image."""
        if self.client is None:
            raise RuntimeError("No OpenAI client. Set OPENAI_API_KEY or pass client=.")

        self._scene_info = dict(scene_info or {})

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        geo_text = _build_scene_geometry_text(self._scene_info)
        prompt = _SYSTEM_PROMPT_TEMPLATE.format(scene_geometry=geo_text or "Not provided.")

        content = [
            {"type": "input_text", "text": f"Task: {text_prompt}\n\n{prompt}"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]

        LOGGER.info("VLM init query: model=%s", self.model)
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        raw = resp.output_text.strip()
        LOGGER.info("VLM init response: %s", raw[:500])
        return _parse_config_response(raw)

    def _ensure_client(self) -> None:
        """Re-initialize OpenAI client if needed, using stored API key."""
        if self.client is not None:
            return
        try:
            import openai
            if self._api_key:
                self.client = openai.OpenAI(api_key=self._api_key)
            else:
                self.client = openai.OpenAI()
            LOGGER.info("Re-initialized OpenAI client for refinement")
        except Exception as exc:
            raise RuntimeError(f"Cannot create OpenAI client: {exc}") from exc

    def refine_phases(
        self,
        *,
        current_config: PhysicsConfig,
        current_phase_name: str,
        phase_index: int,
        steps_executed: int,
        monitor_feedback: Dict[str, Any],
        recent_metrics: Dict[str, Any],
        estimator_state: Optional[Dict[str, Any]] = None,
        image: Any = None,
    ) -> PhysicsConfig:
        """LLM revision — re-query GPT-4o with failure feedback.

        This is FORTE's equivalent of CoRAL's `refine_plan()`.
        The LLM can change cost weights, force bands, contact strategy,
        task_frame, stiffness prior, phases — everything.
        """
        self._ensure_client()

        geo_text = _build_scene_geometry_text(self._scene_info)
        config_json = json.dumps(current_config.to_dict(), indent=2, default=str)
        feedback_text = json.dumps(monitor_feedback, indent=2, default=str)
        metrics_text = json.dumps(recent_metrics, indent=2, default=str)
        estimator_text = json.dumps(estimator_state or {}, indent=2, default=str)

        prompt = _REFINEMENT_PROMPT_TEMPLATE.format(
            scene_geometry=geo_text,
            current_config=config_json,
            current_phase=current_phase_name,
            phase_index=phase_index + 1,
            total_phases=len(current_config.phases),
            steps_executed=steps_executed,
            monitor_feedback=feedback_text,
            recent_metrics=metrics_text,
            estimator_state=estimator_text,
        )

        content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": prompt},
        ]
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
            )

        LOGGER.info("VLM refinement query at step %d (phase: %s)", steps_executed, current_phase_name)
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        raw = resp.output_text.strip()
        LOGGER.info("VLM refinement response: %s", raw[:500])
        return _parse_config_response(raw)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_config_response(raw_text: str) -> PhysicsConfig:
    """Parse a VLM/LLM response into PhysicsConfig."""
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
            gripper_command=first.contact_strategy.gripper_command,
        ),
    )


def _parse_phase(raw: Any, defaults: PhysicsConfig) -> TaskPhase:
    """Parse a single phase dict from VLM output."""
    if not isinstance(raw, dict):
        return defaults.phases[0]

    name = str(raw.get("name", "unknown"))
    trigger = str(raw.get("trigger", "initial"))

    # Cost weights — keep canonical keys and accept explicit extras from VLM.
    default_weights = defaults.phases[0].cost_weights
    canonical_defaults = {
        "task_height": float(default_weights.get("task_height", 0.0)),
        "task_contact": float(default_weights.get("task_contact", 0.0)),
        "task_pose": float(default_weights.get("task_pose", 0.0)),
        "task_lateral": float(default_weights.get("task_lateral", 0.0)),
        "task_tilt": float(default_weights.get("task_tilt", 0.0)),
        "energy": float(default_weights.get("energy", 0.0)),
        "force_upper": float(default_weights.get("force_upper", 0.0)),
        "force_lower": float(default_weights.get("force_lower", 0.0)),
    }
    cw = raw.get("cost_weights", {})
    if not isinstance(cw, dict):
        cw = {}
    cost_weights = {k: float(cw.get(k, v)) for k, v in canonical_defaults.items()}
    for key, val in cw.items():
        if key not in cost_weights:
            cost_weights[str(key)] = float(val)

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
        gripper_command=float(cs_raw.get("gripper_command", cs_def.gripper_command)),
        metadata={
            str(k): v
            for k, v in cs_raw.items()
            if k not in {
                "approach_face_axis",
                "approach_face_sign",
                "contact_standoff",
                "contact_vertical_offset_scale",
                "gripper_command",
            }
        },
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

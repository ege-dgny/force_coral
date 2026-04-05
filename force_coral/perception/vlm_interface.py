"""VLM-based semantic parser for the Phase-1 FORTE pipeline."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class ForceBand:
    lower: float
    upper: float

    def to_dict(self) -> Dict[str, float]:
        return {"lower": float(self.lower), "upper": float(self.upper)}


@dataclasses.dataclass
class ContactStrategy:
    """Which face to approach, standoff distance, vertical offset."""

    approach_face_axis: int = 1        # 0=x, 1=y, 2=z
    approach_face_sign: float = -1.0   # direction along axis
    contact_standoff: float = 0.03     # meters from face surface
    contact_vertical_offset_scale: float = 0.0  # fraction of half-extent

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PhysicsConfig:
    """Structured semantic package emitted by the VLM."""

    stiffness_prior: Dict[str, str]
    force_band: ForceBand
    task_frame_euler: List[float]
    task_frame: np.ndarray
    goal: Dict[str, Any]
    cost_weights: Dict[str, float]
    recovery_hints: List[str]
    contact_strategy: ContactStrategy = dataclasses.field(
        default_factory=ContactStrategy,
    )

    def __post_init__(self):
        if self.task_frame is None and self.task_frame_euler is not None:
            self.task_frame = Rotation.from_euler(
                "xyz", self.task_frame_euler, degrees=True
            ).as_matrix()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stiffness": self.stiffness_prior,
            "force_band": self.force_band.to_dict(),
            "task_frame_euler": self.task_frame_euler,
            "goal": self.goal,
            "cost_weights": self.cost_weights,
            "recovery_hints": self.recovery_hints,
            "contact_strategy": self.contact_strategy.to_dict(),
        }


def default_physics_config() -> PhysicsConfig:
    return PhysicsConfig(
        stiffness_prior={"x": "HIGH", "y": "LOW", "z": "MEDIUM"},
        force_band=ForceBand(lower=2.5, upper=12.0),
        task_frame_euler=[0.0, 0.0, 0.0],
        task_frame=np.eye(3),
        goal={"target_height": 0.50},
        cost_weights={
            "task_height": 14.0,
            "task_contact": 18.0,
            "task_pose": 4.0,
            "energy": 0.2,
            "force_upper": 25.0,
            "force_lower": 12.0,
        },
        recovery_hints=["re_establish_contact", "reduce_normal_force_if_stalled"],
    )


_SYSTEM_PROMPT_TEMPLATE = """\
You are a robotics physics expert. A Panda robot must perform a contact-rich \
manipulation task. You are given an image of the scene, the task description, \
and the 3D positions/dimensions of all objects.

## Scene geometry (meters, world frame: +x=right, +y=forward, +z=up)
{scene_geometry}

## Instructions
Determine the full physical interaction package. Output a single JSON object \
inside a ```json block with this exact schema:

```json
{{
  "contact_strategy": {{
    "approach_face_axis": 0|1|2,
    "approach_face_sign": -1.0|1.0,
    "contact_standoff": 0.03,
    "contact_vertical_offset_scale": 0.0
  }},
  "stiffness": {{
    "x": "HIGH|MEDIUM|LOW",
    "y": "HIGH|MEDIUM|LOW",
    "z": "HIGH|MEDIUM|LOW"
  }},
  "force_band": {{"lower": 2.5, "upper": 12.0}},
  "task_frame_euler": [0.0, 0.0, 0.0],
  "goal": {{"target_height": 0.50}},
  "cost_weights": {{
    "task_height": 14.0,
    "task_contact": 18.0,
    "task_pose": 4.0,
    "energy": 0.2,
    "force_upper": 25.0,
    "force_lower": 12.0
  }},
  "recovery_hints": ["re_establish_contact"]
}}
```

## Field definitions
- contact_strategy: which box face the robot should approach.
  - approach_face_axis: 0=x, 1=y, 2=z of the box body frame.
  - approach_face_sign: -1 or +1 — the direction the robot comes from.
  - contact_standoff: how far from the face surface (meters) the EEF target sits.
  - contact_vertical_offset_scale: vertical offset as fraction of box half-height \
    (0.0=center, -1.0=bottom edge, +1.0=top edge). Use negative for lifting tasks \
    (push from below center of mass).
- stiffness: per-axis contact stiffness prior in the task frame.
  x=wall-normal, y=upward sliding, z=lateral.
  HIGH=rigid contact (wall), LOW=free sliding, MEDIUM=moderate resistance.
- force_band.lower: minimum normal force (N) to maintain wall contact.
- force_band.upper: maximum normal force (N) before jamming.
- goal.target_height: desired box-top height in meters.
- cost_weights: relative weights for the MPPI cost function.
  task_contact should dominate task_pose to ensure the robot makes contact.
- recovery_hints: chosen from \
  ["re_establish_contact", "reduce_normal_force_if_stalled", "raise_subgoal"]
"""


def _build_scene_geometry_text(scene_info: Dict[str, Any]) -> str:
    """Format scene geometry dict into readable text for the VLM prompt."""
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
    """Parse a task image and prompt into a bounded semantic package."""

    def __init__(self, client=None, model: str = "gpt-4o") -> None:
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
        image,
        text_prompt: str,
        scene_info: Optional[Dict[str, Any]] = None,
    ) -> PhysicsConfig:
        if self.client is None:
            raise RuntimeError(
                "No OpenAI client available. Set OPENAI_API_KEY or pass client=."
            )

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
            raise ValueError(
                f"No JSON block found in VLM response. Raw: {raw_text[:300]}"
            )
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in VLM response: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        defaults = default_physics_config()

        stiffness = data.get("stiffness", {})
        if not isinstance(stiffness, dict):
            stiffness = {}
        stiffness_norm = {}
        for axis in ["x", "y", "z"]:
            stiffness_norm[axis] = str(stiffness.get(axis, "MEDIUM")).upper()

        force_band_raw = data.get("force_band", {})
        if not isinstance(force_band_raw, dict):
            force_band_raw = {}
        lower = max(0.0, float(force_band_raw.get("lower", defaults.force_band.lower)))
        upper = float(force_band_raw.get("upper", defaults.force_band.upper))
        if upper <= lower:
            upper = lower + 1.0

        euler = data.get("task_frame_euler", defaults.task_frame_euler)
        if not isinstance(euler, list) or len(euler) != 3:
            euler = list(defaults.task_frame_euler)
        euler = [float(v) for v in euler]

        goal = data.get("goal", defaults.goal)
        if not isinstance(goal, dict):
            goal = dict(defaults.goal)
        goal = {"target_height": float(goal.get("target_height", defaults.goal["target_height"]))}

        cost_weights = data.get("cost_weights", {})
        if not isinstance(cost_weights, dict):
            cost_weights = {}
        merged_weights = {
            key: float(cost_weights.get(key, value))
            for key, value in defaults.cost_weights.items()
        }

        recovery_hints = data.get("recovery_hints", defaults.recovery_hints)
        if not isinstance(recovery_hints, list):
            recovery_hints = [str(recovery_hints)]
        recovery_hints = [str(item) for item in recovery_hints]

        cs_raw = data.get("contact_strategy", {})
        if not isinstance(cs_raw, dict):
            cs_raw = {}
        cs_defaults = ContactStrategy()
        contact_strategy = ContactStrategy(
            approach_face_axis=int(cs_raw.get("approach_face_axis", cs_defaults.approach_face_axis)),
            approach_face_sign=float(cs_raw.get("approach_face_sign", cs_defaults.approach_face_sign)),
            contact_standoff=float(cs_raw.get("contact_standoff", cs_defaults.contact_standoff)),
            contact_vertical_offset_scale=float(
                cs_raw.get("contact_vertical_offset_scale", cs_defaults.contact_vertical_offset_scale)
            ),
        )

        return PhysicsConfig(
            stiffness_prior=stiffness_norm,
            force_band=ForceBand(lower=lower, upper=upper),
            task_frame_euler=euler,
            task_frame=Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
            goal=goal,
            cost_weights=merged_weights,
            recovery_hints=recovery_hints,
            contact_strategy=contact_strategy,
        )

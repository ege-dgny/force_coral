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
class PhysicsConfig:
    """Structured semantic package emitted by the VLM."""

    stiffness_prior: Dict[str, str]
    force_band: ForceBand
    task_frame_euler: List[float]
    task_frame: np.ndarray
    goal: Dict[str, Any]
    cost_weights: Dict[str, float]
    recovery_hints: List[str]

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
            "task_contact": 8.0,
            "task_pose": 18.0,
            "energy": 0.2,
            "force_upper": 25.0,
            "force_lower": 12.0,
        },
        recovery_hints=["re_establish_contact", "reduce_normal_force_if_stalled"],
    )


_SYSTEM_PROMPT = """\
You are a robotics physics expert. Given an image of a manipulation scene and \
a task description, determine the physical interaction package for a wall-assisted \
contact-rich manipulation controller.

Output a single JSON object inside a ```json block with this exact schema:

```json
{
  "stiffness": {
    "x": "HIGH|MEDIUM|LOW",
    "y": "HIGH|MEDIUM|LOW",
    "z": "HIGH|MEDIUM|LOW"
  },
  "force_band": {"lower": 2.0, "upper": 12.0},
  "task_frame_euler": [0.0, 0.0, 0.0],
  "goal": {"target_height": 0.50},
  "cost_weights": {
    "task_height": 14.0,
    "task_contact": 8.0,
    "task_pose": 18.0,
    "energy": 0.2,
    "force_upper": 25.0,
    "force_lower": 12.0
  },
  "recovery_hints": ["re_establish_contact"]
}
```

Definitions:
- x = wall-normal, y = upward sliding direction, z = remaining tangential axis.
- force_band.lower keeps the object pinned against the wall.
- force_band.upper prevents over-force and wall-friction jamming.
- recovery_hints must be short phrases chosen from:
  ["re_establish_contact", "reduce_normal_force_if_stalled", "raise_subgoal"]
"""


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

    def parse_task(self, image, text_prompt: str) -> PhysicsConfig:
        if self.client is None:
            raise RuntimeError(
                "No OpenAI client available. Set OPENAI_API_KEY or pass client=."
            )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        content = [
            {"type": "input_text", "text": f"Task: {text_prompt}\n\n{_SYSTEM_PROMPT}"},
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

        return PhysicsConfig(
            stiffness_prior=stiffness_norm,
            force_band=ForceBand(lower=lower, upper=upper),
            task_frame_euler=euler,
            task_frame=Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
            goal=goal,
            cost_weights=merged_weights,
            recovery_hints=recovery_hints,
        )

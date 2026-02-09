"""
VLM-based physics parser for FORTE.

Queries a Vision-Language Model (GPT-4o) with a task image and description,
and returns a structured PhysicsConfig containing:
    - Stiffness prior (HIGH/MEDIUM/LOW per axis)
    - Safety constraints (force limits)
    - Task frame (rotation matrix aligning world → task)

This replaces heuristic Python code generation with structured JSON output,
ensuring the VLM output is always a valid physical prior.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PhysicsConfig:
    """Structured output from VLM physics parsing.

    Attributes
    ----------
    stiffness_prior : dict
        Per-axis stiffness level.
        Keys: "x", "y", "z", "rx", "ry", "rz".
        Values: "HIGH", "MEDIUM", or "LOW".
    constraints : list of str
        Safety constraint strings, e.g. ["force_z < 10.0", "torque_x < 2.0"].
    task_frame_euler : list of float
        Euler angles [rx, ry, rz] in degrees (XYZ convention) defining
        the rotation from world frame to task-aligned frame.
    task_frame : np.ndarray, shape (3, 3)
        Rotation matrix derived from task_frame_euler.
    """

    stiffness_prior: Dict[str, str]
    constraints: List[str]
    task_frame_euler: List[float]
    task_frame: np.ndarray

    def __post_init__(self):
        if self.task_frame is None and self.task_frame_euler is not None:
            self.task_frame = Rotation.from_euler(
                "xyz", self.task_frame_euler, degrees=True
            ).as_matrix()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stiffness": self.stiffness_prior,
            "constraints": self.constraints,
            "task_frame_euler": self.task_frame_euler,
        }


def default_physics_config() -> PhysicsConfig:
    """Fallback config: medium stiffness on all axes, identity task frame."""
    return PhysicsConfig(
        stiffness_prior={
            "x": "MEDIUM", "y": "MEDIUM", "z": "MEDIUM",
            "rx": "MEDIUM", "ry": "MEDIUM", "rz": "MEDIUM",
        },
        constraints=["force_z < 15.0"],
        task_frame_euler=[0.0, 0.0, 0.0],
        task_frame=np.eye(3),
    )


# ---------------------------------------------------------------------------
# VLM prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robotics physics expert. Given an image of a manipulation scene and \
a task description, determine the **Interaction Topology** for contact-rich \
manipulation.

Output a single JSON object inside a ```json block with this exact schema:

```json
{
  "stiffness": {
    "x": "HIGH|MEDIUM|LOW",
    "y": "HIGH|MEDIUM|LOW",
    "z": "HIGH|MEDIUM|LOW",
    "rx": "HIGH|MEDIUM|LOW",
    "ry": "HIGH|MEDIUM|LOW",
    "rz": "HIGH|MEDIUM|LOW"
  },
  "constraints": [
    "force_z < 10.0"
  ],
  "task_frame_euler": [0.0, 0.0, 0.0]
}
```

Definitions:
- **HIGH stiffness**: Rigid contact direction (e.g. pushing against a wall, \
pressing down on a surface). The robot should NOT push hard in this direction.
- **MEDIUM stiffness**: Moderate resistance expected (e.g. sliding with \
friction, light contact).
- **LOW stiffness**: Free motion direction (e.g. moving through air, sliding \
along a frictionless surface).
- **constraints**: Force/torque safety limits in Newtons or Nm. Format: \
"force_<axis> < <limit>" or "torque_<axis> < <limit>".
- **task_frame_euler**: Euler XYZ rotation (degrees) aligning the world frame \
to the task-relevant frame. Use [0,0,0] if the world frame is already aligned.

Reasoning guidelines:
- If the task involves pushing an object toward a wall, the axis perpendicular \
to the wall surface should be HIGH stiffness.
- If the task involves sliding along a surface, the tangential axes should be \
LOW stiffness and the normal axis HIGH.
- If the task involves flipping or rotating, rotational stiffness axes should \
be set according to which rotations are constrained vs free.
- Always set at least one force constraint to prevent damage.
"""


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------

class TaskPhysicsParser:
    """Parse task images into physical priors using a VLM.

    Parameters
    ----------
    client : openai.OpenAI or None
        OpenAI client instance.  If None, attempts to create one from env.
    model : str
        Model identifier (default: "gpt-4o").
    """

    def __init__(self, client=None, model: str = "gpt-4o") -> None:
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                import openai
                self.client = openai.OpenAI()  # reads OPENAI_API_KEY from env
            except Exception:
                self.client = None

    def parse_task(
        self,
        image,  # PIL.Image.Image
        text_prompt: str,
    ) -> PhysicsConfig:
        """Query VLM and return a PhysicsConfig.

        Parameters
        ----------
        image : PIL.Image.Image
            Scene image.
        text_prompt : str
            Natural-language task description.

        Returns
        -------
        PhysicsConfig

        Raises
        ------
        RuntimeError
            If VLM client is unavailable.
        ValueError
            If VLM response cannot be parsed.
        """
        if self.client is None:
            raise RuntimeError(
                "No OpenAI client available. Set OPENAI_API_KEY or pass client=."
            )

        # Encode image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        content = [
            {
                "type": "input_text",
                "text": f"Task: {text_prompt}\n\n{_SYSTEM_PROMPT}",
            },
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            },
        ]

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=0.0,
        )

        raw = resp.output_text.strip()
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw_text: str) -> PhysicsConfig:
        """Parse VLM JSON response into PhysicsConfig.

        Parameters
        ----------
        raw_text : str
            Raw VLM output containing a ```json ... ``` block.

        Returns
        -------
        PhysicsConfig
        """
        # Extract JSON block
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON block found in VLM response. Raw: {raw_text[:300]}"
            )
        json_str = match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in VLM response: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        # Extract fields with defaults
        stiffness = data.get("stiffness", {})
        # Normalize keys to lowercase, values to uppercase
        stiffness_norm = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            val = stiffness.get(axis, "MEDIUM")
            stiffness_norm[axis] = str(val).upper()

        constraints = data.get("constraints", [])
        if not isinstance(constraints, list):
            constraints = [str(constraints)]

        euler = data.get("task_frame_euler", [0.0, 0.0, 0.0])
        if not isinstance(euler, list) or len(euler) != 3:
            euler = [0.0, 0.0, 0.0]
        euler = [float(e) for e in euler]

        task_frame = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()

        return PhysicsConfig(
            stiffness_prior=stiffness_norm,
            constraints=constraints,
            task_frame_euler=euler,
            task_frame=task_frame,
        )

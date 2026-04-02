"""Pure geometry helpers for object-centric wall-contact planning."""

from __future__ import annotations

from typing import Dict

import numpy as np


def build_wall_lift_task_frame() -> np.ndarray:
    """Task frame for wall-assisted lifting.

    Rows are task axes expressed in world coordinates:
    - x_task: wall normal (+y in world)
    - y_task: upward lift (+z in world)
    - z_task: lateral (+x in world)
    """
    return np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def world_to_task(task_frame: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    return np.asarray(task_frame, dtype=np.float64) @ np.asarray(vector_world, dtype=np.float64).ravel()


def compute_box_face_anchor(
    box_pos: np.ndarray,
    box_rotmat: np.ndarray,
    half_extents: np.ndarray,
    *,
    face_axis: int = 1,
    face_sign: float = -1.0,
    standoff: float = 0.0,
    vertical_offset_scale: float = 0.0,
) -> np.ndarray:
    """Return a world-space point anchored to a face of the oriented box."""
    local = np.zeros(3, dtype=np.float64)
    local[face_axis] = face_sign * (half_extents[face_axis] + standoff)
    local[2] = vertical_offset_scale * half_extents[2]
    return np.asarray(box_pos, dtype=np.float64) + np.asarray(box_rotmat, dtype=np.float64) @ local


def compute_wall_gap(
    box_pos: np.ndarray,
    box_half_extents: np.ndarray,
    wall_pos: np.ndarray,
    wall_half_extents: np.ndarray,
) -> float:
    wall_front_y = float(wall_pos[1] - wall_half_extents[1])
    box_front_y = float(box_pos[1] + box_half_extents[1])
    return float(wall_front_y - box_front_y)


def compute_wall_lift_task_cost(
    *,
    box_top_height: float,
    target_height: float,
    wall_gap: float,
    eef_to_contact_distance: float,
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Return the object-aware geometric task cost terms."""
    height_cost = max(0.0, float(target_height) - float(box_top_height)) ** 2
    contact_cost = max(0.0, float(wall_gap)) ** 2
    pose_cost = float(eef_to_contact_distance) ** 2
    task = (
        float(weights.get("task_height", 14.0)) * height_cost
        + float(weights.get("task_contact", 8.0)) * contact_cost
        + float(weights.get("task_pose", 18.0)) * pose_cost
    )
    return {
        "height": height_cost,
        "contact": contact_cost,
        "pose": pose_cost,
        "total": task,
    }

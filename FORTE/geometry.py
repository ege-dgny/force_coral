"""Pure geometry helpers for wall-contact planning."""

from __future__ import annotations

from typing import Dict

import numpy as np


def build_wall_lift_task_frame() -> np.ndarray:
    """Task frame for wall-assisted lifting.

    Rows = task axes in world coords:
      x_task = wall normal (+y world)
      y_task = upward lift (+z world)
      z_task = lateral (+x world)
    """
    return np.asarray(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]],
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
    """World-space point anchored to a face of the oriented box."""
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


def compute_approach_face_sign(
    box_rotmat: np.ndarray,
    wall_pos: np.ndarray,
    box_pos: np.ndarray,
    face_axis: int = 1,
) -> float:
    """Return +1 or -1 so that the chosen face points AWAY from the wall.

    We want the robot to approach the face that faces the robot (opposite the wall).
    The wall is in the +Y direction relative to the box. We pick the face_sign
    such that the face normal in world coords points away from the wall.
    """
    # Wall direction in world frame (wall relative to box)
    wall_dir = np.asarray(wall_pos[:3], dtype=np.float64) - np.asarray(box_pos[:3], dtype=np.float64)
    # Face normal in world for face_sign=+1
    axis_vec = np.zeros(3, dtype=np.float64)
    axis_vec[face_axis] = 1.0
    face_normal_world = np.asarray(box_rotmat, dtype=np.float64) @ axis_vec
    # If +1 face points toward wall (dot > 0), we want -1 (opposite face)
    # If +1 face points away from wall (dot < 0), we want +1
    dot = float(np.dot(face_normal_world, wall_dir))
    return -1.0 if dot > 0 else 1.0


def compute_wall_lift_task_cost(
    *,
    box_top_height: float,
    target_height: float,
    wall_gap: float,
    eef_to_contact_distance: float,
    weights: Dict[str, float],
    gap_target: float = 0.0,
    box_tilt_deg: float = 0.0,
) -> Dict[str, float]:
    """Geometric task cost terms for wall-lift.

    gap_target < 0 means the box should be pushed PAST the wall plane
    by |gap_target| meters. This creates sustained wall-normal force
    via contact mechanics.

    box_tilt_deg: deviation from upright in degrees (0 = perfect).
    Penalized to prevent tipping during friction-based lift.
    """
    height_cost = max(0.0, float(target_height) - float(box_top_height)) ** 2
    contact_cost = max(0.0, float(wall_gap) - float(gap_target)) ** 2
    pose_cost = float(eef_to_contact_distance) ** 2
    # Tilt penalty: penalize deviations beyond a threshold
    tilt_threshold = 15.0  # degrees — allow minor tilt
    tilt_cost = max(0.0, float(box_tilt_deg) - tilt_threshold) ** 2 / 1000.0
    total = (
        float(weights.get("task_height", 14.0)) * height_cost
        + float(weights.get("task_contact", 18.0)) * contact_cost
        + float(weights.get("task_pose", 4.0)) * pose_cost
        + float(weights.get("task_tilt", 8.0)) * tilt_cost
    )
    return {
        "height": height_cost, "contact": contact_cost,
        "pose": pose_cost, "tilt": tilt_cost, "total": total,
    }

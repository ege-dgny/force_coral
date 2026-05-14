"""Pure geometry helpers for wall-contact planning."""

from __future__ import annotations

from typing import Dict, Optional

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
    box_rotmat: Optional[np.ndarray] = None,
    wall_rotmat: Optional[np.ndarray] = None,
) -> float:
    """Surface-to-surface gap between oriented box and wall geoms.

    Computes distance along the instantaneous box->wall direction using
    OBB support radii. This is robust to geom offsets and rotation.
    """
    box_center = np.asarray(box_pos, dtype=np.float64)
    wall_center = np.asarray(wall_pos, dtype=np.float64)
    u = wall_center - box_center
    norm = float(np.linalg.norm(u))
    if norm < 1e-8:
        # Fallback to +y when centers are degenerate.
        u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        u = u / norm

    box_half = np.asarray(box_half_extents, dtype=np.float64)
    wall_half = np.asarray(wall_half_extents, dtype=np.float64)
    box_R = np.asarray(box_rotmat, dtype=np.float64) if box_rotmat is not None else np.eye(3)
    wall_R = np.asarray(wall_rotmat, dtype=np.float64) if wall_rotmat is not None else np.eye(3)

    # OBB support radius along direction u.
    box_radius = float(np.abs(box_R.T @ u) @ box_half)
    wall_radius = float(np.abs(wall_R.T @ u) @ wall_half)

    # Closest points on each object along u.
    box_surface_toward_wall = box_center + u * box_radius
    wall_surface_toward_box = wall_center - u * wall_radius
    gap = float(np.dot(wall_surface_toward_box - box_surface_toward_wall, u))
    return gap


def compute_approach_face_sign(
    box_rotmat: np.ndarray,
    wall_pos: np.ndarray,
    box_pos: np.ndarray,
    face_axis: int = 1,
) -> float:
    """Return +1 or -1 so that the chosen face points AWAY from the wall."""
    wall_dir = np.asarray(wall_pos[:3], dtype=np.float64) - np.asarray(box_pos[:3], dtype=np.float64)
    axis_vec = np.zeros(3, dtype=np.float64)
    axis_vec[face_axis] = 1.0
    face_normal_world = np.asarray(box_rotmat, dtype=np.float64) @ axis_vec
    dot = float(np.dot(face_normal_world, wall_dir))
    return -1.0 if dot > 0 else 1.0


def compute_best_approach_face(
    box_rotmat: np.ndarray,
    wall_pos: np.ndarray,
    box_pos: np.ndarray,
) -> tuple:
    """Find which box face best faces AWAY from the wall.

    Returns (face_axis, face_sign) for the face whose normal in world
    coordinates has the largest negative dot product with the wall direction.
    This dynamically adapts to box rotation — if the box rotates, the
    approach face updates to whichever face now faces away from the wall.
    """
    rotmat = np.asarray(box_rotmat, dtype=np.float64)
    wall_dir = np.asarray(wall_pos[:3], dtype=np.float64) - np.asarray(box_pos[:3], dtype=np.float64)
    wall_dir_norm = wall_dir / max(np.linalg.norm(wall_dir), 1e-8)

    best_axis = 1
    best_sign = -1.0
    best_dot = float("inf")  # most negative = best (points away from wall)

    for axis in range(3):
        axis_vec = np.zeros(3, dtype=np.float64)
        axis_vec[axis] = 1.0
        face_normal = rotmat @ axis_vec

        for sign in [1.0, -1.0]:
            dot = float(np.dot(sign * face_normal, wall_dir_norm))
            if dot < best_dot:
                best_dot = dot
                best_axis = axis
                best_sign = sign

    return best_axis, best_sign


def compute_wall_lift_task_cost(
    *,
    box_top_height: float,
    target_height: float,
    wall_gap: float,
    eef_to_contact_distance: float,
    weights: Dict[str, float],
    gap_target: float = 0.0,
    box_tilt_deg: float = 0.0,
    lateral_offset: float = 0.0,
) -> Dict[str, float]:
    """Geometric task cost terms for wall-lift.

    gap_target < 0 means the box should be pushed PAST the wall plane
    by |gap_target| meters. This creates sustained wall-normal force
    via contact mechanics.

    box_tilt_deg: deviation from upright in degrees (0 = perfect).
    Penalized to prevent tipping during friction-based lift.

    lateral_offset: box x-position minus initial x-position (meters).
    Penalizes lateral drift that degrades push angle to wall.
    """
    height_cost = max(0.0, float(target_height) - float(box_top_height)) ** 2
    contact_cost = max(0.0, float(wall_gap) - float(gap_target)) ** 2
    pose_cost = float(eef_to_contact_distance) ** 2
    # Tilt penalty: penalize deviations beyond threshold
    tilt_threshold = 10.0  # degrees
    tilt_cost = max(0.0, float(box_tilt_deg) - tilt_threshold) ** 2 / 100.0
    # Lateral stability: penalize drift from initial x-position
    lateral_cost = float(lateral_offset) ** 2
    total = (
        float(weights.get("task_height", 14.0)) * height_cost
        + float(weights.get("task_contact", 18.0)) * contact_cost
        + float(weights.get("task_pose", 4.0)) * pose_cost
        + float(weights.get("task_tilt", 8.0)) * tilt_cost
        + float(weights.get("task_lateral", 0.0)) * lateral_cost
    )
    return {
        "height": height_cost, "contact": contact_cost,
        "pose": pose_cost, "tilt": tilt_cost, "lateral": lateral_cost,
        "total": total,
    }

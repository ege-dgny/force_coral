"""ForteWrapper: force-augmented rollout cost for MPPI workers.

Cost: J = J_task + J_energy
J_task uses geometric costs with a gap_target that pushes the box INTO
the wall by a target amount, creating sustained wall-normal force via
contact mechanics. Force barriers are handled in the main loop for
diagnostics; workers use smooth geometric costs only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from FORTE.env import ObjectCentricWrapper
from FORTE.geometry import (
    build_wall_lift_task_frame,
    compute_approach_face_sign,
    compute_wall_lift_task_cost,
    world_to_task,
)
from FORTE.types import PhysicsConfig, default_physics_config


class ForteWrapper(ObjectCentricWrapper):
    """Force-augmented wrapper. Workers call rollout_cost()."""

    def __init__(self, env: Any, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self.semantic_config: PhysicsConfig = default_physics_config()
        self.task_frame = build_wall_lift_task_frame()
        self.stiffness: Optional[np.ndarray] = None
        self.approach_face_sign = compute_approach_face_sign(
            self.get_box_rotmat(), self.get_wall_pos(), self.get_box_pos(),
            face_axis=self.approach_face_axis,
        )

    def configure_runtime(self, runtime_data: Optional[Dict[str, Any]]) -> None:
        if runtime_data is None:
            return
        if "semantic_config" in runtime_data:
            self.semantic_config = runtime_data["semantic_config"]
            cs = self.semantic_config.contact_strategy
            self.approach_face_axis = int(cs.approach_face_axis)
            self.contact_standoff = float(cs.contact_standoff)
            self.contact_vertical_offset_scale = float(cs.contact_vertical_offset_scale)
            self.approach_face_sign = compute_approach_face_sign(
                self.get_box_rotmat(), self.get_wall_pos(), self.get_box_pos(),
                face_axis=self.approach_face_axis,
            )
        if "stiffness" in runtime_data:
            self.stiffness = runtime_data["stiffness"]
        if not np.allclose(self.semantic_config.task_frame, np.eye(3)):
            self.task_frame = np.asarray(self.semantic_config.task_frame, dtype=np.float64)

    # -- Force helpers (for main loop) --

    def get_force_task(self) -> np.ndarray:
        """Measured force in task frame."""
        return world_to_task(self.task_frame, self.get_wrench_world()[:3])

    def get_delta_task(self) -> np.ndarray:
        """Penetration vector (EEF - contact anchor) in task frame."""
        return world_to_task(
            self.task_frame,
            self.get_eef_pos() - self.get_contact_anchor_world(),
        )

    # -- Cost function --

    def rollout_cost(self) -> float:
        """J_task + J_energy. No force barriers — smooth geometric cost only.

        Gap target drives box INTO wall, creating sustained contact force
        through contact mechanics rather than explicit force barriers.
        """
        weights = self.semantic_config.cost_weights
        goal = self.semantic_config.goal

        # gap_target: negative = push INTO wall. Only for contact/lift phases.
        # Default to 0 (reach wall surface, don't push past) unless goal specifies.
        gap_target = float(goal.get("gap_target", 0.0))

        # Compute box tilt from upright
        from scipy.spatial.transform import Rotation as R_conv
        box_quat_wxyz = self.get_box_quat()
        rot = R_conv.from_quat(
            [box_quat_wxyz[1], box_quat_wxyz[2], box_quat_wxyz[3], box_quat_wxyz[0]]
        )
        # Tilt = angle between box z-axis and world z-axis
        box_z = rot.as_matrix()[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(box_z[2]), 0, 1))))

        task_terms = compute_wall_lift_task_cost(
            box_top_height=self.get_box_top_height(),
            target_height=float(goal.get("target_height", 0.50)),
            wall_gap=self.wall_gap(),
            eef_to_contact_distance=float(
                np.linalg.norm(self.get_eef_pos() - self.get_desired_contact_world())
            ),
            weights=weights,
            gap_target=gap_target,
            box_tilt_deg=tilt_deg,
        )
        task_cost = float(task_terms["total"])

        # Energy term (deformation penalty) — only when contact established
        if self.stiffness is not None:
            delta_task = self.get_delta_task()
            delta_norm = float(np.linalg.norm(delta_task))
            if delta_norm > 0.1:
                delta_task = delta_task * (0.1 / delta_norm)
            energy = float(weights.get("energy", 0.2)) * float(
                delta_task @ self.stiffness @ delta_task
            )
            task_cost += energy

        return task_cost

"""ForteWrapper: force-augmented rollout cost for MPPI workers.

Cost: J = J_task + J_energy + J_force_lower + J_force_upper
J_task: geometric costs (height, contact/gap, pose, lateral, tilt)
J_energy: stiffness-aware deformation penalty (contact-gated)
J_force_lower: sim-force barrier maintaining minimum wall-normal force
J_force_upper: sim-force barrier preventing jamming

Force barriers use MuJoCo sim forces (ground truth in rollout),
NOT K@delta predictions. Contact-gated: only active when
stiffness is set (= main loop detected real contact).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from FORTE.env import ObjectCentricWrapper
from FORTE.geometry import (
    build_wall_lift_task_frame,
    compute_approach_face_sign,
    compute_best_approach_face,
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
        self.box_x_init: float = float(self.get_box_pos()[0])
        self._update_approach_face()

    def configure_runtime(self, runtime_data: Optional[Dict[str, Any]]) -> None:
        if runtime_data is None:
            return
        if "semantic_config" in runtime_data:
            self.semantic_config = runtime_data["semantic_config"]
            cs = self.semantic_config.contact_strategy
            self.contact_standoff = float(cs.contact_standoff)
            self.contact_vertical_offset_scale = float(cs.contact_vertical_offset_scale)
            self._gripper_command = float(cs.gripper_command)
        if "stiffness" in runtime_data:
            self.stiffness = runtime_data["stiffness"]
        if "box_x_init" in runtime_data:
            self.box_x_init = float(runtime_data["box_x_init"])
        if not np.allclose(self.semantic_config.task_frame, np.eye(3)):
            self.task_frame = np.asarray(self.semantic_config.task_frame, dtype=np.float64)
        # Dynamically pick the face that currently points away from wall
        self._update_approach_face()

    def _update_approach_face(self) -> None:
        """Pick box face that currently points away from wall."""
        axis, sign = compute_best_approach_face(
            self.get_box_rotmat(), self.get_wall_pos(), self.get_box_pos(),
        )
        self.approach_face_axis = axis
        self.approach_face_sign = sign

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

    def get_sim_wall_normal_force(self) -> float:
        """Wall-normal force from sim (task frame x-axis = world y-axis)."""
        force_task = world_to_task(self.task_frame, self.get_wrench_world()[:3])
        return abs(float(force_task[0]))

    # -- Cost function --

    def rollout_cost(self) -> float:
        """J_task + J_energy + J_force (contact-gated).

        Force terms use MuJoCo sim forces from the rollout, which are
        ground truth within the simulated trajectory. This lets MPPI
        discover that wall-normal force is needed for friction-based lift.
        """
        return float(self.compute_cost_terms()["total"])

    def compute_cost_terms(self) -> Dict[str, float]:
        """Return the exact per-term costs used by rollout_cost()."""
        weights = self.semantic_config.cost_weights
        goal = self.semantic_config.goal
        gap_target = float(goal.get("gap_target", 0.0))

        from scipy.spatial.transform import Rotation as R_conv

        box_quat_wxyz = self.get_box_quat()
        rot = R_conv.from_quat(
            [box_quat_wxyz[1], box_quat_wxyz[2], box_quat_wxyz[3], box_quat_wxyz[0]]
        )
        box_z = rot.as_matrix()[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(box_z[2]), 0, 1))))
        lateral_offset = float(self.get_box_pos()[0]) - self.box_x_init
        eef_to_contact = float(np.linalg.norm(self.get_eef_pos() - self.get_desired_contact_world()))

        task_terms = compute_wall_lift_task_cost(
            box_top_height=self.get_box_top_height(),
            target_height=float(goal.get("target_height", 0.50)),
            wall_gap=self.wall_gap(),
            eef_to_contact_distance=eef_to_contact,
            weights=weights,
            gap_target=gap_target,
            box_tilt_deg=tilt_deg,
            lateral_offset=lateral_offset,
        )

        task_cost = float(task_terms["total"])
        energy = 0.0
        force_upper = 0.0
        force_lower = 0.0
        f_n = 0.0

        if self.stiffness is not None:
            delta_task = self.get_delta_task()
            delta_norm = float(np.linalg.norm(delta_task))
            if delta_norm > 0.1:
                delta_task = delta_task * (0.1 / delta_norm)
            energy = float(weights.get("energy", 0.2)) * float(delta_task @ self.stiffness @ delta_task)

            f_n = self.get_sim_wall_normal_force()
            f_min = float(self.semantic_config.force_band.lower)
            f_max = float(self.semantic_config.force_band.upper)
            w_lower = float(weights.get("force_lower", 0.0))
            w_upper = float(weights.get("force_upper", 0.0))
            max_barrier = 500.0

            if w_lower > 0 and f_min > 0:
                force_lower = min(max_barrier, w_lower * max(0.0, f_min - f_n) ** 2)
            if w_upper > 0 and f_max < 100:
                force_upper = min(max_barrier, w_upper * max(0.0, f_n - f_max) ** 2)

        total = task_cost + energy + force_upper + force_lower
        return {
            "task": task_cost,
            "energy": energy,
            "force_upper": force_upper,
            "force_lower": force_lower,
            "sim_force_normal": f_n,
            "eef_to_contact": eef_to_contact,
            "tilt_deg": tilt_deg,
            "lateral_offset": lateral_offset,
            "height_term": float(task_terms["height"]),
            "contact_term": float(task_terms["contact"]),
            "pose_term": float(task_terms["pose"]),
            "tilt_term": float(task_terms["tilt"]),
            "lateral_term": float(task_terms["lateral"]),
            "total": total,
        }

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
    compute_wall_flip_task_cost,
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
        # spring_press scenes have no box/wall — skip the geometry-init calls.
        self.has_box = self.box_body_id is not None
        self.has_wall = self.wall_body_id is not None
        if self.has_box:
            self.box_x_init: float = float(self.get_box_pos()[0])
            self.box_top_height_init: float = float(self.get_box_top_height())
        else:
            self.box_x_init = 0.0
            self.box_top_height_init = 0.0
        if self.has_box and self.has_wall:
            self._update_approach_face()

    def configure_runtime(self, runtime_data: Optional[Dict[str, Any]]) -> None:
        if runtime_data is None:
            return
        if "semantic_config" in runtime_data:
            self.semantic_config = runtime_data["semantic_config"]
            cs = self.semantic_config.contact_strategy
            # Honor semantic/contact-hypothesis face selection for this step.
            # Recomputing the face here causes target jumps and contact drift.
            self.approach_face_axis = int(cs.approach_face_axis)
            self.approach_face_sign = float(cs.approach_face_sign)
            self.contact_standoff = float(cs.contact_standoff)
            self.contact_vertical_offset_scale = float(cs.contact_vertical_offset_scale)
            self._gripper_command = float(cs.gripper_command)
            world_offset = cs.metadata.get("world_offset")
            if world_offset is not None:
                self._contact_world_offset = np.asarray(world_offset, dtype=np.float64).reshape(3)
            else:
                self._contact_world_offset = None
        if "stiffness" in runtime_data:
            self.stiffness = runtime_data["stiffness"]
        if "box_x_init" in runtime_data:
            self.box_x_init = float(runtime_data["box_x_init"])
        if "box_top_height_init" in runtime_data:
            self.box_top_height_init = float(runtime_data["box_top_height_init"])
        if not np.allclose(self.semantic_config.task_frame, np.eye(3)):
            self.task_frame = np.asarray(self.semantic_config.task_frame, dtype=np.float64)

    def get_box_lift_height(self) -> float:
        """Top height relative to initial top height (starts near 0)."""
        return float(self.get_box_top_height() - self.box_top_height_init)

    def get_box_tilt_deg(self) -> float:
        """Absolute tilt from upright in degrees (0=upright, 90=on side)."""
        if not self.has_box:
            return 0.0
        from scipy.spatial.transform import Rotation as R_conv

        box_quat_wxyz = self.get_box_quat()
        rot = R_conv.from_quat(
            [box_quat_wxyz[1], box_quat_wxyz[2], box_quat_wxyz[3], box_quat_wxyz[0]]
        )
        box_z = rot.as_matrix()[:, 2]
        return float(np.degrees(np.arccos(np.clip(abs(box_z[2]), 0, 1))))

    # -- Override env-level accessors for the no-box spring_press scene --

    def get_wrench_world(self) -> np.ndarray:
        if self.has_box:
            return super().get_wrench_world()
        # Synthesize a +z wrench from the spring force F = k·d.
        F_z = self.get_button_force()
        return np.asarray([0.0, 0.0, F_z, 0.0, 0.0, 0.0], dtype=np.float64)

    def wall_gap(self) -> float:
        if self.has_box and self.has_wall:
            return super().wall_gap()
        # No wall in the scene: treat "gap" as how far the EEF is above the
        # button cap (positive when above, used for the contact_force trigger).
        eef_z = float(self.get_eef_pos()[2])
        cap_z = float(self._spring_press_button_top()[2])
        return max(0.0, eef_z - cap_z)

    def has_wall_contact(self, tolerance: float = 0.005) -> bool:
        if self.has_box and self.has_wall:
            return super().has_wall_contact(tolerance=tolerance)
        return self.get_button_depth() > 1e-3

    def get_desired_contact_world(self) -> np.ndarray:
        if self.has_box:
            return super().get_desired_contact_world()
        # spring_press: target = button cap top (+ standoff in +z direction).
        target = self._spring_press_button_top()
        target[2] += float(self.contact_standoff)
        return target

    def get_contact_anchor_world(self) -> np.ndarray:
        if self.has_box:
            return super().get_contact_anchor_world()
        return self._spring_press_button_top()

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

    # -- spring_press helpers (slide joint on button) --

    def get_button_depth(self, joint_suffix: str = "button_z") -> float:
        """Compression depth in meters (positive when pressed in)."""
        model = self.env.sim.model
        data = self.env.sim.data
        for jid in range(model.njnt):
            jname = model.joint_id2name(jid) or ""
            if jname.endswith(joint_suffix):
                qpos_adr = int(model.jnt_qposadr[jid])
                # qpos is negative when pressed (range [-0.06, 0]); flip sign.
                return float(-data.qpos[qpos_adr])
        return 0.0

    def get_button_stiffness(self, joint_suffix: str = "button_z") -> float:
        model = self.env.sim.model
        for jid in range(model.njnt):
            jname = model.joint_id2name(jid) or ""
            if jname.endswith(joint_suffix):
                return float(model.jnt_stiffness[jid])
        return 0.0

    def get_button_force(self) -> float:
        """Reaction force magnitude from the button spring: F = k·d."""
        return self.get_button_stiffness() * self.get_button_depth()

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

        # spring_press has no box/wall; cost is built from button depth.
        if not self.has_box:
            return self._compute_spring_press_terms(weights=weights, goal=goal)

        gap_target = float(goal.get("gap_target", 0.0))
        tilt_deg = self.get_box_tilt_deg()
        lateral_offset = float(self.get_box_pos()[0]) - self.box_x_init
        eef_to_contact = float(np.linalg.norm(self.get_eef_pos() - self.get_desired_contact_world()))
        box_lift_height = self.get_box_lift_height()
        target_height = float(goal.get("target_height", 0.50))
        target_tilt_deg = float(goal.get("target_tilt_deg", -1.0))

        if target_tilt_deg >= 0.0:
            task_terms = compute_wall_flip_task_cost(
                box_tilt_deg=tilt_deg,
                target_tilt_deg=target_tilt_deg,
                wall_gap=self.wall_gap(),
                eef_to_contact_distance=eef_to_contact,
                weights=weights,
                gap_target=gap_target,
                lateral_offset=lateral_offset,
                box_top_height=box_lift_height,
                target_height=target_height,
            )
        else:
            task_terms = compute_wall_lift_task_cost(
                box_top_height=box_lift_height,
                target_height=target_height,
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

    def _compute_spring_press_terms(
        self, *, weights: Dict[str, float], goal: Dict[str, Any],
    ) -> Dict[str, float]:
        """Cost terms for the spring_press task family.

        Eq. 5 reduces to a 1-D problem: F = k·d. We use the same key names
        as the wall_lift output so the rest of the pipeline (logger, plots)
        does not have to special-case the schema.
        """
        # Locate button cap centre in world. We pre-compute once via a body
        # name lookup so we don't re-scan every step.
        button_top = self._spring_press_button_top()
        eef_pos = self.get_eef_pos()
        # Pose cost: drive EEF to the cap, with the press axis (z) handled
        # by the force terms below — only XY distance enters here.
        xy_err = float(np.linalg.norm(eef_pos[:2] - button_top[:2]))
        pose_cost = xy_err ** 2

        depth = self.get_button_depth()
        k = self.get_button_stiffness()
        f_n = k * depth
        f_min = float(self.semantic_config.force_band.lower)
        f_max = float(self.semantic_config.force_band.upper)
        w_pose = float(weights.get("task_pose", 4.0))
        w_lower = float(weights.get("force_lower", 0.0))
        w_upper = float(weights.get("force_upper", 0.0))
        w_energy = float(weights.get("energy", 0.0))
        max_barrier = 500.0

        force_lower = 0.0
        force_upper = 0.0
        if w_lower > 0 and f_min > 0:
            force_lower = min(max_barrier, w_lower * max(0.0, f_min - f_n) ** 2)
        if w_upper > 0 and f_max < 100:
            force_upper = min(max_barrier, w_upper * max(0.0, f_n - f_max) ** 2)

        # Interaction energy proxy: 0.5·k·d² (the spring's stored energy).
        energy = w_energy * 0.5 * k * depth * depth

        task_cost = w_pose * pose_cost
        total = task_cost + energy + force_upper + force_lower
        return {
            "task": task_cost,
            "energy": energy,
            "force_upper": force_upper,
            "force_lower": force_lower,
            "sim_force_normal": f_n,
            "eef_to_contact": xy_err,
            "tilt_deg": 0.0,
            "lateral_offset": float(eef_pos[0] - button_top[0]),
            "height_term": 0.0,
            "contact_term": 0.0,
            "pose_term": pose_cost,
            "tilt_term": 0.0,
            "lateral_term": float((eef_pos[0] - button_top[0]) ** 2),
            "button_depth": depth,
            "button_stiffness": k,
            "total": total,
        }

    def _spring_press_button_top(self) -> np.ndarray:
        """World position of the button cap top, used as the EEF target."""
        try:
            bid = self.env.sim.model.body_name2id("spring_button_1_main")
        except Exception:
            return self.get_eef_pos()
        pos = self.env.sim.data.body_xpos[bid].copy()
        # Cap is a cylinder with size=(r=0.04, half_height=0.02), placed at
        # local pos (0,0,0.02). The top surface is roughly at body_z + 0.04.
        pos[2] = float(pos[2]) + 0.04
        return pos

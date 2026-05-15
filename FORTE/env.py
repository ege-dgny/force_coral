"""Environment construction and object-centric wrapper base.

Only external dependency: force_coral (LIBERO env + plugin registration).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

import force_coral

force_coral.bootstrap_libero_extensions(require=True)

from force_coral.libero_ext.env_wrapper import SegmentationRenderEnv  # noqa: E402
from force_coral.libero_ext.init_loader import load_init_bundle_by_name  # noqa: E402

from FORTE.geometry import compute_box_face_anchor, compute_wall_gap  # noqa: E402
from FORTE.types import infer_task_family  # noqa: E402

LOGGER = logging.getLogger(__name__)

# Box mass cap for friction-based wall_lift. OSC saturates ~7.5 N/axis;
# with μ_wall=0.5, lifting mg ≤ μ·N means m ≤ 0.5·7.5/g ≈ 380 g in the
# best case, but coupled control needs slack. 90 g is well inside budget.
WALL_LIFT_BOX_MASS_KG = 0.09

# Soft-wall contact dynamics for the force_hold task (foam pad, k≈200 N/m).
FORCE_HOLD_WALL_OVERRIDES = {
    "solref": "0.02 1",
    "solimp": "0.85 0.92 0.001",
}

# Some FORTE task families reuse an existing BDDL/init bundle (only the
# runtime config differs). The alias is consulted both for BDDL lookup and
# for init-file resolution.
_TASK_BDDL_ALIAS = {
    "force_hold_against_compliant_wall": "push_the_box_up_along_the_wall_while_maintaining_contact",
}


def _resolve_asset_task(task_name: str) -> str:
    return _TASK_BDDL_ALIAS.get(task_name, task_name)


def _canonical_bddl(problem_folder: str, task_name: str) -> str:
    asset_task = _resolve_asset_task(task_name)
    return force_coral.get_data_path("bddl_files") + f"/{problem_folder}/{asset_task}.bddl"


def _scale_body_mass(env: SegmentationRenderEnv, body_name: str, target_kg: float) -> None:
    """Set a body's mass to ``target_kg`` and scale inertia tensor proportionally."""
    bid = env.sim.model.body_name2id(body_name)
    current = float(env.sim.model.body_mass[bid])
    if current <= 1e-9 or target_kg <= 0.0:
        return
    scale = target_kg / current
    env.sim.model.body_mass[bid] = target_kg
    env.sim.model.body_inertia[bid] = env.sim.model.body_inertia[bid] * scale
    env.sim.forward()


def _patch_geom_contact(
    env: SegmentationRenderEnv,
    body_name: str,
    *,
    solref: Optional[np.ndarray] = None,
    solimp: Optional[np.ndarray] = None,
) -> None:
    """Override collision-geom solref/solimp on a body (works for fixtures
    that LIBERO's plugin pipeline does not pass kwargs through to)."""
    model = env.sim.model
    bid = model.body_name2id(body_name)
    start = int(model.body_geomadr[bid])
    num = int(model.body_geomnum[bid])
    for gid in range(start, start + num):
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue  # visual-only geom
        # geom_solref is shape (2,); geom_solimp is shape (5,). User-supplied
        # arrays may be shorter — only update the leading components, leaving
        # the rest at MuJoCo's defaults.
        if solref is not None:
            arr = np.asarray(solref, dtype=np.float64).ravel()
            existing = np.array(model.geom_solref[gid], dtype=np.float64)
            n = min(len(arr), existing.size)
            existing[:n] = arr[:n]
            model.geom_solref[gid] = existing
        if solimp is not None:
            arr = np.asarray(solimp, dtype=np.float64).ravel()
            existing = np.array(model.geom_solimp[gid], dtype=np.float64)
            n = min(len(arr), existing.size)
            existing[:n] = arr[:n]
            model.geom_solimp[gid] = existing


def _patch_slide_joint_stiffness(
    env: SegmentationRenderEnv,
    joint_name_suffix: str,
    *,
    stiffness: Optional[float] = None,
    damping: Optional[float] = None,
) -> None:
    """Override a slide joint's stiffness/damping (spring_press sweep)."""
    model = env.sim.model
    for jid in range(model.njnt):
        jname = model.joint_id2name(jid) or ""
        if jname.endswith(joint_name_suffix):
            if stiffness is not None:
                model.jnt_stiffness[jid] = float(stiffness)
            if damping is not None:
                # damping lives in dof_damping, indexed by jnt_dofadr.
                dof_adr = int(model.jnt_dofadr[jid])
                model.dof_damping[dof_adr] = float(damping)
            return


def build_inner_env(
    *,
    task_name: str,
    controller: str = "OSC_POSE",
    offscreen: bool = False,
    gui: bool = True,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    use_camera_obs: bool = False,
    camera_depths: bool = False,
    camera_heights: int = 240,
    camera_widths: int = 320,
) -> SegmentationRenderEnv:
    task_family = infer_task_family(task_name)
    asset_task = _resolve_asset_task(task_name)
    try:
        overrides, state = load_init_bundle_by_name(
            problem_folder=problem_folder, task_name=asset_task, init_idx=init_idx,
        )
    except (FileNotFoundError, AssertionError):
        overrides, state = {}, None

    env = SegmentationRenderEnv(
        bddl_file_name=_canonical_bddl(problem_folder, task_name),
        robots=["Panda"],
        controller=controller,
        has_renderer=gui,
        has_offscreen_renderer=offscreen,
        ignore_done=True,
        use_camera_obs=use_camera_obs,
        control_freq=20,
        camera_names=["frontview"],
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        camera_depths=camera_depths,
        camera_segmentations="instance",
        **({"object_overrides": overrides} if overrides else {}),
    )
    env.robots[0].controller_config["control_ori"] = True
    env.seed(0)
    env.reset()
    if state is not None:
        env.set_init_state(state)

    # Force-budget fix: light box keeps friction-lift within OSC's reach.
    if task_family == "wall_lift":
        _scale_body_mass(env, "block_1_main", WALL_LIFT_BOX_MASS_KG)

    # Soft wall for force_hold. LIBERO loads wall2_1 as a fixture and does
    # not thread object_overrides through to fixtures, so we patch the sim
    # model directly. This must be applied to every env (real + workers).
    if task_family == "force_hold":
        _patch_geom_contact(
            env, "wall2_1_main",
            solref=np.fromstring(FORCE_HOLD_WALL_OVERRIDES["solref"], sep=" "),
            solimp=np.fromstring(FORCE_HOLD_WALL_OVERRIDES["solimp"], sep=" "),
        )

    return env


class ObjectCentricWrapper:
    """Object-aware wrapper with action scaling, geometry helpers, and state sync."""

    action_position_scale = 8.0
    action_rotation_scale = 0.5
    _gripper_command = -1.0

    def __init__(
        self,
        env: SegmentationRenderEnv,
        *,
        box_body_name: str = "block_1_main",
        wall_body_name: str = "wall2_1_main",
        eef_site_name: str = "gripper0_grip_site",
        approach_face_axis: int = 1,
        approach_face_sign: float = -1.0,
        contact_standoff: float = 0.03,
        contact_vertical_offset_scale: float = 0.0,
    ) -> None:
        self.env = env
        self.box_body_name = box_body_name
        self.wall_body_name = wall_body_name
        self.eef_site_name = eef_site_name
        self.approach_face_axis = int(approach_face_axis)
        self.approach_face_sign = float(approach_face_sign)
        self.contact_standoff = float(contact_standoff)
        self.contact_vertical_offset_scale = float(contact_vertical_offset_scale)
        self._contact_world_offset: Optional[np.ndarray] = None

        self.box_body_id = env.sim.model.body_name2id(box_body_name)
        self.wall_body_id = env.sim.model.body_name2id(wall_body_name)
        self.box_geom_id = self._select_collision_geom_id(self.box_body_id)
        self.wall_geom_id = self._select_collision_geom_id(self.wall_body_id)
        self.box_half_extents = np.array(
            env.sim.model.geom_size[self.box_geom_id],
            dtype=np.float64,
        )
        self.wall_half_extents = np.array(
            env.sim.model.geom_size[self.wall_geom_id],
            dtype=np.float64,
        )
        self._sync_warned = False

    def _select_collision_geom_id(self, body_id: int) -> int:
        """Pick a collision geom for a body (prefer contype/conaffinity-enabled geoms)."""
        model = self.env.sim.model
        start = int(model.body_geomadr[body_id])
        num = int(model.body_geomnum[body_id])
        if num <= 0:
            return start
        geom_ids = list(range(start, start + num))
        collision_ids = [
            gid for gid in geom_ids
            if int(model.geom_contype[gid]) != 0 and int(model.geom_conaffinity[gid]) != 0
        ]
        if collision_ids:
            return collision_ids[0]
        group0_ids = [gid for gid in geom_ids if int(model.geom_group[gid]) == 0]
        if group0_ids:
            return group0_ids[0]
        return geom_ids[0]

    def step(self, action: np.ndarray) -> np.ndarray:
        action7 = np.zeros(7, dtype=np.float64)
        action7[:3] = self.action_position_scale * np.asarray(action[:3], dtype=np.float64)
        action7[3:6] = self.action_rotation_scale * np.asarray(action[3:6], dtype=np.float64)
        action7[-1] = self._gripper_command
        self.env.step(action7)
        return self.get_box_pos()

    def get_box_pos(self) -> np.ndarray:
        return self.env.sim.data.body_xpos[self.box_body_id].copy()

    def get_box_quat(self) -> np.ndarray:
        return self.env.sim.data.body_xquat[self.box_body_id].copy()

    def get_box_rotmat(self) -> np.ndarray:
        return self.env.sim.data.body_xmat[self.box_body_id].reshape(3, 3).copy()

    def get_box_top_height(self) -> float:
        """Top of oriented box along world z-axis."""
        rotmat = self.get_box_rotmat()
        z_extent = float(np.abs(rotmat[2, :]) @ self.box_half_extents)
        return float(self.get_box_pos()[2] + z_extent)

    def get_eef_pos(self) -> np.ndarray:
        sid = self.env.sim.model.site_name2id(self.eef_site_name)
        return self.env.sim.data.site_xpos[sid].copy()

    def get_wall_pos(self) -> np.ndarray:
        return self.env.sim.data.geom_xpos[self.wall_geom_id].copy()

    def get_wall_rotmat(self) -> np.ndarray:
        return self.env.sim.data.geom_xmat[self.wall_geom_id].reshape(3, 3).copy()

    def get_contact_anchor_world(self) -> np.ndarray:
        return compute_box_face_anchor(
            self.get_box_pos(), self.get_box_rotmat(), self.box_half_extents,
            face_axis=self.approach_face_axis, face_sign=self.approach_face_sign,
            standoff=0.0, vertical_offset_scale=self.contact_vertical_offset_scale,
            world_offset=self._contact_world_offset,
        )

    def get_desired_contact_world(self) -> np.ndarray:
        return compute_box_face_anchor(
            self.get_box_pos(), self.get_box_rotmat(), self.box_half_extents,
            face_axis=self.approach_face_axis, face_sign=self.approach_face_sign,
            standoff=self.contact_standoff,
            vertical_offset_scale=self.contact_vertical_offset_scale,
            world_offset=self._contact_world_offset,
        )

    def wall_gap(self) -> float:
        return compute_wall_gap(
            self.get_box_pos(), self.box_half_extents,
            self.get_wall_pos(), self.wall_half_extents,
            box_rotmat=self.get_box_rotmat(),
            wall_rotmat=self.get_wall_rotmat(),
        )

    def has_wall_contact(self, tolerance: float = 0.005) -> bool:
        return self.wall_gap() <= float(tolerance)

    def get_wrench_world(self) -> np.ndarray:
        return self.env.get_body_wrench(self.box_body_name)

    def configure_runtime(self, runtime_data: Optional[Dict[str, Any]]) -> None:
        """Override in subclasses to receive per-step runtime state."""

    def rollout_cost(self) -> float:
        raise NotImplementedError

    def sync_robot_from_real(
        self, real_env: Any, *, include_vel: bool = True, include_gripper: bool = True,
    ) -> None:
        try:
            robot = self.env.robots[0]
            if hasattr(robot, "_ref_joint_pos_indexes") and robot._ref_joint_pos_indexes is not None:
                idx = robot._ref_joint_pos_indexes
                self.env.sim.data.qpos[idx] = real_env.sim.data.qpos[idx]
            if include_vel and hasattr(robot, "_ref_joint_vel_indexes") and robot._ref_joint_vel_indexes is not None:
                idx = robot._ref_joint_vel_indexes
                self.env.sim.data.qvel[idx] = real_env.sim.data.qvel[idx]
            if include_gripper and hasattr(robot, "gripper") and robot.gripper is not None:
                g = robot.gripper
                if hasattr(g, "_ref_gripper_joint_pos_indexes") and g._ref_gripper_joint_pos_indexes is not None:
                    self.env.sim.data.qpos[g._ref_gripper_joint_pos_indexes] = (
                        real_env.sim.data.qpos[g._ref_gripper_joint_pos_indexes]
                    )
                if include_vel and hasattr(g, "_ref_gripper_joint_vel_indexes") and g._ref_gripper_joint_vel_indexes is not None:
                    self.env.sim.data.qvel[g._ref_gripper_joint_vel_indexes] = (
                        real_env.sim.data.qvel[g._ref_gripper_joint_vel_indexes]
                    )
            self.env.sim.forward()
        except Exception as exc:
            if not self._sync_warned:
                LOGGER.warning("Robot sync from real env failed: %s", exc)
                self._sync_warned = True

    def sync_box_from_real(self, real_env: Any) -> None:
        joint_name = "block_1_joint0"
        qpos_addr, _ = self.env.sim.model.get_joint_qpos_addr(joint_name)
        pose = np.concatenate([
            real_env.sim.data.body_xpos[self.box_body_id],
            real_env.sim.data.body_xquat[self.box_body_id],
        ])
        self.update_inner_from_pose(pose)

    def update_inner_from_pose(self, pose: np.ndarray, size: Optional[np.ndarray] = None) -> None:
        """Apply observed object pose to inner world (CoRAL-style dual-world sync)."""
        joint_name = "block_1_joint0"
        qpos_addr, _ = self.env.sim.model.get_joint_qpos_addr(joint_name)
        self.env.sim.data.qpos[qpos_addr: qpos_addr + 7] = np.asarray(pose, dtype=np.float64)
        if size is not None:
            geom_id = self.box_geom_id
            self.env.sim.model.geom_size[geom_id] = np.asarray(size, dtype=np.float64)
            self.box_half_extents = np.asarray(size, dtype=np.float64)
        self.env.sim.forward()

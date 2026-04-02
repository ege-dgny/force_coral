"""Shared controller core for object-aware MPPI inside force_coral."""

from __future__ import annotations

import atexit
import multiprocessing as mp
from typing import Any, Dict, Optional, Type

import numpy as np

import force_coral

force_coral.bootstrap_libero_extensions(require=True)

from force_coral.controllers.task_geometry import compute_box_face_anchor, compute_wall_gap
from force_coral.libero_ext.env_wrapper import SegmentationRenderEnv
from force_coral.libero_ext.init_loader import load_init_bundle_by_name


def _canonical_bddl(problem_folder: str, task_name: str) -> str:
    return force_coral.get_data_path("bddl_files") + f"/{problem_folder}/{task_name}.bddl"


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
    overrides, state = load_init_bundle_by_name(
        problem_folder=problem_folder,
        task_name=task_name,
        init_idx=init_idx,
    )
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
    env.set_init_state(state)
    return env


class ObjectCentricWrapperBase:
    """Shared object-aware wrapper utilities for baseline and FORTE controllers."""

    action_position_scale = 8.0
    action_rotation_scale = 0.5
    gripper_command = -1.0

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

        self.box_body_id = env.sim.model.body_name2id(self.box_body_name)
        self.wall_body_id = env.sim.model.body_name2id(self.wall_body_name)
        self.box_half_extents = np.array(
            env.sim.model.geom_size[env.sim.model.body_geomadr[self.box_body_id]],
            dtype=np.float64,
        )
        self.wall_half_extents = np.array(
            env.sim.model.geom_size[env.sim.model.body_geomadr[self.wall_body_id]],
            dtype=np.float64,
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        action7 = np.zeros(7, dtype=np.float64)
        action7[:3] = self.action_position_scale * np.asarray(action[:3], dtype=np.float64)
        action7[3:6] = self.action_rotation_scale * np.asarray(action[3:6], dtype=np.float64)
        action7[-1] = self.gripper_command
        self.env.step(action7)
        return self.get_box_pos()

    def get_box_pos(self) -> np.ndarray:
        return self.env.sim.data.body_xpos[self.box_body_id].copy()

    def get_box_quat(self) -> np.ndarray:
        return self.env.sim.data.body_xquat[self.box_body_id].copy()

    def get_box_rotmat(self) -> np.ndarray:
        return self.env.sim.data.body_xmat[self.box_body_id].reshape(3, 3).copy()

    def get_box_top_height(self) -> float:
        return float(self.get_box_pos()[2] + self.box_half_extents[2])

    def get_eef_pos(self) -> np.ndarray:
        sid = self.env.sim.model.site_name2id(self.eef_site_name)
        return self.env.sim.data.site_xpos[sid].copy()

    def get_wall_pos(self) -> np.ndarray:
        return self.env.sim.data.body_xpos[self.wall_body_id].copy()

    def get_contact_anchor_world(self) -> np.ndarray:
        return compute_box_face_anchor(
            self.get_box_pos(),
            self.get_box_rotmat(),
            self.box_half_extents,
            face_axis=self.approach_face_axis,
            face_sign=self.approach_face_sign,
            standoff=0.0,
            vertical_offset_scale=self.contact_vertical_offset_scale,
        )

    def get_desired_contact_world(self) -> np.ndarray:
        return compute_box_face_anchor(
            self.get_box_pos(),
            self.get_box_rotmat(),
            self.box_half_extents,
            face_axis=self.approach_face_axis,
            face_sign=self.approach_face_sign,
            standoff=self.contact_standoff,
            vertical_offset_scale=self.contact_vertical_offset_scale,
        )

    def wall_gap(self) -> float:
        return compute_wall_gap(
            self.get_box_pos(),
            self.box_half_extents,
            self.get_wall_pos(),
            self.wall_half_extents,
        )

    def has_wall_contact(self, tolerance: float = 0.005) -> bool:
        return self.wall_gap() <= float(tolerance)

    def get_wrench_world(self) -> np.ndarray:
        return self.env.get_body_wrench(self.box_body_name)

    def configure_runtime(self, runtime_data: Optional[Dict[str, Any]]) -> None:
        """Override in subclasses to receive per-step runtime state."""

    def rollout_cost(self):
        raise NotImplementedError

    def sync_robot_from_real(self, real_env, *, include_vel: bool = True, include_gripper: bool = True):
        try:
            robot = self.env.robots[0]
            if hasattr(robot, "_ref_joint_pos_indexes") and robot._ref_joint_pos_indexes is not None:
                idx = robot._ref_joint_pos_indexes
                self.env.sim.data.qpos[idx] = real_env.sim.data.qpos[idx]
            if include_vel and hasattr(robot, "_ref_joint_vel_indexes") and robot._ref_joint_vel_indexes is not None:
                idx = robot._ref_joint_vel_indexes
                self.env.sim.data.qvel[idx] = real_env.sim.data.qvel[idx]
            if include_gripper and hasattr(robot, "gripper") and robot.gripper is not None:
                if hasattr(robot.gripper, "_ref_gripper_joint_pos_indexes") and robot.gripper._ref_gripper_joint_pos_indexes is not None:
                    gidx = robot.gripper._ref_gripper_joint_pos_indexes
                    self.env.sim.data.qpos[gidx] = real_env.sim.data.qpos[gidx]
                if include_vel and hasattr(robot.gripper, "_ref_gripper_joint_vel_indexes") and robot.gripper._ref_gripper_joint_vel_indexes is not None:
                    gvidx = robot.gripper._ref_gripper_joint_vel_indexes
                    self.env.sim.data.qvel[gvidx] = real_env.sim.data.qvel[gvidx]
            self.env.sim.forward()
        except Exception:
            pass

    def sync_box_from_real(self, real_env) -> None:
        joint_name = "block_1_joint0"
        qpos_addr, _ = self.env.sim.model.get_joint_qpos_addr(joint_name)
        pose = np.concatenate(
            [
                real_env.sim.data.body_xpos[self.box_body_id],
                real_env.sim.data.body_xquat[self.box_body_id],
            ]
        )
        self.env.sim.data.qpos[qpos_addr : qpos_addr + 7] = pose
        self.env.sim.forward()

    def update_inner_from_pose(self, pose: np.ndarray, size: Optional[np.ndarray] = None) -> None:
        joint_name = "block_1_joint0"
        qpos_addr, _ = self.env.sim.model.get_joint_qpos_addr(joint_name)
        self.env.sim.data.qpos[qpos_addr : qpos_addr + 7] = np.asarray(pose, dtype=np.float64)
        if size is not None:
            geom_id = self.env.sim.model.body_geomadr[self.box_body_id]
            self.env.sim.model.geom_size[geom_id] = np.asarray(size, dtype=np.float64)
            self.box_half_extents = np.asarray(size, dtype=np.float64)
        self.env.sim.forward()


_worker_env = None
_worker_wrapper = None


def _init_worker(
    controller: str,
    control_freq: int,
    init_idx: int,
    problem_folder: str,
    task_name: str,
    wrapper_cls: Type[ObjectCentricWrapperBase],
    wrapper_kwargs: Dict[str, Any],
):
    del control_freq  # control_freq is fixed by env construction today.
    global _worker_env, _worker_wrapper
    _worker_env = build_inner_env(
        task_name=task_name,
        controller=controller,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    _worker_wrapper = wrapper_cls(_worker_env, **wrapper_kwargs)


def _evaluate_rollout(args):
    global _worker_env, _worker_wrapper
    action_sequence, initial_state, runtime_data = args
    _worker_env.sim.set_state(initial_state)
    _worker_env.sim.forward()
    _worker_wrapper.configure_runtime(runtime_data)

    total_cost = 0.0
    for action in action_sequence:
        _worker_wrapper.step(action)
        cost = _worker_wrapper.rollout_cost()
        total_cost += float(getattr(cost, "total", cost))
    return total_cost


class ParallelMPPI:
    """Generic multiprocessing MPPI for force_coral wrappers."""

    def __init__(
        self,
        env_wrapper: ObjectCentricWrapperBase,
        *,
        wrapper_cls: Optional[Type[ObjectCentricWrapperBase]] = None,
        wrapper_kwargs: Optional[Dict[str, Any]] = None,
        horizon: int = 10,
        num_samples: int = 64,
        noise_scale: float = 1.0,
        controller: str = "OSC_POSE",
        control_freq: int = 20,
        num_workers: Optional[int] = None,
        seed: int = 0,
        init_idx: int = 0,
        problem_folder: str = "my_suite",
        task_name: Optional[str] = None,
    ) -> None:
        if task_name is None:
            raise ValueError("task_name is required for ParallelMPPI")
        self.envw = env_wrapper
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.noise_scale = float(noise_scale)
        self.rng = np.random.default_rng(seed)
        self.position_scale = float(getattr(env_wrapper, "action_position_scale", 8.0))
        self.wrapper_cls = wrapper_cls or type(env_wrapper)
        self.wrapper_kwargs = dict(wrapper_kwargs or {})

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = int(num_workers)

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(
                controller,
                control_freq,
                init_idx,
                problem_folder,
                task_name,
                self.wrapper_cls,
                self.wrapper_kwargs,
            ),
        )
        atexit.register(self.close)

    def compute_control(self, runtime_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        initial_state = self.envw.env.sim.get_state()
        actions = (
            self.rng.uniform(-1.0, 1.0, size=(self.num_samples, self.horizon, 6))
            * (self.noise_scale / self.position_scale)
        )
        tasks = [(actions[i], initial_state, runtime_data) for i in range(self.num_samples)]
        costs = self.pool.map(_evaluate_rollout, tasks)
        best_idx = int(np.argmin(costs))
        return actions[best_idx, 0]

    def close(self) -> None:
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass

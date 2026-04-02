"""Full Phase-1 FORTE controller built on the force_coral MPPI backbone."""

from __future__ import annotations

import json
import logging
import os
import platform
from typing import Any, Dict

import cv2
import numpy as np

import force_coral
from force_coral.controllers.forte_support import ArtifactManager, CostBreakdown, WallLiftTaskMonitor
from force_coral.controllers.mppi_core import ObjectCentricWrapperBase, ParallelMPPI, build_inner_env
from force_coral.controllers.task_geometry import (
    build_wall_lift_task_frame,
    compute_wall_lift_task_cost,
    world_to_task,
)
from force_coral.dynamics.estimator import RiemannianStiffnessEstimator
from force_coral.perception.semantic_manager import SemanticManager
from force_coral.perception.vlm_interface import PhysicsConfig, TaskPhysicsParser, default_physics_config
from force_coral.utils.device import resolve_device


if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

logging.getLogger("robosuite").setLevel(logging.ERROR)

force_coral.bootstrap_libero_extensions(require=True)


TASK_NAME = "push_the_box_up_along_the_wall_while_maintaining_contact"


class ForteWrapper(ObjectCentricWrapperBase):
    """Object-aware wall-lift wrapper with FORTE augmentation."""

    def __init__(self, env) -> None:
        super().__init__(
            env,
            approach_face_axis=1,
            approach_face_sign=-1.0,
            contact_standoff=0.03,
            contact_vertical_offset_scale=0.0,
        )
        self.semantic_config: PhysicsConfig = default_physics_config()
        self.task_frame = build_wall_lift_task_frame()
        self.stiffness = None
        self.current_breakdown = None

    def configure(self, semantic_config: PhysicsConfig, stiffness: np.ndarray | None) -> None:
        self.semantic_config = semantic_config
        self.task_frame = np.asarray(semantic_config.task_frame, dtype=np.float64)
        self.stiffness = None if stiffness is None else np.asarray(stiffness, dtype=np.float64)

    def configure_runtime(self, runtime_data: Dict[str, Any] | None) -> None:
        if runtime_data is None:
            return
        self.configure(
            runtime_data.get("semantic_config", self.semantic_config),
            runtime_data.get("stiffness", self.stiffness),
        )

    def get_force_task(self) -> np.ndarray:
        return world_to_task(self.task_frame, self.get_wrench_world()[:3])

    def get_delta_task(self) -> np.ndarray:
        return world_to_task(self.task_frame, self.get_eef_pos() - self.get_contact_anchor_world())

    def compute_cost_breakdown(self) -> CostBreakdown:
        desired_contact = self.get_desired_contact_world()
        eef_pos = self.get_eef_pos()
        eef_to_contact = float(np.linalg.norm(eef_pos - desired_contact))
        wall_gap = self.wall_gap()
        task_terms = compute_wall_lift_task_cost(
            box_top_height=self.get_box_top_height(),
            target_height=float(self.semantic_config.goal.get("target_height", 0.50)),
            wall_gap=wall_gap,
            eef_to_contact_distance=eef_to_contact,
            weights=self.semantic_config.cost_weights,
        )
        task_cost = float(task_terms["total"])

        if self.stiffness is None:
            breakdown = CostBreakdown(
                task=task_cost,
                energy=0.0,
                force_upper=0.0,
                force_lower=0.0,
                total=task_cost,
                predicted_force_normal=0.0,
                predicted_force_task=np.zeros(3, dtype=np.float64),
                delta_task=self.get_delta_task(),
            )
            self.current_breakdown = breakdown
            return breakdown

        delta_task = self.get_delta_task()
        predicted_force_task = self.stiffness @ delta_task
        predicted_force_normal = float(predicted_force_task[0])
        weights = self.semantic_config.cost_weights
        energy = float(weights.get("energy", 0.2)) * float(delta_task @ self.stiffness @ delta_task)
        upper = float(weights.get("force_upper", 25.0)) * max(
            0.0, predicted_force_normal - float(self.semantic_config.force_band.upper)
        ) ** 2
        lower = float(weights.get("force_lower", 12.0)) * max(
            0.0, float(self.semantic_config.force_band.lower) - predicted_force_normal
        ) ** 2
        breakdown = CostBreakdown(
            task=task_cost,
            energy=energy,
            force_upper=upper,
            force_lower=lower,
            total=task_cost + energy + upper + lower,
            predicted_force_normal=predicted_force_normal,
            predicted_force_task=predicted_force_task,
            delta_task=delta_task,
        )
        self.current_breakdown = breakdown
        return breakdown

    def rollout_cost(self) -> CostBreakdown:
        return self.compute_cost_breakdown()


def _semantic_snapshot(semantic_config: PhysicsConfig) -> Dict[str, Any]:
    return {
        "force_band": semantic_config.force_band.to_dict(),
        "goal": dict(semantic_config.goal),
        "cost_weights": dict(semantic_config.cost_weights),
        "recovery_hints": list(semantic_config.recovery_hints),
        "task_frame": np.asarray(semantic_config.task_frame).tolist(),
    }


def run_forte(
    *,
    task_name: str = TASK_NAME,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    num_steps: int = 150,
    use_vlm: bool = False,
    show: bool = False,
    save_video: bool = True,
    device: str = "auto",
    pose_source: str = "ground_truth",
    eta: float = 0.005,
    min_eigenvalue: float = 0.1,
    horizon: int = 10,
    num_samples: int = 64,
    noise_scale: float = 1.0,
    review_interval: int = 10,
    num_workers: int | None = None,
) -> Dict[str, Any]:
    if pose_source != "ground_truth":
        raise NotImplementedError("Phase 1 only supports pose_source='ground_truth'")

    selected_device = resolve_device(device)
    out_dir = os.path.join("my_runs", f"forte_phase1_{task_name}")
    os.makedirs(out_dir, exist_ok=True)
    needs_offscreen = bool(use_vlm or save_video or show)

    real_env = build_inner_env(
        task_name=task_name,
        offscreen=needs_offscreen,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    inner_env = build_inner_env(
        task_name=task_name,
        offscreen=needs_offscreen,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    real_wrapper = ForteWrapper(real_env)
    inner_wrapper = ForteWrapper(inner_env)

    parser = TaskPhysicsParser() if use_vlm else None
    semantic = SemanticManager(parser=parser, review_interval=review_interval)
    first_frame = None
    if use_vlm:
        rgb = real_env.sim.render(camera_name="frontview", width=320, height=240)
        from PIL import Image

        first_frame = Image.fromarray(np.flipud(rgb))
    semantic_config = semantic.initialize(image=first_frame, task_prompt=task_name.replace("_", " "))
    if np.allclose(semantic_config.task_frame, np.eye(3)):
        semantic_config.task_frame = build_wall_lift_task_frame()

    estimator = RiemannianStiffnessEstimator.from_vlm_prior(
        semantic_config.stiffness_prior,
        eta=eta,
        min_eigenvalue=min_eigenvalue,
    )
    runtime_data = {
        "semantic_config": semantic_config,
        "stiffness": estimator.get_stiffness(),
    }
    real_wrapper.configure_runtime(runtime_data)
    inner_wrapper.configure_runtime(runtime_data)

    monitor = WallLiftTaskMonitor(
        target_height=float(semantic_config.goal["target_height"]),
        force_lower=semantic_config.force_band.lower,
        force_upper=semantic_config.force_band.upper,
    )
    mppi = ParallelMPPI(
        env_wrapper=inner_wrapper,
        wrapper_cls=ForteWrapper,
        horizon=horizon,
        num_samples=num_samples,
        noise_scale=noise_scale,
        controller="OSC_POSE",
        control_freq=20,
        num_workers=num_workers,
        seed=42,
        init_idx=init_idx,
        problem_folder=problem_folder,
        task_name=task_name,
    )
    artifacts = ArtifactManager(out_dir=out_dir, save_video=save_video, overlay=True)

    with open(os.path.join(out_dir, "run_config.json"), "w") as handle:
        json.dump(
            {
                "task_name": task_name,
                "device": selected_device,
                "pose_source": pose_source,
                "semantic": _semantic_snapshot(semantic_config),
            },
            handle,
            indent=2,
        )

    try:
        for step_idx in range(num_steps):
            monitor.target_height = float(semantic.active_config.goal["target_height"])
            monitor.force_lower = float(semantic.active_config.force_band.lower)
            monitor.force_upper = float(semantic.active_config.force_band.upper)

            measured_force_task = real_wrapper.get_force_task()
            delta_task = real_wrapper.get_delta_task()
            stiffness = estimator.update(measured_force_task, delta_task)
            runtime_data = {
                "semantic_config": semantic.active_config,
                "stiffness": stiffness,
            }
            real_wrapper.configure_runtime(runtime_data)
            inner_wrapper.configure_runtime(runtime_data)
            inner_wrapper.sync_robot_from_real(real_env)
            inner_wrapper.sync_box_from_real(real_env)

            action = mppi.compute_control(runtime_data=runtime_data)
            real_wrapper.step(action)
            breakdown = real_wrapper.compute_cost_breakdown()
            measured_force_task = real_wrapper.get_force_task()
            wall_contact = real_wrapper.has_wall_contact()
            contact_anchor_world = real_wrapper.get_contact_anchor_world()
            desired_contact_world = real_wrapper.get_desired_contact_world()
            eef_contact_error = float(np.linalg.norm(real_wrapper.get_eef_pos() - desired_contact_world))
            status = monitor.update(
                box_height=real_wrapper.get_box_top_height(),
                normal_force=float(measured_force_task[0]),
                wall_contact=wall_contact,
            )

            if semantic.should_review(step_idx, status):
                revision = semantic.revise(
                    monitor_status=status,
                    recent_metrics={
                        "box_height": real_wrapper.get_box_top_height(),
                        "measured_force_normal": float(measured_force_task[0]),
                    },
                )
                runtime_data = {
                    "semantic_config": semantic.active_config,
                    "stiffness": stiffness,
                }
                real_wrapper.configure_runtime(runtime_data)
                inner_wrapper.configure_runtime(runtime_data)
            else:
                revision = None

            record = {
                "step": step_idx,
                "box_height": real_wrapper.get_box_top_height(),
                "box_center_height": float(real_wrapper.get_box_pos()[2]),
                "wall_gap": float(real_wrapper.wall_gap()),
                "wall_contact": bool(wall_contact),
                "contact_anchor_world": np.asarray(contact_anchor_world).tolist(),
                "desired_contact_world": np.asarray(desired_contact_world).tolist(),
                "eef_contact_error": eef_contact_error,
                "measured_force_task": measured_force_task.tolist(),
                "measured_force_normal": float(measured_force_task[0]),
                "predicted_force_task": breakdown.predicted_force_task.tolist(),
                "predicted_force_normal": float(breakdown.predicted_force_normal),
                "delta_task": breakdown.delta_task.tolist(),
                "sigma_eigenvalues": np.linalg.eigvalsh(stiffness).tolist(),
                "cost_task": float(breakdown.task),
                "cost_energy": float(breakdown.energy),
                "cost_force_upper": float(breakdown.force_upper),
                "cost_force_lower": float(breakdown.force_lower),
                "cost_total": float(breakdown.total),
                "force_band_lower": float(semantic.active_config.force_band.lower),
                "force_band_upper": float(semantic.active_config.force_band.upper),
                "semantic_goal_height": float(semantic.active_config.goal["target_height"]),
                "semantic_revision": None if revision is None else dataclass_to_jsonable(revision),
                **status,
            }
            artifacts.log_step(record)

            if save_video or show:
                frame = real_env.sim.render(camera_name="frontview", width=320, height=240)
                frame_bgr = cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR)
                artifacts.add_frame(frame_bgr, record)
                if show:
                    cv2.imshow("FORTE Phase 1", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if status["success"]:
                break

        return artifacts.finalize()
    finally:
        cv2.destroyAllWindows()
        real_env.close()
        inner_env.close()
        mppi.close()


def dataclass_to_jsonable(instance) -> Dict[str, Any]:
    if instance is None:
        return {}
    return {key: value for key, value in instance.__dict__.items() if value is not None}


if __name__ == "__main__":
    run_forte()

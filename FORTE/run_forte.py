"""FORTE controller: force-augmented MPPI with phase-based cost switching.

The VLM (or defaults) generates a sequence of TaskPhases. Each phase carries
its own Q-weights, x_goal, [F_min, F_max], and contact strategy. The FORTE
cost structure (Eq. 5) is unchanged — only its parameters rotate when the
SemanticManager detects a phase trigger.

Usage:
    python -m FORTE.run_forte
"""

from __future__ import annotations

import json
import logging
import os
import platform
from typing import Any, Dict, Optional

import cv2
import numpy as np

from FORTE.artifacts import ArtifactManager
from FORTE.env import build_inner_env
from FORTE.estimator import RiemannianStiffnessEstimator
from FORTE.forte_wrapper import ForteWrapper
from FORTE.geometry import build_wall_lift_task_frame
from FORTE.monitor import WallLiftTaskMonitor
from FORTE.mppi import ParallelMPPI
from FORTE.semantic import SemanticManager
from FORTE.vlm import TaskPhysicsParser

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

logging.getLogger("robosuite").setLevel(logging.ERROR)

TASK_NAME = "push_the_box_up_along_the_wall_while_maintaining_contact"


def run_forte(
    *,
    task_name: str = TASK_NAME,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    num_steps: int = 200,
    show: bool = False,
    save_video: bool = True,
    use_vlm: bool = False,
    # MPPI
    horizon: int = 10,
    num_samples: int = 64,
    noise_scale: float = 1.0,
    action_multiplier: float = 80.0,
    num_workers: Optional[int] = None,
    # Stiffness estimator
    eta: float = 0.005,
    min_eigenvalue: float = 0.1,
    # Contact gating
    contact_threshold: float = 0.5,
    # Semantic revision
    review_interval: int = 10,
) -> Dict[str, Any]:
    out_dir = os.path.join("my_runs", f"forte_{task_name}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Environments ----
    real_env = build_inner_env(
        task_name=task_name, offscreen=True, gui=False,
        init_idx=init_idx, problem_folder=problem_folder,
    )
    inner_env = build_inner_env(
        task_name=task_name, offscreen=True, gui=False,
        init_idx=init_idx, problem_folder=problem_folder,
    )
    real_wrapper = ForteWrapper(real_env)
    inner_wrapper = ForteWrapper(inner_env)

    # ---- Scene geometry for VLM ----
    scene_info = {
        "box_position": real_wrapper.get_box_pos(),
        "box_half_extents": real_wrapper.box_half_extents.copy(),
        "wall_position": real_wrapper.get_wall_pos(),
        "wall_half_extents": real_wrapper.wall_half_extents.copy(),
        "eef_position": real_wrapper.get_eef_pos(),
        "wall_gap": real_wrapper.wall_gap(),
        "box_top_height": real_wrapper.get_box_top_height(),
    }

    # ---- Semantic initialization ----
    parser = TaskPhysicsParser() if use_vlm else None
    semantic = SemanticManager(parser=parser, review_interval=review_interval)

    first_frame = None
    if use_vlm:
        from PIL import Image
        rgb = real_env.sim.render(camera_name="frontview", width=320, height=240)
        first_frame = Image.fromarray(np.flipud(rgb))

    semantic_config = semantic.initialize(
        image=first_frame, task_prompt=task_name.replace("_", " "), scene_info=scene_info,
    )
    if np.allclose(semantic_config.task_frame, np.eye(3)):
        semantic_config.task_frame = build_wall_lift_task_frame()

    print(f"[FORTE] PhysicsConfig: {json.dumps(semantic_config.to_dict(), indent=2)}")
    print(f"[FORTE] Phase plan: {[p.name for p in semantic_config.phases]}")
    print(f"[FORTE] Active phase: {semantic.current_phase.name}")

    # ---- Stiffness estimator ----
    estimator = RiemannianStiffnessEstimator.from_vlm_prior(
        semantic_config.stiffness_prior, eta=eta, min_eigenvalue=min_eigenvalue,
    )

    # ---- Initial runtime (stiffness=None = pre-contact) ----
    runtime_data: Dict[str, Any] = {"semantic_config": semantic_config, "stiffness": None}
    real_wrapper.configure_runtime(runtime_data)
    inner_wrapper.configure_runtime(runtime_data)

    # ---- Task monitor ----
    monitor = WallLiftTaskMonitor(
        target_height=float(semantic_config.goal["target_height"]),
        force_lower=semantic_config.force_band.lower,
        force_upper=semantic_config.force_band.upper,
    )

    # ---- MPPI ----
    mppi = ParallelMPPI(
        env_wrapper=inner_wrapper,
        wrapper_cls=ForteWrapper,
        horizon=horizon,
        num_samples=num_samples,
        noise_scale=noise_scale,
        action_multiplier=action_multiplier,
        controller="OSC_POSE",
        control_freq=20,
        num_workers=num_workers,
        seed=42,
        init_idx=init_idx,
        problem_folder=problem_folder,
        task_name=task_name,
    )

    # ---- Artifacts ----
    artifacts = ArtifactManager(out_dir=out_dir, save_video=save_video, overlay=True)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        scene_ser = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in scene_info.items()
        }
        json.dump({
            "task_name": task_name,
            "scene_info": scene_ser,
            "semantic": semantic_config.to_dict(),
            "K_init": estimator.get_stiffness().tolist(),
        }, f, indent=2)

    # ---- Contact latch ----
    contact_latched = False
    stiffness: Optional[np.ndarray] = None

    try:
        for step in range(num_steps):
            # 1) Read force + displacement
            measured_force_task = real_wrapper.get_force_task()
            delta_task = real_wrapper.get_delta_task()

            # 2) Contact-gated stiffness (Eq. 4)
            wall_normal_force = abs(float(measured_force_task[0]))
            wall_contact = real_wrapper.has_wall_contact()
            force_detected = wall_normal_force > contact_threshold

            if wall_contact and force_detected:
                contact_latched = True
                stiffness = estimator.update(measured_force_task, delta_task)
            elif contact_latched:
                stiffness = estimator.get_stiffness()
            else:
                stiffness = None

            # 3) Build metrics for phase transition check
            eef_pos = real_wrapper.get_eef_pos()
            box_height = real_wrapper.get_box_top_height()
            eef_to_contact = float(np.linalg.norm(
                eef_pos - real_wrapper.get_desired_contact_world()
            ))
            phase_metrics = {
                "wall_contact": wall_contact,
                "wall_normal_force": wall_normal_force,
                "box_top_height": box_height,
                "eef_to_contact_distance": eef_to_contact,
            }

            # 4) Check phase transition
            new_phase = semantic.check_phase_transition(phase_metrics)
            if new_phase is not None:
                print(f"[FORTE] === Phase transition → {new_phase} at step {step} ===")
                monitor.stall_counter = 0
                monitor.over_force_counter = 0
                monitor.prev_height = None

            # Sync monitor with active phase's force band
            monitor.target_height = float(semantic.active_config.goal["target_height"])
            monitor.force_lower = float(semantic.active_config.force_band.lower)
            monitor.force_upper = float(semantic.active_config.force_band.upper)

            # 5) Runtime data for workers
            runtime_data = {"semantic_config": semantic.active_config, "stiffness": stiffness}
            real_wrapper.configure_runtime(runtime_data)
            inner_wrapper.configure_runtime(runtime_data)

            # 6) Sync inner world
            inner_wrapper.sync_robot_from_real(real_env)
            inner_wrapper.sync_box_from_real(real_env)

            # 7) MPPI
            # Pass phase-specific action prior to bias MPPI sampling
            phase_prior = np.array(semantic.current_phase.action_prior, dtype=np.float64)
            action = mppi.compute_control(
                runtime_data=runtime_data,
                action_prior=phase_prior if np.any(phase_prior != 0) else None,
            )

            # 8) Execute (80x outer * 8x inner = 640x total)
            real_wrapper.step(80.0 * action)

            # 9) Monitor
            box_height = real_wrapper.get_box_top_height()
            wall_contact = real_wrapper.has_wall_contact()
            status = monitor.update(
                box_height=box_height,
                normal_force=float(measured_force_task[0]),
                wall_contact=wall_contact,
            )

            # 10) Within-phase semantic revision
            revision = None
            did_revise = semantic.should_review(step, status)
            if did_revise:
                revision = semantic.revise(
                    monitor_status=status,
                    recent_metrics={
                        "box_height": box_height,
                        "measured_force_normal": float(measured_force_task[0]),
                    },
                )

            # 11) Build full debug record
            K_cur = stiffness if stiffness is not None else estimator.get_stiffness()
            K_eig = np.linalg.eigvalsh(K_cur).tolist()
            delta_t = real_wrapper.get_delta_task()
            F_pred_t = K_cur @ delta_t if stiffness is not None else np.zeros(3)
            F_pred_n_raw = float(F_pred_t[0])
            F_pred_n = max(0.0, F_pred_n_raw)  # clamp: negative normal force is unphysical

            # Per-term cost breakdown
            from FORTE.geometry import compute_wall_lift_task_cost
            weights = semantic.active_config.cost_weights
            goal = semantic.active_config.goal
            contact_anchor = real_wrapper.get_contact_anchor_world()
            desired_contact = real_wrapper.get_desired_contact_world()
            wg = real_wrapper.wall_gap()
            task_terms = compute_wall_lift_task_cost(
                box_top_height=box_height,
                target_height=float(goal.get("target_height", 0.50)),
                wall_gap=wg,
                eef_to_contact_distance=float(np.linalg.norm(
                    real_wrapper.get_eef_pos() - desired_contact
                )),
                weights=weights,
            )
            max_barrier = 20.0
            cost_energy = (
                float(weights.get("energy", 0.2)) * float(delta_t @ K_cur @ delta_t)
                if stiffness is not None else 0.0
            )
            cost_force_upper = (
                min(max_barrier, float(weights.get("force_upper", 25.0))
                * max(0.0, F_pred_n - float(semantic.active_config.force_band.upper)) ** 2)
                if stiffness is not None else 0.0
            )
            cost_force_lower = (
                min(max_barrier, float(weights.get("force_lower", 12.0))
                * max(0.0, float(semantic.active_config.force_band.lower) - F_pred_n) ** 2)
                if stiffness is not None else 0.0
            )

            # Box orientation
            from scipy.spatial.transform import Rotation as R_conv
            box_quat_wxyz = real_wrapper.get_box_quat()
            box_euler = R_conv.from_quat(
                [box_quat_wxyz[1], box_quat_wxyz[2], box_quat_wxyz[3], box_quat_wxyz[0]]
            ).as_euler("ZYX", degrees=True).tolist()

            record = {
                "step": step,
                "phase": semantic.current_phase.name,
                "phase_transition": new_phase is not None,
                "phase_transition_to": new_phase,
                "semantic_revision": did_revise,
                "revision_detail": (
                    {"reason": revision.review_reason, "recovery": revision.recovery_mode,
                     "force_band": revision.force_band, "cost_weights": revision.cost_weights}
                    if revision else None
                ),
                # Box state
                "box_pos": real_wrapper.get_box_pos().tolist(),
                "box_quat": box_quat_wxyz.tolist(),
                "box_euler_deg": box_euler,
                "box_height": box_height,
                # Contact geometry
                "wall_gap": float(wg),
                "wall_contact": bool(wall_contact),
                "wall_pos": real_wrapper.get_wall_pos().tolist(),
                "contact_latched": contact_latched,
                "contact_anchor_world": contact_anchor.tolist(),
                "desired_contact_world": desired_contact.tolist(),
                "eef_pos": real_wrapper.get_eef_pos().tolist(),
                "eef_to_contact": float(np.linalg.norm(
                    real_wrapper.get_eef_pos() - desired_contact
                )),
                "delta_task": delta_t.tolist(),
                # Forces
                "measured_force_task": measured_force_task.tolist(),
                "measured_force_normal": float(measured_force_task[0]),
                "predicted_force_normal": F_pred_n,
                "wall_normal_force": wall_normal_force,
                "force_band_lower": float(semantic.active_config.force_band.lower),
                "force_band_upper": float(semantic.active_config.force_band.upper),
                # Cost breakdown
                "cost_height": float(task_terms["height"]) * float(weights.get("task_height", 14.0)),
                "cost_contact": float(task_terms["contact"]) * float(weights.get("task_contact", 18.0)),
                "cost_pose": float(task_terms["pose"]) * float(weights.get("task_pose", 4.0)),
                "cost_energy": cost_energy,
                "cost_force_upper": cost_force_upper,
                "cost_force_lower": cost_force_lower,
                "cost_total": float(task_terms["total"]) + cost_energy + cost_force_upper + cost_force_lower,
                # MPPI
                "action": action.tolist(),
                # Stiffness
                "sigma_eigenvalues": K_eig,
                # Monitor
                **status,
            }
            artifacts.log_step(record)

            print(
                f"Step {step:03d} | {semantic.current_phase.name:14s} | "
                f"h={box_height:.3f}m | gap={wg:.4f}m | F_n={wall_normal_force:.2f}N | "
                f"contact={'Y' if contact_latched else 'N'} | {status['reason']}"
            )

            if save_video or show:
                frame = real_env.sim.render(camera_name="frontview", width=320, height=240)
                frame_bgr = cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR)
                artifacts.add_frame(frame_bgr, record, sim=real_env.sim)
                if show:
                    cv2.imshow("FORTE", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if status["success"]:
                print(f"Success at step {step}: h={box_height:.3f}m")
                break

        return artifacts.finalize()
    finally:
        cv2.destroyAllWindows()
        real_env.close()
        inner_env.close()
        mppi.close()


if __name__ == "__main__":
    run_forte()

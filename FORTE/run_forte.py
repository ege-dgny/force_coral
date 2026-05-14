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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from FORTE.artifacts import ArtifactManager
from FORTE.env import build_inner_env
from FORTE.estimator import RiemannianStiffnessEstimator
from FORTE.forte_wrapper import ForteWrapper
from FORTE.geometry import build_wall_lift_task_frame, compute_box_face_anchor
from FORTE.monitor import WallLiftTaskMonitor
from FORTE.mppi import ParallelMPPI
from FORTE.semantic import SemanticManager
from FORTE.types import ContactBelief, ContactHypothesis, ContactStrategy
from FORTE.vlm import TaskPhysicsParser

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

logging.getLogger("robosuite").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

TASK_NAME = "push_the_box_up_along_the_wall_while_maintaining_contact"


def _camera_world_transform(env, camera_name: str) -> np.ndarray:
    """Return camera pose in world frame (OpenCV convention)."""
    sim = env.sim
    model, data = sim.model, sim.data
    cam_id = model.camera_name2id(camera_name)
    r_gl_to_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    r_w_c_gl = data.cam_xmat[cam_id].reshape(3, 3)
    t_w_c = data.cam_xpos[cam_id]
    w_t_c = np.eye(4, dtype=np.float64)
    w_t_c[:3, :3] = r_w_c_gl @ r_gl_to_cv
    w_t_c[:3, 3] = t_w_c
    return w_t_c


def _quat_wxyz_from_rotmat(rotmat: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(rotmat).as_quat()
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


class _ObservedPoseProvider:
    """Perception-to-inner pose bridge (CoRAL-style update point)."""

    def __init__(
        self,
        *,
        pose_source: str,
        camera_name: str,
        object_name: str,
        mesh_path: Optional[str],
        track_iters: int,
    ) -> None:
        self.pose_source = pose_source
        self.camera_name = camera_name
        self.object_name = object_name
        self.mesh_path = mesh_path
        self.track_iters = int(track_iters)
        self.estimator = None
        self._warned = False
        self._initialized = False
        self._K = None

    def _gt_pose(self, real_env: Any, real_wrapper: ForteWrapper) -> np.ndarray:
        return np.concatenate(
            [
                real_env.sim.data.body_xpos[real_wrapper.box_body_id],
                real_env.sim.data.body_xquat[real_wrapper.box_body_id],
            ]
        )

    def _try_init_fp(self, real_env: Any, width: int, height: int) -> None:
        if self.pose_source != "foundationpose" or self._initialized:
            return
        self._initialized = True
        try:
            from estimater import FoundationPose
            from predictor import PoseRefinePredictor, ScorePredictor
        except Exception as exc:
            if not self._warned:
                LOGGER.warning("FoundationPose import failed, fallback to GT pose: %s", exc)
                self._warned = True
            return

        mesh_path = self.mesh_path
        if mesh_path is None:
            default_mesh = Path("data/assets/models/cube/cube.obj")
            mesh_path = str(default_mesh.resolve()) if default_mesh.exists() else None
        if mesh_path is None or not os.path.exists(mesh_path):
            if not self._warned:
                LOGGER.warning("FoundationPose mesh missing, fallback to GT pose")
                self._warned = True
            return

        cam_id = real_env.sim.model.camera_name2id(self.camera_name)
        fovy_rad = np.deg2rad(float(real_env.sim.model.cam_fovy[cam_id]))
        fy = 0.5 * float(height) / np.tan(0.5 * fovy_rad)
        fx = fy
        cx = float(width) * 0.5
        cy = float(height) * 0.5
        self._K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self.estimator = FoundationPose(
            model_pts=None,
            model_normals=None,
            mesh=mesh_path,
            scorer=scorer,
            refiner=refiner,
            debug_dir=None,
            debug=False,
        )

    def observe(self, real_env: Any, real_wrapper: ForteWrapper, frame_idx: int) -> Tuple[np.ndarray, str]:
        if self.pose_source != "foundationpose":
            return self._gt_pose(real_env, real_wrapper), "ground_truth"

        try:
            real_env._post_process()
            real_env._update_observables(force=True)
            obs = real_env.env._get_observations()
            rgb_key = f"{self.camera_name}_image"
            depth_key = f"{self.camera_name}_depth"
            seg_key = None
            for key in (f"{self.camera_name}_segmentation_instance", f"{self.camera_name}_segmentation"):
                if key in obs:
                    seg_key = key
                    break
            if rgb_key not in obs or depth_key not in obs:
                raise RuntimeError("Missing RGB/depth observations for FoundationPose")

            rgb = obs[rgb_key]
            if rgb.ndim == 3 and rgb.shape[-1] == 1:
                rgb = np.squeeze(rgb, axis=-1)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            depth = obs[depth_key]
            if depth.ndim == 3:
                depth = np.squeeze(depth, axis=-1)
            depth_m = depth.astype(np.float32)
            self._try_init_fp(real_env, width=int(rgb.shape[1]), height=int(rgb.shape[0]))
            if self.estimator is None or self._K is None:
                return self._gt_pose(real_env, real_wrapper), "ground_truth_fallback"

            mask_bool = None
            if frame_idx == 0:
                if seg_key is None:
                    raise RuntimeError("Missing segmentation for first FP registration frame")
                seg = obs[seg_key]
                if seg.ndim == 3:
                    seg = np.squeeze(seg, axis=-1)
                if self.object_name not in real_env.instance_to_id:
                    raise RuntimeError(f"Object '{self.object_name}' not in instance_to_id")
                inst_id = real_env.instance_to_id[self.object_name]
                mask_bool = (seg == inst_id)

            rgb_fp = np.ascontiguousarray(np.flipud(rgb))
            depth_fp = np.ascontiguousarray(np.flipud(depth_m))
            mask_fp = np.ascontiguousarray(np.flipud(mask_bool)) if mask_bool is not None else None

            if frame_idx == 0:
                c_t_o = self.estimator.register(
                    K=self._K,
                    rgb=rgb_fp,
                    depth=depth_fp,
                    ob_mask=mask_fp,
                    iteration=5,
                )
            else:
                c_t_o = self.estimator.track_one(
                    rgb=rgb_fp,
                    depth=depth_fp,
                    K=self._K,
                    iteration=self.track_iters,
                )
            w_t_c = _camera_world_transform(real_env, self.camera_name)
            w_t_o = w_t_c @ c_t_o
            pose = np.concatenate([w_t_o[:3, 3], _quat_wxyz_from_rotmat(w_t_o[:3, :3])])
            return pose, "foundationpose"
        except Exception as exc:
            if not self._warned:
                LOGGER.warning("FoundationPose step failed, fallback to GT pose: %s", exc)
                self._warned = True
            return self._gt_pose(real_env, real_wrapper), "ground_truth_fallback"


def _candidate_strategies(base: ContactStrategy) -> List[ContactStrategy]:
    """Generate top-K contact candidates around semantic proposal."""
    candidates: List[ContactStrategy] = []
    standoff_candidates = [base.contact_standoff, base.contact_standoff + 0.01]
    vertical_offsets = [base.contact_vertical_offset_scale, -0.2, 0.0, 0.2]
    for standoff in standoff_candidates:
        for vertical in vertical_offsets:
            cs = ContactStrategy(
                approach_face_axis=base.approach_face_axis,
                approach_face_sign=base.approach_face_sign,
                contact_standoff=float(np.clip(standoff, 0.0, 0.08)),
                contact_vertical_offset_scale=float(np.clip(vertical, -0.5, 0.5)),
                gripper_command=base.gripper_command,
            )
            candidates.append(cs)
    seen = set()
    unique: List[ContactStrategy] = []
    for cs in candidates:
        key = (
            cs.approach_face_axis,
            round(cs.approach_face_sign, 3),
            round(cs.contact_standoff, 3),
            round(cs.contact_vertical_offset_scale, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cs)
    return unique[:6]


def _select_contact_hypothesis(
    wrapper: ForteWrapper,
    measured_force_normal: float,
) -> Tuple[ContactHypothesis, List[ContactHypothesis]]:
    """Physics filter + force-consistency rerank."""
    base = wrapper.semantic_config.contact_strategy
    force_band = wrapper.semantic_config.force_band
    band_mid = 0.5 * (float(force_band.lower) + float(force_band.upper))
    scored: List[ContactHypothesis] = []
    eef_pos = wrapper.get_eef_pos()
    box_pos = wrapper.get_box_pos()
    box_rot = wrapper.get_box_rotmat()
    wall_gap = max(0.0, wrapper.wall_gap())

    for cs in _candidate_strategies(base):
        desired_contact = compute_box_face_anchor(
            box_pos=box_pos,
            box_rotmat=box_rot,
            half_extents=wrapper.box_half_extents,
            face_axis=cs.approach_face_axis,
            face_sign=cs.approach_face_sign,
            standoff=cs.contact_standoff,
            vertical_offset_scale=cs.contact_vertical_offset_scale,
        )
        eef_dist = float(np.linalg.norm(eef_pos - desired_contact))
        feasible = (eef_dist <= 0.45) and (desired_contact[2] >= (box_pos[2] - 0.7 * wrapper.box_half_extents[2]))
        if not feasible:
            continue
        normal_push_proxy = abs(float(wrapper.task_frame[0] @ (desired_contact - eef_pos)))
        desired_push_proxy = 0.005 * band_mid
        measured_push_proxy = 0.005 * abs(measured_force_normal)
        force_error = abs(normal_push_proxy - measured_push_proxy) + 0.5 * abs(
            normal_push_proxy - desired_push_proxy
        )
        score = 1.5 * eef_dist + 4.0 * force_error + 2.0 * wall_gap
        scored.append(ContactHypothesis(contact_strategy=cs, score=score, reason="pose+force_consistency"))

    if not scored:
        fallback = ContactHypothesis(contact_strategy=base, score=999.0, reason="fallback_base_strategy")
        return fallback, [fallback]

    scored.sort(key=lambda item: item.score)
    return scored[0], scored[:3]


def _update_contact_belief(
    previous: ContactBelief,
    *,
    wall_contact: bool,
    force_detected: bool,
    best_score: float,
) -> ContactBelief:
    if wall_contact and force_detected:
        mode = "contact"
    elif wall_contact:
        mode = "pre_contact"
    else:
        mode = "free"
    confidence = float(np.exp(-max(0.0, best_score)))
    uncertain_steps = previous.uncertain_steps + 1 if confidence < 0.45 else 0
    return ContactBelief(mode=mode, confidence=confidence, uncertain_steps=uncertain_steps)


def run_forte(
    *,
    task_name: str = TASK_NAME,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    num_steps: int = 200,
    show: bool = False,
    save_video: bool = True,
    use_vlm: bool = False,
    pose_source: str = "ground_truth",
    camera_name: str = "frontview",
    fp_object_name: str = "block_1_main",
    fp_mesh_path: Optional[str] = None,
    fp_track_iters: int = 5,
    # MPPI
    horizon: int = 12,
    num_samples: int = 128,
    noise_scale: float = 1.0,
    action_multiplier: float = 10.0,
    num_workers: Optional[int] = None,
    # Stiffness estimator
    eta: float = 0.005,
    min_eigenvalue: float = 0.1,
    # Contact gating
    contact_threshold: float = 0.5,
    # MPPI refinement
    mppi_iters: int = 3,
    # Semantic revision
    review_interval: int = 20,
) -> Dict[str, Any]:
    if pose_source not in {"ground_truth", "foundationpose"}:
        raise ValueError("pose_source must be 'ground_truth' or 'foundationpose'")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("my_runs", f"forte_{task_name}", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    use_camera_obs = (pose_source == "foundationpose")

    # ---- Environments ----
    real_env = build_inner_env(
        task_name=task_name, offscreen=True, gui=False,
        init_idx=init_idx, problem_folder=problem_folder,
        use_camera_obs=use_camera_obs,
        camera_depths=use_camera_obs,
        camera_heights=320,
        camera_widths=320,
    )
    inner_env = build_inner_env(
        task_name=task_name, offscreen=True, gui=False,
        init_idx=init_idx, problem_folder=problem_folder,
        use_camera_obs=False,
        camera_depths=False,
        camera_heights=320,
        camera_widths=320,
    )
    real_wrapper = ForteWrapper(real_env)
    inner_wrapper = ForteWrapper(inner_env)

    # ---- Scene geometry + physics for VLM ----
    # Extract MuJoCo physics so VLM can reason about forces
    box_body_id = real_wrapper.box_body_id
    box_mass = float(real_env.sim.model.body_mass[box_body_id])
    box_geom_id = real_env.sim.model.body_geomadr[box_body_id]
    box_friction = float(real_env.sim.model.geom_friction[box_geom_id, 0])
    wall_body_id = real_wrapper.wall_body_id
    wall_geom_id = real_env.sim.model.body_geomadr[wall_body_id]
    wall_friction = float(real_env.sim.model.geom_friction[wall_geom_id, 0])
    effective_friction = max(box_friction, wall_friction)

    scene_info = {
        "box_position": real_wrapper.get_box_pos(),
        "box_half_extents": real_wrapper.box_half_extents.copy(),
        "box_mass_kg": box_mass,
        "box_weight_N": box_mass * 9.81,
        "box_friction": box_friction,
        "wall_position": real_wrapper.get_wall_pos(),
        "wall_half_extents": real_wrapper.wall_half_extents.copy(),
        "wall_friction": wall_friction,
        "effective_contact_friction_mu": effective_friction,
        "eef_position": real_wrapper.get_eef_pos(),
        "wall_gap": real_wrapper.wall_gap(),
        "box_top_height_abs": real_wrapper.get_box_top_height(),
        "box_top_height_rel": 0.0,
        "gripper_max_opening_m": 0.02,
        "osc_position_gain_kp": 150,
        "osc_output_max_m_per_step": 0.05,
        "approx_max_force_per_axis_N": 7.5,
    }

    # ---- Semantic initialization ----
    parser = TaskPhysicsParser() if use_vlm else None
    semantic = SemanticManager(parser=parser, review_interval=review_interval)

    first_frame = None
    if use_vlm:
        from PIL import Image
        rgb = real_env.sim.render(camera_name=camera_name, width=320, height=240)
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
    box_x_init = float(real_wrapper.get_box_pos()[0])
    box_top_height_init = float(real_wrapper.get_box_top_height())
    runtime_data: Dict[str, Any] = {
        "semantic_config": semantic_config, "stiffness": None,
        "box_x_init": box_x_init,
        "box_top_height_init": box_top_height_init,
    }
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
    pose_provider = _ObservedPoseProvider(
        pose_source=pose_source,
        camera_name=camera_name,
        object_name=fp_object_name,
        mesh_path=fp_mesh_path,
        track_iters=fp_track_iters,
    )
    contact_belief = ContactBelief(mode="free", confidence=0.0, uncertain_steps=0)

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
            "pose_source": pose_source,
        }, f, indent=2)

    # ---- Contact latch ----
    contact_latched = False
    stiffness: Optional[np.ndarray] = None

    try:
        for step in range(num_steps):
            # 1) Read force + displacement (pre-action measurements)
            measured_force_task_pre = real_wrapper.get_force_task()
            delta_task = real_wrapper.get_delta_task()

            # 2) Contact-gated stiffness (Eq. 4)
            wall_normal_force = abs(float(measured_force_task_pre[0]))
            wall_contact = real_wrapper.has_wall_contact()
            force_detected = wall_normal_force > contact_threshold

            if wall_contact and force_detected:
                contact_latched = True
                stiffness = estimator.update(measured_force_task_pre, delta_task)
            elif contact_latched:
                stiffness = estimator.get_stiffness()
            else:
                stiffness = None

            # 3) Build metrics for phase transition check
            eef_pos = real_wrapper.get_eef_pos()
            box_height = real_wrapper.get_box_top_height()
            box_height_rel = real_wrapper.get_box_lift_height()
            eef_to_contact = float(np.linalg.norm(
                eef_pos - real_wrapper.get_desired_contact_world()
            ))
            phase_metrics = {
                "wall_contact": wall_contact,
                "wall_normal_force": wall_normal_force,
                "box_top_height": box_height_rel,
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

            # 5) Contact-point hypotheses: propose -> filter -> rerank
            best_hypothesis, top_hypotheses = _select_contact_hypothesis(
                real_wrapper, measured_force_normal=wall_normal_force
            )
            semantic.active_config.contact_strategy = best_hypothesis.contact_strategy
            contact_belief = _update_contact_belief(
                contact_belief,
                wall_contact=wall_contact,
                force_detected=force_detected,
                best_score=best_hypothesis.score,
            )

            # 6) Runtime data for workers
            runtime_data = {
                "semantic_config": semantic.active_config,
                "stiffness": stiffness,
                "box_x_init": box_x_init,
                "box_top_height_init": box_top_height_init,
            }
            real_wrapper.configure_runtime(runtime_data)
            inner_wrapper.configure_runtime(runtime_data)

            # 7) CoRAL-style dual-world sync: observed pose -> inner state
            observed_pose, observed_pose_source = pose_provider.observe(real_env, real_wrapper, step)
            inner_wrapper.sync_robot_from_real(real_env)
            inner_wrapper.update_inner_from_pose(observed_pose)

            # 8) MPPI
            # Phase-specific action prior. If VLM provided one, use it.
            # Otherwise, compute geometric prior from EEF→contact direction.
            phase_prior = np.array(semantic.current_phase.action_prior, dtype=np.float64)
            if not np.any(phase_prior[:3] != 0):
                # No position prior from VLM — compute from geometry
                contact_target = real_wrapper.get_desired_contact_world()
                direction = contact_target - eef_pos
                dist = float(np.linalg.norm(direction))
                if dist > 0.02:
                    phase_prior[:3] = (direction / dist) * 0.8

            fallback_active = contact_belief.uncertain_steps >= 5
            if fallback_active:
                phase_prior[:3] = np.array([0.0, 0.15, 0.25], dtype=np.float64)

            action = mppi.compute_control(
                runtime_data=runtime_data,
                action_prior=phase_prior if np.any(phase_prior != 0) else None,
                num_iters=1 if fallback_active else mppi_iters,
            )

            # 9) Execute — SAME multiplier as workers (matched scaling)
            action_scale = 1.0
            real_wrapper.step(action_multiplier * action_scale * action)

            # 10) Monitor uses post-action force/state from same step
            measured_force_task_post = real_wrapper.get_force_task()
            box_height = real_wrapper.get_box_top_height()
            box_height_rel = real_wrapper.get_box_lift_height()
            wall_contact = real_wrapper.has_wall_contact()
            status = monitor.update(
                box_height=box_height_rel,
                normal_force=float(measured_force_task_post[0]),
                wall_contact=wall_contact,
            )

            # 11) Semantic revision (LLM re-query when use_vlm=True)
            revision = None
            did_revise = semantic.should_review(step, status)
            if did_revise:
                # Capture scene image for LLM revision
                revision_image = None
                if use_vlm:
                    from PIL import Image
                    rgb = real_env.sim.render(camera_name=camera_name, width=320, height=240)
                    revision_image = Image.fromarray(np.flipud(rgb))

                # Estimator diagnostics for LLM context
                K_diag = estimator.get_stiffness()
                est_state = {
                    "eigenvalues": np.linalg.eigvalsh(K_diag).tolist(),
                    "contact_latched": contact_latched,
                }

                revision = semantic.revise(
                    monitor_status=status,
                    recent_metrics={
                        "box_height": box_height_rel,
                        "box_tilt_deg": float(np.degrees(np.arccos(np.clip(
                            abs(real_wrapper.get_box_rotmat()[2, 2]), 0, 1
                        )))),
                        "measured_force_normal": float(measured_force_task_post[0]),
                        "wall_gap": float(real_wrapper.wall_gap()),
                        "lateral_offset": float(real_wrapper.get_box_pos()[0]) - box_x_init,
                        "eef_to_contact": eef_to_contact,
                    },
                    step_idx=step,
                    estimator_state=est_state,
                    image=revision_image,
                )

                # After LLM revision, re-sync stiffness prior if changed
                if revision.review_reason.startswith("llm_revision"):
                    new_prior = semantic.active_config.stiffness_prior
                    estimator.reset(
                        RiemannianStiffnessEstimator.from_vlm_prior(
                            new_prior, eta=eta, min_eigenvalue=min_eigenvalue,
                        ).get_stiffness()
                    )
                    print(f"[FORTE] LLM revision applied: {revision.review_reason}")

            # 12) Build full debug record (same objective as rollout_cost)
            K_cur = stiffness if stiffness is not None else estimator.get_stiffness()
            K_eig = np.linalg.eigvalsh(K_cur).tolist()
            delta_t = real_wrapper.get_delta_task()
            weights = semantic.active_config.cost_weights
            cost_terms = real_wrapper.compute_cost_terms()
            contact_anchor = real_wrapper.get_contact_anchor_world()
            desired_contact = real_wrapper.get_desired_contact_world()
            wg = real_wrapper.wall_gap()
            lateral_offset = cost_terms["lateral_offset"]

            # Box orientation
            box_quat_wxyz = real_wrapper.get_box_quat()
            box_euler = Rotation.from_quat(
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
                "box_height": box_height_rel,
                "box_height_abs": box_height,
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
                "measured_force_task": measured_force_task_post.tolist(),
                "measured_force_task_pre": measured_force_task_pre.tolist(),
                "measured_force_task_post": measured_force_task_post.tolist(),
                "measured_force_normal": float(measured_force_task_post[0]),
                "measured_force_normal_pre": float(measured_force_task_pre[0]),
                "measured_force_normal_post": float(measured_force_task_post[0]),
                "predicted_force_normal": float(cost_terms["sim_force_normal"]),
                "sim_force_normal_rollout_cost": float(cost_terms["sim_force_normal"]),
                "wall_normal_force_pre": wall_normal_force,
                "force_band_lower": float(semantic.active_config.force_band.lower),
                "force_band_upper": float(semantic.active_config.force_band.upper),
                "lateral_offset": lateral_offset,
                # Cost breakdown
                "cost_height": float(cost_terms["height_term"]) * float(weights.get("task_height", 14.0)),
                "cost_contact": float(cost_terms["contact_term"]) * float(weights.get("task_contact", 18.0)),
                "cost_pose": float(cost_terms["pose_term"]) * float(weights.get("task_pose", 4.0)),
                "cost_tilt": float(cost_terms["tilt_term"]) * float(weights.get("task_tilt", 8.0)),
                "cost_lateral": float(cost_terms["lateral_term"]) * float(weights.get("task_lateral", 0.0)),
                "cost_energy": float(cost_terms["energy"]),
                "cost_force_upper": float(cost_terms["force_upper"]),
                "cost_force_lower": float(cost_terms["force_lower"]),
                "cost_total": float(cost_terms["total"]),
                # MPPI
                "action": action.tolist(),
                "action_scale": action_scale,
                "fallback_active": fallback_active,
                # Contact inference
                "selected_contact_hypothesis": best_hypothesis.to_dict(),
                "top_contact_hypotheses": [h.to_dict() for h in top_hypotheses],
                "contact_belief": contact_belief.to_dict(),
                # Observation
                "observed_pose": observed_pose.tolist(),
                "observed_pose_source": observed_pose_source,
                # Stiffness
                "sigma_eigenvalues": K_eig,
                # Monitor
                **status,
            }
            artifacts.log_step(record)

            print(
                f"Step {step:03d} | {semantic.current_phase.name:14s} | "
                f"h={box_height_rel:.3f}m | gap={wg:.4f}m | lat={lateral_offset:+.3f}m | "
                f"F_n={float(measured_force_task_post[0]):.2f}N | "
                f"contact={'Y' if contact_latched else 'N'} | {status['reason']}"
            )

            if save_video or show:
                frame = real_env.sim.render(camera_name=camera_name, width=320, height=240)
                frame_bgr = cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR)
                artifacts.add_frame(frame_bgr, record, sim=real_env.sim)
                if show:
                    cv2.imshow("FORTE", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if status["success"]:
                print(f"Success at step {step}: h={box_height_rel:.3f}m")
                break

        return artifacts.finalize()
    finally:
        cv2.destroyAllWindows()
        real_env.close()
        inner_env.close()
        mppi.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", action="store_true", help="Enable VLM (GPT-4o)")
    p.add_argument("--pose-source", type=str, default="ground_truth", choices=["ground_truth", "foundationpose"])
    p.add_argument("--camera", type=str, default="frontview")
    p.add_argument("--fp-object", type=str, default="block_1_main")
    p.add_argument("--fp-mesh", type=str, default=None)
    p.add_argument("--fp-track-iters", type=int, default=5)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--multiplier", type=float, default=10.0)
    args = p.parse_args()
    run_forte(
        use_vlm=args.vlm,
        pose_source=args.pose_source,
        camera_name=args.camera,
        fp_object_name=args.fp_object,
        fp_mesh_path=args.fp_mesh,
        fp_track_iters=args.fp_track_iters,
        num_steps=args.steps,
        show=args.show,
        save_video=not args.no_video,
        num_workers=args.workers,
        num_samples=args.samples,
        horizon=args.horizon,
        action_multiplier=args.multiplier,
    )

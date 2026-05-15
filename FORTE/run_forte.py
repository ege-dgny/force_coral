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
from FORTE.geometry import (
    build_wall_lift_task_frame,
    compute_box_face_anchor,
    compute_wall_contact_face,
    coral_wall_contact_offset,
)
from FORTE.monitor import WallLiftTaskMonitor
from FORTE.mppi import ParallelMPPI
from FORTE.semantic import SemanticManager
from FORTE.types import (
    ContactBelief,
    ContactHypothesis,
    ContactSelectorState,
    ContactStrategy,
    infer_task_family,
)
from FORTE.vlm import TaskPhysicsParser

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

logging.getLogger("robosuite").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

TASK_NAME = "push_the_box_up_along_the_wall_while_maintaining_contact"
WALL_FLIP_TASK_NAME = "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side"
DEMO_TASKS = {
    "wall_lift": TASK_NAME,
    "wall_flip": WALL_FLIP_TASK_NAME,
}


def _load_dotenv_if_present() -> None:
    """Best-effort .env loader for server runs without shell export."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path.home() / ".env",
    ]
    for dotenv_path in candidates:
        if not dotenv_path.exists():
            continue
        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Keep quoted values intact, strip lightweight inline comments otherwise.
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
                if key:
                    os.environ.setdefault(key, value)
            if os.environ.get("OPENAI_API_KEY"):
                LOGGER.info("Loaded OPENAI_API_KEY from %s", dotenv_path)
                return
        except Exception as exc:
            LOGGER.warning("Failed reading %s: %s", dotenv_path, exc)


def _goal_uses_tilt(goal: Dict[str, Any]) -> bool:
    return "target_tilt_deg" in goal and float(goal.get("target_tilt_deg", -1.0)) >= 0.0


def _target_progress_from_goal(goal: Dict[str, Any]) -> float:
    if _goal_uses_tilt(goal):
        return float(goal.get("target_tilt_deg", 80.0))
    return float(goal.get("target_height", 0.50))


def _monitor_config_from_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
    if _goal_uses_tilt(goal):
        return {
            "target_metric_name": "tilt_deg",
            "progress_eps": 0.2,       # degrees
            "drop_threshold": 2.0,     # degrees
            "success_requires_contact": False,
        }
    return {
        "target_metric_name": "height_m",
        "progress_eps": 1e-3,         # meters
        "drop_threshold": 0.01,       # meters
        "success_requires_contact": True,
    }


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


def _face_axis_sign_order(wrapper: ForteWrapper, base: ContactStrategy) -> List[Tuple[int, float]]:
    """Ordered face candidates: semantic seed, wall-contact geometry, then all others."""
    geom_axis, geom_sign = compute_wall_contact_face(
        wrapper.get_box_rotmat(), wrapper.get_wall_pos(), wrapper.get_box_pos()
    )
    ordered: List[Tuple[int, float]] = [
        (int(base.approach_face_axis), float(np.sign(base.approach_face_sign) or -1.0)),
        (int(geom_axis), float(np.sign(geom_sign) or -1.0)),
    ]
    for axis in (0, 1, 2):
        for sign in (-1.0, 1.0):
            ordered.append((axis, sign))
    seen = set()
    out: List[Tuple[int, float]] = []
    for axis, sign in ordered:
        key = (int(axis), float(sign))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _candidate_strategies(wrapper: ForteWrapper, base: ContactStrategy) -> List[ContactStrategy]:
    """Generate face-aware contact candidates around semantic proposal."""
    candidates: List[ContactStrategy] = []
    standoff_candidates = [
        base.contact_standoff,
        base.contact_standoff + 0.01,
        base.contact_standoff - 0.01,
    ]
    vertical_offsets = [base.contact_vertical_offset_scale, -0.2, 0.0, 0.2]
    for axis, sign in _face_axis_sign_order(wrapper, base):
        for standoff in standoff_candidates:
            for vertical in vertical_offsets:
                cs = ContactStrategy(
                    approach_face_axis=int(axis),
                    approach_face_sign=float(sign),
                    contact_standoff=float(np.clip(standoff, 0.0, 0.08)),
                    contact_vertical_offset_scale=float(np.clip(vertical, -0.5, 0.5)),
                    gripper_command=base.gripper_command,
                    metadata=dict(base.metadata),
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
    return unique[:24]


def _select_contact_hypothesis(
    wrapper: ForteWrapper,
    measured_force_normal: float,
) -> Tuple[ContactHypothesis, List[ContactHypothesis]]:
    """Physics filter + semantic-biased force-consistency rerank."""
    base = wrapper.semantic_config.contact_strategy
    force_band = wrapper.semantic_config.force_band
    band_mid = 0.5 * (float(force_band.lower) + float(force_band.upper))
    raw_conf = base.metadata.get("confidence", 0.7)
    try:
        semantic_conf = float(raw_conf)
    except (TypeError, ValueError):
        semantic_conf = 0.7
    semantic_conf = float(np.clip(semantic_conf, 0.0, 1.0))
    geom_axis, geom_sign = compute_wall_contact_face(
        wrapper.get_box_rotmat(), wrapper.get_wall_pos(), wrapper.get_box_pos()
    )
    scored: List[ContactHypothesis] = []
    eef_pos = wrapper.get_eef_pos()
    box_pos = wrapper.get_box_pos()
    box_rot = wrapper.get_box_rotmat()
    wall_gap = max(0.0, wrapper.wall_gap())

    for cs in _candidate_strategies(wrapper, base):
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
        feasible = (eef_dist <= 0.45) and (
            desired_contact[2] >= (box_pos[2] - 0.8 * wrapper.box_half_extents[2])
        )
        if not feasible:
            continue
        normal_push_proxy = abs(float(wrapper.task_frame[0] @ (desired_contact - eef_pos)))
        desired_push_proxy = 0.005 * band_mid
        measured_push_proxy = 0.005 * abs(measured_force_normal)
        force_error = abs(normal_push_proxy - measured_push_proxy) + 0.5 * abs(
            normal_push_proxy - desired_push_proxy
        )
        same_semantic_face = (
            cs.approach_face_axis == base.approach_face_axis
            and np.sign(cs.approach_face_sign) == np.sign(base.approach_face_sign)
        )
        same_geom_face = (
            cs.approach_face_axis == int(geom_axis)
            and np.sign(cs.approach_face_sign) == np.sign(geom_sign)
        )
        semantic_penalty = 0.0 if same_semantic_face else (0.4 + 0.8 * semantic_conf)
        geom_bonus = -0.15 if same_geom_face else 0.0
        score = 1.8 * eef_dist + 3.5 * force_error + 1.5 * wall_gap + semantic_penalty + geom_bonus
        scored.append(ContactHypothesis(contact_strategy=cs, score=score, reason="pose+force_consistency"))

    if not scored:
        fallback = ContactHypothesis(contact_strategy=base, score=999.0, reason="fallback_base_strategy")
        return fallback, [fallback]

    scored.sort(key=lambda item: item.score)
    return scored[0], scored


def _best_hypothesis_for_face(
    hypotheses: List[ContactHypothesis], axis: int, sign: float
) -> Optional[ContactHypothesis]:
    for hyp in hypotheses:
        cs = hyp.contact_strategy
        if cs.approach_face_axis == int(axis) and np.sign(cs.approach_face_sign) == np.sign(sign):
            return hyp
    return None


def _temporal_contact_selection(
    selector_state: ContactSelectorState,
    best_hypothesis: ContactHypothesis,
    ranked_hypotheses: List[ContactHypothesis],
    *,
    step: int,
    dwell_steps: int = 6,
) -> ContactHypothesis:
    active = selector_state.active_strategy
    best_cs = best_hypothesis.contact_strategy
    same_face = (
        active.approach_face_axis == best_cs.approach_face_axis
        and np.sign(active.approach_face_sign) == np.sign(best_cs.approach_face_sign)
    )
    if same_face:
        return best_hypothesis
    can_switch = selector_state.last_switch_step < 0 or (step - selector_state.last_switch_step) >= dwell_steps
    if can_switch:
        selector_state.last_switch_step = step
        selector_state.switch_count += 1
        return best_hypothesis
    keep_face_hyp = _best_hypothesis_for_face(
        ranked_hypotheses, active.approach_face_axis, active.approach_face_sign
    )
    if keep_face_hyp is not None:
        return keep_face_hyp
    return ContactHypothesis(
        contact_strategy=active,
        score=best_hypothesis.score,
        reason="dwell_hold_previous_face",
    )


def _align_semantic_contact_to_wall(wrapper: ForteWrapper, config) -> None:
    """Ensure semantic/VLM contact strategy targets the wall-facing face (CoRAL convention)."""
    axis, sign = compute_wall_contact_face(
        wrapper.get_box_rotmat(), wrapper.get_wall_pos(), wrapper.get_box_pos()
    )
    standoff, vertical = coral_wall_contact_offset(wrapper.box_half_extents)
    config.contact_strategy.approach_face_axis = int(axis)
    config.contact_strategy.approach_face_sign = float(sign)
    if float(config.contact_strategy.contact_standoff) <= 0.0:
        config.contact_strategy.contact_standoff = float(standoff)
    if abs(float(config.contact_strategy.contact_vertical_offset_scale)) < 1e-6:
        config.contact_strategy.contact_vertical_offset_scale = float(vertical)
    for phase in config.phases:
        phase.contact_strategy.approach_face_axis = int(axis)
        phase.contact_strategy.approach_face_sign = float(sign)


def _geometric_fallback_strategy(wrapper: ForteWrapper, base: ContactStrategy) -> ContactStrategy:
    axis, sign = compute_wall_contact_face(
        wrapper.get_box_rotmat(), wrapper.get_wall_pos(), wrapper.get_box_pos()
    )
    standoff, vertical = coral_wall_contact_offset(wrapper.box_half_extents)
    return ContactStrategy(
        approach_face_axis=int(axis),
        approach_face_sign=float(sign),
        contact_standoff=float(standoff),
        contact_vertical_offset_scale=float(vertical),
        gripper_command=base.gripper_command,
        metadata={**dict(base.metadata), "fallback": "geometry"},
    )


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
    _load_dotenv_if_present()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("my_runs", f"forte_{task_name}", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    use_camera_obs = (pose_source == "foundationpose")
    task_family = infer_task_family(task_name)

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
    box_geom_id = real_wrapper.box_geom_id
    box_friction = float(real_env.sim.model.geom_friction[box_geom_id, 0])
    wall_body_id = real_wrapper.wall_body_id
    wall_geom_id = real_wrapper.wall_geom_id
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
        "task_family": task_family,
    }

    # ---- Semantic initialization ----
    if use_vlm and not os.environ.get("OPENAI_API_KEY"):
        LOGGER.warning(
            "OPENAI_API_KEY not found after .env load; continuing with default non-VLM semantics."
        )
        parser = None
    else:
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
    if task_family in {"wall_lift", "wall_flip"}:
        _align_semantic_contact_to_wall(real_wrapper, semantic_config)
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
    monitor_cfg = _monitor_config_from_goal(semantic_config.goal)
    monitor = WallLiftTaskMonitor(
        target_height=_target_progress_from_goal(semantic_config.goal),
        force_lower=semantic_config.force_band.lower,
        force_upper=semantic_config.force_band.upper,
        target_metric_name=monitor_cfg["target_metric_name"],
        success_requires_contact=bool(monitor_cfg["success_requires_contact"]),
        progress_eps=float(monitor_cfg["progress_eps"]),
        drop_threshold=float(monitor_cfg["drop_threshold"]),
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
    selector_state = ContactSelectorState(
        active_strategy=ContactStrategy(
            approach_face_axis=semantic_config.contact_strategy.approach_face_axis,
            approach_face_sign=semantic_config.contact_strategy.approach_face_sign,
            contact_standoff=semantic_config.contact_strategy.contact_standoff,
            contact_vertical_offset_scale=semantic_config.contact_strategy.contact_vertical_offset_scale,
            gripper_command=semantic_config.contact_strategy.gripper_command,
            metadata=dict(semantic_config.contact_strategy.metadata),
        )
    )

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

            eef_pos = real_wrapper.get_eef_pos()
            box_height = real_wrapper.get_box_top_height()
            box_height_rel = real_wrapper.get_box_lift_height()

            # 3) Contact-point hypotheses: propose -> filter -> rerank -> temporal select
            best_hypothesis, ranked_hypotheses = _select_contact_hypothesis(
                real_wrapper, measured_force_normal=wall_normal_force
            )
            selected_hypothesis = _temporal_contact_selection(
                selector_state,
                best_hypothesis,
                ranked_hypotheses,
                step=step,
            )
            semantic.active_config.contact_strategy = selected_hypothesis.contact_strategy

            # Tracking-first fallback gate: if EEF cannot track selected target for
            # sustained steps, temporarily recover face from geometry.
            selected_cs = semantic.active_config.contact_strategy
            box_pos = real_wrapper.get_box_pos()
            box_rot = real_wrapper.get_box_rotmat()
            desired_now = compute_box_face_anchor(
                box_pos=box_pos,
                box_rotmat=box_rot,
                half_extents=real_wrapper.box_half_extents,
                face_axis=selected_cs.approach_face_axis,
                face_sign=selected_cs.approach_face_sign,
                standoff=selected_cs.contact_standoff,
                vertical_offset_scale=selected_cs.contact_vertical_offset_scale,
            )
            tracking_error = float(np.linalg.norm(eef_pos - desired_now))
            selector_state.tracking_error_ema = 0.9 * selector_state.tracking_error_ema + 0.1 * tracking_error
            high_error = tracking_error > 0.09
            selector_state.high_error_steps = selector_state.high_error_steps + 1 if high_error else 0
            selector_state.fallback_active = selector_state.high_error_steps >= 8
            selector_state.fallback_reason = (
                "high_tracking_error"
                if selector_state.fallback_active
                else ""
            )
            if selector_state.fallback_active:
                semantic.active_config.contact_strategy = _geometric_fallback_strategy(
                    real_wrapper, semantic.active_config.contact_strategy
                )
            selector_state.active_strategy = semantic.active_config.contact_strategy

            runtime_data = {
                "semantic_config": semantic.active_config,
                "stiffness": stiffness,
                "box_x_init": box_x_init,
                "box_top_height_init": box_top_height_init,
                "contact_fallback_mode": selector_state.fallback_active,
            }
            real_wrapper.configure_runtime(runtime_data)
            inner_wrapper.configure_runtime(runtime_data)

            # 4) Phase transition metrics use the selected contact target (not stale face).
            eef_to_contact = float(np.linalg.norm(
                eef_pos - real_wrapper.get_desired_contact_world()
            ))
            phase_metrics = {
                "wall_contact": wall_contact,
                "wall_normal_force": wall_normal_force,
                "box_top_height": box_height_rel,
                "eef_to_contact_distance": eef_to_contact,
            }
            new_phase = semantic.check_phase_transition(phase_metrics)
            if new_phase is not None:
                print(f"[FORTE] === Phase transition → {new_phase} at step {step} ===")
                if task_family in {"wall_lift", "wall_flip"}:
                    _align_semantic_contact_to_wall(real_wrapper, semantic.active_config)
                monitor.stall_counter = 0
                monitor.over_force_counter = 0
                monitor.prev_height = None

            # Sync monitor with active phase's force band
            monitor.target_height = _target_progress_from_goal(semantic.active_config.goal)
            monitor.force_lower = float(semantic.active_config.force_band.lower)
            monitor.force_upper = float(semantic.active_config.force_band.upper)
            monitor_cfg = _monitor_config_from_goal(semantic.active_config.goal)
            monitor.target_metric_name = monitor_cfg["target_metric_name"]
            monitor.progress_eps = float(monitor_cfg["progress_eps"])
            monitor.drop_threshold = float(monitor_cfg["drop_threshold"])
            monitor.success_requires_contact = bool(monitor_cfg["success_requires_contact"])

            contact_belief = _update_contact_belief(
                contact_belief,
                wall_contact=wall_contact,
                force_detected=force_detected,
                best_score=selected_hypothesis.score,
            )

            # 6) CoRAL-style dual-world sync: observed pose -> inner state
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

            fallback_active = contact_belief.uncertain_steps >= 5 or selector_state.fallback_active
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
            box_tilt_deg = real_wrapper.get_box_tilt_deg()
            progress_value = box_tilt_deg if _goal_uses_tilt(semantic.active_config.goal) else box_height_rel
            wall_contact = real_wrapper.has_wall_contact()
            status = monitor.update(
                box_height=progress_value,
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
                        "box_tilt_deg": box_tilt_deg,
                        "task_progress": progress_value,
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
                "box_tilt_deg": box_tilt_deg,
                "task_progress": progress_value,
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
                "applied_contact_hypothesis": ContactHypothesis(
                    contact_strategy=selector_state.active_strategy,
                    score=selected_hypothesis.score,
                    reason=(
                        selector_state.fallback_reason
                        if selector_state.fallback_reason
                        else selected_hypothesis.reason
                    ),
                ).to_dict(),
                "top_contact_hypotheses": [h.to_dict() for h in ranked_hypotheses[:3]],
                "contact_belief": contact_belief.to_dict(),
                "contact_selector_state": selector_state.to_dict(),
                "eef_to_contact_error": tracking_error,
                "face_switch_count": int(selector_state.switch_count),
                "selected_axis_sign": [
                    int(selector_state.active_strategy.approach_face_axis),
                    float(selector_state.active_strategy.approach_face_sign),
                ],
                "fallback_reason": selector_state.fallback_reason,
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
                f"h={box_height_rel:.3f}m | tilt={box_tilt_deg:5.1f}deg | "
                f"prog={progress_value:.3f} | gap={wg:.4f}m | lat={lateral_offset:+.3f}m | "
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
                print(
                    f"Success at step {step}: "
                    f"{monitor.target_metric_name}={progress_value:.3f} "
                    f"(target={monitor.target_height:.3f})"
                )
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
    p.add_argument(
        "--demo",
        type=str,
        default="wall_lift",
        choices=sorted(DEMO_TASKS.keys()),
        help="Demo preset task to run.",
    )
    p.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Explicit task name override (takes precedence over --demo).",
    )
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
    task_name = args.task_name if args.task_name else DEMO_TASKS[args.demo]
    print(f"[FORTE] Running demo='{args.demo}' task='{task_name}'")
    run_forte(
        task_name=task_name,
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

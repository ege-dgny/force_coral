import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MUJOCO_GL"] = "egl"   # glfw yerine egl: headless + daha hızlı
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"  # GPU 0
os.environ.setdefault('FOUNDATIONPOSE_LOG_LEVEL', 'WARNING')
os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("robosuite").setLevel(logging.ERROR)

import numpy as np
from IPython.display import clear_output, display
import pandas as pd
#import openai
from openai import OpenAI
import json
from pathlib import Path
import types
import textwrap
import inspect
import math
import sys
import mujoco
import glob

# Bootstrap force_coral (sets up sys.path for LIBERO + registers extensions)
import force_coral

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from force_coral.controllers.mppi_core import (
    ParallelMPPI as SharedParallelMPPI,
    build_inner_env as shared_build_inner_env,
)
from force_coral.libero_ext.env_wrapper import SegmentationRenderEnv
from force_coral.libero_ext.init_loader import load_init_bundle_by_name

import contextlib, io
_silent = io.StringIO()
with contextlib.redirect_stdout(_silent), contextlib.redirect_stderr(_silent):
    import robosuite as _robosuite
    load_controller_config = _robosuite.load_controller_config
from dotenv import load_dotenv


import re, io, base64, json
import openai
from PIL import Image

load_dotenv()
client = openai.OpenAI()  # reads OPENAI_API_KEY from environment

# ---------------- FoundationPose integration config (no CLI) ----------------
# This config controls how we capture frames from MuJoCo and run FoundationPose.
FPOSE_CONFIG = {
    "camera": "frontview",
    "height": 512,
    "width": 512,
    # Target object key from SegmentationRenderEnv.instance_to_id
    "object_name": "block_1",
    # Optional: input mesh for FoundationPose; if scaling is enabled, a scaled copy
    # is emitted under the run directory and used for pose estimation
    "mesh_in": os.path.join("my_object_models", "cube", "textured_cube_FP_atlas.obj"),
    "emit_scaled_mesh": True,
    # Enable image + overlay saving under my_runs/<task>
    "save_debug": True,
    # Run FP's refine on track_one
    "track_iters": 5,
    # Report per-frame pose deltas vs MuJoCo GT (translation [m], rotation [deg])
    "report_pose_deltas": True,
}

# Toggle: if True, use FoundationPose estimate for inner-world pose.
# If False, use MuJoCo ground truth directly. You can change this by hand.
USE_FOUNDATIONPOSE_FOR_INNER = True

 

 

# Extra deps used by FP helpers
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
import trimesh  # mesh scaling
import transformations as T  # quaternion utilities

_THIS_DIR = Path(__file__).resolve().parent
_CORAL_ROOT = _THIS_DIR.parent.parent  # controllers/ -> force_coral/ -> CoRAL/
_FP_DIR = Path(os.environ.get("FOUNDATIONPOSE_DIR", _CORAL_ROOT / "FoundationPose")).resolve()
if str(_FP_DIR) not in sys.path:
    sys.path.insert(0, str(_FP_DIR))


import importlib
from scipy.spatial.transform import Rotation as R


desired_orientation = R.from_euler("xyz", [90, 0, 0], degrees=True)

 

 

def _import_foundationpose_modules():
    """Lazy-import FoundationPose and optional Utils from the sibling repo.

    Returns (FoundationPose, ScorePredictor, PoseRefinePredictor, draw_xyz_axis, draw_posed_3d_box)
    Any missing item will be returned as None and callers should handle gracefully.
    """
    try:
        EST = importlib.import_module("estimater")
        FP = getattr(EST, "FoundationPose", None)
        Score = getattr(EST, "ScorePredictor", None)
        Refine = getattr(EST, "PoseRefinePredictor", None)
    except Exception as e:
        print(f"[WARN] Could not import 'estimater' from FoundationPose: {e}")
        return None, None, None, None, None
    try:
        UTL = importlib.import_module("Utils")
        draw_axis = getattr(UTL, "draw_xyz_axis", None)
        draw_box = getattr(UTL, "draw_posed_3d_box", None)
    except Exception:
        draw_axis = None
        draw_box = None
    return FP, Score, Refine, draw_axis, draw_box
    

def ask_vlm_for_object_state(image, task_desc=""):
    """
    VLM’e görüntüyü gönder, objelerin pose + size tahminini JSON olarak döndür.
    """
    # 1) resmi base64 encode et
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2) prompt hazırla
    content = [
        {
            "type": "input_text",
            "text": (
                f"Task: {task_desc}\n"
                "For each object in the scene, return JSON array with:\n"
                "{label, pose: [x,y,z,qw,qx,qy,qz], size: [sx,sy,sz]}.\n"
                "Units: meters. Pose is in world coordinates.\n"
                "If exact numbers are not possible, approximate or use defaults, but always return valid JSON inside ```json ... ```."
                "Only output JSON inside a ```json block."
            ),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
        }
    ]

    # 3) model çağrısı
    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        temperature=0.0,
    )

    raw = resp.output_text.strip()
    print("📦 VLM raw output:\n", raw[:500])
    return raw

def ask_vlm_for_object_state_pose(image, robot_pose=None, task_desc=""):
    """
    VLM: sadece pose tahmini, robot pose referansı ile
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Robot pose bilgisi string olarak hazırla
    robot_pose_str = ""
    if robot_pose is not None:
        robot_pose_str = (
            f"\nRobot base pose (world coords): {robot_pose} "
            "(format: [x,y,z,qw,qx,qy,qz]). Use this as reference frame."
        )

    content = [
        {
            "type": "input_text",
            "text": (
                f"Task: {task_desc}\n"
                "For each object, return JSON array with:\n"
                "{label, pose: [x,y,z,qw,qx,qy,qz]}.\n"
                "Units: meters. Pose is in world coordinates.\n"
                "Always output JSON inside a ```json block."
                + robot_pose_str
            ),
        },
        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
    ]

    resp = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        temperature=0.0,
    )

    raw = resp.output_text.strip()
    print("📦 VLM raw output (pose-only):\n", raw[:500])
    return raw

def extract_json_block_for_state(text):
    """
    VLM’den gelen yanıtı parse edip JSON array döndürür.
    """
    match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("JSON bloğu bulunamadı.")
    json_str = match.group(1)

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, list):
            raise ValueError("Beklenen JSON dizisi (liste).")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError("Geçersiz JSON formatı.") from e


# ---------------- FoundationPose helpers ----------------
def _get_box_extents_for_body(env, body_name: str) -> np.ndarray:
    """Return (sx, sy, sz) full box extents [m] for geoms on the given body.

    Tries env.env.obj_body_id mapping if available, else falls back to MuJoCo ids.
    """
    sim = env.sim
    model = sim.model

    # Prefer explicit mapping from the environment if present
    bid = None
    obj_map = getattr(env, "env", None)
    if obj_map is not None:
        obj_map = getattr(obj_map, "obj_body_id", None)
        if isinstance(obj_map, dict) and (body_name in obj_map):
            bid = int(obj_map[body_name])

    # Fallback to direct name
    if bid is None:
        try:
            bid = int(model.body_name2id(body_name))
        except Exception:
            # Last resort: direct known name used in this repo
            try:
                bid = int(model.body_name2id(f"{body_name}_main"))
            except Exception:
                raise RuntimeError(f"Body '{body_name}' not found for mesh scaling")

    mjGEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
    extents = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != int(bid):
            continue
        if int(model.geom_type[gid]) != mjGEOM_BOX:
            continue
        half = np.array(model.geom_size[gid][:3], dtype=np.float64)
        extents.append(2.0 * half)
    if not extents:
        raise RuntimeError(f"No BOX geoms found for body id {bid} (name '{body_name}')")
    return np.asarray(extents).max(axis=0)


def _scale_and_emit_mesh(src_obj: str, target_extents: np.ndarray, out_dir: str) -> str:
    """Scale OBJ to target_extents (x,y,z) meters and write to out_dir.

    Uses trimesh if available; otherwise performs a simple OBJ vertex scaling.
    Returns the path to the scaled .obj.
    """
    if trimesh is None:
        raise RuntimeError("trimesh not available for mesh scaling")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    src_obj = os.path.abspath(src_obj)
    dst_obj = os.path.join(out_dir, os.path.splitext(os.path.basename(src_obj))[0] + "_scaled.obj")

    mesh = trimesh.load(src_obj, force='mesh')
    orig_ext = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(orig_ext <= 1e-12):
        raise RuntimeError(f"Source mesh has near-zero extents: {orig_ext}")
    scale = (target_extents / orig_ext).astype(np.float64)
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = scale.tolist()
    mesh.apply_transform(S)
    mesh.export(dst_obj)
    # Try to fix relative texture paths in MTL to point near dst
    dst_mtl = os.path.splitext(dst_obj)[0] + ".mtl"
    if os.path.exists(dst_mtl):
        try:
            with open(dst_mtl, 'r') as f:
                lines = f.read().splitlines()
            new_lines = []
            src_dir = os.path.dirname(src_obj)
            dst_dir = os.path.dirname(dst_obj)
            for ln in lines:
                if ln.strip().lower().startswith('map_kd'):
                    parts = ln.split(maxsplit=1)
                    tex_rel = parts[1].strip() if len(parts) > 1 else ''
                    cand = os.path.join(dst_dir, os.path.basename(tex_rel))
                    # copy from source if missing
                    if not os.path.exists(cand):
                        cand_src = os.path.join(src_dir, os.path.basename(tex_rel))
                        if os.path.exists(cand_src):
                            try:
                                import shutil
                                shutil.copy2(cand_src, cand)
                            except Exception:
                                pass
                    ln = f"map_Kd {os.path.basename(cand)}"
                new_lines.append(ln)
            with open(dst_mtl, 'w') as f:
                f.write("\n".join(new_lines) + "\n")
        except Exception:
            pass
    return dst_obj


def _camera_world_transform(env, camera_name: str) -> np.ndarray:
    """Return W_T_C as 4x4 from MuJoCo (convert GL -> CV convention)."""
    sim = env.sim
    model, data = sim.model, sim.data
    cam_id = model.camera_name2id(camera_name)
    R_gl_to_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    R_w_c_gl = data.cam_xmat[cam_id].reshape(3, 3)
    t_w_c = data.cam_xpos[cam_id]
    W_T_C = np.eye(4, dtype=np.float64)
    W_T_C[:3, :3] = R_w_c_gl @ R_gl_to_cv
    W_T_C[:3, 3] = t_w_c
    return W_T_C


def _canonical_bddl(problem_folder: str, task_name: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), problem_folder, f"{task_name}.bddl")


def build_inner_env(
    *,
    task_name: str,
    controller: str = "OSC_POSE",
    offscreen: bool = False,
    gui: bool = True,
    init_idx: int = None,
    problem_folder: str = "my_suite",
    use_camera_obs: bool = False,
    camera_depths: bool = False,
    camera_heights: int = 512,
    camera_widths: int = 512,
):
    if init_idx is None:
        raise ValueError("init_idx must be provided for build_inner_env (no fallback path)")
    return shared_build_inner_env(
        task_name=task_name,
        controller=controller,
        offscreen=offscreen,
        gui=gui,
        init_idx=init_idx,
        problem_folder=problem_folder,
        use_camera_obs=use_camera_obs,
        camera_depths=camera_depths,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )

class SimpleWrapper:
    def __init__(self, env):
        self.env = env
        self.box_body_name = "block_1_main"
        self.box_body_id = env.sim.model.body_name2id(self.box_body_name)
        self.panda_eef_name = "gripper0_grip_site"
        geom_id = env.sim.model.body_geomadr[self.box_body_id]
        half_size = env.sim.model.geom_size[geom_id]
        self.half_extents = np.array(half_size)
        self.current_phase = "push"

    def reset(self, modify_scene=False):
        self.env.reset()
        base_id = self.env.sim.model.body_name2id("robot0_base")
       # base'in world pozisyonunu değiştir
        #self.env.sim.model.body_pos[base_id][1] -= 0.50   # –y yönünde 0.50 m kaydır
        #self.env.sim.forward()

        if modify_scene:
            # sadece outer world için uygula
            wall_id = self.env.sim.model.body_name2id("wall2_1_main")
            self.env.sim.model.body_pos[wall_id][1] += 0.10

            joint_name = "block_1_joint0"
            qpos_addr, size = self.env.sim.model.get_joint_qpos_addr(joint_name)
            self.env.sim.data.qpos[qpos_addr + 0] -= 0.15
            self.env.sim.data.qpos[qpos_addr + 1] += 0.15
            self.env.sim.forward()

    def step(self, action, render=False):
        action7 = np.zeros(7)
        action7[:3] = 8.0 * action[:3]       # pozisyon
        action7[3:6] = 0.5 * action[3:6]     # rotasyon (küçük gain ile)
        action7[-1] = -1.0                   # gripper kapalı
        self.env.step(action7)
        return self.env.sim.data.body_xpos[self.box_body_id]

    def contact_strategy(self, phase="push"):
        box_pos = self.env.sim.data.body_xpos[self.box_body_id]
        box_mat = self.env.sim.data.body_xmat[self.box_body_id].reshape(3, 3)

        if phase == "push":
            # kutunun dışarıya bakan yüzünden it
            local_point = np.array([0, +self.half_extents[1], 0])
        elif phase == "flip":
            # duvara bakan yüzeyin ortasını hedef al
            local_point = np.array([0, -self.half_extents[1], 0])
        else:
            raise ValueError(f"Unknown phase: {phase}")

        return box_pos + box_mat @ local_point
    def state_cost(self):
        box_pos = self.env.sim.data.body_xpos[self.box_body_id]
        box_quat = self.env.sim.data.body_xquat[self.box_body_id]

        # Kutu rotasyonu
        Rmat = R.from_quat(box_quat).as_matrix()

        # --- 1) Duvara mesafe (kutu duvara yaklaşsın) ---
        dist_to_wall = (0.20 - (box_pos[1] + self.half_extents[1]))
        dist_cost = dist_to_wall**2

        # --- 2) EEF konumunun kutunun -y yüzeyine yakınlığı ---
        eef_pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.panda_eef_name)]
        target_contact = box_pos + np.array([0, -self.half_extents[1] - 0.025, -0.05])  # kutu -y yüzeyinden biraz arkada
        contact_cost = np.linalg.norm(eef_pos - target_contact)


        # --- Toplam cost ---
        cost = (8.0 * dist_cost +
                2.0 * contact_cost )  # ağırlıklar: hizalama > pozisyon > tilt

        return cost



    def _site_rotmat(self, site_name: str):
        sid = self.env.sim.model.site_name2id(site_name)
        # MuJoCo site_xmat düz bir 9-lu vektördür (row-major); 3x3'e çevir
        return self.env.sim.data.site_xmat[sid].reshape(3, 3)



    def update_inner_from_vision(self, pose, size=None):
        """
        Vision çıktısıyla inner env güncelle.
        pose: (x, y, z, qw, qx, qy, qz)
        size: (sx, sy, sz) -> half extents (opsiyonel)
        """
        joint_name = "block_1_joint0"
        qpos_addr, size_q = self.env.sim.model.get_joint_qpos_addr(joint_name)

        # pozisyon & oryantasyon
        self.env.sim.data.qpos[qpos_addr:qpos_addr+7] = pose

        # boyut güncellemesi gerekiyorsa
        if size is not None:
            geom_id = self.env.sim.model.body_geomadr[self.box_body_id]
            self.env.sim.model.geom_size[geom_id] = np.array(size)
            self.half_extents = np.array(size)

        self.env.sim.forward()

    def sync_robot_from_real(self, real_env, *, include_vel=True, include_gripper=True):
        """
        Copy only the robot joints (and optionally gripper) qpos/qvel
        from the real_env into this inner env. This avoids a full-state copy
        and keeps control over what is synchronized.

        Tries to use robosuite robot reference index mappings. If unavailable,
        this will silently skip (so it is safe to call even if the underlying
        attributes differ across versions).
        """
        try:
            robot = self.env.robots[0]
            # Joint positions / velocities
            if hasattr(robot, "_ref_joint_pos_indexes") and robot._ref_joint_pos_indexes is not None:
                pos_idx = robot._ref_joint_pos_indexes
                self.env.sim.data.qpos[pos_idx] = real_env.sim.data.qpos[pos_idx]
            if include_vel and hasattr(robot, "_ref_joint_vel_indexes") and robot._ref_joint_vel_indexes is not None:
                vel_idx = robot._ref_joint_vel_indexes
                self.env.sim.data.qvel[vel_idx] = real_env.sim.data.qvel[vel_idx]

            # Gripper joints (if present)
            if include_gripper and hasattr(robot, "gripper") and robot.gripper is not None:
                if hasattr(robot.gripper, "_ref_gripper_joint_pos_indexes") and robot.gripper._ref_gripper_joint_pos_indexes is not None:
                    gpos_idx = robot.gripper._ref_gripper_joint_pos_indexes
                    self.env.sim.data.qpos[gpos_idx] = real_env.sim.data.qpos[gpos_idx]
                if include_vel and hasattr(robot.gripper, "_ref_gripper_joint_vel_indexes") and robot.gripper._ref_gripper_joint_vel_indexes is not None:
                    gvel_idx = robot.gripper._ref_gripper_joint_vel_indexes
                    self.env.sim.data.qvel[gvel_idx] = real_env.sim.data.qvel[gvel_idx]

            self.env.sim.forward()
        except Exception:
            print("Warning: could not sync robot state from real_env.", "@"*20)
            pass


# ---- Parallel MPPI (CPU multi-process) ----
import multiprocessing as mp
import atexit

# Worker tarafında global env tutacağız (pickle etmeyelim)
_worker_env = None
_worker_wrapper = None

def _init_worker(controller, control_freq, init_idx=None, problem_folder="my_suite", task_name=None):
    """Her process başında kendi env'ini kur."""
    global _worker_env, _worker_wrapper
    # Reuse the same builder used in the main process; requires task_name
    _worker_env = build_inner_env(
        task_name=task_name,
        controller=controller,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )

    # Senin SimpleWrapper'ını yeniden kullanıyoruz:
    _worker_wrapper = SimpleWrapper(_worker_env)
    #_worker_wrapper.reset(modify_scene=False)

def _evaluate_one(args):
    """Tek bir action_sequence için rollout cost'u döndür."""
    global _worker_env, _worker_wrapper
    action_sequence, initial_state = args  # (H,3), mujoco.MJSTATE

    # Env durumunu initial'e geri sar
    _worker_env.sim.set_state(initial_state)
    _worker_env.sim.forward()

    total_cost = 0.0
    # Rollout
    for a in action_sequence:
        _worker_wrapper.step(a)           # senin step() 7-D eyleme map ediyor
        total_cost += _worker_wrapper.state_cost()

    # En sondaki cost'u döndürmek istersen total yerine avg/last seçebilirsin
    return total_cost

class ParallelMPPI(SharedParallelMPPI):
    def __init__(self, env_wrapper,
                 horizon=10, num_samples=64, noise_scale=1.0,
                 controller="OSC_POSE", control_freq=20,
                 num_workers=None, seed=0,
                 init_idx=None, problem_folder="my_suite", task_name=None):
        super().__init__(
            env_wrapper=env_wrapper,
            wrapper_cls=SimpleWrapper,
            horizon=horizon,
            num_samples=num_samples,
            noise_scale=noise_scale,
            controller=controller,
            control_freq=control_freq,
            num_workers=num_workers,
            seed=seed,
            init_idx=init_idx,
            problem_folder=problem_folder,
            task_name=task_name,
        )


import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

import cv2

def vision_mppi_visual_fast(
    with_size=False,
    show=True,
    *,
    init_idx=None,
    problem_folder="my_suite",
    task_name=None,
    sync_robot=True,
    sync_robot_vel=True,
    sync_gripper=True,
):
    if task_name is None:
        raise ValueError("task_name is required (BDDL path support dropped)")

    inner_env = build_inner_env(
        task_name=task_name,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    inner_wrapper = SimpleWrapper(inner_env)

    # YENİ (paralel):
    mppi = ParallelMPPI(
        env_wrapper=inner_wrapper,          # sadece wrapper API'si lazım
        horizon=10,
        num_samples=64,                     # çekirdek sayına göre arttır
        noise_scale=1.0,                    # step'te 8x var; burada küçük kalıyor
        controller="OSC_POSE",
        control_freq=20,
        num_workers=None,                   # otomatik: cpu_count()-1
        seed=42,
        init_idx=init_idx,
        problem_folder=problem_folder,
        task_name=task_name,
    )
    # Real environment (offscreen render, GUI yok) with camera obs + depth for FP
    real_env = build_inner_env(
        task_name=task_name,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
        use_camera_obs=True,
        camera_depths=True,
        camera_heights=FPOSE_CONFIG["height"],
        camera_widths=FPOSE_CONFIG["width"],
    )
    real_wrapper = SimpleWrapper(real_env)
    #real_wrapper.reset(modify_scene=False)
    #inner_wrapper.reset(modify_scene=False)

    # Robot visibility is always default (no hiding)

    # Set up FoundationPose estimator and output dirs
    cam = FPOSE_CONFIG["camera"]
    H, W = FPOSE_CONFIG["height"], FPOSE_CONFIG["width"]
    obj_name = FPOSE_CONFIG["object_name"]
    out_root = os.path.join("my_runs", task_name)
    os.makedirs(out_root, exist_ok=True)
    rgb_dir = os.path.join(out_root, "rgb")
    depth_dir = os.path.join(out_root, "depth")
    mask_dir = os.path.join(out_root, "masks")
    for d in (rgb_dir, depth_dir, mask_dir):
        os.makedirs(d, exist_ok=True)

    # Intrinsics
    K = get_camera_intrinsic_matrix(real_env.sim, cam, H, W)
    K = np.asarray(K, dtype=np.float32)
    np.savetxt(os.path.join(out_root, "cam_K.txt"), K, fmt="%.18e")

    # Resolve mesh path for FoundationPose
    mesh_dir = os.path.join(out_root, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path_for_fp = None
    # Prefer an existing scaled mesh in the run directory
    try:
        existing_scaled = sorted(glob(os.path.join(mesh_dir, "*_scaled.obj")))
    except Exception:
        existing_scaled = []
    if existing_scaled:
        mesh_path_for_fp = existing_scaled[0]
        print(f"[FP] Using existing scaled mesh: {mesh_path_for_fp}")
    else:
        # Determine source mesh candidates
        cand_srcs = []
        cfg_mesh = FPOSE_CONFIG.get("mesh_in")
        if cfg_mesh:
            cand_srcs.append(cfg_mesh)
        # Sibling FoundationPose default
        fp_default = Path(_FP_DIR) / "my_object_models" / "cube" / "textured_cube_FP_atlas.obj"
        cand_srcs.append(str(fp_default))
        # Any .obj in FoundationPose cube directory (last resort)
        cube_dir = Path(_FP_DIR) / "my_object_models" / "cube"
        if cube_dir.exists():
            for p in cube_dir.glob("*.obj"):
                cand_srcs.append(str(p))

        # Pick the first existing candidate (absolute or relative)
        src_mesh = None
        for p in cand_srcs:
            if not p:
                continue
            if os.path.isabs(p) and os.path.exists(p):
                src_mesh = p
                break
            # Try relative to FoundationPose dir
            rel_fp = os.path.join(str(_FP_DIR), p)
            if os.path.exists(rel_fp):
                src_mesh = rel_fp
                break
            # Try CWD
            if os.path.exists(p):
                src_mesh = p
                break

        if src_mesh is None:
            print("[WARN] No valid mesh source found for FoundationPose. Looked in:")
            for p in cand_srcs:
                print("   ", p)
        else:
            # Optionally scale to match MuJoCo object and save under run/mesh
            if FPOSE_CONFIG.get("emit_scaled_mesh", True):
                try:
                    target_ext = _get_box_extents_for_body(real_env, obj_name)
                    mesh_path_for_fp = _scale_and_emit_mesh(src_mesh, target_ext, mesh_dir)
                    print(f"[FP] Scaled mesh -> {mesh_path_for_fp} (target extents: {target_ext.tolist()})")
                except Exception as e:
                    print(f"[FP] Mesh scaling failed: {e}")
                    mesh_path_for_fp = src_mesh
            else:
                mesh_path_for_fp = src_mesh

    # Prepare estimator (lazy import of nvdiffrast)
    # Lazy import FP modules here (avoid importing in worker processes)
    FoundationPose, ScorePredictor, PoseRefinePredictor, draw_xyz_axis, draw_posed_3d_box = _import_foundationpose_modules()
    est = None
    try:
        # Avoid stale torch-extensions locks by using a task-local build dir
        torch_ext_dir = os.path.join(out_root, "torch_extensions")
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", torch_ext_dir)
        os.environ.setdefault("NVDR_TORCH_FORCE_BUILD", "1")
        import nvdiffrast.torch as dr
        try:
            glctx = dr.RasterizeCudaContext()
        except Exception as e_cuda:
            glctx = dr.RasterizeGLContext()
            print(f"[FP] Falling back to GL rasterizer: {e_cuda}")

        # Load mesh (required for FP)
        if not (trimesh is not None and mesh_path_for_fp and os.path.exists(mesh_path_for_fp)):
            raise RuntimeError(f"Mesh path invalid or trimesh unavailable: {mesh_path_for_fp}")
        mesh = trimesh.load(mesh_path_for_fp, force='mesh')
        if mesh is None:
            raise RuntimeError(f"Failed to load mesh at {mesh_path_for_fp}")

        scorer = ScorePredictor() if ScorePredictor is not None else None
        refiner = PoseRefinePredictor() if PoseRefinePredictor is not None else None
        scorer.cfg['crop_ratio'] = 1.5
        refiner.cfg['crop_ratio'] = 1.5
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=(mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=int(bool(FPOSE_CONFIG.get("save_debug", True))),
            debug_dir=os.path.join(out_root, "fp_debug"),
        )
        print(f"[FP] FoundationPose estimator initialized with mesh: {mesh_path_for_fp}")
    except Exception as e:
        print(f"[WARN] FoundationPose init failed: {e}")

    try:
        frame_idx = 0
        for t in range(200):
            # --- Grab RGB, depth, seg from real env ---
            real_env._post_process()
            real_env._update_observables(force=True)
            obs = real_env.env._get_observations()
            rgb_key = f"{cam}_image"
            depth_key = f"{cam}_depth"
            seg_key = None
            if frame_idx == 0:
                # Only need mask on the first frame for initial registration
                for k in (f"{cam}_segmentation_instance", f"{cam}_segmentation"):
                    if k in obs:
                        seg_key = k
                        break
                if (rgb_key not in obs) or (depth_key not in obs) or (seg_key is None):
                    raise RuntimeError("[FP] Missing obs keys for RGB/Depth/Segmentation on first frame")
            else:
                if (rgb_key not in obs) or (depth_key not in obs):
                    raise RuntimeError("[FP] Missing obs keys for RGB/Depth")

            rgb = obs[rgb_key]
            if rgb.ndim == 3 and rgb.shape[-1] == 1:
                rgb = np.squeeze(rgb, axis=-1)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            depth = obs[depth_key]
            if depth.ndim == 3:
                depth = np.squeeze(depth, axis=-1)
            depth_m = depth.astype(np.float32)
            mask_bool = None
            if frame_idx == 0:
                seg = obs[seg_key]
                if seg.ndim == 3:
                    seg = np.squeeze(seg, axis=-1)
                if obj_name not in real_env.instance_to_id:
                    raise RuntimeError(f"[FP] Object '{obj_name}' not found in instance_to_id mapping")
                inst_id = real_env.instance_to_id[obj_name]
                mask_bool = (seg == inst_id)

            # Save demo_data-style images
            if FPOSE_CONFIG.get("save_debug", True):
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.png"), rgb)
                # Save mask only for the initial frame; keep RGB per frame for debugging
                depth_mm = np.clip(depth_m * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
                imageio.imwrite(os.path.join(depth_dir, f"{frame_idx:06d}.png"), depth_mm)
                if frame_idx == 0 and mask_bool is not None:
                    imageio.imwrite(os.path.join(mask_dir, f"{frame_idx:06d}.png"), (mask_bool.astype(np.uint8) * 255))

            # Convert to top-left origin for FP
            rgb_fp = np.ascontiguousarray(np.flipud(rgb))
            depth_fp = np.ascontiguousarray(np.flipud(depth_m))
            mask_fp = np.ascontiguousarray(np.flipud(mask_bool)) if (mask_bool is not None) else None

            # Run FoundationPose on the real frame to estimate object pose in camera
            use_fp = (est is not None)
            W_T_O_est = None
            if use_fp:
                try:
                    if frame_idx == 0:
                        C_T_O = est.register(K=K, rgb=rgb_fp, depth=depth_fp, ob_mask=mask_fp, iteration=5)
                    else:
                        # Tracking: no mask required; keep K as is; images flipped already
                        C_T_O = est.track_one(rgb=rgb_fp, depth=depth_fp, K=K, iteration=int(FPOSE_CONFIG.get("track_iters", 5)))

                    W_T_C = _camera_world_transform(real_env, cam)
                    W_T_O_est = W_T_C @ C_T_O

                    # Compute GT world pose from MuJoCo for error reporting
                    if FPOSE_CONFIG.get("report_pose_deltas", False):
                        data = real_env.sim.data
                        body_id_gt = getattr(real_wrapper, "box_body_id", None)
                        if body_id_gt is None:
                            try:
                                body_id_gt = real_env.sim.model.body_name2id("block_1_main")
                            except Exception:
                                body_id_gt = None
                        if body_id_gt is not None:
                            R_w_o_true = data.body_xmat[body_id_gt].reshape(3, 3)
                            t_w_o_true = data.body_xpos[body_id_gt]
                            W_T_O_true = np.eye(4, dtype=np.float64)
                            W_T_O_true[:3, :3] = R_w_o_true
                            W_T_O_true[:3, 3] = t_w_o_true
                            D = np.linalg.inv(W_T_O_true) @ W_T_O_est
                            t_err = float(np.linalg.norm(D[:3, 3]))
                            R_D = D[:3, :3]
                            c = float(np.clip((np.trace(R_D) - 1.0) * 0.5, -1.0, 1.0))
                            r_err_deg = float(np.degrees(np.arccos(c)))
                            print(f"[FP] frame {frame_idx:06d} | pose error: trans={t_err:.4f} m, rot={r_err_deg:.2f} deg")

                    # Overlays (pred vs GT) for diagnostics
                    if FPOSE_CONFIG.get("save_debug", True) and (draw_posed_3d_box is not None) and (draw_xyz_axis is not None):
                        try:
                            ov_root = os.path.join(out_root, "fp_debug")
                            ov_est_dir = os.path.join(ov_root, "overlay_pred")
                            ov_gt_dir = os.path.join(ov_root, "overlay_gt")
                            ov_comp_dir = os.path.join(ov_root, "overlay_comp")
                            for d in (ov_est_dir, ov_gt_dir, ov_comp_dir):
                                os.makedirs(d, exist_ok=True)

                            # Load mesh extents for 3D box
                            if trimesh is not None and mesh_path_for_fp and os.path.exists(mesh_path_for_fp):
                                m = trimesh.load(mesh_path_for_fp, force='mesh')
                                ext_dbg = np.asarray(m.extents, dtype=np.float32)
                            else:
                                # fallback: read extents from MuJoCo
                                ext_dbg = _get_box_extents_for_body(real_env, obj_name).astype(np.float32)
                            bbox_dbg = np.stack([-ext_dbg / 2.0, ext_dbg / 2.0], axis=0).reshape(2, 3)

                            # Pred overlay (camera frame image coords use flipped rgb)
                            est_img = draw_posed_3d_box(K, img=rgb_fp.copy(), ob_in_cam=C_T_O, bbox=bbox_dbg)
                            est_img = draw_xyz_axis(est_img, ob_in_cam=C_T_O, scale=float(max(ext_dbg)) * 0.5, K=K, thickness=2, transparency=0, is_input_rgb=True)

                            # GT overlay from MuJoCo
                            data = real_env.sim.data
                            # get body id for GT 'block_1_main'
                            try:
                                body_id = real_env.sim.model.body_name2id("block_1_main")
                            except Exception:
                                body_id = None
                            if body_id is not None:
                                R_w_o_true = data.body_xmat[body_id].reshape(3, 3)
                                t_w_o_true = data.body_xpos[body_id]
                                W_T_O_true = np.eye(4, dtype=np.float64)
                                W_T_O_true[:3, :3] = R_w_o_true
                                W_T_O_true[:3, 3] = t_w_o_true
                                C_T_O_true = np.linalg.inv(W_T_C) @ W_T_O_true
                                gt_img = draw_posed_3d_box(K, img=rgb_fp.copy(), ob_in_cam=C_T_O_true, bbox=bbox_dbg)
                                gt_img = draw_xyz_axis(gt_img, ob_in_cam=C_T_O_true, scale=float(max(ext_dbg)) * 0.5, K=K, thickness=2, transparency=0, is_input_rgb=True)
                            else:
                                gt_img = rgb_fp.copy()

                            comp = np.concatenate([est_img, gt_img], axis=1)
                            import imageio.v2 as imageio
                            imageio.imwrite(os.path.join(ov_est_dir, f"{frame_idx:06d}.png"), est_img)
                            imageio.imwrite(os.path.join(ov_gt_dir, f"{frame_idx:06d}.png"), gt_img)
                            imageio.imwrite(os.path.join(ov_comp_dir, f"{frame_idx:06d}.png"), comp)
                        except Exception as e_ov:
                            print(f"[FP] overlay failed @ frame {frame_idx:06d}: {e_ov}")
                except Exception as e_reg:
                    print(f"[FP] register/track failed @ frame {frame_idx:06d}: {e_reg}")
                    W_T_O_est = None

            # Render a small preview for OpenCV window (optional)
            if show:
                frame = real_env.sim.render(camera_name=cam, width=320, height=240)
                frame = np.flipud(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Simulation", frame)
                # 1 ms bekle, ESC (27) basılırsa çık
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # --- Pose for inner worlds: toggle between FP estimate and MuJoCo GT ---
            if USE_FOUNDATIONPOSE_FOR_INNER and (W_T_O_est is not None):
                pos = W_T_O_est[:3, 3]
                # Convert rotation to [w,x,y,z] quaternion
                if T is not None:
                    quat_wxyz = T.quaternion_from_matrix(W_T_O_est)
                else:
                    # quick fallback via direct 3x3 -> quat
                    Rm = W_T_O_est[:3, :3]
                    qw = np.sqrt(1.0 + np.trace(Rm)) / 2.0
                    qx = (Rm[2, 1] - Rm[1, 2]) / (4.0 * qw)
                    qy = (Rm[0, 2] - Rm[2, 0]) / (4.0 * qw)
                    qz = (Rm[1, 0] - Rm[0, 1]) / (4.0 * qw)
                    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
                pose = np.concatenate([pos, quat_wxyz])
            else:
                # Fallback: use MuJoCo pose directly
                box_pos = real_env.sim.data.body_xpos[real_wrapper.box_body_id]
                box_quat = real_env.sim.data.body_xquat[real_wrapper.box_body_id]
                pose = np.concatenate([box_pos, box_quat])
            
            # 1) (Opsiyonel) Robot durumunu gerçek dünyadan iç dünyaya senkronize et
            if sync_robot:
                inner_wrapper.sync_robot_from_real(
                    real_env,
                    include_vel=sync_robot_vel,
                    include_gripper=sync_gripper,
                )

            # 2) Kutu durumunu (görüşten veya gerçek dünyadan) iç dünyaya uygula
            inner_wrapper.update_inner_from_vision(pose)

            u = mppi.compute_control()
            box_pos_t = real_wrapper.step(80.0 * u)
            cost = real_wrapper.state_cost()

            print(f"Step {t:03d} | box_pos={box_pos_t.round(3)} | cost={cost:.3f}")
            frame_idx += 1

        real_env.close()
        mppi.close()

    except KeyboardInterrupt:
        print("⛔ Çalışma kullanıcı tarafından durduruldu.")
    finally:
        cv2.destroyAllWindows()
        real_env.close()
        mppi.close()


if __name__ == "__main__":
    TASK_NAME = "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side"
    vision_mppi_visual_fast(with_size=False, show=True, init_idx=0, problem_folder="my_suite", task_name=TASK_NAME)

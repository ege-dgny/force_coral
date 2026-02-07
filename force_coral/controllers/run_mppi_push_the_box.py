import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MUJOCO_GL"] = "egl"   # glfw yerine egl: headless + daha hızlı
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"  # GPU 0

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
import types
import textwrap
import inspect
import math
import sys
import mujoco

# Bootstrap force_coral (sets up sys.path for LIBERO + registers extensions)
import force_coral

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from force_coral.libero_ext.env_wrapper import SegmentationRenderEnv
from force_coral.libero_ext.init_loader import load_init_bundle_by_name
from robosuite import load_controller_config
from dotenv import load_dotenv


import re, io, base64, json
import openai
from PIL import Image

from scipy.spatial.transform import Rotation as R
import numpy as np

load_dotenv()
client = openai.OpenAI()  # reads OPENAI_API_KEY from environment

#goal_quat = [0.7071068, 0, 0.7071068, 0]   # senin verdiğin örnek
#scipy_goal_quat = [goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]]
#desired_orientation = R.from_quat(goal_quat)

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
):
    """
    Build an environment strictly from a saved initialization bundle.
    Requires init_idx; raises if not provided.
    """
    if init_idx is None:
        raise ValueError("init_idx must be provided for build_inner_env (no fallback path)")

    # Use canonical BDDL path under LIBERO's bddl_files/<problem_folder>/
    bddl_file = _canonical_bddl(problem_folder, task_name)

    # Load overrides + init state, then construct env and set state
    overrides, state = load_init_bundle_by_name(
        problem_folder=problem_folder,
        task_name=task_name,
        init_idx=init_idx,
    )



    env = SegmentationRenderEnv(
        bddl_file_name=bddl_file,
        robots=["Panda"],
        controller=controller,
        has_renderer=gui,
        has_offscreen_renderer=offscreen,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        camera_names=["frontview"],
        camera_heights=240,
        camera_widths=320,
        camera_depths=False,
        camera_segmentations="instance",
        **({"object_overrides": overrides} if overrides else {}),
    )
    env.robots[0].controller_config["control_ori"] = True
    env.seed(0)
    env.reset()
    env.set_init_state(state)
    return env

class SimpleWrapper:
    def __init__(self, env):
        self.env = env
        self.box_body_name = "block_1_main"
        self.box_body_id = env.sim.model.body_name2id(self.box_body_name)
        self.panda_eef_name = "gripper0_grip_site"
        geom_id = env.sim.model.body_geomadr[self.box_body_id]
        half_size = env.sim.model.geom_size[geom_id]
        self.half_extents = np.array(half_size)

    def reset(self, modify_scene=False):
        self.env.reset()

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
    _worker_wrapper.reset(modify_scene=False)

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

class ParallelMPPI:
    def __init__(self, env_wrapper,
                 horizon=10, num_samples=64, noise_scale=1.0,
                 controller="OSC_POSE", control_freq=20,
                 num_workers=None, seed=0,
                 init_idx=None, problem_folder="my_suite", task_name=None):
        """
        env_wrapper: gerçek (real) env'i saran senin SimpleWrapper'ının örneği.
        task_name: worker env'lerin kuracağı görev adı (BDDL ismi ile aynı).
        """
        self.envw = env_wrapper
        self.horizon = horizon
        self.num_samples = num_samples
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.scale7 = 8.0  # sen action'u step'te 8.0 ile çarpıyorsun; noise'u küçük tut
        atexit.register(self.close)

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(controller, control_freq, init_idx, problem_folder, task_name),
        )

    def close(self):
        self.pool.terminate()
        self.pool.join()

    def compute_control(self):
        """
        Mevcut env durumundan başlayıp num_samples rollout'u paralel değerlendirir.
        En iyi birinci adım aksiyonunu döndürür (3D).
        """
        # Mevcut state'sini kopyala (tek seferde)
        initial_state = self.envw.env.sim.get_state()

        # NumPy ile tüm örneklemeleri hazırla (H,3) shape; küçük noise koy
        # Not: step() içinde 8.0 ile çarpılıyor; burada noise_scale'i düşük tut
        actions = self.rng.uniform(-1, 1, size=(self.num_samples, self.horizon, 6)) \
                * (self.noise_scale / self.scale7)

        # Paralel değerlendir
        tasks = [(actions[i], initial_state) for i in range(self.num_samples)]
        costs = self.pool.map(_evaluate_one, tasks)

        best_idx = int(np.argmin(costs))
        best_first_action = actions[best_idx, 0]
        return best_first_action


def print_box_and_wall_info(env, box_body_name="block_1_main", wall_body_name="wall2_1_main"):
    # Box bilgileri
    box_id = env.sim.model.body_name2id(box_body_name)
    box_pos = env.sim.data.body_xpos[box_id]
    box_quat = env.sim.data.body_xquat[box_id]
    box_geom_id = env.sim.model.body_geomadr[box_id]
    box_size = env.sim.model.geom_size[box_geom_id]  # half extents

    # Wall bilgileri
    wall_id = env.sim.model.body_name2id(wall_body_name)
    wall_pos = env.sim.data.body_xpos[wall_id]       # DÜZELTME: body_xpos
    wall_quat = env.sim.data.body_xquat[wall_id]
    wall_geom_id = env.sim.model.body_geomadr[wall_id]
    wall_size = env.sim.model.geom_size[wall_geom_id]

    print("\n--- Environment Info ---")
    print(f"Box Pose: pos={box_pos.round(3)}, quat={box_quat.round(3)}")
    print(f"Box Size (half extents): {box_size.round(3)}")
    print(f"Wall Pose: pos={wall_pos.round(3)}, quat={wall_quat.round(3)}")
    print(f"Wall Size (half extents): {wall_size.round(3)}")
    print("------------------------\n")


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
    # Real environment (offscreen render, GUI yok)
    real_env = build_inner_env(
        task_name=task_name,
        offscreen=True,
        gui=False,
        init_idx=init_idx,
        problem_folder=problem_folder,
    )
    real_wrapper = SimpleWrapper(real_env)

    #real_wrapper.reset(modify_scene=False)
    #inner_wrapper.reset(modify_scene=False)

    print_box_and_wall_info(real_env)

    try:
        for t in range(200):


            # --- Render frame (küçük çözünürlük) ---
            frame = real_env.sim.render(camera_name="frontview", width=320, height=240)
            frame = np.flipud(frame)  # mujoco output ters
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV için BGR

            if show:
                cv2.imshow("Simulation", frame)
                # 1 ms bekle, ESC (27) basılırsa çık
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # --- Robot ve kutu bilgisi ---
            #robot_body_id = real_env.sim.model.body_name2id("robot0_base")
            #robot_pos = real_env.sim.data.body_xpos[robot_body_id]
            #robot_quat = real_env.sim.data.body_xquat[robot_body_id]
            #robot_pose = np.concatenate([robot_pos, robot_quat]).tolist()

            # --- Burası artık FP den gelecek ---
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


            # Current box orientation (quat → Euler)
            box_quat = real_env.sim.data.body_xquat[real_wrapper.box_body_id]
            current_rot = R.from_quat(box_quat)
            current_euler = current_rot.as_euler("xyz", degrees=True)

            # Euler unwrap: ani -179/+179 sıçramalarını düzelt
            #current_euler_unwrapped = np.unwrap(current_euler, discont=180, axis=0)

            # Desired orientation
            #target_euler = desired_orientation.as_euler("xyz", degrees=True)
            #target_euler_unwrapped = np.unwrap(target_euler, discont=180, axis=0)

            #print(f"Step {t:03d} | box_pos={box_pos_t.round(3)} | cost={cost:.3f}")
            #print(f"   Current box orientation (Euler xyz, deg): {current_euler_unwrapped.round(1)}")
            #print(f"   Current box quaternion: {box_quat.round(3)}")
            #print(f"   Goal orientation (Euler xyz, deg):       {target_euler_unwrapped.round(1)}")
            #print(f"   Goal orientation quaternion:             {desired_orientation.as_quat().round(3)}")

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

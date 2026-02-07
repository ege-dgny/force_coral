#!/usr/bin/env python3
"""
Quick utility to load a LIBERO task and save a screenshot.

Two ways to resolve the task:
1) Suite-based: select a suite + task_id
2) Name-based: select problem_folder + task_name

Optionally, you can load a specific initialization bundle (overrides + state)
to visualize exactly what evaluation will run.

Edit CONFIG below in your editor, or pass minimal CLI overrides.
"""

import argparse
import os
from typing import Any, Dict

import numpy as np
import imageio.v2 as imageio

import force_coral
from libero.libero import get_libero_path, benchmark
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
from force_coral.libero_ext.init_loader import (
    make_env_for_init,
    make_env_for_task_name,
)


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG: Dict[str, Any] = {
    # Choose task resolution mode: "suite" or "name"
    "mode": "suite",

    # Suite-based
    "suite": "my_suite",
    "task_id": 2,

    # Cameras
    "camera_names": ["frontview"],
    "camera_height": 1024,
    "camera_width": 1024,
    # If True, vertically flip images before saving (was 180° rotate)
    "rotate_180": True,

    # Output
    "out_dir": "renders/see_task",

    # Optional: load a specific init bundle (overrides + state)
    "use_init_bundle": True,
    "init_idx": 0,
}
# ----------------re--------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def suite_bddl_path(suite, task_id: int) -> str:
    return suite.get_task_bddl_file_path(task_id)


def name_bddl_path(problem_folder: str, task_name: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), problem_folder, f"{task_name}.bddl")


def save_obs_images(obs: Dict[str, Any], out_dir: str, cameras, rotate_180: bool):
    ensure_dir(out_dir)
    for cam in cameras:
        key = f"{cam}_image"
        if key not in obs:
            continue
        img = obs[key]
        if rotate_180:
            # Apply vertical flip instead of 180° rotation
            img = np.flipud(img).copy()
        imageio.imwrite(os.path.join(out_dir, f"{cam}.png"), img)


def _quat_xyzw_to_rot(q):
    """Return a 3x3 rotation matrix from an XYZW quaternion."""
    q = np.asarray(q, dtype=float)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must be length-4 [x,y,z,w]")
    x, y, z, w = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def _rot_to_euler_rpy_deg(R: np.ndarray):
    """Return roll, pitch, yaw in degrees from a 3x3 rotation matrix.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ZYX yaw-pitch-roll),
    returned as (roll_x, pitch_y, yaw_z) in degrees.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # pitch from -asin(R[2,0]) with clamping for numerical safety
    sp = -R[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)
    eps = 1e-8
    if abs(cp) > eps:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        # Gimbal lock: pitch ~= +/-90 deg
        # Set yaw = 0, compute roll from alternative terms.
        yaw = 0.0
        roll = np.arctan2(-R[1, 2], R[1, 1])
    return np.degrees([roll, pitch, yaw])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default=CONFIG.get("mode", "suite"))
    parser.add_argument("--suite", default=CONFIG.get("suite", None))
    parser.add_argument("--task_id", type=int, default=CONFIG.get("task_id", 0))
    parser.add_argument("--problem_folder", default=CONFIG.get("problem_folder", None))
    parser.add_argument("--task_name", default=CONFIG.get("task_name", None))
    parser.add_argument("--out_dir", default=CONFIG.get("out_dir", "renders/see_task"))
    parser.add_argument("--init_idx", type=int, default=CONFIG.get("init_idx", 0))
    parser.add_argument("--use_init_bundle", type=lambda x: str(x).lower() == "true", default=CONFIG.get("use_init_bundle", False))
    args, _ = parser.parse_known_args()

    # Materialize effective cfg
    cfg = dict(CONFIG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    base_env_args = {
        "camera_names": cfg["camera_names"],
        "camera_heights": cfg["camera_height"],
        "camera_widths": cfg["camera_width"],
        "camera_depths": False,
        "robots": ["Panda"],
        "controller": "OSC_POSE",
    }

    # Resolve and create env
    mode = cfg["mode"].lower()
    if cfg["use_init_bundle"]:
        if mode == "suite":
            suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
            suite = suite_cls()
            env, obs = make_env_for_init(suite, int(cfg["task_id"]), int(cfg["init_idx"]), base_env_args)
            task = suite.get_task(int(cfg["task_id"]))
            out_dir = os.path.join(cfg["out_dir"], cfg["suite"], task.name, f"init_{cfg['init_idx']}")
        else:
            env, obs = make_env_for_task_name(cfg["problem_folder"], cfg["task_name"], int(cfg["init_idx"]), base_env_args)
            out_dir = os.path.join(cfg["out_dir"], cfg["problem_folder"], cfg["task_name"], f"init_{cfg['init_idx']}")
    else:
        # Plain env reset (no specific init state)
        if mode == "suite":
            suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
            suite = suite_cls()
            bddl = suite_bddl_path(suite, int(cfg["task_id"]))
            task = suite.get_task(int(cfg["task_id"]))
            env = OffScreenRenderEnv(bddl_file_name=bddl, **base_env_args)
            obs = env.reset()
            out_dir = os.path.join(cfg["out_dir"], cfg["suite"], task.name)
        else:
            bddl = name_bddl_path(cfg["problem_folder"], cfg["task_name"])
            env = OffScreenRenderEnv(bddl_file_name=bddl, **base_env_args)
            obs = env.reset()
            out_dir = os.path.join(cfg["out_dir"], cfg["problem_folder"], cfg["task_name"])

    # Save images
    save_obs_images(obs, out_dir, cfg["camera_names"], cfg["rotate_180"])
    print(f"Saved screenshots to: {out_dir}")

    # If object 'block_1' exists, print its world pose
    try:
        # Underlying env keeps a mapping from object name -> MuJoCo body id
        body_map = getattr(env.env, "obj_body_id", {}) if hasattr(env, "env") else {}
        if "block_1" in body_map:
            body_id = body_map["block_1"]
            pos = np.array(env.sim.data.body_xpos[body_id]).copy()
            rot_flat = np.array(env.sim.data.body_xmat[body_id]).copy()
            rot = rot_flat.reshape(3, 3)
            rpy_deg = _rot_to_euler_rpy_deg(rot)
            print("block_1 position (x, y, z):", pos)
            print("block_1 rotation (deg) [roll_x, pitch_y, yaw_z]:", rpy_deg)
        else:
            # Fallback: try observation keys if available
            pos_key, quat_key = "block_1_pos", "block_1_quat"
            if pos_key in obs and quat_key in obs:
                pos = np.array(obs[pos_key])
                rot = _quat_xyzw_to_rot(obs[quat_key])
                rpy_deg = _rot_to_euler_rpy_deg(rot)
                print("block_1 position (x, y, z):", pos)
                print("block_1 rotation (deg) [roll_x, pitch_y, yaw_z]:", rpy_deg)
            else:
                print("Object 'block_1' not found in this task.")
    except Exception as e:
        print(f"Warning: failed to query 'block_1' pose: {e}")
    env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Move the end effector by a delta (dx, dy, dz) relative to its reset pose,
then print final joint qpos and EEF pose.

Uses your current my_tabletop_manipulation environment (hard-coded MyMountedPanda).
It drives the robot with OSC_POSE deltas until within tolerance of the target,
then prints:
- Final EEF pos / quat
- Arm joint positions (qpos)
"""

import argparse
import os
from typing import Tuple, List

import numpy as np

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
import imageio.v2 as imageio

# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG = {
    "task": "push_the_card_to_the_edge_of_the_table_and_pick_the_card",
    # Desired delta (dx, dy, dz) in meters, world frame
    "dx": 0.0,
    "dy": 0.30,
    "dz": 0.0,
    # Control loop params
    "max_steps": 500,
    "tol": 0.00005,
    "per_step_max": 0.3,
    # Video recording (frontview)
    "record_mp4": True,
    "video_dir": "renders/move_eef_to_xyz_get_qpos",
    "video_fps": 20,
    "video_res": 256,
}
# ------------------------------------------------------


def bddl_path(task: str) -> str:
    return os.path.join(
        force_coral.get_data_path("bddl_files"),
        "my_suite",
        f"{task}.bddl",
    )


def parse_args():
    """Optional CLI overrides; defaults come from CONFIG so running in VS Code just works."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", default=CONFIG["task"], help="Task name in my_suite")
    p.add_argument("--dx", type=float, default=CONFIG["dx"], help="Delta x (m), world frame")
    p.add_argument("--dy", type=float, default=CONFIG["dy"], help="Delta y (m), world frame")
    p.add_argument("--dz", type=float, default=CONFIG["dz"], help="Delta z (m), world frame")
    p.add_argument("--max-steps", type=int, default=CONFIG["max_steps"], help="Max control steps")
    p.add_argument("--tol", type=float, default=CONFIG["tol"], help="Position tolerance (m)")
    p.add_argument("--per-step-max", type=float, default=CONFIG["per_step_max"],
                   help="Clamp per-step positional delta (m)")
    p.add_argument("--record-mp4", type=lambda x: str(x).lower()=="true", default=CONFIG["record_mp4"], help="Record frontview MP4")
    p.add_argument("--video-dir", default=CONFIG["video_dir"], help="Output directory for MP4")
    p.add_argument("--video-fps", type=int, default=CONFIG["video_fps"], help="MP4 frames per second")
    p.add_argument("--video-res", type=int, default=CONFIG["video_res"], help="Square resolution for MP4")
    args, _ = p.parse_known_args()
    return args


def clamp_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-9:
        return v
    if n > max_norm:
        return v * (max_norm / n)
    return v


def get_arm_qpos_from_obs_or_sim(env, obs) -> List[float]:
    # Preferred: observation key if present
    if isinstance(obs, dict):
        for k in ("robot0_joint_pos", "robot0_joint_positions"):
            if k in obs:
                arr = np.array(obs[k], dtype=float).tolist()
                # Many setups expose arm+gripper; keep first 7 for Panda arm
                return arr[:7]
    # Fallback: read from sim using known ref indices if available
    try:
        idx = env.env.robots[0]._ref_joint_pos_indexes  # type: ignore[attr-defined]
        qpos = env.env.sim.data.qpos[idx]
        return np.array(qpos, dtype=float).tolist()[:7]
    except Exception:
        pass
    # Last resort: return empty list
    return []


def main():
    args = parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path(args.task),
        robots=["Panda"],            # my_tabletop_manipulation hard-codes MyMountedPanda
        controller="OSC_POSE",
        camera_names=["agentview"],  # minimal rendering footprint
        camera_heights=64,
        camera_widths=64,
        camera_depths=False,
    )
    env.seed(0)
    obs = env.reset()

    # Determine initial EEF position and construct the target by applying delta
    init_pos = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
    delta = np.array([args.dx, args.dy, args.dz], dtype=float)
    target = init_pos + delta

    # Print initial joint configuration (arm qpos)
    init_qpos = get_arm_qpos_from_obs_or_sim(env, obs)
    print("Initial arm qpos (7):", init_qpos)

    # Optional video recorder (frontview)
    writer = None
    if args.record_mp4:
        os.makedirs(args.video_dir, exist_ok=True)
        mp4_path = os.path.join(
            args.video_dir,
            f"{args.task}_dx{args.dx:+.3f}_dy{args.dy:+.3f}_dz{args.dz:+.3f}.mp4",
        )
        try:
            writer = imageio.get_writer(mp4_path, fps=int(args.video_fps), codec="libx264")
            frame0 = env.sim.render(camera_name="frontview", height=int(args.video_res), width=int(args.video_res))
            frame0 = np.rot90(frame0, 2).copy()
            writer.append_data(frame0)
        except Exception:
            writer = None

    # Introspect action dim
    try:
        action_dim = int(env.env.action_dim)
    except Exception:
        action_dim = 7

    # Drive towards target
    reached = False
    for step in range(args.max_steps):
        cur = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
        dpos = target - cur
        if np.linalg.norm(dpos) <= args.tol:
            reached = True
            break
        # Clamp per-step delta and form action
        dpos = clamp_vec(dpos, args.per_step_max)
        a = np.zeros(action_dim, dtype=float)
        a[:3] = dpos
        if action_dim >= 7:
            a[3:6] = 0.0  # no rotation change
            a[6] = 0.0    # gripper still
        obs, _, _, _ = env.step(a)

        # Append frame to video
        if writer is not None:
            try:
                f = env.sim.render(camera_name="frontview", height=int(args.video_res), width=int(args.video_res))
                f = np.rot90(f, 2).copy()
                writer.append_data(f)
            except Exception:
                pass

    # One or two settle steps
    zero = np.zeros(action_dim, dtype=float)
    for _ in range(2):
        obs, _, _, _ = env.step(zero)
        if writer is not None:
            try:
                f = env.sim.render(camera_name="frontview", height=int(args.video_res), width=int(args.video_res))
                f = np.rot90(f, 2).copy()
                writer.append_data(f)
            except Exception:
                pass

    final_pos = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
    final_quat = np.array(obs.get("robot0_eef_quat", [1, 0, 0, 0]), dtype=float)
    arm_qpos = get_arm_qpos_from_obs_or_sim(env, obs)

    print("Init EEF position (x, y, z):", init_pos.tolist())
    print("Requested delta (dx, dy, dz):", delta.tolist())
    print("Target EEF position (x, y, z):", target.tolist())
    print("Reached:", bool(reached))
    print("Final EEF position (x, y, z):", final_pos.tolist())
    print("Final EEF orientation (w, x, y, z):", final_quat.tolist())
    print("Arm joint qpos (7):", arm_qpos)

    # Finalize video
    if writer is not None:
        try:
            writer.close()
            print("Saved MP4:", mp4_path)
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Flip the box by applying external torque / force to visualize the OnSide predicate.

This script loads the "flip_the_blue_box_onto_its_side" task and applies a
brief lateral torque to tip the block onto its side, then lets it settle.

Outputs a few screenshots to verify the behavior.
"""

import os
from typing import Dict, Any
import numpy as np
import imageio.v2 as imageio

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG: Dict[str, Any] = {
    "problem_folder": "my_suite",
    "task_name": "flip_the_blue_box_onto_its_side",

    # Camera settings
    "camera_names": ["frontview"],
    "camera_height": 256,
    "camera_width": 256,
    "rotate_180": True,

    # Force / torque application
    # Torque impulse: apply a pure tipping moment about +y for a few steps
    "apply_mode": "moment",      # "moment" (torque only) or "force"
    "moment_axis": [0.0, 1.0, 0.0],
    "moment_magnitude": 1.0,     # N·m (short impulse)

    # Alternative force mode (linear force + optional downward bias)
    # If you switch apply_mode to "force", these fields are used instead:
    "force_xy_dir": [1.0, 0.0, 0.0],  # horizontal push direction
    "force_xy_mag": 8.0,              # N
    "force_down_mag": 10.0,           # N (downward only)
    # Optional: emulate applying force at a point above COM by adding the equivalent torque
    # tau = r x F, with r = [0, 0, height]
    "force_offset_height": 0.05, # meters; set 0.0 to disable additional torque
    "apply_steps": 8,            # number of steps to apply external input (impulse)
    "settle_before": 20,         # settle steps before actuation
    "settle_after": 50,          # settle steps after actuation

    # Output
    "out_dir": "renders/flip_box_demo",
    "save_video": True,
    "video_camera": "frontview",  # which camera stream to record
    "video_fps": 20,
}
# ------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def bddl_path(problem_folder: str, task_name: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), problem_folder, f"{task_name}.bddl")


def save_obs_images(obs: Dict[str, Any], out_dir: str, idx: int, cameras, rotate_180: bool):
    ensure_dir(out_dir)
    for cam in cameras:
        key = f"{cam}_image"
        if key not in obs:
            continue
        img = obs[key]
        if rotate_180:
            img = np.rot90(img, 2).copy()
        imageio.imwrite(os.path.join(out_dir, f"{idx:03d}_{cam}.png"), img)


def get_video_frame(obs: Dict[str, Any], cam: str, rotate_180: bool):
    key = f"{cam}_image"
    if key not in obs:
        return None
    img = obs[key]
    if rotate_180:
        img = np.rot90(img, 2).copy()
    return img


def main():
    cfg = CONFIG

    # Build env
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path(cfg["problem_folder"], cfg["task_name"]),
        camera_names=cfg["camera_names"],
        camera_heights=cfg["camera_height"],
        camera_widths=cfg["camera_width"],
        camera_depths=False,
        robots=["Panda"],
        controller="OSC_POSE",
    )
    env.seed(0)
    obs = env.reset()

    # Prepare output dir
    out_dir = os.path.join(cfg["out_dir"], cfg["problem_folder"], cfg["task_name"])
    ensure_dir(out_dir)
    save_obs_images(obs, out_dir, 0, cfg["camera_names"], cfg["rotate_180"])

    # Optional video recording setup
    frames = []
    if cfg.get("save_video", False):
        vf = get_video_frame(obs, cfg["video_camera"], cfg["rotate_180"])
        if vf is not None:
            frames.append(vf)

    # Get body id for the box
    block_body_id = env.env.obj_body_id.get("block_1")
    assert block_body_id is not None, "block_1 body id not found"

    # Helper: apply external input this step
    def apply_external():
        mode = cfg["apply_mode"].lower()
        if mode == "moment":
            axis = np.array(cfg["moment_axis"], dtype=float)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            tau = axis * float(cfg["moment_magnitude"])
            env.env.sim.data.xfrc_applied[block_body_id, 0:3] = tau
            # no net linear force -> avoids launching
        elif mode == "force":
            # Compose force with independent horizontal and downward components
            dir_xy = np.array(cfg["force_xy_dir"], dtype=float)
            dir_xy[2] = 0.0  # ensure purely horizontal
            if np.linalg.norm(dir_xy) < 1e-8:
                dir_xy = np.array([1.0, 0.0, 0.0])
            dir_xy = dir_xy / np.linalg.norm(dir_xy)
            F_xy = dir_xy * float(cfg["force_xy_mag"])
            F_down = np.array([0.0, 0.0, -abs(float(cfg["force_down_mag"]))])
            F = F_xy + F_down
            env.env.sim.data.xfrc_applied[block_body_id, 3:6] = F
            # Add equivalent torque for off-COM application
            h = float(cfg.get("force_offset_height", 0.0))
            if h != 0.0:
                # r = [0, 0, h]; tau = r x F = [-h*Fy, h*Fx, 0]
                tau = np.array([-h * F[1], h * F[0], 0.0], dtype=float)
                env.env.sim.data.xfrc_applied[block_body_id, 0:3] += tau
        else:
            raise ValueError(f"Unknown apply_mode: {mode}")

    # Dummy robot action (zeros)
    action_dim = env.env.action_dim
    dummy = np.zeros(action_dim, dtype=float)

    # Settle
    for i in range(1, cfg["settle_before"] + 1):
        obs, _, _, _ = env.step(dummy)
        if i % 5 == 0:
            save_obs_images(obs, out_dir, i, cfg["camera_names"], cfg["rotate_180"])
        if cfg.get("save_video", False):
            vf = get_video_frame(obs, cfg["video_camera"], cfg["rotate_180"])
            if vf is not None:
                frames.append(vf)

    # Apply external input to tip the block
    for i in range(cfg["apply_steps"]):
        # Set the external torque / force for this step
        apply_external()
        obs, _, _, _ = env.step(dummy)
        # Clear external input for next step
        env.env.sim.data.xfrc_applied[block_body_id, :] = 0.0
        save_obs_images(obs, out_dir, cfg["settle_before"] + 1 + i, cfg["camera_names"], cfg["rotate_180"])
        if cfg.get("save_video", False):
            vf = get_video_frame(obs, cfg["video_camera"], cfg["rotate_180"])
            if vf is not None:
                frames.append(vf)

    # Let it settle and check success
    success_seen = False
    for i in range(cfg["settle_after"]):
        obs, _, done, _ = env.step(dummy)
        # done flags success per your debounce logic
        success_seen = success_seen or bool(done)
        idx = cfg["settle_before"] + 1 + cfg["apply_steps"] + i
        if i % 2 == 0:
            save_obs_images(obs, out_dir, idx, cfg["camera_names"], cfg["rotate_180"])
        if cfg.get("save_video", False):
            vf = get_video_frame(obs, cfg["video_camera"], cfg["rotate_180"])
            if vf is not None:
                frames.append(vf)

    print(f"Flip success (with hold logic): {success_seen}")
    # Save MP4 video if requested
    if cfg.get("save_video", False):
        if len(frames) == 0:
            print("[warn] no frames collected for video; skipping save")
        else:
            # Ensure uint8 frames
            frames_uint8 = []
            for f in frames:
                if f.dtype != np.uint8:
                    f = np.clip(f, 0, 255).astype(np.uint8)
                frames_uint8.append(f)
            video_path = os.path.join(out_dir, f"flip_demo_{cfg['video_camera']}.mp4")
            try:
                imageio.mimwrite(video_path, frames_uint8, fps=int(cfg["video_fps"]))
                print(f"Saved video: {video_path}")
            except Exception as e:
                print(f"[warn] could not save MP4 (ffmpeg missing?): {e}")
                # Fallback to GIF
                gif_path = os.path.splitext(video_path)[0] + ".gif"
                try:
                    imageio.mimsave(gif_path, frames_uint8, duration=1.0/float(cfg["video_fps"]))
                    print(f"Saved GIF fallback: {gif_path}")
                except Exception as e2:
                    print(f"[error] could not save GIF fallback either: {e2}")
    # Always report where images were written
    print(f"Images saved under: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()

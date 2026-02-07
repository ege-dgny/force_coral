#!/usr/bin/env python3
"""
Synthetic test for the new "flip with wall support" predicate.

What it does:
- Loads the my_suite task: push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side
- Applies external forces / torques directly to the block to push it into the wall
  and then tip it onto its side.
- Records a video from the specified camera.
- Prints when support gets latched and when success is achieved (done=True).

Notes:
- All timing is in control steps (env.step quanta). The predicate uses step-based
  windows and latching, so it is robust to the success-hold requirement.
"""

import os
import argparse
from typing import Tuple

import numpy as np
import imageio.v2 as imageio

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
from force_coral.libero_ext.init_loader import make_env_for_task_name


TASK = "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side"


def bddl_path(task: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), "my_suite", f"{task}.bddl")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# Centralized configuration defaults. Can be overridden via CLI flags.
DEFAULT_CONFIG = {
    "task": TASK,
    "out": "renders/videos/flip_with_wall_synthetic.mp4",
    "camera": "frontview",
    "height": 512,
    "width": 512,
    "fps": 20,
    "steps": 200,
    "support_pre_steps": 20,
    "support_post_steps": 10,
    # Set to an integer to start from a stored init state
    "init_idx": 0,
}


def force_schedule(t: int, hz: int = 20) -> Tuple[float, float, float, float, float, float]:
    """
    Simple piecewise force / torque schedule to:
    1) push block +y toward the wall
    2) add a torque to tip it onto its side near/after wall contact

    Returns (fx, fy, fz, tx, ty, tz) in world coordinates, applied at CoM.

    Time windows (in control steps and seconds at the given control frequency):
    - Stage A: steps [0, 119]   -> [0.00 s, 120 / hz)    e.g., ~0.0–6.0 s at 20 Hz
    - Stage B: steps [120, 179] -> [120 / hz, 180 / hz)  e.g., ~6.0–9.0 s at 20 Hz
    - Stage C: steps [180, 239] -> [180 / hz, 240 / hz)  e.g., ~9.0–12.0 s at 20 Hz
    """
    # Default no forces/torques
    fx = fy = fz = tx = ty = tz = 0.0

    # Stage A: accelerate toward wall
    if 0 <= t < 25:  # ~0.0–6.0 s at 20 Hz
        pass
    # Stage B: keep pushing and start tipping
    elif 25 <= t < 55:  # ~6.0–9.0 s at 20 Hz
        fy = 30  # push toward +y (wall located around y ~ 0.20)
        fz = -0.01   # slight lift to reduce friction
    # Stage C: mostly torque to encourage settling on side
    elif 55 <= t < 60:  # ~9.0–12.0 s at 20 Hz
        fy = 0.08
        fz = 0.
        tx = -0 # tip around x to encourage a side flip
        ty = 0.04

    return fx, fy, fz, tx, ty, tz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--camera", type=str, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--init_idx", type=int, default=None, help="Start from saved init state index (my_suite)")
    # You can tweak these to make flipping easier or harder
    ap.add_argument("--support_pre_steps", type=int, default=None)
    ap.add_argument("--support_post_steps", type=int, default=None)
    args = ap.parse_args()

    # Build config: defaults -> CLI overrides
    cfg = DEFAULT_CONFIG.copy()
    # Apply CLI args if provided
    for k in [
        "task",
        "out",
        "camera",
        "height",
        "width",
        "fps",
        "steps",
        "init_idx",
        "support_pre_steps",
        "support_post_steps",
    ]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    ensure_dir(os.path.dirname(cfg["out"]))

    if cfg["init_idx"] is None:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path(cfg["task"]),
            camera_names=[cfg["camera"]],
            camera_heights=cfg["height"],
            camera_widths=cfg["width"],
            camera_depths=False,
            horizon=1000,
            robots=["Panda"],
            controller="OSC_POSE",
            # Step-based windows for support latching (consistent with predicate)
            support_pre_steps=cfg["support_pre_steps"],
            support_post_steps=cfg["support_post_steps"],
            # Optional: tweak block mass/size if needed
            # object_overrides={"block_1": {"size": [0.08, 0.08, 0.08], "density": 300}},
        )
        env.seed(0)
        env.reset()
    else:
        # Build env from a stored initialization bundle (my_suite)
        base_env_args = dict(
            camera_names=[cfg["camera"]],
            camera_heights=cfg["height"],
            camera_widths=cfg["width"],
            camera_depths=False,
            horizon=1000,
            robots=["Panda"],
            controller="OSC_POSE",
            support_pre_steps=cfg["support_pre_steps"],
            support_post_steps=cfg["support_post_steps"],
        )
        env, _ = make_env_for_task_name(
            problem_folder="my_suite",
            task_name=cfg["task"],
            init_idx=cfg["init_idx"],
            base_env_args=base_env_args,
        )
        print(f"Loaded initial state index: {cfg['init_idx']}")

    # Determine control frequency for time annotations
    control_hz = int(getattr(env.env, "control_freq", 20))
    print(
        f"Control frequency: {control_hz} Hz\n"
        f"Stage A: steps [0,119]   => ~0.00–{120/control_hz:.2f} s\n"
        f"Stage B: steps [120,179] => ~{120/control_hz:.2f}–{180/control_hz:.2f} s\n"
        f"Stage C: steps [180,239] => ~{180/control_hz:.2f}–{240/control_hz:.2f} s"
    )

    # Body id for external force application
    block_name = "block_1"
    wall_name = "wall2_1"
    bid = env.env.obj_body_id[block_name]

    # Action vector (hold robot still)
    zero_action = np.zeros(env.env.action_dim, dtype=float)

    writer = imageio.get_writer(cfg["out"], fps=cfg["fps"])

    support_announced = False
    success_at = None

    for t in range(cfg["steps"]):
        # Clear any previous external force
        env.env.sim.data.xfrc_applied[bid] = np.zeros(6)
        # Apply synthetic force / torque
        env.env.sim.data.xfrc_applied[bid] = np.array(force_schedule(t, control_hz), dtype=float)

        obs, reward, done, info = env.step(zero_action)

        # Grab frame and write
        key = f"{cfg['camera']}_image"
        img = obs[key]
        img_rot = np.rot90(img, 2).copy()
        writer.append_data(img_rot)

        if t % 5 == 0:
            block_pos = env.env.sim.data.body_xpos[bid]
            print(f"[t={t}] block_1 position: {block_pos}")

        # Print support latch and success for debugging
        if hasattr(env.env, "_support_granted"):
            latched = env.env._support_granted.get((block_name, wall_name), False)
            if latched and not support_announced:
                print(f"[t={t}] Support latched: block used the wall during flip")
                support_announced = True

        if done and success_at is None:
            success_at = t
            print(f"[t={t}] Success condition met (predicate True). Holding for env's debounce.")

    writer.close()

    print("Video saved to:", cfg["out"])
    if success_at is not None:
        print("First success at step:", success_at)
    else:
        print("Did not reach success within the simulated steps. Consider increasing steps or forces.")

    env.close()


if __name__ == "__main__":
    main()

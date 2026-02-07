#!/usr/bin/env python3
"""
Example: Load LIBERO task initializations (overrides + states) as if from
an external model repo, and create ready-to-step environments per init.

- Uses the Init Bundles + Loader utilities added to LIBERO
  (libero.libero.utils.init_loader).
 - Suite-based path (recommended): get suite, then make_env_for_init(...)
 - Name-based path (alternative): make_env_for_task_name(problem_folder, task_name, ...)

Edit CONFIG below in your editor, or pass minimal CLI overrides.
"""

import argparse
import os
from typing import Dict, Any

import numpy as np
import imageio.v2 as imageio

import force_coral
from libero.libero import benchmark
from force_coral.libero_ext.init_loader import (
    make_env_for_init,
    get_num_inits,
    make_env_for_task_name,
)


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG: Dict[str, Any] = {
    # Choose one of the two flows below.
    # 1) Suite-based (recommended for registered suites)
    "suite": "my_suite",
    "task_id": 0,

    # 2) Name-based (if you prefer not to instantiate a suite)
    # "problem_folder": "my_suite",
    # "task_name": "push_the_blue_box_to_the_front_of_the_wall",

    # Evaluation behavior
    "iterate_all": True,   # set False to only run a single init_idx
    "init_idx": 0,
    "num_steps": 10,

    # Env visual settings
    "camera_names": ["agentview"],
    "camera_height": 128,
    "camera_width": 128,
    # Screenshot saving
    "save_images": True,
    "save_dir": "renders/eval_loader",
    "rotate_180": True,
}
# ------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_obs_images(obs: Dict[str, Any], out_dir: str, idx: int, camera_names, rotate_180: bool):
    ensure_dir(out_dir)
    for cam in camera_names:
        key = f"{cam}_image"
        if key not in obs:
            continue
        img = obs[key]
        if rotate_180:
            img = np.rot90(img, 2).copy()
        imageio.imwrite(os.path.join(out_dir, f"{idx:02d}_{cam}.png"), img)


def run_suite_flow(cfg: Dict[str, Any]):
    bdict = benchmark.get_benchmark_dict()
    SuiteCls = bdict[cfg["suite"]]
    suite = SuiteCls()
    task_id = int(cfg["task_id"])  # ensure int

    base_env_args = {
        "camera_names": cfg["camera_names"],
        "camera_heights": cfg["camera_height"],
        "camera_widths": cfg["camera_width"],
        "camera_depths": False,
        "robots": ["Panda"],
        "controller": "OSC_POSE",
    }

    if cfg["iterate_all"]:
        n = get_num_inits(suite, task_id)
        print(f"Found {n} init states for task_id={task_id} in suite={cfg['suite']}")
        # Build base output dir using suite + task name
        task = suite.get_task(task_id)
        out_dir = os.path.join(cfg["save_dir"], cfg["suite"], task.name)
        for idx in range(n):
            env, obs = make_env_for_init(suite, task_id, idx, base_env_args)
            print(f"[init {idx}] language='{env.language_instruction}' action_dim={env.env.action_dim}")
            if cfg["save_images"]:
                save_obs_images(obs, out_dir, idx, cfg["camera_names"], cfg["rotate_180"]) 
            dummy = np.zeros(env.env.action_dim, dtype=float)
            for _ in range(cfg["num_steps"]):
                env.step(dummy)
            env.close()
    else:
        idx = int(cfg["init_idx"])  # ensure int
        env, obs = make_env_for_init(suite, task_id, idx, base_env_args)
        print(f"[init {idx}] language='{env.language_instruction}' action_dim={env.env.action_dim}")
        if cfg["save_images"]:
            task = suite.get_task(task_id)
            out_dir = os.path.join(cfg["save_dir"], cfg["suite"], task.name)
            save_obs_images(obs, out_dir, idx, cfg["camera_names"], cfg["rotate_180"]) 
        dummy = np.zeros(env.env.action_dim, dtype=float)
        for _ in range(cfg["num_steps"]):
            env.step(dummy)
        env.close()


def run_name_flow(cfg: Dict[str, Any]):
    base_env_args = {
        "camera_names": cfg["camera_names"],
        "camera_heights": cfg["camera_height"],
        "camera_widths": cfg["camera_width"],
        "camera_depths": False,
        "robots": ["Panda"],
        "controller": "OSC_POSE",
    }

    problem_folder = cfg["problem_folder"]
    task_name = cfg["task_name"]

    if cfg["iterate_all"]:
        # We don't know n directly without the suite; try increasing idx until failure
        idx = 0
        print(f"Iterating init states for {problem_folder}/{task_name} by name")
        out_dir = os.path.join(cfg["save_dir"], problem_folder, task_name)
        while True:
            try:
                env, obs = make_env_for_task_name(problem_folder, task_name, idx, base_env_args)
            except Exception as e:
                if "out of range" in str(e):
                    print(f"Reached end at init_idx={idx}")
                    break
                raise
            print(f"[init {idx}] language='{env.language_instruction}' action_dim={env.env.action_dim}")
            if cfg["save_images"]:
                save_obs_images(obs, out_dir, idx, cfg["camera_names"], cfg["rotate_180"]) 
            dummy = np.zeros(env.env.action_dim, dtype=float)
            for _ in range(cfg["num_steps"]):
                env.step(dummy)
            env.close()
            idx += 1
    else:
        idx = int(cfg["init_idx"])  # ensure int
        env, obs = make_env_for_task_name(problem_folder, task_name, idx, base_env_args)
        print(f"[init {idx}] language='{env.language_instruction}' action_dim={env.env.action_dim}")
        if cfg["save_images"]:
            out_dir = os.path.join(cfg["save_dir"], problem_folder, task_name)
            save_obs_images(obs, out_dir, idx, cfg["camera_names"], cfg["rotate_180"]) 
        dummy = np.zeros(env.env.action_dim, dtype=float)
        for _ in range(cfg["num_steps"]):
            env.step(dummy)
        env.close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", default=CONFIG.get("suite", None))
    parser.add_argument("--task_id", type=int, default=CONFIG.get("task_id", 0))
    parser.add_argument("--iterate_all", type=lambda x: str(x).lower() == "true", default=CONFIG.get("iterate_all", True))
    parser.add_argument("--init_idx", type=int, default=CONFIG.get("init_idx", 0))
    parser.add_argument("--num_steps", type=int, default=CONFIG.get("num_steps", 10))
    parser.add_argument("--problem_folder", default=CONFIG.get("problem_folder", None))
    parser.add_argument("--task_name", default=CONFIG.get("task_name", None))
    args, _ = parser.parse_known_args()

    # Materialize effective cfg
    cfg = dict(CONFIG)
    # Only override keys present in args.__dict__ that are not None
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    has_suite = cfg.get("suite") is not None
    has_name = cfg.get("problem_folder") is not None and cfg.get("task_name") is not None

    if has_suite:
        run_suite_flow(cfg)
    elif has_name:
        run_name_flow(cfg)
    else:
        raise SystemExit("Please set either suite+task_id or problem_folder+task_name.")


if __name__ == "__main__":
    main()

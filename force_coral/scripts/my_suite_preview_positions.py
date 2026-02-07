#!/usr/bin/env python3
"""
Preview manually specified placements for a my_suite task by saving screenshots.

Two ways to set variables:
1) Edit the CONFIG dict below in your editor and just run the file.
2) (Optional) Provide CLI args to override CONFIG when desired.

Edit placements in scripts/my_suite_positions.py.
This renders one image per placement to help you visually verify the setup.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import imageio.v2 as imageio

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

from force_coral.scripts.my_suite_positions import TASK_POSITIONS, Placement


# --------- EDIT THESE DEFAULTS IN YOUR EDITOR ---------
CONFIG = {
    "task": "pick_the_blue_box_and_place_it_in_the_basket",
    "out": "renders/my_suite",
    "camera": "frontview",  # e.g., agentview | frontview | galleryview
    "height": 256,
    "width": 256,
    # Optional per-object overrides applied at env construction time.
    # Keys are object instance names from BDDL (e.g., "block_1").
    # Example: set a smaller, lighter block.
    "overrides": {
        # "block_1": {"size": [0.06, 0.06, 0.06], "density": 200},
    },
}
# ------------------------------------------------------


def bddl_path(task: str) -> str:
    return os.path.join(
        force_coral.get_data_path("bddl_files"),
        "my_suite",
        f"{task}.bddl",
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_obj_pose(
    env: OffScreenRenderEnv,
    obj_name: str,
    xy: Optional[List[float]] = None,
    xyz: Optional[List[float]] = None,
    yaw: Optional[float] = None,
):
    """Set object pose from either xy (2) or xyz (3); keep unspecified parts.

    - If `xyz` is provided, it must have length 3 and sets x, y, z.
    - Else if `xy` is provided, the first two values set x, y and z is preserved.
    - If `yaw` is provided, set yaw-only orientation.
    """
    geom = env.env.object_states_dict[obj_name].get_geom_state()
    pos = np.array(geom["pos"]).copy()
    quat = np.array(geom["quat"]).copy()

    if xyz is not None:
        arr = np.asarray(xyz, dtype=float)
        if arr.shape[0] != 3:
            raise ValueError(f"xyz must have 3 values, got {arr}")
        pos[:3] = arr
    elif xy is not None:
        arr = np.asarray(xy, dtype=float)
        if arr.shape[0] < 2:
            raise ValueError(f"xy must have at least 2 values, got {arr}")
        pos[:2] = arr[:2]

    if yaw is not None:
        # yaw-only quaternion (w, x, y, z) around z-axis
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        quat = np.array([cy, 0.0, 0.0, sy], dtype=float)

    joint = env.env.get_object(obj_name).joints[-1]
    env.env.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))
    env.env.sim.forward()


def choose_target_obj(env: OffScreenRenderEnv, placement: Placement) -> str:
    # 1) Respect explicit placement selection if provided
    explicit = placement.get("obj")
    if explicit and explicit in env.env.object_states_dict:
        return explicit

    keys = list(env.env.object_states_dict.keys())

    # 2) Prefer common movable instance names, in order
    preferred_exact = [
        "blue_box_1",
        "green_box_1",
        "block_1",
        "card_1",
    ]
    for k in preferred_exact:
        if k in keys:
            return k

    # 3) Pattern-based fallbacks
    #    - prefer anything with "box" in name that isn't a container synonym
    #    - then anything with "card"
    containers = ["basket", "rack", "table", "wall", "cabinet", "tray"]
    def is_container(name: str) -> bool:
        return any(c in name for c in containers)

    for k in keys:
        if ("box" in k) and not is_container(k):
            return k
    for k in keys:
        if ("card" in k) and not is_container(k):
            return k

    # 4) Last resort: first non-container object
    for k in keys:
        if not is_container(k):
            return k

    # If everything else fails, just return the first key
    return keys[0]



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default=CONFIG["task"], help="Task name in my_suite")
    parser.add_argument("--out", default=CONFIG["out"], help="Output folder for screenshots")
    parser.add_argument("--camera", default=CONFIG["camera"], help="Which camera to save")
    parser.add_argument("--height", type=int, default=CONFIG["height"], help="Camera height")
    parser.add_argument("--width", type=int, default=CONFIG["width"], help="Camera width")
    # Parse known args but allow running without any CLI usage
    args, _ = parser.parse_known_args()

    task = args.task
    assert task in TASK_POSITIONS, f"No placements configured for task: {task}"
    placements = TASK_POSITIONS[task]

    out_dir = os.path.join(args.out, task)
    ensure_dir(out_dir)

    for i, p in enumerate(placements):
        # Build env per placement to honor per-initialization overrides
        overrides = p.get("overrides", CONFIG.get("overrides", {}))
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path(task),
            camera_names=[args.camera],
            camera_heights=args.height,
            camera_widths=args.width,
            camera_depths=False,
            horizon=1000,
            robots=["Panda"],
            controller="OSC_POSE",
            object_overrides=overrides,
        )
        env.seed(0)
        env.reset()
        # Choose object instance name to move (generalized)
        obj_name = choose_target_obj(env, p)
        set_obj_pose(
            env,
            obj_name=obj_name,
            xy=p.get("xy"),
            xyz=p.get("xyz"),
            yaw=p.get("yaw"),
        )
        # Generate observations from the current sim state
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        key = f"{args.camera}_image"
        assert key in obs, f"Camera key {key} not in observations. Available: {list(obs.keys())}"
        img = obs[key]
        # Rotate 180 degrees prior to saving
        img_rot = np.rot90(img, 2).copy()
        imageio.imwrite(os.path.join(out_dir, f"{i:02d}.png"), img_rot)
        env.close()
    print(f"Saved {len(placements)} previews to {out_dir}")


if __name__ == "__main__":
    main()

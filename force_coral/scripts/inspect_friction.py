#!/usr/bin/env python3
"""
Inspect friction coefficients (sliding, torsional, rolling) for key geoms in a task.

Reports friction for:
- floor geom (plane)
- all geoms belonging to the wall fixture (e.g., wall2_1)
- all geoms belonging to the block object (block_1)

Values are printed directly from MuJoCo's model: sim.model.geom_friction[geom_id]
Units follow MuJoCo's convention; these are dimensionless coefficients.

Usage:
  python scripts/inspect_friction.py \
    --task push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side

Notes:
- If a geom doesn't explicitly specify friction in XML, MuJoCo defaults are used.
- Effective contact friction in simulation is derived per contact; here we report
  the stored per-geom parameters that go into that computation.
"""

import argparse
import os

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


def bddl_path(task: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), "my_suite", f"{task}.bddl")


def print_geom_friction(env: OffScreenRenderEnv, geom_id: int, label: str):
    model = env.env.sim.model
    try:
        name = model.geom_id2name(geom_id)
    except Exception:
        name = f"geom_{geom_id}"
    fric = model.geom_friction[geom_id].copy()
    print(f"{label}: name={name:<40} id={geom_id:>3}  friction=[{fric[0]:.6g}, {fric[1]:.6g}, {fric[2]:.6g}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        default="push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side",
        help="my_suite task name",
    )
    args = ap.parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path(args.task),
        camera_names=["frontview"],
        camera_heights=64,
        camera_widths=64,
        camera_depths=False,
        horizon=50,
        robots=["Panda"],
        controller="OSC_POSE",
    )
    env.seed(0)
    env.reset()

    model = env.env.sim.model
    data = env.env.sim.data

    # Floor friction (named 'floor' in the scene XML)
    try:
        floor_gid = model.geom_name2id("floor")
        print_geom_friction(env, floor_gid, label="floor")
    except Exception as e:
        print("Could not locate floor geom by name 'floor':", e)

    # Robust helper: collect geoms whose owning body name starts with the object's name.
    # This avoids edge-cases with child / sibling traversal and is fast.
    def geoms_for_object_name_prefix(obj_root_name: str):
        body_ids = []
        for bid in range(model.nbody):
            try:
                bname = model.body_id2name(bid)
            except Exception:
                bname = None
            if bname and bname.startswith(obj_root_name):
                body_ids.append(bid)
        body_ids = set(body_ids)
        return [gid for gid in range(model.ngeom) if model.geom_bodyid[gid] in body_ids]

    # Wall and block body ids
    wall_name = "wall2_1"
    block_name = "block_1"
    try:
        wall_bid = env.env.obj_body_id[wall_name]
        wall_gids = geoms_for_object_name_prefix(wall_name)
        print(f"\nWall '{wall_name}' body_id={wall_bid} (by name prefix) -> {len(wall_gids)} geoms:")
        for gid in wall_gids:
            print_geom_friction(env, gid, label="  wall-geom")
    except Exception as e:
        print(f"Could not find wall '{wall_name}':", e)

    try:
        block_bid = env.env.obj_body_id[block_name]
        block_gids = geoms_for_object_name_prefix(block_name)
        print(f"\nBlock '{block_name}' body_id={block_bid} (by name prefix) -> {len(block_gids)} geoms:")
        for gid in block_gids:
            print_geom_friction(env, gid, label="  block-geom")
    except Exception as e:
        print(f"Could not find block '{block_name}':", e)

    env.close()


if __name__ == "__main__":
    main()

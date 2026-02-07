#!/usr/bin/env python3
"""
Confirm that object density is interpreted as kg/m^3 by comparing MuJoCo body mass
to density * volume for a BoxObject-based block.

Method:
- Create OffScreenRenderEnv for a simple my_suite task with per-object overrides
  setting block size (half-extents, meters) and density (candidate kg/m^3).
- After reset, read the MuJoCo body mass for the block and compare it to
  expected_mass = density * (2*sx) * (2*sy) * (2*sz).
- Print both and the absolute / relative error. A close match empirically
  confirms the density unit is kg/m^3.

Run:
  python scripts/confirm_density_units.py \
    --size 0.05 0.05 0.05 \
    --density 1000

Expected:
- For size=(0.05,0.05,0.05) (i.e., a 10 cm cube), volume = 0.1^3 = 0.001 m^3.
  With density=1000 kg/m^3 -> mass ≈ 1.0 kg. The printed body mass should be ~1.0.
"""

import argparse
import os
import numpy as np

import force_coral
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv

TASK = "push_the_blue_box_to_the_front_of_the_wall"  # any my_suite task with block_1 is fine


def bddl_path(task: str) -> str:
    return os.path.join(force_coral.get_data_path("bddl_files"), "my_suite", f"{task}.bddl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", nargs=3, type=float, default=[0.05, 0.05, 0.05], help="half-extents (m) sx sy sz")
    ap.add_argument("--density", type=float, default=1000.0, help="candidate density (kg/m^3)")
    args = ap.parse_args()

    overrides = {"block_1": {"size": args.size, "density": args.density}}

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path(TASK),
        camera_names=["frontview"],
        camera_heights=64,
        camera_widths=64,
        camera_depths=False,
        horizon=50,
        robots=["Panda"],
        controller="OSC_POSE",
        object_overrides=overrides,
    )
    env.seed(0)
    env.reset()

    block = "block_1"
    bid = env.env.obj_body_id[block]

    # Report mass from MuJoCo and expected mass from density * volume
    mj_mass = float(env.env.sim.model.body_mass[bid])
    sx, sy, sz = [float(x) for x in args.size]
    volume = (2.0 * sx) * (2.0 * sy) * (2.0 * sz)
    expected_mass = args.density * volume

    abs_err = abs(mj_mass - expected_mass)
    rel_err = abs_err / max(expected_mass, 1e-9)

    print("--- Density Unit Check (kg/m^3 hypothesis) ---")
    print(f"size (half-extents): {args.size}")
    print(f"density (passed):    {args.density} kg/m^3")
    print(f"volume (m^3):        {volume:.6f}")
    print(f"expected mass (kg):  {expected_mass:.6f}")
    print(f"mujoco mass (kg):    {mj_mass:.6f}")
    print(f"abs error:           {abs_err:.6e}")
    print(f"rel error:           {rel_err:.6e}")
    if rel_err < 1e-3:
        print("Conclusion: Density is interpreted as kg/m^3 (within tolerance).")
    else:
        print("Conclusion: Mismatch detected. Investigate object composition / multiple geoms.")

    env.close()


if __name__ == "__main__":
    main()


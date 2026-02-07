#!/usr/bin/env python3
"""
Uniformly scale an OBJ's vertex positions and write a new OBJ.

Usage:
  python3 scripts/scale_obj_uniform.py \
      --in libero/libero/assets/primitives/textured_cube/textured_cube_FP_atlas.obj \
      --scale 0.16 \
      --out libero/libero/assets/primitives/textured_cube/textured_cube_FP_atlas_016.obj

Notes:
  - Only 'v' (vertex) positions are scaled. 'vn' and 'vt' are kept as-is.
  - Keep the .mtl reference line (mtllib ...) unchanged; place the new OBJ in the
    same folder as the referenced MTL and PNG so relative paths still resolve.
  - For uniform scaling, normals remain valid. For non-uniform scaling, you should
    recompute normals offline.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def scale_obj(in_path: Path, out_path: Path, s: float) -> None:
    lines = in_path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for line in lines:
        if line.startswith("v "):
            try:
                _, x, y, z, *rest = line.split()
                x, y, z = float(x) * s, float(y) * s, float(z) * s
                if rest:
                    out_lines.append(f"v {x:.9f} {y:.9f} {z:.9f} {' '.join(rest)}")
                else:
                    out_lines.append(f"v {x:.9f} {y:.9f} {z:.9f}")
            except Exception:
                out_lines.append(line)
        else:
            out_lines.append(line)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input OBJ path")
    ap.add_argument("--out", dest="out", required=True, help="Output OBJ path")
    ap.add_argument("--scale", type=float, required=True, help="Uniform scale factor (e.g., 0.16)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale_obj(in_path, out_path, args.scale)
    print(f"Wrote {out_path} (scaled by {args.scale})")


if __name__ == "__main__":
    main()


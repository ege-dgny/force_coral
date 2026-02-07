#!/usr/bin/env python3
"""
Very simple JPG→PNG converter using OpenCV.

Usage:
  python -m force_coral.scripts.jpg_to_png \
    --in force_coral/data/assets/textures/my_texture_1.jpg \
    --out force_coral/data/assets/textures/my_texture_1.png

Both flags are optional; if only --in is given, the output path is inferred by
replacing the extension with .png.
"""

import argparse
import os
import sys

import cv2

import force_coral

_DEFAULT_INPUT = os.path.join(
    force_coral.get_data_path("assets"), "textures", "my_texture_1.jpg"
)


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--in", dest="inp", required=False,
                        default=_DEFAULT_INPUT,
                        help="Input JPG path")
    parser.add_argument("--out", dest="out", required=False,
                        help="Output PNG path (defaults to input with .png)")
    args = parser.parse_args()

    inp = args.inp
    out = args.out

    if out is None:
        root, _ = os.path.splitext(inp)
        out = root + ".png"

    img = cv2.imread(inp, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read input: {inp}", file=sys.stderr)
        sys.exit(1)

    ok = cv2.imwrite(out, img)
    if not ok:
        print(f"Failed to write PNG: {out}", file=sys.stderr)
        sys.exit(2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


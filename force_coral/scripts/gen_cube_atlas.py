#!/usr/bin/env python3
"""
Generate a 3x2 UV atlas PNG for the cube with distinct panels.
Output: libero/libero/assets/primitives/textured_cube/textured_cube_FP_atlas.png
"""
from pathlib import Path

W, H = 1024, 1024
GRID_COLS, GRID_ROWS = 3, 2

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    raise SystemExit("Pillow (PIL) is required. Install via: pip install pillow")

out_path = Path("libero/libero/assets/primitives/textured_cube/textured_cube_FP_atlas.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

img = Image.new("RGB", (W, H), (240, 240, 240))
draw = ImageDraw.Draw(img)

# Panel assignments matching OBJ comments
panels = [
    (0, 0, "L", (220, 70, 70)),   # Left
    (1, 0, "F", (70, 160, 70)),   # Front
    (2, 0, "R", (70, 70, 220)),   # Right
    (0, 1, "B", (210, 160, 60)),  # Back
    (1, 1, "T", (160, 70, 160)),  # Top
    (2, 1, "D", (70, 160, 160)),  # Bottom (Down)
]

cell_w = W // GRID_COLS
cell_h = H // GRID_ROWS

# Try a reasonably large font; fall back to default if missing
def load_font():
    for size in (180, 140, 100, 72, 48):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            continue
    return ImageFont.load_default()

font = load_font()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    try:
        return font.getsize(text)
    except Exception:
        return (64, 64)

for cx, cy, label, color in panels:
    x0, y0 = cx * cell_w, cy * cell_h
    x1, y1 = x0 + cell_w, y0 + cell_h

    # Fill panel
    draw.rectangle([x0, y0, x1, y1], fill=color)

    # Add a small checker overlay for more features
    check_size = 24
    for yy in range(y0, y1, check_size):
        for xx in range(x0, x1, check_size):
            if ((xx // check_size) + (yy // check_size)) % 2 == 0:
                draw.rectangle([xx, yy, min(xx+check_size-1, x1-1), min(yy+check_size-1, y1-1)], outline=None, fill=(255,255,255,))

    # Panel border
    draw.rectangle([x0, y0, x1-1, y1-1], outline=(0,0,0), width=4)

    # Center label
    tw, th = _text_size(draw, label, font)
    tx = x0 + (cell_w - tw) // 2
    ty = y0 + (cell_h - th) // 2
    draw.text((tx, ty), label, fill=(0,0,0), font=font)

img.save(out_path)
print(f"Wrote {out_path} ({W}x{H})")

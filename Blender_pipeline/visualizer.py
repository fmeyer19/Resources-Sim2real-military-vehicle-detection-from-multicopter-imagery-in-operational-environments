"""
visualise_bboxes.py
Quick sanity-check: draws YOLO bounding boxes on rendered images.

Usage:
    python visualise_bboxes.py                        # show all images
    python visualise_bboxes.py --idx 0 5 12           # specific frames
    python visualise_bboxes.py --idx 0 --save         # save annotated image
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
CLASSES    = (OUTPUT_DIR / "classes.txt").read_text().splitlines()

COLORS = [
    (255, 50, 50),
    (50, 200, 50),
    (50, 150, 255),
    (255, 200, 0),
    (200, 50, 255),
    (0, 220, 220),
]


def color_for(class_id: int):
    """
    Return a stable RGB colour for *class_id*, cycling through the palette.
    """
    return COLORS[class_id % len(COLORS)]


def load_label(label_path: Path, W: int, H: int) -> list[tuple]:
    """
    Return the label file's boxes as ``(class_id, x0, y0, x1, y1)`` in pixel
    coordinates, or an empty list when the file is missing.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        x0 = int((cx - bw / 2) * W)
        y0 = int((cy - bh / 2) * H)
        x1 = int((cx + bw / 2) * W)
        y1 = int((cy + bh / 2) * H)
        boxes.append((cls, x0, y0, x1, y1))
    return boxes


def annotate_image(img_path: Path) -> Image.Image:
    """
    Return a copy of the image at *img_path* with its YOLO boxes and class labels
    drawn on.
    """
    img   = Image.open(img_path).convert("RGB")
    draw  = ImageDraw.Draw(img)
    W, H  = img.size
    label_path = LABELS_DIR / img_path.with_suffix(".txt").name
    boxes = load_label(label_path, W, H)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for cls, x0, y0, x1, y1 in boxes:
        color     = color_for(cls)
        label_str = CLASSES[cls] if cls < len(CLASSES) else str(cls)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.rectangle([x0, y0 - 16, x0 + len(label_str) * 8, y0], fill=color)
        draw.text((x0 + 2, y0 - 15), label_str, fill=(0, 0, 0), font=font)

    return img


def main():
    """
    Parse the command-line arguments and show or save the annotated frames.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", nargs="*", type=int, default=None,
                        help="Frame indices to show (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images to output/debug/")
    args = parser.parse_args()

    all_images = sorted(IMAGES_DIR.glob("*.png"))
    if not all_images:
        print("No images found.")
        return

    if args.idx is not None:
        images = [IMAGES_DIR / f"{i:06d}.png" for i in args.idx]
    else:
        images = all_images

    if args.save:
        debug_dir = OUTPUT_DIR / "debug"
        debug_dir.mkdir(exist_ok=True)

    for img_path in images:
        if not img_path.exists():
            print(f"Not found: {img_path}")
            continue

        annotated = annotate_image(img_path)
        print(f"{img_path.name} — {len(load_label(LABELS_DIR / img_path.with_suffix('.txt').name, *annotated.size))} boxes")

        if args.save:
            out_path = debug_dir / img_path.name
            annotated.save(out_path)
            print(f"  Saved → {out_path}")
        else:
            annotated.show()


if __name__ == "__main__":
    main()

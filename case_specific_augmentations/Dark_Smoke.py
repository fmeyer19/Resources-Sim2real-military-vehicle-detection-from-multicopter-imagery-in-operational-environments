"""
Dark_Smoke.py

Loads every supported image from a given folder, subfolders excluded, and adds
a random cloud-like dark smoke effect to each of them. The pattern is based on
fractal noise, which produces more realistic structures than plain random noise.

The processed images are written to the subfolder "edited" of the input folder.
The file name of each image carries the smoke strength that was applied, for
example 'image1_dark_smoke_1.24.png'.

Every image receives its own random seed, so that each smoke pattern is unique.
A global seed can optionally be set in order to obtain reproducible results.

The original images are left unchanged.

Adjust the settings above:
- EXTS: supported file extensions
- INPUT_FOLDER: path to the input folder
- MIN_SMOKE / MAX_SMOKE: range of the random smoke strength
- GLOBAL_SEED: optional seed for reproducible runs (None = fully random)
- BASE_SCALE: affects the size and structure of the smoke (larger = softer clouds)
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter

INPUT_FOLDER = r"<path to input folder>"

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

MIN_SMOKE = "<min smoke strength>"
MAX_SMOKE = "<max smoke strength>"

BASE_SCALE = 9

# Dark smoke shade in the RGB range [0, 1]
# Lower values = darker smoke, higher values = grayer smoke
SMOKE_COLOR = np.array([0.12, 0.12, 0.12], dtype=np.float32)

# Additional local darkening caused by smoke
# Higher values make the effect darker and denser
DARKENING_FACTOR = 0.45

# If set (e.g. 42), run is reproducible
# If None, every run is random
GLOBAL_SEED = None


def is_image(filename):
    return os.path.splitext(filename.lower())[1] in EXTS


def generate_cloud_noise(shape, octaves=5, seed=None):
    """
    Fractal, cloud-like noise with a seed.
    """
    rng = np.random.default_rng(seed)

    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)

    frequency = 1.0
    amplitude = 1.0

    for _ in range(octaves):
        small_h = max(1, int(h / (BASE_SCALE * frequency)))
        small_w = max(1, int(w / (BASE_SCALE * frequency)))

        layer = rng.random((small_h, small_w), dtype=np.float32)

        layer_img = Image.fromarray((layer * 255).astype(np.uint8))
        layer_img = layer_img.resize((w, h), Image.BILINEAR)

        layer = np.array(layer_img).astype(np.float32) / 255.0

        noise += layer * amplitude

        frequency *= 2.0
        amplitude *= 0.5

    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise


def apply_dark_smoke(image, smoke_strength, seed):
    img_np = np.array(image).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    cloud = generate_cloud_noise((h, w), seed=seed)

    rng = random.Random(seed)
    blur_radius = rng.uniform(10, 20)

    cloud_img = Image.fromarray((cloud * 255).astype(np.uint8))
    cloud_img = cloud_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    cloud = np.array(cloud_img).astype(np.float32) / 255.0

    exponent = rng.uniform(1.3, 1.8)
    cloud = cloud ** exponent

    alpha = cloud * smoke_strength

    alpha = np.clip(alpha, 0.0, 0.85)

    smoke_color = np.ones_like(img_np) * SMOKE_COLOR

    darkened_img = img_np * (1.0 - alpha[..., None] * DARKENING_FACTOR)

    result = darkened_img * (1 - alpha[..., None]) + smoke_color * alpha[..., None]

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    if GLOBAL_SEED is not None:
        random.seed(GLOBAL_SEED)

    output_folder = os.path.join(INPUT_FOLDER, "edited")
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(INPUT_FOLDER)

    for file in files:
        if not is_image(file):
            continue

        input_path = os.path.join(INPUT_FOLDER, file)

        try:
            img = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"Error at {file}: {e}")
            continue

        # Seed per image
        image_seed = random.randint(0, 1_000_000)

        smoke_strength = random.uniform(MIN_SMOKE, MAX_SMOKE)

        edited_img = apply_dark_smoke(img, smoke_strength, seed=image_seed)

        name, ext = os.path.splitext(file)
        new_filename = f"{name}_dark_smoke_{smoke_strength:.2f}{ext}"
        output_path = os.path.join(output_folder, new_filename)

        edited_img.save(output_path)

        print(f"Saved: {new_filename} (seed={image_seed})")


if __name__ == "__main__":
    main()
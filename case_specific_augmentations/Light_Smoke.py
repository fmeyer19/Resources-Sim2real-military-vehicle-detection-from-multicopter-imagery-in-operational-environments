"""
Light_Smoke.py

Loads every supported image from a given folder, subfolders excluded, and adds
a random cloud-like light smoke effect to each of them. The pattern is based on
fractal noise, which produces more realistic structures than plain random noise.

The processed images are written to the subfolder "edited" of the input folder.
The file name of each image carries the smoke strength that was applied, for
example 'image1_fog_0.73.png'.

On naming: "fog" is the working name of this method, which is referred to as
"light smoke" in the thesis. The internal names and the file name token were
left unchanged, because they are part of the file names of the datasets already
published.

Every image receives its own random seed, so that each smoke pattern is unique.
A global seed can optionally be set in order to obtain reproducible results.

The original images are left unchanged.

Adjust the settings above:
- EXTS: supported file extensions
- INPUT_FOLDER: path to the input folder
- MIN_FOG / MAX_FOG: range of the random smoke strength
- GLOBAL_SEED: optional seed for reproducible runs (None = fully random)
- BASE_SCALE: affects the size and structure of the smoke (larger = softer clouds)
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter

INPUT_FOLDER = r"<path to input folder>"

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

MIN_FOG = "<min smoke strength>"
MAX_FOG = "<max smoke strength>"

BASE_SCALE = 9

# If set (e.g. 42), the run is reproducible
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


def apply_fog(image, fog_strength, seed):
    img_np = np.array(image).astype(np.float32) / 255.0
    h, w, _ = img_np.shape

    # Cloud pattern with a fixed seed
    cloud = generate_cloud_noise((h, w), seed=seed)

    # Blur (slightly randomized, but deterministic based on the seed)
    rng = random.Random(seed)
    blur_radius = rng.uniform(10, 20)

    cloud_img = Image.fromarray((cloud * 255).astype(np.uint8))
    cloud_img = cloud_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    cloud = np.array(cloud_img).astype(np.float32) / 255.0

    # Contrast curve
    exponent = rng.uniform(1.3, 1.8)
    cloud = cloud ** exponent

    fog_color = np.ones_like(img_np)

    alpha = cloud * fog_strength

    result = img_np * (1 - alpha[..., None]) + fog_color * alpha[..., None]

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

        # Seed per image (critical)
        image_seed = random.randint(0, 1_000_000)

        fog_strength = random.uniform(MIN_FOG, MAX_FOG)

        edited_img = apply_fog(img, fog_strength, seed=image_seed)

        name, ext = os.path.splitext(file)
        new_filename = f"{name}_fog_{fog_strength:.2f}{ext}"
        output_path = os.path.join(output_folder, new_filename)

        edited_img.save(output_path)

        print(f"Saved: {new_filename} (seed={image_seed})")


if __name__ == "__main__":
    main()
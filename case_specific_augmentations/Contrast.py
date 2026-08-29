"""
Contrast.py

Loads every image from a given folder, subfolders excluded, and applies a
random contrast change to each of them. The processed images are written to
the subfolder "edited" of the input folder.

The file name of each image carries the contrast factor that was applied, for
example 'image1_contrast1.23.png'. Existing files are not overwritten but
renamed automatically.

The original images are left unchanged.

Adjust the settings above:
- INPUT_FOLDER: path to the input folder
- EXTS: supported image formats
- MIN_CONTRAST / MAX_CONTRAST: range of the random contrast factor
"""

import os
import random
from PIL import Image, ImageEnhance

INPUT_FOLDER = r"<path to input folder>"

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

MIN_CONTRAST = "<min contrast factor>"
MAX_CONTRAST = "<max contrast factor>"

def is_image_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in EXTS


def get_unique_filename(folder, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    path = os.path.join(folder, filename)

    while os.path.exists(path):
        filename = f"{base}_{counter}{ext}"
        path = os.path.join(folder, filename)
        counter += 1

    return path


def process_images():
    if not os.path.exists(INPUT_FOLDER):
        return

    output_folder = os.path.join(INPUT_FOLDER, "edited")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, filename)

        if not os.path.isfile(input_path):
            continue

        if not is_image_file(filename):
            continue

        try:
            img = Image.open(input_path)

            contrast_factor = random.uniform(MIN_CONTRAST, MAX_CONTRAST)

            enhancer = ImageEnhance.Contrast(img)
            img_enhanced = enhancer.enhance(contrast_factor)

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_contrast{contrast_factor:.2f}{ext}"

            output_path = get_unique_filename(output_folder, new_filename)

            img_enhanced.save(output_path)

            print(f"Done! {filename} -> {os.path.basename(output_path)} ({contrast_factor:.2f})")

        except Exception:
            pass  # the image is skipped without a message

if __name__ == "__main__":
    process_images()
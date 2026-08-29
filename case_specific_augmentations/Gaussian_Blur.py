"""
Gaussian_Blur.py

Loads every image in a given folder and applies a random Gaussian blur to
each of them. The processed images are written to the subfolder
"gaussian_blurred" of the input folder. The file name of each image carries
the standard deviation that was applied.

Adjust the settings above:
- EXTS: supported file extensions
- INPUT_FOLDER: path to the input folder
- OUTPUT_FOLDER: optional, path to the output folder
- blur_range: range of the random blur strength
"""

import os
import random
from PIL import Image, ImageFilter

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
INPUT_FOLDER = r"<path to input folder>"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "gaussian_blurred")

blur_range = ("<min sigma>", "<max sigma>")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)

    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in EXTS:
        try:
            img = Image.open(file_path)

            blur_radius = random.uniform(blur_range[0], blur_range[1])

            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_blur_{blur_radius:.2f}{ext}"
            new_path = os.path.join(OUTPUT_FOLDER, new_filename)

            blurred_img.save(new_path)

            print(f"{filename} -> {new_filename} (sigma={blur_radius:.2f})")

        except Exception as e:
            print(f"Error at {filename}: {e}")
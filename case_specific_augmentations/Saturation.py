"""
Saturation.py

Loads every image in a given folder, subfolders excluded, and changes the color
saturation of each of them at random within a defined range. The processed
images are written to the subfolder "edited" of the input folder.

The file name of each image carries the saturation factor that was applied, for
example 'image1_saturation_0.87.png'.

The original images are left unchanged.

Adjust the settings above:
- EXTS: supported file extensions
- INPUT_FOLDER: path to the input folder
- SATURATION_MIN: minimum saturation (0 = fully desaturated)
- SATURATION_MAX: maximum saturation (above 1 = stronger colors)
"""

import os
import random
from PIL import Image, ImageEnhance

INPUT_FOLDER = r"<path to input folder>"  # folder holding the images

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

SATURATION_MIN = "<min saturation factor>" # 0 =  no saturation (gray)
SATURATION_MAX = "<max saturation factor>" # >1 = strong saturation

output_folder = os.path.join(INPUT_FOLDER, "edited")

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)

    if not os.path.isfile(file_path):
        continue

    name, ext = os.path.splitext(filename)

    if ext.lower() not in EXTS:
        continue

    try:
        img = Image.open(file_path)

        saturation_factor = random.uniform(SATURATION_MIN, SATURATION_MAX)

        enhancer = ImageEnhance.Color(img)
        edited_img = enhancer.enhance(saturation_factor)

        new_filename = f"{name}_saturation_{round(saturation_factor, 2)}{ext}"
        output_path = os.path.join(output_folder, new_filename)
        
        edited_img.save(output_path)

        print(f"Done: {filename} -> {new_filename}")

    except Exception as e:
        print(f"Error at {filename}: {e}")

print("Done!")
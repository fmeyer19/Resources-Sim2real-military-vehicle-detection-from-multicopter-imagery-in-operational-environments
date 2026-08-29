"""
Compression_Artefacts.py

Loads every image in a given folder and applies random JPEG compression to
each of them, which produces compression artifacts. The processed images are
written to the subfolder "compressed_images" of the input folder. The file
name of each image carries the quality that was applied, for example
'image1_q42_compressed.jpg'. The original images are left unchanged.

Adjust the settings above:
- EXTS: supported file extensions
- INPUT_FOLDER: path to the input folder
- OUTPUT_FOLDER: optional, path to the output folder
- quality range: range of the random JPEG quality, for example 10 to 90
"""

import os
import random
from PIL import Image

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
INPUT_FOLDER = r"<path to input folder>"  # <-- enter the folder path here
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "compressed_images")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    file_path = os.path.join(INPUT_FOLDER, filename)
    
    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in EXTS:
        try:
            img = Image.open(file_path)
            
            quality = random.randint("<min quality>", "<max quality>")
            
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_q{quality}_compressed.jpg"
            new_path = os.path.join(OUTPUT_FOLDER, new_filename)
            
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            img.save(new_path, "JPEG", quality=quality)
            print(f"{filename} -> {new_filename} (quality={quality})")
        except Exception as e:
            print(f"Error at {filename}: {e}")
"""
Gaussian_Noise_RGB.py

Loads every image in a given folder and adds colored Gaussian noise to each
of them. A separate random perturbation is generated for every RGB channel.

The processed images are written to the subfolder "noisy_images_rgb" of the
input folder. The original images are left unchanged.

The file name of each image carries the noise strength that was applied, for
example:
- image1_std12.34_rgb.png

Adjust the settings above:
- EXTS: supported file extensions
- folder: path to the input folder
- STD_RANGE: range of the random noise strength
"""

import os
import random
import numpy as np
from PIL import Image

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

STD_RANGE = ("<min std>", "<max std>")


def add_gaussian_noise_rgb(image_array, mean=0, std_range=STD_RANGE):
    std = random.uniform(*std_range)

    noise = np.random.normal(mean, std, image_array.shape)

    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8), std


def process_images(input_folder):
    output_folder = os.path.join(input_folder, "noisy_images_rgb")
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()

        if ext in EXTS:
            input_path = os.path.join(input_folder, file)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img_array = np.array(img)

                    noisy_array, std = add_gaussian_noise_rgb(img_array)
                    noisy_img = Image.fromarray(noisy_array)

                    base_name, ext = os.path.splitext(file)
                    save_name = f"{base_name}_std{std:.2f}_rgb{ext}"
                    save_path = os.path.join(output_folder, save_name)

                    noisy_img.save(save_path)
                    print(f"Augmented: {input_path} (Noise std={std:.2f}, type=rgb)")

            except Exception as e:
                print(f"Error at: {input_path}: {e}")


if __name__ == "__main__":
    folder = r"<path to input folder>"

    if os.path.isdir(folder):
        process_images(folder)
        print("Done")
    else:
        print("Invalid Folder")
"""
02_extract_features_all_datasets.py

Creates one .npy feature file for each of the image folders listed below.
All images are processed with the same YOLOv26n model.
The feature maps immediately preceding the detection head are captured
and reduced to one feature vector per image by global average pooling.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# SETTINGS
# =========================

MODEL_PATH = r"<path to yolo26n.pt>"

OUTPUT_FOLDER = Path(r"<path to features_all output>")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 960

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


DATASETS = {
    # <path> stands for the directory holding the image folders named below.
    #
    # On naming: "white_fog" is the working name of the method referred to as
    # "light smoke" in the thesis. Internal names and folder names were left
    # unchanged, because they become part of the file names of the generated
    # .npy files and are read back in by script 03.
    "arma_compression_artefacts_value_a": r"<path>\Arma_Compression_Artefacts_value_a=20",  # images: 500
    "arma_compression_artefacts_value_b": r"<path>\Arma_Compression_Artefacts_value_b=50",  # images: 500
    "arma_compression_artefacts_value_c": r"<path>\Arma_Compression_Artefacts_value_c=80",  # images: 500
    "arma_contrast_value_a": r"<path>\Arma_Contrast_value_0.2",  # images: 500
    "arma_contrast_value_b": r"<path>\Arma_Contrast_value_0.8",  # images: 500
    "arma_contrast_value_c": r"<path>\Arma_Contrast_value_1.4",  # images: 500
    "arma_dark_smoke_value_a": r"<path>\Arma_Dark_Smoke_value_a=1.0",  # images: 500
    "arma_dark_smoke_value_b": r"<path>\Arma_Dark_Smoke_value_b=1.5",  # images: 500
    "arma_dark_smoke_value_c": r"<path>\Arma_Dark_Smoke_value_c=2.0",  # images: 500
    "arma_gaussian_blur_value_a": r"<path>\Arma_Gaussian_Blur_value_a=0.5",  # images: 500
    "arma_gaussian_blur_value_b": r"<path>\Arma_Gaussian_Blur_value_b=1.5",  # images: 500
    "arma_gaussian_blur_value_c": r"<path>\Arma_Gaussian_Blur_value_c=2.5",  # images: 500
    "arma_gaussian_noise_color_value_a": r"<path>\Arma_Gaussian_Noise_Color_value_a=5",  # images: 500
    "arma_gaussian_noise_color_value_b": r"<path>\Arma_Gaussian_Noise_Color_value_b=15",  # images: 500
    "arma_gaussian_noise_color_value_c": r"<path>\Arma_Gaussian_Noise_Color_value_c=25",  # images: 500
    "arma_gaussian_noise_grey_value_a": r"<path>\Arma_Gaussian_Noise_Grey_value_a=5",  # images: 500
    "arma_gaussian_noise_grey_value_b": r"<path>\Arma_Gaussian_Noise_Grey_value_b=15",  # images: 500
    "arma_gaussian_noise_grey_value_c": r"<path>\Arma_Gaussian_Noise_Grey_value_c=25",  # images: 500
    "arma_raw": r"<path>\Arma_Raw_1",  # images: 500
    "arma_saturation_value_a": r"<path>\Arma_Saturation_value_a=0.2",  # images: 500
    "arma_saturation_value_b": r"<path>\Arma_Saturation_value_b=0.8",  # images: 500
    "arma_saturation_value_c": r"<path>\Arma_Saturation_value_c=1.4",  # images: 500
    "arma_white_fog_value_a": r"<path>\Arma_White_Fog_value_a=1",  # images: 500
    "arma_white_fog_value_b": r"<path>\Arma_White_Fog_value_b=1.5",  # images: 500
    "arma_white_fog_value_c": r"<path>\Arma_White_Fog_value_c=2",  # images: 500
    "blender_compression_artefacts_value_a": r"<path>\Blender_Compression_Artefacts_value_a=20",  # images: 500
    "blender_compression_artefacts_value_b": r"<path>\Blender_Compression_Artefacts_value_b=50",  # images: 500
    "blender_compression_artefacts_value_c": r"<path>\Blender_Compression_Artefacts_value_c=80",  # images: 500
    "blender_contrast_value_a": r"<path>\Blender_Contrast_value_a=0.2",  # images: 500
    "blender_contrast_value_b": r"<path>\Blender_Contrast_value_b=0.8",  # images: 500
    "blender_contrast_value_c": r"<path>\Blender_Contrast_value_c=1.4",  # images: 500
    "blender_dark_smoke_value_a": r"<path>\Blender_Dark_Smoke_value_a=1",  # images: 500
    "blender_dark_smoke_value_b": r"<path>\Blender_Dark_Smoke_value_b=1.5",  # images: 500
    "blender_dark_smoke_value_c": r"<path>\Blender_Dark_Smoke_value_c=2",  # images: 500
    "blender_gaussian_blur_value_a": r"<path>\Blender_Gaussian_Blur_value_a=0.5",  # images: 500
    "blender_gaussian_blur_value_b": r"<path>\Blender_Gaussian_Blur_value_b=1.5",  # images: 500
    "blender_gaussian_blur_value_c": r"<path>\Blender_Gaussian_Blur_value_c=2.5",  # images: 500
    "blender_gaussian_noise_color_value_a": r"<path>\Blender_Gaussian_Noise_Color_value_a=5",  # images: 500
    "blender_gaussian_noise_color_value_b": r"<path>\Blender_Gaussian_Noise_Color_value_b=15",  # images: 500
    "blender_gaussian_noise_color_value_c": r"<path>\Blender_Gaussian_Noise_Color_value_c=25",  # images: 500
    "blender_gaussian_noise_grey_value_a": r"<path>\Blender_Gaussian_Noise_Grey_value_a=5",  # images: 500
    "blender_gaussian_noise_grey_value_b": r"<path>\Blender_Gaussian_Noise_Grey_value_b=15",  # images: 500
    "blender_gaussian_noise_grey_value_c": r"<path>\Blender_Gaussian_Noise_Grey_value_c=25",  # images: 500
    "blender_raw": r"<path>\Blender_Raw",  # images: 500
    "blender_saturation_value_a": r"<path>\Blender_Saturation_value_a=0.2",  # images: 500
    "blender_saturation_value_b": r"<path>\Blender_Saturation_value_b=0.8",  # images: 500
    "blender_saturation_value_c": r"<path>\Blender_Saturation_value_c=1.4",  # images: 500
    "blender_white_fog_value_a": r"<path>\Blender_White_Fog_value_a=1",  # images: 500
    "blender_white_fog_value_b": r"<path>\Blender_White_Fog_value_b=1.5",  # images: 500
    "blender_white_fog_value_c": r"<path>\Blender_White_Fog_value_c=2",  # images: 500
    "mixed_compression_artefacts_value_a": r"<path>\Mixed_Compression_Artefacts_value_a=20",  # images: 500
    "mixed_compression_artefacts_value_b": r"<path>\Mixed_Compression_Artefacts_value_b=50",  # images: 500
    "mixed_compression_artefacts_value_c": r"<path>\Mixed_Compression_Artefacts_value_c=80",  # images: 500
    "mixed_contrast_value_a": r"<path>\Mixed_Contrast_value_a=0.2",  # images: 500
    "mixed_contrast_value_b": r"<path>\Mixed_Contrast_value_b=0.8",  # images: 500
    "mixed_contrast_value_c": r"<path>\Mixed_Contrast_value_c=1.4",  # images: 500
    "mixed_dark_smoke_value_a": r"<path>\Mixed_Dark_Smoke_value_a=1",  # images: 500
    "mixed_dark_smoke_value_b": r"<path>\Mixed_Dark_Smoke_value_b=1.5",  # images: 500
    "mixed_dark_smoke_value_c": r"<path>\Mixed_Dark_Smoke_value_c=2",  # images: 500
    "mixed_gaussian_blur_value_a": r"<path>\Mixed_Gaussian_Blur_value_a=0.5",  # images: 500
    "mixed_gaussian_blur_value_b": r"<path>\Mixed_Gaussian_Blur_value_b=1.5",  # images: 500
    "mixed_gaussian_blur_value_c": r"<path>\Mixed_Gaussian_Blur_value_c=2.5",  # images: 500
    "mixed_gaussian_noise_color_value_a": r"<path>\Mixed_Gaussian_Noise_Color_value_a=5",  # images: 500
    "mixed_gaussian_noise_color_value_b": r"<path>\Mixed_Gaussian_Noise_Color_value_b=15",  # images: 500
    "mixed_gaussian_noise_color_value_c": r"<path>\Mixed_Gaussian_Noise_Color_value_c=25",  # images: 500
    "mixed_gaussian_noise_grey_value_a": r"<path>\Mixed_Gaussian_Noise_Grey_value_a=5",  # images: 500
    "mixed_gaussian_noise_grey_value_b": r"<path>\Mixed_Gaussian_Noise_Grey_value_b=15",  # images: 500
    "mixed_gaussian_noise_grey_value_c": r"<path>\Mixed_Gaussian_Noise_Grey_value_c=25",  # images: 500
    "mixed_raw": r"<path>\Mixed_Raw",  # images: 500
    "mixed_saturation_value_a": r"<path>\Mixed_Saturation_value_a=0.2",  # images: 500
    "mixed_saturation_value_b": r"<path>\Mixed_Saturation_value_b=0.8",  # images: 500
    "mixed_saturation_value_c": r"<path>\Mixed_Saturation_value_c=1.4",  # images: 500
    "mixed_white_fog_value_a": r"<path>\Mixed_White_Fog_value_a=1",  # images: 500
    "mixed_white_fog_value_b": r"<path>\Mixed_White_Fog_value_b=1.5",  # images: 500
    "mixed_white_fog_value_c": r"<path>\Mixed_White_Fog_value_c=2",  # images: 500
    "realworld_A": r"<path>\RealWorld_500Images_1",  # images: 500
    "realworld_B": r"<path>\RealWorld_500Images_2",  # images: 500
}


# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)
model.model.to(device)
model.model.eval()

print("YOLO model loaded successfully.")


# =========================
# FEATURE HOOK
# =========================

captured_features = None


def hook_detect_input(module, inputs):
    global captured_features
    captured_features = inputs[0]


detect_layer = model.model.model[-1]
detect_layer.register_forward_pre_hook(hook_detect_input)

print("Feature hook registered successfully.")


# =========================
# FUNCTIONS
# =========================

def get_image_paths(folder):
    folder = Path(folder)

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def extract_feature_vector(image_path):
    global captured_features

    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img.to(device)

    captured_features = None

    with torch.no_grad():
        _ = model.model(img)

    if captured_features is None:
        raise RuntimeError("No features were captured.")

    pooled_features = []

    if isinstance(captured_features, (list, tuple)):
        for feature_map in captured_features:
            pooled = feature_map.mean(dim=(2, 3))
            pooled_features.append(pooled)

        feature_vector = torch.cat(pooled_features, dim=1)

    else:
        feature_vector = captured_features.mean(dim=(2, 3))

    return feature_vector.squeeze(0).cpu().numpy()


def extract_features_for_dataset(dataset_name, folder):
    image_paths = get_image_paths(folder)

    print("\n========================================")
    print(f"Dataset: {dataset_name}")
    print(f"Folder:  {folder}")
    print(f"Images:  {len(image_paths)}")
    print("========================================")

    if len(image_paths) == 0:
        print(f"WARNING: No images found for dataset: {dataset_name}")
        return

    features = []
    valid_image_names = []

    for image_path in tqdm(image_paths):
        try:
            feature_vector = extract_feature_vector(image_path)
            features.append(feature_vector)
            valid_image_names.append(image_path.name)
        except Exception as e:
            print(f"Skipping {image_path.name}: {e}")

    features = np.array(features, dtype=np.float32)

    print(f"Feature matrix shape for {dataset_name}: {features.shape}")

    feature_output_path = OUTPUT_FOLDER / f"{dataset_name}_features.npy"
    names_output_path = OUTPUT_FOLDER / f"{dataset_name}_image_names.txt"

    np.save(feature_output_path, features)

    with open(names_output_path, "w", encoding="utf-8") as f:
        for name in valid_image_names:
            f.write(name + "\n")

    print(f"Saved features:    {feature_output_path}")
    print(f"Saved image names: {names_output_path}")


# =========================
# MAIN
# =========================

for dataset_name, folder in DATASETS.items():
    extract_features_for_dataset(dataset_name, folder)

print("\nDone.")
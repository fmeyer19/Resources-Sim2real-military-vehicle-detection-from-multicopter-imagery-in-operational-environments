from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# EINSTELLUNGEN
# =========================

MODEL_PATH = r"<path to yolo26n.pt>"

REAL_A_FOLDER = Path("<path to REAL_A Dataset for Baseline>")
REAL_B_FOLDER = Path("<path to REAL_B Dataset for Baseline>")

OUTPUT_FOLDER = Path("<path to output directory>")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# MODELL LADEN
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
# HILFSFUNKTIONEN
# =========================

def get_image_paths(folder):
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

    feature_vector = feature_vector.squeeze(0).cpu().numpy()

    return feature_vector


def extract_features_for_folder(folder, dataset_name):
    image_paths = get_image_paths(folder)

    print(f"\nDataset: {dataset_name}")
    print(f"Found images: {len(image_paths)}")

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

    return features, valid_image_names


# =========================
# MAIN
# =========================

real_a_features, real_a_names = extract_features_for_folder(
    REAL_A_FOLDER,
    "real_A"
)

real_b_features, real_b_names = extract_features_for_folder(
    REAL_B_FOLDER,
    "real_B"
)

np.save(OUTPUT_FOLDER / "real_A_features.npy", real_a_features)
np.save(OUTPUT_FOLDER / "real_B_features.npy", real_b_features)

with open(OUTPUT_FOLDER / "real_A_image_names.txt", "w", encoding="utf-8") as f:
    for name in real_a_names:
        f.write(name + "\n")

with open(OUTPUT_FOLDER / "real_B_image_names.txt", "w", encoding="utf-8") as f:
    for name in real_b_names:
        f.write(name + "\n")

print("\nSaved feature files:")
print(OUTPUT_FOLDER / "real_A_features.npy")
print(OUTPUT_FOLDER / "real_B_features.npy")

print("\nDone.")
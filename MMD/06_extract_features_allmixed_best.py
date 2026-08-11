"""
02_extract_features_all_datasets.py

Erzeugt für jeden angegebenen Bildordner eine .npy-Feature-Datei.
Alle Bilder werden mit demselben YOLOv26n-Modell verarbeitet.
Die Feature Maps unmittelbar vor dem Detection Head werden abgegriffen
und mittels Global Average Pooling zu einem Feature-Vektor pro Bild
zusammengefasst.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# =========================
# EINSTELLUNGEN
# =========================

MODEL_PATH = r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\yolo26n.pt"

OUTPUT_FOLDER = Path(
    r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\features_all"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# Trage hier alle vorhandenen Datensatzordner ein.
# Links: sauberer interner Name
# Rechts: exakter Windows-Pfad zum Bildordner
DATASETS = {
    "mixed_arma_blender_best_values": r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\Mixed_Arma_Blender_Best_Values",
}


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
# FUNKTIONEN
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
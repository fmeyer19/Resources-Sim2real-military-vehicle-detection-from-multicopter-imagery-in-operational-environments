from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# =========================
# EINSTELLUNGEN
# =========================

# Wenn yolo26n.pt im gleichen Ordner wie dieses Skript liegt:
MODEL_PATH = r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\yolo26n.pt"

# Falls das Modell woanders liegt, stattdessen z. B.:
# MODEL_PATH = r"C:\Users\admin\Desktop\MMD_Analyse\yolo26n.pt"

REAL_A_FOLDER = Path(r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\RealWorld_500Images_1")

IMAGE_SIZE = 960

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


# Der letzte Layer ist normalerweise der Detection Head.
# Wir greifen die Eingabe dieses Detection Heads ab.
detect_layer = model.model.model[-1]
detect_layer.register_forward_pre_hook(hook_detect_input)

print("Feature hook registered successfully.")


# =========================
# TESTBILD LADEN
# =========================

image_paths = [
    p for p in REAL_A_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if len(image_paths) == 0:
    raise RuntimeError("No images found in Real A folder.")

test_image_path = image_paths[0]
print("Test image:", test_image_path.name)


# =========================
# PREPROCESSING
# =========================

img = cv2.imread(str(test_image_path))

if img is None:
    raise RuntimeError(f"Could not read image: {test_image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Wichtig: alle Bilder werden auf dieselbe Eingabegröße gebracht
img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

img = img.astype(np.float32) / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
img = img.to(device)


# =========================
# FORWARD PASS
# =========================

with torch.no_grad():
    _ = model.model(img)


# =========================
# FEATURE VECTOR ERZEUGEN
# =========================

if captured_features is None:
    raise RuntimeError("No features were captured. The hook did not work.")

print("Captured feature type:", type(captured_features))

pooled_features = []

if isinstance(captured_features, (list, tuple)):
    print("Number of feature maps:", len(captured_features))

    for i, feature_map in enumerate(captured_features):
        print(f"Feature map {i} shape:", feature_map.shape)

        # Global Average Pooling:
        # [Batch, Channels, Height, Width] -> [Batch, Channels]
        pooled = feature_map.mean(dim=(2, 3))
        pooled_features.append(pooled)

    # Mehrere Feature Maps werden zu einem Vektor zusammengefügt
    feature_vector = torch.cat(pooled_features, dim=1)

else:
    print("Single feature map shape:", captured_features.shape)

    # Falls nur eine Feature Map vorliegt
    feature_vector = captured_features.mean(dim=(2, 3))


feature_vector = feature_vector.squeeze(0).cpu().numpy()

print("Feature vector shape:", feature_vector.shape)
print("First 10 values:")
print(feature_vector[:10])
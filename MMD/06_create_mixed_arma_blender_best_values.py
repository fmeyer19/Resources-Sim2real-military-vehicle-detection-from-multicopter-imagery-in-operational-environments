from pathlib import Path
import random
import shutil


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(r"<path to base directory>")

ARMA_BEST_FOLDER = BASE_DIR / "Arma_AllMixed_Best_Values"
BLENDER_BEST_FOLDER = BASE_DIR / "Blender_AllMixed_Best_Values"

OUTPUT_FOLDER = BASE_DIR / "Mixed_Arma_Blender_Best_Values"

RANDOM_SEED = 42

N_ARMA = 250
N_BLENDER = 250

CLEAR_OUTPUT_FOLDER = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# =========================
# FUNCTIONS
# =========================

def get_images(folder: Path):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def copy_sampled_images(source_images, n_images, prefix, output_folder):
    if len(source_images) < n_images:
        raise ValueError(
            f"Not enough images available: {len(source_images)} found, "
            f"{n_images} required."
        )

    sampled = random.sample(source_images, n_images)

    for idx, src in enumerate(sampled, start=1):
        dst_name = f"{prefix}_{idx:04d}_{src.name}"
        dst = output_folder / dst_name
        shutil.copy2(src, dst)

    return sampled


# =========================
# MAIN
# =========================

random.seed(RANDOM_SEED)

if CLEAR_OUTPUT_FOLDER and OUTPUT_FOLDER.exists():
    shutil.rmtree(OUTPUT_FOLDER)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

arma_images = get_images(ARMA_BEST_FOLDER)
blender_images = get_images(BLENDER_BEST_FOLDER)

print(f"Arma best-value images found: {len(arma_images)}")
print(f"Blender best-value images found: {len(blender_images)}")

copy_sampled_images(
    source_images=arma_images,
    n_images=N_ARMA,
    prefix="arma",
    output_folder=OUTPUT_FOLDER
)

copy_sampled_images(
    source_images=blender_images,
    n_images=N_BLENDER,
    prefix="blender",
    output_folder=OUTPUT_FOLDER
)

final_images = get_images(OUTPUT_FOLDER)

print(f"\nNew dataset created: {OUTPUT_FOLDER}")
print(f"Total number of images: {len(final_images)}")
print(f"Arma share: {N_ARMA}")
print(f"Blender share: {N_BLENDER}")
from pathlib import Path
import shutil
import random
import csv


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(r"<path to base directory>")

RANDOM_SEED = 42
TARGET_IMAGE_COUNT = 500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Safety option:
# False = the script aborts if the target folder already contains images
# True  = the target folder is emptied beforehand
CLEAR_OUTPUT_FOLDERS = False


# =========================
# BEST-VALUE SOURCE FOLDERS
# =========================

ALLMIXED_CONFIGS = {
    "Arma_AllMixed_Best_Values": {
        "compression_artefacts_value_a_20": BASE_DIR / "Arma_Compression_Artefacts_value_a=20",
        "contrast_value_a_0_2": BASE_DIR / "Arma_Contrast_value_0.2",
        "dark_smoke_value_b_1_5": BASE_DIR / "Arma_Dark_Smoke_value_b=1.5",
        "gaussian_blur_value_c_2_5": BASE_DIR / "Arma_Gaussian_Blur_value_c=2.5",
        "gaussian_noise_color_value_b_15": BASE_DIR / "Arma_Gaussian_Noise_Color_value_b=15",
        "gaussian_noise_grey_value_b_15": BASE_DIR / "Arma_Gaussian_Noise_Grey_value_b=15",
        "saturation_value_a_0_2": BASE_DIR / "Arma_Saturation_value_a=0.2",
        "white_fog_value_a_1": BASE_DIR / "Arma_White_Fog_value_a=1",
    },

    "Blender_AllMixed_Best_Values": {
        "compression_artefacts_value_b_50": BASE_DIR / "Blender_Compression_Artefacts_value_b=50",
        "contrast_value_b_0_8": BASE_DIR / "Blender_Contrast_value_b=0.8",
        "saturation_value_a_0_2": BASE_DIR / "Blender_Saturation_value_a=0.2",
    },

    "Mixed_AllMixed_Best_Values": {
        "compression_artefacts_value_a_20": BASE_DIR / "Mixed_Compression_Artefacts_value_a=20",
        "contrast_value_b_0_8": BASE_DIR / "Mixed_Contrast_value_b=0.8",
        "dark_smoke_value_a_1": BASE_DIR / "Mixed_Dark_Smoke_value_a=1",
        "gaussian_blur_value_b_1_5": BASE_DIR / "Mixed_Gaussian_Blur_value_b=1.5",
        "gaussian_noise_color_value_a_5": BASE_DIR / "Mixed_Gaussian_Noise_Color_value_a=5",
        "gaussian_noise_grey_value_a_5": BASE_DIR / "Mixed_Gaussian_Noise_Grey_value_a=5",
        "saturation_value_a_0_2": BASE_DIR / "Mixed_Saturation_value_a=0.2",
        "white_fog_value_a_1": BASE_DIR / "Mixed_White_Fog_value_a=1",
    },
}


# =========================
# FUNCTIONS
# =========================

def get_images(folder: Path):
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def prepare_output_folder(output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)

    existing_images = get_images(output_folder)

    if existing_images and not CLEAR_OUTPUT_FOLDERS:
        raise RuntimeError(
            f"Target folder already contains images: {output_folder}\n"
            f"Either empty the folder or set CLEAR_OUTPUT_FOLDERS = True."
        )

    if CLEAR_OUTPUT_FOLDERS:
        for file in output_folder.iterdir():
            if file.is_file():
                file.unlink()


def distribute_counts(total_count: int, number_of_sources: int):
    """
    Distributes 500 images as evenly as possible across the selected methods.
    Example with 8 methods: 63, 63, 63, 63, 62, 62, 62, 62
    Example with 3 methods: 167, 167, 166
    """
    base = total_count // number_of_sources
    remainder = total_count % number_of_sources

    counts = []

    for i in range(number_of_sources):
        count = base + (1 if i < remainder else 0)
        counts.append(count)

    return counts


def create_allmixed_dataset(output_name: str, source_folders: dict, rng: random.Random):
    output_folder = BASE_DIR / output_name
    prepare_output_folder(output_folder)

    log_rows = []

    source_items = list(source_folders.items())
    counts = distribute_counts(TARGET_IMAGE_COUNT, len(source_items))

    copied_total = 0

    print("\n========================================")
    print(f"Creating: {output_name}")
    print("========================================")

    for (method_name, source_folder), count in zip(source_items, counts):
        if not source_folder.exists():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        images = get_images(source_folder)

        if len(images) < count:
            raise RuntimeError(
                f"Not enough images in {source_folder}. "
                f"Required: {count}, found: {len(images)}"
            )

        selected_images = rng.sample(images, count)

        print(f"{method_name:40s} -> {count:3d} images")

        for image_path in selected_images:
            copied_total += 1

            new_name = f"{copied_total:04d}_{method_name}_{image_path.name}"
            target_path = output_folder / new_name

            shutil.copy2(image_path, target_path)

            log_rows.append({
                "output_dataset": output_name,
                "output_file": target_path.name,
                "source_method": method_name,
                "source_folder": str(source_folder),
                "source_file": image_path.name,
                "source_path": str(image_path),
            })

    log_path = BASE_DIR / f"{output_name}_creation_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "output_dataset",
            "output_file",
            "source_method",
            "source_folder",
            "source_file",
            "source_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(log_rows)

    final_count = len(get_images(output_folder))

    print(f"\nFinished: {output_name}")
    print(f"Images in target folder: {final_count}")
    print(f"Log saved: {log_path}")

    if final_count != TARGET_IMAGE_COUNT:
        raise RuntimeError(
            f"Error: {output_name} contains {final_count} images instead of {TARGET_IMAGE_COUNT}."
        )


# =========================
# MAIN
# =========================

rng = random.Random(RANDOM_SEED)

for output_name, source_folders in ALLMIXED_CONFIGS.items():
    create_allmixed_dataset(output_name, source_folders, rng)

print("\nAll AllMixed_Best_Values datasets have been created.")
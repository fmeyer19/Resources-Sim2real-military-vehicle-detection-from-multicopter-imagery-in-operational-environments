"""
Creates a separately augmented copy of every input image for each enabled
GeneticAugment operation and copies the corresponding YOLO annotation under
the same new file stem.

Properties:
- A strength range is given per operation.
- A random strength value is drawn from that range on every application.
- Each operation is applied independently to the unchanged original image.
- The operations are not combined with one another.
- The original images and annotations are left unchanged.
- Output images and output labels are written to a subfolder named "output".
- Image and label always receive the same file stem.
- The name states the operation and the strength value actually used.
- Roboflow suffixes from "_png.rf" onwards are removed from the output name.
- On a name conflict, (1), (2), ... is inserted before the augmentation suffix.
- The one-to-one pairing of images and labels is checked before and after the run.

This script has to be placed inside the GeneticAugment repository of Vanherle
et al., since it imports get_augmentation from there. That repository is not
redistributed with this thesis.

The assignment of images to blocks is not part of this script. Each block was
processed in a separate run with exactly one operation enabled.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


# =============================================================================
# PATHS
# =============================================================================

GENETIC_AUGMENT_REPO_DIR = Path(
    r"<path to the GeneticAugment repository "
    r"of Vanherle et al.>"
)

INPUT_IMAGES_DIR = Path(
    r"<path to the input images>"
)

INPUT_LABELS_DIR = Path(
    r"<path to the input labels>"
)

OUTPUT_IMAGES_DIR = INPUT_IMAGES_DIR / "output"
OUTPUT_LABELS_DIR = INPUT_LABELS_DIR / "output"


# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================

AUGMENTATIONS: dict[str, dict[str, bool | tuple[float, float]]] = {
    # Blur and defocus
    "gaussian_blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "defocus": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "glass_blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "median_blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "motion_blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "zoom_blur": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },

    # Color and intensity
    "channel_shuffle": {
        "enabled": True,
        "strength_range": (2.0, 2.0),   # the strength has no effect on the result
    },
    "clahe": {
        "enabled": True,
        "strength_range": (4.0, 8.0),
    },
    "brightness": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "contrast": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "saturation": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "hue": {
        "enabled": True,
        "strength_range": (2.0, 14.0),
    },
    "equalize": {
        "enabled": True,
        "strength_range": (2.0, 2.0),   # the strength has no effect on the result 
    },
    "fancy_pca": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "invert": {
        "enabled": True,
        "strength_range": (2.0, 2.0),   # the strength has no effect on the result
    },
    "posterize": {
        "enabled": True,
        "strength_range": (1.0, 1.5),
    },
    "gamma": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "tone_curve": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },

    # Edge and sharpness
    "emboss": {
        "enabled": True,
        "strength_range": (2.0, 2.0),   # the strength barely affects the result
    },
    "sharpen": {
        "enabled": True,
        "strength_range": (1.0, 2.0),
    },
    "unsharp_mask": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },

    # Noise and pixel degradation
    "gauss_noise": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "iso_noise": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "multiplicative_noise": {
        "enabled": True,
        "strength_range": (1.5, 2.0),
    },
    "dropout": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
    "uniform_noise": {
        "enabled": False,
        "strength_range": (0.5, 1.5),
    },
}


# =============================================================================
# FURTHER SETTINGS
# =============================================================================

USE_FIXED_SEED = False
RANDOM_SEED = 42
VARIANTS_PER_METHOD = 1
OVERWRITE_EXISTING = False

# Corresponds to the earlier setting CLEAR_OUTPUT_DIRECTORY, but now clears
# both output folders.
CLEAR_OUTPUT_DIRECTORIES = False

MAX_LISTED_PROBLEMS = 50

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


# =============================================================================
# IMPORT OF THE ORIGINAL GENETICAUGMENT FUNCTION
# =============================================================================

if not GENETIC_AUGMENT_REPO_DIR.is_dir():
    raise NotADirectoryError(
        "The GeneticAugment repository folder was not found:\n"
        f"{GENETIC_AUGMENT_REPO_DIR}"
    )

if not (GENETIC_AUGMENT_REPO_DIR / "augmentation").is_dir():
    raise NotADirectoryError(
        "The folder 'augmentation' is missing in the given repository path:\n"
        f"{GENETIC_AUGMENT_REPO_DIR}"
    )

repository_path = str(GENETIC_AUGMENT_REPO_DIR.resolve())
if repository_path not in sys.path:
    sys.path.insert(0, repository_path)

from augmentation import COLLECTION
from augmentation.loading import get_augmentation

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DatasetAudit:
    image_paths: list[Path]
    label_paths: list[Path]
    image_map: dict[str, list[Path]]
    label_map: dict[str, list[Path]]
    images_without_labels: list[Path]
    labels_without_images: list[Path]
    duplicate_image_stems: dict[str, list[Path]]
    duplicate_label_stems: dict[str, list[Path]]

    @property
    def has_complete_one_to_one_pairing(self) -> bool:
        return (
            not self.images_without_labels
            and not self.labels_without_images
            and not self.duplicate_image_stems
            and not self.duplicate_label_stems
            and len(self.image_paths) == len(self.label_paths)
        )


# =============================================================================
# CHECKS AND FILE NAMES
# =============================================================================

def normalize_stem(path: Path) -> str:
    return path.stem.casefold()


def get_output_base_stem(original_stem: str) -> str:
    """
    Removes "_png.rf" and everything after it from Roboflow file names.

    imagename_png.rf.abc123 -> imagename
    """
    marker = "_png.rf"
    marker_index = original_stem.casefold().find(marker)

    if marker_index == -1:
        return original_stem

    cleaned_stem = original_stem[:marker_index]
    if not cleaned_stem:
        raise ValueError(
            f"The file stem '{original_stem}' contains no usable name before '{marker}' "
            "and cannot be used as an output name."
        )

    return cleaned_stem


def collect_dataset_audit(
    image_folder: Path,
    label_folder: Path,
) -> DatasetAudit:
    image_paths = sorted(
        (
            path
            for path in image_folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )

    label_paths = sorted(
        (
            path
            for path in label_folder.iterdir()
            if path.is_file() and path.suffix.casefold() == ".txt"
        ),
        key=lambda path: path.name.casefold(),
    )

    image_map_temp: dict[str, list[Path]] = defaultdict(list)
    label_map_temp: dict[str, list[Path]] = defaultdict(list)

    for path in image_paths:
        image_map_temp[normalize_stem(path)].append(path)

    for path in label_paths:
        label_map_temp[normalize_stem(path)].append(path)

    image_map = dict(image_map_temp)
    label_map = dict(label_map_temp)

    images_without_labels = [
        path
        for stem, paths in image_map.items()
        if stem not in label_map
        for path in paths
    ]

    labels_without_images = [
        path
        for stem, paths in label_map.items()
        if stem not in image_map
        for path in paths
    ]

    duplicate_image_stems = {
        stem: paths for stem, paths in image_map.items() if len(paths) > 1
    }
    duplicate_label_stems = {
        stem: paths for stem, paths in label_map.items() if len(paths) > 1
    }

    return DatasetAudit(
        image_paths=image_paths,
        label_paths=label_paths,
        image_map=image_map,
        label_map=label_map,
        images_without_labels=images_without_labels,
        labels_without_images=labels_without_images,
        duplicate_image_stems=duplicate_image_stems,
        duplicate_label_stems=duplicate_label_stems,
    )


def print_limited_paths(title: str, paths: list[Path]) -> None:
    if not paths:
        return

    print(f"\n{title} ({len(paths)}):")
    for path in paths[:MAX_LISTED_PROBLEMS]:
        print(f"    - {path.name}")

    hidden_count = len(paths) - MAX_LISTED_PROBLEMS
    if hidden_count > 0:
        print(f"    ... and {hidden_count} further file(s)")


def print_duplicate_stems(
    title: str,
    duplicates: dict[str, list[Path]],
) -> None:
    if not duplicates:
        return

    print(f"\n{title} ({len(duplicates)}):")
    shown = 0

    for stem, paths in sorted(duplicates.items()):
        if shown >= MAX_LISTED_PROBLEMS:
            break
        print(f"    - '{stem}': {', '.join(path.name for path in paths)}")
        shown += 1

    hidden_count = len(duplicates) - shown
    if hidden_count > 0:
        print(f"    ... and {hidden_count} further file stems")


def print_audit(title: str, audit: DatasetAudit) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Images:                       {len(audit.image_paths)}")
    print(f"Annotation files (.txt):      {len(audit.label_paths)}")
    print(f"Images without annotation:    {len(audit.images_without_labels)}")
    print(f"Annotations without image:    {len(audit.labels_without_images)}")
    print(f"Duplicate image stems:        {len(audit.duplicate_image_stems)}")
    print(f"Duplicate label stems:        {len(audit.duplicate_label_stems)}")
    print(
        "Complete one-to-one pairing:  "
        f"{'YES' if audit.has_complete_one_to_one_pairing else 'NO'}"
    )

    print_limited_paths(
        "Images without a matching annotation",
        audit.images_without_labels,
    )
    print_limited_paths(
        "Annotations without a matching image",
        audit.labels_without_images,
    )
    print_duplicate_stems(
        "Several images with the same file stem",
        audit.duplicate_image_stems,
    )
    print_duplicate_stems(
        "Several annotations with the same file stem",
        audit.duplicate_label_stems,
    )


def output_stem_exists(candidate_stem: str) -> bool:
    if (OUTPUT_LABELS_DIR / f"{candidate_stem}.txt").exists():
        return True

    return any(
        (OUTPUT_IMAGES_DIR / f"{candidate_stem}{extension}").exists()
        for extension in IMAGE_EXTENSIONS
    )


def get_available_output_stem(
    output_base_stem: str,
    augmentation_suffix: str,
) -> str:
    """
    Example:
        image_hue_s4.123
        image(1)_hue_s4.123
        image(2)_hue_s4.123
    """
    candidate_stem = f"{output_base_stem}_{augmentation_suffix}"

    if OVERWRITE_EXISTING or not output_stem_exists(candidate_stem):
        return candidate_stem

    counter = 1
    while True:
        candidate_stem = (
            f"{output_base_stem}({counter})_{augmentation_suffix}"
        )
        if not output_stem_exists(candidate_stem):
            return candidate_stem
        counter += 1


# =============================================================================
# CONFIGURATION AND RANDOMNESS
# =============================================================================

def validate_configuration() -> list[tuple[str, float, float]]:
    if not INPUT_IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {INPUT_IMAGES_DIR}"
        )

    if not INPUT_LABELS_DIR.is_dir():
        raise FileNotFoundError(
            f"Annotation folder not found: {INPUT_LABELS_DIR}"
        )

    if VARIANTS_PER_METHOD < 1:
        raise ValueError("VARIANTS_PER_METHOD has to be at least 1.")

    active_augmentations: list[tuple[str, float, float]] = []

    for name, setting in AUGMENTATIONS.items():
        if name not in COLLECTION:
            raise ValueError(
                f"The operation '{name}' is not part of the GeneticAugment "
                "search space."
            )

        enabled = setting.get("enabled")
        strength_range = setting.get("strength_range")

        if not isinstance(enabled, bool):
            raise TypeError(
                f"'enabled' has to be True or False for '{name}'."
            )

        if (
            not isinstance(strength_range, tuple)
            or len(strength_range) != 2
        ):
            raise TypeError(
                f"'strength_range' for '{name}' has to be a tuple of two "
                "numbers."
            )

        minimum, maximum = strength_range

        if not isinstance(minimum, (int, float)):
            raise TypeError(
                f"The minimum value of '{name}' has to be numeric."
            )
        if not isinstance(maximum, (int, float)):
            raise TypeError(
                f"The maximum value of '{name}' has to be numeric."
            )

        minimum = float(minimum)
        maximum = float(maximum)

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(
                f"The strength range of '{name}' has to be finite."
            )
        if minimum <= 0 or maximum <= 0:
            raise ValueError(
                f"The strength range of '{name}' has to be greater than 0."
            )
        if minimum > maximum:
            raise ValueError(
                f"For '{name}' the minimum value exceeds the maximum value."
            )

        if enabled:
            active_augmentations.append((name, minimum, maximum))

    if not active_augmentations:
        raise RuntimeError("No augmentation operation is enabled.")

    return active_augmentations


def prepare_output_directories() -> None:
    if CLEAR_OUTPUT_DIRECTORIES:
        if OUTPUT_IMAGES_DIR.exists():
            shutil.rmtree(OUTPUT_IMAGES_DIR)
        if OUTPUT_LABELS_DIR.exists():
            shutil.rmtree(OUTPUT_LABELS_DIR)

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def initialize_random_generator() -> random.Random:
    if USE_FIXED_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        if torch is not None:
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)

        return random.Random(RANDOM_SEED)

    return random.Random()


def draw_strength(
    rng: random.Random,
    minimum: float,
    maximum: float,
) -> float:
    if minimum == maximum:
        return minimum
    return rng.uniform(minimum, maximum)


def format_strength_for_filename(strength: float) -> str:
    return f"{strength:.3f}"


# =============================================================================
# IMAGE PROCESSING AND PAIRED SAVING
# =============================================================================

def normalize_augmented_image(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        if "image" not in result:
            raise RuntimeError(
                "The augmentation returned a dictionary without 'image'."
            )
        result = result["image"]

    if isinstance(result, Image.Image):
        result = np.asarray(result.convert("RGB"))

    if torch is not None and isinstance(result, torch.Tensor):
        tensor = result.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        result = tensor.numpy()

    array = np.asarray(result)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 4:
        array = array[:, :, :3]

    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(
            f"Unexpected output shape of the augmentation: {array.shape}"
        )

    if array.dtype != np.uint8:
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        if np.issubdtype(array.dtype, np.floating):
            maximum = float(np.max(array)) if array.size else 0.0
            if maximum <= 1.0:
                array = array * 255.0

        array = np.clip(array, 0, 255).round().astype(np.uint8)

    return array


def save_image(image_array: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(image_array, mode="RGB")
    extension = output_path.suffix.casefold()

    if extension in {".jpg", ".jpeg"}:
        image.save(
            output_path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
    elif extension == ".png":
        image.save(output_path, format="PNG")
    elif extension == ".bmp":
        image.save(output_path, format="BMP")
    elif extension in {".tif", ".tiff"}:
        image.save(output_path, format="TIFF")
    elif extension == ".webp":
        image.save(output_path, format="WEBP", quality=95)
    else:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )


def create_augmented_pair(
    original_array: np.ndarray,
    source_image_path: Path,
    source_label_path: Path,
    augmentation_name: str,
    strength: float,
    variant_index: int,
) -> tuple[Path, Path]:
    """
    Creates the augmented image and copies the unchanged YOLO label.

    Both files are written to a temporary name first. On an error this prevents
    a single image or label from being left behind without its counterpart.
    """
    strength_text = format_strength_for_filename(strength)

    variant_suffix = ""
    if VARIANTS_PER_METHOD > 1:
        variant_suffix = f"_v{variant_index:02d}"

    augmentation_suffix = (
        f"{augmentation_name}_s{strength_text}{variant_suffix}"
    )

    output_base_stem = get_output_base_stem(source_image_path.stem)
    output_stem = get_available_output_stem(
        output_base_stem,
        augmentation_suffix,
    )

    output_image_path = OUTPUT_IMAGES_DIR / (
        f"{output_stem}{source_image_path.suffix}"
    )
    output_label_path = OUTPUT_LABELS_DIR / f"{output_stem}.txt"

    unique_id = uuid.uuid4().hex
    temporary_image_path = OUTPUT_IMAGES_DIR / (
        f".{unique_id}_image_tmp{source_image_path.suffix}"
    )
    temporary_label_path = OUTPUT_LABELS_DIR / (
        f".{unique_id}_label_tmp.txt"
    )

    image_committed = False
    label_committed = False

    try:
        augmenter = get_augmentation(
            name=augmentation_name,
            strength=strength,
            probability=1.0,
        )

        # Albumentations is called with a named image argument.
        result = augmenter(image=original_array.copy())
        augmented_array = normalize_augmented_image(result)

        save_image(augmented_array, temporary_image_path)
        shutil.copy2(source_label_path, temporary_label_path)

        os.replace(temporary_image_path, output_image_path)
        image_committed = True

        os.replace(temporary_label_path, output_label_path)
        label_committed = True

        if output_image_path.stem.casefold() != output_label_path.stem.casefold():
            raise RuntimeError(
                "Image and annotation have different file stems."
            )

        return output_image_path, output_label_path

    except Exception:
        for temporary_path in (
            temporary_image_path,
            temporary_label_path,
        ):
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass

        if image_committed and not label_committed:
            try:
                if output_image_path.exists():
                    output_image_path.unlink()
            except OSError:
                pass

        if label_committed and not image_committed:
            try:
                if output_label_path.exists():
                    output_label_path.unlink()
            except OSError:
                pass

        raise


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    active_augmentations = validate_configuration()
    prepare_output_directories()
    rng = initialize_random_generator()

    input_audit = collect_dataset_audit(
        INPUT_IMAGES_DIR,
        INPUT_LABELS_DIR,
    )
    print_audit(
        "INPUT DATASET BEFORE AUGMENTATION",
        input_audit,
    )

    if not input_audit.image_paths:
        print("\nNo supported images were found.")
        return

    maximum_outputs = (
        len(input_audit.image_paths)
        * len(active_augmentations)
        * VARIANTS_PER_METHOD
    )

    created = 0
    skipped = 0
    failed = 0

    print("\n" + "=" * 72)
    print("GENETICAUGMENT OPERATION EVALUATION")
    print("=" * 72)
    print(f"Input images:       {len(input_audit.image_paths)}")
    print(f"Enabled operations: {len(active_augmentations)}")
    print(f"Variants/operation: {VARIANTS_PER_METHOD}")
    print(f"Maximum outputs:    {maximum_outputs}")
    print(f"Output images:      {OUTPUT_IMAGES_DIR}")
    print(f"Output labels:      {OUTPUT_LABELS_DIR}")

    for image_index, image_path in enumerate(
        input_audit.image_paths,
        start=1,
    ):
        stem = normalize_stem(image_path)
        matching_images = input_audit.image_map.get(stem, [])
        matching_labels = input_audit.label_map.get(stem, [])

        if len(matching_images) != 1:
            print(
                f"\nSKIPPED: '{image_path.name}' does not have a unique "
                "image file stem."
            )
            skipped += len(active_augmentations) * VARIANTS_PER_METHOD
            continue

        if len(matching_labels) != 1:
            print(
                f"\nSKIPPED: for '{image_path.name}' no unique annotation was "
                "found."
            )
            skipped += len(active_augmentations) * VARIANTS_PER_METHOD
            continue

        label_path = matching_labels[0]

        try:
            with Image.open(image_path) as opened_image:
                opened_image = ImageOps.exif_transpose(opened_image)
                opened_image.load()
                original_array = np.asarray(
                    opened_image.convert("RGB")
                ).copy()
        except Exception as error:
            failed += len(active_augmentations) * VARIANTS_PER_METHOD
            print(
                f"\n[LOADING ERROR]\n"
                f"Image: {image_path}\n"
                f"{type(error).__name__}: {error}"
            )
            continue

        for augmentation_name, minimum, maximum in active_augmentations:
            for variant_index in range(1, VARIANTS_PER_METHOD + 1):
                strength = draw_strength(
                    rng,
                    minimum,
                    maximum,
                )

                try:
                    output_image_path, output_label_path = (
                        create_augmented_pair(
                            original_array=original_array,
                            source_image_path=image_path,
                            source_label_path=label_path,
                            augmentation_name=augmentation_name,
                            strength=strength,
                            variant_index=variant_index,
                        )
                    )

                    created += 1
                    print(
                        f"OK: {image_path.name} + {label_path.name} -> "
                        f"{output_image_path.name} + "
                        f"{output_label_path.name}"
                    )

                except Exception as error:
                    failed += 1
                    print(
                        f"\n[AUGMENTATION ERROR]\n"
                        f"Image:     {image_path.name}\n"
                        f"Label:     {label_path.name}\n"
                        f"Operation: {augmentation_name}\n"
                        f"Strength:  {strength:.6f}\n"
                        f"{type(error).__name__}: {error}"
                    )

        print(
            f"{image_index}/{len(input_audit.image_paths)} images checked | "
            f"{created} pairs created | "
            f"{skipped} skipped | "
            f"{failed} failed"
        )

    output_audit = collect_dataset_audit(
        OUTPUT_IMAGES_DIR,
        OUTPUT_LABELS_DIR,
    )
    print_audit(
        "OUTPUT DATASET AFTER AUGMENTATION",
        output_audit,
    )

    print("\n" + "=" * 72)
    print("FINAL STATISTICS")
    print("=" * 72)
    print(f"Newly created image-label pairs: {created}")
    print(f"Skipped outputs:                 {skipped}")
    print(f"Failed outputs:                  {failed}")
    print(f"Images in the output folder:     {len(output_audit.image_paths)}")
    print(f"Labels in the output folder:     {len(output_audit.label_paths)}")
    print(
        "One-to-one pairing in output:    "
        f"{'YES' if output_audit.has_complete_one_to_one_pairing else 'NO'}"
    )
    print("\nThe original images and annotations were not modified.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 72)
        print("PROGRAM ABORTED")
        print("=" * 72)
        print(f"ERROR: {error}")
        sys.exit(1)

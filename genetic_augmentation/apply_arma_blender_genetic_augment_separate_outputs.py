"""
Produces two separate augmented YOLO datasets from the separate Arma and
Blender raw datasets, each with the policy learned for it.

Important:
- The raw datasets are read only and are never modified.
- augmentation_arma.pkl is used for Arma.
- augmentation_blender.pkl is used for Blender.
- The YOLO txt annotations are copied unchanged, since GeneticAugment uses
  non-geometric augmentations only.
- The file name states the augmentation entries that were actually selected,
  including their stored selection weights p and strengths s.
- Arma and Blender are written to separate output folders.
- Image and label always receive exactly the same file stem.
- A CSV file records the complete assignment.

This script has to be placed in the main folder of the GeneticAugment
repository of Vanherle et al., since it imports get_augmentation from there.
That repository is not redistributed with this thesis.
"""

from __future__ import annotations

import csv
import pickle
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# Original function of the GeneticAugment repository.
# It builds a single augmentation using the repository's own translation of
# the strength value into concrete Albumentations parameters.
GENETIC_AUGMENT_REPO_DIR = Path(
    r"<path to the GeneticAugment repository>"
)

repo_path_string = str(GENETIC_AUGMENT_REPO_DIR.resolve())

if repo_path_string not in sys.path:
    sys.path.insert(0, repo_path_string)

from augmentation.loading import get_augmentation


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================
# Adjust this section only to match your folder structure.

# ------------------------------
# Arma raw dataset
# ------------------------------
ARMA_IMAGES_DIR = Path(
    r"<path to the Arma raw train images>"
)
ARMA_LABELS_DIR = Path(
    r"<path to the Arma raw train labels>"
)
ARMA_POLICY_PATH = Path(
    r"<path to augmentation_arma.pkl>"
)

# ------------------------------
# Blender raw dataset
# ------------------------------
BLENDER_IMAGES_DIR = Path(
    r"<path to the Blender raw train images>"
)
BLENDER_LABELS_DIR = Path(
    r"<path to the Blender raw train labels>"
)
BLENDER_POLICY_PATH = Path(
    r"<path to augmentation_blender.pkl>"
)

# ------------------------------
# Separate output folders for Arma
# ------------------------------
ARMA_OUTPUT_IMAGES_DIR = Path(
    r"<path to the Arma output images>"
)
ARMA_OUTPUT_LABELS_DIR = Path(
    r"<path to the Arma output labels>"
)
ARMA_MANIFEST_PATH = (
    ARMA_OUTPUT_IMAGES_DIR.parent / "genetic_augment_manifest_arma.csv"
)

# ------------------------------
# Separate output folders for Blender
# ------------------------------
BLENDER_OUTPUT_IMAGES_DIR = Path(
    r"<path to the Blender output images>"
)
BLENDER_OUTPUT_LABELS_DIR = Path(
    r"<path to the Blender output labels>"
)
BLENDER_MANIFEST_PATH = (
    BLENDER_OUTPUT_IMAGES_DIR.parent / "genetic_augment_manifest_blender.csv"
)


# Reproducibility:
# Identical input data, policies and seeds produce the same selections and the
# same random sequences.
ARMA_RANDOM_SEED = 42
BLENDER_RANDOM_SEED = 43

# Supported input formats.
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# True: existing output files may be overwritten.
# False: on a name conflict, (1), (2), ... is inserted automatically.
OVERWRITE_EXISTING = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class SelectedAugmentation:
    """An entry actually drawn from a GeneticAugment policy."""

    index: int
    name: str
    probability: float
    strength: float

    def filename_token(self) -> str:
        """
        Builds a Windows-compatible file name component.

        Example:
        defocus_p0p75_s1p52
        """
        safe_name = sanitize_filename_component(self.name)
        p_text = format_number_for_filename(self.probability)
        s_text = format_number_for_filename(self.strength)
        return f"{safe_name}_p{p_text}_s{s_text}"

    def readable_description(self) -> str:
        """Human-readable form for the CSV record."""
        return (
            f"{self.name}(index={self.index}, "
            f"p={self.probability:.6f}, s={self.strength:.6f})"
        )


@dataclass(frozen=True)
class DatasetSource:
    """Configuration of one synthetic source domain."""

    name: str
    filename_prefix: str
    images_dir: Path
    labels_dir: Path
    policy_path: Path
    output_images_dir: Path
    output_labels_dir: Path
    manifest_path: Path
    random_seed: int


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_filename_component(value: str) -> str:
    """
    Removes characters that are problematic in Windows file names.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def format_number_for_filename(value: float) -> str:
    """
    Formats a parameter compactly and safely for a file name.

    Examples:
    0.75 -> 0p75
    1.0  -> 1p00
    """
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def validate_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(
            f"{description} was not found or is not a folder:\n{path}"
        )


def validate_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def find_images(images_dir: Path) -> list[Path]:
    """Searches recursively for all supported image files."""
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"No supported images were found in the image folder:\n"
            f"{images_dir}"
        )

    return images


def build_label_index(labels_dir: Path) -> dict[str, list[Path]]:
    """
    Indexes the labels by file stem.

    This supports both flat label folders and nested subfolders.
    """
    index: dict[str, list[Path]] = {}

    for label_path in labels_dir.rglob("*.txt"):
        if label_path.is_file():
            index.setdefault(label_path.stem, []).append(label_path)

    return index


def find_matching_label(
    image_path: Path,
    images_dir: Path,
    labels_dir: Path,
    label_index: dict[str, list[Path]],
) -> Path:
    """
    Determines the matching YOLO annotation.

    The same relative subfolder structure is tried first. If no file is found
    there, a search by identical file stem follows.
    """
    relative_image_path = image_path.relative_to(images_dir)
    expected_relative_label = (
        labels_dir / relative_image_path.with_suffix(".txt")
    )

    if expected_relative_label.is_file():
        return expected_relative_label

    candidates = label_index.get(image_path.stem, [])

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No matching label file was found for:\n{image_path}"
        )

    candidate_text = "\n".join(str(path) for path in candidates)
    raise RuntimeError(
        f"Several labels share the file stem "
        f"'{image_path.stem}'. An unambiguous assignment is not possible:\n"
        f"{candidate_text}"
    )


def load_setting(policy_path: Path):
    """
    Loads a serialized GeneticAugment policy.
    """
    validate_file(policy_path, "PKL policy file")

    with policy_path.open("rb") as file:
        setting = pickle.load(file)

    required_attributes = (
        "augmentations",
        "strengths",
        "probabilities",
        "pick_n",
    )
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(setting, attribute)
    ]

    if missing:
        raise TypeError(
            f"The PKL file does not contain a valid AugmentationSetting structure. "
            f"Missing attributes: {missing}"
        )

    if setting.pick_n is None:
        raise RuntimeError(
            f"The policy {policy_path.name} was not stored in pick-n mode. "
            f"Pick-2 in particular was expected."
        )

    if int(setting.pick_n) < 1:
        raise RuntimeError(
            f"Invalid pick_n value in {policy_path.name}: {setting.pick_n}"
        )

    # This application script is written for the policies that were actually
    # learned, which use nested-size=1.
    if any(isinstance(item, list) for item in setting.augmentations):
        raise NotImplementedError(
            f"{policy_path.name} contains a nested policy. "
            f"This script expects nested-size=1."
        )

    lengths = {
        len(setting.augmentations),
        len(setting.strengths),
        len(setting.probabilities),
    }
    if len(lengths) != 1:
        raise RuntimeError(
            f"Operations, strengths and probabilities in "
            f"{policy_path.name} have different lengths."
        )

    if not setting.augmentations:
        raise RuntimeError(
            f"The policy {policy_path.name} contains no augmentations."
        )

    if sum(float(p) for p in setting.probabilities) <= 0:
        raise RuntimeError(
            f"The selection weights in {policy_path.name} do not add up to a "
            f"valid positive sum."
        )

    return setting


def select_policy_entries(setting) -> list[SelectedAugmentation]:
    """
    Reproduces the pick-n mechanism of the original repository.

    The draw uses random.choices and is therefore performed with replacement.
    The same policy entry can consequently be selected more than once.
    """
    indices = random.choices(
        population=range(len(setting.augmentations)),
        weights=[float(p) for p in setting.probabilities],
        k=int(setting.pick_n),
    )

    return [
        SelectedAugmentation(
            index=index,
            name=str(setting.augmentations[index]),
            probability=float(setting.probabilities[index]),
            strength=float(setting.strengths[index]),
        )
        for index in indices
    ]


def apply_selected_augmentations(
    image_array: np.ndarray,
    selections: Sequence[SelectedAugmentation],
) -> np.ndarray:
    """
    Applies the selected policy entries one after another.

    In pick-n mode, p serves as the selection weight. Once an entry is drawn,
    its augmentation is executed with probability=1.0, exactly as in the
    original function get_augmentation_policy().
    """
    result = image_array.copy()

    for selection in selections:
        augmenter = get_augmentation(
            name=selection.name,
            strength=selection.strength,
            probability=1.0,
        )
        result = augmenter(result)

        if not isinstance(result, np.ndarray):
            result = np.asarray(result)

    # Safeguard against grayscale or RGBA output.
    if result.ndim == 2:
        result = np.repeat(result[:, :, None], 3, axis=2)
    elif result.ndim == 3 and result.shape[2] == 1:
        result = np.repeat(result, 3, axis=2)
    elif result.ndim == 3 and result.shape[2] >= 4:
        result = result[:, :, :3]

    if result.dtype != np.uint8:
        if np.issubdtype(result.dtype, np.floating):
            maximum = float(np.nanmax(result)) if result.size else 0.0
            if maximum <= 1.0:
                result = result * 255.0

        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        result = np.clip(result, 0, 255).round().astype(np.uint8)

    return result


def build_output_stem(
    source_prefix: str,
    original_stem: str,
    selections: Sequence[SelectedAugmentation],
) -> str:
    """
    Builds the shared stem for image and label.

    Example:
    arma_scene01__defocus_p0p75_s1p52__gaussian_blur_p0p95_s0p11
    """
    safe_source = sanitize_filename_component(source_prefix)
    safe_original = sanitize_filename_component(original_stem)
    augmentation_suffix = "__".join(
        selection.filename_token() for selection in selections
    )
    return f"{safe_source}_{safe_original}__{augmentation_suffix}"


def make_unique_output_paths(
    desired_stem: str,
    image_suffix: str,
    overwrite: bool,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> tuple[Path, Path]:
    """
    Builds collision-free output paths.

    On a conflict, the number is inserted between the original stem and the
    augmentation suffix, for example:
    arma_scene01(1)__defocus_p...__blur_p...
    """
    image_suffix = image_suffix.lower()
    if image_suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        image_suffix = ".png"

    def paths_for(stem: str) -> tuple[Path, Path]:
        return (
            output_images_dir / f"{stem}{image_suffix}",
            output_labels_dir / f"{stem}.txt",
        )

    image_path, label_path = paths_for(desired_stem)

    if overwrite or (not image_path.exists() and not label_path.exists()):
        return image_path, label_path

    # Split off the augmentation suffix so that (1) is inserted before it.
    if "__" in desired_stem:
        base_stem, augmentation_suffix = desired_stem.split("__", 1)
        separator_suffix = f"__{augmentation_suffix}"
    else:
        base_stem = desired_stem
        separator_suffix = ""

    counter = 1
    while True:
        candidate_stem = (
            f"{base_stem}({counter}){separator_suffix}"
        )
        image_path, label_path = paths_for(candidate_stem)

        if not image_path.exists() and not label_path.exists():
            return image_path, label_path

        counter += 1


def save_rgb_image(image_array: np.ndarray, destination: Path) -> None:
    """
    Saves an RGB image, keeping the chosen file format.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_array, mode="RGB")

    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(
            destination,
            quality=95,
            subsampling=0,
        )
    elif suffix in {".tif", ".tiff"}:
        image.save(destination, compression="tiff_deflate")
    else:
        image.save(destination)


def initialize_random_generators(seed: int) -> None:
    """
    Initializes the Python and NumPy random generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def process_dataset(
    source: DatasetSource,
    manifest_writer: csv.DictWriter,
) -> dict[str, int]:
    """
    Augments one complete source domain and copies the labels.
    """
    print("\n" + "=" * 80)
    print(f"Processing source: {source.name}")
    print("=" * 80)
    print(f"Images:   {source.images_dir}")
    print(f"Labels:   {source.labels_dir}")
    print(f"Policy:   {source.policy_path}")

    validate_directory(source.images_dir, f"{source.name} image folder")
    validate_directory(source.labels_dir, f"{source.name} label folder")

    setting = load_setting(source.policy_path)
    initialize_random_generators(source.random_seed)

    print("\nLoaded policy:")
    print(setting)

    images = find_images(source.images_dir)
    label_index = build_label_index(source.labels_dir)

    statistics = {
        "images_found": len(images),
        "success": 0,
        "missing_or_invalid_labels": 0,
        "errors": 0,
    }

    for position, image_path in enumerate(images, start=1):
        try:
            label_path = find_matching_label(
                image_path=image_path,
                images_dir=source.images_dir,
                labels_dir=source.labels_dir,
                label_index=label_index,
            )

            selections = select_policy_entries(setting)

            with Image.open(image_path) as opened_image:
                source_rgb = opened_image.convert("RGB")
                source_array = np.asarray(source_rgb).copy()

            augmented_array = apply_selected_augmentations(
                image_array=source_array,
                selections=selections,
            )

            desired_stem = build_output_stem(
                source_prefix=source.filename_prefix,
                original_stem=image_path.stem,
                selections=selections,
            )

            output_image_path, output_label_path = make_unique_output_paths(
                desired_stem=desired_stem,
                image_suffix=image_path.suffix,
                overwrite=OVERWRITE_EXISTING,
                output_images_dir=source.output_images_dir,
                output_labels_dir=source.output_labels_dir,
            )

            # Write image and label to the output folders first.
            save_rgb_image(augmented_array, output_image_path)
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, output_label_path)

            # After writing, verify once more that the pair shares the same
            # file stem.
            if output_image_path.stem != output_label_path.stem:
                raise RuntimeError(
                    "Internal error: image and label have different file "
                    "stems."
                )

            manifest_writer.writerow(
                {
                    "source_domain": source.name,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "output_image": str(output_image_path),
                    "output_label": str(output_label_path),
                    "policy_file": str(source.policy_path),
                    "pick_n": int(setting.pick_n),
                    "selected_augmentations": " | ".join(
                        selection.readable_description()
                        for selection in selections
                    ),
                }
            )

            statistics["success"] += 1

        except FileNotFoundError as error:
            statistics["missing_or_invalid_labels"] += 1
            print(f"\n[LABEL ERROR] {error}")

        except Exception as error:
            statistics["errors"] += 1
            print(
                f"\n[ERROR] The image could not be processed:\n"
                f"{image_path}\n{type(error).__name__}: {error}"
            )

        if position % 100 == 0 or position == len(images):
            print(
                f"{source.name}: {position}/{len(images)} checked | "
                f"{statistics['success']} succeeded | "
                f"{statistics['missing_or_invalid_labels']} label problems | "
                f"{statistics['errors']} other errors"
            )

    return statistics


def count_output_files(
    output_images_dir: Path,
    output_labels_dir: Path,
) -> tuple[int, int]:
    image_count = sum(
        1
        for path in output_images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    label_count = sum(
        1
        for path in output_labels_dir.rglob("*.txt")
        if path.is_file()
    )
    return image_count, label_count


def main() -> None:
    """
    Runs Arma first and Blender afterwards.
    """
    sources = [
        DatasetSource(
            name="Arma",
            filename_prefix="arma",
            images_dir=ARMA_IMAGES_DIR,
            labels_dir=ARMA_LABELS_DIR,
            policy_path=ARMA_POLICY_PATH,
            output_images_dir=ARMA_OUTPUT_IMAGES_DIR,
            output_labels_dir=ARMA_OUTPUT_LABELS_DIR,
            manifest_path=ARMA_MANIFEST_PATH,
            random_seed=ARMA_RANDOM_SEED,
        ),
        DatasetSource(
            name="Blender",
            filename_prefix="blender",
            images_dir=BLENDER_IMAGES_DIR,
            labels_dir=BLENDER_LABELS_DIR,
            policy_path=BLENDER_POLICY_PATH,
            output_images_dir=BLENDER_OUTPUT_IMAGES_DIR,
            output_labels_dir=BLENDER_OUTPUT_LABELS_DIR,
            manifest_path=BLENDER_MANIFEST_PATH,
            random_seed=BLENDER_RANDOM_SEED,
        ),
    ]

    # Prevents writing output directly into a raw input folder by accident.
    input_directories = {
        source.images_dir.resolve() for source in sources
    } | {
        source.labels_dir.resolve() for source in sources
    }

    output_directories = {
        directory.resolve()
        for source in sources
        for directory in (
            source.output_images_dir,
            source.output_labels_dir,
        )
    }

    if input_directories & output_directories:
        raise RuntimeError(
            "An output folder is identical to a raw input folder. "
            "The run was aborted in order to protect the original data."
        )

    for source in sources:
        source.output_images_dir.mkdir(parents=True, exist_ok=True)
        source.output_labels_dir.mkdir(parents=True, exist_ok=True)
        source.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "source_domain",
        "source_image",
        "source_label",
        "output_image",
        "output_label",
        "policy_file",
        "pick_n",
        "selected_augmentations",
    ]

    all_statistics: dict[str, dict[str, int]] = {}
    output_counts: dict[str, tuple[int, int]] = {}

    for source in sources:
        with source.manifest_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=manifest_fields,
                delimiter=";",
            )
            writer.writeheader()

            all_statistics[source.name] = process_dataset(
                source=source,
                manifest_writer=writer,
            )

        output_counts[source.name] = count_output_files(
            output_images_dir=source.output_images_dir,
            output_labels_dir=source.output_labels_dir,
        )

    print("\n" + "=" * 80)
    print("OVERALL RESULT")
    print("=" * 80)

    total_success = 0
    total_label_errors = 0
    total_other_errors = 0
    count_mismatch = False

    for source in sources:
        statistics = all_statistics[source.name]
        image_count, label_count = output_counts[source.name]

        print(
            f"\n{source.name}: "
            f"{statistics['success']} succeeded, "
            f"{statistics['missing_or_invalid_labels']} label problems, "
            f"{statistics['errors']} other errors"
        )
        print(f"  Output images: {image_count}")
        print(f"  Output labels: {label_count}")
        print(f"  Image output:  {source.output_images_dir}")
        print(f"  Label output:  {source.output_labels_dir}")
        print(f"  Record:        {source.manifest_path}")

        if image_count != label_count:
            count_mismatch = True

        total_success += statistics["success"]
        total_label_errors += statistics["missing_or_invalid_labels"]
        total_other_errors += statistics["errors"]

    print(f"\nNewly created image-label pairs in total: {total_success}")
    print(f"Label problems in total:                  {total_label_errors}")
    print(f"Other errors in total:                    {total_other_errors}")

    if count_mismatch:
        print(
            "\nWARNING: at least one output dataset holds a different number "
            "of images than labels. Check the records "
            "and the error messages."
        )
        sys.exit(1)

    if total_label_errors or total_other_errors:
        print(
            "\nThe run finished, but not all source files could be processed. "
            "Check the messages above."
        )
        sys.exit(1)

    print(
        "\nDone. The raw datasets were not modified. "
        "Arma and Blender were written to separate output folders."
    )


if __name__ == "__main__":
    main()

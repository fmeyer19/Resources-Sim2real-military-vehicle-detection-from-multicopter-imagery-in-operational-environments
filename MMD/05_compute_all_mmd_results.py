"""
05_compute_all_mmd_results.py

Lädt mehrere zuvor extrahierte Feature-Dateien (.npy), normalisiert alle
Features gemeinsam und berechnet anschließend MMD^2-Werte zwischen einem
Referenzdatensatz und beliebig vielen Vergleichsdatensätzen.

Beispiel:
- Referenz: real_A_features.npy
- Vergleich: real_B_features.npy
- Vergleich: synthetic_raw_features.npy
- Vergleich: synthetic_blur_features.npy
- Vergleich: synthetic_compression_features.npy

Die gemeinsame Normalisierung stellt sicher, dass alle MMD-Werte auf derselben
Feature-Skalierung beruhen und dadurch besser vergleichbar sind.
"""

from pathlib import Path
import csv

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

FEATURE_FILES = {
    "RealWorldDirty A": r"<path to real_A_features.npy>",
    
    "RealWorldDirty B": r"<path to real_B_features.npy>",

    # Später ergänzen:          
    # "Synthetic Raw": r"<path to synthetic_raw_features.npy>",
    # "Synthetic Blur": r"<path to synthetic_blur_features.npy>",
    # "Synthetic Compression": r"<synthetic_compression_features.npy>",
    # "Synthetic Fog": r"<synthetic_fog_features.npy>",
    # "Synthetic AllMixed": r"<synthetic_allmixed_features.npy>",
}

REFERENCE_DATASET = "RealWorldDirty A"

RAW_SYNTHETIC_DATASET = "Synthetic Raw"  # für spätere Prozentreduktion

OUTPUT_FOLDER = Path(
    "<path to output directory>"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MMD-FUNKTIONEN
# ============================================================

def median_heuristic_gamma(X, Y):
    """
    Bestimmt gamma für den RBF-Kernel über die Median-Heuristik.
    Dadurch wird gamma nicht manuell festgelegt.
    """
    Z = np.vstack([X, Y])

    distances = pairwise_distances(Z, Z, metric="euclidean")
    distances = distances[distances > 0]

    median_distance = np.median(distances)

    if median_distance <= 0:
        raise ValueError("Median distance is zero. Cannot compute gamma.")

    gamma = 1.0 / (2.0 * median_distance ** 2)

    return gamma


def rbf_kernel_matrix(X, Y, gamma):
    """
    Berechnet die RBF-Kernelmatrix zwischen X und Y.
    """
    distances_squared = pairwise_distances(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * distances_squared)


def compute_mmd_rbf(X, Y):
    """
    Berechnet MMD^2 zwischen zwei Feature-Matrizen X und Y.
    """
    gamma = median_heuristic_gamma(X, Y)

    K_XX = rbf_kernel_matrix(X, X, gamma)
    K_YY = rbf_kernel_matrix(Y, Y, gamma)
    K_XY = rbf_kernel_matrix(X, Y, gamma)

    mmd_squared = K_XX.mean() + K_YY.mean() - 2.0 * K_XY.mean()

    return float(mmd_squared), float(gamma)


# ============================================================
# 3. FEATURE-DATEIEN LADEN
# ============================================================

def load_feature_files(feature_files):
    features = {}

    for dataset_name, file_path in feature_files.items():
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        data = np.load(file_path)

        if len(data.shape) != 2:
            raise ValueError(
                f"Feature file must be 2D, but got shape {data.shape}: {file_path}"
            )

        features[dataset_name] = data.astype(np.float32)

        print(f"{dataset_name}: {data.shape}")

    return features


# ============================================================
# 4. GEMEINSAME NORMALISIERUNG
# ============================================================

def normalize_all_features_jointly(features):
    """
    Normalisiert alle Feature-Dateien gemeinsam.
    Das ist wichtig, damit alle MMD-Vergleiche auf derselben Skalierung beruhen.
    """
    dataset_names = list(features.keys())

    combined = np.vstack([features[name] for name in dataset_names])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    normalized_features = {}

    start = 0
    for name in dataset_names:
        n = len(features[name])
        normalized_features[name] = combined_scaled[start:start + n]
        start += n

    return normalized_features


# ============================================================
# 5. MMD-ERGEBNISSE BERECHNEN
# ============================================================

def compute_all_mmd_results(features, reference_dataset):
    if reference_dataset not in features:
        raise ValueError(f"Reference dataset not found: {reference_dataset}")

    reference_features = features[reference_dataset]

    results = []

    for dataset_name, dataset_features in features.items():
        if dataset_name == reference_dataset:
            continue

        mmd_value, gamma_value = compute_mmd_rbf(
            reference_features,
            dataset_features
        )

        results.append({
            "comparison": f"{reference_dataset} vs. {dataset_name}",
            "dataset": dataset_name,
            "mmd_squared": mmd_value,
            "gamma": gamma_value,
            "reduction_vs_raw_percent": None,
        })

    return results


# ============================================================
# 6. REDUKTION GEGENÜBER SYNTHETIC RAW BERECHNEN
# ============================================================

def add_reduction_vs_raw(results, raw_dataset_name):
    """
    Berechnet die prozentuale MMD-Reduktion gegenüber Synthetic Raw.

    Formel:
    reduction = (MMD_raw - MMD_method) / MMD_raw * 100
    """
    raw_result = None

    for result in results:
        if result["dataset"] == raw_dataset_name:
            raw_result = result
            break

    if raw_result is None:
        print(
            f"\nHinweis: '{raw_dataset_name}' wurde nicht gefunden. "
            "Reduktion gegenüber Raw wird übersprungen."
        )
        return results

    raw_mmd = raw_result["mmd_squared"]

    for result in results:
        if result["dataset"] == raw_dataset_name:
            result["reduction_vs_raw_percent"] = 0.0
        else:
            reduction = ((raw_mmd - result["mmd_squared"]) / raw_mmd) * 100.0
            result["reduction_vs_raw_percent"] = reduction

    return results


# ============================================================
# 7. CSV SPEICHERN
# ============================================================

def save_results_csv(results, output_path):
    results_sorted = sorted(results, key=lambda x: x["mmd_squared"])

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Comparison",
            "Dataset",
            "MMD_squared",
            "Gamma",
            "Reduction_vs_Synthetic_Raw_percent"
        ])

        for result in results_sorted:
            reduction = result["reduction_vs_raw_percent"]

            if reduction is None:
                reduction_str = ""
            else:
                reduction_str = f"{reduction:.2f}"

            writer.writerow([
                result["comparison"],
                result["dataset"],
                f"{result['mmd_squared']:.8f}",
                f"{result['gamma']:.8f}",
                reduction_str
            ])


# ============================================================
# 8. PLOT ERSTELLEN
# ============================================================

def plot_mmd_results(results, output_path):
    results_sorted = sorted(results, key=lambda x: x["mmd_squared"])

    labels = [r["comparison"] for r in results_sorted]
    values = [r["mmd_squared"] for r in results_sorted]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)

    plt.xlabel("MMD$^2$")
    plt.ylabel("Comparison")
    plt.title("MMD-based feature-space distance to RealWorldDirty")

    # Niedrigster Wert oben
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("\nLoading feature files...")
    features = load_feature_files(FEATURE_FILES)

    print("\nNormalizing all features jointly...")
    features_normalized = normalize_all_features_jointly(features)

    print("\nComputing MMD results...")
    results = compute_all_mmd_results(
        features_normalized,
        REFERENCE_DATASET
    )

    results = add_reduction_vs_raw(
        results,
        RAW_SYNTHETIC_DATASET
    )

    print("\n================ MMD RESULTS ================\n")

    for result in sorted(results, key=lambda x: x["mmd_squared"]):
        reduction = result["reduction_vs_raw_percent"]

        if reduction is None:
            reduction_text = ""
        else:
            reduction_text = f", reduction vs. raw = {reduction:.2f}%"

        print(
            f"{result['comparison']}: "
            f"MMD^2 = {result['mmd_squared']:.8f}, "
            f"gamma = {result['gamma']:.8f}"
            f"{reduction_text}"
        )

    csv_path = OUTPUT_FOLDER / "mmd_results.csv"
    png_path = OUTPUT_FOLDER / "mmd_barplot.png"
    pdf_path = OUTPUT_FOLDER / "mmd_barplot.pdf"

    save_results_csv(results, csv_path)
    plot_mmd_results(results, png_path)
    plot_mmd_results(results, pdf_path)

    print("\nSaved files:")
    print(csv_path)
    print(png_path)
    print(pdf_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
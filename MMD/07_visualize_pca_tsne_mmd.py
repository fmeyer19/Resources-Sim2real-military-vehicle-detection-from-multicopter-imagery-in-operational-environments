from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================
# EINSTELLUNGEN
# =========================

BASE_DIR = Path(r"<path to base directory>")

FEATURE_FOLDER = BASE_DIR / "features_all"
MMD_CSV = BASE_DIR / "mmd_results" / "mmd_results_all.csv"

OUTPUT_FOLDER = BASE_DIR / "mmd_results" / "feature_visualizations"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REFERENCE_DATASET = "realworld_A"

RANDOM_SEED = 42

# t-SNE ist stochastisch und etwas rechenintensiver.
# Bei 3 Datensätzen mit je 500 Bildern = 1500 Punkte pro Plot ist das gut machbar.
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = "auto"
TSNE_INIT = "pca"


# =========================
# VERGLEICHE
# =========================

COMPARISONS = {
    "arma": [
        "realworld_A",
        "arma_raw",
        "arma_allmixed_best_values",
    ],
    "blender": [
        "realworld_A",
        "blender_raw",
        "blender_allmixed_best_values",
    ],
    "mixed": [
        "realworld_A",
        "mixed_raw",
        "mixed_allmixed_best_values",
    ],
}


# =========================
# FUNKTIONEN
# =========================

def load_features(feature_folder: Path):
    features = {}

    for file in sorted(feature_folder.glob("*_features.npy")):
        dataset_name = file.name.replace("_features.npy", "")
        features[dataset_name] = np.load(file)

    return features


def load_mmd_values(csv_path: Path):
    df = pd.read_csv(csv_path, sep=";")
    df["mmd2"] = df["mmd2"].astype(float)

    values = {}

    for _, row in df.iterrows():
        comparison_dataset = row["comparison_dataset"]
        values[comparison_dataset] = row["mmd2"]

    return values


def build_plot_data(features_dict, dataset_names):
    X_list = []
    labels = []

    for dataset_name in dataset_names:
        X = features_dict[dataset_name]
        X_list.append(X)
        labels.extend([dataset_name] * X.shape[0])

    X_all = np.vstack(X_list)
    labels = np.array(labels)

    return X_all, labels


def fit_global_scaler(features_dict):
    """
    Standardisierung wie in der MMD-Berechnung:
    Alle verfügbaren Feature-Dateien werden gemeinsam zur Bestimmung
    von Mittelwert und Standardabweichung verwendet.
    """
    all_features = np.vstack(list(features_dict.values()))

    scaler = StandardScaler()
    scaler.fit(all_features)

    return scaler


def plot_embedding(embedding, labels, title, output_path):
    plt.figure(figsize=(8, 6))

    unique_labels = list(dict.fromkeys(labels))

    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.7,
            label=label
        )

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))

    plt.close()


def make_title(method_name, group_name, dataset_names, mmd_values):
    title = f"{method_name}: {group_name}"

    mmd_parts = []

    for dataset_name in dataset_names:
        if dataset_name == REFERENCE_DATASET:
            continue

        if dataset_name in mmd_values:
            mmd_parts.append(f"{dataset_name}: MMD²={mmd_values[dataset_name]:.6f}")

    if mmd_parts:
        title += "\n" + " | ".join(mmd_parts)

    return title


def save_embedding_csv(embedding, labels, output_path):
    df = pd.DataFrame({
        "dataset": labels,
        "dim_1": embedding[:, 0],
        "dim_2": embedding[:, 1],
    })

    df.to_csv(output_path, sep=";", index=False, encoding="utf-8")


# =========================
# MAIN
# =========================

print("Lade Features...")
features = load_features(FEATURE_FOLDER)

print(f"Geladene Feature-Dateien: {len(features)}")

print("Lade MMD²-Werte...")
mmd_values = load_mmd_values(MMD_CSV)

print("Standardisiere Features global...")
scaler = fit_global_scaler(features)

features_scaled = {
    name: scaler.transform(X)
    for name, X in features.items()
}


for group_name, dataset_names in COMPARISONS.items():
    print("\n========================================")
    print(f"Visualisierung: {group_name}")
    print("========================================")

    for dataset_name in dataset_names:
        if dataset_name not in features_scaled:
            raise KeyError(
                f"Feature-Datei für '{dataset_name}' nicht gefunden. "
                f"Verfügbare Datensätze: {list(features_scaled.keys())}"
            )

    X, labels = build_plot_data(features_scaled, dataset_names)

    print(f"Datenmatrix: {X.shape}")

    # =========================
    # PCA
    # =========================

    print("Berechne PCA...")

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")

    pca_title = make_title("PCA", group_name, dataset_names, mmd_values)
    pca_output_base = OUTPUT_FOLDER / f"{group_name}_pca"

    plot_embedding(
        X_pca,
        labels,
        pca_title,
        pca_output_base
    )

    save_embedding_csv(
        X_pca,
        labels,
        OUTPUT_FOLDER / f"{group_name}_pca_coordinates.csv"
    )

    # =========================
    # t-SNE
    # =========================

    print("Berechne t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INIT,
        random_state=RANDOM_SEED
    )

    X_tsne = tsne.fit_transform(X)

    tsne_title = make_title("t-SNE", group_name, dataset_names, mmd_values)
    tsne_output_base = OUTPUT_FOLDER / f"{group_name}_tsne"

    plot_embedding(
        X_tsne,
        labels,
        tsne_title,
        tsne_output_base
    )

    save_embedding_csv(
        X_tsne,
        labels,
        OUTPUT_FOLDER / f"{group_name}_tsne_coordinates.csv"
    )


print("\nFertig.")
print(f"Visualisierungen gespeichert unter: {OUTPUT_FOLDER}")
from pathlib import Path
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(r"<path to base directory>")

FEATURE_FOLDER = BASE_DIR / "features_all"
OUTPUT_FOLDER = BASE_DIR / "mmd_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

REFERENCE_DATASET = "realworld_A"

OUTPUT_CSV = OUTPUT_FOLDER / "mmd_results_all.csv"
OUTPUT_SORTED_CSV = OUTPUT_FOLDER / "mmd_results_all_sorted.csv"

RANDOM_SEED = 42
GAMMA_SAMPLE_SIZE = 3000


# =========================
# FUNCTIONS
# =========================

def load_feature_files(feature_folder):
    feature_files = sorted(feature_folder.glob("*_features.npy"))

    features = {}

    for file in feature_files:
        dataset_name = file.name.replace("_features.npy", "")
        features[dataset_name] = np.load(file)

    return features


def normalize_features_jointly(features_dict):
    """
    All feature vectors are standardized jointly.
    All datasets therefore lie in the same normalized feature space.
    """
    all_features = np.vstack(list(features_dict.values()))

    scaler = StandardScaler()
    scaler.fit(all_features)

    normalized = {
        name: scaler.transform(features)
        for name, features in features_dict.items()
    }

    return normalized


def estimate_global_gamma(features_dict, sample_size=3000, seed=42):
    """
    Estimates a shared RBF gamma value by the median heuristic.
    The value is determined globally and then used for all MMD^2 comparisons.
    """
    rng = np.random.default_rng(seed)

    all_features = np.vstack(list(features_dict.values()))
    n = all_features.shape[0]

    sample_size = min(sample_size, n)
    sample_indices = rng.choice(n, size=sample_size, replace=False)
    sample = all_features[sample_indices]

    distances_sq = pairwise_distances(sample, sample, metric="sqeuclidean")

    # Remove the diagonal, where the distance is always 0
    distances_sq = distances_sq[np.triu_indices_from(distances_sq, k=1)]

    median_dist_sq = np.median(distances_sq)

    if median_dist_sq <= 0:
        raise ValueError("Median distance is zero. Gamma cannot be estimated.")

    gamma = 1.0 / (2.0 * median_dist_sq)

    return gamma, median_dist_sq


def rbf_kernel_matrix(X, Y, gamma):
    distances_sq = pairwise_distances(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * distances_sq)


def compute_mmd2(X, Y, gamma):
    """
    Biased MMD^2 estimate:
    mean(Kxx) + mean(Kyy) - 2 * mean(Kxy)
    """
    K_xx = rbf_kernel_matrix(X, X, gamma)
    K_yy = rbf_kernel_matrix(Y, Y, gamma)
    K_xy = rbf_kernel_matrix(X, Y, gamma)

    mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()

    return float(mmd2)


def infer_group(dataset_name):
    if dataset_name.startswith("arma_"):
        return "arma"
    if dataset_name.startswith("blender_"):
        return "blender"
    if dataset_name.startswith("mixed_"):
        return "mixed"
    if dataset_name.startswith("realworld_"):
        return "realworld"
    return "unknown"


def infer_variant(dataset_name):
    if dataset_name.endswith("_raw"):
        return "raw"
    if "raw" in dataset_name:
        return "raw"
    if "value_a" in dataset_name:
        return "value_a"
    if "value_b" in dataset_name:
        return "value_b"
    if "value_c" in dataset_name:
        return "value_c"
    return ""


# =========================
# MAIN
# =========================

print("\nLoading feature files...")
features = load_feature_files(FEATURE_FOLDER)

print(f"Feature files found: {len(features)}")

if REFERENCE_DATASET not in features:
    raise KeyError(
        f"Reference dataset '{REFERENCE_DATASET}' was not found. "
        f"Available datasets: {list(features.keys())}"
    )

print("\nFeature Shapes:")
for name, feat in features.items():
    print(f"{name:45s} {feat.shape}")

print("\nStandardizing all features jointly...")
features_norm = normalize_features_jointly(features)

print("\nEstimating global RBF gamma value...")
gamma, median_dist_sq = estimate_global_gamma(
    features_norm,
    sample_size=GAMMA_SAMPLE_SIZE,
    seed=RANDOM_SEED
)

print(f"Global median squared distance: {median_dist_sq:.6f}")
print(f"Global gamma: {gamma:.10f}")

reference_features = features_norm[REFERENCE_DATASET]

results = []

print("\nComputing MMD^2 comparisons...\n")

for dataset_name, dataset_features in features_norm.items():
    if dataset_name == REFERENCE_DATASET:
        continue

    mmd2 = compute_mmd2(reference_features, dataset_features, gamma)

    group = infer_group(dataset_name)
    variant = infer_variant(dataset_name)

    results.append({
        "reference_dataset": REFERENCE_DATASET,
        "comparison_dataset": dataset_name,
        "group": group,
        "variant": variant,
        "mmd2": mmd2,
        "gamma": gamma,
        "median_dist_sq": median_dist_sq,
        "n_reference": reference_features.shape[0],
        "n_comparison": dataset_features.shape[0],
        "feature_dim": reference_features.shape[1],
    })

    print(f"{REFERENCE_DATASET:20s} vs {dataset_name:45s} MMD² = {mmd2:.8f}")


# =========================
# SAVE CSV
# =========================

fieldnames = [
    "reference_dataset",
    "comparison_dataset",
    "group",
    "variant",
    "mmd2",
    "gamma",
    "median_dist_sq",
    "n_reference",
    "n_comparison",
    "feature_dim",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(results)

results_sorted = sorted(results, key=lambda x: x["mmd2"])

with open(OUTPUT_SORTED_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(results_sorted)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_SORTED_CSV)

print("\nDone.")
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# =========================
# EINSTELLUNGEN
# =========================

FEATURE_FOLDER = Path(
    r"C:\Users\admin\Desktop\Doktorarbeit\Maximum_Mean_Discrepancy\features_RW-RW_Baseline"
)

REAL_A_FEATURES = FEATURE_FOLDER / "real_A_features.npy"
REAL_B_FEATURES = FEATURE_FOLDER / "real_B_features.npy"


# =========================
# MMD FUNKTIONEN
# =========================

def median_heuristic_gamma(X, Y):
    """
    Bestimmt den gamma-Wert für den RBF-Kernel automatisch
    über die Median-Heuristik.
    """
    Z = np.vstack([X, Y])

    distances = pairwise_distances(Z, Z, metric="euclidean")

    # Null-Distanzen auf der Diagonale entfernen
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

    return mmd_squared, gamma


# =========================
# MAIN
# =========================

real_A = np.load(REAL_A_FEATURES)
real_B = np.load(REAL_B_FEATURES)

print("real_A shape:", real_A.shape)
print("real_B shape:", real_B.shape)

# Wichtig: Features gemeinsam normalisieren
combined = np.vstack([real_A, real_B])

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

real_A_scaled = combined_scaled[:len(real_A)]
real_B_scaled = combined_scaled[len(real_A):]

mmd_value, gamma_value = compute_mmd_rbf(real_A_scaled, real_B_scaled)

print("\n================ MMD BASELINE ================")
print(f"MMD^2 Real_A vs. Real_B: {mmd_value:.6f}")
print(f"RBF gamma: {gamma_value:.8f}")
print("================================================")
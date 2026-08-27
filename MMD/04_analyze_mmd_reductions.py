from pathlib import Path
import pandas as pd


# =========================
# EINSTELLUNGEN
# =========================

BASE_DIR = Path(r"<path to base directory>")

INPUT_CSV = BASE_DIR / "mmd_results" / "mmd_results_all.csv"

OUTPUT_DETAILED_CSV = BASE_DIR / "mmd_results" / "mmd_reductions_detailed.csv"
OUTPUT_BEST_CSV = BASE_DIR / "mmd_results" / "mmd_best_values_per_method.csv"
OUTPUT_TOP_CSV = BASE_DIR / "mmd_results" / "mmd_top_overall.csv"


# =========================
# HILFSFUNKTIONEN
# =========================

def infer_method_name(dataset_name: str, group: str) -> str:
    """
    Entfernt Gruppenpräfix und value-Suffix.
    Beispiel:
    arma_gaussian_blur_value_a -> gaussian_blur
    mixed_white_fog_value_c    -> white_fog
    """
    prefix = f"{group}_"

    if dataset_name.startswith(prefix):
        name = dataset_name[len(prefix):]
    else:
        name = dataset_name

    for suffix in ["_value_a", "_value_b", "_value_c"]:
        if name.endswith(suffix):
            name = name.replace(suffix, "")

    return name


def interpret_reduction(reduction_percent: float) -> str:
    if reduction_percent > 0:
        return "better_than_raw"
    elif reduction_percent < 0:
        return "worse_than_raw"
    else:
        return "same_as_raw"


# =========================
# CSV LADEN
# =========================

df = pd.read_csv(INPUT_CSV, sep=";")

# Nur Arma, Blender und Mixed betrachten.
# RealWorld_B ist nur Baseline und wird hier nicht als Augmentation analysiert.
df = df[df["group"].isin(["arma", "blender", "mixed"])].copy()

df["mmd2"] = df["mmd2"].astype(float)


# =========================
# RAW-WERTE EXTRAHIEREN
# =========================

raw_values = {}

for group in ["arma", "blender", "mixed"]:
    raw_row = df[(df["group"] == group) & (df["variant"] == "raw")]

    if raw_row.empty:
        raise ValueError(f"Kein Raw-Wert für Gruppe gefunden: {group}")

    raw_values[group] = float(raw_row.iloc[0]["mmd2"])


print("\nRaw-MMD²-Werte:")
for group, value in raw_values.items():
    print(f"{group:8s}: {value:.8f}")


# =========================
# REDUKTIONEN BERECHNEN
# =========================

rows = []

for _, row in df.iterrows():
    group = row["group"]
    dataset_name = row["comparison_dataset"]
    variant = row["variant"]
    mmd2 = float(row["mmd2"])

    # Raw-Zeilen selbst überspringen
    if variant == "raw":
        continue

    raw_mmd2 = raw_values[group]
    delta_mmd2 = raw_mmd2 - mmd2
    reduction_percent = (delta_mmd2 / raw_mmd2) * 100.0

    method = infer_method_name(dataset_name, group)

    rows.append({
        "group": group,
        "method": method,
        "variant": variant,
        "comparison_dataset": dataset_name,
        "mmd2": mmd2,
        "raw_mmd2": raw_mmd2,
        "delta_mmd2": delta_mmd2,
        "reduction_percent": reduction_percent,
        "interpretation": interpret_reduction(reduction_percent),
    })


reduction_df = pd.DataFrame(rows)

# Sortierung: Gruppe, Methode, Variante
reduction_df = reduction_df.sort_values(
    by=["group", "method", "variant"]
)

reduction_df.to_csv(
    OUTPUT_DETAILED_CSV,
    sep=";",
    index=False,
    encoding="utf-8"
)


# =========================
# BESTER VALUE PRO METHODE
# =========================

best_df = (
    reduction_df
    .sort_values(by=["group", "method", "mmd2"], ascending=[True, True, True])
    .groupby(["group", "method"], as_index=False)
    .first()
)

best_df = best_df.sort_values(
    by=["group", "reduction_percent"],
    ascending=[True, False]
)

best_df.to_csv(
    OUTPUT_BEST_CSV,
    sep=";",
    index=False,
    encoding="utf-8"
)


# =========================
# TOP OVERALL
# =========================

top_df = reduction_df.sort_values(
    by="reduction_percent",
    ascending=False
)

top_df.to_csv(
    OUTPUT_TOP_CSV,
    sep=";",
    index=False,
    encoding="utf-8"
)


# =========================
# AUSGABE
# =========================

print("\nBeste Augmentation pro Methode:")
print(
    best_df[
        ["group", "method", "variant", "mmd2", "raw_mmd2", "reduction_percent", "interpretation"]
    ].to_string(index=False)
)

print("\nTop 15 insgesamt:")
print(
    top_df[
        ["group", "method", "variant", "mmd2", "reduction_percent", "interpretation"]
    ].head(15).to_string(index=False)
)

print("\nGespeichert:")
print(OUTPUT_DETAILED_CSV)
print(OUTPUT_BEST_CSV)
print(OUTPUT_TOP_CSV)

print("\nDone.")
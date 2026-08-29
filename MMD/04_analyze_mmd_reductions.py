from pathlib import Path
import pandas as pd


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(r"<path to base directory>")

INPUT_CSV = BASE_DIR / "mmd_results" / "mmd_results_all.csv"

OUTPUT_DETAILED_CSV = BASE_DIR / "mmd_results" / "mmd_reductions_detailed.csv"
OUTPUT_BEST_CSV = BASE_DIR / "mmd_results" / "mmd_best_values_per_method.csv"
OUTPUT_TOP_CSV = BASE_DIR / "mmd_results" / "mmd_top_overall.csv"


# =========================
# HELPER FUNCTIONS
# =========================

def infer_method_name(dataset_name: str, group: str) -> str:
    """
    Removes the group prefix and the value suffix.
    Example:
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
# LOAD CSV
# =========================

df = pd.read_csv(INPUT_CSV, sep=";")

# Consider only Arma, Blender and Mixed.
# RealWorld_B is a baseline only and is not analyzed as an augmentation here.
df = df[df["group"].isin(["arma", "blender", "mixed"])].copy()

df["mmd2"] = df["mmd2"].astype(float)


# =========================
# EXTRACT RAW VALUES
# =========================

raw_values = {}

for group in ["arma", "blender", "mixed"]:
    raw_row = df[(df["group"] == group) & (df["variant"] == "raw")]

    if raw_row.empty:
        raise ValueError(f"No raw value found for group: {group}")

    raw_values[group] = float(raw_row.iloc[0]["mmd2"])


print("\nRaw MMD^2 values:")
for group, value in raw_values.items():
    print(f"{group:8s}: {value:.8f}")


# =========================
# COMPUTE REDUCTIONS
# =========================

rows = []

for _, row in df.iterrows():
    group = row["group"]
    dataset_name = row["comparison_dataset"]
    variant = row["variant"]
    mmd2 = float(row["mmd2"])

    # Skip the raw rows themselves
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

# Sort order: group, method, variant
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
# BEST VALUE PER METHOD
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
# OUTPUT
# =========================

print("\nBest augmentation per method:")
print(
    best_df[
        ["group", "method", "variant", "mmd2", "raw_mmd2", "reduction_percent", "interpretation"]
    ].to_string(index=False)
)

print("\nTop 15 overall:")
print(
    top_df[
        ["group", "method", "variant", "mmd2", "reduction_percent", "interpretation"]
    ].head(15).to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_DETAILED_CSV)
print(OUTPUT_BEST_CSV)
print(OUTPUT_TOP_CSV)

print("\nDone.")
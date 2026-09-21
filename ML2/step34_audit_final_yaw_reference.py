from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Maverick\ML2")
INPUT = ROOT / "step33_yaw_reference" / "final_yaw_reference.csv"

OUTDIR = ROOT / "step34_yaw_reference_audit"
OUTDIR.mkdir(exist_ok=True)

SUMMARY = OUTDIR / "step34_summary.txt"
LARGE_JUMPS = OUTDIR / "large_reference_jumps.csv"
AMBIGUOUS = OUTDIR / "ambiguous_reference_cases.csv"
DISAGREEMENTS = OUTDIR / "reference_signal_disagreements.csv"
DATASET_STATS = OUTDIR / "reference_dataset_statistics.csv"


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def pct(n, total):
    return 100.0 * n / total if total else 0.0


print("=" * 70)
print("STEP 34 - FINAL YAW REFERENCE AUDIT")
print("=" * 70)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required = [
    "dataset",
    "window_index",
    "mean_speed_kmh",
    "heading_change_deg",
    "yawrate_aligned_change_deg",
    "gnss_direction_change_deg",
    "heading_yaw_error_deg",
    "heading_gnss_error_deg",
    "yawrate_gnss_error_deg",
    "reference_source",
    "reference_delta_yaw_deg",
    "reference_confidence",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing columns:\n" + "\n".join(missing)
    )

# ------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------

numeric = [
    "mean_speed_kmh",
    "heading_change_deg",
    "yawrate_aligned_change_deg",
    "gnss_direction_change_deg",
    "heading_yaw_error_deg",
    "heading_gnss_error_deg",
    "yawrate_gnss_error_deg",
    "reference_delta_yaw_deg",
    "reference_confidence",
]

for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ============================================================
# 1. REFERENCE VALIDITY
# ============================================================

df["reference_valid"] = (
    np.isfinite(df["reference_delta_yaw_deg"])
)

valid = df[df["reference_valid"]].copy()

print()
print("VALIDITY")
print("--------")
print(f"Valid references : {len(valid):,}")
print(f"Invalid          : {len(df)-len(valid):,}")


# ============================================================
# 2. LARGE REFERENCE JUMPS
# ============================================================

df["abs_reference_yaw"] = np.abs(
    df["reference_delta_yaw_deg"]
)

jump45 = df["abs_reference_yaw"] >= 45
jump90 = df["abs_reference_yaw"] >= 90
jump120 = df["abs_reference_yaw"] >= 120
jump150 = df["abs_reference_yaw"] >= 150

print()
print("REFERENCE JUMPS")
print("---------------")
print(f">=45°  : {jump45.sum():,}")
print(f">=90°  : {jump90.sum():,}")
print(f">=120° : {jump120.sum():,}")
print(f">=150° : {jump150.sum():,}")

jump_cols = [
    "dataset",
    "window_index",
    "start_row",
    "end_row",
    "mean_speed_kmh",
    "heading_change_deg",
    "yawrate_aligned_change_deg",
    "gnss_direction_change_deg",
    "heading_yaw_error_deg",
    "heading_gnss_error_deg",
    "yawrate_gnss_error_deg",
    "reference_source",
    "reference_delta_yaw_deg",
    "reference_confidence",
]

jump_columns = [
    c for c in jump_cols
    if c in df.columns
]

jump_df = (
    df.loc[jump45, jump_columns + ["abs_reference_yaw"]]
    .sort_values(
        "abs_reference_yaw",
        ascending=False
    )
)

jump_df.to_csv(
    LARGE_JUMPS,
    index=False,
    float_format="%.8f"
)


# ============================================================
# 3. REFERENCE VS EACH SIGNAL
# ============================================================

df["reference_vs_heading_error_deg"] = np.abs(
    wrap180(
        df["reference_delta_yaw_deg"] -
        df["heading_change_deg"]
    )
)

df["reference_vs_yawrate_error_deg"] = np.abs(
    wrap180(
        df["reference_delta_yaw_deg"] -
        df["yawrate_aligned_change_deg"]
    )
)

df["reference_vs_gnss_error_deg"] = np.abs(
    wrap180(
        df["reference_delta_yaw_deg"] -
        df["gnss_direction_change_deg"]
    )
)


def print_stats(name, s):

    s = s[np.isfinite(s)]

    print()
    print(name)
    print(f"  Median : {np.median(s):.6f}°")
    print(f"  P90    : {np.percentile(s,90):.6f}°")
    print(f"  P95    : {np.percentile(s,95):.6f}°")
    print(f"  P99    : {np.percentile(s,99):.6f}°")
    print(f"  >=5°   : {(s>=5).sum():,}")
    print(f"  >=10°  : {(s>=10).sum():,}")
    print(f"  >=20°  : {(s>=20).sum():,}")
    print(f"  >=45°  : {(s>=45).sum():,}")


print()
print("=" * 70)
print("3. REFERENCE AGREEMENT")
print("=" * 70)

print_stats(
    "Reference vs Heading",
    df["reference_vs_heading_error_deg"].to_numpy()
)

print_stats(
    "Reference vs aligned YawRate",
    df["reference_vs_yawrate_error_deg"].to_numpy()
)

print_stats(
    "Reference vs GNSS",
    df["reference_vs_gnss_error_deg"].to_numpy()
)


# ============================================================
# 4. AGREEMENT BY REFERENCE SOURCE
# ============================================================

print()
print("=" * 70)
print("4. AGREEMENT BY REFERENCE SOURCE")
print("=" * 70)

source_rows = []

for source, g in df.groupby(
    "reference_source",
    dropna=False
):

    source_rows.append({
        "reference_source": source,
        "count": len(g),
        "percentage": pct(len(g), len(df)),

        "heading_median_error":
            g["reference_vs_heading_error_deg"].median(),

        "yawrate_median_error":
            g["reference_vs_yawrate_error_deg"].median(),

        "gnss_median_error":
            g["reference_vs_gnss_error_deg"].median(),

        "mean_confidence":
            g["reference_confidence"].mean(),

        "median_reference_abs_yaw":
            g["abs_reference_yaw"].median(),

        "p90_reference_abs_yaw":
            g["abs_reference_yaw"].quantile(.90)
    })

source_stats = pd.DataFrame(source_rows)

print(source_stats.to_string(index=False))

source_stats.to_csv(
    DATASET_STATS,
    index=False
)


# ============================================================
# 5. AMBIGUOUS CASES
# ============================================================

ambiguous = df[
    df["reference_source"] == "AMBIGUOUS"
].copy()

print()
print("=" * 70)
print("5. AMBIGUOUS CASES")
print("=" * 70)

print(f"Ambiguous rows: {len(ambiguous):,}")

if len(ambiguous):

    ambiguous["max_vehicle_signal_difference"] = np.maximum(
        ambiguous["heading_yaw_error_deg"],
        0
    )

    amb_cols = [
        "dataset",
        "window_index",
        "start_row",
        "end_row",
        "mean_speed_kmh",
        "heading_change_deg",
        "yawrate_aligned_change_deg",
        "gnss_direction_change_deg",
        "heading_yaw_error_deg",
        "heading_gnss_error_deg",
        "yawrate_gnss_error_deg",
        "reference_source",
        "reference_confidence",
    ]

    ambiguous[
        [c for c in amb_cols if c in ambiguous.columns]
    ].to_csv(
        AMBIGUOUS,
        index=False,
        float_format="%.8f"
    )


# ============================================================
# 6. IMPORTANT DISAGREEMENTS
# ============================================================

disagreement = df[
    (
        df["heading_yaw_error_deg"] > 10
    ) |
    (
        df["heading_gnss_error_deg"] > 20
    ) |
    (
        df["yawrate_gnss_error_deg"] > 20
    )
].copy()

print()
print("=" * 70)
print("6. SIGNAL DISAGREEMENTS")
print("=" * 70)

print(
    f"Significant disagreement rows: "
    f"{len(disagreement):,}"
)

dis_cols = [
    "dataset",
    "window_index",
    "start_row",
    "end_row",
    "mean_speed_kmh",
    "heading_change_deg",
    "yawrate_aligned_change_deg",
    "gnss_direction_change_deg",
    "heading_yaw_error_deg",
    "heading_gnss_error_deg",
    "yawrate_gnss_error_deg",
    "reference_source",
    "reference_delta_yaw_deg",
    "reference_confidence",
]

disagreement[
    [c for c in dis_cols if c in disagreement.columns]
].to_csv(
    DISAGREEMENTS,
    index=False,
    float_format="%.8f"
)


# ============================================================
# 7. DATASET-BY-DATASET STATISTICS
# ============================================================

print()
print("=" * 70)
print("7. DATASET STATISTICS")
print("=" * 70)

dataset_rows = []

for dataset, g in df.groupby("dataset"):

    dataset_rows.append({
        "dataset": dataset,
        "windows": len(g),

        "valid_reference":
            g["reference_valid"].sum(),

        "ambiguous":
            (g["reference_source"] == "AMBIGUOUS").sum(),

        "reference_ge_45":
            (g["abs_reference_yaw"] >= 45).sum(),

        "reference_ge_90":
            (g["abs_reference_yaw"] >= 90).sum(),

        "median_reference_abs_yaw":
            g["abs_reference_yaw"].median(),

        "p95_reference_abs_yaw":
            g["abs_reference_yaw"].quantile(.95),

        "median_heading_error":
            g["reference_vs_heading_error_deg"].median(),

        "median_yawrate_error":
            g["reference_vs_yawrate_error_deg"].median(),

        "median_gnss_error":
            g["reference_vs_gnss_error_deg"].median(),
    })

dataset_stats = pd.DataFrame(dataset_rows)

print(
    dataset_stats.sort_values(
        "reference_ge_45",
        ascending=False
    ).head(20).to_string(index=False)
)


# ============================================================
# 8. VTB02 SPECIFIC AUDIT
# ============================================================

vtb02 = df[
    df["dataset"].astype(str).str.lower().isin(
        ["vtb02", "vtb2"]
    )
].copy()

print()
print("=" * 70)
print("8. VTB02 AUDIT")
print("=" * 70)

if len(vtb02):

    print(f"Vtb02 windows: {len(vtb02):,}")

    print(
        f"Reference median abs yaw: "
        f"{vtb02['abs_reference_yaw'].median():.6f}°"
    )

    print(
        f"Reference >=45°: "
        f"{(vtb02['abs_reference_yaw'] >= 45).sum():,}"
    )

    print(
        f"Ambiguous: "
        f"{(vtb02['reference_source']=='AMBIGUOUS').sum():,}"
    )

    print()
    print("Largest Vtb02 reference changes:")

    cols = [
        "window_index",
        "mean_speed_kmh",
        "heading_change_deg",
        "yawrate_aligned_change_deg",
        "gnss_direction_change_deg",
        "reference_delta_yaw_deg",
        "reference_source",
        "reference_confidence"
    ]

    print(
        vtb02.sort_values(
            "abs_reference_yaw",
            ascending=False
        )[cols].head(20).to_string(index=False)
    )

else:
    print("Vtb02 was not found by dataset name.")


# ============================================================
# 9. FINAL REFERENCE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("9. FINAL REFERENCE DISTRIBUTION")
print("=" * 70)

bins = [
    -180,
    -90,
    -45,
    -20,
    -10,
    -5,
    5,
    10,
    20,
    45,
    90,
    180
]

labels = [
    "-180 to -90",
    "-90 to -45",
    "-45 to -20",
    "-20 to -10",
    "-10 to -5",
    "-5 to +5",
    "+5 to +10",
    "+10 to +20",
    "+20 to +45",
    "+45 to +90",
    "+90 to +180"
]

distribution = pd.cut(
    df["reference_delta_yaw_deg"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

dist_counts = distribution.value_counts(
    sort=False
)

for label, count in dist_counts.items():

    print(
        f"{str(label):15s} "
        f"{count:8,d} "
        f"({pct(count,len(df)):6.2f}%)"
    )


# ============================================================
# 10. SUMMARY FILE
# ============================================================

summary = []

summary.append("=" * 70)
summary.append("STEP 34 - FINAL YAW REFERENCE AUDIT")
summary.append("=" * 70)
summary.append("")

summary.append(f"Total windows: {len(df):,}")
summary.append(
    f"Valid references: {df['reference_valid'].sum():,}"
)
summary.append(
    f"Ambiguous: {(df['reference_source']=='AMBIGUOUS').sum():,}"
)

summary.append("")
summary.append("REFERENCE JUMPS")
summary.append("----------------")
summary.append(
    f">=45°  : {jump45.sum():,}"
)
summary.append(
    f">=90°  : {jump90.sum():,}"
)
summary.append(
    f">=120° : {jump120.sum():,}"
)
summary.append(
    f">=150° : {jump150.sum():,}"
)

summary.append("")
summary.append("REFERENCE AGREEMENT")
summary.append("--------------------")

for name, column in [
    ("Heading", "reference_vs_heading_error_deg"),
    ("Aligned YawRate", "reference_vs_yawrate_error_deg"),
    ("GNSS", "reference_vs_gnss_error_deg")
]:

    s = df[column].dropna()

    summary.append(
        f"{name}: median={s.median():.6f}°, "
        f"P90={s.quantile(.90):.6f}°, "
        f"P95={s.quantile(.95):.6f}°, "
        f"P99={s.quantile(.99):.6f}°"
    )

summary.append("")
summary.append("REFERENCE SOURCES")
summary.append("-----------------")

for source, count in (
    df["reference_source"]
    .value_counts()
    .items()
):

    summary.append(
        f"{str(source):45s} "
        f"{count:8,d} "
        f"({pct(count,len(df)):6.2f}%)"
    )

summary.append("")
summary.append("INTERPRETATION")
summary.append("--------------")
summary.append(
    "This audit does not modify the Step 33 reference."
)
summary.append(
    "It only evaluates continuity and agreement with "
    "Heading, aligned YawRate and GNSS."
)
summary.append(
    "The reference is dataset-derived and should not be "
    "described as the original paper's external ground truth."
)

with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(summary))


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 34 COMPLETE")
print("=" * 70)

print()
print("Files created:")
print(SUMMARY)
print(LARGE_JUMPS)
print(AMBIGUOUS)
print(DISAGREEMENTS)
print(DATASET_STATS)

print()
print("Do NOT modify final_yaw_reference.csv yet.")
print("Audit it first.")
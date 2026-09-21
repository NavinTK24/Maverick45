import os
import numpy as np
import pandas as pd

ROOT = r"D:\Maverick\ML2"

STEP26_FILE = os.path.join(
    ROOT,
    "step26_gnss_reliability_analysis.csv"
)

OUTPUT_CSV = os.path.join(
    ROOT,
    "step30_physical_turning_consistency.csv"
)

OUTPUT_SUMMARY = os.path.join(
    ROOT,
    "step30_physical_turning_summary.txt"
)


# ============================================================
# STEP 30
# Physical Turning Consistency Analysis
# ============================================================

print("=" * 75)
print("STEP 30 - PHYSICAL TURNING CONSISTENCY ANALYSIS")
print("=" * 75)


# ============================================================
# Load Step 26
# ============================================================

if not os.path.exists(STEP26_FILE):
    raise FileNotFoundError(
        f"Step 26 file not found:\n{STEP26_FILE}"
    )

df = pd.read_csv(STEP26_FILE)

print(f"\nStep 26 windows loaded: {len(df):,}")


# ============================================================
# Actual Step 26 columns
# ============================================================

required = [
    "dataset",
    "window_index",
    "mean_speed_kmh",
    "gnss_displacement_m",
    "abs_gnss_direction_change_deg",
    "gnss_bearing_spread_deg",
    "heading_change_deg",
    "yawrate_integrated_change_deg",
    "heading_vs_yawrate_inverted_error_deg",
    "heading_vs_gnss_absolute_error_deg",
    "yawrate_vs_gnss_inverted_error_deg",
    "mean_abs_lateral_acc"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    print("\nMissing columns:")

    for c in missing:
        print("  ", c)

    print("\nAvailable columns:")

    for c in df.columns:
        print("  ", c)

    raise RuntimeError(
        "Step 26 output does not contain all required columns."
    )


# ============================================================
# Create convenient names
# ============================================================

df["abs_heading_change_deg"] = (
    df["heading_change_deg"].abs()
)

df["abs_yawrate_change_deg"] = (
    df["yawrate_integrated_change_deg"].abs()
)

df["abs_heading_yawrate_error_deg"] = (
    df["heading_vs_yawrate_inverted_error_deg"].abs()
)

df["max_abs_lateral_acc"] = (
    df["mean_abs_lateral_acc"].abs()
)

df["abs_gnss_direction_change_deg"] = (
    df["abs_gnss_direction_change_deg"].abs()
)


# ============================================================
# Physical classification
# ============================================================

def classify_window(row):

    speed = row["mean_speed_kmh"]

    heading = row["abs_heading_change_deg"]

    yaw = row["abs_yawrate_change_deg"]

    lat_acc = row["max_abs_lateral_acc"]

    heading_yaw = (
        row["abs_heading_yawrate_error_deg"]
    )


    # --------------------------------------------------------
    # Straight / low rotational motion
    # --------------------------------------------------------

    if (
        heading < 5
        and yaw < 5
        and lat_acc < 0.15
    ):
        return "STRAIGHT_OR_LOW_TURN"


    # --------------------------------------------------------
    # Strong physical turning evidence
    # --------------------------------------------------------

    if (
        heading >= 10
        and yaw >= 10
        and heading_yaw <= 15
        and lat_acc >= 0.10
    ):
        return "PHYSICALLY_CONSISTENT_TURN"


    # --------------------------------------------------------
    # Heading and yaw-rate agree
    # --------------------------------------------------------

    if (
        heading >= 10
        and yaw >= 10
        and heading_yaw <= 15
    ):
        return "ROTATIONALLY_CONSISTENT_TURN"


    # --------------------------------------------------------
    # Heading says turn, yaw-rate does not
    # --------------------------------------------------------

    if (
        heading >= 10
        and yaw < 5
    ):
        return "HEADING_ONLY_TURN"


    # --------------------------------------------------------
    # Yaw-rate says turn, heading does not
    # --------------------------------------------------------

    if (
        yaw >= 10
        and heading < 5
    ):
        return "YAWRATE_ONLY_TURN"


    # --------------------------------------------------------
    # Both say turn but disagree
    # --------------------------------------------------------

    if (
        heading >= 10
        and yaw >= 10
        and heading_yaw > 15
    ):
        return "HEADING_YAWRATE_DISAGREEMENT"


    return "WEAK_OR_AMBIGUOUS"


# ============================================================
# Apply classification
# ============================================================

df["physical_class"] = df.apply(
    classify_window,
    axis=1
)


# ============================================================
# Moving windows
# ============================================================

moving = df[
    df["mean_speed_kmh"] >= 5.0
].copy()

print(
    f"Moving windows >=5 km/h: {len(moving):,}"
)


# ============================================================
# Classification counts
# ============================================================

print("\n" + "=" * 75)
print("PHYSICAL CLASSIFICATION")
print("=" * 75)

counts = (
    moving["physical_class"]
    .value_counts()
)

for cls, count in counts.items():

    percentage = (
        100.0 * count / len(moving)
    )

    print(
        f"{cls:35s}: "
        f"{count:7d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# Statistics helper
# ============================================================

def print_stats(name, data):

    print("\n" + name)

    if len(data) == 0:
        print("  No samples")
        return

    columns = [
        "mean_speed_kmh",
        "abs_heading_change_deg",
        "abs_yawrate_change_deg",
        "abs_heading_yawrate_error_deg",
        "max_abs_lateral_acc",
        "abs_gnss_direction_change_deg",
        "gnss_bearing_spread_deg"
    ]

    for col in columns:

        values = pd.to_numeric(
            data[col],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        print(
            f"  {col:38s} "
            f"median={values.median():8.3f} "
            f"P90={values.quantile(.90):8.3f} "
            f"P95={values.quantile(.95):8.3f} "
            f"max={values.max():8.3f}"
        )


# ============================================================
# Overall moving statistics
# ============================================================

print_stats(
    "ALL MOVING WINDOWS",
    moving
)


# ============================================================
# Physically consistent turns
# ============================================================

physical = moving[
    moving["physical_class"] ==
    "PHYSICALLY_CONSISTENT_TURN"
]

print_stats(
    "PHYSICALLY CONSISTENT TURNS",
    physical
)


# ============================================================
# Rotationally consistent turns
# ============================================================

rotational = moving[
    moving["physical_class"] ==
    "ROTATIONALLY_CONSISTENT_TURN"
]

print_stats(
    "ROTATIONALLY CONSISTENT TURNS",
    rotational
)


# ============================================================
# Heading-only
# ============================================================

heading_only = moving[
    moving["physical_class"] ==
    "HEADING_ONLY_TURN"
]

print_stats(
    "HEADING ONLY TURNS",
    heading_only
)


# ============================================================
# Yaw-rate-only
# ============================================================

yaw_only = moving[
    moving["physical_class"] ==
    "YAWRATE_ONLY_TURN"
]

print_stats(
    "YAWRATE ONLY TURNS",
    yaw_only
)


# ============================================================
# Strong physical turns
# ============================================================

print("\n" + "=" * 75)
print("STRONG PHYSICALLY CONSISTENT TURNS")
print("=" * 75)

strong = physical.sort_values(
    [
        "abs_heading_change_deg",
        "max_abs_lateral_acc"
    ],
    ascending=False
)

for _, r in strong.head(30).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['mean_speed_kmh']:6.2f} | "
        f"Head={r['heading_change_deg']:8.2f} | "
        f"Yaw={r['yawrate_integrated_change_deg']:8.2f} | "
        f"H-Y={r['heading_vs_yawrate_inverted_error_deg']:7.2f} | "
        f"LatAcc={r['mean_abs_lateral_acc']:6.3f}"
    )


# ============================================================
# Heading-only cases
# ============================================================

print("\n" + "=" * 75)
print("HEADING-ONLY TURN CASES")
print("=" * 75)

heading_only = heading_only.sort_values(
    "abs_heading_change_deg",
    ascending=False
)

for _, r in heading_only.head(30).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['mean_speed_kmh']:6.2f} | "
        f"Head={r['heading_change_deg']:8.2f} | "
        f"Yaw={r['yawrate_integrated_change_deg']:8.2f} | "
        f"H-Y={r['heading_vs_yawrate_inverted_error_deg']:7.2f} | "
        f"LatAcc={r['mean_abs_lateral_acc']:6.3f}"
    )


# ============================================================
# Yaw-rate-only cases
# ============================================================

print("\n" + "=" * 75)
print("YAWRATE-ONLY TURN CASES")
print("=" * 75)

yaw_only = yaw_only.sort_values(
    "abs_yawrate_change_deg",
    ascending=False
)

for _, r in yaw_only.head(30).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['mean_speed_kmh']:6.2f} | "
        f"Head={r['heading_change_deg']:8.2f} | "
        f"Yaw={r['yawrate_integrated_change_deg']:8.2f} | "
        f"H-Y={r['heading_vs_yawrate_inverted_error_deg']:7.2f} | "
        f"LatAcc={r['mean_abs_lateral_acc']:6.3f}"
    )


# ============================================================
# Save complete CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Save summary
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 30 - PHYSICAL TURNING CONSISTENCY SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total Step 26 windows: {len(df):,}\n"
    )

    f.write(
        f"Moving windows >=5 km/h: {len(moving):,}\n\n"
    )

    f.write(
        "PHYSICAL CLASSIFICATION COUNTS\n"
    )

    f.write("-" * 70 + "\n")

    for cls, count in counts.items():

        percentage = (
            100.0 * count / len(moving)
        )

        f.write(
            f"{cls}: {count} "
            f"({percentage:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "INTERPRETATION NOTE\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "These classifications are reliability-analysis labels only.\n"
    )

    f.write(
        "They are NOT ground-truth labels.\n"
    )

    f.write(
        "No data was deleted or modified.\n"
    )

    f.write(
        "No single sensor was selected as ground truth.\n"
    )


print("\n" + "=" * 75)
print("STEP 30 COMPLETE")
print("=" * 75)

print(
    f"\nCSV:\n{OUTPUT_CSV}"
)

print(
    f"\nSummary TXT:\n{OUTPUT_SUMMARY}"
)

print("\nNo data was deleted.")
print("No ground-truth signal was selected.")
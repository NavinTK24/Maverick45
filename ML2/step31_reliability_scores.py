import os
import numpy as np
import pandas as pd

ROOT = r"D:\Maverick\ML2"

INPUT_FILE = os.path.join(
    ROOT,
    "step30_physical_turning_consistency.csv"
)

OUTPUT_CSV = os.path.join(
    ROOT,
    "step31_reliability_scores.csv"
)

OUTPUT_SUMMARY = os.path.join(
    ROOT,
    "step31_reliability_summary.txt"
)


print("=" * 75)
print("STEP 31 - CONTINUOUS RELIABILITY ANALYSIS")
print("=" * 75)


# ============================================================
# Load Step 30
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Step 30 file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nWindows loaded: {len(df):,}")


# ============================================================
# Required columns
# ============================================================

required = [
    "dataset",
    "window_index",
    "mean_speed_kmh",
    "gnss_displacement_m",
    "gnss_bearing_spread_deg",
    "abs_gnss_direction_change_deg",
    "heading_change_deg",
    "yawrate_integrated_change_deg",
    "heading_vs_yawrate_inverted_error_deg",
    "heading_vs_gnss_absolute_error_deg",
    "yawrate_vs_gnss_inverted_error_deg",
    "mean_abs_lateral_acc",
    "physical_class"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    print("\nMissing columns:")

    for c in missing:
        print(" ", c)

    print("\nAvailable columns:")

    for c in df.columns:
        print(" ", c)

    raise RuntimeError(
        "Required Step 30 columns are missing."
    )


# ============================================================
# Utility functions
# ============================================================

def clip01(x):
    return np.clip(x, 0.0, 1.0)


def agreement_score(error_deg, good=5.0, bad=30.0):
    """
    1.0 when error <= good degrees.
    0.0 when error >= bad degrees.
    Linear transition between them.
    """

    error = np.abs(error_deg)

    score = (
        (bad - error) /
        (bad - good)
    )

    return clip01(score)


def spread_score(spread_deg, good=5.0, bad=45.0):
    """
    High score when GNSS bearing is stable.
    """

    score = (
        (bad - np.abs(spread_deg)) /
        (bad - good)
    )

    return clip01(score)


def displacement_score(
    displacement_m,
    low=1.0,
    high=10.0
):
    """
    GNSS direction becomes more meaningful
    as displacement increases.
    """

    x = np.asarray(displacement_m)

    score = (
        (x - low) /
        (high - low)
    )

    return clip01(score)


def lateral_support_score(
    lateral_acc,
    low=0.03,
    high=0.20
):
    """
    Gives stronger support when lateral acceleration
    indicates actual vehicle turning.

    This is an evidence score, not a ground-truth test.
    """

    x = np.abs(lateral_acc)

    score = (
        x - low
    ) / (
        high - low
    )

    return clip01(score)


# ============================================================
# Basic evidence
# ============================================================

df["heading_yawrate_agreement"] = agreement_score(
    df["heading_vs_yawrate_inverted_error_deg"],
    good=5.0,
    bad=30.0
)

df["heading_gnss_agreement"] = agreement_score(
    df["heading_vs_gnss_absolute_error_deg"],
    good=5.0,
    bad=30.0
)

df["yawrate_gnss_agreement"] = agreement_score(
    df["yawrate_vs_gnss_inverted_error_deg"],
    good=5.0,
    bad=30.0
)

df["gnss_stability_score"] = spread_score(
    df["gnss_bearing_spread_deg"],
    good=5.0,
    bad=45.0
)

df["gnss_displacement_score"] = displacement_score(
    df["gnss_displacement_m"],
    low=1.0,
    high=10.0
)

df["turning_physical_support"] = lateral_support_score(
    df["mean_abs_lateral_acc"],
    low=0.03,
    high=0.20
)


# ============================================================
# Speed evidence for GNSS
# ============================================================

# At very low speed GNSS direction is inherently less useful.
#
# This is not saying GNSS position is bad.
# It specifically concerns using trajectory direction as yaw evidence.

speed = df["mean_speed_kmh"].abs()

df["gnss_speed_support"] = np.select(
    [
        speed < 1.0,
        speed < 5.0,
        speed < 20.0
    ],
    [
        0.0,
        (speed - 1.0) / 4.0,
        0.75 + 0.25 * ((speed - 5.0) / 15.0)
    ],
    default=1.0
)

df["gnss_speed_support"] = clip01(
    df["gnss_speed_support"]
)


# ============================================================
# Heading reliability evidence
# ============================================================

df["heading_physical_support"] = (
    0.5 * df["heading_yawrate_agreement"]
    +
    0.5 * df["turning_physical_support"]
)

df["heading_reliability"] = (
    0.5 * df["heading_yawrate_agreement"]
    +
    0.3 * df["heading_physical_support"]
    +
    0.2 * df["heading_gnss_agreement"]
)

df["heading_reliability"] = clip01(
    df["heading_reliability"]
)


# ============================================================
# Yaw-rate reliability evidence
# ============================================================

df["yawrate_physical_support"] = (
    0.5 * df["heading_yawrate_agreement"]
    +
    0.5 * df["turning_physical_support"]
)

df["yawrate_reliability"] = (
    0.5 * df["heading_yawrate_agreement"]
    +
    0.3 * df["yawrate_physical_support"]
    +
    0.2 * df["yawrate_gnss_agreement"]
)

df["yawrate_reliability"] = clip01(
    df["yawrate_reliability"]
)


# ============================================================
# GNSS reliability evidence
# ============================================================

df["gnss_reliability"] = (
    0.30 * df["gnss_stability_score"]
    +
    0.25 * df["gnss_displacement_score"]
    +
    0.15 * df["gnss_speed_support"]
    +
    0.15 * df["heading_gnss_agreement"]
    +
    0.15 * df["yawrate_gnss_agreement"]
)

df["gnss_reliability"] = clip01(
    df["gnss_reliability"]
)


# ============================================================
# Evidence labels
# ============================================================

def evidence_label(score):

    if score >= 0.80:
        return "HIGH"

    if score >= 0.60:
        return "MEDIUM"

    if score >= 0.40:
        return "LOW"

    return "VERY_LOW"


df["heading_reliability_level"] = (
    df["heading_reliability"]
    .apply(evidence_label)
)

df["yawrate_reliability_level"] = (
    df["yawrate_reliability"]
    .apply(evidence_label)
)

df["gnss_reliability_level"] = (
    df["gnss_reliability"]
    .apply(evidence_label)
)


# ============================================================
# Relative reliability
# ============================================================

scores = df[
    [
        "heading_reliability",
        "yawrate_reliability",
        "gnss_reliability"
    ]
].values

df["most_reliable_source"] = np.select(
    [
        (scores[:, 0] >= scores[:, 1]) &
        (scores[:, 0] >= scores[:, 2]),

        (scores[:, 1] >= scores[:, 0]) &
        (scores[:, 1] >= scores[:, 2])
    ],
    [
        "HEADING",
        "YAWRATE"
    ],
    default="GNSS"
)


# ============================================================
# Important: ambiguous relative reliability
# ============================================================

sorted_scores = np.sort(
    scores,
    axis=1
)

df["top_two_score_difference"] = (
    sorted_scores[:, -1]
    -
    sorted_scores[:, -2]
)

df["relative_confidence"] = clip01(
    df["top_two_score_difference"] / 0.50
)


# ============================================================
# Moving subset
# ============================================================

moving = df[
    df["mean_speed_kmh"] >= 5.0
].copy()

print(
    f"Moving windows >=5 km/h: {len(moving):,}"
)


# ============================================================
# Overall score statistics
# ============================================================

print("\n" + "=" * 75)
print("RELIABILITY SCORE STATISTICS")
print("=" * 75)

for name in [
    "heading_reliability",
    "yawrate_reliability",
    "gnss_reliability"
]:

    values = pd.to_numeric(
        moving[name],
        errors="coerce"
    ).dropna()

    print(
        f"\n{name}"
    )

    print(
        f"  median = {values.median():.4f}"
    )

    print(
        f"  P10    = {values.quantile(.10):.4f}"
    )

    print(
        f"  P90    = {values.quantile(.90):.4f}"
    )

    print(
        f"  min    = {values.min():.4f}"
    )

    print(
        f"  max    = {values.max():.4f}"
    )


# ============================================================
# Relative source counts
# ============================================================

print("\n" + "=" * 75)
print("RELATIVE MOST-RELIABLE SOURCE")
print("=" * 75)

source_counts = (
    moving["most_reliable_source"]
    .value_counts()
)

for source, count in source_counts.items():

    pct = (
        100.0 *
        count /
        len(moving)
    )

    print(
        f"{source:10s}: "
        f"{count:7d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# Relative confidence
# ============================================================

print("\n" + "=" * 75)
print("RELATIVE CONFIDENCE")
print("=" * 75)

conf = moving[
    "relative_confidence"
]

print(
    f"median = {conf.median():.4f}"
)

print(
    f"P90    = {conf.quantile(.90):.4f}"
)

print(
    f"P95    = {conf.quantile(.95):.4f}"
)


# ============================================================
# Interesting disagreement cases
# ============================================================

print("\n" + "=" * 75)
print("LOW-CONFIDENCE MOVING WINDOWS")
print("=" * 75)

low_conf = moving[
    moving["relative_confidence"] < 0.20
].copy()

print(
    f"Count: {len(low_conf):,}"
)

low_conf = low_conf.sort_values(
    "relative_confidence"
)

for _, r in low_conf.head(40).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['mean_speed_kmh']:6.2f} | "
        f"H={r['heading_reliability']:.3f} | "
        f"Y={r['yawrate_reliability']:.3f} | "
        f"G={r['gnss_reliability']:.3f} | "
        f"conf={r['relative_confidence']:.3f} | "
        f"class={r['physical_class']}"
    )


# ============================================================
# High-confidence examples
# ============================================================

print("\n" + "=" * 75)
print("HIGH-CONFIDENCE MOVING WINDOWS")
print("=" * 75)

high_conf = moving[
    moving["relative_confidence"] >= 0.80
].copy()

print(
    f"Count: {len(high_conf):,}"
)

high_conf = high_conf.sort_values(
    "relative_confidence",
    ascending=False
)

for _, r in high_conf.head(40).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['mean_speed_kmh']:6.2f} | "
        f"H={r['heading_reliability']:.3f} | "
        f"Y={r['yawrate_reliability']:.3f} | "
        f"G={r['gnss_reliability']:.3f} | "
        f"conf={r['relative_confidence']:.3f} | "
        f"source={r['most_reliable_source']}"
    )


# ============================================================
# Save CSV
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
        "STEP 31 - CONTINUOUS RELIABILITY ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total windows: {len(df):,}\n"
    )

    f.write(
        f"Moving windows >=5 km/h: {len(moving):,}\n\n"
    )

    f.write(
        "MOST RELIABLE SOURCE COUNTS\n"
    )

    f.write("-" * 70 + "\n")

    for source, count in source_counts.items():

        pct = (
            100.0 *
            count /
            len(moving)
        )

        f.write(
            f"{source}: {count} "
            f"({pct:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "RELIABILITY SCORE MEDIANS\n"
    )

    f.write("-" * 70 + "\n")

    for name in [
        "heading_reliability",
        "yawrate_reliability",
        "gnss_reliability"
    ]:

        f.write(
            f"{name}: "
            f"{moving[name].median():.4f}\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANT NOTE\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "These are reliability scores for analysis.\n"
    )

    f.write(
        "They are NOT ground-truth labels.\n"
    )

    f.write(
        "The scoring rules are analysis assumptions and are not part of the original paper.\n"
    )

    f.write(
        "No data was deleted or modified.\n"
    )

    f.write(
        "No final yaw reference was created in Step 31.\n"
    )


print("\n" + "=" * 75)
print("STEP 31 COMPLETE")
print("=" * 75)

print(
    f"\nCSV:\n{OUTPUT_CSV}"
)

print(
    f"\nSummary TXT:\n{OUTPUT_SUMMARY}"
)

print(
    "\nNo data was deleted."
)

print(
    "No final yaw reference was created."
)
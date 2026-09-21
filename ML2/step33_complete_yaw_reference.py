from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Maverick\ML2")

STEP26 = ROOT / "step26_gnss_reliability_analysis.csv"
STEP30 = ROOT / "step30_physical_turning_consistency.csv"

OUTDIR = ROOT / "step33_yaw_reference"
OUTDIR.mkdir(exist_ok=True)

FINAL_CSV = OUTDIR / "final_yaw_reference.csv"
SUMMARY_TXT = OUTDIR / "step33_summary.txt"
RELIABILITY_CSV = OUTDIR / "reference_reliability_statistics.csv"
DISAGREEMENT_CSV = OUTDIR / "reference_disagreement_cases.csv"
TURNING_CSV = OUTDIR / "reference_turning_cases.csv"


# ============================================================
# Utilities
# ============================================================

def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def abs_circular(x):
    return np.abs(wrap180(x))


def pct(n, total):
    return 100.0 * n / total if total else 0.0


# ============================================================
# Load files
# ============================================================

print("=" * 70)
print("STEP 33 - COMPLETE YAW REFERENCE ANALYSIS")
print("=" * 70)

df26 = pd.read_csv(STEP26)
df30 = pd.read_csv(STEP30)

print(f"Step26 rows : {len(df26):,}")
print(f"Step30 rows : {len(df30):,}")

keys = [
    "dataset",
    "window_index",
    "start_row",
    "end_row"
]

df = df26.merge(
    df30[keys + [
        "abs_heading_change_deg",
        "abs_yawrate_change_deg",
        "abs_heading_yawrate_error_deg",
        "max_abs_lateral_acc",
        "physical_class"
    ]],
    on=keys,
    how="left"
)

print(f"Merged rows : {len(df):,}")

if len(df) != len(df26):
    raise RuntimeError("Merge changed the number of rows.")


# ============================================================
# Required actual columns
# ============================================================

required = [
    "mean_speed_kmh",
    "gnss_displacement_m",
    "gnss_bearing_spread_deg",
    "gnss_direction_change_deg",
    "heading_change_deg",
    "yawrate_integrated_change_deg",
    "mean_abs_lateral_acc",
    "max_abs_lateral_acc",
    "physical_class"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing expected columns:\n" +
        "\n".join(missing)
    )

print()
print("All required columns found.")


# ============================================================
# Numeric columns
# ============================================================

numeric_cols = [
    "mean_speed_kmh",
    "gnss_displacement_m",
    "gnss_bearing_spread_deg",
    "gnss_direction_change_deg",
    "heading_change_deg",
    "yawrate_integrated_change_deg",
    "mean_abs_lateral_acc",
    "max_abs_lateral_acc",
    "abs_heading_change_deg",
    "abs_yawrate_change_deg",
    "abs_heading_yawrate_error_deg"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# ============================================================
# A. Determine yaw-rate sign convention
#
# Test:
#
#   signed    = Heading - YawRate
#   inverted  = Heading + YawRate
#
# Only moving windows >= 5 km/h.
# ============================================================

moving = df["mean_speed_kmh"] >= 5.0

h = df.loc[moving, "heading_change_deg"].to_numpy()
y = df.loc[moving, "yawrate_integrated_change_deg"].to_numpy()

valid = np.isfinite(h) & np.isfinite(y)

h = h[valid]
y = y[valid]

same_error = abs_circular(h - y)
inverted_error = abs_circular(h + y)

same_median = np.median(same_error)
same_p90 = np.percentile(same_error, 90)

inv_median = np.median(inverted_error)
inv_p90 = np.percentile(inverted_error, 90)

print()
print("=" * 70)
print("A. HEADING / YAWRATE SIGN CONVENTION")
print("=" * 70)

print(f"Moving valid windows : {len(h):,}")
print()
print("Same sign:")
print(f"  Median error : {same_median:.6f} deg")
print(f"  P90         : {same_p90:.6f} deg")

print()
print("Inverted sign:")
print(f"  Median error : {inv_median:.6f} deg")
print(f"  P90         : {inv_p90:.6f} deg")

if inv_median < same_median:
    selected_sign = -1.0
    sign_name = "INVERTED"
else:
    selected_sign = 1.0
    sign_name = "SAME"

print()
print(f"SELECTED: {sign_name}")


# ============================================================
# B. Create aligned yaw-rate signal
# ============================================================

df["yawrate_aligned_change_deg"] = (
    selected_sign *
    df["yawrate_integrated_change_deg"]
)

df["heading_yaw_error_deg"] = abs_circular(
    df["heading_change_deg"] -
    df["yawrate_aligned_change_deg"]
)

df["heading_gnss_error_deg"] = abs_circular(
    df["heading_change_deg"] -
    df["gnss_direction_change_deg"]
)

df["yawrate_gnss_error_deg"] = abs_circular(
    df["yawrate_aligned_change_deg"] -
    df["gnss_direction_change_deg"]
)


# ============================================================
# C. Vehicle consensus
#
# Strong:
# Heading and aligned YawRate within 5 degrees
#
# Moderate:
# 5-10 degrees
#
# Disagreement:
# >10 degrees
# ============================================================

df["vehicle_consensus"] = (
    df["heading_yaw_error_deg"] <= 5.0
)

df["vehicle_consensus_moderate"] = (
    (df["heading_yaw_error_deg"] > 5.0) &
    (df["heading_yaw_error_deg"] <= 10.0)
)

df["vehicle_disagreement"] = (
    df["heading_yaw_error_deg"] > 10.0
)

df["vehicle_consensus_state"] = np.select(
    [
        df["vehicle_consensus"],
        df["vehicle_consensus_moderate"],
        df["vehicle_disagreement"]
    ],
    [
        "STRONG",
        "MODERATE",
        "DISAGREEMENT"
    ],
    default="UNKNOWN"
)


# ============================================================
# D. Physical turning state
#
# We preserve Step30's physical classification.
# ============================================================

df["physical_turn_state"] = df["physical_class"]


# ============================================================
# E. GNSS reliability
#
# This is NOT claiming GNSS is ground truth.
#
# STRONG:
#   speed >= 20
#   displacement >= 5 m
#   bearing spread <= 20 deg
#
# MODERATE:
#   speed >= 5
#   displacement >= 1 m
#   bearing spread <= 45 deg
#
# otherwise weak.
# ============================================================

df["gnss_strong"] = (
    (df["mean_speed_kmh"] >= 20.0) &
    (df["gnss_displacement_m"] >= 5.0) &
    (df["gnss_bearing_spread_deg"] <= 20.0)
)

df["gnss_moderate"] = (
    (df["mean_speed_kmh"] >= 5.0) &
    (df["gnss_displacement_m"] >= 1.0) &
    (df["gnss_bearing_spread_deg"] <= 45.0)
)

df["gnss_state"] = np.select(
    [
        df["gnss_strong"],
        df["gnss_moderate"]
    ],
    [
        "STRONG_TRAJECTORY",
        "MODERATE_TRAJECTORY"
    ],
    default="WEAK_OR_UNSTABLE"
)


# ============================================================
# F. Turning determination
#
# 10 degrees is used only as a classification threshold.
# ============================================================

TURN_THRESHOLD = 10.0

df["heading_turn"] = (
    df["abs_heading_change_deg"] >= TURN_THRESHOLD
)

df["yawrate_turn"] = (
    df["abs_yawrate_change_deg"] >= TURN_THRESHOLD
)

df["substantial_turn"] = (
    df["heading_turn"] |
    df["yawrate_turn"]
)


# ============================================================
# G. Three-signal agreement
# ============================================================

df["heading_gnss_agree"] = (
    df["heading_gnss_error_deg"] <= 10.0
)

df["yawrate_gnss_agree"] = (
    df["yawrate_gnss_error_deg"] <= 10.0
)


def classify_consensus(row):

    vehicle = row["vehicle_consensus"]
    hg = row["heading_gnss_agree"]
    yg = row["yawrate_gnss_agree"]
    turn = row["substantial_turn"]

    if not turn:

        if vehicle and (hg or yg):
            return "STRAIGHT_ALL_AGREE"

        if vehicle:
            return "STRAIGHT_VEHICLE_CONSENSUS"

        if row["gnss_state"] == "WEAK_OR_UNSTABLE":
            return "STRAIGHT_NO_RELIABLE_GNSS"

        return "STRAIGHT_UNRESOLVED"

    # Turning

    if vehicle and hg and yg:
        return "TURN_ALL_THREE_AGREE"

    if vehicle and not hg and not yg:

        if row["gnss_state"] == "WEAK_OR_UNSTABLE":
            return "TURN_VEHICLE_CONSENSUS_GNSS_UNRELIABLE"

        return "TURN_VEHICLE_CONSENSUS_GNSS_DISAGREE"

    if vehicle and hg:
        return "TURN_HEADING_GNSS_AGREE"

    if vehicle and yg:
        return "TURN_YAWRATE_GNSS_AGREE"

    if row["heading_turn"] and not row["yawrate_turn"]:

        if hg:
            return "HEADING_ONLY_GNSS_AGREE"

        return "HEADING_ONLY_UNRESOLVED"

    if row["yawrate_turn"] and not row["heading_turn"]:

        if yg:
            return "YAWRATE_ONLY_GNSS_AGREE"

        return "YAWRATE_ONLY_UNRESOLVED"

    return "TURN_VEHICLE_DISAGREEMENT"


df["three_signal_consensus"] = df.apply(
    classify_consensus,
    axis=1
)


# ============================================================
# H. Candidate reference selection
#
# IMPORTANT:
#
# This is a DATASET CONSTRUCTION.
#
# It is NOT the paper's original ground truth.
#
# Strong vehicle consensus is preferred because Heading
# and YawRate are two independent vehicle-rotation signals.
#
# GNSS is used as supporting evidence where reliable.
# ============================================================

def choose_reference(row):

    h = row["heading_change_deg"]
    y = row["yawrate_aligned_change_deg"]
    g = row["gnss_direction_change_deg"]

    vehicle_error = row["heading_yaw_error_deg"]
    h_g_error = row["heading_gnss_error_deg"]
    y_g_error = row["yawrate_gnss_error_deg"]

    vehicle_strong = vehicle_error <= 5.0

    gnss_strong = row["gnss_state"] == "STRONG_TRAJECTORY"
    gnss_available = row["gnss_state"] != "WEAK_OR_UNSTABLE"

    # --------------------------------------------------------
    # 1. All three agree
    # --------------------------------------------------------

    if (
        vehicle_strong and
        gnss_available and
        h_g_error <= 10.0 and
        y_g_error <= 10.0
    ):
        return (
            "THREE_SIGNAL_SUPPORTED",
            h,
            1.00
        )

    # --------------------------------------------------------
    # 2. Vehicle consensus + GNSS unreliable
    # --------------------------------------------------------

    if vehicle_strong and not gnss_available:
        return (
            "VEHICLE_CONSENSUS_GNSS_UNRELIABLE",
            h,
            0.95
        )

    # --------------------------------------------------------
    # 3. Vehicle consensus + GNSS disagreement
    #
    # Both vehicle rotational signals agree, so preserve
    # the vehicle rotational estimate.
    # --------------------------------------------------------

    if vehicle_strong and h_g_error > 10.0 and y_g_error > 10.0:
        return (
            "VEHICLE_CONSENSUS_GNSS_DISAGREEMENT",
            h,
            0.90
        )

    # --------------------------------------------------------
    # 4. Vehicle consensus but partial GNSS agreement
    # --------------------------------------------------------

    if vehicle_strong:
        return (
            "VEHICLE_CONSENSUS",
            h,
            0.85
        )

    # --------------------------------------------------------
    # 5. Vehicle disagreement but strong GNSS
    #
    # GNSS becomes a candidate only when the vehicle signals
    # disagree and trajectory information is strong.
    # --------------------------------------------------------

    if (
        gnss_strong and
        h_g_error <= 10.0 and
        y_g_error <= 10.0
    ):
        return (
            "GNSS_SUPPORTED_VEHICLE_DISAGREEMENT",
            g,
            0.70
        )

    # --------------------------------------------------------
    # 6. Very small motion
    #
    # For near-zero rotation, use zero rather than creating
    # artificial rotation.
    # --------------------------------------------------------

    if (
        abs(h) < 5.0 and
        abs(y) < 5.0
    ):
        return (
            "LOW_ROTATION",
            0.0,
            0.60
        )

    # --------------------------------------------------------
    # 7. Unresolved
    # --------------------------------------------------------

    return (
        "AMBIGUOUS",
        np.nan,
        0.0
    )


ref = df.apply(
    choose_reference,
    axis=1,
    result_type="expand"
)

ref.columns = [
    "reference_source",
    "reference_delta_yaw_deg",
    "reference_confidence"
]

df = pd.concat(
    [df, ref],
    axis=1
)


# ============================================================
# I. Final yaw normalization
# ============================================================

df["reference_delta_yaw_deg"] = wrap180(
    df["reference_delta_yaw_deg"]
)


# ============================================================
# J. Yaw-only quaternion
#
# q = [qw, qx, qy, qz]
# ============================================================

yaw_rad = np.deg2rad(
    df["reference_delta_yaw_deg"]
)

df["dq_w"] = np.cos(yaw_rad / 2.0)
df["dq_x"] = 0.0
df["dq_y"] = 0.0
df["dq_z"] = np.sin(yaw_rad / 2.0)


# ============================================================
# K. Validation
# ============================================================

df["quaternion_norm"] = np.sqrt(
    df["dq_w"]**2 +
    df["dq_x"]**2 +
    df["dq_y"]**2 +
    df["dq_z"]**2
)

df["quaternion_norm_error"] = (
    np.abs(df["quaternion_norm"] - 1.0)
)

df["reference_valid"] = (
    np.isfinite(df["reference_delta_yaw_deg"]) &
    np.isfinite(df["dq_w"]) &
    np.isfinite(df["dq_z"])
)


# ============================================================
# L. Save COMPLETE reference dataset
# ============================================================

df.to_csv(
    FINAL_CSV,
    index=False,
    float_format="%.8f"
)


# ============================================================
# M. Save disagreement cases
# ============================================================

disagreement = (
    (df["heading_yaw_error_deg"] > 10.0) |
    (df["heading_gnss_error_deg"] > 20.0) |
    (df["yawrate_gnss_error_deg"] > 20.0)
)

df.loc[
    disagreement
].to_csv(
    DISAGREEMENT_CSV,
    index=False,
    float_format="%.8f"
)


# ============================================================
# N. Save turning cases
# ============================================================

turning = (
    df["substantial_turn"]
)

df.loc[
    turning
].to_csv(
    TURNING_CSV,
    index=False,
    float_format="%.8f"
)


# ============================================================
# O. Reliability statistics
# ============================================================

total = len(df)

stats = []

for source, g in df.groupby(
    "reference_source",
    dropna=False
):

    stats.append({
        "reference_source": source,
        "count": len(g),
        "percentage": pct(len(g), total),
        "mean_confidence":
            g["reference_confidence"].mean(),
        "median_confidence":
            g["reference_confidence"].median(),
        "median_abs_reference_yaw":
            np.median(
                np.abs(
                    g["reference_delta_yaw_deg"]
                )
            ),
        "p90_abs_reference_yaw":
            np.percentile(
                np.abs(
                    g["reference_delta_yaw_deg"]
                ),
                90
            )
    })

pd.DataFrame(stats).to_csv(
    RELIABILITY_CSV,
    index=False
)


# ============================================================
# P. Summary
# ============================================================

summary = []

summary.append("=" * 70)
summary.append("STEP 33 - COMPLETE YAW REFERENCE ANALYSIS")
summary.append("=" * 70)
summary.append("")

summary.append(f"Total windows: {total:,}")
summary.append(
    f"Moving windows >=5 km/h: "
    f"{int((df['mean_speed_kmh'] >= 5).sum()):,}"
)
summary.append("")

summary.append("SIGN CONVENTION")
summary.append("----------------")
summary.append(
    f"Same-sign median error     : "
    f"{same_median:.6f} deg"
)
summary.append(
    f"Same-sign P90              : "
    f"{same_p90:.6f} deg"
)
summary.append(
    f"Inverted-sign median error : "
    f"{inv_median:.6f} deg"
)
summary.append(
    f"Inverted-sign P90          : "
    f"{inv_p90:.6f} deg"
)
summary.append(
    f"Selected convention        : "
    f"{sign_name}"
)
summary.append("")

summary.append("REFERENCE SOURCES")
summary.append("-----------------")

for source, count in (
    df["reference_source"]
    .value_counts(dropna=False)
    .items()
):

    summary.append(
        f"{str(source):45s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

summary.append("")

summary.append("THREE-SIGNAL CONSENSUS")
summary.append("----------------------")

for source, count in (
    df["three_signal_consensus"]
    .value_counts(dropna=False)
    .items()
):

    summary.append(
        f"{str(source):45s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

summary.append("")

summary.append("GNSS STATES")
summary.append("-----------")

for source, count in (
    df["gnss_state"]
    .value_counts(dropna=False)
    .items()
):

    summary.append(
        f"{str(source):30s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

summary.append("")

summary.append("PHYSICAL STATES")
summary.append("---------------")

for source, count in (
    df["physical_turn_state"]
    .value_counts(dropna=False)
    .items()
):

    summary.append(
        f"{str(source):35s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

summary.append("")

valid_count = int(
    df["reference_valid"].sum()
)

ambiguous_count = int(
    (df["reference_source"] == "AMBIGUOUS").sum()
)

summary.append("FINAL REFERENCE")
summary.append("---------------")
summary.append(
    f"Valid reference rows : "
    f"{valid_count:,} "
    f"({pct(valid_count,total):.2f}%)"
)
summary.append(
    f"Ambiguous rows       : "
    f"{ambiguous_count:,} "
    f"({pct(ambiguous_count,total):.2f}%)"
)

summary.append(
    f"Rows lost            : "
    f"{len(df26) - len(df):,}"
)

summary.append(
    f"Maximum quaternion norm error : "
    f"{df['quaternion_norm_error'].max():.12f}"
)

summary.append("")

valid_yaw = df.loc[
    df["reference_valid"],
    "reference_delta_yaw_deg"
]

summary.append("REFERENCE ΔYAW")
summary.append("--------------")
summary.append(
    f"Min    : {valid_yaw.min():.6f} deg"
)
summary.append(
    f"Max    : {valid_yaw.max():.6f} deg"
)
summary.append(
    f"Median : {valid_yaw.median():.6f} deg"
)
summary.append(
    f"P90    : {valid_yaw.quantile(.90):.6f} deg"
)
summary.append(
    f"P95    : {valid_yaw.quantile(.95):.6f} deg"
)
summary.append(
    f"P99    : {valid_yaw.quantile(.99):.6f} deg"
)

summary.append("")
summary.append("IMPORTANT")
summary.append("---------")
summary.append(
    "The final reference is a dataset-derived yaw reference."
)
summary.append(
    "It is not the original paper's external ground-truth attitude."
)
summary.append(
    "All original windows are retained."
)
summary.append(
    "Reference source and supporting measurements are retained "
    "for every window."
)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 33 COMPLETE")
print("=" * 70)

print()
print(f"Total windows : {total:,}")
print(f"Rows lost    : {len(df26)-len(df):,}")

print()
print("Selected sign:")
print(f"  {sign_name}")

print()
print("REFERENCE SOURCE COUNTS")

for source, count in (
    df["reference_source"]
    .value_counts(dropna=False)
    .items()
):

    print(
        f"  {str(source):45s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

print()
print("THREE-SIGNAL CONSENSUS")

for source, count in (
    df["three_signal_consensus"]
    .value_counts(dropna=False)
    .items()
):

    print(
        f"  {str(source):45s} "
        f"{count:8,d} "
        f"({pct(count,total):6.2f}%)"
    )

print()
print("FINAL REFERENCE")
print(
    f"  Valid     : {valid_count:,} "
    f"({pct(valid_count,total):.2f}%)"
)
print(
    f"  Ambiguous : {ambiguous_count:,} "
    f"({pct(ambiguous_count,total):.2f}%)"
)

print()
print("OUTPUT FILES")
print("------------")
print(FINAL_CSV)
print(SUMMARY_TXT)
print(RELIABILITY_CSV)
print(DISAGREEMENT_CSV)
print(TURNING_CSV)

print()
print("Done.")
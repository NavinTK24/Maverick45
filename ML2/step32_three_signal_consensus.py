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
    "step32_three_signal_consensus.csv"
)

OUTPUT_SUMMARY = os.path.join(
    ROOT,
    "step32_three_signal_summary.txt"
)


print("=" * 75)
print("STEP 32 - THREE-SIGNAL CONSENSUS ANALYSIS")
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

    raise RuntimeError(
        "Required columns are missing."
    )


# ============================================================
# Absolute values
# ============================================================

df["abs_heading_change"] = (
    df["heading_change_deg"].abs()
)

df["abs_yawrate_change"] = (
    df["yawrate_integrated_change_deg"].abs()
)

df["heading_yawrate_error"] = (
    df["heading_vs_yawrate_inverted_error_deg"].abs()
)

df["heading_gnss_error"] = (
    df["heading_vs_gnss_absolute_error_deg"].abs()
)

df["yawrate_gnss_error"] = (
    df["yawrate_vs_gnss_inverted_error_deg"].abs()
)

df["gnss_direction_change"] = (
    df["abs_gnss_direction_change_deg"].abs()
)

df["gnss_spread"] = (
    df["gnss_bearing_spread_deg"].abs()
)

df["lateral_acc"] = (
    df["mean_abs_lateral_acc"].abs()
)

df["speed"] = (
    df["mean_speed_kmh"].abs()
)


# ============================================================
# PART 1
# Vehicle rotational consensus
# ============================================================

def vehicle_consensus(error):

    if error <= 5:
        return "STRONG"

    if error <= 10:
        return "MODERATE"

    if error <= 20:
        return "DISAGREEMENT"

    return "STRONG_DISAGREEMENT"


df["vehicle_rotational_consensus"] = (
    df["heading_yawrate_error"]
    .apply(vehicle_consensus)
)


# ============================================================
# PART 2
# Physical turning state
# ============================================================

def physical_state(row):

    h = row["abs_heading_change"]
    y = row["abs_yawrate_change"]
    lat = row["lateral_acc"]

    # No substantial rotation
    if h < 5 and y < 5:
        return "NO_SUBSTANTIAL_TURN"

    # Both indicate rotation
    if h >= 10 and y >= 10:

        if lat >= 0.10:
            return "BOTH_ROTATE_LATERAL_SUPPORT"

        return "BOTH_ROTATE_WEAK_LATERAL"

    # Heading only
    if h >= 10 and y < 5:
        return "HEADING_ONLY"

    # Yaw rate only
    if y >= 10 and h < 5:
        return "YAWRATE_ONLY"

    # Intermediate motion
    return "PARTIAL_ROTATION"


df["physical_turning_state"] = (
    df.apply(
        physical_state,
        axis=1
    )
)


# ============================================================
# PART 3
# GNSS trajectory validity
# ============================================================

def gnss_state(row):

    speed = row["speed"]
    displacement = row["gnss_displacement_m"]
    spread = row["gnss_spread"]

    # GNSS direction is not meaningful at essentially
    # stationary / negligible-displacement conditions.
    if speed < 1.0 or displacement < 0.1:
        return "INSUFFICIENT_TRAJECTORY"

    # Strong trajectory evidence
    if (
        displacement >= 5.0
        and spread <= 10.0
    ):
        return "STRONG_TRAJECTORY"

    # Moderate trajectory evidence
    if (
        displacement >= 1.0
        and spread <= 20.0
    ):
        return "MODERATE_TRAJECTORY"

    # Weak but usable trajectory
    if (
        displacement >= 0.5
        and spread <= 45.0
    ):
        return "WEAK_TRAJECTORY"

    # Unstable
    return "UNSTABLE_TRAJECTORY"


df["gnss_state"] = (
    df.apply(
        gnss_state,
        axis=1
    )
)


# ============================================================
# PART 4
# Pairwise agreement
# ============================================================

df["heading_yawrate_agree"] = (
    df["heading_yawrate_error"] <= 10
)

df["heading_gnss_agree"] = (
    df["heading_gnss_error"] <= 10
)

df["yawrate_gnss_agree"] = (
    df["yawrate_gnss_error"] <= 10
)


# ============================================================
# PART 5
# Three-signal consensus
# ============================================================

def consensus(row):

    H_Y = row["heading_yawrate_agree"]
    H_G = row["heading_gnss_agree"]
    Y_G = row["yawrate_gnss_agree"]

    gnss_good = row["gnss_state"] in [
        "STRONG_TRAJECTORY",
        "MODERATE_TRAJECTORY",
        "WEAK_TRAJECTORY"
    ]

    gnss_bad = row["gnss_state"] in [
        "INSUFFICIENT_TRAJECTORY",
        "UNSTABLE_TRAJECTORY"
    ]

    physical = row["physical_turning_state"]


    # --------------------------------------------------------
    # No substantial turn
    # --------------------------------------------------------

    if physical == "NO_SUBSTANTIAL_TURN":

        if gnss_bad:
            return "STRAIGHT_NO_GNSS_DIRECTION"

        if H_Y and H_G and Y_G:
            return "STRAIGHT_ALL_AGREE"

        if H_Y and gnss_good:
            return "STRAIGHT_VEHICLE_CONSENSUS"

        return "STRAIGHT_OR_UNRESOLVED"


    # --------------------------------------------------------
    # Genuine vehicle rotational evidence
    # --------------------------------------------------------

    if physical in [
        "BOTH_ROTATE_LATERAL_SUPPORT",
        "BOTH_ROTATE_WEAK_LATERAL"
    ]:

        if H_Y:

            if H_G and Y_G:
                return "TURN_ALL_THREE_AGREE"

            if gnss_bad:
                return "TURN_VEHICLE_CONSENSUS_GNSS_WEAK"

            if not H_G and not Y_G:
                return "TURN_VEHICLE_CONSENSUS_GNSS_DISAGREE"

            if H_G:
                return "TURN_HEADING_GNSS_AGREE"

            if Y_G:
                return "TURN_YAWRATE_GNSS_AGREE"

        return "TURN_VEHICLE_DISAGREEMENT"


    # --------------------------------------------------------
    # Heading-only
    # --------------------------------------------------------

    if physical == "HEADING_ONLY":

        if H_G and gnss_good:
            return "HEADING_ONLY_GNSS_AGREE"

        if gnss_bad:
            return "HEADING_ONLY_GNSS_WEAK"

        return "HEADING_ONLY_UNRESOLVED"


    # --------------------------------------------------------
    # Yaw-rate-only
    # --------------------------------------------------------

    if physical == "YAWRATE_ONLY":

        if Y_G and gnss_good:
            return "YAWRATE_ONLY_GNSS_AGREE"

        if gnss_bad:
            return "YAWRATE_ONLY_GNSS_WEAK"

        return "YAWRATE_ONLY_UNRESOLVED"


    # --------------------------------------------------------
    # Partial rotation
    # --------------------------------------------------------

    if H_Y:

        if H_G and Y_G:
            return "PARTIAL_ALL_AGREE"

        if gnss_bad:
            return "PARTIAL_VEHICLE_CONSENSUS"

        return "PARTIAL_VEHICLE_DISAGREEMENT"

    return "UNRESOLVED"


df["three_signal_consensus"] = (
    df.apply(
        consensus,
        axis=1
    )
)


# ============================================================
# Moving subset
# ============================================================

moving = df[
    df["speed"] >= 5
].copy()

print(
    f"Moving windows >=5 km/h: {len(moving):,}"
)


# ============================================================
# Print vehicle consensus
# ============================================================

print("\n" + "=" * 75)
print("VEHICLE HEADING vs YAWRATE")
print("=" * 75)

vc = (
    moving[
        "vehicle_rotational_consensus"
    ]
    .value_counts()
)

for name, count in vc.items():

    pct = (
        100 * count / len(moving)
    )

    print(
        f"{name:25s}: "
        f"{count:7d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# Physical states
# ============================================================

print("\n" + "=" * 75)
print("PHYSICAL TURNING STATES")
print("=" * 75)

pc = (
    moving[
        "physical_turning_state"
    ]
    .value_counts()
)

for name, count in pc.items():

    pct = (
        100 * count / len(moving)
    )

    print(
        f"{name:35s}: "
        f"{count:7d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# GNSS states
# ============================================================

print("\n" + "=" * 75)
print("GNSS TRAJECTORY STATES")
print("=" * 75)

gc = (
    moving[
        "gnss_state"
    ]
    .value_counts()
)

for name, count in gc.items():

    pct = (
        100 * count / len(moving)
    )

    print(
        f"{name:30s}: "
        f"{count:7d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# Final three-signal consensus
# ============================================================

print("\n" + "=" * 75)
print("THREE-SIGNAL CONSENSUS")
print("=" * 75)

cc = (
    moving[
        "three_signal_consensus"
    ]
    .value_counts()
)

for name, count in cc.items():

    pct = (
        100 * count / len(moving)
    )

    print(
        f"{name:40s}: "
        f"{count:7d} "
        f"({pct:6.2f}%)"
    )


# ============================================================
# Important reference candidates
# ============================================================

candidate_states = [
    "TURN_ALL_THREE_AGREE",
    "TURN_VEHICLE_CONSENSUS_GNSS_WEAK",
    "TURN_VEHICLE_CONSENSUS_GNSS_DISAGREE",
    "STRAIGHT_ALL_AGREE",
    "STRAIGHT_VEHICLE_CONSENSUS"
]

print("\n" + "=" * 75)
print("CONSENSUS CANDIDATE WINDOWS")
print("=" * 75)

candidates = moving[
    moving["three_signal_consensus"]
    .isin(candidate_states)
].copy()

print(
    f"Candidate windows: {len(candidates):,}"
)

for _, r in candidates.head(50).iterrows():

    print(
        f"{str(r['dataset']):8s} "
        f"w{int(r['window_index']):5d} | "
        f"speed={r['speed']:6.2f} | "
        f"Head={r['heading_change_deg']:8.2f} | "
        f"Yaw={r['yawrate_integrated_change_deg']:8.2f} | "
        f"GNSS={r['gnss_direction_change']:8.2f} | "
        f"H-Y={r['heading_yawrate_error']:6.2f} | "
        f"H-G={r['heading_gnss_error']:6.2f} | "
        f"Y-G={r['yawrate_gnss_error']:6.2f} | "
        f"state={r['three_signal_consensus']}"
    )


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Summary
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 32 - THREE-SIGNAL CONSENSUS SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total windows: {len(df):,}\n"
    )

    f.write(
        f"Moving windows >=5 km/h: {len(moving):,}\n\n"
    )

    f.write(
        "VEHICLE ROTATIONAL CONSENSUS\n"
    )

    f.write("-" * 70 + "\n")

    for name, count in vc.items():

        pct = (
            100 * count / len(moving)
        )

        f.write(
            f"{name}: {count} "
            f"({pct:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "PHYSICAL TURNING STATES\n"
    )

    f.write("-" * 70 + "\n")

    for name, count in pc.items():

        pct = (
            100 * count / len(moving)
        )

        f.write(
            f"{name}: {count} "
            f"({pct:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "GNSS TRAJECTORY STATES\n"
    )

    f.write("-" * 70 + "\n")

    for name, count in gc.items():

        pct = (
            100 * count / len(moving)
        )

        f.write(
            f"{name}: {count} "
            f"({pct:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "THREE-SIGNAL CONSENSUS\n"
    )

    f.write("-" * 70 + "\n")

    for name, count in cc.items():

        pct = (
            100 * count / len(moving)
        )

        f.write(
            f"{name}: {count} "
            f"({pct:.2f}%)\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANT NOTE\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "These are descriptive reliability/consensus classifications.\n"
    )

    f.write(
        "They are NOT ground-truth labels.\n"
    )

    f.write(
        "No data was deleted or modified.\n"
    )

    f.write(
        "No final yaw reference was created.\n"
    )


print("\n" + "=" * 75)
print("STEP 32 COMPLETE")
print("=" * 75)

print(
    f"\nCSV:\n{OUTPUT_CSV}"
)

print(
    f"\nSummary TXT:\n{OUTPUT_SUMMARY}"
)

print("\nNo data was deleted.")
print("No final yaw reference was created.")
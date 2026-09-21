import os
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE = r"D:\Maverick\ML2"

INPUT_FILE = os.path.join(
    BASE,
    "step25_gnss_trajectory_analysis.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "step26_gnss_reliability_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    BASE,
    "step26_gnss_reliability_summary.csv"
)


# =========================================================
# LOAD STEP 25 RESULTS
# =========================================================

print("=" * 70)
print("STEP 26 - GNSS REFERENCE RELIABILITY ANALYSIS")
print("=" * 70)

print()
print("Input:")
print(INPUT_FILE)


if not os.path.exists(INPUT_FILE):

    print()
    print("ERROR: Step 25 output file not found.")
    raise SystemExit(1)


df = pd.read_csv(
    INPUT_FILE
)


print()
print(
    f"Rows loaded: {len(df):,}"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "dataset",
    "window_index",
    "mean_speed_kmh",
    "gnss_displacement_m",
    "gnss_total_distance_m",
    "gnss_direction_change_deg",
    "gnss_bearing_spread_deg",
    "heading_change_deg",
    "yawrate_integrated_change_deg",
    "heading_vs_gnss_error_deg",
    "yawrate_vs_gnss_signed_error_deg",
    "yawrate_vs_gnss_inverted_error_deg",
    "heading_vs_yawrate_signed_error_deg",
    "heading_vs_yawrate_inverted_error_deg",
    "mean_abs_lateral_acc",
    "mean_gps_satellites"
]


missing = [
    c for c in required_columns
    if c not in df.columns
]


if missing:

    print()
    print("ERROR: Missing columns:")
    for c in missing:
        print("  ", c)

    raise SystemExit(1)


# =========================================================
# BASIC VALIDITY
# =========================================================

df["valid_gnss_direction"] = (
    np.isfinite(
        df["gnss_direction_change_deg"]
    )
)

df["valid_gnss_displacement"] = (
    np.isfinite(
        df["gnss_displacement_m"]
    )
)


# =========================================================
# GNSS PATH / DISPLACEMENT RELATION
# =========================================================

df["path_to_displacement_ratio"] = (
    df["gnss_total_distance_m"]
    /
    df["gnss_displacement_m"]
)


df.loc[
    ~np.isfinite(
        df["path_to_displacement_ratio"]
    ),
    "path_to_displacement_ratio"
] = np.nan


# =========================================================
# DESCRIPTIVE DISPLACEMENT BINS
#
# These are analysis bins only.
# They do NOT mean "good" or "bad".
# =========================================================

def displacement_category(x):

    if not np.isfinite(x):
        return "invalid"

    if x == 0:
        return "0 m"

    if x < 0.1:
        return "0-0.1 m"

    if x < 1:
        return "0.1-1 m"

    if x < 5:
        return "1-5 m"

    if x < 10:
        return "5-10 m"

    return ">10 m"


df["displacement_category"] = (
    df["gnss_displacement_m"]
    .apply(displacement_category)
)


# =========================================================
# SPEED CATEGORIES
# =========================================================

def speed_category(x):

    if not np.isfinite(x):
        return "invalid"

    if x <= 1:
        return "0-1 km/h"

    if x <= 5:
        return "1-5 km/h"

    if x <= 10:
        return "5-10 km/h"

    if x <= 20:
        return "10-20 km/h"

    if x <= 40:
        return "20-40 km/h"

    if x <= 60:
        return "40-60 km/h"

    return ">60 km/h"


df["speed_category"] = (
    df["mean_speed_kmh"]
    .apply(speed_category)
)


# =========================================================
# GNSS DIRECTION CHANGE MAGNITUDE
# =========================================================

df["abs_gnss_direction_change_deg"] = (
    np.abs(
        df["gnss_direction_change_deg"]
    )
)


# =========================================================
# AGREEMENT CATEGORIES
#
# Again, descriptive only.
# =========================================================

def error_category(x):

    if not np.isfinite(x):
        return "invalid"

    if x <= 2:
        return "<=2 deg"

    if x <= 5:
        return "2-5 deg"

    if x <= 10:
        return "5-10 deg"

    if x <= 20:
        return "10-20 deg"

    if x <= 45:
        return "20-45 deg"

    return ">45 deg"


df["heading_gnss_error_category"] = (
    df["heading_vs_gnss_error_deg"]
    .apply(error_category)
)


df["yawrate_gnss_inverted_error_category"] = (
    df[
        "yawrate_vs_gnss_inverted_error_deg"
    ]
    .apply(error_category)
)


# =========================================================
# DIRECTION-STABILITY CATEGORY
# =========================================================

def stability_category(x):

    if not np.isfinite(x):
        return "invalid"

    if x <= 2:
        return "<=2 deg spread"

    if x <= 5:
        return "2-5 deg spread"

    if x <= 10:
        return "5-10 deg spread"

    if x <= 20:
        return "10-20 deg spread"

    if x <= 45:
        return "20-45 deg spread"

    return ">45 deg spread"


df["gnss_stability_category"] = (
    df["gnss_bearing_spread_deg"]
    .apply(stability_category)
)


# =========================================================
# LARGE GNSS DIRECTION CHANGE FLAG
# =========================================================

df["large_gnss_change_45deg"] = (
    df["abs_gnss_direction_change_deg"]
    >= 45
)


df["large_gnss_change_90deg"] = (
    df["abs_gnss_direction_change_deg"]
    >= 90
)


# =========================================================
# VERY SMALL DISPLACEMENT FLAG
#
# This is a descriptive flag only.
# =========================================================

df["very_small_displacement"] = (
    df["gnss_displacement_m"]
    < 0.1
)


df["zero_displacement"] = (
    df["gnss_displacement_m"]
    == 0
)


# =========================================================
# AGREEMENT FLAGS
# =========================================================

df["heading_gnss_within_5deg"] = (
    df["heading_vs_gnss_error_deg"]
    <= 5
)


df["heading_gnss_within_10deg"] = (
    df["heading_vs_gnss_error_deg"]
    <= 10
)


df["yawrate_gnss_inverted_within_5deg"] = (
    df[
        "yawrate_vs_gnss_inverted_error_deg"
    ]
    <= 5
)


df["yawrate_gnss_inverted_within_10deg"] = (
    df[
        "yawrate_vs_gnss_inverted_error_deg"
    ]
    <= 10
)


# =========================================================
# SAVE DETAILED WINDOW-LEVEL RESULTS
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    f"Detailed reliability results saved:"
)

print(
    OUTPUT_FILE
)


# =========================================================
# HELPER FOR SUMMARY PRINTING
# =========================================================

def stats(values):

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return "no valid data"

    return (
        f"N={len(values):,}, "
        f"median={np.percentile(values,50):.3f}, "
        f"P90={np.percentile(values,90):.3f}, "
        f"P95={np.percentile(values,95):.3f}, "
        f"P99={np.percentile(values,99):.3f}, "
        f"max={np.max(values):.3f}"
    )


# =========================================================
# 1. DISPLACEMENT DISTRIBUTION
# =========================================================

print()
print("=" * 70)
print("1. GNSS DISPLACEMENT DISTRIBUTION")
print("=" * 70)

print(
    stats(
        df["gnss_displacement_m"]
    )
)


print()
print(
    df[
        "displacement_category"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)


# =========================================================
# 2. GNSS DIRECTION STABILITY
# =========================================================

print()
print("=" * 70)
print("2. GNSS DIRECTION STABILITY")
print("=" * 70)

print(
    stats(
        df[
            "gnss_bearing_spread_deg"
        ]
    )
)


print()
print(
    df[
        "gnss_stability_category"
    ]
    .value_counts(
        sort=False
    )
    .to_string()
)


# =========================================================
# 3. SPEED VS GNSS RELIABILITY
# =========================================================

print()
print("=" * 70)
print("3. GNSS BEHAVIOUR BY VEHICLE SPEED")
print("=" * 70)


speed_order = [
    "0-1 km/h",
    "1-5 km/h",
    "5-10 km/h",
    "10-20 km/h",
    "20-40 km/h",
    "40-60 km/h",
    ">60 km/h"
]


for category in speed_order:

    subset = df[
        df["speed_category"]
        ==
        category
    ]


    print()
    print(
        f"--- {category} ---"
    )

    print(
        f"Windows: {len(subset):,}"
    )


    if len(subset) == 0:
        continue


    print(
        "GNSS displacement:",
        stats(
            subset[
                "gnss_displacement_m"
            ]
        )
    )


    print(
        "GNSS direction change:",
        stats(
            subset[
                "abs_gnss_direction_change_deg"
            ]
        )
    )


    print(
        "GNSS bearing spread:",
        stats(
            subset[
                "gnss_bearing_spread_deg"
            ]
        )
    )


    print(
        "Heading-GNSS error:",
        stats(
            subset[
                "heading_vs_gnss_error_deg"
            ]
        )
    )


    print(
        "YawRate-GNSS inverted error:",
        stats(
            subset[
                "yawrate_vs_gnss_inverted_error_deg"
            ]
        )
    )


# =========================================================
# 4. DISPLACEMENT VS HEADING AGREEMENT
# =========================================================

print()
print("=" * 70)
print("4. HEADING VS GNSS BY GNSS DISPLACEMENT")
print("=" * 70)


displacement_order = [
    "0 m",
    "0-0.1 m",
    "0.1-1 m",
    "1-5 m",
    "5-10 m",
    ">10 m"
]


for category in displacement_order:

    subset = df[
        df[
            "displacement_category"
        ]
        ==
        category
    ]


    print()
    print(
        f"--- {category} ---"
    )


    print(
        f"Windows: {len(subset):,}"
    )


    if len(subset) == 0:
        continue


    print(
        "Heading-GNSS error:",
        stats(
            subset[
                "heading_vs_gnss_error_deg"
            ]
        )
    )


    print(
        "GNSS direction change:",
        stats(
            subset[
                "abs_gnss_direction_change_deg"
            ]
        )
    )


    print(
        "GNSS bearing spread:",
        stats(
            subset[
                "gnss_bearing_spread_deg"
            ]
        )
    )


# =========================================================
# 5. LOW DISPLACEMENT + LARGE GNSS CHANGE
# =========================================================

print()
print("=" * 70)
print("5. LOW-DISPLACEMENT WINDOWS WITH LARGE GNSS CHANGES")
print("=" * 70)


suspicious_low_motion = df[
    (
        df[
            "gnss_displacement_m"
        ]
        < 1
    )
    &
    (
        df[
            "abs_gnss_direction_change_deg"
        ]
        >= 45
    )
]


print(
    f"Windows with displacement <1 m "
    f"and GNSS direction change >=45°: "
    f"{len(suspicious_low_motion):,}"
)


if len(suspicious_low_motion) > 0:

    print()

    print(
        suspicious_low_motion[
            [
                "dataset",
                "window_index",
                "mean_speed_kmh",
                "gnss_displacement_m",
                "gnss_direction_change_deg",
                "gnss_bearing_spread_deg",
                "heading_change_deg",
                "yawrate_integrated_change_deg",
                "heading_vs_gnss_error_deg",
                "yawrate_vs_gnss_inverted_error_deg"
            ]
        ]
        .sort_values(
            "gnss_displacement_m"
        )
        .head(30)
        .to_string(
            index=False
        )
    )


# =========================================================
# 6. LARGE GNSS CHANGES AT HIGHER SPEED
# =========================================================

print()
print("=" * 70)
print("6. LARGE GNSS CHANGES DURING MOTION")
print("=" * 70)


moving_large = df[
    (
        df["mean_speed_kmh"]
        >= 5
    )
    &
    (
        df[
            "abs_gnss_direction_change_deg"
        ]
        >= 45
    )
]


print(
    f"Windows with speed >=5 km/h "
    f"and GNSS direction change >=45°: "
    f"{len(moving_large):,}"
)


if len(moving_large) > 0:

    print()

    print(
        moving_large[
            [
                "dataset",
                "window_index",
                "mean_speed_kmh",
                "gnss_displacement_m",
                "gnss_direction_change_deg",
                "gnss_bearing_spread_deg",
                "heading_change_deg",
                "yawrate_integrated_change_deg",
                "heading_vs_gnss_error_deg",
                "yawrate_vs_gnss_inverted_error_deg",
                "mean_abs_lateral_acc"
            ]
        ]
        .sort_values(
            "gnss_direction_change_deg"
        )
        .tail(30)
        .to_string(
            index=False
        )
    )


# =========================================================
# 7. LARGE GNSS CHANGES WITH STRONG AGREEMENT
# =========================================================

print()
print("=" * 70)
print("7. LARGE GNSS CHANGES WHERE OTHER SIGNALS AGREE")
print("=" * 70)


candidate_turns = df[
    (
        df[
            "abs_gnss_direction_change_deg"
        ]
        >= 20
    )
    &
    (
        df[
            "heading_vs_gnss_error_deg"
        ]
        <= 10
    )
    &
    (
        df[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
        <= 10
    )
]


print(
    f"Windows with GNSS change >=20° "
    f"and both Heading/YawRate within 10°: "
    f"{len(candidate_turns):,}"
)


if len(candidate_turns) > 0:

    print()

    candidate_display = candidate_turns[
    [
        "dataset",
        "window_index",
        "mean_speed_kmh",
        "gnss_displacement_m",
        "gnss_direction_change_deg",
        "gnss_bearing_spread_deg",
        "heading_change_deg",
        "yawrate_integrated_change_deg",
        "heading_vs_gnss_error_deg",
        "yawrate_vs_gnss_inverted_error_deg",
        "mean_abs_lateral_acc"
    ]
].copy()

candidate_display[
    "abs_gnss_direction_change_deg"
] = np.abs(
    candidate_display[
        "gnss_direction_change_deg"
    ]
)

print(
    candidate_display
    .sort_values(
        "abs_gnss_direction_change_deg",
        ascending=False
    )
    .head(30)
    .to_string(
        index=False
    )
)

# =========================================================
# 8. GNSS / HEADING / YAWRATE THREE-WAY AGREEMENT
# =========================================================

print()
print("=" * 70)
print("8. THREE-WAY AGREEMENT")
print("=" * 70)


three_way = df[
    (
        df[
            "heading_vs_gnss_error_deg"
        ]
        <= 10
    )
    &
    (
        df[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
        <= 10
    )
]


print(
    f"Windows where Heading and inverted "
    f"YawRate both agree with GNSS within 10°: "
    f"{len(three_way):,}"
)


if len(three_way) > 0:

    print(
        f"Percentage of all windows: "
        f"{100 * len(three_way) / len(df):.3f}%"
    )


# =========================================================
# 9. THREE-WAY DISAGREEMENT
# =========================================================

print()
print("=" * 70)
print("9. THREE-WAY DISAGREEMENT")
print("=" * 70)


three_way_disagreement = df[
    (
        df[
            "heading_vs_gnss_error_deg"
        ]
        > 20
    )
    &
    (
        df[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
        > 20
    )
]


print(
    f"Windows where both Heading and "
    f"inverted YawRate disagree with GNSS by >20°: "
    f"{len(three_way_disagreement):,}"
)


if len(three_way_disagreement) > 0:

    print(
        f"Percentage of all windows: "
        f"{100 * len(three_way_disagreement) / len(df):.3f}%"
    )


# =========================================================
# 10. DATASET-LEVEL SUMMARY
# =========================================================

summary_rows = []


for dataset, group in df.groupby(
    "dataset"
):

    summary_rows.append(
        {
            "dataset":
                dataset,

            "windows":
                len(group),

            "median_speed_kmh":
                group[
                    "mean_speed_kmh"
                ].median(),

            "median_gnss_displacement_m":
                group[
                    "gnss_displacement_m"
                ].median(),

            "median_gnss_direction_change_deg":
                np.nanmedian(
                    group[
                        "abs_gnss_direction_change_deg"
                    ]
                ),

            "median_gnss_bearing_spread_deg":
                np.nanmedian(
                    group[
                        "gnss_bearing_spread_deg"
                    ]
                ),

            "median_heading_gnss_error_deg":
                np.nanmedian(
                    group[
                        "heading_vs_gnss_error_deg"
                    ]
                ),

            "median_yawrate_gnss_inverted_error_deg":
                np.nanmedian(
                    group[
                        "yawrate_vs_gnss_inverted_error_deg"
                    ]
                ),

            "windows_displacement_lt_1m":
                np.sum(
                    group[
                        "gnss_displacement_m"
                    ]
                    < 1
                ),

            "windows_gnss_change_ge_45":
                np.sum(
                    group[
                        "abs_gnss_direction_change_deg"
                    ]
                    >= 45
                ),

            "windows_heading_gnss_within_10":
                np.sum(
                    group[
                        "heading_vs_gnss_error_deg"
                    ]
                    <= 10
                ),

            "windows_yawrate_gnss_within_10":
                np.sum(
                    group[
                        "yawrate_vs_gnss_inverted_error_deg"
                    ]
                    <= 10
                )
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================================
# FINAL
# =========================================================

print()
print("=" * 70)
print("STEP 26 COMPLETE")
print("=" * 70)

print()
print(
    f"Detailed file:"
)

print(
    OUTPUT_FILE
)

print()
print(
    f"Dataset summary:"
)

print(
    SUMMARY_FILE
)

print()
print(
    "No data was deleted."
)

print(
    "No ground-truth signal was selected."
)

print(
    "All classifications are descriptive reliability "
    "categories for the next reference-formulation step."
)

print("=" * 70)
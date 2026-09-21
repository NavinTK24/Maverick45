import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 15
# Compare:
#
#       Vehicle V Yaw Rate
#              VS
#       Phone Gyroscope X/Y/Z
#
# Purpose:
# Determine whether any phone gyro axis tracks the vehicle
# yaw-rate signal.
#
# IMPORTANT:
# - S and V are assumed synchronized.
# - Sampling frequency = 10 Hz.
# - No AVNet input is changed.
# - No DDATT target is changed.
# - We do NOT assume X/Y/Z is the vehicle yaw axis.
# ============================================================


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "step15_vehicle_yawrate_vs_phone_gyro.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FS = 10.0


# ============================================================
# VERIFIED COLUMN POSITIONS
#
# S dataset:
#
# Column 16 = Gyroscope Yaw
# Column 17 = Gyroscope Pitch
# Column 18 = Gyroscope Roll
#
# V dataset:
#
# Column 15 = Yaw Rate
#
# Python indexing is zero-based.
# ============================================================

S_GYRO_X_COL = 15
S_GYRO_Y_COL = 16
S_GYRO_Z_COL = 17

V_YAW_RATE_COL = 14


# ============================================================
# FUNCTIONS
# ============================================================

def read_csv(path):

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        return pd.read_csv(
            path,
            encoding="cp1252"
        )


# ------------------------------------------------------------

def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


# ------------------------------------------------------------

def find_pairs():

    s_files = {}
    v_files = {}

    for root, dirs, files in os.walk(
        DATASET_ROOT
    ):

        for filename in files:

            if not filename.lower().endswith(".csv"):
                continue

            stem = os.path.splitext(
                filename
            )[0].lower()

            path = os.path.join(
                root,
                filename
            )

            if stem.startswith("s-"):

                key = stem[2:]

                s_files[key] = path

            elif stem.startswith("v-"):

                key = stem[2:]

                v_files[key] = path

    keys = sorted(
        set(s_files)
        &
        set(v_files)
    )

    return [
        (
            key,
            s_files[key],
            v_files[key]
        )
        for key in keys
    ]


# ============================================================
# PEARSON CORRELATION
# ============================================================

def correlation(x, y):

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    return np.corrcoef(
        x,
        y
    )[0, 1]


# ============================================================
# LINEAR FIT
#
# Vehicle yaw rate ≈ slope * phone gyro + intercept
#
# This is ONLY a diagnostic relationship.
# ============================================================

def linear_fit(x, y):

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    if np.std(x) == 0:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    prediction = (
        slope * x
        +
        intercept
    )

    ss_res = np.sum(
        (y - prediction) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = (
            1
            -
            ss_res / ss_tot
        )

    return slope, intercept, r2


# ============================================================
# ERROR CALCULATIONS
# ============================================================

def calculate_errors(
    phone,
    vehicle
):

    mask = (
        np.isfinite(phone)
        &
        np.isfinite(vehicle)
    )

    phone = phone[mask]
    vehicle = vehicle[mask]

    if len(phone) == 0:

        return {
            "count": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs": np.nan
        }

    slope, intercept, r2 = linear_fit(
        phone,
        vehicle
    )

    predicted = (
        slope * phone
        +
        intercept
    )

    error = (
        predicted
        -
        vehicle
    )

    return {
        "count": len(phone),

        "mae": np.mean(
            np.abs(error)
        ),

        "rmse": np.sqrt(
            np.mean(
                error ** 2
            )
        ),

        "median_abs": np.median(
            np.abs(error)
        )
    }


# ============================================================
# ANALYZE ONE PAIR
# ============================================================

def analyze_pair(
    key,
    s_path,
    v_path
):

    s = read_csv(s_path)
    v = read_csv(v_path)

    n = min(
        len(s),
        len(v)
    )

    if n < 2:
        return [], None

    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    gyro_x = numeric(
        s.iloc[:, S_GYRO_X_COL]
    )[:n]

    gyro_y = numeric(
        s.iloc[:, S_GYRO_Y_COL]
    )[:n]

    gyro_z = numeric(
        s.iloc[:, S_GYRO_Z_COL]
    )[:n]

    vehicle_yaw_rate = numeric(
        v.iloc[:, V_YAW_RATE_COL]
    )[:n]


    # --------------------------------------------------------
    # Sample-by-sample results
    # --------------------------------------------------------

    rows = []

    for i in range(n):

        rows.append({

            "dataset":
                key,

            "sample_index":
                i,

            "phone_gyro_x":
                gyro_x[i],

            "phone_gyro_y":
                gyro_y[i],

            "phone_gyro_z":
                gyro_z[i],

            "vehicle_yaw_rate":
                vehicle_yaw_rate[i],
        })


    # --------------------------------------------------------
    # Pair-level statistics
    # --------------------------------------------------------

    axes = {
        "X": gyro_x,
        "Y": gyro_y,
        "Z": gyro_z
    }

    stats = []

    for axis, gyro in axes.items():

        mask = (
            np.isfinite(gyro)
            &
            np.isfinite(vehicle_yaw_rate)
        )

        x = gyro[mask]
        y = vehicle_yaw_rate[mask]

        if len(x) < 2:

            continue

        corr = correlation(
            x,
            y
        )

        slope, intercept, r2 = linear_fit(
            x,
            y
        )

        predicted = (
            slope * x
            +
            intercept
        )

        error = (
            predicted
            -
            y
        )

        stats.append({

            "dataset":
                key,

            "axis":
                axis,

            "valid_samples":
                len(x),

            "correlation":
                corr,

            "absolute_correlation":
                abs(corr),

            "slope":
                slope,

            "intercept":
                intercept,

            "R2":
                r2,

            "MAE":
                np.mean(
                    np.abs(error)
                ),

            "RMSE":
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                ),

            "median_absolute_error":
                np.median(
                    np.abs(error)
                ),

            "phone_gyro_mean":
                np.mean(x),

            "phone_gyro_std":
                np.std(x),

            "vehicle_yaw_rate_mean":
                np.mean(y),

            "vehicle_yaw_rate_std":
                np.std(y),
        })

    return rows, stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "STEP 15 - VEHICLE YAW RATE VS PHONE GYROSCOPE"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.isdir(
        DATASET_ROOT
    ):

        print(
            "\nERROR: Dataset folder not found:"
        )

        print(
            DATASET_ROOT
        )

        return


    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = find_pairs()

    print(
        f"\nMatched S/V pairs: "
        f"{len(pairs)}"
    )


    if not pairs:

        print(
            "\nNo S/V pairs found."
        )

        return


    all_rows = []
    all_stats = []


    # ========================================================
    # PROCESS ALL DATASETS
    # ========================================================

    for i, (
        key,
        s_path,
        v_path
    ) in enumerate(
        pairs,
        start=1
    ):

        rows, stats = analyze_pair(
            key,
            s_path,
            v_path
        )

        all_rows.extend(
            rows
        )

        all_stats.extend(
            stats
        )

        print(
            f"[{i:02d}/{len(pairs):02d}] "
            f"{key}: "
            f"{len(rows):,} samples"
        )


    # ========================================================
    # SAVE SAMPLE DATA
    # ========================================================

    sample_df = pd.DataFrame(
        all_rows
    )

    sample_output = os.path.join(
        SCRIPT_DIR,
        "step15_sample_level.csv"
    )

    sample_df.to_csv(
        sample_output,
        index=False,
        float_format="%.8f"
    )


    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    stats_df = pd.DataFrame(
        all_stats
    )

    stats_df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8f"
    )


    # ========================================================
    # OVERALL AXIS ANALYSIS
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "OVERALL AXIS COMPARISON"
    )

    print("=" * 72)


    for axis in [
        "X",
        "Y",
        "Z"
    ]:

        axis_df = stats_df[
            stats_df["axis"] == axis
        ]

        if len(axis_df) == 0:
            continue

        # Weight the correlations by number of samples
        # only for descriptive reporting.

        weights = (
            axis_df["valid_samples"]
            /
            axis_df["valid_samples"].sum()
        )

        weighted_corr = np.sum(
            axis_df["correlation"]
            *
            weights
        )

        weighted_abs_corr = np.sum(
            axis_df["absolute_correlation"]
            *
            weights
        )

        print(
            f"\nPHONE GYRO {axis}"
        )

        print(
            f"  Mean correlation      : "
            f"{axis_df['correlation'].mean():.6f}"
        )

        print(
            f"  Mean |correlation|    : "
            f"{axis_df['absolute_correlation'].mean():.6f}"
        )

        print(
            f"  Weighted correlation  : "
            f"{weighted_corr:.6f}"
        )

        print(
            f"  Weighted |correlation|: "
            f"{weighted_abs_corr:.6f}"
        )

        print(
            f"  Median R2             : "
            f"{axis_df['R2'].median():.6f}"
        )

        print(
            f"  Median MAE            : "
            f"{axis_df['MAE'].median():.6f}"
        )

        print(
            f"  Median RMSE           : "
            f"{axis_df['RMSE'].median():.6f}"
        )


    # ========================================================
    # HOW MANY DATASETS HAVE POSITIVE / NEGATIVE RELATION
    # ========================================================

    print("\n" + "-" * 72)

    print(
        "CORRELATION DIRECTION BY DATASET"
    )

    print("-" * 72)


    for axis in [
        "X",
        "Y",
        "Z"
    ]:

        a = stats_df[
            stats_df["axis"] == axis
        ]

        positive = (
            a["correlation"] > 0.5
        ).sum()

        negative = (
            a["correlation"] < -0.5
        ).sum()

        weak = (
            a["correlation"].abs() <= 0.5
        ).sum()

        print(
            f"\nGyro {axis}:"
        )

        print(
            f"  correlation > +0.5 : "
            f"{positive:2d}"
        )

        print(
            f"  correlation < -0.5 : "
            f"{negative:2d}"
        )

        print(
            f"  |correlation| <= .5: "
            f"{weak:2d}"
        )


    # ========================================================
    # BEST CORRELATED AXIS PER DATASET
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "BEST PHONE GYRO AXIS PER DATASET"
    )

    print("=" * 72)


    best_rows = []

    for dataset in sorted(
        stats_df["dataset"].unique()
    ):

        d = stats_df[
            stats_df["dataset"] == dataset
        ]

        if len(d) == 0:
            continue

        best = d.loc[
            d["absolute_correlation"].idxmax()
        ]

        best_rows.append(best)

        print(
            f"{dataset:8s} -> "
            f"Gyro {best['axis']}  "
            f"corr={best['correlation']:.4f}  "
            f"|corr|={best['absolute_correlation']:.4f}  "
            f"R2={best['R2']:.4f}"
        )


    # ========================================================
    # DISTRIBUTION OF BEST AXIS
    # ========================================================

    best_df = pd.DataFrame(
        best_rows
    )

    print("\n" + "-" * 72)

    print(
        "BEST-AXIS COUNTS"
    )

    print("-" * 72)

    print(
        best_df["axis"]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ========================================================
    # VERY STRONG RELATIONSHIPS
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "STRONGEST DATASET-LEVEL CORRELATIONS"
    )

    print("=" * 72)


    strongest = (
        stats_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .head(30)
    )


    print(
        strongest[
            [
                "dataset",
                "axis",
                "correlation",
                "R2",
                "slope",
                "MAE",
                "RMSE"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5f}"
        )
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "STEP 15 COMPLETE"
    )

    print("=" * 72)

    print(
        f"\nDataset-level results:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"\nSample-level results:"
    )

    print(
        sample_output
    )

    print(
        "\nNo original dataset files were modified."
    )

    print(
        "No AVNet inputs or DDATT targets were modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
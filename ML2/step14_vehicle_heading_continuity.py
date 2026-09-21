import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 14
# Analyze raw Vehicle Heading continuity against Vehicle
# Yaw Rate.
#
# Purpose:
# Determine whether large changes in V Heading correspond
# to actual vehicle rotation or are representation/data
# discontinuities.
#
# Dataset:
# D:\Maverick\IO-VNBD\Synchronised V abd S datasets\
#     Categorised IOVNB Dataset
#
# Assumptions:
# - S/V are synchronized
# - Sampling frequency = 10 Hz
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "step14_vehicle_heading_continuity.csv"
)

FS = 10.0
DT = 1.0 / FS

# Verified V columns:
# Column 6  = Heading
# Column 15 = Yaw Rate

V_HEADING_COL = 5       # zero-based
V_YAW_RATE_COL = 14     # zero-based


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(dtype=float)


def circular_difference_deg(a, b):
    """
    Smallest signed angular difference.
    Output range [-180, 180).
    """
    return (a - b + 180.0) % 360.0 - 180.0


def find_vehicle_files():

    files = []

    for root, dirs, filenames in os.walk(DATASET_ROOT):

        for filename in filenames:

            if not filename.lower().endswith(".csv"):
                continue

            if filename.lower().startswith("v-"):

                files.append(
                    (
                        os.path.splitext(filename)[0][2:].lower(),
                        os.path.join(root, filename)
                    )
                )

    return sorted(files)


# ============================================================
# ANALYZE ONE VEHICLE FILE
# ============================================================

def analyze_file(key, path):

    df = read_csv(path)

    if len(df) < 2:
        return []

    heading = numeric(
        df.iloc[:, V_HEADING_COL]
    )

    yaw_rate = numeric(
        df.iloc[:, V_YAW_RATE_COL]
    )

    n = min(
        len(heading),
        len(yaw_rate)
    )

    heading = heading[:n]
    yaw_rate = yaw_rate[:n]

    results = []

    # ========================================================
    # CONSECUTIVE 10 Hz SAMPLES
    # ========================================================

    for i in range(n - 1):

        h1 = heading[i]
        h2 = heading[i + 1]

        yr1 = yaw_rate[i]
        yr2 = yaw_rate[i + 1]

        if not (
            np.isfinite(h1)
            and np.isfinite(h2)
        ):
            continue

        # ----------------------------------------------------
        # Raw heading change
        # ----------------------------------------------------

        raw_change = h2 - h1

        # ----------------------------------------------------
        # Circular heading change
        # ----------------------------------------------------

        circular_change = circular_difference_deg(
            h2,
            h1
        )

        # ----------------------------------------------------
        # Average yaw rate during interval
        # ----------------------------------------------------

        if (
            np.isfinite(yr1)
            and np.isfinite(yr2)
        ):

            avg_yaw_rate = (
                yr1 + yr2
            ) / 2.0

            # Expected rotation during 0.1 sec
            expected_change = (
                avg_yaw_rate
                *
                DT
            )

        else:

            avg_yaw_rate = np.nan
            expected_change = np.nan

        # ----------------------------------------------------
        # Difference between heading change and yaw-rate
        # ----------------------------------------------------

        if np.isfinite(
            expected_change
        ):

            consistency_error = (
                circular_difference_deg(
                    circular_change,
                    expected_change
                )
            )

        else:

            consistency_error = np.nan

        results.append({

            "dataset": key,

            "sample_index": i,

            "heading_start_deg": h1,

            "heading_end_deg": h2,

            "raw_heading_change_deg":
                raw_change,

            "circular_heading_change_deg":
                circular_change,

            "yaw_rate_start":
                yr1,

            "yaw_rate_end":
                yr2,

            "yaw_rate_mean":
                avg_yaw_rate,

            "yaw_rate_expected_change_deg":
                expected_change,

            "heading_vs_yawrate_error_deg":
                consistency_error,

            "raw_jump_abs_deg":
                abs(raw_change),

            "circular_change_abs_deg":
                abs(circular_change),
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STEP 14 - VEHICLE HEADING CONTINUITY ANALYSIS")
    print("=" * 72)

    if not os.path.isdir(DATASET_ROOT):

        print("\nERROR: Dataset folder not found:")
        print(DATASET_ROOT)

        return

    files = find_vehicle_files()

    print(
        f"\nVehicle files found: "
        f"{len(files)}"
    )

    if not files:

        print("\nNo V files found.")
        return

    all_results = []

    for i, (key, path) in enumerate(
        files,
        start=1
    ):

        rows = analyze_file(
            key,
            path
        )

        all_results.extend(rows)

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{key}: "
            f"{len(rows):,} intervals"
        )

    df = pd.DataFrame(
        all_results
    )

    if len(df) == 0:

        print("\nNo valid data.")
        return

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8f"
    )

    # ========================================================
    # OVERALL
    # ========================================================

    print("\n" + "=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)

    print(
        f"\nTotal 10 Hz intervals: "
        f"{len(df):,}"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    # ========================================================
    # RAW HEADING JUMPS
    # ========================================================

    raw = df[
        "raw_jump_abs_deg"
    ]

    print("\n" + "-" * 72)
    print("RAW HEADING JUMPS")
    print("-" * 72)

    for threshold in [
        10,
        20,
        45,
        90,
        120,
        150,
        170,
        175,
        179
    ]:

        count = (
            raw >= threshold
        ).sum()

        pct = (
            count
            /
            len(df)
            *
            100
        )

        print(
            f"|raw heading jump| >= "
            f"{threshold:>3}° : "
            f"{count:>7,} "
            f"({pct:7.4f}%)"
        )

    # ========================================================
    # CIRCULAR HEADING CHANGES
    # ========================================================

    circ = df[
        "circular_change_abs_deg"
    ]

    print("\n" + "-" * 72)
    print("CIRCULAR HEADING CHANGES")
    print("-" * 72)

    for threshold in [
        5,
        10,
        20,
        45,
        90,
        120,
        150
    ]:

        count = (
            circ >= threshold
        ).sum()

        pct = (
            count
            /
            len(df)
            *
            100
        )

        print(
            f"|circular heading change| >= "
            f"{threshold:>3}° : "
            f"{count:>7,} "
            f"({pct:7.4f}%)"
        )

    # ========================================================
    # YAW RATE CONSISTENCY
    # ========================================================

    error = (
        df[
            "heading_vs_yawrate_error_deg"
        ]
        .abs()
    )

    print("\n" + "-" * 72)
    print("HEADING vs YAW-RATE CONSISTENCY")
    print("-" * 72)

    for threshold in [
        1,
        2,
        5,
        10,
        20,
        45
    ]:

        count = (
            error <= threshold
        ).sum()

        pct = (
            count
            /
            len(df)
            *
            100
        )

        print(
            f"Error <= {threshold:>2}° : "
            f"{count:>7,} "
            f"({pct:7.2f}%)"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    print("\n" + "-" * 72)
    print("HEADING CHANGE STATISTICS")
    print("-" * 72)

    print(
        f"Raw |change| median : "
        f"{np.percentile(raw, 50):.6f}°"
    )

    print(
        f"Raw |change| P90    : "
        f"{np.percentile(raw, 90):.6f}°"
    )

    print(
        f"Raw |change| P95    : "
        f"{np.percentile(raw, 95):.6f}°"
    )

    print(
        f"Raw |change| P99    : "
        f"{np.percentile(raw, 99):.6f}°"
    )

    print(
        f"Circular |change| median : "
        f"{np.percentile(circ, 50):.6f}°"
    )

    print(
        f"Circular |change| P90    : "
        f"{np.percentile(circ, 90):.6f}°"
    )

    print(
        f"Circular |change| P95    : "
        f"{np.percentile(circ, 95):.6f}°"
    )

    print(
        f"Circular |change| P99    : "
        f"{np.percentile(circ, 99):.6f}°"
    )

    # ========================================================
    # LARGEST RAW JUMPS
    # ========================================================

    print("\n" + "=" * 72)
    print("20 LARGEST RAW HEADING JUMPS")
    print("=" * 72)

    largest = (
        df.sort_values(
            "raw_jump_abs_deg",
            ascending=False
        )
        .head(20)
    )

    columns = [

        "dataset",

        "sample_index",

        "heading_start_deg",

        "heading_end_deg",

        "raw_heading_change_deg",

        "circular_heading_change_deg",

        "yaw_rate_mean",

        "yaw_rate_expected_change_deg",

        "heading_vs_yawrate_error_deg",
    ]

    print(
        largest[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # IMPORTANT CASES:
    # LARGE HEADING JUMP BUT SMALL YAW RATE
    # ========================================================

    suspicious = df[
        (df["raw_jump_abs_deg"] >= 90)
        &
        (df["yaw_rate_expected_change_deg"].abs() <= 10)
    ].copy()

    print("\n" + "=" * 72)
    print(
        "LARGE HEADING JUMP (>90°) BUT "
        "SMALL YAW-RATE ROTATION (<=10°)"
    )
    print("=" * 72)

    print(
        f"\nCount: "
        f"{len(suspicious):,}"
    )

    if len(suspicious) > 0:

        suspicious = suspicious.sort_values(
            "raw_jump_abs_deg",
            ascending=False
        ).head(30)

        print(
            suspicious[
                columns
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 72)
    print("STEP 14 COMPLETE")
    print("=" * 72)

    print(
        "\nOriginal V files were not modified."
    )


if __name__ == "__main__":
    main()
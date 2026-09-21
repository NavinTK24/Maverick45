import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 13
# Compare Vehicle Heading / Yaw Rate with:
#   1. Phone Orientation Yaw
#   2. Phone Gyroscope X/Y/Z
#
# Dataset:
# Synchronised V abd S datasets
#   -> Categorised IOVNB Dataset
#
# Assumptions:
# - S and V are synchronized
# - Sampling frequency = 10 Hz
# - Window = 1 second = 10 samples
#
# IMPORTANT:
# This is ANALYSIS ONLY.
# We are NOT changing AVNet inputs or deleting samples.
# We are NOT assuming which phone gyro axis represents
# vehicle yaw.
# ============================================================


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "step13_vehicle_heading_yaw_comparison.csv"
)


# ------------------------------------------------------------
# DATASET SETTINGS
# ------------------------------------------------------------

WINDOW_SIZE = 10
FS = 10.0


# ------------------------------------------------------------
# VERIFIED CSV COLUMN POSITIONS
#
# Python uses zero-based indexing.
#
# Smartphone S:
#
# CSV column 16 -> Gyroscope Yaw
# CSV column 17 -> Gyroscope Pitch
# CSV column 18 -> Gyroscope Roll
# CSV column 22 -> Orientation Yaw
#
# Vehicle V:
#
# CSV column 6  -> Heading
# CSV column 15 -> Yaw Rate
# ------------------------------------------------------------

S_GYRO_YAW_COL = 15
S_GYRO_PITCH_COL = 16
S_GYRO_ROLL_COL = 17

S_ORIENTATION_YAW_COL = 21

V_HEADING_COL = 5
V_YAW_RATE_COL = 14


# ============================================================
# FUNCTIONS
# ============================================================

def read_csv(path):
    """
    Read CSV while handling possible encoding problems
    in the dataset.
    """

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

def to_numeric(series):
    """
    Convert a pandas column into numeric values.
    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(dtype=float)


# ------------------------------------------------------------

def circular_difference_deg(a, b):
    """
    Calculate the smallest signed angular difference.

    Example:

    359 -> 1

    Normal subtraction:
        1 - 359 = -358°

    Circular difference:
        +2°

    Output range:
        [-180°, +180°)
    """

    return (
        (a - b + 180.0)
        % 360.0
        - 180.0
    )


# ------------------------------------------------------------

def circular_abs_difference_deg(a, b):
    """
    Absolute circular angular difference.
    """

    return abs(
        circular_difference_deg(a, b)
    )


# ------------------------------------------------------------

def find_matching_pairs():
    """
    Find matching S-xxxxx.csv and V-xxxxx.csv files.
    """

    s_files = {}
    v_files = {}

    for root, dirs, files in os.walk(DATASET_ROOT):

        for filename in files:

            if not filename.lower().endswith(".csv"):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            stem = os.path.splitext(
                filename
            )[0].lower()

            if stem.startswith("s-"):

                key = stem[2:]

                s_files[key] = full_path

            elif stem.startswith("v-"):

                key = stem[2:]

                v_files[key] = full_path

    common_keys = sorted(
        set(s_files.keys())
        &
        set(v_files.keys())
    )

    pairs = []

    for key in common_keys:

        pairs.append(
            (
                key,
                s_files[key],
                v_files[key]
            )
        )

    return pairs


# ============================================================
# ANALYZE ONE S/V PAIR
# ============================================================

def analyze_pair(
    key,
    s_path,
    v_path
):

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    s = read_csv(s_path)
    v = read_csv(v_path)

    # Because the files are assumed synchronized,
    # use the common length.

    n = min(
        len(s),
        len(v)
    )

    if n < WINDOW_SIZE:

        return []


    # --------------------------------------------------------
    # Extract required smartphone data
    # --------------------------------------------------------

    phone_gyro_yaw = to_numeric(
        s.iloc[:, S_GYRO_YAW_COL]
    )[:n]

    phone_gyro_pitch = to_numeric(
        s.iloc[:, S_GYRO_PITCH_COL]
    )[:n]

    phone_gyro_roll = to_numeric(
        s.iloc[:, S_GYRO_ROLL_COL]
    )[:n]

    phone_yaw = to_numeric(
        s.iloc[:, S_ORIENTATION_YAW_COL]
    )[:n]


    # --------------------------------------------------------
    # Extract vehicle data
    # --------------------------------------------------------

    vehicle_heading = to_numeric(
        v.iloc[:, V_HEADING_COL]
    )[:n]

    vehicle_yaw_rate = to_numeric(
        v.iloc[:, V_YAW_RATE_COL]
    )[:n]


    # --------------------------------------------------------
    # Number of complete 1-second windows
    # --------------------------------------------------------

    number_of_windows = (
        n // WINDOW_SIZE
    )

    results = []


    # ========================================================
    # WINDOW LOOP
    # ========================================================

    for window_index in range(
        number_of_windows
    ):

        start = (
            window_index
            *
            WINDOW_SIZE
        )

        end = (
            start
            +
            WINDOW_SIZE
            -
            1
        )


        # ----------------------------------------------------
        # Extract one second
        # ----------------------------------------------------

        py = phone_yaw[
            start:end + 1
        ]

        vh = vehicle_heading[
            start:end + 1
        ]

        gy = phone_gyro_yaw[
            start:end + 1
        ]

        gp = phone_gyro_pitch[
            start:end + 1
        ]

        gr = phone_gyro_roll[
            start:end + 1
        ]

        vyr = vehicle_yaw_rate[
            start:end + 1
        ]


        # ----------------------------------------------------
        # Check endpoint values
        # ----------------------------------------------------

        if not (
            np.isfinite(py[0])
            and
            np.isfinite(py[-1])
            and
            np.isfinite(vh[0])
            and
            np.isfinite(vh[-1])
        ):

            continue


        # ====================================================
        # 1. PHONE YAW CHANGE
        # ====================================================

        phone_yaw_change = (
            circular_difference_deg(
                py[-1],
                py[0]
            )
        )


        # ====================================================
        # 2. VEHICLE HEADING CHANGE
        # ====================================================

        vehicle_heading_change = (
            circular_difference_deg(
                vh[-1],
                vh[0]
            )
        )


        # ====================================================
        # 3. PHONE VS VEHICLE HEADING CHANGE
        # ====================================================

        heading_change_error = (
            circular_difference_deg(
                phone_yaw_change,
                vehicle_heading_change
            )
        )


        # ====================================================
        # 4. INTEGRATE PHONE GYROSCOPE
        #
        # Gyroscope is angular velocity.
        #
        # Approximate:
        #
        # angle = integral(angular velocity dt)
        #
        # Since:
        #
        # Fs = 10 Hz
        #
        # dt = 0.1 sec
        # ====================================================

        gyro_yaw_integrated = (
            np.nansum(gy)
            /
            FS
        )

        gyro_pitch_integrated = (
            np.nansum(gp)
            /
            FS
        )

        gyro_roll_integrated = (
            np.nansum(gr)
            /
            FS
        )


        # ====================================================
        # 5. PHONE GYRO VECTOR RMS
        # ====================================================

        gyro_vector_rms = np.sqrt(
            np.nanmean(
                gy**2
                +
                gp**2
                +
                gr**2
            )
        )


        # ====================================================
        # 6. VEHICLE YAW RATE
        # ====================================================

        vehicle_yaw_rate_mean = (
            np.nanmean(vyr)
        )

        vehicle_yaw_rate_rms = np.sqrt(
            np.nanmean(
                vyr**2
            )
        )


        # ====================================================
        # 7. INTEGRATE VEHICLE YAW RATE
        #
        # 10 Hz -> dt = 0.1 sec
        #
        # angle =
        # sum(yaw_rate * dt)
        #
        # Equivalent to:
        #
        # sum(yaw_rate) / 10
        # ====================================================

        valid_yaw_rate = vyr[
            np.isfinite(vyr)
        ]

        if len(valid_yaw_rate) > 0:

            vehicle_yaw_integrated = (
                np.sum(
                    valid_yaw_rate
                )
                /
                FS
            )

        else:

            vehicle_yaw_integrated = np.nan


        # ====================================================
        # 8. VEHICLE HEADING vs VEHICLE YAW RATE
        #
        # This is an internal consistency check
        # using ONLY V data.
        # ====================================================

        if np.isfinite(
            vehicle_yaw_integrated
        ):

            vehicle_heading_yawrate_error = (
                circular_abs_difference_deg(
                    vehicle_heading_change,
                    vehicle_yaw_integrated
                )
            )

        else:

            vehicle_heading_yawrate_error = np.nan


        # ====================================================
        # SAVE WINDOW
        # ====================================================

        results.append({

            "dataset":
                key,

            "window_index":
                window_index,

            "start_row":
                start,

            "end_row":
                end,


            # ------------------------------
            # PHONE ORIENTATION YAW
            # ------------------------------

            "phone_yaw_start_deg":
                py[0],

            "phone_yaw_end_deg":
                py[-1],

            "phone_yaw_change_deg":
                phone_yaw_change,


            # ------------------------------
            # VEHICLE HEADING
            # ------------------------------

            "vehicle_heading_start_deg":
                vh[0],

            "vehicle_heading_end_deg":
                vh[-1],

            "vehicle_heading_change_deg":
                vehicle_heading_change,


            # ------------------------------
            # PHONE VS VEHICLE
            # ------------------------------

            "phone_vs_vehicle_heading_change_error_deg":
                heading_change_error,


            # ------------------------------
            # PHONE GYRO INTEGRATION
            # ------------------------------

            "phone_gyro_yaw_integrated_deg":
                gyro_yaw_integrated,

            "phone_gyro_pitch_integrated_deg":
                gyro_pitch_integrated,

            "phone_gyro_roll_integrated_deg":
                gyro_roll_integrated,


            # ------------------------------
            # PHONE GYRO RMS
            # ------------------------------

            "phone_gyro_vector_rms_rad_s":
                gyro_vector_rms,


            # ------------------------------
            # VEHICLE YAW RATE
            # ------------------------------

            "vehicle_yaw_rate_mean":
                vehicle_yaw_rate_mean,

            "vehicle_yaw_rate_rms":
                vehicle_yaw_rate_rms,

            "vehicle_yaw_rate_integrated_deg":
                vehicle_yaw_integrated,


            # ------------------------------
            # V INTERNAL CONSISTENCY
            # ------------------------------

            "vehicle_heading_vs_yawrate_error_deg":
                vehicle_heading_yawrate_error,


            # ------------------------------
            # ABSOLUTE CHANGES
            # ------------------------------

            "phone_yaw_change_abs_deg":
                abs(
                    phone_yaw_change
                ),

            "vehicle_heading_change_abs_deg":
                abs(
                    vehicle_heading_change
                ),
        })


    return results


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    df,
    column
):

    data = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna().to_numpy()

    if len(data) == 0:

        return


    print(
        f"\n{column}"
    )

    print(
        f"  Minimum : "
        f"{np.min(data):.6f}"
    )

    print(
        f"  Median  : "
        f"{np.percentile(data, 50):.6f}"
    )

    print(
        f"  P90     : "
        f"{np.percentile(data, 90):.6f}"
    )

    print(
        f"  P95     : "
        f"{np.percentile(data, 95):.6f}"
    )

    print(
        f"  P99     : "
        f"{np.percentile(data, 99):.6f}"
    )

    print(
        f"  Maximum : "
        f"{np.max(data):.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "STEP 13 - VEHICLE HEADING / YAW RATE ANALYSIS"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.isdir(
        DATASET_ROOT
    ):

        print(
            "\nERROR:"
        )

        print(
            "Dataset folder not found:"
        )

        print(
            DATASET_ROOT
        )

        print(
            "\nCheck DATASET_ROOT in the script."
        )

        return


    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = find_matching_pairs()


    print(
        f"\nDataset:"
    )

    print(
        DATASET_ROOT
    )

    print(
        f"\nMatched S/V pairs: "
        f"{len(pairs)}"
    )


    if not pairs:

        print(
            "\nNo matching S/V files found."
        )

        return


    # --------------------------------------------------------
    # Analyze all pairs
    # --------------------------------------------------------

    all_results = []


    for i, (
        key,
        s_path,
        v_path
    ) in enumerate(
        pairs,
        start=1
    ):

        rows = analyze_pair(
            key,
            s_path,
            v_path
        )

        all_results.extend(
            rows
        )


        print(
            f"[{i:02d}/{len(pairs):02d}] "
            f"{key}: "
            f"{len(rows):,} windows"
        )


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )


    if len(df) == 0:

        print(
            "\nNo valid windows."
        )

        return


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8f"
    )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=" * 72
    )


    print(
        f"\nTotal windows: "
        f"{len(df):,}"
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nPHONE YAW CHANGE"
    )

    print_statistics(
        df,
        "phone_yaw_change_abs_deg"
    )


    print(
        "\nVEHICLE HEADING CHANGE"
    )

    print_statistics(
        df,
        "vehicle_heading_change_abs_deg"
    )


    print(
        "\nPHONE VS VEHICLE HEADING CHANGE ERROR"
    )

    print_statistics(
        df,
        "phone_vs_vehicle_heading_change_error_deg"
    )


    print(
        "\nVEHICLE HEADING VS YAW-RATE ERROR"
    )

    print_statistics(
        df,
        "vehicle_heading_vs_yawrate_error_deg"
    )


    print(
        "\nPHONE GYRO VECTOR RMS"
    )

    print_statistics(
        df,
        "phone_gyro_vector_rms_rad_s"
    )


    # ========================================================
    # PHONE vs VEHICLE AGREEMENT
    # ========================================================

    print(
        "\n"
        +
        "-" * 72
    )

    print(
        "PHONE YAW vs VEHICLE HEADING AGREEMENT"
    )

    print(
        "-" * 72
    )


    error = (
        df[
            "phone_vs_vehicle_heading_change_error_deg"
        ]
        .abs()
    )


    for threshold in [
        5,
        10,
        20,
        30,
        45,
        90
    ]:

        count = (
            error <= threshold
        ).sum()

        percentage = (
            count
            /
            len(df)
            *
            100
        )


        print(
            f"Error <= {threshold:>3}° : "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )


    # ========================================================
    # VEHICLE HEADING vs YAW RATE
    # ========================================================

    print(
        "\n"
        +
        "-" * 72
    )

    print(
        "VEHICLE HEADING vs VEHICLE YAW RATE"
    )

    print(
        "-" * 72
    )


    v_error = (
        df[
            "vehicle_heading_vs_yawrate_error_deg"
        ]
        .abs()
    )


    for threshold in [
        1,
        2,
        5,
        10,
        20,
        45
    ]:

        count = (
            v_error <= threshold
        ).sum()

        percentage = (
            count
            /
            len(df)
            *
            100
        )


        print(
            f"Error <= {threshold:>2}° : "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )


    # ========================================================
    # LARGEST PHONE-VS-VEHICLE DISAGREEMENTS
    # ========================================================

    print(
        "\n"
        +
        "-" * 72
    )

    print(
        "20 LARGEST PHONE-VS-VEHICLE DISAGREEMENTS"
    )

    print(
        "-" * 72
    )


    top = (
        df.sort_values(
            "phone_vs_vehicle_heading_change_error_deg",
            ascending=False
        )
        .head(20)
    )


    display_columns = [

        "dataset",

        "window_index",

        "phone_yaw_change_deg",

        "vehicle_heading_change_deg",

        "phone_vs_vehicle_heading_change_error_deg",

        "vehicle_yaw_rate_integrated_deg",

        "phone_gyro_vector_rms_rad_s",
    ]


    print(
        top[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    # ========================================================
    # LARGEST V INTERNAL DISAGREEMENTS
    # ========================================================

    print(
        "\n"
        +
        "-" * 72
    )

    print(
        "20 LARGEST VEHICLE HEADING-V-YAW-RATE DISAGREEMENTS"
    )

    print(
        "-" * 72
    )


    top_v = (
        df.sort_values(
            "vehicle_heading_vs_yawrate_error_deg",
            ascending=False
        )
        .head(20)
    )


    display_columns_v = [

        "dataset",

        "window_index",

        "vehicle_heading_change_deg",

        "vehicle_yaw_rate_integrated_deg",

        "vehicle_heading_vs_yawrate_error_deg",

        "vehicle_yaw_rate_mean",

        "vehicle_yaw_rate_rms",
    ]


    print(
        top_v[
            display_columns_v
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "STEP 13 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nNo original dataset files were modified."
    )

    print(
        "No AVNet inputs or targets were modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
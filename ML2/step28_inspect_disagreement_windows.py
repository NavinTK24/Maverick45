import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 28
# INSPECT THREE-WAY DISAGREEMENT WINDOWS
# ============================================================

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

STEP26_FILE = (
    r"D:\Maverick\ML2\step26_gnss_reliability_analysis.csv"
)

OUTPUT_FILE = (
    r"D:\Maverick\ML2\step28_disagreement_window_inspection.csv"
)


# ============================================================
# VEHICLE COLUMN POSITIONS
# ============================================================

COL_TIME = 2
COL_VELOCITY = 5
COL_HEADING = 6
COL_YAW_RATE = 15
COL_LONG_ACC = 17
COL_LAT_ACC = 18
COL_LATITUDE = 3
COL_LONGITUDE = 4


# ============================================================
# LOAD STEP 26
# ============================================================

df = pd.read_csv(STEP26_FILE)

print("=" * 70)
print("STEP 28 - THREE-WAY DISAGREEMENT INSPECTION")
print("=" * 70)

print(
    f"\nTotal Step 26 windows: {len(df):,}"
)


# ============================================================
# THREE-WAY DISAGREEMENT
#
# Both Heading and inverted Yaw Rate disagree
# with GNSS by more than 20 degrees.
# ============================================================

disagreement = df[
    (df["heading_vs_gnss_error_deg"] > 20.0)
    &
    (df["yawrate_vs_gnss_inverted_error_deg"] > 20.0)
].copy()


disagreement["abs_gnss_change"] = np.abs(
    disagreement["gnss_direction_change_deg"]
)

disagreement["abs_heading_change"] = np.abs(
    disagreement["heading_change_deg"]
)

disagreement["abs_yawrate_change"] = np.abs(
    disagreement["yawrate_integrated_change_deg"]
)


print(
    f"Three-way disagreement windows: "
    f"{len(disagreement):,}"
)

print(
    f"Percentage: "
    f"{100 * len(disagreement) / len(df):.3f}%"
)


# ============================================================
# CREATE DIFFERENT CATEGORIES
# ============================================================

# ------------------------------------------------------------
# A. Low speed
# ------------------------------------------------------------

low_speed = disagreement[
    disagreement["mean_speed_kmh"] < 5.0
].copy()


# ------------------------------------------------------------
# B. Moving
# ------------------------------------------------------------

moving = disagreement[
    disagreement["mean_speed_kmh"] >= 5.0
].copy()


# ------------------------------------------------------------
# C. Large GNSS change
# ------------------------------------------------------------

large_gnss = disagreement[
    np.abs(
        disagreement["gnss_direction_change_deg"]
    ) >= 45.0
].copy()


# ------------------------------------------------------------
# D. Large Heading change
# ------------------------------------------------------------

large_heading = disagreement[
    np.abs(
        disagreement["heading_change_deg"]
    ) >= 20.0
].copy()


# ------------------------------------------------------------
# E. GNSS bearing unstable
# ------------------------------------------------------------

unstable_gnss = disagreement[
    disagreement["gnss_bearing_spread_deg"] >= 45.0
].copy()


# ============================================================
# PRINT CATEGORY COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("DISAGREEMENT CATEGORIES")
print("=" * 70)

print(
    f"\nLow speed <5 km/h:"
    f" {len(low_speed):,}"
)

print(
    f"Moving >=5 km/h:"
    f" {len(moving):,}"
)

print(
    f"GNSS change >=45°:"
    f" {len(large_gnss):,}"
)

print(
    f"Heading change >=20°:"
    f" {len(large_heading):,}"
)

print(
    f"GNSS bearing spread >=45°:"
    f" {len(unstable_gnss):,}"
)


# ============================================================
# SELECT REPRESENTATIVE WINDOWS
# ============================================================

selected = []


def add_rows(
    dataframe,
    count,
    label,
    sort_column,
    ascending=False
):

    if len(dataframe) == 0:
        return

    temp = dataframe.sort_values(
        sort_column,
        ascending=ascending
    ).head(count).copy()

    temp["selection_category"] = label

    selected.append(temp)


# ------------------------------------------------------------
# 1. Largest GNSS changes
# ------------------------------------------------------------

add_rows(
    disagreement,
    10,
    "largest_gnss_change",
    "abs_gnss_change"
)


# ------------------------------------------------------------
# 2. Largest disagreement
# ------------------------------------------------------------

disagreement[
    "max_three_way_error"
] = disagreement[
    [
        "heading_vs_gnss_error_deg",
        "yawrate_vs_gnss_inverted_error_deg"
    ]
].max(axis=1)


add_rows(
    disagreement,
    10,
    "largest_three_way_error",
    "max_three_way_error"
)


# ------------------------------------------------------------
# 3. Highest-speed disagreements
# ------------------------------------------------------------

add_rows(
    disagreement,
    10,
    "highest_speed",
    "mean_speed_kmh"
)


# ------------------------------------------------------------
# 4. Low-speed disagreements
# ------------------------------------------------------------

add_rows(
    low_speed,
    10,
    "low_speed",
    "mean_speed_kmh",
    ascending=True
)


# ------------------------------------------------------------
# 5. Most unstable GNSS
# ------------------------------------------------------------

add_rows(
    disagreement,
    10,
    "largest_gnss_bearing_spread",
    "gnss_bearing_spread_deg"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

if len(selected) > 0:

    selected_df = pd.concat(
        selected,
        ignore_index=True
    )

    selected_df = selected_df.drop_duplicates(
        subset=["dataset", "window_index"]
    )

else:

    selected_df = pd.DataFrame()


print(
    f"\nRepresentative unique windows selected: "
    f"{len(selected_df):,}"
)


# ============================================================
# FIND RAW V FILE
# ============================================================

all_csv_files = []

for root, dirs, files in os.walk(DATASET_ROOT):

    for filename in files:

        if filename.lower().endswith(".csv"):

            all_csv_files.append(
                os.path.join(
                    root,
                    filename
                )
            )


v_files = []

for path in all_csv_files:

    filename = os.path.basename(
        path
    ).lower()

    if filename.startswith("v-"):

        v_files.append(path)


def normalize(text):

    return (
        str(text)
        .lower()
        .replace(".csv", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def find_vehicle_file(dataset):

    key = normalize(dataset)

    matches = []

    for path in v_files:

        filename = normalize(
            os.path.basename(path)
        )

        if key in filename:

            matches.append(path)


    if len(matches) == 1:

        return matches[0]


    # Exact parent-folder match

    folder_matches = []

    for path in matches:

        folder = os.path.basename(
            os.path.dirname(path)
        )

        if normalize(folder) == key:

            folder_matches.append(path)


    if len(folder_matches) == 1:

        return folder_matches[0]


    return None


# ============================================================
# INSPECT RAW WINDOWS
# ============================================================

results = []


print("\n")
print("=" * 70)
print("RAW DISAGREEMENT WINDOWS")
print("=" * 70)


for number, (_, row) in enumerate(
    selected_df.iterrows(),
    start=1
):

    dataset = str(
        row["dataset"]
    )

    window_index = int(
        row["window_index"]
    )

    start_row = window_index * 10
    end_row = start_row + 9


    vehicle_file = find_vehicle_file(
        dataset
    )


    if vehicle_file is None:

        print(
            f"\nWARNING: vehicle file not found "
            f"for {dataset}"
        )

        continue


    v = pd.read_csv(
        vehicle_file,
        header=None,
        low_memory=False
    )


    if end_row >= len(v):

        print(
            f"\nWARNING: window outside file:"
            f" {dataset} {window_index}"
        )

        continue


    # ========================================================
    # WINDOW SUMMARY
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        f"Case {number}/{len(selected_df)}"
    )

    print(
        f"Category       : "
        f"{row['selection_category']}"
    )

    print(
        f"Dataset        : "
        f"{dataset}"
    )

    print(
        f"Window         : "
        f"{window_index}"
    )

    print(
        f"Speed          : "
        f"{row['mean_speed_kmh']:.3f} km/h"
    )

    print(
        f"GNSS displacement: "
        f"{row['gnss_displacement_m']:.3f} m"
    )

    print(
        f"GNSS Δdirection: "
        f"{row['gnss_direction_change_deg']:.3f}°"
    )

    print(
        f"GNSS spread    : "
        f"{row['gnss_bearing_spread_deg']:.3f}°"
    )

    print(
        f"Heading Δ      : "
        f"{row['heading_change_deg']:.3f}°"
    )

    print(
        f"Yaw-rate Δ     : "
        f"{row['yawrate_integrated_change_deg']:.3f}°"
    )

    print(
        f"Heading-GNSS error: "
        f"{row['heading_vs_gnss_error_deg']:.3f}°"
    )

    print(
        f"YawRate-GNSS error: "
        f"{row['yawrate_vs_gnss_inverted_error_deg']:.3f}°"
    )


    # ========================================================
    # RAW SAMPLES
    # ========================================================

    print("\nRaw 10 Hz samples:")

    print(
        "sample | velocity | heading | yaw_rate | "
        "lat_acc | long_acc | latitude | longitude"
    )


    for local_i, raw_index in enumerate(
        range(start_row, end_row + 1)
    ):

        r = v.iloc[raw_index]


        def val(position):

            return pd.to_numeric(
                r.iloc[position - 1],
                errors="coerce"
            )


        velocity = val(
            COL_VELOCITY
        )

        heading = val(
            COL_HEADING
        )

        yaw_rate = val(
            COL_YAW_RATE
        )

        lat_acc = val(
            COL_LAT_ACC
        )

        long_acc = val(
            COL_LONG_ACC
        )

        latitude = val(
            COL_LATITUDE
        )

        longitude = val(
            COL_LONGITUDE
        )


        print(
            f"{local_i:6d} | "
            f"{velocity:8.3f} | "
            f"{heading:8.3f} | "
            f"{yaw_rate:8.3f} | "
            f"{lat_acc:7.3f} | "
            f"{long_acc:8.3f} | "
            f"{latitude:10.6f} | "
            f"{longitude:11.6f}"
        )


        results.append({

            "case_number":
                number,

            "selection_category":
                row["selection_category"],

            "dataset":
                dataset,

            "window_index":
                window_index,

            "local_sample":
                local_i,

            "raw_row":
                raw_index,

            "velocity_kmh":
                velocity,

            "heading_deg":
                heading,

            "yaw_rate_deg_s":
                yaw_rate,

            "lateral_acc":
                lat_acc,

            "longitudinal_acc":
                long_acc,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "gnss_direction_change_deg":
                row["gnss_direction_change_deg"],

            "heading_change_deg":
                row["heading_change_deg"],

            "yawrate_integrated_change_deg":
                row["yawrate_integrated_change_deg"],

            "gnss_displacement_m":
                row["gnss_displacement_m"],

            "gnss_bearing_spread_deg":
                row["gnss_bearing_spread_deg"],

            "heading_vs_gnss_error_deg":
                row["heading_vs_gnss_error_deg"],

            "yawrate_vs_gnss_inverted_error_deg":
                row[
                    "yawrate_vs_gnss_inverted_error_deg"
                ]
        })


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("STEP 28 COMPLETE")
print("=" * 70)

print(
    f"\nThree-way disagreement windows:"
    f" {len(disagreement):,}"
)

print(
    f"Representative windows selected:"
    f" {len(selected_df):,}"
)

print(
    f"Raw samples inspected:"
    f" {len(result_df):,}"
)

print(
    f"\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNo data was deleted."
)

print(
    "No ground-truth signal was selected."
)
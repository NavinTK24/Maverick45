import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 27
# INSPECT RAW VEHICLE DATA FOR RELIABLE GNSS TURNS
# ============================================================

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

STEP26_FILE = (
    r"D:\Maverick\ML2\step26_gnss_reliability_analysis.csv"
)

OUTPUT_FILE = (
    r"D:\Maverick\ML2\step27_reference_window_inspection.csv"
)


# ============================================================
# VEHICLE COLUMN POSITIONS
# 1-based positions from dataset documentation
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

df26 = pd.read_csv(STEP26_FILE)

print("=" * 70)
print("STEP 27 - RAW VEHICLE TURN INSPECTION")
print("=" * 70)

print(f"\nStep 26 rows loaded: {len(df26):,}")


# ============================================================
# SELECT 29 RELIABLE TURNING WINDOWS
# ============================================================

candidate = df26[
    (np.abs(df26["gnss_direction_change_deg"]) >= 20.0)
    & (df26["heading_vs_gnss_error_deg"] <= 10.0)
    & (df26["yawrate_vs_gnss_inverted_error_deg"] <= 10.0)
].copy()

candidate["abs_gnss_change"] = np.abs(
    candidate["gnss_direction_change_deg"]
)

candidate = candidate.sort_values(
    "abs_gnss_change",
    ascending=False
)

print(
    f"\nReliable candidate turning windows: "
    f"{len(candidate):,}"
)


# ============================================================
# FIND ALL CSV FILES RECURSIVELY
# ============================================================

print("\nSearching dataset recursively for CSV files...")

all_csv_files = []

for root, dirs, files in os.walk(DATASET_ROOT):

    for filename in files:

        if filename.lower().endswith(".csv"):

            full_path = os.path.join(
                root,
                filename
            )

            all_csv_files.append(full_path)


print(
    f"CSV files found: {len(all_csv_files):,}"
)


# ============================================================
# BUILD V FILE INDEX
# ============================================================

v_files = []

for path in all_csv_files:

    filename = os.path.basename(path).lower()

    # Vehicle files generally start with v-
    # but we also retain paths containing /V/
    if filename.startswith("v-"):

        v_files.append(path)


print(
    f"Vehicle CSV candidates found: {len(v_files):,}"
)


# ============================================================
# NORMALIZE NAME
# ============================================================

def normalize_name(text):

    text = str(text).lower()

    return (
        text
        .replace(".csv", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


# ============================================================
# FIND VEHICLE FILE FOR DATASET
# ============================================================

def find_vehicle_file(dataset_name):

    key = normalize_name(dataset_name)

    matches = []

    for path in v_files:

        filename = os.path.basename(path)

        normalized_filename = normalize_name(
            filename
        )

        # First preference:
        # filename contains dataset key
        if key in normalized_filename:

            matches.append(path)


    # --------------------------------------------------------
    # Exact / strongest match
    # --------------------------------------------------------

    if len(matches) == 1:

        return matches[0]


    # --------------------------------------------------------
    # If multiple candidates exist,
    # prefer one inside matching folder
    # --------------------------------------------------------

    folder_matches = []

    for path in matches:

        parent_folder = os.path.basename(
            os.path.dirname(path)
        )

        if normalize_name(parent_folder) == key:

            folder_matches.append(path)


    if len(folder_matches) == 1:

        return folder_matches[0]


    # --------------------------------------------------------
    # If still ambiguous, return None
    # --------------------------------------------------------

    if len(matches) > 1:

        print(
            f"\nWARNING: multiple V files for "
            f"{dataset_name}:"
        )

        for m in matches:

            print(
                "   ",
                m
            )

        return None


    return None


# ============================================================
# RAW SAMPLE RESULTS
# ============================================================

results = []


# ============================================================
# INSPECT CANDIDATE WINDOWS
# ============================================================

print("\n")
print("=" * 70)
print("RELIABLE TURNING WINDOWS")
print("=" * 70)


for candidate_number, (_, row) in enumerate(
    candidate.iterrows(),
    start=1
):

    dataset = str(row["dataset"])

    window_index = int(
        row["window_index"]
    )

    start_row = window_index * 10
    end_row = start_row + 9


    # --------------------------------------------------------
    # Find V file
    # --------------------------------------------------------

    vehicle_file = find_vehicle_file(
        dataset
    )


    if vehicle_file is None:

        print(
            f"\nWARNING: could not uniquely "
            f"find vehicle file for {dataset}"
        )

        continue


    # --------------------------------------------------------
    # Read V file
    # --------------------------------------------------------

    try:

        v = pd.read_csv(
            vehicle_file,
            header=None,
            low_memory=False
        )

    except Exception as e:

        print(
            f"\nERROR reading:"
            f"\n{vehicle_file}"
            f"\n{e}"
        )

        continue


    # --------------------------------------------------------
    # Check window bounds
    # --------------------------------------------------------

    if end_row >= len(v):

        print(
            f"\nWARNING:"
            f"\n{dataset}"
            f"\nwindow={window_index}"
            f"\nrequested rows={start_row}-{end_row}"
            f"\nfile rows={len(v)}"
        )

        continue


    samples = v.iloc[
        start_row:end_row + 1
    ]


    # ========================================================
    # WINDOW HEADER
    # ========================================================

    print("\n")
    print("-" * 70)

    print(
        f"Candidate {candidate_number}/"
        f"{len(candidate)}"
    )

    print(
        f"Dataset       : {dataset}"
    )

    print(
        f"Vehicle file  : "
        f"{os.path.basename(vehicle_file)}"
    )

    print(
        f"Window        : {window_index}"
    )

    print(
        f"Rows          : "
        f"{start_row} - {end_row}"
    )

    print(
        f"GNSS Δdirection: "
        f"{row['gnss_direction_change_deg']:.3f} deg"
    )

    print(
        f"Heading Δ      : "
        f"{row['heading_change_deg']:.3f} deg"
    )

    print(
        f"Yaw-rate Δ     : "
        f"{row['yawrate_integrated_change_deg']:.3f} deg"
    )

    print(
        f"GNSS displacement: "
        f"{row['gnss_displacement_m']:.3f} m"
    )

    print(
        f"GNSS bearing spread: "
        f"{row['gnss_bearing_spread_deg']:.3f} deg"
    )

    print(
        f"Mean speed     : "
        f"{row['mean_speed_kmh']:.3f} km/h"
    )


    # ========================================================
    # RAW 10 Hz DATA
    # ========================================================

    print("\nRaw 10 Hz samples:")

    print(
        "sample | time | velocity | heading | "
        "yaw_rate | lat_acc | long_acc | "
        "latitude | longitude"
    )


    for local_i, raw_index in enumerate(
        range(start_row, end_row + 1)
    ):

        r = v.iloc[raw_index]


        def number(position):

            return pd.to_numeric(
                r.iloc[position - 1],
                errors="coerce"
            )


        time_val = number(
            COL_TIME
        )

        velocity = number(
            COL_VELOCITY
        )

        heading = number(
            COL_HEADING
        )

        yaw_rate = number(
            COL_YAW_RATE
        )

        long_acc = number(
            COL_LONG_ACC
        )

        lat_acc = number(
            COL_LAT_ACC
        )

        latitude = number(
            COL_LATITUDE
        )

        longitude = number(
            COL_LONGITUDE
        )


        print(
            f"{local_i:6d} | "
            f"{time_val:8.3f} | "
            f"{velocity:8.3f} | "
            f"{heading:8.3f} | "
            f"{yaw_rate:8.3f} | "
            f"{lat_acc:7.3f} | "
            f"{long_acc:8.3f} | "
            f"{latitude:10.6f} | "
            f"{longitude:11.6f}"
        )


        results.append({

            "candidate_number":
                candidate_number,

            "dataset":
                dataset,

            "window_index":
                window_index,

            "local_sample":
                local_i,

            "raw_row":
                raw_index,

            "time":
                time_val,

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

            "window_gnss_direction_change_deg":
                row["gnss_direction_change_deg"],

            "window_heading_change_deg":
                row["heading_change_deg"],

            "window_yawrate_change_deg":
                row["yawrate_integrated_change_deg"],

            "window_gnss_displacement_m":
                row["gnss_displacement_m"],

            "window_gnss_bearing_spread_deg":
                row["gnss_bearing_spread_deg"]
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


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 27 COMPLETE")
print("=" * 70)

print(
    f"\nCandidate windows: "
    f"{len(candidate):,}"
)

print(
    f"Raw samples inspected: "
    f"{len(result_df):,}"
)

print(
    f"Expected maximum: "
    f"{len(candidate) * 10:,}"
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
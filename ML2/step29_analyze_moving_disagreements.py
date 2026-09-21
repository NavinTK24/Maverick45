import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 29
# ANALYZE ALL MOVING THREE-WAY DISAGREEMENT WINDOWS
# ============================================================

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

STEP26_FILE = (
    r"D:\Maverick\ML2\step26_gnss_reliability_analysis.csv"
)

OUTPUT_FILE = (
    r"D:\Maverick\ML2\step29_moving_disagreement_analysis.csv"
)

SUMMARY_FILE = (
    r"D:\Maverick\ML2\step29_classification_summary.txt"
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


# ============================================================
# HELPERS
# ============================================================

def circular_difference_deg(a, b):
    """
    Smallest signed angular difference a-b.
    Result is in [-180, 180).
    """
    return (
        (a - b + 180.0) % 360.0
    ) - 180.0


def circular_abs_difference_deg(a, b):
    return abs(
        circular_difference_deg(a, b)
    )


def normalize(text):
    return (
        str(text)
        .lower()
        .replace(".csv", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


# ============================================================
# LOAD STEP 26
# ============================================================

df = pd.read_csv(
    STEP26_FILE
)

print("=" * 75)
print("STEP 29 - MOVING DISAGREEMENT ANALYSIS")
print("=" * 75)

print(
    f"\nTotal Step 26 windows: {len(df):,}"
)


# ============================================================
# SELECT THREE-WAY DISAGREEMENTS
# ============================================================

disagreement = df[
    (df["heading_vs_gnss_error_deg"] > 20.0)
    &
    (df["yawrate_vs_gnss_inverted_error_deg"] > 20.0)
].copy()


# ============================================================
# ONLY MOVING WINDOWS
# ============================================================

moving = disagreement[
    disagreement["mean_speed_kmh"] >= 5.0
].copy()


print(
    f"Three-way disagreement: "
    f"{len(disagreement):,}"
)

print(
    f"Moving disagreement >=5 km/h: "
    f"{len(moving):,}"
)


# ============================================================
# DERIVED METRICS
# ============================================================

moving["abs_gnss_change"] = np.abs(
    moving["gnss_direction_change_deg"]
)

moving["abs_heading_change"] = np.abs(
    moving["heading_change_deg"]
)

moving["abs_yawrate_change"] = np.abs(
    moving["yawrate_integrated_change_deg"]
)

moving["heading_yawrate_error"] = np.abs(
    moving["heading_vs_yawrate_inverted_error_deg"]
)


# ============================================================
# FIND VEHICLE FILES
# ============================================================

all_csv_files = []

for root, dirs, files in os.walk(
    DATASET_ROOT
):

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


def find_vehicle_file(dataset):

    key = normalize(dataset)

    exact = []

    for path in v_files:

        filename = normalize(
            os.path.basename(path)
        )

        if filename == "v" + key:

            exact.append(path)


    if len(exact) == 1:

        return exact[0]


    # Parent-folder matching

    folder_matches = []

    for path in v_files:

        folder = os.path.basename(
            os.path.dirname(path)
        )

        if normalize(folder) == key:

            folder_matches.append(path)


    if len(folder_matches) == 1:

        return folder_matches[0]


    # Partial filename fallback

    partial = []

    for path in v_files:

        filename = normalize(
            os.path.basename(path)
        )

        if key in filename:

            partial.append(path)


    if len(partial) == 1:

        return partial[0]


    return None


# ============================================================
# CACHE VEHICLE FILES
# ============================================================

vehicle_cache = {}

for dataset in moving["dataset"].unique():

    path = find_vehicle_file(
        dataset
    )

    if path is not None:

        vehicle_cache[dataset] = pd.read_csv(
            path,
            header=None,
            low_memory=False
        )

    else:

        print(
            f"WARNING: vehicle file not found "
            f"for {dataset}"
        )


# ============================================================
# INSPECT EACH MOVING WINDOW
# ============================================================

results = []


for _, row in moving.iterrows():

    dataset = str(
        row["dataset"]
    )

    window_index = int(
        row["window_index"]
    )

    start_row = (
        window_index * 10
    )

    end_row = (
        start_row + 9
    )


    if dataset not in vehicle_cache:

        continue


    v = vehicle_cache[
        dataset
    ]


    if end_row >= len(v):

        continue


    # --------------------------------------------------------
    # Extract raw window
    # --------------------------------------------------------

    raw = v.iloc[
        start_row:end_row + 1
    ]


    velocity = pd.to_numeric(
        raw.iloc[:, COL_VELOCITY - 1],
        errors="coerce"
    )

    heading = pd.to_numeric(
        raw.iloc[:, COL_HEADING - 1],
        errors="coerce"
    )

    yaw_rate = pd.to_numeric(
        raw.iloc[:, COL_YAW_RATE - 1],
        errors="coerce"
    )

    lat_acc = pd.to_numeric(
        raw.iloc[:, COL_LAT_ACC - 1],
        errors="coerce"
    )

    long_acc = pd.to_numeric(
        raw.iloc[:, COL_LONG_ACC - 1],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    mean_speed = velocity.mean()

    max_speed = velocity.max()

    min_speed = velocity.min()

    mean_abs_yaw_rate = (
        np.abs(yaw_rate).mean()
    )

    max_abs_yaw_rate = (
        np.abs(yaw_rate).max()
    )

    mean_abs_lat_acc = (
        np.abs(lat_acc).mean()
    )

    max_abs_lat_acc = (
        np.abs(lat_acc).max()
    )

    mean_abs_long_acc = (
        np.abs(long_acc).mean()
    )

    max_abs_long_acc = (
        np.abs(long_acc).max()
    )


    # --------------------------------------------------------
    # Heading continuity
    # --------------------------------------------------------

    heading_deltas = []

    for i in range(1, len(heading)):

        if (
            pd.notna(
                heading.iloc[i]
            )
            and
            pd.notna(
                heading.iloc[i - 1]
            )
        ):

            heading_deltas.append(
                circular_difference_deg(
                    heading.iloc[i],
                    heading.iloc[i - 1]
                )
            )


    heading_deltas = np.array(
        heading_deltas,
        dtype=float
    )


    if len(heading_deltas) > 0:

        max_heading_step = np.max(
            np.abs(heading_deltas)
        )

    else:

        max_heading_step = np.nan


    # --------------------------------------------------------
    # Yaw-rate expected change
    # --------------------------------------------------------

    # 10 Hz data = 0.1 s/sample
    #
    # Integrated yaw change across
    # the 10 samples.

    yaw_integrated_deg = (
        np.nansum(
            yaw_rate
        ) * 0.1
    )


    # The Step 25/26 analysis found that
    # inverted yaw-rate convention agrees
    # better with the GNSS/Heading convention.
    yaw_integrated_inverted_deg = (
        -yaw_integrated_deg
    )


    # --------------------------------------------------------
    # Heading total change directly from
    # first and last raw samples
    # --------------------------------------------------------

    if (
        pd.notna(heading.iloc[0])
        and
        pd.notna(heading.iloc[-1])
    ):

        raw_heading_change = (
            circular_difference_deg(
                heading.iloc[-1],
                heading.iloc[0]
            )
        )

    else:

        raw_heading_change = np.nan


    # --------------------------------------------------------
    # Compare Heading and inverted Yaw Rate
    # --------------------------------------------------------

    heading_yaw_error = (
        circular_abs_difference_deg(
            raw_heading_change,
            yaw_integrated_inverted_deg
        )
    )


    # --------------------------------------------------------
    # GNSS measurements from Step 26
    # --------------------------------------------------------

    gnss_change = float(
        row[
            "gnss_direction_change_deg"
        ]
    )

    gnss_abs_change = abs(
        gnss_change
    )

    gnss_spread = float(
        row[
            "gnss_bearing_spread_deg"
        ]
    )

    gnss_displacement = float(
        row[
            "gnss_displacement_m"
        ]
    )


    # --------------------------------------------------------
    # Existing pairwise errors
    # --------------------------------------------------------

    heading_gnss_error = float(
        row[
            "heading_vs_gnss_error_deg"
        ]
    )

    yaw_gnss_error = float(
        row[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================
    #
    # These are NOT ground-truth labels.
    #
    # They only describe which signal looks
    # internally inconsistent.
    # ========================================================

    classification = (
        "E_AMBIGUOUS"
    )


    evidence = []


    # --------------------------------------------------------
    # A: GNSS likely unreliable
    #
    # Conditions:
    # - Heading and yaw rate agree
    # - GNSS has large bearing spread
    # - vehicle is moving
    # --------------------------------------------------------

    if (
        heading_yaw_error <= 10.0
        and
        gnss_spread >= 45.0
        and
        gnss_displacement >= 1.0
    ):

        classification = (
            "A_GNSS_LIKELY_UNRELIABLE"
        )

        evidence.append(
            "Heading/YawRate agree"
        )

        evidence.append(
            "GNSS bearing unstable"
        )


    # --------------------------------------------------------
    # B: Heading likely unreliable
    #
    # Conditions:
    # - GNSS and inverted yaw-rate agree
    # - Heading differs strongly
    # --------------------------------------------------------

    elif (
        yaw_gnss_error <= 10.0
        and
        heading_gnss_error > 20.0
    ):

        classification = (
            "B_HEADING_LIKELY_UNRELIABLE"
        )

        evidence.append(
            "GNSS/YawRate agree"
        )

        evidence.append(
            "Heading disagrees"
        )


    # --------------------------------------------------------
    # C: Yaw Rate likely unreliable
    #
    # Conditions:
    # - GNSS and Heading agree
    # - Yaw Rate differs strongly
    # --------------------------------------------------------

    elif (
        heading_gnss_error <= 10.0
        and
        yaw_gnss_error > 20.0
    ):

        classification = (
            "C_YAWRATE_LIKELY_UNRELIABLE"
        )

        evidence.append(
            "GNSS/Heading agree"
        )

        evidence.append(
            "YawRate disagrees"
        )


    # --------------------------------------------------------
    # D: internally consistent vehicle signals
    #
    # Heading and Yaw Rate agree,
    # while GNSS is the outlier.
    # --------------------------------------------------------

    elif (
        heading_yaw_error <= 10.0
        and
        heading_gnss_error > 20.0
        and
        yaw_gnss_error > 20.0
    ):

        classification = (
            "A_GNSS_LIKELY_UNRELIABLE"
        )

        evidence.append(
            "Heading/YawRate mutually agree"
        )

        evidence.append(
            "GNSS disagrees with both"
        )


    # --------------------------------------------------------
    # E: ambiguous
    # --------------------------------------------------------

    else:

        classification = (
            "E_AMBIGUOUS"
        )

        evidence.append(
            "No two-signal consensus"
        )


    # ========================================================
    # SAVE
    # ========================================================

    results.append({

        "dataset":
            dataset,

        "window_index":
            window_index,

        "classification":
            classification,

        "evidence":
            "; ".join(evidence),

        "mean_speed_kmh":
            mean_speed,

        "min_speed_kmh":
            min_speed,

        "max_speed_kmh":
            max_speed,

        "gnss_displacement_m":
            gnss_displacement,

        "gnss_direction_change_deg":
            gnss_change,

        "gnss_abs_direction_change_deg":
            gnss_abs_change,

        "gnss_bearing_spread_deg":
            gnss_spread,

        "heading_change_step15_deg":
            row[
                "heading_change_deg"
            ],

        "raw_heading_change_deg":
            raw_heading_change,

        "yawrate_integrated_change_deg":
            yaw_integrated_deg,

        "yawrate_integrated_inverted_deg":
            yaw_integrated_inverted_deg,

        "heading_yawrate_error_deg":
            heading_yaw_error,

        "heading_vs_gnss_error_deg":
            heading_gnss_error,

        "yawrate_vs_gnss_error_deg":
            yaw_gnss_error,

        "mean_abs_yaw_rate_deg_s":
            mean_abs_yaw_rate,

        "max_abs_yaw_rate_deg_s":
            max_abs_yaw_rate,

        "mean_abs_lateral_acc":
            mean_abs_lat_acc,

        "max_abs_lateral_acc":
            max_abs_lat_acc,

        "mean_abs_longitudinal_acc":
            mean_abs_long_acc,

        "max_abs_longitudinal_acc":
            max_abs_long_acc,

        "max_heading_step_deg":
            max_heading_step
    })


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

counts = (
    result[
        "classification"
    ]
    .value_counts()
)


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 29 - MOVING DISAGREEMENT SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Total Step 26 windows: "
        f"{len(df):,}\n"
    )

    f.write(
        f"Three-way disagreement: "
        f"{len(disagreement):,}\n"
    )

    f.write(
        f"Moving disagreement >=5 km/h: "
        f"{len(moving):,}\n"
    )

    f.write(
        f"Successfully analyzed: "
        f"{len(result):,}\n\n"
    )


    f.write(
        "CLASSIFICATION COUNTS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )


    for category, count in counts.items():

        percentage = (
            100.0
            * count
            / len(result)
        )

        f.write(
            f"{category}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )


    f.write(
        "\n"
    )


    # --------------------------------------------------------
    # Statistical summary
    # --------------------------------------------------------

    f.write(
        "MOVING WINDOW STATISTICS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )


    columns = [
        "mean_speed_kmh",
        "gnss_displacement_m",
        "gnss_abs_direction_change_deg",
        "gnss_bearing_spread_deg",
        "raw_heading_change_deg",
        "yawrate_integrated_inverted_deg",
        "heading_yawrate_error_deg",
        "heading_vs_gnss_error_deg",
        "yawrate_vs_gnss_error_deg",
        "mean_abs_yaw_rate_deg_s",
        "max_abs_lateral_acc",
        "max_heading_step_deg"
    ]


    for col in columns:

        if col not in result.columns:

            continue


        s = pd.to_numeric(
            result[col],
            errors="coerce"
        ).dropna()


        if len(s) == 0:

            continue


        f.write(
            f"\n{col}\n"
        )

        f.write(
            f"  median = "
            f"{s.median():.4f}\n"
        )

        f.write(
            f"  P90    = "
            f"{s.quantile(.90):.4f}\n"
        )

        f.write(
            f"  P95    = "
            f"{s.quantile(.95):.4f}\n"
        )

        f.write(
            f"  max    = "
            f"{s.max():.4f}\n"
        )


    # --------------------------------------------------------
    # List all classified cases
    # --------------------------------------------------------

    f.write(
        "\n\nALL ANALYZED WINDOWS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )


    display_columns = [
        "dataset",
        "window_index",
        "classification",
        "mean_speed_kmh",
        "gnss_displacement_m",
        "gnss_direction_change_deg",
        "gnss_bearing_spread_deg",
        "raw_heading_change_deg",
        "yawrate_integrated_inverted_deg",
        "heading_yawrate_error_deg",
        "heading_vs_gnss_error_deg",
        "yawrate_vs_gnss_error_deg",
        "max_abs_lateral_acc",
        "max_heading_step_deg"
    ]


    for _, r in result[
        display_columns
    ].iterrows():

        f.write(
            f"{r['dataset']:8s} "
            f"w{int(r['window_index']):5d} | "
            f"{r['classification']:32s} | "
            f"speed={r['mean_speed_kmh']:.2f} | "
            f"GNSS={r['gnss_direction_change_deg']:.2f} | "
            f"Head={r['raw_heading_change_deg']:.2f} | "
            f"Yaw={r['yawrate_integrated_inverted_deg']:.2f} | "
            f"H-Y={r['heading_yawrate_error_deg']:.2f}\n"
        )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("STEP 29 COMPLETE")
print("=" * 75)

print(
    f"\nMoving disagreement windows: "
    f"{len(moving):,}"
)

print(
    f"Successfully analyzed: "
    f"{len(result):,}"
)

print(
    "\nClassification:"
)

for category, count in counts.items():

    percentage = (
        100.0
        * count
        / len(result)
    )

    print(
        f"  {category}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


print(
    f"\nCSV:"
)

print(
    OUTPUT_FILE
)

print(
    f"\nSummary TXT:"
)

print(
    SUMMARY_FILE
)

print(
    "\nNo data was deleted."
)

print(
    "No ground-truth signal was selected."
)
import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Maverick\ML2"

NORMALIZED_DIR = os.path.join(
    BASE,
    "step44_normalized_data"
)

STATS_FILE = os.path.join(
    BASE,
    "step43_normalization_statistics.npz"
)

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

MAPPING_FILE = os.path.join(
    BASE,
    "step58_v_mapping",
    "complete_v_file_mapping.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step59_full_disturbance_audit"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


print("=" * 100)
print("STEP 59 - FULL 105,048 WINDOW DISTURBANCE AUDIT")
print("=" * 100)


# ============================================================
# 1. LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(
    STATS_FILE
)

mean = stats["mean"].astype(
    np.float64
)

std = stats["std"].astype(
    np.float64
)


# ============================================================
# 2. LOAD ALL AVNET INPUT WINDOWS
# ============================================================

X_parts = []

split_parts = []


for split in [
    "train",
    "validation",
    "test"
]:

    path = os.path.join(
        NORMALIZED_DIR,
        split + ".npz"
    )

    data = np.load(
        path
    )

    X_norm = data[
        "X"
    ].astype(
        np.float64
    )

    X_raw = (
        X_norm * std
        + mean
    )

    X_parts.append(
        X_raw
    )

    split_parts.extend(
        [split] * len(X_raw)
    )

    print(
        f"{split:12s}: {X_raw.shape}"
    )


X = np.concatenate(
    X_parts,
    axis=0
)

split_labels = np.array(
    split_parts
)


print(
    f"\nTotal AVNet windows: {len(X)}"
)


# ============================================================
# 3. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

metadata = metadata.sort_values(
    "sample_index"
).reset_index(
    drop=True
)


if len(metadata) != len(X):

    raise RuntimeError(
        "Metadata/X length mismatch: "
        f"{len(metadata)} vs {len(X)}"
    )


# ============================================================
# 4. LOAD EXACT V-FILE MAPPING
# ============================================================

mapping_df = pd.read_csv(
    MAPPING_FILE
)

mapping_df[
    "dataset"
] = (
    mapping_df[
        "dataset"
    ]
    .astype(str)
    .str.lower()
)


mapping_df = mapping_df[
    mapping_df[
        "matched"
    ] == True
]


V_PATHS = dict(
    zip(
        mapping_df[
            "dataset"
        ],
        mapping_df[
            "v_file"
        ]
    )
)


print(
    f"V files available from mapping: "
    f"{len(V_PATHS)}"
)


# ============================================================
# 5. LOAD EVERY V FILE
# ============================================================

V_DATA = {}


print(
    "\nLoading V files..."
)


for dataset, path in sorted(
    V_PATHS.items()
):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"V file does not exist:\n{path}"
        )


    df = pd.read_csv(
        path
    )

    V_DATA[
        dataset
    ] = df


    print(
        f"{dataset:10s} "
        f"{len(df):8d} rows"
    )


print(
    f"\nLoaded V files: {len(V_DATA)}"
)


# ============================================================
# 6. PHONE IMU METRICS
# ============================================================

print(
    "\nCalculating phone IMU metrics..."
)


# X[:,:,0:3] = Accelerometer XYZ
# X[:,:,3:6] = Gyroscope XYZ

acc_mag = np.sqrt(
    np.sum(
        X[:, :, 0:3] ** 2,
        axis=2
    )
)

gyro_mag = np.sqrt(
    np.sum(
        X[:, :, 3:6] ** 2,
        axis=2
    )
)


phone_max_gyro = np.max(
    gyro_mag,
    axis=1
)

phone_mean_gyro = np.mean(
    gyro_mag,
    axis=1
)

phone_max_acc = np.max(
    acc_mag,
    axis=1
)

phone_min_acc = np.min(
    acc_mag,
    axis=1
)

phone_max_acc_deviation = np.max(
    np.abs(
        acc_mag - 9.80665
    ),
    axis=1
)


# ============================================================
# 7. VEHICLE COLUMN IDENTIFICATION
# ============================================================

def find_column(
    columns,
    candidates
):

    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }


    # Exact matching first

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:

            return normalized[
                key
            ]


    # Partial matching second

    for candidate in candidates:

        key = candidate.lower()

        for norm, original in normalized.items():

            if key in norm:

                return original


    return None


# ============================================================
# 8. PROCESS ALL 105,048 WINDOWS
# ============================================================

results = []

skipped = 0


print(
    "\nProcessing every AVNet window..."
)


for idx in range(
    len(X)
):

    meta = metadata.iloc[
        idx
    ]


    dataset = str(
        meta[
            "dataset"
        ]
    ).lower()


    if dataset not in V_DATA:

        skipped += 1

        continue


    vdf = V_DATA[
        dataset
    ]


    start_row = int(
        meta[
            "start_row"
        ]
    )

    end_row = int(
        meta[
            "end_row"
        ]
    )


    if (
        start_row < 0
        or
        end_row >= len(vdf)
        or
        end_row < start_row
    ):

        skipped += 1

        continue


    vwin = vdf.iloc[
        start_row:end_row + 1
    ]


    if len(vwin) != 10:

        skipped += 1

        continue


    # --------------------------------------------------------
    # FIND VEHICLE COLUMNS
    # --------------------------------------------------------

    yaw_col = find_column(
        vdf.columns,
        [
            "Yaw Rate (deg/sec)",
            "Yaw Rate"
        ]
    )


    long_col = find_column(
        vdf.columns,
        [
            "Indicated Longitudinal Acceleration (g)",
            "Longitudinal Acceleration"
        ]
    )


    lat_col = find_column(
        vdf.columns,
        [
            "Indicated Lateral Acceleration (g)",
            "Lateral Acceleration"
        ]
    )


    vel_col = find_column(
        vdf.columns,
        [
            "Velocity (km/hr)",
            "Velocity"
        ]
    )


    # --------------------------------------------------------
    # EXTRACT NUMERIC VALUES
    # --------------------------------------------------------

    def numeric_values(
        col
    ):

        if col is None:

            return np.full(
                len(vwin),
                np.nan
            )


        return pd.to_numeric(
            vwin[col],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )


    yawrate = numeric_values(
        yaw_col
    )

    longacc = numeric_values(
        long_col
    )

    latacc = numeric_values(
        lat_col
    )

    velocity = numeric_values(
        vel_col
    )


    # --------------------------------------------------------
    # VEHICLE METRICS
    # --------------------------------------------------------

    def max_abs(
        values
    ):

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            return np.nan

        return float(
            np.max(
                np.abs(values)
            )
        )


    max_vehicle_yawrate = max_abs(
        yawrate
    )

    max_vehicle_longacc = max_abs(
        longacc
    )

    max_vehicle_latacc = max_abs(
        latacc
    )


    if np.any(
        np.isfinite(
            velocity
        )
    ):

        mean_vehicle_speed = float(
            np.nanmean(
                velocity
            )
        )

    else:

        mean_vehicle_speed = np.nan


    # --------------------------------------------------------
    # THRESHOLD FLAGS
    # --------------------------------------------------------

    high_gyro = (
        phone_max_gyro[idx]
        >= 2.0
    )

    very_high_gyro = (
        phone_max_gyro[idx]
        >= 5.0
    )

    extreme_gyro = (
        phone_max_gyro[idx]
        >= 10.0
    )


    high_phone_acc = (
        phone_max_acc_deviation[idx]
        >= 5.0
    )


    low_vehicle_yaw = (
        np.isfinite(
            max_vehicle_yawrate
        )
        and
        max_vehicle_yawrate
        < 1.0
    )


    low_vehicle_acc = (
        np.isfinite(
            max_vehicle_longacc
        )
        and
        np.isfinite(
            max_vehicle_latacc
        )
        and
        max_vehicle_longacc
        < 0.20
        and
        max_vehicle_latacc
        < 0.20
    )


    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    if (
        extreme_gyro
        and
        low_vehicle_yaw
    ):

        category = (
            "EXTREME_GYRO_LOW_VEHICLE_YAW"
        )

    elif (
        very_high_gyro
        and
        low_vehicle_yaw
    ):

        category = (
            "VERY_HIGH_GYRO_LOW_VEHICLE_YAW"
        )

    elif (
        high_gyro
        and
        low_vehicle_yaw
    ):

        category = (
            "HIGH_GYRO_LOW_VEHICLE_YAW"
        )

    elif (
        high_phone_acc
        and
        low_vehicle_acc
    ):

        category = (
            "HIGH_PHONE_ACC_LOW_VEHICLE_ACC"
        )

    else:

        category = (
            "NO_STRONG_DISAGREEMENT"
        )


    results.append({

        "sample_index":
            int(
                meta[
                    "sample_index"
                ]
            ),

        "dataset":
            dataset,

        "split":
            split_labels[idx],

        "window_index":
            int(
                meta[
                    "window_index"
                ]
            ),

        "start_row":
            start_row,

        "end_row":
            end_row,

        "reference_delta_yaw_deg":
            float(
                meta[
                    "reference_delta_yaw_deg"
                ]
            ),

        "phone_max_gyro_rad_s":
            phone_max_gyro[idx],

        "phone_mean_gyro_rad_s":
            phone_mean_gyro[idx],

        "phone_max_acc_m_s2":
            phone_max_acc[idx],

        "phone_min_acc_m_s2":
            phone_min_acc[idx],

        "phone_max_acc_deviation":
            phone_max_acc_deviation[idx],

        "vehicle_max_yawrate_deg_s":
            max_vehicle_yawrate,

        "vehicle_max_longacc_g":
            max_vehicle_longacc,

        "vehicle_max_latacc_g":
            max_vehicle_latacc,

        "mean_vehicle_speed_kmh":
            mean_vehicle_speed,

        "high_gyro":
            high_gyro,

        "very_high_gyro":
            very_high_gyro,

        "extreme_gyro":
            extreme_gyro,

        "high_phone_acc":
            high_phone_acc,

        "low_vehicle_yaw":
            low_vehicle_yaw,

        "low_vehicle_acc":
            low_vehicle_acc,

        "category":
            category
    })


# ============================================================
# 9. BUILD RESULT DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


print(
    "\n"
    + "=" * 100
)

print(
    "COVERAGE CHECK"
)

print(
    "=" * 100
)

print(
    f"Expected windows : {len(X)}"
)

print(
    f"Processed windows: {len(df)}"
)

print(
    f"Skipped windows  : {skipped}"
)


# ============================================================
# 10. SAVE COMPLETE WINDOW DATA
# ============================================================

full_file = os.path.join(
    OUT_DIR,
    "full_105048_disturbance_audit.csv"
)

df.to_csv(
    full_file,
    index=False
)


# ============================================================
# 11. GLOBAL CATEGORY COUNTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "GLOBAL DISTURBANCE COUNTS"
)

print(
    "=" * 100
)


category_counts = (
    df[
        "category"
    ]
    .value_counts()
)


print(
    category_counts.to_string()
)


category_counts.to_csv(
    os.path.join(
        OUT_DIR,
        "global_category_counts.csv"
    )
)


# ============================================================
# 12. GYRO THRESHOLD COUNTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "PHONE GYRO THRESHOLDS"
)

print(
    "=" * 100
)


thresholds = [
    1,
    2,
    3,
    5,
    7,
    10,
    15,
    20
]


threshold_rows = []


for threshold in thresholds:

    count = np.sum(
        df[
            "phone_max_gyro_rad_s"
        ]
        >= threshold
    )


    percentage = (
        100.0
        *
        count
        /
        len(df)
    )


    threshold_rows.append({

        "threshold_rad_s":
            threshold,

        "count":
            int(count),

        "percentage":
            percentage
    })


    print(
        f">= {threshold:5.1f} rad/s : "
        f"{count:6d} "
        f"({percentage:.5f}%)"
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_df.to_csv(
    os.path.join(
        OUT_DIR,
        "gyro_threshold_counts.csv"
    ),
    index=False
)


# ============================================================
# 13. TOP 50 GYRO EVENTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TOP 50 PHONE GYRO EVENTS"
)

print(
    "=" * 100
)


top_gyro = (
    df.sort_values(
        "phone_max_gyro_rad_s",
        ascending=False
    )
    .head(50)
)


print(
    top_gyro[
        [
            "sample_index",
            "dataset",
            "split",
            "window_index",
            "phone_max_gyro_rad_s",
            "phone_max_acc_deviation",
            "vehicle_max_yawrate_deg_s",
            "vehicle_max_longacc_g",
            "vehicle_max_latacc_g",
            "mean_vehicle_speed_kmh",
            "category"
        ]
    ].to_string(
        index=False
    )
)


top_gyro.to_csv(
    os.path.join(
        OUT_DIR,
        "top_50_gyro_events.csv"
    ),
    index=False
)


# ============================================================
# 14. EXTREME GYRO + LOW VEHICLE YAW
# ============================================================

extreme = df[
    (
        df[
            "phone_max_gyro_rad_s"
        ]
        >= 5.0
    )
    &
    (
        df[
            "vehicle_max_yawrate_deg_s"
        ]
        < 1.0
    )
].copy()


print(
    "\n"
    + "=" * 100
)

print(
    "GYRO >= 5 rad/s + VEHICLE YAW RATE < 1 deg/s"
)

print(
    "=" * 100
)

print(
    f"Count: {len(extreme)}"
)


if len(extreme) > 0:

    print(
        extreme[
            [
                "sample_index",
                "dataset",
                "split",
                "window_index",
                "phone_max_gyro_rad_s",
                "phone_max_acc_deviation",
                "vehicle_max_yawrate_deg_s",
                "vehicle_max_longacc_g",
                "vehicle_max_latacc_g",
                "mean_vehicle_speed_kmh"
            ]
        ]
        .sort_values(
            "phone_max_gyro_rad_s",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


extreme.to_csv(
    os.path.join(
        OUT_DIR,
        "gyro_ge_5_low_vehicle_yaw.csv"
    ),
    index=False
)


# ============================================================
# 15. HIGH PHONE ACC + LOW VEHICLE ACC
# ============================================================

acc_disagreement = df[
    (
        df[
            "phone_max_acc_deviation"
        ]
        >= 5.0
    )
    &
    (
        df[
            "vehicle_max_longacc_g"
        ]
        < 0.20
    )
    &
    (
        df[
            "vehicle_max_latacc_g"
        ]
        < 0.20
    )
].copy()


print(
    "\n"
    + "=" * 100
)

print(
    "HIGH PHONE ACC + LOW VEHICLE ACC"
)

print(
    "=" * 100
)

print(
    f"Count: {len(acc_disagreement)}"
)


acc_disagreement.to_csv(
    os.path.join(
        OUT_DIR,
        "high_phone_acc_low_vehicle_acc.csv"
    ),
    index=False
)


# ============================================================
# 16. DATASET DISTRIBUTION
# ============================================================

dataset_summary = (
    df.groupby(
        "dataset"
    )
    .agg(

        windows=(
            "sample_index",
            "count"
        ),

        gyro_ge_2=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 2)
        ),

        gyro_ge_5=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 5)
        ),

        gyro_ge_10=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 10)
        ),

        acc_disagreement=(
            "phone_max_acc_deviation",
            lambda x:
            np.sum(
                x >= 5
            )
        )
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "DISTURBANCE DISTRIBUTION BY DATASET"
)

print(
    "=" * 100
)

print(
    dataset_summary.to_string()
)


dataset_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "dataset_summary.csv"
    )
)


# ============================================================
# 17. SPLIT DISTRIBUTION
# ============================================================

split_summary = (
    df.groupby(
        "split"
    )
    .agg(

        windows=(
            "sample_index",
            "count"
        ),

        gyro_ge_2=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 2)
        ),

        gyro_ge_5=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 5)
        ),

        gyro_ge_10=(
            "phone_max_gyro_rad_s",
            lambda x:
            np.sum(x >= 10)
        ),

        acc_disagreement=(
            "phone_max_acc_deviation",
            lambda x:
            np.sum(
                x >= 5
            )
        )
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "DISTURBANCE DISTRIBUTION BY SPLIT"
)

print(
    "=" * 100
)

print(
    split_summary.to_string()
)


split_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "split_summary.csv"
    )
)


# ============================================================
# 18. FINAL
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STEP 59 COMPLETE"
)

print(
    "=" * 100
)

print(
    f"Expected windows : {len(X)}"
)

print(
    f"Processed windows: {len(df)}"
)

print(
    f"Skipped windows  : {skipped}"
)

print(
    "\nComplete audit:"
)

print(
    full_file
)
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

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

OUT_DIR = os.path.join(
    BASE,
    "step56_global_disturbance_search"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ALL THREE SPLITS
# ============================================================

print("=" * 100)
print("STEP 56 - GLOBAL PHONE DISTURBANCE SEARCH")
print("=" * 100)


stats = np.load(
    STATS_FILE
)

mean = stats["mean"].astype(
    np.float64
)

std = stats["std"].astype(
    np.float64
)


splits = [
    "train",
    "validation",
    "test"
]


X_all = []
split_labels = []


for split in splits:

    path = os.path.join(
        NORMALIZED_DIR,
        split + ".npz"
    )

    data = np.load(
        path
    )

    X_norm = data["X"].astype(
        np.float64
    )

    X_raw = (
        X_norm * std
        + mean
    )

    X_all.append(
        X_raw
    )

    split_labels.extend(
        [split] * len(X_raw)
    )

    print(
        f"{split:12s}: {X_raw.shape}"
    )


X = np.concatenate(
    X_all,
    axis=0
)

split_labels = np.array(
    split_labels
)


print(
    f"\nTotal windows: {len(X)}"
)


# ============================================================
# LOAD METADATA
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
        f"Metadata/X mismatch: "
        f"{len(metadata)} vs {len(X)}"
    )


# ============================================================
# PHONE METRICS
# ============================================================

print(
    "\nCalculating phone IMU metrics..."
)


# Accelerometer magnitude
acc_mag = np.sqrt(
    np.sum(
        X[:, :, 0:3] ** 2,
        axis=2
    )
)


# Gyroscope magnitude
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

phone_max_acc_deviation = np.max(
    np.abs(
        acc_mag - 9.80665
    ),
    axis=1
)


# ============================================================
# DATASET -> V FILE MAPPING
# ============================================================

# Exact files for the complete synchronized dataset.
# The paths are based on the discovered Categorised IOVNB
# Dataset structure.

V_FILES = {}


def add_file(
    dataset,
    path
):

    V_FILES[
        dataset
    ] = path


# ------------------------------------------------------------
# S
# ------------------------------------------------------------

for name in [
    "s1",
    "s2",
    "s3a",
    "s3b",
    "s3c",
    "s4"
]:

    folder = name.upper()

    path = os.path.join(
        DATASET_ROOT,
        "S (Driver A)",
        folder,
        f"V-{folder}.csv"
    )

    if os.path.exists(path):

        add_file(
            name,
            path
        )


# ------------------------------------------------------------
# VTA
# ------------------------------------------------------------

for n in range(
    1,
    31
):

    if n == 18:

        continue

    if n == 19:

        pass

    dataset = (
        f"vta{n}"
    )

    if n < 10:

        folder = (
            f"Vta0{n}"
        )

        filename = (
            f"V-vta{n}.csv"
        )

    else:

        folder = (
            f"Vta{n}"
        )

        filename = (
            f"V-vta{n}.csv"
        )


    path = os.path.join(
        DATASET_ROOT,
        "Vta (Driver E)",
        folder,
        filename
    )

    if os.path.exists(path):

        add_file(
            dataset,
            path
        )


# ------------------------------------------------------------
# VFA
# ------------------------------------------------------------

for n in [
    "01",
    "02"
]:

    dataset = (
        "vfa" + n
    )

    folder = (
        "V-Vfa" + n
    )

    filename = (
        "V-Vfa" + n + ".csv"
    )

    path = os.path.join(
        DATASET_ROOT,
        "Vf (Driver E)",
        folder,
        filename
    )

    if os.path.exists(path):

        add_file(
            dataset,
            path
        )


# ------------------------------------------------------------
# VTB
# ------------------------------------------------------------

for n in range(
    1,
    13
):

    dataset = (
        f"vtb{n}"
    )

    folder = (
        f"Vtb{n:02d}"
    )

    filename = (
        f"V-vtb{n}.csv"
    )

    path = os.path.join(
        DATASET_ROOT,
        "Vtb (Driver E)",
        folder,
        filename
    )

    if os.path.exists(path):

        add_file(
            dataset,
            path
        )


# ------------------------------------------------------------
# VW
# ------------------------------------------------------------

vw_names = [
    "vw1",
    "vw2",
    "vw3",
    "vw4",
    "vw5",
    "vw6",
    "vw7",
    "vw8",
    "vw9",
    "vw10",
    "vw11",
    "vw12",
    "vw13",
    "vw14a",
    "vw14b",
    "vw14c",
    "vw15",
    "vw16a",
    "vw16b",
    "vw17"
]


for dataset in vw_names:

    suffix = dataset[2:]

    folder = (
        "Vw"
        + suffix
    )

    filename = (
        "V-Vw"
        + suffix
        + ".csv"
    )

    path = os.path.join(
        DATASET_ROOT,
        "Vw (Driver E)",
        folder,
        filename
    )

    if os.path.exists(path):

        add_file(
            dataset,
            path
        )


# ------------------------------------------------------------
# Y
# ------------------------------------------------------------

path = os.path.join(
    DATASET_ROOT,
    "Y (Driver D)",
    "Y1",
    "V-Y1.csv"
)

if os.path.exists(path):

    add_file(
        "y1",
        path
    )


print(
    f"\nV files discovered: {len(V_FILES)}"
)


# ============================================================
# LOAD V DATA ONCE
# ============================================================

V_DATA = {}


for dataset, path in V_FILES.items():

    try:

        V_DATA[
            dataset
        ] = pd.read_csv(
            path
        )

    except Exception as e:

        print(
            f"ERROR loading {dataset}: {e}"
        )


print(
    f"V files loaded: {len(V_DATA)}"
)


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(
    columns,
    candidates
):

    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:

            return normalized[key]


    for candidate in candidates:

        key = candidate.lower()

        for norm, original in normalized.items():

            if key in norm:

                return original


    return None


# ============================================================
# PROCESS ALL WINDOWS
# ============================================================

results = []

skipped = 0


print(
    "\nProcessing synchronized windows..."
)


for idx in range(
    len(X)
):

    meta = metadata.iloc[
        idx
    ]

    dataset = str(
        meta["dataset"]
    ).lower()

    start_row = int(
        meta["start_row"]
    )

    end_row = int(
        meta["end_row"]
    )


    if dataset not in V_DATA:

        skipped += 1

        continue


    vdf = V_DATA[
        dataset
    ]


    if end_row >= len(vdf):

        skipped += 1

        continue


    vwin = vdf.iloc[
        start_row:end_row + 1
    ]


    if len(vwin) != 10:

        skipped += 1

        continue


    # --------------------------------------------------------
    # VEHICLE COLUMNS
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


    def get_numeric(
        col
    ):

        if col is None:

            return np.full(
                10,
                np.nan
            )

        return pd.to_numeric(
            vwin[col],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )


    yawrate = get_numeric(
        yaw_col
    )

    longacc = get_numeric(
        long_col
    )

    latacc = get_numeric(
        lat_col
    )

    velocity = get_numeric(
        vel_col
    )


    # --------------------------------------------------------
    # VEHICLE METRICS
    # --------------------------------------------------------

    def max_abs(
        a
    ):

        a = a[
            np.isfinite(a)
        ]

        if len(a) == 0:

            return np.nan

        return float(
            np.max(
                np.abs(a)
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

    mean_vehicle_speed = np.nanmean(
        velocity
    )


    # --------------------------------------------------------
    # DISTURBANCE CONDITIONS
    # --------------------------------------------------------

    high_gyro = (
        phone_max_gyro[idx]
        >= 2.0
    )

    very_high_gyro = (
        phone_max_gyro[idx]
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

    high_phone_acc = (
        phone_max_acc_deviation[idx]
        >= 5.0
    )


    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    if (
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

    elif (
        high_gyro
        and
        high_phone_acc
        and
        low_vehicle_yaw
        and
        low_vehicle_acc
    ):

        category = (
            "COMBINED_DISAGREEMENT"
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

        "phone_max_gyro_rad_s":
            phone_max_gyro[idx],

        "phone_max_acc_m_s2":
            phone_max_acc[idx],

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

        "category":
            category
    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

output = os.path.join(
    OUT_DIR,
    "global_disturbance_windows.csv"
)

df.to_csv(
    output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "GLOBAL DISTURBANCE SUMMARY"
)

print(
    "=" * 100
)

print(
    f"Processed: {len(df)}"
)

print(
    f"Skipped:   {skipped}"
)


print(
    "\nCATEGORY COUNTS:"
)

print(
    df[
        "category"
    ].value_counts()
)


# ============================================================
# VERY HIGH GYRO
# ============================================================

very_high = df[
    df[
        "category"
    ]
    ==
    "VERY_HIGH_GYRO_LOW_VEHICLE_YAW"
].copy()


print(
    "\n"
    + "=" * 100
)

print(
    "VERY HIGH GYRO + LOW VEHICLE YAW"
)

print(
    "=" * 100
)

print(
    f"Count: {len(very_high)}"
)


if len(very_high) > 0:

    print(
        very_high[
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
        .head(50)
        .to_string(
            index=False
        )
    )


# ============================================================
# HIGH GYRO + LOW VEHICLE YAW
# ============================================================

high = df[
    df[
        "category"
    ]
    ==
    "HIGH_GYRO_LOW_VEHICLE_YAW"
].copy()


print(
    "\n"
    + "=" * 100
)

print(
    "HIGH GYRO + LOW VEHICLE YAW"
)

print(
    "=" * 100
)

print(
    f"Count: {len(high)}"
)


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

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


dataset_summary = (
    df.groupby(
        "dataset"
    )
    .agg(
        windows=(
            "sample_index",
            "count"
        ),

        very_high_gyro_low_yaw=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "VERY_HIGH_GYRO_LOW_VEHICLE_YAW"
            )
        ),

        high_gyro_low_yaw=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "HIGH_GYRO_LOW_VEHICLE_YAW"
            )
        ),

        high_phone_acc_low_vehicle_acc=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "HIGH_PHONE_ACC_LOW_VEHICLE_ACC"
            )
        )
    )
)

print(
    dataset_summary.to_string()
)


dataset_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "dataset_disturbance_summary.csv"
    )
)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

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


split_summary = (
    df.groupby(
        "split"
    )
    .agg(
        windows=(
            "sample_index",
            "count"
        ),

        very_high_gyro_low_yaw=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "VERY_HIGH_GYRO_LOW_VEHICLE_YAW"
            )
        ),

        high_gyro_low_yaw=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "HIGH_GYRO_LOW_VEHICLE_YAW"
            )
        ),

        high_phone_acc_low_vehicle_acc=(
            "category",
            lambda x:
            np.sum(
                x
                ==
                "HIGH_PHONE_ACC_LOW_VEHICLE_ACC"
            )
        )
    )
)

print(
    split_summary.to_string()
)


split_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "split_disturbance_summary.csv"
    )
)


# ============================================================
# TOP 30 PHONE GYRO WINDOWS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TOP 30 PHONE GYRO WINDOWS"
)

print(
    "=" * 100
)


print(
    df[
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
    ]
    .sort_values(
        "phone_max_gyro_rad_s",
        ascending=False
    )
    .head(30)
    .to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "STEP 56 COMPLETE"
)

print(
    "=" * 100
)

print(
    f"\nSaved:"
)

print(
    output
)
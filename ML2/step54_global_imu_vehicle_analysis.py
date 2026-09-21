import os
import glob
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

DDODO_FILE = os.path.join(
    BASE,
    "step45_ddodo_data",
    "test.npz"
)

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\S (Driver A)"
)

OUT_DIR = os.path.join(
    BASE,
    "step54_global_imu_vehicle_analysis"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# TEST SEQUENCES
# ============================================================

TEST_SEQUENCES = [
    "vta5",
    "vta1a",
    "s2",
    "vta4",
    "vfa01",
    "vw14b",
    "vta28",
    "vw8",
    "vta9"
]


# ============================================================
# LOAD NORMALIZED TEST IMU
# ============================================================

print("=" * 90)
print("STEP 54 - GLOBAL PHONE IMU / VEHICLE DYNAMICS ANALYSIS")
print("=" * 90)

data = np.load(
    os.path.join(
        NORMALIZED_DIR,
        "test.npz"
    )
)

X_norm = data["X"].astype(
    np.float32
)

print(
    f"\nNormalized test X: {X_norm.shape}"
)


# ============================================================
# RESTORE RAW IMU
# ============================================================

stats = np.load(
    STATS_FILE
)

means = stats["mean"].astype(
    np.float32
)

stds = stats["std"].astype(
    np.float32
)

X = (
    X_norm * stds
    + means
)

print(
    f"Raw test X restored: {X.shape}"
)


# ============================================================
# LOAD VELOCITY TARGET
# ============================================================

odo = np.load(
    DDODO_FILE
)

velocity = odo["Y"][:, 0]

print(
    f"Velocity samples: {len(velocity)}"
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

metadata = metadata[
    metadata["dataset"].isin(
        TEST_SEQUENCES
    )
].copy()

metadata = metadata.sort_values(
    "sample_index"
).reset_index(
    drop=True
)

print(
    f"Metadata samples: {len(metadata)}"
)


if len(metadata) != len(X):

    raise RuntimeError(
        f"\nERROR: Metadata/test mismatch\n"
        f"Metadata = {len(metadata)}\n"
        f"Test X   = {len(X)}"
    )


# ============================================================
# PHONE IMU METRICS
# ============================================================

print(
    "\nCalculating phone IMU metrics..."
)

# X columns:
#
# 0 = Acc X
# 1 = Acc Y
# 2 = Acc Z
# 3 = Gyro X
# 4 = Gyro Y
# 5 = Gyro Z


gyro_mag = np.sqrt(
    np.sum(
        X[:, :, 3:6] ** 2,
        axis=2
    )
)

acc_mag = np.sqrt(
    np.sum(
        X[:, :, 0:3] ** 2,
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

phone_mean_acc = np.mean(
    acc_mag,
    axis=1
)

phone_max_acc_dev = np.max(
    np.abs(
        acc_mag - 9.80665
    ),
    axis=1
)


# ============================================================
# EXACT SYNCHRONIZED V FILES
# ============================================================

V_FILES = {

    "vta5":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vta (Driver E)\Vta05\V-vta5.csv",

    "vta1a":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vta (Driver E)\Vta01a\V-Vta1a.csv",

    "s2":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\S (Driver A)\S2\V-S2.csv",

    "vta4":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vta (Driver E)\Vta04\V-vta4.csv",

    "vfa01":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vf (Driver E)\V-Vfa01\V-Vfa01.csv",

    "vw14b":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vw (Driver E)\Vw14b\V-Vw14b.csv",

    "vta28":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vta (Driver E)\Vta28\V-vta28.csv",

    "vw8":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vw (Driver E)\Vw08\V-Vw8.csv",

    "vta9":
        r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets"
        r"\Categorised IOVNB Dataset\Vta (Driver E)\Vta09\V-vta9.csv"
}


print(
    "\n"
    + "=" * 90
)

print(
    "CHECKING EXACT SYNCHRONIZED V FILES"
)

print(
    "=" * 90
)

for dataset, path in V_FILES.items():

    if os.path.exists(path):

        print(
            f"[OK] {dataset}: {path}"
        )

    else:

        print(
            f"[MISSING] {dataset}: {path}"
        )


# ============================================================
# LOAD EACH V FILE ONLY ONCE
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "LOADING V FILES ONCE"
)

print(
    "=" * 90
)

V_DATA = {}

for dataset, path in V_FILES.items():

    if not os.path.exists(path):

        print(
            f"Skipping {dataset} - file not found"
        )

        continue

    print(
        f"Loading {dataset}: {path}"
    )

    try:

        vdf = pd.read_csv(
            path
        )

        V_DATA[
            dataset
        ] = vdf

        print(
            f"  Rows: {len(vdf)}"
        )

    except Exception as e:

        print(
            f"  ERROR loading {dataset}: {e}"
        )


print(
    "\nV files successfully loaded:",
    len(V_DATA),
    "/",
    len(TEST_SEQUENCES)
)

# ============================================================
# LOAD EACH V FILE ONLY ONCE
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "LOADING V FILES ONCE"
)

print(
    "=" * 90
)


V_DATA = {}

for dataset, path in V_FILES.items():

    print(
        f"Loading {dataset}: {path}"
    )

    try:

        vdf = pd.read_csv(
            path
        )

        V_DATA[
            dataset
        ] = vdf

        print(
            f"  Rows: {len(vdf)}"
        )

    except Exception as e:

        print(
            f"  ERROR: {e}"
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

    # Exact match
    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:

            return normalized[key]


    # Partial match
    for candidate in candidates:

        key = candidate.lower()

        for norm, original in normalized.items():

            if key in norm:

                return original


    return None


# ============================================================
# PROCESS WINDOWS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PROCESSING TEST WINDOWS"
)

print(
    "=" * 90
)


results = []

skipped = 0


for idx in range(
    len(X)
):

    meta = metadata.iloc[
        idx
    ]

    dataset = str(
        meta["dataset"]
    )

    window_index = int(
        meta["window_index"]
    )

    start_row = int(
        meta["start_row"]
    )

    end_row = int(
        meta["end_row"]
    )

    yaw_reference = float(
        meta[
            "reference_delta_yaw_deg"
        ]
    )

    speed_reference = float(
        velocity[idx]
    )


    # --------------------------------------------------------
    # V DATA ALREADY LOADED
    # --------------------------------------------------------

    if dataset not in V_DATA:

        skipped += 1

        continue


    vdf = V_DATA[
        dataset
    ]


    # --------------------------------------------------------
    # EXACT SYNCHRONIZED WINDOW
    # --------------------------------------------------------

    if (
        start_row < 0
        or end_row >= len(vdf)
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
    # FIND COLUMNS
    # --------------------------------------------------------

    yaw_col = find_column(
        vdf.columns,
        [
            "Yaw Rate (deg/sec)",
            "Yaw Rate (degrees/sec)",
            "Yaw Rate"
        ]
    )

    long_col = find_column(
        vdf.columns,
        [
            "Indicated Longitudinal Acceleration (g)",
            "Longitudinal Acceleration (g)",
            "Longitudinal Acceleration"
        ]
    )

    lat_col = find_column(
        vdf.columns,
        [
            "Indicated Lateral Acceleration (g)",
            "Lateral Acceleration (g)",
            "Lateral Acceleration"
        ]
    )

    steer_col = find_column(
        vdf.columns,
        [
            "Steering Angle (degrees)",
            "Steering Angle"
        ]
    )

    vel_col = find_column(
        vdf.columns,
        [
            "Velocity (km/hr)",
            "Velocity (km/h)",
            "Velocity"
        ]
    )


    # --------------------------------------------------------
    # CONVERT COLUMN
    # --------------------------------------------------------

    def numeric(
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


    vehicle_yawrate = numeric(
        yaw_col
    )

    vehicle_longacc = numeric(
        long_col
    )

    vehicle_latacc = numeric(
        lat_col
    )

    vehicle_steering = numeric(
        steer_col
    )

    vehicle_velocity = numeric(
        vel_col
    )


    # --------------------------------------------------------
    # METRIC FUNCTIONS
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


    def mean_abs(
        a
    ):

        a = a[
            np.isfinite(a)
        ]

        if len(a) == 0:

            return np.nan

        return float(
            np.mean(
                np.abs(a)
            )
        )


    def value_range(
        a
    ):

        a = a[
            np.isfinite(a)
        ]

        if len(a) == 0:

            return np.nan

        return float(
            np.max(a)
            -
            np.min(a)
        )


    # --------------------------------------------------------
    # DIAGNOSTIC CATEGORY
    # --------------------------------------------------------

    pg = phone_max_gyro[
        idx
    ]

    if pg < 1.0:

        imu_category = (
            "LOW_IMU"
        )

    elif pg < 2.0:

        imu_category = (
            "NORMAL_IMU"
        )

    elif pg < 5.0:

        imu_category = (
            "HIGH_IMU"
        )

    else:

        imu_category = (
            "EXTREME_IMU"
        )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "sample_index":
            int(
                meta[
                    "sample_index"
                ]
            ),

        "dataset":
            dataset,

        "window_index":
            window_index,

        "start_row":
            start_row,

        "end_row":
            end_row,

        "reference_yaw_deg":
            yaw_reference,

        "reference_velocity_kmh":
            speed_reference,

        "phone_max_gyro_rad_s":
            phone_max_gyro[idx],

        "phone_mean_gyro_rad_s":
            phone_mean_gyro[idx],

        "phone_max_acc_m_s2":
            phone_max_acc[idx],

        "phone_mean_acc_m_s2":
            phone_mean_acc[idx],

        "phone_max_acc_deviation":
            phone_max_acc_dev[idx],

        "vehicle_max_abs_yawrate_deg_s":
            max_abs(
                vehicle_yawrate
            ),

        "vehicle_mean_abs_yawrate_deg_s":
            mean_abs(
                vehicle_yawrate
            ),

        "vehicle_max_abs_longacc_g":
            max_abs(
                vehicle_longacc
            ),

        "vehicle_max_abs_latacc_g":
            max_abs(
                vehicle_latacc
            ),

        "vehicle_steering_range_deg":
            value_range(
                vehicle_steering
            ),

        "vehicle_velocity_range_kmh":
            value_range(
                vehicle_velocity
            ),

        "imu_category":
            imu_category
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

output_file = os.path.join(
    OUT_DIR,
    "global_imu_vehicle_windows.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT COUNTS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "PROCESSING COMPLETE"
)

print(
    "=" * 90
)

print(
    f"Processed windows: {len(df)}"
)

print(
    f"Skipped windows:   {skipped}"
)


# ============================================================
# IMU CATEGORY DISTRIBUTION
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "IMU CATEGORY DISTRIBUTION"
)

print(
    "=" * 90
)

print(
    df[
        "imu_category"
    ].value_counts()
)


# ============================================================
# CATEGORY STATISTICS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "VEHICLE DYNAMICS BY IMU CATEGORY"
)

print(
    "=" * 90
)


category_summary = (
    df.groupby(
        "imu_category"
    )
    .agg({

        "phone_max_gyro_rad_s":
            [
                "count",
                "mean",
                "median",
                "max"
            ],

        "reference_velocity_kmh":
            [
                "mean",
                "median"
            ],

        "vehicle_max_abs_yawrate_deg_s":
            [
                "mean",
                "median",
                "max"
            ],

        "vehicle_max_abs_longacc_g":
            [
                "mean",
                "median",
                "max"
            ],

        "vehicle_max_abs_latacc_g":
            [
                "mean",
                "median",
                "max"
            ],

        "vehicle_steering_range_deg":
            [
                "mean",
                "median",
                "max"
            ]
    })
)


print(
    category_summary.to_string()
)


# ============================================================
# EXTREME IMU WINDOWS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "EXTREME IMU WINDOWS (>= 5 rad/s)"
)

print(
    "=" * 90
)


extreme = df[
    df[
        "phone_max_gyro_rad_s"
    ] >= 5.0
].copy()


if len(extreme) > 0:

    print(
        extreme[
            [
                "dataset",
                "sample_index",
                "window_index",
                "reference_velocity_kmh",
                "reference_yaw_deg",
                "phone_max_gyro_rad_s",
                "phone_max_acc_deviation",
                "vehicle_max_abs_yawrate_deg_s",
                "vehicle_max_abs_longacc_g",
                "vehicle_max_abs_latacc_g",
                "vehicle_steering_range_deg"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No extreme IMU windows."
    )


# ============================================================
# LOW SPEED + EXTREME IMU
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "EXTREME PHONE IMU AT LOW VEHICLE SPEED (<5 km/h)"
)

print(
    "=" * 90
)


low_speed_extreme = df[
    (
        df[
            "phone_max_gyro_rad_s"
        ] >= 5.0
    )
    &
    (
        df[
            "reference_velocity_kmh"
        ] < 5.0
    )
]


if len(
    low_speed_extreme
) > 0:

    print(
        low_speed_extreme[
            [
                "dataset",
                "sample_index",
                "window_index",
                "reference_velocity_kmh",
                "reference_yaw_deg",
                "phone_max_gyro_rad_s",
                "vehicle_max_abs_yawrate_deg_s",
                "vehicle_max_abs_longacc_g",
                "vehicle_max_abs_latacc_g"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No low-speed extreme IMU windows."
    )


# ============================================================
# CORRELATIONS
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "GLOBAL CORRELATIONS"
)

print(
    "=" * 90
)


correlation_columns = [

    "phone_max_gyro_rad_s",

    "phone_mean_gyro_rad_s",

    "phone_max_acc_deviation",

    "reference_velocity_kmh",

    "reference_yaw_deg",

    "vehicle_max_abs_yawrate_deg_s",

    "vehicle_max_abs_longacc_g",

    "vehicle_max_abs_latacc_g",

    "vehicle_steering_range_deg"
]


corr = df[
    correlation_columns
].corr()


print(
    corr[
        "phone_max_gyro_rad_s"
    ].sort_values(
        ascending=False
    )
)


# ============================================================
# SAVE SUMMARY FILES
# ============================================================

category_summary_file = os.path.join(
    OUT_DIR,
    "imu_category_vehicle_summary.csv"
)

category_summary.to_csv(
    category_summary_file
)


corr_file = os.path.join(
    OUT_DIR,
    "global_correlations.csv"
)

corr.to_csv(
    corr_file
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 90
)

print(
    output_file
)

print(
    category_summary_file
)

print(
    corr_file
)

print(
    "\nSTEP 54 COMPLETE"
)

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
    "step53_imu_vehicle_comparison"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(
    os.path.join(
        NORMALIZED_DIR,
        "test.npz"
    )
)

X_norm = data["X"].astype(
    np.float32
)

Y_att = data["Y"].astype(
    np.float32
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


# ============================================================
# VELOCITY
# ============================================================

odo = np.load(
    DDODO_FILE
)

velocity = odo["Y"][:, 0]


# ============================================================
# METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

# IMPORTANT:
# Metadata contains ALL 105048 samples.
# We need only the test samples.

# Test sequences from Step 38
test_sequences = [
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

test_metadata = metadata[
    metadata["dataset"].isin(
        test_sequences
    )
].copy()

test_metadata = test_metadata.sort_values(
    "sample_index"
).reset_index(
    drop=True
)


if len(test_metadata) != len(X):

    raise RuntimeError(
        f"Metadata/test mismatch: "
        f"{len(test_metadata)} vs {len(X)}"
    )


# ============================================================
# FIND EXTREME GYRO WINDOWS
# ============================================================

gyro_mag = np.sqrt(
    np.sum(
        X[:, :, 3:6] ** 2,
        axis=2
    )
)

max_gyro = np.max(
    gyro_mag,
    axis=1
)

extreme_indices = np.argsort(
    max_gyro
)[-10:][::-1]


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
# VEHICLE COLUMN NAMES
# ============================================================

VELOCITY_NAMES = [
    "Velocity (km/hr)"
]

HEADING_NAMES = [
    "Heading (degrees)"
]

YAWRATE_NAMES = [
    "Yaw Rate (deg/sec)"
]

LONGACC_NAMES = [
    "Indicated Longitudinal Acceleration (g)"
]

LATACC_NAMES = [
    "Indicated Lateral Acceleration (g)"
]

STEERING_NAMES = [
    "Steering Angle (degrees)"
]


# ============================================================
# LOAD V FILE
# ============================================================

def load_v_file(
    dataset
):

    folder = os.path.join(
        DATASET_ROOT,
        dataset.upper()
    )

    path = os.path.join(
        folder,
        "V-" + dataset.upper() + ".csv"
    )

    if not os.path.exists(path):

        print(
            "V file not found:",
            path
        )

        return None, None

    df = pd.read_csv(
        path
    )

    return path, df


# ============================================================
# SAFE NUMERIC
# ============================================================

def numeric_values(
    df,
    column
):

    if column is None:

        return np.full(
            len(df),
            np.nan
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

results = []

print("=" * 90)
print("STEP 53D - EXACT SYNCHRONIZED IMU / VEHICLE COMPARISON")
print("=" * 90)


for rank, test_idx in enumerate(
    extreme_indices,
    start=1
):

    row = test_metadata.iloc[
        test_idx
    ]

    dataset = row["dataset"]

    sample_index = int(
        row["sample_index"]
    )

    window_index = int(
        row["window_index"]
    )

    start_row = int(
        row["start_row"]
    )

    end_row = int(
        row["end_row"]
    )

    reference_yaw = float(
        row["reference_delta_yaw_deg"]
    )

    reference_velocity = float(
        velocity[test_idx]
    )

    phone_window = X[
        test_idx
    ]

    print("\n")
    print("=" * 90)

    print(
        f"EVENT #{rank}"
    )

    print("=" * 90)

    print(
        f"Dataset              : {dataset}"
    )

    print(
        f"Global AVNet sample  : {sample_index}"
    )

    print(
        f"Window index         : {window_index}"
    )

    print(
        f"Original S/V rows    : "
        f"{start_row} - {end_row}"
    )

    print(
        f"Reference yaw change : "
        f"{reference_yaw:.3f} deg"
    )

    print(
        f"Reference velocity   : "
        f"{reference_velocity:.3f} km/h"
    )

    print(
        f"Phone max gyro       : "
        f"{max_gyro[test_idx]:.6f} rad/s"
    )


    # --------------------------------------------------------
    # LOAD V
    # --------------------------------------------------------

    v_path, vdf = load_v_file(
        dataset
    )

    if vdf is None:

        continue


    print(
        "\nV file:",
        v_path
    )

    print(
        "V rows:",
        len(vdf)
    )


    # --------------------------------------------------------
    # EXACT SYNCHRONIZED WINDOW
    # --------------------------------------------------------

    v_window = vdf.iloc[
        start_row:end_row + 1
    ].copy()


    print(
        f"Selected V rows: "
        f"{start_row} - {end_row}"
    )

    print(
        f"Number of V rows: "
        f"{len(v_window)}"
    )


    # --------------------------------------------------------
    # FIND COLUMNS
    # --------------------------------------------------------

    velocity_col = find_column(
        vdf.columns,
        VELOCITY_NAMES
    )

    heading_col = find_column(
        vdf.columns,
        HEADING_NAMES
    )

    yawrate_col = find_column(
        vdf.columns,
        YAWRATE_NAMES
    )

    longacc_col = find_column(
        vdf.columns,
        LONGACC_NAMES
    )

    latacc_col = find_column(
        vdf.columns,
        LATACC_NAMES
    )

    steering_col = find_column(
        vdf.columns,
        STEERING_NAMES
    )


    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    vv = numeric_values(
        v_window,
        velocity_col
    )

    vh = numeric_values(
        v_window,
        heading_col
    )

    vyr = numeric_values(
        v_window,
        yawrate_col
    )

    vlong = numeric_values(
        v_window,
        longacc_col
    )

    vlat = numeric_values(
        v_window,
        latacc_col
    )

    vst = numeric_values(
        v_window,
        steering_col
    )


    # --------------------------------------------------------
    # PRINT VEHICLE DATA
    # --------------------------------------------------------

    print("\nSynchronized V data:")

    print(
        "No | Velocity | Heading | "
        "YawRate | LongAcc(g) | LatAcc(g) | Steering"
    )

    for j in range(
        len(v_window)
    ):

        print(
            f"{j+1:2d} | "
            f"{vv[j]:9.3f} | "
            f"{vh[j]:9.3f} | "
            f"{vyr[j]:9.3f} | "
            f"{vlong[j]:10.4f} | "
            f"{vlat[j]:9.4f} | "
            f"{vst[j]:9.3f}"
        )


    # --------------------------------------------------------
    # PRINT PHONE IMU
    # --------------------------------------------------------

    print("\nPhone IMU data:")

    print(
        "No | AccX | AccY | AccZ | "
        "GyroX | GyroY | GyroZ | GyroMag"
    )

    for j in range(10):

        ax, ay, az = (
            phone_window[j, 0:3]
        )

        gx, gy, gz = (
            phone_window[j, 3:6]
        )

        gm = np.sqrt(
            gx**2
            + gy**2
            + gz**2
        )

        print(
            f"{j+1:2d} | "
            f"{ax:8.3f} | "
            f"{ay:8.3f} | "
            f"{az:8.3f} | "
            f"{gx:8.3f} | "
            f"{gy:8.3f} | "
            f"{gz:8.3f} | "
            f"{gm:8.3f}"
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def max_abs(
        a
    ):

        a = a[
            np.isfinite(a)
        ]

        if len(a) == 0:
            return np.nan

        return np.max(
            np.abs(a)
        )


    results.append({

        "dataset":
            dataset,

        "sample_index":
            sample_index,

        "window_index":
            window_index,

        "start_row":
            start_row,

        "end_row":
            end_row,

        "phone_max_gyro":
            max_gyro[test_idx],

        "reference_yaw_deg":
            reference_yaw,

        "reference_velocity_kmh":
            reference_velocity,

        "vehicle_max_abs_yawrate":
            max_abs(vyr),

        "vehicle_max_abs_longacc_g":
            max_abs(vlong),

        "vehicle_max_abs_latacc_g":
            max_abs(vlat),

        "vehicle_max_abs_steering_deg":
            max_abs(vst)
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

output = os.path.join(
    OUT_DIR,
    "imu_vehicle_event_summary.csv"
)

summary.to_csv(
    output,
    index=False
)


print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    output
)

print(
    "\nSTEP 53D COMPLETE"
)
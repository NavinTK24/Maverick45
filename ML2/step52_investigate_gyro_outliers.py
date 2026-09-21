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

SPLIT_FILE = os.path.join(
    BASE,
    "step38_avnet_data",
    "sequence_split.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step52_gyro_outlier_investigation"
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

X_normalized = data["X"].astype(
    np.float32
)

Y_att = data["Y"].astype(
    np.float32
)


# ============================================================
# RESTORE RAW IMU VALUES
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
    X_normalized * stds
    + means
)


# ============================================================
# LOAD VELOCITY
# ============================================================

odo = np.load(
    DDODO_FILE
)

velocity = odo["Y"][:, 0]


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

split = pd.read_csv(
    SPLIT_FILE
)

test_sequences = split.loc[
    split["split"].str.upper() == "TEST",
    "dataset"
].tolist()


metadata = metadata[
    metadata["dataset"].isin(
        test_sequences
    )
].copy()

metadata = metadata.sort_values(
    "sample_index"
).reset_index(drop=True)


if len(metadata) != len(X):

    raise RuntimeError(
        "Metadata and test data size mismatch."
    )


# ============================================================
# REFERENCE YAW
# ============================================================

reference_yaw = (
    2
    * np.arcsin(
        np.clip(
            Y_att[:, 2],
            -1,
            1
        )
    )
    * 180
    / np.pi
)


# ============================================================
# FIND EXTREME GYRO WINDOWS
# ============================================================

# Maximum absolute gyro value inside each 1-second window

max_abs_gyro = np.max(
    np.abs(
        X[:, :, 3:6]
    ),
    axis=(1, 2)
)

# Maximum gyro magnitude

gyro_magnitude = np.sqrt(
    np.sum(
        X[:, :, 3:6] ** 2,
        axis=2
    )
)

max_gyro_magnitude = np.max(
    gyro_magnitude,
    axis=1
)


# ============================================================
# SELECT EXTREME WINDOWS
# ============================================================

largest_indices = np.argsort(
    max_gyro_magnitude
)[-10:][::-1]


print("=" * 70)
print("STEP 52 - GYROSCOPE OUTLIER INVESTIGATION")
print("=" * 70)

print("\nTop 10 windows by maximum gyro magnitude:")

for rank, idx in enumerate(
    largest_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"dataset="
        f"{metadata.iloc[idx]['dataset']}, "
        f"sample="
        f"{metadata.iloc[idx]['sample_index']}, "
        f"max gyro magnitude="
        f"{max_gyro_magnitude[idx]:.5f}, "
        f"yaw="
        f"{reference_yaw[idx]:.3f} deg, "
        f"velocity="
        f"{velocity[idx]:.3f} km/h"
    )


# ============================================================
# WINDOW PRINTER
# ============================================================

def print_window(
    idx,
    label
):

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    dataset = metadata.iloc[idx][
        "dataset"
    ]

    sample_index = metadata.iloc[idx][
        "sample_index"
    ]

    print(
        "Dataset:",
        dataset
    )

    print(
        "Sample index:",
        sample_index
    )

    print(
        f"Reference yaw change: "
        f"{reference_yaw[idx]:.6f} deg"
    )

    print(
        f"Reference velocity: "
        f"{velocity[idx]:.6f} km/h"
    )

    print(
        f"Maximum gyro magnitude: "
        f"{max_gyro_magnitude[idx]:.6f}"
    )

    print(
        f"Maximum absolute gyro: "
        f"{max_abs_gyro[idx]:.6f}"
    )

    print("\n10-sample window:")

    print(
        "No. | "
        "AccX | AccY | AccZ | "
        "GyroX | GyroY | GyroZ | "
        "GyroMag"
    )

    for i in range(10):

        ax = X[idx, i, 0]
        ay = X[idx, i, 1]
        az = X[idx, i, 2]

        gx = X[idx, i, 3]
        gy = X[idx, i, 4]
        gz = X[idx, i, 5]

        gm = np.sqrt(
            gx**2
            + gy**2
            + gz**2
        )

        print(
            f"{i+1:3d} | "
            f"{ax:8.4f} | "
            f"{ay:8.4f} | "
            f"{az:8.4f} | "
            f"{gx:9.5f} | "
            f"{gy:9.5f} | "
            f"{gz:9.5f} | "
            f"{gm:9.5f}"
        )


# ============================================================
# PRINT TOP EXTREME WINDOWS
# ============================================================

for rank, idx in enumerate(
    largest_indices[:5],
    start=1
):

    print_window(
        idx,
        f"EXTREME WINDOW #{rank}"
    )


# ============================================================
# NEIGHBOUR ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("NEIGHBOURING WINDOWS AROUND EXTREME EVENTS")
print("=" * 70)


def find_neighbor_indices(
    idx,
    radius=3
):

    start = max(
        0,
        idx - radius
    )

    end = min(
        len(X),
        idx + radius + 1
    )

    return range(
        start,
        end
    )


for rank, idx in enumerate(
    largest_indices[:3],
    start=1
):

    print("\n")
    print(
        f"Extreme event #{rank}: "
        f"{metadata.iloc[idx]['dataset']} "
        f"sample "
        f"{metadata.iloc[idx]['sample_index']}"
    )

    print(
        "Relative | "
        "Sample | "
        "MaxGyro | "
        "YawChange | "
        "Velocity"
    )

    center_dataset = metadata.iloc[idx][
        "dataset"
    ]

    for j in find_neighbor_indices(
        idx,
        radius=3
    ):

        # Only compare within same sequence.
        if (
            metadata.iloc[j]["dataset"]
            != center_dataset
        ):
            continue

        print(
            f"{j-idx:+8d} | "
            f"{metadata.iloc[j]['sample_index']} | "
            f"{max_gyro_magnitude[j]:8.4f} | "
            f"{reference_yaw[j]:9.3f} | "
            f"{velocity[j]:8.3f}"
        )


# ============================================================
# OUTLIER THRESHOLD COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("GYRO MAGNITUDE DISTRIBUTION")
print("=" * 70)

thresholds = [
    1,
    2,
    3,
    5,
    10,
    15
]

for threshold in thresholds:

    count = np.sum(
        max_gyro_magnitude
        >= threshold
    )

    percentage = (
        100
        * count
        / len(max_gyro_magnitude)
    )

    print(
        f">= {threshold:5.1f} rad/s : "
        f"{count:6d} windows "
        f"({percentage:.4f}%)"
    )


# ============================================================
# TOP INDIVIDUAL GYRO SAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 INDIVIDUAL GYRO SAMPLES")
print("=" * 70)

individual_values = []

for window_idx in range(
    len(X)
):

    for sample_idx in range(10):

        gx = X[
            window_idx,
            sample_idx,
            3
        ]

        gy = X[
            window_idx,
            sample_idx,
            4
        ]

        gz = X[
            window_idx,
            sample_idx,
            5
        ]

        magnitude = np.sqrt(
            gx**2
            + gy**2
            + gz**2
        )

        individual_values.append(
            (
                magnitude,
                window_idx,
                sample_idx,
                gx,
                gy,
                gz
            )
        )


individual_values.sort(
    key=lambda x: x[0],
    reverse=True
)


print(
    "Rank | Dataset | Window | "
    "Sample | Gx | Gy | Gz | Magnitude"
)

for rank, item in enumerate(
    individual_values[:20],
    start=1
):

    magnitude, w, s, gx, gy, gz = item

    print(
        f"{rank:4d} | "
        f"{metadata.iloc[w]['dataset']} | "
        f"{metadata.iloc[w]['sample_index']} | "
        f"{s+1:6d} | "
        f"{gx:8.4f} | "
        f"{gy:8.4f} | "
        f"{gz:8.4f} | "
        f"{magnitude:9.4f}"
    )


# ============================================================
# SAVE OUTLIER SUMMARY
# ============================================================

outlier_rows = []

for idx in largest_indices:

    outlier_rows.append({

        "dataset":
            metadata.iloc[idx]["dataset"],

        "sample_index":
            metadata.iloc[idx]["sample_index"],

        "max_gyro_magnitude":
            max_gyro_magnitude[idx],

        "max_absolute_gyro":
            max_abs_gyro[idx],

        "reference_yaw_deg":
            reference_yaw[idx],

        "velocity_kmh":
            velocity[idx]
    })


pd.DataFrame(
    outlier_rows
).to_csv(
    os.path.join(
        OUT_DIR,
        "extreme_gyro_windows.csv"
    ),
    index=False
)


print("\n")
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    os.path.join(
        OUT_DIR,
        "extreme_gyro_windows.csv"
    )
)

print("\nSTEP 52 COMPLETE")
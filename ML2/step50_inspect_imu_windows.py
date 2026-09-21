import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Maverick\ML2"

DATA_DIR = os.path.join(
    BASE,
    "step38_avnet_data"
)

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

DDODO_DIR = os.path.join(
    BASE,
    "step45_ddodo_data"
)

OUT_DIR = os.path.join(
    BASE,
    "step50_window_inspection"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD NORMALIZED TEST INPUT
# ============================================================

# We need the normalized input used by the model.
# However, for physical interpretation we also need the
# original unnormalized IMU values.

normalized = np.load(
    os.path.join(
        BASE,
        "step44_normalized_data",
        "test.npz"
    )
)

X_normalized = normalized["X"].astype(
    np.float32
)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(
    os.path.join(
        BASE,
        "step43_normalization_statistics.npz"
    )
)

means = stats["mean"].astype(
    np.float32
)

stds = stats["std"].astype(
    np.float32
)


# Recover original physical IMU values

X_raw = (
    X_normalized * stds
    + means
)


# ============================================================
# LOAD TARGETS
# ============================================================

att_data = np.load(
    os.path.join(
        BASE,
        "step44_normalized_data",
        "test.npz"
    )
)

Y_att = att_data["Y"].astype(
    np.float32
)

odo_data = np.load(
    os.path.join(
        DDODO_DIR,
        "test.npz"
    )
)

Y_odo = odo_data["Y"].astype(
    np.float32
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

split = pd.read_csv(
    os.path.join(
        BASE,
        "step38_avnet_data",
        "sequence_split.csv"
    )
)

test_sequences = split.loc[
    split["split"].str.upper() == "TEST",
    "dataset"
].tolist()


test_metadata = metadata[
    metadata["dataset"].isin(
        test_sequences
    )
].copy()

test_metadata = test_metadata.sort_values(
    "sample_index"
).reset_index(drop=True)


if len(test_metadata) != len(X_raw):

    raise RuntimeError(
        "Metadata and test input size mismatch."
    )


# ============================================================
# CONVERT YAW REFERENCE
# ============================================================

reference_yaw_deg = (
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

reference_velocity = Y_odo[:, 0]


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("STEP 50 - IMU WINDOW INSPECTION")
print("=" * 70)

print("\nInput shape:")
print(X_raw.shape)

print("\nChannels:")
print("0 = Accelerometer X")
print("1 = Accelerometer Y")
print("2 = Accelerometer Z")
print("3 = Gyroscope X")
print("4 = Gyroscope Y")
print("5 = Gyroscope Z")


# ============================================================
# WINDOW SUMMARY FUNCTION
# ============================================================

def window_summary(index):

    X = X_raw[index]

    summary = {}

    channel_names = [
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ"
    ]

    for c, name in enumerate(channel_names):

        values = X[:, c]

        summary[
            name + "_mean"
        ] = np.mean(values)

        summary[
            name + "_std"
        ] = np.std(values)

        summary[
            name + "_min"
        ] = np.min(values)

        summary[
            name + "_max"
        ] = np.max(values)

        summary[
            name + "_range"
        ] = (
            np.max(values)
            - np.min(values)
        )

    summary[
        "yaw_change_deg"
    ] = reference_yaw_deg[index]

    summary[
        "velocity_kmh"
    ] = reference_velocity[index]

    summary[
        "dataset"
    ] = test_metadata.iloc[index]["dataset"]

    summary[
        "sample_index"
    ] = test_metadata.iloc[index][
        "sample_index"
    ]

    return summary


# ============================================================
# SELECT REPRESENTATIVE WINDOWS
# ============================================================

selected = []


# ------------------------------------------------------------
# 1. Nearly stationary / very low yaw
# ------------------------------------------------------------

mask = (
    np.abs(reference_yaw_deg) < 1
) & (
    reference_velocity < 5
)

indices = np.where(mask)[0]

if len(indices) > 0:

    selected.append(
        ("LOW_SPEED_LOW_TURN", indices[0])
    )


# ------------------------------------------------------------
# 2. Straight driving around 40-60 km/h
# ------------------------------------------------------------

mask = (
    np.abs(reference_yaw_deg) < 1
) & (
    reference_velocity >= 40
) & (
    reference_velocity < 60
)

indices = np.where(mask)[0]

if len(indices) > 0:

    selected.append(
        ("STRAIGHT_40_60", indices[0])
    )


# ------------------------------------------------------------
# 3. Moderate turn
# ------------------------------------------------------------

mask = (
    np.abs(reference_yaw_deg) >= 5
) & (
    np.abs(reference_yaw_deg) < 10
)

indices = np.where(mask)[0]

if len(indices) > 0:

    selected.append(
        ("MODERATE_TURN_5_10", indices[0])
    )


# ------------------------------------------------------------
# 4. Strong turn
# ------------------------------------------------------------

mask = (
    np.abs(reference_yaw_deg) >= 10
) & (
    np.abs(reference_yaw_deg) < 20
)

indices = np.where(mask)[0]

if len(indices) > 0:

    selected.append(
        ("STRONG_TURN_10_20", indices[0])
    )


# ------------------------------------------------------------
# 5. Very strong turn
# ------------------------------------------------------------

mask = (
    np.abs(reference_yaw_deg) >= 20
)

indices = np.where(mask)[0]

if len(indices) > 0:

    selected.append(
        ("VERY_STRONG_TURN_20_PLUS", indices[0])
    )


# ============================================================
# PRINT SELECTED WINDOWS
# ============================================================

all_rows = []

for label, index in selected:

    X = X_raw[index]

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        "Dataset:",
        test_metadata.iloc[index]["dataset"]
    )

    print(
        "Sample index:",
        test_metadata.iloc[index][
            "sample_index"
        ]
    )

    print(
        f"Reference velocity: "
        f"{reference_velocity[index]:.3f} km/h"
    )

    print(
        f"Reference yaw change: "
        f"{reference_yaw_deg[index]:.3f} deg"
    )

    print("\n10-sample IMU window:")

    print(
        "Sample | "
        "AccX | AccY | AccZ | "
        "GyroX | GyroY | GyroZ"
    )

    for i in range(10):

        print(
            f"{i+1:6d} | "
            f"{X[i,0]:8.4f} | "
            f"{X[i,1]:8.4f} | "
            f"{X[i,2]:8.4f} | "
            f"{X[i,3]:8.5f} | "
            f"{X[i,4]:8.5f} | "
            f"{X[i,5]:8.5f}"
        )

    print("\nStatistics:")

    for c, name in enumerate([
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ"
    ]):

        values = X[:, c]

        print(
            f"{name:6s}: "
            f"mean={np.mean(values):9.5f}, "
            f"std={np.std(values):9.5f}, "
            f"range="
            f"{np.max(values)-np.min(values):9.5f}"
        )

    row = window_summary(index)
    row["category"] = label

    all_rows.append(row)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    all_rows
)

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "selected_window_summary.csv"
    ),
    index=False
)


# ============================================================
# FIND MOST EXTREME TURN WINDOWS
# ============================================================

print("\n")
print("=" * 70)
print("LARGEST YAW-CHANGE WINDOWS")
print("=" * 70)

largest_indices = np.argsort(
    np.abs(reference_yaw_deg)
)[-10:][::-1]

for rank, index in enumerate(
    largest_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"dataset="
        f"{test_metadata.iloc[index]['dataset']}, "
        f"sample="
        f"{test_metadata.iloc[index]['sample_index']}, "
        f"yaw="
        f"{reference_yaw_deg[index]:.3f} deg, "
        f"velocity="
        f"{reference_velocity[index]:.3f} km/h"
    )


# ============================================================
# FIND MOST EXTREME GYRO WINDOWS
# ============================================================

print("\n")
print("=" * 70)
print("LARGEST GYROSCOPE-MAGNITUDE WINDOWS")
print("=" * 70)

gyro_magnitude = np.sqrt(
    X_raw[:, :, 3] ** 2
    + X_raw[:, :, 4] ** 2
    + X_raw[:, :, 5] ** 2
)

gyro_window_max = np.max(
    gyro_magnitude,
    axis=1
)

largest_gyro_indices = np.argsort(
    gyro_window_max
)[-10:][::-1]

for rank, index in enumerate(
    largest_gyro_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"dataset="
        f"{test_metadata.iloc[index]['dataset']}, "
        f"sample="
        f"{test_metadata.iloc[index]['sample_index']}, "
        f"max gyro magnitude="
        f"{gyro_window_max[index]:.5f}, "
        f"yaw="
        f"{reference_yaw_deg[index]:.3f} deg, "
        f"velocity="
        f"{reference_velocity[index]:.3f} km/h"
    )


print("\n")
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    os.path.join(
        OUT_DIR,
        "selected_window_summary.csv"
    )
)

print("\nSTEP 50 COMPLETE")
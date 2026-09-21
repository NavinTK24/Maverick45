from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 18
# Build AVNet dataset from:
#
#   windows/
#       -> 10 x 6 smartphone IMU windows
#
#   targets_yaw_only/
#       -> DDATT + DDODO targets
#
# No normalization
# No shuffling
# No train/test split
# No outlier removal
#
# ============================================================

ML2 = Path(r"D:\Maverick\ML2")

WINDOW_DIR = ML2 / "windows"
TARGET_DIR = ML2 / "targets_yaw_only"

OUTPUT_FILE = ML2 / "avnet_yaw_only_dataset.npz"

# ------------------------------------------------------------
# Find files
# ------------------------------------------------------------

window_files = sorted(
    WINDOW_DIR.glob("*_windows.npz")
)

target_files = sorted(
    TARGET_DIR.glob("*_targets_yaw_only.csv")
)

print("Window files :", len(window_files))
print("Target files :", len(target_files))

if len(window_files) != 72:
    print(
        f"WARNING: expected 72 window files, "
        f"found {len(window_files)}"
    )

if len(target_files) != 72:
    print(
        f"WARNING: expected 72 target files, "
        f"found {len(target_files)}"
    )


# ------------------------------------------------------------
# Create lookup by dataset name
# ------------------------------------------------------------

window_map = {
    f.name.replace("_windows.npz", ""): f
    for f in window_files
}

target_map = {
    f.name.replace("_targets_yaw_only.csv", ""): f
    for f in target_files
}

keys = sorted(
    set(window_map.keys()) &
    set(target_map.keys())
)

print("Matched datasets:", len(keys))


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

X_all = []
Y_all = []

dataset_names = []
window_indices_all = []

total_windows = 0


# ------------------------------------------------------------
# Process every dataset
# ------------------------------------------------------------

for key in keys:

    window_file = window_map[key]
    target_file = target_map[key]

    # --------------------------------------------------------
    # Load IMU windows
    # --------------------------------------------------------

    data = np.load(window_file)

    X = data["X"]
    start_rows = data["start_row"]

    # Expected shape:
    # (N, 10, 6)

    if X.ndim != 3:
        raise ValueError(
            f"{key}: X must be 3D, got {X.shape}"
        )

    if X.shape[1:] != (10, 6):
        raise ValueError(
            f"{key}: expected X shape (N,10,6), "
            f"got {X.shape}"
        )

    # --------------------------------------------------------
    # Load targets
    # --------------------------------------------------------

    target = pd.read_csv(target_file)

    required = [
        "window_index",
        "start_row",
        "end_row",
        "ddatt_qx",
        "ddatt_qy",
        "ddatt_qz",
        "ddodo_velocity_kmh",
    ]

    missing = [
        c for c in required
        if c not in target.columns
    ]

    if missing:
        raise ValueError(
            f"{key}: missing target columns {missing}"
        )

    # --------------------------------------------------------
    # Verify number of windows
    # --------------------------------------------------------

    if len(X) != len(target):
        raise ValueError(
            f"{key}: window/target count mismatch: "
            f"{len(X)} vs {len(target)}"
        )

    # --------------------------------------------------------
    # Verify window mapping
    # --------------------------------------------------------

    target_start_rows = target[
        "start_row"
    ].to_numpy()

    if not np.array_equal(
        start_rows,
        target_start_rows
    ):
        raise ValueError(
            f"{key}: start_row mapping mismatch"
        )

    # --------------------------------------------------------
    # Construct target
    #
    # Y =
    # [dq_x, dq_y, dq_z, velocity]
    # --------------------------------------------------------

    Y = target[
        [
            "ddatt_qx",
            "ddatt_qy",
            "ddatt_qz",
            "ddodo_velocity_kmh",
        ]
    ].to_numpy(dtype=np.float64)

    X = X.astype(np.float32)

    # --------------------------------------------------------
    # Numeric validity
    # --------------------------------------------------------

    if not np.isfinite(X).all():
        raise ValueError(
            f"{key}: X contains NaN or infinity"
        )

    if not np.isfinite(Y).all():
        raise ValueError(
            f"{key}: Y contains NaN or infinity"
        )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    X_all.append(X)
    Y_all.append(Y)

    dataset_names.extend(
        [key] * len(X)
    )

    window_indices_all.extend(
        target["window_index"].tolist()
    )

    total_windows += len(X)

    print(
        f"[OK] {key}: "
        f"X={X.shape}, "
        f"Y={Y.shape}"
    )


# ------------------------------------------------------------
# Combine all datasets
# ------------------------------------------------------------

X_all = np.concatenate(
    X_all,
    axis=0
)

Y_all = np.concatenate(
    Y_all,
    axis=0
)

dataset_names = np.asarray(
    dataset_names
)

window_indices_all = np.asarray(
    window_indices_all,
    dtype=np.int64
)


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL DATASET VALIDATION")
print("============================================================")

print("X shape :", X_all.shape)
print("Y shape :", Y_all.shape)

print(
    "Expected X shape: "
    "(107043, 10, 6)"
)

print(
    "Expected Y shape: "
    "(107043, 4)"
)

if X_all.shape != (107043, 10, 6):
    raise ValueError(
        f"Unexpected X shape: {X_all.shape}"
    )

if Y_all.shape != (107043, 4):
    raise ValueError(
        f"Unexpected Y shape: {Y_all.shape}"
    )

if len(dataset_names) != 107043:
    raise ValueError(
        "Dataset-name count mismatch"
    )

if not np.isfinite(X_all).all():
    raise ValueError(
        "Final X contains invalid values"
    )

if not np.isfinite(Y_all).all():
    raise ValueError(
        "Final Y contains invalid values"
    )


# ------------------------------------------------------------
# Print target statistics
# ------------------------------------------------------------

print("\nDDATT TARGET")
print("----------------------------")

print(
    "dq_x:",
    Y_all[:, 0].min(),
    "to",
    Y_all[:, 0].max()
)

print(
    "dq_y:",
    Y_all[:, 1].min(),
    "to",
    Y_all[:, 1].max()
)

print(
    "dq_z:",
    Y_all[:, 2].min(),
    "to",
    Y_all[:, 2].max()
)

print("\nDDODO TARGET")
print("----------------------------")

print(
    "Velocity:",
    Y_all[:, 3].min(),
    "to",
    Y_all[:, 3].max(),
    "km/h"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.savez_compressed(
    OUTPUT_FILE,
    X=X_all,
    Y=Y_all,
    dataset_names=dataset_names,
    window_indices=window_indices_all,
)

print("\n============================================================")
print("STEP 18 COMPLETE")
print("============================================================")

print(
    f"Saved:\n{OUTPUT_FILE}"
)

print(
    "\nX = smartphone IMU windows "
    "(Accelerometer XYZ + Gyroscope XYZ)"
)

print(
    "X shape = (107043, 10, 6)"
)

print(
    "\nY = [DDATT qx, DDATT qy, DDATT qz, DDODO velocity]"
)

print(
    "Y shape = (107043, 4)"
)

print(
    "\nNo normalization, "
    "shuffling, filtering, or splitting was performed."
)

print("============================================================")
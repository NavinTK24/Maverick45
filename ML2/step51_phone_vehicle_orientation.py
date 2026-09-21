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

ATT_FILE = os.path.join(
    NORMALIZED_DIR,
    "test.npz"
)

ODO_FILE = os.path.join(
    BASE,
    "step45_ddodo_data",
    "test.npz"
)

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

SPLIT_FILE = os.path.join(
    BASE,
    "step38_avnet_data",
    "sequence_split.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step51_orientation_diagnostics"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(ATT_FILE)

X_normalized = data["X"].astype(
    np.float32
)

Y_att = data["Y"].astype(
    np.float32
)

odo = np.load(ODO_FILE)

Y_odo = odo["Y"].astype(
    np.float32
)


# ============================================================
# RESTORE RAW VALUES
# ============================================================

stats = np.load(STATS_FILE)

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


test_metadata = metadata[
    metadata["dataset"].isin(
        test_sequences
    )
].copy()

test_metadata = test_metadata.sort_values(
    "sample_index"
).reset_index(drop=True)


if len(test_metadata) != len(X):

    raise RuntimeError(
        "Metadata and test data size mismatch."
    )


datasets = (
    test_metadata["dataset"]
    .astype(str)
    .to_numpy()
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

reference_velocity = Y_odo[:, 0]


# ============================================================
# CHANNELS
# ============================================================

# X shape:
#
# samples x 10 x 6
#
# 0 = Acc X
# 1 = Acc Y
# 2 = Acc Z
# 3 = Gyro X
# 4 = Gyro Y
# 5 = Gyro Z

acc = X[:, :, 0:3]
gyro = X[:, :, 3:6]


# ============================================================
# GRAVITY-BASED PHONE ORIENTATION
# ============================================================

# Gravity components are not directly part of the selected
# AVNet input.
#
# Here we estimate the average direction of gravity from the
# accelerometer during each window as a diagnostic only.
#
# Because accelerometer includes vehicle motion, this is only
# an approximate diagnostic and is NOT treated as ground truth.

gravity_estimate = np.mean(
    acc,
    axis=1
)

gravity_norm = np.linalg.norm(
    gravity_estimate,
    axis=1
)

gravity_x = gravity_estimate[:, 0]
gravity_y = gravity_estimate[:, 1]
gravity_z = gravity_estimate[:, 2]


# ============================================================
# GYRO WINDOW STATISTICS
# ============================================================

gyro_mean = np.mean(
    gyro,
    axis=1
)

gyro_std = np.std(
    gyro,
    axis=1
)

gyro_rms = np.sqrt(
    np.mean(
        gyro ** 2,
        axis=1
    )
)

gyro_magnitude = np.sqrt(
    np.sum(
        gyro ** 2,
        axis=2
    )
)

gyro_mag_mean = np.mean(
    gyro_magnitude,
    axis=1
)

gyro_mag_max = np.max(
    gyro_magnitude,
    axis=1
)


# ============================================================
# CORRELATION FUNCTION
# ============================================================

def safe_corr(a, b):

    if len(a) < 3:
        return np.nan

    if np.std(a) == 0:
        return np.nan

    if np.std(b) == 0:
        return np.nan

    return np.corrcoef(
        a,
        b
    )[0, 1]


# ============================================================
# PER-SEQUENCE ANALYSIS
# ============================================================

rows = []

print("=" * 70)
print("STEP 51 - PHONE / VEHICLE ORIENTATION DIAGNOSTICS")
print("=" * 70)

print("\nTest sequences:")

for sequence in test_sequences:
    print(" ", sequence)


for sequence in test_sequences:

    mask = (
        datasets == sequence
    )

    if mask.sum() < 3:
        continue

    yaw = reference_yaw[mask]

    gx = gyro_mean[mask, 0]
    gy = gyro_mean[mask, 1]
    gz = gyro_mean[mask, 2]

    # Correlation between gyro axis and reference yaw change

    corr_gx = safe_corr(
        gx,
        yaw
    )

    corr_gy = safe_corr(
        gy,
        yaw
    )

    corr_gz = safe_corr(
        gz,
        yaw
    )

    corr_gmag = safe_corr(
        gyro_mag_mean[mask],
        np.abs(yaw)
    )

    # Which gyro axis has strongest absolute correlation?

    correlations = np.array([
        abs(corr_gx)
        if not np.isnan(corr_gx)
        else -1,

        abs(corr_gy)
        if not np.isnan(corr_gy)
        else -1,

        abs(corr_gz)
        if not np.isnan(corr_gz)
        else -1
    ])

    dominant_axis = [
        "Gyro X",
        "Gyro Y",
        "Gyro Z"
    ][np.argmax(correlations)]


    rows.append({

        "dataset":
            sequence,

        "samples":
            mask.sum(),

        "mean_velocity_kmh":
            np.mean(
                reference_velocity[mask]
            ),

        "mean_abs_yaw_change_deg":
            np.mean(
                np.abs(yaw)
            ),

        "mean_gravity_x":
            np.mean(
                gravity_x[mask]
            ),

        "mean_gravity_y":
            np.mean(
                gravity_y[mask]
            ),

        "mean_gravity_z":
            np.mean(
                gravity_z[mask]
            ),

        "mean_gravity_norm":
            np.mean(
                gravity_norm[mask]
            ),

        "gyro_x_mean":
            np.mean(gx),

        "gyro_y_mean":
            np.mean(gy),

        "gyro_z_mean":
            np.mean(gz),

        "gyro_x_std":
            np.mean(
                gyro_std[mask, 0]
            ),

        "gyro_y_std":
            np.mean(
                gyro_std[mask, 1]
            ),

        "gyro_z_std":
            np.mean(
                gyro_std[mask, 2]
            ),

        "gyro_mag_mean":
            np.mean(
                gyro_mag_mean[mask]
            ),

        "gyro_mag_max":
            np.max(
                gyro_mag_max[mask]
            ),

        "corr_gyro_x_yaw":
            corr_gx,

        "corr_gyro_y_yaw":
            corr_gy,

        "corr_gyro_z_yaw":
            corr_gz,

        "corr_gyro_magnitude_abs_yaw":
            corr_gmag,

        "dominant_gyro_axis":
            dominant_axis
    })


results = pd.DataFrame(
    rows
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PER-SEQUENCE ORIENTATION / GYRO RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}"
    )
)


# ============================================================
# GLOBAL CORRELATIONS
# ============================================================

print("\n")
print("=" * 70)
print("GLOBAL GYRO / YAW CORRELATION")
print("=" * 70)

global_gx = safe_corr(
    gyro_mean[:, 0],
    reference_yaw
)

global_gy = safe_corr(
    gyro_mean[:, 1],
    reference_yaw
)

global_gz = safe_corr(
    gyro_mean[:, 2],
    reference_yaw
)

global_gmag = safe_corr(
    gyro_mag_mean,
    np.abs(reference_yaw)
)

print(
    f"Gyro X vs yaw change: "
    f"{global_gx:.6f}"
)

print(
    f"Gyro Y vs yaw change: "
    f"{global_gy:.6f}"
)

print(
    f"Gyro Z vs yaw change: "
    f"{global_gz:.6f}"
)

print(
    f"Gyro magnitude vs |yaw|: "
    f"{global_gmag:.6f}"
)


# ============================================================
# DOMINANT AXIS COUNT
# ============================================================

print("\n")
print("=" * 70)
print("DOMINANT GYRO AXIS BY SEQUENCE")
print("=" * 70)

print(
    results[
        [
            "dataset",
            "dominant_gyro_axis",
            "corr_gyro_x_yaw",
            "corr_gyro_y_yaw",
            "corr_gyro_z_yaw"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}"
    )
)


# ============================================================
# GRAVITY SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("GRAVITY-DIRECTION SUMMARY")
print("=" * 70)

print(
    results[
        [
            "dataset",
            "mean_gravity_x",
            "mean_gravity_y",
            "mean_gravity_z",
            "mean_gravity_norm"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.5f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    os.path.join(
        OUT_DIR,
        "sequence_orientation_diagnostics.csv"
    ),
    index=False
)


# ============================================================
# SAVE WINDOW LEVEL DATA
# ============================================================

window_results = pd.DataFrame({

    "sample_index":
        test_metadata[
            "sample_index"
        ].to_numpy(),

    "dataset":
        datasets,

    "velocity_kmh":
        reference_velocity,

    "yaw_change_deg":
        reference_yaw,

    "gravity_x":
        gravity_x,

    "gravity_y":
        gravity_y,

    "gravity_z":
        gravity_z,

    "gravity_norm":
        gravity_norm,

    "gyro_x_mean":
        gyro_mean[:, 0],

    "gyro_y_mean":
        gyro_mean[:, 1],

    "gyro_z_mean":
        gyro_mean[:, 2],

    "gyro_x_std":
        gyro_std[:, 0],

    "gyro_y_std":
        gyro_std[:, 1],

    "gyro_z_std":
        gyro_std[:, 2],

    "gyro_magnitude_mean":
        gyro_mag_mean,

    "gyro_magnitude_max":
        gyro_mag_max
})

window_results.to_csv(
    os.path.join(
        OUT_DIR,
        "window_orientation_diagnostics.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    os.path.join(
        OUT_DIR,
        "sequence_orientation_diagnostics.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "window_orientation_diagnostics.csv"
    )
)

print("\nSTEP 51 COMPLETE")
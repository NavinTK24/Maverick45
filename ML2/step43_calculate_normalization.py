import numpy as np
from pathlib import Path


print("=" * 75)
print("STEP 43 - CALCULATE TRAINING-SET NORMALIZATION STATISTICS")
print("=" * 75)


# ============================================================
# PATH
# ============================================================

BASE = Path(r"D:\Maverick\ML2\step38_avnet_data")

train_file = BASE / "train.npz"


# ============================================================
# LOAD TRAINING DATA ONLY
# ============================================================

data = np.load(train_file)

X_train = data["X"]

print("\nTraining input shape:")
print(X_train.shape)


# ============================================================
# SENSOR NAMES
# ============================================================

sensor_names = [
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
    "Gyroscope X",
    "Gyroscope Y",
    "Gyroscope Z"
]


# ============================================================
# CALCULATE MEAN AND STD
# ============================================================

# X shape:
# samples × 10 time samples × 6 channels
#
# We calculate one mean/std for each sensor channel
# across ALL training samples and ALL 10 time points.

flat = X_train.reshape(-1, 6)

mean = np.mean(flat, axis=0)
std = np.std(flat, axis=0)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("TRAINING-SET NORMALIZATION PARAMETERS")
print("=" * 75)

print("\nFormula:")
print("normalized_value = (value - training_mean) / training_std")

print("\n")

for i, name in enumerate(sensor_names):

    print(
        f"{name:20s} : "
        f"mean = {mean[i]: .9f}    "
        f"std = {std[i]: .9f}"
    )


# ============================================================
# CHECK RESULT OF NORMALIZATION
# ============================================================

X_normalized = (
    X_train - mean.reshape(1, 1, 6)
) / std.reshape(1, 1, 6)


flat_normalized = X_normalized.reshape(-1, 6)

normalized_mean = np.mean(
    flat_normalized,
    axis=0
)

normalized_std = np.std(
    flat_normalized,
    axis=0
)


print("\n" + "=" * 75)
print("NORMALIZATION CHECK")
print("=" * 75)

for i, name in enumerate(sensor_names):

    print(
        f"{name:20s} : "
        f"mean = {normalized_mean[i]: .9f}    "
        f"std = {normalized_std[i]: .9f}"
    )


# ============================================================
# CHECK FOR ZERO STD
# ============================================================

print("\n" + "=" * 75)
print("ZERO-STANDARD-DEVIATION CHECK")
print("=" * 75)

for i, name in enumerate(sensor_names):

    if std[i] == 0:
        print(f"WARNING: {name} has zero standard deviation")
    else:
        print(f"{name}: OK")


# ============================================================
# SAVE STATISTICS
# ============================================================

output_file = Path(
    r"D:\Maverick\ML2\step43_normalization_statistics.npz"
)

np.savez(
    output_file,
    mean=mean.astype(np.float32),
    std=std.astype(np.float32)
)

print("\nSaved:")
print(output_file)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 43 COMPLETE")
print("=" * 75)

print("\nNo original data was modified.")
print("Validation/test data was NOT used.")
print("Only training-set statistics were calculated.")
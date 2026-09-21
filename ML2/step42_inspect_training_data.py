import numpy as np
from pathlib import Path


print("=" * 75)
print("STEP 42 - INSPECT FINAL AVNET TRAINING DATA")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\Maverick\ML2\step38_avnet_data")

train_file = BASE / "train.npz"
val_file = BASE / "validation.npz"
test_file = BASE / "test.npz"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = np.load(train_file)
val = np.load(val_file)
test = np.load(test_file)

X_train = train["X"]
Y_train = train["Y"]

X_val = val["X"]
Y_val = val["Y"]

X_test = test["X"]
Y_test = test["Y"]


# ============================================================
# SHAPES
# ============================================================

print("\n" + "=" * 75)
print("DATASET SHAPES")
print("=" * 75)

print(f"TRAIN:")
print(f"  X = {X_train.shape}")
print(f"  Y = {Y_train.shape}")

print(f"\nVALIDATION:")
print(f"  X = {X_val.shape}")
print(f"  Y = {Y_val.shape}")

print(f"\nTEST:")
print(f"  X = {X_test.shape}")
print(f"  Y = {Y_test.shape}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 75)
print("DATA TYPES")
print("=" * 75)

print("X train:", X_train.dtype)
print("Y train:", Y_train.dtype)

print("X validation:", X_val.dtype)
print("Y validation:", Y_val.dtype)

print("X test:", X_test.dtype)
print("Y test:", Y_test.dtype)


# ============================================================
# NaN / INF CHECK
# ============================================================

print("\n" + "=" * 75)
print("NUMERICAL VALIDITY")
print("=" * 75)

for name, X, Y in [
    ("TRAIN", X_train, Y_train),
    ("VALIDATION", X_val, Y_val),
    ("TEST", X_test, Y_test)
]:

    print(f"\n{name}")

    print("  X NaN :", np.isnan(X).sum())
    print("  X Inf :", np.isinf(X).sum())
    print("  Y NaN :", np.isnan(Y).sum())
    print("  Y Inf :", np.isinf(Y).sum())


# ============================================================
# INPUT STATISTICS
# ============================================================

sensor_names = [
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
    "Gyroscope X",
    "Gyroscope Y",
    "Gyroscope Z"
]


def print_input_statistics(name, X):

    print("\n" + "=" * 75)
    print(f"{name} INPUT STATISTICS")
    print("=" * 75)

    # Flatten samples while retaining sensor channel
    flat = X.reshape(-1, 6)

    for i, sensor in enumerate(sensor_names):

        values = flat[:, i]

        print(
            f"{sensor:20s} : "
            f"min={np.min(values):12.6f}  "
            f"max={np.max(values):12.6f}  "
            f"mean={np.mean(values):12.6f}  "
            f"std={np.std(values):12.6f}  "
            f"median={np.median(values):12.6f}"
        )


print_input_statistics("TRAIN", X_train)
print_input_statistics("VALIDATION", X_val)
print_input_statistics("TEST", X_test)


# ============================================================
# TARGET STATISTICS
# ============================================================

target_names = [
    "DDATT qx",
    "DDATT qy",
    "DDATT qz"
]


def print_attitude_statistics(name, Y):

    print("\n" + "=" * 75)
    print(f"{name} DDATT TARGET STATISTICS")
    print("=" * 75)

    for i, target in enumerate(target_names):

        values = Y[:, i]

        print(
            f"{target:12s} : "
            f"min={np.min(values):12.6f}  "
            f"max={np.max(values):12.6f}  "
            f"mean={np.mean(values):12.6f}  "
            f"std={np.std(values):12.6f}  "
            f"median={np.median(values):12.6f}"
        )


print_attitude_statistics("TRAIN", Y_train)
print_attitude_statistics("VALIDATION", Y_val)
print_attitude_statistics("TEST", Y_test)


# ============================================================
# QUATERNION TARGET CHECK
# ============================================================

print("\n" + "=" * 75)
print("DDATT QUATERNION CHECK")
print("=" * 75)

for name, Y in [
    ("TRAIN", Y_train),
    ("VALIDATION", Y_val),
    ("TEST", Y_test)
]:

    qx = Y[:, 0]
    qy = Y[:, 1]
    qz = Y[:, 2]

    # Our yaw-only quaternion:
    #
    # q = [qw, qx, qy, qz]
    #
    # qw = sqrt(1 - qz²)
    #
    # because qx=qy=0

    qw = np.sqrt(
        np.maximum(
            0.0,
            1.0 - qx**2 - qy**2 - qz**2
        )
    )

    norm = np.sqrt(
        qw**2 +
        qx**2 +
        qy**2 +
        qz**2
    )

    print(f"\n{name}")
    print("  Maximum quaternion norm error:",
          np.max(np.abs(norm - 1.0)))

    print("  qx maximum absolute value:",
          np.max(np.abs(qx)))

    print("  qy maximum absolute value:",
          np.max(np.abs(qy)))


# ============================================================
# SAMPLE TARGETS
# ============================================================

print("\n" + "=" * 75)
print("FIRST 10 TRAINING TARGETS")
print("=" * 75)

for i in range(min(10, len(Y_train))):

    print(
        f"{i:4d} : "
        f"qx={Y_train[i,0]: .8f}   "
        f"qy={Y_train[i,1]: .8f}   "
        f"qz={Y_train[i,2]: .8f}"
    )


# ============================================================
# VELOCITY
#
# The current Y contains DDATT only.
#
# DDODO velocity is therefore not present in these files.
# This is important and will be handled in the next preparation
# step rather than assuming a velocity target exists here.
# ============================================================

print("\n" + "=" * 75)
print("DDODO CHECK")
print("=" * 75)

print(
    "The Step 38 Y arrays contain the 3-value DDATT target "
    "[qx, qy, qz]."
)

print(
    "Vehicle velocity for DDODO must therefore be loaded "
    "separately from the original target/reference data."
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 42 COMPLETE")
print("=" * 75)

print("\nNo normalization was performed.")
print("No samples were modified.")
print("No training was performed.")
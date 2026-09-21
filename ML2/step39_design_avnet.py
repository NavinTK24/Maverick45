import torch
import torch.nn as nn


print("=" * 75)
print("STEP 39 - AVNET ARCHITECTURE DESIGN")
print("=" * 75)

# ============================================================
# YOUR DATA
# ============================================================

BATCH = 4
WINDOW = 10
CHANNELS = 6

print("\nINPUT")
print("-" * 75)
print(f"Input shape: ({BATCH}, {WINDOW}, {CHANNELS})")
print("Meaning: batch × 10 time samples × 6 IMU channels")
print("Channels:")
print("  1. Accelerometer X")
print("  2. Accelerometer Y")
print("  3. Accelerometer Z")
print("  4. Gyroscope X")
print("  5. Gyroscope Y")
print("  6. Gyroscope Z")


# ============================================================
# IMPORTANT:
# PyTorch Conv1D expects:
# batch × channels × sequence_length
# ============================================================

x = torch.randn(BATCH, CHANNELS, WINDOW)

print("\nPyTorch Conv1D representation")
print("-" * 75)
print(f"Shape: {tuple(x.shape)}")


# ============================================================
# CNN ARCHITECTURE
#
# We cannot directly use the paper's:
# Conv kernel 11 -> kernel 9
#
# because our input has only 10 samples.
#
# Therefore we test:
#
# Conv1D 6 -> 128, kernel 3
# MaxPool 2
# Conv1D 128 -> 256, kernel 2
# MaxPool 2
# ============================================================

conv1 = nn.Conv1d(
    in_channels=6,
    out_channels=128,
    kernel_size=3
)

relu1 = nn.ReLU()

pool1 = nn.MaxPool1d(kernel_size=2)

conv2 = nn.Conv1d(
    in_channels=128,
    out_channels=256,
    kernel_size=2
)

relu2 = nn.ReLU()

pool2 = nn.MaxPool1d(kernel_size=2)


print("\nCNN ARCHITECTURE")
print("-" * 75)

x = conv1(x)
print(f"Conv1D 6 -> 128, kernel=3 : {tuple(x.shape)}")

x = relu1(x)
print(f"ReLU                     : {tuple(x.shape)}")

x = pool1(x)
print(f"MaxPool kernel=2         : {tuple(x.shape)}")

x = conv2(x)
print(f"Conv1D 128 -> 256, k=2   : {tuple(x.shape)}")

x = relu2(x)
print(f"ReLU                     : {tuple(x.shape)}")

x = pool2(x)
print(f"MaxPool kernel=2         : {tuple(x.shape)}")


# ============================================================
# FLATTEN
# ============================================================

flatten_size = x.shape[1] * x.shape[2]

print("\nFLATTEN")
print("-" * 75)
print(f"Flatten size: {flatten_size}")

x = torch.flatten(x, start_dim=1)

print(f"Flattened shape: {tuple(x.shape)}")


# ============================================================
# FULLY CONNECTED LAYERS
# ============================================================

fc1 = nn.Linear(flatten_size, 1024)
fc2 = nn.Linear(1024, 512)

x = fc1(x)

print("\nFULLY CONNECTED")
print("-" * 75)
print(f"FC {flatten_size} -> 1024 : {tuple(x.shape)}")

x = fc2(x)

print(f"FC 1024 -> 512           : {tuple(x.shape)}")


# ============================================================
# GRU
#
# IMPORTANT:
# We are NOT deciding the GRU time dimension yet.
#
# This section only demonstrates the feature size.
# ============================================================

print("\nGRU")
print("-" * 75)
print("Feature size entering GRU: 512")
print("Hidden size proposed for paper-compatible implementation: 512")

print("\nIMPORTANT")
print("-" * 75)
print("The paper's figure shows the CNN/FC features being passed")
print("to a GRU, but the exact GRU sequence construction is not")
print("explicitly specified in the paper text available to us.")

print("\nTherefore:")
print("1. CNN dimensions can be fixed now.")
print("2. FC dimensions can be fixed now.")
print("3. GRU sequence construction must be decided separately.")
print("4. Training must NOT start until this is decided.")


# ============================================================
# TWO OUTPUT NETWORKS
# ============================================================

print("\nOUTPUTS")
print("-" * 75)

print("DDATT:")
print("  Output = 3")
print("  [qx, qy, qz]")
print("  For our yaw-only target:")
print("      qx = 0")
print("      qy = 0")
print("      qz = sin(delta_yaw / 2)")

print("\nDDODO:")
print("  Output = 1")
print("  Vehicle velocity (km/h)")


print("\n" + "=" * 75)
print("STEP 39 DESIGN CHECK COMPLETE")
print("=" * 75)
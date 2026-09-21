import numpy as np
from pathlib import Path


print("=" * 75)
print("STEP 44 - CREATE NORMALIZED AVNET DATA")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\Maverick\ML2\step38_avnet_data")

STATS_FILE = Path(
    r"D:\Maverick\ML2\step43_normalization_statistics.npz"
)

OUTPUT = Path(
    r"D:\Maverick\ML2\step44_normalized_data"
)

OUTPUT.mkdir(exist_ok=True)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

stats = np.load(STATS_FILE)

mean = stats["mean"]
std = stats["std"]

print("\nNormalization statistics loaded.")

print("Mean:")
print(mean)

print("\nStd:")
print(std)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(X):

    return (
        X - mean.reshape(1, 1, 6)
    ) / std.reshape(1, 1, 6)


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

splits = [
    ("train", "train.npz"),
    ("validation", "validation.npz"),
    ("test", "test.npz")
]


for split_name, filename in splits:

    print("\n" + "=" * 75)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 75)

    source = BASE / filename

    data = np.load(source)

    X = data["X"]
    Y = data["Y"]

    print("Original X:", X.shape)
    print("Original Y:", Y.shape)

    # Normalize ONLY X
    X_norm = normalize(X).astype(np.float32)

    # Keep targets unchanged
    Y = Y.astype(np.float32)

    print("Normalized X:", X_norm.shape)
    print("Target Y:", Y.shape)

    # Numerical check
    print(
        "X NaN:",
        np.isnan(X_norm).sum(),
        "X Inf:",
        np.isinf(X_norm).sum()
    )

    print(
        "Y NaN:",
        np.isnan(Y).sum(),
        "Y Inf:",
        np.isinf(Y).sum()
    )

    output_file = OUTPUT / filename

    np.savez(
        output_file,
        X=X_norm,
        Y=Y
    )

    print("Saved:", output_file)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 75)
print("FINAL NORMALIZATION CHECK")
print("=" * 75)


for split_name, filename in splits:

    data = np.load(OUTPUT / filename)

    X = data["X"]

    flat = X.reshape(-1, 6)

    print(f"\n{split_name.upper()}")

    for i in range(6):

        print(
            f"Channel {i}: "
            f"mean={np.mean(flat[:, i]): .6f}, "
            f"std={np.std(flat[:, i]): .6f}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 44 COMPLETE")
print("=" * 75)

print("\nNormalized files created in:")
print(OUTPUT)

print("\nOriginal Step 38 data was NOT modified.")
print("Only IMU inputs X were normalized.")
print("DDATT targets Y were kept unchanged.")
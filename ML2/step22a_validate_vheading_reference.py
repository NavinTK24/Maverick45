import os
import glob
import numpy as np
import pandas as pd

TARGET_DIR = r"D:\Maverick\ML2\targets_vheading"

files = sorted(
    glob.glob(
        os.path.join(
            TARGET_DIR,
            "*_targets_vheading.csv"
        )
    )
)

print("=" * 75)
print("STEP 22A - VALIDATE V-HEADING GROUND REFERENCE")
print("=" * 75)

print(f"Target files found: {len(files)}")

all_yaw = []
all_qz = []
all_qw = []
all_velocity = []

total_rows = 0
invalid_rows = 0
quaternion_errors = 0

# For identifying unusually large yaw changes
large_examples = []


# =========================================================
# Read all target files
# =========================================================

for path in files:

    key = os.path.basename(path).replace(
        "_targets_vheading.csv",
        ""
    )

    df = pd.read_csv(path)

    total_rows += len(df)

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------
    required = [
        "heading_start_deg",
        "heading_end_deg",
        "delta_yaw_deg",
        "dq_w",
        "ddatt_qx",
        "ddatt_qy",
        "ddatt_qz",
        "ddodo_velocity_kmh"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        print(
            f"ERROR {key}: missing columns {missing}"
        )

        continue


    # -----------------------------------------------------
    # Convert numeric
    # -----------------------------------------------------
    yaw = pd.to_numeric(
        df["delta_yaw_deg"],
        errors="coerce"
    ).to_numpy()

    qw = pd.to_numeric(
        df["dq_w"],
        errors="coerce"
    ).to_numpy()

    qx = pd.to_numeric(
        df["ddatt_qx"],
        errors="coerce"
    ).to_numpy()

    qy = pd.to_numeric(
        df["ddatt_qy"],
        errors="coerce"
    ).to_numpy()

    qz = pd.to_numeric(
        df["ddatt_qz"],
        errors="coerce"
    ).to_numpy()

    velocity = pd.to_numeric(
        df["ddodo_velocity_kmh"],
        errors="coerce"
    ).to_numpy()


    # -----------------------------------------------------
    # Invalid values
    # -----------------------------------------------------
    valid = (
        np.isfinite(yaw)
        & np.isfinite(qw)
        & np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(qz)
        & np.isfinite(velocity)
    )

    invalid_rows += np.sum(~valid)


    # -----------------------------------------------------
    # Quaternion norm
    # -----------------------------------------------------
    qnorm = np.sqrt(
        qw**2
        + qx**2
        + qy**2
        + qz**2
    )

    quaternion_error = np.abs(
        qnorm - 1.0
    )

    quaternion_errors += np.sum(
        quaternion_error > 1e-5
    )


    # -----------------------------------------------------
    # Collect
    # -----------------------------------------------------
    all_yaw.extend(
        yaw[valid]
    )

    all_qw.extend(
        qw[valid]
    )

    all_qz.extend(
        qz[valid]
    )

    all_velocity.extend(
        velocity[valid]
    )


    # -----------------------------------------------------
    # Large yaw examples
    # -----------------------------------------------------
    abs_yaw = np.abs(yaw)

    idx = np.where(
        valid & (abs_yaw >= 90.0)
    )[0]

    for i in idx[:10]:

        large_examples.append({
            "dataset": key,
            "window": int(
                df.iloc[i]["window_index"]
            ),
            "heading_start": float(
                df.iloc[i]["heading_start_deg"]
            ),
            "heading_end": float(
                df.iloc[i]["heading_end_deg"]
            ),
            "delta_yaw": float(
                df.iloc[i]["delta_yaw_deg"]
            ),
            "qz": float(
                df.iloc[i]["ddatt_qz"]
            )
        })


# =========================================================
# Convert to arrays
# =========================================================

all_yaw = np.asarray(all_yaw)

all_qw = np.asarray(all_qw)

all_qz = np.asarray(all_qz)

all_velocity = np.asarray(all_velocity)


# =========================================================
# Statistics helper
# =========================================================

def stats(x):

    return {
        "min": np.min(x),
        "max": np.max(x),
        "mean": np.mean(x),
        "median": np.median(x),
        "p90": np.percentile(x, 90),
        "p95": np.percentile(x, 95),
        "p99": np.percentile(x, 99),
        "p99.9": np.percentile(x, 99.9)
    }


# =========================================================
# Print yaw statistics
# =========================================================

print()
print("-" * 75)
print("DATASET CHECK")
print("-" * 75)

print(
    f"Total target rows: {total_rows}"
)

print(
    f"Invalid numeric rows: {invalid_rows}"
)

print(
    f"Quaternion norm errors (>1e-5): "
    f"{quaternion_errors}"
)


print()
print("-" * 75)
print("DELTA YAW STATISTICS")
print("-" * 75)

s = stats(all_yaw)

for k, v in s.items():

    print(
        f"{k:8s}: {v:.6f} deg"
    )


# =========================================================
# Absolute yaw thresholds
# =========================================================

print()
print("-" * 75)
print("ABSOLUTE ΔYAW THRESHOLDS")
print("-" * 75)

for threshold in [
    1,
    2,
    5,
    10,
    20,
    45,
    90,
    120,
    150
]:

    count = np.sum(
        np.abs(all_yaw) >= threshold
    )

    percentage = (
        count / len(all_yaw) * 100
    )

    print(
        f">= {threshold:3d}° : "
        f"{count:6d} "
        f"({percentage:.4f}%)"
    )


# =========================================================
# qz statistics
# =========================================================

print()
print("-" * 75)
print("QZ STATISTICS")
print("-" * 75)

s = stats(all_qz)

for k, v in s.items():

    print(
        f"{k:8s}: {v:.8f}"
    )


# =========================================================
# Velocity statistics
# =========================================================

print()
print("-" * 75)
print("VELOCITY STATISTICS")
print("-" * 75)

s = stats(all_velocity)

for k, v in s.items():

    print(
        f"{k:8s}: {v:.6f} km/h"
    )


# =========================================================
# Large yaw examples
# =========================================================

print()
print("-" * 75)
print("EXAMPLES WITH |ΔYAW| >= 90°")
print("-" * 75)

if len(large_examples) == 0:

    print("None found.")

else:

    for x in large_examples[:20]:

        print(
            f"{x['dataset']:8s} "
            f"window={x['window']:5d} | "
            f"heading "
            f"{x['heading_start']:9.3f} -> "
            f"{x['heading_end']:9.3f} | "
            f"Δyaw={x['delta_yaw']:9.3f}° | "
            f"qz={x['qz']: .6f}"
        )


# =========================================================
# Final checks
# =========================================================

print()
print("=" * 75)
print("FINAL CHECK")
print("=" * 75)

checks_passed = True


if len(files) != 72:

    print(
        f"FAIL: expected 72 target files, "
        f"found {len(files)}"
    )

    checks_passed = False


if total_rows != 107043:

    print(
        f"FAIL: expected 107043 rows, "
        f"found {total_rows}"
    )

    checks_passed = False


if invalid_rows != 0:

    print(
        "FAIL: invalid numeric values found"
    )

    checks_passed = False


if quaternion_errors != 0:

    print(
        "FAIL: quaternion norm errors found"
    )

    checks_passed = False


# Circular yaw should be within [-180, 180]
if np.min(all_yaw) < -180.000001:

    print("FAIL: yaw below -180°")

    checks_passed = False


if np.max(all_yaw) > 180.000001:

    print("FAIL: yaw above +180°")

    checks_passed = False


if checks_passed:

    print("ALL BASIC CHECKS PASSED")

else:

    print("SOME CHECKS FAILED")


print("=" * 75)
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# STEP 17
# Verify YAW-ONLY DDATT targets
# ============================================================

ML2 = Path(r"D:\Maverick\ML2")
TARGET_DIR = ML2 / "targets_yaw_only"

files = sorted(TARGET_DIR.glob("*_targets_yaw_only.csv"))

if not files:
    raise FileNotFoundError(
        f"No yaw-only target files found in {TARGET_DIR}"
    )

total_rows = 0
invalid_rows = 0
norm_errors = 0
qx_errors = 0
qy_errors = 0
yaw_errors = 0
velocity_errors = 0

all_delta_yaw = []

for src in files:

    df = pd.read_csv(src)

    required = [
        "window_index",
        "start_row",
        "end_row",
        "yaw_start_deg",
        "yaw_end_deg",
        "delta_yaw_deg",
        "dq_w",
        "dq_x",
        "dq_y",
        "dq_z",
        "ddatt_qx",
        "ddatt_qy",
        "ddatt_qz",
        "ddodo_velocity_kmh",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[ERROR] {src.name}: missing {missing}")
        invalid_rows += 1
        continue

    n = len(df)
    total_rows += n

    # --------------------------------------------------------
    # Numeric values
    # --------------------------------------------------------

    qw = pd.to_numeric(df["dq_w"], errors="coerce").to_numpy()
    qx = pd.to_numeric(df["dq_x"], errors="coerce").to_numpy()
    qy = pd.to_numeric(df["dq_y"], errors="coerce").to_numpy()
    qz = pd.to_numeric(df["dq_z"], errors="coerce").to_numpy()

    dyaw = pd.to_numeric(
        df["delta_yaw_deg"],
        errors="coerce"
    ).to_numpy()

    yaw_start = pd.to_numeric(
        df["yaw_start_deg"],
        errors="coerce"
    ).to_numpy()

    yaw_end = pd.to_numeric(
        df["yaw_end_deg"],
        errors="coerce"
    ).to_numpy()

    velocity = pd.to_numeric(
        df["ddodo_velocity_kmh"],
        errors="coerce"
    ).to_numpy()

    valid = (
        np.isfinite(qw)
        & np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(qz)
        & np.isfinite(dyaw)
        & np.isfinite(yaw_start)
        & np.isfinite(yaw_end)
        & np.isfinite(velocity)
    )

    invalid_rows += int((~valid).sum())

    # --------------------------------------------------------
    # 1. Quaternion norm
    # --------------------------------------------------------

    norm = np.sqrt(
        qw**2 +
        qx**2 +
        qy**2 +
        qz**2
    )

    norm_bad = np.abs(norm - 1.0) > 1e-10
    norm_errors += int(norm_bad.sum())

    # --------------------------------------------------------
    # 2. Pure yaw means qx = 0 and qy = 0
    # --------------------------------------------------------

    qx_bad = np.abs(qx) > 1e-12
    qy_bad = np.abs(qy) > 1e-12

    qx_errors += int(qx_bad.sum())
    qy_errors += int(qy_bad.sum())

    # --------------------------------------------------------
    # 3. Check quaternion against delta yaw
    #
    # Expected:
    #
    # qw = cos(delta_yaw / 2)
    # qz = sin(delta_yaw / 2)
    # --------------------------------------------------------

    expected_qw = np.cos(np.deg2rad(dyaw) / 2.0)
    expected_qz = np.sin(np.deg2rad(dyaw) / 2.0)

    yaw_bad = (
        (np.abs(qw - expected_qw) > 1e-10)
        |
        (np.abs(qz - expected_qz) > 1e-10)
    )

    yaw_errors += int(yaw_bad.sum())

    # --------------------------------------------------------
    # 4. Check DDATT copies
    # --------------------------------------------------------

    ddatt_qx = pd.to_numeric(
        df["ddatt_qx"],
        errors="coerce"
    ).to_numpy()

    ddatt_qy = pd.to_numeric(
        df["ddatt_qy"],
        errors="coerce"
    ).to_numpy()

    ddatt_qz = pd.to_numeric(
        df["ddatt_qz"],
        errors="coerce"
    ).to_numpy()

    if (
        np.any(np.abs(ddatt_qx - qx) > 1e-12)
        or
        np.any(np.abs(ddatt_qy - qy) > 1e-12)
        or
        np.any(np.abs(ddatt_qz - qz) > 1e-12)
    ):
        print(f"[ERROR] DDATT mismatch in {src.name}")

    # --------------------------------------------------------
    # 5. Check yaw difference independently
    # --------------------------------------------------------

    independent_dyaw = (
        (yaw_end - yaw_start + 180.0)
        % 360.0
        - 180.0
    )

    yaw_difference_bad = (
        np.abs(independent_dyaw - dyaw) > 1e-10
    )

    yaw_errors += int(yaw_difference_bad.sum())

    # --------------------------------------------------------
    # 6. Check velocity is finite
    # --------------------------------------------------------

    velocity_errors += int(
        np.sum(~np.isfinite(velocity))
    )

    all_delta_yaw.extend(dyaw[valid])

    print(
        f"[OK] {src.name}: "
        f"{n:,} rows | "
        f"Δyaw min={np.min(dyaw):.6f}° | "
        f"max={np.max(dyaw):.6f}°"
    )


# ============================================================
# SUMMARY
# ============================================================

all_delta_yaw = np.asarray(all_delta_yaw)

print("\n============================================================")
print("STEP 17 COMPLETE — YAW-ONLY TARGET VERIFICATION")
print("============================================================")

print(f"Files checked          : {len(files)}")
print(f"Total rows             : {total_rows:,}")
print(f"Invalid numeric rows   : {invalid_rows:,}")
print(f"Quaternion norm errors : {norm_errors:,}")
print(f"qx != 0 errors         : {qx_errors:,}")
print(f"qy != 0 errors         : {qy_errors:,}")
print(f"Quaternion/yaw errors  : {yaw_errors:,}")
print(f"Velocity errors        : {velocity_errors:,}")

if len(all_delta_yaw) > 0:

    print("\nΔYAW STATISTICS")
    print("----------------------------")
    print(f"Minimum : {np.min(all_delta_yaw):.6f}°")
    print(f"Maximum : {np.max(all_delta_yaw):.6f}°")
    print(f"Mean    : {np.mean(all_delta_yaw):.6f}°")
    print(f"Median  : {np.median(all_delta_yaw):.6f}°")
    print(f"P90     : {np.percentile(all_delta_yaw, 90):.6f}°")
    print(f"P95     : {np.percentile(all_delta_yaw, 95):.6f}°")
    print(f"P99     : {np.percentile(all_delta_yaw, 99):.6f}°")

    print("\nTHRESHOLDS")
    print("----------------------------")

    for threshold in [1, 2, 5, 10, 20, 45, 90, 120, 150]:

        count = np.sum(
            np.abs(all_delta_yaw) >= threshold
        )

        percentage = (
            count / len(all_delta_yaw) * 100
        )

        print(
            f"|Δyaw| >= {threshold:3d}° : "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

print("\n============================================================")

if (
    invalid_rows == 0
    and norm_errors == 0
    and qx_errors == 0
    and qy_errors == 0
    and yaw_errors == 0
    and velocity_errors == 0
):
    print("RESULT: ALL CHECKS PASSED")
else:
    print("RESULT: CHECKS FAILED — inspect errors above")

print("============================================================")
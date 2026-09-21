from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# STEP 16
# Create separate YAW-DOMINANT DDATT targets
#
# Existing full-3D targets are NOT modified.
#
# Experiment:
#   - Yaw is allowed to change.
#   - Pitch is treated as constant.
#   - Roll is treated as constant.
#
# Therefore:
#   attitude change = pure yaw rotation
#
# Quaternion order:
#   [qw, qx, qy, qz]
#
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ML2 = Path(r"D:\Maverick\ML2")

INPUT_DIR = ML2 / "targets_raw"
OUTPUT_DIR = ML2 / "targets_yaw_only"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# ACTUAL COLUMNS IN targets_raw
# ------------------------------------------------------------

required_columns = [
    "window_index",
    "start_row",
    "end_row",
    "yaw_start_deg",
    "pitch_start_deg",
    "roll_start_deg",
    "yaw_end_deg",
    "pitch_end_deg",
    "roll_end_deg",
    "velocity_target_kmh",
]


# ------------------------------------------------------------
# FIND ALL TARGET FILES
# ------------------------------------------------------------

files = sorted(INPUT_DIR.glob("*_targets_raw.csv"))

if not files:
    raise FileNotFoundError(
        f"No *_targets_raw.csv files found in:\n{INPUT_DIR}"
    )


# ------------------------------------------------------------
# COUNTERS
# ------------------------------------------------------------

total_windows = 0
total_invalid = 0
successful_files = 0


# ------------------------------------------------------------
# PROCESS EACH DATASET
# ------------------------------------------------------------

for src in files:

    print(f"\nProcessing: {src.name}")

    df = pd.read_csv(src)

    # Check required columns
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"[SKIP] Missing columns: {missing}")
        total_invalid += 1
        continue

    # --------------------------------------------------------
    # Read required values
    # --------------------------------------------------------

    yaw_start = pd.to_numeric(
        df["yaw_start_deg"],
        errors="coerce"
    ).to_numpy()

    yaw_end = pd.to_numeric(
        df["yaw_end_deg"],
        errors="coerce"
    ).to_numpy()

    velocity = pd.to_numeric(
        df["velocity_target_kmh"],
        errors="coerce"
    ).to_numpy()

    # Pitch and roll are retained only for analysis/traceability.
    pitch_start = pd.to_numeric(
        df["pitch_start_deg"],
        errors="coerce"
    ).to_numpy()

    pitch_end = pd.to_numeric(
        df["pitch_end_deg"],
        errors="coerce"
    ).to_numpy()

    roll_start = pd.to_numeric(
        df["roll_start_deg"],
        errors="coerce"
    ).to_numpy()

    roll_end = pd.to_numeric(
        df["roll_end_deg"],
        errors="coerce"
    ).to_numpy()


    # --------------------------------------------------------
    # VALIDITY CHECK
    # --------------------------------------------------------

    valid = (
        np.isfinite(yaw_start)
        & np.isfinite(yaw_end)
        & np.isfinite(velocity)
        & np.isfinite(pitch_start)
        & np.isfinite(pitch_end)
        & np.isfinite(roll_start)
        & np.isfinite(roll_end)
    )

    invalid_count = int((~valid).sum())

    if invalid_count > 0:
        print(f"  Invalid rows: {invalid_count}")

    total_invalid += invalid_count


    # --------------------------------------------------------
    # CIRCULAR YAW DIFFERENCE
    #
    # Example:
    #
    # 359° -> 1°
    #
    # Normal subtraction:
    #     1 - 359 = -358°
    #
    # Correct circular difference:
    #     +2°
    #
    # Result range:
    #     [-180°, +180°)
    # --------------------------------------------------------

    delta_yaw_deg = (
        (yaw_end - yaw_start + 180.0)
        % 360.0
        - 180.0
    )


    # --------------------------------------------------------
    # CONVERT YAW CHANGE TO RADIANS
    # --------------------------------------------------------

    delta_yaw_rad = np.deg2rad(delta_yaw_deg)


    # --------------------------------------------------------
    # PURE-YAW QUATERNION
    #
    # Quaternion:
    #
    #   dq = [
    #       cos(theta/2),
    #       0,
    #       0,
    #       sin(theta/2)
    #   ]
    #
    # where theta = delta yaw.
    #
    # [qw, qx, qy, qz]
    # --------------------------------------------------------

    half_angle = delta_yaw_rad / 2.0

    dq_w = np.cos(half_angle)
    dq_x = np.zeros_like(dq_w)
    dq_y = np.zeros_like(dq_w)
    dq_z = np.sin(half_angle)


    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    output = pd.DataFrame({

        # Window identification
        "window_index": df["window_index"],
        "start_row": df["start_row"],
        "end_row": df["end_row"],

        # Original yaw values
        "yaw_start_deg": yaw_start,
        "yaw_end_deg": yaw_end,

        # Our new yaw-only target
        "delta_yaw_deg": delta_yaw_deg,

        # Retain pitch/roll for inspection
        # They DO NOT contribute to dq.
        "pitch_start_deg": pitch_start,
        "pitch_end_deg": pitch_end,
        "roll_start_deg": roll_start,
        "roll_end_deg": roll_end,

        # Pure-yaw quaternion
        "dq_w": dq_w,
        "dq_x": dq_x,
        "dq_y": dq_y,
        "dq_z": dq_z,

        # DDATT target
        "ddatt_qx": dq_x,
        "ddatt_qy": dq_y,
        "ddatt_qz": dq_z,

        # DDODO target remains unchanged
        "ddodo_velocity_kmh": velocity,
    })


    # --------------------------------------------------------
    # OUTPUT FILE NAME
    # --------------------------------------------------------

    stem = src.name.replace(
        "_targets_raw.csv",
        ""
    )

    dst = OUTPUT_DIR / (
        f"{stem}_targets_yaw_only.csv"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output.to_csv(
        dst,
        index=False
    )


    successful_files += 1
    total_windows += len(output)

    print(
        f"  [OK] {len(output):,} windows"
    )
    print(
        f"  Saved: {dst.name}"
    )


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("STEP 16 COMPLETE")
print("============================================================")

print(f"Input files processed : {successful_files}")
print(f"Total target windows  : {total_windows:,}")
print(f"Invalid rows          : {total_invalid:,}")

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nOriginal full-3D targets were NOT modified.")

print("\nYaw-dominant DDATT:")
print("  Yaw     -> allowed to change")
print("  Pitch   -> treated as constant")
print("  Roll    -> treated as constant")

print("\nQuaternion:")
print("  [qw, qx, qy, qz]")

print("============================================================")
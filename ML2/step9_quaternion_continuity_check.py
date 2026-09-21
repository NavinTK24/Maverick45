import os
import glob
import numpy as np
import pandas as pd

# ================================================================
# MAVeriCK ML2 - STEP 9
# CHECK QUATERNION CONTINUITY FROM ORIGINAL PHONE ORIENTATION
#
# Purpose:
#   Use the original phone Yaw/Pitch/Roll sequence and examine
#   whether the large attitude changes seen in Step 7 are mainly
#   caused by Euler-angle representation/wrap-around/singularity.
#
# IMPORTANT:
#   This script DOES NOT modify windows or target files.
#   It only creates an analysis CSV.
# ================================================================

DATASET_ROOT = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
OUTPUT_CSV = r"D:\Maverick\ML2\step9_quaternion_continuity_analysis.csv"

# Actual 1-based columns confirmed in Step 2/8
YAW_COL = 22
PITCH_COL = 23
ROLL_COL = 24

def euler_zyx_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    """
    ZYX convention:
        yaw   -> Z
        pitch -> Y
        roll  -> X

    Returns quaternion [qw, qx, qy, qz].
    This is the same convention used when Step 5 generated
    the current DDATT targets.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.column_stack((qw, qx, qy, qz))
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    return q

def quaternion_angle_deg(q0, q1):
    """
    Rotation angle between two unit quaternions.
    Uses |dot| because q and -q represent the same rotation.
    """
    dots = np.sum(q0 * q1, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))

def circular_difference_deg(a, b):
    """Small signed difference b-a in [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0

print("=" * 72)
print("MAVeriCK ML2 - STEP 9: QUATERNION CONTINUITY CHECK")
print("=" * 72)

s_files = sorted(
    glob.glob(os.path.join(DATASET_ROOT, "**", "S-*.csv"), recursive=True)
)

print(f"Smartphone files: {len(s_files)}")
print("Sampling rate: 10 Hz")
print()

records = []

all_angles = []
all_qdot_abs = []
all_near_180 = 0
all_raw_yaw_wrap = 0
all_raw_roll_wrap = 0

for s_path in s_files:
    pair = os.path.splitext(os.path.basename(s_path))[0][2:]

    try:
        df = pd.read_csv(s_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(s_path, encoding="latin1", low_memory=False)

    if df.shape[1] < ROLL_COL:
        print(f"SKIP {pair}: insufficient columns")
        continue

    euler = df.iloc[:, [YAW_COL-1, PITCH_COL-1, ROLL_COL-1]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.all(np.isfinite(euler), axis=1)
    euler = euler[valid]

    if len(euler) < 2:
        print(f"SKIP {pair}: insufficient valid orientation rows")
        continue

    q = euler_zyx_to_quaternion(euler[:,0], euler[:,1], euler[:,2])

    q0 = q[:-1]
    q1 = q[1:]

    dots = np.sum(q0 * q1, axis=1)
    qdot_abs = np.clip(np.abs(dots), 0.0, 1.0)
    angles = np.rad2deg(2.0 * np.arccos(qdot_abs))

    yaw_raw = np.abs(np.diff(euler[:,0]))
    roll_raw = np.abs(np.diff(euler[:,2]))

    yaw_circ = np.abs(circular_difference_deg(euler[:-1,0], euler[1:,0]))
    roll_circ = np.abs(circular_difference_deg(euler[:-1,2], euler[1:,2]))

    raw_yaw_wrap = int(np.sum(yaw_raw > 180.0))
    raw_roll_wrap = int(np.sum(roll_raw > 180.0))
    near_180 = int(np.sum(angles >= 170.0))

    all_angles.extend(angles.tolist())
    all_qdot_abs.extend(qdot_abs.tolist())
    all_near_180 += near_180
    all_raw_yaw_wrap += raw_yaw_wrap
    all_raw_roll_wrap += raw_roll_wrap

    records.append({
        "pair": pair,
        "samples": len(euler),
        "median_quaternion_change_deg": float(np.percentile(angles, 50)),
        "p95_quaternion_change_deg": float(np.percentile(angles, 95)),
        "p99_quaternion_change_deg": float(np.percentile(angles, 99)),
        "max_quaternion_change_deg": float(np.max(angles)),
        "changes_ge_45deg": int(np.sum(angles >= 45.0)),
        "changes_ge_90deg": int(np.sum(angles >= 90.0)),
        "changes_ge_120deg": int(np.sum(angles >= 120.0)),
        "changes_ge_150deg": int(np.sum(angles >= 150.0)),
        "changes_ge_170deg": near_180,
        "raw_yaw_wrap_gt_180": raw_yaw_wrap,
        "raw_roll_wrap_gt_180": raw_roll_wrap,
        "median_abs_yaw_circular_deg": float(np.percentile(yaw_circ, 50)),
        "p95_abs_yaw_circular_deg": float(np.percentile(yaw_circ, 95)),
        "median_abs_roll_circular_deg": float(np.percentile(roll_circ, 50)),
        "p95_abs_roll_circular_deg": float(np.percentile(roll_circ, 95)),
    })

all_angles = np.asarray(all_angles, dtype=float)
all_qdot_abs = np.asarray(all_qdot_abs, dtype=float)

print("1. QUATERNION-BASED CONSECUTIVE ROTATION CHANGE")
print("-" * 72)
for p in [0, 50, 90, 95, 99, 99.9, 100]:
    print(f"{p:5.1f} percentile : {np.percentile(all_angles, p):.6f} deg")

print()
print(f"Change >= 45 deg  : {np.sum(all_angles >= 45):8d} "
      f"({100*np.mean(all_angles >= 45):.4f}%)")
print(f"Change >= 90 deg  : {np.sum(all_angles >= 90):8d} "
      f"({100*np.mean(all_angles >= 90):.4f}%)")
print(f"Change >= 120 deg : {np.sum(all_angles >= 120):8d} "
      f"({100*np.mean(all_angles >= 120):.4f}%)")
print(f"Change >= 150 deg : {np.sum(all_angles >= 150):8d} "
      f"({100*np.mean(all_angles >= 150):.4f}%)")
print(f"Change >= 170 deg : {all_near_180:8d} "
      f"({100*all_near_180/len(all_angles):.4f}%)")

print()
print("2. QUATERNION SIGN CONTINUITY")
print("-" * 72)
print("Quaternion dot products are checked using |dot(q_t,q_t+1)|.")
print(f"Minimum |dot| : {np.min(all_qdot_abs):.9f}")
print(f"1st percentile: {np.percentile(all_qdot_abs, 1):.9f}")
print(f"Median        : {np.percentile(all_qdot_abs, 50):.9f}")
print(f"5th percentile: {np.percentile(all_qdot_abs, 5):.9f}")

print()
print("3. COMPARISON WITH EULER WRAP INDICATORS")
print("-" * 72)
print(f"Yaw raw jumps >180 deg  : {all_raw_yaw_wrap}")
print(f"Roll raw jumps >180 deg : {all_raw_roll_wrap}")
print()
print("A raw 360-degree Euler jump can correspond to a small quaternion")
print("rotation. This comparison lets us separate representation jumps")
print("from genuinely large 3-D rotations.")

out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_CSV, index=False)

print()
print("=" * 72)
print("STEP 9 COMPLETE")
print("=" * 72)
print(f"Pair analysis saved: {OUTPUT_CSV}")
print("No source, window, or target files were modified.")

import os
import glob
import numpy as np
import pandas as pd

# ================================================================
# MAVeriCK ML2 - STEP 10
# IMU vs ORIGINAL PHONE ORIENTATION
#
# Purpose:
#   For each synchronized smartphone sequence, compare the
#   gyroscope activity with the attitude change obtained from the
#   original phone Yaw/Pitch/Roll.
#
#   This is an ANALYSIS ONLY step.
#   It does not modify windows or target files.
# ================================================================

DATASET_ROOT = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
OUTPUT_CSV = r"D:\Maverick\ML2\step10_imu_orientation_analysis.csv"

# Confirmed 1-based columns
ACC_COLS = [10, 11, 12]
GYRO_COLS = [16, 17, 18]
ORIENT_COLS = [22, 23, 24]

FS = 10.0

def euler_zyx_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    q = np.column_stack((qw, qx, qy, qz))
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q

def quaternion_angle_deg(q0, q1):
    dot = np.sum(q0 * q1, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.rad2deg(2*np.arccos(dot))

print("=" * 72)
print("MAVeriCK ML2 - STEP 10: IMU vs ORIGINAL PHONE ORIENTATION")
print("=" * 72)

s_files = sorted(
    glob.glob(os.path.join(DATASET_ROOT, "**", "S-*.csv"), recursive=True)
)

print(f"Smartphone files: {len(s_files)}")
print(f"Sampling rate: {FS:g} Hz")
print()

records = []

all_gyro_mag = []
all_acc_mag = []
all_rot_change = []

# Windows are 1 second = 10 samples.
# For each complete 10-sample window:
#   gyro_activity = RMS gyro magnitude
#   accel_activity = RMS deviation of accel magnitude from its median
#   orientation_change = quaternion angle from first to last sample
#
# We intentionally do NOT call acceleration magnitude "vehicle
# acceleration"; it is only smartphone accelerometer activity.

for s_path in s_files:
    pair = os.path.splitext(os.path.basename(s_path))[0][2:]

    try:
        df = pd.read_csv(s_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(s_path, encoding="latin1", low_memory=False)

    acc = df.iloc[:, [c-1 for c in ACC_COLS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    gyro = df.iloc[:, [c-1 for c in GYRO_COLS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    orient = df.iloc[:, [c-1 for c in ORIENT_COLS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.all(np.isfinite(acc), axis=1) &
        np.all(np.isfinite(gyro), axis=1) &
        np.all(np.isfinite(orient), axis=1)
    )

    acc = acc[valid]
    gyro = gyro[valid]
    orient = orient[valid]

    n = (len(acc) // 10) * 10
    if n < 10:
        continue

    acc = acc[:n].reshape(-1, 10, 3)
    gyro = gyro[:n].reshape(-1, 10, 3)
    orient = orient[:n].reshape(-1, 10, 3)

    for i in range(len(acc)):
        # Gyroscope magnitude at each sample.
        gyro_mag = np.linalg.norm(gyro[i], axis=1)

        # Accelerometer magnitude.
        # This is NOT interpreted as vehicle linear acceleration.
        acc_mag = np.linalg.norm(acc[i], axis=1)

        # Remove the static level approximately by measuring variation
        # around the window median.
        acc_activity = np.sqrt(
            np.mean((acc_mag - np.median(acc_mag))**2)
        )

        gyro_rms = np.sqrt(np.mean(gyro_mag**2))

        q = euler_zyx_to_quaternion(
            orient[i,:,0],
            orient[i,:,1],
            orient[i,:,2]
        )

        # Orientation change over the complete 1-second window.
        rot_change = float(quaternion_angle_deg(q[0:1], q[-1:])[0])

        all_gyro_mag.extend(gyro_mag.tolist())
        all_acc_mag.extend(acc_mag.tolist())
        all_rot_change.append(rot_change)

        records.append({
            "pair": pair,
            "window_index": i,
            "gyro_rms_rad_s": gyro_rms,
            "gyro_max_mag_rad_s": float(np.max(gyro_mag)),
            "acc_mag_median_m_s2": float(np.median(acc_mag)),
            "acc_activity_rms_m_s2": float(acc_activity),
            "orientation_change_deg_1s": rot_change,
            "start_yaw_deg": float(orient[i,0,0]),
            "end_yaw_deg": float(orient[i,-1,0]),
            "start_pitch_deg": float(orient[i,0,1]),
            "end_pitch_deg": float(orient[i,-1,1]),
            "start_roll_deg": float(orient[i,0,2]),
            "end_roll_deg": float(orient[i,-1,2]),
        })

out = pd.DataFrame(records)

print("1. ALL 1-SECOND WINDOWS")
print("-" * 72)
print(f"Complete 1-second windows: {len(out)}")

for col, name in [
    ("gyro_rms_rad_s", "Gyro RMS"),
    ("acc_activity_rms_m_s2", "Accel activity RMS"),
    ("orientation_change_deg_1s", "Orientation change")
]:
    x = out[col].to_numpy()
    print(f"\n{name}:")
    for p in [50, 90, 95, 99, 99.9, 100]:
        print(f"  {p:5.1f} percentile : {np.percentile(x,p):.6f}")

print()
print("2. LARGE 1-SECOND ORIENTATION CHANGES")
print("-" * 72)

for threshold in [45, 90, 120, 150]:
    subset = out[out["orientation_change_deg_1s"] >= threshold]
    print(
        f">= {threshold:3d} deg : {len(subset):6d} "
        f"({100*len(subset)/len(out):.4f}%)"
    )

print()
print("3. WINDOWS WITH LARGE ORIENTATION CHANGE")
print("-" * 72)

large = out[out["orientation_change_deg_1s"] >= 90].copy()

if len(large):
    # Show the largest 20. This is an analysis table only.
    largest = large.sort_values(
        "orientation_change_deg_1s", ascending=False
    ).head(20)

    print(
        largest[
            [
                "pair",
                "window_index",
                "orientation_change_deg_1s",
                "gyro_rms_rad_s",
                "gyro_max_mag_rad_s",
                "start_yaw_deg",
                "end_yaw_deg",
                "start_pitch_deg",
                "end_pitch_deg",
                "start_roll_deg",
                "end_roll_deg",
            ]
        ].to_string(index=False)
    )
else:
    print("No >=90 degree 1-second orientation changes found.")

print()
print("4. CORRELATION CHECK")
print("-" * 72)

corr = out[
    ["gyro_rms_rad_s", "acc_activity_rms_m_s2", "orientation_change_deg_1s"]
].corr()

print(corr.to_string())

print()
print("Interpretation of this step:")
print("- Gyro RMS represents measured angular-rate activity.")
print("- Accel activity represents variation in smartphone acceleration magnitude.")
print("- Orientation change is computed from the original phone orientation.")
print("- No variable is being treated as vehicle linear acceleration here.")
print("- Correlation is descriptive only; it does not prove causation.")

out.to_csv(OUTPUT_CSV, index=False)

print()
print("=" * 72)
print("STEP 10 COMPLETE")
print("=" * 72)
print(f"Analysis saved: {OUTPUT_CSV}")
print("No source, window, or target files were modified.")

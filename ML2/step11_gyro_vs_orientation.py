import os
import glob
import numpy as np
import pandas as pd

# ================================================================
# MAVeriCK ML2 - STEP 11
# GYRO-INTEGRATED ROTATION vs PHONE-ORIENTATION ROTATION
#
# Purpose:
#   Compare the rotation implied by the raw smartphone gyroscope
#   with the 3-D rotation obtained from the original phone
#   Yaw/Pitch/Roll.
#
#   This is an analysis-only step.
#   It DOES NOT modify windows or target files.
#
# IMPORTANT:
#   The gyroscope integration below is intentionally simple:
#       theta = sum(gyro * dt)
#   and is used only as a diagnostic comparison.
#   It is NOT being declared to be the final paper's attitude
#   estimation method.
# ================================================================

DATASET_ROOT = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
OUTPUT_CSV = r"D:\Maverick\ML2\step11_gyro_vs_orientation.csv"

GYRO_COLS = [16, 17, 18]       # Yaw/Pitch/Roll gyro, rad/s
ORIENT_COLS = [22, 23, 24]     # Yaw/Pitch/Roll orientation, deg

FS = 10.0
DT = 1.0 / FS

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
    qz = cr*cp*cy - sr*sp*sy

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= max(np.linalg.norm(q), 1e-12)
    return q

def quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ])

def axis_angle_quaternion(axis, angle):
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = axis / n
    s = np.sin(angle/2.0)
    return np.array([
        np.cos(angle/2.0),
        u[0]*s, u[1]*s, u[2]*s
    ])

def gyro_integrated_quaternion(gyro_window):
    """
    Diagnostic integration of body angular velocity.

    For each 10 Hz sample:
        incremental angle = omega * dt
        incremental quaternion is formed from that axis-angle
        and accumulated.

    This is deliberately kept as a simple diagnostic and is not
    claimed to reproduce the paper's InEKF.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for omega in gyro_window:
        delta = omega * DT
        angle = np.linalg.norm(delta)
        dq = axis_angle_quaternion(delta, angle)
        q = quat_multiply(q, dq)
        q /= max(np.linalg.norm(q), 1e-12)

    return q

def rotation_angle_deg(q):
    q = q / max(np.linalg.norm(q), 1e-12)
    w = np.clip(abs(q[0]), 0.0, 1.0)
    return float(np.rad2deg(2*np.arccos(w)))

print("=" * 72)
print("MAVeriCK ML2 - STEP 11: GYRO vs PHONE ORIENTATION")
print("=" * 72)
print(f"Smartphone files: {len(glob.glob(os.path.join(DATASET_ROOT, '**', 'S-*.csv'), recursive=True))}")
print(f"Sampling rate: {FS:g} Hz")
print("Window: 1 second = 10 samples")
print()

records = []

s_files = sorted(
    glob.glob(os.path.join(DATASET_ROOT, "**", "S-*.csv"), recursive=True)
)

for s_path in s_files:
    pair = os.path.splitext(os.path.basename(s_path))[0][2:]

    try:
        df = pd.read_csv(s_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(s_path, encoding="latin1", low_memory=False)

    gyro = df.iloc[:, [c-1 for c in GYRO_COLS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    orient = df.iloc[:, [c-1 for c in ORIENT_COLS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.all(np.isfinite(gyro), axis=1) & np.all(np.isfinite(orient), axis=1)

    gyro = gyro[valid]
    orient = orient[valid]

    n = (len(gyro) // 10) * 10
    gyro = gyro[:n]
    orient = orient[:n]

    gyro_w = gyro.reshape(-1, 10, 3)
    orient_w = orient.reshape(-1, 10, 3)

    for i in range(len(gyro_w)):
        gw = gyro_w[i]
        ow = orient_w[i]

        # --------------------------------------------------------
        # A. Rotation implied by gyro integration
        # --------------------------------------------------------
        q_gyro = gyro_integrated_quaternion(gw)
        gyro_angle = rotation_angle_deg(q_gyro)

        # --------------------------------------------------------
        # B. Rotation obtained from original phone orientation
        # --------------------------------------------------------
        q_start = euler_zyx_to_quaternion(
            ow[0, 0], ow[0, 1], ow[0, 2]
        )
        q_end = euler_zyx_to_quaternion(
            ow[-1, 0], ow[-1, 1], ow[-1, 2]
        )

        q_orientation_change = quat_multiply(
            quat_inverse(q_start), q_end
        )
        q_orientation_change /= max(
            np.linalg.norm(q_orientation_change), 1e-12
        )

        orientation_angle = rotation_angle_deg(q_orientation_change)

        # Difference in magnitudes only.
        angle_difference = abs(orientation_angle - gyro_angle)

        gyro_rms = float(
            np.sqrt(np.mean(np.sum(gw**2, axis=1)))
        )

        records.append({
            "pair": pair,
            "window_index": i,
            "gyro_integrated_rotation_deg": gyro_angle,
            "orientation_rotation_deg": orientation_angle,
            "absolute_angle_difference_deg": angle_difference,
            "gyro_rms_rad_s": gyro_rms,
            "start_yaw_deg": ow[0,0],
            "end_yaw_deg": ow[-1,0],
            "start_pitch_deg": ow[0,1],
            "end_pitch_deg": ow[-1,1],
            "start_roll_deg": ow[0,2],
            "end_roll_deg": ow[-1,2],
        })

out = pd.DataFrame(records)

print("1. OVERALL ROTATION COMPARISON")
print("-" * 72)

for col, name in [
    ("gyro_integrated_rotation_deg", "Gyro-integrated rotation"),
    ("orientation_rotation_deg", "Orientation-derived rotation"),
    ("absolute_angle_difference_deg", "Absolute difference")
]:
    x = out[col].to_numpy()
    print(f"\n{name}:")
    for p in [50, 90, 95, 99, 99.9, 100]:
        print(f"  {p:5.1f} percentile : {np.percentile(x,p):.6f} deg")

print()
print("2. LARGE ORIENTATION ROTATIONS vs GYRO")
print("-" * 72)

for threshold in [45, 90, 120, 150]:
    subset = out[out["orientation_rotation_deg"] >= threshold]
    print(f"\nOrientation >= {threshold}°: {len(subset)} windows")

    if len(subset):
        print(
            f"  Median gyro-integrated rotation : "
            f"{np.median(subset['gyro_integrated_rotation_deg']):.6f}°"
        )
        print(
            f"  Median angle difference          : "
            f"{np.median(subset['absolute_angle_difference_deg']):.6f}°"
        )

print()
print("3. LARGEST DISAGREEMENTS")
print("-" * 72)

largest = out.sort_values(
    "absolute_angle_difference_deg", ascending=False
).head(20)

print(
    largest[
        [
            "pair",
            "window_index",
            "gyro_integrated_rotation_deg",
            "orientation_rotation_deg",
            "absolute_angle_difference_deg",
            "gyro_rms_rad_s",
            "start_yaw_deg",
            "end_yaw_deg",
            "start_pitch_deg",
            "end_pitch_deg",
            "start_roll_deg",
            "end_roll_deg",
        ]
    ].to_string(index=False)
)

print()
print("4. HOW OFTEN DO THEY AGREE ROUGHLY?")
print("-" * 72)

for tolerance in [5, 10, 20, 30, 45]:
    ok = out["absolute_angle_difference_deg"] <= tolerance
    print(
        f"Difference <= {tolerance:2d}° : "
        f"{ok.sum():7d} / {len(ok)} "
        f"({100*ok.mean():.4f}%)"
    )

print()
print("NOTE")
print("-" * 72)
print("This is a diagnostic comparison only.")
print("It does not claim that simple gyro integration is the paper's")
print("InEKF or final attitude estimator.")
print("No source, window, or target files were modified.")

out.to_csv(OUTPUT_CSV, index=False)

print()
print("=" * 72)
print("STEP 11 COMPLETE")
print("=" * 72)
print(f"Analysis saved: {OUTPUT_CSV}")

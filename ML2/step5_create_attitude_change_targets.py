from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# MAVeriCK ML2 - STEP 5
#
# Create DDATT attitude-change targets.
#
# IMPORTANT:
# We use the IO-VNBD phone Orientation values already selected by the user:
#   Yaw, Pitch, Roll (degrees)
#
# This script intentionally keeps the Euler-angle convention explicit.
# It uses the common aerospace sequence:
#       Z (yaw) -> Y (pitch) -> X (roll)
#
# Quaternion convention:
#       q = [qw, qx, qy, qz]
#
# For each 1-second window:
#       q_start = quaternion(yaw_start, pitch_start, roll_start)
#       q_end   = quaternion(yaw_end,   pitch_end,   roll_end)
#
# Relative attitude change:
#       dq = inverse(q_start) * q_end
#
# The AVNet DDATT target is:
#       [dq_x, dq_y, dq_z]
#
# DDODO remains:
#       velocity_target_kmh
#
# NOTE:
# This is a target-generation convention for our IO-VNBD recreation.
# The paper's own ground-truth attitude comes from NovAtel SPAN +
# ISA-100C IMU, rather than these phone Orientation columns.
# ---------------------------------------------------------------------------

WINDOW_SIZE = 10


def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def find_pairs(root):
    s_files, v_files = {}, {}

    for p in root.rglob("*.csv"):
        if p.name.startswith("S-"):
            s_files[p.stem[2:].lower()] = p
        elif p.name.startswith("V-"):
            v_files[p.stem[2:].lower()] = p

    keys = sorted(set(s_files) & set(v_files))
    return [(k, s_files[k], v_files[k]) for k in keys]


def euler_zyx_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    """Convert ZYX yaw/pitch/roll to [qw,qx,qy,qz]."""
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

    q = np.array([qw, qx, qy, qz], dtype=np.float64)

    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion.")

    return q / norm


def quaternion_inverse(q):
    """Inverse of a unit quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(q1, q2):
    """Hamilton product for [w,x,y,z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python step5_create_attitude_change_targets.py '
            '"PATH_TO_CATEGORISED_IOVNB_DATASET"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    pairs = find_pairs(root)

    if not pairs:
        raise RuntimeError("No matched S/V pairs found.")

    input_target_dir = Path(__file__).resolve().parent / "targets_raw"
    output_dir = Path(__file__).resolve().parent / "targets"

    if not input_target_dir.exists():
        raise RuntimeError(
            "targets_raw directory not found. Run Step 4 first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    invalid = 0

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 5: CREATE DDATT ATTITUDE-CHANGE TARGETS")
    print("=" * 72)
    print("Euler convention : ZYX (Yaw -> Pitch -> Roll)")
    print("Quaternion       : [qw, qx, qy, qz]")
    print("Relative change  : inverse(q_start) * q_end")
    print("DDATT target     : [dq_x, dq_y, dq_z]")
    print("DDODO target     : velocity_target_kmh")
    print()

    for key, _, _ in pairs:
        raw_path = input_target_dir / f"{key}_targets_raw.csv"

        if not raw_path.exists():
            print(f"{key:12s} | SKIPPED: raw target file missing")
            continue

        df = pd.read_csv(raw_path)

        rows = []

        for _, r in df.iterrows():
            start = np.array([
                r["yaw_start_deg"],
                r["pitch_start_deg"],
                r["roll_start_deg"],
            ], dtype=np.float64)

            end = np.array([
                r["yaw_end_deg"],
                r["pitch_end_deg"],
                r["roll_end_deg"],
            ], dtype=np.float64)

            if not np.isfinite(start).all() or not np.isfinite(end).all():
                invalid += 1
                continue

            q_start = euler_zyx_to_quaternion(*start)
            q_end = euler_zyx_to_quaternion(*end)

            dq = quaternion_multiply(
                quaternion_inverse(q_start),
                q_end
            )

            # Normalize numerical round-off.
            dq /= np.linalg.norm(dq)

            rows.append([
                int(r["window_index"]),
                int(r["start_row"]),
                int(r["end_row"]),
                dq[0],
                dq[1],
                dq[2],
                dq[3],
                dq[1],
                dq[2],
                dq[3],
                float(r["velocity_target_kmh"]),
            ])

        columns = [
            "window_index",
            "start_row",
            "end_row",
            "dq_w",
            "dq_x",
            "dq_y",
            "dq_z",
            "ddatt_qx",
            "ddatt_qy",
            "ddatt_qz",
            "ddodo_velocity_kmh",
        ]

        out = pd.DataFrame(rows, columns=columns)
        out_path = output_dir / f"{key}_targets.csv"
        out.to_csv(out_path, index=False)

        total += len(out)

        print(
            f"{key:12s} | targets={len(out):6d} "
            f"| {out_path.name}"
        )

    print()
    print("=" * 72)
    print("STEP 5 COMPLETE")
    print("=" * 72)
    print(f"Total target rows: {total}")
    print(f"Invalid rows skipped: {invalid}")
    print(f"Saved in: {output_dir}")
    print()
    print("Each target row contains:")
    print("  dq_w, dq_x, dq_y, dq_z  -> relative attitude change")
    print("  ddatt_qx, qy, qz        -> AVNet DDATT target")
    print("  ddodo_velocity_kmh      -> AVNet DDODO target")
    print()
    print("No neural-network training has been performed yet.")


if __name__ == "__main__":
    main()

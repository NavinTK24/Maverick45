from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MAVeriCK ML2 - STEP 6
#
# Verify the generated DDATT + DDODO targets.
#
# This is a READ/VALIDATION step only.
# It does not modify the target files.
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) != 1:
        # Optional argument is accepted so the script can be run consistently
        # with the previous steps, but the targets directory is derived from
        # this script's location.
        pass

    base = Path(__file__).resolve().parent
    windows_dir = base / "windows"
    targets_dir = base / "targets"

    if not windows_dir.exists():
        raise RuntimeError(f"Missing directory: {windows_dir}")

    if not targets_dir.exists():
        raise RuntimeError(f"Missing directory: {targets_dir}")

    target_files = sorted(targets_dir.glob("*_targets.csv"))

    if not target_files:
        raise RuntimeError("No target files found.")

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 6: VERIFY DDATT + DDODO TARGETS")
    print("=" * 72)
    print(f"Target files found: {len(target_files)}")
    print()

    required = [
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

    total = 0
    invalid = 0
    norm_errors = 0
    ddatt_mismatch = 0
    window_mismatch = 0

    # Collect global statistics.
    velocities = []
    dq_xyz = []
    dq_w_values = []

    first_example = None

    for target_path in target_files:
        key = target_path.name.replace("_targets.csv", "")
        window_path = windows_dir / f"{key}_windows.npz"

        if not window_path.exists():
            print(f"{key:12s} | ERROR: matching window file missing")
            window_mismatch += 1
            continue

        df = pd.read_csv(target_path)

        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"{key:12s} | ERROR: missing columns {missing}")
            invalid += len(df)
            continue

        with np.load(window_path) as data:
            X = data["X"]
            start_rows = data["start_row"]

        if len(df) != len(X):
            print(
                f"{key:12s} | ERROR: target/window count mismatch "
                f"targets={len(df)} windows={len(X)}"
            )
            window_mismatch += 1

        n = min(len(df), len(X))

        for i in range(n):
            r = df.iloc[i]

            vals = r[required].to_numpy(dtype=float)

            if not np.isfinite(vals).all():
                invalid += 1
                continue

            q = np.array([
                r["dq_w"],
                r["dq_x"],
                r["dq_y"],
                r["dq_z"],
            ], dtype=float)

            q_norm = np.linalg.norm(q)

            if not np.isfinite(q_norm) or abs(q_norm - 1.0) > 1e-5:
                norm_errors += 1

            if (
                abs(r["dq_x"] - r["ddatt_qx"]) > 1e-12
                or abs(r["dq_y"] - r["ddatt_qy"]) > 1e-12
                or abs(r["dq_z"] - r["ddatt_qz"]) > 1e-12
            ):
                ddatt_mismatch += 1

            if int(r["start_row"]) != int(start_rows[i]):
                window_mismatch += 1

            velocities.append(float(r["ddodo_velocity_kmh"]))
            dq_xyz.append([
                float(r["dq_x"]),
                float(r["dq_y"]),
                float(r["dq_z"]),
            ])
            dq_w_values.append(float(r["dq_w"]))

            total += 1

            if first_example is None:
                first_example = (key, r, X[i])

        print(
            f"{key:12s} | rows={len(df):6d} | "
            f"window_shape={X.shape[1:] if X.ndim == 3 and len(X) else 'N/A'}"
        )

    print()
    print("=" * 72)
    print("GLOBAL VALIDATION")
    print("=" * 72)

    print(f"Validated target rows : {total}")
    print(f"Invalid numeric rows  : {invalid}")
    print(f"Quaternion norm errors: {norm_errors}")
    print(f"DDATT copy mismatches : {ddatt_mismatch}")
    print(f"Window mapping errors : {window_mismatch}")

    if velocities:
        v = np.asarray(velocities, dtype=float)
        qxyz = np.asarray(dq_xyz, dtype=float)
        qw = np.asarray(dq_w_values, dtype=float)

        print()
        print("DDODO velocity statistics (km/hr)")
        print(f"  minimum : {v.min():.6f}")
        print(f"  maximum : {v.max():.6f}")
        print(f"  mean    : {v.mean():.6f}")
        print(f"  median  : {np.median(v):.6f}")

        print()
        print("DDATT quaternion-change statistics")
        print(
            f"  dq_w range : {qw.min():.8f} to {qw.max():.8f}"
        )
        print(
            f"  dq_x range : {qxyz[:,0].min():.8f} "
            f"to {qxyz[:,0].max():.8f}"
        )
        print(
            f"  dq_y range : {qxyz[:,1].min():.8f} "
            f"to {qxyz[:,1].max():.8f}"
        )
        print(
            f"  dq_z range : {qxyz[:,2].min():.8f} "
            f"to {qxyz[:,2].max():.8f}"
        )

    if first_example is not None:
        key, r, x = first_example

        print()
        print("=" * 72)
        print("FIRST WINDOW EXAMPLE")
        print("=" * 72)

        print(f"Pair              : {key}")
        print(f"Window index      : {int(r['window_index'])}")
        print(
            f"Rows              : {int(r['start_row'])} "
            f"to {int(r['end_row'])}"
        )

        print()
        print("IMU input X (10 x 6)")
        print("Columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z")

        for j, row in enumerate(x):
            print(
                f"{j:2d}: "
                f"{row[0]: .6f} {row[1]: .6f} {row[2]: .6f} "
                f"{row[3]: .6f} {row[4]: .6f} {row[5]: .6f}"
            )

        print()
        print("Relative quaternion")
        print(f"  dq_w = {r['dq_w']:.12f}")
        print(f"  dq_x = {r['dq_x']:.12f}")
        print(f"  dq_y = {r['dq_y']:.12f}")
        print(f"  dq_z = {r['dq_z']:.12f}")

        print()
        print("DDATT target")
        print(f"  qx = {r['ddatt_qx']:.12f}")
        print(f"  qy = {r['ddatt_qy']:.12f}")
        print(f"  qz = {r['ddatt_qz']:.12f}")

        print()
        print("DDODO target")
        print(f"  velocity = {r['ddodo_velocity_kmh']:.6f} km/hr")

    print()
    print("=" * 72)

    if (
        invalid == 0
        and norm_errors == 0
        and ddatt_mismatch == 0
        and window_mismatch == 0
    ):
        print("STEP 6 PASSED")
        print("All generated targets are numerically consistent.")
    else:
        print("STEP 6 FOUND ISSUES")
        print("Review the validation values above before training.")

    print("=" * 72)


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# MAVeriCK ML2 - STEP 7
#
# Analyze DDATT targets before neural-network training.
#
# This version reads:
#   targets/     -> quaternion + DDODO values
#   targets_raw/ -> original start/end Yaw/Pitch/Roll
#
# It does NOT modify either dataset.
# ---------------------------------------------------------------------------


def main():
    base = Path(__file__).resolve().parent
    targets_dir = base / "targets"
    raw_dir = base / "targets_raw"

    if not targets_dir.exists():
        raise RuntimeError(f"Missing directory: {targets_dir}")

    if not raw_dir.exists():
        raise RuntimeError(f"Missing directory: {raw_dir}")

    files = sorted(targets_dir.glob("*_targets.csv"))

    if not files:
        raise RuntimeError("No target files found.")

    frames = []

    required_target = [
        "window_index",
        "start_row",
        "end_row",
        "dq_w",
        "dq_x",
        "dq_y",
        "dq_z",
        "ddodo_velocity_kmh",
    ]

    required_raw = [
        "yaw_start_deg",
        "pitch_start_deg",
        "roll_start_deg",
        "yaw_end_deg",
        "pitch_end_deg",
        "roll_end_deg",
    ]

    for f in files:
        key = f.name.replace("_targets.csv", "")
        raw_path = raw_dir / f"{key}_targets_raw.csv"

        if not raw_path.exists():
            print(f"Skipping {key}: raw target file missing")
            continue

        target = pd.read_csv(f)
        raw = pd.read_csv(raw_path)

        missing_target = [c for c in required_target if c not in target.columns]
        missing_raw = [c for c in required_raw if c not in raw.columns]

        if missing_target or missing_raw:
            print(
                f"Skipping {key}: "
                f"missing target={missing_target}, raw={missing_raw}"
            )
            continue

        if len(target) != len(raw):
            print(
                f"Skipping {key}: target/raw row mismatch "
                f"{len(target)} vs {len(raw)}"
            )
            continue

        # Join by row position because Step 4 and Step 5 preserve the same
        # one-row-per-window ordering.
        combined = pd.concat(
            [
                target.reset_index(drop=True),
                raw[required_raw].reset_index(drop=True),
            ],
            axis=1,
        )

        combined["pair"] = key
        frames.append(combined)

    if not frames:
        raise RuntimeError("No usable target files.")

    data = pd.concat(frames, ignore_index=True)

    q_w = data["dq_w"].to_numpy(dtype=float)
    q_x = data["dq_x"].to_numpy(dtype=float)
    q_y = data["dq_y"].to_numpy(dtype=float)
    q_z = data["dq_z"].to_numpy(dtype=float)

    # Smallest equivalent quaternion rotation angle.
    theta_rad = 2.0 * np.arccos(
        np.clip(np.abs(q_w), 0.0, 1.0)
    )
    theta_deg = np.rad2deg(theta_rad)

    data["attitude_change_deg"] = theta_deg

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 7: DDATT TARGET ANALYSIS")
    print("=" * 72)
    print(f"Total target windows: {len(data)}")
    print()

    print("1. ATTITUDE-CHANGE ANGLE")
    print("-" * 72)

    percentiles = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]

    for p in percentiles:
        print(
            f"{p:5.1f} percentile : "
            f"{np.percentile(theta_deg, p):.6f} deg"
        )

    print()
    print("Counts by attitude-change magnitude:")

    bins = [0, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180]

    for lo, hi in zip(bins[:-1], bins[1:]):
        count = np.sum((theta_deg >= lo) & (theta_deg < hi))
        pct = 100.0 * count / len(theta_deg)
        print(
            f"  {lo:3d} <= angle < {hi:3d} deg : "
            f"{count:7d} ({pct:6.2f}%)"
        )

    print()
    count_45 = np.sum(theta_deg >= 45)
    count_90 = np.sum(theta_deg >= 90)

    print(
        f"  angle >= 45 deg : {count_45} "
        f"({100*count_45/len(theta_deg):.4f}%)"
    )
    print(
        f"  angle >= 90 deg : {count_90} "
        f"({100*count_90/len(theta_deg):.4f}%)"
    )

    print()
    print("2. DDATT COMPONENT STATISTICS")
    print("-" * 72)

    for name, values in [
        ("dq_x", q_x),
        ("dq_y", q_y),
        ("dq_z", q_z),
    ]:
        print(f"{name}:")
        print(f"  min    = {values.min(): .9f}")
        print(f"  max    = {values.max(): .9f}")
        print(f"  mean   = {values.mean(): .9f}")
        print(f"  median = {np.median(values): .9f}")
        print(f"  std    = {values.std(): .9f}")

    print()
    print("3. POSSIBLE YAW WRAP-AROUND CASES")
    print("-" * 72)

    yaw_start = data["yaw_start_deg"].to_numpy(dtype=float)
    yaw_end = data["yaw_end_deg"].to_numpy(dtype=float)

    raw_yaw_diff = np.abs(yaw_end - yaw_start)
    data["raw_yaw_difference_deg"] = raw_yaw_diff

    candidates = data[raw_yaw_diff > 180].copy()

    print(
        f"Cases with |yaw_end - yaw_start| > 180 deg: "
        f"{len(candidates)}"
    )

    if len(candidates):
        print()
        print("First 10 candidates:")
        cols = [
            "pair",
            "window_index",
            "yaw_start_deg",
            "yaw_end_deg",
            "pitch_start_deg",
            "pitch_end_deg",
            "roll_start_deg",
            "roll_end_deg",
            "attitude_change_deg",
        ]
        print(candidates[cols].head(10).to_string(index=False))

    print()
    print("4. LARGEST ATTITUDE-CHANGE WINDOWS")
    print("-" * 72)

    largest = data.nlargest(10, "attitude_change_deg")

    cols = [
        "pair",
        "window_index",
        "start_row",
        "end_row",
        "yaw_start_deg",
        "pitch_start_deg",
        "roll_start_deg",
        "yaw_end_deg",
        "pitch_end_deg",
        "roll_end_deg",
        "dq_w",
        "dq_x",
        "dq_y",
        "dq_z",
        "attitude_change_deg",
        "ddodo_velocity_kmh",
    ]

    print(largest[cols].to_string(index=False))

    print()
    print("5. FIRST 10 WINDOWS OF FIRST PAIR")
    print("-" * 72)

    first_pair = sorted(data["pair"].unique())[0]
    first = data[data["pair"] == first_pair].head(10)

    print(
        first[
            [
                "pair",
                "window_index",
                "yaw_start_deg",
                "pitch_start_deg",
                "roll_start_deg",
                "yaw_end_deg",
                "pitch_end_deg",
                "roll_end_deg",
                "dq_w",
                "dq_x",
                "dq_y",
                "dq_z",
                "attitude_change_deg",
                "ddodo_velocity_kmh",
            ]
        ].to_string(index=False)
    )

    out = base / "step7_ddatt_analysis.csv"
    data.to_csv(out, index=False)

    print()
    print("=" * 72)
    print("STEP 7 COMPLETE")
    print("=" * 72)
    print(f"Analysis table saved to: {out}")
    print("No target files were modified.")


if __name__ == "__main__":
    main()

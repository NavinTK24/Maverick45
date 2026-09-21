from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# MAVeriCK ML2 - STEP 4
#
# Prepare the target information for DDODO and DDATT.
#
# IMPORTANT:
# The paper defines DDATT as an ATTITUDE CHANGE represented by quaternion
# x,y,z components. The paper excerpt available to us does not specify the
# exact Euler-angle -> quaternion convention needed for our IO-VNBD phone
# Orientation (Yaw/Pitch/Roll) columns.
#
# Therefore this step DOES NOT guess that convention.
#
# It prepares:
#   - DDODO reference velocity: V column 5
#   - DDATT reference orientation: S columns 22,23,24
#   - beginning and ending orientation of every 1-second window
#
# The final quaternion attitude-change labels will be created only after the
# convention is explicitly fixed.
# ---------------------------------------------------------------------------

S_ORIENTATION_POSITIONS = [22, 23, 24]  # Yaw, Pitch, Roll
V_VELOCITY_POSITION = 5                 # Velocity (km/hr)
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


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python step4_prepare_target_data.py '
            '"PATH_TO_CATEGORISED_IOVNB_DATASET"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    pairs = find_pairs(root)

    if not pairs:
        raise RuntimeError("No matched S/V pairs found.")

    output_dir = Path(__file__).resolve().parent / "targets_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    total_invalid = 0

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 4: PREPARE DDODO + DDATT TARGET DATA")
    print("=" * 72)
    print(f"Input dataset: {root}")
    print(f"Matched S/V pairs: {len(pairs)}")
    print()
    print("DDODO reference : V column 5 -> Velocity (km/hr)")
    print("DDATT reference : S columns 22-24 -> Orientation Yaw/Pitch/Roll")
    print("Window          : 10 samples = 1 second")
    print()
    print("NOTE: Quaternion conversion is intentionally NOT performed here.")
    print("The paper requires quaternion x,y,z attitude-change targets, but")
    print("the exact Euler-to-quaternion convention is not specified in the")
    print("paper excerpt. We will not guess it.")
    print()

    for key, s_path, v_path in pairs:
        s = read_csv(s_path)
        v = read_csv(v_path)

        n_usable = min(len(s), len(v))
        s = s.iloc[:n_usable].copy()
        v = v.iloc[:n_usable].copy()

        orient_cols = [s.columns[p - 1] for p in S_ORIENTATION_POSITIONS]
        velocity_col = v.columns[V_VELOCITY_POSITION - 1]

        orientation = s[orient_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)

        velocity = pd.to_numeric(
            v[velocity_col], errors="coerce"
        ).to_numpy(dtype=np.float64)

        n_full = n_usable // WINDOW_SIZE

        rows = []

        for w in range(n_full):
            start = w * WINDOW_SIZE
            end = start + WINDOW_SIZE - 1

            ori_start = orientation[start]
            ori_end = orientation[end]
            vel_target = velocity[end]

            values = np.concatenate(
                [ori_start, ori_end, np.asarray([vel_target])]
            )

            if not np.isfinite(values).all():
                total_invalid += 1
                continue

            rows.append(
                [
                    w,
                    start,
                    end,
                    ori_start[0], ori_start[1], ori_start[2],
                    ori_end[0], ori_end[1], ori_end[2],
                    vel_target,
                ]
            )

        columns = [
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

        target_df = pd.DataFrame(rows, columns=columns)

        out_path = output_dir / f"{key}_targets_raw.csv"
        target_df.to_csv(out_path, index=False)

        total_windows += len(target_df)

        print(
            f"{key:12s} | windows={len(target_df):6d} "
            f"| {out_path.name}"
        )

    print()
    print("=" * 72)
    print("STEP 4 PREPARATION COMPLETE")
    print("=" * 72)
    print(f"Prepared target rows: {total_windows}")
    print(f"Invalid target windows skipped: {total_invalid}")
    print(f"Saved in: {output_dir}")
    print()
    print("Each row corresponds to one 1-second AVNet window.")
    print()
    print("Columns:")
    print("  velocity_target_kmh -> DDODO reference")
    print("  yaw/pitch/roll start -> beginning attitude")
    print("  yaw/pitch/roll end   -> ending attitude")
    print()
    print("Next step: convert the attitude change to the paper's")
    print("quaternion x,y,z target after fixing the required convention.")


if __name__ == "__main__":
    main()

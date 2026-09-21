from pathlib import Path
import sys
import pandas as pd


# We now use the column positions confirmed from the real CSV headers in Step 1.
# This avoids failures caused by harmless encoding/character differences such
# as m/s², Â°, and µT in the CSV header text.
#
# S-file:
#   columns 10-12 = Accelerometer X/Y/Z
#   columns 16-18 = Gyroscope Yaw/Pitch/Roll
#   columns 22-24 = Orientation Yaw/Pitch/Roll
#
# V-file:
#   column 5 = Velocity (km/hr)
#
# These are 1-based positions as displayed by Step 1.

S_IMU_POSITIONS = [10, 11, 12, 16, 17, 18]
S_ORIENTATION_POSITIONS = [22, 23, 24]
V_VELOCITY_POSITION = 5


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


def columns_at_positions(df, positions):
    """Convert 1-based column positions to actual column names."""
    names = list(df.columns)
    return [names[p - 1] for p in positions]


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python step2_verify_imu_columns.py '
            '"PATH_TO_CATEGORISED_IOVNB_DATASET"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    pairs = find_pairs(root)

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 2: VERIFY AVNET INPUT COLUMNS")
    print("=" * 72)
    print(f"Matched pairs: {len(pairs)}")

    if not pairs:
        print("No synchronized S/V pairs found.")
        sys.exit(1)

    print("\nColumn selection is based on the actual positions confirmed in Step 1.")
    print("This avoids CSV header encoding differences; original CSV files are untouched.")

    failures = []
    row_counts = []

    for key, s_path, v_path in pairs:
        try:
            s = read_csv(s_path)
            v = read_csv(v_path)

            if len(s.columns) < 24:
                failures.append((key, f"S file has only {len(s.columns)} columns"))
                continue

            if len(v.columns) < 5:
                failures.append((key, f"V file has only {len(v.columns)} columns"))
                continue

            s_imu_cols = columns_at_positions(s, S_IMU_POSITIONS)
            s_orientation_cols = columns_at_positions(s, S_ORIENTATION_POSITIONS)
            v_velocity_col = columns_at_positions(v, [V_VELOCITY_POSITION])[0]

            # Check that the selected columns contain usable numeric data.
            for col in s_imu_cols + s_orientation_cols:
                numeric = pd.to_numeric(s[col], errors="coerce")
                if numeric.notna().sum() == 0:
                    failures.append((key, f"S column is not numeric: {col}"))
                    break

            numeric_v = pd.to_numeric(v[v_velocity_col], errors="coerce")
            if numeric_v.notna().sum() == 0:
                failures.append((key, f"V velocity column is not numeric: {v_velocity_col}"))
                continue

            row_counts.append((key, len(s), len(v)))

        except Exception as exc:
            failures.append((key, repr(exc)))

    print("\n" + "=" * 72)
    print("VALIDATION RESULT")
    print("=" * 72)

    if failures:
        print(f"FAILED PAIRS: {len(failures)}")
        for key, reason in failures:
            print(f"  {key}: {reason}")
        sys.exit(2)

    print(f"All {len(pairs)} S/V pairs passed the required-column validation.")

    different_lengths = [
        (key, ns, nv) for key, ns, nv in row_counts if ns != nv
    ]

    print(f"Pairs with equal S/V row counts: {len(row_counts) - len(different_lengths)}")
    print(f"Pairs with different S/V row counts: {len(different_lengths)}")

    if different_lengths:
        print("\nDifferent row counts:")
        for key, ns, nv in different_lengths:
            print(f"  {key}: S={ns}, V={nv}")

    # Show the exact columns selected from the first pair.
    key, s_path, v_path = pairs[0]
    s = read_csv(s_path)
    v = read_csv(v_path)

    s_imu_cols = columns_at_positions(s, S_IMU_POSITIONS)
    s_orientation_cols = columns_at_positions(s, S_ORIENTATION_POSITIONS)
    v_velocity_col = columns_at_positions(v, [V_VELOCITY_POSITION])[0]

    print("\n" + "=" * 72)
    print(f"FIRST PAIR: {key}")
    print("=" * 72)

    print("\nAVNet input columns (6):")
    for i, col in enumerate(s_imu_cols, 1):
        print(f"  {i}. {col}")

    print("\nDDATT reference columns (3):")
    for col in s_orientation_cols:
        print(f"  - {col}")

    print("\nDDODO reference column:")
    print(f"  - {v_velocity_col}")

    print("\nFirst 5 rows of the 6 AVNet input channels:")
    print(s[s_imu_cols].head(5).to_string(index=False))

    print("\nFirst 5 DDATT reference rows:")
    print(s[s_orientation_cols].head(5).to_string(index=False))

    print("\nFirst 5 DDODO reference values:")
    print(v[[v_velocity_col]].head(5).to_string(index=False))

    print("\nNo filtering, normalization, resampling, or windowing was performed.")
    print("Step 2 only verifies the six AVNet inputs and the two reference sources.")
    print("\nSTEP 2 COMPLETE.")
    print("Next: create the 10-sample (1-second) windows.")


if __name__ == "__main__":
    main()

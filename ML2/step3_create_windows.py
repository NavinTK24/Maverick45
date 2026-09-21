from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# MAVeriCK ML2 - STEP 3
#
# Create the 1-second AVNet input windows.
#
# Confirmed project setup:
#   - synchronized S/V pairs from Categorised IOVNB Dataset
#   - smartphone sampling: 10 Hz
#   - window length: 1 second
#   - 10 samples/window
#   - 6 AVNet input channels
#
# This step ONLY creates the IMU windows.
# It does NOT:
#   - normalize
#   - filter
#   - resample
#   - create quaternion labels
#   - train a model
#
# For S/V pairs with different row counts, only the common synchronized
# portion is used:
#       N_usable = min(N_S, N_V)
# ---------------------------------------------------------------------------

S_IMU_POSITIONS = [10, 11, 12, 16, 17, 18]  # 1-based CSV positions
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
            'Usage: python step3_create_windows.py '
            '"PATH_TO_CATEGORISED_IOVNB_DATASET"'
        )
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    pairs = find_pairs(root)

    if not pairs:
        raise RuntimeError("No matched S/V pairs found.")

    # Output directory is created automatically.
    output_dir = Path(__file__).resolve().parent / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_window_files = []
    total_windows = 0

    print("=" * 72)
    print("MAVeriCK ML2 - STEP 3: CREATE 1-SECOND IMU WINDOWS")
    print("=" * 72)
    print(f"Input dataset: {root}")
    print(f"Matched S/V pairs: {len(pairs)}")
    print("Sampling rate: 10 Hz")
    print("Window length: 1 second")
    print("Samples per window: 10")
    print("AVNet features per sample: 6")
    print("Window shape: 10 x 6")
    print()

    for key, s_path, v_path in pairs:
        s = read_csv(s_path)
        v = read_csv(v_path)

        # Use the synchronized common portion.
        n_usable = min(len(s), len(v))
        s = s.iloc[:n_usable].copy()
        v = v.iloc[:n_usable].copy()

        imu_cols = [s.columns[p - 1] for p in S_IMU_POSITIONS]

        # Convert selected columns to numeric.
        imu = s[imu_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )

        # A complete window must contain 10 rows and no NaN/inf values.
        windows = []
        start_rows = []

        n_full = n_usable // WINDOW_SIZE

        for w in range(n_full):
            start = w * WINDOW_SIZE
            end = start + WINDOW_SIZE

            x = imu[start:end]

            if x.shape != (WINDOW_SIZE, 6):
                continue

            if not np.isfinite(x).all():
                continue

            windows.append(x)
            start_rows.append(start)

        if windows:
            arr = np.stack(windows).astype(np.float32)
        else:
            arr = np.empty((0, WINDOW_SIZE, 6), dtype=np.float32)

        # Save each S/V pair separately.
        npz_path = output_dir / f"{key}_windows.npz"
        np.savez_compressed(
            npz_path,
            X=arr,
            start_row=np.asarray(start_rows, dtype=np.int64),
        )

        all_window_files.append(npz_path)
        total_windows += len(arr)

        print(
            f"{key:12s} | S={len(s):7d} V={len(v):7d} "
            f"| windows={len(arr):6d} | {npz_path.name}"
        )

    print()
    print("=" * 72)
    print("STEP 3 COMPLETE")
    print("=" * 72)
    print(f"Total complete 10 x 6 windows: {total_windows}")
    print(f"Window files saved in: {output_dir}")
    print()
    print("Each .npz file contains:")
    print("  X         -> shape (number_of_windows, 10, 6)")
    print("  start_row -> starting row of each window in the common S/V portion")
    print()
    print("No labels have been created yet.")
    print("No normalization/filtering/resampling has been performed.")
    print("Next: define and create the DDODO and DDATT targets.")


if __name__ == "__main__":
    main()

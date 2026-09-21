import os
import glob
import numpy as np
import pandas as pd

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
WINDOW_DIR = r"D:\Maverick\ML2\windows"
OUT_DIR = r"D:\Maverick\ML2\targets_vheading"

WINDOW = 10

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# Robust CSV reader
# =========================================================
def read_csv_robust(path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


# =========================================================
# Find V files
# =========================================================
v_files = []

for root, dirs, files in os.walk(BASE):

    for f in files:

        if f.lower().startswith("v-") and f.lower().endswith(".csv"):

            v_files.append(
                os.path.join(root, f)
            )

v_files.sort()


print("=" * 70)
print("STEP 22 - V HEADING GROUND REFERENCE")
print("=" * 70)

print(f"V files found: {len(v_files)}")

total_windows = 0
processed = 0
skipped = 0


# =========================================================
# Process each V dataset
# =========================================================
for v_path in v_files:

    filename = os.path.basename(v_path)

    # Example:
    # V-vtb2.csv
    #
    # key:
    # vtb2
    key = os.path.splitext(filename)[0][2:].lower()


    # -----------------------------------------------------
    # Existing phone-window file
    # -----------------------------------------------------
    window_file = os.path.join(
        WINDOW_DIR,
        f"{key}_windows.npz"
    )


    if not os.path.exists(window_file):

        print(
            f"{key:10s} SKIPPED - "
            f"phone window file not found"
        )

        skipped += 1
        continue


    # -----------------------------------------------------
    # Load phone windows
    #
    # We do NOT use their sensor values here.
    #
    # We only use the number of windows so that
    # the V reference has exactly the same number.
    # -----------------------------------------------------
    try:

        window_data = np.load(window_file)

        X_windows = window_data["X"]

        n_phone_windows = len(X_windows)

    except Exception as e:

        print(
            f"{key:10s} ERROR loading windows: {e}"
        )

        skipped += 1
        continue


    # -----------------------------------------------------
    # Read V dataset
    # -----------------------------------------------------
    try:

        V = read_csv_robust(v_path)

    except Exception as e:

        print()
        print(f"ERROR reading {filename}")
        print(e)

        skipped += 1
        continue


    # -----------------------------------------------------
    # V columns
    #
    # Column 5 = Velocity (km/hr)
    # Column 6 = Heading (degrees)
    #
    # Python zero-based:
    # Velocity = iloc[:,4]
    # Heading  = iloc[:,5]
    # -----------------------------------------------------
    velocity = pd.to_numeric(
        V.iloc[:, 4],
        errors="coerce"
    ).to_numpy()

    heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()


    # -----------------------------------------------------
    # Number of windows MUST match phone windows
    # -----------------------------------------------------
    n_windows = n_phone_windows


    # Safety check
    required_rows = n_windows * WINDOW

    if len(V) < required_rows:

        print(
            f"{key:10s} ERROR - V dataset has only "
            f"{len(V)} rows, but "
            f"{required_rows} are required"
        )

        skipped += 1
        continue


    rows = []


    # =====================================================
    # Create exactly the same windows as phone data
    # =====================================================
    for w in range(n_windows):

        start = w * WINDOW

        end = start + WINDOW - 1


        h_start = heading[start]

        h_end = heading[end]

        velocity_target = velocity[end]


        if not np.isfinite(h_start):
            raise ValueError(
                f"{key}: invalid heading at row {start}"
            )

        if not np.isfinite(h_end):
            raise ValueError(
                f"{key}: invalid heading at row {end}"
            )

        if not np.isfinite(velocity_target):
            raise ValueError(
                f"{key}: invalid velocity at row {end}"
            )


        # =================================================
        # Circular heading difference
        #
        # Correctly handles:
        #
        # 359 -> 1
        #
        # as +2 degrees rather than -358 degrees.
        # =================================================
        delta_yaw_deg = (
            (h_end - h_start + 180.0) % 360.0
        ) - 180.0


        delta_yaw_rad = np.deg2rad(
            delta_yaw_deg
        )


        # =================================================
        # Pure yaw quaternion
        #
        # [qw, qx, qy, qz]
        #
        # Yaw only:
        #
        # qx = 0
        # qy = 0
        # qz = sin(delta_yaw / 2)
        # =================================================
        qw = np.cos(
            delta_yaw_rad / 2.0
        )

        qx = 0.0

        qy = 0.0

        qz = np.sin(
            delta_yaw_rad / 2.0
        )


        rows.append([
            w,
            start,
            end,
            h_start,
            h_end,
            delta_yaw_deg,
            qw,
            qx,
            qy,
            qz,
            velocity_target
        ])


    # =====================================================
    # Create target dataframe
    # =====================================================
    out = pd.DataFrame(
        rows,
        columns=[
            "window_index",
            "start_row",
            "end_row",
            "heading_start_deg",
            "heading_end_deg",
            "delta_yaw_deg",
            "dq_w",
            "ddatt_qx",
            "ddatt_qy",
            "ddatt_qz",
            "ddodo_velocity_kmh"
        ]
    )


    # =====================================================
    # Final safety check
    # =====================================================
    if len(out) != n_phone_windows:

        raise ValueError(
            f"{key}: target/window mismatch: "
            f"targets={len(out)}, "
            f"phone_windows={n_phone_windows}"
        )


    # =====================================================
    # Save
    # =====================================================
    out_path = os.path.join(
        OUT_DIR,
        f"{key}_targets_vheading.csv"
    )


    out.to_csv(
        out_path,
        index=False
    )


    processed += 1

    total_windows += len(out)


    print(
        f"{key:10s} "
        f"phone_windows={n_phone_windows:6d} "
        f"targets={len(out):6d}"
    )


# =========================================================
# Final summary
# =========================================================
print("-" * 70)

print(f"Processed: {processed}")

print(f"Skipped:   {skipped}")

print(f"Total target windows: {total_windows}")

print()

print("Expected phone-window count: 107043")

if total_windows == 107043:

    print("MATCH: 107043 targets = 107043 phone windows")

else:

    print(
        f"WARNING: mismatch! "
        f"targets={total_windows}"
    )

print()

print("Saved to:")
print(OUT_DIR)

print("=" * 70)
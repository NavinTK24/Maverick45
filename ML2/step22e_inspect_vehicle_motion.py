import os
import glob
import numpy as np
import pandas as pd

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

WINDOW = 10
TOP_N = 20


# =========================================================
# Robust CSV reader
# =========================================================

def read_csv_robust(path):

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"Could not decode: {path}")


# =========================================================
# Circular heading difference
# =========================================================

def circular_difference(a, b):
    return ((b - a + 180.0) % 360.0) - 180.0


# =========================================================
# Find V files
# =========================================================

v_files = sorted(
    glob.glob(
        os.path.join(
            BASE,
            "**",
            "V-*.csv"
        ),
        recursive=True
    )
)


print("=" * 90)
print("STEP 22E - HEADING vs VEHICLE MOTION")
print("=" * 90)

print(f"V files found: {len(v_files)}")
print()


# =========================================================
# Calculate suspicious windows
# =========================================================

results = []


for v_path in v_files:

    filename = os.path.basename(v_path)

    key = os.path.splitext(
        filename
    )[0][2:].lower()

    V = read_csv_robust(v_path)


    # -----------------------------------------------------
    # Verified V columns
    #
    # 2  = Time since start of day
    # 5  = Velocity (km/h)
    # 6  = Heading (deg)
    # 9  = Sample period (s)
    # 15 = Yaw rate (deg/s)
    # 17 = Longitudinal acceleration
    # 18 = Lateral acceleration
    #
    # Python zero-based:
    # -----------------------------------------------------

    time = pd.to_numeric(
        V.iloc[:, 1],
        errors="coerce"
    ).to_numpy()

    velocity = pd.to_numeric(
        V.iloc[:, 4],
        errors="coerce"
    ).to_numpy()

    heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    yaw_rate = pd.to_numeric(
        V.iloc[:, 14],
        errors="coerce"
    ).to_numpy()

    long_acc = pd.to_numeric(
        V.iloc[:, 16],
        errors="coerce"
    ).to_numpy()

    lat_acc = pd.to_numeric(
        V.iloc[:, 17],
        errors="coerce"
    ).to_numpy()


    n_windows = len(V) // WINDOW


    for window in range(n_windows):

        start = window * WINDOW
        end = start + WINDOW

        h = heading[start:end]
        yr = yaw_rate[start:end]
        vel = velocity[start:end]
        la = long_acc[start:end]
        lta = lat_acc[start:end]
        tm = time[start:end]


        if len(h) != WINDOW:
            continue


        if not (
            np.isfinite(h[0])
            and np.isfinite(h[-1])
        ):
            continue


        # Heading change
        heading_change = circular_difference(
            h[0],
            h[-1]
        )


        # Integrate yaw rate using 0.1 sec
        # where actual sample period is unavailable.
        valid_yr = np.isfinite(yr)

        yawrate_change = np.sum(
            np.where(
                valid_yr,
                yr * 0.1,
                0.0
            )
        )


        difference = (
            heading_change
            - yawrate_change
        )


        results.append({

            "dataset": key,

            "window": window,

            "start_row": start,

            "end_row": end - 1,

            "heading_change_deg":
                heading_change,

            "yawrate_change_deg":
                yawrate_change,

            "difference_deg":
                difference,

            "absolute_difference_deg":
                abs(difference)

        })


# =========================================================
# Sort by disagreement
# =========================================================

results = pd.DataFrame(results)

results = results.sort_values(
    "absolute_difference_deg",
    ascending=False
).reset_index(drop=True)


# =========================================================
# Inspect top cases
# =========================================================

for rank, (_, r) in enumerate(
    results.head(TOP_N).iterrows(),
    start=1
):

    key = r["dataset"]
    window = int(r["window"])

    matching = [

        p for p in v_files

        if os.path.splitext(
            os.path.basename(p)
        )[0][2:].lower()
        == key

    ]


    if not matching:
        continue


    V = read_csv_robust(
        matching[0]
    )


    start = int(r["start_row"])
    end = int(r["end_row"]) + 1


    block = V.iloc[
        start:end
    ]


    time = pd.to_numeric(
        block.iloc[:, 1],
        errors="coerce"
    ).to_numpy()

    velocity = pd.to_numeric(
        block.iloc[:, 4],
        errors="coerce"
    ).to_numpy()

    heading = pd.to_numeric(
        block.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    yaw_rate = pd.to_numeric(
        block.iloc[:, 14],
        errors="coerce"
    ).to_numpy()

    long_acc = pd.to_numeric(
        block.iloc[:, 16],
        errors="coerce"
    ).to_numpy()

    lat_acc = pd.to_numeric(
        block.iloc[:, 17],
        errors="coerce"
    ).to_numpy()


    print()
    print("=" * 90)
    print(f"CASE #{rank}")
    print("=" * 90)

    print(f"Dataset : {key}")
    print(f"Window  : {window}")
    print(
        f"Rows    : {start} -> {end - 1}"
    )

    print()

    print(
        f"Heading Δ  : "
        f"{r['heading_change_deg']:.3f}°"
    )

    print(
        f"Yaw-rate Δ : "
        f"{r['yawrate_change_deg']:.3f}°"
    )

    print(
        f"Difference : "
        f"{r['difference_deg']:.3f}°"
    )

    print()

    print(
        "Row       Time(s)   Velocity   Heading   "
        "YawRate   LongAcc   LatAcc"
    )

    print(
        "----      -------   --------   -------   "
        "-------   -------   ------"
    )


    for i in range(len(block)):

        print(
            f"{start+i:5d}   "
            f"{time[i]:9.3f}   "
            f"{velocity[i]:8.3f}   "
            f"{heading[i]:8.3f}   "
            f"{yaw_rate[i]:7.3f}   "
            f"{long_acc[i]:7.3f}   "
            f"{lat_acc[i]:7.3f}"
        )


    # -----------------------------------------------------
    # Heading changes between consecutive samples
    # -----------------------------------------------------

    changes = []

    for i in range(1, len(heading)):

        if (
            np.isfinite(heading[i-1])
            and np.isfinite(heading[i])
        ):

            changes.append(
                circular_difference(
                    heading[i-1],
                    heading[i]
                )
            )


    print()

    print(
        "Consecutive Heading changes (deg):"
    )

    print(
        np.round(
            changes,
            3
        )
    )


    # -----------------------------------------------------
    # Maximum motion values in this window
    # -----------------------------------------------------

    print()

    print(
        f"Maximum |Yaw Rate| : "
        f"{np.nanmax(np.abs(yaw_rate)):.3f} °/s"
    )

    print(
        f"Maximum |Lat Acc|  : "
        f"{np.nanmax(np.abs(lat_acc)):.3f}"
    )

    print(
        f"Maximum |Long Acc| : "
        f"{np.nanmax(np.abs(long_acc)):.3f}"
    )

    print(
        f"Velocity range     : "
        f"{np.nanmin(velocity):.3f} -> "
        f"{np.nanmax(velocity):.3f} km/h"
    )


print()
print("=" * 90)
print("STEP 22E COMPLETE")
print("=" * 90)

print()
print(
    "No data was removed or modified."
)

print(
    "This step only compares Heading with "
    "independent vehicle-motion measurements."
)
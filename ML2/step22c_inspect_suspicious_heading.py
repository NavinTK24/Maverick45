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
# Circular angle difference
# =========================================================

def circular_difference(a, b):

    return ((b - a + 180.0) % 360.0) - 180.0


# =========================================================
# Find all V files
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


print("=" * 80)
print("STEP 22C - INSPECT SUSPICIOUS V HEADING CHANGES")
print("=" * 80)

print(
    f"V files found: {len(v_files)}"
)

print()


# =========================================================
# Recalculate all window comparisons
# =========================================================

all_results = []


for v_path in v_files:

    filename = os.path.basename(v_path)

    # Example:
    # V-M.csv -> M
    key = os.path.splitext(filename)[0][2:].lower()

    V = read_csv_robust(v_path)

    # -----------------------------------------------------
    # Required V columns by verified positions
    #
    # 2 = Time since start of day
    # 6 = Heading
    # 9 = Sample period
    # 15 = Yaw rate
    #
    # Python zero-based:
    # heading  -> 5
    # sample   -> 8
    # yaw rate -> 14
    # -----------------------------------------------------

    heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    sample_period = pd.to_numeric(
        V.iloc[:, 8],
        errors="coerce"
    ).to_numpy()

    yaw_rate = pd.to_numeric(
        V.iloc[:, 14],
        errors="coerce"
    ).to_numpy()


    # -----------------------------------------------------
    # Complete 10-sample windows
    # -----------------------------------------------------

    n_windows = len(V) // WINDOW


    for window in range(n_windows):

        start = window * WINDOW
        end = start + WINDOW

        h = heading[start:end]
        yr = yaw_rate[start:end]
        dt = sample_period[start:end]


        if len(h) != WINDOW:
            continue


        # -------------------------------------------------
        # Need valid heading endpoints
        # -------------------------------------------------

        if not (
            np.isfinite(h[0])
            and np.isfinite(h[-1])
        ):
            continue


        # -------------------------------------------------
        # Heading change
        # -------------------------------------------------

        heading_change = circular_difference(
            h[0],
            h[-1]
        )


        # -------------------------------------------------
        # Integrate yaw rate
        #
        # If sample period is available, use it.
        # Otherwise use 0.1 s.
        # -------------------------------------------------

        valid_yr = np.isfinite(yr)

        if not np.any(valid_yr):
            continue


        if np.any(np.isfinite(dt)):

            dt_clean = np.where(
                np.isfinite(dt),
                dt,
                0.1
            )

        else:

            dt_clean = np.full(
                WINDOW,
                0.1
            )


        yawrate_change = np.sum(
            np.where(
                valid_yr,
                yr * dt_clean,
                0.0
            )
        )


        difference = (
            heading_change
            - yawrate_change
        )


        all_results.append({

            "dataset": key,

            "window": window,

            "start_row": start,

            "end_row": end - 1,

            "heading_start_deg": h[0],

            "heading_end_deg": h[-1],

            "heading_change_deg": heading_change,

            "yawrate_integrated_change_deg":
                yawrate_change,

            "difference_deg":
                difference,

            "absolute_difference_deg":
                abs(difference)

        })


# =========================================================
# Create results table
# =========================================================

results = pd.DataFrame(all_results)

results = results.sort_values(
    "absolute_difference_deg",
    ascending=False
).reset_index(drop=True)


print(
    f"Total windows analyzed: {len(results)}"
)

print()


# =========================================================
# Save suspicious-window table
# =========================================================

output_file = (
    r"D:\Maverick\ML2"
    r"\step22c_suspicious_heading_windows.csv"
)

results.head(TOP_N).to_csv(
    output_file,
    index=False
)


# =========================================================
# Print top suspicious windows
# =========================================================

print("=" * 80)
print(
    f"TOP {TOP_N} SUSPICIOUS WINDOWS"
)
print("=" * 80)


for rank, (_, r) in enumerate(
    results.head(TOP_N).iterrows(),
    start=1
):

    print()

    print("-" * 80)

    print(
        f"CASE #{rank}"
    )

    print(
        f"Dataset : {r['dataset']}"
    )

    print(
        f"Window  : {int(r['window'])}"
    )

    print(
        f"Rows    : "
        f"{int(r['start_row'])} -> "
        f"{int(r['end_row'])}"
    )

    print()

    print(
        f"Heading start : "
        f"{r['heading_start_deg']:.3f}°"
    )

    print(
        f"Heading end   : "
        f"{r['heading_end_deg']:.3f}°"
    )

    print(
        f"Heading Δ     : "
        f"{r['heading_change_deg']:.3f}°"
    )

    print(
        f"Yaw-rate Δ    : "
        f"{r['yawrate_integrated_change_deg']:.3f}°"
    )

    print(
        f"Difference    : "
        f"{r['difference_deg']:.3f}°"
    )


    # =====================================================
    # Find V file
    # =====================================================

    matching = [
        p for p in v_files
        if os.path.splitext(
            os.path.basename(p)
        )[0][2:].lower()
        == r["dataset"]
    ]


    if not matching:

        print(
            "V file not found."
        )

        continue


    v_path = matching[0]

    V = read_csv_robust(v_path)


    start = int(r["start_row"])
    end = int(r["end_row"]) + 1


    sample_rows = V.iloc[
        start:end
    ]


    h = pd.to_numeric(
        sample_rows.iloc[:, 5],
        errors="coerce"
    ).to_numpy()


    yr = pd.to_numeric(
        sample_rows.iloc[:, 14],
        errors="coerce"
    ).to_numpy()


    time = pd.to_numeric(
        sample_rows.iloc[:, 1],
        errors="coerce"
    ).to_numpy()


    # =====================================================
    # Print actual 10 Hz samples
    # =====================================================

    print()

    print(
        "Raw 10 Hz samples:"
    )

    print()

    print(
        "Row        Time(s)       Heading(°)       YawRate(°/s)"
    )

    print(
        "---        -------       ----------       -----------"
    )


    for i in range(len(h)):

        print(
            f"{start+i:6d}    "
            f"{time[i]:10.3f}    "
            f"{h[i]:12.3f}    "
            f"{yr[i]:13.3f}"
        )


    # =====================================================
    # Consecutive heading changes
    # =====================================================

    print()

    print(
        "Consecutive circular heading changes:"
    )


    changes = []


    for i in range(1, len(h)):

        if (
            np.isfinite(h[i - 1])
            and np.isfinite(h[i])
        ):

            d = circular_difference(
                h[i - 1],
                h[i]
            )

            changes.append(d)


    print(
        np.round(
            changes,
            3
        )
    )


print()
print("=" * 80)
print("STEP 22C COMPLETE")
print("=" * 80)

print()

print(
    "Saved:"
)

print(
    output_file
)

print()

print(
    "The purpose of this step is ONLY inspection."
)

print(
    "No samples have been removed or modified."
)
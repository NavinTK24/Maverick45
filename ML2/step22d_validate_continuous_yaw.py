import os
import glob
import numpy as np
import pandas as pd

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

OUTPUT = r"D:\Maverick\ML2\step22d_continuous_yaw_results.csv"


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
# Circular difference
# =========================================================

def circular_difference(a, b):

    return ((b - a + 180.0) % 360.0) - 180.0


# =========================================================
# Find V datasets
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
print("STEP 22D - CONTINUOUS YAW VALIDATION")
print("=" * 80)

print(
    f"V files found: {len(v_files)}"
)

print()


all_results = []
dataset_summary = []


# =========================================================
# Process every dataset
# =========================================================

for v_path in v_files:

    filename = os.path.basename(v_path)

    key = os.path.splitext(
        filename
    )[0][2:].lower()

    V = read_csv_robust(v_path)


    # -----------------------------------------------------
    # Verified V columns
    #
    # Heading = column 6
    # Sample period = column 9
    # Yaw rate = column 15
    #
    # zero-based:
    # heading = 5
    # dt      = 8
    # yawrate = 14
    # -----------------------------------------------------

    heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    dt = pd.to_numeric(
        V.iloc[:, 8],
        errors="coerce"
    ).to_numpy()

    yaw_rate = pd.to_numeric(
        V.iloc[:, 14],
        errors="coerce"
    ).to_numpy()


    N = len(V)


    # =====================================================
    # Original heading → continuous/unwrapped heading
    # =====================================================

    valid_heading = np.isfinite(heading)

    if np.sum(valid_heading) < 2:
        continue


    # Fill missing heading values by interpolation
    heading_series = pd.Series(
        heading
    )

    heading_filled = (
        heading_series
        .interpolate()
        .ffill()
        .bfill()
        .to_numpy()
    )


    # Convert degrees to continuous representation
    heading_unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(
                heading_filled
            )
        )
    )


    # =====================================================
    # Integrate yaw rate
    # =====================================================

    dt_clean = np.where(
        np.isfinite(dt) & (dt > 0),
        dt,
        0.1
    )


    yaw_clean = np.where(
        np.isfinite(yaw_rate),
        yaw_rate,
        0.0
    )


    integrated_delta = np.zeros(N)

    if N > 1:

        integrated_delta[1:] = np.cumsum(
            yaw_clean[:-1]
            * dt_clean[:-1]
        )


    # Start integrated yaw at first heading
    integrated_heading = (
        heading_filled[0]
        + integrated_delta
    )


    # =====================================================
    # Align integrated yaw with heading
    #
    # Difference is allowed to contain an initial
    # constant offset. We compare relative evolution.
    # =====================================================

    original_relative = (
        heading_unwrapped
        - heading_unwrapped[0]
    )

    integrated_relative = (
        integrated_heading
        - integrated_heading[0]
    )


    error = (
        original_relative
        - integrated_relative
    )


    abs_error = np.abs(error)


    # =====================================================
    # Statistics
    # =====================================================

    median_error = np.median(
        abs_error
    )

    p90 = np.percentile(
        abs_error,
        90
    )

    p95 = np.percentile(
        abs_error,
        95
    )

    p99 = np.percentile(
        abs_error,
        99
    )

    maximum = np.max(
        abs_error
    )


    dataset_summary.append({

        "dataset": key,

        "samples": N,

        "median_abs_error_deg":
            median_error,

        "p90_abs_error_deg":
            p90,

        "p95_abs_error_deg":
            p95,

        "p99_abs_error_deg":
            p99,

        "max_abs_error_deg":
            maximum

    })


    # =====================================================
    # Save sampled result
    # =====================================================

    # Save every 10th sample to keep output manageable
    step = 10

    for i in range(
        0,
        N,
        step
    ):

        all_results.append({

            "dataset": key,

            "row": i,

            "time_s":
                i * 0.1,

            "heading_deg":
                heading_filled[i],

            "heading_unwrapped_deg":
                heading_unwrapped[i],

            "integrated_heading_deg":
                integrated_heading[i],

            "heading_relative_deg":
                original_relative[i],

            "integrated_relative_deg":
                integrated_relative[i],

            "error_deg":
                error[i],

            "absolute_error_deg":
                abs_error[i]

        })


# =========================================================
# Save detailed results
# =========================================================

detail_df = pd.DataFrame(
    all_results
)

detail_file = (
    r"D:\Maverick\ML2"
    r"\step22d_continuous_yaw_samples.csv"
)

detail_df.to_csv(
    detail_file,
    index=False
)


# =========================================================
# Save dataset summary
# =========================================================

summary_df = pd.DataFrame(
    dataset_summary
)

summary_df = summary_df.sort_values(
    "median_abs_error_deg",
    ascending=False
)

summary_df.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# Print results
# =========================================================

print()
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


# =========================================================
# Overall statistics
# =========================================================

print()
print("=" * 80)
print("OVERALL")
print("=" * 80)


print(
    f"Datasets analyzed : "
    f"{len(summary_df)}"
)

print(
    f"Median of dataset median errors : "
    f"{summary_df['median_abs_error_deg'].median():.3f}°"
)

print(
    f"Median dataset P90 error : "
    f"{summary_df['p90_abs_error_deg'].median():.3f}°"
)

print(
    f"Median dataset P95 error : "
    f"{summary_df['p95_abs_error_deg'].median():.3f}°"
)

print(
    f"Median dataset P99 error : "
    f"{summary_df['p99_abs_error_deg'].median():.3f}°"
)


print()
print(
    "Worst 10 datasets by median error:"
)

print()

print(
    summary_df.head(10).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("STEP 22D COMPLETE")
print("=" * 80)

print()
print("Saved:")
print(OUTPUT)
print(detail_file)
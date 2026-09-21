import os
import glob
import numpy as np
import pandas as pd

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
TARGET_DIR = r"D:\Maverick\ML2\targets_vheading"

WINDOW = 10

# ---------------------------------------------------------
# Robust CSV reader
# ---------------------------------------------------------
def read_csv_robust(path):

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass

    raise RuntimeError(f"Could not decode: {path}")


# ---------------------------------------------------------
# Find V files
# ---------------------------------------------------------
v_files = sorted(
    glob.glob(
        os.path.join(BASE, "**", "V-*.csv"),
        recursive=True
    )
)

print("=" * 75)
print("STEP 22B - V HEADING VS V YAW RATE")
print("=" * 75)

print(f"V files found: {len(v_files)}")


all_heading_change = []
all_yawrate_change = []
all_error = []

large_disagreements = []

total_windows = 0


# =========================================================
# Process each dataset
# =========================================================

for v_path in v_files:

    filename = os.path.basename(v_path)

    key = os.path.splitext(filename)[0][2:].lower()

    target_path = os.path.join(
        TARGET_DIR,
        f"{key}_targets_vheading.csv"
    )

    if not os.path.exists(target_path):

        print(
            f"{key:10s} target missing - skipped"
        )

        continue


    V = read_csv_robust(v_path)

    target = pd.read_csv(target_path)

    # -----------------------------------------------------
    # V columns
    #
    # Column 6 = Heading
    # Column 15 = Yaw Rate
    #
    # Python:
    # Heading = iloc[:,5]
    # Yaw Rate = iloc[:,14]
    # -----------------------------------------------------

    heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    yaw_rate = pd.to_numeric(
        V.iloc[:, 14],
        errors="coerce"
    ).to_numpy()


    n_windows = len(target)


    dataset_heading = []
    dataset_yawrate = []
    dataset_error = []


    # =====================================================
    # Analyze every 1-second window
    # =====================================================

    for w in range(n_windows):

        start = w * WINDOW
        end = start + WINDOW

        if end > len(V):
            break


        # -------------------------------------------------
        # Heading change from start to end
        #
        # Use the same circular definition as Step 22A.
        # -------------------------------------------------

        h_start = heading[start]

        h_end = heading[end - 1]


        if not np.isfinite(h_start):
            continue

        if not np.isfinite(h_end):
            continue


        heading_change = (
            (h_end - h_start + 180.0) % 360.0
        ) - 180.0


        # -------------------------------------------------
        # Yaw-rate integration
        #
        # Dataset is 10 Hz.
        # Therefore:
        #
        # dt = 0.1 second
        #
        # Integrate the 10 yaw-rate samples.
        # -------------------------------------------------

        rates = yaw_rate[start:end]

        if not np.all(np.isfinite(rates)):
            continue


        yawrate_change = np.sum(rates) * 0.1


        # -------------------------------------------------
        # Difference
        # -------------------------------------------------

        error = (
            heading_change
            - yawrate_change
        )


        dataset_heading.append(
            heading_change
        )

        dataset_yawrate.append(
            yawrate_change
        )

        dataset_error.append(
            error
        )


        all_heading_change.append(
            heading_change
        )

        all_yawrate_change.append(
            yawrate_change
        )

        all_error.append(
            error
        )


        # -------------------------------------------------
        # Store large disagreements
        # -------------------------------------------------

        if abs(error) >= 30:

            large_disagreements.append({
                "dataset": key,
                "window": w,
                "heading_change": heading_change,
                "yawrate_change": yawrate_change,
                "error": error,
                "heading_start": h_start,
                "heading_end": h_end
            })


    total_windows += len(dataset_heading)


    # -----------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------

    if len(dataset_error) > 0:

        dataset_error = np.asarray(
            dataset_error
        )

        print(
            f"{key:10s} "
            f"windows={len(dataset_error):6d} "
            f"median_err="
            f"{np.median(np.abs(dataset_error)):.3f}° "
            f"p95_abs="
            f"{np.percentile(np.abs(dataset_error),95):.3f}°"
        )


# =========================================================
# Convert to arrays
# =========================================================

heading_change = np.asarray(
    all_heading_change
)

yawrate_change = np.asarray(
    all_yawrate_change
)

error = np.asarray(
    all_error
)

abs_error = np.abs(error)


# =========================================================
# Overall statistics
# =========================================================

print()
print("-" * 75)
print("OVERALL HEADING vs YAW-RATE")
print("-" * 75)

print(
    f"Windows analyzed: {len(error)}"
)

print(
    f"Median absolute error: "
    f"{np.median(abs_error):.6f}°"
)

print(
    f"P90 absolute error: "
    f"{np.percentile(abs_error,90):.6f}°"
)

print(
    f"P95 absolute error: "
    f"{np.percentile(abs_error,95):.6f}°"
)

print(
    f"P99 absolute error: "
    f"{np.percentile(abs_error,99):.6f}°"
)

print(
    f"Maximum absolute error: "
    f"{np.max(abs_error):.6f}°"
)


# =========================================================
# Agreement thresholds
# =========================================================

print()
print("-" * 75)
print("AGREEMENT THRESHOLDS")
print("-" * 75)

for threshold in [1, 2, 5, 10, 20, 30, 45, 90]:

    count = np.sum(
        abs_error <= threshold
    )

    percentage = (
        count / len(error) * 100
    )

    print(
        f"Error <= {threshold:3d}° : "
        f"{count:7d} "
        f"({percentage:.4f}%)"
    )


# =========================================================
# Heading change statistics
# =========================================================

print()
print("-" * 75)
print("HEADING-CHANGE STATISTICS")
print("-" * 75)

for name, values in [
    ("Heading Δ", heading_change),
    ("YawRate Δ", yawrate_change)
]:

    print()
    print(name)

    print(
        f"  median = "
        f"{np.median(values):.6f}°"
    )

    print(
        f"  P90    = "
        f"{np.percentile(np.abs(values),90):.6f}°"
    )

    print(
        f"  P95    = "
        f"{np.percentile(np.abs(values),95):.6f}°"
    )

    print(
        f"  P99    = "
        f"{np.percentile(np.abs(values),99):.6f}°"
    )

    print(
        f"  max    = "
        f"{np.max(np.abs(values)):.6f}°"
    )


# =========================================================
# Correlation
# =========================================================

if len(error) > 1:

    corr = np.corrcoef(
        heading_change,
        yawrate_change
    )[0, 1]

else:

    corr = np.nan


print()
print("-" * 75)
print("CORRELATION")
print("-" * 75)

print(
    f"Heading Δ vs YawRate Δ correlation: "
    f"{corr:.6f}"
)


# =========================================================
# Large disagreements
# =========================================================

print()
print("-" * 75)
print("LARGEST HEADING / YAW-RATE DISAGREEMENTS")
print("-" * 75)

large_disagreements.sort(
    key=lambda x: abs(x["error"]),
    reverse=True
)


for x in large_disagreements[:30]:

    print(
        f"{x['dataset']:8s} "
        f"window={x['window']:5d} | "
        f"heading "
        f"{x['heading_start']:9.3f} -> "
        f"{x['heading_end']:9.3f} | "
        f"Δheading="
        f"{x['heading_change']:9.3f}° | "
        f"Δyaw-rate="
        f"{x['yawrate_change']:9.3f}° | "
        f"error="
        f"{x['error']:9.3f}°"
    )


# =========================================================
# Save detailed results
# =========================================================

result_df = pd.DataFrame({
    "heading_change_deg": heading_change,
    "yawrate_integrated_change_deg": yawrate_change,
    "difference_deg": error,
    "absolute_difference_deg": abs_error
})

output_file = (
    r"D:\Maverick\ML2"
    r"\step22b_heading_vs_yawrate.csv"
)

result_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# Final
# =========================================================

print()
print("=" * 75)
print("STEP 22B COMPLETE")
print("=" * 75)

print(
    f"Saved detailed results to:"
)

print(output_file)

print("=" * 75)
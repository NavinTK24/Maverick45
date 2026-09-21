import os
import glob
import numpy as np
import pandas as pd

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

WINDOW = 10


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
print("STEP 22F - HEADING TRANSITION STATISTICS")
print("=" * 90)

print(
    f"V files found: {len(v_files)}"
)

print()


# =========================================================
# Collect all windows
# =========================================================

records = []


for v_path in v_files:

    filename = os.path.basename(v_path)

    key = os.path.splitext(
        filename
    )[0][2:].lower()

    V = read_csv_robust(v_path)


    # -----------------------------------------------------
    # Verified V columns
    #
    # Heading       = 6
    # Sample period = 9
    # Yaw rate      = 15
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


    n_windows = len(V) // WINDOW


    for window in range(n_windows):

        start = window * WINDOW
        end = start + WINDOW

        h = heading[start:end]
        yr = yaw_rate[start:end]
        d = dt[start:end]


        if len(h) != WINDOW:
            continue


        if not (
            np.isfinite(h[0])
            and np.isfinite(h[-1])
        ):
            continue


        # -------------------------------------------------
        # Heading change over 1 second
        # -------------------------------------------------

        heading_change = circular_difference(
            h[0],
            h[-1]
        )


        # -------------------------------------------------
        # Yaw-rate integration
        # -------------------------------------------------

        d = np.where(
            np.isfinite(d) & (d > 0),
            d,
            0.1
        )

        yr = np.where(
            np.isfinite(yr),
            yr,
            0.0
        )


        yawrate_change = np.sum(
            yr * d
        )


        difference = (
            heading_change
            - yawrate_change
        )


        records.append({

            "dataset": key,

            "window": window,

            "heading_change_deg":
                heading_change,

            "abs_heading_change_deg":
                abs(heading_change),

            "yawrate_change_deg":
                yawrate_change,

            "abs_yawrate_change_deg":
                abs(yawrate_change),

            "difference_deg":
                difference,

            "abs_difference_deg":
                abs(difference)

        })


# =========================================================
# DataFrame
# =========================================================

df = pd.DataFrame(records)


print(
    f"Total windows analyzed: {len(df)}"
)

print()


# =========================================================
# Save complete analysis
# =========================================================

output_file = (
    r"D:\Maverick\ML2"
    r"\step22f_heading_transition_analysis.csv"
)

df.to_csv(
    output_file,
    index=False
)


# =========================================================
# Heading-change bins
# =========================================================

bins = [
    0,
    5,
    10,
    20,
    45,
    90,
    120,
    150,
    180.0001
]

labels = [
    "0-5",
    "5-10",
    "10-20",
    "20-45",
    "45-90",
    "90-120",
    "120-150",
    "150-180"
]


df["heading_bin"] = pd.cut(
    df["abs_heading_change_deg"],
    bins=bins,
    labels=labels,
    right=False
)


# =========================================================
# Print heading distribution
# =========================================================

print("=" * 90)
print("HEADING CHANGE DISTRIBUTION")
print("=" * 90)

print()

bin_counts = (
    df["heading_bin"]
    .value_counts(
        sort=False
    )
)

for label, count in bin_counts.items():

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"{label:>8}° : "
        f"{count:7d} windows "
        f"({percentage:8.3f}%)"
    )


# =========================================================
# Agreement analysis
#
# Difference thresholds:
# <= 2°
# <= 5°
# <= 10°
# <= 20°
#
# =========================================================

print()
print("=" * 90)
print("HEADING CHANGE vs YAW-RATE AGREEMENT")
print("=" * 90)

print()

agreement_thresholds = [
    2,
    5,
    10,
    20
]


for label in labels:

    subset = df[
        df["heading_bin"] == label
    ]

    if len(subset) == 0:
        continue


    print(
        f"\nHeading change {label}° "
        f"({len(subset)} windows)"
    )


    print(
        f"  Median |difference| : "
        f"{subset['abs_difference_deg'].median():.3f}°"
    )

    print(
        f"  P90 |difference|    : "
        f"{subset['abs_difference_deg'].quantile(.90):.3f}°"
    )

    print(
        f"  P95 |difference|    : "
        f"{subset['abs_difference_deg'].quantile(.95):.3f}°"
    )


    for threshold in agreement_thresholds:

        percentage = (
            np.mean(
                subset[
                    "abs_difference_deg"
                ] <= threshold
            )
            * 100
        )

        print(
            f"  Agreement <= {threshold:2d}° : "
            f"{percentage:7.3f}%"
        )


# =========================================================
# Large-transition analysis
# =========================================================

print()
print("=" * 90)
print("LARGE HEADING TRANSITIONS")
print("=" * 90)

print()


large_thresholds = [
    45,
    90,
    120,
    150
]


for threshold in large_thresholds:

    subset = df[
        df[
            "abs_heading_change_deg"
        ] >= threshold
    ]


    print(
        f"\n|Heading change| >= {threshold}°"
    )

    print(
        f"  Windows : "
        f"{len(subset)}"
    )

    print(
        f"  Percentage : "
        f"{len(subset)/len(df)*100:.4f}%"
    )


    if len(subset) > 0:

        print(
            f"  Median |Yaw-rate change| : "
            f"{subset['abs_yawrate_change_deg'].median():.3f}°"
        )

        print(
            f"  Median |difference| : "
            f"{subset['abs_difference_deg'].median():.3f}°"
        )

        print(
            f"  <=5° agreement : "
            f"{np.mean(subset['abs_difference_deg'] <= 5)*100:.3f}%"
        )

        print(
            f"  <=10° agreement : "
            f"{np.mean(subset['abs_difference_deg'] <= 10)*100:.3f}%"
        )

        print(
            f"  <=20° agreement : "
            f"{np.mean(subset['abs_difference_deg'] <= 20)*100:.3f}%"
        )


# =========================================================
# Suspicious population
#
# Large Heading change AND poor agreement
# =========================================================

print()
print("=" * 90)
print("SUSPICIOUS POPULATIONS")
print("=" * 90)

print()


conditions = [

    (
        "|Heading| >= 45° "
        "AND |difference| > 20°",

        (
            df["abs_heading_change_deg"] >= 45
        )
        &
        (
            df["abs_difference_deg"] > 20
        )
    ),

    (
        "|Heading| >= 90° "
        "AND |difference| > 20°",

        (
            df["abs_heading_change_deg"] >= 90
        )
        &
        (
            df["abs_difference_deg"] > 20
        )
    ),

    (
        "|Heading| >= 120° "
        "AND |difference| > 20°",

        (
            df["abs_heading_change_deg"] >= 120
        )
        &
        (
            df["abs_difference_deg"] > 20
        )
    ),

    (
        "|Heading| >= 150° "
        "AND |difference| > 20°",

        (
            df["abs_heading_change_deg"] >= 150
        )
        &
        (
            df["abs_difference_deg"] > 20
        )
    )

]


for description, condition in conditions:

    subset = df[condition]


    print(
        f"{description}"
    )

    print(
        f"  Count      : {len(subset)}"
    )

    print(
        f"  Percentage : "
        f"{len(subset)/len(df)*100:.4f}%"
    )

    print()


# =========================================================
# Top 30 worst windows
# =========================================================

print("=" * 90)
print("TOP 30 WORST WINDOWS")
print("=" * 90)

print()

top = df.sort_values(
    "abs_difference_deg",
    ascending=False
).head(30)


print(
    top[
        [
            "dataset",
            "window",
            "heading_change_deg",
            "yawrate_change_deg",
            "difference_deg",
            "abs_difference_deg"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# Overall statistics
# =========================================================

print()
print("=" * 90)
print("OVERALL STATISTICS")
print("=" * 90)

print()

for column, name in [

    (
        "abs_heading_change_deg",
        "|Heading change|"
    ),

    (
        "abs_yawrate_change_deg",
        "|Yaw-rate change|"
    ),

    (
        "abs_difference_deg",
        "|Heading - Yaw-rate|"
    )

]:

    values = df[column]


    print(name)

    print(
        f"  Median : "
        f"{values.median():.3f}°"
    )

    print(
        f"  P90    : "
        f"{values.quantile(.90):.3f}°"
    )

    print(
        f"  P95    : "
        f"{values.quantile(.95):.3f}°"
    )

    print(
        f"  P99    : "
        f"{values.quantile(.99):.3f}°"
    )

    print(
        f"  Max    : "
        f"{values.max():.3f}°"
    )

    print()


print("=" * 90)
print("STEP 22F COMPLETE")
print("=" * 90)

print()

print(
    "Saved:"
)

print(
    output_file
)

print()

print(
    "No data was removed or modified."
)
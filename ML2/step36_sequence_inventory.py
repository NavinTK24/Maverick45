from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(r"D:\Maverick\ML2")

METADATA_FILE = (
    ROOT / "avnet_yaw_reference_metadata.csv"
)

OUTPUT_FILE = (
    ROOT / "step36_sequence_inventory.csv"
)

SUMMARY_FILE = (
    ROOT / "step36_sequence_inventory_summary.txt"
)


print("=" * 75)
print("STEP 36 - FINAL SEQUENCE INVENTORY")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    METADATA_FILE
)

print()
print(
    f"Total validated samples: {len(df):,}"
)


required = [
    "sample_index",
    "dataset",
    "window_index",
    "start_row",
    "end_row",
    "reference_delta_yaw_deg",
    "reference_source",
    "reference_confidence"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# SEQUENCE INVENTORY
# ============================================================

groups = []

for dataset, g in df.groupby(
    "dataset",
    sort=True
):

    g = g.sort_values(
        "window_index"
    )

    sample_count = len(g)

    first_window = int(
        g["window_index"].iloc[0]
    )

    last_window = int(
        g["window_index"].iloc[-1]
    )

    first_row = int(
        g["start_row"].iloc[0]
    )

    last_row = int(
        g["end_row"].iloc[-1]
    )

    # --------------------------------------------------------
    # Check whether windows are continuous
    # --------------------------------------------------------

    window_indices = (
        g["window_index"]
        .to_numpy()
    )

    window_diffs = np.diff(
        window_indices
    )

    continuous = bool(
        np.all(window_diffs == 1)
    )

    # --------------------------------------------------------
    # Number of missing windows
    # --------------------------------------------------------

    expected_windows = (
        last_window -
        first_window +
        1
    )

    missing_windows = (
        expected_windows -
        sample_count
    )

    # --------------------------------------------------------
    # Yaw statistics
    # --------------------------------------------------------

    yaw = g[
        "reference_delta_yaw_deg"
    ].to_numpy()

    abs_yaw = np.abs(yaw)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = g[
        "reference_confidence"
    ].to_numpy()

    # --------------------------------------------------------
    # Reference sources
    # --------------------------------------------------------

    source_counts = (
        g["reference_source"]
        .value_counts()
    )

    dominant_source = (
        source_counts.index[0]
    )

    dominant_source_count = (
        source_counts.iloc[0]
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    groups.append({

        "dataset":
            dataset,

        "samples":
            sample_count,

        "first_window":
            first_window,

        "last_window":
            last_window,

        "first_row":
            first_row,

        "last_row":
            last_row,

        "expected_windows":
            expected_windows,

        "missing_windows":
            missing_windows,

        "continuous":
            continuous,

        "yaw_min_deg":
            yaw.min(),

        "yaw_max_deg":
            yaw.max(),

        "yaw_median_deg":
            np.median(yaw),

        "yaw_p90_deg":
            np.percentile(yaw, 90),

        "yaw_p95_deg":
            np.percentile(yaw, 95),

        "yaw_p99_deg":
            np.percentile(yaw, 99),

        "abs_yaw_ge_45":
            int(
                np.sum(abs_yaw >= 45)
            ),

        "abs_yaw_ge_90":
            int(
                np.sum(abs_yaw >= 90)
            ),

        "mean_confidence":
            np.mean(confidence),

        "min_confidence":
            np.min(confidence),

        "dominant_reference_source":
            dominant_source,

        "dominant_source_count":
            int(
                dominant_source_count
            ),

        "dominant_source_percent":
            100.0 *
            dominant_source_count /
            sample_count
    })


inventory = pd.DataFrame(
    groups
)

inventory = inventory.sort_values(
    "samples",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 75)
print("SEQUENCE INVENTORY")
print("=" * 75)

print()

print(
    inventory[
        [
            "dataset",
            "samples",
            "first_window",
            "last_window",
            "continuous",
            "missing_windows",
            "mean_confidence"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print()
print("=" * 75)
print("GLOBAL STATISTICS")
print("=" * 75)

print(
    f"Number of sequences : "
    f"{len(inventory)}"
)

print(
    f"Total samples       : "
    f"{inventory['samples'].sum():,}"
)

print(
    f"Largest sequence    : "
    f"{inventory['samples'].max():,}"
)

print(
    f"Smallest sequence   : "
    f"{inventory['samples'].min():,}"
)

print(
    f"Median sequence     : "
    f"{inventory['samples'].median():.1f}"
)

print(
    f"Mean sequence       : "
    f"{inventory['samples'].mean():.1f}"
)

print(
    f"Continuous sequences: "
    f"{inventory['continuous'].sum()}"
)

print(
    f"Non-continuous      : "
    f"{(~inventory['continuous']).sum()}"
)

print(
    f"Total missing windows: "
    f"{inventory['missing_windows'].sum():,}"
)


# ============================================================
# SORTING VIEWS
# ============================================================

print()
print("=" * 75)
print("LARGEST SEQUENCES")
print("=" * 75)

print(
    inventory[
        [
            "dataset",
            "samples"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("=" * 75)
print("SMALLEST SEQUENCES")
print("=" * 75)

print(
    inventory[
        [
            "dataset",
            "samples"
        ]
    ]
    .sort_values(
        "samples"
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

inventory.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 36 - FINAL SEQUENCE INVENTORY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Number of sequences: "
        f"{len(inventory)}\n"
    )

    f.write(
        f"Total samples: "
        f"{inventory['samples'].sum():,}\n"
    )

    f.write(
        f"Largest sequence: "
        f"{inventory['samples'].max():,}\n"
    )

    f.write(
        f"Smallest sequence: "
        f"{inventory['samples'].min():,}\n"
    )

    f.write(
        f"Median sequence size: "
        f"{inventory['samples'].median():.1f}\n"
    )

    f.write(
        f"Mean sequence size: "
        f"{inventory['samples'].mean():.1f}\n"
    )

    f.write(
        f"Continuous sequences: "
        f"{inventory['continuous'].sum()}\n"
    )

    f.write(
        f"Non-continuous sequences: "
        f"{(~inventory['continuous']).sum()}\n"
    )

    f.write(
        f"Missing windows: "
        f"{inventory['missing_windows'].sum():,}\n"
    )


print()
print("=" * 75)
print("STEP 36A COMPLETE")
print("=" * 75)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Saved: {SUMMARY_FILE}"
)
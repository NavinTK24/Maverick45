from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(r"D:\Maverick\ML2")

METADATA_FILE = (
    ROOT / "avnet_yaw_reference_metadata.csv"
)

INVENTORY_FILE = (
    ROOT / "step36_sequence_inventory.csv"
)

OUTPUT_FILE = (
    ROOT / "step37_split_distribution.csv"
)

SUMMARY_FILE = (
    ROOT / "step37_split_distribution_summary.txt"
)

SEED = 246


# ============================================================
# LOAD
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

inventory = pd.read_csv(
    INVENTORY_FILE
)

total = len(metadata)

counts = dict(
    zip(
        inventory["dataset"],
        inventory["samples"]
    )
)

names = inventory[
    "dataset"
].tolist()


# ============================================================
# RECREATE SEED 246 SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

shuffled = np.array(names)

rng.shuffle(
    shuffled
)

target_train = 0.70 * total
target_val = 0.15 * total
target_test = 0.15 * total

train = []
validation = []
test = []

train_count = 0
validation_count = 0
test_count = 0


for name in shuffled:

    n = counts[name]

    remaining = [
        target_train - train_count,
        target_val - validation_count,
        target_test - test_count
    ]

    choice = int(
        np.argmax(remaining)
    )

    if choice == 0:

        train.append(name)
        train_count += n

    elif choice == 1:

        validation.append(name)
        validation_count += n

    else:

        test.append(name)
        test_count += n


split_map = {}

for name in train:
    split_map[name] = "TRAIN"

for name in validation:
    split_map[name] = "VALIDATION"

for name in test:
    split_map[name] = "TEST"


metadata["split"] = (
    metadata["dataset"]
    .map(split_map)
)


# ============================================================
# CHECK
# ============================================================

if metadata["split"].isna().any():

    missing = metadata[
        metadata["split"].isna()
    ]["dataset"].unique()

    raise RuntimeError(
        "Some datasets were not assigned:\n"
        + "\n".join(missing)
    )


# ============================================================
# STATISTICS
# ============================================================

rows = []


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    g = metadata[
        metadata["split"] == split
    ]

    yaw = g[
        "reference_delta_yaw_deg"
    ].to_numpy()

    abs_yaw = np.abs(yaw)

    confidence = g[
        "reference_confidence"
    ].to_numpy()

    source_counts = (
        g["reference_source"]
        .value_counts()
    )

    dominant_source = (
        source_counts.index[0]
    )

    rows.append({

        "split":
            split,

        "sequences":
            g["dataset"].nunique(),

        "samples":
            len(g),

        "percentage":
            100 * len(g) / total,

        "yaw_mean_deg":
            np.mean(yaw),

        "yaw_std_deg":
            np.std(yaw),

        "yaw_median_deg":
            np.median(yaw),

        "yaw_p90_deg":
            np.percentile(yaw, 90),

        "yaw_p95_deg":
            np.percentile(yaw, 95),

        "yaw_p99_deg":
            np.percentile(yaw, 99),

        "abs_yaw_ge_5":
            np.sum(abs_yaw >= 5),

        "abs_yaw_ge_10":
            np.sum(abs_yaw >= 10),

        "abs_yaw_ge_20":
            np.sum(abs_yaw >= 20),

        "abs_yaw_ge_45":
            np.sum(abs_yaw >= 45),

        "confidence_mean":
            np.mean(confidence),

        "confidence_median":
            np.median(confidence),

        "confidence_min":
            np.min(confidence),

        "dominant_reference_source":
            dominant_source,

        "dominant_source_percent":
            100 *
            source_counts.iloc[0] /
            len(g)
    })


result = pd.DataFrame(
    rows
)


# ============================================================
# PRINT
# ============================================================

print("=" * 75)
print("STEP 37 - SPLIT DISTRIBUTION CHECK")
print("=" * 75)

print()

print(
    result.to_string(
        index=False
    )
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("REFERENCE SOURCE DISTRIBUTION")
print("=" * 75)

source_table = pd.crosstab(
    metadata["reference_source"],
    metadata["split"]
)

print()

print(
    source_table.to_string()
)


# ============================================================
# SEQUENCE DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("SEQUENCE COUNTS")
print("=" * 75)

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    g = metadata[
        metadata["split"] == split
    ]

    print()
    print(
        split,
        ":"
    )

    print(
        g.groupby(
            "dataset"
        ).size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 37 - SPLIT DISTRIBUTION CHECK\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        result.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "Seed: 246\n"
    )

    f.write(
        "Sequence-level split.\n"
    )

    f.write(
        "No individual sequence is divided between splits.\n"
    )


print()
print("=" * 75)
print("STEP 37 COMPLETE")
print("=" * 75)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Saved: {SUMMARY_FILE}"
)
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 19
# Create TRAIN / VALIDATION / TEST split by complete dataset
#
# IMPORTANT:
#   Windows from the same driving dataset are NEVER separated.
#
# Split target:
#   ~70% train
#   ~15% validation
#   ~15% test
#
# The split is deterministic.
# ============================================================

ML2 = Path(r"D:\Maverick\ML2")

DATASET_FILE = ML2 / "avnet_yaw_only_dataset.npz"

OUTPUT_FILE = ML2 / "avnet_dataset_split.csv"

# ------------------------------------------------------------
# Load combined dataset
# ------------------------------------------------------------

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

dataset_names = data["dataset_names"]

window_indices = data["window_indices"]

total_samples = len(dataset_names)

print("Total samples:", total_samples)

# ------------------------------------------------------------
# Count windows per dataset
# ------------------------------------------------------------

counts = (
    pd.Series(dataset_names)
    .value_counts()
    .sort_index()
)

print("\nDatasets:", len(counts))

print("\nWINDOW COUNTS")
print("------------------------------------------------")

for name, count in counts.items():
    print(f"{name:8s} : {count:6d}")


# ------------------------------------------------------------
# Deterministic assignment
#
# We use a greedy balancing method.
#
# Each complete dataset is assigned to the split
# that currently has the smallest number of samples,
# while respecting approximate 70/15/15 targets.
# ------------------------------------------------------------

targets = {
    "train": total_samples * 0.70,
    "validation": total_samples * 0.15,
    "test": total_samples * 0.15,
}

split_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

# Process largest datasets first.
ordered_datasets = (
    counts
    .sort_values(ascending=False)
    .index
    .tolist()
)

assignments = {}

for dataset in ordered_datasets:

    count = int(counts[dataset])

    # Calculate how far each split is from its target.
    deficits = {
        split: targets[split] - split_counts[split]
        for split in targets
    }

    # Prefer splits that still have the largest deficit.
    # Only use a split if adding this complete dataset does
    # not make the imbalance unnecessarily large.
    selected = max(
        deficits,
        key=deficits.get
    )

    assignments[dataset] = selected

    split_counts[selected] += count


# ------------------------------------------------------------
# Create split table
# ------------------------------------------------------------

rows = []

for dataset in sorted(assignments):

    split = assignments[dataset]

    rows.append({
        "dataset": dataset,
        "windows": int(counts[dataset]),
        "split": split,
    })

split_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Add split to every individual sample
# ------------------------------------------------------------

split_lookup = dict(
    zip(
        split_df["dataset"],
        split_df["split"]
    )
)

sample_split = np.array([
    split_lookup[name]
    for name in dataset_names
])

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print("\n============================================================")
print("SPLIT SUMMARY")
print("============================================================")

for split in ["train", "validation", "test"]:

    mask = sample_split == split

    n = int(mask.sum())

    percentage = (
        n / total_samples * 100
    )

    datasets_in_split = sorted(
        set(dataset_names[mask])
    )

    print(
        f"{split:10s}: "
        f"{n:7d} windows "
        f"({percentage:6.2f}%) "
        f"| {len(datasets_in_split)} datasets"
    )

# ------------------------------------------------------------
# Verify no dataset is in multiple splits
# ------------------------------------------------------------

print("\n============================================================")
print("DATASET SEPARATION CHECK")
print("============================================================")

overlap_found = False

for dataset in sorted(set(dataset_names)):

    splits = set(
        sample_split[
            dataset_names == dataset
        ]
    )

    if len(splits) != 1:

        print(
            f"[ERROR] {dataset} appears in {splits}"
        )

        overlap_found = True

if not overlap_found:
    print(
        "PASS: Every dataset belongs to exactly one split."
    )

# ------------------------------------------------------------
# Create per-sample split information
# ------------------------------------------------------------

sample_df = pd.DataFrame({
    "dataset": dataset_names,
    "window_index": window_indices,
    "split": sample_split,
})

# ------------------------------------------------------------
# Save both dataset-level and sample-level information
# ------------------------------------------------------------

split_df.to_csv(
    OUTPUT_FILE,
    index=False
)

sample_output = (
    ML2 / "avnet_sample_split.csv"
)

sample_df.to_csv(
    sample_output,
    index=False
)

# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n============================================================")
print("STEP 19 COMPLETE")
print("============================================================")

print(
    f"Dataset-level split:\n{OUTPUT_FILE}"
)

print(
    f"Sample-level split:\n{sample_output}"
)

print("\nSplit counts:")

for split, count in split_counts.items():
    print(
        f"  {split:10s}: {count:,}"
    )

print("\nNo IMU values or targets were modified.")

print("============================================================")
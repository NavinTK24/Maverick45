import numpy as np
import pandas as pd
from pathlib import Path


print("=" * 75)
print("STEP 45 - PREPARE DDODO VELOCITY DATA")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\Maverick\ML2")

NORMALIZED_DIR = ROOT / "step44_normalized_data"

SPLIT_FILE = (
    ROOT /
    "step38_avnet_data" /
    "sequence_split.csv"
)

METADATA_FILE = (
    ROOT /
    "avnet_yaw_reference_metadata.csv"
)

OUTPUT_DIR = (
    ROOT /
    "step45_ddodo_data"
)

OUTPUT_DIR.mkdir(exist_ok=True)


DATASET_ROOT = Path(
    r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
)


# ============================================================
# LOAD NORMALIZED INPUTS
# ============================================================

print("\nLoading normalized IMU datasets...")

train_data = np.load(
    NORMALIZED_DIR / "train.npz"
)

val_data = np.load(
    NORMALIZED_DIR / "validation.npz"
)

test_data = np.load(
    NORMALIZED_DIR / "test.npz"
)

X_train = train_data["X"]
X_val = val_data["X"]
X_test = test_data["X"]

print("Train X:", X_train.shape)
print("Validation X:", X_val.shape)
print("Test X:", X_test.shape)


# ============================================================
# LOAD SEQUENCE SPLIT
# ============================================================

print("\nLoading sequence split...")

split_df = pd.read_csv(SPLIT_FILE)

print("Columns:")
print(list(split_df.columns))

print("\nSequence split:")

print(split_df)


# ============================================================
# CORRECT COLUMN NAME
# ============================================================

if "dataset" not in split_df.columns:
    raise RuntimeError(
        "Expected 'dataset' column in sequence_split.csv"
    )

if "split" not in split_df.columns:
    raise RuntimeError(
        "Expected 'split' column in sequence_split.csv"
    )


# Convert split labels to lowercase

sequence_to_split = {}

for _, row in split_df.iterrows():

    dataset = str(
        row["dataset"]
    ).strip()

    split = str(
        row["split"]
    ).strip().lower()

    sequence_to_split[dataset] = split


# ============================================================
# SHOW SPLIT COUNTS
# ============================================================

print("\n" + "=" * 75)
print("SEQUENCE PARTITION")
print("=" * 75)

for split in [
    "train",
    "validation",
    "test"
]:

    sequences = [
        dataset
        for dataset, assigned_split
        in sequence_to_split.items()
        if assigned_split == split
    ]

    print(
        f"{split.upper():12s}: "
        f"{len(sequences)} sequences"
    )

    print(
        "  " +
        ", ".join(sequences)
    )


# ============================================================
# LOAD AVNET METADATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING AVNET METADATA")
print("=" * 75)

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    "Metadata shape:",
    metadata.shape
)

print(
    "\nMetadata columns:"
)

print(
    list(metadata.columns)
)


# ============================================================
# REQUIRED METADATA COLUMNS
# ============================================================

required_columns = [
    "dataset",
    "window_index",
    "start_row",
    "end_row"
]

missing = [
    c for c in required_columns
    if c not in metadata.columns
]

if missing:

    raise RuntimeError(
        f"Missing metadata columns: {missing}"
    )


# ============================================================
# VERIFY EVERY DATASET HAS A SPLIT
# ============================================================

metadata_datasets = set(
    metadata["dataset"]
    .astype(str)
    .unique()
)

split_datasets = set(
    sequence_to_split.keys()
)

missing_split = (
    metadata_datasets -
    split_datasets
)

if missing_split:

    raise RuntimeError(
        "These metadata datasets have no split: "
        + str(sorted(missing_split))
    )


print(
    "\nAll metadata datasets have a split."
)


# ============================================================
# FIND V FILE
# ============================================================

def find_v_file(dataset):

    target = f"V-{dataset}.csv"

    # Exact recursive filename match
    matches = [
        p
        for p in DATASET_ROOT.rglob("*")
        if p.is_file()
        and p.name.lower() == target.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"V file not found for dataset '{dataset}' "
            f"(expected {target})"
        )

    raise RuntimeError(
        f"Multiple V files found for '{dataset}': "
        f"{matches}"
    )


# ============================================================
# LOAD VELOCITY FOR ALL DATASETS
# ============================================================

print("\n" + "=" * 75)
print("LOADING VEHICLE VELOCITY")
print("=" * 75)

velocity_by_dataset = {}

for dataset in sorted(metadata_datasets):

    v_file = find_v_file(dataset)

    try:

        v_df = pd.read_csv(
            v_file,
            header=None,
            encoding="cp1252"
        )

    except UnicodeDecodeError:

        v_df = pd.read_csv(
            v_file,
            header=None,
            encoding="latin1"
        )


    # --------------------------------------------------------
    # V column 5 = Velocity (km/hr)
    #
    # Python index 4
    # --------------------------------------------------------

    velocity = pd.to_numeric(
        v_df.iloc[:, 4],
        errors="coerce"
    ).to_numpy(
        dtype=np.float32
    )


    velocity_by_dataset[dataset] = velocity


    print(
        f"{dataset:10s} : "
        f"rows={len(velocity):7d}  "
        f"min={np.nanmin(velocity):8.3f}  "
        f"max={np.nanmax(velocity):8.3f}"
    )


# ============================================================
# BUILD TARGETS FOR A SPECIFIC SPLIT
# ============================================================

def build_targets(split_name):

    # Metadata in its ORIGINAL order
    # is important because Step 38 uses this same dataset
    # ordering when creating its partitions.

    subset = metadata[
        metadata["dataset"]
        .astype(str)
        .map(
            lambda d:
            sequence_to_split[d] == split_name
        )
    ].copy()

    print(
        f"\n{split_name.upper()} metadata rows:",
        len(subset)
    )


    targets = []

    target_metadata_rows = []

    for metadata_index, row in subset.iterrows():

        dataset = str(
            row["dataset"]
        )

        end_row = int(
            row["end_row"]
        )

        velocity = velocity_by_dataset[
            dataset
        ]


        # ----------------------------------------------------
        # Target = velocity at END of 1-second window
        # ----------------------------------------------------

        if end_row < 0:
            raise RuntimeError(
                f"Negative end_row: {end_row}"
            )

        if end_row >= len(velocity):

            raise RuntimeError(
                f"{dataset}: "
                f"end_row={end_row} "
                f"but V rows={len(velocity)}"
            )


        value = velocity[end_row]


        if not np.isfinite(value):

            raise RuntimeError(
                f"Non-finite velocity at "
                f"{dataset}, row {end_row}"
            )


        targets.append(
            float(value)
        )

        target_metadata_rows.append(
            metadata_index
        )


    Y = np.asarray(
        targets,
        dtype=np.float32
    ).reshape(-1, 1)


    return subset, Y


# ============================================================
# BUILD ALL THREE TARGET SETS
# ============================================================

results = {}

for split_name in [
    "train",
    "validation",
    "test"
]:

    subset, Y = build_targets(
        split_name
    )

    results[split_name] = (
        subset,
        Y
    )


# ============================================================
# SAMPLE COUNT CHECK
# ============================================================

print("\n" + "=" * 75)
print("SAMPLE COUNT CHECK")
print("=" * 75)

expected = {
    "train": len(X_train),
    "validation": len(X_val),
    "test": len(X_test)
}

for split_name in [
    "train",
    "validation",
    "test"
]:

    subset, Y = results[
        split_name
    ]

    x_count = expected[
        split_name
    ]

    y_count = len(Y)

    print(
        f"{split_name.upper():12s}: "
        f"X={x_count:6d}  "
        f"Y={y_count:6d}"
    )

    if x_count != y_count:

        raise RuntimeError(
            f"{split_name}: X/Y count mismatch!"
        )


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("DDODO TARGET STATISTICS")
print("=" * 75)

for split_name in [
    "train",
    "validation",
    "test"
]:

    _, Y = results[
        split_name
    ]

    print(
        f"\n{split_name.upper()}"
    )

    print(
        f"min    : {np.min(Y):.6f} km/h"
    )

    print(
        f"max    : {np.max(Y):.6f} km/h"
    )

    print(
        f"mean   : {np.mean(Y):.6f} km/h"
    )

    print(
        f"median : {np.median(Y):.6f} km/h"
    )

    print(
        f"std    : {np.std(Y):.6f} km/h"
    )

    print(
        "NaN    :",
        np.isnan(Y).sum()
    )

    print(
        "Inf    :",
        np.isinf(Y).sum()
    )


# ============================================================
# FIRST TARGETS
# ============================================================

print("\n" + "=" * 75)
print("FIRST 10 DDODO TARGETS")
print("=" * 75)

_, train_Y = results["train"]

for i in range(
    min(10, len(train_Y))
):

    print(
        f"{i:4d}: "
        f"{train_Y[i, 0]:.6f} km/h"
    )


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 75)
print("SAVING DDODO DATA")
print("=" * 75)


X_by_split = {
    "train": X_train,
    "validation": X_val,
    "test": X_test
}


for split_name in [
    "train",
    "validation",
    "test"
]:

    X = X_by_split[
        split_name
    ]

    _, Y = results[
        split_name
    ]

    output_file = (
        OUTPUT_DIR /
        f"{split_name}.npz"
    )

    np.savez(
        output_file,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32)
    )

    print(
        f"{split_name.upper():12s}: "
        f"X={X.shape}, "
        f"Y={Y.shape}"
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 45 COMPLETE")
print("=" * 75)

print(
    "\nDDODO target:"
)

print(
    "V dataset column 5 = Velocity (km/hr)"
)

print(
    "Target sample = velocity at the END "
    "of each 1-second window."
)

print(
    "\nThe exact Step 38 sequence split was preserved."
)

print(
    "\nNo new random split was created."
)
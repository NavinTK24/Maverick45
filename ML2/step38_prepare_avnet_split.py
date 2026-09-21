from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Maverick\ML2")

DATASET_FILE = (
    ROOT / "avnet_yaw_reference_dataset.npz"
)

METADATA_FILE = (
    ROOT / "avnet_yaw_reference_metadata.csv"
)

OUTPUT_DIR = (
    ROOT / "step38_avnet_data"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


print("=" * 75)
print("STEP 38 - PREPARE FINAL AVNET SPLITS")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

data = np.load(
    DATASET_FILE
)

X = data["X"]
Y = data["Y"]

metadata = pd.read_csv(
    METADATA_FILE
)


print()
print(
    f"Complete X: {X.shape}"
)

print(
    f"Complete Y: {Y.shape}"
)

print(
    f"Metadata:   {metadata.shape}"
)


# ============================================================
# RECREATE FROZEN SEED-246 SPLIT
# ============================================================

inventory = (
    metadata[
        [
            "dataset"
        ]
    ]
    .drop_duplicates()
)

# Number of samples per sequence
sequence_counts = (
    metadata
    .groupby("dataset")
    .size()
    .to_dict()
)

sequence_names = (
    inventory[
        "dataset"
    ]
    .tolist()
)

total = len(metadata)

rng = np.random.default_rng(
    246
)

shuffled = np.array(
    sequence_names
)

rng.shuffle(
    shuffled
)

target_train = 0.70 * total
target_val = 0.15 * total
target_test = 0.15 * total

train_sequences = []
validation_sequences = []
test_sequences = []

train_count = 0
validation_count = 0
test_count = 0


for dataset in shuffled:

    n = sequence_counts[
        dataset
    ]

    remaining = [

        target_train -
        train_count,

        target_val -
        validation_count,

        target_test -
        test_count
    ]

    choice = int(
        np.argmax(
            remaining
        )
    )

    if choice == 0:

        train_sequences.append(
            dataset
        )

        train_count += n

    elif choice == 1:

        validation_sequences.append(
            dataset
        )

        validation_count += n

    else:

        test_sequences.append(
            dataset
        )

        test_count += n


# ============================================================
# CREATE MASKS
# ============================================================

train_mask = (
    metadata["dataset"]
    .isin(train_sequences)
    .to_numpy()
)

validation_mask = (
    metadata["dataset"]
    .isin(validation_sequences)
    .to_numpy()
)

test_mask = (
    metadata["dataset"]
    .isin(test_sequences)
    .to_numpy()
)


# ============================================================
# VERIFY PARTITION
# ============================================================

if (
    train_mask
    & validation_mask
).any():

    raise RuntimeError(
        "Train/validation overlap."
    )


if (
    train_mask
    & test_mask
).any():

    raise RuntimeError(
        "Train/test overlap."
    )


if (
    validation_mask
    & test_mask
).any():

    raise RuntimeError(
        "Validation/test overlap."
    )


combined = (
    train_mask.astype(int)
    + validation_mask.astype(int)
    + test_mask.astype(int)
)

if not np.all(
    combined == 1
):

    raise RuntimeError(
        "Some samples are not assigned "
        "to exactly one partition."
    )


# ============================================================
# EXTRACT
# ============================================================

X_train = X[
    train_mask
]

Y_train = Y[
    train_mask
]

X_validation = X[
    validation_mask
]

Y_validation = Y[
    validation_mask
]

X_test = X[
    test_mask
]

Y_test = Y[
    test_mask
]


metadata_train = metadata[
    train_mask
].copy()

metadata_validation = metadata[
    validation_mask
].copy()

metadata_test = metadata[
    test_mask
].copy()


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 75)
print("FINAL PARTITIONS")
print("=" * 75)

print()

print(
    f"TRAIN       : "
    f"X={X_train.shape}, "
    f"Y={Y_train.shape}"
)

print(
    f"VALIDATION  : "
    f"X={X_validation.shape}, "
    f"Y={Y_validation.shape}"
)

print(
    f"TEST        : "
    f"X={X_test.shape}, "
    f"Y={Y_test.shape}"
)


# ============================================================
# SAVE ARRAYS
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "train.npz",
    X=X_train,
    Y=Y_train
)

np.savez_compressed(
    OUTPUT_DIR / "validation.npz",
    X=X_validation,
    Y=Y_validation
)

np.savez_compressed(
    OUTPUT_DIR / "test.npz",
    X=X_test,
    Y=Y_test
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_train.to_csv(
    OUTPUT_DIR / "train_metadata.csv",
    index=False
)

metadata_validation.to_csv(
    OUTPUT_DIR / "validation_metadata.csv",
    index=False
)

metadata_test.to_csv(
    OUTPUT_DIR / "test_metadata.csv",
    index=False
)


# ============================================================
# SAVE SPLIT DEFINITION
# ============================================================

split_rows = []

for dataset in train_sequences:

    split_rows.append({
        "dataset": dataset,
        "split": "TRAIN",
        "samples": sequence_counts[dataset]
    })

for dataset in validation_sequences:

    split_rows.append({
        "dataset": dataset,
        "split": "VALIDATION",
        "samples": sequence_counts[dataset]
    })

for dataset in test_sequences:

    split_rows.append({
        "dataset": dataset,
        "split": "TEST",
        "samples": sequence_counts[dataset]
    })


split_df = pd.DataFrame(
    split_rows
)

split_df.to_csv(
    OUTPUT_DIR / "sequence_split.csv",
    index=False
)


# ============================================================
# FINAL CHECKS
# ============================================================

print()
print("=" * 75)
print("CHECKS")
print("=" * 75)

print(
    "Total samples:"
)

print(
    len(X_train)
    + len(X_validation)
    + len(X_test)
)

print(
    "Expected:"
)

print(
    len(X)
)

print()

print(
    "Train sequences:",
    len(train_sequences)
)

print(
    "Validation sequences:",
    len(validation_sequences)
)

print(
    "Test sequences:",
    len(test_sequences)
)

print()

print(
    "Test sequence list:"
)

for dataset in sorted(
    test_sequences
):

    print(
        f"  {dataset}"
    )


# ============================================================
# SUMMARY
# ============================================================

with open(
    OUTPUT_DIR /
    "split_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 38 - FINAL AVNET SPLIT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Train samples: {len(X_train):,}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(X_validation):,}\n"
    )

    f.write(
        f"Test samples: {len(X_test):,}\n"
    )

    f.write("\n")

    f.write(
        f"Train sequences: "
        f"{len(train_sequences)}\n"
    )

    f.write(
        f"Validation sequences: "
        f"{len(validation_sequences)}\n"
    )

    f.write(
        f"Test sequences: "
        f"{len(test_sequences)}\n"
    )

    f.write("\nTEST SEQUENCES\n")

    for dataset in sorted(
        test_sequences
    ):

        f.write(
            f"{dataset}: "
            f"{sequence_counts[dataset]:,}\n"
        )


print()
print("=" * 75)
print("STEP 38 COMPLETE")
print("=" * 75)

print()
print(
    f"Output directory: {OUTPUT_DIR}"
)
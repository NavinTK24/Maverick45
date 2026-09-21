from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(r"D:\Maverick\ML2")

INVENTORY = (
    ROOT / "step36_sequence_inventory.csv"
)

OUTPUT = (
    ROOT / "step36b_split_candidates.csv"
)


print("=" * 75)
print("STEP 36B - GENERATE SEQUENCE-LEVEL SPLIT CANDIDATES")
print("=" * 75)


df = pd.read_csv(INVENTORY)

datasets = df["dataset"].tolist()
counts = df["samples"].to_numpy()

total = counts.sum()

print()
print(f"Sequences : {len(datasets)}")
print(f"Samples   : {total:,}")


# ============================================================
# We create deterministic candidate splits.
#
# No individual sequence is divided.
# ============================================================

rng = np.random.default_rng(2026)


def evaluate(train_idx, val_idx, test_idx):

    train = counts[train_idx].sum()
    val = counts[val_idx].sum()
    test = counts[test_idx].sum()

    train_pct = 100 * train / total
    val_pct = 100 * val / total
    test_pct = 100 * test / total

    # Difference from desired 70/15/15
    error = (
        abs(train_pct - 70)
        + abs(val_pct - 15)
        + abs(test_pct - 15)
    )

    return (
        train,
        val,
        test,
        train_pct,
        val_pct,
        test_pct,
        error
    )


candidates = []


# ============================================================
# Candidate generation
# ============================================================

for seed in range(1000):

    rng = np.random.default_rng(seed)

    shuffled = np.array(
        datasets
    )

    rng.shuffle(
        shuffled
    )

    # Greedy assignment toward target sample totals.
    target_train = 0.70 * total
    target_val = 0.15 * total
    target_test = 0.15 * total

    train_names = []
    val_names = []
    test_names = []

    train_count = 0
    val_count = 0
    test_count = 0

    lookup = dict(
        zip(
            df["dataset"],
            df["samples"]
        )
    )

    for name in shuffled:

        n = lookup[name]

        # Choose the partition with the
        # largest remaining target.

        remaining = [
            target_train - train_count,
            target_val - val_count,
            target_test - test_count
        ]

        choice = int(
            np.argmax(remaining)
        )

        if choice == 0:

            train_names.append(name)
            train_count += n

        elif choice == 1:

            val_names.append(name)
            val_count += n

        else:

            test_names.append(name)
            test_count += n

    train_pct = 100 * train_count / total
    val_pct = 100 * val_count / total
    test_pct = 100 * test_count / total

    error = (
        abs(train_pct - 70)
        + abs(val_pct - 15)
        + abs(test_pct - 15)
    )

    candidates.append({

        "seed":
            seed,

        "train_sequences":
            len(train_names),

        "validation_sequences":
            len(val_names),

        "test_sequences":
            len(test_names),

        "train_samples":
            train_count,

        "validation_samples":
            val_count,

        "test_samples":
            test_count,

        "train_percent":
            train_pct,

        "validation_percent":
            val_pct,

        "test_percent":
            test_pct,

        "balance_error":
            error
    })


result = pd.DataFrame(
    candidates
)

result = result.sort_values(
    "balance_error"
).reset_index(
    drop=True
)


# ============================================================
# Show best candidates
# ============================================================

print()
print("=" * 75)
print("BEST CANDIDATES")
print("=" * 75)

print()

print(
    result.head(20).to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)


print()
print("=" * 75)
print("STEP 36B COMPLETE")
print("=" * 75)

print()
print(
    f"Saved: {OUTPUT}"
)
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(r"D:\Maverick\ML2")

INVENTORY = (
    ROOT / "step36_sequence_inventory.csv"
)

SEED = 246


print("=" * 75)
print("STEP 36C - INSPECT SPLIT CANDIDATE")
print("=" * 75)

df = pd.read_csv(INVENTORY)

names = df["dataset"].tolist()
counts = dict(
    zip(
        df["dataset"],
        df["samples"]
    )
)

rng = np.random.default_rng(SEED)

shuffled = np.array(names)
rng.shuffle(shuffled)

total = df["samples"].sum()

target_train = 0.70 * total
target_val = 0.15 * total
target_test = 0.15 * total

train = []
val = []
test = []

train_count = 0
val_count = 0
test_count = 0

for name in shuffled:

    n = counts[name]

    remaining = [
        target_train - train_count,
        target_val - val_count,
        target_test - test_count
    ]

    choice = int(
        np.argmax(remaining)
    )

    if choice == 0:
        train.append(name)
        train_count += n

    elif choice == 1:
        val.append(name)
        val_count += n

    else:
        test.append(name)
        test_count += n


def show_group(title, group, count):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    print(
        f"Sequences: {len(group)}"
    )

    print(
        f"Samples: {count:,}"
    )

    print(
        f"Percentage: {100*count/total:.4f}%"
    )

    print()

    for name in sorted(group):
        print(
            f"{name:15s} {counts[name]:8,d}"
        )


show_group(
    "TRAIN",
    train,
    train_count
)

show_group(
    "VALIDATION",
    val,
    val_count
)

show_group(
    "TEST",
    test,
    test_count
)


print()
print("=" * 75)
print("CHECK")
print("=" * 75)

print(
    "Train + Validation + Test = "
    f"{train_count + val_count + test_count:,}"
)

print(
    "Expected = "
    f"{total:,}"
)

print(
    "Sequence overlap:"
)

print(
    "Train ∩ Validation:",
    set(train) & set(val)
)

print(
    "Train ∩ Test:",
    set(train) & set(test)
)

print(
    "Validation ∩ Test:",
    set(val) & set(test)
)
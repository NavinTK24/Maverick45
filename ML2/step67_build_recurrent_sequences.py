import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = r"D:\Maverick\ML2"

X_FILE = os.path.join(
    ROOT,
    "avnet_yaw_reference_dataset.npz"
)

NORMALIZED_DIR = os.path.join(
    ROOT,
    "step44_normalized_data"
)

DDODO_DIR = os.path.join(
    ROOT,
    "step45_ddodo_data"
)

METADATA_FILE = os.path.join(
    ROOT,
    "avnet_yaw_reference_metadata.csv"
)

RUNS_FILE = os.path.join(
    ROOT,
    "step66_continuous_runs",
    "continuous_runs.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "step67_recurrent_sequences"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 100)
print("STEP 67 - BUILD RECURRENT TEMPORAL SEQUENCES")
print("=" * 100)


# ============================================================
# LOAD COMPLETE DATASET
# ============================================================

print("\nLoading complete AVNet dataset...")

data = np.load(
    X_FILE
)

X_all = data["X"].astype(
    np.float32
)

Y_ddatt_all = data["Y"].astype(
    np.float32
)

print(
    "X:",
    X_all.shape
)

print(
    "DDATT Y:",
    Y_ddatt_all.shape
)


# ============================================================
# LOAD METADATA
# ============================================================

print(
    "\nLoading metadata..."
)

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    "Metadata:",
    metadata.shape
)


# ============================================================
# BASIC ALIGNMENT CHECK
# ============================================================

if len(X_all) != len(metadata):

    raise ValueError(
        "X and metadata row counts do not match!"
    )

if len(Y_ddatt_all) != len(metadata):

    raise ValueError(
        "DDATT Y and metadata row counts do not match!"
    )


# ============================================================
# LOAD RUN INFORMATION
# ============================================================

print(
    "\nLoading continuous runs..."
)

runs = pd.read_csv(
    RUNS_FILE
)

print(
    "Continuous runs:",
    len(runs)
)


required_run_columns = [
    "run_id",
    "dataset",
    "split",
    "num_windows",
    "first_sample_index",
    "last_sample_index"
]

missing = [
    c
    for c in required_run_columns
    if c not in runs.columns
]

if missing:

    raise ValueError(
        "Missing run columns: "
        + str(missing)
    )


# ============================================================
# LOAD DDODO TARGETS
# ============================================================

print(
    "\nLoading DDODO targets..."
)


ddodo_files = {
    "TRAIN": os.path.join(
        DDODO_DIR,
        "train.npz"
    ),

    "VALIDATION": os.path.join(
        DDODO_DIR,
        "validation.npz"
    ),

    "TEST": os.path.join(
        DDODO_DIR,
        "test.npz"
    )
}


# We will reconstruct DDODO using the velocity
# already present in the metadata/source dataset.
#
# The existing Step 45 files are split-specific and
# therefore cannot safely be concatenated according to
# the global sample_index without their metadata.
#
# Instead, use the velocity target from the original
# synchronized V/S data through the existing metadata
# mapping created in earlier steps.


# ============================================================
# FIND EXISTING GLOBAL VELOCITY TARGET
# ============================================================

possible_velocity_files = []

for dirpath, dirnames, filenames in os.walk(ROOT):

    for filename in filenames:

        lower = filename.lower()

        if (
            "velocity" in lower
            and lower.endswith(".csv")
        ):

            possible_velocity_files.append(
                os.path.join(
                    dirpath,
                    filename
                )
            )


print(
    "\nVelocity CSV files found:",
    len(possible_velocity_files)
)


# ============================================================
# IMPORTANT:
# USE EXISTING STEP 45 SPLIT DATA
# ============================================================

# Build a global DDODO vector by loading each split file
# together with its corresponding metadata.
#
# Step 45 saved:
#
# train.npz
# validation.npz
# test.npz
#
# Their sample order follows the Step 38 partition.


ddodo_split_data = {}

for split_name, path in ddodo_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            "\nMissing DDODO file:\n"
            + path
        )

    d = np.load(
        path
    )

    print(
        f"\n{split_name} DDODO file:"
    )

    for key in d.files:

        print(
            " ",
            key,
            d[key].shape
        )

    ddodo_split_data[
        split_name
    ] = d


# ============================================================
# CONSTRUCT GLOBAL SAMPLE -> DDODO MAPPING
# ============================================================

print(
    "\nConstructing global DDODO mapping..."
)


# We use the sequence/dataset ordering from metadata and
# the exact Step 38 split ordering.
#
# The DDODO files were generated in the same partition order,
# so the safest reconstruction is performed by dataset and
# sample order.


global_ddodo = np.full(
    len(metadata),
    np.nan,
    dtype=np.float32
)


for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    split_metadata = metadata[
        metadata["dataset"].isin(
            runs.loc[
                runs["split"] == split_name,
                "dataset"
            ].astype(str)
        )
    ].copy()


    # Preserve global sample order.
    split_metadata = (
        split_metadata
        .sort_values(
            "sample_index"
        )
    )


    ddodo_npz = (
        ddodo_split_data[
            split_name
        ]
    )


    # --------------------------------------------------------
    # Identify target array
    # --------------------------------------------------------

    if "Y" in ddodo_npz.files:

        target = ddodo_npz["Y"]

    elif "y" in ddodo_npz.files:

        target = ddodo_npz["y"]

    elif "velocity" in ddodo_npz.files:

        target = ddodo_npz["velocity"]

    elif "target" in ddodo_npz.files:

        target = ddodo_npz["target"]

    else:

        raise ValueError(
            "Could not identify DDODO target in "
            + split_name
            + ".npz"
        )


    target = np.asarray(
        target
    ).reshape(
        -1
    )


    print(
        f"{split_name}:"
    )

    print(
        "  metadata samples:",
        len(split_metadata)
    )

    print(
        "  DDODO targets:",
        len(target)
    )


    if len(split_metadata) != len(target):

        raise ValueError(
            f"{split_name}: metadata count "
            f"does not match DDODO target count!"
        )


    global_indices = (
        split_metadata[
            "sample_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    global_ddodo[
        global_indices
    ] = target.astype(
        np.float32
    )


# ============================================================
# DDODO ALIGNMENT CHECK
# ============================================================

if np.isnan(
    global_ddodo
).any():

    missing_count = int(
        np.isnan(
            global_ddodo
        ).sum()
    )

    raise ValueError(
        f"DDODO mapping has {missing_count} missing samples!"
    )


print(
    "\nGlobal DDODO mapping: PASS"
)

print(
    "Global DDODO shape:",
    global_ddodo.shape
)


# ============================================================
# GLOBAL DATA CHECK
# ============================================================

if not np.isfinite(
    X_all
).all():

    raise ValueError(
        "X contains NaN/Inf!"
    )

if not np.isfinite(
    Y_ddatt_all
).all():

    raise ValueError(
        "DDATT Y contains NaN/Inf!"
    )

if not np.isfinite(
    global_ddodo
).all():

    raise ValueError(
        "DDODO Y contains NaN/Inf!"
    )


# ============================================================
# CREATE OUTPUT CONTAINERS
# ============================================================

split_outputs = {
    "TRAIN": [],
    "VALIDATION": [],
    "TEST": []
}


# ============================================================
# BUILD EACH CONTINUOUS RUN
# ============================================================

print(
    "\nBuilding recurrent sequences..."
)


for _, run in runs.iterrows():

    run_id = int(
        run["run_id"]
    )

    dataset = str(
        run["dataset"]
    )

    split_name = str(
        run["split"]
    ).upper()

    first_sample = int(
        run["first_sample_index"]
    )

    last_sample = int(
        run["last_sample_index"]
    )

    expected_length = int(
        run["num_windows"]
    )


    # --------------------------------------------------------
    # Locate exact global samples.
    #
    # IMPORTANT:
    # We use sample_index values from the run metadata,
    # not a blind integer range.
    # --------------------------------------------------------

    run_metadata = metadata[
        (metadata["dataset"] == dataset)
        &
        (
            metadata["sample_index"]
            >= first_sample
        )
        &
        (
            metadata["sample_index"]
            <= last_sample
        )
    ].copy()


    run_metadata = (
        run_metadata
        .sort_values(
            "sample_index"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Verify exact run length
    # --------------------------------------------------------

    if len(run_metadata) != expected_length:

        raise ValueError(
            f"Run {run_id} ({dataset}) expected "
            f"{expected_length} windows but found "
            f"{len(run_metadata)}"
        )


    # --------------------------------------------------------
    # Verify consecutive global samples
    # --------------------------------------------------------

    sample_indices = (
        run_metadata[
            "sample_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    if len(sample_indices) > 1:

        if not np.all(
            np.diff(
                sample_indices
            ) == 1
        ):

            raise ValueError(
                f"Run {run_id} ({dataset}) "
                "contains non-consecutive global samples!"
            )


    # --------------------------------------------------------
    # Extract global array positions
    # --------------------------------------------------------

    indices = sample_indices


    X_run = X_all[
        indices
    ]

    Y_ddatt_run = Y_ddatt_all[
        indices
    ]

    Y_ddodo_run = global_ddodo[
        indices
    ]


    # --------------------------------------------------------
    # Final shape checks
    # --------------------------------------------------------

    if X_run.shape != (
        expected_length,
        10,
        6
    ):

        raise ValueError(
            f"Run {run_id} X shape incorrect: "
            + str(X_run.shape)
        )


    if Y_ddatt_run.shape != (
        expected_length,
        3
    ):

        raise ValueError(
            f"Run {run_id} DDATT shape incorrect: "
            + str(Y_ddatt_run.shape)
        )


    if Y_ddodo_run.shape != (
        expected_length,
    ):

        raise ValueError(
            f"Run {run_id} DDODO shape incorrect: "
            + str(Y_ddodo_run.shape)
        )


    # --------------------------------------------------------
    # Store run
    # --------------------------------------------------------

    split_outputs[
        split_name
    ].append(
        {
            "run_id": run_id,
            "dataset": dataset,
            "X": X_run,
            "Y_DDATT": Y_ddatt_run,
            "Y_DDODO": Y_ddodo_run
        }
    )


# ============================================================
# PRINT RUN COUNTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "RECURRENT SEQUENCE COUNTS"
)

print(
    "=" * 100
)


for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    sequences = split_outputs[
        split_name
    ]

    total_windows = sum(
        len(item["X"])
        for item in sequences
    )

    print(
        f"\n{split_name}"
    )

    print(
        "  Runs:",
        len(sequences)
    )

    print(
        "  Windows:",
        total_windows
    )


# ============================================================
# GLOBAL CONSERVATION CHECK
# ============================================================

total_reconstructed = sum(
    sum(
        len(item["X"])
        for item in split_outputs[
            split_name
        ]
    )
    for split_name in split_outputs
)


print(
    "\n"
    + "=" * 100
)

print(
    "GLOBAL CONSERVATION CHECK"
)

print(
    "=" * 100
)

print(
    "Original windows:",
    len(X_all)
)

print(
    "Reconstructed windows:",
    total_reconstructed
)


if total_reconstructed != len(X_all):

    raise ValueError(
        "WINDOW COUNT MISMATCH!"
    )


print(
    "Window count: PASS"
)


# ============================================================
# CHECK THAT EVERY GLOBAL SAMPLE IS USED ONCE
# ============================================================

used_indices = []


for split_name in split_outputs:

    for item in split_outputs[
        split_name
    ]:

        run_id = item["run_id"]

        run = runs[
            runs["run_id"] == run_id
        ].iloc[0]


        run_indices = np.arange(
            int(
                run["first_sample_index"]
            ),
            int(
                run["last_sample_index"]
            ) + 1
        )

        used_indices.extend(
            run_indices.tolist()
        )


used_indices = np.array(
    used_indices,
    dtype=np.int64
)


print(
    "\nUnique reconstructed sample indices:",
    len(
        np.unique(
            used_indices
        )
    )
)

print(
    "Total reconstructed sample indices:",
    len(
        used_indices
    )
)


if (
    len(
        np.unique(
            used_indices
        )
    )
    !=
    len(
        used_indices
    )
):

    raise ValueError(
        "Duplicate sample indices detected!"
    )


if not np.array_equal(
    np.sort(
        used_indices
    ),
    np.arange(
        len(X_all)
    )
):

    raise ValueError(
        "Not every global sample index was represented exactly once!"
    )


print(
    "Sample-index conservation: PASS"
)


# ============================================================
# SAVE EACH RUN AS INDIVIDUAL NPZ
# ============================================================

print(
    "\nSaving individual recurrent sequences..."
)


RUN_DIR = os.path.join(
    OUTPUT_DIR,
    "runs"
)

os.makedirs(
    RUN_DIR,
    exist_ok=True
)


sequence_inventory = []


for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    for item in split_outputs[
        split_name
    ]:

        run_id = item[
            "run_id"
        ]

        dataset = item[
            "dataset"
        ]

        X_run = item[
            "X"
        ]

        Y_att = item[
            "Y_DDATT"
        ]

        Y_odo = item[
            "Y_DDODO"
        ]


        filename = (
            f"run_{run_id:04d}_"
            f"{dataset}_"
            f"{split_name.lower()}.npz"
        )


        output_path = os.path.join(
            RUN_DIR,
            filename
        )


        np.savez_compressed(
            output_path,
            X=X_run,
            Y_DDATT=Y_att,
            Y_DDODO=Y_odo
        )


        sequence_inventory.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "split": split_name,
                "num_windows": len(X_run),
                "X_shape": str(
                    X_run.shape
                ),
                "Y_DDATT_shape": str(
                    Y_att.shape
                ),
                "Y_DDODO_shape": str(
                    Y_odo.shape
                ),
                "file": output_path
            }
        )


# ============================================================
# SAVE INVENTORY
# ============================================================

inventory_df = pd.DataFrame(
    sequence_inventory
)


inventory_file = os.path.join(
    OUTPUT_DIR,
    "recurrent_sequence_inventory.csv"
)

inventory_df.to_csv(
    inventory_file,
    index=False
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

split_summary = []


for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    part = inventory_df[
        inventory_df[
            "split"
        ] == split_name
    ]


    split_summary.append(
        {
            "split": split_name,

            "runs": len(part),

            "windows": int(
                part[
                    "num_windows"
                ].sum()
            ),

            "minimum_run": int(
                part[
                    "num_windows"
                ].min()
            ),

            "median_run": float(
                part[
                    "num_windows"
                ].median()
            ),

            "maximum_run": int(
                part[
                    "num_windows"
                ].max()
            )
        }
    )


split_summary_df = pd.DataFrame(
    split_summary
)


split_summary_file = os.path.join(
    OUTPUT_DIR,
    "recurrent_sequence_split_summary.csv"
)

split_summary_df.to_csv(
    split_summary_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 67 - RECURRENT TEMPORAL SEQUENCE DATASET\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        f"Total original windows: "
        f"{len(X_all):,}\n"
    )

    f.write(
        f"Total recurrent runs: "
        f"{len(runs):,}\n"
    )

    f.write(
        f"Total reconstructed windows: "
        f"{total_reconstructed:,}\n"
    )

    f.write(
        "\n"
    )

    for row in split_summary:

        f.write(
            f"{row['split']}\n"
        )

        f.write(
            f"Runs: {row['runs']}\n"
        )

        f.write(
            f"Windows: {row['windows']}\n"
        )

        f.write(
            f"Minimum run: {row['minimum_run']}\n"
        )

        f.write(
            f"Median run: {row['median_run']}\n"
        )

        f.write(
            f"Maximum run: {row['maximum_run']}\n"
        )

        f.write(
            "\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STEP 67 COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nOriginal windows:",
    len(X_all)
)

print(
    "Reconstructed windows:",
    total_reconstructed
)

print(
    "Total recurrent runs:",
    len(runs)
)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
)

print(
    inventory_file
)

print(
    split_summary_file
)

print(
    summary_file
)

print(
    "\nNo training was performed."
)

print(
    "No existing model was modified."
)
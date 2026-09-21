import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = r"D:\Maverick\ML2"

METADATA_FILE = os.path.join(
    ROOT,
    "avnet_yaw_reference_metadata.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "step66_continuous_runs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 100)
print("STEP 66 - CONTINUOUS VALID TEMPORAL RUN ANALYSIS")
print("=" * 100)


# ============================================================
# FIND sequence_split.csv AUTOMATICALLY
# ============================================================

print("\nSearching for sequence_split.csv...")

split_candidates = []

for dirpath, dirnames, filenames in os.walk(ROOT):

    for filename in filenames:

        if filename.lower() == "sequence_split.csv":

            split_candidates.append(
                os.path.join(
                    dirpath,
                    filename
                )
            )


if len(split_candidates) == 0:

    raise FileNotFoundError(
        "\nCould not find sequence_split.csv anywhere under:\n"
        + ROOT
        + "\n\nPlease check that the Step 38 output exists."
    )


print(
    "\nFound sequence_split.csv:"
)

for path in split_candidates:

    print(
        " ",
        path
    )


# If more than one exists, prefer the one inside a step38 folder.
step38_candidates = [
    p for p in split_candidates
    if "step38" in p.lower()
]


if len(step38_candidates) == 1:

    SPLIT_FILE = step38_candidates[0]

elif len(split_candidates) == 1:

    SPLIT_FILE = split_candidates[0]

else:

    # Prefer the shortest path if multiple copies exist.
    SPLIT_FILE = sorted(
        split_candidates,
        key=lambda p: len(p)
    )[0]


print(
    "\nUsing:"
)

print(
    SPLIT_FILE
)


# ============================================================
# LOAD AVNET METADATA
# ============================================================

print("\nLoading AVNet metadata...")

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    "Metadata rows:",
    len(metadata)
)


required_columns = [
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
    c
    for c in required_columns
    if c not in metadata.columns
]

if missing:

    raise ValueError(
        "Missing metadata columns: "
        + str(missing)
    )


# ============================================================
# LOAD SEQUENCE SPLIT
# ============================================================

print(
    "\nLoading sequence split..."
)

split = pd.read_csv(
    SPLIT_FILE
)

print(
    "Split rows:",
    len(split)
)

required_split_columns = [
    "dataset",
    "split"
]

missing_split = [
    c
    for c in required_split_columns
    if c not in split.columns
]

if missing_split:

    raise ValueError(
        "Missing split columns: "
        + str(missing_split)
    )


# Normalize names
split["dataset"] = (
    split["dataset"]
    .astype(str)
    .str.strip()
)

split["split"] = (
    split["split"]
    .astype(str)
    .str.strip()
    .str.upper()
)


sequence_to_split = dict(
    zip(
        split["dataset"],
        split["split"]
    )
)


# ============================================================
# ASSIGN SPLIT TO EVERY VALID WINDOW
# ============================================================

metadata["dataset"] = (
    metadata["dataset"]
    .astype(str)
    .str.strip()
)

metadata["split"] = (
    metadata["dataset"]
    .map(sequence_to_split)
)


if metadata["split"].isna().any():

    missing_datasets = (
        metadata.loc[
            metadata["split"].isna(),
            "dataset"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "\nDatasets missing from sequence_split.csv:\n"
        + str(missing_datasets)
    )


# ============================================================
# SORT TEMPORALLY
# ============================================================

metadata = (
    metadata
    .sort_values(
        [
            "dataset",
            "window_index"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# BUILD CONTINUOUS RUNS
# ============================================================

print(
    "\nBuilding continuous runs..."
)

runs = []

run_id = 0


for dataset, group in metadata.groupby(
    "dataset",
    sort=False
):

    group = (
        group
        .sort_values(
            "window_index"
        )
        .reset_index(
            drop=True
        )
    )

    window_indices = (
        group[
            "window_index"
        ]
        .to_numpy()
    )

    if len(window_indices) == 0:

        continue


    # --------------------------------------------------------
    # A new temporal run starts whenever window_index
    # is not exactly one greater than the previous window.
    # --------------------------------------------------------

    breaks = (
        np.where(
            np.diff(
                window_indices
            ) != 1
        )[0] + 1
    )


    boundaries = np.concatenate(
        [
            np.array([0]),
            breaks,
            np.array([len(group)])
        ]
    )


    for i in range(
        len(boundaries) - 1
    ):

        start = boundaries[i]

        end = boundaries[i + 1]

        run = group.iloc[
            start:end
        ]

        run_id += 1


        runs.append(
            {
                "run_id": run_id,

                "dataset": dataset,

                "split": run[
                    "split"
                ].iloc[0],

                "num_windows": len(run),

                "first_window_index": int(
                    run[
                        "window_index"
                    ].iloc[0]
                ),

                "last_window_index": int(
                    run[
                        "window_index"
                    ].iloc[-1]
                ),

                "first_sample_index": int(
                    run[
                        "sample_index"
                    ].iloc[0]
                ),

                "last_sample_index": int(
                    run[
                        "sample_index"
                    ].iloc[-1]
                ),

                "first_start_row": int(
                    run[
                        "start_row"
                    ].iloc[0]
                ),

                "last_end_row": int(
                    run[
                        "end_row"
                    ].iloc[-1]
                )
            }
        )


runs_df = pd.DataFrame(
    runs
)


# ============================================================
# BASIC SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "CONTINUOUS RUN SUMMARY"
)

print(
    "=" * 100
)


print(
    "\nTotal valid windows:",
    len(metadata)
)

print(
    "Total continuous runs:",
    len(runs_df)
)

print(
    "Total datasets:",
    metadata[
        "dataset"
    ].nunique()
)


# ============================================================
# WINDOW CONSERVATION CHECK
# ============================================================

run_window_total = int(
    runs_df[
        "num_windows"
    ].sum()
)


print(
    "\nWindows represented by runs:",
    run_window_total
)


if run_window_total != len(metadata):

    raise ValueError(
        "Run window count does not match metadata!"
    )


print(
    "Window conservation check: PASS"
)


# ============================================================
# RUN LENGTH STATISTICS
# ============================================================

lengths = (
    runs_df[
        "num_windows"
    ]
    .to_numpy()
)


print(
    "\nRun length statistics:"
)

print(
    "Minimum:",
    int(
        np.min(lengths)
    )
)

print(
    "Maximum:",
    int(
        np.max(lengths)
    )
)

print(
    "Mean:",
    round(
        float(
            np.mean(lengths)
        ),
        2
    )
)

print(
    "Median:",
    float(
        np.median(lengths)
    )
)

print(
    "P25:",
    float(
        np.percentile(
            lengths,
            25
        )
    )
)

print(
    "P75:",
    float(
        np.percentile(
            lengths,
            75
        )
    )
)

print(
    "P90:",
    float(
        np.percentile(
            lengths,
            90
        )
    )
)

print(
    "P95:",
    float(
        np.percentile(
            lengths,
            95
        )
    )
)

print(
    "P99:",
    float(
        np.percentile(
            lengths,
            99
        )
    )
)


# ============================================================
# RUN LENGTH CATEGORIES
# ============================================================

print(
    "\nRun-length categories:"
)


categories = [
    ("1 window", 1, 1),
    ("2-4 windows", 2, 4),
    ("5-9 windows", 5, 9),
    ("10-19 windows", 10, 19),
    ("20-49 windows", 20, 49),
    ("50-99 windows", 50, 99),
    ("100-499 windows", 100, 499),
    ("500-999 windows", 500, 999),
    ("1000+ windows", 1000, np.inf),
]


category_rows = []


for name, low, high in categories:

    mask = (
        (runs_df["num_windows"] >= low)
        &
        (runs_df["num_windows"] <= high)
    )

    run_count = int(
        mask.sum()
    )

    window_count = int(
        runs_df.loc[
            mask,
            "num_windows"
        ].sum()
    )


    category_rows.append(
        {
            "category": name,
            "runs": run_count,
            "windows": window_count
        }
    )


    print(
        f"{name:20s} "
        f"runs={run_count:5d} "
        f"windows={window_count:7d}"
    )


category_df = pd.DataFrame(
    category_rows
)


# ============================================================
# SPLIT-WISE ANALYSIS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SPLIT-WISE CONTINUOUS RUNS"
)

print(
    "=" * 100
)


for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST"
]:

    part = runs_df[
        runs_df[
            "split"
        ] == split_name
    ]


    print(
        f"\n{split_name}"
    )

    print(
        "  Runs:",
        len(part)
    )

    print(
        "  Windows:",
        int(
            part[
                "num_windows"
            ].sum()
        )
    )


    if len(part) > 0:

        print(
            "  Min run:",
            int(
                part[
                    "num_windows"
                ].min()
            )
        )

        print(
            "  Median run:",
            float(
                part[
                    "num_windows"
                ].median()
            )
        )

        print(
            "  Mean run:",
            round(
                float(
                    part[
                        "num_windows"
                    ].mean()
                ),
                2
            )
        )

        print(
            "  Max run:",
            int(
                part[
                    "num_windows"
                ].max()
            )
        )


# ============================================================
# DATASET-WISE ANALYSIS
# ============================================================

dataset_summary = (
    runs_df
    .groupby(
        [
            "split",
            "dataset"
        ],
        as_index=False
    )
    .agg(
        continuous_runs=(
            "run_id",
            "count"
        ),

        valid_windows=(
            "num_windows",
            "sum"
        ),

        shortest_run=(
            "num_windows",
            "min"
        ),

        median_run=(
            "num_windows",
            "median"
        ),

        longest_run=(
            "num_windows",
            "max"
        )
    )
)


# ============================================================
# DATASETS WITH MOST RUNS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DATASETS WITH MOST CONTINUOUS RUNS"
)

print(
    "=" * 100
)


print(
    dataset_summary
    .sort_values(
        "continuous_runs",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SHORTEST RUNS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SHORTEST CONTINUOUS RUNS"
)

print(
    "=" * 100
)


print(
    runs_df[
        [
            "run_id",
            "dataset",
            "split",
            "num_windows",
            "first_window_index",
            "last_window_index"
        ]
    ]
    .sort_values(
        "num_windows"
    )
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# LONGEST RUNS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LONGEST CONTINUOUS RUNS"
)

print(
    "=" * 100
)


print(
    runs_df[
        [
            "run_id",
            "dataset",
            "split",
            "num_windows",
            "first_window_index",
            "last_window_index"
        ]
    ]
    .sort_values(
        "num_windows",
        ascending=False
    )
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

runs_file = os.path.join(
    OUTPUT_DIR,
    "continuous_runs.csv"
)

dataset_file = os.path.join(
    OUTPUT_DIR,
    "dataset_continuous_run_summary.csv"
)

category_file = os.path.join(
    OUTPUT_DIR,
    "run_length_categories.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)


runs_df.to_csv(
    runs_file,
    index=False
)

dataset_summary.to_csv(
    dataset_file,
    index=False
)

category_df.to_csv(
    category_file,
    index=False
)


# ============================================================
# WRITE SUMMARY FILE
# ============================================================

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 66 - CONTINUOUS VALID TEMPORAL RUN ANALYSIS\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        f"Metadata file: {METADATA_FILE}\n"
    )

    f.write(
        f"Sequence split file: {SPLIT_FILE}\n"
    )

    f.write(
        f"Total valid windows: {len(metadata):,}\n"
    )

    f.write(
        f"Total continuous runs: {len(runs_df):,}\n"
    )

    f.write(
        f"Total datasets: "
        f"{metadata['dataset'].nunique():,}\n"
    )

    f.write(
        f"Minimum run length: "
        f"{int(np.min(lengths))}\n"
    )

    f.write(
        f"Maximum run length: "
        f"{int(np.max(lengths))}\n"
    )

    f.write(
        f"Mean run length: "
        f"{np.mean(lengths):.2f}\n"
    )

    f.write(
        f"Median run length: "
        f"{np.median(lengths):.2f}\n"
    )

    f.write(
        f"P90 run length: "
        f"{np.percentile(lengths, 90):.2f}\n"
    )

    f.write(
        f"P95 run length: "
        f"{np.percentile(lengths, 95):.2f}\n"
    )

    f.write(
        f"P99 run length: "
        f"{np.percentile(lengths, 99):.2f}\n"
    )

    f.write(
        "\nSplit summary:\n"
    )


    for split_name in [
        "TRAIN",
        "VALIDATION",
        "TEST"
    ]:

        part = runs_df[
            runs_df[
                "split"
            ] == split_name
        ]

        f.write(
            f"\n{split_name}\n"
        )

        f.write(
            f"Runs: {len(part)}\n"
        )

        f.write(
            f"Windows: "
            f"{int(part['num_windows'].sum())}\n"
        )


print(
    "\n"
    + "=" * 100
)

print(
    "STEP 66 COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nSaved:"
)

print(
    runs_file
)

print(
    dataset_file
)

print(
    category_file
)

print(
    summary_file
)
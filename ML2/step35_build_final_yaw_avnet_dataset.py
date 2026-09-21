from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Maverick\ML2")

DATASET_ROOT = (
    Path(r"D:\Maverick\IO-VNBD")
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
)

REFERENCE_FILE = (
    ROOT
    / "step33_yaw_reference"
    / "final_yaw_reference.csv"
)

OUTPUT = (
    ROOT
    / "avnet_yaw_reference_dataset.npz"
)

METADATA_OUTPUT = (
    ROOT
    / "avnet_yaw_reference_metadata.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "step35_yaw_dataset_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 10

# 1-based column positions from the verified dataset structure
# Accelerometer X/Y/Z = 10,11,12
# Gyroscope X/Y/Z     = 16,17,18

IMU_COLUMNS = [9, 10, 11, 15, 16, 17]


# ============================================================
# FUNCTIONS
# ============================================================

def find_s_file(dataset_name):

    """
    Find the corresponding smartphone S CSV file.

    Searches only inside the selected:
    Categorised IOVNB Dataset
    """

    matches = list(
        DATASET_ROOT.rglob(
            f"S-{dataset_name}.csv"
        )
    )

    # Exact expected filename
    if matches:
        return matches[0]

    # Case-insensitive fallback
    target = f"s-{dataset_name}.csv".lower()

    for path in DATASET_ROOT.rglob("*.csv"):

        if path.name.lower() == target:
            return path

    return None


def wrap180(x):

    return (x + 180.0) % 360.0 - 180.0


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STEP 35 - BUILD FINAL AVNET YAW REFERENCE DATASET")
print("=" * 75)

print()
print("Dataset root:")
print(DATASET_ROOT)

print()
print("Reference file:")
print(REFERENCE_FILE)


# ============================================================
# LOAD FINAL REFERENCE
# ============================================================

reference = pd.read_csv(
    REFERENCE_FILE
)

print()
print(
    f"Reference rows loaded: "
    f"{len(reference):,}"
)


required_columns = [
    "dataset",
    "window_index",
    "start_row",
    "end_row",
    "reference_valid",
    "reference_delta_yaw_deg",
    "reference_source",
    "reference_confidence",
]

missing = [
    c for c in required_columns
    if c not in reference.columns
]

if missing:

    raise RuntimeError(
        "Missing columns in final_yaw_reference.csv:\n"
        + "\n".join(missing)
    )


# ============================================================
# KEEP ONLY VALIDATED REFERENCES
# ============================================================

valid_reference = reference[
    reference["reference_valid"].astype(bool)
].copy()

invalid_count = (
    len(reference) -
    len(valid_reference)
)

print()
print(
    f"Valid reference windows   : "
    f"{len(valid_reference):,}"
)

print(
    f"Ambiguous/invalid windows : "
    f"{invalid_count:,}"
)


# ============================================================
# CACHE DATASET FILES
# ============================================================

# We load each S file only once.
# This avoids repeatedly reading the same CSV.

dataset_cache = {}


# ============================================================
# BUILD AVNET DATASET
# ============================================================

X_list = []
Y_list = []
metadata_list = []

missing_files = []
invalid_windows = []
shape_errors = []


current_dataset = None
current_dataframe = None


for counter, (_, row) in enumerate(
    valid_reference.iterrows(),
    start=1
):

    dataset = str(
        row["dataset"]
    )

    window_index = int(
        row["window_index"]
    )

    start_row = int(
        row["start_row"]
    )

    end_row = int(
        row["end_row"]
    )


    # --------------------------------------------------------
    # Load S file only when dataset changes
    # --------------------------------------------------------

    if dataset != current_dataset:

        if dataset not in dataset_cache:

            s_file = find_s_file(
                dataset
            )

            if s_file is None:

                missing_files.append(
                    dataset
                )

                current_dataframe = None
                current_dataset = dataset

                continue

            print()
            print(
                f"Loading {dataset}:"
            )
            print(
                f"  {s_file}"
            )

            try:

                df_s = pd.read_csv(
                    s_file,
                    encoding="cp1252"
                )

            except Exception as e:

                raise RuntimeError(
                    f"Could not read {s_file}\n"
                    f"{e}"
                )

            dataset_cache[
                dataset
            ] = df_s

        current_dataframe = (
            dataset_cache[dataset]
        )

        current_dataset = dataset


    if current_dataframe is None:

        continue


    # --------------------------------------------------------
    # Check row bounds
    # --------------------------------------------------------

    if (
        start_row < 0
        or end_row >= len(current_dataframe)
        or end_row - start_row + 1
        != WINDOW_SIZE
    ):

        invalid_windows.append(
            (
                dataset,
                window_index,
                start_row,
                end_row
            )
        )

        continue


    # --------------------------------------------------------
    # Extract six IMU columns
    # --------------------------------------------------------

    try:

        x = current_dataframe.iloc[
            start_row:end_row + 1,
            IMU_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not extract IMU data for "
            f"{dataset}, window {window_index}\n"
            f"{e}"
        )


    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    if x.shape != (10, 6):

        shape_errors.append(
            (
                dataset,
                window_index,
                x.shape
            )
        )

        continue


    # --------------------------------------------------------
    # Check numeric validity
    # --------------------------------------------------------

    if not np.isfinite(x).all():

        invalid_windows.append(
            (
                dataset,
                window_index,
                start_row,
                end_row
            )
        )

        continue


    # --------------------------------------------------------
    # Final validated yaw reference
    # --------------------------------------------------------

    delta_yaw = float(
        row["reference_delta_yaw_deg"]
    )

    delta_yaw = wrap180(
        delta_yaw
    )


    # --------------------------------------------------------
    # Yaw-only quaternion
    #
    # q = [qw, qx, qy, qz]
    #
    # qw = cos(delta_yaw/2)
    # qx = 0
    # qy = 0
    # qz = sin(delta_yaw/2)
    # --------------------------------------------------------

    yaw_rad = np.deg2rad(
        delta_yaw
    )

    qw = float(
        np.cos(yaw_rad / 2.0)
    )

    qx = 0.0
    qy = 0.0

    qz = float(
        np.sin(yaw_rad / 2.0)
    )


    # AVNet target
    y = np.array(
        [qx, qy, qz],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    X_list.append(
        x
    )

    Y_list.append(
        y
    )

    metadata_list.append({

        "sample_index":
            len(X_list) - 1,

        "dataset":
            dataset,

        "window_index":
            window_index,

        "start_row":
            start_row,

        "end_row":
            end_row,

        "reference_delta_yaw_deg":
            delta_yaw,

        "reference_source":
            row["reference_source"],

        "reference_confidence":
            float(
                row["reference_confidence"]
            ),

        "dq_w":
            qw,

        "dq_x":
            qx,

        "dq_y":
            qy,

        "dq_z":
            qz
    })


# ============================================================
# FINAL ARRAYS
# ============================================================

if not X_list:

    raise RuntimeError(
        "No samples were created."
    )


X = np.stack(
    X_list
).astype(
    np.float32
)

Y = np.stack(
    Y_list
).astype(
    np.float32
)

metadata = pd.DataFrame(
    metadata_list
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print("=" * 75)
print("DATASET CREATED")
print("=" * 75)

print(
    f"X shape : {X.shape}"
)

print(
    f"Y shape : {Y.shape}"
)

print(
    f"Samples : {len(X):,}"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION")
print("=" * 75)

assert X.ndim == 3
assert X.shape[1:] == (10, 6)

assert Y.ndim == 2
assert Y.shape[1] == 3

print(
    f"X expected : (N, 10, 6)"
)

print(
    f"X actual   : {X.shape}"
)

print(
    f"Y expected : (N, 3)"
)

print(
    f"Y actual   : {Y.shape}"
)


print()
print(
    f"X NaN : {np.isnan(X).sum():,}"
)

print(
    f"X Inf : {np.isinf(X).sum():,}"
)

print(
    f"Y NaN : {np.isnan(Y).sum():,}"
)

print(
    f"Y Inf : {np.isinf(Y).sum():,}"
)


# ============================================================
# QUATERNION VALIDATION
# ============================================================

qnorm = np.sqrt(
    metadata["dq_w"] ** 2
    + metadata["dq_x"] ** 2
    + metadata["dq_y"] ** 2
    + metadata["dq_z"] ** 2
)

norm_error = np.abs(
    qnorm - 1.0
)

print()
print(
    "Maximum quaternion norm error:"
)

print(
    f"{norm_error.max():.12f}"
)


# ============================================================
# YAW STATISTICS
# ============================================================

yaw = metadata[
    "reference_delta_yaw_deg"
].to_numpy()

abs_yaw = np.abs(yaw)

print()
print("=" * 75)
print("FINAL YAW REFERENCE STATISTICS")
print("=" * 75)

print(
    f"Minimum : {yaw.min():.6f}°"
)

print(
    f"Maximum : {yaw.max():.6f}°"
)

print(
    f"Median  : {np.median(yaw):.6f}°"
)

print(
    f"P90     : {np.percentile(yaw,90):.6f}°"
)

print(
    f"P95     : {np.percentile(yaw,95):.6f}°"
)

print(
    f"P99     : {np.percentile(yaw,99):.6f}°"
)

print(
    f"|yaw| >= 45° : "
    f"{(abs_yaw >= 45).sum():,}"
)

print(
    f"|yaw| >= 90° : "
    f"{(abs_yaw >= 90).sum():,}"
)


# ============================================================
# REFERENCE SOURCES
# ============================================================

print()
print("=" * 75)
print("REFERENCE SOURCE DISTRIBUTION")
print("=" * 75)

source_counts = (
    metadata[
        "reference_source"
    ]
    .value_counts()
)

for source, count in source_counts.items():

    percentage = (
        100.0 *
        count /
        len(metadata)
    )

    print(
        f"{str(source):50s}"
        f"{count:8,d}"
        f"  ({percentage:6.2f}%)"
    )


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("DATASET DISTRIBUTION")
print("=" * 75)

dataset_counts = (
    metadata[
        "dataset"
    ]
    .value_counts()
)

for dataset, count in dataset_counts.items():

    print(
        f"{dataset:15s}"
        f"{count:8,d}"
    )


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT,
    X=X,
    Y=Y
)

metadata.to_csv(
    METADATA_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 35 - FINAL AVNET YAW REFERENCE DATASET\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Total valid samples: {len(X):,}\n"
    )

    f.write(
        f"Input shape: {X.shape}\n"
    )

    f.write(
        f"Target shape: {Y.shape}\n\n"
    )

    f.write(
        "Input channels:\n"
    )

    f.write(
        "1. Accelerometer X\n"
    )

    f.write(
        "2. Accelerometer Y\n"
    )

    f.write(
        "3. Accelerometer Z\n"
    )

    f.write(
        "4. Gyroscope X\n"
    )

    f.write(
        "5. Gyroscope Y\n"
    )

    f.write(
        "6. Gyroscope Z\n\n"
    )

    f.write(
        "Sampling frequency: 10 Hz\n"
    )

    f.write(
        "Window duration: 1 second\n"
    )

    f.write(
        "Samples per window: 10\n\n"
    )

    f.write(
        "Target:\n"
    )

    f.write(
        "Yaw-only quaternion change [qx,qy,qz]\n"
    )

    f.write(
        "qx = 0\n"
    )

    f.write(
        "qy = 0\n"
    )

    f.write(
        "qz = sin(delta_yaw/2)\n\n"
    )

    f.write(
        "Reference source:\n"
    )

    f.write(
        "Dataset-derived yaw reference from Steps 33-34.\n"
    )

    f.write(
        "This is an adaptation for the dataset and is NOT "
        "the original paper's external attitude ground truth.\n\n"
    )

    f.write(
        f"Missing S files: {len(set(missing_files))}\n"
    )

    f.write(
        f"Invalid windows: {len(invalid_windows)}\n"
    )

    f.write(
        f"Shape errors: {len(shape_errors)}\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 35 COMPLETE")
print("=" * 75)

print(
    f"Final samples: {len(X):,}"
)

print(
    f"X: {X.shape}"
)

print(
    f"Y: {Y.shape}"
)

print()
print("Saved:")

print(
    OUTPUT
)

print(
    METADATA_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print()

if missing_files:

    print(
        "WARNING: Missing S files:"
    )

    for name in sorted(
        set(missing_files)
    ):

        print(
            f"  {name}"
        )

if invalid_windows:

    print(
        f"WARNING: Invalid windows: "
        f"{len(invalid_windows)}"
    )

if shape_errors:

    print(
        f"WARNING: Shape errors: "
        f"{len(shape_errors)}"
    )
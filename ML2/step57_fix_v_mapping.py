import os
import pandas as pd


BASE = r"D:\Maverick\ML2"

DATASET_ROOT = (
    r"D:\Maverick\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
)

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step58_v_mapping"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


print("=" * 100)
print("STEP 58 - DETERMINISTIC V FILE MAPPING")
print("=" * 100)


# ============================================================
# LOAD DATASET NAMES
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

datasets = sorted(
    metadata["dataset"]
    .astype(str)
    .str.lower()
    .unique()
)

print(
    f"\nUnique AVNet datasets: {len(datasets)}"
)


# ============================================================
# SCAN ALL CSV FILES
# ============================================================

all_csv = []

for root, dirs, files in os.walk(
    DATASET_ROOT
):

    for file in files:

        if file.lower().endswith(
            ".csv"
        ):

            all_csv.append(
                os.path.join(
                    root,
                    file
                )
            )


print(
    f"Total CSV files found: {len(all_csv)}"
)


# ============================================================
# NORMALIZE DATASET NAME
# ============================================================

def normalize(
    text
):

    return (
        str(text)
        .lower()
        .replace(
            ".csv",
            ""
        )
        .replace(
            "-",
            ""
        )
        .replace(
            "_",
            ""
        )
        .replace(
            " ",
            ""
        )
    )


# ============================================================
# IDENTIFY V FILES ONLY
# ============================================================

# Important:
#
# We NEVER match S files.
#
# A vehicle file is identified by:
#
#       filename starts with V-
#
# Examples:
#
# V-S2.csv
# V-vta5.csv
# V-Vta1a.csv
# V-Vfa01.csv
# V-Vw14b.csv
#
# This prevents S/V ambiguity.

v_files = []


for path in all_csv:

    filename = os.path.basename(
        path
    )

    lower = filename.lower()

    if lower.startswith(
        "v-"
    ):

        v_files.append(
            path
        )


print(
    f"V candidate files: {len(v_files)}"
)


# ============================================================
# CREATE DATASET IDENTIFIER FROM V FILENAME
# ============================================================

def dataset_from_v_filename(
    filename
):

    name = os.path.splitext(
        filename
    )[0]

    name = name.lower()

    # Remove V- prefix
    if name.startswith(
        "v-"
    ):

        name = name[2:]


    # --------------------------------------------------------
    # V-S2
    # --------------------------------------------------------

    if name.startswith(
        "s"
    ):

        return name


    # --------------------------------------------------------
    # V-Vta05
    # V-vta5
    # V-Vta1a
    # --------------------------------------------------------

    if name.startswith(
        "vta"
    ):

        return name


    # --------------------------------------------------------
    # V-Vfa01
    # --------------------------------------------------------

    if name.startswith(
        "vfa"
    ):

        return name


    # --------------------------------------------------------
    # V-Vtb01
    # --------------------------------------------------------

    if name.startswith(
        "vtb"
    ):

        return name


    # --------------------------------------------------------
    # V-Vw14b
    # --------------------------------------------------------

    if name.startswith(
        "vw"
    ):

        return name


    # --------------------------------------------------------
    # V-M
    # --------------------------------------------------------

    if name == "m":

        return "m"


    # --------------------------------------------------------
    # V-Y1
    # --------------------------------------------------------

    if name.startswith(
        "y"
    ):

        return name


    return None


# ============================================================
# BUILD V FILE MAP
# ============================================================

v_map = {}


for path in v_files:

    filename = os.path.basename(
        path
    )

    dataset_name = (
        dataset_from_v_filename(
            filename
        )
    )

    if dataset_name is None:

        continue


    dataset_name = normalize(
        dataset_name
    )


    # --------------------------------------------------------
    # Convert zero-padded names
    # --------------------------------------------------------

    # vta05 -> vta5
    # vtb01 -> vtb1
    # vw08 -> vw8

    import re

    match = re.match(
        r"^(vta|vtb|vw)(\d+)(.*)$",
        dataset_name
    )

    if match:

        prefix = match.group(
            1
        )

        number = match.group(
            2
        )

        suffix = match.group(
            3
        )

        dataset_name = (
            prefix
            + str(
                int(number)
            )
            + suffix
        )


    if dataset_name in v_map:

        print(
            "\nDUPLICATE V FILE:"
        )

        print(
            dataset_name
        )

        print(
            v_map[
                dataset_name
            ]
        )

        print(
            path
        )

        raise RuntimeError(
            "Duplicate V mapping detected."
        )


    v_map[
        dataset_name
    ] = path


# ============================================================
# MATCH AVNET DATASETS
# ============================================================

mapping = {}

unmatched = []


for dataset in datasets:

    dataset = normalize(
        dataset
    )


    if dataset in v_map:

        mapping[
            dataset
        ] = v_map[
            dataset
        ]

    else:

        unmatched.append(
            dataset
        )


# ============================================================
# PRINT RESULT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MAPPING RESULT"
)

print(
    "=" * 100
)

print(
    f"AVNet datasets : {len(datasets)}"
)

print(
    f"V files matched: {len(mapping)}"
)

print(
    f"Unmatched      : {len(unmatched)}"
)


# ============================================================
# SHOW ALL MAPPINGS
# ============================================================

for dataset in sorted(
    mapping
):

    print(
        f"{dataset:10s} -> "
        f"{mapping[dataset]}"
    )


# ============================================================
# UNMATCHED
# ============================================================

if unmatched:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "UNMATCHED DATASETS"
    )

    print(
        "=" * 100
    )

    for dataset in sorted(
        unmatched
    ):

        print(
            dataset
        )


# ============================================================
# SAVE MAPPING
# ============================================================

rows = []

for dataset in datasets:

    d = normalize(
        dataset
    )

    rows.append({

        "dataset":
            d,

        "matched":
            d in mapping,

        "v_file":
            mapping.get(
                d,
                ""
            )
    })


mapping_df = pd.DataFrame(
    rows
)

mapping_file = os.path.join(
    OUT_DIR,
    "complete_v_file_mapping.csv"
)

mapping_df.to_csv(
    mapping_file,
    index=False
)


# ============================================================
# VERIFY ALL V FILES
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "VERIFYING V FILES"
)

print(
    "=" * 100
)


verification = []


for dataset, path in sorted(
    mapping.items()
):

    try:

        df = pd.read_csv(
            path,
            nrows=5
        )

        verification.append({

            "dataset":
                dataset,

            "readable":
                True,

            "columns":
                len(df.columns),

            "path":
                path
        })

    except Exception as e:

        verification.append({

            "dataset":
                dataset,

            "readable":
                False,

            "columns":
                0,

            "path":
                path,

            "error":
                str(e)
        })


verification_df = pd.DataFrame(
    verification
)


if len(
    verification_df
) > 0:

    print(
        verification_df[
            [
                "dataset",
                "readable",
                "columns"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nReadable:"
    )

    print(
        verification_df[
            "readable"
        ].sum(),
        "/",
        len(
            verification_df
        )
    )


    verification_file = os.path.join(
        OUT_DIR,
        "v_file_verification.csv"
    )

    verification_df.to_csv(
        verification_file,
        index=False
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STEP 58 COMPLETE"
)

print(
    "=" * 100
)

print(
    f"AVNet datasets : {len(datasets)}"
)

print(
    f"V files matched: {len(mapping)}"
)

print(
    f"Unmatched      : {len(unmatched)}"
)

print(
    "\nMapping saved:"
)

print(
    mapping_file
)
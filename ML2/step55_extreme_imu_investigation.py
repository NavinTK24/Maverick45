import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Maverick\ML2"

NORMALIZED_FILE = os.path.join(
    BASE,
    "step44_normalized_data",
    "test.npz"
)

STATS_FILE = os.path.join(
    BASE,
    "step43_normalization_statistics.npz"
)

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

V_FILE = (
    r"D:\Maverick\IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Categorised IOVNB Dataset"
    r"\S (Driver A)"
    r"\S2"
    r"\V-S2.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step55_extreme_imu_investigation"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD PHONE DATA
# ============================================================

data = np.load(
    NORMALIZED_FILE
)

X_norm = data["X"].astype(
    np.float64
)

stats = np.load(
    STATS_FILE
)

mean = stats["mean"].astype(
    np.float64
)

std = stats["std"].astype(
    np.float64
)

X = (
    X_norm * std
    + mean
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

metadata = metadata[
    metadata["dataset"].str.lower() == "s2"
].copy()

metadata = metadata.sort_values(
    "sample_index"
).reset_index(
    drop=True
)


# ============================================================
# LOAD V DATA
# ============================================================

vdf = pd.read_csv(
    V_FILE
)


# ============================================================
# TARGET WINDOWS
# ============================================================

target_samples = [
    15636,
    15637
]


# ============================================================
# FIND GLOBAL INDEX
# ============================================================

def find_global_index(
    sample_index
):

    matches = np.where(
        metadata[
            "sample_index"
        ].to_numpy()
        == sample_index
    )[0]

    if len(matches) == 0:

        raise RuntimeError(
            f"Sample {sample_index} "
            "not found."
        )

    return int(
        matches[0]
    )


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 100)
print("STEP 55 - EXTREME PHONE IMU INVESTIGATION")
print("=" * 100)


# ============================================================
# PROCESS EACH EXTREME WINDOW
# ============================================================

all_rows = []


for sample_index in target_samples:

    idx = find_global_index(
        sample_index
    )

    meta = metadata.iloc[
        idx
    ]

    window_index = int(
        meta["window_index"]
    )

    start_row = int(
        meta["start_row"]
    )

    end_row = int(
        meta["end_row"]
    )

    ref_yaw = float(
        meta["reference_delta_yaw_deg"]
    )


    # --------------------------------------------------------
    # Phone window
    # --------------------------------------------------------

    phone_window = X[
        idx
    ]


    # --------------------------------------------------------
    # Vehicle window
    # --------------------------------------------------------

    vehicle_window = vdf.iloc[
        start_row:end_row + 1
    ].copy()


    print("\n")
    print("=" * 100)

    print(
        f"SAMPLE INDEX : {sample_index}"
    )

    print(
        f"WINDOW INDEX : {window_index}"
    )

    print(
        f"V ROWS      : {start_row} - {end_row}"
    )

    print(
        f"REFERENCE YAW CHANGE : {ref_yaw:.6f} deg"
    )

    print("=" * 100)


    # --------------------------------------------------------
    # PHONE TABLE
    # --------------------------------------------------------

    phone_df = pd.DataFrame({

        "sample_in_window":
            np.arange(1, 11),

        "AccX":
            phone_window[:, 0],

        "AccY":
            phone_window[:, 1],

        "AccZ":
            phone_window[:, 2],

        "GyroX":
            phone_window[:, 3],

        "GyroY":
            phone_window[:, 4],

        "GyroZ":
            phone_window[:, 5]
    })


    phone_df[
        "Acc_Magnitude"
    ] = np.sqrt(
        phone_df["AccX"] ** 2
        + phone_df["AccY"] ** 2
        + phone_df["AccZ"] ** 2
    )

    phone_df[
        "Gyro_Magnitude"
    ] = np.sqrt(
        phone_df["GyroX"] ** 2
        + phone_df["GyroY"] ** 2
        + phone_df["GyroZ"] ** 2
    )


    print("\nPHONE IMU:")
    print(
        phone_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    # --------------------------------------------------------
    # VEHICLE TABLE
    # --------------------------------------------------------

    print("\nVEHICLE DATA:")

    print(
        vehicle_window.to_string(
            index=True
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nPHONE SUMMARY")

    print(
        f"Max acceleration magnitude : "
        f"{phone_df['Acc_Magnitude'].max():.6f} m/s²"
    )

    print(
        f"Min acceleration magnitude : "
        f"{phone_df['Acc_Magnitude'].min():.6f} m/s²"
    )

    print(
        f"Max gyro magnitude : "
        f"{phone_df['Gyro_Magnitude'].max():.6f} rad/s"
    )

    print(
        f"Mean gyro magnitude : "
        f"{phone_df['Gyro_Magnitude'].mean():.6f} rad/s"
    )


    # --------------------------------------------------------
    # VEHICLE SUMMARY
    # --------------------------------------------------------

    def numeric_column(
        names
    ):

        for name in names:

            if name in vehicle_window.columns:

                return pd.to_numeric(
                    vehicle_window[name],
                    errors="coerce"
                )

        return None


    yawrate = numeric_column([
        "Yaw Rate (deg/sec)"
    ])

    longacc = numeric_column([
        "Indicated Longitudinal Acceleration (g)"
    ])

    latacc = numeric_column([
        "Indicated Lateral Acceleration (g)"
    ])

    steering = numeric_column([
        "Steering Angle (degrees)"
    ])

    velocity = numeric_column([
        "Velocity (km/hr)"
    ])


    print("\nVEHICLE SUMMARY")


    if yawrate is not None:

        print(
            f"Max |yaw rate| : "
            f"{np.nanmax(np.abs(yawrate)):.6f} deg/s"
        )


    if longacc is not None:

        print(
            f"Max |longitudinal acceleration| : "
            f"{np.nanmax(np.abs(longacc)):.6f} g"
        )


    if latacc is not None:

        print(
            f"Max |lateral acceleration| : "
            f"{np.nanmax(np.abs(latacc)):.6f} g"
        )


    if steering is not None:

        print(
            f"Steering range : "
            f"{np.nanmax(steering) - np.nanmin(steering):.6f} deg"
        )


    if velocity is not None:

        print(
            f"Vehicle speed range : "
            f"{np.nanmin(velocity):.6f} - "
            f"{np.nanmax(velocity):.6f} km/h"
        )


    # --------------------------------------------------------
    # SAVE PHONE DATA
    # --------------------------------------------------------

    phone_output = os.path.join(
        OUT_DIR,
        f"s2_sample_{sample_index}_phone.csv"
    )

    phone_df.to_csv(
        phone_output,
        index=False
    )


    # --------------------------------------------------------
    # SAVE VEHICLE DATA
    # --------------------------------------------------------

    vehicle_output = os.path.join(
        OUT_DIR,
        f"s2_sample_{sample_index}_vehicle.csv"
    )

    vehicle_window.to_csv(
        vehicle_output,
        index=False
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 100)
print("STEP 55 COMPLETE")
print("=" * 100)

print(
    f"Results saved to:\n{OUT_DIR}"
)
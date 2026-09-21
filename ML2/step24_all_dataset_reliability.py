import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"

WINDOW_DIR = r"D:\Maverick\ML2\windows"

OUTPUT_DIR = r"D:\Maverick\ML2"

WINDOW_SIZE = 10


# ============================================================
# ROBUST CSV READER
# ============================================================

def read_csv_robust(path):

    for enc in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]:

        try:
            return pd.read_csv(
                path,
                encoding=enc
            )

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not decode: {path}"
    )


# ============================================================
# CIRCULAR DIFFERENCE
# ============================================================

def circular_difference(a, b):

    return (
        (b - a + 180.0) % 360.0
    ) - 180.0


# ============================================================
# BEARING FROM TWO LAT/LON POINTS
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(
        lon2 - lon1
    )

    x = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    y = (
        np.cos(lat1)
        * np.sin(lat2)
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return (
        bearing + 360.0
    ) % 360.0


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1.0 - a)
        )
    )

    return R * c


# ============================================================
# FIND ALL S/V FILES
# ============================================================

s_files = sorted(
    glob.glob(
        os.path.join(
            BASE,
            "**",
            "S-*.csv"
        ),
        recursive=True
    )
)

v_files = sorted(
    glob.glob(
        os.path.join(
            BASE,
            "**",
            "V-*.csv"
        ),
        recursive=True
    )
)


# ============================================================
# CREATE DICTIONARIES
# ============================================================

s_dict = {}

for path in s_files:

    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    key = name[2:].lower()

    s_dict[key] = path


v_dict = {}

for path in v_files:

    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    key = name[2:].lower()

    v_dict[key] = path


keys = sorted(
    set(s_dict)
    &
    set(v_dict)
)


print("=" * 100)
print("STEP 24 - ALL DATASET RELIABILITY ANALYSIS")
print("=" * 100)

print()

print(
    f"S files : {len(s_dict)}"
)

print(
    f"V files : {len(v_dict)}"
)

print(
    f"Matched : {len(keys)}"
)

print()


# ============================================================
# PROCESS DATASETS
# ============================================================

records = []


for dataset_index, key in enumerate(keys, start=1):

    print(
        f"[{dataset_index:02d}/{len(keys):02d}] "
        f"Processing {key}"
    )


    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    S = read_csv_robust(
        s_dict[key]
    )

    V = read_csv_robust(
        v_dict[key]
    )


    # --------------------------------------------------------
    # Verified V columns
    #
    # 1-based:
    #
    # 2  = Time
    # 3  = Latitude
    # 4  = Longitude
    # 5  = Velocity
    # 6  = Heading
    # 15 = Yaw Rate
    # 18 = Lateral acceleration
    #
    # --------------------------------------------------------

    v_lat = pd.to_numeric(
        V.iloc[:, 2],
        errors="coerce"
    ).to_numpy()

    v_lon = pd.to_numeric(
        V.iloc[:, 3],
        errors="coerce"
    ).to_numpy()

    v_velocity = pd.to_numeric(
        V.iloc[:, 4],
        errors="coerce"
    ).to_numpy()

    v_heading = pd.to_numeric(
        V.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    v_yaw_rate = pd.to_numeric(
        V.iloc[:, 14],
        errors="coerce"
    ).to_numpy()

    v_lat_acc = pd.to_numeric(
        V.iloc[:, 17],
        errors="coerce"
    ).to_numpy()


    # --------------------------------------------------------
    # V sample period
    # --------------------------------------------------------

    v_dt = pd.to_numeric(
        V.iloc[:, 8],
        errors="coerce"
    ).to_numpy()


    # --------------------------------------------------------
    # Smartphone GPS fields
    #
    # 1-based:
    #
    # 1 latitude
    # 2 longitude
    # 4 GPS speed
    # 5 GPS accuracy
    # 6 GPS orientation
    # 7 satellites
    #
    # --------------------------------------------------------

    s_lat = pd.to_numeric(
        S.iloc[:, 0],
        errors="coerce"
    ).to_numpy()

    s_lon = pd.to_numeric(
        S.iloc[:, 1],
        errors="coerce"
    ).to_numpy()

    s_gps_speed = pd.to_numeric(
        S.iloc[:, 3],
        errors="coerce"
    ).to_numpy()

    s_gps_accuracy = pd.to_numeric(
        S.iloc[:, 4],
        errors="coerce"
    ).to_numpy()

    s_gps_orientation = pd.to_numeric(
        S.iloc[:, 5],
        errors="coerce"
    ).to_numpy()

    s_satellites = pd.to_numeric(
        S.iloc[:, 6],
        errors="coerce"
    ).to_numpy()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use existing smartphone windows to determine EXACT
    # windows used by AVNet.
    # --------------------------------------------------------

    window_file = os.path.join(
        WINDOW_DIR,
        f"{key}_windows.npz"
    )


    if not os.path.exists(window_file):

        print(
            f"  WARNING: missing {window_file}"
        )

        continue


    W = np.load(
        window_file
    )


    X = W["X"]


    n_windows = X.shape[0]


    # --------------------------------------------------------
    # Check bounds
    # --------------------------------------------------------

    n_common = min(
        len(S),
        len(V)
    )


    # --------------------------------------------------------
    # Process exact windows
    # --------------------------------------------------------

    for w in range(n_windows):

        start = w * WINDOW_SIZE

        end = start + WINDOW_SIZE - 1


        if end >= n_common:

            continue


        # ----------------------------------------------------
        # GPS positions at window boundaries
        # ----------------------------------------------------

        lat1 = v_lat[start]
        lon1 = v_lon[start]

        lat2 = v_lat[end]
        lon2 = v_lon[end]


        # ----------------------------------------------------
        # GNSS displacement
        # ----------------------------------------------------

        if (
            np.isfinite(lat1)
            and np.isfinite(lon1)
            and np.isfinite(lat2)
            and np.isfinite(lon2)
        ):

            displacement = haversine_distance(
                lat1,
                lon1,
                lat2,
                lon2
            )

            route_bearing = calculate_bearing(
                lat1,
                lon1,
                lat2,
                lon2
            )

        else:

            displacement = np.nan
            route_bearing = np.nan


        # ----------------------------------------------------
        # GNSS route bearing using a larger local baseline
        #
        # Use first and last sample of the 1-second window.
        # ----------------------------------------------------

        if np.isfinite(route_bearing):

            # Heading at beginning/end from route itself
            route_direction_start = route_bearing
            route_direction_end = route_bearing

            route_yaw_change = 0.0

        else:

            route_direction_start = np.nan
            route_direction_end = np.nan
            route_yaw_change = np.nan


        # ----------------------------------------------------
        # V Heading
        # ----------------------------------------------------

        h1 = v_heading[start]
        h2 = v_heading[end]


        if (
            np.isfinite(h1)
            and np.isfinite(h2)
        ):

            heading_change = circular_difference(
                h1,
                h2
            )

        else:

            heading_change = np.nan


        # ----------------------------------------------------
        # Yaw-rate integration
        # ----------------------------------------------------

        yr = v_yaw_rate[
            start:end + 1
        ]

        dt = v_dt[
            start:end + 1
        ]


        valid_dt = (
            np.isfinite(dt)
            &
            (dt > 0)
        )


        dt_use = np.where(
            valid_dt,
            dt,
            0.1
        )


        yr_use = np.where(
            np.isfinite(yr),
            yr,
            0.0
        )


        yawrate_change = np.sum(
            yr_use
            * dt_use
        )


        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        velocity_window = v_velocity[
            start:end + 1
        ]

        valid_velocity = (
            velocity_window[
                np.isfinite(
                    velocity_window
                )
            ]
        )


        if len(valid_velocity) > 0:

            mean_velocity = np.mean(
                valid_velocity
            )

            max_velocity = np.max(
                valid_velocity
            )

        else:

            mean_velocity = np.nan
            max_velocity = np.nan


        # ----------------------------------------------------
        # Lateral acceleration
        # ----------------------------------------------------

        lat_acc_window = v_lat_acc[
            start:end + 1
        ]

        valid_lat_acc = (
            lat_acc_window[
                np.isfinite(
                    lat_acc_window
                )
            ]
        )


        if len(valid_lat_acc) > 0:

            max_abs_lat_acc = np.max(
                np.abs(
                    valid_lat_acc
                )
            )

        else:

            max_abs_lat_acc = np.nan


        # ----------------------------------------------------
        # Smartphone GPS
        # ----------------------------------------------------

        s_speed_window = s_gps_speed[
            start:end + 1
        ]

        s_accuracy_window = s_gps_accuracy[
            start:end + 1
        ]

        s_orientation_window = (
            s_gps_orientation[
                start:end + 1
            ]
        )

        s_sat_window = (
            s_satellites[
                start:end + 1
            ]
        )


        def finite_median(x):

            y = x[
                np.isfinite(x)
            ]

            if len(y) == 0:
                return np.nan

            return np.median(y)


        gps_speed = finite_median(
            s_speed_window
        )

        gps_accuracy = finite_median(
            s_accuracy_window
        )

        gps_orientation = finite_median(
            s_orientation_window
        )

        satellites = finite_median(
            s_sat_window
        )


        # ----------------------------------------------------
        # Heading vs yaw-rate agreement
        # ----------------------------------------------------

        if (
            np.isfinite(
                heading_change
            )
            and np.isfinite(
                yawrate_change
            )
        ):

            heading_yawrate_difference = abs(
                circular_difference(
                    yawrate_change,
                    heading_change
                )
            )

        else:

            heading_yawrate_difference = np.nan


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        records.append({

            "dataset": key,

            "window": w,

            "start_row": start,

            "end_row": end,

            "mean_velocity_kmh":
                mean_velocity,

            "max_velocity_kmh":
                max_velocity,

            "gnss_displacement_m":
                displacement,

            "gnss_route_bearing_deg":
                route_bearing,

            "v_heading_start_deg":
                h1,

            "v_heading_end_deg":
                h2,

            "v_heading_change_deg":
                heading_change,

            "v_yawrate_change_deg":
                yawrate_change,

            "heading_yawrate_difference_deg":
                heading_yawrate_difference,

            "max_abs_lateral_acc":
                max_abs_lat_acc,

            "phone_gps_speed":
                gps_speed,

            "phone_gps_accuracy_m":
                gps_accuracy,

            "phone_gps_orientation_deg":
                gps_orientation,

            "phone_satellites":
                satellites
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


print()
print("=" * 100)

print(
    f"Exact AVNet windows analyzed: {len(df)}"
)

print("=" * 100)


# ============================================================
# SAVE COMPLETE DATA
# ============================================================

full_output = os.path.join(
    OUTPUT_DIR,
    "step24_all_dataset_reliability.csv"
)

df.to_csv(
    full_output,
    index=False
)


# ============================================================
# SPEED CATEGORY
# ============================================================

speed_bins = [
    -np.inf,
    1,
    5,
    10,
    20,
    40,
    60,
    np.inf
]

speed_labels = [
    "0-1",
    "1-5",
    "5-10",
    "10-20",
    "20-40",
    "40-60",
    ">60"
]


df["speed_category"] = pd.cut(
    df["mean_velocity_kmh"],
    bins=speed_bins,
    labels=speed_labels,
    right=False
)


# ============================================================
# OVERALL SPEED DISTRIBUTION
# ============================================================

print()
print("=" * 100)
print("WINDOW DISTRIBUTION BY VEHICLE SPEED")
print("=" * 100)

speed_counts = (
    df["speed_category"]
    .value_counts(
        sort=False
    )
)


for category, count in speed_counts.items():

    print(
        f"{str(category):>8} km/h : "
        f"{count:7d} "
        f"({count / len(df) * 100:7.3f}%)"
    )


# ============================================================
# RELIABILITY BY SPEED
# ============================================================

print()
print("=" * 100)
print("RELIABILITY BY SPEED")
print("=" * 100)


for category in speed_labels:

    sub = df[
        df["speed_category"]
        == category
    ]


    if len(sub) == 0:
        continue


    print()
    print(
        f"--- {category} km/h "
        f"({len(sub)} windows) ---"
    )


    # GNSS displacement

    print(
        f"GNSS displacement median : "
        f"{sub['gnss_displacement_m'].median():.4f} m"
    )


    # Heading change

    print(
        f"|Heading change| median  : "
        f"{sub['v_heading_change_deg'].abs().median():.3f}°"
    )


    # Yaw rate change

    print(
        f"|Yaw-rate change| median  : "
        f"{sub['v_yawrate_change_deg'].abs().median():.3f}°"
    )


    # Heading/Yaw agreement

    print(
        f"|Heading-YawRate| median  : "
        f"{sub['heading_yawrate_difference_deg'].median():.3f}°"
    )


    print(
        f"|Heading-YawRate| P90     : "
        f"{sub['heading_yawrate_difference_deg'].quantile(.90):.3f}°"
    )


    # GPS accuracy

    print(
        f"GPS accuracy median      : "
        f"{sub['phone_gps_accuracy_m'].median():.3f} m"
    )


# ============================================================
# LARGE HEADING TRANSITIONS BY SPEED
# ============================================================

print()
print("=" * 100)
print("LARGE HEADING TRANSITIONS BY SPEED")
print("=" * 100)


for category in speed_labels:

    sub = df[
        df["speed_category"]
        == category
    ]


    if len(sub) == 0:
        continue


    large = sub[
        sub[
            "v_heading_change_deg"
        ].abs()
        >= 45
    ]


    print(
        f"{str(category):>8} km/h : "
        f"{len(large):6d} "
        f"/ {len(sub):6d} "
        f"= {len(large)/len(sub)*100:7.3f}%"
    )


# ============================================================
# STRONG HEADING/YAW AGREEMENT
# ============================================================

print()
print("=" * 100)
print("HEADING vs YAWRATE AGREEMENT")
print("=" * 100)


for threshold in [
    2,
    5,
    10,
    20
]:

    valid = df[
        np.isfinite(
            df[
                "heading_yawrate_difference_deg"
            ]
        )
    ]


    percentage = (
        np.mean(
            valid[
                "heading_yawrate_difference_deg"
            ]
            <= threshold
        )
        * 100
    )


    print(
        f"Difference <= {threshold:2d}° : "
        f"{percentage:.3f}%"
    )


# ============================================================
# WORST HEADING/YAWRATE AGREEMENT
# ============================================================

print()
print("=" * 100)
print("TOP 30 HEADING / YAWRATE DISAGREEMENTS")
print("=" * 100)


top = df.sort_values(
    "heading_yawrate_difference_deg",
    ascending=False
).head(30)


print(
    top[
        [
            "dataset",
            "window",
            "mean_velocity_kmh",
            "gnss_displacement_m",
            "v_heading_change_deg",
            "v_yawrate_change_deg",
            "heading_yawrate_difference_deg",
            "max_abs_lateral_acc"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# LARGE HEADING CHANGES WHILE VEHICLE IS MOVING
# ============================================================

print()
print("=" * 100)
print("LARGE HEADING CHANGES WHILE MOVING")
print("=" * 100)


moving_large = df[
    (
        df["mean_velocity_kmh"] >= 5
    )
    &
    (
        df[
            "v_heading_change_deg"
        ].abs()
        >= 45
    )
]


print(
    f"Count: {len(moving_large)}"
)


if len(moving_large) > 0:

    print()

    print(
        moving_large[
            [
                "dataset",
                "window",
                "mean_velocity_kmh",
                "gnss_displacement_m",
                "v_heading_change_deg",
                "v_yawrate_change_deg",
                "heading_yawrate_difference_deg",
                "max_abs_lateral_acc"
            ]
        ]
        .sort_values(
            "mean_velocity_kmh"
        )
        .head(50)
        .to_string(
            index=False
        )
    )


# ============================================================
# HIGH-MOTION CONSISTENCY
# ============================================================

print()
print("=" * 100)
print("HIGH-MOTION WINDOWS (>20 km/h)")
print("=" * 100)


high_motion = df[
    df["mean_velocity_kmh"]
    >= 20
]


print(
    f"Windows: {len(high_motion)}"
)


if len(high_motion) > 0:

    print(
        f"Heading/YawRate median difference: "
        f"{high_motion['heading_yawrate_difference_deg'].median():.3f}°"
    )

    print(
        f"Heading/YawRate P90 difference: "
        f"{high_motion['heading_yawrate_difference_deg'].quantile(.90):.3f}°"
    )

    print(
        f"GNSS displacement median: "
        f"{high_motion['gnss_displacement_m'].median():.3f} m"
    )


# ============================================================
# DATASET-LEVEL SUMMARY
# ============================================================

print()
print("=" * 100)
print("DATASET-LEVEL SUMMARY")
print("=" * 100)


dataset_rows = []


for key, sub in df.groupby(
    "dataset"
):

    dataset_rows.append({

        "dataset": key,

        "windows": len(sub),

        "median_velocity_kmh":
            sub[
                "mean_velocity_kmh"
            ].median(),

        "median_gnss_displacement_m":
            sub[
                "gnss_displacement_m"
            ].median(),

        "median_heading_yawrate_difference_deg":
            sub[
                "heading_yawrate_difference_deg"
            ].median(),

        "p90_heading_yawrate_difference_deg":
            sub[
                "heading_yawrate_difference_deg"
            ].quantile(.90),

        "large_heading_change_count":
            np.sum(
                sub[
                    "v_heading_change_deg"
                ].abs()
                >= 45
            ),

        "large_heading_change_percent":
            np.mean(
                sub[
                    "v_heading_change_deg"
                ].abs()
                >= 45
            )
            * 100
    })


dataset_summary = pd.DataFrame(
    dataset_rows
)


dataset_output = os.path.join(
    OUTPUT_DIR,
    "step24_dataset_summary.csv"
)


dataset_summary.to_csv(
    dataset_output,
    index=False
)


print(
    dataset_summary.to_string(
        index=False
    )
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 100)
print("STEP 24 COMPLETE")
print("=" * 100)

print()

print(
    "Saved:"
)

print(
    full_output
)

print(
    dataset_output
)

print()

print(
    "IMPORTANT:"
)

print(
    "No samples were removed."
)

print(
    "No ground truth was assigned."
)

print(
    "This step only measures signal reliability."
)
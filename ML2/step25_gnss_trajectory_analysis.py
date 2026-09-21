import os
import sys
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE = r"D:\Maverick\ML2"

WINDOW_DIR = os.path.join(
    BASE,
    "windows"
)

OUT_FILE = os.path.join(
    BASE,
    "step25_gnss_trajectory_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    BASE,
    "step25_dataset_summary.csv"
)


# =========================================================
# DATASET PATH
# =========================================================

if len(sys.argv) < 2:

    print("=" * 70)
    print("STEP 25 - ACTUAL GNSS TRAJECTORY-DIRECTION ANALYSIS")
    print("=" * 70)

    print()
    print("Usage:")
    print(
        'python step25_gnss_trajectory_analysis.py '
        '"PATH_TO_CATEGORISED_IOVNB_DATASET"'
    )

    print()
    print("Example:")
    print(
        r'python step25_gnss_trajectory_analysis.py '
        r'"D:\Maverick\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"'
    )

    raise SystemExit(1)


SYNC_ROOT = os.path.abspath(sys.argv[1])


# =========================================================
# CHECK PATHS
# =========================================================

if not os.path.isdir(SYNC_ROOT):

    print()
    print("ERROR: Dataset directory does not exist:")
    print(SYNC_ROOT)

    raise SystemExit(1)


if not os.path.isdir(WINDOW_DIR):

    print()
    print("ERROR: Existing AVNet window directory does not exist:")
    print(WINDOW_DIR)

    print()
    print("Expected:")
    print(
        r"D:\Maverick\ML2\windows"
    )

    raise SystemExit(1)


# =========================================================
# COLUMN POSITIONS
#
# These are the verified 1-based V dataset positions.
# =========================================================

V_SATELLITES = 1
V_TIME = 2
V_LATITUDE = 3
V_LONGITUDE = 4
V_VELOCITY = 5
V_HEADING = 6
V_YAW_RATE = 15
V_LATERAL_ACCELERATION = 18


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def read_csv(path):

    try:

        return pd.read_csv(
            path,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        return pd.read_csv(
            path,
            encoding="cp1252"
        )


def numeric_column(df, one_based_position):

    return pd.to_numeric(
        df.iloc[:, one_based_position - 1],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


def circular_difference_deg(a, b):

    """
    Signed circular difference:

        a - b

    Result is in:

        [-180, +180)
    """

    return (
        (a - b + 180.0) % 360.0
    ) - 180.0


def circular_absolute_difference_deg(a, b):

    return np.abs(
        circular_difference_deg(
            a,
            b
        )
    )


def haversine_distance_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    """
    Distance between two latitude/longitude
    coordinates in metres.
    """

    R = 6371000.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lat = np.radians(
        lat2 - lat1
    )

    delta_lon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(delta_lat / 2.0) ** 2
        +
        np.cos(lat1_rad)
        *
        np.cos(lat2_rad)
        *
        np.sin(delta_lon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0
    )

    return (
        2.0
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


def bearing_deg(
    lat1,
    lon1,
    lat2,
    lon2
):

    """
    Initial bearing from point 1 to point 2.

    Bearing convention:

        0   = North
        90  = East
        180 = South
        270 = West
    """

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lon_rad = np.radians(
        lon2 - lon1
    )

    x = (
        np.sin(delta_lon_rad)
        *
        np.cos(lat2_rad)
    )

    y = (
        np.cos(lat1_rad)
        *
        np.sin(lat2_rad)
        -
        np.sin(lat1_rad)
        *
        np.cos(lat2_rad)
        *
        np.cos(delta_lon_rad)
    )

    bearing = np.degrees(
        np.arctan2(
            x,
            y
        )
    )

    return (
        bearing + 360.0
    ) % 360.0


def circular_median_deg(values):

    """
    Circular median.

    Used only as a descriptive statistic
    for the local GNSS segment bearings.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    best_value = values[0]
    best_cost = np.inf

    for candidate in values:

        cost = np.sum(
            np.abs(
                circular_difference_deg(
                    values,
                    candidate
                )
            )
        )

        if cost < best_cost:

            best_cost = cost
            best_value = candidate

    return best_value % 360.0


def circular_spread_deg(values):

    """
    Maximum pairwise circular separation
    among the provided angles.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:

        return 0.0

    maximum = 0.0

    for i in range(len(values)):

        for j in range(i + 1, len(values)):

            difference = abs(
                circular_difference_deg(
                    values[i],
                    values[j]
                )
            )

            maximum = max(
                maximum,
                difference
            )

    return maximum


def percentile_text(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return "No valid data"

    return (
        f"median={np.percentile(values, 50):.3f}, "
        f"P90={np.percentile(values, 90):.3f}, "
        f"P95={np.percentile(values, 95):.3f}, "
        f"P99={np.percentile(values, 99):.3f}, "
        f"max={np.max(values):.3f}"
    )


def print_stats(
    name,
    values
):

    print(
        f"{name}: "
        f"{percentile_text(values)}"
    )


# =========================================================
# FIND ALL S/V FILES
#
# S and V files may be in different folders.
# We therefore search recursively and globally.
# =========================================================

print("=" * 70)
print("STEP 25 - ACTUAL GNSS TRAJECTORY-DIRECTION ANALYSIS")
print("=" * 70)

print()
print("Dataset root:")
print(SYNC_ROOT)

print()
print("Existing AVNet windows:")
print(WINDOW_DIR)


all_s_files = {}
all_v_files = {}


for root, dirs, files in os.walk(
    SYNC_ROOT
):

    for filename in files:

        if not filename.lower().endswith(".csv"):
            continue

        full_path = os.path.join(
            root,
            filename
        )

        lower_name = filename.lower()

        if lower_name.startswith("s-"):

            key = os.path.splitext(
                filename[2:]
            )[0].lower()

            all_s_files[key] = full_path

        elif lower_name.startswith("v-"):

            key = os.path.splitext(
                filename[2:]
            )[0].lower()

            all_v_files[key] = full_path


matched_keys = sorted(
    set(all_s_files)
    &
    set(all_v_files)
)


print()
print(
    f"S files found: {len(all_s_files)}"
)

print(
    f"V files found: {len(all_v_files)}"
)

print(
    f"Matched S/V pairs: {len(matched_keys)}"
)


if len(matched_keys) == 0:

    print()
    print("ERROR: No S/V pairs were found.")

    print()
    print("Example S files found:")

    for key in list(
        all_s_files.keys()
    )[:10]:

        print(
            key,
            "->",
            all_s_files[key]
        )

    print()
    print("Example V files found:")

    for key in list(
        all_v_files.keys()
    )[:10]:

        print(
            key,
            "->",
            all_v_files[key]
        )

    raise SystemExit(1)


# =========================================================
# CHECK WINDOW FILES
# =========================================================

window_keys = set()

for filename in os.listdir(
    WINDOW_DIR
):

    if (
        filename.endswith(
            "_windows.npz"
        )
    ):

        key = filename[
            :-len("_windows.npz")
        ].lower()

        window_keys.add(
            key
        )


usable_keys = sorted(
    set(matched_keys)
    &
    window_keys
)


print()
print(
    f"Matched pairs with existing window files: "
    f"{len(usable_keys)}"
)


if len(usable_keys) == 0:

    print()
    print(
        "ERROR: No matched S/V datasets have "
        "corresponding AVNet window files."
    )

    raise SystemExit(1)


# =========================================================
# ANALYSIS
# =========================================================

all_rows = []


dataset_summary = []


for dataset_number, dataset_key in enumerate(
    usable_keys,
    start=1
):

    print()
    print(
        f"[{dataset_number}/{len(usable_keys)}] "
        f"Processing {dataset_key}"
    )

    s_path = all_s_files[
        dataset_key
    ]

    v_path = all_v_files[
        dataset_key
    ]

    window_path = os.path.join(
        WINDOW_DIR,
        f"{dataset_key}_windows.npz"
    )


    # -----------------------------------------------------
    # Load existing windows
    # -----------------------------------------------------

    try:

        window_data = np.load(
            window_path
        )

        X = window_data["X"]

    except Exception as error:

        print(
            f"  ERROR loading window file: {error}"
        )

        continue


    n_windows = X.shape[0]


    # -----------------------------------------------------
    # Load V data
    # -----------------------------------------------------

    try:

        v_df = read_csv(
            v_path
        )

    except Exception as error:

        print(
            f"  ERROR reading V file: {error}"
        )

        continue


    # -----------------------------------------------------
    # Extract verified V columns
    # -----------------------------------------------------

    try:

        satellites = numeric_column(
            v_df,
            V_SATELLITES
        )

        time_values = numeric_column(
            v_df,
            V_TIME
        )

        latitude = numeric_column(
            v_df,
            V_LATITUDE
        )

        longitude = numeric_column(
            v_df,
            V_LONGITUDE
        )

        velocity = numeric_column(
            v_df,
            V_VELOCITY
        )

        heading = numeric_column(
            v_df,
            V_HEADING
        )

        yaw_rate = numeric_column(
            v_df,
            V_YAW_RATE
        )

        lateral_acceleration = numeric_column(
            v_df,
            V_LATERAL_ACCELERATION
        )

    except Exception as error:

        print(
            f"  ERROR extracting V columns: {error}"
        )

        continue


    # -----------------------------------------------------
    # Common length
    # -----------------------------------------------------

    N = min(
        len(satellites),
        len(time_values),
        len(latitude),
        len(longitude),
        len(velocity),
        len(heading),
        len(yaw_rate),
        len(lateral_acceleration)
    )


    latitude = latitude[:N]
    longitude = longitude[:N]
    velocity = velocity[:N]
    heading = heading[:N]
    yaw_rate = yaw_rate[:N]
    lateral_acceleration = (
        lateral_acceleration[:N]
    )
    time_values = time_values[:N]
    satellites = satellites[:N]


    # -----------------------------------------------------
    # Analyse each exact 10-sample AVNet window
    # -----------------------------------------------------

    actual_windows = min(
        n_windows,
        N // 10
    )


    for window_index in range(
        actual_windows
    ):

        start_row = (
            window_index * 10
        )

        end_row = (
            start_row + 9
        )


        # =================================================
        # Window data
        # =================================================

        lat_w = latitude[
            start_row:end_row + 1
        ]

        lon_w = longitude[
            start_row:end_row + 1
        ]

        velocity_w = velocity[
            start_row:end_row + 1
        ]

        heading_w = heading[
            start_row:end_row + 1
        ]

        yaw_w = yaw_rate[
            start_row:end_row + 1
        ]

        lat_acc_w = lateral_acceleration[
            start_row:end_row + 1
        ]

        time_w = time_values[
            start_row:end_row + 1
        ]

        satellites_w = satellites[
            start_row:end_row + 1
        ]


        # =================================================
        # 1. GNSS segment bearings
        #
        # Each consecutive GPS position pair gives
        # a local trajectory direction.
        # =================================================

        segment_bearings = []
        segment_distances = []


        for i in range(9):

            required = [
                lat_w[i],
                lon_w[i],
                lat_w[i + 1],
                lon_w[i + 1]
            ]


            if not np.all(
                np.isfinite(required)
            ):

                continue


            distance = haversine_distance_m(
                lat_w[i],
                lon_w[i],
                lat_w[i + 1],
                lon_w[i + 1]
            )


            bearing = bearing_deg(
                lat_w[i],
                lon_w[i],
                lat_w[i + 1],
                lon_w[i + 1]
            )


            segment_distances.append(
                distance
            )

            segment_bearings.append(
                bearing
            )


        segment_bearings = np.asarray(
            segment_bearings,
            dtype=float
        )

        segment_distances = np.asarray(
            segment_distances,
            dtype=float
        )


        # =================================================
        # 2. GNSS start direction
        #
        # First half of the window:
        # sample 0 -> sample 4
        # =================================================

        start_bearing = np.nan


        if np.all(
            np.isfinite(
                [
                    lat_w[0],
                    lon_w[0],
                    lat_w[4],
                    lon_w[4]
                ]
            )
        ):

            start_distance = (
                haversine_distance_m(
                    lat_w[0],
                    lon_w[0],
                    lat_w[4],
                    lon_w[4]
                )
            )


            if start_distance > 0:

                start_bearing = bearing_deg(
                    lat_w[0],
                    lon_w[0],
                    lat_w[4],
                    lon_w[4]
                )


        # =================================================
        # 3. GNSS end direction
        #
        # Second half:
        # sample 5 -> sample 9
        # =================================================

        end_bearing = np.nan


        if np.all(
            np.isfinite(
                [
                    lat_w[5],
                    lon_w[5],
                    lat_w[9],
                    lon_w[9]
                ]
            )
        ):

            end_distance = (
                haversine_distance_m(
                    lat_w[5],
                    lon_w[5],
                    lat_w[9],
                    lon_w[9]
                )
            )


            if end_distance > 0:

                end_bearing = bearing_deg(
                    lat_w[5],
                    lon_w[5],
                    lat_w[9],
                    lon_w[9]
                )


        # =================================================
        # 4. Whole-window GNSS direction
        #
        # sample 0 -> sample 9
        # =================================================

        whole_bearing = np.nan


        if np.all(
            np.isfinite(
                [
                    lat_w[0],
                    lon_w[0],
                    lat_w[9],
                    lon_w[9]
                ]
            )
        ):

            whole_distance = (
                haversine_distance_m(
                    lat_w[0],
                    lon_w[0],
                    lat_w[9],
                    lon_w[9]
                )
            )


            if whole_distance > 0:

                whole_bearing = bearing_deg(
                    lat_w[0],
                    lon_w[0],
                    lat_w[9],
                    lon_w[9]
                )


        # =================================================
        # 5. GNSS direction change
        # =================================================

        if (
            np.isfinite(start_bearing)
            and
            np.isfinite(end_bearing)
        ):

            gnss_direction_change = (
                circular_difference_deg(
                    end_bearing,
                    start_bearing
                )
            )

        else:

            gnss_direction_change = np.nan


        # =================================================
        # 6. Total GNSS displacement
        # =================================================

        if np.all(
            np.isfinite(
                [
                    lat_w[0],
                    lon_w[0],
                    lat_w[9],
                    lon_w[9]
                ]
            )
        ):

            gnss_displacement = (
                haversine_distance_m(
                    lat_w[0],
                    lon_w[0],
                    lat_w[9],
                    lon_w[9]
                )
            )

        else:

            gnss_displacement = np.nan


        # =================================================
        # 7. Total path length inside window
        # =================================================

        if len(segment_distances):

            gnss_total_distance = (
                np.nansum(
                    segment_distances
                )
            )

        else:

            gnss_total_distance = np.nan


        # =================================================
        # 8. GNSS direction stability
        # =================================================

        if len(segment_bearings):

            gnss_bearing_median = (
                circular_median_deg(
                    segment_bearings
                )
            )

            gnss_bearing_spread = (
                circular_spread_deg(
                    segment_bearings
                )
            )

        else:

            gnss_bearing_median = np.nan
            gnss_bearing_spread = np.nan


        # =================================================
        # 9. Vehicle Heading change
        # =================================================

        valid_heading_indices = np.where(
            np.isfinite(
                heading_w
            )
        )[0]


        if len(
            valid_heading_indices
        ) >= 2:

            first_heading_index = (
                valid_heading_indices[0]
            )

            last_heading_index = (
                valid_heading_indices[-1]
            )

            heading_start = heading_w[
                first_heading_index
            ]

            heading_end = heading_w[
                last_heading_index
            ]

            heading_change = (
                circular_difference_deg(
                    heading_end,
                    heading_start
                )
            )

        else:

            heading_start = np.nan
            heading_end = np.nan
            heading_change = np.nan


        # =================================================
        # 10. Integrate vehicle yaw rate
        # =================================================

        valid_yaw = (
            np.isfinite(yaw_w)
            &
            np.isfinite(time_w)
        )


        if np.sum(valid_yaw) >= 2:

            t = time_w[
                valid_yaw
            ]

            y = yaw_w[
                valid_yaw
            ]


            # Dataset time is normally seconds.
            # If the difference is unusually large,
            # treat it as milliseconds.
            if len(t) >= 2:

                median_dt = np.nanmedian(
                    np.diff(t)
                )

            else:

                median_dt = np.nan


            if (
                np.isfinite(median_dt)
                and
                median_dt > 10
            ):

                t = t / 1000.0


            # Remove duplicate/non-increasing time points
            valid_time = np.ones(
                len(t),
                dtype=bool
            )

            if len(t) > 1:

                valid_time[1:] = (
                    np.diff(t) > 0
                )


            t = t[
                valid_time
            ]

            y = y[
                valid_time
            ]


            if len(t) >= 2:

                try:

                    yaw_integrated_change = (
                        np.trapezoid(
                            y,
                            t
                        )
                    )

                except AttributeError:

                    yaw_integrated_change = (
                        np.trapz(
                            y,
                            t
                        )
                    )

            else:

                yaw_integrated_change = np.nan

        else:

            yaw_integrated_change = np.nan


        # =================================================
        # 11. Heading vs GNSS
        # =================================================

        if (
            np.isfinite(
                heading_change
            )
            and
            np.isfinite(
                gnss_direction_change
            )
        ):

            heading_vs_gnss_error = (
                abs(
                    circular_difference_deg(
                        heading_change,
                        gnss_direction_change
                    )
                )
            )

        else:

            heading_vs_gnss_error = np.nan


        # =================================================
        # 12. Yaw Rate vs GNSS
        #
        # Both possible sign conventions are retained.
        # =================================================

        if (
            np.isfinite(
                yaw_integrated_change
            )
            and
            np.isfinite(
                gnss_direction_change
            )
        ):

            yaw_vs_gnss_signed_error = abs(
                gnss_direction_change
                -
                yaw_integrated_change
            )

            yaw_vs_gnss_inverted_error = abs(
                gnss_direction_change
                +
                yaw_integrated_change
            )

        else:

            yaw_vs_gnss_signed_error = np.nan
            yaw_vs_gnss_inverted_error = np.nan


        # =================================================
        # 13. Heading vs Yaw Rate
        #
        # Again retain both sign conventions.
        # =================================================

        if (
            np.isfinite(
                heading_change
            )
            and
            np.isfinite(
                yaw_integrated_change
            )
        ):

            heading_vs_yaw_signed_error = abs(
                heading_change
                -
                yaw_integrated_change
            )

            heading_vs_yaw_inverted_error = abs(
                heading_change
                +
                yaw_integrated_change
            )

        else:

            heading_vs_yaw_signed_error = np.nan
            heading_vs_yaw_inverted_error = np.nan


        # =================================================
        # 14. Absolute Heading vs GNSS direction
        # =================================================

        if (
            np.isfinite(
                heading_end
            )
            and
            np.isfinite(
                whole_bearing
            )
        ):

            heading_vs_gnss_absolute_error = (
                abs(
                    circular_difference_deg(
                        heading_end,
                        whole_bearing
                    )
                )
            )

        else:

            heading_vs_gnss_absolute_error = np.nan


        # =================================================
        # 15. Window statistics
        # =================================================

        if np.any(
            np.isfinite(
                velocity_w
            )
        ):

            mean_speed = np.nanmean(
                velocity_w
            )

        else:

            mean_speed = np.nan


        if np.any(
            np.isfinite(
                satellites_w
            )
        ):

            mean_satellites = np.nanmean(
                satellites_w
            )

        else:

            mean_satellites = np.nan


        if np.any(
            np.isfinite(
                lat_acc_w
            )
        ):

            mean_abs_lateral_acc = (
                np.nanmean(
                    np.abs(
                        lat_acc_w
                    )
                )
            )

        else:

            mean_abs_lateral_acc = np.nan


        # =================================================
        # SAVE ROW
        # =================================================

        all_rows.append(
            {
                "dataset":
                    dataset_key,

                "window_index":
                    window_index,

                "start_row":
                    start_row,

                "end_row":
                    end_row,

                "mean_speed_kmh":
                    mean_speed,

                "gnss_displacement_m":
                    gnss_displacement,

                "gnss_total_distance_m":
                    gnss_total_distance,

                "gnss_start_bearing_deg":
                    start_bearing,

                "gnss_end_bearing_deg":
                    end_bearing,

                "gnss_whole_bearing_deg":
                    whole_bearing,

                "gnss_direction_change_deg":
                    gnss_direction_change,

                "gnss_bearing_median_deg":
                    gnss_bearing_median,

                "gnss_bearing_spread_deg":
                    gnss_bearing_spread,

                "heading_start_deg":
                    heading_start,

                "heading_end_deg":
                    heading_end,

                "heading_change_deg":
                    heading_change,

                "yawrate_integrated_change_deg":
                    yaw_integrated_change,

                "heading_vs_gnss_error_deg":
                    heading_vs_gnss_error,

                "yawrate_vs_gnss_signed_error_deg":
                    yaw_vs_gnss_signed_error,

                "yawrate_vs_gnss_inverted_error_deg":
                    yaw_vs_gnss_inverted_error,

                "heading_vs_yawrate_signed_error_deg":
                    heading_vs_yaw_signed_error,

                "heading_vs_yawrate_inverted_error_deg":
                    heading_vs_yaw_inverted_error,

                "heading_vs_gnss_absolute_error_deg":
                    heading_vs_gnss_absolute_error,

                "mean_abs_lateral_acc":
                    mean_abs_lateral_acc,

                "mean_gps_satellites":
                    mean_satellites
            }
        )


    print(
        f"  Windows analysed: {actual_windows:,}"
    )


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    all_rows
)


if len(df) == 0:

    print()
    print(
        "ERROR: Zero windows were analysed."
    )

    raise SystemExit(1)


# =========================================================
# SAVE DETAILED RESULTS
# =========================================================

df.to_csv(
    OUT_FILE,
    index=False
)


print()
print(
    f"Exact windows analysed: {len(df):,}"
)

print(
    f"Saved detailed results:"
)

print(
    OUT_FILE
)


# =========================================================
# OVERALL RESULTS
# =========================================================

print()
print("=" * 70)
print("OVERALL GNSS TRAJECTORY RESULTS")
print("=" * 70)


print_stats(
    "GNSS direction change |deg|",
    np.abs(
        df[
            "gnss_direction_change_deg"
        ]
    )
)


print_stats(
    "Heading vs GNSS error |deg|",
    df[
        "heading_vs_gnss_error_deg"
    ]
)


print_stats(
    "YawRate vs GNSS signed error |deg|",
    df[
        "yawrate_vs_gnss_signed_error_deg"
    ]
)


print_stats(
    "YawRate vs GNSS inverted error |deg|",
    df[
        "yawrate_vs_gnss_inverted_error_deg"
    ]
)


print_stats(
    "Heading vs YawRate signed error |deg|",
    df[
        "heading_vs_yawrate_signed_error_deg"
    ]
)


print_stats(
    "Heading vs YawRate inverted error |deg|",
    df[
        "heading_vs_yawrate_inverted_error_deg"
    ]
)


# =========================================================
# SPEED BIN ANALYSIS
# =========================================================

print()
print("=" * 70)
print("RESULTS BY VEHICLE SPEED")
print("=" * 70)


speed_bins = [
    (-np.inf, 1.0, "0-1"),
    (1.0, 5.0, "1-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 20.0, "10-20"),
    (20.0, 40.0, "20-40"),
    (40.0, 60.0, "40-60"),
    (60.0, np.inf, ">60")
]


for lower, upper, label in speed_bins:

    subset = df[
        (
            df["mean_speed_kmh"]
            >
            lower
        )
        &
        (
            df["mean_speed_kmh"]
            <=
            upper
        )
    ]


    print()
    print(
        f"Speed {label} km/h: "
        f"{len(subset):,} windows"
    )


    if len(subset) == 0:

        continue


    print_stats(
        "  GNSS direction change |deg|",
        np.abs(
            subset[
                "gnss_direction_change_deg"
            ]
        )
    )


    print_stats(
        "  Heading-GNSS error |deg|",
        subset[
            "heading_vs_gnss_error_deg"
        ]
    )


    print_stats(
        "  YawRate-GNSS signed |deg|",
        subset[
            "yawrate_vs_gnss_signed_error_deg"
        ]
    )


    print_stats(
        "  YawRate-GNSS inverted |deg|",
        subset[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
    )


# =========================================================
# HEADING AGREEMENT WITH GNSS
# =========================================================

print()
print("=" * 70)
print("HEADING VS GNSS AGREEMENT")
print("=" * 70)


valid_heading_gnss = df[
    np.isfinite(
        df[
            "heading_vs_gnss_error_deg"
        ]
    )
]


for threshold in [
    2,
    5,
    10,
    20,
    45
]:

    if len(valid_heading_gnss) == 0:

        break


    percentage = (
        np.mean(
            valid_heading_gnss[
                "heading_vs_gnss_error_deg"
            ]
            <= threshold
        )
        *
        100.0
    )


    print(
        f"<= {threshold:>2}° : "
        f"{percentage:.3f}%"
    )


# =========================================================
# YAW RATE SIGN CONVENTION CHECK
# =========================================================

print()
print("=" * 70)
print("YAW RATE SIGN-CONVENTION CHECK")
print("=" * 70)


signed_valid = df[
    np.isfinite(
        df[
            "yawrate_vs_gnss_signed_error_deg"
        ]
    )
]


inverted_valid = df[
    np.isfinite(
        df[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
    )
]


if (
    len(signed_valid) > 0
    and
    len(inverted_valid) > 0
):

    signed_median = np.median(
        signed_valid[
            "yawrate_vs_gnss_signed_error_deg"
        ]
    )


    inverted_median = np.median(
        inverted_valid[
            "yawrate_vs_gnss_inverted_error_deg"
        ]
    )


    print(
        f"Signed yaw-rate median error: "
        f"{signed_median:.3f}°"
    )


    print(
        f"Inverted yaw-rate median error: "
        f"{inverted_median:.3f}°"
    )


    if signed_median < inverted_median:

        print(
            "Signed convention has the smaller "
            "median error."
        )

    elif inverted_median < signed_median:

        print(
            "Inverted convention has the smaller "
            "median error."
        )

    else:

        print(
            "The two sign conventions have equal "
            "median error."
        )


# =========================================================
# LARGE GNSS DIRECTION CHANGES
# =========================================================

print()
print("=" * 70)
print("LARGE GNSS TRAJECTORY DIRECTION CHANGES")
print("=" * 70)


large_gnss = df[
    np.abs(
        df[
            "gnss_direction_change_deg"
        ]
    )
    >= 45.0
]


print(
    f">=45° GNSS direction changes: "
    f"{len(large_gnss):,}"
)


if len(large_gnss) > 0:

    display_columns = [
        "dataset",
        "window_index",
        "mean_speed_kmh",
        "gnss_displacement_m",
        "gnss_direction_change_deg",
        "heading_change_deg",
        "yawrate_integrated_change_deg",
        "heading_vs_gnss_error_deg"
    ]


    print(
        large_gnss[
            display_columns
        ]
        .sort_values(
            "gnss_displacement_m"
        )
        .tail(20)
        .to_string(
            index=False
        )
    )


# =========================================================
# WORST HEADING VS GNSS CASES
# =========================================================

print()
print("=" * 70)
print("WORST HEADING VS GNSS DISAGREEMENTS")
print("=" * 70)


worst_heading = df[
    np.isfinite(
        df[
            "heading_vs_gnss_error_deg"
        ]
    )
].sort_values(
    "heading_vs_gnss_error_deg",
    ascending=False
).head(20)


if len(worst_heading) > 0:

    print(
        worst_heading[
            [
                "dataset",
                "window_index",
                "mean_speed_kmh",
                "gnss_displacement_m",
                "gnss_start_bearing_deg",
                "gnss_end_bearing_deg",
                "gnss_direction_change_deg",
                "heading_change_deg",
                "heading_vs_gnss_error_deg"
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# WORST YAW RATE VS GNSS CASES
# =========================================================

print()
print("=" * 70)
print("WORST YAW RATE VS GNSS DISAGREEMENTS")
print("=" * 70)


worst_yaw = df[
    np.isfinite(
        df[
            "yawrate_vs_gnss_signed_error_deg"
        ]
    )
].sort_values(
    "yawrate_vs_gnss_signed_error_deg",
    ascending=False
).head(20)


if len(worst_yaw) > 0:

    print(
        worst_yaw[
            [
                "dataset",
                "window_index",
                "mean_speed_kmh",
                "gnss_displacement_m",
                "gnss_direction_change_deg",
                "yawrate_integrated_change_deg",
                "yawrate_vs_gnss_signed_error_deg",
                "yawrate_vs_gnss_inverted_error_deg"
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# DATASET-LEVEL SUMMARY
# =========================================================

for dataset_key, group in df.groupby(
    "dataset"
):

    dataset_summary.append(
        {
            "dataset":
                dataset_key,

            "windows":
                len(group),

            "median_speed_kmh":
                group[
                    "mean_speed_kmh"
                ].median(),

            "median_gnss_displacement_m":
                group[
                    "gnss_displacement_m"
                ].median(),

            "median_gnss_direction_change_deg":
                np.nanmedian(
                    np.abs(
                        group[
                            "gnss_direction_change_deg"
                        ]
                    )
                ),

            "median_heading_vs_gnss_error_deg":
                np.nanmedian(
                    group[
                        "heading_vs_gnss_error_deg"
                    ]
                ),

            "median_yawrate_vs_gnss_signed_error_deg":
                np.nanmedian(
                    group[
                        "yawrate_vs_gnss_signed_error_deg"
                    ]
                ),

            "median_yawrate_vs_gnss_inverted_error_deg":
                np.nanmedian(
                    group[
                        "yawrate_vs_gnss_inverted_error_deg"
                    ]
                ),

            "median_heading_vs_yawrate_signed_error_deg":
                np.nanmedian(
                    group[
                        "heading_vs_yawrate_signed_error_deg"
                    ]
                ),

            "large_gnss_direction_changes":
                np.sum(
                    np.abs(
                        group[
                            "gnss_direction_change_deg"
                        ]
                    )
                    >= 45.0
                )
        }
    )


summary_df = pd.DataFrame(
    dataset_summary
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print()
print(
    f"Saved dataset summary:"
)

print(
    SUMMARY_FILE
)


# =========================================================
# FINAL CHECKS
# =========================================================

print()
print("=" * 70)
print("FINAL CHECKS")
print("=" * 70)


print(
    f"Datasets analysed: "
    f"{df['dataset'].nunique()}"
)


print(
    f"Total windows: "
    f"{len(df):,}"
)


print(
    f"Expected AVNet windows: "
    f"107,043"
)


if len(df) == 107043:

    print(
        "✓ Window count matches the "
        "existing AVNet dataset."
    )

else:

    print(
        "WARNING: Window count does not "
        "match 107,043."
    )


print()
print(
    "Step 25 completed successfully."
)

print()
print(
    "IMPORTANT:"
)

print(
    "No signal has been declared as ground truth."
)

print(
    "GNSS trajectory direction, V Heading, "
    "and V Yaw Rate remain comparison candidates."
)

print("=" * 70)
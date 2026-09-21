from flask import Flask, request, jsonify
import csv
import os
import threading
from datetime import datetime


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FILE NAMES
# ============================================================

SENSOR_CSV = "sensor_data.csv"
GNSS_CSV = "gps_gnss_data.csv"


# ============================================================
# LOCK
# ============================================================

csv_lock = threading.Lock()


# ============================================================
# SENSOR CSV COLUMNS
# ============================================================

SENSOR_HEADERS = [
    "timestamp",

    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",

    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",

    "gravity_x",
    "gravity_y",
    "gravity_z",

    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",

    "magnetic_field_x",
    "magnetic_field_y",
    "magnetic_field_z",

    "game_rotation_qx",
    "game_rotation_qy",
    "game_rotation_qz",
    "game_rotation_qw",

    "rotation_qx",
    "rotation_qy",
    "rotation_qz",
    "rotation_qw",

    "geomag_rotation_qx",
    "geomag_rotation_qy",
    "geomag_rotation_qz",
    "geomag_rotation_qw",

    "orientation_azimuth",
    "orientation_pitch",
    "orientation_roll"
]


# ============================================================
# GNSS CSV COLUMNS
# ============================================================

GNSS_HEADERS = [
    "timestamp",

    # Device
    "device_id",

    # Location
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "bearing",
    "accuracy",
    "vertical_accuracy",
    "speed_accuracy",
    "bearing_accuracy",

    # GNSS clock
    "time_nanos",
    "full_bias_nanos",
    "bias_nanos",
    "bias_uncertainty_nanos",
    "drift_nanos_per_second",
    "drift_uncertainty",
    "time_uncertainty_nanos",

    # Satellite
    "svid",
    "constellation",
    "cn0_dbhz",
    "carrier_frequency_hz",

    "pseudorange_rate_mps",
    "pseudorange_rate_uncertainty_mps",

    "received_sv_time_nanos",
    "received_sv_time_uncertainty_nanos",

    "state",

    "accumulated_delta_range_m",
    "accumulated_delta_range_uncertainty_m",
    "accumulated_delta_range_state",

    "multipath"
]


# ============================================================
# CREATE CSV IF IT DOES NOT EXIST
# ============================================================

def create_csv_if_needed(filename, headers):

    if not os.path.exists(filename):

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(headers)

        print(f"Created: {filename}")

        return


    # File exists but may be empty
    if os.path.getsize(filename) == 0:

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(headers)

        print(f"Initialized: {filename}")


# ============================================================
# FORMAT FLOAT
# ============================================================

def format_float(value):

    if value is None:
        return ""

    try:

        return f"{float(value):.4f}"

    except (ValueError, TypeError):

        return ""


# ============================================================
# GET VALUE
# ============================================================

def get_value(dictionary, key):

    if not isinstance(dictionary, dict):
        return None

    return dictionary.get(key)


# ============================================================
# GET SENSOR VALUE
# ============================================================

def get_sensor_value(sensor_data, sensor_name, value_name):

    sensor = sensor_data.get(sensor_name)

    if not isinstance(sensor, dict):
        return None

    return sensor.get(value_name)


# ============================================================
# APPEND SENSOR ROW
# ============================================================

def append_sensor_row(data):

    payload = data.get("payload", [])

    sensor_values = {}

    if isinstance(payload, list):

        for sensor in payload:

            if not isinstance(sensor, dict):
                continue

            name = sensor.get("name")

            values = sensor.get("values")

            if name and isinstance(values, dict):

                sensor_values[name] = values


    timestamp = data.get(
        "timestamp",
        int(datetime.now().timestamp() * 1000)
    )


    row = [

        timestamp,

        # Accelerometer
        format_float(
            get_sensor_value(
                sensor_values,
                "accelerometer",
                "x"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "accelerometer",
                "y"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "accelerometer",
                "z"
            )
        ),


        # Gyroscope
        format_float(
            get_sensor_value(
                sensor_values,
                "gyroscope",
                "x"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "gyroscope",
                "y"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "gyroscope",
                "z"
            )
        ),


        # Gravity
        format_float(
            get_sensor_value(
                sensor_values,
                "gravity",
                "x"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "gravity",
                "y"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "gravity",
                "z"
            )
        ),


        # Linear acceleration
        format_float(
            get_sensor_value(
                sensor_values,
                "linear_acceleration",
                "x"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "linear_acceleration",
                "y"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "linear_acceleration",
                "z"
            )
        ),


        # Magnetic field
        format_float(
            get_sensor_value(
                sensor_values,
                "magnetic_field",
                "x"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "magnetic_field",
                "y"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "magnetic_field",
                "z"
            )
        ),


        # Game rotation vector
        format_float(
            get_sensor_value(
                sensor_values,
                "game_rotation",
                "qx"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "game_rotation",
                "qy"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "game_rotation",
                "qz"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "game_rotation",
                "qw"
            )
        ),


        # Rotation vector
        format_float(
            get_sensor_value(
                sensor_values,
                "rotation",
                "qx"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "rotation",
                "qy"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "rotation",
                "qz"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "rotation",
                "qw"
            )
        ),


        # GeoMagnetic rotation vector
        format_float(
            get_sensor_value(
                sensor_values,
                "geomag_rotation",
                "qx"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "geomag_rotation",
                "qy"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "geomag_rotation",
                "qz"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "geomag_rotation",
                "qw"
            )
        ),


        # Orientation
        format_float(
            get_sensor_value(
                sensor_values,
                "orientation",
                "azimuth"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "orientation",
                "pitch"
            )
        ),

        format_float(
            get_sensor_value(
                sensor_values,
                "orientation",
                "roll"
            )
        )
    ]


    with csv_lock:

        with open(
            SENSOR_CSV,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(row)


# ============================================================
# APPEND GNSS ROWS
# ============================================================

def append_gnss_rows(data):

    timestamp = data.get(
        "timestamp",
        int(datetime.now().timestamp() * 1000)
    )


    device_id = data.get(
        "deviceId",
        ""
    )


    location = data.get(
        "location",
        {}
    )


    clock = data.get(
        "clock",
        {}
    )


    satellites = data.get(
        "satellites",
        []
    )


    if not isinstance(satellites, list):

        satellites = []


    # --------------------------------------------------------
    # If there are no satellites, still save one row
    # --------------------------------------------------------

    if len(satellites) == 0:

        row = [

            timestamp,

            device_id,

            format_float(
                get_value(location, "latitude")
            ),

            format_float(
                get_value(location, "longitude")
            ),

            format_float(
                get_value(location, "altitude")
            ),

            format_float(
                get_value(location, "speed")
            ),

            format_float(
                get_value(location, "bearing")
            ),

            format_float(
                get_value(location, "accuracy")
            ),

            format_float(
                get_value(
                    location,
                    "vertical_accuracy"
                )
            ),

            format_float(
                get_value(
                    location,
                    "speed_accuracy"
                )
            ),

            format_float(
                get_value(
                    location,
                    "bearing_accuracy"
                )
            ),

            get_value(
                clock,
                "time_nanos"
            ),

            get_value(
                clock,
                "full_bias_nanos"
            ),

            get_value(
                clock,
                "bias_nanos"
            ),

            get_value(
                clock,
                "bias_uncertainty_nanos"
            ),

            get_value(
                clock,
                "drift_nanos_per_second"
            ),

            get_value(
                clock,
                "drift_uncertainty"
            ),

            get_value(
                clock,
                "time_uncertainty_nanos"
            ),

            "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        ]


        with csv_lock:

            with open(
                GNSS_CSV,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow(row)

        return


    # --------------------------------------------------------
    # One row per satellite
    # --------------------------------------------------------

    with csv_lock:

        with open(
            GNSS_CSV,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)


            for satellite in satellites:

                if not isinstance(
                    satellite,
                    dict
                ):

                    continue


                row = [

                    timestamp,

                    device_id,


                    # -----------------------------
                    # Location
                    # -----------------------------

                    format_float(
                        get_value(
                            location,
                            "latitude"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "longitude"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "altitude"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "speed"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "bearing"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "accuracy"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "vertical_accuracy"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "speed_accuracy"
                        )
                    ),

                    format_float(
                        get_value(
                            location,
                            "bearing_accuracy"
                        )
                    ),


                    # -----------------------------
                    # GNSS clock
                    # -----------------------------

                    get_value(
                        clock,
                        "time_nanos"
                    ),

                    get_value(
                        clock,
                        "full_bias_nanos"
                    ),

                    get_value(
                        clock,
                        "bias_nanos"
                    ),

                    get_value(
                        clock,
                        "bias_uncertainty_nanos"
                    ),

                    get_value(
                        clock,
                        "drift_nanos_per_second"
                    ),

                    get_value(
                        clock,
                        "drift_uncertainty"
                    ),

                    get_value(
                        clock,
                        "time_uncertainty_nanos"
                    ),


                    # -----------------------------
                    # Satellite
                    # -----------------------------

                    get_value(
                        satellite,
                        "svid"
                    ),

                    get_value(
                        satellite,
                        "constellation"
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "cn0_dbhz"
                        )
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "carrier_frequency_hz"
                        )
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "pseudorange_rate_mps"
                        )
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "pseudorange_rate_uncertainty_mps"
                        )
                    ),

                    get_value(
                        satellite,
                        "received_sv_time_nanos"
                    ),

                    get_value(
                        satellite,
                        "received_sv_time_uncertainty_nanos"
                    ),

                    get_value(
                        satellite,
                        "state"
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "accumulated_delta_range_m"
                        )
                    ),

                    format_float(
                        get_value(
                            satellite,
                            "accumulated_delta_range_uncertainty_m"
                        )
                    ),

                    get_value(
                        satellite,
                        "accumulated_delta_range_state"
                    ),

                    get_value(
                        satellite,
                        "multipath"
                    )
                ]


                writer.writerow(row)


# ============================================================
# SENSOR ENDPOINT
# ============================================================

@app.route(
    "/data",
    methods=["POST"]
)
def receive_sensor_data():

    try:

        data = request.get_json(
            silent=True
        )


        if not isinstance(data, dict):

            return jsonify({

                "status": "error",

                "message":
                    "Invalid JSON data"

            }), 400


        append_sensor_row(data)


        return jsonify({

            "status": "ok",

            "message":
                "Sensor data received"

        }), 200


    except Exception as e:

        print(
            "Sensor error:",
            str(e)
        )


        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ============================================================
# GNSS ENDPOINT
# ============================================================

@app.route(
    "/gnss-data",
    methods=["POST"]
)
def receive_gnss_data():

    try:

        data = request.get_json(
            silent=True
        )


        if not isinstance(data, dict):

            return jsonify({

                "status": "error",

                "message":
                    "Invalid JSON data"

            }), 400


        append_gnss_rows(data)


        return jsonify({

            "status": "ok",

            "message":
                "GNSS data received"

        }), 200


    except Exception as e:

        print(
            "GNSS error:",
            str(e)
        )


        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return """

    <html>

    <head>
        <title>MAVERICK DataSense Server</title>
    </head>

    <body>

        <h1>MAVERICK DataSense Server is running!</h1>

        <p>Phone sensor endpoint:</p>
        <code>POST /data</code>

        <p>GPS/GNSS endpoint:</p>
        <code>POST /gnss-data</code>

        <p>Sensor CSV:</p>
        <code>sensor_data.csv</code>

        <p>GNSS CSV:</p>
        <code>gps_gnss_data.csv</code>

    </body>

    </html>

    """


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MAVERICK DataSense Server")
    print("=" * 60)


    # Create CSV files only if necessary
    create_csv_if_needed(
        SENSOR_CSV,
        SENSOR_HEADERS
    )


    create_csv_if_needed(
        GNSS_CSV,
        GNSS_HEADERS
    )


    print()
    print(
        "Sensor CSV :",
        os.path.abspath(SENSOR_CSV)
    )

    print(
        "GNSS CSV   :",
        os.path.abspath(GNSS_CSV)
    )

    print()
    print(
        "Server running on:"
    )

    print(
        "http://0.0.0.0:8000"
    )

    print(
        "Phone should use:"
    )

    print(
        "http://192.168.0.100:8000"
    )

    print()
    print("=" * 60)
    print()


    app.run(

        host="0.0.0.0",

        port=8000,

        threaded=True
    )
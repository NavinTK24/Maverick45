import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# PATHS
# ============================================================

BASE = r"D:\Maverick\ML2"

DATA_DIR = os.path.join(BASE, "step44_normalized_data")
DDODO_DIR = os.path.join(BASE, "step45_ddodo_data")
MODEL_DIR = os.path.join(BASE, "step47_training")

METADATA_FILE = os.path.join(
    BASE,
    "avnet_yaw_reference_metadata.csv"
)

SPLIT_FILE = os.path.join(
    BASE,
    "step38_avnet_data",
    "sequence_split.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "step49_diagnostics"
)

os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 1024
DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class AVNet(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.conv1 = nn.Conv1d(
            6, 128, kernel_size=3
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            128, 256, kernel_size=2
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(
            kernel_size=2
        )

        self.fc1 = nn.Linear(
            256, 1024
        )

        self.fc2 = nn.Linear(
            1024, 512
        )

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=512,
            batch_first=True
        )

        self.output = nn.Linear(
            512, output_size
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = x.unsqueeze(1)

        x, _ = self.gru(x)

        x = x[:, -1, :]

        return self.output(x)


# ============================================================
# LOAD TEST DATA
# ============================================================

att_data = np.load(
    os.path.join(
        DATA_DIR,
        "test.npz"
    )
)

X_att = att_data["X"].astype(np.float32)
Y_att = att_data["Y"].astype(np.float32)


odo_data = np.load(
    os.path.join(
        DDODO_DIR,
        "test.npz"
    )
)

X_odo = odo_data["X"].astype(np.float32)
Y_odo = odo_data["Y"].astype(np.float32)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

split = pd.read_csv(
    SPLIT_FILE
)

test_sequences = split.loc[
    split["split"].str.upper() == "TEST",
    "dataset"
].tolist()

print("=" * 70)
print("STEP 49 - AVNET SEQUENCE DIAGNOSTICS")
print("=" * 70)

print("\nTest sequences:")
for name in test_sequences:
    print(" ", name)

print("\nTotal test samples:", len(X_att))


# ============================================================
# CHECK METADATA ORDER
# ============================================================

test_meta = metadata[
    metadata["dataset"].isin(test_sequences)
].copy()

test_meta = test_meta.sort_values(
    "sample_index"
).reset_index(drop=True)


if len(test_meta) != len(X_att):

    raise RuntimeError(
        f"Metadata/test mismatch: "
        f"{len(test_meta)} vs {len(X_att)}"
    )


metadata_dataset = (
    test_meta["dataset"]
    .astype(str)
    .to_numpy()
)

# Verify the metadata corresponds to the test data
# in the expected sample-index order.

sample_indices = (
    test_meta["sample_index"]
    .to_numpy()
)

print(
    "\nMetadata rows:",
    len(test_meta)
)

print(
    "Sample index range:",
    sample_indices.min(),
    "to",
    sample_indices.max()
)


# ============================================================
# LOAD MODELS
# ============================================================

ddatt_model = AVNet(3).to(DEVICE)
ddodo_model = AVNet(1).to(DEVICE)

ddatt_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "best_ddatt_model.pt"
        ),
        map_location=DEVICE
    )
)

ddodo_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "best_ddodo_model.pt"
        ),
        map_location=DEVICE
    )
)

ddatt_model.eval()
ddodo_model.eval()


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X):

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    output = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(DEVICE)

            pred = model(batch)

            output.append(
                pred.cpu().numpy()
            )

    return np.concatenate(
        output,
        axis=0
    )


print("\nGenerating predictions...")

pred_att = predict(
    ddatt_model,
    X_att
)

pred_odo = predict(
    ddodo_model,
    X_odo
)


# ============================================================
# CONVERT DDATT TO YAW
# ============================================================

reference_yaw = (
    2
    * np.arcsin(
        np.clip(
            Y_att[:, 2],
            -1,
            1
        )
    )
    * 180
    / np.pi
)

predicted_yaw = (
    2
    * np.arcsin(
        np.clip(
            pred_att[:, 2],
            -1,
            1
        )
    )
    * 180
    / np.pi
)

yaw_error = (
    predicted_yaw
    - reference_yaw
)

yaw_abs_error = np.abs(
    yaw_error
)


# ============================================================
# DDODO
# ============================================================

reference_velocity = Y_odo[:, 0]

predicted_velocity = pred_odo[:, 0]

velocity_error = (
    predicted_velocity
    - reference_velocity
)

velocity_abs_error = np.abs(
    velocity_error
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def metrics(error):

    absolute = np.abs(error)

    return {
        "samples": len(error),
        "MAE": np.mean(absolute),
        "RMSE": np.sqrt(
            np.mean(error ** 2)
        ),
        "Median": np.median(absolute),
        "P90": np.percentile(
            absolute,
            90
        ),
        "P95": np.percentile(
            absolute,
            95
        ),
        "P99": np.percentile(
            absolute,
            99
        ),
        "Maximum": np.max(absolute)
    }


# ============================================================
# PER-SEQUENCE ANALYSIS
# ============================================================

rows = []

print("\n")
print("=" * 70)
print("PER-SEQUENCE RESULTS")
print("=" * 70)

for dataset in test_sequences:

    mask = (
        metadata_dataset
        == dataset
    )

    if mask.sum() == 0:
        continue

    att_metrics = metrics(
        yaw_error[mask]
    )

    odo_metrics = metrics(
        velocity_error[mask]
    )

    rows.append({
        "dataset": dataset,
        "samples": mask.sum(),

        "DDATT_MAE_deg":
            att_metrics["MAE"],

        "DDATT_RMSE_deg":
            att_metrics["RMSE"],

        "DDATT_Median_deg":
            att_metrics["Median"],

        "DDATT_P90_deg":
            att_metrics["P90"],

        "DDODO_MAE_kmh":
            odo_metrics["MAE"],

        "DDODO_RMSE_kmh":
            odo_metrics["RMSE"],

        "DDODO_Median_kmh":
            odo_metrics["Median"],

        "DDODO_P90_kmh":
            odo_metrics["P90"],

        "Reference_velocity_mean_kmh":
            np.mean(
                reference_velocity[mask]
            ),

        "Reference_velocity_max_kmh":
            np.max(
                reference_velocity[mask]
            ),

        "Reference_yaw_abs_mean_deg":
            np.mean(
                np.abs(
                    reference_yaw[mask]
                )
            )
    })


sequence_results = pd.DataFrame(rows)

print(
    sequence_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# VELOCITY-RANGE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("DDODO PERFORMANCE BY VELOCITY")
print("=" * 70)


velocity_ranges = [
    ("0-5 km/h", 0, 5),
    ("5-20 km/h", 5, 20),
    ("20-40 km/h", 20, 40),
    ("40-60 km/h", 40, 60),
    ("60-80 km/h", 60, 80),
    ("80+ km/h", 80, np.inf)
]

velocity_rows = []

for label, low, high in velocity_ranges:

    mask = (
        (reference_velocity >= low)
        &
        (reference_velocity < high)
    )

    if mask.sum() == 0:
        continue

    m = metrics(
        velocity_error[mask]
    )

    velocity_rows.append({
        "velocity_range": label,
        "samples": mask.sum(),
        "MAE_kmh": m["MAE"],
        "RMSE_kmh": m["RMSE"],
        "Median_kmh": m["Median"],
        "P90_kmh": m["P90"]
    })

velocity_results = pd.DataFrame(
    velocity_rows
)

print(
    velocity_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DDATT BY YAW-CHANGE MAGNITUDE
# ============================================================

print("\n")
print("=" * 70)
print("DDATT PERFORMANCE BY REFERENCE YAW CHANGE")
print("=" * 70)


yaw_ranges = [
    ("0-1 deg", 0, 1),
    ("1-5 deg", 1, 5),
    ("5-10 deg", 5, 10),
    ("10-20 deg", 10, 20),
    ("20-45 deg", 20, 45),
    ("45+ deg", 45, np.inf)
]

yaw_rows = []

reference_yaw_abs = np.abs(
    reference_yaw
)

for label, low, high in yaw_ranges:

    mask = (
        (reference_yaw_abs >= low)
        &
        (reference_yaw_abs < high)
    )

    if mask.sum() == 0:
        continue

    m = metrics(
        yaw_error[mask]
    )

    yaw_rows.append({
        "yaw_range": label,
        "samples": mask.sum(),
        "MAE_deg": m["MAE"],
        "RMSE_deg": m["RMSE"],
        "Median_deg": m["Median"],
        "P90_deg": m["P90"]
    })

yaw_results = pd.DataFrame(
    yaw_rows
)

print(
    yaw_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE DETAILED SAMPLE RESULTS
# ============================================================

sample_results = pd.DataFrame({

    "sample_index":
        sample_indices,

    "dataset":
        metadata_dataset,

    "reference_yaw_deg":
        reference_yaw,

    "predicted_yaw_deg":
        predicted_yaw,

    "yaw_error_deg":
        yaw_error,

    "yaw_abs_error_deg":
        yaw_abs_error,

    "reference_velocity_kmh":
        reference_velocity,

    "predicted_velocity_kmh":
        predicted_velocity,

    "velocity_error_kmh":
        velocity_error,

    "velocity_abs_error_kmh":
        velocity_abs_error
})

sample_results.to_csv(
    os.path.join(
        OUT_DIR,
        "sample_level_diagnostics.csv"
    ),
    index=False
)

sequence_results.to_csv(
    os.path.join(
        OUT_DIR,
        "sequence_results.csv"
    ),
    index=False
)

velocity_results.to_csv(
    os.path.join(
        OUT_DIR,
        "velocity_range_results.csv"
    ),
    index=False
)

yaw_results.to_csv(
    os.path.join(
        OUT_DIR,
        "yaw_range_results.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    os.path.join(
        OUT_DIR,
        "sample_level_diagnostics.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "sequence_results.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "velocity_range_results.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "yaw_range_results.csv"
    )
)

print("\nSTEP 49 COMPLETE")
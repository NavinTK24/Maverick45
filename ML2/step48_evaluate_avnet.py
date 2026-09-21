import os
import numpy as np
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

OUT_DIR = os.path.join(BASE, "step48_evaluation")
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
    os.path.join(DATA_DIR, "test.npz")
)

X_att = att_data["X"].astype(np.float32)
Y_att = att_data["Y"].astype(np.float32)

odo_data = np.load(
    os.path.join(DDODO_DIR, "test.npz")
)

X_odo = odo_data["X"].astype(np.float32)
Y_odo = odo_data["Y"].astype(np.float32)


print("=" * 70)
print("STEP 48 - AVNET TEST EVALUATION")
print("=" * 70)

print("\nDDATT test:")
print("X:", X_att.shape)
print("Y:", Y_att.shape)

print("\nDDODO test:")
print("X:", X_odo.shape)
print("Y:", Y_odo.shape)


# ============================================================
# LOAD BEST MODELS
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
# PREDICTION FUNCTION
# ============================================================

def predict(model, X):

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():

        for (batch_X,) in loader:

            batch_X = batch_X.to(DEVICE)

            pred = model(batch_X)

            predictions.append(
                pred.cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating DDATT predictions...")

pred_att = predict(
    ddatt_model,
    X_att
)

print("DDATT prediction shape:", pred_att.shape)


print("\nGenerating DDODO predictions...")

pred_odo = predict(
    ddodo_model,
    X_odo
)

print("DDODO prediction shape:", pred_odo.shape)


# ============================================================
# DDATT
# ============================================================

# Reference:
#
# q = [qx, qy, qz]
#
# Our reference is pure yaw:
#
# qx = 0
# qy = 0
# qz = sin(delta_yaw / 2)

reference_qz = Y_att[:, 2]

# Predicted qz

predicted_qz = pred_att[:, 2]


# Convert qz to yaw change
#
# delta_yaw = 2 * asin(qz)

predicted_qz_clipped = np.clip(
    predicted_qz,
    -1.0,
    1.0
)

predicted_yaw_deg = (
    2.0
    * np.arcsin(predicted_qz_clipped)
    * 180.0
    / np.pi
)


reference_yaw_deg = (
    2.0
    * np.arcsin(
        np.clip(
            reference_qz,
            -1.0,
            1.0
        )
    )
    * 180.0
    / np.pi
)


yaw_error = (
    predicted_yaw_deg
    - reference_yaw_deg
)

yaw_abs_error = np.abs(
    yaw_error
)


# ============================================================
# DDATT METRICS
# ============================================================

att_mae = np.mean(
    yaw_abs_error
)

att_rmse = np.sqrt(
    np.mean(
        yaw_error ** 2
    )
)

att_median = np.median(
    yaw_abs_error
)

att_p90 = np.percentile(
    yaw_abs_error,
    90
)

att_p95 = np.percentile(
    yaw_abs_error,
    95
)

att_p99 = np.percentile(
    yaw_abs_error,
    99
)

att_max = np.max(
    yaw_abs_error
)


# ============================================================
# DDODO METRICS
# ============================================================

reference_velocity = (
    Y_odo[:, 0]
)

predicted_velocity = (
    pred_odo[:, 0]
)

velocity_error = (
    predicted_velocity
    - reference_velocity
)

velocity_abs_error = np.abs(
    velocity_error
)


odo_mae = np.mean(
    velocity_abs_error
)

odo_rmse = np.sqrt(
    np.mean(
        velocity_error ** 2
    )
)

odo_median = np.median(
    velocity_abs_error
)

odo_p90 = np.percentile(
    velocity_abs_error,
    90
)

odo_p95 = np.percentile(
    velocity_abs_error,
    95
)

odo_p99 = np.percentile(
    velocity_abs_error,
    99
)

odo_max = np.max(
    velocity_abs_error
)


# ============================================================
# CORRELATION
# ============================================================

att_corr = np.corrcoef(
    reference_yaw_deg,
    predicted_yaw_deg
)[0, 1]

odo_corr = np.corrcoef(
    reference_velocity,
    predicted_velocity
)[0, 1]


# ============================================================
# PRINT DDATT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DDATT RESULTS")
print("=" * 70)

print(
    f"MAE:       {att_mae:.6f} deg"
)

print(
    f"RMSE:      {att_rmse:.6f} deg"
)

print(
    f"Median:    {att_median:.6f} deg"
)

print(
    f"P90:       {att_p90:.6f} deg"
)

print(
    f"P95:       {att_p95:.6f} deg"
)

print(
    f"P99:       {att_p99:.6f} deg"
)

print(
    f"Maximum:   {att_max:.6f} deg"
)

print(
    f"Correlation: {att_corr:.6f}"
)


# ============================================================
# PRINT DDODO RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DDODO RESULTS")
print("=" * 70)

print(
    f"MAE:       {odo_mae:.6f} km/h"
)

print(
    f"RMSE:      {odo_rmse:.6f} km/h"
)

print(
    f"Median:    {odo_median:.6f} km/h"
)

print(
    f"P90:       {odo_p90:.6f} km/h"
)

print(
    f"P95:       {odo_p95:.6f} km/h"
)

print(
    f"P99:       {odo_p99:.6f} km/h"
)

print(
    f"Maximum:   {odo_max:.6f} km/h"
)

print(
    f"Correlation: {odo_corr:.6f}"
)


# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("FIRST 20 DDATT PREDICTIONS")
print("=" * 70)

print(
    "No. | Reference yaw | Predicted yaw | Abs error"
)

for i in range(
    min(20, len(reference_yaw_deg))
):

    print(
        f"{i+1:3d} | "
        f"{reference_yaw_deg[i]:13.6f} | "
        f"{predicted_yaw_deg[i]:13.6f} | "
        f"{yaw_abs_error[i]:9.6f}"
    )


print("\n")
print("=" * 70)
print("FIRST 20 DDODO PREDICTIONS")
print("=" * 70)

print(
    "No. | Reference km/h | Predicted km/h | Abs error"
)

for i in range(
    min(20, len(reference_velocity))
):

    print(
        f"{i+1:3d} | "
        f"{reference_velocity[i]:15.6f} | "
        f"{predicted_velocity[i]:15.6f} | "
        f"{velocity_abs_error[i]:9.6f}"
    )


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

results = np.column_stack([
    reference_yaw_deg,
    predicted_yaw_deg,
    yaw_error,
    yaw_abs_error,
    reference_velocity,
    predicted_velocity,
    velocity_error,
    velocity_abs_error
])

np.savetxt(
    os.path.join(
        OUT_DIR,
        "test_predictions.csv"
    ),
    results,
    delimiter=",",
    header=(
        "reference_yaw_deg,"
        "predicted_yaw_deg,"
        "yaw_error_deg,"
        "yaw_abs_error_deg,"
        "reference_velocity_kmh,"
        "predicted_velocity_kmh,"
        "velocity_error_kmh,"
        "velocity_abs_error_kmh"
    ),
    comments=""
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    os.path.join(
        OUT_DIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write("STEP 48 AVNET TEST RESULTS\n\n")

    f.write("DDATT\n")
    f.write(f"MAE = {att_mae}\n")
    f.write(f"RMSE = {att_rmse}\n")
    f.write(f"Median = {att_median}\n")
    f.write(f"P90 = {att_p90}\n")
    f.write(f"P95 = {att_p95}\n")
    f.write(f"P99 = {att_p99}\n")
    f.write(f"Maximum = {att_max}\n")
    f.write(f"Correlation = {att_corr}\n\n")

    f.write("DDODO\n")
    f.write(f"MAE = {odo_mae}\n")
    f.write(f"RMSE = {odo_rmse}\n")
    f.write(f"Median = {odo_median}\n")
    f.write(f"P90 = {odo_p90}\n")
    f.write(f"P95 = {odo_p95}\n")
    f.write(f"P99 = {odo_p99}\n")
    f.write(f"Maximum = {odo_max}\n")
    f.write(f"Correlation = {odo_corr}\n")


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    os.path.join(
        OUT_DIR,
        "test_predictions.csv"
    )
)

print(
    os.path.join(
        OUT_DIR,
        "metrics.txt"
    )
)

print("\nSTEP 48 COMPLETE")
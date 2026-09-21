import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


print("=" * 75)
print("STEP 46 - TRAINING CONFIGURATION AND ONE-BATCH TEST")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 1024
LEARNING_RATE = 0.0001

print("\nTraining configuration")
print("-" * 75)

print("Optimizer     : Adam")
print("Learning rate : 0.0001")
print("Batch size    : 1024")
print("Loss          : MSE")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"D:\Maverick\ML2"
)

DDATT_DIR = ROOT / "step44_normalized_data"
DDODO_DIR = ROOT / "step45_ddodo_data"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)


ddatt_train = np.load(
    DDATT_DIR / "train.npz"
)

ddodo_train = np.load(
    DDODO_DIR / "train.npz"
)


X_att = ddatt_train["X"]
Y_att = ddatt_train["Y"]

X_odo = ddodo_train["X"]
Y_odo = ddodo_train["Y"]


print(
    "DDATT X:",
    X_att.shape
)

print(
    "DDATT Y:",
    Y_att.shape
)

print(
    "DDODO X:",
    X_odo.shape
)

print(
    "DDODO Y:",
    Y_odo.shape
)


# ============================================================
# VERIFY X ALIGNMENT
# ============================================================

print("\n" + "=" * 75)
print("DDATT / DDODO INPUT ALIGNMENT")
print("=" * 75)


if X_att.shape != X_odo.shape:

    raise RuntimeError(
        "DDATT and DDODO X shapes do not match."
    )


max_difference = np.max(
    np.abs(X_att - X_odo)
)

print(
    "Maximum difference between DDATT X and DDODO X:",
    max_difference
)

if max_difference != 0:

    raise RuntimeError(
        "DDATT and DDODO inputs are not identical."
    )

print("Input alignment: PASS")


# ============================================================
# MODEL
# ============================================================

class AVNet(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        self.conv1 = nn.Conv1d(
            6,
            128,
            kernel_size=3
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            128,
            256,
            kernel_size=2
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(
            kernel_size=2
        )

        self.fc1 = nn.Linear(
            256,
            1024
        )

        self.fc2 = nn.Linear(
            1024,
            512
        )

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(
            512,
            output_size
        )


    def forward(self, x):

        # batch × 10 × 6
        x = x.transpose(1, 2)

        # CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # FC
        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        # Sequence length = 1
        x = x.unsqueeze(1)

        # GRU
        x, _ = self.gru(x)

        x = x[:, -1, :]

        # Output
        x = self.output(x)

        return x


# ============================================================
# CREATE MODELS
# ============================================================

print("\n" + "=" * 75)
print("CREATING MODELS")
print("=" * 75)


ddatt_model = AVNet(
    output_size=3
)

ddodo_model = AVNet(
    output_size=1
)


print("DDATT model: created")
print("DDODO model: created")


# ============================================================
# OPTIMIZERS
# ============================================================

ddatt_optimizer = torch.optim.Adam(
    ddatt_model.parameters(),
    lr=LEARNING_RATE
)

ddodo_optimizer = torch.optim.Adam(
    ddodo_model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# DATA LOADERS
# ============================================================

X_att_tensor = torch.from_numpy(
    X_att
)

Y_att_tensor = torch.from_numpy(
    Y_att
)

X_odo_tensor = torch.from_numpy(
    X_odo
)

Y_odo_tensor = torch.from_numpy(
    Y_odo
)


att_dataset = TensorDataset(
    X_att_tensor,
    Y_att_tensor
)

odo_dataset = TensorDataset(
    X_odo_tensor,
    Y_odo_tensor
)


att_loader = DataLoader(
    att_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

odo_loader = DataLoader(
    odo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# ONE DDATT BATCH
# ============================================================

print("\n" + "=" * 75)
print("DDATT ONE-BATCH TEST")
print("=" * 75)


ddatt_model.train()

X_batch, Y_batch = next(
    iter(att_loader)
)


print(
    "Input batch:",
    X_batch.shape
)

print(
    "Target batch:",
    Y_batch.shape
)


ddatt_optimizer.zero_grad()

prediction = ddatt_model(
    X_batch
)

loss = criterion(
    prediction,
    Y_batch
)

print(
    "Prediction:",
    prediction.shape
)

print(
    "Initial MSE:",
    loss.item()
)


loss.backward()

ddatt_optimizer.step()

print(
    "Backward pass: PASS"
)

print(
    "Optimizer step: PASS"
)


# ============================================================
# ONE DDODO BATCH
# ============================================================

print("\n" + "=" * 75)
print("DDODO ONE-BATCH TEST")
print("=" * 75)


ddodo_model.train()

X_batch, Y_batch = next(
    iter(odo_loader)
)


print(
    "Input batch:",
    X_batch.shape
)

print(
    "Target batch:",
    Y_batch.shape
)


ddodo_optimizer.zero_grad()

prediction = ddodo_model(
    X_batch
)

loss = criterion(
    prediction,
    Y_batch
)

print(
    "Prediction:",
    prediction.shape
)

print(
    "Initial MSE:",
    loss.item()
)


loss.backward()

ddodo_optimizer.step()

print(
    "Backward pass: PASS"
)

print(
    "Optimizer step: PASS"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 46 COMPLETE")
print("=" * 75)

print("\nBoth models successfully completed:")
print("  Forward pass")
print("  MSE calculation")
print("  Backward pass")
print("  Adam optimizer step")

print("\nNO full training was performed.")
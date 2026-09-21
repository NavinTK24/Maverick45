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
OUT_DIR = os.path.join(BASE, "step47_training")

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
EPOCHS = 100

DEVICE = torch.device("cpu")

print("=" * 70)
print("STEP 47 - FULL AVNET TRAINING")
print("=" * 70)

print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)
print("Loss: MSE")
print("Optimizer: Adam")

# ============================================================
# LOAD DATA
# ============================================================

def load_npz(path):
    data = np.load(path)
    return data["X"].astype(np.float32), data["Y"].astype(np.float32)


# DDATT data
X_train_att, Y_train_att = load_npz(
    os.path.join(DATA_DIR, "train.npz")
)

X_val_att, Y_val_att = load_npz(
    os.path.join(DATA_DIR, "validation.npz")
)

X_test_att, Y_test_att = load_npz(
    os.path.join(DATA_DIR, "test.npz")
)


# DDODO data
X_train_odo, Y_train_odo = load_npz(
    os.path.join(DDODO_DIR, "train.npz")
)

X_val_odo, Y_val_odo = load_npz(
    os.path.join(DDODO_DIR, "validation.npz")
)

X_test_odo, Y_test_odo = load_npz(
    os.path.join(DDODO_DIR, "test.npz")
)


print("\nDDATT:")
print("Train:", X_train_att.shape, Y_train_att.shape)
print("Validation:", X_val_att.shape, Y_val_att.shape)
print("Test:", X_test_att.shape, Y_test_att.shape)

print("\nDDODO:")
print("Train:", X_train_odo.shape, Y_train_odo.shape)
print("Validation:", X_val_odo.shape, Y_val_odo.shape)
print("Test:", X_test_odo.shape, Y_test_odo.shape)


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_att_dataset = TensorDataset(
    torch.from_numpy(X_train_att),
    torch.from_numpy(Y_train_att)
)

val_att_dataset = TensorDataset(
    torch.from_numpy(X_val_att),
    torch.from_numpy(Y_val_att)
)

test_att_dataset = TensorDataset(
    torch.from_numpy(X_test_att),
    torch.from_numpy(Y_test_att)
)


train_odo_dataset = TensorDataset(
    torch.from_numpy(X_train_odo),
    torch.from_numpy(Y_train_odo)
)

val_odo_dataset = TensorDataset(
    torch.from_numpy(X_val_odo),
    torch.from_numpy(Y_val_odo)
)

test_odo_dataset = TensorDataset(
    torch.from_numpy(X_test_odo),
    torch.from_numpy(Y_test_odo)
)


train_att_loader = DataLoader(
    train_att_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_att_loader = DataLoader(
    val_att_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_att_loader = DataLoader(
    test_att_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


train_odo_loader = DataLoader(
    train_odo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_odo_loader = DataLoader(
    val_odo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_odo_loader = DataLoader(
    test_odo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# AVNET MODEL
# ============================================================

class AVNet(nn.Module):

    def __init__(self, output_size):

        super().__init__()

        # Input:
        # batch x 10 x 6
        #
        # Conv1d expects:
        # batch x channels x sequence
        #
        # Therefore input becomes:
        # batch x 6 x 10

        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=128,
            kernel_size=3
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=2
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(
            kernel_size=2
        )

        # After convolution/pooling:
        #
        # 10
        # -> 8
        # -> 4
        # -> 3
        # -> 1
        #
        # channels = 256
        #
        # flattened size = 256

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
            batch_first=True
        )

        self.output = nn.Linear(
            512,
            output_size
        )

    def forward(self, x):

        # x:
        # batch x 10 x 6

        x = x.transpose(1, 2)

        # batch x 6 x 10

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # batch x 256 x 1

        x = x.flatten(start_dim=1)

        # batch x 256

        x = self.fc1(x)
        x = torch.relu(x)

        # batch x 1024

        x = self.fc2(x)
        x = torch.relu(x)

        # batch x 512

        # GRU expects:
        # batch x sequence x features
        #
        # Our current least-assumption adaptation:
        # sequence length = 1

        x = x.unsqueeze(1)

        # batch x 1 x 512

        x, _ = self.gru(x)

        # Take final sequence output

        x = x[:, -1, :]

        # batch x 512

        x = self.output(x)

        return x


# ============================================================
# CREATE TWO INDEPENDENT MODELS
# ============================================================

ddatt_model = AVNet(
    output_size=3
).to(DEVICE)

ddodo_model = AVNet(
    output_size=1
).to(DEVICE)


print("\nModel parameter counts:")

print(
    "DDATT:",
    sum(p.numel() for p in ddatt_model.parameters())
)

print(
    "DDODO:",
    sum(p.numel() for p in ddodo_model.parameters())
)


# ============================================================
# LOSS + OPTIMIZERS
# ============================================================

criterion_att = nn.MSELoss()

criterion_odo = nn.MSELoss()

optimizer_att = torch.optim.Adam(
    ddatt_model.parameters(),
    lr=LEARNING_RATE
)

optimizer_odo = torch.optim.Adam(
    ddodo_model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, Y in loader:

        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(X)

        loss = criterion(
            prediction,
            Y
        )

        loss.backward()

        optimizer.step()

        batch_size = X.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X, Y in loader:

            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            prediction = model(X)

            loss = criterion(
                prediction,
                Y
            )

            batch_size = X.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_att_val_loss = float("inf")
best_odo_val_loss = float("inf")

best_att_epoch = 0
best_odo_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


history = []


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # DDATT
    # --------------------------------------------------------

    train_att_loss = train_one_epoch(
        ddatt_model,
        train_att_loader,
        criterion_att,
        optimizer_att
    )

    val_att_loss = evaluate(
        ddatt_model,
        val_att_loader,
        criterion_att
    )


    # --------------------------------------------------------
    # DDODO
    # --------------------------------------------------------

    train_odo_loss = train_one_epoch(
        ddodo_model,
        train_odo_loader,
        criterion_odo,
        optimizer_odo
    )

    val_odo_loss = evaluate(
        ddodo_model,
        val_odo_loader,
        criterion_odo
    )


    # --------------------------------------------------------
    # SAVE BEST DDATT
    # --------------------------------------------------------

    if val_att_loss < best_att_val_loss:

        best_att_val_loss = val_att_loss
        best_att_epoch = epoch

        torch.save(
            ddatt_model.state_dict(),
            os.path.join(
                OUT_DIR,
                "best_ddatt_model.pt"
            )
        )


    # --------------------------------------------------------
    # SAVE BEST DDODO
    # --------------------------------------------------------

    if val_odo_loss < best_odo_val_loss:

        best_odo_val_loss = val_odo_loss
        best_odo_epoch = epoch

        torch.save(
            ddodo_model.state_dict(),
            os.path.join(
                OUT_DIR,
                "best_ddodo_model.pt"
            )
        )


    history.append([
        epoch,
        train_att_loss,
        val_att_loss,
        train_odo_loss,
        val_odo_loss
    ])


    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"DDATT "
        f"Train={train_att_loss:.8f} "
        f"Val={val_att_loss:.8f} | "
        f"DDODO "
        f"Train={train_odo_loss:.8f} "
        f"Val={val_odo_loss:.8f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history = np.array(
    history,
    dtype=np.float64
)

np.savetxt(
    os.path.join(
        OUT_DIR,
        "training_history.csv"
    ),
    history,
    delimiter=",",
    header="epoch,train_att_loss,val_att_loss,train_odo_loss,val_odo_loss",
    comments=""
)


# ============================================================
# LOAD BEST MODELS
# ============================================================

ddatt_model.load_state_dict(
    torch.load(
        os.path.join(
            OUT_DIR,
            "best_ddatt_model.pt"
        ),
        map_location=DEVICE
    )
)

ddodo_model.load_state_dict(
    torch.load(
        os.path.join(
            OUT_DIR,
            "best_ddodo_model.pt"
        ),
        map_location=DEVICE
    )
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_att_loss = evaluate(
    ddatt_model,
    test_att_loader,
    criterion_att
)

test_odo_loss = evaluate(
    ddodo_model,
    test_odo_loader,
    criterion_odo
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nDDATT:")
print("Best validation epoch:", best_att_epoch)
print("Best validation MSE:", best_att_val_loss)
print("Test MSE:", test_att_loss)

print("\nDDODO:")
print("Best validation epoch:", best_odo_epoch)
print("Best validation MSE:", best_odo_val_loss)
print("Test MSE:", test_odo_loss)


# ============================================================
# SAVE FINAL BEST MODELS
# ============================================================

torch.save(
    ddatt_model.state_dict(),
    os.path.join(
        OUT_DIR,
        "ddatt_final_best.pt"
    )
)

torch.save(
    ddodo_model.state_dict(),
    os.path.join(
        OUT_DIR,
        "ddodo_final_best.pt"
    )
)


print("\nSaved to:")
print(OUT_DIR)

print("\nFiles:")
print("best_ddatt_model.pt")
print("best_ddodo_model.pt")
print("ddatt_final_best.pt")
print("ddodo_final_best.pt")
print("training_history.csv")

print("\nSTEP 47 COMPLETE")
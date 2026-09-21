import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"D:\Maverick\ML2"

SEQUENCE_DIR = os.path.join(
    ROOT,
    "step67_recurrent_sequences",
    "runs"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "step68B_recurrent_training"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


LEARNING_RATE = 0.0001
BATCH_SIZE = 1
EPOCHS = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 100)
print("STEP 68B - TRAIN RECURRENT AVNET")
print("=" * 100)

print(
    "\nDevice:",
    device
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)


# ============================================================
# MODEL
# ============================================================

class RecurrentAVNet(nn.Module):

    def __init__(
        self,
        output_size
    ):

        super().__init__()


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=128,
            kernel_size=3
        )

        self.pool1 = nn.MaxPool1d(
            kernel_size=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=2
        )

        self.pool2 = nn.MaxPool1d(
            kernel_size=2
        )


        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            256,
            1024
        )

        self.fc2 = nn.Linear(
            1024,
            512
        )


        # ----------------------------------------------------
        # GRU CELL
        # ----------------------------------------------------

        self.gru_cell = nn.GRUCell(
            input_size=512,
            hidden_size=512
        )


        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.output = nn.Linear(
            512,
            output_size
        )


    def extract_feature(
        self,
        window
    ):

        # ----------------------------------------------------
        # Input window:
        #
        # (10, 6)
        #
        # 10 samples
        # 6 IMU channels
        # ----------------------------------------------------

        x = window.permute(
            1,
            0
        )

        # (6,10)

        x = x.unsqueeze(
            0
        )

        # (1,6,10)


        x = self.conv1(
            x
        )

        x = torch.relu(
            x
        )

        x = self.pool1(
            x
        )


        x = self.conv2(
            x
        )

        x = torch.relu(
            x
        )

        x = self.pool2(
            x
        )


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )


        # ----------------------------------------------------
        # FC
        # ----------------------------------------------------

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )


        # (1,512) -> (512)

        return x.squeeze(
            0
        )


    def forward(
        self,
        sequence
    ):

        # sequence:
        #
        # (N,10,6)
        #
        # N = number of consecutive 1-second windows


        sequence_length = sequence.shape[0]


        # ----------------------------------------------------
        # Original QDeepOdo-style hidden state
        # ----------------------------------------------------

        hx = torch.randn(
            512,
            device=sequence.device
        )


        outputs = []


        # ----------------------------------------------------
        # Recurrent processing
        # ----------------------------------------------------

        for t in range(
            sequence_length
        ):

            feature = self.extract_feature(
                sequence[t]
            )


            hx = self.gru_cell(
                feature,
                hx
            )


            output = self.output(
                hx
            )


            outputs.append(
                output
            )


        return torch.stack(
            outputs,
            dim=0
        )


# ============================================================
# LOAD RUN FILES
# ============================================================

print(
    "\nLoading recurrent sequences..."
)


all_files = [
    f
    for f in os.listdir(
        SEQUENCE_DIR
    )
    if f.endswith(".npz")
]


if len(all_files) == 0:

    raise FileNotFoundError(
        "No Step 67 recurrent sequence files found."
    )


train_files = []
validation_files = []
test_files = []


for filename in all_files:

    lower = filename.lower()


    if "_train.npz" in lower:

        train_files.append(
            filename
        )

    elif "_validation.npz" in lower:

        validation_files.append(
            filename
        )

    elif "_test.npz" in lower:

        test_files.append(
            filename
        )


train_files.sort()
validation_files.sort()
test_files.sort()


print(
    "Training runs:",
    len(train_files)
)

print(
    "Validation runs:",
    len(validation_files)
)

print(
    "Test runs:",
    len(test_files)
)


# ============================================================
# FUNCTION TO LOAD A RUN
# ============================================================

def load_run(
    filename
):

    path = os.path.join(
        SEQUENCE_DIR,
        filename
    )


    data = np.load(
        path
    )


    X = data["X"].astype(
        np.float32
    )

    Y_DDATT = data["Y_DDATT"].astype(
        np.float32
    )

    Y_DDODO = data["Y_DDODO"].astype(
        np.float32
    )


    return (
        X,
        Y_DDATT,
        Y_DDODO
    )


# ============================================================
# CREATE MODELS
# ============================================================

print(
    "\nCreating DDATT model..."
)

ddatt_model = RecurrentAVNet(
    output_size=3
).to(device)


print(
    "Creating DDODO model..."
)

ddodo_model = RecurrentAVNet(
    output_size=1
).to(device)


print(
    "\nDDATT parameters:",
    sum(
        p.numel()
        for p in ddatt_model.parameters()
    )
)

print(
    "DDODO parameters:",
    sum(
        p.numel()
        for p in ddodo_model.parameters()
    )
)


# ============================================================
# OPTIMIZERS
# ============================================================

ddatt_optimizer = Adam(
    ddatt_model.parameters(),
    lr=LEARNING_RATE
)

ddodo_optimizer = Adam(
    ddodo_model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# SEQUENCE TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    files,
    target_name
):

    model.train()


    total_squared_error = 0.0
    total_elements = 0


    for run_number, filename in enumerate(
        files,
        start=1
    ):


        X_np, Y_att_np, Y_odo_np = load_run(
            filename
        )


        # ----------------------------------------------------
        # Select target
        # ----------------------------------------------------

        if target_name == "DDATT":

            Y_np = Y_att_np

        else:

            Y_np = Y_odo_np.reshape(
                -1,
                1
            )


        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        X = torch.from_numpy(
            X_np
        ).to(device)

        Y = torch.from_numpy(
            Y_np
        ).to(device)


        # ----------------------------------------------------
        # One complete continuous run
        # ----------------------------------------------------

        optimizer.zero_grad()


        prediction = model(
            X
        )


        loss = criterion(
            prediction,
            Y
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # Weighted accumulation
        # ----------------------------------------------------

        num_elements = Y.numel()


        total_squared_error += (
            loss.item()
            *
            num_elements
        )


        total_elements += (
            num_elements
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            run_number % 100 == 0
            or
            run_number == len(files)
        ):

            print(
                f"    {target_name}: "
                f"{run_number}/{len(files)} runs",
                end="\r"
            )


    epoch_loss = (
        total_squared_error
        /
        total_elements
    )


    print(
        ""
    )


    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    files,
    target_name
):

    model.eval()


    total_squared_error = 0.0
    total_elements = 0


    with torch.no_grad():

        for filename in files:


            X_np, Y_att_np, Y_odo_np = load_run(
                filename
            )


            if target_name == "DDATT":

                Y_np = Y_att_np

            else:

                Y_np = Y_odo_np.reshape(
                    -1,
                    1
                )


            X = torch.from_numpy(
                X_np
            ).to(device)

            Y = torch.from_numpy(
                Y_np
            ).to(device)


            prediction = model(
                X
            )


            loss = criterion(
                prediction,
                Y
            )


            num_elements = Y.numel()


            total_squared_error += (
                loss.item()
                *
                num_elements
            )


            total_elements += (
                num_elements
            )


    return (
        total_squared_error
        /
        total_elements
    )


# ============================================================
# TRAIN DDATT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TRAINING DDATT"
)

print(
    "=" * 100
)


ddatt_history = []

best_ddatt_val = float(
    "inf"
)

best_ddatt_epoch = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )


    train_loss = train_one_epoch(
        ddatt_model,
        ddatt_optimizer,
        train_files,
        "DDATT"
    )


    val_loss = evaluate(
        ddatt_model,
        validation_files,
        "DDATT"
    )


    print(
        f"DDATT train MSE: "
        f"{train_loss:.10f}"
    )

    print(
        f"DDATT validation MSE: "
        f"{val_loss:.10f}"
    )


    ddatt_history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss
        }
    )


    if val_loss < best_ddatt_val:

        best_ddatt_val = val_loss

        best_ddatt_epoch = epoch


        torch.save(
            ddatt_model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_ddatt_recurrent_model.pt"
            )
        )


        print(
            "  *** New best DDATT model saved ***"
        )


# ============================================================
# TRAIN DDODO
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TRAINING DDODO"
)

print(
    "=" * 100
)


ddodo_history = []

best_ddodo_val = float(
    "inf"
)

best_ddodo_epoch = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )


    train_loss = train_one_epoch(
        ddodo_model,
        ddodo_optimizer,
        train_files,
        "DDODO"
    )


    val_loss = evaluate(
        ddodo_model,
        validation_files,
        "DDODO"
    )


    print(
        f"DDODO train MSE: "
        f"{train_loss:.10f}"
    )

    print(
        f"DDODO validation MSE: "
        f"{val_loss:.10f}"
    )


    ddodo_history.append(
        {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss
        }
    )


    if val_loss < best_ddodo_val:

        best_ddodo_val = val_loss

        best_ddodo_epoch = epoch


        torch.save(
            ddodo_model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_ddodo_recurrent_model.pt"
            )
        )


        print(
            "  *** New best DDODO model saved ***"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

ddatt_history_df = pd.DataFrame(
    ddatt_history
)

ddodo_history_df = pd.DataFrame(
    ddodo_history
)


ddatt_history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ddatt_training_history.csv"
    ),
    index=False
)


ddodo_history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ddodo_training_history.csv"
    ),
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "training_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 68B - RECURRENT AVNET TRAINING\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        f"Training runs: {len(train_files)}\n"
    )

    f.write(
        f"Validation runs: {len(validation_files)}\n"
    )

    f.write(
        f"Test runs: {len(test_files)}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        f"Best DDATT validation MSE: "
        f"{best_ddatt_val}\n"
    )

    f.write(
        f"Best DDATT epoch: "
        f"{best_ddatt_epoch}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        f"Best DDODO validation MSE: "
        f"{best_ddodo_val}\n"
    )

    f.write(
        f"Best DDODO epoch: "
        f"{best_ddodo_epoch}\n"
    )


# ============================================================
# FINAL     
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STEP 68B TRAINING COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nBest DDATT epoch:",
    best_ddatt_epoch
)

print(
    "Best DDATT validation MSE:",
    best_ddatt_val
)

print(
    "\nBest DDODO epoch:",
    best_ddodo_epoch
)

print(
    "Best DDODO validation MSE:",
    best_ddodo_val
)

print(
    "\nModels saved in:"
)

print(
    OUTPUT_DIR
)
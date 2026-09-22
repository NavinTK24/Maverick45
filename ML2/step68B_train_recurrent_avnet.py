import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\STUDENT\Desktop\Maverick\Maverick45\ML2"

SEQUENCE_DIR = os.path.join(
    ROOT,
    "step67_recurrent_sequences",
    "runs"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "step68B_ddodo_recurrent_training"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

LEARNING_RATE = 0.0001
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
print("STEP 68B - RECURRENT DDODO ONLY")
print("=" * 100)

print(
    "\nDevice:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Epochs:",
    EPOCHS
)


# ============================================================
# TIME FORMATTER
# ============================================================

def format_time(seconds):

    seconds = int(
        max(
            0,
            seconds
        )
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# MODEL
# ============================================================

class RecurrentDDODO(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.fc1 = nn.Linear(
            256,
            1024
        )

        self.fc2 = nn.Linear(
            1024,
            512
        )

        self.gru_cell = nn.GRUCell(
            input_size=512,
            hidden_size=512
        )

        self.output = nn.Linear(
            512,
            1
        )


    def extract_feature(
        self,
        window
    ):

        x = window.permute(
            1,
            0
        )

        x = x.unsqueeze(
            0
        )

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

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = torch.relu(
            self.fc2(x)
        )

        return x.squeeze(
            0
        )


    def forward(
        self,
        sequence
    ):

        sequence_length = sequence.shape[0]

        # QDeepOdo-style hidden state
        hx = torch.randn(
            512,
            device=sequence.device
        )

        outputs = []

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

            velocity = self.output(
                hx
            )

            outputs.append(
                velocity
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

train_files = sorted(
    [
        f
        for f in all_files
        if "_train.npz" in f.lower()
    ]
)

validation_files = sorted(
    [
        f
        for f in all_files
        if "_validation.npz" in f.lower()
    ]
)

test_files = sorted(
    [
        f
        for f in all_files
        if "_test.npz" in f.lower()
    ]
)


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


if len(train_files) != 1030:
    raise ValueError(
        f"Expected 1030 training runs, "
        f"found {len(train_files)}"
    )

if len(validation_files) != 172:
    raise ValueError(
        f"Expected 172 validation runs, "
        f"found {len(validation_files)}"
    )

if len(test_files) != 193:
    raise ValueError(
        f"Expected 193 test runs, "
        f"found {len(test_files)}"
    )


# ============================================================
# LOAD RUN
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

    X = data[
        "X"
    ].astype(
        np.float32
    )

    Y = data[
        "Y_DDODO"
    ].astype(
        np.float32
    ).reshape(
        -1,
        1
    )

    return X, Y


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "\nCreating DDODO model..."
)

model = RecurrentDDODO().to(
    device
)

print(
    "DDODO parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    files
):

    model.train()

    total_squared_error = 0.0
    total_elements = 0

    epoch_start = time.perf_counter()


    for run_number, filename in enumerate(
        files,
        start=1
    ):

        X_np, Y_np = load_run(
            filename
        )

        X = torch.from_numpy(
            X_np
        ).to(
            device
        )

        Y = torch.from_numpy(
            Y_np
        ).to(
            device
        )

        optimizer.zero_grad()

        prediction = model(
            X
        )

        loss = criterion(
            prediction,
            Y
        )

        loss.backward()

        optimizer.step()


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
        # Live progress
        # ----------------------------------------------------

        if (
            run_number % 25 == 0
            or
            run_number == len(files)
        ):

            elapsed = (
                time.perf_counter()
                -
                epoch_start
            )

            runs_per_second = (
                run_number
                /
                elapsed
            )

            remaining_runs = (
                len(files)
                -
                run_number
            )

            eta = (
                remaining_runs
                /
                runs_per_second
                if runs_per_second > 0
                else 0
            )


            print(
                f"    DDODO: "
                f"{run_number:4d}/{len(files)} "
                f"| Elapsed: {format_time(elapsed)} "
                f"| ETA: {format_time(eta)}",
                end="\r",
                flush=True
            )


    epoch_time = (
        time.perf_counter()
        -
        epoch_start
    )

    print(
        ""
    )

    epoch_loss = (
        total_squared_error
        /
        total_elements
    )

    return (
        epoch_loss,
        epoch_time
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    files
):

    model.eval()

    total_squared_error = 0.0
    total_elements = 0

    validation_start = time.perf_counter()


    with torch.no_grad():

        for filename in files:

            X_np, Y_np = load_run(
                filename
            )

            X = torch.from_numpy(
                X_np
            ).to(
                device
            )

            Y = torch.from_numpy(
                Y_np
            ).to(
                device
            )

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


    validation_time = (
        time.perf_counter()
        -
        validation_start
    )

    validation_loss = (
        total_squared_error
        /
        total_elements
    )

    return (
        validation_loss,
        validation_time
    )


# ============================================================
# TRAINING
# ============================================================

history = []

best_val = float(
    "inf"
)

best_epoch = 0

overall_start = time.perf_counter()


print(
    "\n"
    + "=" * 100
)

print(
    "TRAINING RECURRENT DDODO"
)

print(
    "=" * 100
)

print(
    "Overall training timer started."
)


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        "-" * 80
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_mse, epoch_train_time = train_one_epoch(
        model,
        optimizer,
        train_files
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_mse, validation_time = evaluate(
        model,
        validation_files
    )


    train_rmse = np.sqrt(
        train_mse
    )

    validation_rmse = np.sqrt(
        validation_mse
    )


    # --------------------------------------------------------
    # Epoch timing
    # --------------------------------------------------------

    epoch_total_time = (
        epoch_train_time
        +
        validation_time
    )


    overall_elapsed = (
        time.perf_counter()
        -
        overall_start
    )


    completed_epochs = epoch

    average_epoch_time = (
        overall_elapsed
        /
        completed_epochs
    )


    remaining_epochs = (
        EPOCHS
        -
        completed_epochs
    )


    overall_eta = (
        remaining_epochs
        *
        average_epoch_time
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nDDODO train MSE: "
        f"{train_mse:.10f}"
    )

    print(
        f"DDODO train RMSE: "
        f"{train_rmse:.6f} km/h"
    )

    print(
        f"DDODO validation MSE: "
        f"{validation_mse:.10f}"
    )

    print(
        f"DDODO validation RMSE: "
        f"{validation_rmse:.6f} km/h"
    )


    print(
        "\nEpoch training time:",
        format_time(
            epoch_train_time
        )
    )

    print(
        "Epoch validation time:",
        format_time(
            validation_time
        )
    )

    print(
        "Total epoch time:",
        format_time(
            epoch_total_time
        )
    )

    print(
        "Overall elapsed:",
        format_time(
            overall_elapsed
        )
    )

    print(
        "Overall ETA:",
        format_time(
            overall_eta
        )
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "validation_mse": validation_mse,
            "validation_rmse": validation_rmse,
            "epoch_train_seconds": epoch_train_time,
            "epoch_validation_seconds": validation_time,
            "epoch_total_seconds": epoch_total_time,
            "overall_elapsed_seconds": overall_elapsed,
            "overall_eta_seconds": overall_eta
        }
    )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if validation_mse < best_val:

        best_val = validation_mse

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_ddodo_recurrent_model.pt"
            )
        )

        print(
            "\n*** New best DDODO model saved ***"
        )


# ============================================================
# FINAL TIME
# ============================================================

overall_total_time = (
    time.perf_counter()
    -
    overall_start
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_file = os.path.join(
    OUTPUT_DIR,
    "ddodo_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "training_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 68B - RECURRENT DDODO ONLY\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    if torch.cuda.is_available():

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

        f.write(
            f"CUDA: "
            f"{torch.version.cuda}\n"
        )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
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
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation MSE: {best_val}\n"
    )

    f.write(
        f"Best validation RMSE: "
        f"{np.sqrt(best_val)} km/h\n"
    )

    f.write(
        f"\nTotal training time: "
        f"{format_time(overall_total_time)}\n"
    )

    f.write(
        f"Total training seconds: "
        f"{overall_total_time:.3f}\n"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STEP 68B DDODO TRAINING COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation MSE:",
    best_val
)

print(
    "Best validation RMSE:",
    np.sqrt(best_val),
    "km/h"
)

print(
    "\nTOTAL TRAINING TIME:",
    format_time(
        overall_total_time
    )
)

print(
    "\nModel saved:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "best_ddodo_recurrent_model.pt"
    )
)

print(
    "\nTraining history:"
)

print(
    history_file
)
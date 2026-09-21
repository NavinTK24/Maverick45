from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 20
# AVNet training
#
# Two independent networks:
#
#   1. AVNet_Attitude
#      input  : 10 x 6
#      output : qx, qy, qz
#
#   2. AVNet_Velocity
#      input  : 10 x 6
#      output : velocity
#
# Architecture adapted from Qian et al. AVNet:
#
#   10 x 6
#      |
#   Conv1D  k=3, 6 -> 128
#      |
#     ReLU
#      |
#   MaxPool k=2
#      |
#   Conv1D  k=2, 128 -> 256
#      |
#     ReLU
#      |
#   Flatten
#      |
#   FC 768 -> 1024
#      |
#   FC 1024 -> 512
#      |
#   GRU 512 -> 512
#      |
#   Output
#
# GRU sequence length = 3 consecutive 1-second windows.
#
# IMPORTANT:
# The kernel adaptation and sequence length are adaptations
# for our 10-Hz dataset. They are not claimed as the paper's
# original numerical settings.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ML2 = Path(r"D:\Maverick\ML2")

DATASET_FILE = ML2 / "avnet_yaw_only_dataset.npz"
SPLIT_FILE = ML2 / "avnet_sample_split.csv"

OUTPUT_DIR = ML2 / "avnet_models"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

SEED = 42

BATCH_SIZE = 1024
LEARNING_RATE = 0.0001

EPOCHS = 100

SEQ_LEN = 3

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("AVNet TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

print("\nNPZ keys:")
print(data.files)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)

dataset_names = data["dataset_names"].astype(str)
window_indices = data["window_indices"].astype(np.int64)


print("\nX shape:", X.shape)
print("Y shape:", Y.shape)

print("\nExpected:")
print("X = (107043, 10, 6)")
print("Y = (107043, 4)")


# ============================================================
# LOAD SPLIT INFORMATION
# ============================================================

split_df = pd.read_csv(
    SPLIT_FILE
)

print("\nSplit file:")
print(SPLIT_FILE)

print("\nSplit summary:")

print(
    split_df.groupby("split")
    .size()
    .rename("samples")
)


# ============================================================
# CREATE SAMPLE SPLIT LOOKUP
# ============================================================

split_lookup = dict(
    zip(
        split_df["dataset"].astype(str),
        split_df["split"].astype(str)
    )
)

sample_split = np.array([
    split_lookup[name]
    for name in dataset_names
])


# ============================================================
# VERIFY DATASET SEPARATION
# ============================================================

print("\nChecking dataset separation...")

for dataset in sorted(set(dataset_names)):

    splits = set(
        sample_split[
            dataset_names == dataset
        ]
    )

    if len(splits) != 1:

        raise RuntimeError(
            f"Dataset {dataset} appears in multiple splits: {splits}"
        )

print("PASS: no dataset crosses train/validation/test.")


# ============================================================
# SORT SAMPLES CHRONOLOGICALLY
# ============================================================

# The combined dataset contains window_index.
# We explicitly sort each dataset by window_index.
#
# This is important because the GRU needs consecutive
# temporal windows.
# ============================================================

order = np.lexsort(
    (
        window_indices,
        dataset_names
    )
)

X = X[order]
Y = Y[order]
dataset_names = dataset_names[order]
window_indices = window_indices[order]
sample_split = sample_split[order]


# ============================================================
# BUILD SEQUENCES
# ============================================================

def build_sequences(
    X,
    Y,
    dataset_names,
    window_indices,
    sample_split,
    split_name,
    sequence_length
):
    """
    Build temporal sequences without crossing:

        - dataset boundaries
        - train/validation/test boundaries

    Each sequence contains consecutive 1-second windows.

    Example for SEQ_LEN = 3:

        window 0
        window 1
        window 2
             |
             v
          GRU sequence

    Targets are retained for all 3 time steps.
    """

    sequences_X = []
    sequences_Y = []

    sequence_dataset = []
    sequence_start_window = []

    unique_datasets = sorted(
        set(
            dataset_names[
                sample_split == split_name
            ]
        )
    )

    for dataset in unique_datasets:

        mask = (
            (dataset_names == dataset)
            &
            (sample_split == split_name)
        )

        indices = np.where(mask)[0]

        # Sort chronologically
        indices = indices[
            np.argsort(
                window_indices[indices]
            )
        ]

        n = len(indices)

        # Non-overlapping sequences.
        #
        # This prevents the same windows from being
        # repeatedly reused in many training sequences.
        #
        # Example:
        #
        # 0 1 2 | 3 4 5 | 6 7 8
        #
        usable = (
            n // sequence_length
        ) * sequence_length

        for start in range(
            0,
            usable,
            sequence_length
        ):

            selected = indices[
                start:start + sequence_length
            ]

            # Check chronological continuity.
            expected = np.arange(
                window_indices[selected[0]],
                window_indices[selected[0]]
                + sequence_length
            )

            actual = window_indices[selected]

            if not np.array_equal(
                actual,
                expected
            ):
                continue

            sequences_X.append(
                X[selected]
            )

            sequences_Y.append(
                Y[selected]
            )

            sequence_dataset.append(
                dataset
            )

            sequence_start_window.append(
                int(actual[0])
            )

    if len(sequences_X) == 0:

        raise RuntimeError(
            f"No sequences created for {split_name}"
        )

    sequences_X = np.stack(
        sequences_X
    ).astype(np.float32)

    sequences_Y = np.stack(
        sequences_Y
    ).astype(np.float32)

    return (
        sequences_X,
        sequences_Y,
        np.array(sequence_dataset),
        np.array(sequence_start_window)
    )


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SEQUENCES
# ============================================================

print("\nBuilding temporal sequences...")

(
    X_train,
    Y_train,
    train_dataset_ids,
    train_start_windows
) = build_sequences(
    X,
    Y,
    dataset_names,
    window_indices,
    sample_split,
    "train",
    SEQ_LEN
)

(
    X_val,
    Y_val,
    val_dataset_ids,
    val_start_windows
) = build_sequences(
    X,
    Y,
    dataset_names,
    window_indices,
    sample_split,
    "validation",
    SEQ_LEN
)

(
    X_test,
    Y_test,
    test_dataset_ids,
    test_start_windows
) = build_sequences(
    X,
    Y,
    dataset_names,
    window_indices,
    sample_split,
    "test",
    SEQ_LEN
)


# ============================================================
# PRINT SEQUENCE SHAPES
# ============================================================

print("\nSequence shapes:")

print(
    "Train X:",
    X_train.shape
)

print(
    "Train Y:",
    Y_train.shape
)

print(
    "Validation X:",
    X_val.shape
)

print(
    "Validation Y:",
    Y_val.shape
)

print(
    "Test X:",
    X_test.shape
)

print(
    "Test Y:",
    Y_test.shape
)


# ============================================================
# EXPECTED SHAPE
# ============================================================

print("\nExpected sequence structure:")

print(
    "(number_of_sequences, 3, 10, 6)"
)

print(
    "(number_of_sequences, 3, 4)"
)


# ============================================================
# DATASET CLASS
# ============================================================

class AVNetDataset(Dataset):

    def __init__(
        self,
        X,
        Y
    ):
        self.X = torch.from_numpy(
            X
        )

        self.Y = torch.from_numpy(
            Y
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.Y[index]
        )


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = AVNetDataset(
    X_train,
    Y_train
)

val_dataset = AVNetDataset(
    X_val,
    Y_val
)

test_dataset = AVNetDataset(
    X_test,
    Y_test
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# AVNET MODEL
# ============================================================

class AVNet(nn.Module):

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
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=2,
            stride=1,
            padding=0
        )

        self.relu2 = nn.ReLU()

        # ----------------------------------------------------
        # For input:
        #
        # 10
        # -> Conv3 = 8
        # -> Pool2 = 4
        # -> Conv2 = 3
        #
        # Therefore:
        #
        # 3 × 256 = 768
        # ----------------------------------------------------

        self.flatten_size = 3 * 256

        # ----------------------------------------------------
        # Fully connected layers
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            self.flatten_size,
            1024
        )

        self.fc2 = nn.Linear(
            1024,
            512
        )

        self.relu_fc1 = nn.ReLU()
        self.relu_fc2 = nn.ReLU()

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Output layer
        #
        # Attitude:
        #     output_size = 3
        #
        # Velocity:
        #     output_size = 1
        # ----------------------------------------------------

        self.output = nn.Linear(
            512,
            output_size
        )


    def extract_window_features(
        self,
        x
    ):
        """
        Input:
            x = (batch, 10, 6)

        Output:
            features = (batch, 512)
        """

        # Conv1D expects:
        #
        # (batch, channels, time)

        x = x.transpose(
            1,
            2
        )

        # ----------------------------------------------------
        # CNN 1
        # ----------------------------------------------------

        x = self.conv1(x)

        x = self.relu1(x)

        x = self.pool(x)

        # ----------------------------------------------------
        # CNN 2
        # ----------------------------------------------------

        x = self.conv2(x)

        x = self.relu2(x)

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = x.flatten(
            start_dim=1
        )

        # ----------------------------------------------------
        # FC 1024
        # ----------------------------------------------------

        x = self.fc1(x)

        x = self.relu_fc1(x)

        # ----------------------------------------------------
        # FC 512
        # ----------------------------------------------------

        x = self.fc2(x)

        x = self.relu_fc2(x)

        return x


    def forward(
        self,
        x
    ):
        """
        Input:

            x =
            (batch, sequence_length, 10, 6)

        Example:

            (1024, 3, 10, 6)

        Output:

            (batch, sequence_length, output_size)
        """

        batch_size = x.shape[0]

        sequence_length = x.shape[1]

        # ----------------------------------------------------
        # Treat every 1-second window independently through
        # the CNN + FC feature extractor.
        # ----------------------------------------------------

        x = x.reshape(
            batch_size * sequence_length,
            10,
            6
        )

        features = self.extract_window_features(
            x
        )

        # features:
        #
        # (batch * sequence_length, 512)

        # ----------------------------------------------------
        # Restore temporal sequence
        # ----------------------------------------------------

        features = features.reshape(
            batch_size,
            sequence_length,
            512
        )

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        gru_output, _ = self.gru(
            features
        )

        # ----------------------------------------------------
        # Output at every time step
        # ----------------------------------------------------

        output = self.output(
            gru_output
        )

        return output


# ============================================================
# CREATE TWO INDEPENDENT NETWORKS
# ============================================================

print("\nCreating AVNet models...")

attitude_model = AVNet(
    output_size=3
).to(DEVICE)

velocity_model = AVNet(
    output_size=1
).to(DEVICE)


# ============================================================
# PRINT MODEL
# ============================================================

print("\n============================================================")
print("ATTITUDE AVNET")
print("============================================================")

print(attitude_model)


print("\n============================================================")
print("VELOCITY AVNET")
print("============================================================")

print(velocity_model)


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# OPTIMIZERS
# ============================================================

attitude_optimizer = torch.optim.Adam(
    attitude_model.parameters(),
    lr=LEARNING_RATE
)

velocity_optimizer = torch.optim.Adam(
    velocity_model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader,
    target_type
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    predictions = []
    targets = []

    with torch.no_grad():

        for batch_X, batch_Y in loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_Y = batch_Y.to(
                DEVICE,
                non_blocking=True
            )

            output = model(
                batch_X
            )

            if target_type == "attitude":

                target = batch_Y[:, :, 0:3]

            else:

                target = batch_Y[:, :, 3:4]

            loss = criterion(
                output,
                target
            )

            batch_size = batch_X.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions.append(
                output.cpu().numpy()
            )

            targets.append(
                target.cpu().numpy()
            )

    average_loss = (
        total_loss /
        total_samples
    )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    targets = np.concatenate(
        targets,
        axis=0
    )

    return (
        average_loss,
        predictions,
        targets
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    target_type,
    output_name
):

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {output_name.upper()} AVNET"
    )
    print("=" * 70)

    best_val_loss = float("inf")

    history = []

    best_model_file = (
        OUTPUT_DIR /
        f"best_{output_name}_avnet.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        sample_count = 0

        for batch_X, batch_Y in train_loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_Y = batch_Y.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Target selection
            # ------------------------------------------------

            if target_type == "attitude":

                target = batch_Y[:, :, 0:3]

            else:

                target = batch_Y[:, :, 3:4]

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction = model(
                batch_X
            )

            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            loss = criterion(
                prediction,
                target
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = batch_X.shape[0]

            running_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

        train_loss = (
            running_loss /
            sample_count
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, _, _ = evaluate_model(
            model,
            val_loader,
            target_type
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": val_loss

        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "validation_loss":
                        val_loss,

                    "train_loss":
                        train_loss,

                    "target_type":
                        target_type,

                    "sequence_length":
                        SEQ_LEN,

                    "input_shape":
                        [10, 6],

                    "learning_rate":
                        LEARNING_RATE,

                    "batch_size":
                        BATCH_SIZE,

                },
                best_model_file
            )

            marker = "  <-- BEST"

        else:

            marker = ""

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.8f} | "
            f"Val {val_loss:.8f}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_file = (
        OUTPUT_DIR /
        f"{output_name}_training_history.csv"
    )

    history_df.to_csv(
        history_file,
        index=False
    )

    print(
        "\nBest validation loss:",
        best_val_loss
    )

    print(
        "Best model:",
        best_model_file
    )

    print(
        "History:",
        history_file
    )

    return best_model_file


# ============================================================
# TRAIN ATTITUDE NETWORK
# ============================================================

best_attitude_model = train_model(
    attitude_model,
    attitude_optimizer,
    train_loader,
    val_loader,
    "attitude",
    "attitude"
)


# ============================================================
# TRAIN VELOCITY NETWORK
# ============================================================

best_velocity_model = train_model(
    velocity_model,
    velocity_optimizer,
    train_loader,
    val_loader,
    "velocity",
    "velocity"
)


# ============================================================
# LOAD BEST ATTITUDE MODEL
# ============================================================

print("\nLoading best attitude model...")

checkpoint = torch.load(
    best_attitude_model,
    map_location=DEVICE
)

attitude_model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# LOAD BEST VELOCITY MODEL
# ============================================================

print("Loading best velocity model...")

checkpoint = torch.load(
    best_velocity_model,
    map_location=DEVICE
)

velocity_model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST")
print("=" * 70)


attitude_test_loss, attitude_pred, attitude_true = (
    evaluate_model(
        attitude_model,
        test_loader,
        "attitude"
    )
)


velocity_test_loss, velocity_pred, velocity_true = (
    evaluate_model(
        velocity_model,
        test_loader,
        "velocity"
    )
)


print(
    "\nAttitude test MSE:",
    attitude_test_loss
)

print(
    "Velocity test MSE:",
    velocity_test_loss
)


# ============================================================
# ATTITUDE METRICS
# ============================================================

attitude_error = (
    attitude_pred -
    attitude_true
)

attitude_mae = np.mean(
    np.abs(attitude_error)
)

attitude_rmse = np.sqrt(
    np.mean(
        attitude_error ** 2
    )
)


print(
    "\nAttitude MAE:",
    attitude_mae
)

print(
    "Attitude RMSE:",
    attitude_rmse
)


# ============================================================
# VELOCITY METRICS
# ============================================================

velocity_error = (
    velocity_pred -
    velocity_true
)

velocity_mae = np.mean(
    np.abs(velocity_error)
)

velocity_rmse = np.sqrt(
    np.mean(
        velocity_error ** 2
    )
)


print(
    "\nVelocity MAE:",
    velocity_mae,
    "km/h"
)

print(
    "Velocity RMSE:",
    velocity_rmse,
    "km/h"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

# Flatten sequence dimension.

attitude_pred_flat = (
    attitude_pred
    .reshape(-1, 3)
)

attitude_true_flat = (
    attitude_true
    .reshape(-1, 3)
)

velocity_pred_flat = (
    velocity_pred
    .reshape(-1, 1)
)

velocity_true_flat = (
    velocity_true
    .reshape(-1, 1)
)


results = pd.DataFrame({

    "true_qx":
        attitude_true_flat[:, 0],

    "pred_qx":
        attitude_pred_flat[:, 0],

    "true_qy":
        attitude_true_flat[:, 1],

    "pred_qy":
        attitude_pred_flat[:, 1],

    "true_qz":
        attitude_true_flat[:, 2],

    "pred_qz":
        attitude_pred_flat[:, 2],

    "true_velocity_kmh":
        velocity_true_flat[:, 0],

    "pred_velocity_kmh":
        velocity_pred_flat[:, 0],

})


results_file = (
    OUTPUT_DIR /
    "avnet_test_predictions.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 30 COMPLETE")
print("=" * 70)

print(
    "\nAttitude model:"
)

print(
    best_attitude_model
)

print(
    "\nVelocity model:"
)

print(
    best_velocity_model
)

print(
    "\nTest predictions:"
)

print(
    results_file
)

print(
    "\nNo original dataset files were modified."
)

print("=" * 70)
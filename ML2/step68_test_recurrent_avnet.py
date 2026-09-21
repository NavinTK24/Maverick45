import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

ROOT = r"D:\Maverick\ML2"

SEQUENCE_DIR = os.path.join(
    ROOT,
    "step67_recurrent_sequences",
    "runs"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("STEP 68A - TEST RECURRENT AVNET")
print("=" * 90)

print(
    "\nDevice:",
    device
)


# ============================================================
# RECURRENT AVNET
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
        # Fully connected layers
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
        #
        # Same recurrent mechanism used by QDeepOdo:
        #
        # feature → GRUCell(feature, hidden)
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

        # window:
        # (10, 6)

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

        # (1,128,8)

        x = torch.relu(
            x
        )

        x = self.pool1(
            x
        )

        # (1,128,4)


        x = self.conv2(
            x
        )

        # (1,256,3)

        x = torch.relu(
            x
        )

        x = self.pool2(
            x
        )

        # (1,256,1)


        x = torch.flatten(
            x,
            start_dim=1
        )

        # (1,256)


        x = torch.relu(
            self.fc1(x)
        )

        # (1,1024)

        x = torch.relu(
            self.fc2(x)
        )

        # (1,512)


        return x.squeeze(
            0
        )


    def forward(
        self,
        sequence
    ):

        # sequence:
        # (N,10,6)

        sequence_length = sequence.shape[0]


        # Same concept as original QDeepOdo:
        # one hidden state for the complete sequence.

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
# CREATE MODELS
# ============================================================

ddatt_model = RecurrentAVNet(
    output_size=3
).to(device)

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
# FIND SEQUENCES
# ============================================================

files = [
    f
    for f in os.listdir(
        SEQUENCE_DIR
    )
    if f.endswith(".npz")
]


if len(files) == 0:

    raise FileNotFoundError(
        "No Step 67 sequence files found!"
    )


# Sort by sequence length

sequence_info = []

for filename in files:

    path = os.path.join(
        SEQUENCE_DIR,
        filename
    )

    d = np.load(
        path
    )

    sequence_info.append(
        (
            len(d["X"]),
            filename
        )
    )


sequence_info.sort()


# ============================================================
# TEST DIFFERENT SEQUENCE LENGTHS
# ============================================================

requested_lengths = [
    1,
    5,
    20,
    100,
    1000
]


print(
    "\n"
    + "=" * 90
)

print(
    "FORWARD-PASS TEST"
)

print(
    "=" * 90
)


for requested_length in requested_lengths:

    candidates = [
        item
        for item in sequence_info
        if item[0] >= requested_length
    ]


    if len(candidates) == 0:

        print(
            f"\nNo sequence >= {requested_length} windows"
        )

        continue


    actual_length, filename = candidates[0]


    path = os.path.join(
        SEQUENCE_DIR,
        filename
    )


    d = np.load(
        path
    )


    X = d["X"][
        :requested_length
    ].astype(
        np.float32
    )


    X_tensor = torch.from_numpy(
        X
    ).to(device)


    print(
        f"\nTesting sequence length = "
        f"{requested_length}"
    )

    print(
        "File:",
        filename
    )

    print(
        "Input:",
        tuple(
            X_tensor.shape
        )
    )


    with torch.no_grad():

        ddatt_output = ddatt_model(
            X_tensor
        )

        ddodo_output = ddodo_model(
            X_tensor
        )


    print(
        "DDATT output:",
        tuple(
            ddatt_output.shape
        )
    )

    print(
        "DDODO output:",
        tuple(
            ddodo_output.shape
        )
    )


    if ddatt_output.shape != (
        requested_length,
        3
    ):

        raise RuntimeError(
            "DDATT output shape incorrect!"
        )


    if ddodo_output.shape != (
        requested_length,
        1
    ):

        raise RuntimeError(
            "DDODO output shape incorrect!"
        )


    if not torch.isfinite(
        ddatt_output
    ).all():

        raise RuntimeError(
            "DDATT produced NaN/Inf!"
        )


    if not torch.isfinite(
        ddodo_output
    ).all():

        raise RuntimeError(
            "DDODO produced NaN/Inf!"
        )


    print(
        "Result: PASS"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "STEP 68A COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nThe recurrent AVNet successfully processes"
)

print(
    "different continuous sequence lengths."
)

print(
    "\nNo training was performed."
)
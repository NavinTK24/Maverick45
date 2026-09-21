import torch
import torch.nn as nn


print("=" * 75)
print("STEP 41 - BUILD BASELINE AVNET")
print("=" * 75)


# ============================================================
# AVNET
# ============================================================

class AVNet(nn.Module):

    def __init__(self, output_size):
        super().__init__()

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

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
        # GRU
        #
        # Baseline interpretation:
        # sequence length = 1
        # feature size = 512
        # hidden size = 512
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output = nn.Linear(
            512,
            output_size
        )


    def forward(self, x):

        # Input:
        # batch × 10 × 6

        # Conv1D requires:
        # batch × channels × samples

        x = x.transpose(1, 2)

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Expected:
        # batch × 256 × 1

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(x, start_dim=1)

        # batch × 256

        # ----------------------------------------------------
        # Fully connected
        # ----------------------------------------------------

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        # batch × 512

        # ----------------------------------------------------
        # Create GRU sequence dimension
        # ----------------------------------------------------

        x = x.unsqueeze(1)

        # batch × sequence_length × features
        #
        # = batch × 1 × 512

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        x, _ = self.gru(x)

        # Take final sequence output

        x = x[:, -1, :]

        # batch × 512

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        x = self.output(x)

        return x


# ============================================================
# CREATE BOTH NETWORKS
# ============================================================

print("\nCreating DDATT network...")

ddatt_model = AVNet(
    output_size=3
)

print("DDATT output size = 3")


print("\nCreating DDODO network...")

ddodo_model = AVNet(
    output_size=1
)

print("DDODO output size = 1")


# ============================================================
# TEST FORWARD PASS
# ============================================================

print("\n" + "=" * 75)
print("FORWARD PASS TEST")
print("=" * 75)

# Four example windows
test_input = torch.randn(
    4,
    10,
    6
)

print("\nInput:")
print(test_input.shape)


# DDATT

ddatt_output = ddatt_model(
    test_input
)

print("\nDDATT output:")
print(ddatt_output.shape)

print("Expected: (4, 3)")


# DDODO

ddodo_output = ddodo_model(
    test_input
)

print("\nDDODO output:")
print(ddodo_output.shape)

print("Expected: (4, 1)")


# ============================================================
# PARAMETER COUNT
# ============================================================

ddatt_parameters = sum(
    p.numel()
    for p in ddatt_model.parameters()
)

ddodo_parameters = sum(
    p.numel()
    for p in ddodo_model.parameters()
)


print("\n" + "=" * 75)
print("MODEL SIZE")
print("=" * 75)

print(f"DDATT parameters: {ddatt_parameters:,}")
print(f"DDODO parameters: {ddodo_parameters:,}")


# ============================================================
# PRINT ARCHITECTURE
# ============================================================

print("\n" + "=" * 75)
print("DDATT ARCHITECTURE")
print("=" * 75)

print(ddatt_model)


print("\n" + "=" * 75)
print("DDODO ARCHITECTURE")
print("=" * 75)

print(ddodo_model)


print("\n" + "=" * 75)
print("STEP 41 COMPLETE")
print("=" * 75)

print("\nNo training was performed.")
print("Only the model architecture and forward pass were tested.")
import torch
import sys
import os


MODEL_PATH = (
    r"D:\Maverick\ML2"
    r"\step47_training"
    r"\best_ddatt_model.pt"
)


print("=" * 100)
print("STEP 60 - INSPECT CURRENT GRU STRUCTURE")
print("=" * 100)


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


print(
    "\nCheckpoint type:"
)

print(
    type(checkpoint)
)


if isinstance(
    checkpoint,
    dict
):

    print(
        "\nCheckpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            " ",
            key
        )


# ============================================================
# TRY TO EXTRACT STATE DICT
# ============================================================

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

elif isinstance(
    checkpoint,
    dict
) and "state_dict" in checkpoint:

    state_dict = (
        checkpoint[
            "state_dict"
        ]
    )

else:

    state_dict = checkpoint


print(
    "\nState dictionary:"
)

if hasattr(
    state_dict,
    "keys"
):

    for key in state_dict.keys():

        print(
            key,
            tuple(
                state_dict[
                    key
                ].shape
            )
        )

else:

    print(
        "Could not identify state dictionary."
    )


print(
    "\n"
    + "=" * 100
)

print(
    "STEP 60 COMPLETE"
)

print(
    "=" * 100
)

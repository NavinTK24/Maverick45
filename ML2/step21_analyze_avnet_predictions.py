from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# STEP 21
# Analyze AVNet test predictions
# ============================================================

ML2 = Path(r"D:\Maverick\ML2")

INPUT_FILE = (
    ML2 /
    "avnet_models" /
    "avnet_test_predictions.csv"
)

OUTPUT_DIR = (
    ML2 /
    "avnet_models" /
    "prediction_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("AVNET PREDICTION ANALYSIS")
print("=" * 70)

print("\nFile:")
print(INPUT_FILE)

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("BASIC STATISTICS")
print("=" * 70)


for column in df.columns:

    values = df[column].to_numpy(
        dtype=np.float64
    )

    print(
        f"\n{column}"
    )

    print(
        f"  min    = {np.min(values):.8f}"
    )

    print(
        f"  max    = {np.max(values):.8f}"
    )

    print(
        f"  mean   = {np.mean(values):.8f}"
    )

    print(
        f"  std    = {np.std(values):.8f}"
    )

    print(
        f"  median = {np.median(values):.8f}"
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    true,
    pred
):

    error = pred - true

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    mse = np.mean(
        error ** 2
    )

    # R2
    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (true - np.mean(true)) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1 -
            ss_res / ss_tot
        )

    else:

        r2 = np.nan

    # Pearson correlation
    if (
        np.std(true) > 0
        and
        np.std(pred) > 0
    ):

        correlation = np.corrcoef(
            true,
            pred
        )[0, 1]

    else:

        correlation = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "Correlation": correlation,
    }


# ============================================================
# ATTITUDE COMPONENT ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("ATTITUDE COMPONENT METRICS")
print("=" * 70)


attitude_components = [
    ("qx", "true_qx", "pred_qx"),
    ("qy", "true_qy", "pred_qy"),
    ("qz", "true_qz", "pred_qz"),
]


attitude_results = []


for name, true_col, pred_col in attitude_components:

    true = df[true_col].to_numpy(
        dtype=np.float64
    )

    pred = df[pred_col].to_numpy(
        dtype=np.float64
    )

    metrics = calculate_metrics(
        true,
        pred
    )

    metrics["component"] = name

    attitude_results.append(
        metrics
    )

    print(
        f"\n{name}"
    )

    for key in [
        "MAE",
        "RMSE",
        "MSE",
        "R2",
        "Correlation"
    ]:

        print(
            f"  {key:12s}: "
            f"{metrics[key]:.8f}"
        )


attitude_results_df = pd.DataFrame(
    attitude_results
)

attitude_results_df = attitude_results_df[
    [
        "component",
        "MAE",
        "RMSE",
        "MSE",
        "R2",
        "Correlation",
    ]
]


# ============================================================
# YAW-ONLY CHECK
# ============================================================

print("\n")
print("=" * 70)
print("YAW-ONLY TARGET CHECK")
print("=" * 70)


for column in [
    "true_qx",
    "true_qy"
]:

    values = np.abs(
        df[column].to_numpy(
            dtype=np.float64
        )
    )

    print(
        f"{column}:"
    )

    print(
        f"  max |value| = {np.max(values):.12f}"
    )

    print(
        f"  mean |value| = {np.mean(values):.12f}"
    )


# ============================================================
# QZ ANALYSIS
# ============================================================

true_qz = df[
    "true_qz"
].to_numpy(
    dtype=np.float64
)

pred_qz = df[
    "pred_qz"
].to_numpy(
    dtype=np.float64
)

qz_error = (
    pred_qz -
    true_qz
)

print("\nQZ prediction range:")
print(
    f"  true min = {true_qz.min():.8f}"
)
print(
    f"  true max = {true_qz.max():.8f}"
)
print(
    f"  pred min = {pred_qz.min():.8f}"
)
print(
    f"  pred max = {pred_qz.max():.8f}"
)

print("\nQZ error percentiles:")

for p in [
    50,
    75,
    90,
    95,
    99,
    99.9
]:

    print(
        f"  P{p:<5} = "
        f"{np.percentile(np.abs(qz_error), p):.8f}"
    )


# ============================================================
# VELOCITY ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("VELOCITY METRICS")
print("=" * 70)


true_v = df[
    "true_velocity_kmh"
].to_numpy(
    dtype=np.float64
)

pred_v = df[
    "pred_velocity_kmh"
].to_numpy(
    dtype=np.float64
)

velocity_metrics = calculate_metrics(
    true_v,
    pred_v
)

for key, value in velocity_metrics.items():

    print(
        f"{key:12s}: {value:.8f}"
    )


# ============================================================
# VELOCITY PREDICTION RANGE
# ============================================================

print("\nVelocity ranges:")

print(
    f"True velocity:"
    f" {true_v.min():.3f}"
    f" → {true_v.max():.3f} km/h"
)

print(
    f"Predicted velocity:"
    f" {pred_v.min():.3f}"
    f" → {pred_v.max():.3f} km/h"
)


# ============================================================
# SPEED-BIN ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("VELOCITY ERROR BY TRUE-SPEED RANGE")
print("=" * 70)


bins = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    120,
    np.inf
]

labels = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-80",
    "80-100",
    "100-120",
    "120+",
]


speed_bins = pd.cut(
    true_v,
    bins=bins,
    labels=labels,
    right=False
)


speed_rows = []


for label in labels:

    mask = (
        speed_bins == label
    )

    count = int(
        np.sum(mask)
    )

    if count == 0:
        continue

    true_bin = true_v[mask]
    pred_bin = pred_v[mask]

    metrics = calculate_metrics(
        true_bin,
        pred_bin
    )

    metrics["speed_range"] = label
    metrics["samples"] = count

    speed_rows.append(
        metrics
    )

    print(
        f"\n{label} km/h"
    )

    print(
        f"  samples = {count}"
    )

    print(
        f"  MAE     = {metrics['MAE']:.4f}"
    )

    print(
        f"  RMSE    = {metrics['RMSE']:.4f}"
    )

    print(
        f"  R2      = {metrics['R2']:.4f}"
    )

    print(
        f"  Corr    = {metrics['Correlation']:.4f}"
    )


speed_results_df = pd.DataFrame(
    speed_rows
)


# ============================================================
# BIAS ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("VELOCITY BIAS")
print("=" * 70)


velocity_error = (
    pred_v -
    true_v
)

print(
    f"Mean error: "
    f"{np.mean(velocity_error):.6f} km/h"
)

print(
    f"Median error: "
    f"{np.median(velocity_error):.6f} km/h"
)


# ============================================================
# UNDER / OVER PREDICTION
# ============================================================

under = np.sum(
    velocity_error < 0
)

over = np.sum(
    velocity_error > 0
)

equal = np.sum(
    velocity_error == 0
)

total = len(
    velocity_error
)

print("\nPrediction direction:")

print(
    f"Under-predicted : "
    f"{under:,} "
    f"({under / total * 100:.2f}%)"
)

print(
    f"Over-predicted  : "
    f"{over:,} "
    f"({over / total * 100:.2f}%)"
)

print(
    f"Exactly equal   : "
    f"{equal:,} "
    f"({equal / total * 100:.2f}%)"
)


# ============================================================
# BASELINE: MEAN VELOCITY PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)


mean_velocity = np.mean(
    true_v
)

baseline_pred = np.full_like(
    true_v,
    mean_velocity
)

baseline_metrics = calculate_metrics(
    true_v,
    baseline_pred
)

print(
    "\nMean-velocity baseline:"
)

for key, value in baseline_metrics.items():

    print(
        f"{key:12s}: {value:.8f}"
    )


print(
    "\nAVNet velocity:"
)

for key, value in velocity_metrics.items():

    print(
        f"{key:12s}: {value:.8f}"
    )


# ============================================================
# IMPROVEMENT OVER BASELINE
# ============================================================

baseline_rmse = (
    baseline_metrics["RMSE"]
)

model_rmse = (
    velocity_metrics["RMSE"]
)

if baseline_rmse > 0:

    improvement = (
        1 -
        model_rmse /
        baseline_rmse
    ) * 100

else:

    improvement = np.nan


print(
    f"\nRMSE improvement over "
    f"mean baseline: "
    f"{improvement:.2f}%"
)


# ============================================================
# LARGE ERRORS
# ============================================================

print("\n")
print("=" * 70)
print("LARGEST VELOCITY ERRORS")
print("=" * 70)


large_error_df = pd.DataFrame({

    "true_velocity_kmh":
        true_v,

    "pred_velocity_kmh":
        pred_v,

    "error_kmh":
        velocity_error,

    "absolute_error_kmh":
        np.abs(velocity_error),

})


large_error_df = (
    large_error_df
    .sort_values(
        "absolute_error_kmh",
        ascending=False
    )
)


print(
    large_error_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

attitude_file = (
    OUTPUT_DIR /
    "attitude_metrics.csv"
)

speed_file = (
    OUTPUT_DIR /
    "velocity_speed_bin_metrics.csv"
)

baseline_file = (
    OUTPUT_DIR /
    "velocity_baseline_metrics.csv"
)

large_error_file = (
    OUTPUT_DIR /
    "largest_velocity_errors.csv"
)


attitude_results_df.to_csv(
    attitude_file,
    index=False
)

speed_results_df.to_csv(
    speed_file,
    index=False
)

pd.DataFrame([
    {
        "model": "mean_velocity_baseline",
        **baseline_metrics
    },
    {
        "model": "AVNet",
        **velocity_metrics
    }
]).to_csv(
    baseline_file,
    index=False
)

large_error_df.head(1000).to_csv(
    large_error_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 31 COMPLETE")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    attitude_file
)

print(
    speed_file
)

print(
    baseline_file
)

print(
    large_error_file
)

print("=" * 70)
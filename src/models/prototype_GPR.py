import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

# === Setup ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"D:\BA\PID-Controller-optimization-with-machine-learning\models\GPR\gpr_model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "training_log.txt")

def log(msg):
    timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(timestamped)
    with open(log_path, "a") as f:
        f.write(timestamped + "\n")

log(" Loading dataset...")
try:
    df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
    df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])
    log(f" Dataset loaded with {len(df)} entries.")
except Exception as e:
    log(f" Failed to load dataset: {e}")
    sys.exit(1)

# === Log-transform targets ===
df["Kp_log"] = np.log10(df["Kp"] + 1e-6)
df["Ki_log"] = np.log10(df["Ki"] + 1e-6)
df["Kd_log"] = np.log10(df["Kd"] + 1e-6)
targets = ["Kp_log", "Ki_log", "Kd_log"]

# === Feature engineering ===
df["K_T1"] = df["K"] * df["T1"]
df["K_T2"] = df["K"] * df["T2"]
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)

# === Sample down for GPR stability ===
if len(df) > 3000:
    df = df.sample(n=3000, random_state=42)
    log("🔁 Sampled down to 3000 entries for GPR efficiency.")

log(f" Output directory created: {output_dir}")

metrics_dict = {}
predictions = {}

# === Train one GPR model per target ===
for target in targets:
    param_name = target.replace("_log", "")
    log(f"\n Training GPR model for target: {param_name}")

    try:
        df_filtered = df[df[target] <= 2].dropna(subset=["K", "T1", "T2", target])
        X = df_filtered[["K", "T1", "T2", "K_T1", "K_T2", "T1_T2_ratio"]]
        y = df_filtered[target].values
        log(f"    Samples after filtering: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === Improved Kernel ===
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * \
                 RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))

        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=42)

        log("    Fitting model...")
        gpr.fit(X_train, y_train)

        log("    Predicting with uncertainty...")
        y_pred_log, y_std = gpr.predict(X_test, return_std=True)
        y_pred = 10 ** y_pred_log
        y_true = 10 ** y_test

        # === Outlier Detection ===
        abs_error = np.abs(y_pred - y_true)
        error_threshold = 3 * np.std(y_true)
        inlier_mask = abs_error < error_threshold

        log(f"    {np.sum(~inlier_mask)} outliers removed from metrics (>{error_threshold:.2f})")

        y_true_filtered = y_true[inlier_mask]
        y_pred_filtered = y_pred[inlier_mask]

        metrics_dict[param_name] = {
            "MAE": mean_absolute_error(y_true_filtered, y_pred_filtered),
            "MSE": mean_squared_error(y_true_filtered, y_pred_filtered),
            "R2": r2_score(y_true_filtered, y_pred_filtered)
        }

        predictions[param_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_std": y_std,
            "mask": inlier_mask
        }

        joblib.dump(gpr, os.path.join(output_dir, f"gpr_{param_name}.pkl"))
        log("    Model saved.")

        # === Plotting with optional clipping ===
        plt.figure(figsize=(7, 5))
        y_pred_plot = np.clip(y_pred, 0, np.percentile(y_true, 99))  # clip for visualization

        plt.errorbar(y_true, y_pred_plot, yerr=y_std, fmt='o', alpha=0.6, label="Predictions ±1σ")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
        plt.xlabel(f"True {param_name}")
        plt.ylabel(f"Predicted {param_name}")
        plt.title(f"GPR: True vs Predicted {param_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param_name}_true_vs_predicted.png"))
        plt.close()
        log("    Plot saved.")

    except Exception as e:
        log(f" Failed to train/predict for {param_name}: {e}")
        continue

# === Save all results ===
log(" Saving summary metrics and predictions...")

metrics_df = pd.DataFrame(metrics_dict).T
metrics_df.to_csv(os.path.join(output_dir, "gpr_metrics.csv"))

all_preds_df = pd.DataFrame()
for param in predictions:
    pred_data = predictions[param]
    temp_df = pd.DataFrame({
        f"{param}_true": pred_data["y_true"],
        f"{param}_pred": pred_data["y_pred"],
        f"{param}_std": pred_data["y_std"],
        f"{param}_inlier": pred_data["mask"]
    })
    all_preds_df = pd.concat([all_preds_df, temp_df], axis=1)

all_preds_df.to_csv(os.path.join(output_dir, "gpr_predictions.csv"), index=False)

log(" GPR training complete. All results saved.")
log(f" Final results in: {output_dir}")

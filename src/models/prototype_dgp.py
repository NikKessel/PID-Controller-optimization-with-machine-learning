import os
import torch
import gpytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import logging

# === Configure Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Load and preprocess data ===
# Use a more robust path handling
# Ensure this path is correct for your environment
data_path = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv"
if not os.path.exists(data_path):
    logging.error(f"Data file not found at: {data_path}")
    raise FileNotFoundError(f"Data file not found at: {data_path}")

df = pd.read_csv(data_path)
df = df.dropna(subset=["K", "T1", "T2", "L" ,"Kp", "Ki", "Kd"])
df = df[df["Kp"] < 20]
df = df[df["Ki"] < 20]
df = df[df["Kd"] < 20]

logging.info(f"Initial DataFrame shape: {df.shape}")
logging.info(df.head())
logging.info(df.describe())




# Feature engineering (Consider if these are truly beneficial after scaling)
# For now, let's keep the original features as the primary focus
# df["K_T1"] = df["K"] * df["T1"]
# df["K_T2"] = df["K"] * df["T2"]
# df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3) # Avoid division by zero

# Feature and target selection
features = ["K", "T1", "T2", "L"]
targets = ["Kp", "Ki", "Kd"]
print(df["Kp"].mean(), df["Kp"].std())
print(df["Ki"].mean(), df["Ki"].std())
print(df["Kd"].mean(), df["Kd"].std())



# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = fr"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGP\dgp_model_original_scaled_targets_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Output directory created: {output_dir}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class DGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, inducing_points_count=256):
        # Initialize inducing points to be random samples from a standard normal distribution
        # This will be scaled by the data during training implicitly
        inducing_points = torch.randn(inducing_points_count, input_dim)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

metrics = {}
df_all_preds = None  # Initialize empty for combined predictions

for target in targets:
    logging.info(f"\n🚀 Training for target: {target}")
    df_target = df.dropna(subset=features + [target])
    
    X = df_target[features].values.astype(np.float32)
    y = df_target[target].values.astype(np.float32)

    logging.info(f"Original {target} stats: Min={y.min():.4f}, Max={y.max():.4f}, Mean={y.mean():.4f}, Std={y.std():.4f}")

    # Initialize scalers for X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Normalize features
    X_scaled = scaler_X.fit_transform(X)

    # Normalize targets
    # Reshape y to (n_samples, 1) for StandardScaler, then flatten back
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    logging.info(f"Scaled {target} stats: Min={y_scaled.min():.4f}, Max={y_scaled.max():.4f}, Mean={y_scaled.mean():.4f}, Std={y_scaled.std():.4f}")


    # Train-test split (get indices)
    train_idx, test_idx = train_test_split(df_target.index, test_size=0.2, random_state=42)
    
    # Use original y for test set for final evaluation
    y_train = y_scaled[df_target.index.isin(train_idx)]
    X_train = X_scaled[df_target.index.isin(train_idx)]
    
    y_test_original = y[df_target.index.isin(test_idx)] # Keep original y for true comparison
    X_test = X_scaled[df_target.index.isin(test_idx)]

    logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    X_test_tensor = torch.tensor(X_test).to(device)

    model = DGPModel(X.shape[1]).to(device)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=torch.ones(X_train_tensor.size(0)).to(device) * 1e-4
).to(device)


    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))

    training_iterations = 300 # You might need to increase this based on loss convergence
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        if (i + 1) % 50 == 0 or i == 0: # Log every 50 iterations and at the start
            logging.info(f"Target: {target} - Iter {i+1}/{training_iterations} - Loss: {loss.item():.4f}")
        optimizer.step()

    logging.info(f"Training complete for {target}.")

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test_tensor))
        y_pred_scaled = preds.mean.cpu().numpy()
        y_std_scaled = preds.variance.sqrt().cpu().numpy()

        # Inverse transform predictions and std to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        # The standard deviation in the original scale is the scaled std multiplied by the scale factor of the original data.
        y_std = y_std_scaled * scaler_y.scale_[0]

    # === Store predictions for regional evaluation ===
    param_name = target
    df_test_part = df_target.loc[test_idx].copy()
    df_test_part[f"{param_name}_true"] = y_test_original
    df_test_part[f"{param_name}_pred"] = y_pred

    if df_all_preds is None:
        df_all_preds = df_test_part.copy()
    else:
        # Merge only the true and predicted columns for the current target
        df_all_preds = df_all_preds.merge(
            df_test_part[[f"{param_name}_true", f"{param_name}_pred"]],
            left_index=True, right_index=True, how='left' # Use 'left' join to keep existing rows
        )

    # === Log metrics ===
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    metrics[param_name] = {
        "R2": r2,
        "MAE": mae
    }
    logging.info(f"Metrics for {target}: R2={r2:.4f}, MAE={mae:.4f}")

    # === Plot
    plt.figure(figsize=(8, 7)) # Increased figure size for better readability
    plt.errorbar(y_test_original, y_pred, yerr=y_std, fmt='o', alpha=0.6, label="Predictions ±σ", capsize=3)
    
    # Determine plot limits based on data
    min_val = min(y_test_original.min(), y_pred.min())
    max_val = max(y_test_original.max(), y_pred.max())
    # Add a small buffer to the limits
    buffer = (max_val - min_val) * 0.1
    plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--', label="Ideal (y=x)")
    
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.title(f"{param_name}: True vs Predicted (R2: {r2:.2f}, MAE: {mae:.2f})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{param_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Plot saved to: {plot_path}")

    # === Save model components and scalers ===
    torch.save(model.state_dict(), os.path.join(output_dir, f"dgp_{param_name}.pth"))
    torch.save(likelihood.state_dict(), os.path.join(output_dir, f"dgp_{param_name}_likelihood.pth"))
    joblib.dump(scaler_X, os.path.join(output_dir, f"dgp_{param_name}_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, f"dgp_{param_name}_scaler_y.pkl")) # Save target scaler!
    logging.info(f"Model, likelihood, and scalers saved for {target}.")


# === Region-wise Evaluation ===
logging.info("\nStarting region-wise evaluation...")

# Ensure these columns exist before binning.
# They should already be in df_all_preds due to the merge operations.
if "K" not in df_all_preds.columns or "T1" not in df_all_preds.columns:
    logging.error("K or T1 not found in df_all_preds for regional analysis.")
    # Exit or handle error appropriately

# Ensure 'K_bin' and 'T1_bin' are created only if they don't exist, or overwrite.
# Using pd.cut on the original 'K' and 'T1' columns in df_all_preds.
df_all_preds["K_bin"] = pd.cut(df_all_preds["K"], bins=[0, 1, 5, 10, 20, 30, np.inf], labels=["K<1", "1-5", "5-10", "10-20", "20-30", ">30"])
df_all_preds["T1_bin"] = pd.cut(df_all_preds["T1"], bins=[0, 1, 10, 30, 75, np.inf], labels=["T1<1", "1-10", "10-30", "30-75", ">75"])

region_metrics = []

for k_bin in df_all_preds["K_bin"].cat.categories:
    for t1_bin in df_all_preds["T1_bin"].cat.categories:
        mask = (df_all_preds["K_bin"] == k_bin) & (df_all_preds["T1_bin"] == t1_bin)
        region_label = f"K: {k_bin} | T1: {t1_bin}"
        
        num_samples_in_region = mask.sum()
        if num_samples_in_region < 10:
            logging.info(f"Skipping region '{region_label}': Not enough samples ({num_samples_in_region} < 10).")
            continue
        
        logging.info(f"Evaluating region: {region_label} with {num_samples_in_region} samples.")

        for param in targets: # Use 'targets' list directly
            y_true = df_all_preds.loc[mask, f"{param}_true"]
            y_pred = df_all_preds.loc[mask, f"{param}_pred"]
            
            # Drop NaN values specifically for this region and parameter before calculating metrics
            valid_indices = y_true.notna() & y_pred.notna()
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]

            if len(y_true) < 2: # Need at least 2 samples for R2
                logging.warning(f"Not enough valid samples for {param} in region '{region_label}' for metric calculation (found {len(y_true)}).")
                continue

            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            region_metrics.append({
                "Region": region_label,
                "Parameter": param,
                "R2": r2,
                "MAE": mae,
                "Samples": len(y_true) # Use actual number of valid samples
            })

if region_metrics:
    region_df = pd.DataFrame(region_metrics)
    region_df.sort_values(by=["Parameter", "R2"], ascending=[True, False], inplace=True) # Sort by parameter then R2
    
    region_out_path = os.path.join(output_dir, "dgp_region_metrics.csv")
    region_df.to_csv(region_out_path, index=False)
    logging.info(f"Region-wise metrics saved to: {region_out_path}")
    logging.info("\nRegion-wise Metrics Summary:")
    logging.info(region_df.head(10)) # Display top 10 regions
else:
    logging.warning("No region-wise metrics were generated.")


# === Save Overall Summary Metrics ===
overall_metrics_df = pd.DataFrame(metrics).T
overall_metrics_path = os.path.join(output_dir, "metrics_original_space.csv")
overall_metrics_df.to_csv(overall_metrics_path)
logging.info(f"Overall metrics saved to: {overall_metrics_path}")
logging.info("\nOverall Metrics:")
logging.info(overall_metrics_df)


logging.info("\n✅ All done. Results saved in: " + output_dir)
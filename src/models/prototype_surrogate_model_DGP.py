# === Surrogate Model to Predict Performance Metrics ===
# Input: [K, T1, T2, Kp, Ki, Kd]
# Output: [ISE, Overshoot, SettlingTime, RiseTime]

import os
import torch
import gpytorch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import matplotlib.pyplot as plt


def inverse_log1p_transform(pred_log):
    """
    Inverse transform of y = log1p(original_value) = log(1 + original_value).
    Ensures no negative values after inversion.
    """
    pred = np.expm1(pred_log)  # inverse of log1p
    pred = np.clip(pred, 0, None)  # clamp negative predictions to zero
    return pred


# === Load dataset ===
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd", "ISE", "Overshoot", "SettlingTime", "RiseTime"])
df = df.sample(n=100000, random_state=42).reset_index(drop=True)

# === Filter outliers ===
print(f"Original dataset: {df.shape}")

df = df[df["ISE"] < 100]
df = df[df["ISE"] > 0.001]
df = df[df["SettlingTime"] < 500]
df = df[df["RiseTime"] < 300]
df = df[df["SettlingTime"] > 0.1000]
df = df[df["RiseTime"] > 0.1000]
df = df[df["Overshoot"] > 0.001]
df = df[df["Kp"] < 200 ]
df = df[df["Kp"] > 0.3 ]
df = df[df["Ki"] < 200]
df = df[df["Kd"] < 200]
print(f"After filtering: {df.shape}")


# === Log-transform targets ===
# Store original values for later use
df["ISE_orig"] = df["ISE"].copy()
df["Overshoot_orig"] = df["Overshoot"].copy()
df["SettlingTime_orig"] = df["SettlingTime"].copy()
df["RiseTime_orig"] = df["RiseTime"].copy()

# Apply log1p transform consistently
df["ISE_log"] = np.log1p(df["ISE"])
df["SettlingTime_log"] = np.log1p(df["SettlingTime"])
df["RiseTime_log"] = np.log1p(df["RiseTime"])
#df["Overshoot"] = np.log1p(df["Overshoot"])  # No log transform for Overshoot
# Visualization of transforms
for col in ["ISE", "SettlingTime", "RiseTime"]:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df[col], bins=100)
    plt.title(f"{col} (Original)")

    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df[col]), bins=100)
    plt.title(f"{col}_log (log1p transformed)")

    plt.tight_layout()
    plt.show()

features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
# Define which targets use log transform and their original column names
target_config = {
    "ISE_log": {"original": "ISE_orig", "use_log": True},
    "Overshoot": {"original": "Overshoot_orig", "use_log": False},
    "SettlingTime_log": {"original": "SettlingTime_orig", "use_log": True},
    "RiseTime_log": {"original": "RiseTime_orig", "use_log": True}
}

# === Output directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGPSurrogate\surrogate_perf_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

class VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim):
        inducing_points = torch.randn(128, input_dim)
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

# === Train separate DGP for each target ===
metrics = {}

for target, config in target_config.items():
    print(f"\n🚀 Training for target: {target}")
    
    # === 1. Preprocess ===
    df_target = df.dropna(subset=features + [target] + [config["original"]])
    X = df_target[features].values.astype(np.float32)
    y = df_target[target].values.astype(np.float32)  # This is the transformed target for training
    y_orig = df_target[config["original"]].values.astype(np.float32)  # Original scale for evaluation

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data consistently
    train_idx, test_idx = train_test_split(range(len(df_target)), test_size=0.2, random_state=42)
    
    X_train = X_scaled[train_idx]
    y_train = y[train_idx]  # Log-transformed for training
    X_test = X_scaled[test_idx]
    y_test = y[test_idx]  # Log-transformed for model output
    y_test_orig = y_orig[test_idx]  # Original scale for evaluation

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # === 2. Train model ===
    model = VariationalGP(X.shape[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))

    for i in range(300):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        if (i + 1) % 50 == 0:
            print(f"Iter {i+1}/300 - Loss: {loss.item():.4f}")
        optimizer.step()

    # === 3. Predict ===
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test_tensor))
        y_pred_transformed = preds.mean.numpy()  # This is in transformed space
        y_std_transformed = preds.variance.sqrt().numpy()

        # Convert predictions back to original scale
        if config["use_log"]:
            y_pred_orig = np.expm1(y_pred_transformed)  # Inverse of log1p
            # For uncertainty, we need to transform the bounds
            y_pred_lower = np.expm1(y_pred_transformed - y_std_transformed)
            y_pred_upper = np.expm1(y_pred_transformed + y_std_transformed)
            y_std_orig = (y_pred_upper - y_pred_lower) / 2  # Approximate std in original space
        else:
            y_pred_orig = y_pred_transformed
            y_std_orig = y_std_transformed

        # Ensure no negative predictions
        y_pred_orig = np.clip(y_pred_orig, 0, None)

    # === 4. Evaluation and diagnostics ===
    # Use original scale for all evaluations
    r2 = r2_score(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    metrics[target] = {"R2": r2, "MAE": mae}

    print(f"📊 {target} - R²: {r2:.4f}, MAE: {mae:.4f}")

    # Create diagnostic dataframe
    diagnostic_df = pd.DataFrame({
        'y_true_orig': y_test_orig,
        'y_pred_orig': y_pred_orig,
        'abs_error': np.abs(y_test_orig - y_pred_orig),
        'rel_error_percent': 100 * np.abs(y_test_orig - y_pred_orig) / np.clip(y_test_orig, 1e-6, None)
    })
    
    diagnostic_df = diagnostic_df.sort_values(by="abs_error", ascending=False)
    diagnostic_df.to_csv(os.path.join(output_dir, f"{target}_diagnostics.csv"), index=False)

    # === 5. Plotting ===
    plt.figure(figsize=(10, 6))
    plt.errorbar(y_test_orig, y_pred_orig, yerr=y_std_orig, fmt='o', alpha=0.6, label="Predictions ±σ")
    
    # Perfect prediction line
    min_val = min(y_test_orig.min(), y_pred_orig.min())
    max_val = max(y_test_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction")
    
    plt.xlabel(f"True {config['original']}")
    plt.ylabel(f"Predicted {config['original']}")
    plt.title(f"{target}: R² = {r2:.3f}, MAE = {mae:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target}_plot.png"), dpi=300)
    plt.close()

    # === 6. Save model artifacts ===
    torch.save(model.state_dict(), os.path.join(output_dir, f"{target}_model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(output_dir, f"{target}_likelihood.pth"))
    joblib.dump(scaler, os.path.join(output_dir, f"{target}_scaler.pkl"))
    
    # Save target configuration for later use
    joblib.dump(config, os.path.join(output_dir, f"{target}_config.pkl"))

# === Save Summary ===
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"))
print("\n✅ All surrogate models trained and saved.")
print("\n📊 Final Performance Summary:")
print(metrics_df)
# Assuming 'Overshoot' is one of the columns
y_os = df["Overshoot"].values

plt.hist(y_os, bins=30)
plt.title("Overshoot Distribution")
plt.xlabel("Overshoot (%)")
plt.hist(y_os, bins=30)
plt.title("Overshoot Distribution")
plt.xlabel("Overshoot (%)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()  # <-- This is what displays the plot!

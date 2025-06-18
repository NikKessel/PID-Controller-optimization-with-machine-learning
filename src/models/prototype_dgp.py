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

# === Load and preprocess data ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])

# Log transform targets
df["Kp_log"] = np.log10(df["Kp"] + 1e-6)
df["Ki_log"] = np.log10(df["Ki"] + 1e-6)
df["Kd_log"] = np.log10(df["Kd"] + 1e-6)

# Feature engineering
df["K_T1"] = df["K"] * df["T1"]
df["K_T2"] = df["K"] * df["T2"]
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)

features = ["K", "T1", "T2", "K_T1", "K_T2", "T1_T2_ratio"]
targets = ["Kp_log", "Ki_log", "Kd_log"]

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"D:/BA/PID-Controller-optimization-with-machine-learning/models/DGP/dgp_model_{timestamp}"
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

metrics = {}
df_all_preds = None  # Initialize empty

for target in targets:
    print(f"\n🚀 Training for target: {target}")
    df_target = df.dropna(subset=features + [target])
    X = df_target[features].values.astype(np.float32)
    y = df_target[target].values.astype(np.float32)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (get indices)
    train_idx, test_idx = train_test_split(df_target.index, test_size=0.2, random_state=42)
    X_train = X_scaled[df_target.index.isin(train_idx)]
    y_train = y[df_target.index.isin(train_idx)]
    X_test = X_scaled[df_target.index.isin(test_idx)]
    y_test = y[df_target.index.isin(test_idx)]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

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
        print(f"Iter {i+1}/300 - Loss: {loss.item():.4f}")
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test_tensor))
        y_pred_log = preds.mean.numpy()
        y_std = preds.variance.sqrt().numpy()

    # === Store predictions for regional evaluation ===
    param_name = target.replace("_log", "")
    df_test_part = df_target.loc[test_idx].copy()
    df_test_part[f"{param_name}_true_log"] = y_test
    df_test_part[f"{param_name}_pred_log"] = y_pred_log

    if df_all_preds is None:
        df_all_preds = df_test_part.copy()
    else:
        df_all_preds = df_all_preds.merge(
            df_test_part[[f"{param_name}_true_log", f"{param_name}_pred_log"]],
            left_index=True, right_index=True
        )

    # === Log metrics ===
    r2_log = r2_score(y_test, y_pred_log)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    metrics[param_name] = {
        "R2_log": r2_log,
        "MAE_log": mae_log
    }

    # === Plot
    plt.figure(figsize=(6, 5))
    plt.errorbar(y_test, y_pred_log, yerr=y_std, fmt='o', alpha=0.6, label="Predictions ±σ")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal")
    plt.xlabel("True log10")
    plt.ylabel("Predicted log10")
    plt.title(f"{param_name}: True vs Predicted (log-domain)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param_name}_log_plot.png"))
    plt.close()

    # === Save model
    torch.save(model.state_dict(), os.path.join(output_dir, f"{param_name}_model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(output_dir, f"{param_name}_likelihood.pth"))
    joblib.dump(scaler, os.path.join(output_dir, f"{param_name}_scaler.pkl"))

# === Region-wise Evaluation ===
df_all_preds["K_bin"] = pd.cut(df_all_preds["K"], bins=[0, 1, 5, 10, 20, 30], labels=["K<1", "1-5", "5-10", "10-20", "20-30"])
df_all_preds["T1_bin"] = pd.cut(df_all_preds["T1"], bins=[0, 1, 10, 30, 75], labels=["T1<1", "1-10", "10-30", "30-75"])

region_metrics = []

for k_bin in df_all_preds["K_bin"].cat.categories:
    for t1_bin in df_all_preds["T1_bin"].cat.categories:
        mask = (df_all_preds["K_bin"] == k_bin) & (df_all_preds["T1_bin"] == t1_bin)
        region_label = f"{k_bin} & {t1_bin}"
        if mask.sum() < 10:
            continue
        for param in ["Kp", "Ki", "Kd"]:
            y_true = df_all_preds.loc[mask, f"{param}_true_log"]
            y_pred = df_all_preds.loc[mask, f"{param}_pred_log"]
            if y_true.isna().any() or y_pred.isna().any():
                continue
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            region_metrics.append({
                "Region": region_label,
                "Parameter": param,
                "R2_log": r2,
                "MAE_log": mae,
                "Samples": mask.sum()
            })

region_df = pd.DataFrame(region_metrics)
region_df.sort_values(by="R2_log", ascending=False, inplace=True)

region_out_path = os.path.join(output_dir, "dgp_region_metrics.csv")
region_df.to_csv(region_out_path, index=False)

# === Save Summary
pd.DataFrame(metrics).T.to_csv(os.path.join(output_dir, "metrics_log_space.csv"))

print("\n✅ All done. Results saved in:", output_dir)
print("✅ Region-wise metrics saved to:", region_out_path)

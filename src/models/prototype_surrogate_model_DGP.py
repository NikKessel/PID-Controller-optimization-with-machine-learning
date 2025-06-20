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

# === Filter outliers ===
df = df[df["ISE"] < 100]
df = df[df["SettlingTime"] < 500]
df = df[df["RiseTime"] < 300]
df = df[df["Overshoot"] > 0.03]
df = df[df["Kp"] < 100]
df = df[df["Ki"] < 100]
df = df[df["Kd"] < 100]


# === Log-transform targets ===
#df["ISE_log"] = np.log10(df["ISE"] + 1e-3)
#df["Overshoot_log"] = np.log10(df["Overshoot"] + 1e-3)
#df["SettlingTime_log"] = np.log10(df["SettlingTime"] + 1e-3)
#df["RiseTime_log"] = np.log10(df["RiseTime"] + 1e-3)
# New code using natural log1p transform
df["ISE_log"] = np.log1p(df["ISE"])            # log(1 + ISE)
#df["Overshoot_log"] = np.log1p(df["Overshoot"])
df["SettlingTime_log"] = np.log1p(df["SettlingTime"])
df["RiseTime_log"] = np.log1p(df["RiseTime"])


# Scale overshoot

features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
targets = ["ISE_log", "Overshoot", "SettlingTime_log", "RiseTime_log"]

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

for target in targets:
    print(f"\n🚀 Training for target: {target}")
    df_target = df.dropna(subset=features + [target])
    X = df_target[features].values.astype(np.float32)
    y = df_target[target].values.astype(np.float32)

    # Normalize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_idx, test_idx = train_test_split(df_target.index, test_size=0.2, random_state=42)
    X_train = X_scaled[df_target.index.isin(train_idx)]
    y_train = y[df_target.index.isin(train_idx)]
    X_test = X_scaled[df_target.index.isin(test_idx)]
    y_test = y[df_target.index.isin(test_idx)]

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
        y_pred_log = preds.mean.numpy()  # model prediction (log-domain or original scale)
        y_std = preds.variance.sqrt().numpy()

        # Conditional inverse transform based on target name
        if "log" in target:
            y_pred = np.expm1(y_pred_log)  # inverse log1p transform
        else:
            y_pred = y_pred_log  # no transform for original scale target

        y_pred = np.clip(y_pred, 0, None)  # clamp negatives to zero (if needed)

        y_test_val = y_test.values if not isinstance(y_test, np.ndarray) else y_test
        if "log" in target:
            y_test_inv = np.expm1(y_test_val)  # inverse transform ground truth
        else:
            y_test_inv = y_test_val  # no transform

    # Compute metrics on consistent scale
    r2 = r2_score(y_test_inv, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred)
    metrics[target] = {"R2": r2, "MAE": mae}

    # Plotting using y_test_inv and y_pred
    plt.errorbar(y_test_inv, y_pred, yerr=y_std, fmt='o', alpha=0.6, label="Predictions ±σ")
    plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--', label="Ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{target}: True vs Predicted (original scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{target}_plot.png"))
    plt.close()



    # === Save
    torch.save(model.state_dict(), os.path.join(output_dir, f"{target}_model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(output_dir, f"{target}_likelihood.pth"))
    joblib.dump(scaler, os.path.join(output_dir, f"{target}_scaler.pkl"))

# === Save Summary ===
pd.DataFrame(metrics).T.to_csv(os.path.join(output_dir, "performance_metrics.csv"))
print("\n✅ All surrogate models trained and saved.")

import os
import torch
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# === Load and preprocess dataset ===
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd", "ISE", "Overshoot", "SettlingTime", "RiseTime"])
df = df.sample(n=4000, random_state=42).reset_index(drop=True)

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

# Add log-transformed targets
df["ISE_log"] = np.log1p(df["ISE"])
df["SettlingTime_log"] = np.log1p(df["SettlingTime"])
df["RiseTime_log"] = np.log1p(df["RiseTime"])

features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
target_config = {
    "ISE_log": {"original": "ISE", "use_log": True},
    "Overshoot": {"original": "Overshoot", "use_log": False},
    "SettlingTime_log": {"original": "SettlingTime", "use_log": True},
    "RiseTime_log": {"original": "RiseTime", "use_log": True}
}

# === Fixed DGP Components ===
class ToyDeepGPHiddenLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing=32):
        # Create inducing points with correct input dimensionality
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        )
        
        # Store dimensions for reshaping
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DGPRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, n_layers=2):
        super().__init__()
        
        # Set hidden dimension (can be different from input)
        if hidden_dim is None:
            hidden_dim = max(4, input_dim // 2)  # Reasonable default
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create layers with proper dimensionality
        self.hidden_layers = torch.nn.ModuleList()
        
        # First hidden layer: input_dim -> hidden_dim
        if n_layers > 1:
            self.hidden_layers.append(
                ToyDeepGPHiddenLayer(input_dim, hidden_dim)
            )
            
            # Additional hidden layers: hidden_dim -> hidden_dim
            for _ in range(n_layers - 2):
                self.hidden_layers.append(
                    ToyDeepGPHiddenLayer(hidden_dim, hidden_dim)
                )
        
        # Output layer: hidden_dim -> 1 (or input_dim -> 1 if no hidden layers)
        final_input_dim = hidden_dim if n_layers > 1 else input_dim
        self.output_layer = ToyDeepGPHiddenLayer(final_input_dim, 1)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        # Pass through hidden layers
        current_x = x
        for layer in self.hidden_layers:
            # Sample from the GP distribution
            gp_dist = layer(current_x)
            current_x = gp_dist.rsample()
            
            # Ensure correct dimensionality for next layer
            if current_x.dim() == 1:
                current_x = current_x.unsqueeze(-1)
            
            # Reshape if needed to match expected hidden dimension
            if current_x.shape[-1] != self.hidden_dim and len(self.hidden_layers) > 1:
                # Simple linear projection to correct dimension
                if not hasattr(self, '_projection'):
                    self._projection = torch.nn.Linear(current_x.shape[-1], self.hidden_dim)
                current_x = self._projection(current_x)
        
        # Final output layer
        return self.output_layer(current_x)

# Alternative simpler DGP model for better stability
class SimpleDGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_inducing=64):
        # Create inducing points
        inducing_points = torch.randn(num_inducing, input_dim)
        
        # Initialize variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) + 
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# === Training ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGPSurrogate\surrogate_perf_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

final_metrics = {}

# Choose model type: 'simple' or 'deep'
model_type = 'simple'  # Change to 'deep' if you want to try the fixed DGP

for target, config in target_config.items():
    print(f"\n🚀 Training target: {target}")

    df_target = df.dropna(subset=features + [target] + [config["original"]])
    X = df_target[features].values.astype(np.float32)
    y = df_target[target].values.astype(np.float32)
    y_orig = df_target[config["original"]].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train_t, y_test_orig = train_test_split(
        X_scaled, y_orig, test_size=0.2, random_state=42)

    y_train_tensor = torch.tensor(np.log1p(y_train_t) if config["use_log"] else y_train_t, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # === DataLoader ===
    batch_size = min(128, len(X_train_tensor) // 4)  # Adaptive batch size
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    # === Model Selection ===
    if model_type == 'simple':
        model = SimpleDGPModel(input_dim=X.shape[1], num_inducing=min(64, len(X_train_tensor) // 2))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train_tensor))
    else:
        model = DGPRegressionModel(input_dim=X.shape[1], hidden_dim=4, n_layers=2)
        likelihood = model.likelihood
        mll = gpytorch.mlls.VariationalELBO(likelihood, model.output_layer, num_data=len(X_train_tensor))

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    # Training loop with better error handling
    for i in range(300):
        model.train()
        likelihood.train()
        total_loss = 0
        
        try:
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                if model_type == 'simple':
                    output = model(X_batch)
                else:
                    output = model(X_batch)
                
                loss = -mll(output, y_batch)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at iteration {i+1}")
                    continue
                    
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
        except RuntimeError as e:
            print(f"Training error at iteration {i+1}: {e}")
            break
            
        scheduler.step()
        
        if (i + 1) % 50 == 0:
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
            print(f"Iter {i+1}/300 - Avg Loss: {avg_loss:.4f}")

    # === Evaluation ===
    model.eval()
    likelihood.eval()
    
    try:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if model_type == 'simple':
                preds = likelihood(model(X_test_tensor))
            else:
                preds = likelihood(model(X_test_tensor))
                
            y_pred = preds.mean.numpy()
            y_std = preds.variance.sqrt().numpy()

            if config["use_log"]:
                y_pred_orig = np.expm1(y_pred)
                y_std_orig = (np.expm1(y_pred + y_std) - np.expm1(y_pred - y_std)) / 2
            else:
                y_pred_orig = y_pred
                y_std_orig = y_std

            y_pred_orig = np.clip(y_pred_orig, 0, None)

            r2 = r2_score(y_test_orig, y_pred_orig)
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            final_metrics[target] = {"R2": r2, "MAE": mae}

            print(f"📊 {target} → R²: {r2:.4f}, MAE: {mae:.4f}")

            # Save diagnostics
            df_diag = pd.DataFrame({
                "y_true": y_test_orig,
                "y_pred": y_pred_orig,
                "y_std": y_std_orig,
                "abs_error": np.abs(y_test_orig - y_pred_orig),
                "rel_error_%": 100 * np.abs(y_test_orig - y_pred_orig) / np.clip(y_test_orig, 1e-6, None)
            })
            df_diag.to_csv(os.path.join(output_dir, f"{target}_diagnostics.csv"), index=False)

            # Create plot
            plt.figure(figsize=(8, 6))
            plt.errorbar(y_test_orig, y_pred_orig, yerr=y_std_orig, fmt='o', alpha=0.6, capsize=3)
            minv, maxv = min(y_test_orig.min(), y_pred_orig.min()), max(y_test_orig.max(), y_pred_orig.max())
            plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.8, label='Perfect Prediction')
            plt.title(f"{target}: R²={r2:.3f}, MAE={mae:.3f}")
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{target}_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Save model components
            torch.save(model.state_dict(), os.path.join(output_dir, f"{target}_model.pth"))
            torch.save(likelihood.state_dict(), os.path.join(output_dir, f"{target}_likelihood.pth"))
            joblib.dump(scaler, os.path.join(output_dir, f"{target}_scaler.pkl"))
            
    except Exception as e:
        print(f"Evaluation error for {target}: {e}")
        final_metrics[target] = {"R2": float('nan'), "MAE": float('nan')}

# Save summary
summary_df = pd.DataFrame(final_metrics).T
summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"))
print("\n✅ All DGP models trained. Results saved to:")
print(output_dir)
print(summary_df)
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

# =========================
# ======= SETTINGS ========
# =========================
# Base directory for your repo (adjust as needed)
BASE = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning"
DATA_DIR = os.path.join(BASE, "src", "data")

CSV_EXT = os.path.join(DATA_DIR, "pid_dataset_pidtune_extended.csv")
CSV_OLD = os.path.join(DATA_DIR, "pid_dataset_pidtune.csv")  # fallback

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(BASE, "models", "DGPSurrogate", f"surrogate_perf_{timestamp}")
os.makedirs(OUT_DIR, exist_ok=True)

# Sampling / training
N_SAMPLE_CAP = 30000          # cap dataset rows for speed
TEST_SIZE = 0.2
BATCH_FRACTION = 0.25         # batch_size ≈ len(X_train) * BATCH_FRACTION (capped)
MAX_ITERS = 300
LR = 1e-2
STEP_SIZE = 100
GAMMA = 0.8
MODEL_TYPE = "simple"         # 'simple' (recommended) or 'deep'
NUM_INDUCING = 64             # inducing points for simple model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# ======= LOADING =========
# =========================
def load_dataset():
    if os.path.exists(CSV_EXT):
        print(f"Loading extended dataset: {CSV_EXT}")
        df = pd.read_csv(CSV_EXT)
        is_extended = True
    elif os.path.exists(CSV_OLD):
        print(f"Extended not found; loading legacy dataset: {CSV_OLD}")
        df = pd.read_csv(CSV_OLD)
        is_extended = False
    else:
        raise FileNotFoundError("No dataset found. Expected extended or legacy CSV in src/data/")
    return df, is_extended

df, is_extended = load_dataset()

# =========================
# ===== PREPROCESS =========
# =========================
# Keep rows with valid controller + metrics
required_cols = ["K", "L", "Kp", "Ki", "Kd", "ISE", "Overshoot", "SettlingTime", "RiseTime"]
# Legacy CSV may not have L; if missing, create L=0
for c in ["L"]:
    if c not in df.columns:
        df[c] = 0.0

df = df.dropna(subset=[c for c in required_cols if c in df.columns])

# Ensure new columns exist, then fill and mark missingness
for c in ["T1", "T2", "w0", "zeta", "Tchar"]:
    if c not in df.columns:
        df[c] = np.nan
    df[f"isnan_{c}"] = df[c].isna().astype(np.float32)
    df[c] = df[c].fillna(0.0)

# Family one-hots (if available); else synthesize a reasonable mapping from SystemType
fam_cols = ["fam_PT1PT2_existing", "fam_PT2_osc", "fam_IT1", "fam_P"]
if "Family" in df.columns:
    fam_dum = pd.get_dummies(df["Family"], prefix="fam")
    for fc in fam_cols:
        if fc not in fam_dum.columns:
            fam_dum[fc] = 0
    df = pd.concat([df, fam_dum[fam_cols]], axis=1)
else:
    # Legacy fallback: infer coarse families from SystemType strings if present
    for fc in fam_cols:
        df[fc] = 0
    if "SystemType" in df.columns:
        st = df["SystemType"].astype(str)
        df.loc[st.str.contains("PT2osc", case=False, na=False), "fam_PT2_osc"] = 1
        df.loc[st.str.contains("IT1", case=False, na=False), "fam_IT1"] = 1
        df.loc[st.str.contains("P", case=False, na=False), "fam_P"] = 1
        # default others to PT1/PT2
        mask_any = (df[fam_cols].sum(axis=1) > 0)
        df.loc[~mask_any, "fam_PT1PT2_existing"] = 1
    else:
        # If truly nothing: assume legacy data are PT1/PT2
        df["fam_PT1PT2_existing"] = 1

# Optional: balanced subsample by family (stratified) if dataset is huge
if len(df) > N_SAMPLE_CAP:
    # Stratify by dominant family dummy (argmax over fam cols)
    fam_idx = df[fam_cols].values.argmax(axis=1)
    df["_fam_idx"] = fam_idx
    # Approx equal per class
    per_class = max(1, N_SAMPLE_CAP // max(1, len(np.unique(fam_idx))))
    sampled = []
    for k in np.unique(fam_idx):
        part = df[df["_fam_idx"] == k]
        sampled.append(part.sample(n=min(per_class, len(part)), random_state=42))
    df = pd.concat(sampled).sample(frac=1.0, random_state=42).drop(columns=["_fam_idx"]).reset_index(drop=True)
else:
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle

print(f"Dataset after load/shuffle: {df.shape}")

# === Outlier / sanity filters (tune if needed) ===
df = df[(df["ISE"] < 50) & (df["ISE"] > 1e-3)]
df = df[(df["SettlingTime"] < 500) & (df["SettlingTime"] > 1e-2)]
df = df[(df["RiseTime"] < 300) & (df["RiseTime"] > 1e-2)]
df = df[df["Overshoot"] < 200]
df = df[(df["Kp"] < 50) & (df["Kp"] > 0.3)]
df = df[(df["Ki"] < 50) & (df["Kd"] < 50)]

print(f"After filtering: {df.shape}")

# === Feature engineering ===
def log_pos(x, eps=1e-8):
    return np.log(np.clip(x, eps, None))

def log1p_pos(x):
    return np.log1p(np.clip(x, 0, None))

df["logK"]      = log_pos(df["K"])
df["logL1p"]    = log1p_pos(df["L"])
df["logT1p"]    = log1p_pos(df["T1"])
df["logT2p"]    = log1p_pos(df["T2"])
df["logw0p"]    = log1p_pos(df["w0"])
df["logTcharp"] = log1p_pos(df["Tchar"])

# Targets (linear; set "use_log": True if you want log-target training)
target_config = {
    "ISE":          {"use_log": False},
    "Overshoot":    {"use_log": False},
    "SettlingTime": {"use_log": False},
    "RiseTime":     {"use_log": False},
}

# Final feature set
FEATURES = [
    # controller
    "Kp", "Ki", "Kd",
    # plant canonical (log-scales)
    "logK", "logL1p", "logT1p", "logT2p", "logw0p", "logTcharp",
    # missingness indicators (so zero-fill is not ambiguous)
    "isnan_T1", "isnan_T2", "isnan_w0", "isnan_zeta", "isnan_Tchar",
    # damping (linear)
    "zeta",
    # family
    *fam_cols,
]

missing = [c for c in FEATURES if c not in df.columns]
if len(missing):
    raise ValueError(f"Missing required features: {missing}")

# =========================
# ======= MODELS ==========
# =========================
class SimpleDGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_inducing=64):
        inducing_points = torch.randn(num_inducing, input_dim)
        q = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        vs = gpytorch.variational.VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) +
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ToyDeepGPHiddenLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing=128):
        inducing_points = torch.randn(num_inducing, input_dims)
        q = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        vs = gpytorch.variational.VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        )
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DGPRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, n_layers=2):
        super().__init__()
        hidden_dim = max(4, input_dim // 2) if hidden_dim is None else hidden_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.hidden_layers = torch.nn.ModuleList()
        if n_layers > 1:
            self.hidden_layers.append(ToyDeepGPHiddenLayer(input_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.hidden_layers.append(ToyDeepGPHiddenLayer(hidden_dim, hidden_dim))

        final_input_dim = hidden_dim if n_layers > 1 else input_dim
        self.output_layer = ToyDeepGPHiddenLayer(final_input_dim, 1)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._projection = None

    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            dist = layer(h)
            h = dist.rsample()
            if h.dim() == 1:
                h = h.unsqueeze(-1)
            if h.shape[-1] != self.hidden_dim and len(self.hidden_layers) > 1:
                if self._projection is None:
                    self._projection = torch.nn.Linear(h.shape[-1], self.hidden_dim).to(h.device)
                h = self._projection(h)
        return self.output_layer(h)

# =========================
# ======= TRAINING ========
# =========================
def train_one_target(target_name, df):
    print(f"\n🚀 Training target: {target_name}")

    # Assemble matrices
    df_t = df.dropna(subset=FEATURES + [target_name])
    X = df_t[FEATURES].values.astype(np.float32)
    y = df_t[target_name].values.astype(np.float32)

    # scaler per target
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=TEST_SIZE, random_state=42,
        stratify=df_t[fam_cols].values.argmax(axis=1) if df_t[fam_cols].values.ndim == 2 else None
    )

    # Optionally log-target
    use_log = target_config[target_name]["use_log"]
    y_train_t = np.log1p(y_train) if use_log else y_train

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_t, dtype=torch.float32, device=DEVICE)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32,  device=DEVICE)

    # DataLoader
    bs = max(16, min(128, int(len(X_train_t) * BATCH_FRACTION)))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=bs, shuffle=True)

    # Model + likelihood
    if MODEL_TYPE == "simple":
        model = SimpleDGPModel(input_dim=X.shape[1], num_inducing=min(NUM_INDUCING, max(8, len(X_train_t)//2))).to(DEVICE)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train_t))
    else:
        model = DGPRegressionModel(input_dim=X.shape[1], hidden_dim=8, n_layers=2).to(DEVICE)
        likelihood = model.likelihood.to(DEVICE)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model.output_layer, num_data=len(X_train_t))

    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    for it in range(1, MAX_ITERS + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = -mll(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        sch.step()
        if it % 50 == 0:
            denom = max(1, len(train_loader))
            print(f"Iter {it:3d}/{MAX_ITERS}  Avg Loss: {total_loss/denom:.4f}")

    # ===== Evaluation =====
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test_t))
        y_pred = preds.mean.detach().cpu().numpy()
        y_std  = preds.variance.sqrt().detach().cpu().numpy()

    if use_log:
        y_pred_orig = np.expm1(y_pred)
        # approx symmetric std mapping in log-space
        y_std_orig = (np.expm1(y_pred + y_std) - np.expm1(y_pred - y_std)) / 2
    else:
        y_pred_orig = y_pred
        y_std_orig = y_std

    y_pred_orig = np.clip(y_pred_orig, 0, None)
    r2 = r2_score(y_test, y_pred_orig)
    mae = mean_absolute_error(y_test, y_pred_orig)
    print(f"📊 {target_name} → R²: {r2:.4f}, MAE: {mae:.4f}")

    # Diagnostics
    diag = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_orig,
        "y_std": y_std_orig,
        "abs_error": np.abs(y_test - y_pred_orig),
        "rel_error_%": 100*np.abs(y_test - y_pred_orig)/np.clip(y_test, 1e-6, None)
    })
    diag.to_csv(os.path.join(OUT_DIR, f"{target_name}_diagnostics.csv"), index=False)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.errorbar(y_test, y_pred_orig, yerr=y_std_orig, fmt='o', alpha=0.55, capsize=2)
    mn, mx = min(y_test.min(), y_pred_orig.min()), max(y_test.max(), y_pred_orig.max())
    plt.plot([mn, mx], [mn, mx], 'r--', alpha=0.8, label='Perfect Prediction')
    plt.title(f"{target_name}: R²={r2:.3f}, MAE={mae:.3f}")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{target_name}_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save weights + scaler
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{target_name}_model.pth"))
    torch.save(likelihood.state_dict(), os.path.join(OUT_DIR, f"{target_name}_likelihood.pth"))
    joblib.dump(scaler, os.path.join(OUT_DIR, f"{target_name}_scaler.pkl"))

    return {"R2": r2, "MAE": mae}

# =========================
# ======= RUN ALL =========
# =========================
final_metrics = {}
for tgt in target_config.keys():
    final_metrics[tgt] = train_one_target(tgt, df)

# Save summary
summary_df = pd.DataFrame(final_metrics).T
summary_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
print("\n✅ All models trained. Results saved to:")
print(OUT_DIR)
print(summary_df)
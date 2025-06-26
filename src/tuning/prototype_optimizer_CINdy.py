## === CINdy-based PID Optimizer Using Surrogate Model ===
# Input: G(s) = [K, T1, T2] and trained surrogate model
# Output: Optimized [Kp, Ki, Kd] minimizing surrogate-predicted cost
import os
import nevergrad as ng
import torch
import joblib
import pandas as pd
import numpy as np
from typing import Dict
import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.spatial.distance import pdist, squareform
import control


evaluated_controllers = []


# === Define model architecture (same as training) ===
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
    
class SimpleDGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_inducing=64):
        inducing_points = torch.randn(num_inducing, input_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) +
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# === Load trained surrogates ===
MODEL_DIR = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGPSurrogate\surrogate_perf_20250626_134644"

#def load_model(param: str):
    #input_dim = 6
    #model = VariationalGP(input_dim)
    #model.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_model.pth"))
    #model.eval()

    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #likelihood.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_likelihood.pth"))
    #likelihood.eval()

    #scaler = joblib.load(f"{MODEL_DIR}/{param}_scaler.pkl")
    #return model, likelihood, scaler

def load_model(param: str):
    input_dim = 6
    model = SimpleDGPModel(input_dim=input_dim, num_inducing=64)  # match training
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_model.pth", map_location='cpu'))
    model.eval()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_likelihood.pth", map_location='cpu'))
    likelihood.eval()

    scaler = joblib.load(f"{MODEL_DIR}/{param}_scaler.pkl")
    return model, likelihood, scaler


print("\U0001F4E6 Loading surrogate models...")
models = {}
for target in ["ISE_log", "Overshoot", "SettlingTime_log", "RiseTime_log"]:
    print(f"   └─ {target}")
    models[target] = load_model(target)

# === Surrogate prediction ===
def predict_metrics(K, T1, T2, Kp, Ki, Kd) -> Dict[str, float]:
    X = np.array([[K, T1, T2, Kp, Ki, Kd]], dtype=np.float32)
    preds = {}
    print(f"\n[🔍 predict_metrics] Input: K={K}, T1={T1}, T2={T2}, Kp={Kp}, Ki={Ki}, Kd={Kd}")
    for target, (model, likelihood, scaler) in models.items():
        try:
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                output = likelihood(model(X_tensor))
                mean = output.mean.item()
                #if "log" in target:
                    #mean = np.power(10, mean) - 1
                preds[target.replace("_log", "")] = mean
                print(f"[✔️ {target}] Prediction: {mean:.6f}")
                print(f"[📊 Raw output.mean] {target}: {output.mean.item():.4f}")

        except Exception as e:
            print(f"[❌ {target}] Error during prediction: {e}")
            preds[target.replace("_log", "")] = np.nan
    return preds


# === Cost function ===
def surrogate_cost(params, plant, weights):
    Kp, Ki, Kd = params
    K, T1, T2 = plant

    # === Stability rejection ===
    if not is_stable(K, T1, T2, Kp, Ki, Kd):
        print(f"🚫 Unstable PID rejected: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")
        return 1e6  # Penalize unstable controllers

    # === Predict surrogate metrics ===
    metrics = predict_metrics(K, T1, T2, Kp, Ki, Kd)

    # === Optional logic to reject implausible combinations ===
    settling = metrics.get("SettlingTime", np.nan)
    rise = metrics.get("RiseTime", np.nan)
    if settling < rise:
        print(f"[❌ Reject] Implausible timing: SettlingTime={settling:.3f} < RiseTime={rise:.3f}")
        return 1e6

    cost = (
        weights["ISE"] * metrics["ISE"] +
        weights["Overshoot"] * metrics["Overshoot"] +
        weights["SettlingTime"] * metrics["SettlingTime"] +
        weights["RiseTime"] * metrics["RiseTime"]
    )

    if cost <= 1:
        print(f"[❌ Skipping] Cost={cost:.3f} too low, ignoring for training signal")
        return 1e6

    print(f"\n[🧮 surrogate_cost] Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} → Cost={cost:.6f}")
    return cost



# === Optimizer with fallback options ===
def optimize_pid(plant, weights, budget=1000, optimizer_name="DE"):
    print(f"\U0001F680 Starting optimization with {optimizer_name}...")
    
    parametrization = ng.p.Instrumentation(
        Kp=ng.p.Scalar(init=1.0).set_bounds(0.30, 10.0),
        Ki=ng.p.Scalar(init=0.1).set_bounds(0.1, 5.0),
        Kd=ng.p.Scalar(init=0.01).set_bounds(0.0, 5.0),
    )

    optimizers_to_try = [
        optimizer_name,
        "DE",
        "PSO",
        "OnePlusOne",
        "RandomSearch"
    ]

    print_counter = 0  # ⬅️ Counter for debug prints (limit to 5)

    for opt_name in optimizers_to_try:
        try:
            print(f"   Trying {opt_name}...")
            if opt_name in ng.optimizers.registry:
                optimizer_cls = ng.optimizers.registry[opt_name]
            else:
                print(f"   {opt_name} not found, skipping...")
                continue

            optimizer = optimizer_cls(parametrization=parametrization, budget=budget)

            def cost_fn(Kp, Ki, Kd):
                nonlocal print_counter
                cost = surrogate_cost([Kp, Ki, Kd], plant, weights)
                
                # Log controller evaluation
                evaluated_controllers.append({
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "cost": cost
                })
                
                if print_counter < 5:
                    print(f"[🧪 Eval {print_counter + 1}] Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} → Cost={cost:.6f}")
                    print_counter += 1
                return cost


            test_candidate = optimizer.ask()
            optimizer.tell(test_candidate, cost_fn(**test_candidate.kwargs))

            print(f"   ✅ {opt_name} works! Running optimization...")
            recommendation = optimizer.minimize(cost_fn)
            result = recommendation.kwargs

            print("✅ Optimization complete.")
            return result, cost_fn(**result), opt_name

        except Exception as e:
            print(f"   ❌ {opt_name} failed: {str(e)}")
            continue

    raise RuntimeError("All optimizers failed! Please check your NumPy/CMA compatibility.")


# === Alternative scipy-based optimizer ===
def optimize_pid_scipy(plant, weights, budget=100):
    """Fallback optimizer using scipy.optimize"""
    try:
        from scipy.optimize import differential_evolution
        print("\U0001F680 Using scipy differential evolution...")
        
        def cost_fn_scipy(params):
            Kp, Ki, Kd = params
            return surrogate_cost([Kp, Ki, Kd], plant, weights)
        
        bounds = [(0.1, 1.0), (0.001, 3), (0.0, 3)]  # Kp, Ki, Kd bounds
        
        result = differential_evolution(
            cost_fn_scipy, 
            bounds, 
            maxiter=budget//10,  # Adjust iterations based on budget
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            best_params = {"Kp": result.x[0], "Ki": result.x[1], "Kd": result.x[2]}
            print("✅ Scipy optimization complete.")
            return best_params, result.fun, "scipy_DE"
        else:
            raise RuntimeError(f"Scipy optimization failed: {result.message}")
            
    except ImportError:
        raise RuntimeError("Scipy not available and nevergrad optimizers failed")

def is_stable(K, T1, T2, Kp, Ki, Kd) -> bool:
    """
    Check if a PID-controlled system is stable.
    
    Plant: G(s) = K / ((T1*s + 1) * (T2*s + 1))
    Controller: C(s) = Kd*s + Kp + Ki/s
    
    Returns True if all closed-loop poles have negative real parts.
    """
    try:
        # Input validation
        if K <= 0:
            print("❌ Plant gain K must be positive")
            return False
        
        if T1 <= 0:
            print("❌ Time constant T1 must be positive")
            return False
        
        if T2 < 0:
            print("❌ Time constant T2 must be non-negative")
            return False
        
        # === 1. Plant G(s) ===
        if T2 == 0 or abs(T2) < 1e-12:  # Handle T2 = 0 case
            # First-order system: G(s) = K/(T1*s + 1)
            den = [T1, 1]
        else:
            # Second-order system: G(s) = K/((T1*s + 1)(T2*s + 1))
            den = np.polymul([T1, 1], [T2, 1])
        
        G = control.TransferFunction([K], den)
        print(f"Plant G(s): {G}")

        # === 2. PID Controller C(s) ===
        # C(s) = (Kd*s² + Kp*s + Ki) / s
        if abs(Ki) < 1e-12 and abs(Kd) < 1e-12:
            # Pure proportional controller
            C = control.TransferFunction([Kp], [1])
        elif abs(Kd) < 1e-12:
            # PI controller: C(s) = (Kp*s + Ki) / s
            C = control.TransferFunction([Kp, Ki], [1, 0])
        else:
            # Full PID controller
            C = control.TransferFunction([Kd, Kp, Ki], [1, 0])
        
        print(f"Controller C(s): {C}")

        # === 3. Closed-loop Transfer Function ===
        # T(s) = C(s)*G(s) / (1 + C(s)*G(s))
        open_loop = C * G
        print(f"Open-loop C(s)*G(s): {open_loop}")
        
        sys_cl = control.feedback(open_loop, 1)
        print(f"Closed-loop system: {sys_cl}")

        # === 4. Get poles and check stability ===
        poles = sys_cl.poles()  # Note: .poles() not .pole()
        print(f"Closed-loop poles: {poles}")
        
        # Check if all poles have negative real parts
        stable = np.all(np.real(poles) < 0)
        
        if stable:
            print("✅ System is stable")
        else:
            print("❌ System is unstable")
            unstable_poles = poles[np.real(poles) >= 0]
            print(f"Unstable poles: {unstable_poles}")
        
        return stable

    except Exception as e:
        print(f"❌ Stability check failed: {e}")
        print(f"Parameters: K={K}, T1={T1}, T2={T2}, Kp={Kp}, Ki={Ki}, Kd={Kd}")
        return False



def diverse_top_controllers(df, top_n=5, min_dist=0.5):
    df_sorted = df.sort_values("cost").copy()
    selected = []

    for _, row in df_sorted.iterrows():
        gains = np.array([row["Kp"], row["Ki"], row["Kd"]])
        if all(np.linalg.norm(gains - np.array([s["Kp"], s["Ki"], s["Kd"]])) > min_dist for s in selected):
            selected.append(row)
        if len(selected) == top_n:
            break

    return pd.DataFrame(selected)


# === Example usage ===
if __name__ == "__main__":
    plant = (1.87, 4.0, 2.0)  # K, T1, T2
    weights = {"ISE": 1.10, "Overshoot": 0.5, "SettlingTime": 1.2, "RiseTime": 1.1}
    print(weights)  # Should show non-zero weights
    print(models.keys())  # Should match: ISE_log, Overshoot, ...
    print(models["ISE_log"][0])  # Print the model to confirm uniqueness

    try:
        # Try nevergrad optimizers first
        best_pid, best_cost, used_optimizer = optimize_pid(plant, weights)
        print(f"\n\U0001F9E0 Best PID (using {used_optimizer}):", best_pid)
    except:
        # Fallback to scipy
        print("Nevergrad failed, trying scipy...")
        best_pid, best_cost, used_optimizer = optimize_pid_scipy(plant, weights)
        print(f"\n\U0001F9E0 Best PID (using {used_optimizer}):", best_pid)
    
    print("\U0001F50D Predicted Cost:", best_cost)

    predicted = predict_metrics(plant[0], plant[1], plant[2], best_pid["Kp"], best_pid["Ki"], best_pid["Kd"])
    print("\n📈 Predicted Performance Metrics:")
    for metric, value in predicted.items():
        if metric in ["ISE", "Overshoot", "SettlingTime", "RiseTime"]:
            original_value = np.expm1(value)
            print(f"{metric}: {value:.4f} → {original_value:.3f} s")
        else:
            print(f"{metric}: {value:.4f}")
            # Convert to DataFrame
    df_eval = pd.DataFrame(evaluated_controllers)

    # === Output directory ===
    plot_dir = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\results\plot"
    os.makedirs(plot_dir, exist_ok=True)

    # === Convert to DataFrame ===
    df_eval = pd.DataFrame(evaluated_controllers)

    # === 1. Histogram of Costs ===
    if not df_eval.empty and "cost" in df_eval.columns:
        fig = plt.figure()
        plt.hist(df_eval["cost"], bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("Cost")
        plt.ylabel("Frequency")
        plt.title("Distribution of Evaluated Costs")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "cost_hist.png"))
        plt.close(fig)
    else:
        print("⚠️ No valid data for histogram.")
    matplotlib.use("Agg")  # Prevent GUI locking if running headless or in VSCode

    # === 2. Pairplot (Seaborn) ===
    try:
        # Clean and filter
        df_clean = df_eval[["Kp", "Ki", "Kd", "cost"]].dropna()
        df_clean = df_clean[df_clean["cost"] < 1e5]  # remove cost outliers, adjust threshold as needed

        if len(df_clean) > 500:
            df_clean = df_clean.sample(n=500, random_state=42)

        print(f"🌀 Generating seaborn pairplot on {len(df_clean)} filtered samples...")

        # Plot
        sns_plot = sns.pairplot(df_clean, diag_kind="kde", corner=True, plot_kws={'alpha': 0.6})
        sns_plot.fig.suptitle("Gain vs Cost Pairplot (Filtered)", y=1.02)

        sns_plot.savefig(os.path.join(plot_dir, "pair_scatter_matrix_filtered.png"))
        plt.close()

        print("✅ Filtered pairplot saved.")
    except Exception as e:
        print(f"❌ Seaborn pairplot failed: {e}")

    # === 3. 3D Scatter: Kp, Ki vs Cost ===
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df_eval["Kp"], df_eval["Ki"], df_eval["cost"], c='blue', alpha=0.6)
        ax.set_xlabel("Kp")
        ax.set_ylabel("Ki")
        ax.set_zlabel("Cost")
        plt.title("3D Scatter: Kp, Ki, Cost")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "3dscatter.png"))
        plt.close(fig)
    except Exception as e:
        print(f"❌ 3D scatter plot failed: {e}")

    # === 4. Radar Plot: Predicted Performance ===
    try:
        labels = list(predicted.keys())
        values = list(predicted.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw={'polar': True})
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title("Predicted Performance Radar Plot")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "Radar.png"))
        plt.close(fig)
    except Exception as e:
        print(f"❌ Radar plot failed: {e}")

    # === 5. Top 5 Controllers Table ===
    # Clean up and get top diverse set
    df_unique = (
        df_eval
        .round({"Kp": 2, "Ki": 2, "Kd": 2})
        .drop_duplicates(subset=["Kp", "Ki", "Kd"])
        .dropna(subset=["Kp", "Ki", "Kd", "cost"])
    )

    top5_unique = diverse_top_controllers(df_unique, top_n=5, min_dist=3.5)

    print("\n🏆 Top 5 Diverse Controllers:")
    print(top5_unique[["Kp", "Ki", "Kd", "cost"]].round(4))

    print("\n📊 Full Metrics for Top 5 Controllers:")
    for i, row in top5_unique.iterrows():
        Kp, Ki, Kd = row["Kp"], row["Ki"], row["Kd"]
        metrics = predict_metrics(plant[0], plant[1], plant[2], Kp, Ki, Kd)

        print(f"\n🔧 Controller #{i + 1}: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}, Cost={row['cost']:.4f}")
        for metric, val in metrics.items():
            if metric in ["SettlingTime", "RiseTime"]:
                val_inv = np.expm1(val)
                print(f"   {metric}: {val:.4f} → {val_inv:.3f} s")
            else:
                print(f"   {metric}: {val:.4f}")


# === Diagnosis and Fix for Settling Time Underprediction ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

def diagnose_settling_time_bias(df, model, likelihood, scaler, target_config, output_dir):
    """
    Comprehensive diagnosis of settling time prediction bias
    """
    print("🔍 DIAGNOSING SETTLING TIME PREDICTION BIAS")
    print("=" * 50)
    
    # 1. Check for systematic bias across different ranges
    target = "SettlingTime_log"
    config = target_config[target]
    
    # Get predictions for full dataset
    features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
    X = df[features].values.astype(np.float32)
    y_true_orig = df[config["original"]].values
    
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_tensor))
        y_pred_log = preds.mean.numpy()
        y_pred_orig = np.expm1(y_pred_log)
        y_pred_orig = np.clip(y_pred_orig, 0, None)
    
    # Create comprehensive analysis dataframe
    analysis_df = pd.DataFrame({
        'true_settling_time': y_true_orig,
        'pred_settling_time': y_pred_orig,
        'absolute_error': np.abs(y_true_orig - y_pred_orig),
        'relative_error': (y_pred_orig - y_true_orig) / y_true_orig * 100,  # Negative = underprediction
        'K': df['K'].values,
        'T1': df['T1'].values,
        'T2': df['T2'].values,
        'Kp': df['Kp'].values,
        'Ki': df['Ki'].values,
        'Kd': df['Kd'].values
    })
    
    # 2. Analyze bias by settling time ranges
    print("\n📊 BIAS ANALYSIS BY SETTLING TIME RANGES:")
    settling_ranges = [
        (0, 10, "Very Fast"),
        (10, 50, "Fast"), 
        (50, 100, "Medium"),
        (100, 200, "Slow"),
        (200, 500, "Very Slow")
    ]
    
    bias_by_range = []
    for min_val, max_val, label in settling_ranges:
        mask = (analysis_df['true_settling_time'] >= min_val) & (analysis_df['true_settling_time'] < max_val)
        if mask.sum() > 0:
            subset = analysis_df[mask]
            mean_bias = subset['relative_error'].mean()
            median_bias = subset['relative_error'].median()
            count = len(subset)
            bias_by_range.append({
                'Range': label,
                'Count': count,
                'Mean_Bias_%': mean_bias,
                'Median_Bias_%': median_bias,
                'Mean_True': subset['true_settling_time'].mean(),
                'Mean_Pred': subset['pred_settling_time'].mean()
            })
            print(f"{label:12} ({min_val:3.0f}-{max_val:3.0f}): "
                  f"Count={count:3d}, Mean_Bias={mean_bias:6.1f}%, "
                  f"True_Avg={subset['true_settling_time'].mean():6.1f}, "
                  f"Pred_Avg={subset['pred_settling_time'].mean():6.1f}")
    
    bias_df = pd.DataFrame(bias_by_range)
    bias_df.to_csv(f"{output_dir}/settling_time_bias_analysis.csv", index=False)
    
    # 3. Check for parameter-dependent bias
    print("\n🔧 PARAMETER-DEPENDENT BIAS ANALYSIS:")
    param_correlations = {}
    for param in ['K', 'T1', 'T2', 'Kp', 'Ki', 'Kd']:
        corr = np.corrcoef(analysis_df[param], analysis_df['relative_error'])[0, 1]
        param_correlations[param] = corr
        print(f"{param:3s} correlation with bias: {corr:6.3f}")
    
    # 4. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: True vs Predicted with bias coloring
    scatter = axes[0, 0].scatter(analysis_df['true_settling_time'], 
                                analysis_df['pred_settling_time'],
                                c=analysis_df['relative_error'], 
                                cmap='RdBu_r', alpha=0.6, s=30)
    axes[0, 0].plot([0, analysis_df['true_settling_time'].max()], 
                    [0, analysis_df['true_settling_time'].max()], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('True Settling Time')
    axes[0, 0].set_ylabel('Predicted Settling Time')
    axes[0, 0].set_title('Predictions Colored by Bias (%)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Relative Error (%)')
    
    # Plot 2: Bias distribution
    axes[0, 1].hist(analysis_df['relative_error'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Bias')
    axes[0, 1].axvline(analysis_df['relative_error'].mean(), color='orange', 
                      linestyle='-', alpha=0.7, label=f'Mean: {analysis_df["relative_error"].mean():.1f}%')
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Bias')
    axes[0, 1].legend()
    
    # Plot 3: Bias vs True Values
    axes[0, 2].scatter(analysis_df['true_settling_time'], analysis_df['relative_error'], alpha=0.6)
    axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('True Settling Time')
    axes[0, 2].set_ylabel('Relative Error (%)')
    axes[0, 2].set_title('Bias vs True Values')
    
    # Plot 4: Parameter correlations with bias
    params = list(param_correlations.keys())
    corrs = list(param_correlations.values())
    colors = ['red' if abs(c) > 0.2 else 'blue' for c in corrs]
    axes[1, 0].bar(params, corrs, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_ylabel('Correlation with Bias')
    axes[1, 0].set_title('Parameter Correlation with Bias')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Residuals vs Fitted
    axes[1, 1].scatter(analysis_df['pred_settling_time'], 
                      analysis_df['true_settling_time'] - analysis_df['pred_settling_time'], alpha=0.6)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Settling Time')
    axes[1, 1].set_ylabel('Residuals (True - Pred)')
    axes[1, 1].set_title('Residual Analysis')
    
    # Plot 6: Bias by ranges
    if len(bias_df) > 0:
        axes[1, 2].bar(bias_df['Range'], bias_df['Mean_Bias_%'], alpha=0.7)
        axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].set_ylabel('Mean Bias (%)')
        axes[1, 2].set_title('Bias by Settling Time Ranges')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/settling_time_bias_diagnosis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary statistics
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"Mean Relative Error: {analysis_df['relative_error'].mean():.2f}%")
    print(f"Median Relative Error: {analysis_df['relative_error'].median():.2f}%")
    print(f"Std of Relative Error: {analysis_df['relative_error'].std():.2f}%")
    print(f"% of Underpredictions: {(analysis_df['relative_error'] < 0).mean() * 100:.1f}%")
    print(f"R²: {r2_score(analysis_df['true_settling_time'], analysis_df['pred_settling_time']):.4f}")
    
    return analysis_df, bias_df


def fix_settling_time_bias(df, features, target_config, output_dir):
    """
    Multiple strategies to fix settling time underprediction bias
    """
    print("\n🔧 IMPLEMENTING BIAS CORRECTION STRATEGIES")
    print("=" * 50)
    
    # Strategy 1: Bias-corrected log transform
    print("\n1. Trying bias-corrected log transform...")
    
    # Find optimal bias correction
    settling_orig = df["SettlingTime_orig"].values
    settling_log_original = np.log1p(settling_orig)
    
    # Try different bias corrections
    bias_corrections = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    best_correction = 0
    best_symmetry = float('inf')
    
    for correction in bias_corrections:
        log_corrected = np.log1p(settling_orig + correction)
        # Measure symmetry of distribution
        symmetry = abs(np.percentile(log_corrected, 75) + np.percentile(log_corrected, 25) - 
                      2 * np.percentile(log_corrected, 50))
        if symmetry < best_symmetry:
            best_symmetry = symmetry
            best_correction = correction
    
    print(f"Best bias correction: {best_correction}")
    df["SettlingTime_log_corrected"] = np.log1p(df["SettlingTime_orig"] + best_correction)
    
    # Strategy 2: Square root transform (alternative to log)
    print("\n2. Trying square root transform...")
    df["SettlingTime_sqrt"] = np.sqrt(df["SettlingTime_orig"])
    
    # Strategy 3: Box-Cox-like transform
    print("\n3. Trying Box-Cox-like transform...")
    lambda_param = 0.3  # Can be optimized
    df["SettlingTime_boxcox"] = (df["SettlingTime_orig"]**lambda_param - 1) / lambda_param
    
    # Strategy 4: Quantile transform
    print("\n4. Trying rank-based transform...")
    from scipy.stats import rankdata
    ranks = rankdata(df["SettlingTime_orig"])
    df["SettlingTime_rank"] = (ranks - 1) / (len(ranks) - 1)  # Normalize to [0,1]
    
    # Updated target configuration
    extended_target_config = {
        "SettlingTime_log": {"original": "SettlingTime_orig", "use_log": True, "inverse": lambda x: np.expm1(x)},
        "SettlingTime_log_corrected": {"original": "SettlingTime_orig", "use_log": True, 
                                     "inverse": lambda x: np.expm1(x) - best_correction},
        "SettlingTime_sqrt": {"original": "SettlingTime_orig", "use_log": False, 
                             "inverse": lambda x: x**2},
        "SettlingTime_boxcox": {"original": "SettlingTime_orig", "use_log": False, 
                               "inverse": lambda x: (x * lambda_param + 1)**(1/lambda_param)},
        "SettlingTime_rank": {"original": "SettlingTime_orig", "use_log": False, 
                             "inverse": lambda x: np.interp(x, 
                                                           np.linspace(0, 1, len(df)), 
                                                           np.sort(df["SettlingTime_orig"]))}
    }
    
    return df, extended_target_config


    def train_improved_settling_time_model(df, features, target_name, config, output_dir):
        """
        Train settling time model with bias correction techniques
        """
        print(f"\n🚀 Training improved model for {target_name}")
        
        # Data preparation
        df_target = df.dropna(subset=features + [target_name] + [config["original"]])
        X = df_target[features].values.astype(np.float32)
        y = df_target[target_name].values.astype(np.float32)
        y_orig = df_target[config["original"]].values.astype(np.float32)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        train_idx, test_idx = train_test_split(range(len(df_target)), test_size=0.2, random_state=42)
        
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_test = X_scaled[test_idx]
        y_test = y[test_idx]
        y_test_orig = y_orig[test_idx]
        
        # Enhanced GP model with better kernel
        class EnhancedVariationalGP(gpytorch.models.ApproximateGP):
            def __init__(self, input_dim):
                inducing_points = torch.randn(256, input_dim)  # More inducing points
                variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                
                # Better mean function
                self.mean_module = gpytorch.means.LinearMean(input_dim)
                
                # Enhanced kernel with multiple components
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel() + 
                    gpytorch.kernels.MaternKernel(nu=2.5) +
                    gpytorch.kernels.LinearKernel()
                )

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)
        
        # Train model
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        model = EnhancedVariationalGP(X.shape[1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Enhanced training with learning rate schedule
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=0.01)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
        
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))
        
        for i in range(500):  # More training iterations
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = -mll(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (i + 1) % 100 == 0:
                print(f"Iter {i+1}/500 - Loss: {loss.item():.4f}")
        
        # Prediction with bias correction
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = likelihood(model(X_test_tensor))
            y_pred_transformed = preds.mean.numpy()
            
            # Apply inverse transform
            y_pred_orig = config["inverse"](y_pred_transformed)
            y_pred_orig = np.clip(y_pred_orig, 0, None)
            
            # Additional bias correction based on training data
            training_preds = likelihood(model(X_train_tensor)).mean.numpy()
            training_pred_orig = config["inverse"](training_preds)
            training_pred_orig = np.clip(training_pred_orig, 0, None)
            
            # Calculate systematic bias on training data
            training_bias = np.mean(y_orig[train_idx] - training_pred_orig)
            print(f"Detected training bias: {training_bias:.3f}")
            
            # Apply bias correction to test predictions
            y_pred_orig_corrected = y_pred_orig + training_bias
            y_pred_orig_corrected = np.clip(y_pred_orig_corrected, 0, None)
        
        # Evaluate both versions
        r2_uncorrected = r2_score(y_test_orig, y_pred_orig)
        r2_corrected = r2_score(y_test_orig, y_pred_orig_corrected)
        
        mae_uncorrected = mean_absolute_error(y_test_orig, y_pred_orig)
        mae_corrected = mean_absolute_error(y_test_orig, y_pred_orig_corrected)
        
        bias_uncorrected = np.mean((y_pred_orig - y_test_orig) / y_test_orig * 100)
        bias_corrected = np.mean((y_pred_orig_corrected - y_test_orig) / y_test_orig * 100)
        
        print(f"\nUncorrected - R²: {r2_uncorrected:.4f}, MAE: {mae_uncorrected:.3f}, Bias: {bias_uncorrected:.2f}%")
        print(f"Corrected   - R²: {r2_corrected:.4f}, MAE: {mae_corrected:.3f}, Bias: {bias_corrected:.2f}%")
        
        # Save the better model
        if abs(bias_corrected) < abs(bias_uncorrected):
            print("✅ Using bias-corrected model")
            final_predictions = y_pred_orig_corrected
            final_bias = training_bias
        else:
            print("⚠️ Bias correction didn't help, using original")
            final_predictions = y_pred_orig
            final_bias = 0
        
        # Save model and correction factor
        torch.save(model.state_dict(), f"{output_dir}/{target_name}_enhanced_model.pth")
        torch.save(likelihood.state_dict(), f"{output_dir}/{target_name}_enhanced_likelihood.pth")
        joblib.dump(scaler, f"{output_dir}/{target_name}_enhanced_scaler.pkl")
        joblib.dump({'config': config, 'bias_correction': final_bias}, 
                    f"{output_dir}/{target_name}_enhanced_config.pkl")
        
        return model, likelihood, scaler, config, final_bias


    # Example usage in your main training loop:
    """
    # After loading your data and before training:
    df, extended_target_config = fix_settling_time_bias(df, features, target_config, output_dir)

    # Train multiple variants and compare:
    settling_variants = [
        "SettlingTime_log",
        "SettlingTime_log_corrected", 
        "SettlingTime_sqrt",
        "SettlingTime_boxcox"
    ]

    best_model = None
    best_bias = float('inf')
    best_variant = None

    for variant in settling_variants:
        if variant in extended_target_config:
            print(f"\n{'='*60}")
            print(f"TESTING VARIANT: {variant}")
            print(f"{'='*60}")
            
            model, likelihood, scaler, config, bias_correction = train_improved_settling_time_model(
                df, features, variant, extended_target_config[variant], output_dir
            )
            
            if abs(bias_correction) < abs(best_bias):
                best_bias = bias_correction
                best_model = (model, likelihood, scaler, config, bias_correction)
                best_variant = variant

    print(f"\n🏆 BEST VARIANT: {best_variant} with bias correction: {best_bias:.3f}")
    """
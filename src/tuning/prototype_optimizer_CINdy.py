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

# === Load trained surrogates ===
MODEL_DIR = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGPSurrogate\surrogate_perf_20250620_140556"

def load_model(param: str):
    input_dim = 6
    model = VariationalGP(input_dim)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_model.pth"))
    model.eval()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(torch.load(f"{MODEL_DIR}/{param}_likelihood.pth"))
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
    plant = (2.87, 7.0, 18.0)  # K, T1, T2
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


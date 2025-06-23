# === CINdy-based PID Optimizer Using Surrogate Model ===
# Input: G(s) = [K, T1, T2] and trained surrogate model
# Output: Optimized [Kp, Ki, Kd] minimizing surrogate-predicted cost

import nevergrad as ng
import torch
import joblib
import pandas as pd
import numpy as np
from typing import Dict
import gpytorch

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
def predict_metrics(K, T1, T2, Kp, Ki, Kd) -> Dict[str, Dict[str, float]]:
    X = np.array([[K, T1, T2, Kp, Ki, Kd]], dtype=np.float32)
    result = {}
    for target, (model, likelihood, scaler) in models.items():
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = likelihood(model(X_tensor))
            mean = output.mean.item()
            std = output.variance.sqrt().item()
            #if "log" in target:
                #mean = 10 ** mean - 1e-3
                #std = (10 ** (mean + std)) - 1e-3 - mean  # Approximate std in linear space
            if "log" in target:
                pred_mean = np.expm1(mean)
                lower = np.expm1(mean - std)
                upper = np.expm1(mean + std)
                std_approx = (upper - lower) / 2
            else:
                pred_mean = mean
                std_approx = std

            result[target.replace("_log", "")] = {
                "mean": mean,
                "std": std,
            }
    return result

top5_controllers = []

# === Cost function ===
def surrogate_cost(params, plant, weights):
    Kp, Ki, Kd = params
    K, T1, T2 = plant
    metrics = predict_metrics(K, T1, T2, Kp, Ki, Kd)

    # Compute weighted cost
    weighted = {
        name: weights[name] * metrics[name]["mean"]
        for name in ["ISE", "Overshoot", "SettlingTime", "RiseTime"]
    }
    cost = sum(weighted.values())

    # Print details
    print(f"\n[Evaluation] Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f} → Cost={cost:.3f}")
    for name in ["ISE", "Overshoot", "SettlingTime", "RiseTime"]:
        print(f" - {name}: {metrics[name]['mean']:.3f} ± {metrics[name]['std']:.3f} × {weights[name]} = {weighted[name]:.3f}")

    # Store for top 5 tracking
    top5_controllers.append((cost, {
        "Kp": Kp, "Ki": Ki, "Kd": Kd, 
        "Metrics": metrics, "Weighted": weighted
    }))

    return cost



# === Optimizer with fallback options ===
def optimize_pid(plant, weights, budget=100, optimizer_name="DE"):
    print(f"\U0001F680 Starting optimization with {optimizer_name}...")
    
    parametrization = ng.p.Instrumentation(
        Kp=ng.p.Scalar(init=1.0).set_bounds(0.01, 5.0),
        Ki=ng.p.Scalar(init=0.1).set_bounds(0.001, 5.0),
        Kd=ng.p.Scalar(init=0.01).set_bounds(0.0, 5.0),
    )
    
    # Try different optimizers in order of preference
    optimizers_to_try = [
        optimizer_name,  # User specified
        "DE",           # Differential Evolution (usually robust)
        "PSO",          # Particle Swarm Optimization
        "OnePlusOne",   # Simple evolutionary strategy
        "RandomSearch"  # Last resort
    ]
    
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
                return surrogate_cost([Kp, Ki, Kd], plant, weights)
            
            # Test the optimizer by trying to get one candidate
            test_candidate = optimizer.ask()
            optimizer.tell(test_candidate, cost_fn(**test_candidate.kwargs))
            
            # If we get here, the optimizer works
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
        
        bounds = [(0.01, 3.0), (0.001, 3), (0.0, 3)]  # Kp, Ki, Kd bounds
        
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
    
top5_sorted = sorted(top5_controllers, key=lambda x: x[0])[:5]
print("\n🏆 Top 5 Controllers Evaluated:")
for i, (c, data) in enumerate(top5_sorted, 1):
    print(f"#{i} → Cost={c:.3f}, Kp={data['Kp']:.3f}, Ki={data['Ki']:.3f}, Kd={data['Kd']:.3f}")
    for name in ["ISE", "Overshoot", "SettlingTime", "RiseTime"]:
        m = data["Metrics"][name]
        w = data["Weighted"][name]
        print(f"   • {name}: {m['mean']:.3f} ± {m['std']:.3f}, weighted: {w:.3f}")


# === Example usage ===
if __name__ == "__main__":
    plant = (35, 50.0, 40.0)  # K, T1, T2
    weights = {"ISE": 1.0, "Overshoot": 0.5, "SettlingTime": 0.2, "RiseTime": 0.1}
    
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
    print("\n\U0001F4C8 Predicted Performance Metrics:")
for metric, result in predicted.items():
    print(f"{metric}: {result['mean']:.4f} ± {result['std']:.4f}")

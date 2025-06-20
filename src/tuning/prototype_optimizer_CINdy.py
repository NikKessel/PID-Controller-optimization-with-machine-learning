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
MODEL_DIR = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\DGPSurrogate\surrogate_perf_20250620_130054"

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

models = {}
for target in ["ISE_log", "Overshoot", "SettlingTime_log", "RiseTime_log"]:
    models[target] = load_model(target)

# === Surrogate prediction ===
def predict_metrics(K, T1, T2, Kp, Ki, Kd) -> Dict[str, float]:
    X = np.array([[K, T1, T2, Kp, Ki, Kd]], dtype=np.float32)
    preds = {}
    for target, (model, likelihood, scaler) in models.items():
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = likelihood(model(X_tensor))
            mean = output.mean.item()
            if "log" in target:
                mean = 10 ** mean - 1e-3
            preds[target.replace("_log", "")] = mean
    return preds

# === Cost function ===
def surrogate_cost(params, plant, weights):
    Kp, Ki, Kd = params
    K, T1, T2 = plant
    metrics = predict_metrics(K, T1, T2, Kp, Ki, Kd)
    return (
        weights["ISE"] * metrics["ISE"] +
        weights["Overshoot"] * metrics["Overshoot"] +
        weights["SettlingTime"] * metrics["SettlingTime"] +
        weights["RiseTime"] * metrics["RiseTime"]
    )

# === Optimizer ===
def optimize_pid(plant, weights, budget=100):
    parametrization = ng.p.Instrumentation(
        Kp=ng.p.Scalar(init=1.0).set_bounds(0.01, 100.0),
        Ki=ng.p.Scalar(init=0.1).set_bounds(0.001, 50.0),
        Kd=ng.p.Scalar(init=0.01).set_bounds(0.0, 50.0),
    )
    optimizer = ng.optimizers.registry.registry['CMA'](parametrization=parametrization, budget=budget)

    def cost_fn(Kp, Ki, Kd):
        return surrogate_cost([Kp, Ki, Kd], plant, weights)

    recommendation = optimizer.minimize(cost_fn)
    result = recommendation.kwargs
    return result, cost_fn(**result)

# === Example usage ===
if __name__ == "__main__":
    plant = (1.2, 15.0, 4.0)  # K, T1, T2
    weights = {"ISE": 1.0, "Overshoot": 0.5, "SettlingTime": 0.2, "RiseTime": 0.1}
    best_pid, best_cost = optimize_pid(plant, weights)
    print("\n🧠 Best PID:", best_pid)
    print("🔍 Predicted Cost:", best_cost)
# === CINdy-based PID Optimizer with Diagnostics ===

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

# === DIAGNOSTIC FUNCTIONS ===
def diagnose_surrogate_model(models, plant, pid_params):
    """Diagnose potential issues with surrogate model predictions"""
    
    K, T1, T2 = plant
    Kp, Ki, Kd = pid_params["Kp"], pid_params["Ki"], pid_params["Kd"]
    
    print("\n" + "="*60)
    print("🔍 SURROGATE MODEL DIAGNOSTICS")
    print("="*60)
    
    # Check input ranges
    print("\n📊 INPUT PARAMETER RANGES:")
    print(f"Plant: K={K:.3f}, T1={T1:.3f}, T2={T2:.3f}")
    print(f"PID:   Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
    
    # Check each model's prediction details
    X = np.array([[K, T1, T2, Kp, Ki, Kd]], dtype=np.float32)
    
    print("\n🎯 DETAILED PREDICTIONS BY MODEL:")
    print("-" * 50)
    
    for target, (model, likelihood, scaler) in models.items():
        print(f"\n📈 {target}:")
        
        # Check original input
        print(f"  Raw input: {X[0]}")
        
        # Check scaler statistics
        try:
            if hasattr(scaler, 'mean_'):
                print(f"  Scaler mean: {scaler.mean_}")
                print(f"  Scaler scale: {scaler.scale_}")
            elif hasattr(scaler, 'data_min_'):
                print(f"  Scaler min: {scaler.data_min_}")
                print(f"  Scaler max: {scaler.data_max_}")
        except:
            print("  Could not access scaler parameters")
        
        # Check scaled input
        X_scaled = scaler.transform(X)
        print(f"  Scaled input: {X_scaled[0]}")
        
        # Check if scaled values are reasonable (typically should be roughly [-3, 3])
        extreme_indices = np.where(np.abs(X_scaled[0]) > 3)[0]
        if len(extreme_indices) > 0:
            print(f"  ⚠️  WARNING: Scaled input contains extreme values at indices {extreme_indices}!")
            print(f"      Values: {X_scaled[0][extreme_indices]}")
            print(f"      This suggests the input is outside the training range.")
        
        # Get prediction with uncertainty
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = likelihood(model(X_tensor))
            raw_mean = output.mean.item()
            raw_std = output.stddev.item()
            
        print(f"  Raw prediction: {raw_mean:.4f} ± {raw_std:.4f}")
        
        # Apply inverse transform
        if "log" in target:
            final_pred = 10 ** raw_mean - 1e-3
            print(f"  After log transform: {final_pred:.4f}")
            print(f"  Log transform: 10^{raw_mean:.4f} - 0.001 = {final_pred:.4f}")
        else:
            final_pred = raw_mean
            
        # Check if prediction is physically reasonable
        if target == "Overshoot" and final_pred < 0:
            print(f"  ❌ CRITICAL ISSUE: Negative overshoot ({final_pred:.4f})!")
            print(f"      Raw prediction before transform: {raw_mean:.4f}")
            if "log" in target:
                print(f"      This means 10^{raw_mean:.4f} - 0.001 = {final_pred:.4f}")
                print(f"      The raw prediction is too negative for log transform!")
            
        elif target == "Overshoot" and final_pred > 100:
            print(f"  ⚠️  WARNING: Very high overshoot ({final_pred:.4f}%)")
            
        print(f"  Model confidence (std): {raw_std:.4f}")
        if raw_std > 1.0:
            print(f"  ⚠️  WARNING: High uncertainty in prediction!")
        elif raw_std > 2.0:
            print(f"  ❌ CRITICAL: Very high uncertainty - model is guessing!")

def test_reasonable_inputs(models):
    """Test the model with some reasonable inputs to see if it behaves well"""
    
    print("\n" + "="*60)
    print("🧪 TESTING WITH REASONABLE INPUTS")
    print("="*60)
    
    # Test cases: [K, T1, T2, Kp, Ki, Kd]
    test_cases = [
        ("Conservative PID", [1.0, 10.0, 2.0, 0.5, 0.1, 0.05]),
        ("Moderate PID", [1.0, 10.0, 2.0, 1.0, 0.5, 0.1]),
        ("Aggressive PID", [1.0, 10.0, 2.0, 2.0, 1.0, 0.2]),
        ("Your Plant + Conservative", [1.2, 15.0, 4.0, 0.5, 0.1, 0.05]),
        ("Your Plant + Moderate", [1.2, 15.0, 4.0, 1.0, 0.5, 0.1]),
        ("Your Optimized Result", [20, 40.0, 44.0, 1.173, 2.716, 1.678]),
    ]
    
    for name, params in test_cases:
        print(f"\n📋 {name}:")
        print(f"   Input: K={params[0]}, T1={params[1]}, T2={params[2]}, Kp={params[3]:.3f}, Ki={params[4]:.3f}, Kd={params[5]:.3f}")
        
        X = np.array([params], dtype=np.float32)
        predictions = {}
        
        for target, (model, likelihood, scaler) in models.items():
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                output = likelihood(model(X_tensor))
                raw_mean = output.mean.item()
                raw_std = output.stddev.item()
                
            if "log" in target:
                final_pred = 10 ** raw_mean - 1e-3
            else:
                final_pred = raw_mean
                
            metric_name = target.replace('_log', '')
            predictions[metric_name] = final_pred
            
            # Flag problematic predictions
            flag = ""
            if metric_name == "Overshoot" and final_pred < 0:
                flag = " ❌ NEGATIVE!"
            elif metric_name == "Overshoot" and final_pred > 50:
                flag = " ⚠️ HIGH"
            elif raw_std > 1.5:
                flag = " ⚠️ UNCERTAIN"
                
            print(f"   {metric_name}: {final_pred:.4f}{flag}")

def analyze_scaler_ranges(models):
    """Analyze the training data ranges from scalers"""
    print("\n" + "="*60)
    print("📏 SCALER ANALYSIS (Training Data Ranges)")
    print("="*60)
    
    param_names = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
    
    for target, (model, likelihood, scaler) in models.items():
        print(f"\n🎯 {target} Model:")
        
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            print("  Training data ranges (min → max):")
            for i, param in enumerate(param_names):
                print(f"    {param}: {scaler.data_min_[i]:.4f} → {scaler.data_max_[i]:.4f}")
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            print("  StandardScaler stats:")
            for i, param in enumerate(param_names):
                # For StandardScaler, we can estimate range as mean ± 2*scale
                mean_val = scaler.mean_[i]
                scale_val = scaler.scale_[i]
                est_min = mean_val - 2*scale_val
                est_max = mean_val + 2*scale_val
                print(f"    {param}: ~{est_min:.4f} → ~{est_max:.4f} (mean±2σ)")
        else:
            print("  Could not determine scaler type or ranges")

def suggest_fixes():
    """Suggest potential fixes for the negative overshoot issue"""
    
    print("\n" + "="*60)
    print("🔧 SUGGESTED FIXES")
    print("="*60)
    print("""
1. 🔍 IMMEDIATE DEBUGGING:
   - Check if overshoot was negative in your training data
   - Verify the log transformation is correct for overshoot
   - Check if the offset (1e-3) in log transform is appropriate

2. 🛠️ QUICK FIXES:
   - Add bounds checking: max(0, predicted_overshoot)
   - Use a different optimizer initialization
   - Constrain the optimizer search space

3. 🔄 MODEL IMPROVEMENTS:
   - Retrain with proper overshoot constraints (≥ 0)
   - Use different preprocessing for overshoot (e.g., no log transform)
   - Add more training data in the problematic region

4. 🎯 OPTIMIZATION IMPROVEMENTS:
   - Add penalty for unrealistic predictions in cost function
   - Use multi-objective optimization with constraint handling
   - Initialize optimizer with known good PID values
""")

# === Original functions with safety bounds ===
def predict_metrics_safe(K, T1, T2, Kp, Ki, Kd) -> Dict[str, float]:
    """Predict metrics with safety bounds"""
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
                
        metric_name = target.replace("_log", "")
        
        # Apply safety bounds
        if metric_name == "Overshoot":
            if mean < 0:
                print(f"⚠️  Clamping negative overshoot {mean:.4f} to 0.0")
            mean = max(0.0, mean)  # Overshoot cannot be negative
        elif metric_name in ["SettlingTime", "RiseTime"]:
            mean = max(0.001, mean)  # Times cannot be zero or negative
        elif metric_name == "ISE":
            mean = max(0.0, mean)  # ISE cannot be negative
            
        preds[metric_name] = mean
    return preds

# === Original prediction function (unchanged) ===
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
    # Predict metrics safely (with clamping)
    metrics = predict_metrics_safe(plant[0], plant[1], plant[2], Kp, Ki, Kd)
    
    # Check for any negative metric values (should not happen after clamping, but just in case)
    for metric_name, value in metrics.items():
        if value < 0:
            # Penalize heavily
            return 1e6  # Large penalty cost to reject this solution
    
    # Optionally also check for NaN or inf
    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
        return 1e6
    
    # Compute weighted cost normally
    cost = (
        weights["ISE"] * metrics["ISE"] +
        weights["Overshoot"] * metrics["Overshoot"] +
        weights["SettlingTime"] * metrics["SettlingTime"] +
        weights["RiseTime"] * metrics["RiseTime"]
    )
    return cost


# === Optimizer ===
def optimize_pid(plant, weights, budget=1000):
    print("\U0001F680 Starting optimization with DE...")
    parametrization = ng.p.Instrumentation(
        Kp=ng.p.Scalar(init=1.0).set_bounds(0.5, 3.0),
        Ki=ng.p.Scalar(init=0.1).set_bounds(0.001, 2.0),
        Kd=ng.p.Scalar(init=0.01).set_bounds(0.0, 5.0),
    )
    
    optimizer_cls = ng.optimizers.registry['DE']  # Use DE instead of CMA
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget)

    def cost_fn(Kp, Ki, Kd):
        return surrogate_cost([Kp, Ki, Kd], plant, weights)

    recommendation = optimizer.minimize(cost_fn)
    result = recommendation.kwargs
    print("✅ Optimization complete.")
    return result, cost_fn(**result)

# === Example usage with diagnostics ===
if __name__ == "__main__":
    print("\U0001F4E6 Loading surrogate models...")
    models = {}
    for target in ["ISE_log", "Overshoot", "SettlingTime_log", "RiseTime_log"]:
        print(f"   └─ {target}")
        models[target] = load_model(target)
    
    plant = (16, 25.0, 30.0)  # K, T1, T2
    weights = {"ISE": 1.0, "Overshoot": 0.5, "SettlingTime": 0.2, "RiseTime": 0.1}
    
    # Run optimization
    best_pid, best_cost = optimize_pid(plant, weights)
    print("\n\U0001F9E0 Best PID:", best_pid)
    print("\U0001F50D Predicted Cost:", best_cost)

    predicted = predict_metrics(plant[0], plant[1], plant[2], best_pid["Kp"], best_pid["Ki"], best_pid["Kd"])
    print("\n\U0001F4C8 Predicted Performance Metrics:")
    for metric, value in predicted.items():
        print(f"{metric}: {value:.4f}")
    
    # === RUN DIAGNOSTICS ===
    print("\n" + "🔍" * 20 + " RUNNING DIAGNOSTICS " + "🔍" * 20)
    
    # 1. Analyze the problematic result
    diagnose_surrogate_model(models, plant, best_pid)
    
    # 2. Test with reasonable inputs
    test_reasonable_inputs(models)
    
    # 3. Analyze scaler ranges
    analyze_scaler_ranges(models)
    
    # 4. Show suggestions
    suggest_fixes()
    
    print("\n" + "="*80)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("="*80)
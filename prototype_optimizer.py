import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
from scipy.optimize import differential_evolution
from control.matlab import tf, feedback, step
import matplotlib.pyplot as plt

# === Helper function to estimate Tu and Tg if missing ===
def estimate_tu_tg(K, T1, T2, Td):
    # Placeholder heuristic estimation (can be replaced by domain-specific logic)
    estimated_tu = 0.2 * (T1 + T2 + Td)
    estimated_tg = 0.6 * (T1 + T2 + Td)
    return estimated_tu, estimated_tg

# === 1. Load surrogate model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_20250608_222802\mlp_surrogate_model.pkl"
surrogate_model = joblib.load(model_path)

# === 2. Prompt user for plant parameters ===
print("Enter plant parameters for G(s):")
tf_type = input("Type (PT1, PT2, PT1+Td, PT2+Td): ")
K = float(input("K: "))
T1 = float(input("T1: "))
T2 = float(input("T2 (0 if not applicable): "))
Td = float(input("Td (0 if not applicable): "))
Tu = float(input("Tu (0 if not applicable): "))
Tg = float(input("Tg (0 if not applicable): "))

# Estimate Tu and Tg if zero
if Tu == 0 or Tg == 0:
    est_tu, est_tg = estimate_tu_tg(K, T1, T2, Td)
    if Tu == 0:
        Tu = est_tu
        print(f"Estimated Tu: {Tu:.4f}")
    if Tg == 0:
        Tg = est_tg
        print(f"Estimated Tg: {Tg:.4f}")

# One-hot encode type
type_encoding = {
    "type_PT1": 0,
    "type_PT1+Td": 0,
    "type_PT2": 0,
    "type_PT2+Td": 0,
}
if tf_type in ["PT1", "PT1+Td", "PT2", "PT2+Td"]:
    type_encoding[f"type_{tf_type}"] = 1
else:
    print("Unknown type. Defaulting to PT1.")
    type_encoding["type_PT1"] = 1

# === 3. Define cost function using surrogate model ===
def predict_ise_from_pid(params):
    Kp, Ki, Kd = params
    input_vec = np.array([
        K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd,
        type_encoding["type_PT1"],
        type_encoding["type_PT1+Td"],
        type_encoding["type_PT2"],
        type_encoding["type_PT2+Td"]
    ]).reshape(1, -1)
    pred_ise = surrogate_model.predict(input_vec)[0]
    return pred_ise  # Do not clip negative values

# === 4. Optimize PID parameters using Differential Evolution ===
param_bounds = [(0, 10), (0, 10), (0, 5)]  # Bounds for Kp, Ki, Kd
result = differential_evolution(predict_ise_from_pid, param_bounds, strategy='best1bin', maxiter=50, popsize=20, tol=1e-6, seed=42)

Kp_opt, Ki_opt, Kd_opt = result.x
print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted ISE: {result.fun:.4f}")

print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")
print("\n=== Summary ===")
print(f"Used Tu: {Tu:.4f}")
print(f"Used Tg: {Tg:.4f}")
print(f"Surrogate predicted ISE: {result.fun:.4f}")
print(f"Optimized PID: Kp={Kp_opt:.4f}, Ki={Ki_opt:.4f}, Kd={Kd_opt:.4f}")



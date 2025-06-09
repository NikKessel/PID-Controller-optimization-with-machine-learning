import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
from scipy.optimize import differential_evolution
from control.matlab import tf, feedback, step
import matplotlib.pyplot as plt
from functools import partial


# === Helper function to estimate Tu and Tg if missing ===
def estimate_tu_tg(K, T1, T2, Td):
    estimated_tu = 0.2 * (T1 + T2 + Td)
    estimated_tg = 0.6 * (T1 + T2 + Td)
    return estimated_tu, estimated_tg

# === 1. Load surrogate model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_20250609_182908\pid_mlp_surrogate_model.pkl"
surrogate_model = joblib.load(model_path)

# === 2. Prompt user for plant parameters ===
print("Enter plant parameters for G(s):")
tf_type = input("Type (PT1, PT2, PT1+Td, PT2+Td): ").strip()
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

# Validate type
valid_types = ["PT1", "PT2", "PT1+Td", "PT2+Td"]
type_str = tf_type if tf_type in valid_types else "PT1"
if tf_type not in valid_types:
    print("Unknown type. Defaulting to PT1.")

# === 3. Define cost function using surrogate model ===
def cost_from_pid(params, plant_params, surrogate_model, weights=None):
    K, T1, T2, Td, Tu, Tg, type_str = plant_params
    Kp, Ki, Kd = params

    if weights is None:
        weights = [0.4, 0.2, 0.2, 0.2]  # ISE, Overshoot, SettlingTime, RiseTime

    try:
        input_df = pd.DataFrame([{
            "K": K, "T1": T1, "T2": T2, "Td": Td, "Tu": Tu, "Tg": Tg,
            "Kp": Kp, "Ki": Ki, "Kd": Kd, "type": type_str
        }])

        prediction = surrogate_model.predict(input_df)[0]
        ise, overshoot, settling, rise = prediction

        cost = (weights[0] * ise +
                weights[1] * overshoot +
                weights[2] * settling +
                weights[3] * rise)
        return cost

    except Exception as e:
        print("❌ Error in cost_from_pid:", e)
        return 1e6  # high penalty for invalid point

# === 4. Optimize PID parameters ===
plant_params = (K, T1, T2, Td, Tu, Tg, type_str)
cost_fn = partial(cost_from_pid, plant_params=plant_params, surrogate_model=surrogate_model)
param_bounds = [(0.1, 10), (0.01, 10), (0, 2)]
result = differential_evolution(cost_fn, param_bounds, strategy='best1bin', maxiter=50, popsize=20, tol=1e-6, seed=42)
Kp_opt, Ki_opt, Kd_opt = result.x
print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted cost: {result.fun:.4f}")
print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")

# === 5. Visualization: ISE vs Kp ===
kp_range = np.linspace(0.1, 10, 100)
ise_values = []
for kp in kp_range:
    ise = cost_from_pid([kp, Ki_opt, Kd_opt], plant_params, surrogate_model, weights=[1, 0, 0, 0])
    ise_values.append(ise)

plt.figure(figsize=(8, 4))
plt.plot(kp_range, ise_values, label="Predicted ISE")
plt.xlabel("Kp")
plt.ylabel("ISE")
plt.title("ISE vs Kp (Ki, Kd fixed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Numerical gradients around optimum ===
def estimate_gradient(pid):
    delta = 1e-2
    base = cost_from_pid(pid, plant_params, surrogate_model, weights=[1, 0, 0, 0])
    grad = []
    for i in range(3):
        shift = pid.copy()
        shift[i] += delta
        grad_val = (cost_from_pid(shift, plant_params, surrogate_model, weights=[1, 0, 0, 0]) - base) / delta
        grad.append(grad_val)
    return grad

grad_kp, grad_ki, grad_kd = estimate_gradient([Kp_opt, Ki_opt, Kd_opt])
print("\nGradient Sensitivity near optimum:")
print(f"dISE/dKp: {grad_kp:.4f}, dISE/dKi: {grad_ki:.4f}, dISE/dKd: {grad_kd:.4f}")

# === 7. Distribution of predicted ISE in optimization ===
pop_samples = np.random.uniform([0, 0, 0], [10, 10, 5], size=(1000, 3))
pred_ises = [cost_from_pid(p, plant_params, surrogate_model, weights=[1, 0, 0, 0]) for p in pop_samples]

plt.figure(figsize=(6, 4))
plt.hist(pred_ises, bins=30, color='steelblue', edgecolor='k')
plt.title("Distribution of Surrogate-Predicted ISE")
plt.xlabel("ISE")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("ise_distribution_histogram.png")
plt.show()

# === 8. Final Output ===
print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted ISE: {result.fun:.4f}")
print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")

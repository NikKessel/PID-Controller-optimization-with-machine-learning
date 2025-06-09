import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
from scipy.optimize import differential_evolution
from control.matlab import tf, feedback, step
import matplotlib.pyplot as plt

# === Helper function to estimate Tu and Tg if missing ===
def estimate_tu_tg(K, T1, T2, Td):
    estimated_tu = 0.2 * (T1 + T2 + Td)
    estimated_tg = 0.6 * (T1 + T2 + Td)
    return estimated_tu, estimated_tg

# === Helper function to simulate response and extract metrics ===
def simulate_metrics(K, T1, T2, Td, Kp, Ki, Kd):
    s = tf([1, 0], [1])
    if T2 > 0:
        plant = tf([K], [T1*T2, T1+T2, 1])
    else:
        plant = tf([K], [T1, 1])
    if Td > 0:
        D_tf = tf([Kd * Td, Kd], [1])
    else:
        D_tf = tf([Kd], [1])
    I_tf = tf([Ki], [1, 0]) if Ki > 0 else 0
    P_tf = tf([Kp], [1])
    PID = P_tf + I_tf + D_tf
    sys_cl = feedback(PID * plant, 1)
    t, y = step(sys_cl, np.linspace(0, 100, 1000))
    rise_time = t[next(i for i, val in enumerate(y) if val >= 0.9)] if any(val >= 0.9 for val in y) else 100
    ss_error = abs(1 - y[-1])
    return rise_time, ss_error

# === 1. Load surrogate model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_20250609_122155\mlp_surrogate_model.pkl"
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
type_encoding = {"type_PT1": 0, "type_PT1+Td": 0, "type_PT2": 0, "type_PT2+Td": 0}
if tf_type in type_encoding:
    type_encoding[f"type_{tf_type}"] = 1
else:
    print("Unknown type. Defaulting to PT1.")
    type_encoding["type_PT1"] = 1

# === 3. Define cost function using surrogate model ===
def predict_ise_from_pid(params):
    Kp, Ki, Kd = params
    rise_time, ss_error = simulate_metrics(K, T1, T2, Td, Kp, Ki, Kd)
    input_vec = np.array([
        K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd,
        rise_time, ss_error,
        type_encoding["type_PT1"],
        type_encoding["type_PT1+Td"],
        type_encoding["type_PT2"],
        type_encoding["type_PT2+Td"]
    ]).reshape(1, -1)
    pred_ise = surrogate_model.predict(input_vec)[0]
    return pred_ise

# === 4. Optimize PID parameters ===
param_bounds = [(0, 10), (0, 10), (0, 5)]
result = differential_evolution(predict_ise_from_pid, param_bounds, strategy='best1bin', maxiter=50, popsize=20, tol=1e-6, seed=42)
Kp_opt, Ki_opt, Kd_opt = result.x
print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted ISE: {result.fun:.4f}")
print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")

# === 5. Surrogate sensitivity visualization ===
kp_range = np.linspace(0, 10, 50)
ki_range = np.linspace(0, 10, 50)
kd_range = np.linspace(0, 5, 50)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

z1 = [predict_ise_from_pid([kp, Ki_opt, Kd_opt]) for kp in kp_range]
axs[0].plot(kp_range, z1)
axs[0].set_title("ISE vs Kp")
axs[0].set_xlabel("Kp")
axs[0].set_ylabel("Predicted ISE")

z2 = [predict_ise_from_pid([Kp_opt, ki, Kd_opt]) for ki in ki_range]
axs[1].plot(ki_range, z2)
axs[1].set_title("ISE vs Ki")
axs[1].set_xlabel("Ki")

z3 = [predict_ise_from_pid([Kp_opt, Ki_opt, kd]) for kd in kd_range]
axs[2].plot(kd_range, z3)
axs[2].set_title("ISE vs Kd")
axs[2].set_xlabel("Kd")
plt.tight_layout()
plt.savefig("ise_sensitivity_curves.png")
plt.show()

# === 6. Numerical gradients around optimum ===
def estimate_gradient(pid):
    delta = 1e-2
    base = predict_ise_from_pid(pid)
    grad = []
    for i in range(3):
        shift = pid.copy()
        shift[i] += delta
        grad_val = (predict_ise_from_pid(shift) - base) / delta
        grad.append(grad_val)
    return grad

grad_kp, grad_ki, grad_kd = estimate_gradient([Kp_opt, Ki_opt, Kd_opt])
print("\nGradient Sensitivity near optimum:")
print(f"dISE/dKp: {grad_kp:.4f}, dISE/dKi: {grad_ki:.4f}, dISE/dKd: {grad_kd:.4f}")

# === 7. Distribution of predicted ISE in optimization ===
pop_samples = np.random.uniform([0, 0, 0], [10, 10, 5], size=(1000, 3))
pred_ises = [predict_ise_from_pid(p) for p in pop_samples]
plt.figure(figsize=(6,4))
plt.hist(pred_ises, bins=30, color='steelblue', edgecolor='k')
plt.title("Distribution of Surrogate-Predicted ISE")
plt.xlabel("ISE")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("ise_distribution_histogram.png")
plt.show()

print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted ISE: {result.fun:.4f}")
print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")
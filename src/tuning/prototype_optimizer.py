import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
from scipy.optimize import differential_evolution
from control.matlab import tf, feedback, step
import matplotlib.pyplot as plt
from functools import partial
import os
from datetime import datetime

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

        # Check for physically invalid values
        if any(x < 0 or np.isnan(x) for x in prediction):
            print(f"⚠️ Invalid prediction discarded (values: {prediction}) for PID: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
            return [1e6, 1e6, 1e6, 1e6]

        # Valid result → log and return
        print(f"PID: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f} → Predicted ISE={prediction[0]:.2f}, OS={prediction[1]:.2f}, ST={prediction[2]:.2f}, RT={prediction[3]:.2f}")
        return prediction


    except Exception as e:
        print("❌ Error in cost_from_pid:", e)
        return [1e6, 1e6, 1e6, 1e6]  # high penalties

# === 4. Optimize PID parameters ===
def cost_only(params):
    ise, overshoot, settling_time, rise_time = cost_from_pid(params, plant_params, surrogate_model)
    return 0.4 * ise + 0.2 * overshoot + 0.2 * settling_time + 0.2 * rise_time

plant_params = (K, T1, T2, Td, Tu, Tg, type_str)
param_bounds = [(0.1, 10), (0.01, 10), (0, 2)]
result = differential_evolution(cost_only, param_bounds, strategy='best1bin', maxiter=50, popsize=20, tol=1e-6, seed=42)
Kp_opt, Ki_opt, Kd_opt = result.x

# === 5. Create timestamped export folder ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_dir = os.path.join("D:/BA/PID-Controller-optimization-with-machine-learning/results", f"run_{timestamp}")
os.makedirs(export_dir, exist_ok=True)

# === 6. Plot ISE vs Kp ===
kp_range = np.linspace(0.1, 10, 100)
ise_values = [cost_from_pid([kp, Ki_opt, Kd_opt], plant_params, surrogate_model)[0] for kp in kp_range]
plt.figure()
plt.plot(kp_range, ise_values, label="Predicted ISE")
plt.xlabel("Kp")
plt.ylabel("ISE")
plt.title("ISE vs Kp (Ki, Kd fixed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "ise_vs_kp.png"))
plt.show()

# === 7. Plot Overshoot and Settling Time vs Kp ===
overshoot_values = [cost_from_pid([kp, Ki_opt, Kd_opt], plant_params, surrogate_model)[1] for kp in kp_range]
settling_values = [cost_from_pid([kp, Ki_opt, Kd_opt], plant_params, surrogate_model)[2] for kp in kp_range]

plt.figure()
plt.plot(kp_range, overshoot_values, label="Overshoot")
plt.plot(kp_range, settling_values, label="Settling Time")
plt.xlabel("Kp")
plt.ylabel("Metric Value")
plt.title("Overshoot and Settling Time vs Kp")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "overshoot_settling_vs_kp.png"))
plt.show()

# === 8. Distribution of ISE ===
pop_samples = np.random.uniform([0, 0, 0], [10, 10, 5], size=(1000, 3))
pred_ises = [cost_from_pid(p, plant_params, surrogate_model)[0] for p in pop_samples]
plt.figure()
plt.hist(pred_ises, bins=30, color='steelblue', edgecolor='k')
plt.title("Distribution of Surrogate-Predicted ISE")
plt.xlabel("ISE")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(export_dir, "ise_distribution_histogram.png"))
plt.show()

# === 9. Save results ===
ise, overshoot, settling_time, rise_time = cost_from_pid([Kp_opt, Ki_opt, Kd_opt], plant_params, surrogate_model)
with open(os.path.join(export_dir, "summary.txt"), "w") as f:
    f.write(f"Optimal PID parameters:\n")
    f.write(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}\n")
    f.write(f"Predicted ISE: {ise:.4f}\n")
    f.write(f"Overshoot: {overshoot:.4f}\n")
    f.write(f"Settling Time: {settling_time:.4f}\n")
    f.write(f"Rise Time: {rise_time:.4f}\n")
    f.write(f"Cost: {result.fun:.4f}\n")
    f.write(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}\n")

# === 10. Console output ===
print("\nOptimal PID parameters:")
print(f"Kp: {Kp_opt:.4f}, Ki: {Ki_opt:.4f}, Kd: {Kd_opt:.4f}")
print(f"Predicted ISE: {ise:.4f}")
print(f"Overshoot: {overshoot:.4f}, Settling Time: {settling_time:.4f}, Rise Time: {rise_time:.4f}")
print(f"Final Tu used: {Tu:.4f}, Final Tg used: {Tg:.4f}")
print(f"Results saved to: {export_dir}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from control.matlab import tf, feedback, step
from scipy.integrate import cumulative_trapezoid

# === 1. Load and preprocess dataset (only 'good' labels) ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = df[df["Label"] == "good"].copy()
df = df.dropna(subset=["K", "T1", "Tu", "Tg", "Kp", "Ki", "Kd"])
df["T2"] = df["T2"].fillna(0)
df["Td"] = df["Td"].fillna(0)
type_dummies = pd.get_dummies(df["type"], prefix="type")
df = pd.concat([df, type_dummies], axis=1)

# === 2. Helper functions for metrics ===
def compute_ise(K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd):
    num = [K]
    den = np.convolve([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
    plant = tf(num, den)
    if Td > 0: plant = plant * tf([Td, 1], [1])
    if Tu > 0: plant = plant * tf([1], [Tu, 1])
    if Tg > 0: plant = plant * tf([1], [Tg, 1])
    pid = tf([Kd, Kp, Ki], [1, 0])
    closed_loop = feedback(pid * plant, 1)
    t = np.linspace(0, 50, 500)
    t, y = step(closed_loop, T=t)
    e = 1 - y
    return np.trapz(e**2, t)

def compute_overshoot(t, y):
    y_final = y[-1]
    y_peak = np.max(y)
    return ((y_peak - y_final) / y_final) * 100 if y_peak > y_final else 0

def compute_settling_time(t, y, tol=0.05):
    y_final = y[-1]
    lower, upper = (1 - tol) * y_final, (1 + tol) * y_final
    for i in reversed(range(len(y))):
        if not (lower <= y[i] <= upper):
            return t[i+1] if i+1 < len(t) else t[-1]
    return 0

def compute_full_metrics(row):
    K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd = row[["K", "T1", "T2", "Td", "Tu", "Tg", "Kp", "Ki", "Kd"]]
    num = [K]
    den = np.convolve([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
    plant = tf(num, den)
    if Td > 0: plant = plant * tf([Td, 1], [1])
    if Tu > 0: plant = plant * tf([1], [Tu, 1])
    if Tg > 0: plant = plant * tf([1], [Tg, 1])
    pid = tf([Kd, Kp, Ki], [1, 0])
    closed_loop = feedback(pid * plant, 1)
    t = np.linspace(0, 50, 500)
    t, y = step(closed_loop, T=t)
    e = 1 - y
    de = np.gradient(e, t)
    integral_e = cumulative_trapezoid(e, t, initial=0)
    u = Kp * e + Ki * integral_e + Kd * de
    rise_time = t[next(i for i in range(len(y)) if y[i] >= 0.9 * y[-1])] - \
                t[next(i for i in range(len(y)) if y[i] >= 0.1 * y[-1])]
    ss_error = abs(1 - y[-1])
    overshoot = compute_overshoot(t, y)
    settling_time = compute_settling_time(t, y)
    control_energy = np.trapz(u**2, t)
    ise = np.trapz(e**2, t)
    return pd.Series([ise, rise_time, ss_error, overshoot, settling_time, control_energy])

# === 3. Compute and normalize all metrics ===
print("Computing control metrics...")
df[["ISE", "rise_time", "ss_error", "overshoot", "settling_time", "control_energy"]] = df.apply(compute_full_metrics, axis=1)
df = df[df["ISE"] > 0.0]
df[["ISE", "overshoot", "settling_time", "control_energy"]] = df[["ISE", "overshoot", "settling_time", "control_energy"]].clip(upper=10000)

# Normalize for weighting
df["ise_norm"] = df["ISE"] / df["ISE"].max()
df["overshoot_norm"] = df["overshoot"] / df["overshoot"].max()
df["settling_time_norm"] = df["settling_time"] / df["settling_time"].max()
df["control_energy_norm"] = df["control_energy"] / df["control_energy"].max()

# Define weighted composite loss
w_ise = 0.5
w_os = 0.2
w_st = 0.2
w_ce = 0.1
df["composite_loss"] = (
    w_ise * df["ise_norm"] +
    w_os * df["overshoot_norm"] +
    w_st * df["settling_time_norm"] +
    w_ce * df["control_energy_norm"]
)

# === 4. Model training ===
features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Kp", "Ki", "Kd",
            "rise_time", "ss_error"] + list(type_dummies.columns)
target = "composite_loss"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=5000, random_state=42)
mlp.fit(X_train, y_train)

# === 5. Evaluation ===
y_pred = mlp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae, "\nMSE:", mse, "\nR²:", r2)

# === 6. Save ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("True Composite Loss")
plt.ylabel("Predicted Composite Loss")
plt.title("Surrogate Model: True vs Predicted Loss")
plt.grid(True)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = fr"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "true_vs_predicted_loss.png"))
joblib.dump(mlp, os.path.join(output_dir, "mlp_surrogate_model.pkl"))

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"MAE: {mae}\nMSE: {mse}\nR²: {r2}\n")

df.to_csv(os.path.join(output_dir, "surrogate_training_data_with_metrics.csv"), index=False)

print(f"Model and results saved in {output_dir}")

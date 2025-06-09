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

# === 1. Load and preprocess dataset (only 'good' labels) ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = df[df["Label"] == "good"].copy()  # Filter for good responses only
df = df.dropna(subset=["K", "T1", "Tu", "Tg", "Kp", "Ki", "Kd"])


# Fill T2 and Td NaNs with 0 for systems without them
df["T2"] = df["T2"].fillna(0)
df["Td"] = df["Td"].fillna(0)

# One-hot encode the system type
type_dummies = pd.get_dummies(df["type"], prefix="type")
df = pd.concat([df, type_dummies], axis=1)

# === 2. Simulate system response to compute ISE ===
def compute_ise(K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd):
    num = [K]
    den = [1]
    if T2 > 0:
        den = np.convolve([T1, 1], [T2, 1])
    else:
        den = [T1, 1]
    plant = tf(num, den)
    if Td > 0:
        plant = plant * tf([Td, 1], [1])
    if Tu > 0:
        plant = plant * tf([1], [Tu, 1])
    if Tg > 0:
        plant = plant * tf([1], [Tg, 1])

    pid = tf([Kd, Kp, Ki], [1, 0])
    closed_loop = feedback(pid * plant, 1)
    t = np.linspace(0, 50, 500)
    t, y = step(closed_loop, T=t)
    e = 1 - y
    ise = np.trapz(e**2, t)
    return ise

print("Computing ISE values for each row...")
df["ISE"] = df.apply(lambda row: compute_ise(row.K, row.T1, row.T2, row.Td, row.Tu, row.Tg, row.Kp, row.Ki, row.Kd), axis=1)
df["ISE"] = df["ISE"].clip(lower=0, upper=10000)  # Remove extreme outliers
df = df[df["ISE"] > 0.0]  # ⬅️ This line excludes zero ISE entries
print("ISE value statistics:")
print(df["ISE"].describe())
print("Any negative ISE values?:", (df["ISE"] < 0).any())

# Optional: Log transform
df["ISE"] = np.log1p(df["ISE"])
print("ISE value statistics:")
print(df["ISE"].describe())
print("Any negative ISE values?:", (df["ISE"] < 0).any())

def compute_rise_time_and_ss_error(K, T1, T2, Td, Tu, Tg, Kp, Ki, Kd):
    num = [K]
    den = [1]
    if T2 > 0:
        den = np.convolve([T1, 1], [T2, 1])
    else:
        den = [T1, 1]
    plant = tf(num, den)
    if Td > 0:
        plant = plant * tf([Td, 1], [1])
    if Tu > 0:
        plant = plant * tf([1], [Tu, 1])
    if Tg > 0:
        plant = plant * tf([1], [Tg, 1])
    
    pid = tf([Kd, Kp, Ki], [1, 0])
    closed_loop = feedback(pid * plant, 1)
    t = np.linspace(0, 50, 500)
    t, y = step(closed_loop, T=t)

    # Rise time (time from 10% to 90% of final value)
    y_final = y[-1]
    y_10 = 0.1 * y_final
    y_90 = 0.9 * y_final
    try:
        t_rise_start = next(t[i] for i in range(len(y)) if y[i] >= y_10)
        t_rise_end = next(t[i] for i in range(len(y)) if y[i] >= y_90)
        rise_time = t_rise_end - t_rise_start
    except StopIteration:
        rise_time = 50  # max time if response too slow

    # Steady-state error
    ss_error = abs(1 - y[-1])
    return rise_time, ss_error

# Compute and add to DataFrame
print("Computing rise time and steady-state error...")
df[["rise_time", "ss_error"]] = df.apply(
    lambda row: pd.Series(compute_rise_time_and_ss_error(
        row.K, row.T1, row.T2, row.Td, row.Tu, row.Tg,
        row.Kp, row.Ki, row.Kd
    )), axis=1)


# === 3. Define features and target ===
features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Kp", "Ki", "Kd", "rise_time", "ss_error"] + list(type_dummies.columns)
target = "ISE"
print("Features used for training:", features)

X = df[features]
y = df[target]

print("Feature columns in X:")
print(X.columns.tolist())

# === 4. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Train MLPRegressor as surrogate model ===
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=5000, random_state=42)
# mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.001, learning_rate_init=0.001, max_iter=5000)

mlp.fit(X_train, y_train)

# === 6. Evaluate ===
y_pred = mlp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

# Plot true vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("True ISE")
plt.ylabel("Predicted ISE")
plt.title("Surrogate Model: True vs Predicted ISE")
plt.grid(True)
plt.tight_layout()

# Save model and plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = fr"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "true_vs_predicted_ise.png"))
joblib.dump(mlp, os.path.join(output_dir, "mlp_surrogate_model.pkl"))

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"MAE: {mae}\nMSE: {mse}\nR²: {r2}\n")
# Save the full dataset with computed ISE values
df.to_csv(os.path.join(output_dir, "surrogate_training_data_with_ise.csv"), index=False)

print(f"Model and results saved in {output_dir}")
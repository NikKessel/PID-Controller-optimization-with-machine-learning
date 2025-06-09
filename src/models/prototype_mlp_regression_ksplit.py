import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import joblib

# Load data
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = df[df["Label"] == "good"].copy()

# Feature and target setup
core_features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Overshoot"]
type_features = [col for col in df.columns if col.startswith("type_")]
features = core_features + type_features

X = df[features].fillna(0)
print("🚨 Final Features used for training:")
print(X.columns.tolist())
y_kp = df["Kp"]
y_ki = df["Ki"]
y_kd = df["Kd"]

# Split for all three targets with consistent shuffling
X_train, X_test, y_kp_train, y_kp_test = train_test_split(X, y_kp, test_size=0.2, random_state=42)
_, _, y_ki_train, y_ki_test = train_test_split(X, y_ki, test_size=0.2, random_state=42)
_, _, y_kd_train, y_kd_test = train_test_split(X, y_kd, test_size=0.2, random_state=42)

# Create and train pipelines
model_kp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42))
model_ki = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42))
model_kd = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42))

model_kp.fit(X_train, y_kp_train)
model_ki.fit(X_train, y_ki_train)
model_kd.fit(X_train, y_kd_train)
# === Train combined multi-output model ===

Y_multi = df[["Kp", "Ki", "Kd"]]
X = df[features].fillna(0)  # Re-define X for clarity

X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(X, Y_multi, test_size=0.2, random_state=42)

multi_output_model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42))
)
multi_output_model.fit(X_train_m, Y_train_m)

# Predict
Y_pred_m = multi_output_model.predict(X_test_m)

# Predictions
pred_kp = model_kp.predict(X_test)
pred_ki = model_ki.predict(X_test)
pred_kd = model_kd.predict(X_test)

# Metrics
mae_kp = mean_absolute_error(y_kp_test, pred_kp)
mae_ki = mean_absolute_error(y_ki_test, pred_ki)
mae_kd = mean_absolute_error(y_kd_test, pred_kd)
mse_kp = mean_squared_error(y_kp_test, pred_kp)
mse_ki = mean_squared_error(y_ki_test, pred_ki)
mse_kd = mean_squared_error(y_kd_test, pred_kd)
r2_kp = r2_score(y_kp_test, pred_kp)
r2_ki = r2_score(y_ki_test, pred_ki)
r2_kd = r2_score(y_kd_test, pred_kd)

# Weighted loss example
weighted_mae = 1.0 * mae_kp + 1.0 * mae_ki + 0.1 * mae_kd

# Output
output_dir = f"D:/BA/PID-Controller-optimization-with-machine-learning/models/mlp/pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model_kp, os.path.join(output_dir, "model_kp.joblib"))
joblib.dump(model_ki, os.path.join(output_dir, "model_ki.joblib"))
joblib.dump(model_kd, os.path.join(output_dir, "model_kd.joblib"))

mae_all = mean_absolute_error(Y_test_m, Y_pred_m, multioutput='raw_values')
r2_all = r2_score(Y_test_m, Y_pred_m, multioutput='raw_values')

metrics_multi = pd.DataFrame({
    "Target": ["Kp", "Ki", "Kd"],
    "MAE": mae_all,
    "R2": r2_all
})
metrics_multi.to_csv(os.path.join(output_dir, "metrics_multi_output.csv"), index=False)

joblib.dump(multi_output_model, os.path.join(output_dir, "model_multi_output.joblib"))

metrics = pd.DataFrame({
    "Target": ["Kp", "Ki", "Kd"],
    "MAE": [mae_kp, mae_ki, mae_kd],
    "MSE": [mse_kp, mse_ki, mse_kd],
    "R2": [r2_kp, r2_ki, r2_kd]
})
metrics.to_csv(os.path.join(output_dir, "weighted_metrics.csv"), index=False)
# === Prediction vs True value plots ===
plt.figure(figsize=(15, 4))

# --- Kp ---
plt.subplot(1, 3, 1)
plt.scatter(y_kp_test, pred_kp, alpha=0.4, color='tab:blue')
plt.plot([y_kp_test.min(), y_kp_test.max()], [y_kp_test.min(), y_kp_test.max()], 'k--', linewidth=1)
plt.title("Kp: True vs Predicted")
plt.xlabel("True Kp")
plt.ylabel("Predicted Kp")
plt.grid(True)

# --- Ki ---
plt.subplot(1, 3, 2)
plt.scatter(y_ki_test, pred_ki, alpha=0.4, color='tab:green')
plt.plot([y_ki_test.min(), y_ki_test.max()], [y_ki_test.min(), y_ki_test.max()], 'k--', linewidth=1)
plt.title("Ki: True vs Predicted")
plt.xlabel("True Ki")
plt.ylabel("Predicted Ki")
plt.grid(True)

# --- Kd ---
plt.subplot(1, 3, 3)
plt.scatter(y_kd_test, pred_kd, alpha=0.4, color='tab:red')
plt.plot([y_kd_test.min(), y_kd_test.max()], [y_kd_test.min(), y_kd_test.max()], 'k--', linewidth=1)
plt.title("Kd: True vs Predicted")
plt.xlabel("True Kd")
plt.ylabel("Predicted Kd")
plt.grid(True)

plt.suptitle("Regression Results: True vs Predicted Values")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.savefig(os.path.join(output_dir, "regression_plots.png"), dpi=300)

weighted_mae, output_dir

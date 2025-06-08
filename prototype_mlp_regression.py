import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1. Load and prepare data ===
df = pd.read_csv(r"D:/BA/PID-Controller-optimization-with-machine-learning/pid_dataset_control.csv")
df = df[df["Label"] == "good"].copy()

core_features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Overshoot"]
type_features = [col for col in df.columns if col.startswith("type_")]
features = core_features + type_features
targets = ["Kp", "Ki", "Kd"]

X = df[features].fillna(0)
y = df[targets]

# === 2. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Define model pipeline ===
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                   max_iter=1000, early_stopping=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), MultiOutputRegressor(mlp))

# === 4. Train model ===
pipeline.fit(X_train, y_train)

# === 5. Predict and evaluate ===
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R^2:", r2)

# === 6. Residual plots ===
output_dir = f"D:/BA/PID-Controller-optimization-with-machine-learning/models/mlp/pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

for i, target in enumerate(targets):
    residuals = y_test[target].values - y_pred[:, i]
    plt.figure()
    plt.scatter(y_test[target].values, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residual Plot for {target}")
    plt.xlabel(f"True {target}")
    plt.ylabel("Residual (True - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"residual_plot_{target}.png"))
    plt.close()

# === 7. True vs Predicted plots ===
plt.figure(figsize=(16, 4))
for i, target in enumerate(targets):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test[target].values, y_pred[:, i], alpha=0.7)
    plt.plot([y_test[target].min(), y_test[target].max()], [y_test[target].min(), y_test[target].max()], 'r--')
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"True vs Predicted {target}")
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_vs_predicted_all.png"))
plt.close()

# === 8. Save model and metrics ===
joblib.dump(pipeline, os.path.join(output_dir, "mlp_pid_model.pkl"))

with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
    f.write(",MAE,MSE,R2\n")
    for i, target in enumerate(targets):
        f.write(f"{target},{mean_absolute_error(y_test[target], y_pred[:, i]):.6f},"
                f"{mean_squared_error(y_test[target], y_pred[:, i]):.6f},"
                f"{r2_score(y_test[target], y_pred[:, i]):.6f}\n")

print(f"✅ MLP PID model saved to: {output_dir}")

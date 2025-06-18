import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
import seaborn as sns

# === 1. Load and prepare data (simplified for new column structure) ===
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_dataset_pidtune.csv")

df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])

# Optional filter if label column exists
if "Label" in df.columns:
    df = df[df["Label"] == "good"].copy()

# === 1b. Feature Engineering ===
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)  # avoid division by zero
df["K_T1"] = df["K"] * df["T1"]

# Check if Td exists
if "Td" not in df.columns:
    df["Td"] = 0.0
df["log_Td"] = np.log1p(df["Td"])

# Feature set and targets
features = ["K", "T1", "T2", "Td"]
targets = ["Kp", "Ki", "Kd"]
for target in ["Kp", "Ki", "Kd"]:
    df = df[df[target] <= 100]
X = df[features].fillna(0)
y = df[targets]

# === 2. Visualization ===
for target in targets:
    plt.figure()
    sns.histplot(df[target], bins=40, kde=True)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Grid Search for MLP Hyperparameters ===
base_mlp = MLPRegressor(max_iter=2000, early_stopping=True, random_state=42)
wrapped = MultiOutputRegressor(base_mlp)
pipeline = make_pipeline(StandardScaler(), wrapped)

param_grid = {
    'multioutputregressor__estimator__hidden_layer_sizes': [(128, 64, 32), (256, 128, 64)],
    'multioutputregressor__estimator__alpha': [1e-5, 1e-4, 1e-3],
    'multioutputregressor__estimator__learning_rate': ['constant', 'adaptive'],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best Params: {grid.best_params_}")
pipeline = grid.best_estimator_

# === 5. Retrain with TransformedTargetRegressor ===
mlp = MLPRegressor(
    hidden_layer_sizes=grid.best_params_['multioutputregressor__estimator__hidden_layer_sizes'],
    activation='relu',
    solver='adam',
    learning_rate=grid.best_params_['multioutputregressor__estimator__learning_rate'],
    alpha=grid.best_params_['multioutputregressor__estimator__alpha'],
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)

pipeline = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(
        TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    )
)
pipeline.fit(X_train, y_train)

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"Cross-validated R²: Mean = {np.mean(cv_scores):.4f}, Std = {np.std(cv_scores):.4f}")

# === 6. Evaluation ===
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

# === 7. Save Plots and Model ===
#output_dir = rf"D:\BA\PID-Controller-optimization-with-machine-learning\models\mlp\pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\mlp\pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\mlp\pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Residual plots
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

# True vs predicted plots
plt.figure(figsize=(16, 4))
for i, target in enumerate(targets):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test[target].values, y_pred[:, i], alpha=0.7)
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()], 'r--')
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"True vs Predicted {target}")
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_vs_predicted_all.png"))
plt.close()

# Save model
joblib.dump(pipeline, os.path.join(output_dir, "mlp_pid_model.pkl"))

# Save metrics
with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
    f.write(",MAE,MSE,R2\n")
    for i, target in enumerate(targets):
        f.write(f"{target},{mean_absolute_error(y_test[target], y_pred[:, i]):.6f},"
                f"{mean_squared_error(y_test[target], y_pred[:, i]):.6f},"
                f"{r2_score(y_test[target], y_pred[:, i]):.6f}\n")

print(f"✅ MLP PID model saved to: {output_dir}")

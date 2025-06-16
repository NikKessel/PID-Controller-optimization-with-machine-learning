import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from xgboost import XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# === 1. Load and preprocess data ===
#csv_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_dataset_control.csv"
#df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
#df = pd.read_csv(csv_path)

# One-hot encode the 'type' column
df = pd.get_dummies(df, columns=["type"])


# === 2. Define input features and targets ===
core_features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Overshoot"]
#core_features = ["K", "T1", "T2", "Td"]
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)  # avoid div by 0
df["K_T1"] = df["K"] * df["T1"]
df["log_Td"] = np.log1p(df["Td"])  # log(1 + Td)
type_features = [col for col in df.columns if col.startswith("type_")]
engineered_features: list[str] = ["T1_T2_ratio", "K_T1", "log_Td"]
#features = core_features + engineered_features + type_features
features = core_features + type_features
targets = ["Kp", "Ki", "Kd"]
df_good = df[df["Label"] == "good"].copy()

X = df_good[features].fillna(0)
y = df_good[targets]



# === 3. Train/Test Split ===

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Train XGBoost Model ===
xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model = MultiOutputRegressor(xgb_base)

param_grid = {
    "estimator__n_estimators": [100, 300, 500],
    "estimator__max_depth": [3, 5, 7],
    "estimator__learning_rate": [0.05, 0.1, 0.2],
    "estimator__subsample": [0.8, 1.0],
    "estimator__colsample_bytree": [0.8, 1.0],
    "estimator__reg_lambda": [0.1, 1.0]
}

grid = GridSearchCV(MultiOutputRegressor(XGBRegressor(random_state=42)), param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best parameters found:")
print(grid.best_params_)

model = grid.best_estimator_
#model.fit(X_train, y_train)

# === 5. Predictions ===


y_pred = model.predict(X_test)

# === 6. Metrics ===
metrics = {}
for i, var in enumerate(targets):
    y_true = y_test.iloc[:, i]
    y_hat = y_pred[:, i]
    metrics[var] = {
        "MAE": mean_absolute_error(y_true, y_hat),
        "MSE": mean_squared_error(y_true, y_hat),
        "R2": r2_score(y_true, y_hat)
    }

metrics_df = pd.DataFrame(metrics).T

# === 7. Save All Outputs ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#output_dir = fr"D:\BA\PID-Controller-optimization-with-machine-learning\models\xgboost\pid_model_{timestamp}"
output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\xgboost\pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

os.makedirs(output_dir, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(output_dir, "xgb_pid_model.pkl"))

# Save metrics
metrics_df.to_csv(os.path.join(output_dir, "xgb_performance_metrics.csv"))

# Save test predictions
pred_df = y_test.copy()
pred_df[["Kp_pred", "Ki_pred", "Kd_pred"]] = y_pred
pred_df.to_csv(os.path.join(output_dir, "xgb_test_predictions.csv"), index=False)

# === 8. True vs Predicted Plot for All ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, param in enumerate(["Kp", "Ki", "Kd"]):
    axes[i].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6)
    axes[i].plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                 'r--', label='Ideal')
    axes[i].set_xlabel(f"True {param}")
    axes[i].set_ylabel(f"Predicted {param}")
    axes[i].set_title(f"True vs. Predicted {param}")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "true_vs_predicted_all.png"))

# === 9. Residual Plots for Ki and Kd ===
residuals_ki = y_test["Ki"] - y_pred[:, 1]
residuals_kd = y_test["Kd"] - y_pred[:, 2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test["Ki"], residuals_ki, alpha=0.6)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_xlabel("True Ki")
axes[0].set_ylabel("Residual (True - Predicted)")
axes[0].set_title("Residual Plot for Ki")
axes[0].grid(True)

axes[1].scatter(y_test["Kd"], residuals_kd, alpha=0.6)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_xlabel("True Kd")
axes[1].set_ylabel("Residual (True - Predicted)")
axes[1].set_title("Residual Plot for Kd")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plots_ki_kd.png"))

# === 10. Feature Importance Plot ===
plt.figure(figsize=(6, 4))
plot_importance(model.estimators_[0], importance_type='gain', show_values=False)
plt.title("XGBoost Feature Importance (First Regressor)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgb_feature_importance.png"))

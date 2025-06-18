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
df = pd.read_csv(filepath_or_buffer=r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_dataset_pidtune.csv")
#df = pd.read_csv(csv_path)
print("Columns in df:", df.columns.tolist())
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])

# One-hot encode the 'type' column
#df = pd.get_dummies(df, columns=["type"])


# === 2. Define input features and targets ===
core_features = ["K", "T1", "T2"]
#core_features = ["K", "T1", "T2", "Td"]
#df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)  # avoid div by 0
#df["K_T1"] = df["K"] * df["T1"]
#df["log_Td"] = np.log1p(df["Td"])  # log(1 + Td)
type_features = [col for col in df.columns if col.startswith("type_")]
engineered_features: list[str] = ["T1_T2_ratio", "K_T1", "log_Td"]
#features = core_features + engineered_features + type_features
features = core_features 
targets = ["Kp", "Ki", "Kd"]
#df_good = df[df["Label"] == "good"].copy()
for target in ["Kp", "Ki", "Kd"]:
    df = df[df[target] <= 100]
epsilon = 1e-6

df["Kp_log"] = np.log10(df["Kp"] + epsilon)
df["Ki_log"] = np.log10(df["Ki"] + epsilon)
df["Kd_log"] = np.log10(df["Kd"] + epsilon)

# Use these as targets
targets = ["Kp_log", "Ki_log", "Kd_log"]
df_good = df
X = df_good[features]
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


#y_pred = model.predict(X_test)
# After model.predict()
y_pred_log = model.predict(X_test)

# Convert back to original scale
y_pred = np.power(10, y_pred_log)  # or: 10 ** y_pred_log
y_true = np.power(10, y_test.to_numpy())

# === 6. Metrics ===
metrics = {}
for i, var in enumerate(["Kp", "Ki", "Kd"]):  # original names
    y_hat = y_pred[:, i]
    y_real = y_true[:, i]
    metrics[var] = {
        "MAE": mean_absolute_error(y_real, y_hat),
        "MSE": mean_squared_error(y_real, y_hat),
        "R2": r2_score(y_real, y_hat)
    }

metrics_df = pd.DataFrame(metrics).T

# === 7. Save All Outputs ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = fr"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\xgboost\pid_model_{timestamp}"
#output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\xgboost\pid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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


plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plots_ki_kd.png"))

# === 10. Feature Importance Plot ===
plt.figure(figsize=(6, 4))
plot_importance(model.estimators_[0], importance_type='gain', show_values=False)
plt.title("XGBoost Feature Importance (First Regressor)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgb_feature_importance.png"))

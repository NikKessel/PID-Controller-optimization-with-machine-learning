import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1. Load Data ===
csv_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_dataset_control.csv"
df = pd.read_csv(csv_path)

# === 2. Filter 'good' labels and define features/targets ===
df_good = df[df["Label"] == "good"].copy()
features = ["K", "T1", "T2", "Td"]
targets = ["Kp", "Ki", "Kd"]

X = df_good[features].fillna(0)   # NaNs → 0
y = df_good[targets]

# === 3. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Train Model ===
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

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

# === 7. Residual Plot for Kp ===
residuals_kp = y_test["Kp"] - y_pred[:, 0]
plt.figure(figsize=(6, 4))
plt.scatter(y_test["Kp"], residuals_kp, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("True Kp")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot for Kp")
plt.grid(True)
plt.tight_layout()

# === 8. Save All Results ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = fr"D:\BA\PID-Controller-optimization-with-machine-learning\models\randomforrest\pid_model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(output_dir, "rf_pid_model.pkl"))

# Save metrics
metrics_df.to_csv(os.path.join(output_dir, "rf_performance_metrics.csv"))

# Save residual plot
plt.savefig(os.path.join(output_dir, "residual_plot_kp.png"))

# Save test predictions
pred_df = y_test.copy()
pred_df[["Kp_pred", "Ki_pred", "Kd_pred"]] = y_pred
pred_df.to_csv(os.path.join(output_dir, "rf_test_predictions.csv"), index=False)

# === 9. Cross-Validation for Kp ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores_kp = cross_val_score(rf_cv_model, X, y["Kp"], cv=cv, scoring="r2")
print(f"Cross-Validation R² (Kp): {cv_scores_kp.mean():.3f} ± {cv_scores_kp.std():.3f}")

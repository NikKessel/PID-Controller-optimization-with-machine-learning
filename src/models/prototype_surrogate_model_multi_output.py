# Full updated script using MLPRegressor with preprocessing and one-hot encoding
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate"
#base_dir = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data"

output_dir = os.path.join(base_dir, f"surrogate_model_multi_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# === Load & Filter Dataset ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_pidtune.csv")
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])

if "Label" in df.columns:
    df = df[df["Label"] == "good"].copy()

# === Feature Engineering ===
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)
df["K_T1"] = df["K"] * df["T1"]
if "Td" not in df.columns:
    df["Td"] = 0.0
df["log_Td"] = np.log1p(df["Td"])

# === Target and Feature Setup ===
target_cols = ["ISE", "SSE", "RiseTime", "SettlingTime", "Overshoot"]
missing_targets = [col for col in target_cols if col not in df.columns]
if missing_targets:
    raise ValueError(f"Missing required target metrics: {missing_targets}")

numeric_features = ["K", "T1", "T2", "Td", "Kp", "Ki", "Kd"]
X = df[numeric_features]
Y = df[target_cols]

# === Preprocessing ===
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ]
)

# === Model ===
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(mlp))
])

# === Split, Train, Predict ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# === Metrics ===
mae = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')
metrics_summary = pd.DataFrame({
    "Target": target_cols,
    "MAE": mae,
    "R2": r2
})
metrics_summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

# === Save model ===
model_path = os.path.join(output_dir, "pid_mlp_surrogate_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# === Plotting ===
Y_test_reset = Y_test.reset_index(drop=True)
Y_pred_df = pd.DataFrame(Y_pred, columns=target_cols)

for target in target_cols:
    plt.figure()
    plt.scatter(Y_test_reset[target], Y_pred_df[target], alpha=0.6)
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target} Prediction Performance (MLP)")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{target.lower()}_prediction_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved for {target}: {plot_path}")

    plt.scatter(Y_test_reset[target], Y_pred_df[target], alpha=0.6)
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target} Prediction Performance (MLP)")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{target.lower()}_prediction_plot.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved for {target}: {plot_path}")
    plt.close()


metrics_summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
# Save plot
plt.scatter(Y_test_reset["ISE"], Y_pred_df["ISE"], alpha=0.6)
plt.xlabel("True ISE")
plt.ylabel("Predicted ISE")
plt.title("ISE Prediction Performance (MLP)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "ise_prediction_plot.png"))
print(f"✅ Plot saved to {output_dir}")
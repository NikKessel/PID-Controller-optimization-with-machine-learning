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
#base_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate"
base_dir = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data"

output_dir = os.path.join(base_dir, f"surrogate_model_multi_{timestamp}")
os.makedirs(output_dir, exist_ok=True)


# Load dataset
#ile_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv"
file_path = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv"
df = pd.read_csv(file_path)
print("✅ Loaded CSV with shape:", df.shape)
print("Columns:", df.columns.tolist())
# Filter for valid and labeled as "good"
# Compute SSE_log first
min_pos_sse = df.loc[df["SSE"] > 0, "SSE"].min()
df["SSE_log"] = np.log1p(df["SSE"].apply(lambda x: min_pos_sse if x <= 0 else x))

# Then filter and keep SSE_log
df_clean = df[df["Label"] == "good"].copy()

# Now drop rows with missing values in targets (including SSE_log)
target_cols = ["ISE", "SSE_log",  "Overshoot", "settling_time",  "rise_time"]
df_clean = df_clean.dropna(subset=target_cols)

print("✅ After filtering 'good' labels:", df_clean.shape)
# Features
numeric_features = ["K", "T1", "T2", "Td", "Tu", "Tg", "Kp", "Ki", "Kd"]
categorical_features = ["type"]
all_features = numeric_features + categorical_features
X = df_clean[all_features]
Y = df_clean[target_cols]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define model pipeline with MLP
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(mlp))
])

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')

metrics_summary = pd.DataFrame({
    "Target": target_cols,
    "MAE": mae,
    "R2": r2
})

print("✅ Checking for target columns:", target_cols)
missing_targets = [col for col in target_cols if col not in df_clean.columns]
if missing_targets:
    print("❌ Missing target columns:", missing_targets)
    raise ValueError("Missing required target metrics.")

# Save model
model_path = os.path.join(output_dir, "pid_mlp_surrogate_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Display performance summary

# === Plotting prediction performance for all target metrics ===
for target in target_cols:
    plt.figure()
    # Predict and evaluate
    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
    r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')

    # Predict once before the loop
    Y_pred = model.predict(X_test)
    Y_test_reset = Y_test.reset_index(drop=True)
    Y_pred_df = pd.DataFrame(Y_pred, columns=target_cols)

    # If using SSE_log, compute real SSE once
    if "SSE_log" in target_cols:
        Y_pred_df["SSE"] = np.expm1(Y_pred_df["SSE_log"])
        Y_test_reset["SSE"] = np.expm1(Y_test_reset["SSE_log"])
        sse_mae = mean_absolute_error(Y_test_reset["SSE"], Y_pred_df["SSE"])
        sse_r2 = r2_score(Y_test_reset["SSE"], Y_pred_df["SSE"])
        print(f"✅ SSE performance → MAE: {sse_mae:.4f}, R²: {sse_r2:.4f}")

    # === Plotting each target ===
    for target in target_cols:
        plt.figure()
        plt.scatter(Y_test_reset[target], Y_pred_df[target], alpha=0.6)
        plt.xlabel(f"True {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target} Prediction Performance (MLP)")
        plt.grid(True)
        plot_path = os.path.join(output_dir, f"{target.lower()}_prediction_plot.png")
        plt.savefig(plot_path)
        print(f"✅ Plot saved for {target}: {plot_path}")
        plt.close()


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
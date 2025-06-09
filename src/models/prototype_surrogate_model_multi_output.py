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

# Load dataset
file_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv"
df = pd.read_csv(file_path)

# Filter for valid and labeled as "good"
df_clean = df[df["Label"] == "good"].copy()
target_cols = ["ISE", "Overshoot", "SettlingTime", "SSE", "RiseTime"]
df_clean = df_clean.dropna(subset=target_cols)

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

# Save model
joblib.dump(model, "/mnt/data/pid_mlp_surrogate_model.pkl")

# Display performance summary
import ace_tools as tools; tools.display_dataframe_to_user(name="MLP Surrogate Model Performance", dataframe=metrics_summary)

# Optional: Plot ISE predictions
Y_test_reset = Y_test.reset_index(drop=True)
Y_pred_df = pd.DataFrame(Y_pred, columns=target_cols)

plt.scatter(Y_test_reset["ISE"], Y_pred_df["ISE"], alpha=0.6)
plt.xlabel("True ISE")
plt.ylabel("Predicted ISE")
plt.title("ISE Prediction Performance (MLP)")
plt.grid(True)
plt.show()

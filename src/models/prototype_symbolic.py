
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# === Load Data ===
print("🔄 Loading dataset...")
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
#df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_multi_ranges.csv")

df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd"])
print(f"✅ Dataset loaded with {len(df)} entries.")

# === Feature Engineering ===
df["K_T1"] = df["K"] * df["T1"]
df["K_T2"] = df["K"] * df["T2"]
df["T1_T2_ratio"] = df["T1"] / (df["T2"] + 1e-3)

# === Log-transform targets ===
df["Kp_log"] = np.log10(df["Kp"] + 1e-6)
df["Ki_log"] = np.log10(df["Ki"] + 1e-6)
df["Kd_log"] = np.log10(df["Kd"] + 1e-6)

# === Use subset of data for efficiency ===
if len(df) > 3000:
    df = df.sample(n=3000, random_state=42)

features = ["K", "T1", "T2", "K_T1", "K_T2", "T1_T2_ratio"]
targets = ["Kp_log", "Ki_log", "Kd_log"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"D:\BA\PID-Controller-optimization-with-machine-learning\models\Symbolic\symbolic_model_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory created: {output_dir}")

results = {}

for target in targets:
    param = target.replace("_log", "")
    print(f"🚀 Fitting Symbolic Regression model for {param}...")

    df_filtered = df[df[target] <= 2].dropna(subset=features + [target])
    X = df_filtered[features]
    y = df_filtered[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = PySRRegressor(
        model_selection="best",
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt"],
        extra_sympy_mappings={"sqrt": lambda x: x**0.5},
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        progress=True,
        temp_equation_file=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(10**y_test, 10**y_pred)
    r2 = r2_score(10**y_test, 10**y_pred)

    results[param] = {
        "Best Equation": str(model.get_best()),
        "R2": r2,
        "MAE": mae
    }

    # Save plot
    plt.figure(figsize=(7, 5))
    plt.scatter(10**y_test, 10**y_pred, alpha=0.6)
    plt.plot([min(10**y_test), max(10**y_test)], [min(10**y_test), max(10**y_test)], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Symbolic Regression: {param}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param}_true_vs_predicted.png"))
    plt.close()
        # === Save Trained Model ===
    model_path = os.path.join(output_dir, f"{param}_symbolic_model.pkl")
    joblib.dump(model, model_path)

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(output_dir, "symbolic_metrics.csv"))

print("✅ Symbolic regression complete. Results saved.")
print(f"📁 Results in: {output_dir}")

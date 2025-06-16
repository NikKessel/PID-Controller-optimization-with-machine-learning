import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import differential_evolution
import joblib

# === User Input via Prompt ===
import builtins
input = builtins.input  # workaround for notebook prompt

K = float(input("Enter system gain K: "))
T1 = float(input("Enter time constant T1: "))
T2 = float(input("Enter time constant T2: "))
Td = float(input("Enter dead time Td: "))

plant_params = {"K": K, "T1": T1, "T2": T2, "Td": Td}

# === Performance Metric Weights ===
weight_vector = {
    "ISE": 0.4,
    "Overshoot": 0.2,
    "SettlingTime": 0.2,
    "RiseTime": 0.2
}

# === PID Parameter Bounds ===
bounds = [(0.01, 5), (0.001, 2), (0.001, 2)]  # Kp, Ki, Kd

# === Load Surrogate Model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_multi_20250616_213302\pid_mlp_surrogate_model.pkl"
surrogate_model = joblib.load(model_path)

# === Create Export Directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_dir = os.path.join(
    r"D:\BA\PID-Controller-optimization-with-machine-learning\results",
    f"run_{timestamp}"
)
os.makedirs(export_dir, exist_ok=True)

# === Cost Function ===
def cost_function(pid_params):
    Kp, Ki, Kd = pid_params
    input_df = pd.DataFrame([{
        **plant_params,
        "Kp": Kp, "Ki": Ki, "Kd": Kd
    }])
    prediction = surrogate_model.predict(input_df)[0]
    ISE, Overshoot, SettlingTime, RiseTime = prediction[:4]
    if any(np.isnan(prediction)) or any(x < 0 for x in prediction[:4]):
        return np.inf
    cost = (weight_vector["ISE"] * ISE +
            weight_vector["Overshoot"] * Overshoot +
            weight_vector["SettlingTime"] * SettlingTime +
            weight_vector["RiseTime"] * RiseTime)
    return cost

# === Optimization ===
results = differential_evolution(cost_function, bounds, maxiter=30, popsize=20, seed=42)
Kp_opt, Ki_opt, Kd_opt = results.x

# === Log All Solutions ===
evaluated_population = []
for sol in results.population:
    Kp, Ki, Kd = sol
    input_df = pd.DataFrame([{
        **plant_params,
        "Kp": Kp, "Ki": Ki, "Kd": Kd
    }])
    pred = surrogate_model.predict(input_df)[0]
    ISE, Overshoot, SettlingTime, RiseTime = pred[:4]
    total_cost = cost_function([Kp, Ki, Kd])
    if not np.isinf(total_cost):
        evaluated_population.append({
            "Kp": Kp, "Ki": Ki, "Kd": Kd,
            "ISE": ISE, "Overshoot": Overshoot,
            "SettlingTime": SettlingTime, "RiseTime": RiseTime,
            "Cost": total_cost
        })

df_results = pd.DataFrame(evaluated_population)
df_results.to_csv(os.path.join(export_dir, "all_solutions.csv"), index=False)

# === Extract Best Results ===
top_5 = df_results.sort_values("Cost").head(5)
top_5.to_csv(os.path.join(export_dir, "top_5_controllers.csv"), index=False)

# === Save Best Per Metric ===
best_per_metric = {
    "best_ise": df_results.loc[df_results["ISE"].idxmin()],
    "best_overshoot": df_results.loc[df_results["Overshoot"].idxmin()],
    "best_settling": df_results.loc[df_results["SettlingTime"].idxmin()],
    "best_rise": df_results.loc[df_results["RiseTime"].idxmin()]
}
best_df = pd.DataFrame(best_per_metric).T
best_df.to_csv(os.path.join(export_dir, "best_per_metric.csv"))

# === Pareto Front ===
pareto_df = df_results.sort_values(["ISE", "Overshoot"])
pareto_set = [pareto_df.iloc[0]]
for _, row in pareto_df.iterrows():
    if row["Overshoot"] < pareto_set[-1]["Overshoot"]:
        pareto_set.append(row)
pareto_df = pd.DataFrame(pareto_set)
pareto_df.to_csv(os.path.join(export_dir, "pareto_front.csv"), index=False)

# === Visualization ===
plt.figure()
plt.hist(df_results["ISE"], bins=30)
plt.title("Histogram of ISE")
plt.xlabel("ISE")
plt.ylabel("Count")
plt.savefig(os.path.join(export_dir, "hist_ise.png"))

plt.figure()
plt.scatter(df_results["ISE"], df_results["Overshoot"], c=df_results["Cost"], cmap="viridis")
plt.xlabel("ISE")
plt.ylabel("Overshoot")
plt.title("Pareto Scatter (ISE vs Overshoot)")
plt.colorbar(label="Cost")
plt.savefig(os.path.join(export_dir, "pareto_scatter.png"))

for param in ["Kp", "Ki", "Kd"]:
    plt.figure()
    plt.scatter(df_results[param], df_results["Cost"])
    plt.xlabel(param)
    plt.ylabel("Cost")
    plt.title(f"{param} vs Cost")
    plt.savefig(os.path.join(export_dir, f"{param}_vs_cost.png"))


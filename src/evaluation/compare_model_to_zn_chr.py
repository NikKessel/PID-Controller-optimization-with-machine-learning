import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import joblib
import os
from datetime import datetime
from scipy.optimize import differential_evolution

# === 1. Load Data ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = df.dropna(subset=["K", "T1", "T2", "Tu", "Tg", "type"]).head(10)
surrogate_model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\surrogate\surrogate_model_20250609_182908\pid_mlp_surrogate_model.pkl"
surrogate_model = joblib.load(surrogate_model_path)
all_results = []


# === 2. Define ZN and CHR formulas ===
def zn_pid(K, Tu, Tg):
    Kp = 1.2 * Tg / (K * Tu)
    Ki = Kp / (2 * Tu)
    Kd = Kp * Tu / 2
    return Kp, Ki, Kd

def chr_pid(K, Tu, Tg):  # Regulation tuning
    Kp = 0.6 * Tg / (K * Tu)
    Ki = Kp / Tu
    Kd = 0.5 * Tu * Kp
    return Kp, Ki, Kd

def optimize_pid_with_surrogate(K, T1, T2, Td, Tu, Tg):
    plant_params = (K, T1, T2, Td, Tu, Tg)

    def cost_fn(params):
        Kp, Ki, Kd = params
        input_vec = pd.DataFrame([{
            "K": K, "T1": T1, "T2": T2, "Td": Td, "Tu": Tu, "Tg": Tg,
            "Kp": Kp, "Ki": Ki, "Kd": Kd
            # ← keep if one-hot encoding is inside pipeline
        }])

        #prediction = surrogate_model.predict(input_df)[0]
        prediction = surrogate_model.predict(input_vec)[0]

        if any(p < 0 for p in prediction):  # Handle invalid values
            return 1e6

        # Weighted sum of metrics: ISE + Overshoot + Settling + Rise
        weights = [0.4, 0.2, 0.2, 0.2]
        return np.dot(weights, prediction)

    bounds = [(0.1, 10), (0.01, 10), (0, 2)]
    result = differential_evolution(cost_fn, bounds, maxiter=50, popsize=20, seed=42)


    return result.x  # returns [Kp_opt, Ki_opt, Kd_opt]

# === 3. Load ML model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\mlp\pid_model_20250609_194935\model_multi_output.joblib"
ml_model = joblib.load(model_path)

# === 4. Define evaluation + simulation ===
def simulate_pid(Kp, Ki, Kd, plant, t):
    pid = tf([Kd, Kp, Ki], [1, 0])
    system = feedback(pid * plant, 1)
    t, y = step(system, T=t)
    return t, y

def extract_metrics(t, y):
    y_final = y[-1]
    overshoot = (np.max(y) - y_final) / y_final * 100
    rise_time = t[np.where(y >= 0.9 * y_final)[0][0]]
    settling_time = t[np.where(np.abs(y - y_final) < 0.05 * y_final)[0][-1]]
    ise = np.sum((1 - y)**2) * (t[1] - t[0])
    return rise_time, overshoot, settling_time, ise

def simulate_and_extract_metrics(plant, Kp, Ki, Kd):
    t = np.linspace(0, 50, 1000)
    t, y = simulate_pid(Kp, Ki, Kd, plant, t)
    rise, overshoot, settling, ise = extract_metrics(t, y)
    return {
        "RiseTime": rise,
        "Overshoot": overshoot,
        "SettlingTime": settling,
        "ISE": ise
    }


# === 5. Loop and evaluate ===
results = []
t = np.linspace(0, 50, 1000)

for idx, row in df.iterrows():
    K, T1, T2, Td, Tu, Tg = row["K"], row["T1"], row["T2"], row["Td"], row["Tu"], row["Tg"]

    # Define plant
    num = [K]
    den = [T1, 1]
    if pd.notna(T2):
        den = np.convolve(den, [T2, 1])
    plant = tf(num, den)
    if pd.notna(Td):
        plant = plant * tf([1], [1, Td])

    # ML prediction
    # Encode type as one-hot vector (based on training time)
    type_encoding = {
        "type_PT1": 0,
        "type_PT2": 0,
        "type_PT1+Td": 0,
        "type_PT2+Td": 0,
    }
    type_key = f"type_{row['type']}"
    if type_key in type_encoding:
        type_encoding[type_key] = 1
    else:
        print(f"⚠️ Unknown type '{row['type']}', using zeros for type encoding.")

    input_features = {
    "K": K,
    "T1": T1,
    "T2": T2 if pd.notna(T2) else 0,
    "Td": Td if pd.notna(Td) else 0,
    "Tu": Tu,
    "Tg": Tg,
    "Overshoot": 0.0  # or an estimated value
    }
    input_df = pd.DataFrame([input_features])



    print("=== DEBUG INPUT TO ML MODEL ===")
    print("Shape:", input_df.shape)
    print("Columns:", input_df.columns.tolist())
    print("Values:\n", input_df.values)


    # If you have access to the pipeline's expected features:
    #try:
        #expected_n_features = ml_model.named_steps['preprocessor'].transformers_[0][2]
        #print("Expected feature names by ColumnTransformer:", expected_n_features)
    #except Exception as e:
        #print("Could not extract expected features from pipeline:", e)

    Kp_ml, Ki_ml, Kd_ml = ml_model.predict(input_df)[0]
    t_ml, y_ml = simulate_pid(Kp_ml, Ki_ml, Kd_ml, plant, t)
    m_ml = extract_metrics(t_ml, y_ml)

    # ZN prediction
    Kp_zn, Ki_zn, Kd_zn = zn_pid(K, Tu, Tg)
    t_zn, y_zn = simulate_pid(Kp_zn, Ki_zn, Kd_zn, plant, t)
    m_zn = extract_metrics(t_zn, y_zn)

    # CHR prediction
    Kp_chr, Ki_chr, Kd_chr = chr_pid(K, Tu, Tg)
    t_chr, y_chr = simulate_pid(Kp_chr, Ki_chr, Kd_chr, plant, t)
    m_chr = extract_metrics(t_chr, y_chr)

    # === GA-Surrogate Method ===
    try:
        Kp_ga, Ki_ga, Kd_ga = optimize_pid_with_surrogate(K, T1, T2, Td, Tu, Tg)
        #metrics_ga = simulate_and_extract_metrics(Gs, Kp_ga, Ki_ga, Kd_ga)
        metrics_ga = simulate_and_extract_metrics(plant, Kp_ga, Ki_ga, Kd_ga)

        all_results.append({
            "System": idx,
            "Method": "GA",
            "Kp": Kp_ga, "Ki": Ki_ga, "Kd": Kd_ga,
            "ISE": metrics_ga["ISE"],
            "Overshoot": metrics_ga["Overshoot"],
            "SettlingTime": metrics_ga["SettlingTime"],
            "RiseTime": metrics_ga["RiseTime"],
        })
    except Exception as e:
        print(f"❌ Error optimizing PID for system {idx}: {e}")

        # Store results
    results.append({
        "System": idx,
        "ML_Metrics": m_ml,
        "ZN_Metrics": m_zn,
        "CHR_Metrics": m_chr
    })

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(t_ml, y_ml, label="ML")
    plt.plot(t_zn, y_zn, label="ZN")
    plt.plot(t_chr, y_chr, label="CHR")
    plt.title(f"Step Response Comparison – System {idx}")
    plt.xlabel("Time [s]")
    plt.ylabel("Output y(t)")
    plt.legend()
    plt.grid(True)
    save_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\benchmark_plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"system_{idx}_comparison.png"))
    plt.close()

# === 6. Export results to CSV ===
out = pd.DataFrame([{**{"System": r["System"]},
                     **{f"ML_{k}": v for k, v in zip(["Rise", "OS", "Settling", "ISE"], r["ML_Metrics"])} ,
                     **{f"ZN_{k}": v for k, v in zip(["Rise", "OS", "Settling", "ISE"], r["ZN_Metrics"])} ,
                     **{f"CHR_{k}": v for k, v in zip(["Rise", "OS", "Settling", "ISE"], r["CHR_Metrics"])} }
                    for r in results])
out.to_csv(os.path.join(save_dir, "benchmark_results.csv"), index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import joblib
import os
from datetime import datetime

# === 1. Load Data ===
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_dataset_control.csv")
df = df.dropna(subset=["K", "T1", "T2", "Tu", "Tg", "type"]).head(10)

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

# === 3. Load ML model ===
model_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\models\xgboost\pid_model_20250608_194624\xgb_pid_model.pkl"
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
    type_encoded = [1 if row['type'] == t else 0 for t in ['PT1', 'PT2', 'PT1+Td', 'PT2+Td', 'Osc2']]
    overshoot_dummy = 0  # Use proper value or compute if needed
    input_vec = np.array([[K, T1, T2 if pd.notna(T2) else 0, Td if pd.notna(Td) else 0, Tu, Tg, overshoot_dummy] + type_encoded])
    Kp_ml, Ki_ml, Kd_ml = ml_model.predict(input_vec)[0]
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

import pandas as pd
import numpy as np
from control.matlab import tf, feedback, step, pade, series
import ast
from tqdm import tqdm


def evaluate_step_response_metrics(num, den, Kp, Ki, Kd, Td=0.0, t_end=200, n_points=1000, setpoint=1.0):
    try:
        print("=== Starting Metric Evaluation ===")

        # --- 1. Build G(s) ---
        G_nom = tf(num, den)
        print(f"Original G_nom(s): {G_nom}")

        if Td > 0:
            delay_num, delay_den = pade(Td, 4)
            G_delay = tf(delay_num, delay_den)
            G = series(G_nom, G_delay)
            print(f"Delay Td = {Td} applied with Padé: {G_delay}")
        else:
            G = G_nom
            print("No delay applied.")

        print(f"Final G(s): {G}")

        # --- 2. Build PID(s) ---
        s = tf([1, 0], [1])
        PID = Kp + Ki / s + Kd * s
        print(f"PID(s): {PID}")

        # --- 3. Closed-loop system ---
        sys_cl = feedback(PID * G, 1)
        print(f"Closed-loop TF: {sys_cl}")

        # --- 4. Simulate step response ---
        t = np.linspace(0, t_end, n_points)
        t_out, y_out = step(sys_cl, T=t)
        print(f"Step response simulated, final y = {y_out[-1]}")

        # --- 5. Compute error ---
        e = setpoint - y_out

        # --- 6. Metrics ---
        ise = np.trapz(e**2, t_out)
        print(f"ISE = {ise}")

        sse = abs(setpoint - y_out[-1])
        print(f"SSE = {sse}")

        overshoot = max(y_out) - setpoint
        print(f"Overshoot = {overshoot}")

        settling_idx = np.where(np.abs(e) < 0.02)[0]
        settling_time = t_out[settling_idx[-1]] if len(settling_idx) > 0 else t_out[-1]
        print(f"Settling Time = {settling_time}")

        try:
            rt_start = np.where(y_out >= 0.1 * setpoint)[0][0]
            rt_end = np.where(y_out >= 0.9 * setpoint)[0][0]
            rise_time = t_out[rt_end] - t_out[rt_start]
        except:
            rise_time = np.nan
        print(f"Rise Time = {rise_time}")

        return ise, overshoot, settling_time, sse, rise_time

    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

# Load dataset
df = pd.read_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv")
df = df[df["Label"] == "good"].copy()

# Initialize result columns
df["ISE"] = np.nan
df["SSE"] = np.nan
df["SettlingTime"] = np.nan
df["RiseTime"] = np.nan

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        num = ast.literal_eval(row["num"])
        den = ast.literal_eval(row["den"])
        Kp, Ki, Kd = row["Kp"], row["Ki"], row["Kd"]

        ise, overshoot, settling_time, sse, rise_time = evaluate_step_response_metrics(num, den, Kp, Ki, Kd)
        df.at[idx, "ISE"] = ise
        df.at[idx, "Overshoot"] = overshoot
        df.at[idx, "SettlingTime"] = settling_time
        df.at[idx, "SSE"] = sse
        df.at[idx, "RiseTime"] = rise_time

    except Exception as e:
        continue

# Save to new file
df.to_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_with_metrics.csv", index=False)
print("✅ Metrics computed and saved to pid_dataset_with_metrics.csv")

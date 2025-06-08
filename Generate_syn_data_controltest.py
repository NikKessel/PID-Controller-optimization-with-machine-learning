import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

from control import tf, feedback, step_response

# === CONFIG ===
num_samples = 100  # how many systems to generate
output_csv_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_dataset_control.csv"
plot_output_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\plots\visul_syn_data"
os.makedirs(plot_output_dir, exist_ok=True)

# === TRANSFER FUNCTION GENERATOR ===
def generate_transfer_function(type_name, sprunghoehe_target, sprungzeit_target):
    K = random.uniform(0.5, 3.0)
    T1 = T2 = Td = zeta = omega_n = None

    if type_name == "PT1":
        T1 = random.uniform(1, 50)
        num = [K]
        den = [T1, 1]
    elif type_name == "PT2":
        T1 = random.uniform(1, 50)
        T2 = random.uniform(1, 50)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])
    elif type_name == "PT1+Td":
        T1 = random.uniform(1, 50)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = [T1, 1]
    elif type_name == "PT2+Td":
        T1 = random.uniform(1, 50)
        T2 = random.uniform(1, 50)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])
    elif type_name == "Osc2":
        zeta = random.uniform(0.5, 0.9)
        omega_n = random.uniform(0.1, 2.0)
        num = [K * omega_n ** 2]
        den = [1, 2 * zeta * omega_n, omega_n ** 2]
    else:
        num = [K]
        den = [1]

    return {
        "type": type_name,
        "K": K,
        "T1": T1,
        "T2": T2,
        "Td": Td,
        "zeta": zeta,
        "omega_n": omega_n,
        "num": num,
        "den": den,
        "sprunghoehe_target": sprunghoehe_target,
        "sprungzeit_target": sprungzeit_target
    }

# === SIMULATION WITH PID CONTROLLER ===
def simulate_and_extract(num, den, Kp, Ki, Kd):
    s = tf([1, 0], [1])
    G = tf(num, den)
    C = Kp + Ki / s + Kd * s
    T = feedback(C * G, 1)

    t, y = step_response(T, T=np.linspace(0, 100, 1000))
    w = np.ones_like(t)
    e = w - y
    dt = t[1] - t[0]
    u = Kp * e + Ki * np.cumsum(e) * dt + Kd * np.gradient(e, dt)

    # Estimate metrics
    y_final = y[-1] if y[-1] != 0 else 1e-6
    overshoot = (np.max(y) - y_final) / y_final
    try:
        tg = t[np.where(y >= 0.63 * y_final)[0][0]]
    except IndexError:
        tg = np.nan
    try:
        tu = t[np.where(np.diff(np.sign(np.diff(y))) < 0)[0][0]]
    except IndexError:
        tu = np.nan

    return {
        "t": t,
        "y": y,
        "u": u,
        "w": w,
        "tg": tg,
        "tu": tu,
        "overshoot": overshoot
    }

# === MAIN GENERATION LOOP ===
sample_data = []

for i in range(num_samples):
    tf_types = ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"]
    tf_type = random.choice(tf_types)
    tf_data = generate_transfer_function(tf_type, sprunghoehe_target=1, sprungzeit_target=20)

    num, den = tf_data["num"], tf_data["den"]

    Kp = random.uniform(0.5, 3.0)
    Ki = random.uniform(0.01, 1.0)
    Kd = random.uniform(0.01, 1.0)

    try:
        sim_data = simulate_and_extract(num, den, Kp, Ki, Kd)

        # Store scalar values
        row = {
            **tf_data,
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "Tu": sim_data["tu"],
            "Tg": sim_data["tg"],
            "Overshoot": sim_data["overshoot"]
        }
        sample_data.append(row)

        # Optional: Save plot
        plt.figure(figsize=(6, 4))
        plt.plot(sim_data["t"], sim_data["w"], 'k--', label='w(t)')
        plt.plot(sim_data["t"], sim_data["y"], 'b', label='y(t)')
        plt.plot(sim_data["t"], sim_data["u"], 'r', label='u(t)')
        plt.title(f"Sample {i}")
        plt.xlabel("Time [s]")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, f"sample_{i}.png"))
        plt.close()

    except Exception as e:
        print(f"⚠️ Sample {i} failed: {e}")
        continue

# === SAVE TO CSV ===
df = pd.DataFrame(sample_data)
df.to_csv(output_csv_path, index=False)

print(df.head())
print(f"\n✅ Saved {len(df)} samples to {output_csv_path}")

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from control import tf, feedback, step_response
from control.matlab import pade

# === CONFIG ===
num_samples = 1000  # how many systems to generate
output_csv_path = r"D:\BA\PID-Controller-optimization-with-machine-learning\data\pid_dataset_control.csv"
plot_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\plots\visul_syn_data"
plot_output_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\plots\visul_syn_data"

os.makedirs(plot_output_dir, exist_ok=True)
# === Create subfolders for labeled plots ===
label_categories = ["good", "slow", "overshoot", "unstable"]
for cat in label_categories:
    os.makedirs(os.path.join(plot_output_dir, cat), exist_ok=True)


# === TRANSFER FUNCTION GENERATOR ===
def generate_transfer_function(type_name, sprunghoehe_target=None, sprungzeit_target=None):
    K = random.uniform(0.5, 3.0)
    T1 = T2 = Td = zeta = omega_n = None

    # Generate targets if not provided
    if sprunghoehe_target is None:
        sprunghoehe_target = random.uniform(0.8, 1.2)  # normalized setpoint

    if type_name == "PT1":
        T1 = random.uniform(1, 50)
        num = [K]
        den = [T1, 1]

    elif type_name == "PT2":
        T1 = random.uniform(5, 50)
        T2 = random.uniform(1, T1)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])

    elif type_name == "PT1+Td":
        T1 = random.uniform(1, 50)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = [T1, 1]
        num_delay, den_delay = pade(Td, 1)
        num = np.polymul(num, num_delay)
        den = np.polymul(den, den_delay)

    elif type_name == "PT2+Td":
        T1 = random.uniform(5, 50)
        T2 = random.uniform(1, T1)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])
        num_delay, den_delay = pade(Td, 1)
        num = np.polymul(num, num_delay)
        den = np.polymul(den, den_delay)

    elif type_name == "Osc2":
        zeta = random.uniform(0.5, 0.9)
        omega_n = random.uniform(0.1, 2.0)
        num = [K * omega_n ** 2]
        den = [1, 2 * zeta * omega_n, omega_n ** 2]

    else:
        num = [K]
        den = [1]
        # === Physically-informed rise time target ===
    if sprungzeit_target is None:
        if type_name == "PT1":
            sprungzeit_target = 2.2 * T1

        elif type_name == "PT2":
            if T1 is not None and T2 is not None:
                sprungzeit_target = 1.8 * (T1 + T2)
            else:
                sprungzeit_target = 10.0  # fallback

        elif type_name == "PT1+Td":
            sprungzeit_target = 2.2 * T1 + Td

        elif type_name == "PT2+Td":
            if T1 is not None and T2 is not None and Td is not None:
                sprungzeit_target = 1.8 * (T1 + T2) + Td
            else:
                sprungzeit_target = 10.0  # fallback

        elif type_name == "Osc2":
            if zeta is not None and omega_n is not None and zeta < 1.0:
                sprungzeit_target = np.pi / (omega_n * np.sqrt(1 - zeta**2))  # time to first peak
            else:
                sprungzeit_target = 10.0

        else:
            sprungzeit_target = 10.0  # safe default

        # Optional: Add 10% noise for variability
        sprungzeit_target *= random.uniform(0.9, 1.1)

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

def simulate_and_extract(num, den, Kp, Ki, Kd, setpoint):
    s = tf([1, 0], [1])
    G = tf(num, den)
    C = Kp + Ki / s + Kd * s
    T_sys = feedback(C * G, 1)

    t, y = step_response(T_sys, T=np.linspace(0, 200, 2000))
    w = np.ones_like(t) * setpoint  # use the actual step height
    e = w - y
    dt = t[1] - t[0]

    # Control effort (not used in metric but helpful for energy calc)
    u = Kp * e + Ki * np.cumsum(e) * dt + Kd * np.gradient(e, dt)

    # === Metrics ===
    y_final = y[-1] if y[-1] != 0 else 1e-6
    overshoot = (np.max(y) - y_final) / y_final

    # Rise time: 10% to 90% of final value
    try:
        t_10 = t[np.where(y >= 0.1 * y_final)[0][0]]
        t_90 = t[np.where(y >= 0.9 * y_final)[0][0]]
        rise_time = t_90 - t_10
    except IndexError:
        rise_time = np.nan

    # Settling time: time when y stays within ±2% of final value
    try:
        tol = 0.02 * y_final
        above = np.where(np.abs(y - y_final) > tol)[0]
        if len(above) == 0:
            settling_time = 0
        else:
            settling_time = t[above[-1]]  # last out-of-bound index
    except IndexError:
        settling_time = np.nan

    # tg: rise to 63% of final value
    try:
        tg = t[np.where(y >= 0.63 * y_final)[0][0]]
    except IndexError:
        tg = np.nan

    # tu: inflection point approximation via second derivative
    try:
        tu = t[np.where(np.diff(np.sign(np.diff(y))) < 0)[0][0]]
    except IndexError:
        tu = np.nan

    # ISE (Integral of Squared Error)
    ISE = np.sum(e**2) * dt

    # SSE (Steady State Error)
    SSE = e[-1]

    return {
        "t": t,
        "y": y,
        "u": u,
        "w": w,
        "tg": tg,
        "tu": tu,
        "overshoot": overshoot,
        "y_final": y_final,
        "ISE": ISE,
        "SSE": SSE,
        "settling_time": settling_time,
        "rise_time": rise_time
    }
def classify_sample(y, u, tg, tu, setpoint, sprungzeit_target):
    try:
        # Check for simulation instability
        if (
            np.any(np.isnan(y)) or np.any(np.isnan(u)) or
            np.any(np.isinf(y)) or np.any(np.isinf(u)) or
            np.isnan(tg) or np.isnan(tu)
        ):
            return "unstable"

        y_final = y[-1]
        # Overshoot compared to the step height, not y_final
        overshoot_pct = (np.max(y) - setpoint) / max(1e-6, setpoint)

        # Overshoot label
        if overshoot_pct > 1.0:
            return "overshoot"

        # Slow label (if response doesn't reach setpoint correctly)
        if y_final < 0.95 * setpoint or y_final > 1.05 * setpoint or tg > 1.5 * sprungzeit_target:
            return "slow"

        return "good"

    except:
        return "unstable"


# === MAIN GENERATION LOOP ===
sample_data = []

for i in range(num_samples):
    tf_types = ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"]
    tf_type = random.choice(tf_types)
    tf_data = generate_transfer_function(tf_type, sprunghoehe_target=None, sprungzeit_target=None)

    num, den = tf_data["num"], tf_data["den"]

    Kp = random.uniform(0.5, 3.0)
    Ki = random.uniform(0.01, 1.0)
    Kd = random.uniform(0.01, 1.0)

    try:
        sim_data = simulate_and_extract(num, den, Kp, Ki, Kd, setpoint=tf_data["sprunghoehe_target"])
        label = classify_sample(sim_data["y"], sim_data["u"], sim_data["tg"], sim_data["tu"], setpoint=tf_data["sprunghoehe_target"], sprungzeit_target=tf_data["sprungzeit_target"])

        # Store scalar values
        row = {
            **tf_data,
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "Tu": sim_data["tu"],
            "Tg": sim_data["tg"],
            "Overshoot": sim_data["overshoot"],
            "ISE": sim_data["ISE"],
            "SSE": sim_data["SSE"],
            "settling_time": sim_data["settling_time"],
            "rise_time": sim_data["rise_time"],
            "y_final": sim_data["y_final"],
            "Label": label
        }

        sample_data.append(row)

        # === Save combined step + control plot (only for first 10) ===
        if i < 10:
            label_dir = os.path.join(plot_output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            plt.figure(figsize=(10, 5))
            ax1 = plt.gca()

            # Step response
            ax1.plot(sim_data["t"], sim_data["y"], label="Output y(t)", color='blue')
            ax1.plot(sim_data["t"], sim_data["w"], '--', label="Setpoint w(t)", color='black')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Output", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Control effort
            ax2 = ax1.twinx()
            ax2.plot(sim_data["t"], sim_data["u"], label="Control u(t)", color='orange')
            ax2.set_ylabel("Control Effort", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            plt.title(f"Sample {i} - Label: {label}")
            plt.tight_layout()
            save_path = os.path.join(label_dir, f"sample_{i}_{label}.png")
            plt.savefig(save_path)
            print(f"✅ Saved: {save_path}")
            plt.close()

    except Exception as e:
        print(f"⚠️ Sample {i} failed: {e}")
        continue

# Save only first 10 plots
if i < 10:
    os.makedirs(os.path.join(plot_output_dir, label), exist_ok=True)

    # === Combined Plot: y(t), w(t), u(t) ===
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()

    # Plot step response
    ax1.plot(sim_data["t"], sim_data["y"], label="Output y(t)", color='blue', linewidth=2)
    ax1.plot(sim_data["t"], sim_data["w"], '--', label="Setpoint w(t)", color='black', linewidth=1.5)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Output", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot control effort on second y-axis
    ax2 = ax1.twinx()
    ax2.plot(sim_data["t"], sim_data["u"], label="Control u(t)", color='orange', linewidth=2, alpha=0.8)
    ax2.set_ylabel("Control Effort", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Title with label and sample index
    plt.title(f"Sample {i} - Label: {label}")
    fig = plt.gcf()
    fig.tight_layout()

    # Save the plot
    save_path = os.path.join(plot_output_dir, label, f"sample_{i}_{label}.png")
    plt.savefig(save_path)
    plt.close()


# === SAVE TO CSV ===
df = pd.DataFrame(sample_data)
df.to_csv(output_csv_path, index=False)

print(df.head())
print(f"\n✅ Saved {len(df)} samples to {output_csv_path}")

import numpy as np
import pandas as pd
import scipy.signal as signal
import random
import matplotlib.pyplot as plt
import os

output_dir = r"D:\BA\PID-Controller-optimization-with-machine-learning\plots\visul_syn_data"
os.makedirs(output_dir, exist_ok=True)  # Create if it doesn't exist


def generate_transfer_function(type_name, sprunghoehe_target, sprungzeit_target):
    K = random.uniform(0.5, 3.0)
    T1 = T2 = Td = zeta = omega_n = None

    if type_name == "PT1":
        T1 = random.uniform(0.1, 50)
        num = [K]
        den = [T1, 1]
    elif type_name == "PT2":
        T1 = random.uniform(0.1, 50)
        T2 = random.uniform(0.1, 50)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])
    elif type_name == "PT1+Td":
        T1 = random.uniform(0.1, 50)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = [T1, 1]
    elif type_name == "PT2+Td":
        T1 = random.uniform(0.1, 50)
        T2 = random.uniform(0.1, 50)
        Td = random.uniform(0.1, 5)
        num = [K]
        den = np.polymul([T1, 1], [T2, 1])
    elif type_name == "D":
        T1 = random.uniform(0.1, 5)
        num = [K * T1, 0]
        den = [1, 1]
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

def calculate_tangent_parameters(t, y):
    dy = np.gradient(y, t)
    max_slope_idx = np.argmax(dy)
    slope = dy[max_slope_idx]
    t_inflect = t[max_slope_idx]
    y_inflect = y[max_slope_idx]

    Tu = t_inflect - (y_inflect / slope)
    Tg = (1.0 - y_inflect) / slope + t_inflect

    Tu = max(Tu, 0.1)  # Clamp Tu to prevent divide-by-zero
    Ks = y[-1]
    return Ks, Tu, Tg

def apply_pid_tuning(method, Ks, Tu, Tg):
    if method == "ZN":
        Kp = 1.2 * (Tg / (Ks * Tu))
        Ki = 0.6 * (Tg / (Ks * Tu ** 2))
        Kd = 0.6 * (Tg / Ks)
    elif method == "CHR":
        Kp = 0.6 * (Tg / (Ks * Tu))
        Ki = 0.6 / (Ks * Tu)
        Kd = 0.3 * (Tg / Ks)
    else:
        Kp = Ki = Kd = 0.0
    return Kp, Ki, Kd

def simulate_step_response(num, den, Td=0, sprunghoehe_target=1.0, duration=100, dt=0.1):
    system = signal.TransferFunction(num, den)
    t = np.arange(0, duration, dt)
    if Td:
        t_shifted = np.arange(0, duration - Td, dt)
        _, y = signal.step(system, T=t_shifted)
        y = np.concatenate((np.zeros(int(Td / dt)), y))
    else:
        _, y = signal.step(system, T=t)

    if y[-1] != 0:
        y *= sprunghoehe_target / y[-1]

    sprunghoehe_actual = y[-1]
    y_10 = 0.1 * sprunghoehe_actual
    y_90 = 0.9 * sprunghoehe_actual
    t_10_idx = np.argmax(y >= y_10)
    t_90_idx = np.argmax(y >= y_90)
    sprungzeit_actual = t[t_90_idx] - t[t_10_idx]

    return t, y, sprunghoehe_actual, sprungzeit_actual

sample_data = []
types = ["PT1", "PT2", "PT1+Td", "PT2+Td", "D", "Osc2"]

while len(sample_data) < 50:
    try:
        type_choice = random.choice(types)
        sprunghoehe_target = round(random.uniform(0.5, 2.0), 2)
        sprungzeit_target = round(random.uniform(2.0, 30.0), 1)
        tf_data = generate_transfer_function(type_choice, sprunghoehe_target, sprungzeit_target)

        t, y, sprunghoehe_actual, sprungzeit_actual = simulate_step_response(
            tf_data["num"], tf_data["den"], tf_data["Td"], tf_data["sprunghoehe_target"]
        )

        Ks, Tu, Tg = calculate_tangent_parameters(t, y)
        method = random.choice(["ZN", "CHR"])
        Kp, Ki, Kd = apply_pid_tuning(method, Ks, Tu, Tg)

        tf_data.update({
            "Ks": Ks, "Tu": Tu, "Tg": Tg,
            "method": method,
            "sprunghoehe_actual": sprunghoehe_actual,
            "sprungzeit_actual": sprungzeit_actual,
            "Kp": Kp, "Ki": Ki, "Kd": Kd
        })

        del tf_data["num"]
        del tf_data["den"]
        sample_data.append(tf_data)
    except Exception as e:
        continue  # Skip invalid simulations

def visualize_control_responses(df, num_samples=5):
    sample = df.sample(num_samples, random_state=42)
    time = np.linspace(0, 100, 1000)

    for plot_idx, (_, row) in enumerate(sample.iterrows()):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plant G(s)
        num = [row["K"]]
        den = [1]
        if pd.notnull(row["T1"]):
            den = np.polymul(den, [row["T1"], 1])
        if pd.notnull(row["T2"]):
            den = np.polymul(den, [row["T2"], 1])
        G = signal.TransferFunction(num, den)

        # PID Controller
        Kp, Ki, Kd = row["Kp"], row["Ki"], row["Kd"]
        C = signal.TransferFunction([Kd, Kp, Ki], [1, 0])

        # Closed-loop
        open_loop_num = np.polymul(C.num, G.num)
        open_loop_den = np.polymul(C.den, G.den)
        CL_num = open_loop_num
        CL_den = np.polyadd(open_loop_den, open_loop_num)
        closed_loop = signal.TransferFunction(CL_num, CL_den)
        t_out, y_out = signal.step(closed_loop, T=time)

        # Control signal
        e_t = 1.0 - y_out
        de_dt = np.gradient(e_t, t_out)
        integral_e = np.cumsum(e_t) * (t_out[1] - t_out[0])
        u_t = Kp * e_t + Ki * integral_e + Kd * de_dt

        # Plot
        ax.plot(t_out, y_out, label=f"y(t) - Output ({row['type']})", color='red')
        ax.plot(t_out, np.ones_like(t_out), '--', color='blue', label="w(t) - Reference")
        ax.plot(t_out, u_t, label="u(t) - Control Signal", color='green')
        ax.set_title(f"System {plot_idx+1}: {row['type']} (Method: {row['method']})")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time [s]")
        ax.legend()
        ax.grid(True)

        # Save as file
        filename = os.path.join(output_dir, f"pid_response_{plot_idx+1}_{row['type']}_{row['method']}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"✅ Saved: {filename}")



df = pd.DataFrame(sample_data)
df.to_csv(r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_dataset.csv", index=False)
print(df.head())
print("\n✅ Dataset saved to 'pid_dataset.csv'")
visualize_control_responses(df)
plt.savefig(r"D:\BA\PID-Controller-optimization-with-machine-learning\pid_closed_loop_examples.png")

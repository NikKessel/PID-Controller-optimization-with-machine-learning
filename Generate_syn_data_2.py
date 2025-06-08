import numpy as np
import pandas as pd
import scipy.signal as signal
import random

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
        T1 = random.uniform(0.1, 5)  # Derivative time
        num = [K * T1, 0]
        den = [1, 1]  # Artificially make it proper
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

    return sprunghoehe_actual, sprungzeit_actual

sample_data = []
types = ["PT1", "PT2", "PT1+Td", "PT2+Td", "D", "Osc2"]

while len(sample_data) < 5:
    try:
        type_choice = random.choice(types)
        sprunghoehe_target = round(random.uniform(0.5, 2.0), 2)
        sprungzeit_target = round(random.uniform(2.0, 30.0), 1)
        tf_data = generate_transfer_function(type_choice, sprunghoehe_target, sprungzeit_target)

        sprunghoehe_actual, sprungzeit_actual = simulate_step_response(
            tf_data["num"], tf_data["den"], tf_data["Td"], tf_data["sprunghoehe_target"]
        )

        tf_data.update({
            "sprunghoehe_actual": sprunghoehe_actual,
            "sprungzeit_actual": sprungzeit_actual,
            "Kp": 0.0, "Ki": 0.0, "Kd": 0.0
        })

        del tf_data["num"]
        del tf_data["den"]
        sample_data.append(tf_data)
    except Exception as e:
        continue  # Skip invalid or failed simulations

df = pd.DataFrame(sample_data)
print(df.head())

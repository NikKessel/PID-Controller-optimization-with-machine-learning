import numpy as np
import matplotlib.pyplot as plt
from control import tf, feedback, step_response

# Plant parameters
K = 1.4
T1 = 27
T2 = 11
num = [K]
den = np.polymul([T1, 1], [T2, 1])
G = tf(num, den)

# PID controller evaluation function
def get_pid_response(Kp, Ki, Kd, system, setpoint=1.0, sim_time=2000):
    s = tf([1, 0], [1])
    C = Kp + Ki / s + Kd * s
    T_sys = feedback(C * system, 1)
    t = np.linspace(0, sim_time, 1000)
    t, y = step_response(T_sys, T=t)

    w = np.ones_like(t) * setpoint
    e = w - y
    dt = t[1] - t[0]
    ise = np.sum(e ** 2) * dt
    y_final = y[-1] if y[-1] != 0 else 1e-6
    overshoot = (np.max(y) - y_final) / y_final

    # Settling time (5%)
    within_band = np.abs(y - y_final) <= 0.05 * np.abs(y_final)
    settling_time = t[np.where(within_band)[0][-1]] if np.any(within_band) else sim_time

    # Rise time (10%–90%)
    try:
        t_10 = t[np.where(y >= 0.1 * y_final)[0][0]]
        t_90 = t[np.where(y >= 0.9 * y_final)[0][0]]
        rise_time = t_90 - t_10
    except IndexError:
        rise_time = np.nan

    return {
        "ISE": ise,
        "Overshoot": overshoot,
        "Settling Time": settling_time,
        "Rise Time": rise_time
    }

# === Run for both controllers ===
pid_zn = {"Kp": 2.88, "Ki": 0.189, "Kd": 10.94}
pid_chr = {"Kp": 1.44, "Ki": 0.063, "Kd": 5.47}

res_zn = get_pid_response(**pid_zn, system=G)
res_chr = get_pid_response(**pid_chr, system=G)

print("Ziegler-Nichols Results:", res_zn)
print("CHR (0%) Results:", res_chr)

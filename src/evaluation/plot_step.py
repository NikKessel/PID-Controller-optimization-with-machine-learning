import numpy as np
import matplotlib.pyplot as plt
from control.matlab import tf, feedback, step

# Define PT2 system
K = 1.4
T1 = 27
T2 = 11
num = [K]
den = np.polymul([T1, 1], [T2, 1])
G = tf(num, den)

# Time vector
t = np.linspace(0, 500, 1000)

# Controller configurations
controllers = {
    "ZN": {"Kp": 2.88, "Ki": 0.189, "Kd": 10.94},
    "CHR 0%": {"Kp": 1.44, "Ki": 0.063, "Kd": 5.47},
    "ML Optimal (Broad)": {"Kp": 3.53, "Ki": 0.01, "Kd": 0.91},
    "Best ISE (Broad)": {"Kp": 11.71, "Ki": 0.313, "Kd": 19.70},
    "Best ST (Broad)": {"Kp": 3.90, "Ki": 0.0102, "Kd": 1.28},
    "Best RT (Broad)": {"Kp": 11.84, "Ki": 4.02, "Kd": 0.58},
}

# Plot step responses
plt.figure(figsize=(10, 6))
for label, gains in controllers.items():
    Kp, Ki, Kd = gains["Kp"], gains["Ki"], gains["Kd"]
    s = tf([1, 0], [1])
    C = Kp + Ki / s + Kd * s
    T_sys = feedback(C * G, 1)
    t_out, y_out = step(T_sys, T=t)
    plt.plot(t_out, y_out, label=label)

plt.title("Step Responses of Different PID Controllers")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

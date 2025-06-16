# utils/optimize_pid.py
import pandas as pd
import joblib
import os


def optimize_pid_for_system(K, T1, T2, Td, surrogate_model, weights):
    from scipy.optimize import differential_evolution
    import numpy as np
    import pandas as pd

    def objective(params):
        Kp, Ki, Kd = params
        X_df = pd.DataFrame([{
            'K': K, 'T1': T1, 'T2': T2, 'Td': Td,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
        }])
        ISE, OS, ST, RT, SSE = surrogate_model.predict(X_df)[0]

        return (
            weights["ISE"] * ISE +
            weights["Overshoot"] * OS +
            weights["SettlingTime"] * ST +
            weights["RiseTime"] * RT
        )

    bounds = [(0.1, 10.0), (0.001, 1.0), (0.0, 10.0)]
    result = differential_evolution(objective, bounds, seed=42)

    best_Kp, best_Ki, best_Kd = result.x
    best_metrics = surrogate_model.predict(pd.DataFrame([{
        'K': K, 'T1': T1, 'T2': T2, 'Td': Td,
        'Kp': best_Kp, 'Ki': best_Ki, 'Kd': best_Kd,
    }]))[0]

    return best_Kp, best_Ki, best_Kd, *best_metrics


# utils/optimize_pid.py
import pandas as pd
import joblib
import os
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd


def optimize_pid_for_system(K, T1, T2, T_d, surrogate_model, weights, constraints):
    from scipy.optimize import differential_evolution
    import numpy as np
    import pandas as pd

    def objective(params):
        Kp, Ki, Kd = params
        X_df = pd.DataFrame([{
            'K': K, 'T1': T1, 'T2': T2, 'Td': T_d,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
        }])
        prediction = surrogate_model.predict(X_df)[0]
        ISE, OS, ST, RT, SSE = prediction

        # === HARD CONSTRAINT CHECK ===
        if (
            ISE > constraints["ISE"] or ISE < 0.01 or
            OS > constraints["Overshoot"] or OS < 0.01 or
            ST > constraints["SettlingTime"] or ST < 0.01 or
            RT > constraints["RiseTime"] or RT < 0.01 or
            SSE > constraints["SSE"] or SSE < 0.01
        ):
            return np.inf  # Reject unfeasible solution


        # === Weighted cost function ===
        return (
            weights["ISE"] * ISE +
            weights["Overshoot"] * OS +
            weights["SettlingTime"] * ST +
            weights["RiseTime"] * RT
        )

    # === PID search bounds ===
    bounds = [(0.1, 10.0), (0.001, 1.0), (0.0, 10.0)]

    # === Run global optimizer ===
    result = differential_evolution(objective, bounds, seed=42)

    # === Best candidate ===
    best_Kp, best_Ki, best_Kd = result.x
    best_metrics = surrogate_model.predict(pd.DataFrame([{
        'K': K, 'T1': T1, 'T2': T2, 'Td': T_d,
        'Kp': best_Kp, 'Ki': best_Ki, 'Kd': best_Kd,
    }]))[0]

    return best_Kp, best_Ki, best_Kd, *best_metrics



# utils/optimize_pid.py
import pandas as pd

def optimize_pid_for_system(K, T1, T2, Td, Tu, Tg, system_type, surrogate_model, weights):
    from scipy.optimize import differential_evolution
    import numpy as np

    def objective(params):
        Kp, Ki, Kd = params
        X = {
            'K': K, 'T1': T1, 'T2': T2, 'Td': Td,
            'Tu': Tu, 'Tg': Tg,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
            'type': system_type
        }
        import pandas as pd
        X_df = pd.DataFrame([X])
        ise, os, stime, rtime = surrogate_model.predict(X_df)[0]
        w_ise, w_os, w_st, w_rt = weights
        return (w_ise * ise) + (w_os * os) + (w_st * stime) + (w_rt * rtime)

    bounds = [(0.1, 10.0), (0.001, 1.0), (0.0, 10.0)]  # Kp, Ki, Kd

    result = differential_evolution(objective, bounds)
    best_Kp, best_Ki, best_Kd = result.x
    best_metrics = surrogate_model.predict(pd.DataFrame([{
        'K': K, 'T1': T1, 'T2': T2, 'Td': Td,
        'Tu': Tu, 'Tg': Tg,
        'Kp': best_Kp, 'Ki': best_Ki, 'Kd': best_Kd,
        'type': system_type
    }]))[0]
    
    return best_Kp, best_Ki, best_Kd, *best_metrics

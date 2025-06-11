# utils/predict_pid.py

import numpy as np

def predict_pid_params(model, X: np.ndarray):
    """
    Predicts Kp, Ki, Kd values using a trained ML model.
    
    Parameters:
        model: trained ML regressor
        X: np.ndarray of shape (1, 4) [K, T1, T2, Td]
    
    Returns:
        Tuple: (Kp, Ki, Kd)
    """
    preds = model.predict(X)
    return tuple(preds[0])

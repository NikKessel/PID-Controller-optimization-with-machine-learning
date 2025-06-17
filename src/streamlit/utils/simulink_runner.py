# src/utils/simulink_runner.py

import subprocess
import scipy.io
import numpy as np
import os

def run_simulink_simulation(K, T1, T2, Kp, Ki, Kd,
                             input_file='C:/Users/KesselN/Documents/GitHub/PID-Controller-optimization-with-machine-learning/src/data/input.mat',
                             output_file='C:/Users/KesselN/Documents/GitHub/PID-Controller-optimization-with-machine-learning/src/data/results.mat',
                             matlab_script='simulate_and_export'):

    from scipy.io import savemat, loadmat

    # === 1. Save inputs to .mat ===
    savemat(input_file, {
        'K': K,
        'T1': T1,
        'T2': T2,
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd
    })

    # === 2. Run MATLAB as subprocess ===
    try:
        subprocess.run([
            "matlab",
            "-batch",
            "cd('C:/Users/KesselN/Documents/GitHub/PID-Controller-optimization-with-machine-learning/src/data'); simulate_and_export"
        ], check=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ MATLAB simulation failed: {e}")

    # === 3. Load results ===
    if not os.path.exists(output_file):
        raise FileNotFoundError("MATLAB simulation did not produce output file.")

    data = loadmat(output_file)
    print({k: np.array(v).shape for k, v in data.items()})

    # === 4. Parse outputs ===
    def _flatten(x): return np.squeeze(x) if isinstance(x, np.ndarray) else x
    def get_scalar(val):
        val = np.array(val).flatten()
        if val.size == 0:
            return float('nan')  # Return NaN if MATLAB didn’t produce a value
        if val.size != 1:
            raise ValueError(f"Expected scalar, got array of size {val.size}")
        return float(val[0])



    result = {
        't': _flatten(data['t']),
        'y': _flatten(data['y']),
        'u': _flatten(data['u']),
        'e': _flatten(data['e']),
        'ISE': get_scalar(data['ise']),
        'SSE': get_scalar(data['sse']),
        'Overshoot': get_scalar(data['os']),
        'RiseTime': get_scalar(data['rise_time']),
        'SettlingTime': get_scalar(data['settle_time']),
    }


    return result

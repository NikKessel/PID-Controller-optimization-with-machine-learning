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

    evaluated_controllers = []

    def objective(params):
        Kp, Ki, Kd = params
        X_df = pd.DataFrame([{
            'K': K, 'T1': T1, 'T2': T2, 'Td': T_d,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
        }])
        prediction = surrogate_model.predict(X_df)[0]
        ISE, OS, ST, RT, SSE = prediction

        if (
            ISE > constraints["ISE"] or ISE < 0.01 or
            OS > constraints["Overshoot"] or OS < 0.01 or
            ST > constraints["SettlingTime"] or ST < 0.01 or
            RT > constraints["RiseTime"] or RT < 0.01 or
            SSE > constraints["SSE"] or SSE < 0.01
        ):
            cost = np.inf
        else:
            cost = (
                weights["ISE"] * ISE +
                weights["Overshoot"] * OS +
                weights["SettlingTime"] * ST +
                weights["RiseTime"] * RT
            )

        evaluated_controllers.append({
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
            'ISE': ISE, 'Overshoot': OS, 'SettlingTime': ST, 'RiseTime': RT, 'SSE': SSE,
            'Cost': cost
        })

        return cost

    bounds = [(0.1, 10.0), (0.001, 1.0), (0.0, 10.0)]
    #result = differential_evolution(objective, bounds, seed=42)
    result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=300,         # ← Main control: number of generations
    popsize=25,          # ← Population size per generation
    tol=0.01,            # ← Convergence tolerance
    mutation=(0.5, 1),   # ← Mutation constant range
    recombination=0.7,   # ← Crossover probability
)


    best_Kp, best_Ki, best_Kd = result.x
    best_metrics = surrogate_model.predict(pd.DataFrame([{
        'K': K, 'T1': T1, 'T2': T2, 'Td': T_d,
        'Kp': best_Kp, 'Ki': best_Ki, 'Kd': best_Kd,
    }]))[0]

    evaluated_df = pd.DataFrame(evaluated_controllers)
    feasible_controllers = evaluated_df[evaluated_df["Cost"] < np.inf].sort_values(by="Cost")

    print("Total feasible controllers found:", len(feasible_controllers))

    unique_controllers = []

    for idx, candidate in feasible_controllers.iterrows():
        candidate_params = candidate[['Kp', 'Ki', 'Kd']].values
        print(f"\nEvaluating candidate #{idx}: Kp={candidate_params[0]}, Ki={candidate_params[1]}, Kd={candidate_params[2]}")

        if not unique_controllers:
            unique_controllers.append(candidate)
            print("→ Added as first controller.")
            continue

        differences = [
            np.abs(candidate_params - np.array(ctrl[['Kp', 'Ki', 'Kd']]))
            for ctrl in unique_controllers
        ]

        for diff_idx, diff in enumerate(differences):
            print(f"   Difference with controller #{diff_idx}: ΔKp={diff[0]:.3f}, ΔKi={diff[1]:.3f}, ΔKd={diff[2]:.3f}")

        is_different = all(np.any(diff >= 0.5) for diff in differences)
        print("→ Is different enough from existing controllers?", is_different)

        if is_different:
            unique_controllers.append(candidate)
            print("→ Controller added.")
        else:
            print("→ Controller skipped (too similar).")

        if len(unique_controllers) >= 5:
            print("→ Found 5 sufficiently distinct controllers. Stopping.")
            break

    top5_df = pd.DataFrame(unique_controllers)

    # Padding if fewer than 5 controllers found
    if len(top5_df) < 5:
        additional_rows = 5 - len(top5_df)
        top5_df = pd.concat([
            top5_df,
            pd.DataFrame([{
                'Kp': np.nan, 'Ki': np.nan, 'Kd': np.nan,
                'ISE': np.nan, 'Overshoot': np.nan, 'SettlingTime': np.nan,
                'RiseTime': np.nan, 'SSE': np.nan, 'Cost': np.nan
            }] * additional_rows)
        ], ignore_index=True)

    print("\nFinal Top 5 Controllers:")
    print(top5_df)


    return best_Kp, best_Ki, best_Kd, best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3], best_metrics[4], top5_df





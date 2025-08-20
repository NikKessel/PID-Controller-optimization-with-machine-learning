import os
import numpy as np
import pandas as pd
import joblib
import torch
import gpytorch
from scipy.optimize import differential_evolution

def optimize_pid_for_system(K, T1, T2, T_d, model_choice, weights, constraints):

    print("🔧 Starting PID optimization")
    
    evaluated_controllers = []

    surrogate_model = None
    dgp_models = {}
    dgp_likelihoods = {}
    dgp_scalers = {}

    # Load models depending on choice
    if model_choice == "MLP":
        print("📊 Loading MLP model")
        model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
        surrogate_model = joblib.load(os.path.join(model_dir, "model_surrogate_mlp.joblib"))
        print("✅ MLP model loaded successfully")

    elif model_choice == "DGP":
        print("📊 Loading DGP model")
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "streamlit_models", "dgp"))
        print("📁 DGP Base Path:", base_path)
        assert os.path.exists(base_path), f"❌ DGP directory not found: {base_path}"

        class SimpleDGPModel(gpytorch.models.ApproximateGP):
            def __init__(self, input_dim, num_inducing=64):
                inducing_points = torch.randn(num_inducing, input_dim)
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True)
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) +
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        metrics = ["ISE_log", "RiseTime_log", "SettlingTime_log", "Overshoot"]

        for m in metrics:
            model = SimpleDGPModel(input_dim=7)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            model.load_state_dict(torch.load(os.path.join(base_path, f"{m}_model.pth")))
            likelihood.load_state_dict(torch.load(os.path.join(base_path, f"{m}_likelihood.pth")))

            model.eval()
            likelihood.eval()

            dgp_models[m] = model
            dgp_likelihoods[m] = likelihood
            dgp_scalers[m] = joblib.load(os.path.join(base_path, f"{m}_scaler.pkl"))

    def predict_surrogate(Kp, Ki, Kd):
        #print(f"🔍 predict_surrogate called with Kp={Kp}, Ki={Ki}, Kd={Kd}, model_choice={model_choice}")

        if model_choice == "MLP":
            K_T1 = K * T1
            K_T2 = K * T2
            T1_T2_ratio = T1 / (T2 + 1e-8)
            X = np.array([[K, T1, T2, T_d, Kp, Ki, Kd, K_T1, K_T2, T1_T2_ratio]])
            # MLP does NOT use scaler
            result = surrogate_model.predict(X)[0]
            print(f"📊 MLP prediction result: {result}")
            return result  # Expected 5 values: ISE, SSE, RT, ST, OS

        elif model_choice == "DGP":
            X_df = pd.DataFrame([{
                'K': K, 'T1': T1, 'T2': T2, 'Td': T_d,
                'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
            }])
            X = X_df.values.astype(np.float32)
            #print(f"📊 DGP input shape: {X.shape}")

            results_mean = []
            results_std = []

            for metric in metrics:
                X_scaled = dgp_scalers[metric].transform(X)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred_dist = dgp_likelihoods[metric](dgp_models[metric](X_tensor))
                    mean = pred_dist.mean.item()
                    std = pred_dist.stddev.item()

                    if "log" in metric:
                        mean = np.exp(mean)
                        #mean = mean
                        # Approximate std for exp-transformed variables (delta method)
                        std = mean * std

                    results_mean.append(mean)
                    results_std.append(std)


            # DGP returns 4 values: ISE, RT, ST, OS (no SSE)
            #print(f"📊 DGP final results: {results}")
            return results_mean, results_std


        else:
            print(f"❌ Unknown model_choice: {model_choice}")
            return None
        
        
    def objective(params):
        try:
            Kp, Ki, Kd = params
            #print(f"🎯 Objective called with params: Kp={Kp}, Ki={Ki}, Kd={Kd}")

            predictions = predict_surrogate(Kp, Ki, Kd)
            print(f"🔍 Raw predictions from surrogate: {predictions}")

            if predictions is None:
                print("❌ predict_surrogate returned None")
                return float('inf')

            if model_choice == "MLP":
                means = predictions
                stds = [0.0] * len(means)
                print(f"📊 MLP means: {means}")
                print(f"📊 MLP stds (zeroed): {stds}")

            elif model_choice == "DGP":
                if not (isinstance(predictions, tuple) and len(predictions) == 2):
                    print(f"❌ DGP predictions not tuple of length 2: {predictions}")
                    return float('inf')
                means, stds = predictions
                #print(f"📊 DGP means: {means}")
                #print(f"📊 DGP stds: {stds}")

            else:
                print(f"❌ Invalid model_choice: {model_choice}")
                return float('inf')

            # Now unpack means and stds carefully with debug prints
            if model_choice == "MLP":
                if len(means) != 5 or len(stds) != 5:
                    print(f"❌ Unexpected length of MLP means/stds: means={len(means)}, stds={len(stds)}")
                    return float('inf')
                ISE, SSE, RT, ST, OS = means
                ISE_std, SSE_std, RT_std, ST_std, OS_std = stds

            else:  # DGP
                if len(means) != 4 or len(stds) != 4:
                    print(f"❌ Unexpected length of DGP means/stds: means={len(means)}, stds={len(stds)}")
                    return float('inf')
                ISE, RT, ST, OS = means
                ISE_std, RT_std, ST_std, OS_std = stds
                SSE = 0.0
                SSE_std = 0.0

            print(f"✅ Unpacked metrics: ISE={ISE}, SSE={SSE}, RT={RT}, ST={ST}, OS={OS}")
            print(f"✅ Unpacked stds: ISE_std={ISE_std}, SSE_std={SSE_std}, RT_std={RT_std}, ST_std={ST_std}, OS_std={OS_std}")

            # Constraint checks (means only)
            if ISE > constraints["ISE"] or ISE < 0.01:
                print(f"🚫 ISE constraint violated: {ISE}")
                return float('inf')
            #if OS > constraints["Overshoot"] or OS < 0.01:
                #print(f"🚫 Overshoot constraint violated: {OS}")
                return float('inf')
            if ST > constraints["SettlingTime"] or ST < 0.01:
                print(f"🚫 SettlingTime constraint violated: {ST}")
                return float('inf')
            if RT > constraints["RiseTime"] or RT < 0.01:
                print(f"🚫 RiseTime constraint violated: {RT}")
                return float('inf')
            if model_choice == "MLP":
                if SSE > constraints["SSE"] or SSE < 0.01:
                    print(f"🚫 SSE constraint violated: {SSE}")
                    return float('inf')

            cost = (weights["ISE"] * ISE +
                    weights["Overshoot"] * OS +
                    weights["SettlingTime"] * ST +
                    weights["RiseTime"] * RT)

            #print(f"✅ Valid solution found with cost: {cost}")

            evaluated_controllers.append({
                'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
                'ISE': ISE, 'Overshoot': OS, 'SettlingTime': ST,
                'RiseTime': RT, 'SSE': SSE, 'Cost': cost,
                'ISE_std': ISE_std, 'Overshoot_std': OS_std,
                'SettlingTime_std': ST_std, 'RiseTime_std': RT_std,
                'SSE_std': SSE_std
            })

            return cost

        except Exception as e:
            print(f"❌ Exception in objective function: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')


        except Exception as e:
            print(f"❌ Exception in objective function: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

    # Define PID parameter bounds
    bounds = [(0.1, 20.0), (0.0, 20.0), (0.0, 15.0)]
    print("🚀 Starting differential_evolution optimization")
    print(f"📊 Bounds: {bounds}")

    # Test objective once before optimization
    print("🧪 Testing objective function with initial parameters...")
    test_params = [1.0, 0.1, 0.1]
    test_cost = objective(test_params)
    print(f"🧪 Test cost: {test_cost}")

    if np.isnan(test_cost) or np.isinf(test_cost):
        print("❌ Test warning - objective function returned invalid cost on initial guess")
        print("🔁 Proceeding with optimization anyway...")

    if len(evaluated_controllers) > 0:
        df = pd.DataFrame(evaluated_controllers)
        df = df[np.isfinite(df["Cost"])]   # filter out invalid
        df = df.sort_values("Cost")        # best first
        print("\n📋 Evaluated Controllers (Top 10):")
        print(df.head(10).to_string(index=False))

    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=150,
            popsize=25,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.9,
        )
        #print("🏁 Optimization finished")
        #print(f"📈 Success: {result.success}")
        #print(f"📈 Message: {result.message}")
        #print(f"📈 Iterations: {result.nit}")
        #print(f"📈 Function evaluations: {result.nfev}")

        if result.success:
            best_Kp, best_Ki, best_Kd = result.x
            best_metrics = predict_surrogate(best_Kp, best_Ki, best_Kd)
            print(f"🎯 Best parameters: Kp={best_Kp}, Ki={best_Ki}, Kd={best_Kd}")
            print(f"🎯 Best metrics: {best_metrics}")

            return {
                'success': True,
                'best_params': result.x,
                'best_metrics': best_metrics,
                'best_cost': result.fun,
                'evaluated_controllers': evaluated_controllers
            }
        else:
            return {
                'success': False,
                'message': result.message,
                'evaluated_controllers': evaluated_controllers
            }

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'evaluated_controllers': evaluated_controllers
        }

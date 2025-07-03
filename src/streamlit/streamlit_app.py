import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import pandas as pd
import control
from control import tf, feedback, step_response, pade
from control.matlab import tf, feedback, step
from utils.predict_pid import predict_pid_params
from utils.simulink_runner import run_simulink_simulation
from scipy.signal import step
from scipy.integrate import simpson
import plotly.graph_objects as go
import torch
import gpytorch
from gpytorch.settings import fast_pred_var


# Set page config###
#test
# === Page Config ===
st.set_page_config(page_title="PID Optimizer", layout="wide", initial_sidebar_state="expanded")




# === Sidebar Navigation ===
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", [
    "🏠 Home",
    "🔍 Predict PID",
    "📊 Evaluate PID",
    "⚙️ Optimize PID",
    "🧪 Simulink Validation"
])

# === Landing Page ===
if mode == "🏠 Home":
    st.title("📘 Machine Learning for Performance-Driven Tuning of PID Controllers in Process Control Applications")

    st.markdown("""
    ## 🎓 Project Overview

    This application is part of my Bachelor thesis in **Bioprocess Engineering** at **Frankfurt University of Applied Sciences**. 

    ### 🤖 Goal:
    The aim is to build a machine learning pipeline that can:
    - Predict suitable PID controller parameters (Kp, Ki, Kd) for dynamic systems
    - Evaluate the performance metrics of user-given controller
    - Optimize controller parameters using surrogate models and genetic algorithms
    - Validate results via MATLAB/Simulink simulations

    ## 🧠 How the App Works

    You can use the navigation bar to explore 4 different modes:

    1. **Predict PID**: Input a transfer function (K, T1, T2, Td), select ML model → get Kp, Ki, Kd
    2. **Evaluate PID**: Evaluate any PID setting using trained surrogate model
    3. **Optimize PID**: Use ML-driven optimization to find best controller (based on weights and constraints)
    4. **Simulink Validation**: Run final controller in MATLAB for verification

    ---

    ## 📊 Model Performance (R² / MAE)

    ### 🔎 Multi-output Surrogate Model (Deep Gaussian Process)
    | Metric        | R²     | MAE       |
    |---------------|---------|-----------|
    | ISE           | 0.853   | 1.302     |
    | Overshoot     | 0.581   | 1.422     |
    | Settling Time | 0.833   | 15.503    |
    | Rise Time     | 0.836   | 4.259     |

    ### 🔍 PID Parameter Prediction
    | Parameter | Model        | R²      | MAE     |
    | --------- | ------------ | ------- | ------- |
    | Kp        | Symbolic Reg | 0.975   | 0.0035  |
    | Ki        | Symbolic Reg | 0.913   | 0.010   |
    | Kd        | Symbolic Reg  | 0.946   | 0.020 |
    | --------- | ------- | ----- | ----- |
    | Kp        | Deep GP | 0.985 | 0.029 |
    | Ki        | Deep GP | 0.955 | 0.061 |
    | Kd        | Deep GP | 0.947 | 0.347 |
    | --------- | ------------- | ----- | ----- |
    | Kp        | Random Forest | 0.984 | 0.032 |
    | Ki        | Random Forest | 0.942 | 0.066 |
    | Kd        | Random Forest | 0.944 | 0.362 |
    | --------- | ----- | ----- | ----- |
    | Kp        | MLP   | 0.973 | 0.043 |
    | Ki        | MLP   | 0.937 | 0.079 |
    | Kd        | MLP   | 0.926 | 0.395 |
    | --------- | ------- | ----- | ----- |
    | Kp        | XGBoost | 0.978 | 0.039 |
    | Ki        | XGBoost | 0.940 | 0.073 |
    | Kd        | XGBoost | 0.932 | 0.376 |


    ---

    """)

    st.markdown("""
    ---
    🔎 For any questions or source code, visit the [GitHub repository](https://github.com/NikKessel/PID-Controller-optimization-with-machine-learning/tree/main) or contact me via my university email: nkessel[a]stud.fra-uas.de.
    """)




# --- Conditional ML model selection ---
if mode == "🔍 Predict PID":
    julia_env_path = os.path.join(os.getcwd(), 'julia_env')

    if not os.path.exists(julia_env_path):
        os.makedirs(julia_env_path)
    model_choice = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "MLP", "XGBoost", "Symbolic", "DGP"], key="model_select")
    if "predict_clicked" not in st.session_state: ####
        st.session_state.predict_clicked = False

    st.sidebar.markdown("**System Parameters**")
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.50)
    T1 = st.sidebar.number_input("T1", min_value=0.1, max_value=50.0, value=2.0)
    T2 = st.sidebar.number_input("T2", min_value=0.0, max_value=50.0, value=0.00)
    Td = st.sidebar.number_input("Td", min_value=0.0, max_value=5.0, value=0.50) 

    st.sidebar.markdown("**Plot Settings**")
    t_max = st.sidebar.slider("Simulation Time [s]", 1, 300, 20, key="slider_t_max")
    y_max = st.sidebar.slider("Y-Axis Max (Output)", 1.0, 5.0, 1.5, step=0.1, key="slider_y_max")

    if st.button("🔍 Predict PID"):
        st.session_state.predict_clicked = True
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")

            if model_choice in ["Random Forest", "MLP"]:
                X = np.array([[K, T1, T2, Td]])
            elif model_choice == "XGBoost":
                X = np.array([[K, T1, T2]])
            else:
                X = np.array([[K, T1, T2, Td]])  # full input for Symbolic and DGP

            def load_and_predict_symb(param, K, T1, T2):
                try:
                    import pysr
                except ImportError:
                    raise ImportError("⚠️ PySR is required to load symbolic models. Install with: `pip install pysr`")

                model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
                model = joblib.load(os.path.join(model_dir, f"symbolic_{param}.pkl"))

                # Only use the features you trained on
                X = np.array([[K, T1, T2]])
                st.write(f"🔍 [SYMB DEBUG] Input X.shape = {X.shape}, X = {X}")

                assert X.shape[1] == 3, f"Expected 3 features for Symbolic model, got {X.shape[1]}"
                # Predict log10-transformed output and invert it
                y_log = model.predict(X)[0]
                y = max(0.0, 10**y_log - 1e-6)

                # Debug print
                st.write(f"📊 Symbolic prediction for **{param}**:")
                st.write(f"- Raw log10 output: `{y_log:.4f}`")
                st.write(f"- Inverted: `{y:.4f}`")
                st.write(f"- Features: K={K}, T1={T1}, T2={T2}")

                return y

            import torch
            from gpytorch.settings import fast_pred_var
            class DGPModel(gpytorch.models.ApproximateGP):
                def __init__(self, input_dim):
                    inducing_points = torch.randn(256, input_dim)
                    variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
                    variational_strategy = gpytorch.variational.VariationalStrategy(
                        self, inducing_points, variational_distribution, learn_inducing_locations=True
                    )
                    super().__init__(variational_strategy)
                    self.mean_module = gpytorch.means.ZeroMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

                def forward(self, x):
                    mean = self.mean_module(x)
                    covar = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean, covar)



            K_T1 = K * T1
            K_T2 = K * T2
            eps = 1e-8
            T1_T2_ratio = T1 / (T2 + eps)

            #X_raw = np.array([[K, T1, T2, K_T1, K_T2, T1_T2_ratio]])
            X_raw = np.array([[K, T1, T2,Td]])

            def load_and_predict_dgp(param, X_raw, return_std=False, return_all=False):
                #base_path = os.path.join(os.path.dirname(__file__), "streamlit_models")

                #base_path = os.path.join(os.getcwd(), "streamlit_models")
                base_path = os.path.join(os.path.dirname(__file__), "streamlit_models")

                #base_path = os.path.abspath(base_path)  # resolve relative path


                # Method 2: Alternative - use path relative to the script file
                # Uncomment this if Method 1 doesn't work
                # base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_models")
                
                # Method 3: For debugging - check if directory exists


                #param = param[0].upper() + param[1:]  # Capitalize first letter: kp → Kp, ki → Ki, etc.
                # Check if model files exist before loading

                def find_existing_file(base_path, patterns):
                    for pattern in patterns:
                        path = os.path.join(base_path, pattern)
                        if os.path.exists(path):
                            return path
                    return None

                param_lower = param.lower()
                param_cap = param.capitalize()

                x_scaler_path = find_existing_file(base_path, [
                    f"dgp_{param_lower}_scaler_X.pkl",
                    f"dgp_{param_cap}_scaler_X.pkl"
                ])

                y_scaler_path = find_existing_file(base_path, [
                    f"dgp_{param_lower}_scaler_y.pkl",
                    f"dgp_{param_cap}_scaler_y.pkl"
                ])

                if not x_scaler_path or not y_scaler_path:
                    raise FileNotFoundError("❌ Scaler file(s) not found")
                #x_scaler_path = os.path.join(base_path, f"dgp_{param}_scaler_X.pkl")
                #y_scaler_path = os.path.join(base_path, f"dgp_{param}_scaler_y.pkl")
                


                if not os.path.exists(x_scaler_path):
                    raise FileNotFoundError(f"X scaler not found at: {x_scaler_path}")
                if not os.path.exists(y_scaler_path):
                    raise FileNotFoundError(f"Y scaler not found at: {y_scaler_path}")
                X_scaler = joblib.load(x_scaler_path)
                y_scaler = joblib.load(y_scaler_path)
                param_lower = param.lower()
                param_cap = param.capitalize()

                model_path = os.path.join(base_path, f"dgp_{param_lower}.pth")
                likelihood_path = os.path.join(base_path, f"dgp_{param_lower}_likelihood.pth")
                #model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")

                # === Load scalers
                #X_scaler = joblib.load(os.path.join(base_path, f"dgp_{param}_scaler_X.pkl"))
                #y_scaler = joblib.load(os.path.join(base_path, f"dgp_{param}_scaler_y.pkl"))
                #X_scaler = joblib.load(os.path.join(model_dir, f"dgp_{param}_scaler_X.pkl"))
                #y_scaler = joblib.load(os.path.join(model_dir, f"dgp_{param}_scaler_y.pkl"))
                # === Preprocess input
                X_scaled = X_scaler.transform(X_raw)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                print("y_scaler mean:", y_scaler.mean_)
                print("y_scaler scale:", y_scaler.scale_)
                # === Instantiate model and likelihood
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = DGPModel(input_dim=X_tensor.shape[1])

                #model = VariationalGP(X_train_tensor.shape[1]).to(device)
                #likelihood = gpytorch.likelihoods.GaussianLikelihood()
                # === Inference dummy noise (required by FixedNoise)
                dummy_noise = torch.ones(X_tensor.size(0)) * 1e-4  # use same shape as X_tensor
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=dummy_noise).to(device)

                #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                #noise=torch.ones(X_train_tensor.size(0)).to(device) * 1e-4
            #).to(device)
                # === Load model weights
                model.load_state_dict(torch.load(os.path.join(model_dir, f"dgp_{param_lower}.pth")))
                #likelihood.load_state_dict(torch.load(os.path.join(model_dir, f"dgp_{param}_likelihood.pth")))
                likelihood.load_state_dict(torch.load(os.path.join(model_dir, f"dgp_{param_lower}_likelihood.pth"), map_location=device))

                model.eval()
                likelihood.eval()

                # === Inference
                with torch.no_grad(), fast_pred_var():
                    preds = likelihood(model(X_tensor))
                    y_pred = preds.mean.item()
                    y_std = preds.variance.sqrt().item() if return_std else None

                # === Inverse transform prediction
                y_pred_inv = y_scaler.inverse_transform(np.array([[y_pred]]))[0][0]
                y_std_inv = None
                if return_std:
                    upper = y_scaler.inverse_transform(np.array([[y_pred + y_std]]))[0][0]
                    lower = y_scaler.inverse_transform(np.array([[y_pred - y_std]]))[0][0]
                    y_std_inv = abs(upper - lower) / 2

                # === Debug info
                #st.write(f"🔍 Scaled prediction for {param}: {y_pred:.4f}")
                #if return_std:
                    #st.write(f"📉 Scaled std: ±{y_std:.4f}")
                #st.write(f"📈 Final prediction for {param}: {y_pred_inv:.6f}" + (f" ± {y_std_inv:.6f}" if return_std else ""))

                # === Return structure
                if return_all:
                    return y_pred, y_pred_inv, y_std_inv
                elif return_std:
                    return {
                        "mean": y_pred_inv,
                        "std": y_std_inv
                    }
                else:
                    return y_pred_inv


            if model_choice in ["Random Forest", "MLP", "XGBoost"]:
                model_filename = f"model_{model_choice.lower().replace(' ', '_')}.joblib"
                model_path = os.path.join(model_dir, model_filename)
                model = joblib.load(model_path)

                from utils.predict_pid import predict_pid_params


                Kp, Ki, Kd = predict_pid_params(model, X)

            elif model_choice == "Symbolic":
                Kp = load_and_predict_symb("kp", K, T1, T2)
                Ki = load_and_predict_symb("ki", K, T1, T2)
                Kd = load_and_predict_symb("kd", K, T1, T2)

            elif model_choice == "DGP":
                Kp = load_and_predict_dgp("kp", X_raw)
                Ki = load_and_predict_dgp("ki", X_raw)
                Kd = load_and_predict_dgp("kd", X_raw)
                Kp_result = load_and_predict_dgp("Kp", X_raw, return_std=True)
                Ki_result = load_and_predict_dgp("Ki", X_raw, return_std=True)
                Kd_result = load_and_predict_dgp("Kd", X_raw, return_std=True)


            else:
                st.error("❌ Unknown model type selected.")
                raise ValueError("Invalid model")

            # only reached if prediction successful:
            st.success("✅ Prediction complete!")
            Kp_ml, Ki_ml, Kd_ml = Kp, Ki, Kd


            # === Extract predicted values
            #Kp_ml, Ki_ml, Kd_ml = Kp_result["mean"], Ki_result["mean"], Kd_result["mean"]
            #Kp_std, Ki_std, Kd_std = Kp_result["std"], Ki_result["std"], Kd_result["std"]
            if model_choice == "DGP":
                Kp_ml, Ki_ml, Kd_ml = Kp_result["mean"], Ki_result["mean"], Kd_result["mean"]
                Kp_std, Ki_std, Kd_std = Kp_result["std"], Ki_result["std"], Kd_result["std"]
                Kp_str = f"{Kp_ml:.3f} ± {Kp_std:.3f}"
                Ki_str = f"{Ki_ml:.5f} ± {Ki_std:.5f}"
                Kd_str = f"{Kd_ml:.2f} ± {Kd_std:.2f}"
            else:
                Kp_ml, Ki_ml, Kd_ml = Kp, Ki, Kd
                Kp_str = f"{Kp_ml:.3f}"
                Ki_str = f"{Ki_ml:.5f}"
                Kd_str = f"{Kd_ml:.2f}"

            # === Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Kp", Kp_str)
            col2.metric("Ki", Ki_str)
            col3.metric("Kd", Kd_str)



            # --- Real simulation ---

            # --- Simulate system response ---
            def simulate_response(K, T1, T2, L, Kp, Ki, Kd, T_final=100):
                t = np.linspace(0, T_final, 1000)
                den = np.polymul([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
                G = tf([K], den)
                if L > 0:
                    num_d, den_d = pade(L, 1)
                    G = G * tf(num_d, den_d)
                s = tf([1, 0], [1])
                C = Kp + Ki/s + Kd*s
                sys = feedback(C * G, 1)
                t, y = step_response(sys, T=t)
                return t, y

            def zn_pid(K, T1, T2, L):
                T = T1 + T2 if T2 > 0 else T1
                Kp = 1.2 * T / (K * L)
                Ti = 2 * L
                Td = 0.5* L
                Ki = Kp / Ti
                Kd = Kp * Td
                return Kp, Ki, Kd
            def chr_pid(K, T1, T2, L, overshoot=0):
                T = T1 + T2 if T2 > 0 else T1
                if overshoot == 0:
                    Kp = 0.6 * T / (K * L)
                    Ti = L
                    Td = 0.5 * L
                else:
                    Kp = 0.95 * T / (K * L)
                    Ti = 1.35 * L
                    Td = 0.47 * L
                Ki = Kp / Ti
                Kd = Kp * Td
                return Kp, Ki, Kd

            # === Use Predicted PID ===
            L = Td  # clarity
            Kp_ml, Ki_ml, Kd_ml = Kp, Ki, Kd
            Kp_zn, Ki_zn, Kd_zn = zn_pid(K, T1, T2, L)
            Kp_chr0, Ki_chr0, Kd_chr0 = chr_pid(K, T1, T2, L, overshoot=0)
            Kp_chr20, Ki_chr20, Kd_chr20 = chr_pid(K, T1, T2, L, overshoot=20)

            # === Simulate Step Responses ===
            t_ml, y_ml = simulate_response(K, T1, T2, L, Kp_ml, Ki_ml, Kd_ml, T_final=t_max)
            t_zn, y_zn = simulate_response(K, T1, T2, L, Kp_zn, Ki_zn, Kd_zn, T_final=t_max)
            t_chr0, y_chr0 = simulate_response(K, T1, T2, L, Kp_chr0, Ki_chr0, Kd_chr0, T_final=t_max)
            t_chr20, y_chr20 = simulate_response(K, T1, T2, L, Kp_chr20, Ki_chr20, Kd_chr20, T_final=t_max)

            # === Debug Print for PID parameters ===
            #st.markdown("### 🔧 PID Parameter Debug")
            #st.markdown("### 📐 Calculation Breakdown")

            T_eff = T1 + T2 if T2 > 0 else T1
            with st.expander("🔧 Show Calculation Details"):

                st.code(f"""
                🔧 Effective Time Constant:
                    T = T1 + T2 = {T1:.3f} + {T2:.3f} = {T_eff:.3f}

                === Ziegler–Nichols ===
                Kp = 1.2 × T / (K × L) = 1.2 × {T_eff:.3f} / ({K:.3f} × {L:.3f}) = {Kp_zn:.4f}
                Ti = 2 × L = 2 × {L:.3f} = {2 * L:.4f}
                Td = 0.5 × L = 0.5 × {L:.3f} = {0.5 * L:.4f}
                Ki = Kp / Ti = {Kp_zn:.4f} / {2 * L:.4f} = {Ki_zn:.4f}
                Kd = Kp × Td = {Kp_zn:.4f} × {0.5 * L:.4f} = {Kd_zn:.4f}

                === CHR (0% Overshoot) ===
                Kp = 0.6 × T / (K × L) = 0.6 × {T_eff:.3f} / ({K:.3f} × {L:.3f}) = {Kp_chr0:.4f}
                Ti = L = {L:.3f}
                Td = 0.5 × L = {0.5 * L:.4f}
                Ki = Kp / Ti = {Kp_chr0:.4f} / {L:.4f} = {Ki_chr0:.4f}
                Kd = Kp × Td = {Kp_chr0:.4f} × {0.5 * L:.4f} = {Kd_chr0:.4f}

                === CHR (20% Overshoot) ===
                Kp = 0.95 × T / (K × L) = 0.95 × {T_eff:.3f} / ({K:.3f} × {L:.3f}) = {Kp_chr20:.4f}
                Ti = 1.35 × L = 1.35 × {L:.3f} = {1.35 * L:.4f}
                Td = 0.47 × L = 0.47 × {L:.3f} = {0.47 * L:.4f}
                Ki = Kp / Ti = {Kp_chr20:.4f} / {1.35 * L:.4f} = {Ki_chr20:.4f}
                Kd = Kp × Td = {Kp_chr20:.4f} × {0.47 * L:.4f} = {Kd_chr20:.4f}
                """, language="text")
            with st.expander("🔧 Show Parameters"):

                st.code(f"""
                🔍 Input Parameters:
                    K  = {K:.3f}
                    T1 = {T1:.3f}
                    T2 = {T2:.3f}
                    L  = {L:.3f}

                📊 ML Predicted:
                    Kp = {Kp_ml:.4f}
                    Ki = {Ki_ml:.4f}
                    Kd = {Kd_ml:.4f}

                📊 Ziegler-Nichols:
                    Kp = {Kp_zn:.4f}
                    Ki = {Ki_zn:.4f}
                    Kd = {Kd_zn:.4f}

                📊 CHR (0% Overshoot):
                    Kp = {Kp_chr0:.4f}
                    Ki = {Ki_chr0:.4f}
                    Kd = {Kd_chr0:.4f}

                📊 CHR (20% Overshoot):
                    Kp = {Kp_chr20:.4f}
                    Ki = {Ki_chr20:.4f}
                    Kd = {Kd_chr20:.4f}
                """, language="text")


            # === Plot Step Responses ===
            #st.markdown("### Step Response")
            #fig, ax = plt.subplots(figsize=(7, 4))
            #ax.plot(t_ml, y_ml, label="ML Predicted PID", linewidth=2)
            #ax.plot(t_zn, y_zn, '--', label="Ziegler–Nichols")
            #ax.plot(t_chr0, y_chr0, ":", label="CHR (0% OS)")
            #ax.plot(t_chr20, y_chr20, "-.", label="CHR (20% OS)")
            #ax.plot(t_ml, np.ones_like(t_ml)*K, "k--", label=f"Step Input ({1:.2f})")
            step_input = np.ones_like(t_ml)
            step_input[t_ml < 0.01] = 0  # Optional: simulate visible step
            #ax.plot(t_ml, step_input, "k--", label="Step Input (0 → 1)")

            #ax.set_xlabel("Time [s]")
            #ax.set_ylabel("Output")
            #ax.set_title("Closed-Loop Step Response")
            #ax.set_ylim(0, y_max)
            #ax.grid(True)
            #ax.legend()
            #st.pyplot(fig)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_ml, y=y_ml, mode='lines', name='ML Predicted PID'))
            fig.add_trace(go.Scatter(x=t_zn, y=y_zn, mode='lines', name='Ziegler–Nichols', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=t_chr0, y=y_chr0, mode='lines', name='CHR (0% OS)', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=t_chr20, y=y_chr20, mode='lines', name='CHR (20% OS)', line=dict(dash='dashdot')))
            fig.add_trace(go.Scatter(x=t_ml, y=step_input, mode='lines', name='Step Input (0 → 1)', line=dict(color='black', dash='dash')))

            fig.update_layout(
                title="Closed-Loop Step Response",
                xaxis=dict(
                    title="Time [s]",
                    tickmode='linear',
                    tick0=0,
                    dtick=1,  # smaller time steps (e.g. 0.5 if you want more granularity)
                ),
                yaxis=dict(
                    title="Output",
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2,  # finer vertical spacing
                    range=[0, y_max],  # y_max as defined earlier
                ),
                legend=dict(
                    x=1,
                    y=0,
                    xanchor='right',
                    yanchor='bottom',
                    orientation='v',  # vertical legend, you can also use 'h'
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='black',
                    borderwidth=1
                ),
                template='plotly_white'
            )
            def compute_ise(t, y):
                e = 1.0 - y  # step input is 1
                ise = simpson(e**2, t)
                return ise

            def compute_metrics(t, y, label=""):
                # === Rise Time (10% to 90%)
                try:
                    t_10 = t[np.where(y >= 0.1 * y[-1])[0][0]]
                    t_90 = t[np.where(y >= 0.9 * y[-1])[0][0]]
                    rise_time = t_90 - t_10
                except IndexError:
                    rise_time = np.nan

                # === Settling Time (±5% band)
                final_val = y[-1]
                tol = 0.05 * abs(final_val)
                lower = final_val - tol
                upper = final_val + tol

                within_bounds = (y >= lower) & (y <= upper)
                settling_time = np.nan
                for i in range(len(y)):
                    if np.all(within_bounds[i:]):
                        settling_time = t[i]
                        break

                # === Overshoot
                overshoot = max(0.0, (np.max(y) - 1.0) * 100)
                ise = compute_ise(t, y)

                return rise_time, settling_time, overshoot, ise

            # === Collect Metrics ===
            metrics = {}
            #metrics["ML"] = compute_metrics(t_ml, y_ml)
            #metrics["ZN"] = compute_metrics(t_zn, y_zn)
            #metrics["CHR 0%"] = compute_metrics(t_chr0, y_chr0)
            #metrics["CHR 20%"] = compute_metrics(t_chr20, y_chr20)
            metrics = {}
            for label, (t, y) in {
                "ML": (t_ml, y_ml),
                "ZN": (t_zn, y_zn),
                "CHR 0%": (t_chr0, y_chr0),
                "CHR 20%": (t_chr20, y_chr20)
            }.items():
                rt, stt, os, ise = compute_metrics(t, y)
                metrics[label] = (rt, stt, os, ise)


            metric_rows = []
            for label, (rt, stt, os, ise) in metrics.items():
                metric_rows.append({
                    "Controller": label,
                    "Rise Time [s]": f"{rt:.2f}" if not np.isnan(rt) else "—",
                    "Settling Time [s]": f"{stt:.2f}" if not np.isnan(stt) else "—",
                    "Overshoot [%]": f"{os:.2f}" if not np.isnan(os) else "—",
                    "ISE": f"{ise:.3f}" if not np.isnan(ise) else "—"

                })

            df_metrics = pd.DataFrame(metric_rows)

            # === Display as nice table ===
            st.markdown("### 📊 Key Performance Metrics (All Controllers)")
            st.table(df_metrics)
            # === Metric Extraction ===
            # Rise Time
            try:
                t_10 = t_ml[np.where(y_ml >= 0.1)[0][0]]
                t_90 = t_ml[np.where(y_ml >= 0.9)[0][0]]
                rise_time = t_90 - t_10
            except IndexError:
                rise_time = np.nan

            # Overshoot
            overshoot_val = (np.max(y_ml) - 1.0) * 100
            overshoot_time = t_ml[np.argmax(y_ml)] if np.max(y_ml) > 1 else np.nan

            # Settling Time (within ±2%)
            #within_bounds = np.abs(y_ml - 1.0) < 0.02
            #settling_time = t_ml[np.where(within_bounds)[-1][-1]] if np.any(within_bounds) else np.nan
            final_val = y_ml[-1]
            tol = 0.05 * abs(final_val)
            lower_bound = final_val - tol
            upper_bound = final_val + tol

            # Debug info
            #st.code(f"""
            #Final Value Estimate: {final_val:.4f}
            #Tolerance (±5%): ±{tol:.4f}
            #Acceptable Range: [{lower_bound:.4f}, {upper_bound:.4f}]
            #""")

            within_bounds = (y_ml >= lower_bound) & (y_ml <= upper_bound)

            # Find the last time index after which the signal always stays within bounds
            settling_time = np.nan
            for i in range(len(y_ml)):
                if np.all(within_bounds[i:]):
                    settling_time = t_ml[i]
                    #st.code(f"Settling starts at index {i}, time = {settling_time:.4f}s")
                    break

            if np.isnan(settling_time):
                st.warning("⚠️ System never fully settles within ±5%.")


            # === Annotate on Plotly Figure ===
            if not np.isnan(rise_time):
                fig.add_vline(
                    x=rise_time,
                    line_width=2, line_dash="dot", line_color="green",
                    annotation_text="Rise Time", annotation_position="top right"
                )

            if not np.isnan(settling_time):
                fig.add_vline(
                    x=settling_time,
                    line_width=2, line_dash="dot", line_color="orange",
                    annotation_text="Settling Time", annotation_position="top right"
                )

            if not np.isnan(overshoot_time):
                fig.add_trace(go.Scatter(
                    x=[overshoot_time], y=[np.max(y_ml)],
                    mode="markers+text",
                    name="Overshoot",
                    text=["Overshoot"],
                    textposition="bottom center",
                    marker=dict(size=10, color="red")
                ))

            # === Add Zoom Slider ===
            #fig.update_layout(
                #xaxis=dict(rangeslider=dict(visible=True))
            #)

            # === Optional: Metric Summary Below Plot ===


            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            Kp_ml = Ki_ml = Kd_ml = None  # prevent downstream crash
            
            def compute_and_plot_control_effort(K, T1, T2, Td, Kp, Ki, Kd, T_final=100, N=1000):
                # Time vector
                t = np.linspace(0, T_final, N)
                dt = t[1] - t[0]

                # Transfer function G(s)
                den = [T1, 1] if T2 == 0 else np.polymul([T1, 1], [T2, 1])
                G = tf([K], den)
                if Td > 0:
                    G *= tf([1], [1, Td])

                # PID Controller C(s)
                s = tf([1, 0], [1])
                C = Kp + Ki / s + Kd * s

                # Closed-loop system and step response
                sys_cl = feedback(C * G, 1)
                t, y = step(sys_cl, T=t)

                # Step input and error signal
                w = np.ones_like(t) * K
                e = w - y

                # Control effort
                u = Kp * e + Ki * np.cumsum(e) * dt + Kd * np.gradient(e, dt)

                # Plot control effort
                """fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(t, u, label="Control Effort $u(t)$", color="tab:red")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Control Signal $u(t)$")
                ax.set_title("Control Effort over Time")
                ax.grid(True)
                ax.legend()

                return fig  # for display in Streamlit
                dt = t_ml[1] - t_ml[0]
                w = np.ones_like(t_ml) * K       # Step input signal
                e = w - y                        # Error signal
                u = Kp_ml * e + Ki_ml * np.cumsum(e) * dt + Kd_ml * np.gradient(e, dt)
                T_final=t
                fig = compute_and_plot_control_effort(K, T1, T2, Td, Kp_ml, Ki_ml, Kd_ml)
                st.pyplot(fig)
                fig_u, ax_u = plt.subplots(figsize=(6, 3))
                ax_u.plot(t_ml, u, label="Control Effort $u(t)$", color="tab:red")
                ax_u.set_xlabel("Time [s]")
                ax_u.set_ylabel("Control Signal $u(t)$")
                ax_u.set_title("Control Effort for ML-PID")
                ax_u.grid(True)
                ax_u.legend()
                st.pyplot(fig_u)  # if using Streamlit' """


        except Exception as e:
            st.error(f"Prediction or simulation failed: {e}")




elif mode == "📊 Evaluate PID":
    st.info("Evaluate performance of a given PID configuration")

    # === Sidebar: Model Selection ===
    model_choice = st.sidebar.selectbox("Surrogate Model", ["MLP", "DGP"])

    # Proper t definition
    t = np.linspace(0, 100, 1000)  # or whatever range is needed

    t_start, t_end = st.sidebar.slider(
    "Time Window [s]",
    min_value=float(t[0]),
    max_value=float(t[-1]),
    value=(float(t[0]), float(t[-1])),
    step=1.0
)
    
    # === User Inputs ===
    K = st.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.0)
    T1 = st.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.number_input("T2 (2nd Time Constant in s)", min_value=0.0, max_value=50.0, value=10.0)
    Td = st.number_input("Td (Dead Time in s)", min_value=0.0, max_value=5.0, value=1.0)

    Kp = st.number_input("Kp", min_value=0.0, max_value=10.0, value=2.0)
    Ki = st.number_input("Ki", min_value=0.0, max_value=10.0, value=0.1)
    Kd = st.number_input("Kd", min_value=0.0, max_value=10.0, value=1.0)



    def predict_dgp(param, log_transform=True):
        base_path = os.path.join(os.path.dirname(__file__), "streamlit_models", "dgp")
        scaler = joblib.load(os.path.join(base_path, f"{param}_scaler.pkl"))
        model = SimpleDGPModel(input_dim=7, num_inducing=64)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.load_state_dict(torch.load(os.path.join(base_path, f"{param}_model.pth")))
        likelihood.load_state_dict(torch.load(os.path.join(base_path, f"{param}_likelihood.pth")))
        model.eval()
        likelihood.eval()

        X_dgp = pd.DataFrame({
            'K': [K], 'T1': [T1], 'T2': [T2],
            'Td': [Td], 'Kp': [Kp], 'Ki': [Ki], 'Kd': [Kd]
        })
        X_scaled = scaler.transform(X_dgp)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(X_tensor))
            mean = pred_dist.mean.item()
            std = pred_dist.stddev.item()

            if log_transform:
                mean_exp = np.exp(mean)
                std_exp = mean_exp * std
                return mean_exp, std_exp
            else:
                return mean, std


    if st.button("📊 Evaluate Performance", key="eval_button"):

        try:
            # === Prepare Input ===
            X_eval = pd.DataFrame({
                'K': [K], 'T1': [T1], 'T2': [T2],
                'Td': [Td], 'Kp': [Kp], 'Ki': [Ki], 'Kd': [Kd]
            })

            if model_choice == "MLP":
                model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
                model_path = os.path.join(model_dir, "model_surrogate_mlp.joblib")
                surrogate_model = joblib.load(model_path)
                prediction = surrogate_model.predict(X_eval)
                ise_pred, sse_pred, rise_time_pred, settling_time_pred, overshoot_pred = prediction[0]

            elif model_choice == "DGP":
                import torch
                import gpytorch
                import joblib
                from gp_model import DGPModel

                class SimpleDGPModel(gpytorch.models.ApproximateGP):
                    def __init__(self, input_dim, num_inducing=64):
                        inducing_points = torch.randn(num_inducing, input_dim)
                        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
                        variational_strategy = gpytorch.variational.VariationalStrategy(
                            self, inducing_points, variational_distribution, learn_inducing_locations=True
                        )
                        super().__init__(variational_strategy)

                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) +
                            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
                        )

                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

                    def predict_dgp(param, log_transform=True):
                        base_path = os.path.join(os.path.dirname(__file__), "streamlit_models", "dgp")

                        # === Load Scaler and Model ===
                        scaler = joblib.load(os.path.join(base_path, f"{param}_scaler.pkl"))
                        model = SimpleDGPModel(input_dim=7, num_inducing=64)
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()

                        model.load_state_dict(torch.load(os.path.join(base_path, f"{param}_model.pth")))
                        likelihood.load_state_dict(torch.load(os.path.join(base_path, f"{param}_likelihood.pth")))
                        model.eval()
                        likelihood.eval()

                        # === Prepare Input ===
                        X_dgp = pd.DataFrame({
                            'K': [K], 'T1': [T1], 'T2': [T2],
                            'Td': [Td],  # or use L if needed
                            'Kp': [Kp], 'Ki': [Ki], 'Kd': [Kd]
                        })
                        X_scaled = scaler.transform(X_dgp)
                        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                        # === Predict with GP ===
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            pred_dist = likelihood(model(X_tensor))
                            mean = pred_dist.mean.item()
                            std = pred_dist.stddev.item()

                            if log_transform:
                                mean_exp = np.exp(mean)
                                std_exp = mean_exp * std  # ∂exp ≈ exp(x)*Δx
                                return mean_exp, std_exp
                            else:
                                return mean, std


                ise_pred = predict_dgp("ISE_log", log_transform=True)
                rise_time_pred = predict_dgp("RiseTime_log", log_transform=True)
                settling_time_pred = predict_dgp("SettlingTime_log", log_transform=True)
                overshoot_pred = predict_dgp("Overshoot", log_transform=False)
                sse_pred = np.nan
                
                # === Predict DGP Surrogate Outputs with Uncertainty ===
                ise_pred, ise_std = predict_dgp("ISE_log", log_transform=True)
                rise_time_pred, rise_time_std = predict_dgp("RiseTime_log", log_transform=True)
                settling_time_pred, settling_time_std = predict_dgp("SettlingTime_log", log_transform=True)
                overshoot_pred, overshoot_std = predict_dgp("Overshoot", log_transform=False)
                sse_pred = np.nan


            # === Simulate Closed-Loop System ===
            if T2 > 0:
                den = np.convolve([T1, 1], [T2, 1])
            else:
                den = [T1, 1]
            G = control.tf([K], den)

            if Td > 0:
                try:
                    num, den_delay = control.pade(Td, 1)
                    G = control.tf(num, den_delay) * G
                except:
                    st.warning("Pade approximation failed; skipping dead time.")

            P = control.tf([Kp], [1])
            I = control.tf([Ki], [1, 0])
            D = control.tf([Kd, 0], [1])
            C = P + I + D
            sys_cl = control.feedback(C * G, 1)

            t = np.linspace(0, 100, 2000)
            t, y = control.step_response(sys_cl, T=t)

            u = np.ones_like(t)
            e = u - y
            ise_true = simpson(e**2, t)
            sse_true = abs(1 - y[-1])
            overshoot_true = (np.max(y) - 1) * 100

            try:
                final_val = y[-1]
                rise_start = np.where(y >= 0.1 * final_val)[0][0]
                rise_end = np.where(y >= 0.9 * final_val)[0][0]
                rise_time_true = t[rise_end] - t[rise_start]
            except:
                rise_time_true = np.nan

            try:
                tolerance = 0.05 * final_val
                within_bounds = np.abs(y - final_val) <= tolerance
                settling_time_true = t[-1]
                for i in range(len(y)):
                    if np.all(within_bounds[i:]):
                        settling_time_true = t[i]
                        break
            except:
                settling_time_true = np.nan

            # === Display Comparison Table ===
            st.markdown("### 📊 Performance: Surrogate vs Simulation")

            # Format predicted values depending on model choice
            def format_metric(value, std=None):
                if std is not None:
                    return f"{value:.2f}".replace(".", ",") + " ± " + f"{std:.2f}".replace(".", ",")
                else:
                    return f"{value:.4f}".replace(".", ",")

            if model_choice == "DGP":
                predicted_values = [
                    format_metric(ise_pred, ise_std),
                    f"{sse_pred:.5f}".replace(".", ",") if not np.isnan(sse_pred) else "—",
                    format_metric(overshoot_pred, overshoot_std),
                    format_metric(settling_time_pred, settling_time_std),
                    format_metric(rise_time_pred, rise_time_std),
                ]
            else:
                predicted_values = [
                    f"{ise_pred:.4f}".replace(".", ","),
                    f"{sse_pred:.5f}".replace(".", ","),
                    f"{overshoot_pred:.2f}".replace(".", ","),
                    f"{settling_time_pred:.2f}".replace(".", ","),
                    f"{rise_time_pred:.2f}".replace(".", ","),
                ]

            df_compare = pd.DataFrame({
                "Metric": ["ISE", "SSE", "Overshoot [%]", "Settling Time [s]", "Rise Time [s]"],
                "Predicted": predicted_values,
                "Simulated": [
                    f"{ise_true:.4f}".replace(".", ","), 
                    f"{sse_true:.5f}".replace(".", ","), 
                    f"{overshoot_true:.2f}".replace(".", ","), 
                    f"{settling_time_true:.2f}".replace(".", ","), 
                    f"{rise_time_true:.2f}".replace(".", ",")
                ]
            })
            
            st.table(df_compare)
            
        

            # === Define step input: constant 1 after t=0 ===
            step_input = np.ones_like(t)

            # === Step Response Plot ===
            st.markdown("#### 🧪 Closed-Loop Step Response")

            fig_step = go.Figure()
            fig_step.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Simulated Response', line=dict(width=2)))
            fig_step.add_trace(go.Scatter(
                x=t, y=step_input,
                mode='lines',
                name='Step Input (0 → 1)',
                line=dict(color='black', dash='dash'),
                opacity=0.6
            ))

            fig_step.update_layout(
                title="Step Response of G(s) + PID",
                xaxis=dict(
                    title="Time [s]",
                    rangeslider=dict(visible=False),
                            range=[t_start, t_end],

                    rangeselector=dict(
                        buttons=list([
                            dict(count=10, label="10s", step="second", stepmode="backward"),
                            dict(count=25, label="25s", step="second", stepmode="backward"),
                            dict(count=50, label="50s", step="second", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                yaxis=dict(
                    title="Output",
                    range=[0, max(1.2, np.max(y))]
                ),
                legend=dict(
                    x=1, y=0, xanchor='right', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
                ),
                template='plotly_white'
            )

            st.plotly_chart(fig_step, use_container_width=True)


            # === Error Curve Plot ===
            st.markdown("#### 📉 Error Curve $e(t)$")

            fig_error = go.Figure()
            fig_error.add_trace(go.Scatter(
                x=t, y=e, mode='lines', name='Tracking Error', line=dict(color='red')
            ))

            fig_error.update_layout(
                title="Error Signal Over Time",
                xaxis=dict(
                    title="Time [s]",
                    rangeslider=dict(visible=False),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=10, label="10s", step="second", stepmode="backward"),
                            dict(count=25, label="25s", step="second", stepmode="backward"),
                            dict(count=50, label="50s", step="second", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                yaxis=dict(title="e(t)"),
                template='plotly_white'
            )

            st.plotly_chart(fig_error, use_container_width=True)

            
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")


elif mode == "⚙️ Optimize PID":
    st.info("Use ML-guided optimization to find best PID")
    
    model_choice = st.sidebar.selectbox("Surrogate Model", ["MLP", "DGP"])
    st.markdown("#### Define Optimization Weights")
    w_ise = st.slider("ISE Weight", 0.0, 1.0, 0.5)
    w_os = st.slider("Overshoot Weight", 0.0, 1.0, 0.2)
    w_st = st.slider("Settling Time Weight", 0.0, 1.0, 0.2)
    w_rt = st.slider("Rise Time Weight", 0.0, 1.0, 0.1)

    st.markdown("#### Define Performance Constraints")
    c1, c2 = st.columns(2)
    with c1:
        max_ise = st.number_input("Max ISE", min_value=0.0, max_value=100.0, value=25.0)
        max_st = st.number_input("Max Settling Time", min_value=0.0, max_value=300.0, value=100.0)
        max_sse = st.number_input("Max SSE", min_value=0.0, max_value=1.0, value=0.5)
    with c2:
        max_os = st.number_input("Max Overshoot (%)", min_value=0.0, max_value=100.0, value=50.0)
        max_rt = st.number_input("Max Rise Time", min_value=0.0, max_value=200.0, value=50.0)

    st.sidebar.markdown("**Plant Parameter**")
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=4.5)
    T1 = st.sidebar.number_input("T1", min_value=1.0, max_value=50.0, value=10.0)
    T2 = st.sidebar.number_input("T2", min_value=0.0, max_value=50.0, value=3.0)
    Td = st.sidebar.number_input("Td (Dead Time)", min_value=0.0, max_value=5.0, value=0.6)
    def predict_dgp(param, log_transform=True):
        #h = os.path.join(os.path.dirname(__file__), "streamlit_models", "dgp")
        base_path = os.path.join(os.path.dirname(__file__), "streamlit", "streamlit_models", "dgp")

        scaler = joblib.load(os.path.join(base_path, f"{param}_scaler.pkl"))
        model = SimpleDGPModel(input_dim=7, num_inducing=64)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        model_file = os.path.join(base_path, f"{param}_model.pth")
        likelihood_file = os.path.join(base_path, f"{param}_likelihood.pth")
        scaler_file = os.path.join(base_path, f"{param}_scaler.pkl")

        print("🔍 Checking paths:")
        print("Model Path      :", model_file)
        print("Likelihood Path :", likelihood_file)
        print("Scaler Path     :", scaler_file)

        assert os.path.exists(model_file), f"❌ Model file not found: {model_file}"
        assert os.path.exists(likelihood_file), f"❌ Likelihood file not found: {likelihood_file}"
        assert os.path.exists(scaler_file), f"❌ Scaler file not found: {scaler_file}"

        
        model.load_state_dict(torch.load(os.path.join(base_path, f"{param}_model.pth")))
        likelihood.load_state_dict(torch.load(os.path.join(base_path, f"{param}_likelihood.pth")))
        model.eval()
        likelihood.eval()

        X_dgp = pd.DataFrame({
            'K': [K], 'T1': [T1], 'T2': [T2],
            'Td': [Td], 'Kp': [Kp], 'Ki': [Ki], 'Kd': [Kd]
        })
        X_scaled = scaler.transform(X_dgp)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(X_tensor))
            mean = pred_dist.mean.item()
            std = pred_dist.stddev.item()

            if log_transform:
                mean_exp = np.exp(mean)
                std_exp = mean_exp * std
                return mean_exp, std_exp
            else:
                return mean, std
            
    if model_choice == "MLP":
                model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
                model_path = os.path.join(model_dir, "model_surrogate_mlp.joblib")
                surrogate_model = joblib.load(model_path)
                #prediction = surrogate_model.predict(X_eval)
                #ise_pred, sse_pred, rise_time_pred, settling_time_pred, overshoot_pred = prediction[0]

    elif model_choice == "DGP":
                import torch
                import gpytorch
                import joblib
                from gp_model import DGPModel

                class SimpleDGPModel(gpytorch.models.ApproximateGP):
                    def __init__(self, input_dim, num_inducing=64):
                        inducing_points = torch.randn(num_inducing, input_dim)
                        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
                        variational_strategy = gpytorch.variational.VariationalStrategy(
                            self, inducing_points, variational_distribution, learn_inducing_locations=True
                        )
                        super().__init__(variational_strategy)

                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) +
                            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
                        )

                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

                    def predict_dgp(param, log_transform=True):
                        base_path = os.path.join(os.path.dirname(__file__), "streamlit_models", "dgp")

                        # === Load Scaler and Model ===
                        scaler = joblib.load(os.path.join(base_path, f"{param}_scaler.pkl"))
                        model = SimpleDGPModel(input_dim=7, num_inducing=64)
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()

                        model.load_state_dict(torch.load(os.path.join(base_path, f"{param}_model.pth")))
                        likelihood.load_state_dict(torch.load(os.path.join(base_path, f"{param}_likelihood.pth")))
                        model.eval()
                        likelihood.eval()

                        # === Prepare Input ===
                        X_dgp = pd.DataFrame({
                            'K': [K], 'T1': [T1], 'T2': [T2],
                            'Td': [Td],  # or use L if needed
                            'Kp': [Kp], 'Ki': [Ki], 'Kd': [Kd]
                        })
                        X_scaled = scaler.transform(X_dgp)
                        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                        # === Predict with GP ===
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            pred_dist = likelihood(model(X_tensor))
                            mean = pred_dist.mean.item()
                            std = pred_dist.stddev.item()

                            if log_transform:
                                mean_exp = np.exp(mean)
                                std_exp = mean_exp * std  # ∂exp ≈ exp(x)*Δx
                                return mean_exp, std_exp
                            else:
                                return mean, std


                #ise_pred = predict_dgp("ISE_log", log_transform=True)
                #rise_time_pred = predict_dgp("RiseTime_log", log_transform=True)
                #settling_time_pred = predict_dgp("SettlingTime_log", log_transform=True)
                #overshoot_pred = predict_dgp("Overshoot", log_transform=False)
                #sse_pred = np.nan
                
                # === Predict DGP Surrogate Outputs with Uncertainty ===
                #ise_pred, ise_std = predict_dgp("ISE_log", log_transform=True)
                #rise_time_pred, rise_time_std = predict_dgp("RiseTime_log", log_transform=True)
                #settling_time_pred, settling_time_std = predict_dgp("SettlingTime_log", log_transform=True)
                #overshoot_pred, overshoot_std = predict_dgp("Overshoot", log_transform=False)
                #sse_pred = np.nan
            
    #model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
    #model_path = os.path.join(model_dir, "model_surrogate_mlp.joblib")
    #surrogate_model = joblib.load(model_path)



    if st.button("⚙️ Run Optimization", key="optimize_button"):

        weights = {
            "ISE": w_ise,
            "Overshoot": w_os,
            "SettlingTime": w_st,
            "RiseTime": w_rt
        }

        constraints = {
            "ISE": max_ise,
            "Overshoot": max_os,
            "SettlingTime": max_st,
            "RiseTime": max_rt,
            "SSE": max_sse
        }

        from utils.optimize_pid import optimize_pid_for_system
        try:
            #Kp, Ki, Kd, ise, os, stime, rtime, sse, top5_df = optimize_pid_for_system(
                        #K, T1, T2, Td, model_choice, weights, constraints
                    #)
            result = optimize_pid_for_system(K, T1, T2, Td, model_choice, weights, constraints)

            if result is None or not result.get('success', False):
                st.error("❌ Optimization failed: " + (result.get('message') or result.get('error') or 'Unknown error'))
            else:
                best_params = result['best_params']
                Kp, Ki, Kd = best_params

                best_metrics = result['best_metrics']

                if model_choice == "MLP":
                    # MLP returns means only (5 values), set std=0
                    ise, sse, rtime, stime, os = best_metrics
                    ise_std = sse_std = rtime_std = stime_std = os_std = 0.0
                elif model_choice == "DGP":
                    # DGP returns tuple (means, stds)
                    means, stds = best_metrics
                    ise, rtime, stime, os = means
                    ise_std, rtime_std, stime_std, os_std = stds
                    sse = 0.0
                    sse_std = 0.0
                else:
                    # Fallback values
                    ise = sse = rtime = stime = os = None
                    ise_std = sse_std = rtime_std = stime_std = os_std = 0.0



                st.success("✅ Optimization complete!")
                st.markdown("#### Optimal PID Parameters")
                st.write(f"Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")

            
            
                        # === Simulate Closed-Loop with Optimal PID for true metrics ===
            if T2 > 0:
                G = control.tf([K], np.convolve([T1, 1], [T2, 1]))
            else:
                G = control.tf([K], [T1, 1])

            P = control.tf([Kp], [1])
            I = control.tf([Ki], [1, 0])
            D = control.tf([Kd, 0], [1])
            C = P + I + D

            if Td > 0:
                num_d, den_d = control.pade(Td, 1)
                G_delay = control.tf(num_d, den_d)
                G = control.series(G_delay, G)

            sys_cl = control.feedback(C * G, 1)
            t_sim = np.linspace(0, max(2 * (T1 + T2 + Td), 100), 1000)
            t_sim, y_sim = control.step_response(sys_cl, t_sim)

            # === Recalculate actual performance metrics ===
            u_sim = np.ones_like(t_sim)
            e_sim = u_sim - y_sim
            ise_sim = simpson(e_sim**2, t_sim)
            sse_sim = abs(1.0 - y_sim[-1])
            overshoot_sim = (np.max(y_sim) - 1.0) * 100

            # Rise Time
            try:
                t_10 = t_sim[np.where(y_sim >= 0.1 * y_sim[-1])[0][0]]
                t_90 = t_sim[np.where(y_sim >= 0.9 * y_sim[-1])[0][0]]
                rise_time_sim = t_90 - t_10
            except:
                rise_time_sim = np.nan

            # Settling Time (±5%)
            tol = 0.05 * abs(y_sim[-1])
            within_bounds = (y_sim >= 1.0 - tol) & (y_sim <= 1.0 + tol)
            settling_time_sim = np.nan
            for i in range(len(y_sim)):
                if np.all(within_bounds[i:]):
                    settling_time_sim = t_sim[i]
                    break

            st.markdown("#### 📊 Performance Comparison: Surrogate vs Simulation")

            
            combined_df = pd.DataFrame({
                "Metric": ["ISE", "Overshoot (%)", "Settling Time (s)", "Rise Time (s)", "SSE"],
                "Optimized (Predicted)": [
                    f"{ise:.2f} ± {ise_std:.2f}",
                    f"{os:.2f} ± {os_std:.2f}",
                    f"{stime:.2f} ± {stime_std:.2f}",
                    f"{rtime:.2f} ± {rtime_std:.2f}",
                    f"{sse:.2f} ± {sse_std:.2f}"
                ],
                "Simulated (True)": [
                    f"{ise_sim:.2f}", f"{overshoot_sim:.2f}",
                    f"{settling_time_sim:.2f}", f"{rise_time_sim:.2f}", f"{sse_sim:.2f}"
                ]
            })

            st.table(data=combined_df)

            
            
            evaluated = result.get('evaluated_controllers', [])
            if not evaluated:
                st.info("No evaluated controllers available.")
            else:
                feasible_controllers = pd.DataFrame(evaluated)

                unique_controllers = []

                for idx, candidate in feasible_controllers.iterrows():
                    candidate_params = candidate[['Kp', 'Ki', 'Kd']].values

                    if not unique_controllers:
                        unique_controllers.append(candidate)
                        continue

                    differences = [
                        np.abs(candidate_params - np.array(ctrl[['Kp', 'Ki', 'Kd']]))
                        for ctrl in unique_controllers
                    ]

                    is_different = all(np.any(diff >= 0.5) for diff in differences)

                    if is_different:
                        unique_controllers.append(candidate)
                    else:
                        st.write("→ Controller skipped (too similar).")

                    if len(unique_controllers) >= 5:
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

                st.markdown("#### 🏆 Top 5 Distinct PID Controllers")
                st.dataframe(top5_df.style.format({
                    'Kp': '{:.3f}', 'Ki': '{:.3f}', 'Kd': '{:.3f}',
                    'ISE': '{:.2f}', 'Overshoot': '{:.2f}', 'SettlingTime': '{:.2f}',
                    'RiseTime': '{:.2f}', 'SSE': '{:.3f}', 'Cost': '{:.2f}'
                }))


            # === Step Response ===

            st.markdown("#### 📈 Step Responses of Top 5 Controllers")

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig2, ax2 = plt.subplots(figsize=(8, 4))

            t = np.linspace(0, max(2 * (T1 + T2 + Td), 100), 1000)
            step_input = np.ones_like(t)

            for idx, row in top5_df.iterrows():
                if pd.isna(row["Kp"]):
                    continue  # skip padded rows

                Kp_i, Ki_i, Kd_i = row["Kp"], row["Ki"], row["Kd"]

                # === Plant
                if T2 > 0:
                    G = control.tf([K], np.convolve([T1, 1], [T2, 1]))
                else:
                    G = control.tf([K], [T1, 1])

                # === PID Controller
                P = control.tf([Kp_i], [1])
                I = control.tf([Ki_i], [1, 0])
                D = control.tf([Kd_i, 0], [1])
                C = P + I + D

                if Td > 0:
                    num, den = control.pade(Td, 1)
                    G_delay = control.tf(num, den)
                    G = control.series(G_delay, G)

                sys_cl = control.feedback(C * G, 1)

                try:
                    t_response, y_response = control.step_response(sys_cl, t)
                    e_response = step_input - y_response

                    ax1.plot(t_response, y_response, label=f"#{idx+1}: Kp={Kp_i:.2f}, Ki={Ki_i:.2f}, Kd={Kd_i:.2f}")
                    ax2.plot(t_response, e_response, label=f"#{idx+1}")
                except Exception as e:
                    print(f"⚠️ Skipped controller #{idx+1} due to instability or simulation error: {e}")

            # Plot setpoint line
            ax1.plot(t, step_input, "--", color="black", label="Setpoint r(t)=1")


            # === Sidebar slider for zooming time
            t_start, t_end = st.sidebar.slider(
                "Time Window [s] (Top 5 Plot)",
                min_value=float(np.min(t)),
                max_value=float(np.max(t)),
                value=(float(np.min(t)), float(np.max(t))),
                step=1.0
            )

            # === Step Response Plot (Top 5 Controllers)
            st.markdown("#### 🧪 Step Responses of Top 5 Controllers")
            fig_step5 = go.Figure()
            fig_error5 = go.Figure()

            #for idx, (Kp_i, Ki_i, Kd_i) in enumerate(top_5_pid_params):
            for idx, row in top5_df.iterrows():
                if pd.isna(row["Kp"]):
                    continue  # skip padded rows

                Kp_i, Ki_i, Kd_i = row["Kp"], row["Ki"], row["Kd"]


                try:
                    P = control.tf([Kp_i], [1])
                    I = control.tf([Ki_i], [1, 0])
                    D = control.tf([Kd_i, 0], [1])
                    C = P + I + D
                    sys_cl = control.feedback(C * G, 1)

                    t_response, y_response = control.step_response(sys_cl, t)
                    e_response = 1.0 - y_response

                    fig_step5.add_trace(go.Scatter(
                        x=t_response,
                        y=y_response,
                        mode='lines',
                        name=f"#{idx+1}: Kp={Kp_i:.2f}, Ki={Ki_i:.2f}, Kd={Kd_i:.2f}"
                    ))

                    fig_error5.add_trace(go.Scatter(
                        x=t_response,
                        y=e_response,
                        mode='lines',
                        name=f"#{idx+1}"
                    ))

                except Exception as e:
                    print(f"⚠️ Skipped controller #{idx+1} due to instability or simulation error: {e}")

            # Add setpoint line
            fig_step5.add_trace(go.Scatter(
                x=t, y=step_input,
                mode='lines',
                name="Setpoint r(t)=1",
                line=dict(color='black', dash='dash'),
                opacity=0.6
            ))

            # === Layout for Step Response
            fig_step5.update_layout(
                title="Step Response of Top 5 Controllers",
                xaxis=dict(title="Time [s]", range=[t_start, t_end], rangeslider=dict(visible=False)),
                yaxis=dict(title="Output y(t)", range=[0, max(1.2, np.max(step_input))]),
                legend=dict(
                    x=1, y=0, xanchor='right', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
                ),
                template="plotly_white"
            )

            # === Layout for Error Curve
            fig_error5.update_layout(
                title="Tracking Error of Top 5 Controllers",
                xaxis=dict(title="Time [s]", range=[t_start, t_end], rangeslider=dict(visible=False)),
                yaxis=dict(title="Error e(t)"),
                legend=dict(
                    x=1, y=0, xanchor='right', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1
                ),
                template="plotly_white"
            )

            # === Display in Streamlit
            st.plotly_chart(fig_step5, use_container_width=True)
            st.plotly_chart(fig_error5, use_container_width=True)


        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")



                
elif mode == "🧪 Simulink Validation":
    st.success("✅ Entered Simulink Validation mode")  # Debug marker
    st.header("🧪 Simulink-in-the-Loop Validation")
    st.markdown("Run your controller on a real Simulink model and compare the result.")
    st.markdown("""To validate the performance of the machine learning–optimized PID controllers under realistic conditions, a **Simulink-in-the-loop** simulation setup was implemented.

    The optimized controller parameters are exported from Python to MATLAB via a `.mat` file. MATLAB then runs a predefined Simulink model using these values and returns the step response of the system.

    ⚠️ **Note**: Due to MATLAB’s academic licensing, this feature **only works locally**. It cannot be run in cloud environments like Streamlit Cloud.

    To run the Simulink evaluation on your own machine:

    1. **Clone the repository**  
    git clone https://github.com/your-username/PID-Controller-optimization-with-machine-learning.git

    2. **Update file paths**  
    Adjust paths in `run_simulink_pid.m` and related Python scripts (e.g. `simulate_controller.py`) to match your local paths.

    3. **Start the app locally**  
    Run `streamlit run app.py` and open the **🧪 Simulink Validation** section.

    This allows you to **compare the ML-optimized controller response directly with MATLAB Simulink simulation**, ensuring practical and reliable validation.
    """)

    # === Input fields ===
    st.subheader("System Parameters")
    K = st.number_input("K (Gain)", min_value=0.1, max_value=5.0, value=1.5)
    T1 = st.number_input("T1 (Time Constant 1)", min_value=0.01, max_value=50.0, value=12.0)
    T2 = st.number_input("T2 (Time Constant 2)", min_value=0.01, max_value=50.0, value=4.0)

    st.subheader("PID Parameters")
    Kp = st.number_input("Kp", min_value=0.0, max_value=20.0, value=1.2)
    Ki = st.number_input("Ki", min_value=0.0, max_value=20.0, value=0.4)
    Kd = st.number_input("Kd", min_value=0.0, max_value=20.0, value=0.2)

    if st.button("▶️ Run Simulation"):
        with st.spinner("Running MATLAB simulation..."):
            try:
                from utils.simulink_runner import run_simulink_simulation
                results = run_simulink_simulation(K, T1, T2, Kp, Ki, Kd)

                st.success("✅ Simulation completed successfully.")
                st.subheader("📈 Step Response")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(results['t'], results['y'], label='y(t)')
                ax.plot(results['t'], results['u'], label='u(t)', linestyle='--')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Output / Control")
                ax.legend()
                st.pyplot(fig)

                st.subheader("📊 Performance Metrics")
                st.write({
                    "ISE": results['ISE'],
                    "SSE": results['SSE'],
                    "Overshoot": results['Overshoot'],
                    "Rise Time": results['RiseTime'],
                    "Settling Time": results['SettlingTime'],
                })

            except Exception as e:
                st.error(f"❌ Simulation failed:\n{e}")
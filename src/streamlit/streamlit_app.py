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
from openai import OpenAI

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
    #st.sidebar.success(f"Loaded Groq API key: {st.secrets['GROQ_API_KEY'][:5]}...✅")

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
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=5.0)
    T1 = st.sidebar.number_input("T1", min_value=0.0, max_value=50.0, value=2.0)
    T2 = st.sidebar.number_input("T2", min_value=0.0, max_value=50.0, value=1.00)
    Td = st.sidebar.number_input("Td", min_value=0.0, max_value=5.0, value=0.0) 
    w0 = st.sidebar.number_input("w0", min_value=0.0, max_value=10.0, value=0.0)
    zeta = st.sidebar.number_input("zeta", min_value=0.0, max_value=1.0, value=0.0)
    Tchar = st.sidebar.number_input("Tchar", min_value=0.0, max_value=50.0, value=0.0)
    Family = st.sidebar.selectbox("Family", ["PT1PT2_existing","PT2_osc", "IT1", "P"])

    st.sidebar.markdown("**Plot Settings**")
    t_max = st.sidebar.slider("Simulation Time [s]", 1, 300, 20, key="slider_t_max")
    y_max = st.sidebar.slider("Y-Axis Max (Output)", 1.0, 50000.0, 1.5, step=0.1, key="slider_y_max")


    if st.button("🔍 Predict PID"):
        st.session_state.predict_clicked = True

        import os, joblib, numpy as np, pandas as pd

        # Where your models live
        model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")

        # ==============================
        # XGB helpers (self-contained)
        # ==============================
        ASSUME_LOG_TARGETS = True  # set False if you trained without log1p(target)

        # Family-specific feature layouts
        _FAMILY_FEATURES = {
            "with": {
                "PT2_osc":         ["logK","logL1p","zeta","logw0p","wc","PhaseMargin","focus_balanced","focus_reference-tracking","focus_disturbance-rejection"],
                "PT1PT2_existing": ["logK","logL1p","logT1p","logT2p","wc","PhaseMargin","focus_balanced","focus_reference-tracking","focus_disturbance-rejection"],
                "IT1":             ["logK","logL1p","logT1p","wc","PhaseMargin","focus_balanced","focus_reference-tracking","focus_disturbance-rejection"],
                "P":               ["logK","logL1p","wc","PhaseMargin","focus_balanced","focus_reference-tracking","focus_disturbance-rejection"],
            },
            "without": {
                "PT2_osc":         ["logK","logL1p","zeta","logw0p"],
                "PT1PT2_existing": ["logK","logL1p","logT1p","logT2p"],
                "IT1":             ["logK","logL1p","logT1p"],
                "P":               ["logK","logL1p"],
            }
        }

        def _log1p0(x: float) -> float:
            return float(np.log1p(max(0.0, x)))

        def _build_row(FamilySel: str, K, Td, T1, T2, w0, zeta, Tchar,
                    wc_default=3.0, pm_default=60, focus_default="balanced"):
            """Build both with/without-tuning feature rows. Returns (df_with, cols_with), (df_wo, cols_wo)."""
            feats = {
                "logK":      _log1p0(K),
                "logL1p":    _log1p0(Td),   # Td == L
                "logT1p":    _log1p0(T1),
                "logT2p":    _log1p0(T2),
                "logw0p":    _log1p0(w0),
                "logTcharp": _log1p0(Tchar),
                "zeta":      float(zeta),
                # tuning defaults (only used if scaler expects them)
                "wc": float(wc_default),
                "PhaseMargin": float(pm_default),
                "focus_balanced": 0.0,
                "focus_reference-tracking": 0.0,
                "focus_disturbance-rejection": 0.0,
            }
            key = f"focus_{focus_default}"
            if key in feats:
                feats[key] = 1.0

            cols_with = _FAMILY_FEATURES["with"][FamilySel]
            cols_wo   = _FAMILY_FEATURES["without"][FamilySel]

            row_with = {c: feats.get(c, 0.0) for c in cols_with}
            row_wo   = {c: feats.get(c, 0.0) for c in cols_wo}

            df_with = pd.DataFrame([row_with], columns=cols_with)
            df_wo   = pd.DataFrame([row_wo],   columns=cols_wo)
            return (df_with, cols_with), (df_wo, cols_wo)

        def _load_family_model(model_dir: str, FamilySel: str, TargetSel: str):
            model_path  = os.path.join(model_dir, f"{FamilySel}_{TargetSel}_xgb.pkl")
            scaler_path = os.path.join(model_dir, f"{FamilySel}_{TargetSel}_scaler.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Missing scaler: {scaler_path}")
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler

        def _predict_one(model, scaler, X_with, X_wo, assume_log_targets=True):
            """
            Choose the correct feature set (with/without tuning) based on scaler dimensionality,
            scale, predict, and (optionally) invert log1p.
            Returns: pred_value (float), debug (dict)
            """
            # Determine expected feature count
            n_expected = getattr(scaler, "n_features_in_", None)
            if n_expected is None:
                n_expected = getattr(scaler, "mean_", None)
                n_expected = len(n_expected) if n_expected is not None else X_with.shape[1]

            if X_with.shape[1] == n_expected:
                Xs = scaler.transform(X_with.values)
                used = "with"; used_cols = list(X_with.columns)
            elif X_wo.shape[1] == n_expected:
                Xs = scaler.transform(X_wo.values)
                used = "without"; used_cols = list(X_wo.columns)
            else:
                raise ValueError(
                    f"Scaler expects {n_expected} features, but got "
                    f"{X_with.shape[1]} (with) and {X_wo.shape[1]} (without)."
                )

            raw_pred = model.predict(Xs)
            pred = np.expm1(raw_pred) if assume_log_targets else raw_pred

            dbg = {
                "expected_features": int(n_expected),
                "used_set": used,
                "used_cols": used_cols,
                "Xs_shape": Xs.shape,
                "raw_pred": float(raw_pred[0]),
                "final_pred": float(pred[0]),
            }
            return float(max(0.0, pred[0])), dbg

        def predict_pid_gains_xgb(
            model_dir: str,
            Family: str,
            K: float, T1: float, T2: float, Td: float, w0: float, zeta: float, Tchar: float,
            assume_log_targets: bool = True,
            debug: bool = False,
        ):
            # Build both feature variants; auto-pick per-scaler inside _predict_one
            (X_with, _), (X_wo, _) = _build_row(
                FamilySel=Family,
                K=K, Td=Td, T1=T1, T2=T2, w0=w0, zeta=zeta, Tchar=Tchar,
                wc_default=3.0, pm_default=60, focus_default="balanced"
            )

            preds = {}
            for TargetSel in ["Kp", "Ki", "Kd"]:
                model, scaler = _load_family_model(model_dir, Family, TargetSel)
                pred_val, dbg = _predict_one(model, scaler, X_with, X_wo, assume_log_targets)

                if debug:
                    st.write(f"🔄 {Family}-{TargetSel}: used={dbg['used_set']} | "
                            f"expected={dbg['expected_features']} | Xs={dbg['Xs_shape']}")
                    st.write(f"    cols={dbg['used_cols']}")
                    st.write(f"    raw={dbg['raw_pred']:.6f} → final={dbg['final_pred']:.6f}")

                preds[TargetSel] = pred_val

            # Optional: enforce trivial values for P
            # if Family == "P":
            #     preds["Ki"] = 0.0
            #     preds["Kd"] = 0.0

            Kp = float(max(0.0, preds.get("Kp", 0.0)))
            Ki = float(max(0.0, preds.get("Ki", 0.0)))
            Kd = float(max(0.0, preds.get("Kd", 0.0)))
            return Kp, Ki, Kd

        # ==============================
        # Prediction branches
        # ==============================
        try:
            if model_choice in ["Random Forest", "MLP"]:
                X = np.array([[K, T1, T2, Td]])
                model_filename = f"model_{model_choice.lower().replace(' ', '_')}.joblib"
                model_path = os.path.join(model_dir, model_filename)
                model = joblib.load(model_path)

                try:
                    from utils.predict_pid import predict_pid_params
                except Exception as e:
                    st.error("⚠️ Could not import predict_pid_params from utils.predict_pid")
                    raise

                Kp, Ki, Kd = predict_pid_params(model, X)

            elif model_choice == "XGBoost":
                DEBUG_XGB = st.sidebar.checkbox("Debug XGBoost", value=False)
                Kp, Ki, Kd = predict_pid_gains_xgb(
                    model_dir=model_dir,
                    Family=Family,
                    K=K, T1=T1, T2=T2, Td=Td, w0=w0, zeta=zeta, Tchar=Tchar,
                    assume_log_targets=ASSUME_LOG_TARGETS,
                    debug=DEBUG_XGB,
                )

            elif model_choice == "Symbolic":
                # Minimal safe loader for symbolic (expects you trained with log10 target and invert with 10**y - 1e-6)
                def load_and_predict_symb(param, K, T1, T2):
                    import numpy as _np
                    try:
                        model = joblib.load(os.path.join(model_dir, f"symbolic_{param}.pkl"))
                    except Exception as _e:
                        raise FileNotFoundError(f"symbolic_{param}.pkl not found in {model_dir}")
                    Xs = np.array([[K, T1, T2]])
                    y_log = model.predict(Xs)[0]
                    return max(0.0, 10**y_log - 1e-6)

                Kp = load_and_predict_symb("kp", K, T1, T2)
                Ki = load_and_predict_symb("ki", K, T1, T2)
                Kd = load_and_predict_symb("kd", K, T1, T2)

            elif model_choice == "DGP":
                # Minimal, fixed-path DGP loader (uses base_path consistently)
                import torch, gpytorch
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

                def load_and_predict_dgp(param, X_raw, return_std=False):
                    base_path = model_dir  # keep consistent
                    x_scaler_path = os.path.join(base_path, f"dgp_{param.lower()}_scaler_X.pkl")
                    y_scaler_path = os.path.join(base_path, f"dgp_{param.lower()}_scaler_y.pkl")
                    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
                        raise FileNotFoundError(f"DGP scalers for {param} not found in {base_path}")
                    X_scaler = joblib.load(x_scaler_path)
                    y_scaler = joblib.load(y_scaler_path)

                    X_scaled = X_scaler.transform(X_raw)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = DGPModel(input_dim=X_tensor.shape[1]).to(device)
                    dummy_noise = torch.ones(X_tensor.size(0)).to(device) * 1e-4
                    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=dummy_noise).to(device)

                    model.load_state_dict(torch.load(os.path.join(base_path, f"dgp_{param.lower()}.pth"), map_location=device))
                    likelihood.load_state_dict(torch.load(os.path.join(base_path, f"dgp_{param.lower()}_likelihood.pth"), map_location=device))

                    model.eval(); likelihood.eval()
                    with torch.no_grad(), fast_pred_var():
                        preds = likelihood(model(X_tensor))
                        y_pred = preds.mean.item()
                        y_std = preds.variance.sqrt().item() if return_std else None

                    y_pred_inv = y_scaler.inverse_transform(np.array([[y_pred]]))[0][0]
                    if return_std:
                        upper = y_scaler.inverse_transform(np.array([[y_pred + y_std]]))[0][0]
                        lower = y_scaler.inverse_transform(np.array([[y_pred - y_std]]))[0][0]
                        return {"mean": y_pred_inv, "std": abs(upper - lower) / 2}
                    return y_pred_inv

                X_raw = np.array([[K, T1, T2, Td]])
                Kp = load_and_predict_dgp("kp", X_raw)
                Ki = load_and_predict_dgp("ki", X_raw)
                Kd = load_and_predict_dgp("kd", X_raw)

            else:
                st.error("❌ Unknown model type selected.")
                raise ValueError("Invalid model")

            # === Success & display ===
                    # === Predict (already done above) ===
            try:
                # Kp, Ki, Kd must be set by your model branches before this point
                Kp_ml, Ki_ml, Kd_ml = float(Kp), float(Ki), float(Kd)
                pred_ok = True
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                pred_ok = False

            if pred_ok:
                # === Success & display ===
                st.success("✅ Prediction complete!")
                Kp_str = f"{Kp_ml:.3f}"
                Ki_str = f"{Ki_ml:.5f}"
                Kd_str = f"{Kd_ml:.2f}"

                col1, col2, col3 = st.columns(3)
                col1.metric("Kp", Kp_str)
                col2.metric("Ki", Ki_str)
                col3.metric("Kd", Kd_str)

                # =========================
                # Simulation & visualization
                # =========================
                import numpy as _np
                import plotly.graph_objects as go
                from scipy.integrate import simpson
                # control library
                from control.matlab import tf, feedback, step as step_response
                from control import pade
                from utils.Wendetangente import ZieglerNicholsTuner
                import matplotlib.pyplot as plt
                from scipy import signal
                import warnings
                warnings.filterwarnings('ignore')
                tuner = ZieglerNicholsTuner()

                # ---------- Plant per Family ----------
                def plant_tf(Family, K, T1, T2, w0, zeta, Tchar, L):
                    """
                    Returns continuous-time plant transfer function for the selected Family.
                    Adds 1st-order Pade delay if L>0.
                    """
                    # Ensure positive-ish parameters to avoid singularities
                    K  = float(K)
                    T1 = float(max(T1, 1e-9))
                    T2 = float(max(T2, 0.0))
                    w0 = float(max(w0, 1e-9))
                    zeta = float(max(zeta, 0.0))
                    L  = float(max(L, 0.0))

                    if Family == "PT1PT2_existing":
                        den = _np.polymul([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
                        G = tf([K], den)

                    elif Family == "PT2_osc":
                        # K * ω0^2 / (s^2 + 2ζω0 s + ω0^2)
                        G = tf([K * (w0**2)], [1.0, 2.0*zeta*w0, w0**2])

                    elif Family == "IT1":
                        # IT1: K / (s*(T1 s + 1))
                        # denominator = s * (T1 s + 1) = T1 s^2 + s
                        G = tf([K], _np.polymul([T1, 1], [1, 0]))

                    elif Family == "P":
                        # Pure gain + optional delay
                        G = tf([K], [1.0])

                    else:
                        # Fallback: PT1/PT2
                        den = _np.polymul([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
                        G = tf([K], den)

                    if L > 0:
                        num_d, den_d = pade(L, 1)
                        G = G * tf(num_d, den_d)

                    return G

                # ---------- Controller & closed loop ----------
                def simulate_response(Family, K, T1, T2, L, w0, zeta, Tchar, Kp, Ki, Kd, T_final=100):
                    """Build plant per family, close loop with PID, return unit-step response."""
                    t = _np.linspace(0, T_final, 1000)
                    G = plant_tf(Family, K, T1, T2, w0, zeta, Tchar, L)

                    s = tf([1, 0], [1])
                    #C = Kp + Ki / s + Kd * s
                    N = 20.0
                    D_f = Kd * (N*s) / (1 + N*s)
                    C  = Kp + Ki/s + D_f
                    sys = feedback(C * G, 1)         # unity feedback
                    y, t = step_response(sys, T=t)    # unit step reference
                    return t, y

                # ---------- FOPDT rules (only when applicable) ----------
                def _safe_div(x, eps=1e-9): 
                    return x if abs(x) > eps else (eps if x >= 0 else -eps)

                def _build_open_loop_sys_for_wende(Family: str, K: float, T1: float, T2: float,
                                                Td: float, w0: float, zeta: float):
                    """
                    Create an open-loop TransferFunction (scipy.signal) + deadtime for Wendetangente.
                    Returns (sys, deadtime_seconds).
                    """
                    # PT1/PT2_existing
                    if Family == "PT1PT2_existing":
                        den = np.convolve([T1, 1.0], [T2, 1.0]) if T2 > 0 else [T1, 1.0]
                        sys = signal.TransferFunction([K], den)
                        return (sys, float(max(0.0, Td)))

                    # IT1
                    if Family == "IT1":
                        # G(s) = K / (s * (T1 s + 1)) -> den = [T1, 1, 0]
                        sys = signal.TransferFunction([K], [T1, 1.0, 0.0])
                        return (sys, float(max(0.0, Td)))

                    # Oscillatory second-order (underdamped)
                    if Family == "PT2_osc":
                        # G(s) = K*w0^2 / (s^2 + 2*zeta*w0*s + w0^2)
                        w0 = float(max(w0, 1e-9))
                        sys = signal.TransferFunction([K * (w0**2)], [1.0, 2.0*zeta*w0, w0**2])
                        return (sys, float(max(0.0, Td)))

                    # Pure gain – Wendetangente is not meaningful here (no dynamics)
                    if Family == "P":
                        sys = signal.TransferFunction([K], [1.0])
                        return (sys, float(max(0.0, Td)))

                    # Fallback: treat like PT1
                    den = [T1, 1.0]
                    sys = signal.TransferFunction([K], den)
                    return (sys, float(max(0.0, Td)))


                def compute_pid_sets_via_wendetangente(Family: str, *, 
                                                    K: float, T1: float, T2: float, Td: float,
                                                    w0: float, zeta: float,
                                                    t_final: float = 50.0, n_points: int = 2000):
                    """
                    Uses your Wendetangente methods to extract Tu, Tg, Ks from the open-loop step
                    and returns PID parameter sets for ZN and CHR.
                    """
                    tuner = ZieglerNicholsTuner()

                    # Build open-loop TF and simulate step with explicit deadtime shift
                    sys_tuple = _build_open_loop_sys_for_wende(Family, K, T1, T2, Td, w0, zeta)
                    t_ol, y_ol = tuner.simulate_step_response(sys_tuple, t_final=float(t_final), n_points=int(n_points))

                    # Extract Tu, Tg, Ks via Wendetangente
                    tuner.find_inflection_point(t_ol, y_ol)
                    tangent_fun = tuner.fit_tangent_line(t_ol, y_ol)
                    Tu, Tg, Ks = tuner.extract_wendetangenten_parameters(t_ol, y_ol, system_type=Family)

                    # Compute PID parameter sets from Tu/Tg/Ks
                    pid_sets = {
                        "ZN":          tuner.calculate_pid_parameters("ZN"),
                        "CHR 0%":      tuner.calculate_pid_parameters("CHR_aperiodic"),
                        "CHR 20%":     tuner.calculate_pid_parameters("CHR_20"),
                    }

                    return {
                        "t_open": t_ol, "y_open": y_ol,
                        "tangent": tangent_fun,
                        "Tu": Tu, "Tg": Tg, "Ks": Ks,
                        "pid_sets": pid_sets,
                        "tuner": tuner,  # to reuse its plotting helpers if desired
                    }

                # === Use Predicted PID ===
                L = Td  # clarity
                Kp_ml, Ki_ml, Kd_ml = Kp, Ki, Kd

                # Baselines only for FOPDT-like families with L>0 and T1>0
                #baselines = {}
                #if Family in {"PT1PT2_existing"} and (L > 0.0) and (T1 > 0.0):
                    #baselines["ZN"]      = zn_pid(K, T1, T2, L)
                    #baselines["CHR 0%"]  = chr_pid(K, T1, T2, L, overshoot=0)
                    #baselines["CHR 20%"] = chr_pid(K, T1, T2, L, overshoot=20)
                # For PT2_osc / IT1 / P we intentionally skip ZN/CHR (not meaningful)
                # --- NEW: Wendetangente-based ZN / CHR from open-loop step ---
                baselines = {}
                try:
                    wende = compute_pid_sets_via_wendetangente(
                        Family=Family,
                        K=float(K), T1=float(T1), T2=float(T2), Td=float(L),
                        w0=float(w0), zeta=float(zeta),
                        t_final=float(t_max), n_points=2000,
                    )
                    # Collect as (Kp, Ki, Kd)
                    for label, params in wende["pid_sets"].items():
                        baselines[label] = (params["Kp"], params["Ki"], params["Kd"])

                    # (Optional) details box
                    with st.expander("🧭 Wendetangente (Tu, Tg, Ks) & PID from step"):
                        st.write(
                            f"Tu = **{wende['Tu']:.3f} s**,  Tg = **{wende['Tg']:.3f} s**,  Ks = **{wende['Ks']:.3f}**"
                        )
                        st.code(
                            "\n".join([
                                "=== Ziegler–Nichols ===",
                                f"Kp = {wende['pid_sets']['ZN']['Kp']:.6f}",
                                f"Ki = {wende['pid_sets']['ZN']['Ki']:.6f}",
                                f"Kd = {wende['pid_sets']['ZN']['Kd']:.6f}",
                                "",
                                "=== CHR (0% OS) ===",
                                f"Kp = {wende['pid_sets']['CHR 0%']['Kp']:.6f}",
                                f"Ki = {wende['pid_sets']['CHR 0%']['Ki']:.6f}",
                                f"Kd = {wende['pid_sets']['CHR 0%']['Kd']:.6f}",
                                "",
                                "=== CHR (20% OS) ===",
                                f"Kp = {wende['pid_sets']['CHR 20%']['Kp']:.6f}",
                                f"Ki = {wende['pid_sets']['CHR 20%']['Ki']:.6f}",
                                f"Kd = {wende['pid_sets']['CHR 20%']['Kd']:.6f}",
                            ]),
                            language="text"
                        )

                        # Optional: show the open-loop step + tangent + Tu/Tg/Ks using your class' plot
                        try:
                            fig_wende = wende["tuner"].plot_analysis(
                                wende["t_open"], wende["y_open"], wende["tangent"],
                                wende["pid_sets"], system_type=Family
                            )
                            st.pyplot(fig_wende, use_container_width=True)
                        except Exception as _:
                            st.info("Open-loop Wendetangente plot skipped.")

                except Exception as ex:
                    st.warning(f"Wendetangente PID baseline unavailable: {ex}")

                # === Simulate ===
                t_ml, y_ml = simulate_response(Family, K, T1, T2, L, w0, zeta, Tchar, Kp_ml, Ki_ml, Kd_ml, T_final=t_max)
                sims = {"ML": (t_ml, y_ml)}

                for label, (Kp_b, Ki_b, Kd_b) in baselines.items():
                    t_b, y_b = simulate_response(Family, K, T1, T2, L, w0, zeta, Tchar, Kp_b, Ki_b, Kd_b, T_final=t_max)
                    sims[label] = (t_b, y_b)

                # === Calculation details expander (show only what exists) ===
                T_eff = T1 + (T2 if T2 > 0 else 0.0)
                with st.expander("🔧 Show Calculation Details"):
                    lines = [f"Effective time constant T ≈ {T_eff:.3f} s (heuristic)"]
                    if "ZN" in baselines:
                        Kp_zn, Ki_zn, Kd_zn = baselines["ZN"]
                        lines += [
                            "",
                            "=== Ziegler–Nichols (FOPDT) ===",
                            f"Kp = {Kp_zn:.6f}",
                            f"Ki = {Ki_zn:.6f}",
                            f"Kd = {Kd_zn:.6f}",
                        ]
                    if "CHR 0%" in baselines:
                        Kp_chr0, Ki_chr0, Kd_chr0 = baselines["CHR 0%"]
                        lines += [
                            "",
                            "=== CHR (0% OS) ===",
                            f"Kp = {Kp_chr0:.6f}",
                            f"Ki = {Ki_chr0:.6f}",
                            f"Kd = {Kd_chr0:.6f}",
                        ]
                    if "CHR 20%" in baselines:
                        Kp_chr20, Ki_chr20, Kd_chr20 = baselines["CHR 20%"]
                        lines += [
                            "",
                            "=== CHR (20% OS) ===",
                            f"Kp = {Kp_chr20:.6f}",
                            f"Ki = {Ki_chr20:.6f}",
                            f"Kd = {Kd_chr20:.6f}",
                        ]
                    st.code("\n".join(lines), language="text")

                with st.expander("🔧 Show Parameters"):
                    st.code(f"""
                    🔍 Inputs:
                        Family = {Family}
                        K  = {K:.3f}
                        T1 = {T1:.3f}
                        T2 = {T2:.3f}
                        L  = {L:.3f}
                        w0 = {w0:.3f}
                        ζ   = {zeta:.3f}
                        Tchar = {Tchar:.3f}

                    📊 ML-PID:
                        Kp = {Kp_ml:.6f}
                        Ki = {Ki_ml:.6f}
                        Kd = {Kd_ml:.6f}
                    """, language="text")

                # === Plot Step Responses ===
                ref = 1.0  # unit step
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_ml, y=y_ml, mode='lines', name='ML PID'))
                for label, (t_b, y_b) in sims.items():
                    if label == "ML": 
                        continue
                    style = dict(dash='dash') if "ZN" in label else dict(dash='dot')
                    fig.add_trace(go.Scatter(x=t_b, y=y_b, mode='lines', name=label, line=style))
                fig.add_trace(go.Scatter(x=t_ml, y=_np.ones_like(t_ml)*ref, mode='lines',
                                        name='Step Input (0→1)', line=dict(color='black', dash='dash')))

                fig.update_layout(
                    title="Closed-Loop Step Response",
                    xaxis=dict(title="Time [s]", tickmode='linear', tick0=0, dtick=1),
                    yaxis=dict(title="Output", tickmode='linear', tick0=0, dtick=0.2, range=[0, y_max]),
                    legend=dict(x=1, y=0, xanchor='right', yanchor='bottom', orientation='v',
                                bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1),
                    template='plotly_white'
                )

                # === Metrics (relative to unit step ref = 1.0) ===
                def compute_ise(t, y, ref=1.0):
                    e = ref - y
                    return simpson(e**2, t)

                def compute_metrics(t, y, ref=1.0):
                    # Rise time (10%->90% of ref)
                    try:
                        t_10 = t[_np.where(y >= 0.1 * ref)[0][0]]
                        t_90 = t[_np.where(y >= 0.9 * ref)[0][0]]
                        rise_time = t_90 - t_10
                    except IndexError:
                        rise_time = _np.nan
                    # Settling (±2% of ref)
                    band = 0.02 * abs(ref)
                    lower, upper = ref - band, ref + band
                    within = (y >= lower) & (y <= upper)
                    settling_time = _np.nan
                    for i in range(len(y)):
                        if _np.all(within[i:]):
                            settling_time = t[i]
                            break
                    # Overshoot relative to ref
                    overshoot = max(0.0, (float(y.max()) - ref) / max(1e-9, abs(ref)) * 100.0)
                    ise = compute_ise(t, y, ref=ref)
                    return rise_time, settling_time, overshoot, ise

                metrics = {}
                for label, (t, y) in sims.items():
                    metrics[label] = compute_metrics(t, y, ref=ref)

                metric_rows = [{
                    "Controller": lbl,
                    "Rise Time [s]": f"{rt:.2f}" if not _np.isnan(rt) else "—",
                    "Settling Time [s]": f"{stt:.2f}" if not _np.isnan(stt) else "—",
                    "Overshoot [%]": f"{os:.2f}" if not _np.isnan(os) else "—",
                    "ISE": f"{ise:.3f}" if not _np.isnan(ise) else "—",
                } for lbl, (rt, stt, os, ise) in metrics.items()]

                st.markdown("### 📊 Key Performance Metrics (All Controllers)")
                st.table(pd.DataFrame(metric_rows))

                # Annotate rise & settling for ML curve
                rt_ml, st_ml, os_ml, _ = metrics["ML"]
                if not _np.isnan(rt_ml):
                    fig.add_vline(x=rt_ml, line_width=2, line_dash="dot", line_color="green",
                                annotation_text="Rise Time", annotation_position="top right")
                if not _np.isnan(st_ml):
                    fig.add_vline(x=st_ml, line_width=2, line_dash="dot", line_color="orange",
                                annotation_text="Settling Time", annotation_position="top right")

                # Overshoot marker for ML
                if y_ml.max() > ref:
                    fig.add_trace(go.Scatter(
                        x=[t_ml[_np.argmax(y_ml)]], y=[float(y_ml.max())],
                        mode="markers+text", name="Overshoot", text=["Overshoot"],
                        textposition="bottom center", marker=dict(size=10)
                    ))

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")


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
                #mean_exp = np.exp(mean)
                mean_exp = mean
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
                                #mean_exp = np.exp(mean)
                                mean_exp = mean
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
                tolerance = 0.02 * final_val
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
    
    def satisfies_constraints(ctrl, constraints):
        for metric, max_val in constraints.items():
            if metric in ctrl and ctrl[metric] is not None:
                if ctrl[metric] > max_val:
                    return False
        return True


    import numpy as np
    from control import tfdata
    from numpy.polynomial import Polynomial

    def is_stable(system):

        _, den = tfdata(system)              # unpack numerator and denominator
        den_coeffs = np.squeeze(den)         # flatten to 1D array
        poles = np.roots(den_coeffs)         # compute poles
        return np.all(np.real(poles) < 0)    # check left-half plane



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
                #mean_exp = np.exp(mean)
                mean_exp = mean

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
                                #mean_exp = np.exp(mean)
                                mean_exp = mean
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
            tol = 0.02 * abs(y_sim[-1])
            within_bounds = (y_sim >= 1.0 - tol) & (y_sim <= 1.0 + tol)
            settling_time_sim = np.nan
            for i in range(len(y_sim)):
                if np.all(within_bounds[i:]):
                    settling_time_sim = t_sim[i]
                    break



                # === Stability check ===
                # === Construct Plant G ===
                if T2 > 0:
                    G_base = control.tf([K], np.convolve([T1, 1], [T2, 1]))
                else:
                    G_base = control.tf([K], [T1, 1])

                if Td > 0:
                    num_d, den_d = control.pade(Td, 1)
                    G_delay = control.tf(num_d, den_d)
                    G_base = G_delay * G_base

                # === Stability check for best_params ===
                def is_pid_stable(Kp, Ki, Kd):
                    P = control.tf([Kp], [1])
                    I = control.tf([Ki], [1, 0])
                    D = control.tf([Kd, 0], [1])
                    C = P + I + D
                    sys_cl = control.feedback(C * G_base, 1)
                    return is_stable(sys_cl), sys_cl

                is_stable_best, sys_cl = is_pid_stable(Kp, Ki, Kd)

                # === If unstable, fallback to stable controller from evaluated_controllers ===
                if not is_stable_best:
                    fallback = None
                    for ctrl in result["evaluated_controllers"]:
                        Kp_i, Ki_i, Kd_i = ctrl["Kp"], ctrl["Ki"], ctrl["Kd"]
                        stable, sys = is_pid_stable(Kp_i, Ki_i, Kd_i)
                        if stable:
                            fallback = (Kp_i, Ki_i, Kd_i, sys)
                            break

                    if fallback is not None:
                        Kp, Ki, Kd, sys_cl = fallback
                        #st.info("⚠️ Best controller was unstable. Using best *stable* fallback controller instead.")
                    else:
                        st.error("❌ All evaluated controllers are unstable. Try different weights or constraints.")
                        st.stop()  # Stop further execution


            st.success("✅ Optimization complete!")
            st.markdown("#### Optimal PID Parameters")
            st.caption("This controller minimizes the weighted cost based on your selected metrics.")

            #col1, col2, col3, col4 = st.columns(4)

            #col1.metric("Kp", f"{Kp:.4f}")
            #col2.metric("Ki", f"{Ki:.4f}")
            #col3.metric("Kd", f"{Kd:.4f}")
            #col4.metric("Cost", f"{Cost:.4f}")
            #st.write(f"Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")

            #st.markdown("#### 📊 Performance Comparison: Surrogate vs Simulation")

            
            #combined_df = pd.DataFrame({
                #"Metric": ["ISE", "Overshoot (%)", "Settling Time (s)", "Rise Time (s)", "SSE"],
                #"Optimized (Predicted)": [
                #    f"{ise:.2f} ± {ise_std:.2f}",
                #    f"{os:.2f} ± {os_std:.2f}",
                #    f"{stime:.2f} ± {stime_std:.2f}",
                #    f"{rtime:.2f} ± {rtime_std:.2f}",
                #    f"{sse:.2f} ± {sse_std:.2f}"
                #],
                #"Simulated (True)": [
                 #   f"{ise_sim:.2f}", f"{overshoot_sim:.2f}",
                  #  f"{settling_time_sim:.2f}", f"{rise_time_sim:.2f}", f"{sse_sim:.2f}"
                #]
            #})

            #st.table(data=combined_df)

            
            # === Collect and SIMULATE up to 50 controllers; then pick best 5 by SIM metrics ===
            evaluated = result.get('evaluated_controllers', [])
            if not evaluated:
                st.info("No evaluated controllers available.")
            else:
                raw_df = pd.DataFrame(evaluated)

                # --- Build a base plant (no controller) once ---
                if T2 > 0:
                    G_base = control.tf([K], np.convolve([T1, 1], [T2, 1]))
                else:
                    G_base = control.tf([K], [T1, 1])

                if Td > 0:
                    num_d, den_d = control.pade(Td, 1)
                    G_delay = control.tf(num_d, den_d)
                    G_base = G_delay * G_base

                # --- Helpers ---
                def simulate_true_metrics(Kp, Ki, Kd):
                    P = control.tf([Kp], [1])
                    I = control.tf([Ki], [1, 0])
                    D = control.tf([Kd, 0], [1])
                    C = P + I + D
                    sys_cl = control.feedback(C * G_base, 1)
                    # Stability gate
                    if not is_stable(sys_cl):
                        return None

                    t_sim = np.linspace(0, max(2.0*(T1 + max(T2,0) + Td), 100.0), 1000)
                    t_sim, y_sim = control.step_response(sys_cl, t_sim)
                    e_sim = 1.0 - y_sim

                    ise_sim = simpson(e_sim**2, t_sim)
                    sse_sim = abs(1.0 - y_sim[-1])
                    overshoot_sim = (np.max(y_sim) - 1.0) * 100.0

                    # Rise time (10–90 % of final value)
                    try:
                        y_final = y_sim[-1]
                        t_10 = t_sim[np.where(y_sim >= 0.1 * y_final)[0][0]]
                        t_90 = t_sim[np.where(y_sim >= 0.9 * y_final)[0][0]]
                        rise_time_sim = float(t_90 - t_10)
                    except Exception:
                        rise_time_sim = np.nan

                    # Settling time (±2 % band)
                    tol = 0.02 * abs(y_sim[-1])
                    settling_time_sim = np.nan
                    for i in range(len(y_sim)):
                        if np.all((y_sim[i:] >= 1.0 - tol) & (y_sim[i:] <= 1.0 + tol)):
                            settling_time_sim = float(t_sim[i])
                            break

                    return {
                        "ISE_sim": float(ise_sim),
                        "SSE_sim": float(sse_sim),
                        "Overshoot_sim": float(overshoot_sim),
                        "RiseTime_sim": float(rise_time_sim) if np.isfinite(rise_time_sim) else np.nan,
                        "SettlingTime_sim": float(settling_time_sim) if np.isfinite(settling_time_sim) else np.nan,
                        "sys_cl": sys_cl  # keep for optional later use
                    }

                def violates_constraints(sim_metrics, constraints):
                    # Use sim metrics only
                    if constraints is None: 
                        return False
                    checks = [
                        ("ISE_sim",       "ISE"),
                        ("Overshoot_sim", "Overshoot"),
                        ("SettlingTime_sim","SettlingTime"),
                        ("RiseTime_sim",  "RiseTime"),
                        ("SSE_sim",       "SSE"),
                    ]
                    for sim_key, c_key in checks:
                        if c_key in constraints and constraints[c_key] is not None:
                            lim = constraints[c_key]
                            val = sim_metrics.get(sim_key, np.nan)
                            if np.isnan(val):
                                return True
                            if val > lim:
                                return True
                    return False

                def true_cost(sim_metrics, weights):
                    # Weighted sum on simulated metrics (lower is better)
                    parts = []
                    for sim_key, w_key in [("ISE_sim","ISE"),
                                        ("Overshoot_sim","Overshoot"),
                                        ("SettlingTime_sim","SettlingTime"),
                                        ("RiseTime_sim","RiseTime")]:
                        w = float(weights.get(w_key, 0.0))
                        v = sim_metrics.get(sim_key, np.nan)
                        if np.isnan(v):
                            return np.inf
                        parts.append(w*v)
                    return float(np.sum(parts))

                # --- Phase 1: simulate until we have up to 50 feasible, stable items ---
                # --- helpers (put near other utils) ---
                def check_constraints(sim, constraints):
                    """
                    Return (violates:boolean, reasons:list[str]).
                    Expects 'sim' to contain the simulated metrics with keys that match your code.
                    """
                    reasons = []

                    # Pull with defaults to avoid KeyError
                    ISE_max  = constraints.get("ISE",           None)
                    OS_max   = constraints.get("Overshoot",     None)
                    ST_max   = constraints.get("SettlingTime",  None)
                    RT_max   = constraints.get("RiseTime",      None)
                    SSE_max  = constraints.get("SSE",           None)

                    # Access sim metrics (adapt names if yours differ)
                    ISE = sim.get("ISE_sim", None)
                    OS  = sim.get("Overshoot_sim", None)
                    ST  = sim.get("SettlingTime_sim", None)
                    RT  = sim.get("RiseTime_sim", None)
                    SSE = sim.get("SSE_sim", None)

                    # Basic domain sanity (optional)
                    if ISE is None or ISE <= 0: reasons.append("ISE<=0/None")
                    if ST  is None or ST  <= 0: reasons.append("ST<=0/None")
                    if RT  is None or RT  <= 0: reasons.append("RT<=0/None")
                    if OS  is None or OS  <  0: reasons.append("OS<0/None")
                    # If you model OS in %, clamp here accordingly.

                    # Upper-bound constraints
                    if ISE_max is not None and ISE is not None and ISE > ISE_max: reasons.append("ISE>max")
                    if OS_max  is not None and OS  is not None and OS  > OS_max:  reasons.append("OS>max")
                    if ST_max  is not None and ST  is not None and ST  > ST_max:  reasons.append("ST>max")
                    if RT_max  is not None and RT  is not None and RT  > RT_max:  reasons.append("RT>max")
                    if SSE_max is not None and SSE is not None and SSE > SSE_max: reasons.append("SSE>max")

                    return (len(reasons) > 0), reasons
                simulated_rows = []
                MAX_SIM = 100
                # Debug counters
                dbg = {
                    "raw_candidates": int(len(raw_df)),
                    "sim_failed": 0,          # simulate_true_metrics returned None / unstable / exception
                    "constraint_ISE": 0,
                    "constraint_OS": 0,
                    "constraint_ST": 0,
                    "constraint_RT": 0,
                    "constraint_SSE": 0,
                    "constraint_other": 0,    # domain sanity (<=0/None) etc.
                    "nonfinite_cost": 0,
                    "accepted": 0,
                    "cap_reached": 0,         # stopped due to MAX_SIM
                    "unique_filtered": 0,     # Phase 3 uniqueness culling
}           
                rejected_samples = []
                MAX_REJECT_SAMPLES = 30

                for idx, row in raw_df.iterrows():
                    # pull Kp,Ki,Kd from evaluated list (predicted candidates)
                    try:
                        Kp_i = float(row["Kp"])
                        Ki_i = float(row["Ki"])
                        Kd_i = float(row["Kd"])

                    except Exception:
                        dbg["sim_failed"] += 1
                        if len(rejected_samples) < MAX_REJECT_SAMPLES:
                            rejected_samples.append({
                                "idx": idx,
                                "Kp": row.get("Kp", np.nan), "Ki": row.get("Ki", np.nan), "Kd": row.get("Kd", np.nan),
                                "ISE_sim": np.nan, "SSE_sim": np.nan, "Overshoot_sim": np.nan,
                                "reason": "parse_params_failed"
                            })
                        continue


                    sim = None
                    try:
                        sim = simulate_true_metrics(Kp_i, Ki_i, Kd_i)
                    except Exception:
                        sim = None

                    if sim is None:
                        dbg["sim_failed"] += 1
                        if len(rejected_samples) < MAX_REJECT_SAMPLES:
                            rejected_samples.append({"idx": idx, "Kp": Kp_i, "Ki": Ki_i, "Kd": Kd_i, "reason": "sim_failed/unstable"})
                        continue  # unstable or failed sim

                    violates, reasons = check_constraints(sim, constraints)
                    if violates:
                        # Tally fine-grained reasons
                        counted_specific = False
                        for r in reasons:
                            if r == "ISE>max": dbg["constraint_ISE"] += 1; counted_specific = True
                            elif r == "OS>max": dbg["constraint_OS"] += 1; counted_specific = True
                            elif r == "ST>max": dbg["constraint_ST"] += 1; counted_specific = True
                            elif r == "RT>max": dbg["constraint_RT"] += 1; counted_specific = True
                            elif r == "SSE>max": dbg["constraint_SSE"] += 1; counted_specific = True

                        # Domain sanity & nulls bucket
                        if not counted_specific:
                            dbg["constraint_other"] += 1

                        if len(rejected_samples) < MAX_REJECT_SAMPLES:
                            rejected_samples.append({
                                "idx": idx, "Kp": Kp_i, "Ki": Ki_i, "Kd": Kd_i,
                                "reason": ",".join(reasons)
                            })
                        continue

                    # attach cost_true
                    cost_true = true_cost(sim, weights)
                    if not np.isfinite(cost_true):
                        dbg["nonfinite_cost"] += 1
                        if len(rejected_samples) < MAX_REJECT_SAMPLES:
                            rejected_samples.append({
                                "idx": idx, "Kp": Kp_i, "Ki": Ki_i, "Kd": Kd_i,
                                         "ISE_sim": sim.get("ISE_sim", np.nan),
            "SSE_sim": sim.get("SSE_sim", np.nan),
            "Overshoot_sim": sim.get("Overshoot_sim", np.nan),
                                "reason": "nonfinite_cost"
                            })
                        continue

                    out = {
                        "Kp": Kp_i, "Ki": Ki_i, "Kd": Kd_i,
                        "ISE_sim": sim["ISE_sim"],
                        "SSE_sim": sim["SSE_sim"],
                        "Overshoot_sim": sim["Overshoot_sim"],
                        "RiseTime_sim": sim["RiseTime_sim"],
                        "SettlingTime_sim": sim["SettlingTime_sim"],
                        "Cost_true": cost_true,
                        # Keep predicted metrics too if present (useful for later comparison)
                        "ISE": row.get("ISE", np.nan),
                        "Overshoot": row.get("Overshoot", np.nan),
                        "SettlingTime": row.get("SettlingTime", np.nan),
                        "RiseTime": row.get("RiseTime", np.nan),
                        "SSE": row.get("SSE", np.nan),
                        "Cost": row.get("Cost", np.nan),
                    }
                    simulated_rows.append(out)
                    dbg["accepted"] += 1

                    if len(simulated_rows) >= MAX_SIM:
                        dbg["cap_reached"] += 1
                        break

                if len(simulated_rows) == 0:
                    st.error("❌ No feasible/stable controllers after simulation and constraints.")
                    # Optional: print debug summary before stop
                    with st.expander("Debug summary"):
                        st.write(dbg)
                        if rejected_samples:
                            st.write(pd.DataFrame(rejected_samples))
                    st.stop()

                sim_df = pd.DataFrame(simulated_rows)

                # --- Phase 2: sort by simulated true cost (best first) ---
                sim_df = sim_df.sort_values("Cost_true", ascending=True).reset_index(drop=True)

                # --- Phase 3: pick 5 UNIQUE controllers AFTER sorting ---
                MIN_DELTA = 0.5  # uniqueness radius in parameter space
                picked = []
                for _, r in sim_df.iterrows():
                    p = np.array([r["Kp"], r["Ki"], r["Kd"]], dtype=float)
                    if any(np.all(np.abs(p - np.array([q["Kp"], q["Ki"], q["Kd"]])) < MIN_DELTA) for q in picked):
                        continue
                    picked.append(r.to_dict())
                    if len(picked) == 5:
                        break

                # Fallback: if uniqueness collapses to <5, we still show what we have (and pad later)
                top5_df = pd.DataFrame(picked)

                # === Padding if fewer than 5
                if len(top5_df) < 5:
                    pad_rows = 5 - len(top5_df)
                    padding = pd.DataFrame([{
                        'Kp': np.nan, 'Ki': np.nan, 'Kd': np.nan,
                        'ISE': np.nan, 'Overshoot': np.nan, 'SettlingTime': np.nan,
                        'RiseTime': np.nan, 'SSE': np.nan, 'Cost': np.nan,
                        'ISE_sim': np.nan, 'Overshoot_sim': np.nan, 'SettlingTime_sim': np.nan,
                        'RiseTime_sim': np.nan, 'SSE_sim': np.nan,
                        'Cost_true': np.nan
                    }] * pad_rows)
                    top5_df = pd.concat([top5_df, padding], ignore_index=True)

                # === Prepare display columns (retain your *_val and ±std formatting logic) ===
                # Raw numeric copies for plotting later:
                top5_df["Kp_val"] = pd.to_numeric(top5_df["Kp"], errors="coerce")
                top5_df["Ki_val"] = pd.to_numeric(top5_df["Ki"], errors="coerce")
                top5_df["Kd_val"] = pd.to_numeric(top5_df["Kd"], errors="coerce")

                # If you still want to show predicted mean ± std for comparison, keep your fmt() logic.
                # Otherwise, just round simulated columns and show:
                for c in ["ISE_sim","Overshoot_sim","SettlingTime_sim","RiseTime_sim","SSE_sim","Cost_true"]:
                    if c in top5_df.columns:
                        top5_df[c] = pd.to_numeric(top5_df[c], errors="coerce").round(3)
                with st.expander("🔎 Debug summary (why fewer than 5?)", expanded=False):
                    st.markdown(f"""
                - Raw candidates: **{dbg['raw_candidates']}**
                - Accepted after sim & constraints: **{dbg['accepted']}** (capped: {dbg['cap_reached']})
                - Rejected due to simulation/unstable: **{dbg['sim_failed']}**
                - Rejected by constraints:
                - ISE>max: **{dbg['constraint_ISE']}**
                - OS>max: **{dbg['constraint_OS']}**
                - ST>max: **{dbg['constraint_ST']}**
                - RT>max: **{dbg['constraint_RT']}**
                - SSE>max: **{dbg['constraint_SSE']}**
                - Other (null/≤0): **{dbg['constraint_other']}**
                - Rejected due to non-finite cost: **{dbg['nonfinite_cost']}**
                - Removed by uniqueness radius (MIN_DELTA={MIN_DELTA}): **{dbg['unique_filtered']}**
                - Picked for Top-5: **{len(top5_df)}**
                """)
                    if rejected_samples:
                        st.caption("Sample of rejected candidates and reasons:")
                        st.dataframe(pd.DataFrame(rejected_samples))


                st.markdown("#### 🏆 Top 5 Distinct PID Controllers (sorted by **simulated** cost)")
                display_cols = [
                    "Kp", "Ki", "Kd",
                    "ISE_sim", "Overshoot_sim", "SettlingTime_sim", "RiseTime_sim", "SSE_sim",
                    "Cost_true"
                ]
                st.dataframe(top5_df[display_cols])

                # Keep using top5_df downstream (plots etc.) with `Kp_val/Ki_val/Kd_val` and *_sim series.




            # === Step Response ===

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            fig2, ax2 = plt.subplots(figsize=(8, 4))

            t = np.linspace(0, max(2 * (T1 + T2 + Td), 100), 1000)
            step_input = np.ones_like(t)

            for idx, row in top5_df.iterrows():
                if pd.isna(row["Kp"]):
                    continue  # skip padded rows

                #Kp_i, Ki_i, Kd_i = row["Kp"], row["Ki"], row["Kd"]
                Kp_i, Ki_i, Kd_i = row["Kp_val"], row["Ki_val"], row["Kd_val"]


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

                #sys_cl = control.feedback(C * G, 1)
                sys_cl = control.feedback(C * G, 1)

                # Check stability
                sys_cl = control.feedback(C * G, 1)

                if not is_stable(sys_cl):
                    print(f"⚠️ Skipping unstable controller.")
                    continue


                try:
                    t_response, y_response = control.step_response(sys_cl, t)
                    e_response = step_input - y_response

                    ax1.plot(t_response, y_response, label=f"#{idx+1}: Kp={Kp_i:.2f}, Ki={Ki_i:.2f}, Kd={Kd_i:.2f}")
                    ax2.plot(t_response, e_response, label=f"#{idx+1}")
                except Exception as e:
                    print(f"⚠️ Skipped controller #{idx+1} due to simulation error: {e}")


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
            # === Step Response Plot (Top 5 Controllers)
            st.markdown("#### 🧪 Step Responses of Top 5 Controllers")
            fig_step5 = go.Figure()
            fig_error5 = go.Figure()

            # === Show raw controller table
            #st.markdown("### 🛠️ Debug: Raw Top 5 Controllers (before plotting)")
            #st.dataframe(top5_df[[
            #    "Kp_val", "Ki_val", "Kd_val",
            #    "Kp", "Ki", "Kd",
            #    "ISE", "ISE_sim",
            #    "Overshoot", "Overshoot_sim",
            #    "SettlingTime", "SettlingTime_sim",
            #    "RiseTime", "RiseTime_sim",
            #    "SSE", "Cost"
            #]])

            # === Filter valid controllers
            valid_rows = top5_df[
                top5_df[["Kp_val", "Ki_val", "Kd_val"]].applymap(
                    lambda x: pd.notna(x) and np.isfinite(x)
                ).all(axis=1)
            ].reset_index(drop=True)

            # === Handle empty case
            if len(valid_rows) == 0:
                st.warning("⚠️ No valid controllers available for plotting.")
            else:
                for idx, row in valid_rows.iterrows():
                    Kp_i = row["Kp_val"]
                    Ki_i = row["Ki_val"]
                    Kd_i = row["Kd_val"]

                    #st.write(f"🔍 Plotting Controller #{idx+1}: Kp={Kp_i:.2f}, Ki={Ki_i:.2f}, Kd={Kd_i:.2f}")

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
                        st.warning(f"⚠️ Skipped controller #{idx+1} due to instability or simulation error: {e}")

            # Add setpoint line
            fig_step5.add_trace(go.Scatter(
                x=t, y=step_input,
                mode='lines',
                name="Setpoint r(t)=1",
                line=dict(color='black', dash='dash'),
                opacity=0.6
            ))

            # === Layouts
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

            # === Plot
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

# === Groq API setup ===
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
model_name = "llama3-8b-8192"

# === Session State ===
if "floating_chat_history" not in st.session_state:
    st.session_state.floating_chat_history = [
        {"role": "system", "content": "You are a helpful assistant for PID tuning and ML optimization."}
    ]

# === Floating Chat Button and Box HTML/CSS ===
floating_chat_html = """
<style>
#floatingChatBtn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #5e81ac;
    color: white;
    border: none;
    padding: 12px 16px;
    border-radius: 20px;
    font-size: 16px;
    z-index: 10000;
    cursor: pointer;
}
#floatingChatBox {
    display: none;
    position: fixed;
    bottom: 70px;
    right: 20px;
    width: 340px;
    max-height: 480px;
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
    padding: 15px;
    overflow-y: auto;
    z-index: 9999;
}
</style>

<button id="floatingChatBtn" onclick="toggleChat()">💬 Ask AI</button>
<div id="floatingChatBox">
    <p><strong>🤖 AI Assistant</strong></p>
    <div id="chatHistory">You can ask about PID tuning, optimization, etc.</div>
</div>

<script>
function toggleChat() {
    var chatBox = document.getElementById("floatingChatBox");
    if (chatBox.style.display === "none" || chatBox.style.display === "") {
        chatBox.style.display = "block";
    } else {
        chatBox.style.display = "none";
    }
}
</script>
"""

# === Chat Session State ===
if "floating_chat_history" not in st.session_state:
    st.session_state.floating_chat_history = [
        {"role": "system", "content": "You are a professional assistant embedded in a Bachelor thesis tool for optimizing PID controller parameters using machine learning."
    "Your answers should be concise, technically sound, and suitable for an academic or engineering audience (e.g., professors, recruiters). "
    "Explain concepts from control theory and machine learning clearly and accurately. Focus on key topics like PID tuning, surrogate models, performance metrics (ISE, overshoot, settling time), and the benefits of data-driven methods over classical techniques like Ziegler-Nichols or CHR. "
    "Respond in a structured, professional tone. Use Markdown for equations or formatting when helpful."}
    ]

# === Initialize Chat History (Session State) ===
if "floating_chat_history" not in st.session_state:
    st.session_state.floating_chat_history = [
        {"role": "system", "content": (
            "You are a professional AI assistant specialized in control theory and machine learning. "
            "Answer briefly and clearly, suitable for academic and engineering use. "
            "The user is working on a bachelor's thesis about 'Machine Learning-Based Optimization of PID Controller Parameters'. "
            "Explain concepts like ISE, surrogate modeling, and PID tuning in a concise, academic tone."
        )}
    ]

# === Floating Chat UI Block ===
# === Ask the AI Assistant UI ===
with st.container(border=True):
    st.markdown("### 🤖 Ask the AI Assistant")
    st.caption("Ask about PID control, optimization, surrogate models, etc.")
    
    # Show previous messages
    if len(st.session_state.floating_chat_history) > 1:
        for msg in st.session_state.floating_chat_history[1:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # === Input form
    with st.form("chat_input_form"):
        user_input = st.text_input("Your question:", key="chat_input")
        submitted = st.form_submit_button("Send")
    
    # === Process immediately when form is submitted ===
    if submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.floating_chat_history.append({"role": "user", "content": question})
        
        with st.spinner("💬 Thinking..."):
            client = OpenAI(
                api_key=st.secrets["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=st.session_state.floating_chat_history,
                temperature=0.5
            )
            reply = response.choices[0].message.content.strip()
            st.session_state.floating_chat_history.append({"role": "assistant", "content": reply})
        
        # Force rerun to show the new messages
        st.rerun()
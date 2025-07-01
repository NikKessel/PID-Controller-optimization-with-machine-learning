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
    
    model_choice = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "MLP", "XGBoost"], key="model_select")
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
            model_filename = f"model_{model_choice.lower().replace(' ', '_')}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            model = joblib.load(model_path)
            # Prepare input vector
            if model_choice == "Random Forest":
                X = np.array([[K, T1, T2, Td]])
            elif model_choice == "MLP":
                X = np.array([[K, T1, T2, Td]])
            elif model_choice == "XGBoost":
                X = np.array([[K, T1, T2]])


            from utils.predict_pid import predict_pid_params
            Kp, Ki, Kd = predict_pid_params(model, X)
            #Kp_ml, Ki_ml, Kd_ml = predict_pid_params(model, X)

            st.success("Prediction complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Kp", f"{Kp:.3f}")
            col2.metric("Ki", f"{Ki:.5f}")
            col3.metric("Kd", f"{Kd:.2f}")

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
            st.markdown("### 🔧 PID Parameter Debug")
            st.markdown("### 📐 Calculation Breakdown")

            T_eff = T1 + T2 if T2 > 0 else T1

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
            st.markdown("### Step Response")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_ml, y_ml, label="ML Predicted PID", linewidth=2)
            ax.plot(t_zn, y_zn, '--', label="Ziegler–Nichols")
            ax.plot(t_chr0, y_chr0, ":", label="CHR (0% OS)")
            ax.plot(t_chr20, y_chr20, "-.", label="CHR (20% OS)")
            #ax.plot(t_ml, np.ones_like(t_ml)*K, "k--", label=f"Step Input ({1:.2f})")
            step_input = np.ones_like(t_ml)
            step_input[t_ml < 0.01] = 0  # Optional: simulate visible step
            ax.plot(t_ml, step_input, "k--", label="Step Input (0 → 1)")

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Output")
            ax.set_title("Closed-Loop Step Response")
            ax.set_ylim(0, y_max)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)




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

    # === User Inputs ===
    K = st.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.0)
    T1 = st.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.number_input("T2 (2nd Time Constant in s)", min_value=0.0, max_value=50.0, value=10.0)
    Td = st.number_input("Td (Dead Time in s)", min_value=0.0, max_value=5.0, value=1.0)

    Kp = st.number_input("Kp", min_value=0.0, max_value=10.0, value=2.0)
    Ki = st.number_input("Ki", min_value=0.0, max_value=10.0, value=0.1)
    Kd = st.number_input("Kd", min_value=0.0, max_value=10.0, value=1.0)

    # === Load Surrogate Model ===
    model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
    model_path = os.path.join(model_dir, "model_surrogate.joblib")

    try:
        surrogate_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load surrogate model: {e}")
        surrogate_model = None

    if st.button("📊 Evaluate Performance", key="eval_button") and surrogate_model:
        try:



            # === Prepare Input ===
            X_eval = pd.DataFrame({
                'K': [K],
                'T1': [T1],
                'T2': [T2],
                'Td': [Td],
                'Kp': [Kp],
                'Ki': [Ki],
                'Kd': [Kd],
            })

            # === Surrogate Prediction ===
            prediction = surrogate_model.predict(X_eval)
            ise_pred, sse_pred, rise_time_pred, settling_time_pred, overshoot_pred = prediction[0]

            # === Simulate Closed-Loop System ===
            if T2 > 0:
                den = np.convolve([T1, 1], [T2, 1])
            else:
                den = [T1, 1]
            G = control.tf([K], den)

            if Td > 0:
                # Optional: Add delay approximation (can skip if unstable)
                try:
                    G = control.pade(Td, 1)[0] * G
                except:
                    st.warning("Pade approximation failed; skipping dead time.")
            #C = control.tf([Kd, Kp, Ki], [1, 0])
            P = control.tf([Kp], [1])
            I = control.tf([Ki], [1, 0])
            D = control.tf([Kd, 0], [1])
            C = P + I + D

            sys_cl = control.feedback(C * G, 1)

            t = np.linspace(0, 1000, 20000)
            t, y = control.step_response(sys_cl, T=t)

            # === Compute Actual Metrics ===
            # === Compute Actual Metrics ===
            u = np.ones_like(t)
            e = u - y
            ise_true = simpson(e**2, t)
            sse_true = abs(1 - y[-1])
            overshoot_true = (np.max(y) - 1) * 100

            # === Robust Rise Time: time from 10% to 90% of final value ===
            try:
                final_val = y[-1]
                rise_start = np.where(y >= 0.1 * final_val)[0][0]
                rise_end = np.where(y >= 0.9 * final_val)[0][0]
                rise_time_true = t[rise_end] - t[rise_start]
            except Exception:
                rise_time_true = np.nan

            # === Robust Settling Time: time after which output stays within ±2% ===
            try:
                tolerance = 0.02 * final_val
                within_bounds = np.abs(y - final_val) <= tolerance

                # Find first index from which all remaining values are within bounds
                settling_time_true = t[-1]  # fallback if never settles
                for i in range(len(y)):
                    if np.all(within_bounds[i:]):
                        settling_time_true = t[i]
                        break
            except Exception:
                settling_time_true = np.nan

            # === Display Comparison Table ===
            st.markdown("### 📊 Performance: Surrogate vs Simulation")
            df_compare = pd.DataFrame({
                "Metric": ["ISE", "SSE", "Overshoot [%]", "Settling Time [s]", "Rise Time [s]"],
                "Predicted": [f"{ise_pred:.4f}", f"{sse_pred:.5f}", f"{overshoot_pred:.2f}",
                              f"{settling_time_pred:.2f}", f"{rise_time_pred:.2f}"],
                "Simulated": [f"{ise_true:.4f}", f"{sse_true:.5f}", f"{overshoot_true:.2f}",
                              f"{settling_time_true:.2f}", f"{rise_time_true:.2f}"]
            })
            st.dataframe(df_compare)

            # === Plot Step Response ===
            st.markdown("#### 🧪 Closed-Loop Step Response")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, y, label="Simulated Response", linewidth=2)
            ax.plot(t, np.ones_like(t), "k--", label="Step Input", alpha=0.6)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Output")
            ax.set_title("Step Response of G(s) + PID")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # === Plot Error Signal ===
            st.markdown("#### 📉 Error Curve $e(t)$")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.plot(t, e, label="Tracking Error", color='red')
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("e(t)")
            ax2.grid(True)
            ax2.set_title("Error Signal Over Time")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")


elif mode == "⚙️ Optimize PID":
    st.info("Use ML-guided optimization to find best PID")

    model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
    model_path = os.path.join(model_dir, "model_surrogate.joblib")
    surrogate_model = joblib.load(model_path)

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

        st.markdown("### 🔍 Debug Info")
        st.write("**Plant:**", {"K": K, "T1": T1, "T2": T2, "Td": Td})
        st.write("**Weights:**", weights)
        st.write("**Constraints:**", constraints)

        from utils.optimize_pid import optimize_pid_for_system

        try:
            Kp, Ki, Kd, ise, os, stime, rtime, sse = optimize_pid_for_system(
                K, T1, T2, Td, surrogate_model, weights, constraints
            )

            st.success("✅ Optimization complete!")
            st.markdown("#### Optimal PID Parameters")
            st.write(f"Kp = {Kp:.4f}, Ki = {Ki:.4f}, Kd = {Kd:.4f}")

            # === Step Response ===
            import matplotlib.pyplot as plt
            import numpy as np
            import control

            if T2 > 0:
                G = control.tf([K], np.convolve([T1, 1], [T2, 1]))
            else:
                G = control.tf([K], [T1, 1])

            P = control.tf([Kp], [1])
            I = control.tf([Ki], [1, 0])
            D = control.tf([Kd, 0], [1])
            C = P + I + D

            if Td > 0:
                #G = control.series(control.pade(Td, 1)[0], G)
                num, den = control.pade(Td, 1)
                G_delay = control.tf(num, den)
                G = control.series(G_delay, G)


            sys_cl = control.feedback(C * G, 1)

            t = np.linspace(0, max(2 * (T1 + T2 + Td), 100), 1000)
            t, y = control.step_response(sys_cl, t)
            u = np.ones_like(t)
            e = u - y

            fig1, ax1 = plt.subplots()
            ax1.plot(t, y, label="Output y(t)")
            ax1.plot(t, u, "--", label="Setpoint r(t)=1", color="black")
            ax1.set_title("Step Response")
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Output")
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(t, e, color="crimson", label="Error e(t)")
            ax2.set_title("Tracking Error")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Error")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

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
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import pandas as pd
from control.matlab import tf, feedback, step
from utils.predict_pid import predict_pid_params

# Set page config
st.set_page_config(
    page_title="PID Optimizer with Machine Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and subtitle
st.title("🎛️ PID Controller Optimization Tool")
st.markdown("_A modern ML-based GUI to Predict, Evaluate, and Optimize PID controllers for your process models._")

# --- Sidebar: Mode selection ---
model_choice = None
mode = st.sidebar.radio("🧠 Select Mode", ["Predict PID", "Evaluate PID", "Optimize PID"])



# --- Conditional ML model selection ---
if mode == "Predict PID":
    model_choice = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "MLP", "XGBoost"], key="model_select")
    if "predict_clicked" not in st.session_state: ####
        st.session_state.predict_clicked = False

    st.sidebar.markdown("**System Parameters**")
    system_type = st.sidebar.selectbox("System Type", ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"])
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.0)
    T1 = st.sidebar.number_input("T1", min_value=0.1, max_value=50.0, value=20.0)
    T2 = st.sidebar.number_input("T2", min_value=0.0, max_value=50.0, value=10.0) if "PT2" in system_type else 0.0
    Td = st.sidebar.number_input("Td", min_value=0.0, max_value=5.0, value=1.0) if "Td" in system_type else 0.0

    # Optional advanced params
    Tu = Tg = overshoot = 0.0
    if model_choice in ["MLP", "XGBoost"]:
        Tu = st.sidebar.number_input("Tu", min_value=0.0, max_value=100.0, value=3.0)
        Tg = st.sidebar.number_input("Tg", min_value=0.0, max_value=100.0, value=10.0)
        overshoot = st.sidebar.number_input("Overshoot", min_value=0.0, max_value=2.0, value=0.1)

    st.sidebar.markdown("**Plot Settings**")
    t_max = st.sidebar.slider("Simulation Time [s]", 20, 300, 100, key="slider_t_max")
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
                X = np.array([[K, T1, T2, Td, Tu, Tg, overshoot]])
            elif model_choice == "XGBoost":
                type_PT1 = 1 if system_type == "PT1" else 0
                type_PT2 = 1 if system_type == "PT2" else 0
                type_PT1_Td = 1 if system_type == "PT1+Td" else 0
                type_PT2_Td = 1 if system_type == "PT2+Td" else 0
                type_Osc2 = 1 if system_type == "Osc2" else 0
                X = np.array([[K, T1, T2, Td, Tu, Tg, overshoot, type_Osc2, type_PT1, type_PT1_Td, type_PT2, type_PT2_Td]])

            from utils.predict_pid import predict_pid_params
            Kp, Ki, Kd = predict_pid_params(model, X)
            #Kp_ml, Ki_ml, Kd_ml = predict_pid_params(model, X)

            st.success("Prediction complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Kp", f"{Kp:.3f}")
            col2.metric("Ki", f"{Ki:.5f}")
            col3.metric("Kd", f"{Kd:.2f}")

            # --- Real simulation ---

            def simulate_response(K, T1, T2, Td, Kp, Ki, Kd, T_final=100):
                t = np.linspace(0, T_final, 1000)
                num = [K]
                den = np.polymul([T1, 1], [T2, 1]) if T2 > 0 else [T1, 1]
                G = tf(num, den)
                num = [1]                  # Unit-gain plant
                G = tf(num, den)
                G = K * G                  # Apply user gain as scalar multiplier

                if Td > 0:
                    G = G * tf([1], [1, Td])
                s = tf([1, 0], [1])
                C = Kp + Ki/s + Kd*s
                sys = feedback(C * G, 1)
                #t = np.linspace(0, T_final, 1000)
                t = np.linspace(0, t_max, 1000)
                T_final = t_max
                #y = y * K 
                t, y = step(sys, T=t)
                 # Scale the output to match input step of magnitude K
                return t, y, G, C

            t, y, G, C = simulate_response(K, T1, T2, Td, Kp, Ki, Kd)

            # Updated implementation of ZN and CHR based on approximated Tu and Tg from T1 and T2

            def estimate_tu_tg(T1, T2):
                Tu = 0.5 * T1
                Tg = T1 + T2
                return Tu, Tg

            def ziegler_nichols_pid(Tu, Tg):
                Kp = 1.2 * Tg / Tu
                Ti = 2 * Tu
                Td = 0.5 * Tu
                Ki = Kp / Ti
                Kd = Kp * Td
                return Kp, Ki, Kd

            def chr_pid(Tu, Tg, overshoot=0):
                if overshoot == 0:
                    # CHR for 0% overshoot
                    Kp = 0.6 * Tg / Tu
                    Ti = Tu
                    Td = 0.5 * Tu
                else:
                    # CHR for ~20% overshoot
                    Kp = 0.95 * Tg / Tu
                    Ti = 1.35 * Tu
                    Td = 0.47 * Tu
                Ki = Kp / Ti
                Kd = Kp * Td
                return Kp, Ki, Kd

            Tu, Tg = estimate_tu_tg(T1, T2)
            Kp_zn, Ki_zn, Kd_zn = ziegler_nichols_pid(Tu, Tg)
            Kp_chr, Ki_chr, Kd_chr = chr_pid(Tu, Tg, overshoot=0)

            # Simulate
            Kp_ml = Kp
            Ki_ml = Ki
            Kd_ml = Kd
            t_ml, y, _, _ = simulate_response(K, T1, T2, Td, Kp_ml, Ki_ml, Kd_ml)
            t_zn, y_zn, _, _ = simulate_response(K, T1, T2, Td, Kp_zn, Ki_zn, Kd_zn)
            t_chr, y_chr, _, _ = simulate_response(K, T1, T2, Td, Kp_chr, Ki_chr, Kd_chr)
            t = t_ml*K
            t_zn = t_zn*K
            t_chr = t_chr*K
            st.markdown("### Step Response")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(y,t,  label="ML Predicted PID")
            ax.plot(y_zn,t_zn,  '--', label="Ziegler–Nichols (approx)")
            ax.plot(y_chr,t_chr,  ":", label="CHR (0%)")
            step_input = np.ones_like(t) * K
            step_input[t < 0.01] = 0  # makes it visibly a 'step'
            ax.plot(t, step_input, "k--", label=f"Step Input (0 → {K:.2f})")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Output")
            ax.set_title("Closed-Loop Step Response")
            ax.set_ylim(0, 1.5 * K)
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




elif mode == "Evaluate PID":
    st.info("Evaluate performance of a given PID configuration")

    system_type = st.selectbox("System Type", ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"])
    K = st.number_input("K (Gain)", min_value=0.1, max_value=5.0, value=1.0)
    T1 = st.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.number_input("T2 (2nd Time Constant in s)", min_value=0.0, max_value=50.0, value=10.0) if "PT2" in system_type else 0.0
    Td = st.number_input("Td (Dead Time in s)", min_value=0.0, max_value=5.0, value=1.0) if "Td" in system_type else 0.0

    Kp = st.number_input("Kp", min_value=0.0, max_value=10.0, value=2.0)
    Ki = st.number_input("Ki", min_value=0.0, max_value=1.0, value=0.1)
    Kd = st.number_input("Kd", min_value=0.0, max_value=10.0, value=1.0)

    model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
    model_path = os.path.join(model_dir, "model_surrogate.joblib")

    try:
        surrogate_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load surrogate model: {e}")
        surrogate_model = None

    if st.button("📊 Evaluate Performance", key="eval_button") and surrogate_model:
        try:
            # Add input fields for Tu and Tg in evaluation mode
            Tu = st.number_input("Tu (Ultimate Period in s)", min_value=0.0, max_value=100.0, value=10.0, key="eval_tu")
            Tg = st.number_input("Tg (Gradient Time in s)", min_value=0.0, max_value=100.0, value=20.0, key="eval_tg")
            
            # Create DataFrame with proper column names
            import pandas as pd
            
            # One-hot encode system type
            type_PT1 = 1 if system_type == "PT1" else 0
            type_PT2 = 1 if system_type == "PT2" else 0
            type_PT1_Td = 1 if system_type == "PT1+Td" else 0
            type_PT2_Td = 1 if system_type == "PT2+Td" else 0
            type_Osc2 = 1 if system_type == "Osc2" else 0
            

            X_eval = pd.DataFrame({
                'K': [K],
                'T1': [T1], 
                'T2': [T2],
                'Td': [Td],
                'Tu': [Tu],
                'Tg': [Tg],
                'Kp': [Kp],
                'Ki': [Ki],
                'Kd': [Kd],
                'type': [system_type]  # <--- This line adds the required 'type' column
            })

            prediction = surrogate_model.predict(X_eval)
            ise, os, stime, rtime = prediction[0]
            st.success("Evaluation complete!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ISE", f"{ise:.2f}")
            col2.metric("Overshoot", f"{os * 100:.1f}%")
            col3.metric("Settling Time", f"{stime:.1f} s")
            col4.metric("Rise Time", f"{rtime:.1f} s")

            st.markdown("#### Simulated Step Response")
            t = np.linspace(0, 100, 500)
            y = 1 - np.exp(-t / 15) * np.cos(t / 10)
            fig, ax = plt.subplots(figsize=(6, 4))  # Width=6, Height=4 inches
            ax.plot(t, y, label="User PID Response")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Output")
            ax.set_title("Simulated Step Response")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

elif mode == "Optimize PID":
    st.info("Use ML-guided optimization to find best PID")
    model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
    st.markdown("#### Define Optimization Weights")
    w_ise = st.slider("ISE Weight", 0.0, 1.0, 0.5)
    w_os = st.slider("Overshoot Weight", 0.0, 1.0, 0.2)
    w_st = st.slider("Settling Time Weight", 0.0, 1.0, 0.2)
    w_rt = st.slider("Rise Time Weight", 0.0, 1.0, 0.1)


    system_type = st.selectbox("System Type", ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"])
    K = st.number_input("K (Gain)", min_value=0.1, max_value=5.0, value=1.0)
    T1 = st.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.number_input("T2 (2nd Time Constant)", min_value=0.0, max_value=50.0, value=10.0) if "PT2" in system_type else 0.0
    Td = st.number_input("Td (Dead Time)", min_value=0.0, max_value=5.0, value=1.0) if "Td" in system_type else 0.0
    Tu = st.number_input("Tu (Ultimate Period)", min_value=0.0, max_value=100.0, value=10.0)
    Tg = st.number_input("Tg (Gradient Time)", min_value=0.0, max_value=100.0, value=20.0)

    model_path = os.path.join(model_dir, "model_surrogate.joblib")
    surrogate_model = joblib.load(model_path)

    if st.button("⚙️ Run Optimization", key="optimize_button"):
            weights = (w_ise, w_os, w_st, w_rt)
            from utils.optimize_pid import optimize_pid_for_system
            try:
                Kp, Ki, Kd, ise, os, stime, rtime = optimize_pid_for_system(
                    K, T1, T2, Td, Tu, Tg, system_type, surrogate_model, weights
                )
                st.success("Optimization complete!")

                st.markdown("#### Optimal PID Parameters")
                col1, col2, col3 = st.columns(3)
                col1.metric("Kp", f"{Kp:.3f}")
                col2.metric("Ki", f"{Ki:.5f}")
                col3.metric("Kd", f"{Kd:.2f}")

                st.markdown("#### Predicted Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ISE", f"{ise:.2f}")
                col2.metric("Overshoot", f"{os * 100:.1f}%")
                col3.metric("Settling Time", f"{stime:.1f} s")
                col4.metric("Rise Time", f"{rtime:.1f} s")

            except Exception as e:
                st.error(f"Optimization failed: {e}")

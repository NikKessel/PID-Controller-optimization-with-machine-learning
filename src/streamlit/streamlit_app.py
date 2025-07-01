import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import pandas as pd
from control import tf, feedback, step_response, pade
from control.matlab import tf, feedback, step
from utils.predict_pid import predict_pid_params
from utils.simulink_runner import run_simulink_simulation


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
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.0)
    T1 = st.sidebar.number_input("T1", min_value=0.1, max_value=50.0, value=20.0)
    T2 = st.sidebar.number_input("T2", min_value=0.0, max_value=50.0, value=10.0)
    Td = st.sidebar.number_input("Td", min_value=0.0, max_value=5.0, value=1.0) 

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
                Td = 0.5 * L
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
            Kp_chr, Ki_chr, Kd_chr = chr_pid(K, T1, T2, L, overshoot=0)

            # === Simulate Step Responses ===
            t_ml, y_ml = simulate_response(K, T1, T2, L, Kp_ml, Ki_ml, Kd_ml, T_final=t_max)
            t_zn, y_zn = simulate_response(K, T1, T2, L, Kp_zn, Ki_zn, Kd_zn, T_final=t_max)
            t_chr, y_chr = simulate_response(K, T1, T2, L, Kp_chr, Ki_chr, Kd_chr, T_final=t_max)

            
            # === Debug Print for PID parameters ===
            st.markdown("### 🔧 PID Parameter Debug")

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

            📊 CHR (0% OS):
                Kp = {Kp_chr:.4f}
                Ki = {Ki_chr:.4f}
                Kd = {Kd_chr:.4f}
            """, language="text")


            # === Plot Step Responses ===
            st.markdown("### Step Response")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t_ml, y_ml, label="ML Predicted PID", linewidth=2)
            ax.plot(t_zn, y_zn, '--', label="Ziegler–Nichols")
            ax.plot(t_chr, y_chr, ":", label="CHR (0% OS)")
            ax.plot(t_ml, np.ones_like(t_ml)*K, "k--", label=f"Step Input ({K:.2f})")
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




elif mode ==  "📊 Evaluate PID":
    st.info("Evaluate performance of a given PID configuration")

    K = st.number_input("K (Gain)", min_value=0.1, max_value=10.0, value=1.0)
    T1 = st.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.number_input("T2 (2nd Time Constant in s)", min_value=0.0, max_value=50.0, value=10.0) 
    Td = st.number_input("Td (Dead Time in s)", min_value=0.0, max_value=5.0, value=1.0) 

    Kp = st.number_input("Kp", min_value=0.0, max_value=10.0, value=2.0)
    Ki = st.number_input("Ki", min_value=0.0, max_value=10.0, value=0.1)
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

            
            # Create DataFrame with proper column names
            import pandas as pd

            

            X_eval = pd.DataFrame({
                'K': [K],
                'T1': [T1], 
                'T2': [T2],
                'Td': [Td],
                'Kp': [Kp],
                'Ki': [Ki],
                'Kd': [Kd],
            })

            prediction = surrogate_model.predict(X_eval)
            ise, sse, rise_time, settling_time, overshoot = prediction[0]            
            st.success("Evaluation complete!")

            # === Display Metrics ===
            st.markdown("### 📈 Predicted Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("ISE", f"{ise:.4f}")
            col2.metric("SSE", f"{sse:.5f}")
            col3.metric("Overshoot", f"{overshoot:.1f} %")

            col4, col5 = st.columns(2)
            col4.metric("Settling Time", f"{settling_time:.2f} s")
            col5.metric("Rise Time", f"{rise_time:.2f} s")

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

elif mode == "⚙️ Optimize PID":
    
    st.info("Use ML-guided optimization to find best PID")
    model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
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
    T1 = st.sidebar.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=10.0)
    T2 = st.sidebar.number_input("T2 (2nd Time Constant)", min_value=0.0, max_value=50.0, value=3.0) #
    Td = st.sidebar.number_input("Td (Dead Time)", min_value=0.0, max_value=5.0, value=0.6) 


    model_path = os.path.join(model_dir, "model_surrogate.joblib")
    surrogate_model = joblib.load(model_path)

    if st.button("⚙️ Run Optimization", key="optimize_button"):
            #weights = (w_ise, w_os, w_st, w_rt)
            weights = {
                        "ISE":  w_ise,
                        "Overshoot": w_os,
                        "SettlingTime": w_st,
                        "RiseTime": w_rt
                    }
            constraints = {
            "ISE": max_ise,
            "Overshoot": max_os ,  # convert from % to 0–1 range
            "SettlingTime": max_st,
            "RiseTime": max_rt,
            "SSE": max_sse,
}

            from utils.optimize_pid import optimize_pid_for_system
            try:
                Kp, Ki, Kd, ise, os, stime, rtime, sse = optimize_pid_for_system(
                    K, T1, T2, Td, surrogate_model, weights, constraints
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
                col2.metric("Overshoot", f"{os :.1f}%")
                col3.metric("Settling Time", f"{stime:.1f} s")
                col4.metric("Rise Time", f"{rtime:.1f} s")

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                
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
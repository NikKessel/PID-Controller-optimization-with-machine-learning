import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import pandas as pd

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

if mode == "Predict PID":
    model_choice = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "MLP", "XGBoost"], key="model_select")

    st.sidebar.markdown("**System Parameters**")
    system_type = st.sidebar.selectbox("System Type", ["PT1", "PT2", "PT1+Td", "PT2+Td", "Osc2"])
    K = st.sidebar.number_input("K (Gain)", min_value=0.1, max_value=5.0, value=1.0)
    T1 = st.sidebar.number_input("T1 (Time Constant in s)", min_value=1.0, max_value=50.0, value=20.0)
    T2 = st.sidebar.number_input("T2 (2nd Time Constant in s)", min_value=0.0, max_value=50.0, value=10.0) if "PT2" in system_type else 0.0
    Td = st.sidebar.number_input("Td (Dead Time in s)", min_value=0.0, max_value=5.0, value=1.0) if "Td" in system_type else 0.0

    st.sidebar.markdown("**Model-Specific Inputs**")
    if model_choice == "XGBoost":
        Tu = st.sidebar.number_input("Tu (Ultimate Period in s)", min_value=0.0, max_value=100.0, value=10.0)
        Tg = st.sidebar.number_input("Tg (Gradient Time in s)", min_value=0.0, max_value=100.0, value=20.0)
        overshoot = st.sidebar.number_input("Overshoot", min_value=0.0, max_value=2.0, value=0.1)
    elif model_choice == "MLP":
        Tu = st.sidebar.number_input("Tu (Ultimate Period in s)", min_value=0.0, max_value=100.0, value=10.0)
        Tg = st.sidebar.number_input("Tg (Gradient Time in s)", min_value=0.0, max_value=100.0, value=20.0)
        overshoot = st.sidebar.number_input("Overshoot", min_value=0.0, max_value=2.0, value=0.1)
    else:
        Tu = Tg = overshoot = sprunghoehe = 0.0

    # One-hot encode type manually
    type_PT1 = 1 if system_type == "PT1" else 0
    type_PT2 = 1 if system_type == "PT2" else 0
    type_PT1_Td = 1 if system_type == "PT1+Td" else 0
    type_PT2_Td = 1 if system_type == "PT2+Td" else 0
    type_Osc2 = 1 if system_type == "Osc2" else 0

# --- Conditional ML model selection ---
if mode == "Predict PID":
    st.info("Predicting optimal Kp, Ki, Kd using selected ML model")
    
    if st.button("🔍 Predict PID"):
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "streamlit_models")
            model_filename = f"model_{model_choice.lower().replace(' ', '_')}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            model = joblib.load(model_path)
            if model_choice == "Random Forest":
                X = np.array([[K, T1, T2, Td]])
            elif model_choice == "XGBoost":
                type_Osc2 = 1 if system_type == "Osc2" else 0
                X = np.array([[K, T1, T2, Td, Tu, Tg, overshoot, type_Osc2, type_PT1, type_PT1_Td, type_PT2, type_PT2_Td]])
            elif model_choice == "MLP":
                X = np.array([[K, T1, T2, Td, Tu, Tg, overshoot]])
            Kp, Ki, Kd = predict_pid_params(model, X)
            st.success("Prediction complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Kp", f"{Kp:.3f}")
            col2.metric("Ki", f"{Ki:.5f}")
            col3.metric("Kd", f"{Kd:.2f}")

            st.markdown("#### Step Response")
            t = np.linspace(0, 100, 500)
            y = 1 - np.exp(-t / 20)
            fig, ax = plt.subplots()
            ax.plot(t, y, label="Predicted Response")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Output")
            ax.set_title("Step Response of Predicted PID")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Model loading or prediction failed: {e}")

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
            fig, ax = plt.subplots()
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

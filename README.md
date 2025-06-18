
# 📡 Data-Driven Optimization of PID Controller Parameters Using Machine Learning

**Bachelor Thesis – Bioprocess Engineering**  
*Intelligent PID Tuning via Surrogate Models, ML, and Optimization*

📎 **Live App**: [Streamlit Interface](https://ml-based-pid-optimizer.streamlit.app/)

---

## 🎯 Objective

This project aims to develop a machine-learning-based system for **automated tuning of PID controller parameters** \((K_p, K_i, K_d)\) using both **simulated** and (optionally) **real-world** plant models \(G(s)\).

Key capabilities:
- Predict PID parameters directly from system dynamics
- Evaluate performance (ISE, overshoot, etc.) without time-domain simulation
- Tune optimal controllers using **surrogate models** and **Genetic Algorithms**
- Compare with classical methods (Ziegler–Nichols, CHR)
- Deliver results through an **interactive GUI**
- (WIP) Validate controllers in **Simulink** using MATLAB backend

---

## 🧭 Methodology

### 1. **Synthetic Dataset Generation**
- Randomized \(G(s)\): PT1, PT2, with/without dead time
- Simulated closed-loop step response
- Extract metrics: ISE, Overshoot, Rise Time, Settling Time, SSE

### 2. **Classical Baseline Tuning**
- Reference tuning via Ziegler–Nichols and CHR methods

### 3. **Machine Learning Models**
#### 🔹 **Model A: PID Parameter Prediction**
- Multi-output regression (MLP, XGBoost, Deep GP, Symbolic Regression)
- Predict Kp, Ki, Kd from system parameters (K, T1, T2, Td)

#### 🔸 **Model B: Surrogate Model**
- Predict full performance metrics from (G(s), PID) tuple
- Models: MLP, GPR, DGP
- Used in optimization pipeline

#### 🔵 **Model C: Performance Classifier**
- Predict quality class: {good, slow, unstable}
- Filters poor-performing data during surrogate model training

### 4. **PID Optimization Engine**
- Surrogate model estimates ISE, OS, etc.
- **Primary Method**: Genetic Algorithm minimizes weighted cost function  
- **(WIP)**: Bayesian Optimization under development to improve convergence speed and constraint handling  
- Supports custom constraints and tuning ranges

### 5. **GUI via Streamlit**
- Users input system parameters
- Visualize recommended PID values, performance metrics, and response plots
- Adjust optimization weights and rerun tuning interactively

### 6. **Validation (Ongoing)**
- Evaluate on known benchmark systems from literature
- Simulink-in-the-loop simulation via `.mat` parameter exchange (WIP)
- Future: System identification from experimental `u(t), y(t)` data

---

## 🔬 Technical Summary

| Classical Tuning        | Data-Driven Tuning                 |
|-------------------------|------------------------------------|
| Ziegler–Nichols         | MLP, XGBoost, DGP, Symbolic Models |
| CHR                     | Surrogate Optimization (GA, BO)    |
| Manual Adjustment       | Performance Classifier             |
| Time-Domain Simulation  | Instant Metric Prediction          |

---

## 📂 Project Structure

```
PID-Controller-optimization/
├── data/                  # Synthetic and real-world system datasets
├── src/                   # All core source code (models, logic, GUI)
├── models/                # Trained regressors, surrogates, classifiers
├── results/               # Optimization and benchmark outputs
├── plots/                 # Step responses, radar charts, performance plots
├── streamlit/             # Streamlit app logic and interface
├── simulink/              # Simulink model + MATLAB interaction scripts
├── requirements.txt       # Dependency list
├── README.md              # Project overview
```

---

## 🚀 Getting Started (WIP)

```bash
# Generate synthetic dataset
python src/data/generate_pid_dataset_full.m

# Train ML models of choice
python src/models/train_dgp.py

# Run PID optimization using surrogate + GA
python src/tuning/run_optimizer.py

# (Optional) Try Bayesian Optimization (WIP)
python src/tuning/run_bo.py

# Launch GUI
streamlit run streamlit/streamlit_app.py
```

---

## 📦 Tech Stack

- Python 3.12+
- Libraries: NumPy, SciPy, Pandas, scikit-learn, XGBoost, PySR, DEAP, PyTorch
- Surrogates: GPR, Deep GP, MLP
- Optimization: Genetic Algorithm (DEAP), Bayesian Optimization (Scikit-Optimize/BoTorch)
- Frontend: Streamlit
- Simulink backend via MATLAB `.mat` interaction (subprocess)
- Control Systems Toolbox (Python) for simulation

---

## 📈 Output & Results

- Interactive GUI for real-time PID tuning
- Metric predictions without full simulation
- Cost-minimized controller tuning using surrogate + GA/BO
- Exportable plots: step responses, error signal, radar plots
- Performance stats: MAE, R², classification accuracy

---

## ✅ Current Status (June 2025)

- ✅ Data generation for wide range of systems complete  
- ✅ PID regressors (XGBoost, DGP) with strong metrics (R² > 0.95)  
- ✅ Surrogate model enables fast cost evaluation  
- ✅ Optimization via Genetic Algorithm stable  
- 🔄 Bayesian Optimization module in development  
- ✅ Streamlit GUI functional and deployable  
- 🔄 Simulink validation and real system test in progress  

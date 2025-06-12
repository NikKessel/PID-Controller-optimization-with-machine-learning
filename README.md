# 📡 Data-Driven Optimization of PID Controller Parameters Using Machine Learning

**Bachelor Thesis – Bioprocess Engineering**  
*Automated PID Tuning via ML Algorithms*

https://ml-based-pid-optimizer.streamlit.app/
---

## 🎯 Objective

The goal of this project is to develop a machine-learning-based system for the **automated tuning of PID controller parameters** \((K_p, K_i, K_d)\), using both synthetic and—if available—real-world process models \(G(s)\).

Key features:
- Predict high-quality PID parameters directly from system models
- Optionally predict **ISE** (Integral of Squared Error) without full simulation
- Compare classical tuning methods (e.g., **Ziegler–Nichols**, **CHR**) against modern approaches (e.g., **XGBoost**, **MLPs**, **Genetic Algorithms**)
- Provide a **user-friendly GUI** for interactive use

---

## 🧭 Methodological Workflow

### 1. **Dataset Generation**
- Generate synthetic transfer functions \(G(s)\)
- Simulate control response with PID settings
- Compute quality metrics: ISE, overshoot, rise time, etc.

### 2. **Classical Baselines**
- Implement and evaluate Ziegler–Nichols and CHR methods as reference

### 3. **Machine Learning Models**
Three ML-based models are developed:

#### 🔹 Model 1: **Direct Parameter Prediction**
- **Input**: Transfer function features  
- **Output**: Kp, Ki, Kd  
- **Goal**: Recommend controller parameters  
- **Metrics**: MAE, RMSE for PID parameters

#### 🔸 Model 2: **ISE Prediction**
- **Input**: System & controller parameters  
- **Output**: ISE prediction  
- **Goal**: Evaluate control quality  
- **Metrics**: MAE, RMSE, \(R^2\) for ISE

#### 🔵 Model 3: **Performance Classification**
- **Input**: PID configuration  
- **Output**: Labels: {good, unstable, slow}  
- **Goal**: Classify controller performance  
- **Metrics**: Accuracy, Precision, Recall

### 4. **Optimization Framework**
- Use surrogate models with **Genetic Algorithms** for PID tuning  
- Minimize predicted ISE to find optimal Kp, Ki, Kd

### 5. **Comparison & Validation**
- Quantitative comparison of classical vs. data-driven approaches  
- Statistical evaluation of control performance and generalization

### 6. **GUI Implementation**
- Build a graphical user interface (GUI)  
- Allow users to input a transfer function and receive PID suggestions and plots

---

## 🔬 Methods Overview

| Classical Methods       | Data-Driven Approaches            |
|------------------------|-----------------------------------|
| Ziegler–Nichols        | Machine Learning (XGBoost, MLPs)  |
| CHR                    | Genetic Algorithms (GA)           |
| Analytical Formulas    | Surrogate-based Optimization      |
| Empirical Tuning       | Classification & Regression       |

---

## 📁 Project Structure

```
PID-Controller-optimization/
├── data/                  # Synthetic and real process datasets
├── src/                   # Source code (data, models, optimization, evaluation)
├── models/                # Trained ML and surrogate models
├── results/               # Benchmark results, ISE values, etc.
├── plots/                 # Step responses, error curves
├── benchmark_plots/       # Visual comparisons of tuning methods
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git exclusions
```

---

## 🚀 Getting Started (WIP)

```bash
# Generate synthetic data
python src/data/generate_data.py

# Train a model
python src/models/train_mlp.py

# Optimize using surrogate + GA
python src/tuning/optimizer.py

# Compare to Ziegler-Nichols
python src/evaluation/compare_to_chr.py
```

---

## 🛠 Technologies Used

- Python 3.12+
- NumPy, SciPy, Pandas
- scikit-learn, XGBoost, MLPRegressor
- DEAP (Genetic Algorithms)
- Control Systems Toolbox
- Matplotlib, seaborn
- Tkinter or Streamlit (planned GUI)

---

## 📊 Planned Output

- Performance plots comparing PID tuning strategies
- A surrogate-based optimizer with GA backend
- GUI to input \(G(s)\) and visualize results
- Statistical tables (MAE, ISE, Stability Classification)

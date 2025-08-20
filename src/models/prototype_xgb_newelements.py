# -*- coding: utf-8 -*-
"""
Per-family XGBoost models to predict PID gains (Kp, Ki, Kd) from plant + tuning features.

Families handled: PT2_osc, PT1PT2_existing, IT1, P
Features are tailored per family (physics-aware) and include tuning settings (wc, PhaseMargin, DesignFocus).
Targets are trained in log-space (log1p) and evaluated on the original scale.

Outputs:
- models/XGB_KpKiKd_Family/xgb_kpkikd_family_<timestamp>/
    <Family>_<Target>_xgb.pkl
    <Family>_<Target>_scaler.pkl
    <Family>_<Target>_diagnostics.csv
    <Family>_metrics_summary.csv
    metrics_summary_all.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# =========================
# ======== CONFIG =========
# =========================
BASE = r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning"
DATA_PATH = os.path.join(BASE, "src", "data", "pid_dataset_pidtune_extended.csv")

USE_LOG_TARGETS = True        # log1p targets for stability
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROWS = 50                 # minimum rows per (family,target) to train

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(BASE, "models", "XGB_KpKiKd_Family", f"xgb_kpkikd_family_{timestamp}")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ["Kp", "Ki", "Kd"]

# =========================
# ======= LOAD DATA =======
# =========================
print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Loaded shape:", df.shape)
df = df[df["ISE"] < 5]
df = df[df["ISE"] > 0.001]
df = df[df["SettlingTime"] < 50]
df = df[df["RiseTime"] < 30]
df = df[df["SettlingTime"] > 0.01000]
df = df[df["RiseTime"] > 0.01000]
df = df[df["Overshoot"] < 40]
df = df[df["Kp"] < 20 ]
df = df[df["Kp"] > 0.3 ]
df = df[df["Ki"] < 20]
df = df[df["Kd"] < 20]
# Ensure required raw columns exist (extended CSV should have these)
for c in ["K", "L", "T1", "T2", "w0", "zeta", "Tchar",  "Family"]:
    if c not in df.columns:
        # Create safe defaults if missing
        if c in ["DesignFocus", "Family"]:
            df[c] = ""
        else:
            df[c] = np.nan

# One-hot DesignFocus (tuning knob from pidtune)
focus_dummies = pd.get_dummies(df["DesignFocus"].astype(str), prefix="focus")
df = pd.concat([df, focus_dummies], axis=1)
FOCUS_COLS = list(focus_dummies.columns)

# =========================
# === Feature engineering
# =========================
def log1p_safe(series):
    return np.log1p(series.fillna(0.0))

# Build log features (NaN-safe)
df["logK"]      = log1p_safe(df["K"])
df["logL1p"]    = log1p_safe(df["L"])
df["logT1p"]    = log1p_safe(df["T1"])
df["logT2p"]    = log1p_safe(df["T2"])
df["logw0p"]    = log1p_safe(df["w0"])
df["logTcharp"] = log1p_safe(df["Tchar"])

# Fill NaNs in numeric columns used below
for c in ["zeta", "T1", "T2", "w0", "Tchar", "L"]:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# =========================
# === Family-specific features (physics-aware)
# =========================
# Common tuning features across families

FAMILY_FEATURES = {
    # Oscillatory second-order: K, L, zeta, w0 + tuning
    "PT2_osc":           ["logK", "logL1p", "zeta", "logw0p"] ,

    # Legacy PT1/PT2 mixture: K, T1, T2, L + tuning
    "PT1PT2_existing":   ["logK", "logL1p", "logT1p", "logT2p"] ,

    # Integrator with lag: K, T (use T1), L + tuning
    "IT1":               ["logK", "logL1p", "logT1p"] ,

    # Pure gain (+ delay): K, L + tuning
    "P":                 ["logK", "logL1p"] ,
}

# If Family missing on any rows, infer coarsely from SystemType as fallback
if (df["Family"].astype(str).str.len() == 0).any():
    if "SystemType" in df.columns:
        st = df["SystemType"].astype(str)
        df.loc[st.str.contains("PT2osc", case=False, na=False), "Family"] = "PT2_osc"
        df.loc[st.str.contains("IT1", case=False, na=False),    "Family"] = "IT1"
        df.loc[st.str.contains("P", case=False, na=False),      "Family"] = "P"
        # default
        df.loc[df["Family"].astype(str).str.len() == 0, "Family"] = "PT1PT2_existing"
    else:
        df["Family"] = "PT1PT2_existing"

families = df["Family"].astype(str).unique().tolist()
print("Families present:", families)

# =========================
# ===== TRAIN/EXPORT ======
# =========================
all_metrics = {}

for fam in families:
    if fam not in FAMILY_FEATURES:
        print(f"Skipping unknown family '{fam}' (no feature map).")
        continue

    feats = [c for c in FAMILY_FEATURES[fam] if c in df.columns]
    if len(feats) == 0:
        print(f"Family {fam}: no usable features found, skipping.")
        continue

    df_fam = df[df["Family"] == fam].copy()
    print(f"\n=== Family: {fam} | rows: {len(df_fam)} | features: {feats}")

    fam_metrics = {}

    for target in TARGETS:
        if target not in df_fam.columns:
            print(f"  ⚠️ {fam}-{target}: target column missing, skip.")
            continue

        # Keep rows where target is available
        df_t = df_fam.dropna(subset=[target])
        if df_t.shape[0] < MIN_ROWS:
            print(f"  ⚠️ {fam}-{target}: only {df_t.shape[0]} rows (<{MIN_ROWS}), skip.")
            continue

        X = df_t[feats].values.astype(np.float32)
        y = df_t[target].values.astype(np.float32)

        # Train/test split
        X_train, X_test, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Target transform
        if USE_LOG_TARGETS:
            y_train = np.log1p(np.clip(y_train_orig, 0, None))
        else:
            y_train = y_train_orig

        # Scale inputs
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # XGBoost config (robust defaults; tune if needed)
        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train_sc, y_train)

        # Predict & invert transform
        y_pred = model.predict(X_test_sc)
        if USE_LOG_TARGETS:
            y_pred_orig = np.expm1(y_pred)
        else:
            y_pred_orig = y_pred

        # Metrics on original scale
        r2 = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        fam_metrics[target] = {"R2": r2, "MAE": mae}
        print(f"  📊 {fam}-{target}: R²={r2:.4f}, MAE={mae:.4f}")

        # Diagnostics
        diag = pd.DataFrame({
            "y_true": y_test_orig,
            "y_pred": y_pred_orig,
            "abs_error": np.abs(y_test_orig - y_pred_orig),
            "rel_error_%": 100 * np.abs(y_test_orig - y_pred_orig) / np.clip(y_test_orig, 1e-6, None)
        })
        diag.to_csv(os.path.join(OUT_DIR, f"{fam}_{target}_diagnostics.csv"), index=False)

        # Plot
        plt.figure(figsize=(6,6))
        plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
        mn, mx = float(min(y_test_orig.min(), y_pred_orig.min())), float(max(y_test_orig.max(), y_pred_orig.max()))
        plt.plot([mn, mx], [mn, mx], "r--", alpha=0.8, label="Perfect")
        plt.title(f"{fam}-{target}: R²={r2:.3f}, MAE={mae:.3f}")
        plt.xlabel("True"); plt.ylabel("Pred"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{fam}_{target}_plot.png"), dpi=300)
        plt.close()

        # Save model + scaler
        joblib.dump(model,  os.path.join(OUT_DIR, f"{fam}_{target}_xgb.pkl"))
        joblib.dump(scaler, os.path.join(OUT_DIR, f"{fam}_{target}_scaler.pkl"))

    fam_summary = pd.DataFrame(fam_metrics).T
    fam_summary.to_csv(os.path.join(OUT_DIR, f"{fam}_metrics_summary.csv"))
    print(f"✅ {fam} summary:\n{fam_summary}\n")
    all_metrics[fam] = fam_summary

# Global summary
if len(all_metrics):
    summary_df = pd.concat(all_metrics, axis=1)
    summary_df.to_csv(os.path.join(OUT_DIR, "metrics_summary_all.csv"))
    print("=== All families — summary ===")
    print(summary_df)
    print("\nSaved to:", OUT_DIR)
else:
    print("No models trained — check data availability and MIN_ROWS.")

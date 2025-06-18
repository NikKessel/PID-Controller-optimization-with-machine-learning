# csv_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Load CSV ===
#file_path = "your_dataset.csv"  # <-- change this#
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_multi_ranges.csv")

# === Basic Info ===
print("✅ Data Overview")
print(df.shape)
print(df.dtypes)
print(df.describe(include='all'))
print("\nMissing values per column:")
print(df.isnull().sum())

# === Numerical Columns ===
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# === Distribution Plots ===
print("\n📊 Plotting distributions...")
df[numeric_cols].hist(bins=50, figsize=(18, 14))
plt.suptitle("Variable Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# === IQR Outlier Detection ===
print("\n🔍 Outlier Summary (IQR Method)")
outlier_stats = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    outlier_stats[col] = len(outliers)

outlier_df = pd.DataFrame.from_dict(outlier_stats, orient='index', columns=['Outlier Count'])
print(outlier_df.sort_values(by='Outlier Count', ascending=False))

# === Correlation Heatmap ===
print("\n📈 Correlation Heatmap")
plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# === Log-Log Scatterplots vs ISE (or target metric) ===
print("\n📐 Scatterplots with Log Scaling (Kp/Ki/Kd vs. ISE)")
targets = ["Kp", "Ki", "Kd"]
if "ISE" in df.columns:
    for col in targets:
        plt.figure()
        sns.scatterplot(x=df[col], y=df["ISE"])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(col)
        plt.ylabel("ISE")
        plt.title(f"{col} vs ISE (Log-Log Scale)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Top Outliers Per Parameter ===
print("\n🚨 Top 5 highest values per parameter")
for col in targets:
    if col in df.columns:
        print(f"\nTop 5 {col}:")
        print(df[[col]].sort_values(by=col, ascending=False).head())

# === PCA Projection ===
print("\n🧠 PCA Projection to detect structural anomalies")
X_clean = df[numeric_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
plt.title("PCA Projection of Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Range Distribution Summary for Kp, Ki, Kd ===
print("\n📊 Range Distribution (%) for Kp, Ki, Kd")

# Define bins and labels
bins = [0, 1, 10, 100, 1000, 10000, float("inf")]
labels = ["0–1", "1.1–10", "11–100", "101–1k", "1k–10k", "10k+"]

# Loop over target columns
for col in ["K", "T1", "T2", "Kp", "Ki", "Kd", "ISE", "SSE", 
            "RiseTime", "SettlingTime", "Overshoot"]:
    # Your logic here
    print(f"\n=== {col} Range Distribution ===")
    
    # Create bin categories
    df[f"{col}_range"] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    
    # Count and percentage
    counts = df[f"{col}_range"].value_counts(sort=False)
    percentages = 100 * counts / len(df)
    
    # Combine and print
    summary = pd.DataFrame({
        "Range": labels,
        "Count": counts.values,
        "Percentage": percentages.round(2).values
    })
    print(summary)

    # Optional: Plot bar chart
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Range", y="Percentage", data=summary)
    plt.title(f"{col} Distribution by Range (%)")
    plt.ylabel("Percentage of Samples")
    plt.xlabel(f"{col} Value Range")
    plt.tight_layout()
    plt.show()
# === Identify K, T1, T2 ranges producing large Kp/Ki/Kd values ===
thresholds = [100, 1000]

for threshold in thresholds:
    print(f"\n🔎 Analyzing input parameters for Kp/Ki/Kd above {threshold}")
    
    # Select data exceeding the threshold
    df_high = df[(df["Kp"] > threshold) | (df["Ki"] > threshold) | (df["Kd"] > threshold)]
    
    if df_high.empty:
        print(f"No samples found with Kp/Ki/Kd above {threshold}.")
        continue
    
    # Summary statistics
    summary_stats = df_high[["K", "T1", "T2"]].describe(percentiles=[0.25, 0.5, 0.75])
    print(f"\n📌 Summary stats for Kp/Ki/Kd above {threshold}:\n{summary_stats}")

    # Visualize distributions of K, T1, T2 in this subset
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    params = ["K", "T1", "T2"]
    for ax, param in zip(axes, params):
        sns.histplot(df_high[param], bins=30, kde=True, ax=ax)
        ax.set_title(f"{param} distribution (Kp/Ki/Kd > {threshold})")
        ax.set_xlabel(param)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

categorical_cols = ["PhaseMargin", "DesignFocus", "Bandwidth", "SystemCategory"]
thresholds = [100, 1000]

for threshold in thresholds:
    print(f"\n🔎 Categorical Analysis for Kp/Ki/Kd values > {threshold}")
    
    df_high = df[(df["Kp"] > threshold) | (df["Ki"] > threshold) | (df["Kd"] > threshold)]
    
    if df_high.empty:
        print(f"No entries found with Kp/Ki/Kd above {threshold}.")
        continue
    
    for cat_col in categorical_cols:
        print(f"\n📌 Category distribution in '{cat_col}' for gains > {threshold}:")
        counts = df_high[cat_col].value_counts()
        percentages = counts / len(df_high) * 100
        summary_df = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": percentages.round(2)
        })
        print(summary_df)

        # Optional visualization:
        plt.figure(figsize=(8, 4))
        sns.barplot(y=summary_df.index, x=summary_df["Percentage (%)"])
        plt.title(f"{cat_col} Distribution for Gains > {threshold}")
        plt.xlabel("Percentage (%)")
        plt.ylabel(cat_col)
        plt.tight_layout()
        plt.show()

thresholds = [100, 1000]

for threshold in thresholds:
    df_high_kp = df[df["Kp"] > threshold]

    count_t2_zero = (df_high_kp["T2"] == 0).sum()
    count_t2_gt_zero = (df_high_kp["T2"] > 0).sum()

    total = len(df_high_kp)

    print(f"\n🔎 Analysis of T2 values for Kp > {threshold}:")
    print(f"Total entries with Kp > {threshold}: {total}")
    print(f"T2 = 0 count: {count_t2_zero} ({(count_t2_zero / total * 100):.2f}%)")
    print(f"T2 > 0 count: {count_t2_gt_zero} ({(count_t2_gt_zero / total * 100):.2f}%)")

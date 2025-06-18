# csv_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Load CSV ===
#file_path = "your_dataset.csv"  # <-- change this#
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\data\pid_dataset_pidtune.csv")

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
for col in ["Kp", "Ki", "Kd"]:
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

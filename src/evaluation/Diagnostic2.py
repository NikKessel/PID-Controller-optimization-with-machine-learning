# === Diagnosis and Fix for Settling Time Underprediction ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import os
import torch
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# === Load and preprocess dataset ===
df = pd.read_csv(r"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\src\data\pid_dataset_pidtune.csv")
df = df.dropna(subset=["K", "T1", "T2", "Kp", "Ki", "Kd", "ISE", "Overshoot", "SettlingTime", "RiseTime"])
df = df.sample(n=4000, random_state=42).reset_index(drop=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\KesselN\Documents\GitHub\PID-Controller-optimization-with-machine-learning\models\test\surrogate_perf_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === Filter outliers ===
print(f"Original dataset: {df.shape}")

df = df[df["ISE"] < 100]
df = df[df["ISE"] > 0.001]
df = df[df["SettlingTime"] < 500]
df = df[df["RiseTime"] < 300]
df = df[df["SettlingTime"] > 0.1000]
df = df[df["RiseTime"] > 0.1000]
df = df[df["Overshoot"] > 0.001]
df = df[df["Kp"] < 200 ]
df = df[df["Kp"] > 0.3 ]
df = df[df["Ki"] < 200]
df = df[df["Kd"] < 200]
print(f"After filtering: {df.shape}")

# Add log-transformed targets
df["ISE_log"] = np.log1p(df["ISE"])
df["SettlingTime_log"] = np.log1p(df["SettlingTime"])
df["RiseTime_log"] = np.log1p(df["RiseTime"])

features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
target_config = {
    "ISE_log": {"original": "ISE", "use_log": True},
    "Overshoot": {"original": "Overshoot", "use_log": False},
    "SettlingTime_log": {"original": "SettlingTime", "use_log": True},
    "RiseTime_log": {"original": "RiseTime", "use_log": True}
}
# === Diagnosis and Fix for Settling Time Underprediction ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

def diagnose_settling_time_bias(df, model, likelihood, scaler, target_config, output_dir):
    """
    Comprehensive diagnosis of settling time prediction bias
    """
    print("🔍 DIAGNOSING SETTLING TIME PREDICTION BIAS")
    print("=" * 50)
    
    # 1. Check for systematic bias across different ranges
    target = "SettlingTime_log"
    config = target_config[target]
    
    # Get predictions for full dataset
    features = ["K", "T1", "T2", "Kp", "Ki", "Kd"]
    X = df[features].values.astype(np.float32)
    y_true_orig = df[config["original"]].values
    
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_tensor))
        y_pred_log = preds.mean.numpy()
        y_pred_orig = np.expm1(y_pred_log)
        y_pred_orig = np.clip(y_pred_orig, 0, None)
    
    # Create comprehensive analysis dataframe
    analysis_df = pd.DataFrame({
        'true_settling_time': y_true_orig,
        'pred_settling_time': y_pred_orig,
        'absolute_error': np.abs(y_true_orig - y_pred_orig),
        'relative_error': (y_pred_orig - y_true_orig) / y_true_orig * 100,  # Negative = underprediction
        'K': df['K'].values,
        'T1': df['T1'].values,
        'T2': df['T2'].values,
        'Kp': df['Kp'].values,
        'Ki': df['Ki'].values,
        'Kd': df['Kd'].values
    })
    
    # 2. Analyze bias by settling time ranges
    print("\n📊 BIAS ANALYSIS BY SETTLING TIME RANGES:")
    settling_ranges = [
        (0, 10, "Very Fast"),
        (10, 50, "Fast"), 
        (50, 100, "Medium"),
        (100, 200, "Slow"),
        (200, 500, "Very Slow")
    ]
    
    bias_by_range = []
    for min_val, max_val, label in settling_ranges:
        mask = (analysis_df['true_settling_time'] >= min_val) & (analysis_df['true_settling_time'] < max_val)
        if mask.sum() > 0:
            subset = analysis_df[mask]
            mean_bias = subset['relative_error'].mean()
            median_bias = subset['relative_error'].median()
            count = len(subset)
            bias_by_range.append({
                'Range': label,
                'Count': count,
                'Mean_Bias_%': mean_bias,
                'Median_Bias_%': median_bias,
                'Mean_True': subset['true_settling_time'].mean(),
                'Mean_Pred': subset['pred_settling_time'].mean()
            })
            print(f"{label:12} ({min_val:3.0f}-{max_val:3.0f}): "
                  f"Count={count:3d}, Mean_Bias={mean_bias:6.1f}%, "
                  f"True_Avg={subset['true_settling_time'].mean():6.1f}, "
                  f"Pred_Avg={subset['pred_settling_time'].mean():6.1f}")
    
    bias_df = pd.DataFrame(bias_by_range)
    bias_df.to_csv(f"{output_dir}/settling_time_bias_analysis.csv", index=False)
    
    # 3. Check for parameter-dependent bias
    print("\n🔧 PARAMETER-DEPENDENT BIAS ANALYSIS:")
    param_correlations = {}
    for param in ['K', 'T1', 'T2', 'Kp', 'Ki', 'Kd']:
        corr = np.corrcoef(analysis_df[param], analysis_df['relative_error'])[0, 1]
        param_correlations[param] = corr
        print(f"{param:3s} correlation with bias: {corr:6.3f}")
    
    # 4. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: True vs Predicted with bias coloring
    scatter = axes[0, 0].scatter(analysis_df['true_settling_time'], 
                                analysis_df['pred_settling_time'],
                                c=analysis_df['relative_error'], 
                                cmap='RdBu_r', alpha=0.6, s=30)
    axes[0, 0].plot([0, analysis_df['true_settling_time'].max()], 
                    [0, analysis_df['true_settling_time'].max()], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('True Settling Time')
    axes[0, 0].set_ylabel('Predicted Settling Time')
    axes[0, 0].set_title('Predictions Colored by Bias (%)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Relative Error (%)')
    
    # Plot 2: Bias distribution
    axes[0, 1].hist(analysis_df['relative_error'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Bias')
    axes[0, 1].axvline(analysis_df['relative_error'].mean(), color='orange', 
                      linestyle='-', alpha=0.7, label=f'Mean: {analysis_df["relative_error"].mean():.1f}%')
    axes[0, 1].set_xlabel('Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Bias')
    axes[0, 1].legend()
    
    # Plot 3: Bias vs True Values
    axes[0, 2].scatter(analysis_df['true_settling_time'], analysis_df['relative_error'], alpha=0.6)
    axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('True Settling Time')
    axes[0, 2].set_ylabel('Relative Error (%)')
    axes[0, 2].set_title('Bias vs True Values')
    
    # Plot 4: Parameter correlations with bias
    params = list(param_correlations.keys())
    corrs = list(param_correlations.values())
    colors = ['red' if abs(c) > 0.2 else 'blue' for c in corrs]
    axes[1, 0].bar(params, corrs, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_ylabel('Correlation with Bias')
    axes[1, 0].set_title('Parameter Correlation with Bias')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Residuals vs Fitted
    axes[1, 1].scatter(analysis_df['pred_settling_time'], 
                      analysis_df['true_settling_time'] - analysis_df['pred_settling_time'], alpha=0.6)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Settling Time')
    axes[1, 1].set_ylabel('Residuals (True - Pred)')
    axes[1, 1].set_title('Residual Analysis')
    
    # Plot 6: Bias by ranges
    if len(bias_df) > 0:
        axes[1, 2].bar(bias_df['Range'], bias_df['Mean_Bias_%'], alpha=0.7)
        axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].set_ylabel('Mean Bias (%)')
        axes[1, 2].set_title('Bias by Settling Time Ranges')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/settling_time_bias_diagnosis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary statistics
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"Mean Relative Error: {analysis_df['relative_error'].mean():.2f}%")
    print(f"Median Relative Error: {analysis_df['relative_error'].median():.2f}%")
    print(f"Std of Relative Error: {analysis_df['relative_error'].std():.2f}%")
    print(f"% of Underpredictions: {(analysis_df['relative_error'] < 0).mean() * 100:.1f}%")
    print(f"R²: {r2_score(analysis_df['true_settling_time'], analysis_df['pred_settling_time']):.4f}")
    
    return analysis_df, bias_df


def fix_settling_time_bias(df, features, target_config, output_dir):
    """
    Multiple strategies to fix settling time underprediction bias
    """
    print("\n🔧 IMPLEMENTING BIAS CORRECTION STRATEGIES")
    print("=" * 50)
    
    # Strategy 1: Bias-corrected log transform
    print("\n1. Trying bias-corrected log transform...")
    
    # Find optimal bias correction
    settling_orig = df["SettlingTime_orig"].values
    settling_log_original = np.log1p(settling_orig)
    
    # Try different bias corrections
    bias_corrections = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    best_correction = 0
    best_symmetry = float('inf')
    
    for correction in bias_corrections:
        log_corrected = np.log1p(settling_orig + correction)
        # Measure symmetry of distribution
        symmetry = abs(np.percentile(log_corrected, 75) + np.percentile(log_corrected, 25) - 
                      2 * np.percentile(log_corrected, 50))
        if symmetry < best_symmetry:
            best_symmetry = symmetry
            best_correction = correction
    
    print(f"Best bias correction: {best_correction}")
    df["SettlingTime_log_corrected"] = np.log1p(df["SettlingTime_orig"] + best_correction)
    
    # Strategy 2: Square root transform (alternative to log)
    print("\n2. Trying square root transform...")
    df["SettlingTime_sqrt"] = np.sqrt(df["SettlingTime_orig"])
    
    # Strategy 3: Box-Cox-like transform
    print("\n3. Trying Box-Cox-like transform...")
    lambda_param = 0.3  # Can be optimized
    df["SettlingTime_boxcox"] = (df["SettlingTime_orig"]**lambda_param - 1) / lambda_param
    
    # Strategy 4: Quantile transform
    print("\n4. Trying rank-based transform...")
    from scipy.stats import rankdata
    ranks = rankdata(df["SettlingTime_orig"])
    df["SettlingTime_rank"] = (ranks - 1) / (len(ranks) - 1)  # Normalize to [0,1]
    
    # Updated target configuration
    extended_target_config = {
        "SettlingTime_log": {"original": "SettlingTime_orig", "use_log": True, "inverse": lambda x: np.expm1(x)},
        "SettlingTime_log_corrected": {"original": "SettlingTime_orig", "use_log": True, 
                                     "inverse": lambda x: np.expm1(x) - best_correction},
        "SettlingTime_sqrt": {"original": "SettlingTime_orig", "use_log": False, 
                             "inverse": lambda x: x**2},
        "SettlingTime_boxcox": {"original": "SettlingTime_orig", "use_log": False, 
                               "inverse": lambda x: (x * lambda_param + 1)**(1/lambda_param)},
        "SettlingTime_rank": {"original": "SettlingTime_orig", "use_log": False, 
                             "inverse": lambda x: np.interp(x, 
                                                           np.linspace(0, 1, len(df)), 
                                                           np.sort(df["SettlingTime_orig"]))}
    }
    
    return df, extended_target_config


def train_improved_settling_time_model(df, features, target_name, config, output_dir):
    """
    Train settling time model with bias correction techniques
    """
    print(f"\n🚀 Training improved model for {target_name}")
    
    # Data preparation
    df_target = df.dropna(subset=features + [target_name] + [config["original"]])
    X = df_target[features].values.astype(np.float32)
    y = df_target[target_name].values.astype(np.float32)
    y_orig = df_target[config["original"]].values.astype(np.float32)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    train_idx, test_idx = train_test_split(range(len(df_target)), test_size=0.2, random_state=42)
    
    X_train = X_scaled[train_idx]
    y_train = y[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y[test_idx]
    y_test_orig = y_orig[test_idx]
    
    # Enhanced GP model with better kernel
    class EnhancedVariationalGP(gpytorch.models.ApproximateGP):
        def __init__(self, input_dim):
            inducing_points = torch.randn(256, input_dim)  # More inducing points
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            
            # Better mean function
            self.mean_module = gpytorch.means.LinearMean(input_dim)
            
            # Enhanced kernel with multiple components
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel() + 
                gpytorch.kernels.MaternKernel(nu=2.5) +
                gpytorch.kernels.LinearKernel()
            )

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    # Train model
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    model = EnhancedVariationalGP(X.shape[1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # Enhanced training with learning rate schedule
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))
    
    for i in range(500):  # More training iterations
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/500 - Loss: {loss.item():.4f}")
    
    # Prediction with bias correction
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test_tensor))
        y_pred_transformed = preds.mean.numpy()
        
        # Apply inverse transform
        y_pred_orig = config["inverse"](y_pred_transformed)
        y_pred_orig = np.clip(y_pred_orig, 0, None)
        
        # Additional bias correction based on training data
        training_preds = likelihood(model(X_train_tensor)).mean.numpy()
        training_pred_orig = config["inverse"](training_preds)
        training_pred_orig = np.clip(training_pred_orig, 0, None)
        
        # Calculate systematic bias on training data
        training_bias = np.mean(y_orig[train_idx] - training_pred_orig)
        print(f"Detected training bias: {training_bias:.3f}")
        
        # Apply bias correction to test predictions
        y_pred_orig_corrected = y_pred_orig + training_bias
        y_pred_orig_corrected = np.clip(y_pred_orig_corrected, 0, None)
    
    # Evaluate both versions
    r2_uncorrected = r2_score(y_test_orig, y_pred_orig)
    r2_corrected = r2_score(y_test_orig, y_pred_orig_corrected)
    
    mae_uncorrected = mean_absolute_error(y_test_orig, y_pred_orig)
    mae_corrected = mean_absolute_error(y_test_orig, y_pred_orig_corrected)
    
    bias_uncorrected = np.mean((y_pred_orig - y_test_orig) / y_test_orig * 100)
    bias_corrected = np.mean((y_pred_orig_corrected - y_test_orig) / y_test_orig * 100)
    
    print(f"\nUncorrected - R²: {r2_uncorrected:.4f}, MAE: {mae_uncorrected:.3f}, Bias: {bias_uncorrected:.2f}%")
    print(f"Corrected   - R²: {r2_corrected:.4f}, MAE: {mae_corrected:.3f}, Bias: {bias_corrected:.2f}%")
    
    # Save the better model
    if abs(bias_corrected) < abs(bias_uncorrected):
        print("✅ Using bias-corrected model")
        final_predictions = y_pred_orig_corrected
        final_bias = training_bias
    else:
        print("⚠️ Bias correction didn't help, using original")
        final_predictions = y_pred_orig
        final_bias = 0
    
    # Save model and correction factor
    torch.save(model.state_dict(), f"{output_dir}/{target_name}_enhanced_model.pth")
    torch.save(likelihood.state_dict(), f"{output_dir}/{target_name}_enhanced_likelihood.pth")
    joblib.dump(scaler, f"{output_dir}/{target_name}_enhanced_scaler.pkl")
    joblib.dump({'config': config, 'bias_correction': final_bias}, 
                f"{output_dir}/{target_name}_enhanced_config.pkl")
    
    return model, likelihood, scaler, config, final_bias


# Example usage in your main training loop:
"""
# After loading your data and before training:
df, extended_target_config = fix_settling_time_bias(df, features, target_config, output_dir)

# Train multiple variants and compare:
settling_variants = [
    "SettlingTime_log",
    "SettlingTime_log_corrected", 
    "SettlingTime_sqrt",
    "SettlingTime_boxcox"
]

best_model = None
best_bias = float('inf')
best_variant = None

for variant in settling_variants:
    if variant in extended_target_config:
        print(f"\n{'='*60}")
        print(f"TESTING VARIANT: {variant}")
        print(f"{'='*60}")
        
        model, likelihood, scaler, config, bias_correction = train_improved_settling_time_model(
            df, features, variant, extended_target_config[variant], output_dir
        )
        
        if abs(bias_correction) < abs(best_bias):
            best_bias = bias_correction
            best_model = (model, likelihood, scaler, config, bias_correction)
            best_variant = variant

print(f"\n🏆 BEST VARIANT: {best_variant} with bias correction: {best_bias:.3f}")
"""
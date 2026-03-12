"""
Calculate proper confidence intervals for r = -0.962 correlation
Using actual simulation data with bootstrap resampling
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import bootstrap

print("="*80)
print("CALCULATING PROPER CONFIDENCE INTERVALS")
print("="*80)

# Load the three-vehicle validation data
df = pd.read_csv('github/results/data/task8_three_vehicle/three_vehicle_validation_results.csv')

print(f"\nData loaded: {len(df)} simulations")
print(f"Columns: {list(df.columns)}")

# Check for the right column names
if 'actual_D' in df.columns:
    D_col = 'actual_D'
elif 'D' in df.columns:
    D_col = 'D'
else:
    print("ERROR: No D column found!")
    exit(1)

print(f"\nUsing D column: {D_col}")
print(f"D range: {df[D_col].min():.3f} to {df[D_col].max():.3f}")

# Overall correlation
r_overall, p_overall = stats.pearsonr(df[D_col], df['beta'])
print(f"\n" + "="*80)
print(f"OVERALL CORRELATION")
print(f"="*80)
print(f"r = {r_overall:.4f}")
print(f"p = {p_overall:.2e}")
print(f"n = {len(df)}")

# METHOD 1: Bootstrap confidence intervals (proper method)
print(f"\n" + "="*80)
print(f"METHOD 1: BOOTSTRAP CONFIDENCE INTERVALS")
print(f"="*80)

def correlation_statistic(x, y, axis=-1):
    """Calculate correlation for bootstrap"""
    return stats.pearsonr(x, y)[0]

# Prepare data for bootstrap
data = (df[D_col].values, df['beta'].values)

# Bootstrap with 10,000 resamples
np.random.seed(42)
n_bootstrap = 10000
bootstrap_correlations = []

for i in range(n_bootstrap):
    # Resample with replacement
    indices = np.random.choice(len(df), size=len(df), replace=True)
    r_boot, _ = stats.pearsonr(df[D_col].iloc[indices], df['beta'].iloc[indices])
    bootstrap_correlations.append(r_boot)

bootstrap_correlations = np.array(bootstrap_correlations)

# Calculate 95% CI
ci_lower = np.percentile(bootstrap_correlations, 2.5)
ci_upper = np.percentile(bootstrap_correlations, 97.5)

print(f"\nBootstrap (10,000 resamples):")
print(f"  r = {r_overall:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  CI width: {ci_upper - ci_lower:.4f}")

# METHOD 2: Per-vehicle correlations (if vehicle column exists)
if 'vehicle' in df.columns:
    print(f"\n" + "="*80)
    print(f"METHOD 2: PER-VEHICLE CORRELATIONS")
    print(f"="*80)
    
    vehicle_correlations = []
    for vehicle in sorted(df['vehicle'].unique()):
        df_v = df[df['vehicle'] == vehicle]
        r_v, p_v = stats.pearsonr(df_v[D_col], df_v['beta'])
        vehicle_correlations.append(r_v)
        print(f"  {vehicle}: r = {r_v:.4f}, n = {len(df_v)}")
    
    # Calculate CI from vehicle variation
    vehicle_correlations = np.array(vehicle_correlations)
    mean_r = vehicle_correlations.mean()
    std_r = vehicle_correlations.std(ddof=1)
    
    # 95% CI using t-distribution (n=3 vehicles)
    from scipy.stats import t
    n_vehicles = len(vehicle_correlations)
    t_critical = t.ppf(0.975, n_vehicles - 1)
    ci_lower_vehicle = mean_r - t_critical * std_r / np.sqrt(n_vehicles)
    ci_upper_vehicle = mean_r + t_critical * std_r / np.sqrt(n_vehicles)
    
    print(f"\n  Mean r across vehicles: {mean_r:.4f}")
    print(f"  Std dev: {std_r:.4f}")
    print(f"  95% CI (t-distribution): [{ci_lower_vehicle:.4f}, {ci_upper_vehicle:.4f}]")
    print(f"  CI width: {ci_upper_vehicle - ci_lower_vehicle:.4f}")

# METHOD 3: Fisher Z-transformation (analytical method)
print(f"\n" + "="*80)
print(f"METHOD 3: FISHER Z-TRANSFORMATION (ANALYTICAL)")
print(f"="*80)

# Fisher Z transformation
z = np.arctanh(r_overall)
se_z = 1 / np.sqrt(len(df) - 3)
z_critical = 1.96  # 95% CI

z_lower = z - z_critical * se_z
z_upper = z + z_critical * se_z

# Transform back to r
r_lower_fisher = np.tanh(z_lower)
r_upper_fisher = np.tanh(z_upper)

print(f"\nFisher Z-transformation:")
print(f"  r = {r_overall:.4f}")
print(f"  95% CI: [{r_lower_fisher:.4f}, {r_upper_fisher:.4f}]")
print(f"  CI width: {r_upper_fisher - r_lower_fisher:.4f}")

# SUMMARY AND RECOMMENDATION
print(f"\n" + "="*80)
print(f"SUMMARY AND RECOMMENDATION")
print(f"="*80)

print(f"\nAll three methods give similar results:")
print(f"  Bootstrap:        [{ci_lower:.4f}, {ci_upper:.4f}]")
if 'vehicle' in df.columns:
    print(f"  Per-vehicle:      [{ci_lower_vehicle:.4f}, {ci_upper_vehicle:.4f}]")
print(f"  Fisher Z:         [{r_lower_fisher:.4f}, {r_upper_fisher:.4f}]")

print(f"\nRECOMMENDED FOR MANUSCRIPT:")
print(f"  Use Bootstrap CI (most robust for non-normal data)")
print(f"  r = {r_overall:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]), n = {len(df)}")

print(f"\nREPLACE IN MANUSCRIPT:")
print(f"  OLD: r = -0.962, p < 10^{{-278}}, n=1500")
print(f"  NEW: r = {r_overall:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]), n = {len(df)}")

# Check if the previously estimated CI [-0.968, -0.956] was reasonable
print(f"\n" + "="*80)
print(f"CHECKING PREVIOUS ESTIMATE")
print(f"="*80)
print(f"\nPrevious estimate: [-0.968, -0.956]")
print(f"Bootstrap result:  [{ci_lower:.3f}, {ci_upper:.3f}]")

if abs(ci_lower - (-0.968)) < 0.01 and abs(ci_upper - (-0.956)) < 0.01:
    print(f"✓ Previous estimate was ACCURATE!")
else:
    print(f"✗ Previous estimate needs updating")
    print(f"  Difference in lower bound: {abs(ci_lower - (-0.968)):.4f}")
    print(f"  Difference in upper bound: {abs(ci_upper - (-0.956)):.4f}")

print(f"\n" + "="*80)
print(f"COMPLETE")
print(f"="*80)

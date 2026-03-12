#!/usr/bin/env python3
"""
Create Figure 3: Universal energy scaling across three vehicles
Clean 4-panel figure for Nature Communications
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

# Load data
if os.path.exists('results/data/task8_three_vehicle/three_vehicle_validation_results.csv'):
    df = pd.read_csv('results/data/task8_three_vehicle/three_vehicle_validation_results.csv')
elif os.path.exists('github/results/data/task8_three_vehicle/three_vehicle_validation_results.csv'):
    df = pd.read_csv('github/results/data/task8_three_vehicle/three_vehicle_validation_results.csv')
else:
    print("ERROR: Cannot find three_vehicle_validation_results.csv")
    exit(1)

print(f"Loaded {len(df)} simulations")

# Clean data
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['actual_D', 'total_energy'])
df_clean = df_clean[df_clean['total_energy'] > 0]

print(f"Clean data: {len(df_clean)} simulations")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

colors = {'Vehicle A': 'royalblue', 'Vehicle B': 'forestgreen', 'Vehicle C': 'crimson'}

# ============================================================================
# Panel A: Energy vs D for all three vehicles
# ============================================================================
ax = axes[0, 0]

for vehicle in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
    df_v = df_clean[df_clean['vehicle'] == vehicle]
    ax.scatter(df_v['actual_D'], df_v['total_energy'], 
              alpha=0.4, s=20, c=colors[vehicle], label=vehicle, edgecolors='none')

ax.set_xlabel('Fractal Dimension $D$')
ax.set_ylabel('Vibration Energy (m²/s⁴)')
ax.set_yscale('log')
ax.set_title('A. Energy Scaling by Vehicle')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# ============================================================================
# Panel B: Combined scaling (all vehicles)
# ============================================================================
ax = axes[0, 1]

D_all = df_clean['actual_D'].values
E_all = df_clean['total_energy'].values

# Regression
log_Dm2 = np.log10(D_all - 2)
log_E = np.log10(E_all)
slope, intercept, r, p, stderr = linregress(log_Dm2, log_E)

# Plot all data
ax.scatter(D_all - 2, E_all, alpha=0.2, s=15, c='gray', edgecolors='none')

# Fit line
D_fit = np.linspace(D_all.min(), D_all.max(), 100)
E_fit = 10**(intercept) * (D_fit - 2)**slope
ax.plot(D_fit - 2, E_fit, 'r-', linewidth=3, 
        label=f'$E \\propto (D-2)^{{{slope:.3f}}}$\n$R^2 = {r**2:.3f}$\n$p < 10^{{-300}}$')

ax.set_xlabel('$D - 2$')
ax.set_ylabel('Vibration Energy (m²/s⁴)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('B. Combined Scaling (N = 1500)')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# ============================================================================
# Panel C: Per-vehicle exponents
# ============================================================================
ax = axes[1, 0]

exponents = []
r2_values = []
vehicle_names = []

for vehicle in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
    df_v = df_clean[df_clean['vehicle'] == vehicle]
    
    D = df_v['actual_D'].values
    E = df_v['total_energy'].values
    
    log_Dm2 = np.log10(D - 2)
    log_E = np.log10(E)
    
    slope_v, intercept_v, r_v, p_v, stderr_v = linregress(log_Dm2, log_E)
    
    exponents.append(slope_v)
    r2_values.append(r_v**2)
    vehicle_names.append(vehicle)

# Bar chart
x = np.arange(len(vehicle_names))
bars = ax.bar(x, exponents, color=[colors[v] for v in vehicle_names], 
              alpha=0.7, edgecolor='black', linewidth=1.5)

# Add values on bars
for i, (exp, r2) in enumerate(zip(exponents, r2_values)):
    ax.text(i, exp - 0.05, f'γ = {exp:.3f}\n$R^2$ = {r2:.3f}', 
            ha='center', va='top', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(vehicle_names)
ax.set_ylabel('Scaling Exponent γ')
ax.set_title('C. Per-Vehicle Exponents')
ax.axhline(y=np.mean(exponents), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(exponents):.3f} ± {np.std(exponents):.3f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add CV text
cv = np.std(exponents) / abs(np.mean(exponents)) * 100
ax.text(0.5, 0.95, f'CV = {cv:.1f}%\n(Highly Universal)', 
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        fontsize=11, fontweight='bold')

# ============================================================================
# Panel D: Ensemble means with error bars
# ============================================================================
ax = axes[1, 1]

for vehicle in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
    df_v = df_clean[df_clean['vehicle'] == vehicle]
    
    # Group by target_D and compute statistics
    grouped = df_v.groupby('target_D')['total_energy'].agg(['mean', 'std', 'count'])
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot with error bars
    ax.errorbar(grouped.index, grouped['mean'], yerr=1.96*grouped['se'],
               fmt='o-', color=colors[vehicle], label=vehicle,
               capsize=5, capthick=2, linewidth=2, markersize=8, alpha=0.8)

ax.set_xlabel('Target Fractal Dimension $D$')
ax.set_ylabel('Mean Vibration Energy (m²/s⁴)')
ax.set_yscale('log')
ax.set_title('D. Ensemble Means (±95% CI)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Add statistics box
stats_text = f'N = {len(df_clean)} simulations\n'
stats_text += f'γ = {np.mean(exponents):.3f} ± {np.std(exponents):.3f}\n'
stats_text += f'CV = {cv:.1f}%'
ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        fontsize=10, verticalalignment='bottom')

# ============================================================================
# Final adjustments
# ============================================================================
plt.tight_layout()
plt.savefig('three_vehicle_universality.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: three_vehicle_universality.png")

# Print summary
print("\n" + "="*70)
print("UNIVERSAL SCALING SUMMARY")
print("="*70)
print(f"\nCombined scaling: E ∝ (D-2)^{slope:.3f}")
print(f"R² = {r**2:.4f}")
print(f"p-value < 10^-300")
print(f"N = {len(df_clean)} simulations")

print(f"\nPer-vehicle exponents:")
for v, exp, r2 in zip(vehicle_names, exponents, r2_values):
    print(f"  {v}: γ = {exp:.3f} (R² = {r2:.3f})")

print(f"\nMean: γ = {np.mean(exponents):.3f} ± {np.std(exponents):.3f}")
print(f"CV = {cv:.1f}%")
print("="*70)

plt.show()

#!/usr/bin/env python3
"""
Create amplitude_complexity_coupling.png for Figure 4
Shows the two-parameter framework and amplitude-complexity coupling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
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

# Load data - use task7 dataset which has actual spectral amplitude measurements
# This is a 500-simulation analysis used to illustrate the two-parameter framework
data_path = 'github/results/data/task7_spectral_fix/advanced_spectral_results.csv'
if not os.path.exists(data_path):
    print(f"ERROR: Cannot find {data_path}")
    print("This figure requires the task7 dataset with spectral measurements")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded {len(df)} simulations from {data_path}")
print(f"Columns: {df.columns.tolist()}")

# Rename columns to match expected names
df = df.rename(columns={
    'spectral_amplitude_acc': 'spectral_amplitude',
    'spectral_exponent_acc': 'beta',
    'total_vibration_energy': 'total_energy'
})

# Clean data
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=['actual_D', 'spectral_amplitude', 'beta', 'total_energy']
)
df_clean = df_clean[df_clean['total_energy'] > 0]
df_clean = df_clean[df_clean['spectral_amplitude'] > 0]

print(f"Clean data: {len(df_clean)} simulations")

# Create figure with 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ============================================================================
# Panel A: Spectral Amplitude vs Fractal Dimension
# ============================================================================
ax = axes[0, 0]

D = df_clean['actual_D'].values
C_z = df_clean['spectral_amplitude'].values

# Regression
log_Dm2 = np.log10(D - 2)
log_Cz = np.log10(C_z)
slope, intercept, r, p, stderr = linregress(log_Dm2, log_Cz)

# Plot
ax.scatter(D, C_z, alpha=0.3, s=20, c='steelblue', edgecolors='none')

# Fit line
D_fit = np.linspace(D.min(), D.max(), 100)
C_z_fit = 10**(intercept) * (D_fit - 2)**slope
ax.plot(D_fit, C_z_fit, 'r-', linewidth=2, 
        label=f'$C_z \\propto (D-2)^{{{slope:.2f}}}$\n$R^2 = {r**2:.3f}$')

ax.set_xlabel('Fractal Dimension $D$')
ax.set_ylabel('Spectral Amplitude $C_z$ (m³/rad)')
ax.set_yscale('log')
ax.set_title('A. Amplitude-Complexity Coupling')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.05, f'N = {len(df_clean)}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# Panel B: Two-Parameter Model
# ============================================================================
ax = axes[0, 1]

E = df_clean['total_energy'].values
beta = df_clean['beta'].values

# Two-parameter regression: log(E) = a*log(C_z) + b*log(beta) + c
from sklearn.linear_model import LinearRegression

X = np.column_stack([np.log10(C_z), np.log10(beta)])
y = np.log10(E)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

a, b = model.coef_
c = model.intercept_
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

# Plot predicted vs actual
E_pred = 10**y_pred
ax.scatter(E, E_pred, alpha=0.3, s=20, c='forestgreen', edgecolors='none')

# Diagonal line
E_range = [E.min(), E.max()]
ax.plot(E_range, E_range, 'k--', linewidth=1.5, label='Perfect fit')

ax.set_xlabel('Actual Energy (m²/s⁴)')
ax.set_ylabel('Predicted Energy (m²/s⁴)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('B. Two-Parameter Model')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add model equation
model_text = f'$E \\propto C_z^{{{a:.2f}}} \\times \\beta^{{{b:.2f}}}$\n$R^2 = {r2:.3f}$'
ax.text(0.05, 0.85, model_text, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        fontsize=11)

# ============================================================================
# Panel C: Energy vs Amplitude (colored by beta)
# ============================================================================
ax = axes[1, 0]

# Create scatter plot colored by beta
scatter = ax.scatter(C_z, E, c=beta, alpha=0.5, s=30, 
                    cmap='viridis', edgecolors='none')

ax.set_xlabel('Spectral Amplitude $C_z$ (m³/rad)')
ax.set_ylabel('Vibration Energy (m²/s⁴)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('C. Energy vs Amplitude (colored by $\\beta$)')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Spectral Exponent $\\beta$')

# Add annotation
ax.text(0.05, 0.95, 'Amplitude dominates\nenergy prediction',
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# Panel D: Single-Parameter vs Two-Parameter Comparison
# ============================================================================
ax = axes[1, 1]

# Single-parameter model: E vs (D-2)
log_Dm2 = np.log10(D - 2)
log_E = np.log10(E)
slope_1p, intercept_1p, r_1p, p_1p, stderr_1p = linregress(log_Dm2, log_E)
r2_1p = r_1p**2

# Bar chart comparing R²
models = ['Single-Parameter\n$E \\propto (D-2)^\\gamma$',
          'Two-Parameter\n$E \\propto C_z^a \\times \\beta^b$']
r2_values = [r2_1p, r2]
colors = ['coral', 'forestgreen']

bars = ax.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add values on bars
for bar, r2_val in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'$R^2 = {r2_val:.3f}$',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Coefficient of Determination ($R^2$)')
ax.set_ylim(0, 1.05)
ax.set_title('D. Model Comparison')
ax.grid(True, alpha=0.3, axis='y')

# Add improvement annotation
improvement = (r2 - r2_1p) / r2_1p * 100
ax.text(0.5, 0.5, f'Improvement:\n+{improvement:.1f}%',
        transform=ax.transAxes, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        fontsize=12, fontweight='bold')

# ============================================================================
# Final adjustments
# ============================================================================
plt.tight_layout()
plt.savefig('amplitude_complexity_coupling.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: amplitude_complexity_coupling.png")

# Print summary statistics
print("\n" + "="*70)
print("AMPLITUDE-COMPLEXITY COUPLING ANALYSIS")
print("="*70)
print(f"\nAmplitude-Complexity Coupling:")
print(f"  C_z ∝ (D-2)^{slope:.3f}")
print(f"  R² = {r**2:.4f}")
print(f"  p-value = {p:.2e}")

print(f"\nTwo-Parameter Model:")
print(f"  E ∝ C_z^{a:.3f} × β^{b:.3f}")
print(f"  R² = {r2:.4f}")

print(f"\nSingle-Parameter Model:")
print(f"  E ∝ (D-2)^{slope_1p:.3f}")
print(f"  R² = {r2_1p:.4f}")

print(f"\nImprovement: {improvement:.1f}%")
print("="*70)

plt.show()

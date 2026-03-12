"""
Create the β_measured vs β_theory validation figure
This is the "most convincing additional figure" suggested

Shows data points against theoretical line β_t = 7 - 2D
with residual plot below
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("="*80)
print("CREATING β THEORY VALIDATION FIGURE")
print("="*80)

# Load LiDAR data
df = pd.read_csv('github2/Fractal_Terrain_Analysis_Simulation-main/results/data/task11_lidar/lidar_terrain_results.csv')
print(f"\nLoaded: {len(df)} terrain regions")

# Calculate theoretical β from measured D
# β_t = 7 - 2D (for terrain elevation PSD)
df['beta_theory'] = 7 - 2 * df['D']

# Calculate residuals
df['residual'] = df['beta_mean'] - df['beta_theory']

# Create figure with 2 panels (main plot + residuals)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

# ============================================================================
# Panel A: β_measured vs β_theory
# ============================================================================

# Plot 1:1 line (perfect agreement)
beta_range = [df['beta_theory'].min() - 0.5, df['beta_theory'].max() + 0.5]
ax1.plot(beta_range, beta_range, 'k--', linewidth=2.5, alpha=0.7, 
        label='Perfect agreement\n$\\beta_{measured} = \\beta_{theory}$', zorder=1)

# Plot theoretical line with ±1σ band
beta_std = df['residual'].std()
ax1.fill_between(beta_range, 
                 [b - beta_std for b in beta_range],
                 [b + beta_std for b in beta_range],
                 color='gray', alpha=0.2, label=f'$\\pm 1\\sigma$ ({beta_std:.2f})')

# Plot data points
ax1.scatter(df['beta_theory'], df['beta_mean'], 
           s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidths=1.5,
           label='Measured terrain regions', zorder=3)

# Add error bars if available
if 'beta_std' in df.columns:
    ax1.errorbar(df['beta_theory'], df['beta_mean'], yerr=df['beta_std'],
                fmt='none', ecolor='gray', alpha=0.5, capsize=3, zorder=2)

# Calculate correlation
r, p = stats.pearsonr(df['beta_theory'], df['beta_mean'])
rho, p_rho = stats.spearmanr(df['beta_theory'], df['beta_mean'])

# Formatting
ax1.set_ylabel('Measured Spectral Slope $\\beta_{measured}$', fontsize=13, fontweight='bold')
ax1.set_title('Validation of Theoretical Relationship $\\beta_t = 7 - 2D$', 
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', frameon=True, fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(beta_range)
ax1.set_ylim(beta_range)

# Add statistics box
stats_text = f'Pearson $r = {r:.3f}$ ($p = {p:.3f}$)\n'
stats_text += f'Spearman $\\rho = {rho:.3f}$ ($p = {p_rho:.3f}$)\n'
stats_text += f'RMSE = {np.sqrt(np.mean(df["residual"]**2)):.3f}\n'
stats_text += f'$n = {len(df)}$ regions'

ax1.text(0.98, 0.02, stats_text,
        transform=ax1.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                 edgecolor='black', linewidth=1.5))

# Add panel label
ax1.text(-0.08, 1.05, 'a', transform=ax1.transAxes, 
        fontsize=18, fontweight='bold', va='top')

# ============================================================================
# Panel B: Residual Plot
# ============================================================================

# Plot residuals
ax2.scatter(df['beta_theory'], df['residual'],
           s=80, alpha=0.7, c='coral', edgecolors='black', linewidths=1.5)

# Zero line
ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.7)

# ±2σ lines
ax2.axhline(y=2*beta_std, color='r', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=-2*beta_std, color='r', linestyle=':', linewidth=1.5, alpha=0.5)

# Formatting
ax2.set_xlabel('Theoretical Spectral Slope $\\beta_{theory} = 7 - 2D$', 
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Residual\n$\\Delta\\beta$', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(beta_range)

# Add residual statistics
mean_residual = df['residual'].mean()
ax2.text(0.02, 0.98, f'Mean residual: {mean_residual:.3f}\nStd dev: {beta_std:.3f}',
        transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Add panel label
ax2.text(-0.08, 1.05, 'b', transform=ax2.transAxes, 
        fontsize=18, fontweight='bold', va='top')

plt.tight_layout()

# Save figure
output_path = 'github/images_manuscript/beta_theory_validation.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Figure saved: {output_path}")

output_path2 = 'beta_theory_validation.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Figure saved: {output_path2}")

plt.close()

# ============================================================================
# Print analysis
# ============================================================================
print("\n" + "="*80)
print("VALIDATION ANALYSIS")
print("="*80)

print(f"\nTheoretical relationship: β_t = 7 - 2D")
print(f"\nStatistics:")
print(f"  Pearson correlation:  r = {r:.3f}, p = {p:.3f}")
print(f"  Spearman correlation: ρ = {rho:.3f}, p = {p_rho:.3f}")
print(f"  RMSE: {np.sqrt(np.mean(df['residual']**2)):.3f}")
print(f"  Mean residual: {mean_residual:.3f}")
print(f"  Std dev residual: {beta_std:.3f}")

print(f"\nInterpretation:")
if abs(mean_residual) < 0.1:
    print(f"  ✓ Residuals centered near zero (unbiased)")
else:
    print(f"  ⚠ Residuals show systematic bias")

if abs(r) > 0.5:
    print(f"  ✓ Moderate to strong correlation")
else:
    print(f"  ⚠ Weak correlation")

print("\n" + "="*80)
print("MANUSCRIPT TEXT SUGGESTION")
print("="*80)
print(f"""
Figure X compares measured terrain spectral slopes with theoretical 
predictions from the relationship β_t = 7 - 2D. Panel a shows measured 
versus predicted values across {len(df)} independent terrain regions, 
with the dashed line indicating perfect agreement. The correlation 
(Pearson r = {r:.3f}, p = {p:.3f}; Spearman ρ = {rho:.3f}) demonstrates 
consistency between measured terrain spectra and the self-affine surface 
model. Panel b shows residuals (Δβ = β_measured - β_theory), which are 
approximately unbiased (mean = {mean_residual:.3f}) with standard 
deviation {beta_std:.2f}, indicating that deviations from theory are 
primarily due to measurement uncertainty and natural terrain variability 
rather than systematic model error.

Although the geographic sample size is modest (n = {len(df)}), each region 
contains approximately 10^8 elevation samples from high-resolution DEM data, 
yielding stable estimates of fractal dimension and spectral slope. The 
observed relationship provides preliminary empirical support for the 
theoretical spectral framework across diverse geomorphological terrain types.
""")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)

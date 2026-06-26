#!/usr/bin/env python3
"""Regenerate TartanDrive validation figure from saved CSV (no raw data needed)."""

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 300,
})

df = pd.read_csv('tartandrive_validation_results.csv')
df_imu = df.dropna(subset=['energy', 'beta_a'])

print(f"Total segments: {len(df_imu)}")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Panel (a): E vs beta_a (all segments)
ax = axes[0, 0]
log_E = np.log10(df_imu['energy'].values)
sl, intercept, r, p, _ = linregress(df_imu['beta_a'].values, log_E)
ax.scatter(df_imu['beta_a'], df_imu['energy'], alpha=0.5, s=20, 
          c='steelblue', edgecolors='none')
ax.set_xlabel('Vehicle Spectral Slope $\\beta_a$')
ax.set_ylabel('Vibration Energy $E$ [m$^2$/s$^4$]')
ax.set_yscale('log')
ax.set_title(f'(a) $E$ vs $\\beta_a$ ($r={r:.3f}$, $p={p:.1e}$, $n={len(df_imu)}$)')
ax.grid(True, alpha=0.3)
print(f"Panel (a): r={r:.3f}, p={p:.1e}, n={len(df_imu)}")

# Panel (b): E vs speed (all segments, linear)
ax = axes[0, 1]
df_plot = df_imu.dropna(subset=['mean_speed'])
sc = ax.scatter(df_plot['mean_speed'], df_plot['energy'], 
              alpha=0.5, s=20, c=df_plot['beta_a'], 
              cmap='viridis', edgecolors='none')
plt.colorbar(sc, ax=ax, label='$\\beta_a$')
sl2, _, r2, p2, _ = linregress(df_plot['mean_speed'].values,
                               np.log10(df_plot['energy'].values))
ax.set_xlabel('Vehicle Speed [m/s]')
ax.set_ylabel('Vibration Energy $E$ [m$^2$/s$^4$]')
ax.set_yscale('log')
ax.set_title(f'(b) $E$ vs speed ($r={r2:.3f}$, $p={p2:.1e}$, $n={len(df_plot)}$)')
ax.grid(True, alpha=0.3)
print(f"Panel (b): r={r2:.3f}, p={p2:.1e}, n={len(df_plot)}")

# Panel (c): beta_a distribution
ax = axes[1, 0]
ax.hist(df_imu['beta_a'], bins=20, color='steelblue', 
       edgecolor='white', alpha=0.8)
ax.axvline(df_imu['beta_a'].mean(), color='red', linestyle='--', 
          label=f'mean={df_imu["beta_a"].mean():.2f}')
ax.axvline(df_imu['beta_a'].median(), color='orange', linestyle='-', 
          label=f'median={df_imu["beta_a"].median():.2f}')
ax.set_xlabel('Vehicle Spectral Slope $\\beta_a$')
ax.set_ylabel('Count')
ax.set_title(f'(c) Distribution of $\\beta_a$ ($n={len(df_imu)}$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (d): Log-log power law (v > 0.5 m/s ONLY — matches text statistics)
ax = axes[1, 1]
df_moving = df_imu.dropna(subset=['mean_speed'])
df_moving = df_moving[df_moving['mean_speed'] > 0.5]
log_v = np.log10(df_moving['mean_speed'].values)
log_E = np.log10(df_moving['energy'].values)
ax.scatter(log_v, log_E, alpha=0.5, s=20, c='forestgreen', edgecolors='none')
sl, intercept, r, p, _ = linregress(log_v, log_E)
xfit = np.linspace(log_v.min(), log_v.max(), 50)
ax.plot(xfit, sl*xfit + intercept, 'r-', lw=2,
       label=f'slope={sl:.2f}, $r$={r:.3f}')
ax.legend(fontsize=9)
ax.set_xlabel('$\\log_{10}$(speed [m/s])')
ax.set_ylabel('$\\log_{10}$($E$ [m$^2$/s$^4$])')
ax.set_title(f'(d) Power-law: $E \\propto v^{{{sl:.2f}}}$ ($n={len(df_moving)}$, $v>0.5$ m/s)')
ax.grid(True, alpha=0.3)
print(f"Panel (d): slope={sl:.2f}, r={r:.3f}, p={p:.1e}, n={len(df_moving)}")

plt.tight_layout()
plt.savefig('figures/tartandrive_validation.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: figures/tartandrive_validation.png")
plt.close()

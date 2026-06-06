#!/usr/bin/env python3
"""
Generate corrected Figure 3 for revised manuscript.

Uses the ORIGINAL simulation data from the project repository:
  https://github.com/GeoGizmodo/fractalFatigueGeo

Data file: three_vehicle_validation_results.csv (1500 simulations)
Columns: vehicle, target_D, actual_D, beta, rms_acceleration, total_energy, rms_force, realization

This script:
  1. Downloads the original data CSV from the repo (if not present locally)
  2. Plots the corrected Figure 3:
     Panel (a): Vibration Energy E vs Fractal Dimension D (log-y scale)
     Panel (b): Variance Decomposition bar chart
  3. Reports the actual regression statistics from the data

The original manuscript Figure 3 panel (a) incorrectly displayed beta_a vs D.
The corrected version shows total_energy vs actual_D, which is the E-D regression
with slope ~ -3.18 and R² ~ 0.915.

IMPORTANT: If the computed statistics differ from the manuscript values,
update the manuscript to match the data — do NOT force the data to match
the manuscript.
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

# =============================================================================
# 1. DATA LOADING
# =============================================================================

DATA_URL = (
    "https://raw.githubusercontent.com/GeoGizmodo/fractalFatigueGeo/main/"
    "data/simulation_results/three_vehicle_validation_results.csv"
)
LOCAL_CSV = "three_vehicle_validation_results.csv"


def load_data():
    """Load the original 1500-simulation dataset."""
    if not os.path.exists(LOCAL_CSV):
        print(f"Downloading data from GitHub repo...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_CSV)
        print(f"  Saved to {LOCAL_CSV}")
    
    df = pd.read_csv(LOCAL_CSV)
    print(f"Loaded {len(df)} simulations")
    
    # Clean: remove inf/nan/zero energy (same as original script)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['actual_D', 'total_energy']
    )
    df_clean = df_clean[df_clean['total_energy'] > 0]
    print(f"After cleaning: {len(df_clean)} simulations")
    
    return df_clean


# =============================================================================
# 2. FIGURE GENERATION
# =============================================================================

def generate_figure3(df):
    """
    Generate 2-panel Figure 3:
      (a) Vibration Energy E vs Fractal Dimension D (log-y, colored by vehicle)
      (b) Variance Decomposition bar chart
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    colors = {
        'Vehicle A': '#1f77b4',  # blue
        'Vehicle B': '#ff7f0e',  # orange
        'Vehicle C': '#2ca02c',  # green
    }
    
    # -------------------------------------------------------------------------
    # Panel (a): Energy vs D
    # -------------------------------------------------------------------------
    ax = axes[0]
    
    D_all = df['actual_D'].values
    E_all = df['total_energy'].values
    
    # Scatter by vehicle
    for veh in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
        mask = df['vehicle'] == veh
        ax.scatter(
            df.loc[mask, 'actual_D'],
            df.loc[mask, 'total_energy'],
            c=colors[veh], alpha=0.35, s=18, label=veh, edgecolors='none'
        )
    
    # Combined power-law regression: log10(E) = slope * log10(D-2) + intercept
    log_Dm2 = np.log10(D_all - 2)
    log_E = np.log10(E_all)
    slope, intercept, r_val, p_val, stderr = linregress(log_Dm2, log_E)
    R2 = r_val**2
    
    # Regression line
    D_fit = np.linspace(D_all.min(), D_all.max(), 200)
    E_fit = 10**intercept * (D_fit - 2)**slope
    ax.plot(D_fit, E_fit, 'k--', linewidth=2.5,
            label=f'$E \\propto (D-2)^{{{slope:.2f}}}$\n$R^2 = {R2:.3f}$')
    
    ax.set_yscale('log')
    ax.set_xlabel('Terrain Fractal Dimension $D$')
    ax.set_ylabel('Vibration Energy $E$ (m²/s⁴)')
    ax.set_title('(a) Vibration Energy Scaling')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    # Statistics annotation
    ax.text(0.05, 0.05,
            f'Combined: slope = {slope:.3f}\n'
            f'$R^2$ = {R2:.3f}\n'
            f'$n$ = {len(df)}',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # -------------------------------------------------------------------------
    # Panel (b): Variance Decomposition
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Terrain D only
    sl_t, _, r_t, _, _ = linregress(D_all, log_E)
    R2_terrain = r_t**2
    
    # Vehicle only (encode as numeric)
    veh_map = {'Vehicle A': 0, 'Vehicle B': 1, 'Vehicle C': 2}
    veh_idx = df['vehicle'].map(veh_map).values.astype(float)
    sl_v, _, r_v, _, _ = linregress(veh_idx, log_E)
    R2_vehicle = r_v**2
    
    # Combined (D + Vehicle)
    X = np.column_stack([D_all, veh_idx, np.ones(len(D_all))])
    beta_hat, residuals, rank, sv = np.linalg.lstsq(X, log_E, rcond=None)
    SS_res = np.sum((log_E - X @ beta_hat)**2)
    SS_tot = np.sum((log_E - np.mean(log_E))**2)
    R2_combined = 1 - SS_res / SS_tot
    
    categories = ['Terrain $D$\nonly', 'Vehicle type\nonly', 'Combined\n($D$ + Vehicle)']
    values = [R2_terrain * 100, R2_vehicle * 100, R2_combined * 100]
    bar_colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    bars = ax2.bar(categories, values, color=bar_colors,
                   edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title('(b) Variance Decomposition')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11,
                 fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig('figures/Figure3.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/Figure3.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to figures/Figure3.png and figures/Figure3.pdf")
    plt.close()
    
    # -------------------------------------------------------------------------
    # Print all statistics for verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FIGURE 3 STATISTICS (verify against manuscript)")
    print("=" * 70)
    print(f"\nPanel (a) - Combined energy regression:")
    print(f"  E ∝ (D-2)^{slope:.3f}")
    print(f"  R² = {R2:.4f}")
    print(f"  r  = {r_val:.4f}")
    print(f"  p  < 10^{int(np.log10(p_val)) if p_val > 0 else -300}")
    print(f"  n  = {len(df)}")
    
    print(f"\nPer-vehicle energy regressions:")
    for veh in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
        mask = df['vehicle'] == veh
        D_v = df.loc[mask, 'actual_D'].values
        E_v = df.loc[mask, 'total_energy'].values
        sl_v, int_v, r_v, p_v, _ = linregress(np.log10(D_v - 2), np.log10(E_v))
        print(f"  {veh}: γ = {sl_v:.3f}, R² = {r_v**2:.3f}")
    
    print(f"\nPanel (b) - Variance decomposition:")
    print(f"  Terrain D only: R² = {R2_terrain:.4f} ({R2_terrain*100:.1f}%)")
    print(f"  Vehicle only:   R² = {R2_vehicle:.4f} ({R2_vehicle*100:.1f}%)")
    print(f"  Combined:       R² = {R2_combined:.4f} ({R2_combined*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("IMPORTANT: If these values differ from the manuscript, update the")
    print("manuscript text/captions to match these actual data-derived values.")
    print("=" * 70)


# =============================================================================
# 3. MAIN
# =============================================================================

if __name__ == '__main__':
    df = load_data()
    generate_figure3(df)

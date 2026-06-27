#!/usr/bin/env python3
"""
Create Source Data Excel file for Nature Communications submission.
Each sheet contains raw data underlying one figure/table.
"""
import pandas as pd
import numpy as np
from pathlib import Path

writer = pd.ExcelWriter('revision_submission/Source_Data.xlsx', engine='openpyxl')

# ========================================
# Figure 2: D vs beta_a (1500 simulations)
# ========================================
try:
    df = pd.read_csv('three_vehicle_validation_results.csv')
    df_fig2 = df[['fractal_dimension', 'spectral_exponent', 'vehicle']].copy()
    df_fig2.columns = ['Fractal_Dimension_D', 'Spectral_Exponent_beta_a', 'Vehicle']
    df_fig2.to_excel(writer, sheet_name='Fig2_D_vs_beta', index=False)
    print(f"Fig 2: {len(df_fig2)} rows")
except Exception as e:
    print(f"Fig 2 skipped: {e}")

# ========================================
# Figure 3a: Energy vs D (1500 simulations)
# ========================================
try:
    df = pd.read_csv('three_vehicle_validation_results.csv')
    df_fig3 = df[['fractal_dimension', 'vibration_energy', 'vehicle']].copy()
    df_fig3.columns = ['Fractal_Dimension_D', 'Vibration_Energy_E_m2s4', 'Vehicle']
    df_fig3.to_excel(writer, sheet_name='Fig3a_Energy_vs_D', index=False)
    print(f"Fig 3a: {len(df_fig3)} rows")
except Exception as e:
    print(f"Fig 3a skipped: {e}")

# ========================================
# Figure 3b: Variance decomposition (summary statistics)
# ========================================
df_var = pd.DataFrame({
    'Predictor': ['Terrain D only', 'Vehicle type only', 'Combined (D + Vehicle)'],
    'R_squared': [0.935, 0.019, 0.954],
    'Variance_Explained_pct': [93.5, 1.9, 95.4],
    'Method': ['Sequential linear regression log10(E) = a*D + b*V + c'] * 3
})
df_var.to_excel(writer, sheet_name='Fig3b_Variance_Decomp', index=False)
print("Fig 3b: variance decomposition (3 rows)")

# ========================================
# Figure 6a: USGS 25-region validation
# ========================================
try:
    df = pd.read_csv('expanded_terrain_validation.csv')
    df.to_excel(writer, sheet_name='Fig6a_USGS_25regions', index=False)
    print(f"Fig 6a: {len(df)} rows")
except Exception as e:
    print(f"Fig 6a skipped: {e}")

# ========================================
# Supp Figure 4: Constant-amplitude decoupling
# ========================================
try:
    df = pd.read_csv('constant_amplitude_results.csv')
    df.to_excel(writer, sheet_name='SuppFig4_Decoupling', index=False)
    print(f"Supp Fig 4: {len(df)} rows")
except Exception as e:
    print(f"Supp Fig 4 skipped: {e}")

# ========================================
# Supp Figure 5: TartanDrive validation
# ========================================
try:
    df = pd.read_csv('tartandrive_validation_results.csv')
    df.to_excel(writer, sheet_name='SuppFig5_TartanDrive', index=False)
    print(f"Supp Fig 5: {len(df)} rows")
except Exception as e:
    print(f"Supp Fig 5 skipped: {e}")

# ========================================
# Table 1: Variance decomposition
# ========================================
df_tab1 = pd.DataFrame({
    'Predictor': ['Terrain D', 'Vehicle type', 'Combined'],
    'R_squared': [0.935, 0.019, 0.954],
    'Variance_Explained_pct': [93.5, 1.9, 95.4],
    'n': [1500, 1500, 1500],
    'Note': ['Sequential regression, terrain factor enters first'] * 3
})
df_tab1.to_excel(writer, sheet_name='Table1_Variance', index=False)

# ========================================
# Table 2: ISO 8608 comparison (qualitative)
# ========================================
df_tab2 = pd.DataFrame({
    'Feature': ['Input requirement', 'Domain', 'Parameters', 'R_squared'],
    'ISO_8608': ['Profilometer measurement', 'Paved roads (Classes A-H)', 
                 'G_d(n_0), w≈2 fixed', '0.91 (amplitude only)'],
    'Fractal_Framework': ['DEM (remote sensing)', 'Natural terrain',
                          'C_z, beta (both fitted)', '0.96 (two-parameter)']
})
df_tab2.to_excel(writer, sheet_name='Table2_ISO_Comparison', index=False)

writer.close()
print("\nSource Data saved: revision_submission/Source_Data.xlsx")

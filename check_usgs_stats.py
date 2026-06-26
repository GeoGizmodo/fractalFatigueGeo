#!/usr/bin/env python3
"""Check the actual statistics from the 25-region USGS CSV."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress

df = pd.read_csv('expanded_terrain_validation.csv')
D1 = df['D1_mean'].values
beta = df['beta_mean'].values

rp, pp = pearsonr(D1, beta)
rho, ps = spearmanr(D1, beta)
sl, ic, rv, pv, _ = linregress(D1, beta)

print(f"n = {len(df)}")
print(f"D1 range: {D1.min():.3f} - {D1.max():.3f}")
print(f"beta range: {beta.min():.3f} - {beta.max():.3f}")
print(f"")
print(f"Pearson:  r = {rp:.4f}, p = {pp:.4f}")
print(f"Spearman: rho = {rho:.4f}, p = {ps:.4f}")
print(f"OLS: slope = {sl:.3f}, intercept = {ic:.3f}, R2 = {rv**2:.4f}")
print()
print("="*60)
print("MANUSCRIPT CLAIMS: rho = -0.402, p = 0.046")
print(f"ACTUAL FROM CSV:   rho = {rho:.3f}, p = {ps:.4f}")
print()
if abs(rho - (-0.402)) < 0.01 and abs(ps - 0.046) < 0.01:
    print("MATCH: Figure and text are consistent.")
else:
    print("MISMATCH: Figure and text DISAGREE.")
    print()
    print("Theory predicts NEGATIVE correlation (higher D1 -> lower beta)")
    print(f"Actual sign: {'NEGATIVE' if rho < 0 else 'POSITIVE'}")
    print()
    print("DECISION NEEDED:")
    print("  Option A: If an earlier run genuinely gave rho=-0.402,")
    print("            find that cached data or re-run with same params.")
    print("  Option B: If THIS is the reproducible result,")
    print("            update ALL 7 manuscript occurrences to match.")
    print()
    print("The data shows D1 values are mostly 1.3-1.6 (not the expected")
    print("1.0-1.2 range from the old 13-tile analysis). This suggests the")
    print("variogram-based D1 estimation gives different values than the")
    print("box-counting method used in the original 13-tile analysis.")

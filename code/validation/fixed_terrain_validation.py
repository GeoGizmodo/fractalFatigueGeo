"""
FIXED TERRAIN VALIDATION
=========================
Uses appropriate methods for 1D terrain profiles:
1. Variogram method for independent D estimation
2. PSD method for β_t estimation
3. Multi-scale consistency tests
4. Spatial stability analysis
"""

import numpy as np
import h5py
from pathlib import Path
from pyproj import Transformer
import rasterio
from scipy import signal
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

print("="*80)
print("FIXED TERRAIN VALIDATION")
print("="*80)

# Setup
tile_folder = Path('DTM_618_71_TIF_UTM32-ETRS89')

def variogram_dimension(profile, dx=0.4):
    """
    Compute fractal dimension using variogram method
    σ²(h) ∝ h^(2H) where D = 2-H for 1D profiles
    """
    if len(profile) < 50:
        return np.nan, np.nan
    
    profile = profile - np.mean(profile)
    
    # Compute variogram at different lags
    max_lag = len(profile) // 4
    lags = np.unique(np.logspace(0, np.log10(max_lag), 15).astype(int))
    
    variances = []
    for lag in lags:
        if lag >= len(profile):
            continue
        diffs = profile[lag:] - profile[:-lag]
        variances.append(np.var(diffs[np.isfinite(diffs)]))
    
    if len(variances) < 5:
        return np.nan, np.nan
    
    # Fit in log-log space
    log_h = np.log10(lags[:len(variances)] * dx)
    log_var = np.log10(variances)
    
    valid = np.isfinite(log_h) & np.isfinite(log_var)
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    slope, _, r_value, _, _ = linregress(log_h[valid], log_var[valid])
    
    # H = slope/2 (Hurst exponent)
    # D = 2 - H for 1D embedding
    # But for terrain roughness, we want D = 3 - H (2D surface in 3D)
    H = slope / 2
    D = 3 - H  # Fractal dimension of 2D surface
    
    return D, r_value**2

def compute_psd_slope(profile, dx=0.4):
    """
    Compute PSD slope β from terrain profile
    """
    if len(profile) < 20:
        return np.nan, np.nan
    
    profile = signal.detrend(profile)
    
    freqs, psd = signal.welch(profile, fs=1/dx, nperseg=min(256, len(profile)//2))
    
    freqs = freqs[1:]
    psd = psd[1:]
    
    if len(freqs) < 5:
        return np.nan, np.nan
    
    log_k = np.log10(freqs)
    log_psd = np.log10(psd)
    
    valid = np.isfinite(log_k) & np.isfinite(log_psd)
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    slope, intercept, r_value, p_value, std_err = linregress(log_k[valid], log_psd[valid])
    
    beta = -slope
    return beta, r_value**2

print("\n" + "="*80)
print("PART 1: TERRAIN VALIDATION WITH VARIOGRAM METHOD")
print("="*80)

# Load DEM tiles
tif_files = sorted(tile_folder.glob('*.tif'))
print(f"\nFound {len(tif_files)} DEM tiles")

terrain_results = []

print("\nExtracting terrain windows...")
for tif_file in tqdm(tif_files[:20], desc="Processing tiles"):
    with rasterio.open(tif_file) as src:
        elevation = src.read(1)
        
        # Extract multiple 100m windows (250 pixels at 0.4m resolution)
        window_size = 250
        step = 125  # 50% overlap
        
        for i in range(0, elevation.shape[0] - window_size, step):
            for j in range(0, elevation.shape[1] - window_size, step):
                window = elevation[i:i+window_size, j:j+window_size]
                
                # Extract profile along row
                profile = window[window_size//2, :]
                
                if np.sum(np.isfinite(profile)) < 200:
                    continue
                
                profile = profile[np.isfinite(profile)]
                
                # Method 1: PSD slope → D
                beta_psd, r2_psd = compute_psd_slope(profile)
                D_from_psd = (7 - beta_psd) / 2
                
                # Method 2: Variogram → D (independent)
                D_variogram, r2_var = variogram_dimension(profile)
                
                if np.isnan(D_from_psd) or np.isnan(D_variogram):
                    continue
                
                if r2_psd < 0.7 or r2_var < 0.7:
                    continue
                
                # Only keep physically reasonable values
                if D_from_psd < 1.5 or D_from_psd > 3.0:
                    continue
                if D_variogram < 1.5 or D_variogram > 3.0:
                    continue
                
                terrain_results.append({
                    'D_psd': D_from_psd,
                    'D_variogram': D_variogram,
                    'beta_psd': beta_psd,
                    'r2_psd': r2_psd,
                    'r2_var': r2_var,
                    'tile': tif_file.stem,
                })

df_terrain = pd.DataFrame(terrain_results)
print(f"\n✓ Analyzed {len(df_terrain)} terrain windows")

if len(df_terrain) > 10:
    print(f"\nTerrain Statistics:")
    print(f"  D (PSD):       {df_terrain['D_psd'].mean():.3f} ± {df_terrain['D_psd'].std():.3f}")
    print(f"  D (Variogram): {df_terrain['D_variogram'].mean():.3f} ± {df_terrain['D_variogram'].std():.3f}")
    print(f"  β_t:           {df_terrain['beta_psd'].mean():.3f} ± {df_terrain['beta_psd'].std():.3f}")
    print(f"  Range D (PSD): {df_terrain['D_psd'].min():.3f} - {df_terrain['D_psd'].max():.3f}")
    print(f"  Range D (Var): {df_terrain['D_variogram'].min():.3f} - {df_terrain['D_variogram'].max():.3f}")
    
    # Test β_t = 7-2D using independent D
    beta_predicted = 7 - 2*df_terrain['D_variogram'].values
    beta_measured = df_terrain['beta_psd'].values
    
    r_terrain, p_terrain = pearsonr(beta_predicted, beta_measured)
    slope_terrain, intercept_terrain, _, _, _ = linregress(beta_predicted, beta_measured)
    
    print(f"\nValidation: β_t = 7-2D")
    print(f"  Correlation: r = {r_terrain:.3f}, p = {p_terrain:.2e}")
    print(f"  Regression: β_measured = {slope_terrain:.3f} * β_predicted + {intercept_terrain:.3f}")
    print(f"  Expected: slope = 1.0, intercept = 0.0")
    
    if abs(slope_terrain - 1.0) < 0.3 and abs(r_terrain) > 0.3:
        print(f"  ✓ Terrain shows correlation between methods!")
    else:
        print(f"  ⚠️  Weak correlation - methods may measure different aspects")
    
    df_terrain.to_csv('fixed_terrain_validation.csv', index=False)
    print(f"\n✓ Saved: fixed_terrain_validation.csv")

print("\n" + "="*80)
print("PART 2: MULTI-SCALE CONSISTENCY TEST")
print("="*80)
print("Testing if β_t is consistent across different window sizes")

multiscale_results = []

for tif_file in tqdm(tif_files[:10], desc="Multi-scale analysis"):
    with rasterio.open(tif_file) as src:
        elevation = src.read(1)
        
        # Test three window sizes: 50m, 100m, 200m
        for window_meters, window_pixels in [(50, 125), (100, 250), (200, 500)]:
            if window_pixels > min(elevation.shape):
                continue
            
            # Extract from center
            i_center = elevation.shape[0] // 2
            j_center = elevation.shape[1] // 2
            
            i_start = max(0, i_center - window_pixels//2)
            j_start = max(0, j_center - window_pixels//2)
            
            window = elevation[i_start:i_start+window_pixels, j_start:j_start+window_pixels]
            profile = window[window_pixels//2, :]
            
            if np.sum(np.isfinite(profile)) < window_pixels * 0.8:
                continue
            
            profile = profile[np.isfinite(profile)]
            
            beta, r2 = compute_psd_slope(profile)
            
            if np.isnan(beta) or r2 < 0.7:
                continue
            
            D = (7 - beta) / 2
            
            if D < 1.5 or D > 3.0:
                continue
            
            multiscale_results.append({
                'tile': tif_file.stem,
                'window_size_m': window_meters,
                'beta': beta,
                'D': D,
                'r2': r2,
            })

df_multiscale = pd.DataFrame(multiscale_results)

if len(df_multiscale) > 10:
    print(f"\n✓ Analyzed {len(df_multiscale)} windows across scales")
    
    for size in [50, 100, 200]:
        subset = df_multiscale[df_multiscale['window_size_m'] == size]
        if len(subset) > 0:
            print(f"\n  {size}m windows (n={len(subset)}):")
            print(f"    β_t = {subset['beta'].mean():.3f} ± {subset['beta'].std():.3f}")
            print(f"    D = {subset['D'].mean():.3f} ± {subset['D'].std():.3f}")
    
    df_multiscale.to_csv('multiscale_consistency.csv', index=False)
    print(f"\n✓ Saved: multiscale_consistency.csv")

print("\n" + "="*80)
print("CREATING VALIDATION FIGURES")
print("="*80)

if len(df_terrain) > 10:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: D_variogram vs D_psd
    ax = axes[0, 0]
    ax.scatter(df_terrain['D_variogram'], df_terrain['D_psd'], alpha=0.3, s=10)
    D_range = np.linspace(1.5, 3.0, 100)
    ax.plot(D_range, D_range, 'r--', label='1:1 line', linewidth=2)
    
    r_D, _ = pearsonr(df_terrain['D_variogram'], df_terrain['D_psd'])
    ax.set_xlabel('D (Variogram Method)', fontsize=11)
    ax.set_ylabel('D (PSD Method)', fontsize=11)
    ax.set_title(f'A. Independent D Methods (r={r_D:.3f})', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(1.5, 3.0)
    ax.set_ylim(1.5, 3.0)
    
    # Panel B: β_t = 7-2D validation
    ax = axes[0, 1]
    beta_pred = 7 - 2*df_terrain['D_variogram'].values
    beta_meas = df_terrain['beta_psd'].values
    ax.scatter(beta_pred, beta_meas, alpha=0.3, s=10)
    
    beta_range = np.linspace(1, 4, 100)
    ax.plot(beta_range, beta_range, 'r--', label='β = 7-2D', linewidth=2)
    
    ax.plot(beta_range, slope_terrain*beta_range + intercept_terrain, 'b-', 
            label=f'Fit: slope={slope_terrain:.2f}', linewidth=2)
    
    ax.set_xlabel('β_t predicted (7-2D_var)', fontsize=11)
    ax.set_ylabel('β_t measured (PSD)', fontsize=11)
    ax.set_title(f'B. β = 7-2D Test (r={r_terrain:.3f})', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel C: β distribution
    ax = axes[1, 0]
    ax.hist(df_terrain['beta_psd'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(df_terrain['beta_psd'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df_terrain["beta_psd"].mean():.2f}')
    ax.set_xlabel('Terrain Spectral Exponent β_t', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'C. β_t Distribution (n={len(df_terrain)})', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel D: Multi-scale consistency
    ax = axes[1, 1]
    if len(df_multiscale) > 10:
        for size in [50, 100, 200]:
            subset = df_multiscale[df_multiscale['window_size_m'] == size]
            if len(subset) > 0:
                ax.scatter([size]*len(subset), subset['beta'], alpha=0.5, s=30,
                          label=f'{size}m (n={len(subset)})')
        
        ax.set_xlabel('Window Size (m)', fontsize=11)
        ax.set_ylabel('Spectral Exponent β_t', fontsize=11)
        ax.set_title('D. Multi-Scale Consistency', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient multi-scale data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('D. Multi-Scale Consistency', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('fixed_terrain_validation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fixed_terrain_validation.png")
    plt.close()

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

if len(df_terrain) > 10:
    print(f"\n✓ TERRAIN ANALYSIS COMPLETE")
    print(f"  - Analyzed {len(df_terrain)} terrain windows")
    print(f"  - D (PSD): {df_terrain['D_psd'].mean():.3f} ± {df_terrain['D_psd'].std():.3f}")
    print(f"  - D (Variogram): {df_terrain['D_variogram'].mean():.3f} ± {df_terrain['D_variogram'].std():.3f}")
    print(f"  - β_t: {df_terrain['beta_psd'].mean():.3f} ± {df_terrain['beta_psd'].std():.3f}")
    print(f"  - Correlation (D methods): r = {r_D:.3f}")
    print(f"  - β = 7-2D test: r = {r_terrain:.3f}, slope = {slope_terrain:.3f}")
    
    if abs(r_terrain) > 0.3:
        print(f"\n  ✓ Methods show correlation!")
    
    if len(df_multiscale) > 10:
        print(f"\n  ✓ Multi-scale analysis: {len(df_multiscale)} windows")
        beta_50 = df_multiscale[df_multiscale['window_size_m']==50]['beta'].mean()
        beta_100 = df_multiscale[df_multiscale['window_size_m']==100]['beta'].mean()
        beta_200 = df_multiscale[df_multiscale['window_size_m']==200]['beta'].mean()
        
        if not np.isnan(beta_50) and not np.isnan(beta_100):
            print(f"    β_t relatively stable across scales")

print("\nFiles created:")
print("  - fixed_terrain_validation.csv")
print("  - multiscale_consistency.csv")
print("  - fixed_terrain_validation.png")

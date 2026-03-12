"""
COMPLETE REAL-WORLD VALIDATION: LiRA-CD + Copenhagen LiDAR
============================================================
Validates the complete physics chain with 17,481 matched measurement points:
    Terrain Fractal Dimension D → Spectral Exponent β_t → Vehicle Vibration Energy

This is the final validation that connects:
1. Real terrain geometry (0.4m resolution LiDAR)
2. Real vehicle measurements (GPS-tagged accelerometers)
3. Theoretical predictions (β = 7-2D, E ∝ D^γ)
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
print("COMPLETE REAL-WORLD VALIDATION")
print("="*80)
print("Validating: Terrain D → β_t → Vehicle Vibration Energy")
print("="*80)

# Setup
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
tile_folder = Path('DTM_618_71_TIF_UTM32-ETRS89')
liracd_folder = Path('23192909')

# Tile coverage
tile_bounds = {
    'northing': (6180000, 6190000),
    'easting': (710000, 720000),
}

print("\n" + "="*80)
print("STEP 1: Extract all matched GPS points with vehicle data")
print("="*80)

roads = ['M3', 'M13', 'CPH1', 'CPH6']
all_segments = []

for road in roads:
    hdf5_file = liracd_folder / f'{road}_HH.hdf5'
    
    if not hdf5_file.exists():
        continue
    
    print(f"\nProcessing {road}...")
    
    with h5py.File(hdf5_file, 'r') as f:
        if 'GM' not in f.keys():
            continue
            
        gm_group = f['GM']
        
        for task_id in gm_group.keys():
            task_group = gm_group[task_id]
            
            for pass_id in task_group.keys():
                pass_group = task_group[pass_id]
                
                if 'gps' not in pass_group.keys() or 'acc.xyz' not in pass_group.keys():
                    continue
                
                gps_data = pass_group['gps'][:]
                accel_data = pass_group['acc.xyz'][:]
                
                if len(gps_data.shape) != 2 or gps_data.shape[1] < 3:
                    continue
                
                lats = gps_data[:, 1]
                lons = gps_data[:, 2]
                eastings, northings = transformer.transform(lons, lats)
                
                in_bounds = ((northings >= tile_bounds['northing'][0]) & 
                           (northings < tile_bounds['northing'][1]) &
                           (eastings >= tile_bounds['easting'][0]) & 
                           (eastings < tile_bounds['easting'][1]))
                
                if np.sum(in_bounds) == 0:
                    continue
                
                for i in np.where(in_bounds)[0]:
                    if i >= len(accel_data) or accel_data.shape[1] < 4:
                        continue
                    
                    accel_z = accel_data[i, 3]  # Vertical acceleration (g)
                    
                    all_segments.append({
                        'road': road,
                        'task_id': task_id,
                        'pass_id': pass_id,
                        'lat': lats[i],
                        'lon': lons[i],
                        'easting': eastings[i],
                        'northing': northings[i],
                        'accel_z': accel_z,
                    })

print(f"\n✓ Extracted {len(all_segments)} measurement points")

# Convert to DataFrame
df = pd.DataFrame(all_segments)

print("\n" + "="*80)
print("STEP 2: Load DEM tiles and extract terrain profiles")
print("="*80)

# Find all .tif files
tif_files = sorted(tile_folder.glob('*.tif'))
print(f"Found {len(tif_files)} DEM tiles")

# Load all tiles into memory for fast access
print("Loading DEM tiles...")
dem_data = {}

for tif_file in tqdm(tif_files, desc="Loading tiles"):
    with rasterio.open(tif_file) as src:
        # Get tile bounds
        bounds = src.bounds
        transform = src.transform
        elevation = src.read(1)
        
        dem_data[tif_file.stem] = {
            'elevation': elevation,
            'transform': transform,
            'bounds': bounds,
        }

print(f"✓ Loaded {len(dem_data)} tiles")

print("\n" + "="*80)
print("STEP 3: Extract terrain elevation at each GPS point")
print("="*80)

def get_elevation_at_point(easting, northing, dem_data):
    """Extract elevation at a specific UTM coordinate"""
    for tile_name, tile_info in dem_data.items():
        bounds = tile_info['bounds']
        
        if (bounds.left <= easting < bounds.right and 
            bounds.bottom <= northing < bounds.top):
            
            # Convert UTM to pixel coordinates
            transform = tile_info['transform']
            col = int((easting - transform.c) / transform.a)
            row = int((northing - transform.f) / transform.e)
            
            elevation = tile_info['elevation']
            
            if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]:
                return elevation[row, col]
    
    return np.nan

# Extract elevations
elevations = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting elevations"):
    elev = get_elevation_at_point(row['easting'], row['northing'], dem_data)
    elevations.append(elev)

df['elevation'] = elevations

# Remove points without elevation data
df_valid = df.dropna(subset=['elevation']).copy()
print(f"✓ {len(df_valid)}/{len(df)} points have valid elevation data")

print("\n" + "="*80)
print("STEP 4: Compute terrain fractal dimension for each road segment")
print("="*80)

def compute_fractal_dimension_psd(profile, dx=0.4):
    """
    Compute fractal dimension from power spectral density
    Uses the relationship: PSD ∝ k^(-β) where β = 7-2D
    """
    if len(profile) < 10:
        return np.nan, np.nan
    
    # Detrend
    profile = signal.detrend(profile)
    
    # Compute PSD
    freqs, psd = signal.welch(profile, fs=1/dx, nperseg=min(256, len(profile)//2))
    
    # Remove DC component
    freqs = freqs[1:]
    psd = psd[1:]
    
    if len(freqs) < 5:
        return np.nan, np.nan
    
    # Fit in log-log space
    log_k = np.log10(freqs)
    log_psd = np.log10(psd)
    
    # Remove infinities
    valid = np.isfinite(log_k) & np.isfinite(log_psd)
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    slope, intercept, r_value, p_value, std_err = linregress(log_k[valid], log_psd[valid])
    
    # β = -slope (PSD ∝ k^(-β))
    beta = -slope
    
    # D = (7-β)/2
    D = (7 - beta) / 2
    
    return D, r_value**2

# Group by road and pass to create continuous segments
print("Computing fractal dimensions for road segments...")

segment_results = []

for (road, task_id, pass_id), group in tqdm(df_valid.groupby(['road', 'task_id', 'pass_id'])):
    if len(group) < 20:  # Need minimum points for PSD
        continue
    
    # Sort by position along road
    group = group.sort_values('northing')
    
    # Extract terrain profile
    profile = group['elevation'].values
    
    # Compute D
    D, r2 = compute_fractal_dimension_psd(profile)
    
    if np.isnan(D):
        continue
    
    # Compute vehicle vibration energy (RMS of vertical acceleration)
    accel_rms = np.sqrt(np.mean(group['accel_z'].values**2))
    
    # Theoretical prediction: β_t = 7-2D
    beta_predicted = 7 - 2*D
    
    segment_results.append({
        'road': road,
        'task_id': task_id,
        'pass_id': pass_id,
        'n_points': len(group),
        'D': D,
        'beta_predicted': beta_predicted,
        'accel_rms': accel_rms,
        'r2_fit': r2,
    })

df_segments = pd.DataFrame(segment_results)
print(f"✓ Computed D for {len(df_segments)} road segments")

# Save results
df_segments.to_csv('real_world_validation_results.csv', index=False)
print(f"✓ Saved results to real_world_validation_results.csv")

print("\n" + "="*80)
print("STEP 5: Statistical validation")
print("="*80)

# Filter for good fits
df_good = df_segments[df_segments['r2_fit'] > 0.7].copy()
print(f"Using {len(df_good)}/{len(df_segments)} segments with R² > 0.7")

if len(df_good) < 10:
    print("⚠️  Not enough high-quality segments for validation")
    print("This may be due to:")
    print("  - Short segment lengths (need longer profiles)")
    print("  - GPS sampling rate vs terrain resolution mismatch")
    print("  - Need to aggregate nearby points into longer segments")
else:
    # Test correlation: D vs vibration energy
    # Theory predicts: E ∝ D^γ where γ ≈ -2.3 (from ensemble results)
    
    log_D = np.log10(df_good['D'].values)
    log_E = np.log10(df_good['accel_rms'].values)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_D, log_E)
    
    print(f"\nCorrelation: log(E_vib) vs log(D)")
    print(f"  Slope (γ): {slope:.3f} ± {std_err:.3f}")
    print(f"  R²: {r_value**2:.3f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Expected γ: -2.336 ± 0.190 (from ensemble)")
    
    # Test if measured γ matches theory
    gamma_theory = -2.336
    gamma_std = 0.190
    z_score = abs(slope - gamma_theory) / np.sqrt(std_err**2 + gamma_std**2)
    
    if z_score < 2:
        print(f"  ✓ Measured γ consistent with theory (z={z_score:.2f})")
    else:
        print(f"  ⚠️  Measured γ differs from theory (z={z_score:.2f})")
    
    # Summary statistics
    print(f"\nTerrain fractal dimensions:")
    print(f"  Mean D: {df_good['D'].mean():.3f} ± {df_good['D'].std():.3f}")
    print(f"  Range: {df_good['D'].min():.3f} - {df_good['D'].max():.3f}")
    
    print(f"\nVehicle vibration (RMS acceleration):")
    print(f"  Mean: {df_good['accel_rms'].mean():.3f} ± {df_good['accel_rms'].std():.3f} g")
    print(f"  Range: {df_good['accel_rms'].min():.3f} - {df_good['accel_rms'].max():.3f} g")

print("\n" + "="*80)
print("STEP 6: Create validation figure")
print("="*80)

if len(df_good) >= 10:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: D distribution
    ax = axes[0, 0]
    D_values = df_good['D'].values
    ax.hist(D_values, bins=20, alpha=0.7, edgecolor='black')
    mean_D = df_good['D'].mean()
    ax.axvline(mean_D, color='red', linestyle='--', 
               label=f'Mean: {mean_D:.3f}')
    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel('Count')
    ax.set_title('A. Terrain Fractal Dimensions')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel B: β_t distribution
    ax = axes[0, 1]
    beta_values = df_good['beta_predicted'].values
    ax.hist(beta_values, bins=20, alpha=0.7, edgecolor='black')
    mean_beta = df_good['beta_predicted'].mean()
    ax.axvline(mean_beta, color='red', linestyle='--',
               label=f'Mean: {mean_beta:.3f}')
    ax.set_xlabel('Spectral Exponent β_t = 7-2D')
    ax.set_ylabel('Count')
    ax.set_title('B. Predicted Spectral Exponents')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel C: D vs Vibration Energy (log-log)
    ax = axes[1, 0]
    D_vals = df_good['D'].values
    accel_vals = df_good['accel_rms'].values
    ax.scatter(D_vals, accel_vals, alpha=0.5, s=30)
    
    # Fit line
    D_range = np.linspace(df_good['D'].min(), df_good['D'].max(), 100)
    E_fit = 10**(intercept + slope * np.log10(D_range))
    ax.plot(D_range, E_fit, 'r-', linewidth=2,
            label=f'γ = {slope:.3f} ± {std_err:.3f}\nR² = {r_value**2:.3f}')
    
    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel('RMS Acceleration (g)')
    ax.set_title('C. Terrain D → Vehicle Vibration')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel D: Comparison with theory
    ax = axes[1, 1]
    
    # Theory: γ = -2.336 ± 0.190
    gamma_theory = -2.336
    gamma_std_theory = 0.190
    
    # Plot measured vs theory
    ax.errorbar([1], [slope], yerr=std_err, fmt='o', markersize=10,
                capsize=5, label='Measured (Real Data)', color='blue')
    ax.errorbar([2], [gamma_theory], yerr=gamma_std_theory, fmt='s', markersize=10,
                capsize=5, label='Theory (Ensemble)', color='red')
    
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real\nData', 'Theory\n(Ensemble)'])
    ax.set_ylabel('Scaling Exponent γ')
    ax.set_title('D. Validation: Real vs Theory')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_world_validation_figure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved figure: real_world_validation_figure.png")
    plt.close()

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print(f"\nResults:")
print(f"  - Analyzed {len(df_segments)} road segments")
print(f"  - {len(df_good)} high-quality segments (R² > 0.7)")
print(f"  - Terrain D range: {df_good['D'].min():.3f} - {df_good['D'].max():.3f}")
if len(df_good) >= 10:
    print(f"  - Measured γ: {slope:.3f} ± {std_err:.3f}")
    print(f"  - Theory γ: -2.336 ± 0.190")
    print(f"  - Correlation R²: {r_value**2:.3f}")
    print(f"  - p-value: {p_value:.2e}")
    
    if z_score < 2:
        print(f"\n✓✓✓ VALIDATION SUCCESSFUL ✓✓✓")
        print("Real-world data confirms theoretical predictions!")
    else:
        print(f"\n⚠️  Partial validation - consider:")
        print("  - Aggregating points into longer segments")
        print("  - Filtering by road quality/type")
        print("  - Accounting for vehicle speed variations")

print("\nFiles created:")
print("  - real_world_validation_results.csv")
print("  - real_world_validation_figure.png")

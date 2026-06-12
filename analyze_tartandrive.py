#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TartanDrive Off-Road Vehicle Validation
========================================

PURPOSE:
  Validate the terrain-vibration spectral framework using real off-road
  vehicle data from the TartanDrive dataset (CMU, 2022).

DATASET:
  TartanDrive: ~200,000 off-road driving interactions
  Vehicle: Yamaha Viking ATV on natural terrain
  Sensors: IMU (200 Hz), GPS/Odom (50 Hz), Heightmaps (501x501, 20 Hz)
  Speed: up to 15 m/s (matches our simulation speed)
  Source: https://github.com/castacks/tartan_drive
  Paper: Triest et al., ICRA 2022

WHAT THIS SCRIPT DOES:
  For each trajectory segment:
  1. Extract vertical (z-axis) acceleration from IMU at 200 Hz
  2. Compute vibration energy E = a_rms^2 (after high-pass filtering)
  3. Compute acceleration PSD spectral slope beta_a (Welch, 0.5-25 Hz)
  4. Extract terrain heightmap along trajectory
  5. Compute terrain PSD spectral slope beta_t
  6. Compute terrain fractal dimension D (1D box-counting on profile)
  7. Correlate: beta_t vs D, E vs beta_t, E vs D

PREREQUISITES:
  1. Download TartanDrive data:
     python download_files.py --download-dir ./tartandrive_data
     (from https://github.com/castacks/tartan_drive)

  2. Convert rosbags to torch format:
     python multi_convert_bag.py --bag_dir ./tartandrive_data \
       --save_to ./tartandrive_torch --use_stamps True --torch True

  3. Or: extract IMU + heightmap data directly from rosbags using
     the extraction functions below.

USAGE:
  python analyze_tartandrive.py --data_dir ./tartandrive_torch

OUTPUT:
  - tartandrive_validation_results.csv
  - figures/tartandrive_validation.png
  - Console statistics for manuscript insertion
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 300,
})


# =============================================================================
# 1. IMU PROCESSING
# =============================================================================

def compute_vibration_energy(imu_z, fs=200.0, highpass_hz=0.5):
    """
    Compute vibration energy from vertical acceleration.
    
    Parameters:
        imu_z: vertical acceleration time series (m/s^2)
        fs: sampling frequency (200 Hz for TartanDrive IMU)
        highpass_hz: high-pass filter cutoff to remove gravity/DC
    
    Returns:
        energy: a_rms^2 (m^2/s^4)
        beta_a: spectral slope of acceleration PSD
    """
    # High-pass filter to remove gravity component
    sos = signal.butter(4, highpass_hz, btype='high', fs=fs, output='sos')
    imu_filtered = signal.sosfilt(sos, imu_z)
    
    # Skip transient from filter (first 1 second)
    skip = int(fs)
    if len(imu_filtered) <= 2 * skip:
        return np.nan, np.nan
    
    imu_steady = imu_filtered[skip:]
    
    # RMS acceleration and energy
    energy = np.mean(imu_steady**2)
    
    # PSD and spectral slope
    nperseg = min(1024, len(imu_steady) // 4)
    if nperseg < 64:
        return energy, np.nan
    
    freqs, psd = signal.welch(imu_steady, fs=fs, nperseg=nperseg,
                              window='hann', noverlap=nperseg//2)
    
    # Fit spectral slope in 0.5-25 Hz range
    mask = (freqs >= 0.5) & (freqs <= 25.0) & (psd > 0)
    if np.sum(mask) < 5:
        return energy, np.nan
    
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _, r, _, _ = linregress(log_f, log_psd)
    
    beta_a = -slope  # convention: positive beta means PSD decreases
    return energy, beta_a


# =============================================================================
# 2. TERRAIN PROCESSING
# =============================================================================

def compute_terrain_spectral_slope(heightmap_profile, pixel_size_m=0.1):
    """
    Compute spectral slope of terrain profile.
    
    Parameters:
        heightmap_profile: 1D elevation profile (meters)
        pixel_size_m: spatial resolution (meters/pixel)
            TartanDrive heightmaps are 501x501 covering local area;
            resolution depends on map configuration
    
    Returns:
        beta_t: terrain PSD spectral slope
        r2: goodness of fit
    """
    if len(heightmap_profile) < 64:
        return np.nan, 0.0
    
    # Remove mean and linear trend
    profile = heightmap_profile - np.mean(heightmap_profile)
    x = np.arange(len(profile))
    slope_lin, intercept_lin = np.polyfit(x, profile, 1)
    profile = profile - (slope_lin * x + intercept_lin)
    
    # Spatial frequency
    fs_spatial = 1.0 / pixel_size_m  # samples per meter
    nperseg = min(256, len(profile) // 4)
    if nperseg < 32:
        return np.nan, 0.0
    
    freqs, psd = signal.welch(profile, fs=fs_spatial, nperseg=nperseg,
                              window='hann', noverlap=nperseg//2)
    
    # Fit in relevant spatial frequency range
    # For vehicle at 15 m/s: 0.5 Hz temporal = 0.033 cycles/m spatial
    #                        25 Hz temporal = 1.67 cycles/m spatial
    mask = (freqs >= 0.01) & (freqs <= 2.0) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan, 0.0
    
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _, r, _, _ = linregress(log_f, log_psd)
    
    beta_t = -slope
    return beta_t, r**2


def compute_fractal_dimension_1d(profile):
    """
    Compute 1D fractal dimension using box-counting.
    
    Parameters:
        profile: 1D elevation profile
    
    Returns:
        D1: 1D fractal dimension (range 1.0 to 2.0)
    """
    if len(profile) < 32:
        return np.nan
    
    # Normalize to [0, 1]
    pmin, pmax = np.min(profile), np.max(profile)
    if pmax == pmin:
        return 1.0
    profile_norm = (profile - pmin) / (pmax - pmin)
    
    n = len(profile_norm)
    box_sizes = [s for s in [2, 4, 8, 16, 32, 64] if s < n // 2]
    
    if len(box_sizes) < 3:
        return np.nan
    
    counts = []
    for eps in box_sizes:
        count = 0
        for i in range(0, n - eps, eps):
            segment = profile_norm[i:i+eps]
            h_range = np.max(segment) - np.min(segment)
            h_boxes = max(1, int(np.ceil(h_range / (eps / n))))
            count += h_boxes
        counts.append(count)
    
    # Regression
    log_eps = np.log(1.0 / np.array(box_sizes))
    log_n = np.log(np.array(counts))
    
    slope, _, r, _, _ = linregress(log_eps, log_n)
    D1 = slope
    
    # Clip to valid range
    D1 = np.clip(D1, 1.0, 2.0)
    return D1


# =============================================================================
# 3. MAIN ANALYSIS (for torch-format trajectories)
# =============================================================================

def analyze_trajectory(traj_data):
    """
    Analyze a single TartanDrive trajectory.
    
    Parameters:
        traj_data: dict with keys 'imu', 'heightmap', 'state', etc.
            imu: shape (T, 6) -- [ax, ay, az, gx, gy, gz] at 200 Hz
            heightmap: shape (T_map, 501, 501) at 20 Hz
            state: shape (T_state, 7) at 50 Hz
    
    Returns:
        dict with computed metrics
    """
    results = {}
    
    # --- IMU analysis ---
    if 'imu' in traj_data and traj_data['imu'] is not None:
        imu = traj_data['imu']
        # z-axis acceleration (index 2 in [ax, ay, az, gx, gy, gz])
        imu_z = imu[:, 2] if imu.ndim == 2 else imu
        
        energy, beta_a = compute_vibration_energy(imu_z, fs=200.0)
        results['energy'] = energy
        results['beta_a'] = beta_a
        results['rms_acc'] = np.sqrt(energy) if energy > 0 else np.nan
    
    # --- Terrain analysis ---
    if 'heightmap' in traj_data and traj_data['heightmap'] is not None:
        hmap = traj_data['heightmap']
        
        # Use middle row of last heightmap as terrain profile
        if hmap.ndim == 3:
            # (T, H, W) -- use mean of several frames for stability
            n_frames = min(5, hmap.shape[0])
            avg_hmap = np.mean(hmap[-n_frames:], axis=0)
        else:
            avg_hmap = hmap
        
        # Extract middle-row profile
        mid = avg_hmap.shape[0] // 2
        profile = avg_hmap[mid, :]
        
        # Terrain spectral slope
        # TartanDrive local heightmap: 501 pixels, ~50m coverage = 0.1 m/pixel
        beta_t, beta_r2 = compute_terrain_spectral_slope(profile, pixel_size_m=0.1)
        results['beta_t'] = beta_t
        results['beta_t_r2'] = beta_r2
        
        # Fractal dimension
        D1 = compute_fractal_dimension_1d(profile)
        results['D1'] = D1
    
    # --- Speed ---
    if 'state' in traj_data and traj_data['state'] is not None:
        state = traj_data['state']
        # state is typically [x, y, z, qx, qy, qz, qw] or [x,y,z,vx,vy,vz,yaw]
        # Compute speed from position differences if velocity not available
        if state.shape[1] >= 6:
            vx = state[:, 3] if state.shape[1] >= 7 else np.diff(state[:, 0]) * 50
            vy = state[:, 4] if state.shape[1] >= 7 else np.diff(state[:, 1]) * 50
            speed = np.sqrt(vx**2 + vy**2)
            results['mean_speed'] = np.mean(speed)
    
    return results


# =============================================================================
# 4. BATCH PROCESSING
# =============================================================================

def process_all_trajectories(data_dir):
    """
    Process all TartanDrive trajectories in a directory.
    
    Parameters:
        data_dir: path to directory of .pt or .npy trajectory files
    """
    import torch
    
    results = []
    files = sorted([f for f in os.listdir(data_dir) 
                    if f.endswith('.pt') or f.endswith('.npy')])
    
    print(f"Found {len(files)} trajectory files in {data_dir}")
    
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        
        try:
            if fname.endswith('.pt'):
                traj = torch.load(fpath, map_location='cpu')
                # Convert torch tensors to numpy
                traj_np = {}
                for key, val in traj.items():
                    if hasattr(val, 'numpy'):
                        traj_np[key] = val.numpy()
                    else:
                        traj_np[key] = val
            else:
                traj_np = np.load(fpath, allow_pickle=True).item()
            
            result = analyze_trajectory(traj_np)
            result['file'] = fname
            results.append(result)
            
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(files)} trajectories...")
    
    return pd.DataFrame(results)


# =============================================================================
# 5. RESULTS AND FIGURE
# =============================================================================

def analyze_results(df):
    """Compute correlations and generate figure."""
    
    print("\n" + "=" * 70)
    print("TARTANDRIVE VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTotal trajectories analyzed: {len(df)}")
    
    # Filter valid data
    df_valid = df.dropna(subset=['energy', 'beta_a', 'beta_t', 'D1'])
    print(f"Valid (all metrics): {len(df_valid)}")
    
    if len(df_valid) < 10:
        print("ERROR: Not enough valid data points for analysis.")
        return
    
    # --- Key correlations ---
    # 1. beta_t vs D1 (terrain: does spectral slope relate to fractal dim?)
    sl1, _, r1, p1, _ = linregress(df_valid['D1'], df_valid['beta_t'])
    print(f"\n1. Terrain: beta_t vs D1")
    print(f"   slope = {sl1:.3f}, r = {r1:.4f}, p = {p1:.2e}, n = {len(df_valid)}")
    
    # 2. Energy vs beta_t (vibration: does spectral slope predict energy?)
    log_E = np.log10(df_valid['energy'].values)
    sl2, _, r2, p2, _ = linregress(df_valid['beta_t'], log_E)
    print(f"\n2. Vibration: log10(E) vs beta_t")
    print(f"   slope = {sl2:.3f}, r = {r2:.4f}, p = {p2:.2e}")
    
    # 3. Energy vs D1 (combined chain)
    sl3, _, r3, p3, _ = linregress(df_valid['D1'], log_E)
    print(f"\n3. Combined: log10(E) vs D1")
    print(f"   slope = {sl3:.3f}, r = {r3:.4f}, p = {p3:.2e}")
    
    # 4. beta_a vs beta_t (spectral transfer)
    sl4, _, r4, p4, _ = linregress(df_valid['beta_t'], df_valid['beta_a'])
    print(f"\n4. Spectral: beta_a vs beta_t")
    print(f"   slope = {sl4:.3f}, r = {r4:.4f}, p = {p4:.2e}")
    
    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.scatter(df_valid['D1'], df_valid['beta_t'], alpha=0.3, s=10)
    ax.set_xlabel('Terrain Fractal Dimension D1')
    ax.set_ylabel('Terrain Spectral Slope beta_t')
    ax.set_title(f'(a) beta_t vs D1 (r={r1:.3f}, p={p1:.1e})')
    ax.grid(True, alpha=0.2)
    
    ax = axes[0, 1]
    ax.scatter(df_valid['beta_t'], df_valid['energy'], alpha=0.3, s=10)
    ax.set_xlabel('Terrain Spectral Slope beta_t')
    ax.set_ylabel('Vibration Energy E (m2/s4)')
    ax.set_yscale('log')
    ax.set_title(f'(b) E vs beta_t (r={r2:.3f}, p={p2:.1e})')
    ax.grid(True, alpha=0.2)
    
    ax = axes[1, 0]
    ax.scatter(df_valid['D1'], df_valid['energy'], alpha=0.3, s=10)
    ax.set_xlabel('Terrain Fractal Dimension D1')
    ax.set_ylabel('Vibration Energy E (m2/s4)')
    ax.set_yscale('log')
    ax.set_title(f'(c) E vs D1 (r={r3:.3f}, p={p3:.1e})')
    ax.grid(True, alpha=0.2)
    
    ax = axes[1, 1]
    ax.scatter(df_valid['beta_t'], df_valid['beta_a'], alpha=0.3, s=10)
    ax.set_xlabel('Terrain Spectral Slope beta_t')
    ax.set_ylabel('Vehicle Accel Spectral Slope beta_a')
    ax.set_title(f'(d) beta_a vs beta_t (r={r4:.3f}, p={p4:.1e})')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/tartandrive_validation.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: figures/tartandrive_validation.png")
    plt.close()
    
    # Save results
    df_valid.to_csv('tartandrive_validation_results.csv', index=False)
    print(f"Data saved: tartandrive_validation_results.csv")
    
    print("\n" + "=" * 70)
    print("MANUSCRIPT TEXT (if correlations are significant):")
    print("=" * 70)
    print(f"""
To validate the framework on real off-road vehicle data, we analyzed
trajectories from the TartanDrive dataset (Triest et al., ICRA 2022):
a Yamaha Viking ATV traversing diverse natural terrain at speeds up to
15 m/s with IMU data at 200 Hz and terrain heightmaps.

Across {len(df_valid)} valid trajectory segments:
- Terrain beta_t vs D1: r = {r1:.3f} (p = {p1:.1e})
- Vibration energy vs beta_t: r = {r2:.3f} (p = {p2:.1e})
- Vibration energy vs D1: r = {r3:.3f} (p = {p3:.1e})
- Vehicle beta_a vs terrain beta_t: r = {r4:.3f} (p = {p4:.1e})
""")


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze TartanDrive data for terrain-vibration validation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to TartanDrive torch trajectory directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Directory not found: {args.data_dir}")
        print("\nTo download TartanDrive data:")
        print("  git clone https://github.com/castacks/tartan_drive")
        print("  cd tartan_drive")
        print("  python download_files.py --download-dir ./data")
        print("\nThen convert to torch format:")
        print("  python multi_convert_bag.py --bag_dir ./data \\")
        print("    --save_to ./torch_data --use_stamps True --torch True")
        sys.exit(1)
    
    df = process_all_trajectories(args.data_dir)
    
    if len(df) > 0:
        analyze_results(df)
    else:
        print("No trajectories processed successfully.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TartanDrive Extraction & Analysis — STREAM + DELETE MODE
=========================================================

Processes bags ONE AT A TIME from the tar.gz archive:
  1. Extract one .bag file
  2. Analyze it (IMU → energy, β_a; heightmap → β_t, D)
  3. DELETE the .bag file immediately
  4. Move to next bag

This keeps disk usage to ~1 bag at a time (max ~13 GB).

USAGE:
    pip install rosbags numpy scipy pandas matplotlib
    python extract_and_analyze_tartandrive.py --tar_file tartan_drive/data/20210826_heightmaps_1.tar.gz --max_bags 50

OUTPUT:
    - tartandrive_validation_results.csv (lightweight, always saved)
    - figures/tartandrive_validation.png (at the end)
"""

import os
import sys
import tarfile
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'figure.dpi': 300,
})


# =============================================================================
# ROSBAG READING (using `rosbags` — pure Python, no ROS needed)
# =============================================================================

def read_rosbag_data(bag_path):
    """
    Read IMU and heightmap data from a rosbag using the `rosbags` library.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    
    data = {
        'imu_z': [],
        'imu_timestamps': [],
        'heightmaps': [],
        'hmap_timestamps': [],
        'speed': [],
    }
    
    try:
        with Reader(bag_path) as reader:
            # List all topics for diagnostics (first bag only)
            topics = set(c.topic for c in reader.connections)
            
            imu_conns = [c for c in reader.connections 
                        if c.topic == '/multisense/imu/imu_data']
            odom_conns = [c for c in reader.connections 
                         if c.topic == '/odometry/filtered_odom']
            hmap_conns = [c for c in reader.connections 
                         if 'height' in c.topic.lower() or 'map' in c.topic.lower()]
            
            # Read IMU
            for conn, timestamp, rawdata in reader.messages(connections=imu_conns):
                try:
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                    data['imu_z'].append(msg.linear_acceleration.z)
                    data['imu_timestamps'].append(timestamp)
                except Exception:
                    continue
            
            # Read odometry for speed
            for conn, timestamp, rawdata in reader.messages(connections=odom_conns):
                try:
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    data['speed'].append(np.sqrt(vx**2 + vy**2))
                except Exception:
                    continue
            
            # Try heightmaps (may fail for custom msg types)
            hmap_count = 0
            for conn, timestamp, rawdata in reader.messages(connections=hmap_conns):
                hmap_count += 1
                if hmap_count % 10 != 0:  # subsample heavily
                    continue
                try:
                    msg = deserialize_cdr(
                        ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                    # Try various GridMap data structures
                    if hasattr(msg, 'data') and len(msg.data) > 0:
                        for layer in msg.data:
                            if hasattr(layer, 'data'):
                                grid_data = np.array(layer.data, dtype=np.float32)
                                side = int(np.sqrt(len(grid_data)))
                                if side > 10 and side * side == len(grid_data):
                                    hmap = grid_data.reshape(side, side)
                                    hmap[hmap > 1e5] = np.nan
                                    if np.sum(np.isfinite(hmap)) > 100:
                                        data['heightmaps'].append(hmap)
                                        data['hmap_timestamps'].append(timestamp)
                                    break
                except Exception:
                    continue
    
    except Exception as e:
        print(f"    Reader error: {e}")
        return None, set()
    
    data['imu_z'] = np.array(data['imu_z'])
    data['speed'] = np.array(data['speed'])
    
    return data, topics


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def compute_vibration_energy(imu_z, fs=200.0, highpass_hz=0.5):
    """Compute vibration energy and spectral slope from z-acceleration."""
    if len(imu_z) < 400:
        return np.nan, np.nan
    
    sos = signal.butter(4, highpass_hz, btype='high', fs=fs, output='sos')
    imu_filtered = signal.sosfilt(sos, imu_z)
    
    skip = int(fs)
    imu_steady = imu_filtered[skip:]
    if len(imu_steady) < 200:
        return np.nan, np.nan
    
    energy = np.mean(imu_steady**2)
    
    nperseg = min(1024, len(imu_steady) // 4)
    if nperseg < 64:
        return energy, np.nan
    
    freqs, psd = signal.welch(imu_steady, fs=fs, nperseg=nperseg,
                              window='hann', noverlap=nperseg//2)
    
    mask = (freqs >= 0.5) & (freqs <= 25.0) & (psd > 0)
    if np.sum(mask) < 5:
        return energy, np.nan
    
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _, r, _, _ = linregress(log_f, log_psd)
    
    beta_a = -slope
    return energy, beta_a


def compute_terrain_spectral_slope(profile, pixel_size_m=0.1):
    """Compute spectral slope of a 1D terrain profile."""
    if len(profile) < 64:
        return np.nan, 0.0
    
    valid = np.isfinite(profile)
    if np.sum(valid) < 64:
        return np.nan, 0.0
    profile = profile[valid]
    
    profile = profile - np.mean(profile)
    x = np.arange(len(profile))
    slope_lin, intercept = np.polyfit(x, profile, 1)
    profile = profile - (slope_lin * x + intercept)
    
    fs_spatial = 1.0 / pixel_size_m
    nperseg = min(256, len(profile) // 4)
    if nperseg < 32:
        return np.nan, 0.0
    
    freqs, psd = signal.welch(profile, fs=fs_spatial, nperseg=nperseg,
                              window='hann', noverlap=nperseg//2)
    
    mask = (freqs >= 0.01) & (freqs <= 2.0) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan, 0.0
    
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, _, r, _, _ = linregress(log_f, log_psd)
    
    return -slope, r**2


def compute_fractal_dimension_1d(profile):
    """1D box-counting fractal dimension."""
    valid = np.isfinite(profile)
    if np.sum(valid) < 32:
        return np.nan
    profile = profile[valid]
    
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
    
    log_eps = np.log(1.0 / np.array(box_sizes))
    log_n = np.log(np.array(counts))
    slope, _, r, _, _ = linregress(log_eps, log_n)
    
    return np.clip(slope, 1.0, 2.0)


# =============================================================================
# SEGMENT ANALYSIS
# =============================================================================

def analyze_bag_segments(data, segment_duration_s=10.0, fs_imu=200.0):
    """Split bag data into fixed-duration segments and analyze each."""
    results = []
    imu_z = data['imu_z']
    
    if len(imu_z) < int(segment_duration_s * fs_imu):
        # Too short — one segment
        energy, beta_a = compute_vibration_energy(imu_z, fs=fs_imu)
        result = {'energy': energy, 'beta_a': beta_a}
        if len(data['speed']) > 0:
            result['mean_speed'] = np.mean(data['speed'])
        if len(data.get('heightmaps', [])) > 0:
            hmap = data['heightmaps'][0]
            mid = hmap.shape[0] // 2
            profile = hmap[mid, :]
            beta_t, r2 = compute_terrain_spectral_slope(profile)
            D1 = compute_fractal_dimension_1d(profile)
            result['beta_t'] = beta_t
            result['D1'] = D1
        results.append(result)
        return results
    
    samples_per_seg = int(segment_duration_s * fs_imu)
    n_segments = len(imu_z) // samples_per_seg
    
    hmaps = data.get('heightmaps', [])
    hmaps_per_seg = max(1, len(hmaps) // max(1, n_segments))
    
    for seg_idx in range(n_segments):
        start = seg_idx * samples_per_seg
        end = start + samples_per_seg
        
        imu_segment = imu_z[start:end]
        energy, beta_a = compute_vibration_energy(imu_segment, fs=fs_imu)
        
        result = {'energy': energy, 'beta_a': beta_a, 'segment': seg_idx}
        
        # Speed
        if len(data['speed']) > 0:
            n_speed = len(data['speed'])
            sp_start = seg_idx * (n_speed // max(1, n_segments))
            sp_end = min(sp_start + n_speed // max(1, n_segments), n_speed)
            if sp_end > sp_start:
                result['mean_speed'] = np.mean(data['speed'][sp_start:sp_end])
        
        # Heightmap
        if len(hmaps) > 0:
            hmap_idx = min(seg_idx * hmaps_per_seg + hmaps_per_seg // 2, 
                          len(hmaps) - 1)
            hmap = hmaps[hmap_idx]
            mid = hmap.shape[0] // 2
            profile = hmap[mid, :]
            beta_t, r2 = compute_terrain_spectral_slope(profile)
            D1 = compute_fractal_dimension_1d(profile)
            result['beta_t'] = beta_t
            result['D1'] = D1
        
        results.append(result)
    
    return results


# =============================================================================
# STREAM-AND-DELETE PROCESSING
# =============================================================================

def stream_process_tar(tar_path, max_bags=50, output_dir='_tmp_bag'):
    """
    Stream bags from tar.gz one at a time:
      extract → analyze → delete → next
    
    Keeps only ~1 bag on disk at a time.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    bag_count = 0
    all_topics = set()
    
    print(f"\nStreaming from: {tar_path}")
    print(f"Max bags to process: {max_bags}")
    print(f"Temp directory: {output_dir}")
    print("-" * 60)
    
    try:
        with tarfile.open(tar_path, 'r|gz') as tar:
            for member in tar:
                if not member.name.endswith('.bag') or not member.isfile():
                    continue
                
                if bag_count >= max_bags:
                    print(f"\n  Reached max_bags limit ({max_bags}). Stopping.")
                    break
                
                bag_name = os.path.basename(member.name)
                bag_size_mb = member.size / 1e6
                
                print(f"\n[{bag_count+1}/{max_bags}] {bag_name} "
                      f"({bag_size_mb:.0f} MB)")
                
                # EXTRACT
                try:
                    tar.extract(member, output_dir)
                    bag_path = os.path.join(output_dir, member.name)
                except OSError as e:
                    if "No space" in str(e):
                        print(f"  DISK FULL — stopping extraction.")
                        break
                    print(f"  Extract error: {e}. Skipping.")
                    continue
                except Exception as e:
                    print(f"  Extract error: {e}. Skipping.")
                    continue
                
                # ANALYZE
                try:
                    data, topics = read_rosbag_data(bag_path)
                    all_topics.update(topics)
                    
                    if data is None or len(data['imu_z']) == 0:
                        print(f"  No IMU data found. Skipping.")
                    else:
                        n_imu = len(data['imu_z'])
                        n_hmap = len(data.get('heightmaps', []))
                        duration = n_imu / 200.0
                        
                        print(f"  IMU: {n_imu} samples ({duration:.1f}s), "
                              f"Heightmaps: {n_hmap}, "
                              f"Speed: {np.mean(data['speed']):.2f} m/s avg")
                        
                        segments = analyze_bag_segments(data)
                        for seg in segments:
                            seg['bag_file'] = bag_name
                        all_results.extend(segments)
                        
                        print(f"  → {len(segments)} segments extracted")
                        bag_count += 1
                
                except Exception as e:
                    print(f"  Analysis error: {e}")
                
                # DELETE immediately
                try:
                    os.remove(bag_path)
                except Exception:
                    pass
                
                # Save intermediate results every 5 bags
                if bag_count % 5 == 0 and all_results:
                    pd.DataFrame(all_results).to_csv(
                        'tartandrive_validation_results.csv', index=False)
                    print(f"  [Checkpoint: {len(all_results)} segments saved]")
    
    except EOFError:
        print(f"\n  Archive truncated at bag {bag_count}. "
              f"Processing {len(all_results)} segments collected so far.")
    except Exception as e:
        print(f"\n  Archive error: {e}. "
              f"Processing {len(all_results)} segments collected so far.")
    
    # Cleanup temp dir
    try:
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Bags processed: {bag_count}")
    print(f"  Total segments: {len(all_results)}")
    print(f"  Topics found: {sorted(all_topics)}")
    print(f"{'='*60}")
    
    return pd.DataFrame(all_results)


# =============================================================================
# RESULTS & FIGURE
# =============================================================================

def generate_results(df):
    """Compute correlations and generate validation figure."""
    
    print("\n" + "=" * 70)
    print("TARTANDRIVE OFF-ROAD VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTotal segments analyzed: {len(df)}")
    
    # Basic stats
    df_imu = df.dropna(subset=['energy', 'beta_a'])
    print(f"Valid IMU (energy + β_a): {len(df_imu)}")
    
    if 'mean_speed' in df.columns:
        speeds = df['mean_speed'].dropna()
        if len(speeds) > 0:
            print(f"Speed range: {speeds.min():.2f} – {speeds.max():.2f} m/s "
                  f"(mean {speeds.mean():.2f} m/s)")
    
    has_terrain = 'beta_t' in df.columns and df['beta_t'].notna().sum() > 5
    
    print(f"\nIMU statistics:")
    print(f"  Energy range: {df_imu['energy'].min():.4f} – "
          f"{df_imu['energy'].max():.4f} m²/s⁴")
    print(f"  β_a range: {df_imu['beta_a'].min():.2f} – "
          f"{df_imu['beta_a'].max():.2f}")
    print(f"  RMS acc: {np.sqrt(df_imu['energy'].min()):.3f} – "
          f"{np.sqrt(df_imu['energy'].max()):.3f} m/s²")
    
    # --- Correlations ---
    print("\n" + "-" * 50)
    print("KEY CORRELATIONS")
    print("-" * 50)
    
    results_summary = {}
    
    # E vs beta_a
    if len(df_imu) >= 10:
        log_E = np.log10(df_imu['energy'].values)
        sl, intercept, r, p, se = linregress(df_imu['beta_a'].values, log_E)
        print(f"\n1. log₁₀(E) vs β_a:")
        print(f"   slope = {sl:.3f} ± {se:.3f}, r = {r:.4f}, "
              f"p = {p:.2e}, n = {len(df_imu)}")
        results_summary['E_vs_beta_a'] = {'r': r, 'p': p, 'slope': sl, 
                                           'n': len(df_imu)}
    
    # E vs speed
    if 'mean_speed' in df_imu.columns:
        df_speed = df_imu.dropna(subset=['mean_speed'])
        # Filter out near-zero speeds (stationary)
        df_moving = df_speed[df_speed['mean_speed'] > 0.5]
        if len(df_moving) >= 10:
            log_E = np.log10(df_moving['energy'].values)
            log_v = np.log10(df_moving['mean_speed'].values)
            sl, intercept, r, p, se = linregress(log_v, log_E)
            print(f"\n2. log₁₀(E) vs log₁₀(speed) [moving only, v > 0.5 m/s]:")
            print(f"   slope = {sl:.3f} ± {se:.3f}, r = {r:.4f}, "
                  f"p = {p:.2e}, n = {len(df_moving)}")
            print(f"   Theory predicts slope = 6-2D (for D~2.2: slope≈1.6)")
            results_summary['E_vs_speed'] = {'r': r, 'p': p, 'slope': sl,
                                             'n': len(df_moving)}
        
        # Also linear E vs speed
        if len(df_speed) >= 10:
            sl2, _, r2, p2, _ = linregress(df_speed['mean_speed'].values,
                                           np.log10(df_speed['energy'].values))
            print(f"\n3. log₁₀(E) vs speed (linear, all segments):")
            print(f"   slope = {sl2:.3f}, r = {r2:.4f}, p = {p2:.2e}, "
                  f"n = {len(df_speed)}")
            results_summary['E_vs_speed_linear'] = {'r': r2, 'p': p2, 
                                                     'slope': sl2, 'n': len(df_speed)}
    
    # beta_a vs speed
    if 'mean_speed' in df_imu.columns:
        df_moving = df_imu[df_imu['mean_speed'] > 0.5].dropna(subset=['mean_speed'])
        if len(df_moving) >= 10:
            sl, _, r, p, _ = linregress(df_moving['mean_speed'].values,
                                        df_moving['beta_a'].values)
            print(f"\n4. β_a vs speed [v > 0.5 m/s]:")
            print(f"   slope = {sl:.3f}, r = {r:.4f}, p = {p:.2e}, "
                  f"n = {len(df_moving)}")
            results_summary['beta_a_vs_speed'] = {'r': r, 'p': p, 'n': len(df_moving)}
    
    # Terrain correlations (if heightmaps available)
    if has_terrain:
        df_full = df.dropna(subset=['energy', 'beta_a', 'beta_t', 'D1'])
        print(f"\n  [Terrain data available: {len(df_full)} full segments]")
        
        if len(df_full) >= 5:
            sl, _, r, p, _ = linregress(df_full['D1'], df_full['beta_t'])
            print(f"\n5. β_t vs D:")
            print(f"   slope = {sl:.3f}, r = {r:.4f}, p = {p:.2e}")
            results_summary['beta_t_vs_D'] = {'r': r, 'p': p, 'n': len(df_full)}
            
            log_E = np.log10(df_full['energy'].values)
            sl, _, r, p, _ = linregress(df_full['D1'], log_E)
            print(f"\n6. log₁₀(E) vs D:")
            print(f"   slope = {sl:.3f}, r = {r:.4f}, p = {p:.2e}")
            results_summary['E_vs_D'] = {'r': r, 'p': p, 'n': len(df_full)}
    
    # --- FIGURE ---
    print("\nGenerating figure...")
    
    try:
        if has_terrain and df['beta_t'].notna().sum() >= 10:
            fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        
        # Panel (a): E vs beta_a
        ax = axes[0, 0]
        ax.scatter(df_imu['beta_a'], df_imu['energy'], alpha=0.5, s=20, 
                  c='steelblue', edgecolors='none')
        if 'E_vs_beta_a' in results_summary:
            r_val = results_summary['E_vs_beta_a']['r']
            p_val = results_summary['E_vs_beta_a']['p']
            ax.set_title(f'(a) $E$ vs $\\beta_a$ ($r={r_val:.3f}$, '
                        f'$p={p_val:.1e}$, $n={len(df_imu)}$)')
        ax.set_xlabel('Vehicle Spectral Slope $\\beta_a$')
        ax.set_ylabel('Vibration Energy $E$ [m$^2$/s$^4$]')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Panel (b): E vs speed
        ax = axes[0, 1]
        if 'mean_speed' in df_imu.columns:
            df_plot = df_imu.dropna(subset=['mean_speed'])
            sc = ax.scatter(df_plot['mean_speed'], df_plot['energy'], 
                          alpha=0.5, s=20, c=df_plot['beta_a'], 
                          cmap='viridis', edgecolors='none')
            plt.colorbar(sc, ax=ax, label='$\\beta_a$')
            if 'E_vs_speed_linear' in results_summary:
                r_val = results_summary['E_vs_speed_linear']['r']
                p_val = results_summary['E_vs_speed_linear']['p']
                ax.set_title(f'(b) $E$ vs speed ($r={r_val:.3f}$, '
                            f'$p={p_val:.1e}$)')
        ax.set_xlabel('Vehicle Speed [m/s]')
        ax.set_ylabel('Vibration Energy $E$ [m$^2$/s$^4$]')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
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
        ax.set_title(f'(c) Distribution of $\\beta_a$ (n={len(df_imu)})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel (d): Log-log power law (v > 0.5 m/s only, matching text)
        ax = axes[1, 1]
        if 'mean_speed' in df_imu.columns:
            df_plot = df_imu.dropna(subset=['mean_speed'])
            df_plot = df_plot[df_plot['mean_speed'] > 0.5]
            if len(df_plot) > 0:
                # Log-log scatter with regression line
                log_v = np.log10(df_plot['mean_speed'].values)
                log_E = np.log10(df_plot['energy'].values)
                ax.scatter(log_v, log_E, alpha=0.5, s=20, c='forestgreen',
                          edgecolors='none')
                if len(df_plot) >= 5:
                    sl, intercept, r, p, _ = linregress(log_v, log_E)
                    xfit = np.linspace(log_v.min(), log_v.max(), 50)
                    ax.plot(xfit, sl*xfit + intercept, 'r-', lw=2,
                           label=f'slope={sl:.2f}, $r$={r:.3f}')
                    ax.legend(fontsize=9)
                ax.set_xlabel('$\\log_{10}$(speed [m/s])')
                ax.set_ylabel('$\\log_{10}$($E$ [m$^2$/s$^4$])')
                ax.set_title(f'(d) Power-law: $E \\propto v^{{{sl:.1f}}}$')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/tartandrive_validation.png', dpi=300, 
                   bbox_inches='tight')
        print(f"Figure saved: figures/tartandrive_validation.png")
        plt.close()
    
    except Exception as e:
        print(f"Figure generation error: {e}")
        print("(Results still saved to CSV)")
    
    # Save final CSV
    df.to_csv('tartandrive_validation_results.csv', index=False)
    print(f"Data saved: tartandrive_validation_results.csv")
    
    # --- Manuscript-ready summary ---
    print("\n" + "=" * 70)
    print("MANUSCRIPT-READY RESULTS SUMMARY")
    print("=" * 70)
    
    n = len(df_imu)
    beta_mean = df_imu['beta_a'].mean()
    beta_std = df_imu['beta_a'].std()
    E_range = (df_imu['energy'].min(), df_imu['energy'].max())
    
    print(f"""
Dataset: TartanDrive (Triest et al., ICRA 2022)
Vehicle: Yamaha Viking ATV, natural off-road terrain
Sensors: IMU at 200 Hz (z-acceleration)
Segments: {n} (10-second windows)

Key findings:
  β_a = {beta_mean:.2f} ± {beta_std:.2f} (range {df_imu['beta_a'].min():.2f}–{df_imu['beta_a'].max():.2f})
  E range: {E_range[0]:.3f}–{E_range[1]:.3f} m²/s⁴
""")
    
    if 'E_vs_beta_a' in results_summary:
        s = results_summary['E_vs_beta_a']
        print(f"  E vs β_a: r = {s['r']:.3f}, p = {s['p']:.1e}, n = {s['n']}")
    
    if 'E_vs_speed' in results_summary:
        s = results_summary['E_vs_speed']
        print(f"  E vs speed (log-log): slope = {s['slope']:.2f}, "
              f"r = {s['r']:.3f}, p = {s['p']:.1e}")
        print(f"    (Theory: slope = 6-2D; observed slope {s['slope']:.2f} "
              f"implies effective D ≈ {(6-s['slope'])/2:.2f})")
    
    if 'E_vs_speed_linear' in results_summary:
        s = results_summary['E_vs_speed_linear']
        print(f"  E vs speed (linear): r = {s['r']:.3f}, p = {s['p']:.1e}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TartanDrive stream-and-delete analysis')
    parser.add_argument('--tar_file', type=str, 
                        default='tartan_drive/data/20210826_heightmaps_1.tar.gz',
                        help='Path to tar.gz file')
    parser.add_argument('--max_bags', type=int, default=50,
                        help='Maximum number of bags to process')
    parser.add_argument('--segment_s', type=float, default=10.0,
                        help='Segment duration in seconds')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TartanDrive Off-Road Validation — STREAM & DELETE MODE")
    print("=" * 70)
    print("Each bag is extracted, analyzed, then DELETED to save disk space.")
    print(f"Max disk usage: ~1 bag at a time (up to ~13 GB)")
    
    # Process
    df = stream_process_tar(args.tar_file, max_bags=args.max_bags)
    
    if len(df) == 0:
        print("\nERROR: No data extracted from any bags.")
        print("Check that `rosbags` is installed: pip install rosbags")
        sys.exit(1)
    
    # Results
    generate_results(df)

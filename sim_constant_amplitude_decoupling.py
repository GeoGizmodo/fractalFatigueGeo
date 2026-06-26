#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant-Amplitude Decoupling Simulation
=========================================

PURPOSE:
  Test whether spectral slope beta independently affects vibration energy
  when amplitude C_z is held constant. Addresses Reviewer #1 Comment (5).

APPROACH:
  In the original 1500 simulations, C_z and beta are strongly correlated
  (r = -0.962). Here we decouple them by normalizing all terrain profiles
  to identical RMS amplitude before vehicle simulation.

DATA SOURCES:
  - Terrain generation: Spectral synthesis (same as repo FractalTerrainGenerator)
  - Vehicle dynamics: 2-DOF quarter-car (same as repo QuarterCarSimulator)
  - Parameters: From manuscript Methods section exactly

WHY NEW SIMULATED DATA:
  The existing CSV has amplitude-complexity coupling baked in. To decouple
  C_z from beta, we must generate new terrain with controlled normalization.

OUTPUT:
  - constant_amplitude_results.csv
  - figures/constant_amplitude_decoupling.png
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import odeint
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

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
# 1. TERRAIN GENERATION
# =============================================================================

def generate_fbm_terrain_constant_amplitude(size, target_D, rms_target, seed):
    """
    Generate fractal terrain with FIXED RMS amplitude.
    Uses spectral synthesis with beta = 7-2D, then normalizes RMS.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))

    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    f_magnitude = np.sqrt(fx**2 + fy**2)
    f_magnitude[0, 0] = 1e-10

    beta = 7.0 - 2.0 * target_D
    filter_func = f_magnitude ** (-beta / 2.0)
    filter_func[0, 0] = 0

    noise_fft = np.fft.fft2(noise)
    filtered_fft = noise_fft * filter_func
    terrain = np.real(np.fft.ifft2(filtered_fft))

    # Extract middle row as 1D profile FIRST
    profile = terrain[size // 2, :]

    # Normalize the PROFILE (not 2D grid) to fixed RMS
    # This ensures the actual vehicle input has constant amplitude
    profile = profile - np.mean(profile)
    rms_actual = np.sqrt(np.mean(profile**2))
    if rms_actual > 0:
        profile = profile * (rms_target / rms_actual)

    return profile


# =============================================================================
# 2. VEHICLE DYNAMICS
# =============================================================================

def simulate_quarter_car(profile, pixel_size, vehicle_speed, ms, mu, ks, cs, kt):
    """
    2-DOF quarter-car simulation using odeint.
    Same physics as repo QuarterCarSimulator.
    """
    dt = pixel_size / vehicle_speed
    n_points = len(profile)
    time = np.arange(n_points) * dt

    def dynamics(state, t):
        z_s, dz_s, z_u, dz_u = state
        idx = min(int(t / dt), n_points - 1)
        z_r = profile[idx]
        F_spring = ks * (z_u - z_s)
        F_damper = cs * (dz_u - dz_s)
        F_tire = kt * (z_r - z_u)
        ddz_s = (F_spring + F_damper) / ms
        ddz_u = (-F_spring - F_damper + F_tire) / mu
        return [dz_s, ddz_s, dz_u, ddz_u]

    x0 = [profile[0], 0.0, profile[0], 0.0]
    solution = odeint(dynamics, x0, time)

    dz_s = solution[:, 1]
    acc = np.gradient(dz_s, dt)

    # Skip transient (first 20%)
    skip = int(0.2 * len(acc))
    acc_steady = acc[skip:]

    rms_acc = np.sqrt(np.mean(acc_steady**2))
    energy = rms_acc**2
    return energy, rms_acc


# =============================================================================
# 3. SPECTRAL ANALYSIS
# =============================================================================

def measure_spectral_slope(profile, pixel_size, vehicle_speed):
    """Measure PSD spectral slope from terrain profile (Welch, 0.1-10 Hz)."""
    dt = pixel_size / vehicle_speed
    fs = 1.0 / dt
    nperseg = min(1024, len(profile) // 4)
    noverlap = nperseg // 2
    freqs, psd = signal.welch(profile, fs=fs, nperseg=nperseg,
                              window='hann', noverlap=noverlap)
    mask = (freqs >= 0.1) & (freqs <= 10.0) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan, 0.0
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask])
    slope, intercept, r, p, se = linregress(log_f, log_psd)
    return -slope, r**2


# =============================================================================
# 4. MAIN SIMULATION
# =============================================================================

TARGET_D = [2.05, 2.15, 2.25, 2.35, 2.45]
N_REALIZATIONS = 30


def run_constant_amplitude_simulations():
    """Run decoupled simulations: vary D, hold RMS amplitude constant."""

    SIZE = 512
    PIXEL_SIZE = 10.0
    SPEED = 15.0
    RMS_TARGET = 5.0  # meters, CONSTANT for all D

    MU = 100.0
    KT = 200000.0
    ZETA = 0.30

    vehicles = {
        'Vehicle A': {'ms': 800.0, 'ks': 25000.0},
        'Vehicle B': {'ms': 1000.0, 'ks': 35000.0},
        'Vehicle C': {'ms': 1200.0, 'ks': 45000.0},
    }
    for veh in vehicles.values():
        veh['cs'] = 2 * ZETA * np.sqrt(veh['ks'] * veh['ms'])

    print("=" * 70)
    print("CONSTANT-AMPLITUDE DECOUPLING SIMULATION")
    print("=" * 70)
    total = len(vehicles) * len(TARGET_D) * N_REALIZATIONS
    print(f"Total simulations: {total}")

    results = []
    count = 0

    for D_target in TARGET_D:
        for real_idx in range(N_REALIZATIONS):
            seed = int(D_target * 10000) + real_idx
            profile = generate_fbm_terrain_constant_amplitude(
                SIZE, D_target, RMS_TARGET, seed
            )
            rms_check = np.sqrt(np.mean((profile - np.mean(profile))**2))
            beta_measured, beta_r2 = measure_spectral_slope(profile, PIXEL_SIZE, SPEED)

            for veh_name, veh_params in vehicles.items():
                energy, rms_acc = simulate_quarter_car(
                    profile, PIXEL_SIZE, SPEED,
                    veh_params['ms'], MU,
                    veh_params['ks'], veh_params['cs'], KT
                )
                results.append({
                    'vehicle': veh_name,
                    'target_D': D_target,
                    'beta_measured': beta_measured,
                    'beta_r2': beta_r2,
                    'rms_amplitude': rms_check,
                    'energy': energy,
                    'rms_acceleration': rms_acc,
                    'seed': seed,
                    'realization': real_idx,
                })
                count += 1

            if (real_idx + 1) % 10 == 0:
                print(f"  D={D_target:.2f}: {real_idx+1}/{N_REALIZATIONS} done")

    print(f"\nCompleted {count} simulations.")
    return pd.DataFrame(results)


# =============================================================================
# 5. ANALYSIS AND FIGURE
# =============================================================================

def analyze_and_plot(df):
    """Analyze results and generate figure."""

    print("\n" + "=" * 70)
    print("AMPLITUDE VERIFICATION")
    print("=" * 70)
    rms_by_D = df.groupby('target_D')['rms_amplitude'].agg(['mean', 'std'])
    print(rms_by_D)
    cv_amp = df['rms_amplitude'].std() / df['rms_amplitude'].mean() * 100
    print(f"CV of amplitude: {cv_amp:.4f}%")

    print("\n" + "=" * 70)
    print("ENERGY VS FRACTAL DIMENSION (CONSTANT AMPLITUDE)")
    print("=" * 70)

    D_all = df['target_D'].values
    E_all = df['energy'].values
    valid = (E_all > 0) & np.isfinite(E_all)
    log_E = np.log10(E_all[valid])
    D_valid = D_all[valid]

    if np.any(~np.isfinite(log_E)):
        finite_mask = np.isfinite(log_E)
        log_E = log_E[finite_mask]
        D_valid = D_valid[finite_mask]

    slope, intercept, r_val, p_val, stderr = linregress(D_valid, log_E)
    R2 = r_val**2
    print(f"\nCombined: log10(E) = {slope:.3f}*D + {intercept:.3f}")
    print(f"  R2 = {R2:.4f}, r = {r_val:.4f}, p = {p_val:.2e}, n = {len(D_valid)}")

    for veh in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
        mask = (df['vehicle'] == veh) & (df['energy'] > 0) & np.isfinite(df['energy'])
        if mask.sum() < 5:
            print(f"  {veh}: insufficient valid data ({mask.sum()} points)")
            continue
        D_v = df.loc[mask, 'target_D'].values
        E_v = df.loc[mask, 'energy'].values
        log_E_v = np.log10(E_v)
        finite = np.isfinite(log_E_v)
        sl, _, rv, pv, _ = linregress(D_v[finite], log_E_v[finite])
        print(f"  {veh}: slope={sl:.3f}, R2={rv**2:.3f}, p={pv:.2e}")

    # Beta vs D
    df_uniq = df.drop_duplicates(subset=['target_D', 'realization'])
    beta_valid = df_uniq['beta_measured'].dropna()
    D_beta = df_uniq.loc[beta_valid.index, 'target_D']
    sl_b, int_b, r_b, p_b, _ = linregress(D_beta, beta_valid)
    print(f"\nbeta vs D: slope={sl_b:.3f}, R2={r_b**2:.3f}")

    # --- FIGURE ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Vehicle A': '#1f77b4', 'Vehicle B': '#ff7f0e', 'Vehicle C': '#2ca02c'}

    # Panel (a)
    ax = axes[0]
    for veh in ['Vehicle A', 'Vehicle B', 'Vehicle C']:
        mask = df['vehicle'] == veh
        ax.scatter(df.loc[mask, 'target_D'], df.loc[mask, 'energy'],
                   c=colors[veh], alpha=0.5, s=25, label=veh, edgecolors='none')
    D_fit = np.linspace(2.0, 2.5, 100)
    E_fit = 10**(slope * D_fit + intercept)
    ax.plot(D_fit, E_fit, 'k--', lw=2.5,
            label=f'slope={slope:.2f}, R2={R2:.3f}')
    ax.set_yscale('log')
    ax.set_xlabel('Terrain Fractal Dimension D')
    ax.set_ylabel('Vibration Energy E (m2/s4)')
    ax.set_title('(a) Energy vs D at Constant Amplitude')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel (b)
    ax2 = axes[1]
    grouped = df.groupby('target_D')['energy'].agg(['mean', 'std'])
    n_per = N_REALIZATIONS * 3
    grouped['se'] = grouped['std'] / np.sqrt(n_per)
    ax2.errorbar(grouped.index, grouped['mean'], yerr=1.96*grouped['se'],
                 fmt='o-', color='darkblue', capsize=5, capthick=2,
                 linewidth=2, markersize=8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Terrain Fractal Dimension D')
    ax2.set_ylabel('Mean Vibration Energy E (m2/s4)')
    ax2.set_title('(b) Energy Trend at Fixed Amplitude')
    ax2.grid(True, alpha=0.2)

    # Panel (c)
    ax3 = axes[2]
    ax3.scatter(df_uniq['target_D'], df_uniq['beta_measured'],
                c='purple', alpha=0.5, s=30, edgecolors='none')
    beta_fit = sl_b * D_fit + int_b
    ax3.plot(D_fit, beta_fit, 'k--', lw=2,
             label=f'beta = {sl_b:.2f}*D + {int_b:.2f}\nR2 = {r_b**2:.3f}')
    ax3.set_xlabel('Terrain Fractal Dimension D')
    ax3.set_ylabel('Measured Spectral Slope beta')
    ax3.set_title('(c) Spectral Slope vs D (amplitude constant)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('figures/constant_amplitude_decoupling.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: figures/constant_amplitude_decoupling.png")
    plt.close()

    df.to_csv('constant_amplitude_results.csv', index=False)
    print(f"Data saved: constant_amplitude_results.csv")

    print("\n" + "=" * 70)
    print("KEY RESULT FOR MANUSCRIPT:")
    print(f"  Terrain spectral structure (fractal dimension) independently")
    print(f"  explains R2 = {R2:.3f} of energy variance")
    print(f"  at constant amplitude (p = {p_val:.1e}, n = {len(D_valid)})")
    print("=" * 70)

    return {'slope': slope, 'R2': R2, 'r': r_val, 'p': p_val, 'n': len(D_valid)}


# =============================================================================
# 6. RUN
# =============================================================================

if __name__ == '__main__':
    df = run_constant_amplitude_simulations()
    stats = analyze_and_plot(df)

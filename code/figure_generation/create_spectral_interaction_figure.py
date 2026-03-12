#!/usr/bin/env python3
"""
Create Spectral Interaction Figure for MSSP Submission
Shows: Terrain PSD → Vehicle Transfer → Spectral Overlap → Acceleration PSD

This is the KEY figure that vibration journals expect to see.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def terrain_psd(k, D, C_z=1e-3):
    """
    Terrain PSD: S_z(k) = C_z * k^(-beta)
    where beta = 7 - 2D (from self-affine theory)
    """
    beta = 7 - 2*D
    return C_z * k**(-beta)

def vehicle_transfer_function(omega, m=1000, k=40000, c=2000):
    """
    Quarter-car transfer function |H(omega)|^2
    For body acceleration response to terrain input
    """
    omega_n = np.sqrt(k/m)  # Natural frequency
    zeta = c / (2*np.sqrt(k*m))  # Damping ratio
    
    # Transfer function magnitude squared
    numerator = (omega**4)
    denominator = (omega_n**2 - omega**2)**2 + (2*zeta*omega_n*omega)**2
    H_squared = numerator / denominator
    
    return H_squared, omega_n, zeta

def acceleration_psd(omega, D, C_z, m, k, c, v=10):
    """
    Acceleration PSD: S_a(omega) = |H(omega)|^2 * S_z(omega)
    Convert spatial to temporal: omega = v*k
    """
    k_spatial = omega / v
    S_z = terrain_psd(k_spatial, D, C_z)
    H_squared, _, _ = vehicle_transfer_function(omega, m, k, c)
    return S_z * H_squared

def main():
    print("=" * 80)
    print("CREATING SPECTRAL INTERACTION FIGURE FOR MSSP")
    print("=" * 80)
    
    # Vehicle parameters (typical HMMWV-class)
    m = 1000  # kg (sprung mass)
    k = 40000  # N/m (suspension stiffness)
    c = 2000  # N·s/m (damping)
    v = 10  # m/s (vehicle speed)
    
    omega_n = np.sqrt(k/m)
    zeta = c / (2*np.sqrt(k*m))
    f_n = omega_n / (2*np.pi)
    
    print(f"\nVehicle parameters:")
    print(f"  Mass: {m} kg")
    print(f"  Stiffness: {k} N/m")
    print(f"  Damping: {c} N·s/m")
    print(f"  Natural frequency: {f_n:.2f} Hz")
    print(f"  Damping ratio: {zeta:.3f}")
    print(f"  Speed: {v} m/s")
    
    # Frequency ranges
    k_spatial = np.logspace(-2, 1, 500)  # Spatial wavenumber (rad/m)
    omega = np.logspace(-1, 2, 500)  # Temporal frequency (rad/s)
    
    # Terrain fractal dimensions
    D_values = [2.05, 2.25, 2.45]
    D_labels = ['Smooth (D=2.05)', 'Moderate (D=2.25)', 'Rough (D=2.45)']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # PANEL A: Terrain PSD
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    C_z = 1e-3  # Amplitude parameter
    
    for D, label, color in zip(D_values, D_labels, colors):
        beta = 7 - 2*D
        S_z = terrain_psd(k_spatial, D, C_z)
        ax1.loglog(k_spatial, S_z, linewidth=2.5, label=label, color=color)
        
        # Add slope annotation
        k_mid = 0.3
        S_mid = terrain_psd(k_mid, D, C_z)
        ax1.text(k_mid, S_mid*2, f'β={beta:.1f}', 
                fontsize=10, color=color, fontweight='bold')
    
    ax1.set_xlabel('Spatial Wavenumber k (rad/m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Terrain PSD $S_z(k)$ (m³/rad)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Terrain Power Spectral Density', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([1e-7, 1e-1])
    
    # Add theory annotation
    ax1.text(0.05, 0.95, r'$S_z(k) = C_z \cdot k^{-\beta}$' + '\n' + r'$\beta = 7 - 2D$',
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================================================
    # PANEL B: Vehicle Transfer Function
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    H_squared, omega_n_val, zeta_val = vehicle_transfer_function(omega, m, k, c)
    
    ax2.loglog(omega/(2*np.pi), H_squared, linewidth=3, color='#C73E1D', label='Quarter-car model')
    
    # Mark resonance
    ax2.axvline(omega_n_val/(2*np.pi), color='black', linestyle='--', linewidth=2, 
                label=f'Resonance: {omega_n_val/(2*np.pi):.2f} Hz')
    
    # Shade resonance band
    f_res = omega_n_val / (2*np.pi)
    ax2.axvspan(f_res*0.7, f_res*1.3, alpha=0.2, color='red', label='Resonance band')
    
    ax2.set_xlabel('Frequency f (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transfer Function $|H(\\omega)|^2$', fontsize=12, fontweight='bold')
    ax2.set_title('B. Vehicle Transfer Function', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([0.1, 100])
    ax2.set_ylim([1e-4, 1e2])
    
    # Add parameters annotation
    ax2.text(0.05, 0.05, f'm = {m} kg\nk = {k} N/m\nc = {c} N·s/m\nζ = {zeta_val:.3f}',
            transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========================================================================
    # PANEL C: Spectral Overlap (Key Insight!)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Convert spatial to temporal for overlay
    f_hz = omega / (2*np.pi)
    k_from_omega = omega / v
    
    # Plot terrain PSDs (converted to temporal)
    for D, label, color in zip(D_values, D_labels, colors):
        S_z_temporal = terrain_psd(k_from_omega, D, C_z) * v  # Scale for temporal
        ax3.loglog(f_hz, S_z_temporal, linewidth=2, label=f'Terrain: {label}', 
                  color=color, linestyle='-', alpha=0.7)
    
    # Overlay transfer function (normalized for visualization)
    H_squared_norm = H_squared / np.max(H_squared) * 1e-4
    ax3.loglog(f_hz, H_squared_norm, linewidth=3, color='#C73E1D', 
              label='Vehicle filter', linestyle='--', alpha=0.9)
    
    # Mark resonance
    ax3.axvline(f_n, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax3.axvspan(f_n*0.7, f_n*1.3, alpha=0.15, color='red')
    
    ax3.set_xlabel('Frequency f (Hz)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Spectral Density (normalized)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Spectral Interaction (Terrain × Vehicle)', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim([0.1, 100])
    
    # Add key insight annotation
    ax3.text(0.5, 0.95, 'Energy concentration near resonance\ndetermines vibration severity',
            transform=ax3.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========================================================================
    # PANEL D: Resulting Acceleration PSD
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for D, label, color in zip(D_values, D_labels, colors):
        S_a = acceleration_psd(omega, D, C_z, m, k, c, v)
        ax4.loglog(f_hz, S_a, linewidth=2.5, label=label, color=color)
        
        # Calculate RMS acceleration
        a_rms = np.sqrt(np.trapz(S_a, omega))
        print(f"\n{label}:")
        print(f"  RMS acceleration: {a_rms:.3f} m/s²")
    
    # Mark resonance
    ax4.axvline(f_n, color='black', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Resonance: {f_n:.2f} Hz')
    ax4.axvspan(f_n*0.7, f_n*1.3, alpha=0.15, color='red')
    
    ax4.set_xlabel('Frequency f (Hz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Acceleration PSD $S_a(\\omega)$ (m²/s⁴/Hz)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Vehicle Acceleration Spectrum', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=11, frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim([0.1, 100])
    
    # Add result annotation
    ax4.text(0.05, 0.95, r'$S_a(\omega) = |H(\omega)|^2 \cdot S_z(\omega)$',
            transform=ax4.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ========================================================================
    # Overall title
    # ========================================================================
    fig.suptitle('Spectral Interaction: Terrain Excitation → Vehicle Response', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig('spectral_interaction_mssp.png', dpi=300, bbox_inches='tight')
    plt.savefig('github/images_manuscript/spectral_interaction_mssp.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "=" * 80)
    print("FIGURE SAVED")
    print("=" * 80)
    print("Files:")
    print("  - spectral_interaction_mssp.png")
    print("  - github/images_manuscript/spectral_interaction_mssp.png")
    
    print("\n" + "=" * 80)
    print("CAPTION FOR MANUSCRIPT")
    print("=" * 80)
    print("""
Figure X. Spectral interaction between terrain excitation and vehicle dynamics.
(a) Terrain power spectral density functions for three fractal dimensions 
(D = 2.05, 2.25, 2.45) showing power-law scaling S_z(k) ∝ k^(-β) with 
β = 7 - 2D. Steeper slopes (higher β) indicate smoother terrain with energy 
concentrated at long wavelengths. (b) Quarter-car vehicle transfer function 
|H(ω)|² showing resonance amplification near the suspension natural frequency 
(f_n = 1.0 Hz). The shaded band indicates the resonance region where terrain 
energy is most efficiently transmitted to vehicle vibration. (c) Spectral 
overlap demonstrating how terrain spectral energy near the vehicle resonance 
frequency dominates the response. The interaction between terrain slope and 
vehicle filtering determines vibration severity. (d) Resulting acceleration 
power spectral density S_a(ω) = |H(ω)|² S_z(ω) for each terrain type. Smooth 
terrain (D = 2.05) produces higher vibration energy due to greater spectral 
amplitude at resonance, despite having steeper spectral roll-off. This 
illustrates why both PSD amplitude (C_z) and slope (β) are required for 
accurate vibration prediction.
    """)
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR REVIEWERS")
    print("=" * 80)
    print("1. Shows complete spectral propagation chain")
    print("2. Visualizes resonance amplification mechanism")
    print("3. Explains why terrain near f_n matters most")
    print("4. Demonstrates two-parameter requirement (C_z, β)")
    print("5. Uses standard vibration engineering language")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

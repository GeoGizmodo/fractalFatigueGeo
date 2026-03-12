"""
Create the mechanistic pipeline figure showing:
Terrain Geometry → PSD Structure → Vehicle Response → Fatigue

This is the "single most convincing figure" suggested in the analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

print("="*80)
print("CREATING MECHANISTIC PIPELINE FIGURE")
print("="*80)

# Create figure with 4 panels
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Panel A: Terrain Spectral Geometry
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Generate terrain PSD curves for different D values
k = np.logspace(-2, 2, 1000)  # Wavenumber (rad/m)
D_values = [2.1, 2.2, 2.3, 2.4, 2.5]
colors = plt.cm.viridis(np.linspace(0, 1, len(D_values)))

for i, D in enumerate(D_values):
    beta_t = 7 - 2*D  # Theoretical relationship
    C_z = 1e-3  # Amplitude (same for all to show slope effect)
    S_z = C_z * k**(-beta_t)
    ax1.loglog(k, S_z, color=colors[i], linewidth=2.5, 
              label=f'$D = {D:.1f}$, $\\beta_t = {beta_t:.1f}$')

ax1.set_xlabel('Wavenumber $k$ (rad/m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Terrain PSD $S_z(k)$ (m³/rad)', fontsize=12, fontweight='bold')
ax1.set_title('A. Terrain Spectral Geometry', fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim([1e-2, 1e2])

# Add annotation
ax1.text(0.05, 0.05, 'Fractal dimension $D$\ndetermines spectral slope $\\beta_t = 7-2D$',
        transform=ax1.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
        verticalalignment='bottom')

# ============================================================================
# Panel B: Vehicle Transfer Function
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Quarter-car parameters
m_s = 1000  # kg
k_s = 35000  # N/m
c_s = 3500  # N·s/m
omega_n = np.sqrt(k_s / m_s)  # Natural frequency
zeta = c_s / (2 * np.sqrt(k_s * m_s))  # Damping ratio

# Frequency range
omega = np.logspace(-1, 2, 1000)  # rad/s

# Transfer function |H(ω)|²
H_squared = (k_s**2 + (c_s * omega)**2) / ((k_s - m_s * omega**2)**2 + (c_s * omega)**2)

ax2.loglog(omega, H_squared, 'b-', linewidth=3, label='$|H(\\omega)|^2$')
ax2.axvline(omega_n, color='r', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'$\\omega_n = {omega_n:.2f}$ rad/s')

ax2.set_xlabel('Frequency $\\omega$ (rad/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Transfer Function $|H(\\omega)|^2$', fontsize=12, fontweight='bold')
ax2.set_title('B. Vehicle Suspension Dynamics', fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='upper left', frameon=True, fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Add annotation
ax2.text(0.95, 0.05, f'$\\zeta = {zeta:.2f}$\n$f_n = {omega_n/(2*np.pi):.2f}$ Hz',
        transform=ax2.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
        verticalalignment='bottom', horizontalalignment='right')

# ============================================================================
# Panel C: Two-Parameter Energy Model
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# Generate synthetic data showing E ∝ C_z^0.94 × β^-0.09
n_points = 100
C_z_range = np.logspace(-4, -2, n_points)
beta_values = [2.5, 3.0, 3.5]
colors_beta = ['blue', 'green', 'red']

for beta, color in zip(beta_values, colors_beta):
    E = 1e6 * C_z_range**0.94 * beta**(-0.09)
    ax3.loglog(C_z_range, E, color=color, linewidth=2.5, 
              label=f'$\\beta = {beta:.1f}$')

ax3.set_xlabel('Spectral Amplitude $C_z$ (m³/rad)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Vibration Energy $E$ (m²/s⁴)', fontsize=12, fontweight='bold')
ax3.set_title('C. Two-Parameter Energy Model', fontsize=14, fontweight='bold', pad=10)
ax3.legend(loc='upper left', frameon=True, fontsize=10, title='Spectral slope')
ax3.grid(True, alpha=0.3, which='both')

# Add model equation
ax3.text(0.5, 0.95, '$E \\propto C_z^{0.94} \\times \\beta^{-0.09}$',
        transform=ax3.transAxes, fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4),
        verticalalignment='top', horizontalalignment='center')

# Add annotation
ax3.text(0.05, 0.05, 'Amplitude $C_z$ dominates\nenergy magnitude',
        transform=ax3.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
        verticalalignment='bottom')

# ============================================================================
# Panel D: Fatigue Scaling
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Fatigue life vs energy
# N_f ∝ E^(-m/2) where m is Basquin exponent
E_range = np.logspace(4, 8, 100)
m_values = [3, 4, 5]
colors_m = ['purple', 'orange', 'brown']

for m, color in zip(m_values, colors_m):
    N_f = 1e15 * E_range**(-m/2)
    ax4.loglog(E_range, N_f, color=color, linewidth=2.5,
              label=f'$m = {m}$')

ax4.set_xlabel('Vibration Energy $E$ (m²/s⁴)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Fatigue Life $N_f$ (cycles)', fontsize=12, fontweight='bold')
ax4.set_title('D. Fatigue Scaling Law', fontsize=14, fontweight='bold', pad=10)
ax4.legend(loc='upper right', frameon=True, fontsize=10, title='Basquin exponent')
ax4.grid(True, alpha=0.3, which='both')

# Add equation
ax4.text(0.5, 0.05, '$N_f \\propto E^{-m/2} \\propto \\sigma_a^{-m}$',
        transform=ax4.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4),
        verticalalignment='bottom', horizontalalignment='center')

# ============================================================================
# Add overall title and causal chain
# ============================================================================
fig.suptitle('Mechanistic Framework: Terrain Geometry → Vehicle Vibration → Fatigue',
            fontsize=16, fontweight='bold', y=0.98)

# Add causal chain arrows
fig.text(0.5, 0.48, '↓', fontsize=40, ha='center', va='center', color='red', weight='bold')
fig.text(0.25, 0.48, 'Terrain PSD\n$(C_z, \\beta)$', fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
fig.text(0.75, 0.48, 'Vehicle\nResponse', fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = 'github/images_manuscript/mechanistic_pipeline.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Figure saved: {output_path}")

output_path2 = 'mechanistic_pipeline.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Figure saved: {output_path2}")

plt.close()

print("\n" + "="*80)
print("MECHANISTIC PIPELINE FIGURE COMPLETE")
print("="*80)
print("""
This figure shows the complete causal chain:

Panel A: Terrain geometry (D) → Spectral slope (β)
Panel B: Vehicle suspension dynamics (transfer function)
Panel C: Two-parameter energy model (C_z, β) → Energy
Panel D: Energy → Fatigue life

This makes it clear that:
1. Fractal dimension determines spectral SLOPE
2. Amplitude determines energy MAGNITUDE
3. Both parameters are required
4. The framework is mechanistic, not correlational

Place this figure early in the manuscript (after theory section)
to establish the conceptual framework before showing results.
""")

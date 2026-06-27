#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Figure 1 with proper (a), (b), (c), (d) panel labels.
Based on the original create_mechanistic_pipeline_figure.py from the repo.

Only change from original: panel labels added as bold letters in top-left corner
of each subplot per Nature Communications figure style.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Panel (a): Terrain Spectral Geometry
ax1 = fig.add_subplot(gs[0, 0])
k = np.logspace(-2, 2, 1000)
D_values = [2.1, 2.2, 2.3, 2.4, 2.5]
colors = plt.cm.viridis(np.linspace(0, 1, len(D_values)))

for i, D in enumerate(D_values):
    beta_t = 7 - 2*D
    C_z = 1e-3
    S_z = C_z * k**(-beta_t)
    ax1.loglog(k, S_z, color=colors[i], linewidth=2.5,
               label=f'$D = {D:.1f}$, $\\beta_t = {beta_t:.1f}$')

ax1.set_xlabel('Wavenumber $k$ (rad/m)')
ax1.set_ylabel('Terrain PSD $S_z(k)$ (m$^3$/rad)')
ax1.set_title('Terrain Spectral Geometry')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim([1e-2, 1e2])
ax1.text(0.05, 0.05, '$\\beta_t = 7-2D$',
         transform=ax1.transAxes, fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         verticalalignment='bottom')
# Panel label
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top')

# Panel (b): Vehicle Transfer Function
ax2 = fig.add_subplot(gs[0, 1])
m_s = 1000
k_s = 35000
c_s = 2 * 0.30 * np.sqrt(k_s * m_s)
omega_n = np.sqrt(k_s / m_s)
zeta = 0.30
omega = np.logspace(-1, 2, 1000)
H_squared = (k_s**2 + (c_s * omega)**2) / ((k_s - m_s * omega**2)**2 + (c_s * omega)**2)

ax2.loglog(omega, H_squared, 'b-', linewidth=3, label='$|H(\\omega)|^2$')
ax2.axvline(omega_n, color='r', linestyle='--', linewidth=2, alpha=0.7,
            label=f'$\\omega_n = {omega_n:.1f}$ rad/s')
ax2.set_xlabel('Frequency $\\omega$ (rad/s)')
ax2.set_ylabel('Transfer Function $|H(\\omega)|^2$')
ax2.set_title('Vehicle Suspension Dynamics')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3, which='both')
ax2.text(0.95, 0.05, f'$\\zeta = {zeta:.2f}$\n$f_n = {omega_n/(2*np.pi):.2f}$ Hz',
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
         verticalalignment='bottom', horizontalalignment='right')
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top')

# Panel (c): Two-Parameter Energy Model
ax3 = fig.add_subplot(gs[1, 0])
n_points = 100
C_z_range = np.logspace(-4, -2, n_points)
beta_values = [2.0, 2.5, 3.0]
colors_beta = ['#0072B2', '#D55E00', '#CC79A7']  # CVD-safe: blue, vermillion, pink

for beta, color in zip(beta_values, colors_beta):
    E = 1e6 * C_z_range**0.94 * beta**(-0.09)
    ax3.loglog(C_z_range, E, color=color, linewidth=2.5,
               label=f'$\\beta_t = {beta:.1f}$')

ax3.set_xlabel('Spectral Amplitude $C_z$ (m$^3$/rad)')
ax3.set_ylabel('Vibration Energy $E$ (m$^2$/s$^4$)')
ax3.set_title('Two-Parameter Energy Model')
ax3.legend(loc='upper left', fontsize=9, title='Spectral slope')
ax3.grid(True, alpha=0.3, which='both')
ax3.text(0.5, 0.95, '$E \\propto C_z^{0.94} \\times \\beta_t^{-0.09}$',
         transform=ax3.transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4),
         verticalalignment='top', horizontalalignment='center')
ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes,
         fontsize=14, fontweight='bold', va='top')

# Panel (d): Fatigue Scaling
ax4 = fig.add_subplot(gs[1, 1])
E_range = np.logspace(4, 8, 100)
m_values = [3, 4, 5]
colors_m = ['purple', 'orange', 'brown']

for m, color in zip(m_values, colors_m):
    N_f = 1e15 * E_range**(-m/2)
    ax4.loglog(E_range, N_f, color=color, linewidth=2.5,
               label=f'$m = {m}$')

ax4.set_xlabel('Vibration Energy $E$ (m$^2$/s$^4$)')
ax4.set_ylabel('Fatigue Life $N_f$ (cycles)')
ax4.set_title('Fatigue Scaling Law')
ax4.legend(loc='upper right', fontsize=9, title='Basquin exponent')
ax4.grid(True, alpha=0.3, which='both')
ax4.text(0.5, 0.05, '$N_f \\propto E^{-m/2}$',
         transform=ax4.transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4),
         verticalalignment='bottom', horizontalalignment='center')
ax4.text(-0.12, 1.05, '(d)', transform=ax4.transAxes,
         fontsize=14, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('figures/Figure1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/Figure1.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 saved with (a), (b), (c), (d) panel labels.")
plt.close()

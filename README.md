# Fractal Terrain Geometry Controls Vehicle Vibration

<div align="center">

![Rotating Earth](images/animate-the-sphere-rotating-slowly-with-the-glowin.gif)

**Two-Parameter Spectral Framework for Predicting Terrain-Induced Mechanical Fatigue**

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[Paper](manuscripts/arxiv_manuscript_combined.pdf) • [Documentation](docs/) • [Data](data/) • [Code](code/)

</div>

---

## 🌍 Overview

Terrain-induced mechanical wear remains unpredictable despite decades of vehicle dynamics research. This work establishes a **two-parameter spectral framework** connecting terrain geometry to vehicle vibration and mechanical fatigue through rigorous physics-based derivation and comprehensive validation.

### 🎯 Core Innovation

Complete terrain characterization requires **two parameters**:

1. **Amplitude (C_z)**: Controls energy magnitude  
2. **Spectral slope (β)**: Determines spectral structure

**Key relationship**: Fractal dimension provides geometric basis for spectral slope through **β_t = 7-2D**

### 📊 Key Results

```
Vibration Energy:  E ∝ C_z^0.94 × β_a^-0.09  (R² = 0.96)
Fractal-Spectral:  β_t = 7 - 2D              (from theory)
Correlation:       r = -0.956                 (95% CI: [-0.959, -0.952])
Ensemble:          γ = -2.34 ± 0.19          (CV = 8.1%, n=100 vehicles)
```

---

## 📁 Repository Structure

```
fractalFatigue/
├── manuscripts/              # LaTeX manuscripts and PDFs
│   ├── arxiv_manuscript_combined.tex
│   ├── arxiv_manuscript_combined.pdf
│   ├── nature_communications_manuscript.tex
│   └── nature_communications_manuscript.pdf
├── figures/                  # All manuscript figures
│   ├── theoretical_validation.png
│   ├── figure1_beta_D_relationship.png
│   ├── three_vehicle_universality.png
│   ├── spectral_framework_validation.png
│   ├── mechanistic_pipeline.png
│   └── ...
├── code/
│   ├── simulations/         # Monte Carlo and ensemble simulations
│   │   ├── sim05_vehicle_ensemble_validation.py
│   │   ├── vehicle_library.py
│   │   ├── fractal_terrain_generator.py
│   │   └── vehicle_dynamics_simulator.py
│   ├── analysis/            # Statistical analysis and verification
│   │   ├── analyze_ensemble_results.py
│   │   ├── improve_lidar_correlation.py
│   │   └── verify_all_equations_sympy.py
│   ├── validation/          # Real-world data validation
│   │   ├── complete_real_world_validation.py
│   │   └── corrected_spectral_validation.py
│   └── figure_generation/   # Scripts to generate all figures
│       ├── create_spectral_interaction_figure.py
│       └── ...
├── data/
│   ├── simulation_results/  # Monte Carlo and ensemble results
│   │   ├── three_vehicle_validation_results.csv
│   │   ├── vehicle_ensemble_results.csv
│   │   └── advanced_spectral_results.csv
│   └── real_world/          # LiDAR and vehicle sensor data
├── docs/                    # Documentation
│   └── ARXIV_READY_FOR_SUBMISSION.md
├── images/                  # Repository images
│   └── animate-the-sphere-rotating-slowly-with-the-glowin.gif
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---


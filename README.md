# Two-Parameter Spectral Framework for Predicting Terrain-Induced Mechanical Fatigue

![Rotating Earth](images/animate-the-sphere-rotating-slowly-with-the-glowin.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)
- [Contact](#contact)

---

## Overview

Terrain-induced mechanical wear remains unpredictable despite decades of vehicle dynamics research. This work establishes a **two-parameter spectral framework** that connects terrain geometry to vehicle vibration and mechanical fatigue through rigorous physics-based derivation and comprehensive validation.

### Core Innovation

We demonstrate that complete terrain characterization requires **two parameters**:
1. **Amplitude (C_z)**: Controls energy magnitude
2. **Spectral slope (β)**: Determines spectral structure

Fractal dimension provides the geometric basis for spectral slope through the relationship **β_t = 7-2D**.

### Scientific Impact

- **Predictive Maintenance**: Enables terrain-aware maintenance scheduling
- **Route Optimization**: Minimize component wear by selecting favorable terrain
- **Vehicle Design**: Optimize suspension for expected terrain profiles
- **Mission Planning**: Balance operational objectives against vehicle longevity

---

## Key Findings

### 1. Two-Parameter Model
```
E ∝ C_z^0.94 × β_a^-0.09  (R² = 0.96)
```
- Amplitude dominates energy magnitude (exponent ≈ 1)
- Spectral slope determines structure (exponent ≈ 0 for direct energy)

### 2. Fractal-Spectral Connection
```
β_t = 7 - 2D
```
- Derived from self-affine surface theory
- Smooth terrain (D=2.0) → steep roll-off (β=3.0)
- Rough terrain (D=2.5) → shallow roll-off (β=2.0)

### 3. Vehicle-Independent Scaling
- **Correlation**: r = -0.956 (95% CI: [-0.959, -0.952])
- **Sample size**: n = 1500 terrain realizations
- **Consistency**: CV = 1.4% across 3 vehicle types

### 4. Ensemble Validation
- **100 vehicles**: 18,000 simulations
- **Fatigue exponent**: γ = -2.34 ± 0.19 (CV = 8.1%)
- **Mass independence**: R² = 0.000
- **Frequency dependence**: γ = 0.36 f_n - 2.93 (R² = 0.879)

### 5. Real-World Validation
- **LiDAR**: 13 terrain regions, r = -0.62 (p < 0.05)
- **Vehicle sensors**: 8,609 road segments
- **IRI correlation**: r = +0.194 (β_a), r = +0.145 (energy)

---

## Repository Structure

```
fractalFatigue/
├── manuscripts/
│   ├── arxiv_manuscript_combined.tex
│   ├── arxiv_manuscript_combined.pdf
│   ├── nature_communications_manuscript.tex
│   └── nature_communications_manuscript.pdf
├── figures/
│   ├── theoretical_validation.png
│   ├── figure1_beta_D_relationship.png
│   ├── three_vehicle_universality.png
│   ├── spectral_framework_validation.png
│   ├── mechanistic_pipeline.png
│   └── beta_theory_validation.png
├── code/
│   ├── simulations/
│   │   ├── sim05_vehicle_ensemble_validation.py
│   │   ├── vehicle_library.py
│   │   └── fractal_terrain_generator.py
│   ├── analysis/
│   │   ├── analyze_ensemble_results.py
│   │   ├── improve_lidar_correlation.py
│   │   └── verify_all_equations_sympy.py
│   └── validation/
│       ├── complete_real_world_validation.py
│       └── corrected_spectral_validation.py
├── data/
│   ├── simulation_results/
│   │   ├── three_vehicle_validation_results.csv
│   │   ├── vehicle_ensemble_results.csv
│   │   └── advanced_spectral_results.csv
│   └── real_world/
│       ├── lidar_validation_results.csv
│       └── liracd_vehicle_data.csv
└── README.md
```

---

## Installation

### Requirements
- Python 3.9+
- Required Python packages (see `requirements.txt`)

### Setup

```bash
# Clone the repository
git clone https://github.com/GeoGizmodo/fractalFatigueGeo.git
cd fractalFatigueGeo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=0.24.0
sympy>=1.8
```

---

## Quick Start

### Run Monte Carlo Validation

```python
from code.simulations import sim05_vehicle_ensemble_validation

# Run ensemble validation (100 vehicles, 18,000 simulations)
results, vehicles = sim05_vehicle_ensemble_validation.run_ensemble_validation()

# Analyze and plot results
sim05_vehicle_ensemble_validation.analyze_and_plot_results(results, vehicles)
```

### Verify Equations

```python
from code.analysis import verify_all_equations_sympy

# Symbolically verify all manuscript equations
verify_all_equations_sympy.verify_all_equations()
```

### Analyze Real-World Data

```python
from code.validation import complete_real_world_validation

# Run LiDAR and vehicle sensor validation
complete_real_world_validation.run_complete_validation()
```

---

## Data

### Simulation Data

All simulation results are provided in `data/simulation_results/`:

- `three_vehicle_validation_results.csv` - 1500 terrain realizations
- `vehicle_ensemble_results.csv` - 100 vehicles, 18,000 simulations
- `advanced_spectral_results.csv` - Detailed spectral analysis

### Real-World Data

Real-world validation data in `data/real_world/`:

- `lidar_validation_results.csv` - 13 USGS 3DEP LiDAR terrain regions
- `liracd_vehicle_data.csv` - 8,609 road segments from LiRA-CD dataset

### Data Availability

- **Simulation code**: Included in this repository (MIT License)
- **LiDAR data**: USGS 3DEP (public domain)
- **Vehicle data**: LiRA-CD dataset (open access)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{zare2026fractal,
  title={Two-Parameter Spectral Framework for Predicting Terrain-Induced Mechanical Fatigue},
  author={Zare, Alex and Meredith, Stanislava and Hariharan, Aneesh},
  journal={Submitted for Publication},
  year={2026},
  institution={GeoGizmodo LLC}
}
```

### Key References

1. **Self-Affine Surface Theory**: Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*
2. **Vehicle Dynamics**: Wong, J. Y. (2022). *Theory of Ground Vehicles* (5th ed.)
3. **Fatigue Mechanics**: Dowling, N. E. (2013). *Mechanical Behavior of Materials* (4th ed.)
4. **Random Vibration**: Bendat, J. S. & Piersol, A. G. (2010). *Random Data* (4th ed.)

---

## Authors

**Alex Zare**  
GeoGizmodo LLC Research Division  
Tacoma, Washington, USA

**Stanislava Meredith**  
GeoGizmodo LLC Product Division  
Tacoma, Washington, USA

**Aneesh Hariharan** (Corresponding Author)  
GeoGizmodo LLC Research Division  
Tacoma, Washington, USA  
Email: hello@geogizmodo.ai

---

## Funding

This work was supported by the United States Air Force SBIR Program:

**Contract**: FA860425CB079  
**Program**: Small Business Innovation Research (SBIR)  
**Topic**: Predictive Maintenance for Ground Vehicles

---

## License

All simulation and analysis code is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 GeoGizmodo LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions, collaborations, or access to additional data:

**Email**: hello@geogizmodo.ai  
**Website**: https://geogizmodo.ai  
**GitHub**: https://github.com/GeoGizmodo

---

## Acknowledgments

- **USAF SBIR Program** for funding support
- **USGS 3DEP** for LiDAR terrain data
- **LiRA-CD Project** for vehicle sensor data
- **Open-source community** for scientific Python tools

---

## Disclaimer

This work was supported by the United States Air Force under Contract FA860425CB079. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the United States Air Force or the U.S. Government.

---

**Last Updated**: March 12, 2026  
**Repository**: https://github.com/GeoGizmodo/fractalFatigueGeo

---

<div align="center">

**GeoGizmodo Research Division**

[Website](https://geogizmodo.ai) • [Email](mailto:hello@geogizmodo.ai) • [GitHub](https://github.com/GeoGizmodo)

</div>

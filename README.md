# Spectral Scaling of Natural-Terrain-Induced Fatigue

Code and data repository for reproducing all results in the manuscript.

## 1. System Requirements

### Software Dependencies
- Python 3.8 or higher
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Pandas >= 2.0.0
- Requests >= 2.28.0 (for USGS DEM downloads)
- Pillow >= 9.0.0 (for DEM image processing)
- openpyxl >= 3.1.0 (for Source Data Excel generation)

### Operating Systems
Tested on:
- Windows 10/11 (Python 3.9, Anaconda)
- Ubuntu 22.04 (Python 3.10)

### Non-standard Hardware
None required. All computations run on standard desktop hardware.

## 2. Installation Guide

### Instructions
```bash
git clone https://anonymous.4open.science/r/fractalFatigueGeo-4834/
cd fractalFatigueGeo
pip install -r requirements.txt
```

### Typical Install Time
< 2 minutes on a normal desktop computer (dependencies only).

## 3. Demo

### Instructions to Run
```bash
# Generate all main figures (uses pre-computed CSV data)
python generate_figure1.py          # Figure 1: Theoretical framework
python generate_figure3.py          # Figure 3: Energy scaling + variance decomposition
python regenerate_figure6.py        # Figure 6: Domain boundary (USGS + Copenhagen)
python regenerate_tartandrive_figure.py  # Supp Figure 5: TartanDrive validation

# Run the constant-amplitude decoupling experiment (Supp Figure 4)
python sim_constant_amplitude_decoupling.py

# Run the USGS 25-region terrain validation (Figure 6a)
# NOTE: requires internet connection for DEM downloads on first run
python expand_usgs_validation.py
```

### Expected Output
- `figures/Figure1.png` — 4-panel theoretical framework
- `figures/Figure3.png` — Energy vs D scatter + variance decomposition
- `figures/Figure6.png` — Domain boundary comparison (USGS vs Copenhagen)
- `figures/tartandrive_validation.png` — TartanDrive 4-panel validation
- `figures/constant_amplitude_decoupling.png` — Decoupling experiment results
- `figures/usgs_25region_validation.png` — USGS 25-region scatter
- `constant_amplitude_results.csv` — 449-simulation decoupling data
- `expanded_terrain_validation.csv` — 25-region USGS results

### Expected Run Time (Demo)
- Figure generation from pre-computed CSVs: < 10 seconds each
- `sim_constant_amplitude_decoupling.py`: ~2–5 minutes (450 vehicle simulations)
- `expand_usgs_validation.py`: ~5–10 minutes (downloads 25 DEM tiles from USGS; cached after first run)

## 4. Instructions for Use

### Running on Your Own Data

**To analyze a new terrain DEM:**
1. Prepare a GeoTIFF or numpy array of elevation data
2. Extract 1D profiles (rows or along-track)
3. Use functions from `expand_usgs_validation.py`:
   - `fractal_dim_1d(profile, dx)` — computes variogram-based D₁
   - `psd_slope(profile, dx)` — computes Welch PSD spectral exponent β_t

**To analyze new vehicle IMU data:**
1. Prepare vertical acceleration time series (≥ 100 Hz recommended)
2. Use functions from `extract_and_analyze_tartandrive.py`:
   - `compute_vibration_energy(imu_z, fs)` — returns E and β_a
   - `compute_terrain_spectral_slope(profile)` — returns β_t from heightmap

### Reproduction Instructions

To reproduce ALL quantitative results in the manuscript:

```bash
# Step 1: Generate simulation data (if CSVs not present)
# The pre-computed CSVs are included in the repository.
# To regenerate from scratch, run the original simulation scripts
# (requires the full simulation codebase in code/ directory)

# Step 2: Run USGS validation (downloads real terrain data)
python expand_usgs_validation.py

# Step 3: Generate all figures
python generate_figure1.py
python generate_figure3.py
python regenerate_figure6.py
python sim_constant_amplitude_decoupling.py
python regenerate_tartandrive_figure.py

# Step 4: Verify statistics
python check_usgs_stats.py

# Step 5: Generate Source Data Excel
pip install openpyxl
python create_source_data.py
```

## Data Files

| File | Description | n |
|------|-------------|---|
| `three_vehicle_validation_results.csv` | 1500-simulation results (3 vehicles × 5 D × 100 realizations) | 1500 |
| `constant_amplitude_results.csv` | Constant-amplitude decoupling experiment | 449 |
| `expanded_terrain_validation.csv` | USGS 3DEP 25-region terrain analysis | 25 |
| `tartandrive_validation_results.csv` | TartanDrive off-road ATV segments | 42 |
| `dem_cache/*.npy` | Cached USGS DEM tiles (25 regions) | 25 |

## Code Description

The computational pipeline implements:
1. **Fractal terrain generation** — Diamond-Square algorithm with target fractal dimension D
2. **Vehicle dynamics** — 2-DOF quarter-car model, 4th-order Runge-Kutta integration (Δt = 0.001s)
3. **Spectral analysis** — Welch PSD estimation, log-log regression for β extraction
4. **Fatigue analysis** — Rainflow cycle counting, Basquin's law (S-N curve), Miner's rule
5. **Terrain validation** — USGS 3DEP DEM download, variogram-based D₁ estimation
6. **Vehicle validation** — TartanDrive rosbag IMU extraction, segment-level analysis

Detailed algorithmic description is provided in the Methods section and Supplementary Notes 1–13 of the manuscript.

## License

MIT License

## Citation

[To be added upon publication]

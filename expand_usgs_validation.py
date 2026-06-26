#!/usr/bin/env python3
"""
USGS 3DEP Validation: 25 Class-Balanced Regions
Uses direct USGS REST API (no py3dep dependency issues).

pip install requests numpy scipy pandas matplotlib Pillow
python expand_usgs_validation.py
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress, pearsonr, spearmanr
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os, argparse, warnings
warnings.filterwarnings('ignore')

REGIONS = {
    "Kansas_Plains": {"bbox":(-99.0,38.5,-98.5,39.0),"class":"smooth_flat","class_id":1},
    "Iowa_Farmland": {"bbox":(-93.5,41.8,-93.0,42.3),"class":"smooth_flat","class_id":1},
    "Nebraska_Sandhills": {"bbox":(-101.5,41.5,-101.0,42.0),"class":"smooth_flat","class_id":1},
    "Florida_Panhandle": {"bbox":(-85.5,30.3,-85.0,30.8),"class":"smooth_flat","class_id":1},
    "Texas_Permian": {"bbox":(-102.5,31.5,-102.0,32.0),"class":"smooth_flat","class_id":1},
    "Oregon_Coast": {"bbox":(-123.8,44.0,-123.3,44.5),"class":"rolling","class_id":2},
    "Virginia_Piedmont": {"bbox":(-78.5,37.5,-78.0,38.0),"class":"rolling","class_id":2},
    "Missouri_Ozarks": {"bbox":(-91.5,37.0,-91.0,37.5),"class":"rolling","class_id":2},
    "Wisconsin_Driftless": {"bbox":(-90.5,43.0,-90.0,43.5),"class":"rolling","class_id":2},
    "Tennessee_Valley": {"bbox":(-85.5,35.0,-85.0,35.5),"class":"rolling","class_id":2},
    "Appalachian_Ridge": {"bbox":(-79.5,38.0,-79.0,38.5),"class":"rocky","class_id":3},
    "Sedona_Arizona": {"bbox":(-111.9,34.7,-111.4,35.2),"class":"rocky","class_id":3},
    "Kentucky_Karst": {"bbox":(-86.5,37.0,-86.0,37.5),"class":"rocky","class_id":3},
    "Black_Hills_SD": {"bbox":(-103.8,43.8,-103.3,44.3),"class":"rocky","class_id":3},
    "Bryce_Canyon_UT": {"bbox":(-112.3,37.5,-111.8,38.0),"class":"rocky","class_id":3},
    "Badlands_SD": {"bbox":(-102.0,43.5,-101.5,44.0),"class":"rough","class_id":4},
    "Cascade_Range_OR": {"bbox":(-121.8,44.0,-121.3,44.5),"class":"rough","class_id":4},
    "Death_Valley_CA": {"bbox":(-117.0,36.0,-116.5,36.5),"class":"rough","class_id":4},
    "Bighorn_Mtns_WY": {"bbox":(-107.5,44.3,-107.0,44.8),"class":"rough","class_id":4},
    "White_Sands_NM": {"bbox":(-106.5,32.7,-106.0,33.2),"class":"rough","class_id":4},
    "Grand_Canyon_AZ": {"bbox":(-112.2,36.0,-111.7,36.5),"class":"very_rough","class_id":5},
    "Glacier_NP_MT": {"bbox":(-114.0,48.5,-113.5,49.0),"class":"very_rough","class_id":5},
    "Colorado_Rockies": {"bbox":(-106.0,39.5,-105.5,40.0),"class":"very_rough","class_id":5},
    "Sierra_Nevada_CA": {"bbox":(-119.0,37.5,-118.5,38.0),"class":"very_rough","class_id":5},
    "North_Cascades_WA": {"bbox":(-121.5,48.3,-121.0,48.8),"class":"very_rough","class_id":5},
}


def download_dem(name, bbox, cache_dir="dem_cache"):
    """Download DEM from USGS 3DEP ImageServer REST API."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{name}.npy")
    if os.path.exists(cache_file):
        print(f"  [cached]", end=" ")
        return np.load(cache_file)
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    xmin, ymin, xmax, ymax = bbox
    lat_mid = (ymin + ymax) / 2
    m_per_deg_lon = 111000 * np.cos(np.radians(lat_mid))
    width = int((xmax - xmin) * m_per_deg_lon / 10)
    height = int((ymax - ymin) * 111000 / 10)
    cap = 2000
    if max(width, height) > cap:
        s = cap / max(width, height)
        width, height = int(width*s), int(height*s)
    params = {"bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": "4326",
              "imageSR": "4326", "size": f"{width},{height}",
              "format": "tiff", "pixelType": "F32",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    try:
        print(f"  [downloading {width}x{height}]...", end=" ", flush=True)
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        from PIL import Image
        img = Image.open(BytesIO(r.content))
        elev = np.array(img, dtype=np.float32)
        elev[(elev < -1000) | (elev > 10000)] = np.nan
        np.save(cache_file, elev)
        print(f"OK {elev.shape}")
        return elev
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def fractal_dim_1d(profile, dx=10.0):
    """Variogram-based D1. Returns (D1, R2)."""
    p = profile[np.isfinite(profile)]
    if len(p) < 100:
        return np.nan, np.nan
    p = p - p.mean()
    ml = len(p) // 4
    lags = np.unique(np.logspace(0, np.log10(ml), 20).astype(int))
    lags = lags[(lags > 0) & (lags < len(p))]
    vs = []
    for lag in lags:
        d = p[lag:] - p[:-lag]
        if len(d) > 10:
            vs.append(np.var(d))
        else:
            vs.append(np.nan)
    vs = np.array(vs)
    ok = np.isfinite(vs) & (vs > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    lx = np.log10(lags[ok].astype(float) * dx)
    ly = np.log10(vs[ok])
    sl, _, rv, _, _ = linregress(lx, ly)
    return 2.0 - sl/2.0, rv**2


def psd_slope(profile, dx=10.0):
    """Welch PSD slope. Returns (beta, R2)."""
    p = profile[np.isfinite(profile)]
    if len(p) < 64:
        return np.nan, np.nan
    p = signal.detrend(p)
    ns = min(256, len(p)//4)
    if ns < 32:
        ns = len(p)//2
    f, pxx = signal.welch(p, fs=1.0/dx, nperseg=ns)
    m = f > 0
    f, pxx = f[m], pxx[m]
    if len(f) < 5:
        return np.nan, np.nan
    lf = np.log10(f)
    lp = np.log10(pxx)
    ok = np.isfinite(lf) & np.isfinite(lp) & (pxx > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    sl, _, rv, _, _ = linregress(lf[ok], lp[ok])
    return -sl, rv**2


def analyze_region(name, info, n_prof=50):
    """Full analysis of one region."""
    elev = download_dem(name, info["bbox"])
    if elev is None:
        return None
    vf = np.isfinite(elev).sum() / elev.size
    if vf < 0.5:
        print(f"  skip ({vf:.0%} valid)")
        return None
    nr, nc = elev.shape
    if nr < 50 or nc < 50:
        print(f"  skip (too small {nr}x{nc})")
        return None
    rows = np.linspace(nr//10, nr - nr//10, n_prof, dtype=int)
    d1s, betas = [], []
    dx = ((info["bbox"][2]-info["bbox"][0]) * 111000 *
          np.cos(np.radians((info["bbox"][1]+info["bbox"][3])/2))) / nc
    for ri in rows:
        prof = elev[ri, :]
        ok = np.isfinite(prof)
        if ok.sum() < nc * 0.7:
            continue
        if (~ok).any():
            x = np.arange(len(prof))
            prof = np.interp(x, x[ok], prof[ok])
        d1, d1r2 = fractal_dim_1d(prof, dx=dx)
        b, br2 = psd_slope(prof, dx=dx)
        if (np.isfinite(d1) and np.isfinite(b) and
            d1r2 > 0.7 and br2 > 0.6 and 1.0 <= d1 <= 2.0 and 0.5 <= b <= 5.0):
            d1s.append(d1)
            betas.append(b)
    if len(d1s) < 5:
        print(f"  insufficient ({len(d1s)} profiles)")
        return None
    ev = elev[np.isfinite(elev)]
    res = {"region": name, "terrain_class": info["class"], "class_id": info["class_id"],
           "D1_mean": np.mean(d1s), "D1_std": np.std(d1s),
           "beta_mean": np.mean(betas), "beta_std": np.std(betas),
           "n_profiles": len(d1s), "elev_std": np.std(ev), "elev_range": np.ptp(ev)}
    print(f"  D1={res['D1_mean']:.3f} beta={res['beta_mean']:.3f} n={res['n_profiles']}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='expanded_terrain_validation.csv')
    args = parser.parse_args()

    print("="*60)
    print(f"USGS 3DEP VALIDATION: {len(REGIONS)} regions")
    print("="*60)

    results = []
    for i, (name, info) in enumerate(REGIONS.items(), 1):
        print(f"\n[{i}/{len(REGIONS)}] {name} ({info['class']}):", end="")
        r = analyze_region(name, info)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n\nSaved: {args.output} ({len(df)} regions)")

    if len(df) < 10:
        print("WARNING: Too few regions for statistics.")
        return

    # Statistics
    D1 = df["D1_mean"].values
    beta = df["beta_mean"].values
    bs = df["beta_std"].values
    sl, ic, rv, pv, _ = linregress(D1, beta)
    rp, pp = pearsonr(D1, beta)
    rho, ps = spearmanr(D1, beta)
    w = 1.0/(bs**2 + 0.01); w = w/w.sum()
    dc = D1 - np.average(D1, weights=w)
    bc = beta - np.average(beta, weights=w)
    rw = np.sum(w*dc*bc) / np.sqrt(np.sum(w*dc**2)*np.sum(w*bc**2))

    print(f"\n{'='*60}")
    print(f"RESULTS (n={len(df)})")
    print(f"{'='*60}")
    print(f"OLS: beta = {sl:.2f}*D1 + {ic:.2f} (R2={rv**2:.3f})")
    print(f"Pearson: r = {rp:.3f}, p = {pp:.4f}")
    print(f"Spearman: rho = {rho:.3f}, p = {ps:.4f}")
    print(f"Weighted: r = {rw:.3f}")

    # Figure
    os.makedirs("figures", exist_ok=True)
    cols = {1:'#2196F3',2:'#4CAF50',3:'#FF9800',4:'#F44336',5:'#9C27B0'}
    labs = {1:'Smooth/flat',2:'Rolling',3:'Rocky',4:'Rough',5:'Very rough'}
    fig, ax = plt.subplots(figsize=(8,6))
    for c in sorted(df["class_id"].unique()):
        s = df[df["class_id"]==c]
        ax.scatter(s["D1_mean"], s["beta_mean"], c=cols[c], label=labs[c],
                  s=80, edgecolors='k', linewidth=0.5, zorder=3)
        ax.errorbar(s["D1_mean"], s["beta_mean"], xerr=s["D1_std"],
                   yerr=s["beta_std"], fmt='none', color=cols[c], alpha=0.3)
    xr = np.linspace(D1.min()-0.05, D1.max()+0.05, 100)
    ax.plot(xr, sl*xr+ic, 'k-', lw=2, label=f'OLS (r={rp:.3f})')
    ax.plot(xr, 7-2*xr, 'r--', lw=1.5, alpha=0.7, label='Theory')
    ax.set_xlabel('Profile Fractal Dimension D1')
    ax.set_ylabel('Spectral Exponent beta_t')
    ax.set_title(f'USGS 3DEP (n={len(df)}): rho={rho:.3f}, p={ps:.4f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/usgs_25region_validation.png", dpi=300, bbox_inches='tight')
    print("Figure: figures/usgs_25region_validation.png")
    plt.close()


if __name__ == '__main__':
    main()

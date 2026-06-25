#!/usr/bin/env python3
"""
Re-analyze cached USGS DEMs with multiple D estimation methods.
Requires: expanded_terrain_validation already run (dem_cache/ populated).

Methods tested:
  A) Variogram (current) - D1 = 2 - H where var(h) ~ h^(2H)
  B) 1D box-counting - count boxes covering the profile graph
  C) PSD-derived D - D1 = (5 - beta) / 2 from the PSD slope directly
  D) Detrended Fluctuation Analysis (DFA) - F(n) ~ n^H
  E) Roughness-Length method - RMS height vs window size

python reanalyze_usgs_methods.py
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress, pearsonr, spearmanr
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

# Load region definitions from the main script
from expand_usgs_validation import REGIONS


def load_cached_dem(name):
    """Load a cached DEM numpy file."""
    f = os.path.join("dem_cache", f"{name}.npy")
    if os.path.exists(f):
        return np.load(f)
    return None


# ============================================================
# METHOD A: Variogram (same as before)
# ============================================================
def method_variogram(profile, dx=10.0):
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


# ============================================================
# METHOD B: 1D Box-Counting
# ============================================================
def method_boxcount(profile, dx=10.0):
    """1D box-counting: cover the profile graph with boxes of size eps."""
    p = profile[np.isfinite(profile)]
    if len(p) < 64:
        return np.nan, np.nan
    # Normalize to [0, 1] range
    pmin, pmax = p.min(), p.max()
    if pmax - pmin < 1e-6:
        return np.nan, np.nan
    pn = (p - pmin) / (pmax - pmin)
    n = len(pn)
    # Box sizes (in terms of x-index spacing)
    sizes = np.unique(np.logspace(0, np.log10(n/4), 15).astype(int))
    sizes = sizes[sizes >= 2]
    counts = []
    valid_sizes = []
    for s in sizes:
        nboxes = 0
        for i in range(0, n, s):
            chunk = pn[i:i+s]
            if len(chunk) == 0:
                continue
            # Number of vertical boxes needed to cover this chunk
            cmin, cmax = chunk.min(), chunk.max()
            # Scale vertical to same units as horizontal
            vrange = (cmax - cmin) * n  # relative to full height in box units
            nboxes += max(1, int(np.ceil(vrange / s)) + 1)
        if nboxes > 0:
            counts.append(nboxes)
            valid_sizes.append(s)
    if len(counts) < 5:
        return np.nan, np.nan
    lx = np.log10(np.array(valid_sizes, dtype=float))
    ly = np.log10(np.array(counts, dtype=float))
    ok = np.isfinite(lx) & np.isfinite(ly)
    if ok.sum() < 5:
        return np.nan, np.nan
    sl, _, rv, _, _ = linregress(lx[ok], ly[ok])
    D1 = -sl  # box-counting dimension
    return D1, rv**2


# ============================================================
# METHOD C: PSD-derived D (D1 = (5 - beta) / 2)
# ============================================================
def method_psd_derived(profile, dx=10.0):
    """Derive D1 directly from PSD slope: D1 = (5 - beta) / 2."""
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
    beta = -sl
    D1 = (5.0 - beta) / 2.0  # From beta = 2H+1 = 2(2-D1)+1 = 5-2D1
    return D1, rv**2


# ============================================================
# METHOD D: Detrended Fluctuation Analysis (DFA)
# ============================================================
def method_dfa(profile, dx=10.0):
    """DFA: F(n) ~ n^H, D1 = 2 - H."""
    p = profile[np.isfinite(profile)]
    if len(p) < 100:
        return np.nan, np.nan
    # Cumulative sum (integration)
    y = np.cumsum(p - p.mean())
    N = len(y)
    # Window sizes
    sizes = np.unique(np.logspace(1, np.log10(N//4), 15).astype(int))
    sizes = sizes[(sizes >= 10) & (sizes < N//2)]
    flucts = []
    valid_sizes = []
    for s in sizes:
        # Divide into windows
        n_windows = N // s
        if n_windows < 2:
            continue
        rms_vals = []
        for w in range(n_windows):
            segment = y[w*s:(w+1)*s]
            # Linear detrend within window
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_vals.append(np.sqrt(np.mean((segment - trend)**2)))
        if len(rms_vals) > 0:
            flucts.append(np.mean(rms_vals))
            valid_sizes.append(s)
    if len(flucts) < 5:
        return np.nan, np.nan
    lx = np.log10(np.array(valid_sizes, dtype=float))
    ly = np.log10(np.array(flucts))
    ok = np.isfinite(lx) & np.isfinite(ly)
    if ok.sum() < 5:
        return np.nan, np.nan
    sl, _, rv, _, _ = linregress(lx[ok], ly[ok])
    H = sl  # DFA exponent = Hurst exponent
    D1 = 2.0 - H
    return D1, rv**2


# ============================================================
# METHOD E: Roughness-Length (RMS height vs window)
# ============================================================
def method_roughness_length(profile, dx=10.0):
    """RMS roughness vs. measurement length: sigma(L) ~ L^H."""
    p = profile[np.isfinite(profile)]
    if len(p) < 100:
        return np.nan, np.nan
    p = p - p.mean()
    N = len(p)
    sizes = np.unique(np.logspace(1, np.log10(N//2), 15).astype(int))
    sizes = sizes[(sizes >= 10) & (sizes < N)]
    rms_vals = []
    valid_sizes = []
    for s in sizes:
        n_win = N // s
        if n_win < 3:
            continue
        sigmas = []
        for w in range(n_win):
            chunk = p[w*s:(w+1)*s]
            # Detrend each window
            x = np.arange(len(chunk))
            if len(chunk) > 2:
                coeffs = np.polyfit(x, chunk, 1)
                chunk = chunk - np.polyval(coeffs, x)
            sigmas.append(np.std(chunk))
        if len(sigmas) > 0:
            rms_vals.append(np.mean(sigmas))
            valid_sizes.append(s * dx)
    if len(rms_vals) < 5:
        return np.nan, np.nan
    lx = np.log10(np.array(valid_sizes))
    ly = np.log10(np.array(rms_vals))
    ok = np.isfinite(lx) & np.isfinite(ly) & (np.array(rms_vals) > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    sl, _, rv, _, _ = linregress(lx[ok], ly[ok])
    H = sl  # slope = Hurst exponent
    D1 = 2.0 - H
    return D1, rv**2


# ============================================================
# PSD slope (same for all methods)
# ============================================================
def psd_slope(profile, dx=10.0):
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


# ============================================================
# MAIN
# ============================================================
METHODS = {
    "variogram": method_variogram,
    "boxcount": method_boxcount,
    "psd_derived": method_psd_derived,
    "dfa": method_dfa,
    "roughness_length": method_roughness_length,
}

def analyze_all():
    print("="*70)
    print("MULTI-METHOD D ESTIMATION ON 25 USGS REGIONS")
    print("="*70)

    all_results = []

    for name, info in REGIONS.items():
        elev = load_cached_dem(name)
        if elev is None:
            print(f"  {name}: no cached DEM, skipping")
            continue

        nr, nc = elev.shape
        bbox = info["bbox"]
        dx = ((bbox[2]-bbox[0]) * 111000 *
              np.cos(np.radians((bbox[1]+bbox[3])/2))) / nc

        # Extract profiles
        rows = np.linspace(nr//10, nr - nr//10, 50, dtype=int)
        profiles = []
        for ri in rows:
            prof = elev[ri, :]
            ok = np.isfinite(prof)
            if ok.sum() < nc * 0.7:
                continue
            if (~ok).any():
                x = np.arange(len(prof))
                prof = np.interp(x, x[ok], prof[ok])
            profiles.append(prof)

        if len(profiles) < 10:
            continue

        # Compute beta (same for all methods)
        betas = []
        for prof in profiles:
            b, r2 = psd_slope(prof, dx=dx)
            if np.isfinite(b) and r2 > 0.6 and 0.5 < b < 6.0:
                betas.append(b)
        beta_mean = np.mean(betas) if betas else np.nan
        beta_std = np.std(betas) if betas else np.nan

        # Compute D1 with each method
        row = {"region": name, "class": info["class"], "class_id": info["class_id"],
               "beta_mean": beta_mean, "beta_std": beta_std}

        for mname, mfunc in METHODS.items():
            d1_vals = []
            for prof in profiles:
                d1, r2 = mfunc(prof, dx=dx)
                if np.isfinite(d1) and r2 > 0.6 and 0.5 < d1 < 2.5:
                    d1_vals.append(d1)
            if d1_vals:
                row[f"D1_{mname}_mean"] = np.mean(d1_vals)
                row[f"D1_{mname}_std"] = np.std(d1_vals)
                row[f"D1_{mname}_n"] = len(d1_vals)
            else:
                row[f"D1_{mname}_mean"] = np.nan
                row[f"D1_{mname}_std"] = np.nan
                row[f"D1_{mname}_n"] = 0

        all_results.append(row)
        print(f"  {name}: beta={beta_mean:.3f}", end="")
        for mn in METHODS:
            v = row.get(f"D1_{mn}_mean", np.nan)
            if np.isfinite(v):
                print(f"  {mn[:4]}={v:.3f}", end="")
        print()

    df = pd.DataFrame(all_results)
    df.to_csv("usgs_multimethod_results.csv", index=False)
    print(f"\nSaved: usgs_multimethod_results.csv ({len(df)} regions)")

    # Compare correlations
    print(f"\n{'='*70}")
    print("CORRELATION OF EACH D METHOD WITH beta_t:")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'r':>8} {'p':>10} {'rho':>8} {'p_s':>10} {'n':>5}")
    print("-"*60)

    best_method = None
    best_rho = 0

    for mname in METHODS:
        col = f"D1_{mname}_mean"
        valid = df[col].notna() & df["beta_mean"].notna()
        if valid.sum() < 10:
            print(f"{mname:<20} insufficient data (n={valid.sum()})")
            continue
        d1 = df.loc[valid, col].values
        beta = df.loc[valid, "beta_mean"].values
        rp, pp = pearsonr(d1, beta)
        rho, ps = spearmanr(d1, beta)
        n = valid.sum()
        sig = "*" if ps < 0.05 else ("~" if ps < 0.10 else " ")
        print(f"{mname:<20} {rp:>8.3f} {pp:>10.4f} {rho:>8.3f} {ps:>10.4f} {n:>5} {sig}")

        if abs(rho) > abs(best_rho):
            best_rho = rho
            best_method = mname

    print(f"\nBest method: {best_method} (Spearman rho = {best_rho:.3f})")

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    cols = {1:'#2196F3',2:'#4CAF50',3:'#FF9800',4:'#F44336',5:'#9C27B0'}

    for i, mname in enumerate(METHODS):
        ax = axes[i]
        col = f"D1_{mname}_mean"
        valid = df[col].notna() & df["beta_mean"].notna()
        if valid.sum() < 5:
            ax.set_title(f"{mname}: insufficient data")
            continue
        d1 = df.loc[valid, col].values
        beta = df.loc[valid, "beta_mean"].values
        cids = df.loc[valid, "class_id"].values
        for c in np.unique(cids):
            m = cids == c
            ax.scatter(d1[m], beta[m], c=cols[int(c)], s=60, edgecolors='k', lw=0.5)
        rp, pp = pearsonr(d1, beta)
        rho, ps = spearmanr(d1, beta)
        sl, ic, _, _, _ = linregress(d1, beta)
        xr = np.linspace(d1.min(), d1.max(), 50)
        ax.plot(xr, sl*xr+ic, 'k-', lw=2)
        ax.set_title(f"{mname}\nr={rp:.3f}, rho={rho:.3f}, p={ps:.3f}")
        ax.set_xlabel("D1")
        ax.set_ylabel("beta_t")
        ax.grid(alpha=0.3)

    # Last panel: legend
    axes[5].axis('off')
    labs = {1:'Smooth',2:'Rolling',3:'Rocky',4:'Rough',5:'V.Rough'}
    for c, lab in labs.items():
        axes[5].scatter([], [], c=cols[c], s=80, label=lab, edgecolors='k')
    axes[5].legend(fontsize=12, loc='center')

    plt.tight_layout()
    plt.savefig("figures/usgs_method_comparison.png", dpi=200, bbox_inches='tight')
    print(f"\nFigure: figures/usgs_method_comparison.png")
    plt.close()


if __name__ == '__main__':
    analyze_all()

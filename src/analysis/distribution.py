"""
distribution.py
---------------
Fit log-normal, gamma, Pearson Type III, and SHASH distributions to 7-day
mean precipitation at every grid point.  Compute goodness-of-fit metrics and
identify the best-fitting distribution via AIC.

Scientific context
------------------
All four distributions are candidates for positive-definite, right-skewed
precipitation data.  AIC is the primary comparison metric (penalises
complexity by the number of free parameters).

Distribution summary
--------------------
  Log-normal  : 2 free params (shape s, scale); floc=0 (precipitation ≥ 0)
  Gamma       : 2 free params (shape a, scale); floc=0
  Pearson III : 3 free params (skew, loc, scale); loc is free (standard
                hydrological convention — lower bound can be negative)
  SHASH       : 3 free params (ε, τ, scale); floc=0
                Sinh-Arcsinh Normal (Jones & Pewsey 2009, Biometrika 96 761)
                CDF: Φ(sinh(arcsinh((x − loc) / scale) / τ − ε))
                τ (tailweight): τ=1 → normal, τ>1 → heavier than normal,
                                τ<1 → lighter than normal

Output variables per grid point
---------------------------------
  lognorm_s, lognorm_scale         log-normal params
  gamma_a, gamma_scale             gamma params
  pearson3_skew, _loc, _scale      Pearson III params
  shash_eps, shash_tailweight, _scale   SHASH params  (tailweight τ = 1/δ_J&P)
  ks_{dist}, pval_{dist}           KS statistic and p-value (all 4 dists)
  delta_aic                        AIC_lognorm − AIC_gamma  (2-way; backward compat)
  best_fit                         0 = log-normal, 1 = gamma  (2-way)
  best_fit_4way                    0-3 → lognorm/gamma/pearson3/shash (4-way)
  aic_confidence                   ΔAIC between 2nd-best and best distribution
  n_wetdays                        wet-day sample count per grid point
"""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


# ---------------------------------------------------------------------------
# SHASH distribution  (Jones & Pewsey 2009)
# ---------------------------------------------------------------------------

class _SHASHDist(stats.rv_continuous):
    """
    Sinh-Arcsinh Normal (SHASH) — Jones & Pewsey (2009), Biometrika 96 761.
    Geosciences / TensorFlow convention: tailweight τ = 1 / δ_J&P.

    Shape params:
      eps        (ε) — skewness  (any real; ε=0 → symmetric)
      tailweight (τ) — tail weight (τ > 0)
                       τ = 1 → normal (when ε = 0)
                       τ > 1 → heavier tails than normal (leptokurtic)
                       τ < 1 → lighter tails than normal (platykurtic)

    Standard (loc=0, scale=1) CDF:
      F(x) = Φ(sinh(arcsinh(x) / τ − ε))
    """

    def _pdf(self, x, eps, tailweight):
        u = np.arcsinh(x)
        z = np.sinh(u / tailweight - eps)
        return (np.cosh(u / tailweight - eps)
                / (tailweight * np.sqrt(1.0 + x ** 2))
                * stats.norm.pdf(z))

    def _cdf(self, x, eps, tailweight):
        return stats.norm.cdf(np.sinh(np.arcsinh(x) / tailweight - eps))

    def _argcheck(self, eps, tailweight):
        return tailweight > 0


_shash = _SHASHDist(name="shash")


# ---------------------------------------------------------------------------
# Fast SHASH MLE — direct numpy, bypasses rv_continuous overhead
# ---------------------------------------------------------------------------

def _shash_nloglik(params: np.ndarray, data: np.ndarray) -> float:
    """
    Negative log-likelihood for SHASH(eps, tailweight, scale) with loc=0.

    Jones & Pewsey (2009) formula, geosciences τ convention:
      F(x) = Φ(sinh(arcsinh(x/scale) / τ − ε))

    Parameters: [eps, log(τ), log(scale)] so τ > 0 and scale > 0.
    """
    eps, log_tailweight, log_scale = params
    tailweight = np.exp(log_tailweight)
    scale = np.exp(log_scale)
    x = data / scale                            # standardised x
    u = np.arcsinh(x)                           # arcsinh(x/scale)
    v = u / tailweight - eps                    # arcsinh(x/scale)/τ − ε
    z = np.sinh(v)                              # normal variate
    # Numerically stable log|cosh(v)|
    log_cosh_v = np.abs(v) + np.log1p(np.exp(-2.0 * np.abs(v))) - np.log(2.0)
    # log pdf = −0.5·log(2π) − 0.5·z² + log|cosh(v)| − log(τ) − log(scale) − 0.5·log(1+x̃²)
    # Use hypot for log(1+x²) to avoid float64 overflow when x = data/scale is large.
    ll = (-0.5 * np.log(2.0 * np.pi)
          - 0.5 * z * z
          + log_cosh_v
          - np.log(tailweight)
          - np.log(scale)
          - np.log(np.hypot(1.0, x)))
    return -np.sum(ll)


def _shash_fit_fast(wet: np.ndarray):
    """
    Fit SHASH via direct numpy MLE.  ~100× faster than rv_continuous.fit().

    Returns (eps, tailweight, scale) or raises ValueError on failure.
    Uses J&P formula with τ (tailweight) convention.
    """
    # Initial guesses: eps≈0.5 (slight right skew), τ≈1.5 (moderate heavy tails),
    # scale ≈ median of data
    x0 = np.array([0.5, np.log(1.5), np.log(np.median(wet))])
    res = minimize(
        _shash_nloglik,
        x0,
        args=(wet,),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5, "disp": False},
    )
    eps, log_tailweight, log_scale = res.x
    tailweight = np.exp(log_tailweight)
    scale = np.exp(log_scale)
    if not res.success and res.fun == _shash_nloglik(x0, wet):
        raise ValueError("SHASH optimisation did not improve on initial guess")
    if tailweight <= 0 or scale <= 0:
        raise ValueError("invalid SHASH params")
    return eps, tailweight, scale


# ---------------------------------------------------------------------------
# Per-row worker  (must be module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

_ROW_KEYS = [
    "lognorm_s", "lognorm_scale", "ks_lognorm", "pval_lognorm",
    "gamma_a", "gamma_scale", "ks_gamma", "pval_gamma",
    "pearson3_skew", "pearson3_loc", "pearson3_scale", "ks_pearson3", "pval_pearson3",
    "shash_eps", "shash_tailweight", "shash_scale", "ks_shash", "pval_shash",
    "delta_aic", "best_fit", "best_fit_4way", "aic_confidence", "n_wetdays",
]


def _fit_lat_row(args):
    """
    Fit all 4 distributions for every longitude in one latitude row.

    Parameters
    ----------
    args : (row_idx, row_data, min_samples)
        row_data : np.ndarray  shape (n_time, n_lon)

    Returns
    -------
    (row_idx, dict)  where each dict value is a 1-D array of length n_lon.
    """
    row_idx, row_data, min_samples = args
    n_lon = row_data.shape[1]

    nan32 = np.full(n_lon, np.nan, dtype=np.float32)
    row = {k: nan32.copy() for k in _ROW_KEYS}
    row["best_fit"]      = np.full(n_lon, -1, dtype=np.int8)
    row["best_fit_4way"] = np.full(n_lon, -1, dtype=np.int8)
    row["n_wetdays"]     = np.zeros(n_lon, dtype=np.int32)

    for j in range(n_lon):
        col = row_data[:, j]
        wet = col[np.isfinite(col) & (col > 0)]
        n = len(wet)
        row["n_wetdays"][j] = n
        if n < min_samples:
            continue

        aics = [np.inf, np.inf, np.inf, np.inf]

        # 1. Log-normal  (floc=0, k=2)
        try:
            s, _, scale_ln = stats.lognorm.fit(wet, floc=0)
            ll_ln = np.sum(stats.lognorm.logpdf(wet, s, loc=0, scale=scale_ln))
            aics[0] = _aic(ll_ln, k=2)
            ks_ln, pv_ln = stats.kstest(wet, "lognorm", args=(s, 0, scale_ln))
            row["lognorm_s"][j]     = s
            row["lognorm_scale"][j] = scale_ln
            row["ks_lognorm"][j]    = ks_ln
            row["pval_lognorm"][j]  = pv_ln
        except Exception:
            pass

        # 2. Gamma  (floc=0, k=2)
        try:
            a, _, scale_gm = stats.gamma.fit(wet, floc=0)
            ll_gm = np.sum(stats.gamma.logpdf(wet, a, loc=0, scale=scale_gm))
            aics[1] = _aic(ll_gm, k=2)
            ks_gm, pv_gm = stats.kstest(wet, "gamma", args=(a, 0, scale_gm))
            row["gamma_a"][j]       = a
            row["gamma_scale"][j]   = scale_gm
            row["ks_gamma"][j]      = ks_gm
            row["pval_gamma"][j]    = pv_gm
        except Exception:
            pass

        # 3. Pearson Type III  (all 3 params free, k=3)
        try:
            skew_p3, loc_p3, scale_p3 = stats.pearson3.fit(wet)
            ll_p3 = np.sum(stats.pearson3.logpdf(wet, skew_p3, loc_p3, scale_p3))
            aics[2] = _aic(ll_p3, k=3)
            ks_p3, pv_p3 = stats.kstest(wet, "pearson3", args=(skew_p3, loc_p3, scale_p3))
            row["pearson3_skew"][j]  = skew_p3
            row["pearson3_loc"][j]   = loc_p3
            row["pearson3_scale"][j] = scale_p3
            row["ks_pearson3"][j]    = ks_p3
            row["pval_pearson3"][j]  = pv_p3
        except Exception:
            pass

        # 4. SHASH  (floc=0, k=3) — fast direct MLE  (J&P formula, τ convention)
        try:
            eps_sh, tailweight_sh, scale_sh = _shash_fit_fast(wet)
            ll_sh = -_shash_nloglik(
                [eps_sh, np.log(tailweight_sh), np.log(scale_sh)], wet
            )
            aics[3] = _aic(ll_sh, k=3)
            ks_sh, pv_sh = stats.kstest(
                wet, _shash.cdf, args=(eps_sh, tailweight_sh, 0.0, scale_sh)
            )
            row["shash_eps"][j]        = eps_sh
            row["shash_tailweight"][j] = tailweight_sh
            row["shash_scale"][j]      = scale_sh
            row["ks_shash"][j]         = ks_sh
            row["pval_shash"][j]       = pv_sh
        except Exception:
            pass

        # --- Determine winners ---
        finite_pairs = [(a, k) for k, a in enumerate(aics) if np.isfinite(a)]
        if not finite_pairs:
            continue

        sorted_aics = sorted(a for a, _ in finite_pairs)
        best_aic_val, best_idx = min(finite_pairs)
        row["best_fit_4way"][j]  = best_idx
        row["aic_confidence"][j] = (
            sorted_aics[1] - sorted_aics[0] if len(sorted_aics) >= 2 else 0.0
        )

        # 2-way lognorm vs gamma (backward compatibility)
        if np.isfinite(aics[0]) and np.isfinite(aics[1]):
            row["delta_aic"][j] = aics[0] - aics[1]
            row["best_fit"][j]  = 0 if aics[0] <= aics[1] else 1

    return row_idx, row


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_distributions(config: dict, dataset: str, n_jobs: int = None) -> xr.Dataset:
    """
    Load the preprocessed file, fit four distributions grid-by-grid, and save.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    dataset : str
        'cpc' or 'imerg'
    n_jobs : int, optional
        Number of parallel worker processes.  Defaults to all logical CPUs.

    Returns
    -------
    xr.Dataset
        Dataset of 2-D result arrays (lat, lon), also saved to output/stats/.
    """
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    project_root = Path(__file__).resolve().parents[2]

    if dataset == "cpc":
        proc_dir = project_root / config["paths"]["processed_cpc"]
    else:
        proc_dir = project_root / config["paths"]["processed_imerg"]

    nc_path = proc_dir / f"{dataset}_7day_wet.nc"
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {nc_path}\n"
            "Run preprocess.run_preprocessing() first."
        )

    print(f"[dist] Loading {nc_path}")
    da = xr.open_dataarray(nc_path)

    lats = da.lat.values
    lons = da.lon.values
    data_np = da.values          # (time, lat, lon)

    min_samples = config["analysis"]["min_samples"]
    n_lat, n_lon = len(lats), len(lons)
    shape = (n_lat, n_lon)

    # --- Allocate output arrays ---
    result = {
        # Log-normal  (k=2)
        "lognorm_s":       np.full(shape, np.nan, dtype=np.float32),
        "lognorm_scale":   np.full(shape, np.nan, dtype=np.float32),
        "ks_lognorm":      np.full(shape, np.nan, dtype=np.float32),
        "pval_lognorm":    np.full(shape, np.nan, dtype=np.float32),
        # Gamma  (k=2)
        "gamma_a":         np.full(shape, np.nan, dtype=np.float32),
        "gamma_scale":     np.full(shape, np.nan, dtype=np.float32),
        "ks_gamma":        np.full(shape, np.nan, dtype=np.float32),
        "pval_gamma":      np.full(shape, np.nan, dtype=np.float32),
        # Pearson Type III  (k=3, loc is free)
        "pearson3_skew":   np.full(shape, np.nan, dtype=np.float32),
        "pearson3_loc":    np.full(shape, np.nan, dtype=np.float32),
        "pearson3_scale":  np.full(shape, np.nan, dtype=np.float32),
        "ks_pearson3":     np.full(shape, np.nan, dtype=np.float32),
        "pval_pearson3":   np.full(shape, np.nan, dtype=np.float32),
        # SHASH  (k=3, floc=0)
        "shash_eps":        np.full(shape, np.nan, dtype=np.float32),
        "shash_tailweight": np.full(shape, np.nan, dtype=np.float32),
        "shash_scale":     np.full(shape, np.nan, dtype=np.float32),
        "ks_shash":        np.full(shape, np.nan, dtype=np.float32),
        "pval_shash":      np.full(shape, np.nan, dtype=np.float32),
        # Comparison metrics
        "delta_aic":       np.full(shape, np.nan, dtype=np.float32),  # lognorm−gamma (2-way)
        "best_fit":        np.full(shape, -1,     dtype=np.int8),      # 0/1 (2-way backward compat)
        "best_fit_4way":   np.full(shape, -1,     dtype=np.int8),      # 0-3 (4-way)
        "aic_confidence":  np.full(shape, np.nan, dtype=np.float32),   # ΔAIC: 2nd-best minus best
        "n_wetdays":       np.full(shape, 0,      dtype=np.int32),
    }

    # --- Parallel grid-point fitting (parallelised over latitude rows) ---
    print(f"[dist] Fitting 4 distributions over {n_lat}×{n_lon} grid points "
          f"({n_jobs} workers) ...")

    row_args = [
        (i, data_np[:, i : i + 1, :].reshape(data_np.shape[0], 1, n_lon), min_samples)
        for i in range(n_lat)
    ]
    # data_np[:, i, :] is (n_time, n_lon); worker expects (n_time, 1, n_lon)
    # Simpler: pass 2-D slice directly, worker indexes [t, j] not [t, 0, j]
    row_args = [
        (i, data_np[:, i, :], min_samples)   # (n_time, n_lon)
        for i in range(n_lat)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        futures = {exe.submit(_fit_lat_row, arg): arg[0] for arg in row_args}
        for future in tqdm(as_completed(futures), total=n_lat, desc="lat rows"):
            row_idx, row = future.result()
            for k in _ROW_KEYS:
                result[k][row_idx, :] = row[k]

    # --- Package into xr.Dataset ---
    coords = {"lat": lats, "lon": lons}
    ds = xr.Dataset(
        {k: xr.DataArray(v, coords=coords, dims=["lat", "lon"]) for k, v in result.items()}
    )
    ds["best_fit"].attrs["legend"]      = "0=log-normal, 1=gamma"
    ds["best_fit_4way"].attrs["legend"] = "0=log-normal, 1=gamma, 2=pearson3, 3=shash"
    ds["delta_aic"].attrs["description"] = (
        "AIC_lognorm - AIC_gamma; negative = log-normal better"
    )
    ds["aic_confidence"].attrs["description"] = (
        "ΔAIC between 2nd-best and best distribution (higher = more decisive)"
    )
    ds.attrs["dataset"]       = dataset
    ds.attrs["wet_threshold"] = config["analysis"]["wet_threshold"]
    ds.attrs["rolling_days"]  = config["analysis"]["rolling_days"]
    ds.attrs["period"]        = f"{config['years']['start']}-{config['years']['end']}"

    # --- Save ---
    out_dir = project_root / config["paths"]["output_stats"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_distribution_stats.nc"
    if out_path.exists():
        out_path.unlink()
    ds.to_netcdf(out_path)
    print(f"[dist] Results saved → {out_path}")

    _print_summary(result, n_lat, n_lon)
    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aic(log_likelihood: float, k: int) -> float:
    """Akaike Information Criterion: 2k - 2·ln(L)."""
    return 2 * k - 2 * log_likelihood


def _print_summary(result: dict, n_lat: int, n_lon: int) -> None:
    """Print a quick 4-way summary of fitting results to stdout."""
    total  = n_lat * n_lon
    fitted = int(np.sum(result["n_wetdays"] >= 1))
    best4  = result["best_fit_4way"]
    names  = ["log-normal", "gamma", "pearson3", "SHASH"]
    counts = [int(np.sum(best4 == i)) for i in range(4)]

    print(f"\n[dist] === Summary ===")
    print(f"  Grid points total : {total}")
    print(f"  Grid points fitted: {fitted} ({100 * fitted / max(total, 1):.1f}%)")
    for name, count in zip(names, counts):
        pct = 100 * count / max(fitted, 1)
        print(f"  Best fit {name:12s}: {count:5d} ({pct:.1f}%)")

    delta = result["delta_aic"]
    valid = delta[np.isfinite(delta)]
    if len(valid):
        print(f"  ΔAIC lognorm−gamma: mean={valid.mean():.2f}, std={valid.std():.2f}")

    conf = result["aic_confidence"]
    valid_conf = conf[np.isfinite(conf)]
    if len(valid_conf):
        print(f"  AIC confidence    : mean={valid_conf.mean():.2f}, std={valid_conf.std():.2f}")
    print(f"  (aic_confidence: how decisively the best distribution wins over 2nd-best)")


def load_stats(config: dict, dataset: str) -> xr.Dataset:
    """Load previously saved distribution stats NetCDF."""
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / config["paths"]["output_stats"] / f"{dataset}_distribution_stats.nc"
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    return xr.open_dataset(path)


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = fit_distributions(cfg, "cpc")
    print(ds)

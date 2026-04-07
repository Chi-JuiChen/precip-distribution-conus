"""
threshold.py
------------
Automatic wet-day threshold selection per grid cell.

Scientific motivation
---------------------
The 7-day mean precipitation at each cell is a zero-inflated mixture:
  - A spike near zero representing dry / trace-precipitation periods
  - A continuous right-skewed distribution (≈ Pearson III) for wet periods

The goal is to find the optimal threshold T* that separates these two
components per cell, enabling a clean 4-output ML prediction target:
  POP (P(precip > T*)),  P3 skewness,  P3 location,  P3 scale

Algorithm (per cell)
--------------------
For each candidate threshold T in THRESHOLDS:
  1. Compute POP = fraction of windows above T
  2. If n_wet < min_samples → skip
  3. Fit Pearson III (loc free) to wet values
  4. Compute KS statistic D (lower = better P3 fit)
Choose T* = argmin(D) among valid thresholds.

Inputs
------
Raw 7-day rolling mean (no threshold applied) — loaded internally from
data/processed/cpc/ by calling preprocess functions, or passed directly.

Outputs (NetCDF)
----------------
  optimal_threshold  mm/day  — chosen T* per cell
  pop                (0–1)   — P(precip > T*)
  p3_skew                    — Pearson III skewness
  p3_loc             mm/day  — Pearson III location
  p3_scale           mm/day  — Pearson III scale
  ks_stat                    — KS statistic at T*
  n_wetdays                  — n windows above T*
  n_total                    — total windows (for POP denominator)

Usage
-----
  from src.analysis.threshold import fit_optimal_thresholds
  ds = fit_optimal_thresholds(config)          # runs + saves
  ds = fit_optimal_thresholds(config, overwrite=False)  # loads cache
"""

import numpy as np
import xarray as xr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
from tqdm import tqdm


# Candidate thresholds (mm/day) to try at each cell
THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

_OUT_VARS = [
    'optimal_threshold', 'pop',
    'p3_skew', 'p3_loc', 'p3_scale',
    'ks_stat', 'n_wetdays', 'n_total',
]


# ---------------------------------------------------------------------------
# Per-cell worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _fit_cell_thresholds(args):
    """
    Try all candidate thresholds for a single grid cell.

    Parameters
    ----------
    args : (i_lat, i_lon, values, thresholds, min_samples, n_total)
        values : 1-D array of 7-day mean precipitation (no NaN masking applied)

    Returns
    -------
    (i_lat, i_lon, result_dict)
    """
    i_lat, i_lon, values, thresholds, min_samples, n_total = args

    nan_row = {v: np.nan for v in _OUT_VARS}
    nan_row['n_total'] = n_total

    # Remove NaN (land/ocean mask) but keep zeros and trace values
    vals = values[np.isfinite(values)]
    if len(vals) < min_samples:
        return i_lat, i_lon, nan_row

    best = {'ks': np.inf, 'threshold': np.nan}
    for T in thresholds:
        wet = vals[vals > T]
        n_wet = len(wet)
        if n_wet < min_samples:
            continue
        try:
            skew, loc, scale = stats.pearson3.fit(wet)
            D, _ = stats.kstest(wet, 'pearson3', args=(skew, loc, scale))
        except Exception:
            continue
        if D < best['ks']:
            best = {
                'ks':        D,
                'threshold': T,
                'pop':       n_wet / n_total,
                'p3_skew':   skew,
                'p3_loc':    loc,
                'p3_scale':  scale,
                'n_wetdays': n_wet,
                'n_total':   n_total,
            }

    if np.isinf(best['ks']):
        return i_lat, i_lon, nan_row

    result = {
        'optimal_threshold': best['threshold'],
        'pop':               best['pop'],
        'p3_skew':           best['p3_skew'],
        'p3_loc':            best['p3_loc'],
        'p3_scale':          best['p3_scale'],
        'ks_stat':           best['ks'],
        'n_wetdays':         best['n_wetdays'],
        'n_total':           best['n_total'],
    }
    return i_lat, i_lon, result


# ---------------------------------------------------------------------------
# Per-latitude row worker (batches cells to reduce IPC overhead)
# ---------------------------------------------------------------------------

def _fit_lat_row_thresholds(args):
    """Fit all cells in one latitude row. Returns (i_lat, list_of_results)."""
    i_lat, row_data, thresholds, min_samples, n_total = args
    n_lon = row_data.shape[1]
    row_results = []
    for i_lon in range(n_lon):
        _, _, res = _fit_cell_thresholds(
            (i_lat, i_lon, row_data[:, i_lon], thresholds, min_samples, n_total)
        )
        row_results.append(res)
    return i_lat, row_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fit_optimal_thresholds(
    config: dict,
    thresholds: list = THRESHOLDS,
    workers: int = 16,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Fit optimal wet-day threshold + P3 parameters at every CPC land cell.

    Parameters
    ----------
    config     : loaded config.yaml
    thresholds : candidate thresholds (mm/day) to try per cell
    workers    : parallel workers
    overwrite  : if False, load cached result if it exists

    Returns
    -------
    xr.Dataset with variables listed in _OUT_VARS
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir  = project_root / config['paths']['output_stats'].replace(
        'unconditional/stats', 'unconditional/stats'
    )
    # Save alongside unconditional stats
    out_dir  = project_root / config['paths']['output_stats']
    out_path = out_dir / 'cpc_optimal_threshold.nc'
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f'[threshold] Loading cached result: {out_path.name}')
        return xr.open_dataset(out_path)

    # ── Load raw 7-day rolling mean (no threshold applied) ────────────────
    # The preprocess pipeline saves only the thresholded file. We re-derive
    # the raw rolling mean from the thresholded file by noting that values
    # below threshold are NaN — for threshold analysis we need to load from
    # the raw CPC annual files instead.
    raw_path = project_root / 'data' / 'processed' / 'cpc' / 'cpc_7day_raw.nc'

    if not raw_path.exists():
        print('[threshold] Raw 7-day file not found — building from annual CPC files ...')
        _build_raw_7day(config, raw_path, project_root)

    print('[threshold] Loading raw 7-day precipitation ...')
    da = xr.open_dataarray(raw_path)

    lats      = da.lat.values
    lons      = da.lon.values
    data_np   = da.values          # (time, lat, lon)
    n_total   = da.sizes['time']
    min_samp  = config['analysis']['min_samples']

    print(f'[threshold] Array shape: {data_np.shape}, '
          f'candidates: {thresholds}, min_samples: {min_samp}')

    # Build row args
    row_args = [
        (i_lat, data_np[:, i_lat, :], thresholds, min_samp, n_total)
        for i_lat in range(len(lats))
    ]

    # ── Parallel fitting ──────────────────────────────────────────────────
    arrays = {v: np.full((len(lats), len(lons)), np.nan) for v in _OUT_VARS}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_lat_row_thresholds, arg): arg[0]
                   for arg in row_args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='  threshold fitting'):
            i_lat, row_results = fut.result()
            for i_lon, res in enumerate(row_results):
                for v in _OUT_VARS:
                    arrays[v][i_lat, i_lon] = res[v]

    # ── Build xr.Dataset ──────────────────────────────────────────────────
    coords = {'lat': lats, 'lon': lons}
    ds = xr.Dataset(
        {v: xr.DataArray(arrays[v], coords=coords, dims=['lat', 'lon'])
         for v in _OUT_VARS}
    )
    ds.attrs['thresholds_tried'] = str(thresholds)
    ds.attrs['min_samples']      = min_samp
    ds.attrs['n_total_windows']  = n_total
    ds.attrs['description']      = (
        'Optimal wet-day threshold + P3 fit per CPC grid cell. '
        'Threshold chosen to minimise Pearson-III KS statistic.'
    )

    ds.to_netcdf(out_path)
    print(f'[threshold] Saved → {out_path}')
    return ds


# ---------------------------------------------------------------------------
# Helper: build raw 7-day rolling mean (no threshold)
# ---------------------------------------------------------------------------

def _build_raw_7day(config: dict, out_path: Path, project_root: Path):
    """
    Load annual CPC files, compute 7-day trailing mean, save without
    threshold masking.  Mirror of preprocess.run_preprocessing but skips
    mask_wet_days().
    """
    from src.data.preprocess import load_cpc, compute_rolling

    print('[threshold] Loading raw CPC annual files ...')
    da_daily = load_cpc(config)           # (time, lat, lon) daily mm/day
    print('[threshold] Computing 7-day rolling mean ...')
    da_7day  = compute_rolling(da_daily, config)   # trailing 7-day mean

    # Drop incomplete leading windows (NaN from rolling)
    da_7day  = da_7day.dropna(dim='time', how='all')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_7day.to_netcdf(out_path)
    print(f'[threshold] Raw 7-day file saved → {out_path}')


# ---------------------------------------------------------------------------
# Convenience: extract single-cell threshold analysis for plotting
# ---------------------------------------------------------------------------

def city_threshold_analysis(
    config: dict,
    lat: float,
    lon: float,
    thresholds: list = THRESHOLDS,
    min_samples: int = 30,
) -> dict:
    """
    Return threshold analysis results for a single lat/lon point.

    Returns
    -------
    dict with keys:
      'values'     : 1-D array of all 7-day means (raw, no threshold)
      'thresholds' : list of candidate thresholds
      'results'    : list of dicts {threshold, pop, p3_skew, p3_loc,
                                    p3_scale, ks_stat, n_wetdays} per T
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / 'data' / 'processed' / 'cpc' / 'cpc_7day_raw.nc'

    if not raw_path.exists():
        _build_raw_7day(config, raw_path, project_root)

    da   = xr.open_dataarray(raw_path)
    ts   = da.sel(lat=lat, lon=lon, method='nearest').values
    vals = ts[np.isfinite(ts)]
    n_total = len(ts)

    results = []
    for T in thresholds:
        wet   = vals[vals > T]
        n_wet = len(wet)
        entry = {'threshold': T, 'pop': n_wet / n_total, 'n_wetdays': n_wet}
        if n_wet >= min_samples:
            try:
                skew, loc, scale = stats.pearson3.fit(wet)
                D, pval          = stats.kstest(wet, 'pearson3',
                                                args=(skew, loc, scale))
                entry.update({'p3_skew': skew, 'p3_loc': loc,
                              'p3_scale': scale, 'ks_stat': D,
                              'ks_pval': pval, 'valid': True})
            except Exception:
                entry['valid'] = False
        else:
            entry['valid'] = False
        results.append(entry)

    return {'values': vals, 'thresholds': thresholds,
            'results': results, 'n_total': n_total}

"""
conditional.py
--------------
Phase 1: label 7-day CPC precipitation windows by conditioning variable and
re-fit 4 distributions (LN, Gamma, P3, SHASH) per stratum.

Three independent conditions (CPC only, as per research design decision D8/D9):
  season  : DJF / MAM / JJA / SON
  enso    : el_nino / neutral / la_nina
  mjo     : phase_1 … phase_8  /  inactive

Scientific basis
----------------
ENSO definition  (decision D10):
  Index     : Niño 3.4 SST anomaly (5°N–5°S, 120–170°W), 3-month running mean (ONI)
  Threshold : ±0.5 °C  (NOAA CPC standard; Johnson et al. 2014)

MJO definition   (decision D11):
  Index     : RMM (Wheeler & Hendon 2004, MWR 132, 1917–1932)
  Active    : RMM amplitude = sqrt(RMM1²+RMM2²) >= 1.0  (unanimous in literature)
  Phases    : all 8 individually (grouped 2s only for low-N discussion)

References
----------
Johnson et al. (2014, Wea. Fcst. 29, 23–38)
Wheeler & Hendon (2004, MWR 132, 1917–1932)
Zheng et al. (2025, GRL 52, e2024GL110925)
Nardi/Baggett et al. (2020, Wea. Fcst. 35, 2179–2198)

Usage
-----
  from src.analysis.conditional import fit_conditional_distributions, load_conditional_stats

  # Fit and save all season strata (CPC)
  results = fit_conditional_distributions(config, dataset='cpc', condition='season')

  # Load previously saved results
  results = load_conditional_stats(config, dataset='cpc', condition='season')

  # results is a dict: {stratum_label: xr.Dataset}
  # Each Dataset has identical structure to the unconditional stats NetCDF,
  # plus two extra attributes: condition_type and stratum_label.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.analysis.distribution import _fit_lat_row, _ROW_KEYS


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3:  "MAM", 4: "MAM", 5: "MAM",
    6:  "JJA", 7: "JJA", 8: "JJA",
    9:  "SON", 10: "SON", 11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
ENSO_ORDER   = ["el_nino", "neutral", "la_nina"]
MJO_ORDER    = ["inactive"] + [f"phase_{i}" for i in range(1, 9)]


def _get_season(month: int) -> str:
    return _MONTH_TO_SEASON[month]


# ---------------------------------------------------------------------------
# Window labelling
# ---------------------------------------------------------------------------

def label_windows(
    da: xr.DataArray,
    condition: str,
    oni_df: pd.DataFrame | None = None,
    rmm_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.Series:
    """
    Assign a condition label to each time step in da.

    Parameters
    ----------
    da        : xr.DataArray with a 'time' dimension (7-day mean windows)
    condition : 'season' | 'enso' | 'mjo'
    oni_df    : output of load_oni() — required for condition='enso'
    rmm_df    : output of load_rmm() — required for condition='mjo'
    config    : loaded config.yaml — used for thresholds

    Returns
    -------
    pd.Series
        Index : same as da.time (as DatetimeIndex)
        Values: string labels, e.g. 'DJF', 'el_nino', 'phase_3', 'inactive'
    """
    times = pd.DatetimeIndex(da.time.values)

    if condition == "season":
        return pd.Series(
            [_get_season(t.month) for t in times],
            index=times,
            name="season",
        )

    elif condition == "enso":
        if oni_df is None:
            raise ValueError("oni_df required for condition='enso'")
        # Match each 7-day window to the ONI for its centre month
        # ONI index is monthly (first day of month); use nearest-month lookup
        oni_monthly = oni_df["enso_phase"].reindex(
            times, method="nearest", tolerance=pd.Timedelta("31D")
        )
        oni_monthly.index = times
        return oni_monthly.rename("enso_phase")

    elif condition == "mjo":
        if rmm_df is None:
            raise ValueError("rmm_df required for condition='mjo'")
        amp_threshold = config["conditional"]["mjo_amp_threshold"] if config else 1.0

        labels = []
        for t in times:
            # Use the centre day of each 7-day window
            centre = t
            if centre in rmm_df.index:
                row = rmm_df.loc[centre]
                if row["mjo_active"] and not np.isnan(row["phase"]):
                    labels.append(f"phase_{int(row['phase'])}")
                else:
                    labels.append("inactive")
            else:
                # Try nearest day within ±3 days
                idx = rmm_df.index.get_indexer([centre], method="nearest")[0]
                if idx >= 0:
                    row = rmm_df.iloc[idx]
                    delta = abs((rmm_df.index[idx] - centre).days)
                    if delta <= 3 and row["mjo_active"] and not np.isnan(row["phase"]):
                        labels.append(f"phase_{int(row['phase'])}")
                    else:
                        labels.append("inactive")
                else:
                    labels.append("inactive")
        return pd.Series(labels, index=times, name="mjo_phase")

    else:
        raise ValueError(f"Unknown condition: {condition!r}. Must be 'season', 'enso', or 'mjo'.")


# ---------------------------------------------------------------------------
# Core fitting (one stratum)
# ---------------------------------------------------------------------------

def _fit_stratum(
    da: xr.DataArray,
    time_mask: np.ndarray,
    config: dict,
    stratum_label: str,
    n_workers: int = 16,
) -> xr.Dataset:
    """
    Fit 4 distributions at every grid cell using only the time steps
    selected by time_mask.  Identical logic to fit_distributions() in
    distribution.py, but operates on a subset of the time dimension.

    Returns an xr.Dataset with the same variables as the unconditional stats,
    plus attributes recording the condition type and stratum label.
    """
    da_sub = da.isel(time=time_mask)
    n_windows = int(time_mask.sum())

    lats = da.lat.values
    lons = da.lon.values
    min_samples = config["analysis"]["min_samples"]

    # Build work units: one per latitude row — must match _fit_lat_row signature
    # args: (row_idx, row_data_2d, min_samples)  shape of row_data: (n_time, n_lon)
    lat_chunks = [
        (i_lat, da_sub.sel(lat=lat).values, min_samples)
        for i_lat, lat in enumerate(lats)
    ]

    # ── parallel fitting ──────────────────────────────────────────────────
    n_lat, n_lon = len(lats), len(lons)
    arrays = {key: np.full((n_lat, n_lon), np.nan) for key in _ROW_KEYS}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_lat_row, chunk): chunk[0] for chunk in lat_chunks}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {stratum_label} (n={n_windows})",
            leave=False,
        ):
            row_idx, row = fut.result()
            for k in _ROW_KEYS:
                arrays[k][row_idx, :] = row[k]

    # ── build xr.Dataset ─────────────────────────────────────────────────
    ds = xr.Dataset(
        {key: xr.DataArray(arr, coords=[da.lat, da.lon], dims=["lat", "lon"])
         for key, arr in arrays.items()},
    )
    ds.attrs["n_windows"]     = n_windows
    ds.attrs["stratum_label"] = stratum_label
    ds.attrs["wet_threshold"] = config["analysis"]["wet_threshold"]
    ds.attrs["min_samples"]   = min_samples

    return ds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fit_conditional_distributions(
    config: dict,
    dataset: str = "cpc",
    condition: str = "season",
    workers: int = 16,
    overwrite: bool = False,
) -> dict[str, xr.Dataset]:
    """
    Fit 4 distributions per conditioning stratum at every CPC land cell.

    Parameters
    ----------
    config    : loaded config.yaml
    dataset   : 'cpc' (IMERG support planned for later phases)
    condition : 'season' | 'enso' | 'mjo'
    workers   : parallel workers for ProcessPoolExecutor
    overwrite : if False, load cached NetCDF if it exists

    Returns
    -------
    dict[str, xr.Dataset]
        Keys   : stratum labels (e.g. 'DJF', 'el_nino', 'phase_3', 'inactive')
        Values : xr.Dataset with same structure as unconditional stats
    """
    if dataset != "cpc":
        raise NotImplementedError("Conditional fitting currently supports 'cpc' only.")

    project_root = Path(__file__).resolve().parents[2]

    # ── load processed precipitation ──────────────────────────────────────
    processed_path = project_root / config["paths"][f"processed_{dataset}"] / f"{dataset}_7day_wet.nc"
    da = xr.open_dataarray(processed_path)

    # ── load index data if needed ─────────────────────────────────────────
    oni_df = rmm_df = None
    if condition == "enso":
        from src.data.download_oni import load_oni
        oni_df = load_oni(config)
    elif condition == "mjo":
        from src.data.download_rmm import load_rmm
        rmm_df = load_rmm(config)

    # ── assign labels ─────────────────────────────────────────────────────
    print(f"[conditional] Labelling {dataset.upper()} windows by {condition} ...")
    labels = label_windows(da, condition, oni_df=oni_df, rmm_df=rmm_df, config=config)

    # ── determine stratum order ────────────────────────────────────────────
    if condition == "season":
        all_strata = SEASON_ORDER
    elif condition == "enso":
        all_strata = ENSO_ORDER
    elif condition == "mjo":
        all_strata = MJO_ORDER

    # ── output directory ──────────────────────────────────────────────────
    out_dir = project_root / config["paths"]["conditional_stats"] / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── sample counts ─────────────────────────────────────────────────────
    print(f"\n[conditional] Sample counts per {condition} stratum:")
    for stratum in all_strata:
        n = int((labels == stratum).sum())
        print(f"  {stratum:<12}: {n:5d} windows")

    # ── fit per stratum ────────────────────────────────────────────────────
    results: dict[str, xr.Dataset] = {}
    print(f"\n[conditional] Fitting distributions ...")
    for stratum in all_strata:
        out_path = out_dir / f"{dataset}_{condition}_{stratum}_distribution_stats.nc"

        if out_path.exists() and not overwrite:
            print(f"  [skip] {stratum} — loading cached {out_path.name}")
            results[stratum] = xr.open_dataset(out_path)
            continue

        mask = (labels == stratum).values
        n = int(mask.sum())
        if n < config["analysis"]["min_samples"]:
            print(f"  [skip] {stratum} — only {n} windows (< min_samples={config['analysis']['min_samples']})")
            continue

        ds = _fit_stratum(da, mask, config, stratum_label=stratum, n_workers=workers)
        ds.attrs["condition_type"] = condition

        ds.to_netcdf(out_path)
        print(f"  [saved] {out_path.name}  (n={n})")
        results[stratum] = ds

    print(f"\n[conditional] Done. {len(results)} strata saved to {out_dir}")
    return results


def load_conditional_stats(
    config: dict,
    dataset: str = "cpc",
    condition: str = "season",
) -> dict[str, xr.Dataset]:
    """
    Load previously computed conditional stats from NetCDF files.

    Returns
    -------
    dict[str, xr.Dataset] — same structure as fit_conditional_distributions()
    """
    project_root = Path(__file__).resolve().parents[2]
    stats_dir = project_root / config["paths"]["conditional_stats"] / condition

    if not stats_dir.exists():
        raise FileNotFoundError(
            f"No conditional stats found at {stats_dir}\n"
            f"Run fit_conditional_distributions(config, '{dataset}', '{condition}') first."
        )

    if condition == "season":
        strata = SEASON_ORDER
    elif condition == "enso":
        strata = ENSO_ORDER
    elif condition == "mjo":
        strata = MJO_ORDER
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    results: dict[str, xr.Dataset] = {}
    for stratum in strata:
        path = stats_dir / f"{dataset}_{condition}_{stratum}_distribution_stats.nc"
        if path.exists():
            results[stratum] = xr.open_dataset(path)
        else:
            print(f"  [missing] {path.name}")

    return results

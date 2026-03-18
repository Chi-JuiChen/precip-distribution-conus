"""
preprocess.py
-------------
Load raw CPC/IMERG NetCDF files, regrid IMERG to 0.25°, compute 7-day rolling
mean precipitation, and mask to wet days only.

Functions
---------
load_cpc(config)        → xr.DataArray (time, lat, lon), daily mm/day
load_imerg(config)      → xr.DataArray (time, lat, lon), daily mm/day
regrid_to_025(da, cfg)  → xr.DataArray on standard 0.25° grid
compute_rolling(da, n)  → xr.DataArray, n-day trailing rolling mean
mask_wet_days(da, thr)  → xr.DataArray with sub-threshold values set to NaN
run_preprocessing(cfg, dataset) → saves processed file, returns path
"""

import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CPC loader
# ---------------------------------------------------------------------------

def load_cpc(config: dict) -> xr.DataArray:
    """
    Load all CPC annual NetCDF files into a single DataArray.

    CPC native grid is already ~0.25° over CONUS.
    Missing values (large negative fill) are replaced with NaN.

    Returns
    -------
    xr.DataArray with dims (time, lat, lon), units mm/day
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / config["paths"]["raw_cpc"]
    template = config["cpc"]["filename_template"]
    fill = config["cpc"]["fill_value"]

    year_start = config["years"]["start"]
    year_end = config["years"]["end"]

    files = sorted([
        raw_dir / template.format(year=y)
        for y in range(year_start, year_end + 1)
    ])

    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"[CPC] Missing {len(missing)} files. Run download first.\n"
            f"  First missing: {missing[0]}"
        )

    print(f"[CPC] Loading {len(files)} files...")
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        engine="netcdf4",
    )

    da = ds[config["cpc"]["variable"]].astype("float32")

    # Replace fill values with NaN
    da = da.where(da > fill * 0.9)

    # Standardize coordinate names
    da = _standardize_coords(da)

    # CPC uses 0–360 longitude convention; convert to -180–180
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")

    # Clip to CONUS domain
    da = _clip_domain(da, config["domain"])

    print(f"[CPC] Loaded: {da.sizes}")
    return da


# ---------------------------------------------------------------------------
# IMERG loader
# ---------------------------------------------------------------------------

def load_imerg(config: dict) -> xr.DataArray:
    """
    Load IMERG v07 daily files into a single DataArray and convert units.

    IMERG daily product 'precipitation' variable is in mm/hr.
    Multiply by 24 to convert to mm/day.

    Returns
    -------
    xr.DataArray with dims (time, lat, lon), units mm/day, 0.1° grid
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / config["paths"]["raw_imerg"]

    files = sorted(raw_dir.glob("3B-DAY.MS.MRG.3IMERG.*.nc4"))

    if not files:
        raise FileNotFoundError(
            f"[IMERG] No files found in {raw_dir}. Run download first."
        )

    year_start = config["years"]["start"]
    year_end = config["years"]["end"]
    # Filename: 3B-DAY.MS.MRG.3IMERG.YYYYMMDD-...
    # "3B-DAY.MS.MRG.3IMERG." is 21 chars → year starts at position 21
    files = [
        f for f in files
        if year_start <= int(f.name[21:25]) <= year_end
    ]

    # Quick size-based sanity check: valid IMERG daily files are ~30-35 MB;
    # a corrupted/truncated file will be much smaller. 20 MB is a safe threshold.
    MIN_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
    bad_files = [f for f in files if f.stat().st_size < MIN_SIZE_BYTES]
    if bad_files:
        print(f"[IMERG] WARNING: {len(bad_files)} likely-corrupted file(s) skipped "
              f"(size < 20 MB):")
        for b in bad_files:
            print(f"  {b.name}  ({b.stat().st_size / 1e6:.1f} MB)")
        print("  → Delete and re-download them, then rerun.")
    files = [f for f in files if f not in bad_files]

    print(f"[IMERG] Loading {len(files)} daily files...")
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        engine="netcdf4",
        chunks={"time": 30},
    )

    var = config["imerg"]["variable"]
    da = ds[var].astype("float32")

    # IMERG stores lon as 0–360 in some versions; convert to -180–180
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")

    # IMERG V07 3B-DAY stores 'precipitation' already in mm/day (changed from
    # mm/hr in V06).  No unit conversion needed — just ensure the attr is set.
    da.attrs["units"] = "mm/day"

    da = _standardize_coords(da)
    da = _clip_domain(da, config["domain"])

    print(f"[IMERG] Loaded: {da.sizes}")
    return da


# ---------------------------------------------------------------------------
# Regridding (IMERG 0.1° → 0.25°)
# ---------------------------------------------------------------------------

def regrid_to_025(da: xr.DataArray, config: dict) -> xr.DataArray:
    """
    Regrid DataArray to a standard 0.25° grid over CONUS using xesmf.

    Uses conservative regridding (mass-conserving), appropriate for precipitation.

    Parameters
    ----------
    da : xr.DataArray
        Input data (any resolution) with dims (time, lat, lon).
    config : dict
        Config dict (used for domain bounds and regrid method).

    Returns
    -------
    xr.DataArray on 0.25° grid.
    """
    try:
        import xesmf as xe
    except ImportError:
        raise ImportError(
            "xesmf not installed. Run:\n"
            "  conda install -c conda-forge xesmf"
        )

    domain = config["domain"]
    res = config["analysis"]["target_resolution"]
    method = config["imerg"]["regrid_method"]

    # Build target grid
    lat_out = np.arange(domain["lat_min"], domain["lat_max"] + res, res)
    lon_out = np.arange(domain["lon_min"], domain["lon_max"] + res, res)
    ds_out = xr.Dataset({"lat": lat_out, "lon": lon_out})

    # xesmf needs a Dataset
    ds_in = da.to_dataset(name="precip")

    print(f"[regrid] {da.sizes} → 0.25° ({method})")
    regridder = xe.Regridder(ds_in, ds_out, method, periodic=False)
    ds_regridded = regridder(ds_in)

    result = ds_regridded["precip"]
    result.attrs["units"] = da.attrs.get("units", "mm/day")
    return result


# ---------------------------------------------------------------------------
# Rolling average
# ---------------------------------------------------------------------------

def compute_rolling(da: xr.DataArray, config: dict) -> xr.DataArray:
    """
    Compute trailing n-day rolling mean.

    A window is valid only if all n days have non-NaN data (min_periods=n).
    This means the first (n-1) time steps will be NaN.

    Returns
    -------
    xr.DataArray of same shape, units mm/day (averaged over the window).
    """
    n = config["analysis"]["rolling_days"]
    print(f"[rolling] Computing {n}-day trailing mean...")
    rolled = da.rolling(time=n, min_periods=n).mean()
    rolled.attrs = da.attrs
    rolled.attrs["rolling_days"] = n
    return rolled


# ---------------------------------------------------------------------------
# Wet-day masking
# ---------------------------------------------------------------------------

def mask_wet_days(da: xr.DataArray, config: dict) -> xr.DataArray:
    """
    Set grid points below wet_threshold to NaN.

    Parameters
    ----------
    da : xr.DataArray
        7-day rolling mean precipitation (mm/day).
    config : dict

    Returns
    -------
    xr.DataArray with sub-threshold values as NaN.
    """
    threshold = config["analysis"]["wet_threshold"]
    masked = da.where(da >= threshold)
    masked.attrs = da.attrs
    masked.attrs["wet_threshold"] = threshold
    print(f"[mask] Wet-day threshold: {threshold} mm/day applied")
    return masked


# ---------------------------------------------------------------------------
# Top-level preprocessing function
# ---------------------------------------------------------------------------

def run_preprocessing(config: dict, dataset: str) -> Path:
    """
    Full preprocessing pipeline for one dataset.

    Steps: load → (regrid if IMERG) → rolling mean → mask → save

    Parameters
    ----------
    config : dict
    dataset : str
        'cpc' or 'imerg'

    Returns
    -------
    Path to saved processed NetCDF file.
    """
    project_root = Path(__file__).resolve().parents[2]

    if dataset == "cpc":
        da = load_cpc(config)
        out_dir = project_root / config["paths"]["processed_cpc"]
    elif dataset == "imerg":
        da = load_imerg(config)
        da = regrid_to_025(da, config)
        out_dir = project_root / config["paths"]["processed_imerg"]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'cpc' or 'imerg'.")

    out_dir.mkdir(parents=True, exist_ok=True)

    da = compute_rolling(da, config)
    da = mask_wet_days(da, config)

    out_path = out_dir / f"{dataset}_7day_wet.nc"
    if out_path.exists():
        out_path.unlink()  # remove stale file to avoid PermissionError on overwrite
    print(f"[save] Writing {out_path} ...")
    da.to_netcdf(out_path)
    print(f"[save] Done → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename coordinate variants to standard 'lat'/'lon'."""
    rename_map = {}
    for coord in da.coords:
        if coord.lower() in ("latitude", "lat"):
            rename_map[coord] = "lat"
        elif coord.lower() in ("longitude", "lon"):
            rename_map[coord] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da


def _clip_domain(da: xr.DataArray, domain: dict) -> xr.DataArray:
    """Clip DataArray to the configured lat/lon bounding box."""
    return da.sel(
        lat=slice(domain["lat_min"], domain["lat_max"]),
        lon=slice(domain["lon_min"], domain["lon_max"]),
    )


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Quick test with CPC only
    cfg["years"]["start"] = 2010
    cfg["years"]["end"] = 2010

    path = run_preprocessing(cfg, "cpc")
    print(f"Processed file: {path}")

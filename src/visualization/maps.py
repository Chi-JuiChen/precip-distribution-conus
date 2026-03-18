"""
maps.py
-------
Generate publication-quality Cartopy maps of the distribution fitting results.

Maps produced
-------------
  1.  best_fit           — Binary: log-normal (blue) vs gamma (red)
  2.  delta_aic          — ΔAIC per grid point (diverging colormap)
  3.  ks_lognorm         — KS statistic for log-normal fit
  4.  ks_gamma           — KS statistic for gamma fit
  5.  pval_lognorm       — KS p-value for log-normal
  6.  pval_gamma         — KS p-value for gamma
  7.  wetday_freq        — Fraction of 7-day windows classified as wet
  8.  best_fit_4way      — 4-colour: lognorm / gamma / pearson3 / SHASH
  9.  aic_confidence     — ΔAIC margin between best and 2nd-best distribution
  10. ks_pearson3        — KS statistic for Pearson III
  11. ks_shash           — KS statistic for SHASH
  12. pval_pearson3      — KS p-value for Pearson III
  13. pval_shash         — KS p-value for SHASH
  (panel)  6-panel summary figure for each dataset  (original, kept)
  (panel)  9-panel summary with all 4 distributions
  (comparison) side-by-side ΔAIC for CPC vs IMERG

All figures saved to output/figures/ as PNG (150 dpi default).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------------------------------------------------------
# Cartopy import with helpful error message
# ---------------------------------------------------------------------------
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


def _require_cartopy() -> None:
    if not _HAS_CARTOPY:
        raise ImportError(
            "cartopy not installed. Run:\n"
            "  conda install -c conda-forge cartopy"
        )


# Distribution metadata used by 4-way maps
_DIST_NAMES  = ["Log-normal", "Gamma", "Pearson III", "SHASH"]
_DIST_COLORS = ["#2166ac", "#d6604d", "#4dac26", "#f1a340"]   # blue, red, green, orange


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_ax(fig, pos, domain: dict):
    """Create a Cartopy axes with state/coast overlays and CONUS extent."""
    _require_cartopy()
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_extent(
        [domain["lon_min"], domain["lon_max"], domain["lat_min"], domain["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="black", alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    return ax


def _pcolormesh(ax, lons, lats, data, **kwargs):
    """Wrapper for pcolormesh on PlateCarree axes."""
    return ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )


def _colorbar(fig, im, ax, label: str):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[maps] Saved → {path}")


# ---------------------------------------------------------------------------
# Individual map functions
# ---------------------------------------------------------------------------

def plot_best_fit(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    Binary map: 0 = log-normal (blue), 1 = gamma (red).
    Masked grid points (n_wetdays < min_samples) shown in grey.
    """
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds["best_fit"].values.astype(float)
    mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]
    data[mask] = np.nan

    cmap = plt.cm.bwr
    cmap.set_bad("lightgrey")

    im = _pcolormesh(ax, ds.lon.values, ds.lat.values, data,
                     cmap=cmap, vmin=-0.5, vmax=1.5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03,
                        shrink=0.7, ticks=[0, 1])
    cbar.ax.set_xticklabels(["Log-normal", "Gamma"], fontsize=9)
    cbar.set_label("Best-fit distribution (ΔAIC criterion)", fontsize=9)

    ax.set_title(f"Best-fit distribution — {dataset.upper()} ({_period(config)})",
                 fontsize=11, fontweight="bold")

    out_path = out_dir / f"{dataset}_best_fit.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


def plot_delta_aic(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    ΔAIC map: negative (blue) = log-normal better, positive (red) = gamma better.
    Symmetric colorbar centered at 0; saturate at ±50 AIC units.
    """
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds["delta_aic"].values.copy()
    mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]
    data[mask] = np.nan

    vmax = 50.0
    cmap = plt.get_cmap(cfg_vis["cmap_delta_aic"])
    cmap.set_bad("lightgrey")

    im = _pcolormesh(ax, ds.lon.values, ds.lat.values, data,
                     cmap=cmap, vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("ΔAIC  (AIC_lognorm − AIC_gamma)\n← log-normal better  |  gamma better →",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(f"ΔAIC — {dataset.upper()} ({_period(config)})",
                 fontsize=11, fontweight="bold")

    out_path = out_dir / f"{dataset}_delta_aic.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


def plot_ks_map(ds: xr.Dataset, config: dict, dataset: str, dist: str) -> Path:
    """KS statistic map for one distribution. Lower = better fit."""
    assert dist in ("lognorm", "gamma", "pearson3", "shash")
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds[f"ks_{dist}"].values.copy()
    mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]
    data[mask] = np.nan

    cmap = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap.set_bad("lightgrey")

    im = _pcolormesh(ax, ds.lon.values, ds.lat.values, data,
                     cmap=cmap, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("KS statistic (lower = better fit)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    label = {"lognorm": "Log-normal", "gamma": "Gamma",
             "pearson3": "Pearson III", "shash": "SHASH"}[dist]
    ax.set_title(f"KS statistic — {label} — {dataset.upper()} ({_period(config)})",
                 fontsize=11, fontweight="bold")

    out_path = out_dir / f"{dataset}_ks_{dist}.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


def plot_pvalue_map(ds: xr.Dataset, config: dict, dataset: str, dist: str) -> Path:
    """p-value map. Values >0.05 = cannot reject the distribution."""
    assert dist in ("lognorm", "gamma", "pearson3", "shash")
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds[f"pval_{dist}"].values.copy()
    mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]
    data[mask] = np.nan

    # Highlight rejection at 0.05 threshold with custom boundaries
    bounds = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    cmap = plt.get_cmap("RdYlGn", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad("lightgrey")

    im = _pcolormesh(ax, ds.lon.values, ds.lat.values, data,
                     cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85,
                      ticks=bounds)
    cb.set_label("KS p-value  (>0.05 = cannot reject distribution)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    label = {"lognorm": "Log-normal", "gamma": "Gamma",
             "pearson3": "Pearson III", "shash": "SHASH"}[dist]
    ax.set_title(f"KS p-value — {label} — {dataset.upper()} ({_period(config)})",
                 fontsize=11, fontweight="bold")

    out_path = out_dir / f"{dataset}_pval_{dist}.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


def plot_wetday_freq(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """Fraction of time steps classified as wet (non-NaN after masking)."""
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    # n_wetdays / total possible time steps
    # Estimate total steps from year range
    n_years = config["years"]["end"] - config["years"]["start"] + 1
    # ~365.25 days/year minus (rolling_days - 1) lost at start of each year
    roll = config["analysis"]["rolling_days"]
    total_steps = n_years * (365 - roll + 1)

    freq = ds["n_wetdays"].values.astype(float) / total_steps
    freq = np.clip(freq, 0, 1)

    fig, ax = plt.subplots(
        1, 1,
        figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    cmap = plt.get_cmap("Blues")
    im = _pcolormesh(ax, ds.lon.values, ds.lat.values, freq,
                     cmap=cmap, vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("Wet-day frequency (fraction of 7-day windows)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(f"Wet-day frequency — {dataset.upper()} ({_period(config)})",
                 fontsize=11, fontweight="bold")

    out_path = out_dir / f"{dataset}_wetday_freq.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# 6-panel summary figure
# ---------------------------------------------------------------------------

def plot_summary_panel(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    6-panel figure:
      [0] Best-fit distribution
      [1] ΔAIC
      [2] KS statistic — log-normal
      [3] KS statistic — gamma
      [4] p-value — log-normal
      [5] Wet-day frequency
    """
    _require_cartopy()
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig = plt.figure(figsize=cfg_vis["figsize_panel"])
    fig.suptitle(
        f"Precipitation Distribution Analysis — {dataset.upper()}\n"
        f"7-day mean, 2001–2020, wet threshold {config['analysis']['wet_threshold']} mm/day",
        fontsize=13, fontweight="bold", y=0.98,
    )

    lons = ds.lon.values
    lats = ds.lat.values
    mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]

    def _make_ax(pos):
        ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
        return _setup_ax(ax, domain)

    # --- Panel 0: best fit ---
    ax = _make_ax(231)
    data = ds["best_fit"].values.astype(float)
    data[mask] = np.nan
    cmap0 = plt.cm.bwr.copy()
    cmap0.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap0, vmin=-0.5, vmax=1.5)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8, ticks=[0, 1])
    cb.ax.set_xticklabels(["LogN", "Gamma"], fontsize=7)
    ax.set_title("Best-fit distribution", fontsize=9, fontweight="bold")

    # --- Panel 1: ΔAIC ---
    ax = _make_ax(232)
    data = ds["delta_aic"].values.copy()
    data[mask] = np.nan
    cmap1 = plt.get_cmap(cfg_vis["cmap_delta_aic"])
    cmap1.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap1, vmin=-50, vmax=50)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
    cb.set_label("ΔAIC (← LogN | Gamma →)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("ΔAIC  (AIC_LogN − AIC_Gamma)", fontsize=9, fontweight="bold")

    # --- Panel 2: KS log-normal ---
    ax = _make_ax(233)
    data = ds["ks_lognorm"].values.copy()
    data[mask] = np.nan
    cmap2 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap2.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap2, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS statistic — Log-normal", fontsize=9, fontweight="bold")

    # --- Panel 3: KS gamma ---
    ax = _make_ax(234)
    data = ds["ks_gamma"].values.copy()
    data[mask] = np.nan
    cmap3 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap3.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap3, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS statistic — Gamma", fontsize=9, fontweight="bold")

    # --- Panel 4: p-value log-normal ---
    ax = _make_ax(235)
    data = ds["pval_lognorm"].values.copy()
    data[mask] = np.nan
    bounds = [0, 0.01, 0.05, 0.1, 0.5, 1.0]
    cmap4 = plt.get_cmap("RdYlGn", len(bounds) - 1)
    norm4 = mcolors.BoundaryNorm(bounds, cmap4.N)
    cmap4.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap4, norm=norm4)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                      ticks=bounds)
    cb.set_label("p-value (>0.05 = OK)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS p-value — Log-normal", fontsize=9, fontweight="bold")

    # --- Panel 5: wet-day frequency ---
    ax = _make_ax(236)
    n_years = config["years"]["end"] - config["years"]["start"] + 1
    roll = config["analysis"]["rolling_days"]
    total_steps = n_years * (365 - roll + 1)
    freq = np.clip(ds["n_wetdays"].values.astype(float) / total_steps, 0, 1)
    cmap5 = plt.get_cmap("Blues")
    im = _pcolormesh(ax, lons, lats, freq, cmap=cmap5, vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8)
    cb.set_label("Wet-day fraction", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Wet-day frequency", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"{dataset}_summary_panel.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# 4-way best-fit map
# ---------------------------------------------------------------------------

def plot_best_fit_4way(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    Categorical map showing which of the 4 distributions wins per grid point:
      blue  = Log-normal  (0)
      red   = Gamma       (1)
      green = Pearson III (2)
      orange= SHASH       (3)
    """
    _require_cartopy()
    if "best_fit_4way" not in ds:
        raise KeyError("'best_fit_4way' not found. Re-run fit_distributions().")

    out_dir  = _out_dir(config)
    domain   = config["domain"]
    cfg_vis  = config["visualization"]
    lons     = ds.lon.values
    lats     = ds.lat.values
    mask     = ds["n_wetdays"].values < config["analysis"]["min_samples"]

    fig, ax = plt.subplots(
        1, 1, figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds["best_fit_4way"].values.astype(float)
    data[mask] = np.nan

    cmap = mcolors.ListedColormap(_DIST_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    cmap.set_bad("lightgrey")

    im = _pcolormesh(ax, lons, lats, data, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.75,
                      ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels(_DIST_NAMES, fontsize=8)
    cb.set_label("Best-fit distribution (lowest AIC)", fontsize=9)

    ax.set_title(
        f"Best-fit distribution (4-way) — {dataset.upper()} ({_period(config)})",
        fontsize=11, fontweight="bold",
    )
    out_path = out_dir / f"{dataset}_best_fit_4way.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# AIC confidence map
# ---------------------------------------------------------------------------

def plot_aic_confidence(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    Map of ΔAIC between the best and 2nd-best distribution.
    Higher values indicate a more decisive winner.
    Saturates at 50 AIC units.
    """
    _require_cartopy()
    if "aic_confidence" not in ds:
        raise KeyError("'aic_confidence' not found. Re-run fit_distributions().")

    out_dir  = _out_dir(config)
    domain   = config["domain"]
    cfg_vis  = config["visualization"]
    lons     = ds.lon.values
    lats     = ds.lat.values
    mask     = ds["n_wetdays"].values < config["analysis"]["min_samples"]

    fig, ax = plt.subplots(
        1, 1, figsize=cfg_vis["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = _setup_ax(ax, domain)

    data = ds["aic_confidence"].values.copy()
    data[mask] = np.nan

    cmap = plt.get_cmap("YlOrRd")
    cmap.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap, vmin=0, vmax=50)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("AIC confidence  (ΔAIC between 1st and 2nd best; higher = clearer winner)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(
        f"AIC confidence — {dataset.upper()} ({_period(config)})",
        fontsize=11, fontweight="bold",
    )
    out_path = out_dir / f"{dataset}_aic_confidence.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# 9-panel summary (all 4 distributions)
# ---------------------------------------------------------------------------

def plot_summary_panel_4dist(ds: xr.Dataset, config: dict, dataset: str) -> Path:
    """
    9-panel figure (3×3) summarising all 4 distributions:

      Row 0: best_fit_4way  |  aic_confidence  |  ΔAIC (lognorm−gamma)
      Row 1: KS lognorm     |  KS gamma        |  KS pearson3
      Row 2: KS shash       |  p-value lognorm |  wet-day frequency
    """
    _require_cartopy()
    if "best_fit_4way" not in ds:
        raise KeyError("'best_fit_4way' not found. Re-run fit_distributions().")

    out_dir  = _out_dir(config)
    domain   = config["domain"]
    cfg_vis  = config["visualization"]
    lons     = ds.lon.values
    lats     = ds.lat.values
    mask     = ds["n_wetdays"].values < config["analysis"]["min_samples"]

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Precipitation Distribution Analysis (4 distributions) — {dataset.upper()}\n"
        f"7-day mean, {_period(config)}, wet threshold "
        f"{config['analysis']['wet_threshold']} mm/day",
        fontsize=13, fontweight="bold", y=0.99,
    )

    def _ax(pos):
        ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
        return _setup_ax(ax, domain)

    def _masked(key):
        d = ds[key].values.copy().astype(float)
        d[mask] = np.nan
        return d

    # --- (0,0) 4-way best fit ---
    ax = _ax(331)
    data = _masked("best_fit_4way")
    cmap0 = mcolors.ListedColormap(_DIST_COLORS)
    norm0 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap0.N)
    cmap0.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap0, norm=norm0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85,
                      ticks=[0, 1, 2, 3])
    cb.ax.set_xticklabels(["LogN", "Gam", "P3", "SHASH"], fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Best-fit (4-way)", fontsize=9, fontweight="bold")

    # --- (0,1) AIC confidence ---
    ax = _ax(332)
    data = _masked("aic_confidence")
    cmap1 = plt.get_cmap("YlOrRd")
    cmap1.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap1, vmin=0, vmax=50)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("ΔAIC (1st vs 2nd)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("AIC confidence", fontsize=9, fontweight="bold")

    # --- (0,2) ΔAIC lognorm − gamma ---
    ax = _ax(333)
    data = _masked("delta_aic")
    cmap2 = plt.get_cmap(cfg_vis["cmap_delta_aic"])
    cmap2.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap2, vmin=-50, vmax=50)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("ΔAIC (← LogN | Gam →)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("ΔAIC  LogN − Gamma", fontsize=9, fontweight="bold")

    # --- (1,0) KS log-normal ---
    ax = _ax(334)
    data = _masked("ks_lognorm")
    cmap3 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap3.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap3, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS — Log-normal", fontsize=9, fontweight="bold")

    # --- (1,1) KS gamma ---
    ax = _ax(335)
    data = _masked("ks_gamma")
    cmap4 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap4.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap4, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS — Gamma", fontsize=9, fontweight="bold")

    # --- (1,2) KS Pearson III ---
    ax = _ax(336)
    data = _masked("ks_pearson3")
    cmap5 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap5.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap5, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS — Pearson III", fontsize=9, fontweight="bold")

    # --- (2,0) KS SHASH ---
    ax = _ax(337)
    data = _masked("ks_shash")
    cmap6 = plt.get_cmap(cfg_vis["cmap_ks"])
    cmap6.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap6, vmin=0, vmax=0.15)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("KS statistic", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS — SHASH", fontsize=9, fontweight="bold")

    # --- (2,1) p-value log-normal ---
    ax = _ax(338)
    data = _masked("pval_lognorm")
    bounds = [0, 0.01, 0.05, 0.1, 0.5, 1.0]
    cmap7 = plt.get_cmap("RdYlGn", len(bounds) - 1)
    norm7 = mcolors.BoundaryNorm(bounds, cmap7.N)
    cmap7.set_bad("lightgrey")
    im = _pcolormesh(ax, lons, lats, data, cmap=cmap7, norm=norm7)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85,
                      ticks=bounds)
    cb.set_label("p-value (>0.05 = OK)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("KS p-value — Log-normal", fontsize=9, fontweight="bold")

    # --- (2,2) wet-day frequency ---
    ax = _ax(339)
    n_years    = config["years"]["end"] - config["years"]["start"] + 1
    roll       = config["analysis"]["rolling_days"]
    total_steps = n_years * (365 - roll + 1)
    freq = np.clip(ds["n_wetdays"].values.astype(float) / total_steps, 0, 1)
    cmap8 = plt.get_cmap("Blues")
    im = _pcolormesh(ax, lons, lats, freq, cmap=cmap8, vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85)
    cb.set_label("Wet-day fraction", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Wet-day frequency", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"{dataset}_summary_panel_4dist.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# CPC vs IMERG comparison
# ---------------------------------------------------------------------------

def plot_comparison(ds_cpc: xr.Dataset, ds_imerg: xr.Dataset,
                    config: dict) -> Path:
    """
    Side-by-side ΔAIC comparison: CPC (left) vs IMERG (right).
    """
    _require_cartopy()
    out_dir = _out_dir(config)
    domain = config["domain"]
    cfg_vis = config["visualization"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.suptitle(
        "ΔAIC Comparison: CPC vs IMERG v07\n"
        f"(blue = log-normal better, red = gamma better)   {_period(config)}",
        fontsize=12, fontweight="bold",
    )

    for ax, ds, label in zip(axes, [ds_cpc, ds_imerg], ["CPC", "IMERG v07"]):
        ax = _setup_ax(ax, domain)
        data = ds["delta_aic"].values.copy()
        mask = ds["n_wetdays"].values < config["analysis"]["min_samples"]
        data[mask] = np.nan
        cmap = plt.get_cmap(cfg_vis["cmap_delta_aic"])
        cmap.set_bad("lightgrey")
        im = _pcolormesh(ax, ds.lon.values, ds.lat.values, data,
                         cmap=cmap, vmin=-50, vmax=50)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85)
        cb.set_label("ΔAIC", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "comparison_delta_aic.png"
    _save(fig, out_path, cfg_vis["dpi"])
    return out_path


# ---------------------------------------------------------------------------
# Convenience: generate all maps for one dataset
# ---------------------------------------------------------------------------

def plot_all(ds: xr.Dataset, config: dict, dataset: str) -> list[Path]:
    """Generate all individual maps + summary panels for one dataset."""
    paths = [
        # --- original figures (kept unchanged) ---
        plot_best_fit(ds, config, dataset),
        plot_delta_aic(ds, config, dataset),
        plot_ks_map(ds, config, dataset, "lognorm"),
        plot_ks_map(ds, config, dataset, "gamma"),
        plot_pvalue_map(ds, config, dataset, "lognorm"),
        plot_pvalue_map(ds, config, dataset, "gamma"),
        plot_wetday_freq(ds, config, dataset),
        plot_summary_panel(ds, config, dataset),
        # --- new figures for P3 and SHASH ---
        plot_ks_map(ds, config, dataset, "pearson3"),
        plot_ks_map(ds, config, dataset, "shash"),
        plot_pvalue_map(ds, config, dataset, "pearson3"),
        plot_pvalue_map(ds, config, dataset, "shash"),
        plot_best_fit_4way(ds, config, dataset),
        plot_aic_confidence(ds, config, dataset),
        plot_summary_panel_4dist(ds, config, dataset),
    ]
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _setup_ax(ax, domain: dict):
    """Add features and set extent on an existing Cartopy axis."""
    import cartopy.io.shapereader as shpreader

    ax.set_extent(
        [domain["lon_min"], domain["lon_max"], domain["lat_min"], domain["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    # Mask ocean (zorder=4, above pcolormesh at zorder≈1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey", zorder=4)
    # Mask Canada and Mexico using Natural Earth country polygons (already in cartopy)
    countries_shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    for rec in shpreader.Reader(countries_shp).records():
        if rec.attributes.get("NAME_EN", "") in ("Canada", "Mexico"):
            ax.add_geometries(
                [rec.geometry], ccrs.PlateCarree(),
                facecolor="lightgrey", edgecolor="none", zorder=4,
            )
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="black", alpha=0.6, zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=5)
    return ax


def _out_dir(config: dict) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / config["paths"]["output_figures"]


def _period(config: dict) -> str:
    return f"{config['years']['start']}–{config['years']['end']}"


if __name__ == "__main__":
    import yaml
    from src.analysis.distribution import load_stats

    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds = load_stats(cfg, "cpc")
    paths = plot_all(ds, cfg, "cpc")
    print(f"Generated {len(paths)} maps")

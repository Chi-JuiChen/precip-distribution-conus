"""
conditional_viz.py
------------------
Generate all conditional distribution figures for season, ENSO, and MJO.

For each condition:
  1. Best-fit distribution map
  2. P3 skewness
  3. P3 scale
  4. SHASH tailweight τ  (SHASH-winning cells only)
  5. SHASH location shift ε (SHASH-winning cells only)
  6. Wet-day frequency (% of windows that are wet at each cell)

Colorbar: one shared bar on the right of the whole figure.
Range: p2–p98 across all strata of a condition (consistent comparison).
Layouts: season 2×2, ENSO 1×3, MJO 3×3.
"""

import os, sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import yaml, warnings
warnings.filterwarnings('ignore')

from src.analysis.conditional import (
    load_conditional_stats, SEASON_ORDER, ENSO_ORDER, MJO_ORDER
)

# ── Config ────────────────────────────────────────────────────────────────
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
MIN_SAMPLES = config['analysis']['min_samples']

DIST_COLORS = ['#2166ac', '#4dac26', '#d6604d', '#762a83']
DIST_LABELS = ['Log-normal', 'Gamma', 'Pearson III', 'SHASH']
DIST_IDX    = {'lognorm': 0, 'gamma': 1, 'pearson3': 2, 'shash': 3}

PC_LO, PC_HI = 2, 98


# ── Helpers ────────────────────────────────────────────────────────────────
def _conus_ax(ax):
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='0.4')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax


def _get_data(ds, var, shash_only=False):
    d = ds[var].values.copy().astype(float)
    d[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
    if shash_only:
        d[ds['best_fit_4way'].values != DIST_IDX['shash']] = np.nan
    return d


def _wet_pct(ds):
    """Wet-day frequency: n_wetdays / n_windows * 100 at each cell."""
    n_windows = ds.attrs['n_windows']
    d = ds['n_wetdays'].values.copy().astype(float)
    d[d < MIN_SAMPLES] = np.nan
    return d / n_windows * 100.0


def _auto_range(results, strata, var, shash_only=False, use_pct=False):
    all_vals = []
    for s in strata:
        if use_pct:
            d = _wet_pct(results[s]).ravel()
        else:
            d = _get_data(results[s], var, shash_only=shash_only).ravel()
        all_vals.append(d[np.isfinite(d)])
    vals = np.concatenate(all_vals)
    if len(vals) == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, PC_LO)), float(np.percentile(vals, PC_HI))


def _label(s):
    return s.replace('_', ' ').title()


def _grid_shape(condition):
    """Return (n_rows, n_cols) for each condition."""
    return {'season': (2, 2), 'enso': (1, 3), 'mjo': (3, 3)}[condition]


def _panel_size(condition):
    """Return (panel_w, panel_h) in inches."""
    return {'season': (6.0, 4.2), 'enso': (6.5, 4.5), 'mjo': (5.5, 3.8)}[condition]


# ── Core panel builder ─────────────────────────────────────────────────────
def make_figure(results, strata, condition, var, cmap_name, cb_label,
                suptitle, out_path,
                shash_only=False, vmin=None, vmax=None,
                use_pct=False, extra_cb_fn=None):

    if vmin is None or vmax is None:
        vmin, vmax = _auto_range(results, strata, var,
                                 shash_only=shash_only, use_pct=use_pct)

    n_rows, n_cols = _grid_shape(condition)
    pw, ph = _panel_size(condition)
    # Extra width on the right for the colorbar
    fig_w = pw * n_cols + 0.9
    fig_h = ph * n_rows + 0.5

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('lightgrey')

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        subplot_kw={'projection': ccrs.PlateCarree()},
        squeeze=False,
    )
    # Reserve space on the right for colorbar
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.05)

    im = None
    for idx, s in enumerate(strata):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        _conus_ax(ax)
        ds = results[s]
        if use_pct:
            data = _wet_pct(ds)
        else:
            data = _get_data(ds, var, shash_only=shash_only)
        data = np.clip(data, vmin, vmax)
        im = ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
        nw = ds.attrs['n_windows']
        n_valid = int(np.isfinite(data).sum())
        ax.set_title(f'{_label(s)}  (n={nw})',
                     fontsize=9, fontweight='bold', pad=3)

    # Hide unused axes (e.g. MJO has 9 panels in 3×3 = exactly filled)
    for idx in range(len(strata), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    # One shared colorbar on the right spanning all rows
    cbar_ax = fig.add_axes([0.90, 0.08, 0.018, 0.82])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cb_label, fontsize=9, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    if extra_cb_fn:
        extra_cb_fn(cb)

    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.005)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(out_path)}  [{vmin:.2f} – {vmax:.2f}]')


def make_bestfit_figure(results, strata, condition, suptitle, out_path):
    n_rows, n_cols = _grid_shape(condition)
    pw, ph = _panel_size(condition)
    fig_w = pw * n_cols + 0.3
    fig_h = ph * n_rows + 0.7

    cmap4 = mcolors.ListedColormap(DIST_COLORS)
    norm4 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap4.N)
    cmap4.set_bad('lightgrey')

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        subplot_kw={'projection': ccrs.PlateCarree()},
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.28, wspace=0.05, bottom=0.10)

    for idx, s in enumerate(strata):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        _conus_ax(ax)
        ds = results[s]
        data = ds['best_fit_4way'].values.astype(float)
        data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
        ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                      transform=ccrs.PlateCarree(), cmap=cmap4, norm=norm4)
        n_tot = max(int(np.isfinite(data).sum()), 1)
        pcts = {k: 100 * int(np.nansum(data == v)) / n_tot
                for k, v in DIST_IDX.items()}
        nw = ds.attrs['n_windows']
        ax.set_title(
            f'{_label(s)}  (n={nw})\n'
            f'P3 {pcts["pearson3"]:.1f}%  SHASH {pcts["shash"]:.1f}%  '
            f'LN {pcts["lognorm"]:.1f}%  Γ {pcts["gamma"]:.1f}%',
            fontsize=8, fontweight='bold', pad=3,
        )

    for idx in range(len(strata), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(DIST_COLORS, DIST_LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=10)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.005)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(out_path)}')


# ── Run ────────────────────────────────────────────────────────────────────
CONDITIONS = [
    ('season', SEASON_ORDER),
    ('enso',   ENSO_ORDER),
    ('mjo',    MJO_ORDER),
]

for cond, strata in CONDITIONS:
    print(f'\n=== {cond.upper()} ===')
    out_dir = f'output/conditional/figures/{cond}'
    results = load_conditional_stats(config, dataset='cpc', condition=cond)

    make_bestfit_figure(
        results, strata, cond,
        suptitle=f'Best-fit distribution by {cond} — CPC 7-day means (2001–2020)',
        out_path=f'{out_dir}/cpc_{cond}_best_fit.png',
    )
    make_figure(
        results, strata, cond,
        var='pearson3_skew', cmap_name='YlOrRd',
        cb_label='P3 skewness γ₁',
        suptitle=f'Pearson III skewness by {cond} — CPC (2001–2020)',
        out_path=f'{out_dir}/cpc_{cond}_p3_skewness.png',
    )
    make_figure(
        results, strata, cond,
        var='pearson3_scale', cmap_name='Blues',
        cb_label='P3 scale (mm/day)',
        suptitle=f'Pearson III scale by {cond} — CPC (2001–2020)',
        out_path=f'{out_dir}/cpc_{cond}_p3_scale.png',
    )
    make_figure(
        results, strata, cond,
        var='shash_tailweight', cmap_name='YlOrRd',
        cb_label='SHASH tailweight τ  (τ > 1 = heavier tails than normal)',
        suptitle=f'SHASH tailweight τ (SHASH-winning cells) by {cond} — CPC',
        out_path=f'{out_dir}/cpc_{cond}_shash_tau.png',
        shash_only=True,
        extra_cb_fn=lambda cb: cb.ax.axhline(1.0, color='k', lw=1.5, ls='--'),
    )
    make_figure(
        results, strata, cond,
        var='shash_eps', cmap_name='PuOr',
        cb_label='SHASH location shift ε',
        suptitle=f'SHASH location shift ε (SHASH-winning cells) by {cond} — CPC',
        out_path=f'{out_dir}/cpc_{cond}_shash_eps.png',
        shash_only=True,
    )
    make_figure(
        results, strata, cond,
        var='n_wetdays', cmap_name='Blues',
        cb_label='Wet-day frequency (% of windows)',
        suptitle=f'Wet-day frequency by {cond} — CPC (2001–2020)',
        out_path=f'{out_dir}/cpc_{cond}_wet_pct.png',
        use_pct=True,
    )

print('\nAll done.')

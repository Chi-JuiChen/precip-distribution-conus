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

from src.analysis.conditional import load_conditional_stats, ENSO_ORDER, MJO_ORDER

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

MIN_SAMPLES = config['analysis']['min_samples']
DIST_COLORS = ['#2166ac', '#4dac26', '#d6604d', '#762a83']
DIST_LABELS = ['Log-normal', 'Gamma', 'Pearson III', 'SHASH']
DIST_IDX    = {'lognorm': 0, 'gamma': 1, 'pearson3': 2, 'shash': 3}

def _conus_ax(ax):
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='0.4')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    return ax

def best_fit_panel(results, strata, out_path, suptitle):
    n = len(strata)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if n == 1: axes = [axes]
    cmap4 = mcolors.ListedColormap(DIST_COLORS)
    norm4 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap4.N)
    cmap4.set_bad('lightgrey')
    for ax, s in zip(axes, strata):
        _conus_ax(ax)
        ds = results[s]
        data = ds['best_fit_4way'].values.astype(float)
        data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
        ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                      transform=ccrs.PlateCarree(), cmap=cmap4, norm=norm4)
        n_tot = max(int(np.isfinite(data).sum()), 1)
        pcts = {k: 100*int(np.nansum(data == v))/n_tot for k, v in DIST_IDX.items()}
        subtitle = f"P3 {pcts['pearson3']:.1f}%  SHASH {pcts['shash']:.1f}%"
        ax.set_title(f'{s.replace("_"," ").title()}\n{subtitle}', fontsize=9, fontweight='bold')
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(DIST_COLORS, DIST_LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {os.path.basename(out_path)}')

def p3skew_panel(results, strata, out_path, suptitle):
    n = len(strata)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if n == 1: axes = [axes]
    cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('lightgrey')
    for ax, s in zip(axes, strata):
        _conus_ax(ax)
        ds = results[s]
        data = ds['pearson3_skew'].values.copy()
        data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
        data = np.clip(data, 0, 4)
        im = ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                           transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=4)
        ax.set_title(s.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    cb = fig.colorbar(im, ax=axes if n>1 else axes[0],
                      orientation='vertical', fraction=0.015, pad=0.02)
    cb.set_label('P3 skewness (0–4)', fontsize=10)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {os.path.basename(out_path)}')

# ── ENSO ──────────────────────────────────────────────────────────────────
enso_results = load_conditional_stats(config, dataset='cpc', condition='enso')
print('ENSO loaded:', list(enso_results.keys()))

best_fit_panel(enso_results, ENSO_ORDER,
    'output/conditional/figures/enso/cpc_enso_best_fit.png',
    'Best-fit distribution by ENSO phase — CPC 7-day means (2001–2020)')

p3skew_panel(enso_results, ENSO_ORDER,
    'output/conditional/figures/enso/cpc_enso_p3_skewness.png',
    'Pearson III skewness by ENSO phase — CPC 7-day means (2001–2020)')

# ── MJO best-fit (2 rows: inactive + phases 1-8) ──────────────────────────
mjo_results = load_conditional_stats(config, dataset='cpc', condition='mjo')
print('MJO loaded:', list(mjo_results.keys()))

# Row layout: inactive on top row alone is odd — use 3+3+3 or 1+8
# Better: inactive | phases 1-4 | phases 5-8 → 2-row layout
fig = plt.figure(figsize=(28, 10))
cmap4 = mcolors.ListedColormap(DIST_COLORS)
norm4 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap4.N)
cmap4.set_bad('lightgrey')

all_mjo = MJO_ORDER  # inactive + phase_1..8 = 9
# Use 2 rows: row1 = inactive, phase_1..4  (5 panels)
#             row2 = phase_5..8 + empty    (4 panels + spacer)
# Simpler: single row of 9 (wide)
fig, axes = plt.subplots(1, 9, figsize=(54, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
for ax, s in zip(axes, all_mjo):
    _conus_ax(ax)
    ds = mjo_results[s]
    data = ds['best_fit_4way'].values.astype(float)
    data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
    ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                  transform=ccrs.PlateCarree(), cmap=cmap4, norm=norm4)
    n_tot = max(int(np.isfinite(data).sum()), 1)
    pcts = {k: 100*int(np.nansum(data == v))/n_tot for k, v in DIST_IDX.items()}
    nw = ds.attrs['n_windows']
    ax.set_title(f'{s.replace("_"," ").title()}\n(n={nw})\nSHASH {pcts["shash"]:.1f}%',
                 fontsize=8, fontweight='bold')

handles = [mpatches.Patch(color=c, label=l) for c, l in zip(DIST_COLORS, DIST_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=12)
fig.suptitle('Best-fit distribution by MJO phase — CPC 7-day means (2001–2020)',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('output/conditional/figures/mjo/cpc_mjo_best_fit.png',
            dpi=120, bbox_inches='tight')
plt.close()
print('Saved: cpc_mjo_best_fit.png')

# ── MJO P3 skewness ───────────────────────────────────────────────────────
p3skew_panel(mjo_results, MJO_ORDER,
    'output/conditional/figures/mjo/cpc_mjo_p3_skewness.png',
    'Pearson III skewness by MJO phase — CPC 7-day means (2001–2020)')

print('\nAll figures done.')

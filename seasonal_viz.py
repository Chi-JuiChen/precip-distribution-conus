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

from src.analysis.conditional import load_conditional_stats, SEASON_ORDER

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

MIN_SAMPLES = config['analysis']['min_samples']
DIST_COLORS = ['#2166ac', '#4dac26', '#d6604d', '#762a83']
DIST_LABELS = ['Log-normal', 'Gamma', 'Pearson III', 'SHASH']
DIST_IDX    = {'lognorm': 0, 'gamma': 1, 'pearson3': 2, 'shash': 3}
OUT = 'output/conditional/figures/season'
os.makedirs(OUT, exist_ok=True)

def _conus_ax(ax):
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='0.4')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    return ax

season_results = load_conditional_stats(config, dataset='cpc', condition='season')
print('Loaded:', list(season_results.keys()))

# ── Figure 1: Best-fit map by season ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
cmap4 = mcolors.ListedColormap(DIST_COLORS)
norm4 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap4.N)
cmap4.set_bad('lightgrey')

for ax, season in zip(axes, SEASON_ORDER):
    _conus_ax(ax)
    ds = season_results[season]
    data = ds['best_fit_4way'].values.astype(float)
    data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
    ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                  transform=ccrs.PlateCarree(), cmap=cmap4, norm=norm4)
    n_tot = int(np.isfinite(data).sum())
    pcts = {k: 100*int(np.nansum(data == v))/n_tot for k, v in DIST_IDX.items()}
    subtitle = f"P3 {pcts['pearson3']:.1f}%  SHASH {pcts['shash']:.1f}%  LN {pcts['lognorm']:.1f}%  Γ {pcts['gamma']:.1f}%"
    ax.set_title(f'{season}\n{subtitle}', fontsize=9, fontweight='bold')

handles = [mpatches.Patch(color=c, label=l) for c, l in zip(DIST_COLORS, DIST_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=11)
fig.suptitle('Best-fit distribution by season — CPC 7-day means (2001–2020)',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/cpc_season_best_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: cpc_season_best_fit.png')

# ── Figure 2: P3 skewness by season ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
cmap_skew = plt.get_cmap('YlOrRd').copy()
cmap_skew.set_bad('lightgrey')

for ax, season in zip(axes, SEASON_ORDER):
    _conus_ax(ax)
    ds = season_results[season]
    data = ds['pearson3_skew'].values.copy()
    data[ds['n_wetdays'].values < MIN_SAMPLES] = np.nan
    data = np.clip(data, 0, 4)
    im = ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                       transform=ccrs.PlateCarree(), cmap=cmap_skew, vmin=0, vmax=4)
    ax.set_title(season, fontsize=12, fontweight='bold')

cb = fig.colorbar(im, ax=axes.tolist(), orientation='vertical', fraction=0.015, pad=0.02)
cb.set_label('P3 skewness (clipped 0–4)', fontsize=10)
fig.suptitle('Pearson III skewness by season — CPC 7-day means (2001–2020)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/cpc_season_p3_skewness.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: cpc_season_p3_skewness.png')

# ── Figure 3: SHASH τ (SHASH-winning cells only) ──────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
cmap_tau = plt.get_cmap('YlOrRd').copy()
cmap_tau.set_bad('lightgrey')

for ax, season in zip(axes, SEASON_ORDER):
    _conus_ax(ax)
    ds = season_results[season]
    data = ds['shash_tailweight'].values.copy()
    mask_n  = ds['n_wetdays'].values < MIN_SAMPLES
    mask_bf = ds['best_fit_4way'].values != DIST_IDX['shash']
    data[mask_n | mask_bf] = np.nan
    data = np.clip(data, 0.5, 3.5)
    im = ax.pcolormesh(ds.lon.values, ds.lat.values, data,
                       transform=ccrs.PlateCarree(), cmap=cmap_tau, vmin=0.5, vmax=3.5)
    n_shash = int(np.isfinite(data).sum())
    ax.set_title(f'{season}  (n={n_shash})', fontsize=11, fontweight='bold')

cb = fig.colorbar(im, ax=axes.tolist(), orientation='vertical', fraction=0.015, pad=0.02)
cb.set_label('SHASH tailweight τ  (0.5–3.5)', fontsize=10)
cb.ax.axhline(1.0, color='k', lw=1.5, ls='--')
fig.suptitle('SHASH tailweight τ (SHASH-winning cells) by season — CPC',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/cpc_season_shash_tau.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: cpc_season_shash_tau.png')

print('\nDone.')

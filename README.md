# Precipitation Distribution Analysis

Grid-by-grid analysis of 7-day precipitation intensity distributions over CONUS,
comparing **log-normal** vs **gamma** fits. Used to inform subseasonal forecast model design.

**Datasets:** CPC Gauge-Based CONUS + IMERG v07
**Period:** 2001–2020
**Resolution:** 0.25°
**Target variable:** 7-day mean precipitation (wet days only, > 0.1 mm/day)

---

## Project Structure

```
first_project/
├── config/config.yaml          ← all tunable parameters
├── data/
│   ├── raw/cpc/                ← downloaded CPC NetCDF files
│   ├── raw/imerg/              ← downloaded IMERG NetCDF files
│   └── processed/              ← regridded + 7-day averaged
├── src/
│   ├── data/
│   │   ├── download_cpc.py     ← download from NOAA PSL (no auth needed)
│   │   ├── download_imerg.py   ← download from NASA GES DISC (auth needed)
│   │   └── preprocess.py       ← regrid, roll, mask
│   ├── analysis/
│   │   └── distribution.py     ← fit log-normal & gamma, compute GOF metrics
│   └── visualization/
│       └── maps.py             ← cartopy maps
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_distribution_analysis.ipynb
│   └── 03_maps.ipynb
├── scripts/run_pipeline.py     ← end-to-end CLI runner
└── output/
    ├── figures/                ← saved PNG/PDF maps
    └── stats/                  ← NetCDF files of fit parameters
```

---

## Setup

### 1. Install packages into `atmo` (all via conda-forge)

```bash
conda activate atmo
conda install -c conda-forge \
    numpy xarray scipy matplotlib \
    cartopy xesmf netCDF4 h5py \
    tqdm pyyaml requests
```

> All packages are on conda-forge — no pip required.
> If you prefer a declarative approach, use `environment.yml` instead:
> ```bash
> conda env update -n atmo -f environment.yml
> ```

### 2. NASA Earthdata account (for IMERG — already done)

Your `~/.netrc` and GESDISC approval are already set up. You're ready to download IMERG.

---

## Usage

### Run the full pipeline

```bash
conda activate atmo

# CPC only (no NASA account needed)
python scripts/run_pipeline.py --dataset cpc

# IMERG only (requires ~/.netrc)
python scripts/run_pipeline.py --dataset imerg

# Both datasets
python scripts/run_pipeline.py --dataset both

# Quick test: single year, skip download if data already present
python scripts/run_pipeline.py --dataset cpc --years 2010 2010 --skip-download
```

### Step-by-step in notebooks

Open VS Code, select **Python (atmo)** kernel, then run:

1. `notebooks/01_data_check.ipynb` — verify download & units
2. `notebooks/02_distribution_analysis.ipynb` — inspect fits at sample grid points
3. `notebooks/03_maps.ipynb` — generate publication-quality maps

---

## Output Maps

| Map | Description |
|-----|-------------|
| Best-fit distribution | Blue = log-normal, Red = gamma |
| ΔAIC | AIC_lognorm − AIC_gamma; negative = log-normal better |
| KS statistic (log-normal) | Lower = better fit |
| KS statistic (gamma) | Lower = better fit |
| p-value (log-normal) | >0.05 = cannot reject distribution |
| Wet-day frequency | Fraction of 7-day windows classified as wet |

---

## Key Parameters (config/config.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wet_threshold` | 0.1 mm/day | Minimum 7-day mean to count as wet |
| `rolling_days` | 7 | Rolling window length |
| `min_samples` | 30 | Min wet samples required per grid point |
| `years.start/end` | 2001/2020 | Analysis period |

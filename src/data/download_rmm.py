"""
download_rmm.py
---------------
Download the Real-time Multivariate MJO (RMM) daily index from the Australian
Bureau of Meteorology (BoM).

Source : http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
Format : space-delimited text
Columns: year  month  day  RMM1  RMM2  phase  amplitude
Missing: 999.000

The RMM index is derived from the two leading EOFs of combined equatorially
averaged OLR, 850-hPa and 200-hPa zonal wind (after ENSO removal).

MJO phase classification:
  Active MJO : amplitude >= mjo_amp_threshold (default 1.0; unanimous in literature)
  Inactive   : amplitude < threshold → labelled phase 0 / 'inactive'
  Phases 1–8 : octants of the RMM1–RMM2 phase space, corresponding to geographic
               location of enhanced MJO convection (Indian Ocean → Pacific → W.Hem.)

References
----------
Wheeler & Hendon (2004, MWR 132, 1917–1932) — RMM index definition
Johnson et al. (2014, Wea. Fcst. 29, 23–38) — amplitude >= 1.0 threshold
Nardi/Baggett et al. (2020, Wea. Fcst. 35, 2179–2198) — all-season RMM usage
BoM MJO monitoring: http://www.bom.gov.au/climate/mjo/
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path


def download_rmm(config: dict) -> Path:
    """
    Download the BoM RMM daily index file and save to data/indices/.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.

    Returns
    -------
    Path
        Path to the saved index file.
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir  = project_root / config["paths"]["indices"]
    out_dir.mkdir(parents=True, exist_ok=True)

    url      = config["rmm"]["url"]
    filename = config["rmm"]["filename"]
    dest     = out_dir / filename

    if dest.exists():
        print(f"[RMM] Already exists: {dest}")
        return dest

    print(f"[RMM] Downloading from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"[RMM] Saved to {dest}")
    return dest


def load_rmm(config: dict) -> pd.DataFrame:
    """
    Load the BoM RMM daily index file and return a tidy DataFrame.

    Missing values (999.0) are replaced with NaN.

    Returns
    -------
    pd.DataFrame
        Index : pandas DatetimeIndex (daily)
        Columns:
          rmm1       — first principal component
          rmm2       — second principal component
          phase      — integer 1–8 (or NaN if missing)
          amplitude  — sqrt(rmm1² + rmm2²)
          mjo_active — bool: amplitude >= mjo_amp_threshold
    """
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / config["paths"]["indices"] / config["rmm"]["filename"]

    if not path.exists():
        raise FileNotFoundError(
            f"RMM file not found: {path}\n"
            f"Run download_rmm(config) first."
        )

    missing = config["rmm"]["missing_value"]

    # Skip header lines starting with '#' or 'year'
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("year"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "year":      int(parts[0]),
                    "month":     int(parts[1]),
                    "day":       int(parts[2]),
                    "rmm1":      float(parts[3]),
                    "rmm2":      float(parts[4]),
                    "phase":     int(parts[5]),
                    "amplitude": float(parts[6]),
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").drop(columns=["year", "month", "day"]).sort_index()

    # Replace missing values
    for col in ["rmm1", "rmm2", "amplitude"]:
        df[col] = df[col].where(df[col].abs() < missing * 0.9, other=np.nan)
    df["phase"] = df["phase"].where(df["amplitude"].notna(), other=np.nan)

    # Recompute amplitude from rmm1/rmm2 (more reliable than file value)
    df["amplitude"] = np.sqrt(df["rmm1"] ** 2 + df["rmm2"] ** 2)

    # Active MJO flag
    threshold = config["conditional"]["mjo_amp_threshold"]
    df["mjo_active"] = df["amplitude"] >= threshold

    return df


if __name__ == "__main__":
    import yaml
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    download_rmm(cfg)
    df = load_rmm(cfg)
    print(df.tail(10))
    print(f"\nActive MJO days: {df['mjo_active'].sum()} / {len(df)} "
          f"({100*df['mjo_active'].mean():.1f}%)")
    print("\nPhase counts (active MJO only):")
    print(df[df["mjo_active"]]["phase"].value_counts().sort_index())

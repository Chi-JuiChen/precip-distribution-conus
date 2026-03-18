"""
download_oni.py
---------------
Download the NOAA CPC Niño 3.4 monthly SST anomaly index used to define ENSO phase.

Source : https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices
Format : fixed-width text, one row per month
Columns: YR  MON  NINO1+2  ANOM  NINO3  ANOM  NINO34  ANOM  NINO4  ANOM

We extract the Niño 3.4 anomaly column (column index 7, 0-based) and compute
the 3-month centred running mean to produce the ONI (Oceanic Niño Index).

ENSO phase classification (ONI standard, NOAA CPC):
  El Niño  : ONI >= +0.5 °C
  La Niña  : ONI <= -0.5 °C
  Neutral  : -0.5 < ONI < +0.5 °C

References
----------
Johnson et al. (2014, Wea. Fcst. 29, 23–38) — ENSO + MJO week 3–4 forecast method
NOAA CPC ONI: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path


def download_oni(config: dict) -> Path:
    """
    Download the NOAA Niño 3.4 monthly SST anomaly file and save to data/indices/.

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

    url      = config["oni"]["url"]
    filename = config["oni"]["filename"]
    dest     = out_dir / filename

    if dest.exists():
        print(f"[ONI] Already exists: {dest}")
        return dest

    print(f"[ONI] Downloading from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"[ONI] Saved to {dest}")
    return dest


def load_oni(config: dict) -> pd.DataFrame:
    """
    Load the Niño 3.4 monthly anomaly file and return a tidy DataFrame.

    Computes the 3-month centred running mean (ONI) and adds ENSO phase label.

    Returns
    -------
    pd.DataFrame
        Index : pandas DatetimeIndex (first day of each month)
        Columns:
          nino34_anom  — raw monthly Niño 3.4 SST anomaly (°C)
          oni          — 3-month centred running mean (ONI)
          enso_phase   — 'el_nino' | 'neutral' | 'la_nina'
    """
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / config["paths"]["indices"] / config["oni"]["filename"]

    if not path.exists():
        raise FileNotFoundError(
            f"ONI file not found: {path}\n"
            f"Run download_oni(config) first."
        )

    # Fixed-width file: YR MON NINO1+2 ANOM NINO3 ANOM NINO34 ANOM NINO4 ANOM
    df_raw = pd.read_csv(
        path, sep=r"\s+", header=0,
        names=["year", "mon",
               "nino12", "nino12_anom",
               "nino3",  "nino3_anom",
               "nino34", "nino34_anom",
               "nino4",  "nino4_anom"],
    )

    # Build DatetimeIndex (first day of each month)
    df_raw["date"] = pd.to_datetime(
        df_raw["year"].astype(str) + "-" + df_raw["mon"].astype(str).str.zfill(2) + "-01"
    )
    df_raw = df_raw.set_index("date").sort_index()

    # Keep only Niño 3.4 anomaly; drop any non-numeric rows
    df = pd.DataFrame({"nino34_anom": pd.to_numeric(df_raw["nino34_anom"], errors="coerce")})
    df = df.dropna()

    # 3-month centred running mean → ONI
    df["oni"] = df["nino34_anom"].rolling(window=3, center=True, min_periods=3).mean()

    # ENSO phase (±0.5 °C threshold, ONI standard)
    threshold = config["conditional"]["enso_threshold"]
    df["enso_phase"] = "neutral"
    df.loc[df["oni"] >=  threshold, "enso_phase"] = "el_nino"
    df.loc[df["oni"] <= -threshold, "enso_phase"] = "la_nina"

    return df


if __name__ == "__main__":
    import yaml
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    download_oni(cfg)
    df = load_oni(cfg)
    print(df.tail(12))
    print("\nENSO phase counts:")
    print(df["enso_phase"].value_counts())

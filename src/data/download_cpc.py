"""
download_cpc.py
---------------
Download CPC Unified Gauge-Based Daily Precipitation over CONUS from NOAA PSL.

Source: https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/
Format: NetCDF (.nc), one file per year
Variable: precip (mm/day), ~0.25° resolution, CONUS only
No account/API key required.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_cpc(config: dict) -> list[Path]:
    """
    Download CPC daily precipitation files for the configured year range.

    Parameters
    ----------
    config : dict
        Loaded config.yaml as a dictionary.

    Returns
    -------
    list[Path]
        Paths to all downloaded (or already existing) NetCDF files.
    """
    base_url: str = config["cpc"]["base_url"]
    rt_url: str = config["cpc"]["rt_url"]
    rt_start: int = config["cpc"]["rt_start_year"]
    template: str = config["cpc"]["filename_template"]
    year_start: int = config["years"]["start"]
    year_end: int = config["years"]["end"]

    # Resolve output directory relative to project root
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / config["paths"]["raw_cpc"]
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    years = range(year_start, year_end + 1)

    print(f"[CPC] Downloading {len(years)} years ({year_start}–{year_end}) → {out_dir}")
    print(f"[CPC] Note: years <{rt_start} → main dir; years >={rt_start} → RT/ dir")

    for year in tqdm(years, desc="CPC years"):
        filename = template.format(year=year)
        # Years 2007+ live in the /RT/ subdirectory
        url = f"{rt_url}/{filename}" if year >= rt_start else f"{base_url}/{filename}"
        dest = out_dir / filename

        if dest.exists():
            tqdm.write(f"  [skip] {filename} already exists")
            downloaded.append(dest)
            continue

        tqdm.write(f"  [download] {filename}  ({url})")
        _download_file(url, dest)
        downloaded.append(dest)

    print(f"[CPC] Done. {len(downloaded)} files in {out_dir}")
    return downloaded


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    fname = dest.name

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"    {fname}",
        leave=False,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


if __name__ == "__main__":
    # Standalone test: download a single year
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override to a single test year
    cfg["years"]["start"] = 2010
    cfg["years"]["end"] = 2010

    files = download_cpc(cfg)
    print(f"Downloaded: {[str(p) for p in files]}")

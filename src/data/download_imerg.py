"""
download_imerg.py
-----------------
Download IMERG v07 daily accumulated precipitation (GPM_3IMERGDF) from NASA GES DISC.

Source: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/
Format: NetCDF4 (.nc4), one file per day
Variable: precipitation (mm/hr) → multiply by 24 to get mm/day
Resolution: 0.1° global

Requirements
------------
- Free NASA Earthdata account: https://urs.earthdata.nasa.gov
- Approve "NASA GESDISC DATA ARCHIVE" under Applications → Authorized Apps
- Create ~/.netrc with:

    machine urs.earthdata.nasa.gov
        login YOUR_USERNAME
        password YOUR_PASSWORD

- Run: chmod 600 ~/.netrc
"""

import os
import datetime
import requests
from pathlib import Path
from tqdm import tqdm


# IMERG v07 daily product base URL on GES DISC
_BASE_URL = (
    "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07"
)


def _build_url(date: datetime.date) -> tuple[str, str]:
    """
    Build the GES DISC URL for one daily IMERG file.

    Directory structure: {BASE}/{YYYY}/{MM}/
    Filename: 3B-DAY.MS.MRG.3IMERG.{YYYYMMDD}-S000000-E235959.V07B.nc4
    """
    fname = (
        f"3B-DAY.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}"
        f"-S000000-E235959.V07B.nc4"
    )
    return f"{_BASE_URL}/{date.year}/{date.month:02d}/{fname}", fname


def download_imerg(config: dict) -> list[Path]:
    """
    Download IMERG v07 daily files for the configured year range.

    Uses ~/.netrc for authentication (no password stored in code).

    Parameters
    ----------
    config : dict
        Loaded config.yaml as a dictionary.

    Returns
    -------
    list[Path]
        Paths to all downloaded (or already existing) NetCDF files.
    """
    year_start: int = config["years"]["start"]
    year_end: int = config["years"]["end"]

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / config["paths"]["raw_imerg"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build list of all dates
    start = datetime.date(year_start, 1, 1)
    end = datetime.date(year_end, 12, 31)
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

    print(f"[IMERG] Downloading {len(dates)} daily files ({year_start}–{year_end}) → {out_dir}")
    print("[IMERG] Authenticating via ~/.netrc")

    _check_netrc()

    downloaded = []
    session = requests.Session()

    for date in tqdm(dates, desc="IMERG days"):
        url, fname = _build_url(date)
        dest = out_dir / fname

        if dest.exists():
            downloaded.append(dest)
            continue

        try:
            _download_file(session, url, dest)
            downloaded.append(dest)
        except requests.HTTPError as e:
            tqdm.write(f"  [warn] {fname}: {e} — skipping")

    print(f"[IMERG] Done. {len(downloaded)} files in {out_dir}")
    return downloaded


def _check_netrc() -> None:
    """Warn if ~/.netrc does not exist (authentication will fail silently)."""
    netrc = Path.home() / ".netrc"
    if not netrc.exists():
        print(
            "\n[IMERG] WARNING: ~/.netrc not found.\n"
            "  Create it with:\n"
            "    machine urs.earthdata.nasa.gov\n"
            "        login YOUR_USERNAME\n"
            "        password YOUR_PASSWORD\n"
            "  Then: chmod 600 ~/.netrc\n"
        )


def _download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    chunk_size: int = 1 << 20,
) -> None:
    """Stream-download with redirect handling (GES DISC uses auth redirects)."""
    response = session.get(url, stream=True, timeout=120)

    # GES DISC redirects to Earthdata login — follow with netrc auth
    if response.status_code == 401:
        from requests.auth import HTTPDigestAuth
        import netrc as _netrc_mod

        try:
            info = _netrc_mod.netrc()
            auth_info = info.authenticators("urs.earthdata.nasa.gov")
            if auth_info:
                session.auth = (auth_info[0], auth_info[2])
                response = session.get(url, stream=True, timeout=120)
        except FileNotFoundError:
            pass

    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"  {dest.name[:50]}",
        leave=False,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Single-month test
    cfg["years"]["start"] = 2010
    cfg["years"]["end"] = 2010

    files = download_imerg(cfg)
    print(f"Downloaded {len(files)} files")

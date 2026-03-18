"""
run_pipeline.py
---------------
End-to-end pipeline for precipitation distribution analysis.

Usage
-----
  conda activate atmo
  python scripts/run_pipeline.py --dataset cpc
  python scripts/run_pipeline.py --dataset imerg
  python scripts/run_pipeline.py --dataset both
  python scripts/run_pipeline.py --dataset cpc --years 2010 2010 --skip-download
  python scripts/run_pipeline.py --dataset both --skip-preprocess

Steps
-----
  1. Download raw data (unless --skip-download)
  2. Preprocess: regrid, 7-day rolling mean, wet-day mask (unless --skip-preprocess)
  3. Fit log-normal & gamma distributions grid-by-grid (unless --skip-fit)
  4. Generate maps (unless --skip-maps)
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to Python path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_cpc import download_cpc
from src.data.download_imerg import download_imerg
from src.data.preprocess import run_preprocessing
from src.analysis.distribution import fit_distributions, load_stats
from src.visualization.maps import plot_all, plot_comparison


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precipitation distribution analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["cpc", "imerg", "both"],
        required=True,
        help="Which dataset to process",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Override year range from config (e.g. --years 2010 2010)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw data (assumes files already exist)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (assumes processed NetCDF already exists)",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Skip distribution fitting (assumes stats NetCDF already exists)",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip map generation",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml (default: config/config.yaml)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # --- Load config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override year range if provided
    if args.years:
        config["years"]["start"], config["years"]["end"] = args.years
        print(f"[pipeline] Year range overridden: {args.years[0]}–{args.years[1]}")

    datasets = ["cpc", "imerg"] if args.dataset == "both" else [args.dataset]

    print(f"\n{'='*60}")
    print(f"  Precipitation Distribution Analysis Pipeline")
    print(f"  Datasets : {', '.join(d.upper() for d in datasets)}")
    print(f"  Years    : {config['years']['start']}–{config['years']['end']}")
    print(f"  Steps    : download={not args.skip_download}, "
          f"preprocess={not args.skip_preprocess}, "
          f"fit={not args.skip_fit}, "
          f"maps={not args.skip_maps}")
    print(f"{'='*60}\n")

    stats = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name.upper()} ---")

        # Step 1: Download
        if not args.skip_download:
            if ds_name == "cpc":
                download_cpc(config)
            else:
                download_imerg(config)
        else:
            print(f"[pipeline] Skipping download for {ds_name.upper()}")

        # Step 2: Preprocess
        if not args.skip_preprocess:
            run_preprocessing(config, ds_name)
        else:
            print(f"[pipeline] Skipping preprocessing for {ds_name.upper()}")

        # Step 3: Fit distributions
        if not args.skip_fit:
            ds_result = fit_distributions(config, ds_name)
        else:
            print(f"[pipeline] Skipping fitting for {ds_name.upper()}, loading cached stats")
            ds_result = load_stats(config, ds_name)

        stats[ds_name] = ds_result

        # Step 4: Maps per dataset
        if not args.skip_maps:
            print(f"[pipeline] Generating maps for {ds_name.upper()}...")
            paths = plot_all(ds_result, config, ds_name)
            print(f"[pipeline] {len(paths)} maps saved for {ds_name.upper()}")

    # Step 4b: Comparison map (only when both datasets processed)
    if not args.skip_maps and "cpc" in stats and "imerg" in stats:
        from src.visualization.maps import plot_comparison
        path = plot_comparison(stats["cpc"], stats["imerg"], config)
        print(f"[pipeline] Comparison map: {path}")

    print(f"\n[pipeline] Done! Results in:")
    print(f"  Stats  : {PROJECT_ROOT / config['paths']['output_stats']}")
    print(f"  Figures: {PROJECT_ROOT / config['paths']['output_figures']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)

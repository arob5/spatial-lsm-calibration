"""Run RWMH and save posterior as ArviZ NetCDF.

Usage
-----
    python run_mcmc.py --run_id rwmh_v1
    python run_mcmc.py --run_id rwmh_v2 --seed 123

Output
------
    outputs/mcmc/<run_id>.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from the experiment directory itself (config.py)
sys.path.insert(0, str(Path(__file__).parent))

import arviz as az
from probpipe import condition_on

from config import load_model, MCMC_KWARGS, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RWMH for synthetic_single_site.")
    parser.add_argument("--run_id", required=True, help="Unique identifier for this run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    args = parser.parse_args()

    print(f"Loading model and data...")
    model, obs_nee = load_model()

    print(f"Running MCMC (run_id={args.run_id!r}, seed={args.seed})...")
    posterior = condition_on(model, obs_nee, random_seed=args.seed, **MCMC_KWARGS)

    out_path = OUTPUT_DIR / "mcmc" / f"{args.run_id}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(posterior.inference_data, str(out_path))

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

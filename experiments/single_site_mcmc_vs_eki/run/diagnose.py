"""The diagnostics of a run: the predictive check, NEE's residuals, the two towers.

Overview
--------
Runs ``model/diagnostics.py`` on one run's predictive and writes its tables
beside the run, as ``MODEL.md``, "Diagnostics", defines them. Every run is
diagnosed the same way, so runs compare directly:

- ``--model <name> --run <run>``: the predictive ``run/predict.py`` wrote
  for a calibration run, against the calibration vector and the held-out
  tower, under the model's ``R`` at the run's posterior median scales;
- ``--run prior``: the prior predictive's ensemble (a prior predictive
  check), under ``--model``'s ``R`` at scale 1.

Input data
----------
The run's ``predictive/`` (``model/outputs.py``'s ``load_predictive``), its
``samples.csv`` for the scales, the observation vectors, and ``R`` as
``config`` and the model set it now.

Output data
-----------
Under the run's directory, ``diagnostics/``:

- ``predictive_check_calibration.csv``, ``predictive_check_validation.csv``:
  ``diagnostics.predictive_check``;
- ``nee_residuals.csv``: each calibration NEE window's residual;
- ``nee_residual_summary.csv``, ``nee_weekly_residuals.csv``,
  ``nee_autocorrelation.csv``, ``nee_slow_fast.csv``, ``nee_night_day.csv``:
  the residuals' size, recurring seasonal part, autocorrelation, slow and
  fast parts, and night-day correlation;
- ``nee_towers.csv``, ``nee_towers_summary.csv``: the two towers' windows
  and what their differences bound (the same for every run).

``figures/diagnostics.py`` draws them.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.diagnose --model long_memory/inferred --run eki_gibbs_common_smc
    uv run python -m experiments.single_site_mcmc_vs_eki.run.diagnose --run prior
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from ..model import diagnostics
from ..model.outputs import load_predictive, run_directory
from ..models import MODEL_NAMES, Model, fixed_posterior, heldout_posterior

__all__ = ["diagnose", "diagnose_run", "main", "posterior_median_scales"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Diagnose one stored run and write its tables."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    try:
        diagnose_run(arguments.model, arguments.run)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def diagnose_run(model_name: str, run: str) -> None:
    """Diagnose run *run* of model *model_name* (``prior``: the prior
    predictive), write its tables under its ``diagnostics/`` and print the
    main ones.

    Raises
    ------
    FileNotFoundError
        If the run has no predictive.
    """
    directory = run_directory(model_name, run)
    predictive = directory if run == "prior" else directory / "predictive"
    model = Model.parse(model_name)
    outputs = load_predictive(predictive)
    scales = {} if run == "prior" else posterior_median_scales(predictive)
    calibration = diagnostics.sources_from_predictive(
        outputs, "calibration", fixed_posterior(model), scales=scales
    )
    validation = diagnostics.sources_from_predictive(
        outputs, "validation", heldout_posterior(model), scales=scales
    )
    tables = diagnose(calibration, validation, model.nee_error, scales)
    write_tables(directory / "diagnostics", tables)
    print_report(f"{model_name} {run}", tables)


# ── the steps ──


def posterior_median_scales(predictive: Path) -> dict[str, float]:
    """The median of each source's scale in a predictive's ``samples.csv``
    (a column per scaled source, ``run/predict.py``), by source; empty for a
    run with fixed noise."""
    samples = pd.read_csv(predictive / "samples.csv")
    return {
        name: float(samples[name].median())
        for name in config.NOISE_SCALED_SOURCES
        if name in samples
    }


def diagnose(
    calibration: dict,
    validation: dict,
    nee_error_model_name: str,
    scales: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """Every table, by the name its file takes."""
    residuals = diagnostics.nee_residuals(calibration)
    tower_windows, tower_summary = diagnostics.tower_comparison(
        config.CALIBRATION_NEE_SERIES,
        config.VALIDATION_NEE_SERIES,
        (config.CALIBRATION_NEE_PERIOD[0], config.VALIDATION_NEE_PERIOD[1]),
    )
    return {
        "predictive_check_calibration": diagnostics.predictive_check(calibration),
        "predictive_check_validation": diagnostics.predictive_check(validation),
        "nee_residuals": residuals,
        "nee_residual_summary": diagnostics.residual_summary(residuals, scales=scales),
        "nee_weekly_residuals": diagnostics.weekly_residuals(residuals),
        "nee_autocorrelation": diagnostics.residual_autocorrelation(
            residuals, nee_error_model_name
        ),
        "nee_slow_fast": diagnostics.slow_fast_split(residuals),
        "nee_night_day": diagnostics.night_day_correlation(residuals).to_frame("value"),
        "nee_towers": tower_windows,
        "nee_towers_summary": tower_summary,
    }


def write_tables(directory: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Each table as ``<name>.csv``; a stale table of an earlier diagnosis is removed."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.csv"):
        path.unlink()
    for name, table in tables.items():
        index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(directory / f"{name}.csv", index=index)
    print(f"wrote {directory}")


def print_report(label: str, tables: dict[str, pd.DataFrame]) -> None:
    """The tables a reader looks at first."""
    with pd.option_context(
        "display.width", 200, "display.max_columns", 20, "display.precision", 3
    ):
        for name in (
            "predictive_check_calibration",
            "predictive_check_validation",
            "nee_residual_summary",
            "nee_slow_fast",
            "nee_night_day",
            "nee_towers_summary",
        ):
            print(f"\n{label}: {name}\n{tables[name].to_string()}")
        autocorrelation = tables["nee_autocorrelation"].unstack("lag_days")
        print(f"\n{label}: nee_autocorrelation\n{autocorrelation.round(3)}")


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: which model and run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="long_memory/fixed",
        help="the model whose R is used; long_memory/fixed by default",
    )
    parser.add_argument(
        "--run", required=True, help="the run's directory name, or prior"
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())

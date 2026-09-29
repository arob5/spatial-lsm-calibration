"""The diagnostics of a run: the predictive check, NEE's residuals, the two towers.

Overview
--------
Runs ``model/diagnostics.py`` on one run and writes its tables beside the
run, as ``MODEL.md``, "Diagnostics", defines them. Every run of this
experiment is diagnosed the same way, so runs compare directly:

- ``--run prior``: the prior predictive's ensemble (a prior predictive
  check, the same computation against the prior's draws);
- ``--run synthetic``, ``--run observed``: an EKI run's final ensemble,
  against the observations it conditioned on, and, for ``observed`` once
  ``scripts/posterior_predictive.py`` has run, against the held-out tower.

Input data
----------
The run's directory (``config.PRIOR_PREDICTIVE_DIRECTORY``, or
``config.EKI_DIRECTORY / <data>``), the observation vectors, and ``R`` as
``config`` sets it now. A run is diagnosed under the noise model it ran
with only if ``config`` still holds it; the script compares ``config``'s
noise settings with the run's ``provenance.json`` and says when they differ.

Output data
-----------
Under the run's directory, ``diagnostics/``:

- ``predictive_check_calibration.csv``, ``predictive_check_validation.csv``
  (when there are validation predictions): ``diagnostics.predictive_check``;
- ``nee_residuals.csv``: each calibration NEE window's residual;
- ``nee_residual_summary.csv``, ``nee_weekly_residuals.csv``,
  ``nee_autocorrelation.csv``, ``nee_night_day.csv``: the residuals'
  size, recurring seasonal part, autocorrelation and night-day correlation;
- ``nee_towers.csv``, ``nee_towers_summary.csv``: the two towers' windows
  and what their differences bound (the same for every run).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.diagnose --run observed
    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.diagnose --run prior
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..model import diagnostics, observations
from ..model.outputs import load_eki_run, load_predictive, run_directory

__all__ = ["NOISE_CONFIG_NAMES", "main"]

#: The config constants the noise model is built from.
NOISE_CONFIG_NAMES = (
    "NEE_DISCREPANCY",
    "LAI_STANDARD_DEVIATION_FLOOR",
    "LAI_DISCREPANCY_STANDARD_DEVIATION",
    "LAI_DISCREPANCY_TIMESCALE",
    "WOOD_CARBON_FRACTION_UNCERTAINTY",
    "LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION",
    "SOIL_CARBON_DISCREPANCY_FRACTION",
)


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Diagnose one run and write its tables."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    run_name = _parser().parse_args(argv).run
    directory = run_directory(run_name)
    try:
        calibration, validation = run_sources(run_name, directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    report_noise_model_changes(directory)
    tables = diagnose(calibration, validation)
    write_tables(directory / "diagnostics", tables)
    print_report(run_name, tables)
    return 0


# ── the steps ──


def run_sources(run_name: str, directory: Path):
    """The run's predictions of the calibration vector, and of the validation vector if any."""
    calibration_vector = observations.calibration_observation_vector()
    validation_vector = observations.validation_observation_vector()
    if run_name == "prior":
        outputs = load_predictive(directory)
        return (
            diagnostics.sources_from_predictive(
                outputs,
                "calibration",
                calibration_vector,
                config.CALIBRATION_NEE_SERIES,
            ),
            diagnostics.sources_from_predictive(
                outputs,
                "validation",
                validation_vector,
                config.VALIDATION_NEE_SERIES,
            ),
        )
    run = load_eki_run(directory)
    check_run_reached_the_posterior(run["beta"], directory)
    y = run["y"] if run["y"] is not None else calibration_vector.y
    calibration = diagnostics.sources_from_flat(
        calibration_vector, y, run["predictions"], config.CALIBRATION_NEE_SERIES
    )
    validation = None
    # The posterior predictive's validation predictions are of the held-out
    # tower's own data, which a synthetic run did not condition on.
    predictive = directory / "posterior_predictive"
    if run_name == "observed" and (predictive / "ensemble_daily.nc").exists():
        validation = diagnostics.sources_from_predictive(
            load_predictive(predictive),
            "validation",
            validation_vector,
            config.VALIDATION_NEE_SERIES,
        )
    return calibration, validation


def report_noise_model_changes(directory: Path) -> None:
    """Say when ``config``'s noise settings are not the ones the run recorded."""
    path = directory / "provenance.json"
    recorded = json.loads(path.read_text())["config"] if path.exists() else {}
    changed = [
        name
        for name in NOISE_CONFIG_NAMES
        if recorded.get(name) != repr(getattr(config, name))
    ]
    if changed:
        print(
            "note: the run's provenance.json does not record these noise settings "
            "as config holds them now, so R here may not be the R it ran with: "
            + ", ".join(changed)
        )


def diagnose(calibration: dict, validation: dict | None) -> dict[str, pd.DataFrame]:
    """Every table, by the name its file takes."""
    residuals = diagnostics.nee_residuals(calibration)
    tower_windows, tower_summary = diagnostics.tower_comparison(
        config.CALIBRATION_NEE_SERIES,
        config.VALIDATION_NEE_SERIES,
        (config.CALIBRATION_NEE_PERIOD[0], config.VALIDATION_NEE_PERIOD[1]),
    )
    tables = {
        "predictive_check_calibration": diagnostics.predictive_check(calibration),
        "nee_residuals": residuals,
        "nee_residual_summary": diagnostics.residual_summary(residuals),
        "nee_weekly_residuals": diagnostics.weekly_residuals(residuals),
        "nee_autocorrelation": diagnostics.residual_autocorrelation(residuals),
        "nee_night_day": diagnostics.night_day_correlation(residuals).to_frame("value"),
        "nee_towers": tower_windows,
        "nee_towers_summary": tower_summary,
    }
    if validation is not None:
        tables["predictive_check_validation"] = diagnostics.predictive_check(validation)
    return tables


def write_tables(directory: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Each table as ``<name>.csv``; a stale table of an earlier diagnosis is removed."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.csv"):
        path.unlink()
    for name, table in tables.items():
        index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(directory / f"{name}.csv", index=index)
    print(f"wrote {directory}")


def print_report(run_name: str, tables: dict[str, pd.DataFrame]) -> None:
    """The tables a reader looks at first."""
    with pd.option_context(
        "display.width", 200, "display.max_columns", 20, "display.precision", 3
    ):
        for name in (
            "predictive_check_calibration",
            "predictive_check_validation",
            "nee_residual_summary",
            "nee_night_day",
            "nee_towers_summary",
        ):
            if name in tables:
                print(f"\n{run_name}: {name}\n{tables[name].to_string()}")
        autocorrelation = tables["nee_autocorrelation"].unstack("lag_days")
        print(f"\n{run_name}: nee_autocorrelation\n{autocorrelation.round(3)}")


# ── helpers ──


def _parser() -> argparse.ArgumentParser:
    """The command line: which run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run", choices=("prior", "synthetic", "observed"), required=True
    )
    return parser


# ── checks ──


def check_run_reached_the_posterior(beta: float, directory: Path) -> None:
    """An EKI run's last step reached beta = 1."""
    if not np.isclose(beta, 1.0):
        raise ValueError(
            f"the run under {directory} ended at beta {beta:g}, not 1; "
            "finish it with scripts/eki.py --resume"
        )


if __name__ == "__main__":
    sys.exit(main())

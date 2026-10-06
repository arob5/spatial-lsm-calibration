"""The diagnostics of a run: the predictive check, NEE's residuals, the two towers.

Overview
--------
Runs ``model/diagnostics.py`` on one run, writes its tables beside the run,
as ``MODEL.md``, "Diagnostics", defines them, and draws their figures
(``figures/diagnostics.py``). Every run of this experiment is diagnosed the
same way, so runs compare directly. The predictive runs diagnose themselves
at their end (:func:`diagnose_run`); this script rediagnoses a stored run:

- ``--run prior``: the prior predictive's ensemble (a prior predictive
  check, the same computation against the prior's draws);
- ``--run synthetic``, ``--run observed``: an EKI run's final ensemble,
  against the observations it conditioned on, and, for ``observed`` once
  ``run/posterior_predictive.py`` has run, against the held-out tower.

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
  ``nee_autocorrelation.csv``, ``nee_slow_fast.csv``, ``nee_night_day.csv``:
  the residuals' size, recurring seasonal part, autocorrelation, slow and
  fast parts, and night-day correlation;
- ``nee_towers.csv``, ``nee_towers_summary.csv``: the two towers' windows
  and what their differences bound (the same for every run).

The figures go where ``figures/diagnostics.py`` says.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.diagnose --run observed
    uv run python -m experiments.single_site_mcmc_vs_eki.run.diagnose --run prior
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from .. import config
from ..figures.diagnostics import draw_diagnostic_figures
from ..model import calibration, diagnostics
from ..model.outputs import (
    check_eki_run_finished,
    load_eki_run,
    load_predictive,
    run_directory,
)

__all__ = ["NOISE_CONFIG_NAMES", "diagnose_run", "main"]

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
    """Diagnose one stored run: write its tables and draw their figures."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    try:
        diagnose_run(_parser().parse_args(argv).run)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def diagnose_run(run_name: str) -> None:
    """Diagnose run *run_name* (``prior``, ``synthetic`` or ``observed``).

    Writes its tables under its ``diagnostics/``, prints the main ones and
    draws their figures.

    Raises
    ------
    FileNotFoundError
        If the run, or for an EKI run its final step, is missing.
    ValueError
        If an EKI run has not reached beta = 1.
    """
    directory = run_directory(run_name)
    calibration, validation = run_sources(run_name, directory)
    report_noise_model_changes(directory)
    tables = diagnose(calibration, validation)
    write_tables(directory / "diagnostics", tables)
    print_report(run_name, tables)
    draw_diagnostic_figures(run_name)


# ── the steps ──


def run_sources(run_name: str, directory: Path):
    """The run's predictions of the calibration vector, and of the validation vector if any."""
    if run_name == "prior":
        outputs = load_predictive(directory)
        return (
            diagnostics.sources_from_predictive(
                outputs, "calibration", calibration.calibration_posterior()
            ),
            diagnostics.sources_from_predictive(
                outputs, "validation", calibration.validation_posterior()
            ),
        )
    run = load_eki_run(directory)
    check_eki_run_finished(run, directory)
    validation = None
    # The posterior predictive's validation predictions are of the held-out
    # tower's own data, which a synthetic run did not condition on.
    predictive = directory / "posterior_predictive"
    if run_name == "observed" and (predictive / "ensemble_daily.nc").exists():
        validation = diagnostics.sources_from_predictive(
            load_predictive(predictive),
            "validation",
            calibration.validation_posterior(),
        )
    return diagnostics.sources_from_eki_run(run), validation


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
        "nee_slow_fast": diagnostics.slow_fast_split(residuals),
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
            "nee_slow_fast",
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


if __name__ == "__main__":
    sys.exit(main())

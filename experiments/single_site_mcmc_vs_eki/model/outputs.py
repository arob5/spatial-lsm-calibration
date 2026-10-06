"""Reading what the runs wrote: a calibration run, a predictive, diagnostics.

The runs write; this module reads, so the diagnostics, the comparison and
the figures read a run one way. Nothing here runs a model or writes a file.

Functions
---------
:func:`run_directory`
    Where a run's outputs are, by model and run name; ``prior`` is the
    prior predictive.
:func:`load_run`
    A calibration run's directory (``algorithms/records.py``): its samples,
    natural values, cost and history.
:func:`load_predictive`
    A predictive's directory (the prior predictive, ``run/_predictive.py``,
    or a run's, ``run/predict.py``), as nested dicts of xarray objects at
    the site.
:func:`load_diagnostics`
    A run's diagnostics (``run/diagnose.py``), one table per file.
:func:`at_site`
    Data at the configured site, the ``site`` dim dropped.
"""

import json
from pathlib import Path

import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.conventions import SITE

from .. import config, models
from ..algorithms.records import load_samples

__all__ = [
    "DIAGNOSTIC_INDEX_COLUMNS",
    "VECTOR_NAMES",
    "at_site",
    "check_run_was_diagnosed",
    "load_diagnostics",
    "load_predictive",
    "load_run",
    "run_directory",
]

#: The observation vectors a predictive predicts, by the names its files use.
VECTOR_NAMES = ("calibration", "validation")

#: The index columns of each diagnostics table, by table name; a table not
#: listed has none.
DIAGNOSTIC_INDEX_COLUMNS = frozendict(
    {
        "predictive_check_calibration": "source",
        "predictive_check_validation": "source",
        "nee_residual_summary": "source",
        "nee_slow_fast": "source",
        "nee_weekly_residuals": "week",
        "nee_autocorrelation": ["source", "lag_days"],
        "nee_night_day": 0,
        "nee_towers_summary": "source",
    }
)

#: The diagnostics tables with a ``time`` column.
_TIMED_TABLES = ("nee_residuals", "nee_towers")


def run_directory(model_name: str, run: str) -> Path:
    """Where run *run* of model *model_name* is (``models.Model.directory``),
    or the prior predictive's directory for ``run == "prior"``."""
    if run == "prior":
        return config.PRIOR_PREDICTIVE_DIRECTORY
    return models.Model.parse(model_name).directory(run)


def load_run(directory: Path) -> dict:
    """A calibration run's outputs (``algorithms/records.py``).

    Returns ``{"samples": Dataset, "natural_values": DataFrame, "cost":
    dict, "history": DataFrame or None}``, the natural values one row per
    sample with its ``log_weight``.
    """
    directory = Path(directory)
    history = directory / "history.csv"
    return {
        "samples": load_samples(directory),
        "natural_values": pd.read_csv(directory / "natural_values.csv", index_col=0),
        "cost": json.loads((directory / "cost.json").read_text()),
        "history": pd.read_csv(history) if history.exists() else None,
    }


def load_predictive(directory: Path | None = None) -> dict:
    """A predictive's outputs, as nested dicts of xarray objects.

    Returns ``{"daily": {run: Dataset}, "predicted": {run: {vector: {source:
    field}}}, "observed": {vector: {source: dataset}}}``, every field at the
    site, the runs being ``ensemble`` and, when the predictive has one run
    by hand, ``single_run``. *directory* defaults to the prior predictive's.

    Raises
    ------
    FileNotFoundError
        If the ensemble's daily file is missing: the predictive has not run.
    """
    directory = directory or config.PRIOR_PREDICTIVE_DIRECTORY
    runs = ["ensemble"]
    if (directory / "single_run_daily.nc").exists():
        runs = ["single_run", *runs]
    outputs = {
        "daily": {
            run: at_site(xr.load_dataset(directory / f"{run}_daily.nc")) for run in runs
        },
        "predicted": {},
        "observed": {},
    }
    for run in runs:
        outputs["predicted"][run] = {
            vector: {
                path.stem: at_site(xr.load_dataarray(path))
                for path in sorted(
                    (directory / "predictions" / run / vector).glob("*.nc")
                )
            }
            for vector in VECTOR_NAMES
        }
    outputs["observed"] = {
        vector: {
            path.stem: at_site(xr.load_dataset(path))
            for path in sorted((directory / "observed" / vector).glob("*.nc"))
        }
        for vector in VECTOR_NAMES
    }
    return outputs


def load_diagnostics(directory: Path) -> dict[str, pd.DataFrame]:
    """A run's diagnostics, ``<directory>/diagnostics/<name>.csv``, by name.

    Each table has the index ``run/diagnose.py`` wrote it with, and the
    ``time`` columns are parsed.

    Raises
    ------
    FileNotFoundError
        If the run has not been diagnosed.
    """
    diagnostics_directory = directory / "diagnostics"
    paths = sorted(diagnostics_directory.glob("*.csv"))
    check_run_was_diagnosed(paths, directory)
    return {
        path.stem: pd.read_csv(
            path,
            index_col=DIAGNOSTIC_INDEX_COLUMNS.get(path.stem),
            parse_dates=["time"] if path.stem in _TIMED_TABLES else False,
        )
        for path in paths
    }


def at_site(data):
    """*data* at the configured site, the ``site`` dim dropped to a scalar."""
    return data.sel({SITE: config.SITE}) if SITE in data.dims else data


# ── checks ──


def check_run_was_diagnosed(paths: list[Path], directory: Path) -> None:
    """A run has diagnostics tables."""
    if not paths:
        raise FileNotFoundError(
            f"no diagnostics under {directory / 'diagnostics'}; run its "
            "predictive (run/prior_predictive.py or run/predict.py), then "
            "run/diagnose.py"
        )

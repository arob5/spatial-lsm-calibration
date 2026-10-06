"""Reading what the runs wrote: a predictive, and an EKI run.

The runs write; this module reads, so the diagnostics and the figures
read a run one way. Nothing here runs a model or writes a file.

Functions
---------
:func:`load_predictive`
    A predictive's directory (``run/_predictive.py``'s layout: the prior
    predictive, or an EKI run's posterior predictive), as nested dicts of
    xarray objects at the site.
:func:`load_eki_run`
    An EKI run's directory (``run/eki.py``): its history, its first and
    final ensembles, the final ensemble's predictions of the calibration
    vector, and the synthetic truth when there is one.
:func:`load_diagnostics`
    A run's diagnostics (``run/diagnose.py``), one table per file.
:func:`run_directory`
    Where a run's outputs are, by the run's name.
:func:`at_site`
    Data at the configured site, the ``site`` dim dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.conventions import SITE

from .. import config

__all__ = [
    "DIAGNOSTIC_INDEX_COLUMNS",
    "VECTOR_NAMES",
    "at_site",
    "check_eki_run_finished",
    "check_eki_run_wrote_a_step",
    "check_run_was_diagnosed",
    "load_diagnostics",
    "load_eki_run",
    "load_predictive",
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


def load_eki_run(directory: Path) -> dict:
    """An EKI run's outputs.

    Returns ``{"history": DataFrame, "prior": DataFrame, "posterior":
    DataFrame or None, "theta_prior": (J, D), "theta_posterior": (J, D),
    "predictions": (J, N), "beta": float, "finished": bool, "y": (N,) or
    None, "truth": DataFrame or None, "theta_true": (D,) or None}``.

    ``theta_prior`` is the initial ensemble as drawn; ``theta_posterior`` the
    ensemble after the last step. ``predictions`` are the last step's, in
    the calibration posterior's y order, a member's row NaN where its sample
    was invalid, since the evaluation moved it to the valid center. ``finished`` says whether that last step is the
    evaluation of the final ensemble, at beta = 1 with no update, so the
    predictions are the posterior's. ``y`` is the synthetic observations, or
    ``None`` for a run on the calibration vector's own.

    Raises
    ------
    FileNotFoundError
        If the run wrote no step.
    """
    steps = sorted((directory / "steps").glob("step_*.npz"))
    check_eki_run_wrote_a_step(steps, directory)
    first, last = np.load(steps[0]), np.load(steps[-1])
    history = pd.read_csv(directory / "history.csv")
    predictions = np.array(last["predictions"])
    if "valid" in last:
        predictions[~last["valid"]] = np.nan
    initial_path = directory / "initial_ensemble.npy"
    synthetic_path = directory / "synthetic.npz"
    synthetic = np.load(synthetic_path) if synthetic_path.exists() else None
    posterior_path = directory / "posterior_ensemble.csv"
    final = history.iloc[-1]
    return {
        "history": history,
        "prior": pd.read_csv(directory / "prior_ensemble.csv", index_col=0),
        "posterior": (
            pd.read_csv(posterior_path, index_col=0)
            if posterior_path.exists()
            else None
        ),
        "theta_prior": (
            np.load(initial_path) if initial_path.exists() else first["ensemble"]
        ),
        "theta_posterior": last["next_ensemble"],
        "predictions": predictions,
        "beta": float(last["next_beta"]),
        "finished": bool(final["increment"] == 0 and np.isclose(final["beta"], 1.0)),
        "y": synthetic["y"] if synthetic is not None else None,
        "truth": (
            pd.read_csv(directory / "truth.csv", index_col=0)
            if synthetic is not None
            else None
        ),
        "theta_true": synthetic["theta_true"] if synthetic is not None else None,
    }


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


def run_directory(run_name: str) -> Path:
    """Where a run's outputs are: ``prior`` (the prior predictive), or an EKI
    run's data, ``synthetic`` or ``observed``."""
    if run_name == "prior":
        return config.PRIOR_PREDICTIVE_DIRECTORY
    return config.EKI_DIRECTORY / run_name


def at_site(data):
    """*data* at the configured site, the ``site`` dim dropped to a scalar."""
    return data.sel({SITE: config.SITE}) if SITE in data.dims else data


# ── checks ──


def check_eki_run_wrote_a_step(steps: list[Path], directory: Path) -> None:
    """An EKI run wrote at least one step."""
    if not steps:
        raise FileNotFoundError(
            f"no step under {directory / 'steps'}; run run/eki.py first"
        )


def check_run_was_diagnosed(paths: list[Path], directory: Path) -> None:
    """A run has diagnostics tables."""
    if not paths:
        raise FileNotFoundError(
            f"no diagnostics under {directory / 'diagnostics'}; run its "
            "predictive (run/prior_predictive.py or run/posterior_predictive.py), "
            "or run/diagnose.py"
        )


def check_eki_run_finished(run: dict, directory: Path) -> None:
    """An EKI run reached beta = 1 and evaluated its final ensemble."""
    if not run["finished"]:
        raise ValueError(
            f"the run under {directory} has not evaluated a final ensemble at "
            "beta = 1; finish it with run/eki.py --resume"
        )

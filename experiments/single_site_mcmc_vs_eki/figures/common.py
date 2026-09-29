"""What every figure of the experiment shares: reading a run's outputs, and the legend.

Functions
---------
:func:`load_predictive`
    What ``scripts/predictive.py`` wrote for a prior or a posterior
    predictive, as nested dicts of xarray objects at the site.
:func:`at_site`
    Data at the configured site, the ``site`` dim dropped.
:func:`one_legend`
    One deduplicated legend for a figure of several panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SITE

from .. import config

__all__ = ["NEE_TITLES", "at_site", "load_predictive", "one_legend"]

#: Panel titles of the NEE sources.
NEE_TITLES = {
    "nee_night_centered": "NEE, 00-12 UTC (night-centered)",
    "nee_day_centered": "NEE, 12-24 UTC (day-centered)",
}


def load_predictive(directory: Path | None = None) -> dict:
    """What ``scripts/predictive.py`` wrote, as nested dicts of xarray objects.

    Returns ``{"daily": {run: Dataset}, "predicted": {run: {vector: {source:
    field}}}, "observed": {vector: {source: dataset}}}``, every field at the
    site, the runs being ``ensemble`` and, when the predictive has one run
    by hand, ``single_run``. *directory* defaults to the prior predictive's.
    Raises ``FileNotFoundError`` when the ensemble's daily file is missing.
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
            for vector in ("calibration", "validation")
        }
    outputs["observed"] = {
        vector: {
            path.stem: at_site(xr.load_dataset(path))
            for path in sorted((directory / "observed" / vector).glob("*.nc"))
        }
        for vector in ("calibration", "validation")
    }
    return outputs


def at_site(data):
    """*data* at the configured site, the ``site`` dim dropped to a scalar."""
    return data.sel({SITE: config.SITE}) if SITE in data.dims else data


def one_legend(figure: plt.Figure, axes) -> None:
    """One legend for the figure, from the entries of every panel, deduplicated."""
    entries = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    figure.legend(
        entries.values(), entries.keys(), loc="outside lower center", ncol=len(entries)
    )

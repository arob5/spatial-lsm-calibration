"""What every figure of the experiment shares: panel titles, the legend, saving.

The figures read what the runs wrote through ``model/outputs.py``.

Functions
---------
:func:`one_legend`
    One deduplicated legend for a figure of several panels.
:func:`save_figure`
    Write a figure into ``config.FIGURE_DIRECTORY``, or an EKI setup's
    subdirectory of it, and close it.
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import config

__all__ = ["NEE_TITLES", "SOURCE_LABELS", "one_legend", "save_figure"]

#: The observation sources' labels, in the order the figures draw them.
SOURCE_LABELS = {
    "nee_night_centered": "NEE, night-centered",
    "nee_day_centered": "NEE, day-centered",
    "modis_leaf_area_index": "MODIS LAI",
    "landtrendr_aboveground_biomass": "LandTrendr biomass",
    "soilgrids_soil_organic_carbon": "SoilGrids soil C",
}

#: Panel titles of the NEE sources.
NEE_TITLES = {
    "nee_night_centered": "NEE, 00-12 UTC (night-centered)",
    "nee_day_centered": "NEE, 12-24 UTC (day-centered)",
}


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


def save_figure(
    figure: plt.Figure,
    name: str,
    *,
    eki_run: bool = False,
    subdirectory: str | None = None,
) -> None:
    """Write *figure* as ``<name>.png`` and close it.

    Into ``config.FIGURE_DIRECTORY``; for a figure of an EKI run (*eki_run*),
    into its subdirectory ``config.EKI_RUN_NAME``, beside the figures of the
    setup's other runs; or into the named *subdirectory*.
    """
    directory = config.FIGURE_DIRECTORY
    if eki_run:
        directory = directory / config.EKI_RUN_NAME
    elif subdirectory is not None:
        directory = directory / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")

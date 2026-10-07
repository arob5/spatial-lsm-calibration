"""What every figure of the experiment shares: panel titles, the legend, saving.

The figures read what the runs wrote through ``model/outputs.py``.

Functions
---------
:func:`one_legend`
    One deduplicated legend for a figure of several panels.
:func:`save_figure`
    Write a figure into a directory, ``config.FIGURE_DIRECTORY`` by
    default, and close it.
:func:`weekly_nee_quantiles`, :func:`week_of_year`
    An NEE source's observed and predicted weekly means over the year.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sipnet_calibration.conventions import SAMPLE, TIME

from .. import config

__all__ = [
    "NEE_TITLES",
    "PARAMETER_TITLES",
    "SLIDE_STYLE",
    "SOURCE_LABELS",
    "one_legend",
    "save_figure",
    "week_of_year",
    "weekly_nee_quantiles",
]

#: The observation sources' labels, in the order the figures draw them.
SOURCE_LABELS = {
    "nee_night_centered": "NEE, night-centered",
    "nee_day_centered": "NEE, day-centered",
    "modis_leaf_area_index": "MODIS LAI",
    "landtrendr_aboveground_biomass": "LandTrendr biomass",
    "soilgrids_soil_organic_carbon": "SoilGrids soil C",
}

#: Font sizes for slides.
SLIDE_STYLE = {
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
}

#: Each calibrated parameter's panel title, with its units.
PARAMETER_TITLES = {
    "photosynthetic_capacity": "Photosynthetic capacity P\n(nmol g⁻¹ s⁻¹)",
    "respiration_share": "Foliar respiration share ρ",
    "optimum_photosynthesis_temperature": "Optimum photosynthesis\ntemperature (°C)",
    "half_saturation_light": "Half-saturation light\n(mol m⁻² d⁻¹)",
    "soil_water_holding_capacity": "Soil water holding\ncapacity (cm)",
    "leaf_on_growth": "Leaf growth at leaf-on\n(g C m⁻²)",
    "leaf_on_growing_degree_days": "Leaf-on growing\ndegree-days (°C d)",
    "allocation.leaf": "Allocation: leaf",
    "allocation.wood": "Allocation: wood",
    "allocation.fine_root": "Allocation: fine root",
    "allocation.coarse_root": "Allocation: coarse root",
    "wood_respiration_rate_at_10c": "Wood respiration rate\nat 10 °C (yr⁻¹)",
    "soil_respiration_flux_at_10c": "Soil respiration flux\nat 10 °C (g C m⁻² yr⁻¹)",
    "soil_respiration_q10": "Soil respiration Q₁₀",
    "initial_wood_carbon": "Initial wood carbon\n(kg C m⁻²)",
    "initial_soil_organic_carbon": "Initial soil carbon\n(kg C m⁻²)",
}

#: Panel titles of the NEE sources: the windows are 00-12 and 12-24 UTC,
#: Harvard Forest's standard time (EST) plus 5 hours.
NEE_TITLES = {
    "nee_night_centered": "NEE, night (7 pm to 7 am EST)",
    "nee_day_centered": "NEE, day (7 am to 7 pm EST)",
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
    figure: plt.Figure, name: str, directory: Path = config.FIGURE_DIRECTORY
) -> None:
    """Write *figure* as ``<directory>/<name>.png`` and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")


def weekly_nee_quantiles(
    outputs: dict, name: str, vector: str = "calibration"
) -> tuple[pd.Series, pd.DataFrame]:
    """One NEE source of *vector* by week of year: the observed means, and the predictive's quantiles.

    The observations are averaged by the week of each window's end, and each
    member's predictions over the same windows; the quantiles, columns 0.05,
    0.25, 0.5, 0.75 and 0.95, are over the members' weekly means.
    """
    observed = outputs["observed"][vector][name]["value"]
    predicted = outputs["predicted"]["ensemble"][vector][name]
    weeks = week_of_year(observed[TIME])
    observed_weekly = pd.Series(observed.to_numpy()).groupby(weeks).mean()
    predicted_weekly = (
        pd.DataFrame(predicted.transpose(SAMPLE, TIME).to_numpy().T)
        .groupby(weeks)
        .mean()
    )
    quantiles = predicted_weekly.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    return observed_weekly, quantiles


def week_of_year(times) -> np.ndarray:
    """The ISO week of each window's end, at most 52."""
    weeks = pd.DatetimeIndex(np.asarray(times)).isocalendar().week.to_numpy()
    return np.minimum(weeks, 52)

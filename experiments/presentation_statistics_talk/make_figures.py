"""The talk's figures that need the model's posteriors built, drawn once.

Overview
--------
The predictive figures replicate the data (``analysis/predictive.py``),
which builds the model's posteriors, a minute or two per error model, too
slow for every render of the deck. This script draws them into
``figures/generated/``, which ``slides.qmd`` includes as images:

- ``prior_predictive_nee.png``: the prior predictive of the NEE windows'
  weekly means (long memory, inferred noise);
- ``posterior_predictive_nee.png``: the same for the posterior (MCMC), with
  the summer uptake it misses marked;
- ``error_models_nee.png``: the three error models' posterior predictives
  (inferred noise, MCMC);
- ``algorithms_nee.png``: EKI's, SMC's and MCMC's posterior predictives
  (long memory, inferred noise);
- ``daily_nee_summer.png``: one summer's daytime NEE windows, observed
  against the posterior median prediction under short and long memory.

Input data
----------
The experiment's runs and prior predictive (``output/``).

Usage
-----
From the repository root::

    uv run python experiments/presentation_statistics_talk/make_figures.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from sipnet_calibration.conventions import SAMPLE, TIME  # noqa: E402
from sipnet_calibration.plotting.style import use_project_style  # noqa: E402

from experiments.single_site_mcmc_vs_eki import config  # noqa: E402
from experiments.single_site_mcmc_vs_eki.figures.algorithms import (  # noqa: E402
    COMPARED_ALGORITHMS,
    draw_band,
    load_algorithm_runs,
    weekly_band,
)
from experiments.single_site_mcmc_vs_eki.figures.common import week_of_year  # noqa: E402
from experiments.single_site_mcmc_vs_eki.figures.error_models import (  # noqa: E402
    ERROR_MODEL_COLORS,
    ERROR_MODEL_LABELS,
    load_error_model_runs,
)
from experiments.single_site_mcmc_vs_eki.figures.prior_predictive import (  # noqa: E402
    prior_predictive_of_the_data,
)
from experiments.single_site_mcmc_vs_eki.model.outputs import load_predictive  # noqa: E402
from experiments.single_site_mcmc_vs_eki.models import Model  # noqa: E402

#: Where the figures go.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures" / "generated"

#: The model of the single-model figures.
MODEL = Model("long_memory", "inferred")

#: The NEE sources, with their panel titles: the windows are 00-12 and 12-24
#: UTC, which is Harvard Forest's standard time (EST) plus 5 hours.
NEE_PANELS = {
    "nee_night_centered": "Night (7 pm to 7 am EST)",
    "nee_day_centered": "Day (7 am to 7 pm EST)",
}

#: The x-axis label of the weekly figures, which pool the calibration years.
WEEK_LABEL = "week of year (mean over {}-{})".format(*config.CALIBRATION_NEE_PERIOD)

#: The colors of the prior and the posterior bands, and of the prior band
#: behind a posterior.
PRIOR_COLOR = "#7f7f7f"
PRIOR_BACKGROUND_COLOR = "#e3e3e3"
POSTERIOR_COLOR = "#0072B2"

#: The summer of the daily figure, and the error models it compares.
DAILY_PERIOD = ("2015-06-01", "2015-09-01")
DAILY_ERROR_MODELS = ("short_memory", "long_memory")

#: Slide-sized fonts.
STYLE = {
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
}


def main() -> None:
    """Draw every figure into :data:`OUTPUT_DIRECTORY`."""
    warnings.filterwarnings("ignore")
    use_project_style()
    plt.rcParams.update(STYLE)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prior = prior_predictive_of_the_data(load_predictive(config.PRIOR_PREDICTIVE_DIRECTORY), MODEL)
    _save(_prior_predictive_nee(prior), "prior_predictive_nee")
    prior_bands = _prior_bands(prior)
    algorithm_runs = load_algorithm_runs(MODEL, COMPARED_ALGORITHMS)
    mcmc = next(run for run in algorithm_runs if run.algorithm.label == "MCMC")
    _save(_posterior_predictive_nee(mcmc, prior_bands), "posterior_predictive_nee")
    _save(_algorithms_nee(algorithm_runs, prior_bands), "algorithms_nee")
    error_model_runs = load_error_model_runs(MODEL.noise)
    _save(_error_models_nee(error_model_runs, prior_bands), "error_models_nee")
    _save(_daily_nee_summer(error_model_runs), "daily_nee_summer")


# ── the figures ──


def _prior_predictive_nee(prior: dict) -> plt.Figure:
    """The prior predictive's weekly means against the observed, by NEE source."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, name in zip(axes, NEE_PANELS, strict=True):
        observed = prior["observed"]["calibration"][name]["value"]
        replicated = prior["predicted"]["ensemble"]["calibration"][name].transpose(SAMPLE, TIME).to_numpy()
        draw_band(ax, *_weekly(replicated, observed[TIME]), PRIOR_COLOR, "prior predictive")
        _finish_panel(ax, name, observed)
    _legend(figure, axes)
    return figure


def _posterior_predictive_nee(run, prior_bands: dict) -> plt.Figure:
    """The posterior predictive's weekly means against the observed, over the
    prior's band, the summer daytime gap marked."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, name in zip(axes, NEE_PANELS, strict=True):
        observed = run.predictive["observed"]["calibration"][name]["value"]
        _draw_prior_background(ax, prior_bands[name])
        positions, band = _weekly(run.replicated["calibration"][name], observed[TIME])
        draw_band(ax, positions, band, POSTERIOR_COLOR, "posterior predictive")
        weekly_observed = _finish_panel(ax, name, observed)
        if name == "nee_day_centered":
            week = int(weekly_observed.idxmin())
            ax.annotate(
                "",
                xy=(week, weekly_observed[week]),
                xytext=(week, band.loc[week, 0.05]),
                arrowprops={"arrowstyle": "<->", "color": "#D55E00", "linewidth": 2},
            )
            # Below the lowest observed week, where no point is.
            ax.set_ylim(bottom=min(ax.get_ylim()[0], weekly_observed[week] - 2.5))
            ax.text(week + 1, weekly_observed[week] - 1.4, "summer uptake missed",
                    color="#D55E00", va="center", fontsize="large")
    _legend(figure, axes)
    return figure


def _algorithms_nee(runs, prior_bands: dict) -> plt.Figure:
    """Each algorithm's posterior predictive weekly means against the
    observed, over the prior's band."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, name in zip(axes, NEE_PANELS, strict=True):
        observed = runs[0].predictive["observed"]["calibration"][name]["value"]
        _draw_prior_background(ax, prior_bands[name])
        for run in runs:
            draw_band(
                ax,
                *_weekly(run.replicated["calibration"][name], observed[TIME]),
                run.algorithm.color,
                run.algorithm.label,
            )
        _finish_panel(ax, name, observed)
    _legend(figure, axes, n_columns=4)
    return figure


def _error_models_nee(runs: dict, prior_bands: dict) -> plt.Figure:
    """Each error model's posterior predictive weekly means against the
    observed, over the prior's band (:data:`MODEL`'s, which the error models
    barely change)."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, name in zip(axes, NEE_PANELS, strict=True):
        observed = next(iter(runs.values())).predictive["observed"]["calibration"][name]["value"]
        _draw_prior_background(ax, prior_bands[name])
        for error_model, run in runs.items():
            draw_band(
                ax,
                *_weekly(run.replicated["calibration"][name], observed[TIME]),
                ERROR_MODEL_COLORS[error_model],
                ERROR_MODEL_LABELS[error_model],
            )
        _finish_panel(ax, name, observed)
    _legend(figure, axes, n_columns=4)
    return figure


def _daily_nee_summer(runs: dict) -> plt.Figure:
    """One summer's daytime windows: observed, and each error model's
    posterior median prediction (MCMC), the noise left out."""
    name = "nee_day_centered"
    figure, ax = plt.subplots(figsize=(15, 5.5))
    observed = next(iter(runs.values())).predictive["observed"]["calibration"][name]["value"]
    times = pd.DatetimeIndex(observed[TIME].to_numpy())
    inside = (times >= DAILY_PERIOD[0]) & (times < DAILY_PERIOD[1])
    ax.plot(times[inside], observed.to_numpy()[inside], "o-", color="black", markersize=4,
            linewidth=0.8, label="observed")
    for error_model in DAILY_ERROR_MODELS:
        predicted = runs[error_model].predictive["predicted"]["ensemble"]["calibration"][name]
        median = np.nanmedian(predicted.transpose(SAMPLE, TIME).to_numpy(), axis=0)
        ax.plot(times[inside], median[inside], "-", color=ERROR_MODEL_COLORS[error_model],
                linewidth=2.2, label=f"{ERROR_MODEL_LABELS[error_model]} (posterior median)")
    ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_ylabel("daytime NEE (µmol CO₂ m⁻² s⁻¹)")
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    _legend(figure, [ax])
    return figure


# ── helpers ──


def _prior_bands(prior: dict) -> dict[str, pd.DataFrame]:
    """The prior predictive's weekly band of each NEE source."""
    return {
        name: weekly_band(
            prior["predicted"]["ensemble"]["calibration"][name].transpose(SAMPLE, TIME).to_numpy(),
            prior["observed"]["calibration"][name]["value"][TIME],
        )
        for name in NEE_PANELS
    }


def _draw_prior_background(ax, band: pd.DataFrame) -> None:
    """The prior predictive's 90% band, light and behind everything else."""
    ax.fill_between(band.index, band[0.05], band[0.95], color=PRIOR_BACKGROUND_COLOR,
                    linewidth=0, zorder=-1, label="prior predictive, 90%")


def _weekly(values: np.ndarray, times) -> tuple[pd.Index, pd.DataFrame]:
    """The weekly band of *values* ``(K, n)``, and its weeks."""
    band = weekly_band(values, times)
    return band.index, band


def _finish_panel(ax, name: str, observed) -> pd.Series:
    """The observed weekly means, the zero line, the title and labels; returns the means."""
    weekly = observed.to_series().groupby(week_of_year(observed[TIME])).mean()
    ax.plot(weekly.index, weekly.to_numpy(), "o", color="black", markersize=4, label="observed")
    ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_title(NEE_PANELS[name])
    ax.set_xlabel(WEEK_LABEL)
    ax.set_ylabel("NEE (µmol CO₂ m⁻² s⁻¹)")
    return weekly


def _legend(figure, axes, n_columns: int | None = None) -> None:
    """One legend below the panels, deduplicated, in *n_columns* (one row by default)."""
    entries = {}
    for ax in np.ravel(axes):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    figure.legend(entries.values(), entries.keys(), loc="outside lower center",
                  ncol=n_columns or len(entries), frameon=False)


def _save(figure: plt.Figure, name: str) -> None:
    """Write *figure* as ``<name>.png`` and close it."""
    path = OUTPUT_DIRECTORY / f"{name}.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

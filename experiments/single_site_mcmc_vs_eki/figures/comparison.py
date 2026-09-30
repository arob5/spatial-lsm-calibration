"""The EKI setups compared: how the NEE error model moves the calibration.

Reads each compared setup's observed-data run (``output/eki/<setup>/observed``:
its posterior ensemble, posterior predictive and diagnostics) and draws,
sized for slides, into ``config.FIGURE_DIRECTORY / "comparison"``:

- :func:`plot_seasonal_cycles` (``comparison_seasonal_cycles``): NEE's
  seasonal cycle, observed against each setup's posterior predictive, weekly
  means over the windows the likelihood reads;
- :func:`plot_moving_parameters` (``comparison_parameters``): the
  parameters the setups move, each setup's posterior 90% interval and median
  against the prior's;
- :func:`plot_slow_fast` (``comparison_slow_fast``): the daytime residual's
  slow and fast variance per setup, and each setup's day-to-day variability
  of NEE relative to the observed (``MODEL.md``, "Diagnostics").

``MODEL.md``, "The error model and the fast-slow trade-off", reads them.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.figures.comparison
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frozendict import frozendict

from sipnet_calibration.plotting.style import (
    CURVE_COLORS,
    role_style,
    use_project_style,
)

from .. import config
from ..model.outputs import load_diagnostics, load_predictive
from .common import NEE_TITLES, one_legend, save_figure
from .prior_predictive import PARAMETER_TITLES, SLIDE_STYLE, weekly_nee_quantiles

__all__ = [
    "COMPARED_SETUPS",
    "MOVING_PARAMETER_NAMES",
    "load_setups",
    "plot_moving_parameters",
    "plot_seasonal_cycles",
    "plot_slow_fast",
]

#: The EKI setups compared, in the order they were run, with their labels.
COMPARED_SETUPS = frozendict(
    {
        "single_term_discrepancy": "one term, 2 days",
        "three_term_discrepancy": "three terms, one recurring",
        "two_term_discrepancy": "two terms, short and long",
    }
)

#: The parameters the setups move most.
MOVING_PARAMETER_NAMES = (
    "photosynthetic_capacity",
    "half_saturation_light",
    "respiration_share",
    "optimum_photosynthesis_temperature",
    "wood_respiration_rate_at_10c",
    "soil_respiration_q10",
)


def load_setups(setups=COMPARED_SETUPS) -> dict:
    """Each setup's observed-data run: its posterior predictive, posterior and diagnostics.

    Returns ``{setup: {"predictive": dict, "posterior": DataFrame, "prior":
    DataFrame, "diagnostics": dict}}``.

    Raises
    ------
    FileNotFoundError
        If a setup's run, posterior predictive or diagnostics is missing.
    """
    runs = {}
    for setup in setups:
        directory = _run_directory(setup)
        runs[setup] = {
            "predictive": load_predictive(directory / "posterior_predictive"),
            "posterior": pd.read_csv(directory / "posterior_ensemble.csv", index_col=0),
            "prior": pd.read_csv(directory / "prior_ensemble.csv", index_col=0),
            "diagnostics": load_diagnostics(directory),
        }
    return runs


def plot_seasonal_cycles(runs: dict) -> plt.Figure:
    """NEE by week of year: the observations, and each setup's posterior predictive."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        observed_weekly = None
        for (setup, run), color in zip(runs.items(), CURVE_COLORS, strict=False):
            observed_weekly, quantiles = weekly_nee_quantiles(run["predictive"], name)
            ax.fill_between(
                quantiles.index,
                quantiles[0.05],
                quantiles[0.95],
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            ax.plot(
                quantiles.index,
                quantiles[0.5],
                color=color,
                linewidth=2,
                label=COMPARED_SETUPS.get(setup, setup),
            )
        ax.plot(
            observed_weekly.index,
            observed_weekly.to_numpy(),
            "o",
            color="black",
            markersize=4,
            label="observed, US-Ha1 2012-2020",
        )
        ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
        ax.set_title(NEE_TITLES[name])
        ax.set_xlabel("week of year")
    axes[0].set_ylabel("NEE (µmol CO₂ m⁻² s⁻¹), weekly mean")
    one_legend(figure, axes)
    figure.suptitle(
        "Posterior predictive by NEE error model: median and 90% of the weekly means"
    )
    return figure


def plot_moving_parameters(runs: dict) -> plt.Figure:
    """The parameters the setups move: each posterior's 90% interval against the prior's."""
    prior = next(iter(runs.values()))["prior"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    labels = ["prior", *[COMPARED_SETUPS.get(setup, setup) for setup in runs]]
    colors = [role_style("prior", "band")["color"], *CURVE_COLORS[: len(runs)]]
    for ax, name in zip(axes.flat, MOVING_PARAMETER_NAMES, strict=True):
        samples = [prior[name], *[run["posterior"][name] for run in runs.values()]]
        for position, (values, color) in enumerate(zip(samples, colors, strict=True)):
            low, middle, high = np.quantile(values, [0.05, 0.5, 0.95])
            ax.plot([low, high], [position, position], color=color, linewidth=6)
            ax.plot(middle, position, "o", color="black", markersize=5)
        ax.set_yticks(range(len(labels)), labels if ax in axes[:, 0] else [])
        ax.invert_yaxis()
        ax.set_title(PARAMETER_TITLES[name], fontsize=12)
    figure.suptitle("The parameters the error model moves: 90% intervals and medians")
    return figure


def plot_slow_fast(runs: dict) -> plt.Figure:
    """The day residual's slow and fast variance, and the model's day-to-day variability."""
    tables = {setup: run["diagnostics"]["nee_slow_fast"] for setup, run in runs.items()}
    labels = [COMPARED_SETUPS.get(setup, setup) for setup in tables]
    positions = np.arange(len(tables))
    figure, (variances, ratios) = plt.subplots(1, 2, figsize=(15, 5.5))
    day = pd.DataFrame(
        {setup: table.loc["nee_day_centered"] for setup, table in tables.items()}
    ).T
    variances.bar(
        positions - 0.2,
        day["slow_variance"],
        width=0.4,
        color="#444444",
        label="slow: 31-day running mean",
    )
    variances.bar(
        positions + 0.2,
        day["fast_variance"],
        width=0.4,
        color="#aaaaaa",
        label="fast: what remains",
    )
    variances.set_xticks(positions, labels, fontsize=11)
    variances.set_ylabel("variance of the day residual ((µmol m⁻² s⁻¹)²)")
    variances.set_title("Day-centered NEE: where the misfit sits")
    variances.legend(fontsize=11)
    for offset, (name, color) in zip(
        (-0.2, 0.2),
        (("nee_day_centered", "#D55E00"), ("nee_night_centered", "#0072B2")),
        strict=True,
    ):
        ratios.bar(
            positions + offset,
            [
                table.loc[name, "fast_standard_deviation_ratio"]
                for table in tables.values()
            ],
            width=0.4,
            color=color,
            label=NEE_TITLES[name],
        )
    ratios.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ratios.set_xticks(positions, labels, fontsize=11)
    ratios.set_ylabel("model / observed, sd of the fast part")
    ratios.set_title("Day-to-day variability of NEE, model against observed")
    ratios.legend(fontsize=11)
    return figure


# ── entry point ──


def main() -> int:
    """Draw the comparison of the setups into the figure directory's ``comparison``."""
    use_project_style()
    try:
        runs = load_setups()
    except FileNotFoundError as error:
        print(
            f"error: {error}; run each setup's EKI, posterior predictive and "
            "diagnostics first",
            file=sys.stderr,
        )
        return 1
    with plt.rc_context(SLIDE_STYLE):
        for name, draw in (
            ("comparison_seasonal_cycles", plot_seasonal_cycles),
            ("comparison_parameters", plot_moving_parameters),
            ("comparison_slow_fast", plot_slow_fast),
        ):
            save_figure(draw(runs), name, subdirectory="comparison")
    return 0


# ── helpers ──


def _run_directory(setup: str) -> Path:
    """A setup's observed-data run."""
    return config.OUTPUT_DIRECTORY / "eki" / setup / "observed"


if __name__ == "__main__":
    sys.exit(main())

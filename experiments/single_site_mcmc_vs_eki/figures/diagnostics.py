"""A run's diagnostic figures, sized for slides.

Reads what ``scripts/diagnose.py`` wrote (``model/outputs.py``'s
``load_diagnostics``) and draws, for ``--run <run>``, into
``config.FIGURE_DIRECTORY`` for the prior predictive and into its
``config.EKI_RUN_NAME`` subdirectory for an EKI run:

- :func:`plot_predictive_check` (``diagnostics_<run>_predictive_check``):
  per observation source, the members' standardized misfit
  ``(misfit - n / 2) / sqrt(n / 2)`` (median, and the range over members)
  against the band of plus or minus 2 the model expects, with the p-values;
- :func:`plot_weekly_residuals` (``diagnostics_<run>_weekly_residuals``):
  per NEE source, each year's weekly mean residual and the recurring
  seasonal part;
- :func:`plot_autocorrelation` (``diagnostics_<run>_autocorrelation``): per
  NEE source, the residuals' autocorrelation, with and without the recurring
  part, against the correlation ``R`` implies;
- :func:`plot_towers` (``diagnostics_towers``): each NEE source's windows at
  one tower against the other's, where both keep one.

``MODEL.md``, "Diagnostics", defines each quantity.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.figures.diagnostics --run observed
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sipnet_calibration.plotting.style import role_style, use_project_style

from .. import config
from ..model.outputs import load_diagnostics, run_directory
from .common import NEE_TITLES, save_figure
from .prior_predictive import SLIDE_STYLE

__all__ = [
    "plot_autocorrelation",
    "plot_predictive_check",
    "plot_towers",
    "plot_weekly_residuals",
]

#: Row labels of the observation sources, in the order they are drawn.
SOURCE_LABELS = {
    "nee_night_centered": "NEE, night-centered",
    "nee_day_centered": "NEE, day-centered",
    "modis_leaf_area_index": "MODIS LAI",
    "landtrendr_aboveground_biomass": "LandTrendr biomass",
    "soilgrids_soil_organic_carbon": "SoilGrids soil C",
    "all": "all sources",
}


def plot_predictive_check(tables: dict) -> plt.Figure:
    """The standardized misfit per source, against the band the model expects."""
    rows = []
    for vector in ("calibration", "validation"):
        table = tables.get(f"predictive_check_{vector}")
        if table is None:
            continue
        for source, label in SOURCE_LABELS.items():
            if source in table.index:
                row = table.loc[source]
                scale = np.sqrt(row["n"] / 2)
                rows.append(
                    {
                        "label": label
                        + (" (held out)" if vector == "validation" else ""),
                        "median": row["standardized"],
                        "low": (row["misfit_min"] - row["n"] / 2) / scale,
                        "high": (row["misfit_max"] - row["n"] / 2) / scale,
                        "p_value": row["p_value"],
                    }
                )
    frame = pd.DataFrame(rows)
    positions = np.arange(len(frame))
    figure, ax = plt.subplots(figsize=(11, 0.55 * len(frame) + 1.5))
    ax.axvspan(-2, 2, color="#dddddd", zorder=0, label="expected under the model (±2)")
    ax.hlines(positions, frame["low"], frame["high"], color="black", linewidth=2)
    ax.plot(frame["median"], positions, "o", color="black", label="median member")
    for position, (high, p_value) in enumerate(
        zip(frame["high"], frame["p_value"], strict=True)
    ):
        ax.text(
            max(high, 2) + 0.6, position, f"p = {p_value:.2g}", va="center", fontsize=11
        )
    ax.axvline(0.0, color="#999999", linewidth=0.6)
    ax.set_yticks(positions, frame["label"])
    ax.invert_yaxis()
    ax.set_xlim(min(-3.0, frame["low"].min() - 1), frame["high"].max() * 1.25 + 4)
    ax.set_xlabel("standardized misfit, (misfit − n/2) / √(n/2)")
    ax.set_title(
        "Posterior predictive check: each source's misfit against its χ² reference"
    )
    ax.legend(loc="lower right", fontsize=10)
    return figure


def plot_weekly_residuals(tables: dict) -> plt.Figure:
    """Each year's weekly mean NEE residual, and the recurring seasonal part."""
    weekly = tables["nee_weekly_residuals"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        table = weekly[weekly["source"] == name]
        years = [column for column in table.columns if column.isdigit()]
        for year in years:
            ax.plot(table.index, table[year], color="#bbbbbb", linewidth=0.8)
        ax.plot([], [], color="#bbbbbb", linewidth=0.8, label="one year's weekly mean")
        ax.plot(
            table.index,
            table["recurring"],
            color=role_style("posterior")["color"],
            linewidth=2.5,
            label="recurring part (all years, 5-week smoothed)",
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(NEE_TITLES[name])
        ax.set_xlabel("week of year")
    axes[0].set_ylabel("residual, observed − model (µmol CO₂ m⁻² s⁻¹)")
    axes[1].legend(loc="lower left", fontsize=10)
    figure.suptitle(
        "NEE residuals of the ensemble's median prediction, by week of year"
    )
    return figure


def plot_autocorrelation(tables: dict) -> plt.Figure:
    """The residuals' autocorrelation against the correlation ``R`` implies."""
    autocorrelation = tables["nee_autocorrelation"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        table = autocorrelation.loc[name]
        lags = table.index.to_numpy()
        ax.plot(lags, table["observed"], "o-", color="black", label="residuals")
        ax.plot(
            lags,
            table["remainder"],
            "s--",
            color="#777777",
            label="residuals less the recurring part",
        )
        ax.plot(
            lags,
            table["modeled"],
            "^:",
            color=role_style("posterior")["color"],
            label="implied by R",
        )
        ax.set_xscale("log")
        ax.set_xticks(lags, [str(lag) for lag in lags])
        ax.axhline(0.0, color="#999999", linewidth=0.6)
        ax.set_title(NEE_TITLES[name])
        ax.set_xlabel("lag (days)")
    axes[0].set_ylabel("autocorrelation")
    axes[1].legend(fontsize=10)
    return figure


def plot_towers(tables: dict) -> plt.Figure:
    """Each NEE window at US-Ha1 against US-xHA, where both keep one."""
    towers = tables["nee_towers"]
    summary = tables["nee_towers_summary"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        both = towers[towers["source"] == name]
        ax.scatter(both["second"], both["first"], s=12, color="black", alpha=0.7)
        limits = [
            min(both["first"].min(), both["second"].min()),
            max(both["first"].max(), both["second"].max()),
        ]
        ax.plot(limits, limits, color="#999999", linewidth=0.8, label="1:1")
        row = summary.loc[name]
        ax.set_title(
            f"{NEE_TITLES[name]}\n{int(row['n'])} windows, "
            f"{int(row['first_year'])}-{int(row['last_year'])}; "
            f"representativeness sd ≥ {row['representativeness_sd']:.2f}"
        )
        ax.set_xlabel("US-xHA (µmol CO₂ m⁻² s⁻¹)")
        ax.set_ylabel("US-Ha1 (µmol CO₂ m⁻² s⁻¹)")
        ax.set_aspect("equal")
    return figure


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Draw one run's diagnostic figures into ``config.FIGURE_DIRECTORY``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run", choices=("prior", "synthetic", "observed"), required=True
    )
    run_name = parser.parse_args(argv).run
    use_project_style()
    try:
        tables = load_diagnostics(run_directory(run_name))
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    eki_run = run_name != "prior"
    with plt.rc_context(SLIDE_STYLE):
        prefix = f"diagnostics_{run_name}"
        for name, draw in (
            ("predictive_check", plot_predictive_check),
            ("weekly_residuals", plot_weekly_residuals),
            ("autocorrelation", plot_autocorrelation),
        ):
            save_figure(draw(tables), f"{prefix}_{name}", eki_run=eki_run)
        save_figure(plot_towers(tables), "diagnostics_towers")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The experiment's figures.

Reads what the scripts wrote and draws it through
:mod:`sipnet_calibration.plotting`; nothing here runs a model. Run as a
script, it draws every figure of the steps run so far into
``config.FIGURE_DIRECTORY``.

Step 3, the forward check (``forward_check.py``):

- :func:`plot_nee_windows`: the two NEE sources, observed against the one
  run and the ensemble, over the whole record and over one year;
- :func:`plot_pool_observations`: leaf area index, biomass and soil carbon,
  observed against the same;
- :func:`plot_daily_trajectories`: the fluxes and pools behind them, day by
  day.

In every figure the ensemble is the prior's draws (role ``prior``: a
median and 50% and 90% bands), the one run is at the prior's center (a solid
line), and observations are black points with error bars of the noise model's
total standard deviation; the held-out tower's are hollow.

Usage
-----
::

    uv run python experiments/single_site_mcmc_vs_eki/plots.py [--calibration stand_in]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import config
from sipnet_calibration.conventions import SAMPLE, SITE
from sipnet_calibration.plotting import plot_time_series
from sipnet_calibration.plotting.style import role_style, use_project_style

__all__ = [
    "load_forward_check",
    "plot_daily_trajectories",
    "plot_nee_windows",
    "plot_pool_observations",
]

#: The legend entries of the three things every forward-check panel shows.
ENSEMBLE_LABEL = "ensemble (prior draws)"
SINGLE_RUN_LABEL = "one run (prior center)"
OBSERVED_LABEL = "observed (US-Ha1, constraints)"
HELD_OUT_LABEL = "held out (US-xHA)"

#: Panel titles of the NEE sources.
NEE_TITLES = {
    "nee_night_centered": "NEE, 00-12 UTC (night-centered)",
    "nee_day_centered": "NEE, 12-24 UTC (day-centered)",
}


def load_forward_check(directory: Path | None = None) -> dict:
    """What ``forward_check.py`` wrote, as nested dicts of xarray objects.

    Returns ``{"daily": {"single_run", "ensemble"}, "predicted": {run:
    {vector: {source: field}}}, "observed": {vector: {source: dataset}}}``,
    every field at the site.
    """
    directory = directory or config.FORWARD_CHECK_DIRECTORY
    check = {
        "daily": {
            "single_run": _at_site(xr.load_dataset(directory / "single_run_daily.nc")),
            "ensemble": _at_site(xr.load_dataset(directory / "ensemble_daily.nc")),
        },
        "predicted": {},
        "observed": {},
    }
    for run in ("single_run", "ensemble"):
        check["predicted"][run] = {
            vector: {
                path.stem: _at_site(xr.load_dataarray(path))
                for path in sorted(
                    (directory / "predictions" / run / vector).glob("*.nc")
                )
            }
            for vector in ("calibration", "validation")
        }
    check["observed"] = {
        vector: {
            path.stem: _at_site(xr.load_dataset(path))
            for path in sorted((directory / "observed" / vector).glob("*.nc"))
        }
        for vector in ("calibration", "validation")
    }
    return check


def plot_nee_windows(check: dict, *, zoom_year: int = 2015) -> plt.Figure:
    """Both NEE sources: the whole record (left) and one year with error bars (right)."""
    figure, axes = plt.subplots(
        len(config.NEE_WINDOWS), 2, figsize=(12, 6), width_ratios=(2.2, 1), sharey="row"
    )
    for row, name in enumerate(config.NEE_WINDOWS):
        whole, year = axes[row]
        for vector in ("calibration", "validation"):
            _draw_predictions(whole, check, vector, name)
            _draw_observed(whole, check, vector, name, error_bars=False)
        _draw_predictions(year, check, "calibration", name)
        _draw_observed(year, check, "calibration", name, error_bars=True)
        year.set_xlim(
            np.datetime64(f"{zoom_year}-01-01"), np.datetime64(f"{zoom_year + 1}-01-01")
        )
        whole.set_title(NEE_TITLES[name])
        year.set_title(f"{NEE_TITLES[name]}, {zoom_year}")
        for ax in (whole, year):
            ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
            ax.set_xlabel("")
        whole.set_ylabel("NEE (umol m-2 s-1 CO2)")
        year.set_ylabel("")
    _one_legend(figure, axes)
    return figure


def plot_pool_observations(check: dict) -> plt.Figure:
    """Leaf area index, aboveground biomass and soil carbon, observed against the runs."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), width_ratios=(2, 1.2, 0.7))
    for ax, name in zip(
        axes[:2],
        ("modis_leaf_area_index", "landtrendr_aboveground_biomass"),
        strict=True,
    ):
        _draw_predictions(ax, check, "calibration", name)
        _draw_observed(ax, check, "calibration", name, error_bars=True)
        ax.set_xlabel("")
    axes[0].set_title("MODIS leaf area index (June-August composites)")
    axes[1].set_title("LandTrendr aboveground biomass (dry)")
    _draw_static(axes[2], check, "soilgrids_soil_organic_carbon")
    axes[2].set_title("SoilGrids soil carbon")
    _one_legend(figure, axes)
    return figure


def plot_daily_trajectories(check: dict) -> plt.Figure:
    """The fluxes and pools behind the observations, as daily model output."""
    names = config.FORWARD_CHECK_OUTPUT_VARIABLE_NAMES
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, name in zip(axes.flat, names, strict=True):
        plot_time_series(
            check["daily"]["ensemble"][name], ax=ax, role="prior", label=ENSEMBLE_LABEL
        )
        plot_time_series(
            check["daily"]["single_run"][name],
            ax=ax,
            role="posterior",
            label=SINGLE_RUN_LABEL,
        )
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("")
    _one_legend(figure, axes)
    return figure


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Draw every figure of one forward check into ``config.FIGURE_DIRECTORY``."""
    parser = argparse.ArgumentParser(description="Draw the forward check's figures.")
    parser.add_argument("--calibration", choices=("prior", "stand_in"), default="prior")
    name_of_check = parser.parse_args(argv).calibration
    use_project_style()
    try:
        check = load_forward_check(config.FORWARD_CHECK_DIRECTORY / name_of_check)
    except FileNotFoundError as error:
        print(f"error: {error}; run forward_check.py first", file=sys.stderr)
        return 1
    config.FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, draw in (
        ("forward_check_nee", plot_nee_windows),
        ("forward_check_pools", plot_pool_observations),
        ("forward_check_trajectories", plot_daily_trajectories),
    ):
        path = config.FIGURE_DIRECTORY / f"{name}_{name_of_check}.png"
        draw(check).savefig(path)
        print(f"wrote {path}")
    return 0


# ── helpers ──


def _at_site(data):
    """*data* at the configured site, the ``site`` dim dropped to a scalar."""
    return data.sel({SITE: config.SITE}) if SITE in data.dims else data


def _draw_predictions(ax, check: dict, vector: str, name: str) -> None:
    """The ensemble's fan and the one run's line at a source's observations."""
    ensemble = check["predicted"]["ensemble"][vector].get(name)
    single = check["predicted"]["single_run"][vector].get(name)
    if ensemble is None:
        return
    plot_time_series(ensemble, ax=ax, role="prior", label=ENSEMBLE_LABEL)
    plot_time_series(
        single, ax=ax, role="posterior", label=SINGLE_RUN_LABEL, linewidth=0.8
    )


def _draw_observed(
    ax, check: dict, vector: str, name: str, *, error_bars: bool
) -> None:
    """A source's observed values, with the noise model's standard deviation."""
    observed = check["observed"][vector].get(name)
    if observed is None:
        return
    held_out = vector == "validation"
    plot_time_series(
        observed["value"],
        ax=ax,
        role="observation",
        show="points",
        standard_deviation=observed["noise_standard_deviation"] if error_bars else None,
        label=HELD_OUT_LABEL if held_out else OBSERVED_LABEL,
        markersize=2.5,
        **({"markerfacecolor": "none"} if held_out else {}),
    )


def _draw_static(ax, check: dict, name: str) -> None:
    """A static source: the ensemble's values as a strip, the run and the observation."""
    ensemble = check["predicted"]["ensemble"]["calibration"][name].squeeze()
    single = float(check["predicted"]["single_run"]["calibration"][name].squeeze())
    observed = check["observed"]["calibration"][name]
    rng = np.random.default_rng(0)
    values = ensemble.transpose(SAMPLE, ...).to_numpy().ravel()
    ax.scatter(
        rng.uniform(-0.15, 0.15, values.size),
        values,
        s=10,
        **{
            key: value
            for key, value in role_style("prior", "line").items()
            if key == "color"
        },
        alpha=0.6,
        label=ENSEMBLE_LABEL,
    )
    ax.scatter(
        [0.0],
        [single],
        marker="_",
        s=400,
        color=role_style("posterior")["color"],
        label=SINGLE_RUN_LABEL,
    )
    ax.errorbar(
        [0.5],
        [float(observed["value"])],
        yerr=[float(observed["noise_standard_deviation"])],
        fmt="o",
        color="black",
        markersize=4,
        capsize=3,
        label=OBSERVED_LABEL,
    )
    ax.set_xlim(-0.5, 1.0)
    ax.set_xticks([0.0, 0.5], ["model", "observed"])
    ax.set_ylabel(f"soil carbon ({observed['value'].attrs.get('units')} C)")


def _one_legend(figure, axes) -> None:
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


if __name__ == "__main__":
    sys.exit(main())

"""The prior predictive's figures.

Reads what ``run/prior_predictive.py`` wrote and draws it through
:mod:`sipnet_calibration.plotting`; nothing here runs a model. Run as a
script, it draws every figure into ``config.FIGURE_DIRECTORY``.

The time series and pools, against the observations:

- :func:`plot_nee_windows` (``prior_predictive_nee``): the two NEE sources,
  observed against the one run and the ensemble, over the whole record and
  over one year;
- :func:`plot_pool_observations` (``prior_predictive_pools``): leaf area
  index, biomass and soil carbon, observed against the same;
- :func:`plot_daily_trajectories` (``prior_predictive_trajectories``): the
  fluxes and pools behind them, day by day.

In these the ensemble is the prior's draws (role ``prior``: a median and 50%
and 90% bands), the one run is at the prior's center (a solid line), and
observations are black points with error bars of the noise model's total
standard deviation; the held-out tower's are hollow.

The summaries, sized for slides (:data:`SLIDE_STYLE`):

- :func:`plot_prior_marginals` (``prior_marginals``): each calibrated
  parameter's prior, as the draws the ensemble ran;
- :func:`plot_nee_seasonal_cycle` (``prior_predictive_nee_seasonal``): NEE's
  seasonal cycle, observed against the prior predictive, both averaged by
  week of year over the windows the likelihood reads;
- :func:`plot_nee_annual` (``prior_predictive_nee_annual``): annual NEE, the
  prior predictive against both towers' annual totals;
- :func:`plot_coverage` (``prior_predictive_coverage``): per observation
  source, the fraction of observations inside the prior predictive's 50% and
  90% intervals, noise included.

:func:`draw_prior_predictive_figures` draws them all; ``run/prior_predictive.py``
calls it at the end of the run, and ``run/draw_figures.py --run prior``
redraws them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frozendict import frozendict

from sipnet_calibration.conventions import SAMPLE, TIME, TIMESTEP_LENGTH, TIMESTEP_START
from sipnet_calibration.observation import aggregate_time
from sipnet_calibration.plotting import plot_time_series
from sipnet_calibration.plotting.style import role_style, use_project_style

from .. import config
from ..model import inputs
from ..model.outputs import load_predictive
from .common import NEE_TITLES, SOURCE_LABELS, one_legend, save_figure

__all__ = [
    "PARAMETER_TITLES",
    "PREDICTIVES",
    "SLIDE_STYLE",
    "draw_prior_predictive_figures",
    "plot_coverage",
    "plot_daily_trajectories",
    "plot_nee_annual",
    "plot_nee_seasonal_cycle",
    "plot_nee_windows",
    "plot_pool_observations",
    "plot_prior_marginals",
    "weekly_nee_quantiles",
]

#: How each predictive is drawn, by its kind: its ensemble's role, its
#: ensemble's legend entry, and its name in titles and legends.
PREDICTIVES = frozendict(
    {
        "prior": frozendict(
            {
                "role": "prior",
                "ensemble_label": "ensemble (prior draws)",
                "name": "prior predictive",
            }
        ),
        "posterior": frozendict(
            {
                "role": "posterior",
                "ensemble_label": "ensemble (EKI posterior)",
                "name": "posterior predictive",
            }
        ),
    }
)

#: The legend entries of the other things a time series panel shows: the one
#: run, which only the prior predictive has, and the observations.
SINGLE_RUN_LABEL = "one run (prior center)"
OBSERVED_LABEL = "observed (US-Ha1, constraints)"
HELD_OUT_LABEL = "held out (US-xHA)"

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

#: g C m-2 in one umol CO2 m-2 s-1 sustained for one second.
_GRAMS_CARBON_PER_UMOL_SECOND = 12.011e-6


def plot_nee_windows(
    outputs: dict, *, kind: str = "prior", zoom_year: int = 2015
) -> plt.Figure:
    """Both NEE sources: the whole record (left) and one year with error bars (right)."""
    figure, axes = plt.subplots(
        len(config.NEE_WINDOWS), 2, figsize=(12, 6), width_ratios=(2.2, 1), sharey="row"
    )
    for row, name in enumerate(config.NEE_WINDOWS):
        whole, year = axes[row]
        for vector in ("calibration", "validation"):
            _draw_predictions(whole, outputs, vector, name, kind)
            _draw_observed(whole, outputs, vector, name, error_bars=False)
        _draw_predictions(year, outputs, "calibration", name, kind)
        _draw_observed(year, outputs, "calibration", name, error_bars=True)
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
    one_legend(figure, axes)
    return figure


def plot_pool_observations(outputs: dict, *, kind: str = "prior") -> plt.Figure:
    """Leaf area index, aboveground biomass and soil carbon, observed against the runs."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), width_ratios=(2, 1.2, 0.7))
    for ax, name in zip(
        axes[:2],
        ("modis_leaf_area_index", "landtrendr_aboveground_biomass"),
        strict=True,
    ):
        _draw_predictions(ax, outputs, "calibration", name, kind)
        _draw_observed(ax, outputs, "calibration", name, error_bars=True)
        ax.set_xlabel("")
    axes[0].set_title("MODIS leaf area index (June-August composites)")
    axes[1].set_title("LandTrendr aboveground biomass (dry)")
    _draw_static(axes[2], outputs, "soilgrids_soil_organic_carbon", kind)
    axes[2].set_title("SoilGrids soil carbon")
    one_legend(figure, axes)
    return figure


def plot_daily_trajectories(outputs: dict, *, kind: str = "prior") -> plt.Figure:
    """The fluxes and pools behind the observations, as daily model output."""
    names = config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, name in zip(axes.flat, names, strict=True):
        plot_time_series(
            outputs["daily"]["ensemble"][name],
            ax=ax,
            role=PREDICTIVES[kind]["role"],
            label=PREDICTIVES[kind]["ensemble_label"],
        )
        if "single_run" in outputs["daily"]:
            plot_time_series(
                outputs["daily"]["single_run"][name],
                ax=ax,
                role="posterior",
                label=SINGLE_RUN_LABEL,
            )
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("")
    one_legend(figure, axes)
    return figure


def plot_prior_marginals(parameters: pd.DataFrame) -> plt.Figure:
    """A histogram of each calibrated parameter's prior draws."""
    names = [name for name in PARAMETER_TITLES if name in parameters]
    draws = parameters.drop(index="single_run")
    figure, axes = plt.subplots(4, 4, figsize=(14, 10))
    color = role_style("prior", "band")["color"]
    for ax, name in zip(axes.flat, names, strict=False):
        values = draws[name].to_numpy()
        log_scale = values.min() > 0 and values.max() / values.min() > 20
        bins = np.geomspace(values.min(), values.max(), 25) if log_scale else 25
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.6)
        if log_scale:
            ax.set_xscale("log")
        ax.axvline(np.median(values), color="black", linewidth=1.0)
        ax.set_title(PARAMETER_TITLES[name])
        ax.set_yticks([])
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    figure.suptitle(
        f"The prior: {len(draws)} draws of 13 calibrated parameters "
        "(the black line is the median)"
    )
    return figure


def plot_nee_seasonal_cycle(outputs: dict, *, kind: str = "prior") -> plt.Figure:
    """NEE by week of year, observed against a predictive, at the observed windows."""
    predictive = PREDICTIVES[kind]["name"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        observed_weekly, quantiles = weekly_nee_quantiles(outputs, name)
        color = role_style(PREDICTIVES[kind]["role"], "band")["color"]
        ax.fill_between(
            quantiles.index,
            quantiles[0.05],
            quantiles[0.95],
            color=color,
            alpha=0.25,
            linewidth=0,
            label=f"{predictive}, 90%",
        )
        ax.fill_between(
            quantiles.index,
            quantiles[0.25],
            quantiles[0.75],
            color=color,
            alpha=0.5,
            linewidth=0,
            label=f"{predictive}, 50%",
        )
        ax.plot(
            quantiles.index,
            quantiles[0.5],
            color="#555555",
            linewidth=1.5,
            label=f"{predictive}, median",
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
        f"NEE's seasonal cycle, {predictive} against observed: weekly means over "
        "the observed windows"
    )
    return figure


def weekly_nee_quantiles(outputs: dict, name: str) -> tuple[pd.Series, pd.DataFrame]:
    """One NEE source by week of year: the observed means, and the predictive's quantiles.

    The observations are averaged by the week of each window's end, and each
    member's predictions over the same windows; the quantiles, columns 0.05,
    0.25, 0.5, 0.75 and 0.95, are over the members' weekly means.
    """
    observed = outputs["observed"]["calibration"][name]["value"]
    predicted = outputs["predicted"]["ensemble"]["calibration"][name]
    weeks = _week_of_year(observed[TIME])
    observed_weekly = pd.Series(observed.to_numpy()).groupby(weeks).mean()
    predicted_weekly = (
        pd.DataFrame(predicted.transpose(SAMPLE, TIME).to_numpy().T)
        .groupby(weeks)
        .mean()
    )
    quantiles = predicted_weekly.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    return observed_weekly, quantiles


def plot_nee_annual(outputs: dict, *, kind: str = "prior") -> plt.Figure:
    """Annual NEE per year: a predictive's spread against both towers' totals."""
    predictive = PREDICTIVES[kind]["name"]
    daily = outputs["daily"]["ensemble"]["net_ecosystem_exchange"]
    annual = aggregate_time(daily, "YS")
    # A cell is labeled with its end; its year is its first step's. The
    # record's first cell is the three hours before 2012 and is dropped; its
    # last is three hours short of 2024's end and is kept.
    years = pd.DatetimeIndex(annual[TIMESTEP_START].values).year.to_numpy()
    days = annual[TIMESTEP_LENGTH].values / np.timedelta64(1, "D")
    values = annual.transpose(SAMPLE, TIME).to_numpy()
    keep = days >= 365
    figure, ax = plt.subplots(figsize=(12, 5))
    color = role_style(PREDICTIVES[kind]["role"], "band")["color"]
    parts = ax.violinplot(
        [values[:, i] for i in np.flatnonzero(keep)],
        positions=years[keep],
        widths=0.7,
        showextrema=False,
        showmedians=True,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.5)
    parts["cmedians"].set_color("#555555")
    for series, label, marker in (
        (config.CALIBRATION_NEE_SERIES, "observed, US-Ha1 (calibration)", "o"),
        (config.VALIDATION_NEE_SERIES, "observed, US-xHA (held out)", "s"),
    ):
        totals = _observed_annual_nee(series)
        ax.plot(
            totals.index,
            totals.to_numpy(),
            marker,
            color="black",
            markersize=7,
            markerfacecolor="black" if marker == "o" else "none",
            label=label,
        )
    ax.plot([], [], color=color, linewidth=8, alpha=0.5, label=predictive)
    ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_xticks(range(2012, 2025))
    ax.set_ylabel("annual NEE (g C m⁻² yr⁻¹)\nnegative = uptake")
    ax.set_title(f"Annual NEE: {predictive} against the towers' gap-filled totals")
    ax.legend(loc="upper left", ncol=3)
    return figure


def plot_coverage(
    outputs: dict, *, kind: str = "prior", vector: str = "calibration"
) -> plt.Figure:
    """Per source, the fraction of observations inside the 50% and 90% predictive intervals."""
    rng = np.random.default_rng(0)
    rows = []
    source_labels = {
        name: label
        for name, label in SOURCE_LABELS.items()
        if name in outputs["observed"][vector]
    }
    for name, label in source_labels.items():
        observed = outputs["observed"][vector][name]
        predicted = outputs["predicted"]["ensemble"][vector][name]
        y = np.atleast_1d(observed["value"].to_numpy())
        sd = np.atleast_1d(observed["noise_standard_deviation"].to_numpy())
        samples = (
            predicted.transpose(SAMPLE, ...)
            .to_numpy()
            .reshape(predicted.sizes[SAMPLE], -1)
        )
        noisy = samples + rng.standard_normal(samples.shape) * sd
        for level in (0.5, 0.9):
            low, high = np.quantile(noisy, [(1 - level) / 2, (1 + level) / 2], axis=0)
            rows.append(
                {
                    "source": label,
                    "level": level,
                    "inside": np.mean((y >= low) & (y <= high)),
                    "n": y.size,
                }
            )
    table = (
        pd.DataFrame(rows)
        .pivot(index="source", columns="level", values="inside")
        .loc[list(source_labels.values())]
    )
    counts = pd.DataFrame(rows).groupby("source")["n"].first()
    figure, ax = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(table))
    color = role_style(PREDICTIVES[kind]["role"], "band")["color"]
    ax.barh(
        positions + 0.2,
        table[0.9],
        height=0.38,
        color=color,
        alpha=0.9,
        label="inside the 90% interval",
    )
    ax.barh(
        positions - 0.2,
        table[0.5],
        height=0.38,
        color=color,
        alpha=0.45,
        label="inside the 50% interval",
    )
    for level, offset in ((0.9, 0.2), (0.5, -0.2)):
        ax.plot(
            [level, level],
            [positions[0] - 0.5, positions[-1] + 0.5],
            color="black",
            linewidth=0.8,
            linestyle="--",
        )
        for position, value in zip(positions, table[level], strict=True):
            ax.text(
                value + 0.01,
                position + offset,
                f"{value:.2f}",
                va="center",
                fontsize=10,
            )
    ax.set_yticks(
        positions, [f"{source} (n={counts[source]})" for source in table.index]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("fraction of observations (dashed: the nominal level)")
    held_out = ", held-out tower" if vector == "validation" else ""
    ax.set_title(
        f"{PREDICTIVES[kind]['name'].capitalize()} coverage{held_out}, "
        "noise model included"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return figure


# ── drawing the run ──


def draw_prior_predictive_figures() -> None:
    """Draw every figure of the prior predictive into ``config.FIGURE_DIRECTORY``.

    Raises
    ------
    FileNotFoundError
        If the prior predictive has not run.
    """
    use_project_style()
    directory = config.PRIOR_PREDICTIVE_DIRECTORY
    outputs = load_predictive(directory)
    parameters = pd.read_csv(directory / "parameters.csv", index_col=0)
    for name, draw in (
        ("prior_predictive_nee", plot_nee_windows),
        ("prior_predictive_pools", plot_pool_observations),
        ("prior_predictive_trajectories", plot_daily_trajectories),
    ):
        save_figure(draw(outputs), name)
    with plt.rc_context(SLIDE_STYLE):
        for name, figure in (
            ("prior_marginals", plot_prior_marginals(parameters)),
            ("prior_predictive_nee_seasonal", plot_nee_seasonal_cycle(outputs)),
            ("prior_predictive_nee_annual", plot_nee_annual(outputs)),
            ("prior_predictive_coverage", plot_coverage(outputs)),
        ):
            save_figure(figure, name)


# ── helpers ──


def _draw_predictions(ax, outputs: dict, vector: str, name: str, kind: str) -> None:
    """The ensemble's fan and the one run's line, if any, at a source's observations."""
    ensemble = outputs["predicted"]["ensemble"][vector].get(name)
    single = outputs["predicted"].get("single_run", {}).get(vector, {}).get(name)
    if ensemble is None:
        return
    plot_time_series(
        ensemble,
        ax=ax,
        role=PREDICTIVES[kind]["role"],
        label=PREDICTIVES[kind]["ensemble_label"],
    )
    if single is not None:
        plot_time_series(
            single, ax=ax, role="posterior", label=SINGLE_RUN_LABEL, linewidth=0.8
        )


def _draw_observed(
    ax, outputs: dict, vector: str, name: str, *, error_bars: bool
) -> None:
    """A source's observed values, with the noise model's standard deviation."""
    observed = outputs["observed"][vector].get(name)
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


def _draw_static(ax, outputs: dict, name: str, kind: str) -> None:
    """A static source: the ensemble's values as a strip, the run and the observation."""
    ensemble = outputs["predicted"]["ensemble"]["calibration"][name].squeeze()
    observed = outputs["observed"]["calibration"][name]
    rng = np.random.default_rng(0)
    values = ensemble.transpose(SAMPLE, ...).to_numpy().ravel()
    ax.scatter(
        rng.uniform(-0.15, 0.15, values.size),
        values,
        s=10,
        **{
            key: value
            for key, value in role_style(PREDICTIVES[kind]["role"], "line").items()
            if key == "color"
        },
        alpha=0.6,
        label=PREDICTIVES[kind]["ensemble_label"],
    )
    if "single_run" in outputs["predicted"]:
        single = outputs["predicted"]["single_run"]["calibration"][name]
        ax.scatter(
            [0.0],
            [float(single.squeeze())],
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


def _week_of_year(times) -> np.ndarray:
    """The ISO week of each window's end, at most 52."""
    weeks = pd.DatetimeIndex(np.asarray(times)).isocalendar().week.to_numpy()
    return np.minimum(weeks, 52)


def _observed_annual_nee(series_name: str) -> pd.Series:
    """A tower's annual NEE, g C m-2 yr-1, from its gap-filled series, whole years only."""
    rates = inputs.observed_nee_fields()[series_name].squeeze().to_series()
    step_seconds = (rates.index[1] - rates.index[0]).total_seconds()
    # A value is the mean rate over the step ending at its label.
    years = (rates.index - pd.Timedelta(seconds=step_seconds)).year
    grouped = rates.groupby(years)
    totals = grouped.sum() * step_seconds * _GRAMS_CARBON_PER_UMOL_SECOND
    complete = grouped.count() == grouped.size()
    complete &= grouped.size() >= 0.99 * 365 * 86400 / step_seconds
    return totals[complete]

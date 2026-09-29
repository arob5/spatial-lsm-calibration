"""Step 5: the prior predictive figures, sized for slides.

Reads what ``forward_check.py --calibration prior`` wrote (run it with
``--ensemble-size 200`` for these figures) and draws, into
``config.FIGURE_DIRECTORY``:

- ``prior_marginals``: each calibrated parameter's prior, as the draws the
  ensemble ran;
- ``prior_predictive_nee_seasonal``: NEE's seasonal cycle, observed against the
  prior predictive, both averaged by week of year over the same windows the
  likelihood reads;
- ``prior_predictive_nee_annual``: annual NEE, the prior predictive against both
  towers' annual totals;
- ``prior_predictive_coverage``: per observation source, the fraction of
  observations inside the prior predictive's 50% and 90% intervals, noise
  included.

``plots.py --calibration prior`` draws the time series and pool figures from
the same run.

Usage
-----
::

    uv run python experiments/single_site_mcmc_vs_eki/forward_check.py --ensemble-size 200
    uv run python experiments/single_site_mcmc_vs_eki/talk_figures.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import inputs
import plots
from sipnet_calibration.conventions import SAMPLE, TIME, TIMESTEP_LENGTH, TIMESTEP_START
from sipnet_calibration.observation import aggregate_time
from sipnet_calibration.plotting.style import role_style, use_project_style

__all__ = [
    "plot_coverage",
    "plot_nee_annual",
    "plot_nee_seasonal_cycle",
    "plot_prior_marginals",
]

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

#: The observation sources' labels on the coverage figure.
_SOURCE_LABELS = {
    "nee_night_centered": "NEE, night-centered",
    "nee_day_centered": "NEE, day-centered",
    "modis_leaf_area_index": "MODIS LAI",
    "landtrendr_aboveground_biomass": "LandTrendr biomass",
    "soilgrids_soil_organic_carbon": "SoilGrids soil C",
}


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


def plot_nee_seasonal_cycle(check: dict) -> plt.Figure:
    """NEE by week of year, observed against the prior predictive, at the observed windows."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, name in zip(axes, config.NEE_WINDOWS, strict=True):
        observed = check["observed"]["calibration"][name]["value"]
        predicted = check["predicted"]["ensemble"]["calibration"][name]
        weeks = _week_of_year(observed[TIME])
        observed_weekly = pd.Series(observed.to_numpy()).groupby(weeks).mean()
        predicted_weekly = (
            pd.DataFrame(predicted.transpose(SAMPLE, TIME).to_numpy().T)
            .groupby(weeks)
            .mean()
        )
        quantiles = predicted_weekly.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
        color = role_style("prior", "band")["color"]
        ax.fill_between(
            quantiles.index,
            quantiles[0.05],
            quantiles[0.95],
            color=color,
            alpha=0.25,
            linewidth=0,
            label="prior predictive, 90%",
        )
        ax.fill_between(
            quantiles.index,
            quantiles[0.25],
            quantiles[0.75],
            color=color,
            alpha=0.5,
            linewidth=0,
            label="prior predictive, 50%",
        )
        ax.plot(
            quantiles.index,
            quantiles[0.5],
            color="#555555",
            linewidth=1.5,
            label="prior predictive, median",
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
        ax.set_title(plots.NEE_TITLES[name])
        ax.set_xlabel("week of year")
    axes[0].set_ylabel("NEE (µmol CO₂ m⁻² s⁻¹), weekly mean")
    plots._one_legend(figure, axes)
    figure.suptitle(
        "NEE's seasonal cycle, prior predictive against observed: weekly means over "
        "the observed windows"
    )
    return figure


def plot_nee_annual(check: dict) -> plt.Figure:
    """Annual NEE per year: the prior predictive's spread against both towers' totals."""
    daily = check["daily"]["ensemble"]["net_ecosystem_exchange"]
    annual = aggregate_time(daily, "YS")
    # A cell is labeled with its end; its year is its first step's. The
    # record's first cell is the three hours before 2012 and is dropped; its
    # last is three hours short of 2024's end and is kept.
    years = pd.DatetimeIndex(annual[TIMESTEP_START].values).year.to_numpy()
    days = annual[TIMESTEP_LENGTH].values / np.timedelta64(1, "D")
    values = annual.transpose(SAMPLE, TIME).to_numpy()
    keep = days >= 365
    figure, ax = plt.subplots(figsize=(12, 5))
    color = role_style("prior", "band")["color"]
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
    ax.plot([], [], color=color, linewidth=8, alpha=0.5, label="prior predictive")
    ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_xticks(range(2012, 2025))
    ax.set_ylabel("annual NEE (g C m⁻² yr⁻¹)\nnegative = uptake")
    ax.set_title("Annual NEE: prior predictive against the towers' gap-filled totals")
    ax.legend(loc="upper left", ncol=3)
    return figure


def plot_coverage(check: dict) -> plt.Figure:
    """Per source, the fraction of observations inside the 50% and 90% predictive intervals."""
    rng = np.random.default_rng(0)
    rows = []
    for name, label in _SOURCE_LABELS.items():
        observed = check["observed"]["calibration"][name]
        predicted = check["predicted"]["ensemble"]["calibration"][name]
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
        .loc[list(_SOURCE_LABELS.values())]
    )
    counts = pd.DataFrame(rows).groupby("source")["n"].first()
    figure, ax = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(table))
    color = role_style("prior", "band")["color"]
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
    ax.set_title("Prior predictive coverage, noise model included")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return figure


# ── entry point ──


def main() -> int:
    """Draw the step-5 figures from the prior's forward check."""
    use_project_style()
    plt.rcParams.update(SLIDE_STYLE)
    directory = config.FORWARD_CHECK_DIRECTORY / "prior"
    try:
        check = plots.load_forward_check(directory)
        parameters = pd.read_csv(directory / "parameters.csv", index_col=0)
    except FileNotFoundError as error:
        print(f"error: {error}; run forward_check.py first", file=sys.stderr)
        return 1
    config.FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, figure in (
        ("prior_marginals", plot_prior_marginals(parameters)),
        ("prior_predictive_nee_seasonal", plot_nee_seasonal_cycle(check)),
        ("prior_predictive_nee_annual", plot_nee_annual(check)),
        ("prior_predictive_coverage", plot_coverage(check)),
    ):
        path = config.FIGURE_DIRECTORY / f"{name}.png"
        figure.savefig(path)
        print(f"wrote {path}")
    return 0


# ── helpers ──


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


if __name__ == "__main__":
    sys.exit(main())

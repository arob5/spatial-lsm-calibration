"""Figures for the statistics talk.

Each public function draws one figure and returns it; ``slides.qmd`` calls
them. Nothing here saves a file: the deck renders what is returned. The
diagrams are SVG files under ``figures/``, drawn by hand so they can be
edited in any SVG editor; the functions here only choose which layers of a
diagram a slide shows.
"""

import functools
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import HTML
from matplotlib.colors import ListedColormap
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.figure import Figure
import jax
import scipy.stats
from pysipnet.units import convert_dataarray_units

from sipnet_calibration.conventions import SAMPLE, SITE
from sipnet_calibration.fields import to_model_output
from sipnet_calibration.net_ecosystem_exchange.towers import read_tower_table
from sipnet_calibration.observation import aggregate_time
from sipnet_calibration.plotting import plot_map
from sipnet_calibration.plotting.series import plot_time_series
from sipnet_calibration.plotting.style import CURVE_COLORS
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import load_sites, site_coordinates

from experiments.single_site_mcmc_vs_eki import config as experiment_config
from experiments.single_site_mcmc_vs_eki.figures.common import PARAMETER_TITLES
from experiments.single_site_mcmc_vs_eki.model import inputs, noise, prior, sipnet

__all__ = [
    "box_model",
    "eddy_covariance",
    "forward_map_schematic",
    "model_output",
    "noise_scale_prior",
    "prior_marginals",
    "sites_and_towers",
    "state_space_graph",
]

#: Where the hand-drawn diagrams are.
FIGURE_DIRECTORY = Path(__file__).resolve().parent / "figures"

#: The layer of ``box_model.svg`` the "Observations" slide adds.
OBSERVATION_LAYER_ID = "observations"

#: The model output variables shown, in the order of the panels, each with the
#: units it is drawn in.
MODEL_OUTPUT_UNITS = {
    "net_ecosystem_exchange": "g m-2",
    "leaf_area_index": "m2 m-2",
    "wood_carbon": "kg m-2",
    "soil_carbon": "kg m-2",
}

#: The span of the run the forward-map schematic shows.
SCHEMATIC_PERIOD = slice("2015-01-01", "2017-12-31")

#: The forward-map schematic's rows: the model state or flux drawn, its label,
#: and the label of the observed quantity reduced from it.
SCHEMATIC_ROWS = (
    ("net_ecosystem_exchange", "NEE", "NEE"),
    ("leaf_carbon", "leaf carbon", "leaf area index"),
    ("wood_carbon", "wood carbon", "biomass"),
    ("soil_carbon", "soil carbon", "soil carbon"),
)

#: The subscript of each row's observation operator.
SCHEMATIC_OPERATOR_NAMES = ("NEE", "LAI", "biomass", "soil\\ C")

#: How many prior draws the marginals figure histograms, and their seed.
PRIOR_MARGINAL_DRAWS = 20_000
PRIOR_MARGINAL_SEED = 0

#: The color of a prior's histogram and density.
PRIOR_COLOR = "#7f7f7f"

#: Tower marker color: the palette's vermillion, away from the sites' gray.
TOWER_COLOR = CURVE_COLORS[1]

#: Site marker color.
SITE_COLOR = "#9a9a9a"


def box_model(*, observations: bool) -> HTML:
    """The pools-and-fluxes diagram, with or without its observation layer."""
    path = FIGURE_DIRECTORY / "box_model.svg"
    if observations:
        return _inline_svg(path.read_text())
    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ElementTree.parse(path)
    root = tree.getroot()
    (layer,) = [
        element
        for element in root
        if element.get("id") == OBSERVATION_LAYER_ID
    ]
    root.remove(layer)
    return _inline_svg(ElementTree.tostring(root, encoding="unicode"))


def eddy_covariance() -> HTML:
    """The labeled eddy-covariance tower diagram."""
    return _inline_svg((FIGURE_DIRECTORY / "eddy_covariance.svg").read_text())


def sites_and_towers() -> Figure:
    """The site pool over North America, with the flux towers matched to it."""
    site_table = load_sites()
    sites = xr.DataArray(
        np.zeros(len(site_table)),
        dims=SITE,
        coords=site_coordinates(site_table["site_id"], site_table),
        name="pool_site",
        attrs={"long_name": "pool site", "units": "1"},
    )
    towers = read_tower_table()
    towers = towers[towers["site_id"].notna()]
    figure, ax = plt.subplots(figsize=(14, 7.4))
    plot_map(
        sites,
        ax=ax,
        cmap=ListedColormap([SITE_COLOR]),
        colorbar=False,
        s=2,
        linewidths=0,
    )
    x, y = SITE_PROJECTION.forward(
        towers["tower_lon"].to_numpy(), towers["tower_lat"].to_numpy()
    )
    ax.scatter(
        x, y, s=38, marker="^", color=TOWER_COLOR, edgecolor="white",
        linewidth=0.5, zorder=5,
    )
    ax.scatter([], [], s=12, color=SITE_COLOR, label=f"{len(site_table):,} sites")
    ax.scatter(
        [], [], s=38, marker="^", color=TOWER_COLOR,
        label=f"{towers['tower'].nunique()} flux towers",
    )
    ax.legend(loc="lower left", fontsize="medium", markerscale=1.5)
    return figure


def state_space_graph() -> Figure:
    """SIPNET as a directed graph: drivers and parameters into fluxes, fluxes into pools.

    Squares are given inputs; circles are computed from their parents, except
    the parameters and the initial pools, which have none.
    """
    figure, ax = plt.subplots(figsize=(11, 5.6))
    columns = {"1": 1, "2": 3, "T": 7}
    states = {"0": 0, "1": 2, "2": 4, "T-1": 6, "T": 8}
    rows = {"theta": 4.6, "driver": 2.4, "flux": 1.2, "state": 0.0}
    theta = (4.5, rows["theta"])
    _graph_node(ax, theta, r"$\theta$", unknown=True)
    for t, x in columns.items():
        _graph_node(ax, (x, rows["driver"]), rf"$u_{{{t}}}$", square=True)
        _graph_node(ax, (x, rows["flux"]), rf"$F_{{{t}}}$")
        _graph_edge(ax, theta, (x, rows["flux"]))
        _graph_edge(ax, (x, rows["driver"]), (x, rows["flux"]))
    for t, x in states.items():
        _graph_node(ax, (x, rows["state"]), rf"$x_{{{t}}}$", unknown=t == "0")
    for t, x in columns.items():
        _graph_edge(ax, (x - 1, rows["state"]), (x, rows["flux"]))
        _graph_edge(ax, (x, rows["flux"]), (x + 1, rows["state"]))
        _graph_edge(ax, (x - 1, rows["state"]), (x + 1, rows["state"]))
    for y in (rows["driver"], rows["flux"], rows["state"]):
        ax.text(5, y, r"$\cdots$", ha="center", va="center", fontsize=28)
    labels = {
        "theta": "parameters",
        "driver": "weather drivers",
        "flux": "fluxes",
        "state": "pools",
    }
    for row, label in labels.items():
        ax.text(9.0, rows[row], label, va="center", fontsize="large", color="#56635b")
    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-0.5, 5.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return figure


def forward_map_schematic() -> Figure:
    """The forward map as SIPNET then one reduction per data type, on one run.

    The reductions are illustrative, of the kind each observation operator
    makes: NEE averaged over windows, leaf area read at dates, biomass
    averaged by year, soil carbon averaged over the run.
    """
    daily = _prior_center_run().sel(time=SCHEMATIC_PERIOD)
    figure = plt.figure(figsize=(14, 6.2))
    grid = figure.add_gridspec(
        len(SCHEMATIC_ROWS), 5, width_ratios=(1.0, 0.45, 2.2, 0.8, 2.2)
    )
    _draw_parameters_into_sipnet(figure.add_subplot(grid[:, 0]))
    _draw_fan(figure.add_subplot(grid[:, 1]), len(SCHEMATIC_ROWS))
    for row, (state_name, label, prediction_label) in enumerate(SCHEMATIC_ROWS):
        trajectory_ax = figure.add_subplot(grid[row, 2])
        operator_ax = figure.add_subplot(grid[row, 3])
        prediction_ax = figure.add_subplot(grid[row, 4], sharey=trajectory_ax)
        trajectory = daily[state_name]
        trajectory_ax.plot(trajectory["time"], trajectory, color=CURVE_COLORS[0], linewidth=1)
        trajectory_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize="large")
        prediction_ax.plot(trajectory["time"], trajectory, color="#cccccc", linewidth=1)
        times, values = _illustrative_reduction(trajectory, state_name)
        prediction_ax.plot(times, values, "o", color=CURVE_COLORS[1], markersize=6)
        prediction_ax.set_ylabel(prediction_label, rotation=0, ha="left", va="center", fontsize="large")
        prediction_ax.yaxis.set_label_position("right")
        for ax in (trajectory_ax, prediction_ax):
            _strip_axes(ax)
        operator_ax.axis("off")
        operator_ax.annotate(
            "", xy=(0.95, 0.45), xytext=(0.05, 0.45), xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "#3b4652", "linewidth": 1.6,
                        "mutation_scale": 18},
        )
        operator_ax.text(
            0.5, 0.62, rf"$\mathcal{{H}}_{{\mathrm{{{SCHEMATIC_OPERATOR_NAMES[row]}}}}}$",
            ha="center", va="bottom", fontsize=20, transform=operator_ax.transAxes,
        )
        if row == 0:
            trajectory_ax.set_title(r"trajectory $\mathcal{M}(\theta)$", fontsize="large")
            prediction_ax.set_title(r"predictions $\mathcal{G}_k(\theta)$", fontsize="large")
    return figure


def prior_marginals() -> Figure:
    """Each calibrated parameter's prior: a histogram of draws, its law's
    density where it has one value, and the draws' median dashed."""
    posterior = prior.prior_alone()
    theta = posterior.sample_prior(
        jax.random.key(PRIOR_MARGINAL_SEED), PRIOR_MARGINAL_DRAWS
    )
    draws = prior.natural_table(posterior, theta)
    figure, axes = plt.subplots(4, 4, figsize=(14, 8.4))
    for ax, column in zip(axes.flat, PARAMETER_TITLES, strict=True):
        values = draws[column].to_numpy()
        low, high = np.quantile(values, [0.002, 0.998])
        ax.hist(values, bins=50, range=(low, high), density=True,
                color=PRIOR_COLOR, alpha=0.35)
        # The allocation's parts are one simplex-valued law, with no density
        # of their own; every other column is its own parameter.
        if column in posterior.parameter_names:
            grid = np.linspace(low, high, 400)
            density = np.exp(np.asarray(
                posterior.model.law(column, given={}).log_prob(grid)
            ))
            ax.plot(grid, density, color=PRIOR_COLOR, linewidth=1.6)
        ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1)
        ax.set_title(PARAMETER_TITLES[column], fontsize="medium")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    return figure


def noise_scale_prior() -> Figure:
    """The noise scales' inverse-gamma prior, its median and central 90%."""
    law = scipy.stats.invgamma(
        experiment_config.NOISE_SCALE_SHAPE, scale=noise.noise_scale_prior_scale()
    )
    grid = np.linspace(0.02, 7.0, 600)
    low, high = law.ppf([0.05, 0.95])
    figure, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(grid, law.pdf(grid), color=CURVE_COLORS[0], linewidth=2)
    inside = (grid >= low) & (grid <= high)
    ax.fill_between(grid[inside], law.pdf(grid[inside]), color=CURVE_COLORS[0], alpha=0.15)
    ax.axvline(law.median(), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$s_k$", fontsize="x-large")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return figure


def model_output() -> Figure:
    """One SIPNET run's NEE, leaf area index, wood and soil carbon, by day."""
    daily = _prior_center_run()
    figure = plt.figure(figsize=(14, 7.4))
    top, bottom = figure.subfigures(2, 1, height_ratios=(1.15, 1))
    nee_ax = top.subplots()
    pool_axes = bottom.subplots(1, 3)
    for ax, name in zip([nee_ax, *pool_axes], MODEL_OUTPUT_UNITS):
        plot_time_series(daily[name], ax=ax, color=CURVE_COLORS[0], linewidth=0.8)
        ax.set_title(_titles()[name], loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(YearLocator(base=2 if ax is nee_ax else 4))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    nee_ax.axhline(0, color="black", linewidth=0.6)
    return figure


# ── helpers ──


def _inline_svg(svg: str) -> HTML:
    """*svg* as inline HTML, so its text takes the slide's font."""
    return HTML(f'<div class="diagram">{svg}</div>')

#: The radius of a node of :func:`state_space_graph`, in data units.
_NODE_RADIUS = 0.36


def _graph_node(ax, center, label, *, square=False, unknown=False) -> None:
    """One node: a square for a given input, a circle otherwise, bold if unknown."""
    style = {
        "facecolor": "#e7ece5" if square else "white",
        "edgecolor": CURVE_COLORS[1] if unknown else "#3b4652",
        "linewidth": 3.0 if unknown else 1.8,
        "zorder": 3,
    }
    if square:
        side = 1.7 * _NODE_RADIUS
        patch = plt.Rectangle(
            (center[0] - side / 2, center[1] - side / 2), side, side, **style
        )
    else:
        patch = plt.Circle(center, _NODE_RADIUS, **style)
    ax.add_patch(patch)
    # A two-character subscript ("T-1") is set smaller, to stay inside its node.
    fontsize = 19 if "-" in label else 24
    ax.text(*center, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def _graph_edge(ax, start, end) -> None:
    """An arrow between two node centers, stopping at the nodes' edges."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = (end - start) / np.linalg.norm(end - start)
    ax.annotate(
        "",
        xy=end - direction * (_NODE_RADIUS + 0.04),
        xytext=start + direction * (_NODE_RADIUS + 0.04),
        arrowprops={"arrowstyle": "-|>", "color": "#3b4652", "linewidth": 1.6,
                    "mutation_scale": 18, "shrinkA": 0, "shrinkB": 0},
        zorder=2,
    )



@functools.cache
def _draw_parameters_into_sipnet(ax) -> None:
    """The schematic's left column: theta, an arrow, and SIPNET as a box."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # A marker, not a patch, so it stays round in axes that are not square.
    ax.scatter([0.5], [0.78], s=5200, facecolor="white", edgecolor=CURVE_COLORS[1],
               linewidth=3, zorder=3)
    ax.text(0.5, 0.78, r"$\theta$", ha="center", va="center", fontsize=26, zorder=4)
    ax.annotate("", xy=(0.5, 0.56), xytext=(0.5, 0.68),
                arrowprops={"arrowstyle": "-|>", "color": "#3b4652", "linewidth": 1.6,
                            "mutation_scale": 18})
    ax.add_patch(plt.Rectangle((0.08, 0.36), 0.84, 0.2, facecolor="#f4f7fa",
                               edgecolor="#3b4652", linewidth=2))
    ax.text(0.5, 0.46, r"SIPNET $\mathcal{M}$", ha="center", va="center", fontsize=15)


def _draw_fan(ax, n_rows: int) -> None:
    """Arrows from SIPNET's box to each row of the trajectory."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for row in range(n_rows):
        target = 1 - (row + 0.5) / n_rows
        ax.annotate("", xy=(0.95, target), xytext=(0.0, 0.46),
                    arrowprops={"arrowstyle": "-|>", "color": "#3b4652",
                                "linewidth": 1.4, "mutation_scale": 15})


def _illustrative_reduction(trajectory: xr.DataArray, state_name: str):
    """Times and values of an illustrative observation of *trajectory*."""
    series = trajectory.to_series()
    if state_name == "net_ecosystem_exchange":
        reduced = series.resample("MS").mean()
        return reduced.index + pd.Timedelta(days=15), reduced.to_numpy()
    if state_name == "leaf_carbon":
        reduced = series.iloc[::24]
        return reduced.index, reduced.to_numpy()
    if state_name == "wood_carbon":
        reduced = series.resample("YS").mean()
        return reduced.index + pd.Timedelta(days=182), reduced.to_numpy()
    middle = series.index[len(series) // 2]
    return [middle], [series.mean()]


def _strip_axes(ax) -> None:
    """No ticks and no frame but the time axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _prior_center_run() -> xr.Dataset:
    """SIPNET run once at the prior's center, its outputs by day.

    The run the experiment's prior predictive makes by hand, here without the
    observation vectors: theta at the prior's center, its natural values and
    the model's inputs to SIPNET parameter fields, one SIPNET call.
    """
    posterior = prior.prior_alone()
    theta = prior.prior_center(posterior)[None, :]
    values = {**posterior.to_labeled(theta), **prior.input_values()}
    sipnet_parameter_fields = prior.sipnet_parameter_map().sipnet_parameter_fields(
        values, site_dims=prior.site_dims()
    )
    at_the_run = sipnet_parameter_fields.isel({SAMPLE: 0}).sel(
        {SITE: experiment_config.SITE}
    )
    sipnet_overrides = {
        name: float(value) for name, value in at_the_run.data_vars.items()
    }
    sipnet_result = sipnet.sipnet_model()(**sipnet_overrides)
    model_output = to_model_output(
        sipnet_result,
        output_variable_names=[
            "net_ecosystem_exchange", "leaf_carbon", "wood_carbon", "soil_carbon"
        ],
        site=experiment_config.SITE,
        site_table=inputs.site_table(),
    )
    daily = xr.Dataset(
        {
            name: aggregate_time(model_output[name], "1D")
            for name in ("net_ecosystem_exchange", "leaf_carbon", "wood_carbon", "soil_carbon")
        }
    )
    leaf_carbon_per_area = sipnet_overrides["leaf_carbon_per_area"]
    leaf_area_index = daily["leaf_carbon"] / leaf_carbon_per_area
    leaf_area_index.attrs = {"long_name": "leaf area index", "units": "m2 m-2"}
    daily["leaf_area_index"] = leaf_area_index
    for name in ("wood_carbon", "soil_carbon"):
        daily[name] = convert_dataarray_units(daily[name], to_units="kg m-2")
    return daily


def _titles() -> dict[str, str]:
    """Each panel's title, with its units."""
    return {
        "net_ecosystem_exchange": "Net ecosystem exchange (g C m$^{-2}$ d$^{-1}$)",
        "leaf_area_index": "Leaf area index (m$^2$ m$^{-2}$)",
        "wood_carbon": "Wood carbon (kg C m$^{-2}$)",
        "soil_carbon": "Soil carbon (kg C m$^{-2}$)",
    }

"""The talk's figures: one function per figure, each reading saved results.

No function here computes a result. Each takes a results dataclass
(``results``) and returns a ``Figure``; ``run/draw_figures.py`` saves them.
The few choices made here, such as which site a figure zooms in on, are
documented where they are made (``showcase_sites``).

Toy maps are drawn on the unit square with the map grid as an image and
sites as markers (``style.SITE_MARKERS``): towers are triangles, sparse
sites circles, held-out sites crosses.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure

from sipnet_calibration.plotting.basemap import draw_basemap
from sipnet_calibration.projection import SITE_PROJECTION

from . import config
from .results import CalibrationOptions, NorthAmericaPrior, ToyResults, Transect
from .style import (
    DEVIATION_COLORMAP,
    FIELD_COLORMAP,
    METHOD_COLORS,
    METHOD_LABELS,
    PALETTE,
    SITE_MARKERS,
    UNCERTAINTY_COLORMAP,
)

__all__ = [
    "ShowcaseSites",
    "cost_bars",
    "equifinality",
    "held_out_errors",
    "ipl_agreement",
    "ipl_convergence",
    "maps_estimate",
    "maps_uncertainty",
    "north_america_prior_draws",
    "option_independent_map",
    "option_pft_map",
    "option_spatial_map",
    "showcase_sites",
    "site_pool_map",
    "toy_decomposition",
    "toy_world",
    "transect",
]

#: The fraction of probability inside each contour of a site's posterior.
CONTOUR_PROBABILITIES = (0.5, 0.9)
#: How many reference draws the equifinality figure scatters.
N_REFERENCE_DRAWS_SHOWN = 1500
#: The size of one calibration-option map, inches.
OPTION_MAP_SIZE = (5.2, 4.6)

# ── North America ───────────────────────────────────────────────────────────


def site_pool_map(north_america: NorthAmericaPrior) -> Figure:
    """The real site pool, with the flux-tower sites marked."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    x, y = SITE_PROJECTION.forward(north_america.lon, north_america.lat)
    is_tower = np.isin(north_america.site_ids, north_america.tower_site_ids)
    _north_america_frame(ax, x, y)
    ax.scatter(
        x[~is_tower], y[~is_tower], s=3, color=PALETTE["stone"], alpha=0.7, linewidth=0,
        label="site (~1 km grid cell)",
    )
    ax.scatter(
        x[is_tower], y[is_tower], **{**SITE_MARKERS["tower"], "s": 60},
        label="site with flux tower",
    )
    legend = ax.legend(loc="lower left", handletextpad=0.3)
    legend.legend_handles[0].set_sizes([40])  # the sites' 3-point dots, legible in the legend
    return fig


def north_america_prior_draws(north_america: NorthAmericaPrior) -> Figure:
    """Three draws of one parameter from a spatial prior, as maps, with the sites over them.

    There is no colorbar: the parameter is generic, and only the pattern
    matters. The draws share one color scale.
    """
    n_draws = len(north_america.draws)
    fig, axes = plt.subplots(1, n_draws, figsize=(5.2 * n_draws, 4.6), gridspec_kw={"wspace": 0.04})
    norm = Normalize(*np.nanpercentile(north_america.draws, [1, 99]))
    for ax, draw in zip(np.atleast_1d(axes), north_america.draws):
        _draw_field_map(ax, north_america, draw, norm)
    return fig


# ── the calibration options ─────────────────────────────────────────────────
# One map each, a parameter at every site under one option, drawn alike:
# no colorbar (the parameter is generic), each on its own 1-99% color range.


def option_independent_map(options: CalibrationOptions) -> Figure:
    """A parameter drawn independently at each site."""
    return _site_values_map(options, options.independent)


def option_pft_map(options: CalibrationOptions) -> Figure:
    """A parameter drawn once per PFT, shared by the PFT's sites."""
    return _site_values_map(options, options.by_pft)


def option_spatial_map(north_america: NorthAmericaPrior) -> Figure:
    """A parameter as a spatial field: the first North America prior draw, defined everywhere."""
    fig, ax = plt.subplots(figsize=OPTION_MAP_SIZE)
    draw = north_america.draws[0]
    _draw_field_map(ax, north_america, draw, Normalize(*np.nanpercentile(draw, [1, 99])))
    return fig


# ── the 1D transect ─────────────────────────────────────────────────────────


def transect(transect_result: Transect, *, stage: str) -> Figure:
    """The transect's prior (``stage="prior"``) or posterior (``"posterior"``).

    Both stages share axes, so the slides can step from one to the other.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = transect_result.locations
    two_sd = 2.0 * transect_result.prior_sd
    if stage == "prior":
        ax.fill_between(x, -two_sd, two_sd, color=PALETTE["mist"], label="prior: 95% range")
        for draw in transect_result.prior_draws:
            ax.plot(x, draw, color=PALETTE["stone"], linewidth=1.3, alpha=0.9)
        ax.plot([], [], color=PALETTE["stone"], linewidth=1.3, label="possible parameter fields")
    elif stage == "posterior":
        mean, sd = transect_result.posterior_mean, transect_result.posterior_sd
        ax.fill_between(x, mean - 2 * sd, mean + 2 * sd, color=PALETTE["forest"], alpha=0.18,
                        label="posterior: 95% range", linewidth=0)
        for draw in transect_result.posterior_draws:
            ax.plot(x, draw, color=PALETTE["forest"], linewidth=1.1, alpha=0.6)
        ax.plot(x, transect_result.true_field, color=PALETTE["ink"], linewidth=1.6, linestyle="--", label="truth")
        ax.scatter(
            transect_result.measurement_locations, transect_result.measurements,
            **SITE_MARKERS["tower"], label="sites that measure it",
        )
    else:
        raise ValueError(f"stage must be 'prior' or 'posterior', not {stage!r}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.6 * two_sd, 1.6 * two_sd)
    ax.set_xlabel("position along a transect")
    ax.set_ylabel("parameter")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
    return fig


# ── the toy: the problem ────────────────────────────────────────────────────


def toy_decomposition(toy: ToyResults, *, parameter: int) -> Figure:
    """The prior's terms for one parameter: trend + smooth variation + site noise = the parameter.

    Each panel has its own color scale, so each term's pattern is visible;
    the trend and the site noise are both small next to the smooth
    variation, which a shared scale would hide.
    """
    truth = toy.truth
    trend_map = truth.trend_map[:, parameter]
    variation_map = truth.smooth_variation_map[:, parameter]
    site_noise = truth.site_noise_training[:, parameter]
    site_value = truth.theta_training[:, parameter]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"wspace": 0.35})
    _image(axes[0], trend_map, toy.map_grid_size, cmap=FIELD_COLORMAP)
    _image(axes[1], variation_map, toy.map_grid_size, cmap=DEVIATION_COLORMAP, norm=_centered_norm(variation_map))
    axes[2].scatter(*toy.training_locations.T, c=site_noise, cmap=DEVIATION_COLORMAP, norm=_centered_norm(site_noise),
                    s=40, edgecolor=PALETTE["stone"], linewidth=0.4)
    axes[3].scatter(*toy.training_locations.T, c=site_value, cmap=FIELD_COLORMAP,
                    s=40, edgecolor=PALETTE["stone"], linewidth=0.4)
    titles = ("climate trend", "smooth variation", "site-to-site noise", "parameter at each site")
    for ax, title in zip(axes, titles):
        _toy_map_frame(ax)
        ax.set_title(title, pad=10)
    for left, right, symbol in zip(axes[:-1], axes[1:], ("+", "+", "=")):
        _symbol_between(fig, left, right, symbol)
    return fig


def toy_world(toy: ToyResults, *, parameter: int) -> Figure:
    """The toy problem: the true field with the sites, and what a tower and a sparse site observe."""
    sites = showcase_sites(toy)
    fig = plt.figure(figsize=(13.5, 5.0))
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 0.045, 0.42, 1.35), hspace=0.75, wspace=0.05)
    map_ax = fig.add_subplot(grid[:, 0])
    colorbar_ax = fig.add_subplot(grid[:, 1])
    nee_ax = fig.add_subplot(grid[0, 3])
    pool_ax = fig.add_subplot(grid[1, 3])

    true_field = toy.truth.trend_map[:, parameter] + toy.truth.smooth_variation_map[:, parameter]
    image = _image(map_ax, true_field, toy.map_grid_size, cmap=FIELD_COLORMAP)
    _draw_sites(map_ax, toy)
    # Label the two sites the right-hand panels show. The two are close
    # together, so the labels go toward the map's center (never clipped) at
    # different distances (never overlapping).
    for site, label, distance in ((sites.tower, "tower", 0.04), (sites.sparse, "sparse site", 0.16)):
        location = toy.training_locations[site]
        toward_center = np.where(location > 0.5, -1.0, 1.0)
        label_location = location + toward_center * [0.25, distance]
        map_ax.annotate(
            label, xy=location, xytext=label_location, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", linewidth=2.0),
        )
    _toy_map_frame(map_ax)
    map_ax.set_title(f"true {config.PARAMETER_NAMES[parameter]}", pad=10)
    fig.colorbar(image, cax=colorbar_ax, label=config.PARAMETER_LABELS[parameter])

    month_in_years = (np.arange(config.N_MONTHS) + 1) / 12
    nee = slice(0, config.N_MONTHS)
    nee_ax.plot(month_in_years, toy.true_model_output[sites.tower, nee], color=PALETTE["ink"], linewidth=1.5)
    nee_ax.scatter(month_in_years, toy.data[sites.tower, nee], color=PALETTE["sun"], edgecolor=PALETTE["ink"],
                   s=28, linewidth=0.6, zorder=3)
    nee_ax.set_title("a tower: monthly net ecosystem exchange", loc="left", fontsize=14)
    nee_ax.set_ylabel("NEE")

    years = np.arange(1, config.N_MONTHS // 12 + 1)
    pools = slice(config.N_MONTHS, None)
    pool_ax.plot(years, toy.true_model_output[sites.sparse, pools], color=PALETTE["ink"], linewidth=1.5)
    pool_ax.errorbar(years, toy.data[sites.sparse, pools], yerr=2 * toy.noise_sd[sites.sparse, pools],
                     fmt="o", color=PALETTE["ink"], markerfacecolor=PALETTE["sun"], markersize=7, capsize=3)
    pool_ax.set_title("a sparse site: the carbon pool once a year", loc="left", fontsize=14)
    pool_ax.set_ylabel("carbon")
    for ax in (nee_ax, pool_ax):
        ax.set_xlim(0, config.N_MONTHS / 12 + 0.2)
        ax.set_xlabel("year")
    # One legend, in the pool panel's empty upper left, covers both panels.
    pool_ax.scatter([], [], color=PALETTE["sun"], edgecolor=PALETTE["ink"], s=28, label="observed")
    pool_ax.plot([], [], color=PALETTE["ink"], linewidth=1.5, label="model at the truth")
    pool_ax.legend(loc="upper left", fontsize=12)
    return fig


# ── the toy: what the spatial prior buys ────────────────────────────────────


def equifinality(toy: ToyResults, *, stage: str) -> Figure:
    """One sparse site's posterior: site by site (``"site_by_site"``), then with the spatial one (``"spatial"``)."""
    site = showcase_sites(toy).sparse
    axis_0, axis_1 = toy.site_grids.grid_axes[site]
    density = toy.site_grids.density[site]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    levels = _highest_density_levels(density, CONTOUR_PROBABILITIES)
    ax.contourf(axis_0, axis_1, density.T, levels=[levels[-1], levels[0], density.max()],
                colors=[to_rgba(PALETTE["clay"], 0.18), to_rgba(PALETTE["clay"], 0.4)])
    ax.contour(axis_0, axis_1, density.T, levels=levels[::-1], colors=PALETTE["clay"], linewidths=1.5)
    ax.plot([], [], color=PALETTE["clay"], linewidth=6, alpha=0.4, label=METHOD_LABELS["site_by_site"])
    if stage == "spatial":
        draws = toy.reference_samples[:, site, :]
        shown = np.random.default_rng(0).choice(len(draws), size=min(N_REFERENCE_DRAWS_SHOWN, len(draws)), replace=False)
        ax.scatter(*draws[shown].T, s=5, color=METHOD_COLORS["spatial_reference"], alpha=0.35, linewidth=0,
                   label="spatial")
    elif stage != "site_by_site":
        raise ValueError(f"stage must be 'site_by_site' or 'spatial', not {stage!r}")
    ax.scatter(*toy.truth.theta_training[site], marker="*", s=380, color=PALETTE["ink"],
               edgecolor="white", linewidth=1.2, zorder=5, label="truth")
    limits = _equifinality_limits(toy, site)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xlabel(config.PARAMETER_LABELS[0])
    ax.set_ylabel(config.PARAMETER_LABELS[1])
    legend = ax.legend(loc="upper left")
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):  # scatter handles: the draws and the star
            handle.set_sizes([60 if handle.get_label() == "spatial" else 200])
            handle.set_alpha(1.0)
    return fig


def maps_estimate(toy: ToyResults, *, parameter: int) -> Figure:
    """The truth, the site-by-site estimates, and the spatial estimate of one parameter."""
    true_field = toy.truth.trend_map[:, parameter] + toy.truth.smooth_variation_map[:, parameter]
    norm = Normalize(*np.percentile(true_field, [0.5, 99.5]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw={"wspace": 0.08})

    image = _image(axes[0], true_field, toy.map_grid_size, cmap=FIELD_COLORMAP, norm=norm)
    axes[0].set_title("truth")

    site_estimate = toy.site_by_site.posterior_mean[:, parameter]
    axes[1].scatter(*toy.training_locations.T, c=site_estimate, cmap=FIELD_COLORMAP, norm=norm, s=55,
                    edgecolor=PALETTE["stone"], linewidth=0.4)
    axes[1].scatter(*toy.held_out_locations.T, **SITE_MARKERS["held_out"], label="no data: no estimate")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=12)
    axes[1].set_title(METHOD_LABELS["site_by_site"])

    _image(axes[2], toy.spatial_reference.map_mean[:, parameter], toy.map_grid_size, cmap=FIELD_COLORMAP, norm=norm)
    axes[2].set_title("spatial")
    for ax in axes:
        _toy_map_frame(ax)
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.015, label=config.PARAMETER_LABELS[parameter])
    return fig


def maps_uncertainty(toy: ToyResults, *, parameter: int) -> Figure:
    """Posterior SD of one parameter: site by site at each site, spatial everywhere."""
    prior_sd = float(toy.prior.held_out_sd[0, parameter])
    norm = Normalize(0.0, prior_sd)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"wspace": 0.08})

    site_sd = np.sqrt(toy.site_by_site.posterior_site_covariance[:, parameter, parameter])
    axes[0].scatter(*toy.training_locations.T, c=site_sd, cmap=UNCERTAINTY_COLORMAP, norm=norm, s=55,
                    edgecolor=PALETTE["stone"], linewidth=0.4)
    axes[0].set_title(METHOD_LABELS["site_by_site"])

    image = _image(axes[1], toy.spatial_reference.map_sd[:, parameter], toy.map_grid_size,
                   cmap=UNCERTAINTY_COLORMAP, norm=norm)
    axes[1].scatter(*toy.training_locations[toy.is_tower].T, **SITE_MARKERS["tower"])
    axes[1].set_title("spatial")
    for ax in axes:
        _toy_map_frame(ax)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="uncertainty (posterior SD)")
    return fig


def held_out_errors(toy: ToyResults) -> Figure:
    """Error at the held-out sites, the sites with no data, for each parameter and method."""
    methods = ("site_by_site", "spatial_reference", "ipl")
    labels = {"site_by_site": "site by site\n(prior only)", "spatial_reference": "spatial\n(exact)",
              "ipl": "spatial\n(scalable)"}
    fig, axes = plt.subplots(1, len(config.PARAMETER_NAMES), figsize=(11, 4.4), sharey=True)
    for parameter, ax in enumerate(axes):
        errors = [getattr(toy, method).held_out_root_mean_square_error[parameter] for method in methods]
        bars = ax.bar(range(len(methods)), errors, color=[METHOD_COLORS[m] for m in methods], width=0.62)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12)
        ax.set_xticks(range(len(methods)), [labels[m] for m in methods], fontsize=12)
        ax.set_title(config.PARAMETER_NAMES[parameter])
        ax.spines["left"].set_visible(parameter == 0)
    axes[0].set_ylabel("error at sites with no data\n(root-mean-square, log units)")
    return fig


# ── the toy: the scalable method ────────────────────────────────────────────


def ipl_agreement(toy: ToyResults, *, parameter: int) -> Figure:
    """The exact and the scalable spatial estimates of one parameter, side by side."""
    reference_map = toy.spatial_reference.map_mean[:, parameter]
    norm = Normalize(*np.percentile(reference_map, [0.5, 99.5]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"wspace": 0.08})
    for ax, method in zip(axes, ("spatial_reference", "ipl")):
        image = _image(ax, getattr(toy, method).map_mean[:, parameter], toy.map_grid_size,
                       cmap=FIELD_COLORMAP, norm=norm)
        _toy_map_frame(ax)
        ax.set_title(METHOD_LABELS[method])
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label=config.PARAMETER_LABELS[parameter])
    return fig


def cost_bars(toy: ToyResults) -> Figure:
    """Model runs per site: the exact spatial posterior by MCMC against IPL."""
    runs = {
        "exact (MCMC)\nneeds model gradients": toy.reference_model_runs_per_site,
        "scalable (IPL)\nno gradients": int(toy.ipl_history.runs_per_site[-1]),
    }
    colors = [METHOD_COLORS["spatial_reference"], METHOD_COLORS["ipl"]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    bars = ax.barh(range(len(runs)), list(runs.values()), color=colors, height=0.55)
    ax.bar_label(bars, labels=[f"{value:,}" for value in runs.values()], padding=6, fontsize=14)
    ax.set_yticks(range(len(runs)), list(runs.keys()))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(10, 20 * max(runs.values()))
    ax.set_xlabel("model runs per site (log scale)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig


def ipl_convergence(toy: ToyResults) -> Figure:
    """IPL's distance from the exact posterior after each iteration (a backup slide)."""
    history = toy.ipl_history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for parameter, ax in enumerate(axes):
        for name, errors, style in (
            ("towers", history.median_standardized_error_towers, "-"),
            ("sparse sites", history.median_standardized_error_sparse, "--"),
        ):
            ax.plot(history.runs_per_site, errors[:, parameter], linestyle=style, marker="o",
                    color=METHOD_COLORS["ipl"], label=name)
        ax.set_title(config.PARAMETER_NAMES[parameter])
        ax.set_xlabel("model runs per site")
        ax.set_yscale("log")
    axes[0].set_ylabel("|mean - exact mean| / exact SD\n(median over sites)")
    axes[0].legend()
    return fig


# ── choices ─────────────────────────────────────────────────────────────────


def _equifinality_limits(toy: ToyResults, site: int) -> list[tuple[float, float]]:
    """Axis limits holding both posteriors of ``site`` (both stages share them)."""
    axis_0, axis_1 = toy.site_grids.grid_axes[site]
    density = toy.site_grids.density[site]
    inside = density >= _highest_density_levels(density, (0.98,))[0]
    draws = toy.reference_samples[:, site, :]
    limits = []
    for axis, grid_axis, other in ((0, axis_0, 1), (1, axis_1, 0)):
        grid_points = grid_axis[inside.any(axis=other)]
        low = min(grid_points.min(), np.percentile(draws[:, axis], 0.5))
        high = max(grid_points.max(), np.percentile(draws[:, axis], 99.5))
        padding = 0.12 * (high - low)
        limits.append((low - padding, high + padding))
    return limits


@dataclass(frozen=True)
class ShowcaseSites:
    """The training sites the zoomed-in figures show."""

    sparse: int
    tower: int


def showcase_sites(toy: ToyResults) -> ShowcaseSites:
    """The sparse site and tower the zoomed-in figures show.

    The sparse site is the one at the 90th percentile of how much the
    spatial posterior narrows turnover compared with site by site (the ratio
    of their SDs): a site where the effect is clear, without taking the
    single most extreme one. Only sparse sites whose site-by-site 90% region
    holds the truth are candidates, so the figure shows the spatial model
    narrowing an answer that was right, not correcting one that was wrong,
    which would overstate the point. The tower is the one nearest to it.
    """
    sparse_sites = np.array([
        site for site in np.flatnonzero(~toy.is_tower) if _site_by_site_region_holds_truth(toy, site, 0.9)
    ])
    site_by_site_sd = np.sqrt(toy.site_by_site.posterior_site_covariance[sparse_sites, 1, 1])
    spatial_sd = np.sqrt(toy.spatial_reference.posterior_site_covariance[sparse_sites, 1, 1])
    narrowing = site_by_site_sd / spatial_sd
    order = np.argsort(narrowing)
    sparse = int(sparse_sites[order[int(0.9 * (len(order) - 1))]])

    towers = np.flatnonzero(toy.is_tower)
    distance = np.linalg.norm(toy.training_locations[towers] - toy.training_locations[sparse], axis=1)
    return ShowcaseSites(sparse=sparse, tower=int(towers[np.argmin(distance)]))


# ── helpers ─────────────────────────────────────────────────────────────────


def _image(ax, values: np.ndarray, grid_size: int, **style):
    """Draw map-grid values as an image over the unit square."""
    return ax.imshow(values.reshape(grid_size, grid_size), origin="lower", extent=(0, 1, 0, 1),
                     interpolation="nearest", **style)


def _draw_sites(ax, toy: ToyResults) -> None:
    ax.scatter(*toy.training_locations[~toy.is_tower].T, **SITE_MARKERS["sparse"])
    ax.scatter(*toy.training_locations[toy.is_tower].T, **SITE_MARKERS["tower"])


def _toy_map_frame(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["stone"])


def _site_values_map(options: CalibrationOptions, values: np.ndarray) -> Figure:
    """One map of a value at every site of the pool."""
    fig, ax = plt.subplots(figsize=OPTION_MAP_SIZE)
    x, y = SITE_PROJECTION.forward(options.lon, options.lat)
    _north_america_frame(ax, x, y)
    ax.scatter(x, y, c=values, cmap=FIELD_COLORMAP, norm=Normalize(*np.percentile(values, [1, 99])),
               s=4, linewidth=0, zorder=3)
    return fig


def _draw_field_map(ax, north_america: NorthAmericaPrior, draw: np.ndarray, norm: Normalize) -> None:
    """One prior draw as a continuous map on ``ax``, with the sites over it."""
    site_x, site_y = SITE_PROJECTION.forward(north_america.lon, north_america.lat)
    half_cell = (north_america.grid_x[1] - north_america.grid_x[0]) / 2
    extent = (
        north_america.grid_x[0] - half_cell, north_america.grid_x[-1] + half_cell,
        north_america.grid_y[0] - half_cell, north_america.grid_y[-1] + half_cell,
    )
    _north_america_frame(ax, site_x, site_y)
    ax.imshow(draw, origin="lower", extent=extent, cmap=FIELD_COLORMAP, norm=norm, interpolation="nearest")
    ax.scatter(site_x, site_y, s=0.4, color=PALETTE["ink"], alpha=0.35, linewidth=0, zorder=3)


def _north_america_frame(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Draw the basemap, framed on the projected site coordinates ``x``, ``y``."""
    margin = 0.03 * max(np.ptp(x), np.ptp(y))
    draw_basemap(ax)
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _centered_norm(values: np.ndarray) -> TwoSlopeNorm:
    """A diverging scale symmetric about 0 that holds every value."""
    largest = float(np.max(np.abs(values)))
    return TwoSlopeNorm(vcenter=0.0, vmin=-largest, vmax=largest)


def _symbol_between(fig, left_ax, right_ax, symbol: str) -> None:
    """Write ``symbol`` in the gap between two axes."""
    left, right = left_ax.get_position(), right_ax.get_position()
    fig.text((left.x1 + right.x0) / 2, (left.y0 + left.y1) / 2, symbol, ha="center", va="center",
             fontsize=34, color=PALETTE["ink"])


def _site_by_site_region_holds_truth(toy: ToyResults, site: int, probability: float) -> bool:
    """Whether the truth lies in the site-by-site posterior's highest-density region."""
    axis_0, axis_1 = toy.site_grids.grid_axes[site]
    density = toy.site_grids.density[site]
    true_0, true_1 = toy.truth.theta_training[site]
    if not (axis_0[0] <= true_0 <= axis_0[-1] and axis_1[0] <= true_1 <= axis_1[-1]):
        return False
    nearest = (np.abs(axis_0 - true_0).argmin(), np.abs(axis_1 - true_1).argmin())
    return bool(density[nearest] >= _highest_density_levels(density, (probability,))[0])


def _highest_density_levels(density: np.ndarray, probabilities) -> list[float]:
    """Density thresholds whose superlevel sets hold each probability, highest first."""
    values = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(values) / values.sum()
    return [float(values[np.searchsorted(cumulative, probability)]) for probability in probabilities]

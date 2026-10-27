"""The talk's figures: one function per figure, each reading saved results.

No function here computes a result. Each takes a results dataclass
(``results``) and returns a ``Figure``; ``run/draw_figures.py`` saves them.
The few choices made here, such as which site a figure zooms in on, are
documented where they are made (``showcase_sites``).

Toy maps are drawn on the unit square with the map grid as an image and
sites as markers (``style.SITE_MARKERS``): towers are triangles, sparse
sites circles, held-out sites crosses.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sipnet_calibration.plotting.basemap import draw_basemap
from sipnet_calibration.projection import SITE_PROJECTION

from . import config
from .inference.density import highest_density_levels
from .results import CalibrationOptions, NorthAmericaPrior, SitePosteriors, ToyResults, UncertaintyMaps
from .showcase import showcase_sites
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
    "cost_bars",
    "held_out_errors",
    "ipl_agreement",
    "ipl_convergence",
    "maps_estimate",
    "maps_uncertainty",
    "north_america_prior_draws",
    "site_posterior_comparison",
    "option_independent_map",
    "option_pft_map",
    "option_spatial_map",
    "site_pool_map",
    "toy_decomposition",
    "toy_world",
    "uncertainty_maps",
]

#: The size of one calibration-option map, inches.
OPTION_MAP_SIZE = (5.2, 4.6)
#: Text sizes on the estimate maps, which fill a slide's width (points).
MAP_TITLE_SIZE = 26
MAP_LABEL_SIZE = 20

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


def uncertainty_maps(uncertainty: UncertaintyMaps) -> Figure:
    """The field's prior and posterior SD side by side, with the sites and the strong-data towers.

    Both maps share one color scale, from 0 to the prior SD, labeled only
    low to high: the parameter is generic, and only the change matters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"wspace": 0.04})
    site_x, site_y = SITE_PROJECTION.forward(uncertainty.site_lon, uncertainty.site_lat)
    tower_x, tower_y = SITE_PROJECTION.forward(uncertainty.tower_lon, uncertainty.tower_lat)
    norm = Normalize(0.0, float(np.nanmax(uncertainty.prior_sd)))
    for ax, sd, title in (
        (axes[0], uncertainty.prior_sd, "prior parameter uncertainty"),
        (axes[1], uncertainty.posterior_sd, "posterior parameter uncertainty"),
    ):
        _north_america_frame(ax, site_x, site_y)
        image = ax.imshow(sd, origin="lower", extent=_raster_extent(uncertainty.grid_x, uncertainty.grid_y),
                          cmap=UNCERTAINTY_COLORMAP, norm=norm, interpolation="nearest")
        ax.scatter(site_x, site_y, s=0.4, color=PALETTE["ink"], alpha=0.3, linewidth=0, zorder=3)
        # Hollow, so the dip in uncertainty around each tower shows through.
        ax.scatter(tower_x, tower_y, marker="^", s=55, facecolor="none", edgecolor=PALETTE["ink"],
                   linewidth=1.3, zorder=4)
        ax.set_title(title, pad=8)
    axes[0].legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", markersize=3, color=PALETTE["ink"], alpha=0.6,
                   label="site (weak data)"),
            Line2D([], [], linestyle="none", marker="^", markersize=8, markerfacecolor="none",
                   markeredgecolor=PALETTE["ink"], markeredgewidth=1.3, label="flux tower (strong data)"),
        ],
        loc="lower left", fontsize=12, handletextpad=0.3,
    )
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40,
                            ticks=[norm.vmin, norm.vmax])
    colorbar.ax.set_xticklabels(["low", "high"])
    colorbar.outline.set_visible(False)
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
    """The toy problem: the true field with the sites, and what a tower and a sparse site observe.

    The map has no colorbar: only the field's pattern matters here. The tower
    and sparse site shown are ``showcase_sites``'.
    """
    sites = showcase_sites(toy)
    fig = plt.figure(figsize=(16, 3.6))
    # Column widths in inches: a square map as tall as the panels, room for the
    # panels' axis labels, the panels.
    grid = fig.add_gridspec(2, 3, width_ratios=(3.0, 1.0, 12.0), hspace=0.55, wspace=0.05)
    map_ax = fig.add_subplot(grid[:, 0])
    nee_ax = fig.add_subplot(grid[0, 2])
    pool_ax = fig.add_subplot(grid[1, 2])

    true_field = toy.truth.trend_map[:, parameter] + toy.truth.smooth_variation_map[:, parameter]
    _image(map_ax, true_field, toy.map_grid_size, cmap=FIELD_COLORMAP)
    _draw_sites(map_ax, toy)
    _toy_map_frame(map_ax)
    map_ax.set_title(f"true {config.PARAMETER_NAMES[parameter]}", pad=10)

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
    # The panels share their years, so only the lower one is labeled.
    pool_ax.set_xlabel("year")
    # One legend, in the pool panel's empty upper left, covers both panels.
    pool_ax.scatter([], [], color=PALETTE["sun"], edgecolor=PALETTE["ink"], s=28, label="observed")
    pool_ax.plot([], [], color=PALETTE["ink"], linewidth=1.5, label="model at the truth")
    pool_ax.legend(loc="upper left", fontsize=12)
    return fig


# ── the toy: what the spatial prior buys ────────────────────────────────────


def site_posterior_comparison(site_posteriors: SitePosteriors, *, which: str) -> Figure:
    """One site's posterior site by site and spatially, as contours, with a map locating the site.

    ``which`` is ``"tower"`` or ``"sparse"``. The contours hold
    ``config.CONTOUR_PROBABILITIES``: exact for site by site, a checked KDE
    for spatial (``run/site_posteriors.py``).
    """
    posterior = getattr(site_posteriors, which)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for axes, density, color in (
        (posterior.site_by_site_axes, posterior.site_by_site_density, METHOD_COLORS["site_by_site"]),
        (posterior.spatial_axes, posterior.spatial_density, METHOD_COLORS["spatial_reference"]),
    ):
        _draw_contours(ax, axes, density, color)
    ax.scatter(*posterior.truth, marker="*", s=380, color=PALETTE["ink"], edgecolor="white", linewidth=1.2,
               zorder=5)
    ax.set_xlim(posterior.spatial_axes[0][[0, -1]])
    ax.set_ylim(posterior.spatial_axes[1][[0, -1]])
    ax.set_xlabel(config.PARAMETER_LABELS[0])
    ax.set_ylabel(config.PARAMETER_LABELS[1])
    ax.legend(
        handles=[
            Patch(facecolor=to_rgba(METHOD_COLORS["site_by_site"], 0.35), edgecolor=METHOD_COLORS["site_by_site"],
                  label=METHOD_LABELS["site_by_site"]),
            Patch(facecolor=to_rgba(METHOD_COLORS["spatial_reference"], 0.35),
                  edgecolor=METHOD_COLORS["spatial_reference"], label="spatial"),
            Line2D([], [], linestyle="none", marker="*", markersize=16, color=PALETTE["ink"], label="truth"),
        ],
        loc="upper left", fontsize=13,
    )
    _draw_locator_map(ax.inset_axes((0.66, 0.04, 0.32, 0.32)), site_posteriors, posterior.site)
    return fig


def maps_estimate(toy: ToyResults, *, parameter: int) -> Figure:
    """The truth, the site-by-site estimates, and the spatial estimate of one parameter."""
    true_field = toy.truth.trend_map[:, parameter] + toy.truth.smooth_variation_map[:, parameter]
    norm = Normalize(*np.percentile(true_field, [0.5, 99.5]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"wspace": 0.05})

    image = _image(axes[0], true_field, toy.map_grid_size, cmap=FIELD_COLORMAP, norm=norm)
    axes[0].set_title("truth", fontsize=MAP_TITLE_SIZE)

    site_estimate = toy.site_by_site.posterior_mean[:, parameter]
    axes[1].scatter(*toy.training_locations.T, c=site_estimate, cmap=FIELD_COLORMAP, norm=norm, s=55,
                    edgecolor=PALETTE["stone"], linewidth=0.4)
    axes[1].scatter(*toy.held_out_locations.T, **SITE_MARKERS["held_out"], label="no data: no estimate")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=MAP_LABEL_SIZE, handletextpad=0.2, markerscale=1.6)
    axes[1].set_title(METHOD_LABELS["site_by_site"], fontsize=MAP_TITLE_SIZE)

    _image(axes[2], toy.spatial_reference.map_mean[:, parameter], toy.map_grid_size, cmap=FIELD_COLORMAP, norm=norm)
    axes[2].set_title("spatial", fontsize=MAP_TITLE_SIZE)
    for ax in axes:
        _toy_map_frame(ax)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.015)
    colorbar.set_label(config.PARAMETER_LABELS[parameter], fontsize=MAP_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=MAP_LABEL_SIZE - 3)
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
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"wspace": 0.06})
    for ax, method in zip(axes, ("spatial_reference", "ipl")):
        image = _image(ax, getattr(toy, method).map_mean[:, parameter], toy.map_grid_size,
                       cmap=FIELD_COLORMAP, norm=norm)
        _toy_map_frame(ax)
        ax.set_title(METHOD_LABELS[method], fontsize=MAP_TITLE_SIZE)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label(config.PARAMETER_LABELS[parameter], fontsize=MAP_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=MAP_LABEL_SIZE - 3)
    return fig


def cost_bars(toy: ToyResults) -> Figure:
    """Model runs per site: the exact spatial posterior by MCMC against IPL."""
    runs = {
        "exact (MCMC)\nwith gradients": toy.reference_model_runs_per_site,
        "scalable (IPL)\nno gradients": int(toy.ipl_history.runs_per_site[-1]),
    }
    colors = [METHOD_COLORS["spatial_reference"], METHOD_COLORS["ipl"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    bars = ax.bar(range(len(runs)), list(runs.values()), color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{value:,}" for value in runs.values()], padding=6, fontsize=MAP_LABEL_SIZE)
    ax.set_xticks(range(len(runs)), list(runs.keys()), fontsize=MAP_LABEL_SIZE - 2)
    ax.set_yscale("log")
    ax.set_ylim(10, 30 * max(runs.values()))
    ax.set_ylabel("model runs per site (log scale)", fontsize=MAP_LABEL_SIZE - 2)
    ax.tick_params(axis="y", labelsize=MAP_LABEL_SIZE - 4)
    ax.tick_params(axis="x", length=0)
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


# ── helpers ─────────────────────────────────────────────────────────────────


def _draw_contours(ax, axes: np.ndarray, density: np.ndarray, color: str) -> None:
    """Filled highest-density regions of a grid density, darker for the more probable one, outlined."""
    levels = highest_density_levels(density, config.CONTOUR_PROBABILITIES)  # one per probability, high to low
    ax.contourf(*axes, density.T, levels=[levels[-1], levels[0], density.max()],
                colors=[to_rgba(color, 0.15), to_rgba(color, 0.35)])
    ax.contour(*axes, density.T, levels=levels[::-1], colors=color, linewidths=1.6)


def _draw_locator_map(ax, site_posteriors: SitePosteriors, site: int) -> None:
    """A small map of the toy region, its true turnover and its sites, with ``site`` ringed."""
    _image(ax, site_posteriors.true_turnover_map, site_posteriors.map_grid_size, cmap=FIELD_COLORMAP, alpha=0.8)
    ax.scatter(*site_posteriors.training_locations.T, s=3, color="white", linewidth=0)
    ax.scatter(*site_posteriors.training_locations[site], s=160, facecolor="none", edgecolor=PALETTE["clay"],
               linewidth=2.5, zorder=4)
    _toy_map_frame(ax)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["ink"])


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
    _north_america_frame(ax, site_x, site_y)
    ax.imshow(draw, origin="lower", extent=_raster_extent(north_america.grid_x, north_america.grid_y),
              cmap=FIELD_COLORMAP, norm=norm, interpolation="nearest")
    ax.scatter(site_x, site_y, s=0.4, color=PALETTE["ink"], alpha=0.35, linewidth=0, zorder=3)


def _raster_extent(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[float, float, float, float]:
    """``imshow``'s extent for a raster with these cell centers: its outer cell edges."""
    half_cell = (grid_x[1] - grid_x[0]) / 2
    return (grid_x[0] - half_cell, grid_x[-1] + half_cell, grid_y[0] - half_cell, grid_y[-1] + half_cell)


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

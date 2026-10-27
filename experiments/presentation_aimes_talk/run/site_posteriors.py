"""Compute two sites' posteriors as densities, and check the KDE behind one of them.

Overview
--------
For the showcase tower and sparse site (``showcase``), lay out each site's
posterior over its two parameters as a density on a grid, two ways:

- **site by site**: the exact posterior, already on its grid in the toy
  results;
- **spatial**: a Gaussian KDE of the reference MCMC draws at that site, on
  a grid holding both posteriors, its bandwidth
  ``config.KDE_BANDWIDTH_FACTOR`` times Scott's rule.

Then check the KDE (``inference.density`` describes the checks):
**reproduction**, applying the same KDE to as many draws from the exact
site-by-site posterior, and **coverage**, of the MCMC draws by the spatial
KDE's regions. Both are printed, and saved with the densities.

Input data
----------
``config.TOY_RESULTS_FILE``, written by ``run/toy_experiment.py``.

Output data
-----------
``config.SITE_POSTERIORS_FILE``: a ``results.SitePosteriors``.

Usage
-----
::

    uv run python -m experiments.presentation_aimes_talk.run.site_posteriors
"""

import numpy as np

from .. import config, results
from ..inference.density import (
    fraction_of_samples_inside,
    gaussian_kde_on_grid,
    highest_density_levels,
    mass_inside,
    sample_from_grid_density,
)
from ..showcase import showcase_sites

#: Of the exact site-by-site posterior, the region the KDE grid must hold.
SITE_BY_SITE_REGION_HELD = 0.99
#: Of the MCMC draws, the central fraction the KDE grid must hold, per axis.
DRAWS_HELD = 0.99
#: Margin added to the KDE grid on each side, as a fraction of its span.
GRID_MARGIN = 0.15

TURNOVER = 1


def main() -> None:
    toy = results.load(results.ToyResults, config.TOY_RESULTS_FILE)
    sites = showcase_sites(toy)
    rng = np.random.default_rng(config.KDE_CHECK_SEED)
    site_posteriors = results.SitePosteriors(
        tower=_site_posterior(toy, sites.tower, rng),
        sparse=_site_posterior(toy, sites.sparse, rng),
        training_locations=toy.training_locations,
        is_tower=toy.is_tower,
        map_grid_size=toy.map_grid_size,
        true_turnover_map=toy.truth.trend_map[:, TURNOVER] + toy.truth.smooth_variation_map[:, TURNOVER],
    )
    results.save(site_posteriors, config.SITE_POSTERIORS_FILE)
    _report(site_posteriors)
    print(f"wrote {config.SITE_POSTERIORS_FILE}")


# ── the steps ───────────────────────────────────────────────────────────────


def _site_posterior(toy: results.ToyResults, site: int, rng: np.random.Generator) -> results.SitePosterior:
    """One site's two densities, and the KDE's checks."""
    exact_axes = toy.site_grids.grid_axes[site]
    exact_density = toy.site_grids.density[site].astype(float)
    draws = toy.reference_samples[:, site, :]

    spatial_axes = _axes_holding(exact_axes, exact_density, draws)
    spatial_density = gaussian_kde_on_grid(draws, *spatial_axes, bandwidth_factor=config.KDE_BANDWIDTH_FACTOR)

    # Reproduction: the same KDE, applied to as many draws from a density we
    # know exactly, should give regions holding that density's probability.
    exact_draws = sample_from_grid_density(exact_density, *exact_axes, rng, len(draws))
    reproduced = gaussian_kde_on_grid(exact_draws, *exact_axes, bandwidth_factor=config.KDE_BANDWIDTH_FACTOR)
    reproduction_mass = [
        mass_inside(exact_density, reproduced >= level)
        for level in highest_density_levels(reproduced, config.CONTOUR_PROBABILITIES)
    ]
    # Coverage: the spatial KDE's regions should hold that share of the draws.
    draw_coverage = [
        fraction_of_samples_inside(draws, spatial_density, *spatial_axes, level)
        for level in highest_density_levels(spatial_density, config.CONTOUR_PROBABILITIES)
    ]
    return results.SitePosterior(
        site=site,
        truth=toy.truth.theta_training[site],
        site_by_site_axes=exact_axes,
        site_by_site_density=exact_density,
        spatial_axes=spatial_axes,
        spatial_density=spatial_density,
        kde_reproduction_mass=np.array(reproduction_mass),
        spatial_draw_coverage=np.array(draw_coverage),
    )


def _axes_holding(exact_axes: np.ndarray, exact_density: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """KDE grid axes ``(2, KDE_GRID_SIZE)`` holding both posteriors, with a margin."""
    held = exact_density >= highest_density_levels(exact_density, (SITE_BY_SITE_REGION_HELD,))[0]
    tail = 100 * (1 - DRAWS_HELD) / 2
    axes = []
    for axis, other in ((0, 1), (1, 0)):
        exact_points = exact_axes[axis][held.any(axis=other)]
        low = min(exact_points.min(), np.percentile(draws[:, axis], tail))
        high = max(exact_points.max(), np.percentile(draws[:, axis], 100 - tail))
        margin = GRID_MARGIN * (high - low)
        axes.append(np.linspace(low - margin, high + margin, config.KDE_GRID_SIZE))
    return np.array(axes)


def _report(site_posteriors: results.SitePosteriors) -> None:
    """Print the KDE's checks, against the probability each region should hold."""
    targets = ", ".join(f"{p:.0%}" for p in config.CONTOUR_PROBABILITIES)
    print(f"KDE checks; each should be close to {targets}")
    for name in ("tower", "sparse"):
        posterior = getattr(site_posteriors, name)
        print(
            f"  {name:6s} site {posterior.site:3d}: exact mass in the KDE's regions "
            f"{np.round(posterior.kde_reproduction_mass, 3)}; MCMC draws in the spatial KDE's regions "
            f"{np.round(posterior.spatial_draw_coverage, 3)}"
        )


if __name__ == "__main__":
    main()

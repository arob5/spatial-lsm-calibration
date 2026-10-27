"""Which toy sites the zoomed-in figures show.

A presentation choice, not a result, made here once so that every figure
and computation that zooms in on a site shows the same ones.
"""

from dataclasses import dataclass

import numpy as np

from .inference.density import highest_density_levels
from .results import ToyResults

__all__ = ["ShowcaseSites", "showcase_sites"]


@dataclass(frozen=True)
class ShowcaseSites:
    """The training sites the zoomed-in figures show, by position."""

    sparse: int
    tower: int


def showcase_sites(toy: ToyResults) -> ShowcaseSites:
    """The sparse site and tower the zoomed-in figures show.

    The sparse site is the one at the 90th percentile of how much the
    spatial posterior narrows turnover compared with site by site (the ratio
    of their SDs): a site where the effect is clear, without taking the
    single most extreme one. Only sparse sites whose site-by-site 90% region
    holds the truth are candidates, so the figures show the spatial model
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


def _site_by_site_region_holds_truth(toy: ToyResults, site: int, probability: float) -> bool:
    """Whether the truth lies in the site-by-site posterior's highest-density region."""
    axis_0, axis_1 = toy.site_grids.grid_axes[site]
    density = toy.site_grids.density[site]
    true_0, true_1 = toy.truth.theta_training[site]
    if not (axis_0[0] <= true_0 <= axis_0[-1] and axis_1[0] <= true_1 <= axis_1[-1]):
        return False
    nearest = (np.abs(axis_0 - true_0).argmin(), np.abs(axis_1 - true_1).argmin())
    return bool(density[nearest] >= highest_density_levels(density, (probability,))[0])

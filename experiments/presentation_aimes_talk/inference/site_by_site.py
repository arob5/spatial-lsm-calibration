"""Site-by-site calibration: each site alone, computed exactly on a grid.

The practice the talk argues against: each site's parameters are
calibrated with only that site's data and its own prior,

    p(theta_s | y_s)  proportional to  p(y_s | theta_s) N(theta_s; m(x_s), diag(prior_sd^2)),

where the prior is the site's marginal under the spatial prior: it keeps the
climate trend, but no site learns from any other. (Giving each site the
trend is generous to this method; a PFT-level prior would be cruder.)

With two parameters the posterior is computed exactly on a grid: the
unnormalized density at every grid point, normalized by its sum. A grid
wide enough for the prior is too coarse for a tower's much narrower
posterior, so the grid is re-centered and narrowed on the posterior
(``config`` sets how). The grid is fine enough that the moments and the
density are exact for the talk's purposes; the cost (``grid_size^2`` model
runs per pass per site) is irrelevant here, as this is a reference for the
idea, not a method that scales.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..toy.domain import Drivers
from ..toy.forward_model import CarbonModel
from ..toy.observations import ObservationDesign, gaussian_log_likelihood
from .moments import PosteriorMoments

__all__ = ["SiteGridPosteriors", "site_by_site_posteriors"]


@dataclass(frozen=True)
class SiteGridPosteriors:
    """Each site's posterior on its own (final) grid."""

    #: ``(n_sites, n_parameters, grid_size)``: each site's grid axes.
    grid_axes: np.ndarray
    #: ``(n_sites, grid_size, grid_size)``: the density on the grid, first
    #: index along parameter 0's axis, second along parameter 1's.
    density: np.ndarray
    moments: PosteriorMoments


def site_by_site_posteriors(
    model: CarbonModel,
    drivers: Drivers,
    design: ObservationDesign,
    data: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    *,
    grid_size: int,
    half_width_in_prior_sds: float,
    half_width_in_posterior_sds: float,
    n_refinements: int,
) -> SiteGridPosteriors:
    """Every training site's posterior, each computed independently.

    ``prior_mean`` is ``(n_sites, 2)``; ``prior_sd`` is ``(2,)``.
    """
    log_posterior_on_grid = _compile_log_posterior(model)
    n_sites = len(prior_mean)
    grid_axes = np.empty((n_sites, 2, grid_size))
    density = np.empty((n_sites, grid_size, grid_size), dtype=np.float32)
    means = np.empty((n_sites, 2))
    covariances = np.empty((n_sites, 2, 2))

    for site in range(n_sites):
        site_inputs = (
            drivers.temperature[site],
            drivers.light[site],
            data[site],
            design.observed[site],
            design.noise_sd[site],
            prior_mean[site],
            prior_sd,
        )
        center, half_width = prior_mean[site], half_width_in_prior_sds * prior_sd
        for _ in range(1 + n_refinements):
            axes = center[:, None] + np.linspace(-1.0, 1.0, grid_size)[None, :] * half_width[:, None]
            weights = _normalized_weights(np.asarray(log_posterior_on_grid(axes, *site_inputs)))
            mean, covariance = _grid_moments(axes, weights)
            # Narrow to the posterior, but never below a few of the current
            # grid's cells, which a posterior narrower than one cell would ask for.
            cell_width = axes[:, 1] - axes[:, 0]
            center = mean
            half_width = np.maximum(half_width_in_posterior_sds * np.sqrt(np.diag(covariance)), 3.0 * cell_width)

        grid_axes[site] = axes
        cell_area = np.prod(axes[:, 1] - axes[:, 0])
        density[site] = weights / cell_area
        means[site], covariances[site] = mean, covariance

    return SiteGridPosteriors(
        grid_axes=grid_axes,
        density=density,
        moments=PosteriorMoments.independent_sites(means, covariances),
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _compile_log_posterior(model: CarbonModel):
    """A jitted ``(axes, *site inputs) -> (grid_size, grid_size)`` unnormalized log posterior."""

    def log_posterior_on_grid(axes, temperature, light, site_data, observed, noise_sd, prior_mean, prior_sd):
        # Every (theta_0, theta_1) pair of the grid, "ij" so [i, j] is (axis 0[i], axis 1[j]).
        theta_0, theta_1 = jnp.meshgrid(axes[0], axes[1], indexing="ij")
        theta = jnp.stack([theta_0, theta_1], axis=-1)
        model_output = model.simulate(theta, temperature, light)
        log_likelihood = gaussian_log_likelihood(model_output, site_data, observed, noise_sd)
        log_prior = -0.5 * jnp.sum(((theta - prior_mean) / prior_sd) ** 2, axis=-1)
        return log_likelihood + log_prior

    return jax.jit(log_posterior_on_grid)


def _normalized_weights(log_density: np.ndarray) -> np.ndarray:
    """Probabilities of the grid points, from the unnormalized log density."""
    weights = np.exp(log_density - log_density.max())
    return weights / weights.sum()


def _grid_moments(axes: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``(2,)`` and covariance ``(2, 2)`` of the grid distribution."""
    theta_0, theta_1 = np.meshgrid(axes[0], axes[1], indexing="ij")
    points = np.stack([theta_0.ravel(), theta_1.ravel()], axis=1)
    flat_weights = weights.ravel()
    mean = flat_weights @ points
    centered = points - mean
    covariance = (centered * flat_weights[:, None]).T @ centered
    return mean, covariance

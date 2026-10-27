"""The 1D transect: how a spatial prior spreads information from measured sites.

A parameter ``theta(x)`` along a line ``x`` in [0, 1] has a Gaussian process
prior, mean 0 and Matern-3/2 covariance. A few sites measure it directly,
with Gaussian noise; in the talk's terms a tower whose data pin a parameter
down acts like such a measurement. The posterior is exact Gaussian process
regression:

    mean(x) = k_xM (K_MM + s^2 I)^{-1} y,
    var(x)  = k_xx - k_xM (K_MM + s^2 I)^{-1} k_Mx,

where ``M`` are the measured locations, ``y`` the measurements and ``s``
their noise SD. The truth the measurements are made of is itself a prior
draw.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky

from ..results import Transect
from ..toy.spatial_prior import CHOLESKY_JITTER, matern32_correlation

__all__ = ["compute_transect"]


def compute_transect(
    rng: np.random.Generator,
    *,
    length_scale: float,
    field_sd: float,
    measurement_locations: np.ndarray,
    measurement_sd: float,
    grid_size: int,
    n_draws: int,
) -> Transect:
    """The prior, a truth, measurements of it, and the posterior along the line."""
    locations = np.linspace(0.0, 1.0, grid_size)

    def covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return field_sd**2 * matern32_correlation(a[:, None], b[:, None], length_scale)

    prior_cholesky = cholesky(covariance(locations, locations) + CHOLESKY_JITTER * np.eye(grid_size), lower=True)
    true_field = prior_cholesky @ rng.standard_normal(grid_size)
    prior_draws = (prior_cholesky @ rng.standard_normal((grid_size, n_draws))).T

    true_at_measurements = np.interp(measurement_locations, locations, true_field)
    measurements = true_at_measurements + measurement_sd * rng.standard_normal(len(measurement_locations))

    measured_covariance = covariance(measurement_locations, measurement_locations) + measurement_sd**2 * np.eye(
        len(measurement_locations)
    )
    cross_covariance = covariance(locations, measurement_locations)
    factor = cho_factor(measured_covariance, lower=True)
    posterior_mean = cross_covariance @ cho_solve(factor, measurements)
    posterior_covariance = covariance(locations, locations) - cross_covariance @ cho_solve(factor, cross_covariance.T)

    posterior_cholesky = cholesky(posterior_covariance + CHOLESKY_JITTER * np.eye(grid_size), lower=True)
    posterior_draws = posterior_mean + (posterior_cholesky @ rng.standard_normal((grid_size, n_draws))).T
    return Transect(
        locations=locations,
        true_field=true_field,
        prior_draws=prior_draws,
        prior_sd=field_sd,
        measurement_locations=measurement_locations,
        measurements=measurements,
        posterior_mean=posterior_mean,
        posterior_sd=np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0)),
        posterior_draws=posterior_draws,
    )

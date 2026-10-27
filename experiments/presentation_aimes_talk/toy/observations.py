"""The observation model: what each site observes, the noise, and the likelihood.

Every training site observes its pool at the end of each year. A random
subset of "tower" sites also observes every month's NEE. Observations are
the model's output at the site's true parameters plus independent Gaussian
noise:

    y_s = G_s(theta_s) + e_s,   e_s ~ N(0, diag(noise_sd_s^2)),

restricted to the outputs site ``s`` observes. The noise SDs are known.

Arrays over sites and outputs are kept rectangular, ``(n_sites,
n_outputs)``, with a boolean ``observed`` mask: an unobserved entry holds a
placeholder (data 0, SD 1) that the likelihood never reads.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .domain import Drivers
from .forward_model import CarbonModel

__all__ = ["ObservationDesign", "design_observations", "gaussian_log_likelihood", "simulate_observations"]


@dataclass(frozen=True)
class ObservationDesign:
    """Which outputs each training site observes, and how noisily."""

    #: ``(n_sites,)``: whether the site is a tower.
    is_tower: np.ndarray
    #: ``(n_sites, n_outputs)``: whether the site observes the output.
    observed: np.ndarray
    #: ``(n_sites, n_outputs)``: the noise SD; 1 where unobserved.
    noise_sd: np.ndarray


def design_observations(
    model: CarbonModel,
    drivers: Drivers,
    prior_mean_theta: np.ndarray,
    rng: np.random.Generator,
    *,
    tower_fraction: float,
    pool_relative_noise: float,
    nee_relative_noise: float,
) -> ObservationDesign:
    """Choose the towers and set each site's noise SDs.

    ``prior_mean_theta`` is ``(n_sites, 2)``: each site's prior mean, at
    which the noise scale is set (``config`` explains the rule). Setting it
    from the prior mean rather than the truth keeps the truth out of the
    data's error model.
    """
    n_sites = len(prior_mean_theta)
    n_towers = round(tower_fraction * n_sites)
    is_tower = np.zeros(n_sites, dtype=bool)
    is_tower[rng.choice(n_sites, size=n_towers, replace=False)] = True

    observed = np.zeros((n_sites, model.n_outputs), dtype=bool)
    observed[:, model.annual_pool_indices] = True
    observed[np.ix_(is_tower, model.nee_indices)] = True

    output_at_prior_mean = np.asarray(model.simulate(prior_mean_theta, drivers.temperature, drivers.light))
    mean_pool = output_at_prior_mean[:, model.annual_pool_indices].mean(axis=1)
    rms_nee = np.sqrt((output_at_prior_mean[:, model.nee_indices] ** 2).mean(axis=1))
    noise_sd = np.ones((n_sites, model.n_outputs))
    noise_sd[:, model.annual_pool_indices] = pool_relative_noise * mean_pool[:, None]
    noise_sd[:, model.nee_indices] = nee_relative_noise * rms_nee[:, None]
    return ObservationDesign(is_tower=is_tower, observed=observed, noise_sd=noise_sd)


def simulate_observations(
    model: CarbonModel,
    true_theta: np.ndarray,
    drivers: Drivers,
    design: ObservationDesign,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic data ``(n_sites, n_outputs)``: the true output plus noise, 0 where unobserved."""
    true_output = np.asarray(model.simulate(true_theta, drivers.temperature, drivers.light))
    noisy = true_output + design.noise_sd * rng.standard_normal(true_output.shape)
    return np.where(design.observed, noisy, 0.0)


def gaussian_log_likelihood(model_output, data, observed, noise_sd):
    """The Gaussian log likelihood, summed over the last (output) axis.

    ``model_output`` is ``(..., n_outputs)``; ``data``, ``observed`` and
    ``noise_sd`` broadcast against it: ``(n_sites, n_outputs)`` for every
    site at once, ``(n_outputs,)`` for one site. Unobserved outputs
    contribute nothing.
    """
    standardized_residual = (data - model_output) / noise_sd
    log_density = -0.5 * standardized_residual**2 - jnp.log(noise_sd) - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(jnp.where(observed, log_density, 0.0), axis=-1)

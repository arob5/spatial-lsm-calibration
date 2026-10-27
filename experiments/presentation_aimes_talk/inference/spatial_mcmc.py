"""The exact spatial posterior, sampled by NUTS: the reference.

The posterior of every training site's parameters jointly, under the
spatial prior:

    p(theta(X) | y)  proportional to  prod_s p(y_s | theta_s)  x  prod_p N(theta_p(X); m_p(X), K_p).

NUTS needs the gradient of the log density, which the toy model has (it is
written in JAX) and SIPNET does not. So this is the yardstick the scalable
method is measured against, not a method for SIPNET.

The sampler works on whitened coordinates: ``theta_p(X) = m_p(X) + L_p
xi_p`` with ``L_p`` the Cholesky factor of ``K_p`` and ``xi_p`` standard
normal under the prior. The prior is then isotropic and NUTS's adaptation has
far less to learn; the posterior is the same.

Notes
-----
**Cost.** Each NUTS leapfrog step evaluates the model and its gradient at
every site at once, so the total number of leapfrog steps over warmup and
sampling is the number of model runs (with gradients) each site needed.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from scipy.linalg import cholesky

from ..toy.domain import Drivers
from ..toy.forward_model import CarbonModel
from ..toy.observations import ObservationDesign, gaussian_log_likelihood
from ..toy.spatial_prior import SpatialPrior
from .moments import PosteriorMoments

__all__ = ["SpatialMCMCResult", "sample_spatial_posterior"]


@dataclass(frozen=True)
class SpatialMCMCResult:
    """NUTS's draws and what they cost."""

    #: ``(n_draws, n_sites, n_parameters)``: every chain's draws, pooled.
    samples: np.ndarray
    moments: PosteriorMoments
    #: Leapfrog steps over warmup and sampling, every chain: model runs, with
    #: gradients, at each site.
    model_runs_per_site: int
    #: The worst split R-hat and the smallest effective sample size over
    #: every site and parameter.
    max_r_hat: float
    min_effective_sample_size: float


def sample_spatial_posterior(
    model: CarbonModel,
    drivers: Drivers,
    design: ObservationDesign,
    data: np.ndarray,
    prior: SpatialPrior,
    site_locations: np.ndarray,
    *,
    seed: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
) -> SpatialMCMCResult:
    """Sample the spatial posterior at the training sites by NUTS."""
    n_sites = len(site_locations)
    prior_mean = jnp.asarray(prior.trend(site_locations).T)  # (n_parameters, n_sites)
    prior_cholesky = jnp.asarray(
        np.stack([cholesky(prior.site_covariance(p, site_locations), lower=True) for p in range(prior.n_parameters)])
    )  # (n_parameters, n_sites, n_sites)

    def theta_from_whitened(whitened):
        """(n_parameters, n_sites) whitened -> (n_sites, n_parameters) theta."""
        return (prior_mean + jnp.einsum("pij,pj->pi", prior_cholesky, whitened)).T

    def numpyro_model():
        whitened = numpyro.sample(
            "whitened", dist.Normal(0.0, 1.0).expand([prior.n_parameters, n_sites]).to_event(2)
        )
        model_output = model.simulate(theta_from_whitened(whitened), drivers.temperature, drivers.light)
        log_likelihood = gaussian_log_likelihood(model_output, data, design.observed, design.noise_sd)
        numpyro.factor("log_likelihood", jnp.sum(log_likelihood))

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    # Warm up and sample as two calls so the warmup's leapfrog steps can be
    # counted too: they are part of the cost.
    warmup_key, sampling_key = jax.random.split(jax.random.PRNGKey(seed))
    mcmc.warmup(warmup_key, extra_fields=("num_steps",), collect_warmup=True)
    warmup_steps = int(np.sum(mcmc.get_extra_fields()["num_steps"]))
    mcmc.run(sampling_key, extra_fields=("num_steps",))
    sampling_steps = int(np.sum(mcmc.get_extra_fields()["num_steps"]))

    whitened_by_chain = mcmc.get_samples(group_by_chain=True)["whitened"]  # (chains, draws, P, S)
    theta_by_chain = np.asarray(jax.vmap(jax.vmap(theta_from_whitened))(whitened_by_chain))
    diagnostics = summary(theta_by_chain, group_by_chain=True)
    samples = theta_by_chain.reshape(-1, n_sites, prior.n_parameters)
    return SpatialMCMCResult(
        samples=samples,
        moments=PosteriorMoments.from_samples(samples),
        model_runs_per_site=warmup_steps + sampling_steps,
        max_r_hat=float(np.max(diagnostics["Param:0"]["r_hat"])),
        min_effective_sample_size=float(np.min(diagnostics["Param:0"]["n_eff"])),
    )

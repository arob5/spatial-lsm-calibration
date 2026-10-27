"""Iterated posterior linearization (IPL): the scalable spatial method.

The idea: the expensive, gradient-free model enters only through **local**
ensembles of runs, one small ensemble per site, all independent; the
**global** step that shares information between sites is cheap Gaussian
algebra. Each iteration:

1. **Local runs.** At each site ``s``, draw ``J`` parameter values from the
   current approximation of that site's posterior, ``N(mu_s, V_s)`` (the
   site's prior at the first iteration), and run the model at each.

2. **Local linearization.** Regress the site's observed outputs on its
   parameters across the ``J`` runs (statistical linear regression):

       G_s(theta) ~ A_s theta + b_s + delta_s,   delta_s ~ N(0, Omega_s),

   where ``A_s, b_s`` are the least-squares fit and ``Omega_s`` the
   covariance of its residuals, the error of the straight-line fit. The
   site's likelihood then becomes Gaussian in ``theta_s``,

       y_s ~ N(A_s theta_s + b_s, Sigma_s + Omega_s),

   a **site factor** ``exp(-theta_s' Lambda_s theta_s / 2 + h_s' theta_s)``
   with ``Lambda_s = A_s' R_s^{-1} A_s`` and ``h_s = A_s' R_s^{-1} (y_s -
   b_s)``, ``R_s = Sigma_s + Omega_s``.

3. **Global step.** Combine every site factor with the spatial prior
   ``N(m, K)``. Everything is Gaussian, so the posterior over all sites is
   exact for the linearized problem:

       precision = K^{-1} + Lambda,   mean = precision^{-1} (K^{-1} m + h),

   with ``Lambda`` block diagonal over sites. This is where a tower's data
   reaches its neighbors.

The new posterior's marginal at each site is the next iteration's sampling
distribution, so the linearization moves to where the posterior is. The
iteration stops once no site's posterior mean moves by more than a
tolerance, in units of that parameter's prior SD.

Notes
-----
**Common random numbers.** The standard normal draws behind each site's
ensemble are drawn once and reused at every iteration: the runs move with
the approximation, but the iteration is a deterministic map, so it settles
rather than jittering by Monte Carlo noise.

**Cost.** ``J`` model runs per site per iteration, no gradients, every run
independent of every other (parallel across sites and ensemble members).

**The global step's algebra.** With ``L`` the Cholesky factor of ``K``
(block diagonal over parameters, since the prior is independent across
them), substituting ``theta = m + L xi`` gives

    precision of xi = I + L' Lambda L =: M,   mean of xi = M^{-1} L' (h - Lambda m),

so ``mean = m + L M^{-1} L' (h - Lambda m)`` and ``covariance = L M^{-1}
L'``, which never inverts ``K`` itself.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, cholesky, solve_triangular

from ..toy.domain import Drivers
from ..toy.forward_model import CarbonModel
from ..toy.observations import ObservationDesign
from ..toy.spatial_prior import SpatialPrior
from .moments import PosteriorMoments, parameter_major_index

__all__ = ["IPLResult", "SiteFactors", "run_ipl"]


@dataclass(frozen=True)
class SiteFactors:
    """Each site's Gaussian likelihood factor in its own parameters."""

    #: ``(n_sites, n_parameters, n_parameters)``.
    precision: np.ndarray
    #: ``(n_sites, n_parameters)``.
    shift: np.ndarray


@dataclass(frozen=True)
class IPLResult:
    """The approximation after every iteration."""

    #: ``history[i]`` is the posterior after iteration ``i + 1``.
    history: list[PosteriorMoments]
    #: ``runs_per_site[i]``: the model runs each site had used after
    #: iteration ``i + 1``.
    runs_per_site: np.ndarray

    @property
    def final(self) -> PosteriorMoments:
        return self.history[-1]


def run_ipl(
    model: CarbonModel,
    drivers: Drivers,
    design: ObservationDesign,
    data: np.ndarray,
    prior: SpatialPrior,
    site_locations: np.ndarray,
    *,
    ensemble_size: int,
    max_iterations: int,
    convergence_tolerance: float,
    seed: int,
) -> IPLResult:
    """Run IPL from the prior until it converges, or for ``max_iterations``.

    Converged means the largest change in any site's posterior mean over
    the last iteration is below ``convergence_tolerance`` prior SDs.
    """
    n_sites, n_parameters = len(site_locations), prior.n_parameters
    prior_mean = prior.trend(site_locations)  # (n_sites, n_parameters)
    prior_cholesky = block_diag(
        *[cholesky(prior.site_covariance(p, site_locations), lower=True) for p in range(n_parameters)]
    )  # parameter-major

    # Common random numbers: one set of standard normal draws, reused at
    # every iteration (module Notes).
    standard_normal_draws = np.random.default_rng(seed).standard_normal((n_sites, ensemble_size, n_parameters))

    # The first iteration samples each site's prior marginal.
    sampling_mean = prior_mean
    sampling_covariance = np.repeat(np.diag(prior.site_marginal_sd() ** 2)[None], n_sites, axis=0)

    prior_sd = prior.site_marginal_sd()
    history = []
    for _ in range(max_iterations):
        ensemble = _site_ensembles(sampling_mean, sampling_covariance, standard_normal_draws)
        model_output = np.asarray(
            model.simulate(ensemble, drivers.temperature[:, None, :], drivers.light[:, None, :])
        )  # (n_sites, ensemble_size, n_outputs)
        factors = _linearized_site_factors(ensemble, model_output, data, design)
        posterior = _global_gaussian_update(prior_mean, prior_cholesky, factors)
        history.append(posterior)
        largest_change = np.max(np.abs(posterior.mean - sampling_mean) / prior_sd)
        sampling_mean, sampling_covariance = posterior.mean, posterior.site_covariances()
        if largest_change < convergence_tolerance:
            break

    runs_per_site = ensemble_size * np.arange(1, len(history) + 1)
    return IPLResult(history=history, runs_per_site=runs_per_site)


# ── the three steps ─────────────────────────────────────────────────────────


def _site_ensembles(mean: np.ndarray, covariance: np.ndarray, standard_normal_draws: np.ndarray) -> np.ndarray:
    """Step 1: ``(n_sites, J, n_parameters)`` draws, ``mean_s + chol(V_s) z``."""
    lower = np.linalg.cholesky(covariance)  # (n_sites, P, P)
    return mean[:, None, :] + np.einsum("spq,sjq->sjp", lower, standard_normal_draws)


def _linearized_site_factors(
    ensemble: np.ndarray,
    model_output: np.ndarray,
    data: np.ndarray,
    design: ObservationDesign,
) -> SiteFactors:
    """Step 2: each site's Gaussian factor from its ensemble (module docstring)."""
    n_sites, ensemble_size, n_parameters = ensemble.shape
    precision = np.empty((n_sites, n_parameters, n_parameters))
    shift = np.empty((n_sites, n_parameters))
    for site in range(n_sites):
        observed = design.observed[site]
        outputs = model_output[site][:, observed]  # (J, N_s)
        slope, intercept, residual_covariance = _statistical_linear_regression(ensemble[site], outputs)
        noise_covariance = np.diag(design.noise_sd[site, observed] ** 2) + residual_covariance
        # R^{-1} [A, y - b] in one solve.
        solved = cho_solve(
            cho_factor(noise_covariance, lower=True),
            np.column_stack([slope, data[site, observed] - intercept]),
        )
        precision[site] = slope.T @ solved[:, :n_parameters]
        shift[site] = slope.T @ solved[:, n_parameters]
    # Symmetrize away rounding.
    precision = 0.5 * (precision + precision.transpose(0, 2, 1))
    return SiteFactors(precision=precision, shift=shift)


def _global_gaussian_update(prior_mean: np.ndarray, prior_cholesky: np.ndarray, factors: SiteFactors) -> PosteriorMoments:
    """Step 3: the Gaussian posterior over all sites (module Notes give the algebra)."""
    n_sites, n_parameters = prior_mean.shape
    dimension = n_sites * n_parameters

    # Place each site's factor in the parameter-major vector.
    factor_precision = np.zeros((dimension, dimension))
    factor_shift = np.zeros(dimension)
    sites = np.arange(n_sites)
    for p in range(n_parameters):
        rows = parameter_major_index(p, sites, n_sites)
        factor_shift[rows] = factors.shift[:, p]
        for q in range(n_parameters):
            factor_precision[rows, parameter_major_index(q, sites, n_sites)] = factors.precision[:, p, q]

    mean_vector = prior_mean.T.ravel()  # parameter-major
    whitened_precision = np.eye(dimension) + prior_cholesky.T @ factor_precision @ prior_cholesky
    whitened_precision_cholesky = cholesky(whitened_precision, lower=True)
    whitened_mean = cho_solve(
        (whitened_precision_cholesky, True),
        prior_cholesky.T @ (factor_shift - factor_precision @ mean_vector),
    )
    posterior_mean = mean_vector + prior_cholesky @ whitened_mean
    # covariance = L M^{-1} L' = W' W with W = chol(M)^{-1} L'.
    half = solve_triangular(whitened_precision_cholesky, prior_cholesky.T, lower=True)
    posterior_covariance = half.T @ half
    return PosteriorMoments(mean=posterior_mean.reshape(n_parameters, n_sites).T, covariance=posterior_covariance)


# ── helpers ─────────────────────────────────────────────────────────────────


def _statistical_linear_regression(parameters: np.ndarray, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit ``outputs ~ A parameters + b`` across an ensemble.

    ``parameters`` is ``(J, P)`` and ``outputs`` ``(J, N)``. Returns the slope
    ``A`` ``(N, P)``, the intercept ``b`` ``(N,)`` and the residual covariance
    ``Omega`` ``(N, N)``, divided by ``J - P - 1``, the residuals' degrees of
    freedom.
    """
    ensemble_size, n_parameters = parameters.shape
    parameter_mean, output_mean = parameters.mean(axis=0), outputs.mean(axis=0)
    centered_parameters = parameters - parameter_mean
    centered_outputs = outputs - output_mean
    # Normal equations: A' = (dtheta' dtheta)^{-1} dtheta' dG.
    slope = np.linalg.solve(centered_parameters.T @ centered_parameters, centered_parameters.T @ centered_outputs).T
    intercept = output_mean - slope @ parameter_mean
    residuals = outputs - parameters @ slope.T - intercept
    residual_covariance = residuals.T @ residuals / (ensemble_size - n_parameters - 1)
    return slope, intercept, residual_covariance

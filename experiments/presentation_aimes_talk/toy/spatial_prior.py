"""The spatial prior: each parameter is a field over the region.

For parameter ``p`` (log productivity, log turnover) at location ``x``:

    theta_p(x) = beta_p0 + beta_p1 z(x)     the climate trend, m_p(x)
               + sigma_p u_p(x)             smooth variation
               + tau_p eta_p(x)             site-to-site noise

where ``z`` is the climate covariate (``domain.climate_covariate``), ``u_p``
is a Gaussian process with unit variance and Matern-3/2 correlation of
length scale ``ell_p``, and ``eta_p`` is independent standard normal at each
site. The parameters are independent of one another a priori.

We call ``f_p(x) = m_p(x) + sigma_p u_p(x)`` the **smooth field**: it is
what the maps show. A site's parameter value is the smooth field plus its own
noise, so at sites ``X`` the prior is

    theta_p(X) ~ N(m_p(X), K_p),   K_p = sigma_p^2 R_p(X, X) + tau_p^2 I.

Every number in the prior (``beta``, ``sigma``, ``ell``, ``tau``) is treated
as known. A real calibration would estimate them too.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.spatial.distance import cdist

from .domain import climate_covariate

__all__ = ["FieldDraw", "SpatialPrior", "matern32_correlation"]

#: Added to a covariance's diagonal before a Cholesky factorization, for
#: numerical stability only (the smooth field's covariance at a dense map grid
#: is nearly singular). Small next to every variance in the problem.
CHOLESKY_JITTER = 1e-8


def matern32_correlation(locations_a: np.ndarray, locations_b: np.ndarray, length_scale: float) -> np.ndarray:
    """Matern-3/2 correlation, ``(1 + r) exp(-r)`` with ``r = sqrt(3) d / length_scale``."""
    scaled_distance = np.sqrt(3.0) * cdist(locations_a, locations_b) / length_scale
    return (1.0 + scaled_distance) * np.exp(-scaled_distance)


@dataclass(frozen=True)
class FieldDraw:
    """One draw of every parameter field, split into the prior's three terms.

    Each array is ``(n_locations, n_parameters)``. At map points the site
    noise is zero, since it belongs to sites only; at sites the value is
    ``trend + smooth_variation + site_noise``.
    """

    trend: np.ndarray
    smooth_variation: np.ndarray
    site_noise: np.ndarray

    @property
    def smooth_field(self) -> np.ndarray:
        return self.trend + self.smooth_variation

    @property
    def value(self) -> np.ndarray:
        return self.trend + self.smooth_variation + self.site_noise


@dataclass(frozen=True)
class SpatialPrior:
    """The prior's numbers, one entry per parameter (see the module docstring)."""

    #: ``(n_parameters, 2)``: the trend's intercept and slope on ``z``.
    beta: np.ndarray
    field_sd: np.ndarray
    length_scale: np.ndarray
    site_noise_sd: np.ndarray

    @property
    def n_parameters(self) -> int:
        return len(self.field_sd)

    # ── the prior at given locations ──

    def trend(self, locations: np.ndarray) -> np.ndarray:
        """The climate trend ``m(x)``, ``(n_locations, n_parameters)``."""
        z = climate_covariate(locations)
        return self.beta[:, 0][None, :] + z[:, None] * self.beta[:, 1][None, :]

    def smooth_field_covariance(self, parameter: int, locations_a: np.ndarray, locations_b: np.ndarray) -> np.ndarray:
        """``cov(f_p(a), f_p(b)) = sigma_p^2 R_p(a, b)``."""
        return self.field_sd[parameter] ** 2 * matern32_correlation(
            locations_a, locations_b, self.length_scale[parameter]
        )

    def site_covariance(self, parameter: int, site_locations: np.ndarray) -> np.ndarray:
        """``K_p``, the prior covariance of ``theta_p`` at the sites."""
        n_sites = len(site_locations)
        return self.smooth_field_covariance(parameter, site_locations, site_locations) + (
            self.site_noise_sd[parameter] ** 2 * np.eye(n_sites)
        )

    def site_marginal_sd(self) -> np.ndarray:
        """The prior SD of one site's parameters, ``sqrt(sigma^2 + tau^2)``."""
        return np.sqrt(self.field_sd**2 + self.site_noise_sd**2)

    # ── drawing ──

    def draw(self, rng: np.random.Generator, site_locations: np.ndarray, map_locations: np.ndarray) -> tuple[FieldDraw, FieldDraw]:
        """Draw every field jointly at the sites and on the map.

        Returns the draw at the sites and the draw at the map points, which
        share one smooth field, so the map shows the field the site values
        were drawn from.
        """
        all_locations = np.vstack([site_locations, map_locations])
        n_sites, n_all = len(site_locations), len(all_locations)

        smooth_variation = np.empty((n_all, self.n_parameters))
        for p in range(self.n_parameters):
            covariance = self.smooth_field_covariance(p, all_locations, all_locations)
            lower = cholesky(covariance + CHOLESKY_JITTER * np.eye(n_all), lower=True)
            smooth_variation[:, p] = lower @ rng.standard_normal(n_all)

        site_noise = self.site_noise_sd[None, :] * rng.standard_normal((n_sites, self.n_parameters))
        trend = self.trend(all_locations)
        at_sites = FieldDraw(
            trend=trend[:n_sites],
            smooth_variation=smooth_variation[:n_sites],
            site_noise=site_noise,
        )
        on_map = FieldDraw(
            trend=trend[n_sites:],
            smooth_variation=smooth_variation[n_sites:],
            site_noise=np.zeros((n_all - n_sites, self.n_parameters)),
        )
        return at_sites, on_map

    # ── prediction away from the sites ──

    def predict(
        self,
        parameter: int,
        site_locations: np.ndarray,
        site_mean: np.ndarray,
        site_covariance: np.ndarray,
        target_locations: np.ndarray,
        *,
        include_site_noise: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD of ``theta_p`` at new locations, given a posterior at the sites.

        Parameters
        ----------
        parameter:
            Which parameter, ``p``.
        site_locations:
            The sites ``X``, ``(n_sites, 2)``.
        site_mean, site_covariance:
            The posterior mean ``(n_sites,)`` and covariance
            ``(n_sites, n_sites)`` of ``theta_p(X)``, from any method.
        target_locations:
            Where to predict, ``(n_targets, 2)``.
        include_site_noise:
            ``False`` predicts the smooth field ``f_p`` (the maps);
            ``True`` predicts the parameter at a new *site*, which carries
            its own noise ``tau_p`` too (the held-out sites).

        Returns
        -------
        mean, sd:
            Each ``(n_targets,)``.

        Notes
        -----
        Given the site values, the field at a target ``g`` is Gaussian
        (kriging): with ``W = k_gX K_p^{-1}``,

            E[f_p(g) | theta_p(X)]   = m_p(g) + W (theta_p(X) - m_p(X))
            var[f_p(g) | theta_p(X)] = sigma_p^2 - W k_Xg.

        The mean is linear in the site values, so averaging over a posterior
        ``theta_p(X) ~ (mu, V)`` needs only its mean and covariance (the law
        of total variance), whether ``(mu, V)`` is a Gaussian approximation
        or the moments of MCMC samples:

            mean = m_p(g) + W (mu - m_p(X)),   var = sigma_p^2 - W k_Xg + W V W^T.
        """
        prior_covariance = self.site_covariance(parameter, site_locations)
        cross_covariance = self.smooth_field_covariance(parameter, target_locations, site_locations)
        weights = cho_solve(cho_factor(prior_covariance, lower=True), cross_covariance.T).T

        trend_at_targets = self.trend(target_locations)[:, parameter]
        trend_at_sites = self.trend(site_locations)[:, parameter]
        mean = trend_at_targets + weights @ (site_mean - trend_at_sites)

        kriging_variance = self.field_sd[parameter] ** 2 - np.einsum("ts,ts->t", weights, cross_covariance)
        propagated_variance = np.einsum("ts,su,tu->t", weights, site_covariance, weights)
        variance = kriging_variance + propagated_variance
        if include_site_noise:
            variance = variance + self.site_noise_sd[parameter] ** 2
        return mean, np.sqrt(np.maximum(variance, 0.0))

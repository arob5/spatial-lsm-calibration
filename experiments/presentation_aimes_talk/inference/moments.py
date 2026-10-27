"""``PosteriorMoments``: the mean and covariance of a posterior over the sites.

Every method's result is reduced to this form: the posterior mean and the
joint covariance of all training sites' parameters. The maps and held-out
predictions (``evaluation``) need nothing more, since prediction away from
the sites is linear in the site values (``SpatialPrior.predict``).

The covariance is stored **parameter-major**: entry ``p * n_sites + s`` is
parameter ``p`` at site ``s``, so one parameter's block over the sites is
contiguous.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["PosteriorMoments", "parameter_major_index"]


def parameter_major_index(parameter: int, site: int | np.ndarray, n_sites: int) -> int | np.ndarray:
    """The position of ``(parameter, site)`` in the parameter-major vector."""
    return parameter * n_sites + site


@dataclass(frozen=True)
class PosteriorMoments:
    """Posterior mean ``(n_sites, n_parameters)`` and parameter-major covariance."""

    mean: np.ndarray
    covariance: np.ndarray

    @property
    def n_sites(self) -> int:
        return self.mean.shape[0]

    @property
    def n_parameters(self) -> int:
        return self.mean.shape[1]

    # ── construction ──

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "PosteriorMoments":
        """Moments of samples ``(n_samples, n_sites, n_parameters)``."""
        n_samples = samples.shape[0]
        # (n_samples, n_sites, n_parameters) -> (n_samples, n_parameters * n_sites), parameter-major.
        flat = samples.transpose(0, 2, 1).reshape(n_samples, -1)
        return cls(mean=samples.mean(axis=0), covariance=np.cov(flat, rowvar=False))

    @classmethod
    def independent_sites(cls, mean: np.ndarray, site_covariances: np.ndarray) -> "PosteriorMoments":
        """Moments of a posterior under which sites are independent.

        ``site_covariances`` is ``(n_sites, n_parameters, n_parameters)``.
        """
        n_sites, n_parameters = mean.shape
        covariance = np.zeros((n_parameters * n_sites, n_parameters * n_sites))
        sites = np.arange(n_sites)
        for p in range(n_parameters):
            for q in range(n_parameters):
                covariance[
                    parameter_major_index(p, sites, n_sites),
                    parameter_major_index(q, sites, n_sites),
                ] = site_covariances[:, p, q]
        return cls(mean=mean, covariance=covariance)

    # ── views ──

    def parameter_covariance(self, parameter: int) -> np.ndarray:
        """One parameter's covariance over the sites, ``(n_sites, n_sites)``."""
        block = slice(parameter * self.n_sites, (parameter + 1) * self.n_sites)
        return self.covariance[block, block]

    def site_covariances(self) -> np.ndarray:
        """Each site's own covariance, ``(n_sites, n_parameters, n_parameters)``."""
        sites = np.arange(self.n_sites)
        positions = np.stack(
            [parameter_major_index(p, sites, self.n_sites) for p in range(self.n_parameters)], axis=1
        )  # (n_sites, n_parameters)
        return self.covariance[positions[:, :, None], positions[:, None, :]]

    @property
    def sd(self) -> np.ndarray:
        """Marginal posterior SDs, ``(n_sites, n_parameters)``."""
        return np.sqrt(np.einsum("spp->sp", self.site_covariances()))

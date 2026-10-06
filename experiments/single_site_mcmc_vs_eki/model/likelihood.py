"""The likelihood of a model, from predictions: Gaussian at given scales, or
with the scales integrated out, and the scales' conditional draws.

Every algorithm here evaluates the forward map once per batch and then needs
the likelihood under several noise treatments; this module computes them
from the predictions alone, with no SIPNET run. For source :math:`k` with
:math:`n_k` observations, residual :math:`r_k = y_k - \\mathcal G_k(\\theta)`
and :math:`q_k = r_k^\\top C_k^{-1} r_k`:

.. math::

    \\log \\mathcal N(y_k; \\mathcal G_k, s_k C_k)
        = -\\tfrac12\\big(q_k/s_k + n_k \\log(2\\pi s_k) + \\log|C_k|\\big),

and, for a scaled source with :math:`s_k \\sim \\mathrm{IG}(a, b)` integrated
out, the multivariate Student-t :math:`t_{2a}(\\mathcal G_k, (b/a) C_k)`,

.. math::

    \\log p(y_k \\mid \\theta) = \\log\\Gamma(a + \\tfrac{n_k}{2}) - \\log\\Gamma(a)
        + a\\log b - \\tfrac{n_k}{2}\\log 2\\pi - \\tfrac12 \\log|C_k|
        - (a + \\tfrac{n_k}{2}) \\log\\big(b + \\tfrac{q_k}{2}\\big).

The tempered conditional of a scale is
:math:`s_k \\mid \\theta, y \\sim \\mathrm{IG}(a + \\varphi n_k/2,\\ b + \\varphi q_k/2)`.
A sample whose predictions hold a ``NaN`` (a failed run) has likelihood
:math:`-\\infty`.

:class:`NoiseModel` reads :math:`y` and every :math:`C_k` from a model's
fixed posterior (``models.fixed_posterior``), whose :math:`R` is
:math:`\\operatorname{diag}(C_k)`; the tests check its log likelihoods against
the posteriors' own.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

from sipnet_calibration.probability import Posterior

from .. import config
from . import noise

__all__ = ["NoiseModel", "SourceNoise"]


@dataclass(frozen=True)
class SourceNoise:
    """One source's :math:`C_k`: its entries of y, Cholesky factor and log determinant."""

    name: str
    positions: slice
    cholesky: np.ndarray
    log_determinant: float
    scaled: bool

    @property
    def size(self) -> int:
        """:math:`n_k`."""
        return self.cholesky.shape[0]


class NoiseModel:
    """The likelihood of a model's observations from its predictions.

    Parameters
    ----------
    posterior : Posterior
        A model's fixed posterior, every scale held at 1.
    inferred : bool
        Whether the scales of ``config.NOISE_SCALED_SOURCES`` are unknown:
        :meth:`log_likelihood` is then the scales' marginal.
    """

    def __init__(self, posterior: Posterior, *, inferred: bool) -> None:
        likelihood = posterior.gaussian_likelihood()
        self.y = np.asarray(likelihood.y)
        self.inferred = inferred
        self.shape = config.NOISE_SCALE_SHAPE
        self.prior_scale = noise.noise_scale_prior_scale()
        sources, start = [], 0
        for name, block in zip(
            posterior.observations.component_names,
            likelihood.noise_covariance.blocks,
            strict=True,
        ):
            cholesky = np.linalg.cholesky(np.asarray(block.to_dense()))
            size = cholesky.shape[0]
            sources.append(
                SourceNoise(
                    name=name,
                    positions=slice(start, start + size),
                    cholesky=cholesky,
                    log_determinant=2.0 * float(np.log(np.diag(cholesky)).sum()),
                    scaled=name in config.NOISE_SCALED_SOURCES,
                )
            )
            start += size
        self.sources = tuple(sources)

    @property
    def scaled_names(self) -> tuple[str, ...]:
        """The sources with a noise scale, in y's order."""
        return tuple(source.name for source in self.sources if source.scaled)

    def quadratic_forms(self, predictions) -> dict[str, np.ndarray]:
        """:math:`q_k = r_k^\\top C_k^{-1} r_k` per source, ``(J,)``, ``NaN`` where a run failed."""
        residuals = self.y - np.atleast_2d(np.asarray(predictions))
        forms = {}
        for source in self.sources:
            whitened = scipy.linalg.solve_triangular(
                source.cholesky, residuals[:, source.positions].T, lower=True, check_finite=False
            )
            forms[source.name] = np.sum(whitened**2, axis=0)
        return forms

    def log_likelihood_at(
        self, predictions, scales: Mapping[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """The Gaussian log likelihood at *scales*, ``{source: (J,)}``, 1 where
        not given; ``(J,)``, :math:`-\\infty` where a run failed."""
        forms = self.quadratic_forms(predictions)
        total = 0.0
        for source in self.sources:
            scale = 1.0 if scales is None else scales.get(source.name, 1.0)
            total = total - 0.5 * (
                forms[source.name] / scale
                + source.size * np.log(2.0 * np.pi * scale)
                + source.log_determinant
            )
        return _failed_as_minus_infinity(total)

    def log_marginal_likelihood(self, predictions) -> np.ndarray:
        """The log likelihood with every scale integrated out over its prior, ``(J,)``."""
        forms = self.quadratic_forms(predictions)
        a, b = self.shape, self.prior_scale
        total = 0.0
        for source in self.sources:
            n, q = source.size, forms[source.name]
            if source.scaled:
                total = total + (
                    scipy.special.gammaln(a + n / 2)
                    - scipy.special.gammaln(a)
                    + a * np.log(b)
                    - n / 2 * np.log(2.0 * np.pi)
                    - source.log_determinant / 2
                    - (a + n / 2) * np.log(b + q / 2)
                )
            else:
                total = total - 0.5 * (q + n * np.log(2.0 * np.pi) + source.log_determinant)
        return _failed_as_minus_infinity(total)

    def log_likelihood(self, predictions) -> np.ndarray:
        """The model's likelihood of theta: the marginal if the scales are inferred,
        else the Gaussian at scale 1."""
        if self.inferred:
            return self.log_marginal_likelihood(predictions)
        return self.log_likelihood_at(predictions)

    def scale_conditional(self, forms: Mapping[str, np.ndarray], phi: float = 1.0):
        """``{source: (shape, scale)}`` of each scale's tempered conditional,
        :math:`\\mathrm{IG}(a + \\varphi n_k/2, b + \\varphi q_k/2)`, from *forms*."""
        return {
            source.name: (
                self.shape + phi * source.size / 2,
                self.prior_scale + phi * np.asarray(forms[source.name]) / 2,
            )
            for source in self.sources
            if source.scaled
        }

    def draw_scales(
        self, rng: np.random.Generator, forms: Mapping[str, np.ndarray], phi: float = 1.0
    ) -> dict[str, np.ndarray]:
        """One draw of each scale from its tempered conditional, per sample."""
        return {
            name: scipy.stats.invgamma(shape, scale=scale).rvs(random_state=rng)
            for name, (shape, scale) in self.scale_conditional(forms, phi).items()
        }

    def draw_prior_scales(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """*n* draws of each scale's prior."""
        return {
            name: scipy.stats.invgamma(self.shape, scale=self.prior_scale).rvs(
                size=n, random_state=rng
            )
            for name in self.scaled_names
        }


def _failed_as_minus_infinity(values) -> np.ndarray:
    """*values* with ``NaN`` (a failed run) replaced by :math:`-\\infty`."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isnan(values), -np.inf, values)

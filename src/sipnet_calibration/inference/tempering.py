"""Tempered SMC and importance sampling: a posterior as the
:class:`~sipnet_calibration.smc.TemperingProblem` :mod:`~sipnet_calibration.smc`
reads.

:mod:`~sipnet_calibration.smc` reads three functions of theta: the
normalized log prior :math:`\\log \\pi_\\theta`, the normalized log
likelihood :math:`\\log L`, ``NaN`` for a failed run, and a normalized base
density :math:`q` to draw from. A posterior gives the first two, each a
density in theta, so the evidence it estimates is :math:`Z = \\int
\\pi_\\theta L \\, d\\theta`. The observed factors with no target ancestor
contribute only a constant :math:`C`, whose log is ``posterior.log_constant``,
so the evidence of every observed value is :math:`C Z`. The base is the prior
unless another is given, such as a Student-t fitted to an EKI ensemble.

The log likelihood is one :meth:`~sipnet_calibration.probability.Posterior.evaluate`
per batch: ``NaN`` where a simulator output it reads was not computed
(``simulator_valid`` false), which :mod:`~sipnet_calibration.smc` counts as
a failed run and scores :math:`-\\infty`, and :math:`-\\infty` where only
the traced part failed. When the likelihood is Gaussian, each sample's
predictions :math:`G(\\theta)`, ``(n, N)`` in y's order, travel with it as
its auxiliary values; otherwise the log likelihood carries none.

Usage
-----
::

    from sipnet_calibration import smc

    base = smc.fit_student_t(eki_ensemble, degrees_of_freedom=5.0)
    problem = tempering_problem(posterior, base=base)
    state = smc.initial_state(problem, smc.SMCSettings(n_samples=400, seed=1))
    for state in smc.run_smc(problem, state):
        ...
    state.log_evidence, state.auxiliary          # log Z, each sample's G(theta)
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sipnet_calibration.inference._validation import check_posterior_is_a_posterior
from sipnet_calibration.probability import GaussianLikelihood, Posterior
from sipnet_calibration.smc import BaseDensity, LikelihoodEvaluation, TemperingProblem

__all__ = [
    "PriorBaseDensity",
    "check_base_has_an_integer_dimension",
    "check_base_has_the_posteriors_dimension",
    "tempering_problem",
]

Array = jax.Array


def tempering_problem(posterior: Posterior, *, base: BaseDensity | None = None) -> TemperingProblem:
    """``TemperingProblem(log_prior=posterior.log_prior, log_likelihood=...,
    base=base)``, the base :class:`PriorBaseDensity` when *base* is ``None``.

    The log likelihood, ``(n, D) -> (n,)``, is ``NaN`` where a simulator
    output it reads was not computed and :math:`-\\infty` where only the
    traced part failed; with a Gaussian likelihood it is an
    ``smc.LikelihoodEvaluation`` whose auxiliary values are the predictions,
    ``(n, N)``, ``NaN`` in an invalid sample.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``, or *base* has no integer
        ``dimension``.
    ValueError
        If *base*'s dimension is not the posterior's ``D``.
    """
    check_posterior_is_a_posterior(posterior)
    base = PriorBaseDensity(posterior) if base is None else base
    check_base_has_an_integer_dimension(base)
    check_base_has_the_posteriors_dimension(base, posterior.dimension)
    return TemperingProblem(
        log_prior=posterior.log_prior,
        log_likelihood=_TemperedLogLikelihood(posterior, _gaussian_likelihood_or_none(posterior)),
        base=base,
    )


@dataclass(frozen=True, eq=False)
class PriorBaseDensity:
    """The posterior's prior :math:`\\pi_\\theta` as an ``smc.BaseDensity``:
    normalized, in theta, drawn with no simulator.

    Parameters
    ----------
    posterior : Posterior

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``.
    """

    posterior: Posterior

    def __post_init__(self) -> None:
        check_posterior_is_a_posterior(self.posterior)

    @property
    def dimension(self) -> int:
        """:math:`D`."""
        return self.posterior.dimension

    def log_prob(self, theta: Any) -> Array:
        """``posterior.log_prior(theta)``, ``(n, D) -> (n,)``."""
        return self.posterior.log_prior(theta)

    def sample(self, rng: np.random.Generator, n_samples: int) -> Array:
        """*n_samples* draws of the prior, ``(n_samples, D)``, keyed by a JAX
        key seeded from one integer *rng* draws.

        Raises
        ------
        TypeError, ValueError
            As :meth:`Posterior.sample_prior`.
        """
        key = jax.random.key(int(rng.integers(np.iinfo(np.int64).max)))
        return self.posterior.sample_prior(key, n_samples)


# ── helpers ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _TemperedLogLikelihood:
    """The posterior's log likelihood as :mod:`smc` reads it, from one
    evaluation per batch."""

    posterior: Posterior
    likelihood: GaussianLikelihood | None

    def __call__(self, theta: np.ndarray) -> Any:
        if self.likelihood is None:
            evaluation = self.posterior.evaluate(theta)
            return _failed_runs_as_nan(evaluation.log_likelihood, evaluation.simulator_valid)
        predictions, _, evaluation = self.likelihood.forward(theta)
        return LikelihoodEvaluation(
            log_likelihood=_failed_runs_as_nan(evaluation.log_likelihood, evaluation.simulator_valid),
            auxiliary=predictions,
        )


def _failed_runs_as_nan(log_likelihood: Array, simulator_valid: Array) -> Array:
    """``NaN`` where a simulator output was not computed, *log_likelihood*
    (``-inf`` where the traced part failed) elsewhere."""
    return jnp.where(simulator_valid, log_likelihood, jnp.nan)


def _gaussian_likelihood_or_none(posterior: Posterior) -> GaussianLikelihood | None:
    """The posterior's Gaussian likelihood, or ``None`` when it has none."""
    # gaussian_likelihood raises ValueError for exactly the three reasons a
    # posterior has none: nothing observed depends on theta, a factor is not
    # Gaussian, or a covariance reads a parameter.
    try:
        return posterior.gaussian_likelihood()
    except ValueError:
        return None


# ── checks ────────────────────────────────────────────────────────────────────


def check_base_has_an_integer_dimension(base: Any) -> None:
    """Check the base density has an integer ``dimension``, as ``smc.BaseDensity`` does."""
    if not hasattr(base, "dimension"):
        raise TypeError(
            f"the base density, a {type(base).__name__}, has no dimension; "
            "pass an smc.BaseDensity: dimension, log_prob(theta) and sample(rng, n_samples)."
        )
    value = base.dimension
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"the base density's dimension must be an integer, not {type(value).__name__}; "
            "pass an smc.BaseDensity."
        )


def check_base_has_the_posteriors_dimension(base: Any, dimension: int) -> None:
    """Check the base density is over theta's ``D`` entries."""
    if base.dimension != dimension:
        raise ValueError(
            f"the base density has dimension {base.dimension}, but theta has D = {dimension} entries; "
            "fit the base to theta in the posterior's layout."
        )

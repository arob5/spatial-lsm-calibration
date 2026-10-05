"""Gradient-free MCMC: a posterior's log density as a sampler calls it, and
starting points where it is finite.

A sampler such as ``emcee`` reads the unnormalized log posterior in theta,
:math:`\\log \\pi_\\theta(\\theta) + \\log L(\\theta)`, ``-inf`` where a
sample is invalid (:attr:`PosteriorEvaluation.valid`), which truncates the
posterior to where the simulators run. :func:`batched_log_density` makes one
:meth:`~sipnet_calibration.probability.Posterior.evaluate`, and so one
simulator call, per batch; :func:`log_density` is the same for a sampler
that evaluates one point at a time. :func:`initial_points` finds prior draws
at which the posterior is finite, so walkers start where SIPNET runs.

Usage
-----
::

    import emcee
    import jax

    n_walkers = 2 * posterior.dimension
    start, _ = initial_points(posterior, jax.random.key(0), n_walkers)
    sampler = emcee.EnsembleSampler(
        n_walkers, posterior.dimension, batched_log_density(posterior), vectorize=True
    )
    sampler.run_mcmc(start, 1000)

``emcee``'s ``vectorize=True`` passes half the walkers per call, so a step is
two batches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict

from sipnet_calibration.inference._validation import check_posterior_is_a_posterior, check_theta_is_shaped
from sipnet_calibration.probability import Posterior, PosteriorEvaluation
from sipnet_calibration.validation import as_bounded_integer, as_positive_integer

__all__ = [
    "batched_log_density",
    "check_enough_draws_are_finite",
    "initial_points",
    "log_density",
]

Array = jax.Array


def batched_log_density(posterior: Posterior) -> Callable[[np.ndarray], np.ndarray]:
    """The log posterior, ``(J, D) -> (J,)`` ``float64``, ``-inf`` in an
    invalid sample, from one :meth:`Posterior.evaluate` per batch.

    The function raises ``ValueError`` for a theta that is not ``(J, D)``,
    and otherwise as :meth:`Posterior.evaluate`.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``.
    """
    check_posterior_is_a_posterior(posterior)

    def batched(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        check_theta_is_shaped(theta.shape, 2, posterior.dimension, message_name="batched_log_density")
        return np.asarray(posterior.evaluate(theta).log_density)

    return batched


def log_density(posterior: Posterior) -> Callable[[np.ndarray], float]:
    """The log posterior at one sample, ``(D,) -> float``, ``-inf`` where it
    is invalid, from one :meth:`Posterior.evaluate` per call.

    The function raises ``ValueError`` for a theta that is not ``(D,)``, and
    otherwise as :meth:`Posterior.evaluate`.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``.
    """
    check_posterior_is_a_posterior(posterior)

    def one(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        check_theta_is_shaped(theta.shape, 1, posterior.dimension, message_name="log_density")
        return float(posterior.evaluate(theta).log_density[0])

    return one


def initial_points(
    posterior: Posterior, key: Array, n: int, *, max_draws: int | None = None
) -> tuple[np.ndarray, PosteriorEvaluation]:
    """The first ``n`` of ``max_draws`` prior draws at which the posterior is
    finite, and their evaluation, to start walkers where SIPNET runs.

    The draws are ``posterior.sample_prior(key, max_draws)``, evaluated in
    order in batches, each as large as the number still needed. A draw is
    kept where its sample is valid and its log density finite.

    Parameters
    ----------
    posterior : Posterior
    key : jax.Array
        A typed JAX key.
    n : int
        At least 1.
    max_draws : int, optional
        At least ``n``; ``4 n`` when ``None``.

    Returns
    -------
    (numpy.ndarray, PosteriorEvaluation)
        Theta ``(n, D)``, and the evaluation at those samples, each array's
        rows theirs and its ``simulator_records`` holding, per simulator, the
        tuple of the records of every batch evaluated, in order.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``, or *n* or *max_draws* not an
        integer.
    ValueError
        If *n* is less than 1 or *max_draws* less than *n*; or as
        :meth:`Posterior.sample_prior` and :meth:`Posterior.evaluate`.
    RuntimeError
        If fewer than ``n`` of the ``max_draws`` draws are finite.

    Notes
    -----
    Batches the size of the shortfall evaluate no draw past the ``n``-th
    finite one, and the draws chosen do not depend on the batches. A batch's
    record describes all its samples, kept or not, and so cannot be cut to
    the kept rows; hence the tuple.
    """
    check_posterior_is_a_posterior(posterior)
    n = as_positive_integer(n, message_name="n")
    max_draws = 4 * n if max_draws is None else as_bounded_integer(max_draws, minimum=n, message_name="max_draws")
    draws = np.asarray(posterior.sample_prior(key, max_draws))
    evaluations, kept = [], []
    start, found = 0, 0
    while found < n and start < max_draws:
        size = min(n - found, max_draws - start)
        evaluation = posterior.evaluate(draws[start : start + size])
        finite = np.asarray(evaluation.valid) & np.isfinite(np.asarray(evaluation.log_density))
        evaluations.append(evaluation)
        kept.append(np.flatnonzero(finite))
        found += int(finite.sum())
        start += size
    check_enough_draws_are_finite(found, n, max_draws)
    evaluation = _rows_of(evaluations, kept)
    return np.asarray(evaluation.theta), evaluation


# ── helpers ───────────────────────────────────────────────────────────────────


def _rows_of(evaluations: Sequence[PosteriorEvaluation], kept: Sequence[np.ndarray]) -> PosteriorEvaluation:
    """One evaluation of the *kept* rows of each of *evaluations*, in order."""

    def stacked(read: Callable[[PosteriorEvaluation], Array]) -> Array:
        return jnp.concatenate([read(e)[jnp.asarray(rows)] for e, rows in zip(evaluations, kept)])

    first = evaluations[0]
    return PosteriorEvaluation(
        theta=stacked(lambda e: e.theta),
        log_prior=stacked(lambda e: e.log_prior),
        log_likelihood=stacked(lambda e: e.log_likelihood),
        log_density=stacked(lambda e: e.log_density),
        valid=stacked(lambda e: e.valid),
        simulator_valid=stacked(lambda e: e.simulator_valid),
        values=frozendict({name: stacked(lambda e, name=name: e.values[name]) for name in first.values}),
        simulator_records=frozendict(
            {name: tuple(e.simulator_records[name] for e in evaluations) for name in first.simulator_records}
        ),
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_enough_draws_are_finite(found: int, n: int, max_draws: int) -> None:
    """Check *n* of the prior draws had a finite posterior density."""
    if found < n:
        raise RuntimeError(
            f"only {found} of {max_draws} prior draws have a finite posterior density, fewer than the "
            f"{n} asked for; raise max_draws, or look at where the simulator fails "
            "(posterior.evaluate(theta).simulator_valid)."
        )

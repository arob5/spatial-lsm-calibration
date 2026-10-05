"""Ensemble Kalman inversion: what EnsKit's EKI driver,
``enskit.algorithms.eki``, reads from a posterior whose likelihood is
Gaussian.

The driver reads four things: a forward map from theta to predictions,
:math:`G : (J, D) \\to (J, N)`, with a non-finite row for a failed sample;
:math:`y`; the base noise covariance :math:`R`; and an initial ensemble, an
EnsKit ``Ensemble`` whose blocks are the parameters. A posterior whose
observed factors are Gaussian with held covariances,
:math:`y \\sim \\mathcal N(G(\\theta), R)`, provides all four
(:meth:`~sipnet_calibration.probability.Posterior.gaussian_likelihood`): the
ensemble has one block,
:data:`~sipnet_calibration.probability.names.THETA`, which the driver
passes to the forward map as its one positional argument (a state holding
other blocks passes ``inputs="theta"`` to the driver); the predictions are
``float64``, as the ensemble must then be; the covariance is already one of
EnsKit's operators, so nothing is converted; and no Gaussian prior is
needed to start.

Usage
-----
::

    import jax
    from enskit import kalman
    from enskit.algorithms import eki

    problem = eki_problem(posterior)
    ensemble_key, run_key = jax.random.split(key)
    state = eki.EKIState(problem.initial_ensemble(ensemble_key, 100), key=run_key)
    result = eki.run(state, problem.forward, problem.y, problem.noise_covariance,
                     update_rule=kalman.Matheron(), schedule=eki.AdaptiveESSSchedule(ess_fraction=0.5),
                     on_failure="repair")
    problem.last_evaluation.valid      # which samples of the last ensemble evaluated ran
    posterior.to_labeled(result.ensemble["theta"])
"""

from __future__ import annotations

from typing import Any

import jax
from enskit.distribution import Ensemble

from sipnet_calibration.inference._validation import check_posterior_is_a_posterior
from sipnet_calibration.probability import GaussianLikelihood, Posterior, PosteriorEvaluation
from sipnet_calibration.probability.names import THETA
from sipnet_calibration.validation import as_bounded_integer

__all__ = [
    "EKIProblem",
    "eki_problem",
]

Array = jax.Array


def eki_problem(posterior: Posterior) -> EKIProblem:
    """The EKI problem of *posterior*.

    Raises
    ------
    TypeError
        If *posterior* is not a ``Posterior``.
    ValueError
        As :meth:`Posterior.gaussian_likelihood`: nothing observed depends on
        the parameters, an observed factor is not Gaussian, or its covariance
        reads a parameter; hold a noise hyperparameter by observing it at a
        value first.
    """
    check_posterior_is_a_posterior(posterior)
    return EKIProblem(posterior.gaussian_likelihood())


class EKIProblem:
    """What an EKI run reads, from a posterior whose likelihood is
    :math:`y \\sim \\mathcal N(G(\\theta), R)`. Made by :func:`eki_problem`.
    Compared and hashed by identity.

    Parameters
    ----------
    likelihood : GaussianLikelihood
        Positional-only.

    Attributes
    ----------
    posterior : Posterior
    likelihood : GaussianLikelihood
    y : jax.Array
        ``(N,)``.
    noise_covariance : PSDLinOp
        :math:`R`, one of EnsKit's positive-definite operators.
    last_evaluation : PosteriorEvaluation or None
        The last :meth:`forward` call's; ``None`` before the first. In a run
        of EnsKit's driver, its last step's: theta after any inflation,
        before any repair and the update. A run that ends on one of EnsKit's
        schedules does not evaluate its final ensemble, whose predictions are
        one more :meth:`forward` call, which then replaces this.
    """

    def __init__(self, likelihood: GaussianLikelihood, /) -> None:
        self._likelihood = likelihood
        self._last_evaluation: PosteriorEvaluation | None = None

    def __repr__(self) -> str:
        return f"EKIProblem(D={self.posterior.dimension}, N={self.y.shape[0]})"

    @property
    def posterior(self) -> Posterior:
        return self._likelihood.posterior

    @property
    def likelihood(self) -> GaussianLikelihood:
        return self._likelihood

    @property
    def y(self) -> Array:
        return self._likelihood.y

    @property
    def noise_covariance(self) -> Any:
        return self._likelihood.noise_covariance

    @property
    def last_evaluation(self) -> PosteriorEvaluation | None:
        return self._last_evaluation

    def forward(self, theta: Any) -> Array:
        """:math:`G(\\theta)`, ``(J, D) -> (J, N)`` in y's order, ``NaN`` in an
        invalid sample, as EnsKit's driver reads a failed sample; one
        :meth:`Posterior.evaluate`, which replaces :attr:`last_evaluation`.

        Raises
        ------
        ValueError, TypeError, Exception
            As :meth:`Posterior.evaluate`.
        """
        predictions, _, evaluation = self._likelihood.forward(theta)
        self._last_evaluation = evaluation
        return predictions

    def initial_ensemble(self, key: Array, n: int) -> Ensemble:
        """``n`` draws of the prior as an unweighted EnsKit ``Ensemble`` with
        one block, :data:`~sipnet_calibration.probability.names.THETA`
        ``(n, D)`` in ``float64``, the ensemble EnsKit's ``EKIState`` starts
        from; no simulator runs.

        Raises
        ------
        TypeError
            If *n* is not an integer.
        ValueError
            If *n* is less than 2, the smallest ensemble EKI updates; or as
            :meth:`Posterior.sample_prior`.
        """
        n = as_bounded_integer(n, minimum=2, message_name="n")
        return Ensemble({THETA: self.posterior.sample_prior(key, n)})

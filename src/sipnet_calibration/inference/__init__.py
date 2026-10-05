"""Inference adapters: a posterior as each algorithm reads it.

Where this sits
---------------
::

    probability.condition_on -> Posterior
      -> inference.eki_problem          EKI (EnsKit): G, y, R and an initial ensemble
      -> inference.tempering_problem    tempered SMC and importance sampling (smc)
      -> inference.batched_log_density  gradient-free MCMC (emcee and the like)
      -> the algorithm

The package is generic, like the probability layer: it reads a
:class:`~sipnet_calibration.probability.Posterior` and nothing of SIPNET, the
sites or the observation vector, so a second experiment uses it unchanged.
It imports the probability layer, :mod:`~sipnet_calibration.smc` and
:mod:`~sipnet_calibration.validation`, and no algorithm package: each
adapter returns the plain functions, arrays and operators its algorithm
takes, and the experiment imports the algorithm.

Every adapter that needs the likelihood evaluates theta through
:meth:`Posterior.evaluate`, one simulator call per batch; the prior's
density and draws run no simulator. What counts as a failed sample is the
posterior's: an invalid sample (``valid`` false) is a ``NaN`` row of
predictions for EKI, a ``NaN`` (a failed run) or ``-inf`` log likelihood for
SMC, and ``-inf`` for MCMC.

Modules
-------
:mod:`~sipnet_calibration.inference.eki`
    :class:`EKIProblem` and :func:`eki_problem`: the forward map, y, the noise
    covariance and an initial ensemble of a posterior whose likelihood is
    Gaussian.
:mod:`~sipnet_calibration.inference.tempering`
    :class:`PriorBaseDensity` and :func:`tempering_problem`: the posterior
    as an ``smc.TemperingProblem``.
:mod:`~sipnet_calibration.inference.mcmc`
    :func:`batched_log_density`, :func:`log_density` and
    :func:`initial_points`.
"""

from sipnet_calibration.inference.eki import EKIProblem, eki_problem
from sipnet_calibration.inference.mcmc import batched_log_density, initial_points, log_density
from sipnet_calibration.inference.tempering import PriorBaseDensity, tempering_problem

__all__ = [
    "EKIProblem",
    "PriorBaseDensity",
    "batched_log_density",
    "eki_problem",
    "initial_points",
    "log_density",
    "tempering_problem",
]

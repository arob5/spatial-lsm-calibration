"""Tempered sequential Monte Carlo from a base density to the posterior.

Where this sits
---------------
::

    a prior p_0, a batched log likelihood log L, a base density q
      -> smc.initial_state, smc.run_smc   (weighted samples of each pi_beta)
      -> samples of the posterior, and its marginal likelihood Z

The module knows nothing of SIPNET or of the package's vectors: it reads
three functions of theta and returns arrays. An experiment's adapter builds
them, the log likelihood from a forward model and the base density from an
earlier ensemble. It imports NumPy, SciPy, JAX (the arrays it returns) and,
for the Pareto diagnostic only, ArviZ; from the package, :mod:`io` and
:mod:`validation`.

What it reads
-------------
A :class:`TemperingProblem`:

- ``log_prior``: the normalized log density of the prior, ``(n, D) -> (n,)``;
- ``log_likelihood``: the normalized log likelihood, ``(n, D) -> (n,)``, the
  one expensive function: called once per batch of new points, never again
  at a point already evaluated. A NaN is a failed run. It may return a
  :class:`LikelihoodEvaluation` instead, whose auxiliary values (a forward
  model's predictions) travel with each sample;
- ``base``: a normalized :class:`BaseDensity` ``q`` that can be sampled,
  positive wherever :math:`p_0 L` is (the stages before the last are
  restricted to where it is):
  :class:`MultivariateStudentT` (the Gaussian included), a
  :class:`DefensiveMixture`, or the prior itself.

The targets
-----------
With :math:`\\ell(\\theta) = \\log p_0(\\theta) + \\log L(\\theta) -
\\log q(\\theta)`, the tempered targets are

.. math::

    \\pi_\\beta(\\theta) \\propto q(\\theta)^{1-\\beta}
        \\big(p_0(\\theta) L(\\theta)\\big)^{\\beta},
    \\qquad
    \\log \\pi_\\beta = (1 - \\beta) \\log q + \\beta (\\log p_0 + \\log L)
        = \\log q + \\beta \\ell + \\text{const},

so :math:`\\pi_0 = q` and :math:`\\pi_1` is the posterior. A failed run has
:math:`\\log L = -\\infty`, so :math:`\\pi_\\beta = 0` there for every
:math:`\\beta > 0`: the posterior is the one truncated to where the model
runs.

The algorithm
-------------
Stage 0 draws :math:`n` samples of :math:`q` with weights :math:`W_i =
1/n` and evaluates :math:`\\log L` at them, one call. Stage
:math:`t \\ge 1` then

1. **chooses the increment** :math:`\\delta_t \\le 1 - \\beta_{t-1}`, the
   largest at which the conditional effective sample size (Zhou, Johansen
   and Aston 2016)

   .. math::

       \\mathrm{CESS}(\\delta) = \\frac{\\big(\\sum_i W_i e^{\\delta\\ell_i}\\big)^2}
           {\\sum_i W_i e^{2\\delta\\ell_i}} \\in (0, 1]

   is at least ``cess_fraction`` times :math:`\\mathrm{CESS}(0^+) =
   \\sum_{i:\\,\\ell_i > -\\infty} W_i`, by bisection (:func:`next_increment`);
   in the one-step mode :math:`\\delta_1 = 1`;
2. **reweights** with the cached :math:`\\ell`, no call:
   :math:`W_i \\leftarrow W_i e^{\\delta_t \\ell_i} / \\sum_j W_j
   e^{\\delta_t \\ell_j}`, and adds the log of that sum to the log
   evidence,

   .. math::

       \\log \\hat Z = \\sum_{t \\ge 1} \\log \\sum_i W_{t-1,i}\\,
           e^{\\delta_t \\ell_i},

   an unbiased estimate (on the :math:`Z` scale) of
   :math:`Z = \\int p_0 L \\, d\\theta`, because :math:`q` and :math:`p_0` are
   normalized;
3. in the one-step mode **stops** here: that is importance sampling from
   :math:`q`, :math:`\\log W_i = \\ell_i - \\log\\sum_j e^{\\ell_j}` and
   :math:`\\log\\hat Z = \\log \\frac{1}{n}\\sum_i e^{\\ell_i}`;
4. otherwise **resamples** systematically (:func:`systematic_resample`), so
   every sample has :math:`W_i = 1/n` and a finite :math:`\\ell_i`, and
5. **moves** every sample by Metropolis-Hastings steps invariant for
   :math:`\\pi_{\\beta_t}`, one call per step, until ``move_probability`` of
   the samples have moved at least once (at least ``min_move_steps``, at
   most ``max_move_steps``). Each step, each sample proposes, with
   probability :math:`p_t`, an independent draw of
   :math:`g_t = t_\\nu(\\hat m_t, \\kappa \\hat C_t)` and otherwise a random
   walk :math:`\\theta' = \\theta + \\lambda_t \\hat L_t z`,
   :math:`z \\sim N(0, I)`, :math:`\\hat L_t \\hat L_t^\\top = \\hat C_t`,
   with :math:`\\hat m_t, \\hat C_t` the weighted mean and covariance of the
   samples before resampling. It accepts with probability
   :math:`\\min(1, \\alpha)`,

   .. math::

       \\log\\alpha = \\log\\pi_\\beta(\\theta') - \\log\\pi_\\beta(\\theta)
           + \\big[\\log g_t(\\theta) - \\log g_t(\\theta')\\big]_{\\text{independent only}},

   every term analytic but :math:`\\log L(\\theta')`, so a failed proposal
   (:math:`\\log\\pi_\\beta = -\\infty`) is rejected and no sample ever
   stands at :math:`-\\infty`.

It stops once a stage ends at :math:`\\beta = 1`, so the tempered result is
an equal-weight sample of the posterior, resampled and moved there.

Between stages the kernel adapts: the random walk's scale by
:math:`\\lambda_{t+1} = \\lambda_t \\exp(\\bar a_{\\mathrm{RW}} - a^*)`, with
:math:`\\bar a_{\\mathrm{RW}}` its mean acceptance probability and
:math:`a^*` = ``random_walk_acceptance`` (from :math:`2.38/\\sqrt{D}`); and
:math:`p_{t+1} = J_{\\mathrm{I}} / (J_{\\mathrm{I}} + J_{\\mathrm{RW}})`,
clamped to ``independent_fraction_bounds`` (from 0.5), with :math:`J` each
kernel's expected squared jump distance per proposal, :math:`\\mathbb E
[\\min(1, \\alpha) \\lVert \\hat L_t^{-1}(\\theta' - \\theta)\\rVert^2]`.

The state
---------
An :class:`SMCState` holds everything a run needs to continue, so that
:func:`save_state` after any operation and :func:`load_state` resume it
bit for bit:

- ``theta`` ``(n, D)``, and per sample ``log_likelihood``, ``log_prior``,
  ``log_base`` and ``log_weights`` (normalized) ``(n,)``, and
  ``auxiliary`` ``(n, ...)`` or ``None``; ``float64`` JAX arrays;
- ``phase``: the next operation, ``"reweight"`` or ``"move"``, or
  ``"done"``; ``stage``, the number of reweights made; ``beta``;
  ``log_evidence``; ``n_evaluations`` and ``n_calls``, the likelihood's
  points and calls so far;
- the current stage's kernel (``proposal_mean``, ``proposal_covariance``,
  ``random_walk_scale``, ``independent_fraction``) and its moves so far
  (:class:`MoveTally`);
- ``records``: one mapping per stage, :data:`RECORD_FIELD_NAMES`;
- ``settings``: the :class:`SMCSettings` it runs under.

Randomness is drawn from a generator seeded by ``(seed, stage, step)``, so
no generator state is kept.

Functions
---------
:func:`initial_state`
    Stage 0: the draws of ``q`` and their log likelihood.
:func:`run_smc`
    Continue a state to :math:`\\beta = 1`, yielding it after every
    operation, each a point to checkpoint at.
:func:`save_state`, :func:`load_state`
    A state on disk, written through :func:`~sipnet_calibration.io.write_checked`.
:func:`fit_student_t`
    A Student-t (or Gaussian) fitted to weighted samples: a base density
    from an ensemble, and the independent proposal.
:func:`next_increment`, :func:`conditional_effective_sample_size`,
:func:`effective_sample_size`, :func:`pareto_k`, :func:`pareto_k_threshold`,
:func:`systematic_resample`
    The pieces, each usable alone.

Notes
-----
**Why the likelihood is cached.** Reweighting and choosing the increment need
:math:`\\ell` at the current samples only, which is kept per sample, and a
move needs :math:`\\log L` at the proposals only. So the likelihood is called
exactly once for stage 0 and once per move step, each call a batch of all
:math:`n` points, and :attr:`SMCState.n_evaluations` is :math:`n` times
:attr:`SMCState.n_calls`.

**Why the target is relative to CESS(0+).** Draws of ``q`` whose runs failed
have zero weight for every :math:`\\delta > 0`, so the CESS jumps from 1 to
the valid mass as soon as :math:`\\delta` leaves 0. A target relative to 1
would have no root when more than ``1 - cess_fraction`` of the draws failed.

**Adaptation and exactness.** The kernel's parameters are fixed within a
stage, set from the population at its start and the previous stage's moves,
so each step leaves :math:`\\pi_{\\beta_t}` invariant given them. The number
of steps depends on the stage's own acceptances: a stopping rule on the whole
population, whose effect on any one sample vanishes as :math:`n` grows, as
the rest of the method's error does. SMC is consistent as :math:`n \\to
\\infty`, not exact at finite :math:`n`; what is exact is the mode that does
no moves, importance sampling.

**Why a mixture of two kernels.** An independent proposal fitted to the
samples moves far in one step when the target is near Gaussian, and almost
never when it is curved, where its draws miss the ridge; a random walk moves
on either, slowly. Weighting them by their jump distances lets each stage use
the one that is moving the samples, and the bounds keep either from being
switched off by one poor stage.

Usage
-----
::

    from sipnet_calibration import smc

    problem = smc.TemperingProblem(
        log_prior=prior_density.log_prob,
        log_likelihood=log_likelihood,          # (n, D) -> (n,)
        base=smc.fit_student_t(ensemble, covariance_inflation=2.0,
                               degrees_of_freedom=5.0),
    )
    settings = smc.SMCSettings(n_samples=1000, seed=1)
    state = smc.initial_state(problem, settings)
    smc.save_state(path, state)
    for state in smc.run_smc(problem, state):
        smc.save_state(path, state)
    # resume: for state in smc.run_smc(problem, smc.load_state(path)): ...
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp

from sipnet_calibration.io import write_checked
from sipnet_calibration.validation import as_batched_flat, as_integer, as_positive_integer

__all__ = [
    "RECORD_FIELD_NAMES",
    "BaseDensity",
    "DefensiveMixture",
    "LikelihoodEvaluation",
    "MoveTally",
    "MultivariateStudentT",
    "SMCSettings",
    "SMCState",
    "TemperingProblem",
    "check_saved_state_equals",
    "conditional_effective_sample_size",
    "effective_sample_size",
    "fit_student_t",
    "initial_state",
    "load_state",
    "next_increment",
    "pareto_k",
    "pareto_k_threshold",
    "run_smc",
    "save_state",
    "systematic_resample",
]

#: The fields of each stage's record, in order. The reweight's: ``stage``;
#: ``beta`` after it and its ``increment``; ``ess``, the effective sample size
#: ``1 / sum W^2`` of the new weights, and ``cess``, ``n`` times the CESS at
#: the increment, both counts; ``pareto_k`` of the incremental log weights,
#: with ``pareto_k_threshold``; ``n_zero_weight``, samples whose runs failed;
#: ``log_evidence`` after it; ``n_unique``, distinct samples after
#: resampling. The moves': ``n_move_steps``; mean acceptance probability
#: ``acceptance`` and per kernel; ``jump_random_walk`` and
#: ``jump_independent``, the expected squared jump distances; ``moved_fraction``;
#: the kernel used, ``random_walk_scale`` and ``independent_fraction``. Both:
#: ``failed_runs`` in the stage's calls, and the cumulative ``n_evaluations``
#: and ``n_calls``. Stage 0 records its draws; a field a stage has no value for
#: is NaN.
RECORD_FIELD_NAMES = (
    "stage",
    "beta",
    "increment",
    "ess",
    "cess",
    "pareto_k",
    "pareto_k_threshold",
    "n_zero_weight",
    "log_evidence",
    "n_unique",
    "n_move_steps",
    "acceptance",
    "acceptance_random_walk",
    "acceptance_independent",
    "jump_random_walk",
    "jump_independent",
    "moved_fraction",
    "random_walk_scale",
    "independent_fraction",
    "failed_runs",
    "n_evaluations",
    "n_calls",
)


# ── densities ─────────────────────────────────────────────────────────────────


class BaseDensity(Protocol):
    """A normalized density on :math:`\\mathbb R^D` that can be sampled."""

    @property
    def dimension(self) -> int:
        """:math:`D`."""
        ...

    def log_prob(self, theta: Any) -> jax.Array:
        """The normalized log density, ``(n, D) -> (n,)``."""
        ...

    def sample(self, rng: np.random.Generator, n_samples: int) -> jax.Array:
        """*n_samples* independent draws, ``(n_samples, D)``."""
        ...


@dataclass(frozen=True, eq=False, kw_only=True)
class MultivariateStudentT:
    """A multivariate Student-t of given mean and covariance, or a Gaussian.

    Parameters
    ----------
    mean
        :math:`m`, ``(D,)``.
    covariance
        :math:`\\Sigma`, ``(D, D)``, symmetric positive definite: the
        distribution's covariance, not its scale matrix.
    degrees_of_freedom
        :math:`\\nu > 2`, or ``None`` for the Gaussian :math:`N(m, \\Sigma)`.

    Raises
    ------
    ValueError
        If the covariance is not a finite symmetric positive definite
        ``(D, D)`` matrix, or :math:`\\nu \\le 2`.

    Notes
    -----
    With scale matrix :math:`S = \\Sigma (\\nu - 2)/\\nu` (:math:`S = \\Sigma`
    for the Gaussian), :math:`S = L L^\\top` and :math:`r^2 = \\lVert
    L^{-1}(\\theta - m)\\rVert^2`,

    .. math::

        \\log t_\\nu = \\log\\Gamma\\!\\left(\\tfrac{\\nu + D}{2}\\right)
            - \\log\\Gamma\\!\\left(\\tfrac{\\nu}{2}\\right)
            - \\tfrac{D}{2}\\log(\\nu\\pi) - \\tfrac12 \\log|S|
            - \\tfrac{\\nu + D}{2} \\log\\left(1 + \\tfrac{r^2}{\\nu}\\right),
        \\qquad
        \\log N = -\\tfrac12\\left(D \\log 2\\pi + \\log|S| + r^2\\right),

    and a draw is :math:`m + L z / \\sqrt{u/\\nu}`, :math:`z \\sim N(0, I)`,
    :math:`u \\sim \\chi^2_\\nu` (:math:`m + Lz` for the Gaussian). The
    covariance is given rather than the scale so that one inflation factor
    means the same spread whatever :math:`\\nu`.
    """

    mean: jax.Array
    covariance: jax.Array
    degrees_of_freedom: float | None = None
    _cholesky: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64)
        covariance = np.array(self.covariance, dtype=np.float64)
        check_student_t_is_valid(mean, covariance, self.degrees_of_freedom)
        nu = self.degrees_of_freedom
        scale = covariance if nu is None else covariance * (nu - 2.0) / nu
        cholesky = np.linalg.cholesky(scale)
        cholesky.setflags(write=False)
        object.__setattr__(self, "mean", jnp.asarray(mean))
        object.__setattr__(self, "covariance", jnp.asarray(covariance))
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def dimension(self) -> int:
        """:math:`D`."""
        return int(self.mean.shape[0])

    def log_prob(self, theta: Any) -> jax.Array:
        """The normalized log density, ``(n, D) -> (n,)``."""
        theta = np.asarray(as_batched_flat(theta, self.dimension, message_name="theta"))
        whitened = solve_triangular(
            self._cholesky, (theta - np.asarray(self.mean)).T, lower=True
        )
        squared_radius = np.sum(whitened**2, axis=0)
        dimension = self.dimension
        log_det = 2.0 * np.sum(np.log(np.diag(self._cholesky)))
        nu = self.degrees_of_freedom
        if nu is None:
            values = -0.5 * (dimension * math.log(2.0 * math.pi) + log_det + squared_radius)
        else:
            values = (
                gammaln(0.5 * (nu + dimension))
                - gammaln(0.5 * nu)
                - 0.5 * dimension * math.log(nu * math.pi)
                - 0.5 * log_det
                - 0.5 * (nu + dimension) * np.log1p(squared_radius / nu)
            )
        return jnp.asarray(values)

    def sample(self, rng: np.random.Generator, n_samples: int) -> jax.Array:
        """*n_samples* independent draws, ``(n_samples, D)``."""
        draws = rng.standard_normal((n_samples, self.dimension)) @ self._cholesky.T
        if self.degrees_of_freedom is not None:
            nu = self.degrees_of_freedom
            draws = draws / np.sqrt(rng.chisquare(nu, n_samples) / nu)[:, None]
        return jnp.asarray(draws + np.asarray(self.mean))


@dataclass(frozen=True, eq=False, kw_only=True)
class DefensiveMixture:
    """A mixture :math:`q = (1 - \\alpha)\\, g + \\alpha\\, h` of two densities.

    With :math:`h` wider than the target (the prior), the importance weights
    of the target against :math:`q` are at most :math:`1/\\alpha` times those
    against :math:`h` (Hesterberg 1995).

    Parameters
    ----------
    component
        :math:`g`, the density fitted to the target.
    defensive
        :math:`h`, the density that bounds the weights.
    defensive_fraction
        :math:`\\alpha \\in (0, 1)`.

    Raises
    ------
    ValueError
        If the two densities differ in dimension, or :math:`\\alpha` is not
        in :math:`(0, 1)`.

    Notes
    -----
    :math:`\\log q = \\operatorname{logaddexp}\\big(\\log(1 - \\alpha) + \\log g,\\
    \\log\\alpha + \\log h\\big)`; a draw is :math:`h`'s with probability
    :math:`\\alpha`, else :math:`g`'s.
    """

    component: BaseDensity
    defensive: BaseDensity
    defensive_fraction: float

    def __post_init__(self) -> None:
        check_mixture_is_valid(self.component, self.defensive, self.defensive_fraction)

    @property
    def dimension(self) -> int:
        """:math:`D`."""
        return self.component.dimension

    def log_prob(self, theta: Any) -> jax.Array:
        """The normalized log density, ``(n, D) -> (n,)``."""
        alpha = self.defensive_fraction
        return jnp.asarray(
            np.logaddexp(
                math.log1p(-alpha) + np.asarray(self.component.log_prob(theta)),
                math.log(alpha) + np.asarray(self.defensive.log_prob(theta)),
            )
        )

    def sample(self, rng: np.random.Generator, n_samples: int) -> jax.Array:
        """*n_samples* independent draws, ``(n_samples, D)``."""
        from_defensive = rng.random(n_samples) < self.defensive_fraction
        component = np.asarray(self.component.sample(rng, n_samples))
        defensive = np.asarray(self.defensive.sample(rng, n_samples))
        return jnp.asarray(np.where(from_defensive[:, None], defensive, component))


def fit_student_t(
    samples: Any,
    *,
    log_weights: Any = None,
    covariance_inflation: float = 1.0,
    degrees_of_freedom: float | None = None,
) -> MultivariateStudentT:
    """A Student-t (or Gaussian) with the weighted samples' mean and inflated covariance.

    .. math::

        \\hat m = \\sum_i W_i \\theta_i, \\qquad
        \\hat C = \\frac{\\sum_i W_i (\\theta_i - \\hat m)(\\theta_i - \\hat m)^\\top}
            {1 - \\sum_i W_i^2},

    the unbiased weighted covariance (:math:`1/(n-1)` for equal weights), and
    the result has mean :math:`\\hat m` and covariance :math:`\\kappa \\hat C`.

    Parameters
    ----------
    samples
        ``(n, D)``.
    log_weights
        ``(n,)``, normalized or not; ``None`` for equal weights.
    covariance_inflation
        :math:`\\kappa > 0`.
    degrees_of_freedom
        :math:`\\nu > 2`, or ``None`` for a Gaussian.

    Raises
    ------
    ValueError
        If :math:`\\kappa \\le 0`, or the fitted covariance is not positive
        definite (fewer than :math:`D + 1` samples of positive weight, or
        samples in a subspace).
    """
    theta = np.asarray(samples, dtype=np.float64)
    weights = (
        np.full(theta.shape[0], 1.0 / theta.shape[0])
        if log_weights is None
        else _normalized_weights(np.asarray(log_weights, dtype=np.float64))
    )
    check_covariance_inflation_is_positive(covariance_inflation)
    mean = weights @ theta
    centered = theta - mean
    covariance = (weights[:, None] * centered).T @ centered / (1.0 - np.sum(weights**2))
    covariance = 0.5 * (covariance + covariance.T)
    return MultivariateStudentT(
        mean=mean,
        covariance=covariance_inflation * covariance,
        degrees_of_freedom=degrees_of_freedom,
    )


# ── the problem and the settings ──────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class LikelihoodEvaluation:
    """A batch of log likelihood values, with auxiliary values kept per sample.

    Parameters
    ----------
    log_likelihood
        ``(n,)``; NaN for a failed run.
    auxiliary
        ``(n, ...)``, such as a forward model's predictions: kept with each
        sample, copied when it is resampled and replaced when a move is
        accepted, so the final samples' values need no further call.
    """

    log_likelihood: Any
    auxiliary: Any


@dataclass(frozen=True, eq=False, kw_only=True)
class TemperingProblem:
    """The prior, the likelihood and the base density the targets are made of.

    Parameters
    ----------
    log_prior
        :math:`\\log p_0`, normalized, ``(n, D) -> (n,)``.
    log_likelihood
        :math:`\\log L`, normalized, ``(n, D)`` NumPy ``float64`` ``-> (n,)``
        or a :class:`LikelihoodEvaluation`; NaN for a failed run.
    base
        :math:`q`, normalized.
    """

    log_prior: Callable[[np.ndarray], Any]
    log_likelihood: Callable[[np.ndarray], Any]
    base: BaseDensity


@dataclass(frozen=True, kw_only=True)
class SMCSettings:
    """How a run proceeds: its size, seed, schedule and moves.

    Parameters
    ----------
    n_samples
        :math:`n`, at least 2.
    seed
        A non-negative integer; with the stage and step, it seeds every draw.
    one_step
        Reweight from :math:`\\beta = 0` to 1 in one stage, with no resampling
        or moves: importance sampling from :math:`q`.
    cess_fraction
        The CESS each increment keeps, relative to :math:`\\mathrm{CESS}(0^+)`,
        in :math:`(0, 1)`.
    max_stages
        The most reweights a run makes before it stops unfinished; a stopped
        run is continued by a state with a larger value.
    min_move_steps, max_move_steps
        The bounds on a stage's move steps, :math:`1 \\le` min :math:`\\le` max.
    move_probability
        A stage's moves stop once this fraction of its samples have moved,
        in :math:`(0, 1]`.
    random_walk_acceptance
        :math:`a^*`, the acceptance probability the random walk's scale is
        adapted towards, in :math:`(0, 1)`.
    independent_fraction_bounds
        The bounds :math:`0 \\le \\text{low} \\le \\text{high} \\le 1` of
        :math:`p_t`, the probability of the independent proposal; ``(0, 0)``
        is a random walk alone, ``(1, 1)`` the independent proposal alone.
    independent_degrees_of_freedom
        :math:`\\nu` of the independent proposal, or ``None`` for a Gaussian.
    independent_covariance_inflation
        :math:`\\kappa` of the independent proposal.
    n_bisect
        The bisections of :func:`next_increment`.

    Raises
    ------
    TypeError
        If an integer setting is not an integer.
    ValueError
        If a setting is outside its range.
    """

    n_samples: int
    seed: int
    one_step: bool = False
    cess_fraction: float = 0.5
    max_stages: int = 500
    min_move_steps: int = 3
    max_move_steps: int = 20
    move_probability: float = 0.99
    random_walk_acceptance: float = 0.234
    independent_fraction_bounds: tuple[float, float] = (0.1, 0.9)
    independent_degrees_of_freedom: float | None = 5.0
    independent_covariance_inflation: float = 1.0
    n_bisect: int = 60

    def __post_init__(self) -> None:
        # Coerced to plain Python values, which a saved state's JSON holds.
        for name in ("n_samples", "max_stages", "min_move_steps", "max_move_steps", "n_bisect"):
            object.__setattr__(self, name, as_positive_integer(getattr(self, name), message_name=f"SMCSettings.{name}"))
        object.__setattr__(self, "seed", as_integer(self.seed, message_name="SMCSettings.seed"))
        check_one_step_is_a_boolean(self.one_step)
        object.__setattr__(self, "one_step", bool(self.one_step))
        for name in ("cess_fraction", "move_probability", "random_walk_acceptance", "independent_covariance_inflation"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.independent_degrees_of_freedom is not None:
            object.__setattr__(self, "independent_degrees_of_freedom", float(self.independent_degrees_of_freedom))
        object.__setattr__(
            self, "independent_fraction_bounds", tuple(float(bound) for bound in self.independent_fraction_bounds)
        )
        check_smc_settings_are_valid(self)


# ── the state ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class MoveTally:
    """The current stage's moves so far.

    ``steps`` taken; proposals of each kernel; the sums over them of the
    acceptance probability :math:`\\min(1, \\alpha)` and of the jump
    :math:`\\min(1, \\alpha) \\lVert \\hat L^{-1}(\\theta' - \\theta)\\rVert^2`;
    ``failed_runs`` among the proposals; and ``moved``, ``(n,)``, whether each
    sample has accepted a proposal in this stage.
    """

    steps: int = 0
    proposed_random_walk: int = 0
    proposed_independent: int = 0
    acceptance_random_walk: float = 0.0
    acceptance_independent: float = 0.0
    jump_random_walk: float = 0.0
    jump_independent: float = 0.0
    failed_runs: int = 0
    moved: jax.Array | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class SMCState:
    """Everything a run needs to continue; the module docstring lists the fields."""

    settings: SMCSettings
    phase: Literal["reweight", "move", "done"]
    stage: int
    beta: float
    theta: jax.Array
    log_likelihood: jax.Array
    log_prior: jax.Array
    log_base: jax.Array
    log_weights: jax.Array
    auxiliary: jax.Array | None
    log_evidence: float
    n_evaluations: int
    n_calls: int
    random_walk_scale: float
    independent_fraction: float
    proposal_mean: jax.Array | None
    proposal_covariance: jax.Array | None
    moves: MoveTally
    records: tuple[frozendict, ...]

    @property
    def finished(self) -> bool:
        """Whether the run has reached :math:`\\beta = 1` and done its last moves."""
        return self.phase == "done"

    @property
    def weights(self) -> jax.Array:
        """The normalized weights :math:`W`, ``(n,)``."""
        return jnp.exp(self.log_weights)

    @property
    def log_target(self) -> jax.Array:
        """:math:`\\log\\pi_\\beta` at the samples, up to its constant, ``(n,)``."""
        return jnp.asarray(
            _log_target(
                self.beta,
                np.asarray(self.log_base),
                np.asarray(self.log_prior),
                np.asarray(self.log_likelihood),
            )
        )


# ── running ───────────────────────────────────────────────────────────────────


def initial_state(problem: TemperingProblem, settings: SMCSettings) -> SMCState:
    """Stage 0: ``n_samples`` draws of the base density and their log likelihood.

    The one call to the likelihood here is stage 0's; its failed runs have
    :math:`\\log L = -\\infty` and so zero weight from stage 1 on.

    Raises
    ------
    ValueError
        If the likelihood returns the wrong shape or :math:`+\\infty`, or a
        log density ratio :math:`\\ell` is NaN or :math:`+\\infty`.
    """
    n = settings.n_samples
    theta = np.asarray(problem.base.sample(_generator(settings.seed, 0, 0), n))
    log_likelihood, auxiliary, failed = _evaluate(problem, theta)
    log_prior, log_base = _log_densities(problem, theta)
    check_log_ratios_are_valid(_log_ratio(log_prior, log_likelihood, log_base))
    low, high = settings.independent_fraction_bounds
    record = _record(
        stage=0,
        beta=0.0,
        n_zero_weight=failed,
        log_evidence=0.0,
        failed_runs=failed,
        n_evaluations=n,
        n_calls=1,
    )
    return SMCState(
        settings=settings,
        phase="reweight",
        stage=0,
        beta=0.0,
        theta=jnp.asarray(theta),
        log_likelihood=jnp.asarray(log_likelihood),
        log_prior=jnp.asarray(log_prior),
        log_base=jnp.asarray(log_base),
        log_weights=jnp.full(n, -math.log(n)),
        auxiliary=None if auxiliary is None else jnp.asarray(auxiliary),
        log_evidence=0.0,
        n_evaluations=n,
        n_calls=1,
        random_walk_scale=2.38 / math.sqrt(theta.shape[1]),
        independent_fraction=float(np.clip(0.5, low, high)),
        proposal_mean=None,
        proposal_covariance=None,
        moves=MoveTally(),
        records=(record,),
    )


def run_smc(problem: TemperingProblem, state: SMCState) -> Iterator[SMCState]:
    """Continue *state* to :math:`\\beta = 1`, yielding the state after every operation.

    Each yield follows one reweight (with its resampling) or one move step,
    and is a point to checkpoint at. The run ends when a stage ends at
    :math:`\\beta = 1` (at once after the reweight, in the one-step mode), or
    when ``max_stages`` reweights have been made, unfinished.

    Raises
    ------
    ValueError
        If every sample's run failed, the increment reaches 0, the likelihood
        returns the wrong shape or :math:`+\\infty`, or a sample about to move
        is not at a finite target density.
    """
    while not state.finished:
        if state.phase == "reweight":
            if state.stage >= state.settings.max_stages:
                return
            state = _reweight(problem, state)
        else:
            state = _move(problem, state)
        yield state


def save_state(path: Path | str, state: SMCState) -> Path:
    """Write *state* to *path* as ``.npz``, through a checked ``.partial`` file.

    The check reads the file back with :func:`load_state` and requires every
    array and value to equal *state*'s exactly.

    Returns
    -------
    pathlib.Path
        *path*.
    """
    arrays, metadata = _state_contents(state)

    def write(partial: Path) -> None:
        with open(partial, "wb") as handle:
            np.savez(handle, **arrays, metadata=np.array(json.dumps(metadata)))

    def check(partial: Path) -> None:
        check_saved_state_equals(load_state(partial), state)

    return write_checked(path, write, check)


def load_state(path: Path | str) -> SMCState:
    """The state :func:`save_state` wrote to *path*."""
    with np.load(path, allow_pickle=False) as contents:
        arrays = {name: np.array(contents[name]) for name in contents.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    optional = {name: arrays.get(name) for name in ("auxiliary", "proposal_mean", "proposal_covariance", "moved")}
    settings = metadata.pop("settings")
    settings["independent_fraction_bounds"] = tuple(settings["independent_fraction_bounds"])
    moves = metadata.pop("moves")
    return SMCState(
        settings=SMCSettings(**settings),
        theta=jnp.asarray(arrays["theta"]),
        log_likelihood=jnp.asarray(arrays["log_likelihood"]),
        log_prior=jnp.asarray(arrays["log_prior"]),
        log_base=jnp.asarray(arrays["log_base"]),
        log_weights=jnp.asarray(arrays["log_weights"]),
        auxiliary=_jax_or_none(optional["auxiliary"]),
        proposal_mean=_jax_or_none(optional["proposal_mean"]),
        proposal_covariance=_jax_or_none(optional["proposal_covariance"]),
        moves=MoveTally(**moves, moved=_jax_or_none(optional["moved"])),
        records=tuple(frozendict(record) for record in metadata.pop("records")),
        **metadata,
    )


# ── the pieces ────────────────────────────────────────────────────────────────


def conditional_effective_sample_size(log_weights: Any, log_ratios: Any, increment: float) -> float:
    """The CESS of reweighting by :math:`e^{\\delta\\ell}`, as a fraction of :math:`n`.

    .. math::

        \\mathrm{CESS}(\\delta) = \\frac{\\big(\\sum_i W_i e^{\\delta\\ell_i}\\big)^2}
            {\\sum_i W_i e^{2\\delta\\ell_i}}
        = \\exp\\big(2\\,\\mathrm{lse}(\\log W + \\delta\\ell)
            - \\mathrm{lse}(\\log W + 2\\delta\\ell)\\big) \\in (0, 1],

    with :math:`W` the normalized weights, computed in the log-space form;
    a sample with :math:`\\ell_i = -\\infty` has weight zero at every
    :math:`\\delta`, 0 included, so the value at 0 is the mass of the others.
    With equal weights it is the effective sample size over :math:`n`.

    Parameters
    ----------
    log_weights
        :math:`\\log W`, ``(n,)``, normalized.
    log_ratios
        :math:`\\ell`, ``(n,)``.
    increment
        :math:`\\delta \\ge 0`.
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_ratios = np.asarray(log_ratios, dtype=np.float64)
    once = _incremental_log_weights(log_weights, log_ratios, increment)
    twice = _incremental_log_weights(log_weights, log_ratios, 2.0 * increment)
    return float(np.exp(2.0 * logsumexp(once) - logsumexp(twice)))


def next_increment(
    log_weights: Any,
    log_ratios: Any,
    *,
    cess_fraction: float,
    max_increment: float,
    n_bisect: int = 60,
) -> float:
    """The largest :math:`\\delta \\le` *max_increment* keeping the CESS at the target.

    The target is ``cess_fraction`` times :math:`\\mathrm{CESS}(0^+)`, which
    :func:`conditional_effective_sample_size` gives at 0. The CESS is
    non-increasing in :math:`\\delta` (its log derivative is twice the
    difference of the means of :math:`\\ell` tilted by :math:`\\delta` and by
    :math:`2\\delta`, which is never positive), so *max_increment* is returned
    when it meets the target, and otherwise the lower end of a bisection
    bracket, which always meets it.

    Parameters
    ----------
    log_weights
        :math:`\\log W`, ``(n,)``, normalized.
    log_ratios
        :math:`\\ell`, ``(n,)``.
    cess_fraction
        In :math:`(0, 1)`.
    max_increment
        :math:`1 - \\beta`, positive.
    n_bisect
        Each halves the bracket.
    """
    target = cess_fraction * conditional_effective_sample_size(log_weights, log_ratios, 0.0)
    if conditional_effective_sample_size(log_weights, log_ratios, max_increment) >= target:
        return float(max_increment)
    low, high = 0.0, float(max_increment)
    for _ in range(n_bisect):
        middle = 0.5 * (low + high)
        if conditional_effective_sample_size(log_weights, log_ratios, middle) >= target:
            low = middle
        else:
            high = middle
    return low


def effective_sample_size(log_weights: Any) -> float:
    """:math:`1 / \\sum_i W_i^2` of weights :math:`W \\propto e^{\\log w}`, a count in :math:`[1, n]`."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    return float(np.exp(-logsumexp(2.0 * (log_weights - logsumexp(log_weights)))))


def pareto_k(log_weights: Any) -> float:
    """The Pareto :math:`\\hat k` of importance weights, by ArviZ's PSIS.

    The shape of a generalized Pareto distribution fitted to the largest
    weights (Vehtari et al. 2024), from ``arviz.psislw``; weights of zero
    (log weight :math:`-\\infty`, a failed run) are left out, being no part
    of the tail. Below :func:`pareto_k_threshold` the weights' estimates are
    reliable. NaN when fewer than ten weights are positive, where the
    threshold is at most 0, or when they are all equal, where there is no
    tail to fit.
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    finite = log_weights[np.isfinite(log_weights)]
    if finite.size < 10 or np.all(finite == finite[0]):
        return math.nan
    # ArviZ warns on import of its coming refactor, which psislw survives.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
        import arviz

        return float(arviz.psislw(finite, reff=1.0)[1])


def pareto_k_threshold(n_samples: int) -> float:
    """:math:`\\min(1 - 1/\\log_{10} n, 0.7)`, the largest reliable :math:`\\hat k` at *n* samples."""
    return min(1.0 - 1.0 / math.log10(n_samples), 0.7) if n_samples > 1 else -math.inf


def systematic_resample(log_weights: Any, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: *n_samples* indices, sample :math:`i` about :math:`n W_i` times.

    With one uniform :math:`u`, index :math:`k` takes the :math:`i` whose
    cumulative weight interval :math:`[c_{i-1}, c_i)` holds
    :math:`(u + k)/n`; a sample of weight zero is never taken.
    """
    weights = _normalized_weights(np.asarray(log_weights, dtype=np.float64))
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    points = (rng.random() + np.arange(n_samples)) / n_samples
    # (u + n - 1) / n rounds to 1 for u within about n 2^-53 of 1, which lies
    # past every interval; it belongs to the last sample of positive weight.
    last = int(np.flatnonzero(weights > 0.0)[-1])
    return np.minimum(np.searchsorted(cumulative, points, side="right"), last)


# ── private helpers ───────────────────────────────────────────────────────────


def _reweight(problem: TemperingProblem, state: SMCState) -> SMCState:
    """Stage ``stage + 1``'s increment and reweight, then its resampling unless one-step."""
    settings = state.settings
    log_weights = np.asarray(state.log_weights)
    log_ratios = _log_ratio(
        np.asarray(state.log_prior), np.asarray(state.log_likelihood), np.asarray(state.log_base)
    )
    check_some_run_succeeded(log_ratios)
    remaining = 1.0 - state.beta
    increment = (
        remaining
        if settings.one_step
        else next_increment(
            log_weights,
            log_ratios,
            cess_fraction=settings.cess_fraction,
            max_increment=remaining,
            n_bisect=settings.n_bisect,
        )
    )
    check_increment_is_positive(increment, state.beta)
    incremental = _incremental_log_weights(log_weights, log_ratios, increment)
    log_normalizer = float(logsumexp(incremental))
    new_log_weights = incremental - log_normalizer
    beta = 1.0 if increment == remaining else state.beta + increment
    stage = state.stage + 1
    n = settings.n_samples
    reweighted = replace(
        state,
        stage=stage,
        beta=beta,
        log_weights=jnp.asarray(new_log_weights),
        log_evidence=state.log_evidence + log_normalizer,
    )
    finite = np.isfinite(incremental)
    record = dict(
        stage=stage,
        beta=beta,
        increment=increment,
        ess=effective_sample_size(new_log_weights),
        cess=n * conditional_effective_sample_size(log_weights, log_ratios, increment),
        pareto_k=pareto_k(incremental),
        pareto_k_threshold=pareto_k_threshold(int(finite.sum())),
        n_zero_weight=int((~finite).sum()),
        log_evidence=reweighted.log_evidence,
    )
    if settings.one_step:
        return replace(
            reweighted,
            phase="done",
            records=state.records
            + (_record(**record, failed_runs=0, n_evaluations=state.n_evaluations, n_calls=state.n_calls),),
        )
    return _resampled(reweighted, record)


def _resampled(state: SMCState, record: dict) -> SMCState:
    """*state* resampled to equal weights, with its stage's proposal fitted before."""
    theta = np.asarray(state.theta)
    fitted = fit_student_t(theta, log_weights=state.log_weights)
    indices = systematic_resample(
        state.log_weights, state.settings.n_samples, _generator(state.settings.seed, state.stage, 0)
    )
    record["n_unique"] = int(np.unique(indices).size)
    n = state.settings.n_samples
    return replace(
        state,
        phase="move",
        theta=jnp.asarray(theta[indices]),
        log_likelihood=state.log_likelihood[indices],
        log_prior=state.log_prior[indices],
        log_base=state.log_base[indices],
        log_weights=jnp.full(n, -math.log(n)),
        auxiliary=None if state.auxiliary is None else state.auxiliary[indices],
        proposal_mean=fitted.mean,
        proposal_covariance=fitted.covariance,
        moves=MoveTally(moved=jnp.zeros(n, dtype=bool)),
        records=state.records + (_record(**record),),
    )


def _move(problem: TemperingProblem, state: SMCState) -> SMCState:
    """One Metropolis-Hastings step of every sample, invariant for :math:`\\pi_\\beta`; one call."""
    settings = state.settings
    n, dimension = state.theta.shape
    rng = _generator(settings.seed, state.stage, state.moves.steps + 1)
    covariance = np.asarray(state.proposal_covariance)
    cholesky = np.linalg.cholesky(covariance)
    independent = MultivariateStudentT(
        mean=state.proposal_mean,
        covariance=settings.independent_covariance_inflation * covariance,
        degrees_of_freedom=settings.independent_degrees_of_freedom,
    )
    theta = np.asarray(state.theta)
    current = _log_target(
        state.beta,
        np.asarray(state.log_base),
        np.asarray(state.log_prior),
        np.asarray(state.log_likelihood),
    )
    check_samples_have_finite_target_density(current)
    use_independent = rng.random(n) < state.independent_fraction
    walk = theta + state.random_walk_scale * rng.standard_normal((n, dimension)) @ cholesky.T
    jump = np.asarray(independent.sample(rng, n))
    proposal = np.where(use_independent[:, None], jump, walk)
    log_likelihood, auxiliary, failed = _evaluate(problem, proposal)
    log_prior, log_base = _log_densities(problem, proposal)
    proposed = _log_target(state.beta, log_base, log_prior, log_likelihood)
    correction = np.asarray(independent.log_prob(theta)) - np.asarray(independent.log_prob(proposal))
    log_alpha = proposed - current + np.where(use_independent, correction, 0.0)
    accept = np.log1p(-rng.random(n)) <= log_alpha
    acceptance = np.exp(np.minimum(log_alpha, 0.0))
    jump_distance = acceptance * np.sum(
        solve_triangular(cholesky, (proposal - theta).T, lower=True) ** 2, axis=0
    )
    moves = state.moves
    tally = MoveTally(
        steps=moves.steps + 1,
        proposed_random_walk=moves.proposed_random_walk + int((~use_independent).sum()),
        proposed_independent=moves.proposed_independent + int(use_independent.sum()),
        acceptance_random_walk=moves.acceptance_random_walk + float(acceptance[~use_independent].sum()),
        acceptance_independent=moves.acceptance_independent + float(acceptance[use_independent].sum()),
        jump_random_walk=moves.jump_random_walk + float(jump_distance[~use_independent].sum()),
        jump_independent=moves.jump_independent + float(jump_distance[use_independent].sum()),
        failed_runs=moves.failed_runs + failed,
        moved=jnp.asarray(np.asarray(moves.moved) | accept),
    )
    moved = replace(
        state,
        theta=jnp.asarray(np.where(accept[:, None], proposal, theta)),
        log_likelihood=jnp.asarray(np.where(accept, log_likelihood, np.asarray(state.log_likelihood))),
        log_prior=jnp.asarray(np.where(accept, log_prior, np.asarray(state.log_prior))),
        log_base=jnp.asarray(np.where(accept, log_base, np.asarray(state.log_base))),
        auxiliary=_accepted_auxiliary(state.auxiliary, auxiliary, accept),
        n_evaluations=state.n_evaluations + n,
        n_calls=state.n_calls + 1,
        moves=tally,
    )
    moved_fraction = float(np.mean(np.asarray(tally.moved)))
    if tally.steps < settings.max_move_steps and (
        tally.steps < settings.min_move_steps or moved_fraction < settings.move_probability
    ):
        return moved
    return _stage_ended(moved, moved_fraction)


def _stage_ended(state: SMCState, moved_fraction: float) -> SMCState:
    """The stage's record completed, and the next stage's kernel adapted from its moves."""
    settings = state.settings
    moves = state.moves
    acceptance_random_walk = _ratio(moves.acceptance_random_walk, moves.proposed_random_walk)
    acceptance_independent = _ratio(moves.acceptance_independent, moves.proposed_independent)
    jump_random_walk = _ratio(moves.jump_random_walk, moves.proposed_random_walk)
    jump_independent = _ratio(moves.jump_independent, moves.proposed_independent)
    n_proposed = moves.proposed_random_walk + moves.proposed_independent
    record = dict(state.records[-1])
    record.update(
        n_move_steps=moves.steps,
        acceptance=(moves.acceptance_random_walk + moves.acceptance_independent) / n_proposed,
        acceptance_random_walk=acceptance_random_walk,
        acceptance_independent=acceptance_independent,
        jump_random_walk=jump_random_walk,
        jump_independent=jump_independent,
        moved_fraction=moved_fraction,
        random_walk_scale=state.random_walk_scale,
        independent_fraction=state.independent_fraction,
        failed_runs=moves.failed_runs,
        n_evaluations=state.n_evaluations,
        n_calls=state.n_calls,
    )
    scale = state.random_walk_scale
    if math.isfinite(acceptance_random_walk):
        scale *= math.exp(acceptance_random_walk - settings.random_walk_acceptance)
    fraction = state.independent_fraction
    total_jump = jump_random_walk + jump_independent
    if math.isfinite(total_jump) and total_jump > 0.0:
        fraction = float(np.clip(jump_independent / total_jump, *settings.independent_fraction_bounds))
    return replace(
        state,
        phase="done" if state.beta == 1.0 else "reweight",
        random_walk_scale=scale,
        independent_fraction=fraction,
        moves=MoveTally(),
        records=state.records[:-1] + (frozendict(record),),
    )


def _evaluate(problem: TemperingProblem, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, int]:
    """The log likelihood at *theta*, NaN made :math:`-\\infty`; its auxiliary values; the failures."""
    result = problem.log_likelihood(np.array(theta))
    auxiliary = None
    if isinstance(result, LikelihoodEvaluation):
        result, auxiliary = result.log_likelihood, np.asarray(result.auxiliary)
        check_auxiliary_values_have_one_row_per_sample(auxiliary.shape, theta.shape[0])
    values = np.asarray(result, dtype=np.float64)
    check_log_likelihood_has_one_value_per_sample(values.shape, theta.shape[0])
    failed = np.isnan(values)
    values = np.where(failed, -np.inf, values)
    check_log_likelihood_is_never_positive_infinity(values)
    return values, auxiliary, int(failed.sum())


def _log_densities(problem: TemperingProblem, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:math:`\\log p_0` and :math:`\\log q` at *theta*, ``(n,)`` each."""
    log_prior = np.asarray(problem.log_prior(theta), dtype=np.float64)
    log_base = np.asarray(problem.base.log_prob(theta), dtype=np.float64)
    check_log_densities_are_not_nan(log_prior, message_name="the log prior")
    check_log_densities_are_not_nan(log_base, message_name="the base density's log density")
    return log_prior, log_base


def _log_ratio(log_prior: np.ndarray, log_likelihood: np.ndarray, log_base: np.ndarray) -> np.ndarray:
    """:math:`\\ell = \\log p_0 + \\log L - \\log q`, :math:`-\\infty` where the run failed."""
    with np.errstate(invalid="ignore"):
        ratios = log_prior + log_likelihood - log_base
    return np.where(np.isneginf(log_likelihood), -np.inf, ratios)


def _log_target(beta: float, log_base: np.ndarray, log_prior: np.ndarray, log_likelihood: np.ndarray) -> np.ndarray:
    """:math:`(1 - \\beta)\\log q + \\beta(\\log p_0 + \\log L)`; :math:`-\\infty` where a run failed, for :math:`\\beta > 0`."""
    # At either end one term is absent, not multiplied by 0, which would make
    # NaN of a log density of -inf there.
    if beta == 0.0:
        return np.array(log_base)
    if beta == 1.0:
        return log_prior + log_likelihood
    return (1.0 - beta) * log_base + beta * (log_prior + log_likelihood)


def _incremental_log_weights(log_weights: np.ndarray, log_ratios: np.ndarray, increment: float) -> np.ndarray:
    """:math:`\\log W + \\delta\\ell`, :math:`-\\infty` where :math:`\\ell` is (at :math:`\\delta = 0` too)."""
    finite = np.isfinite(log_ratios)
    return np.where(finite, log_weights + increment * np.where(finite, log_ratios, 0.0), -np.inf)


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """:math:`W_i = e^{\\log w_i} / \\sum_j e^{\\log w_j}`."""
    return np.exp(log_weights - logsumexp(log_weights))


def _accepted_auxiliary(current: jax.Array | None, proposed: np.ndarray | None, accept: np.ndarray) -> jax.Array | None:
    """The auxiliary values of the samples after the step: the proposal's where accepted."""
    if current is None:
        return None
    mask = accept.reshape((-1,) + (1,) * (proposed.ndim - 1))
    return jnp.asarray(np.where(mask, proposed, np.asarray(current)))


def _record(**values: Any) -> frozendict:
    """A stage's record: every field of :data:`RECORD_FIELD_NAMES`, NaN where not given."""
    return frozendict({name: values.get(name, math.nan) for name in RECORD_FIELD_NAMES})


def _ratio(total: float, count: int) -> float:
    """*total* over *count*, NaN for no count."""
    return total / count if count else math.nan


def _generator(seed: int, stage: int, step: int) -> np.random.Generator:
    """The generator of stage *stage*'s step *step* (0: its draw or resampling)."""
    return np.random.default_rng([seed, stage, step])


def _state_contents(state: SMCState) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """*state* as the arrays and the JSON-able values :func:`save_state` writes."""
    arrays = {
        name: np.asarray(getattr(state, name))
        for name in ("theta", "log_likelihood", "log_prior", "log_base", "log_weights")
    }
    for name in ("auxiliary", "proposal_mean", "proposal_covariance"):
        if getattr(state, name) is not None:
            arrays[name] = np.asarray(getattr(state, name))
    if state.moves.moved is not None:
        arrays["moved"] = np.asarray(state.moves.moved)
    moves = {f.name: getattr(state.moves, f.name) for f in dataclasses.fields(MoveTally) if f.name != "moved"}
    metadata = {
        "settings": dataclasses.asdict(state.settings),
        "phase": state.phase,
        "stage": state.stage,
        "beta": state.beta,
        "log_evidence": state.log_evidence,
        "n_evaluations": state.n_evaluations,
        "n_calls": state.n_calls,
        "random_walk_scale": state.random_walk_scale,
        "independent_fraction": state.independent_fraction,
        "moves": moves,
        "records": [dict(record) for record in state.records],
    }
    return arrays, metadata


def _jax_or_none(array: np.ndarray | None) -> jax.Array | None:
    """*array* as JAX, or ``None``."""
    return None if array is None else jnp.asarray(array)


# ── checks ────────────────────────────────────────────────────────────────────


def check_student_t_is_valid(mean: np.ndarray, covariance: np.ndarray, degrees_of_freedom: float | None) -> None:
    """A Student-t's covariance is a finite symmetric positive definite matrix
    of its mean's dimension, and its degrees of freedom exceed 2, so that its
    density is finite everywhere and its covariance exists."""
    dimension = mean.shape[0] if mean.ndim == 1 else -1
    if mean.ndim != 1 or covariance.shape != (dimension, dimension):
        raise ValueError(
            f"a Student-t's mean is (D,) and its covariance (D, D), got {mean.shape} and "
            f"{covariance.shape}; pass matching shapes."
        )
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))):
        raise ValueError("a Student-t's mean and covariance are finite; fit them to finite samples.")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
        raise ValueError("a Student-t's covariance is symmetric; symmetrize it.")
    if np.linalg.eigvalsh(covariance)[0] <= 0.0:
        raise ValueError(
            "a Student-t's covariance is positive definite; fit it to more samples of positive "
            "weight than dimensions, not lying in a subspace."
        )
    if degrees_of_freedom is not None and not degrees_of_freedom > 2.0:
        raise ValueError(
            f"a Student-t's degrees of freedom exceed 2, where its covariance exists, got "
            f"{degrees_of_freedom}; pass more, or None for a Gaussian."
        )


def check_mixture_is_valid(component: BaseDensity, defensive: BaseDensity, fraction: float) -> None:
    """A mixture's two densities share a dimension and its fraction is in (0, 1),
    where both logarithms of its weights are finite."""
    if component.dimension != defensive.dimension:
        raise ValueError(
            f"a mixture's densities share a dimension, got {component.dimension} and "
            f"{defensive.dimension}; pass densities over one space."
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"a defensive fraction is in (0, 1), got {fraction}; use the component alone for 0."
        )


def check_covariance_inflation_is_positive(inflation: float) -> None:
    """A covariance inflation is positive, so the covariance it scales stays positive definite."""
    if not inflation > 0.0:
        raise ValueError(f"a covariance inflation is positive, got {inflation}; pass 1 for none.")


def check_one_step_is_a_boolean(one_step: Any) -> None:
    """``one_step`` is a boolean, which a truthy string or number would pass for."""
    if not isinstance(one_step, (bool, np.bool_)):
        raise TypeError(f"SMCSettings.one_step is a boolean, got {one_step!r}; pass True or False.")


def check_smc_settings_are_valid(settings: SMCSettings) -> None:
    """The settings' fractions, step bounds and sample count are in range, where
    the schedule terminates and the moves are defined."""
    if settings.n_samples < 2:
        raise ValueError(f"SMCSettings.n_samples is at least 2, got {settings.n_samples}.")
    if settings.seed < 0:
        raise ValueError(f"SMCSettings.seed is non-negative, as NumPy's seeds are, got {settings.seed}.")
    if not 0.0 < settings.cess_fraction < 1.0:
        raise ValueError(
            f"SMCSettings.cess_fraction is in (0, 1), got {settings.cess_fraction}; at 1 no "
            "increment is positive."
        )
    if settings.min_move_steps > settings.max_move_steps:
        raise ValueError(
            f"SMCSettings.min_move_steps is at most max_move_steps, got {settings.min_move_steps} "
            f"and {settings.max_move_steps}."
        )
    if not 0.0 < settings.move_probability <= 1.0:
        raise ValueError(f"SMCSettings.move_probability is in (0, 1], got {settings.move_probability}.")
    if not 0.0 < settings.random_walk_acceptance < 1.0:
        raise ValueError(
            f"SMCSettings.random_walk_acceptance is in (0, 1), got {settings.random_walk_acceptance}."
        )
    low, high = settings.independent_fraction_bounds
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(
            f"SMCSettings.independent_fraction_bounds are 0 <= low <= high <= 1, got "
            f"{settings.independent_fraction_bounds}."
        )
    # Refused here rather than at the first move, after stage 0's calls.
    check_covariance_inflation_is_positive(settings.independent_covariance_inflation)
    nu = settings.independent_degrees_of_freedom
    if nu is not None and not nu > 2.0:
        raise ValueError(
            f"SMCSettings.independent_degrees_of_freedom exceed 2, where the proposal's covariance "
            f"exists, got {nu}; pass more, or None for a Gaussian."
        )


def check_log_likelihood_has_one_value_per_sample(shape: tuple[int, ...], n_samples: int) -> None:
    """The log likelihood has shape ``(n,)``, where an ``(n, 1)`` would broadcast
    against the samples' ``(n,)`` arrays into ``(n, n)`` without an error."""
    if shape != (n_samples,):
        raise ValueError(
            f"the log likelihood has one value per sample, shape ({n_samples},), got {shape}; "
            "return one value per row of theta."
        )


def check_auxiliary_values_have_one_row_per_sample(shape: tuple[int, ...], n_samples: int) -> None:
    """The auxiliary values have one row per sample, which resampling and
    accepted moves index by sample."""
    if shape[:1] != (n_samples,):
        raise ValueError(
            f"the auxiliary values have one row per sample, {n_samples} rows, got shape {shape}; "
            "return one row per row of theta."
        )


def check_log_likelihood_is_never_positive_infinity(values: np.ndarray) -> None:
    """No log likelihood is +inf, which would take all the weight and make it NaN."""
    if np.any(np.isposinf(values)):
        raise ValueError("a log likelihood is +inf; a normalized likelihood is finite or a failure (NaN).")


def check_log_densities_are_not_nan(values: np.ndarray, *, message_name: str) -> None:
    """No log density is NaN, which a Metropolis ratio would silently reject
    and which would make the stage's acceptance record NaN."""
    if np.any(np.isnan(values)):
        raise ValueError(f"{message_name} is NaN at {int(np.isnan(values).sum())} points; return -inf outside its support.")


def check_log_ratios_are_valid(log_ratios: np.ndarray) -> None:
    """No log density ratio is NaN or +inf, as a base density that is 0 at its
    own draws, or a prior and a base both 0, would make one; the weights would
    be NaN."""
    if np.any(np.isnan(log_ratios) | np.isposinf(log_ratios)):
        raise ValueError(
            "a log density ratio log p_0 + log L - log q is NaN or +inf; the base density is 0 "
            "where it was sampled, or the prior and base are both 0 there."
        )


def check_increment_is_positive(increment: float, beta: float) -> None:
    """An increment is positive, so the ladder advances: a log likelihood
    spread too wide for float64 bisection would give 0 and stall it."""
    if not increment > 0.0:
        raise ValueError(
            f"the tempering increment from beta = {beta} is 0; the log likelihoods' spread is too "
            "wide to bisect, so raise n_bisect or lower cess_fraction."
        )


def check_some_run_succeeded(log_ratios: np.ndarray) -> None:
    """Some sample's run succeeded, without which no weight is positive and
    the increment, the weights and the resampling would all be NaN."""
    if not np.any(np.isfinite(log_ratios)):
        raise ValueError("every sample's run failed; no weight is positive, so nothing can be resampled.")


def check_samples_have_finite_target_density(log_target: np.ndarray) -> None:
    """Every sample about to move is at a finite target density: from -inf a
    Metropolis ratio is NaN, never accepted, and the sample would never move."""
    if not np.all(np.isfinite(log_target)):
        raise ValueError(
            f"{int((~np.isfinite(log_target)).sum())} samples about to move have a target density "
            "of 0 or NaN; only samples of positive weight are resampled, so the state is corrupt."
        )


def check_saved_state_equals(loaded: SMCState, state: SMCState) -> None:
    """A state read back from its file equals the one written, bit for bit, so
    a resumed run continues the same run."""
    written, written_metadata = _state_contents(state)
    read, read_metadata = _state_contents(loaded)
    same_arrays = written.keys() == read.keys() and all(
        written[name].dtype == read[name].dtype and np.array_equal(written[name], read[name], equal_nan=True)
        for name in written
    )
    if not same_arrays or json.dumps(written_metadata) != json.dumps(read_metadata):
        raise ValueError("the state read back differs from the state written; the file is not a faithful copy.")

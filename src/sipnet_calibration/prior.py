"""The prior over a parameter vector: what is believed before the data.

Where this sits
---------------
::

    parameter_vector.ParameterVector      (the unknowns x, and theta = T^-1(x))
      -> prior.Prior                      (pi(x), and so the density of theta)
      -> pyEKI (sample, gaussian), an MCMC target (log_prob)

This module imports TFP's distributions and pyEKI's ``Gaussian`` and
operators, and not pySIPNET.

What it reads
-------------
A :class:`~sipnet_calibration.parameter_vector.ParameterVector` and one
:class:`PriorTerm` per parameter. A term's distribution is the prior of the
parameter's whole natural value:

- **a parameter without a dim**: a TFP distribution whose event shape is the
  parameter's value shape, with batch shape ``()``, in ``float64``; or a
  :data:`PriorFunction` called with ``dim_index=None``;
- **a parameter with a dim**: a :data:`PriorFunction`, called as
  ``f(dim_index, site_table)`` with the vector's
  :meth:`~sipnet_calibration.parameter_vector.ParameterVector.dim_index` and
  site table, returning such a distribution over every dim label at once.
  :func:`iid_over_dim` and :func:`independent_over_dim` make one.

The distributions of the family builders (:func:`log_normal`,
:func:`logit_normal`, :func:`softmax_normal` and their ``_from_*`` forms)
are pushforwards of a Gaussian through their support's default bijector.

The density
-----------
With parameters :math:`x`, prior :math:`\\pi(x) = \\prod_b \\pi_b(x_{B_b})`
and coordinates :math:`\\theta = T^{-1}(x)`,

.. math::

    \\log \\pi_\\theta(\\theta) = \\sum_b \\Big[ \\log \\pi_b\\big(T(\\theta)_{B_b}\\big)
        + \\sum_{p \\in B_b} \\log J_p(\\theta_p) \\Big],

each density and :math:`J_p` against the reference measure of
:meth:`~sipnet_calibration.parameter_vector.Support.log_jacobian`.
:meth:`Prior.log_prob` evaluates each term by its base density where it is a
pushforward through its parameter's own bijector, and by change of variables
otherwise.

Functions and classes
---------------------
:class:`Prior`
    ``sample``, ``log_prob``, ``gaussian``, ``select``, ``describe``.
:class:`PriorTerm`, :data:`PriorFunction`
    One parameter's prior and its provenance.
:func:`iid_over_dim`, :func:`independent_over_dim`
    Priors over a dim, independent across dim labels.
The family builders
    :func:`log_normal`, :func:`log_normal_from_interval`,
    :func:`log_normal_from_samples`, :func:`logit_normal`,
    :func:`logit_normal_from_interval`, :func:`logit_normal_from_samples`,
    :func:`softmax_normal`.
:func:`check_prior_term_is_valid`
    The checks a term passes at construction.

Notes
-----
**Gaussians are declared, not detected.** The builders know the Gaussian in
theta they built: :func:`iid_over_dim` and :func:`independent_over_dim` over
a family builder's distribution, and a family builder's distribution given
directly, declare it. A declaration is honored when the parameter's bijector
agrees with the family's at the probe points, and is checked against
``log_prob`` at construction. It feeds only :meth:`Prior.gaussian`, so a
wrong declaration could distort the Gaussian approximation but never
``log_prob``, and the check catches it first.

**A prior function must marginalize consistently.** ``Prior(vector.select(...),
prior.terms)`` rebuilds each term on fewer dim labels, which is the exact
marginal only if the distribution at a set of dim labels is the marginal of
the one at any larger set. The builders here satisfy this. A user-written
function that normalizes over the labels present, or standardizes a
covariate inside, does not, and changes meaning under ``select``.

Usage
-----
::

    import jax
    from sipnet_calibration.prior import (
        Prior, PriorTerm, iid_over_dim, log_normal_from_interval, logit_normal,
        softmax_normal,
    )

    prior = Prior(vector, {
        "respiration_share": PriorTerm(
            logit_normal(median=0.18, logit_sd=0.35), provenance="..."),
        "allocation": PriorTerm(
            iid_over_dim(softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5)),
            provenance="..."),
        "initial_soil_carbon": PriorTerm(
            iid_over_dim(log_normal_from_interval(lower=5e3, upper=8e4)), provenance="..."),
    })
    theta = prior.sample(jax.random.key(0), 50)   # (50, D)
    prior.log_prob(theta)                         # (50,)
    prior.gaussian()                              # pyeki.gauss.Gaussian, exact here
    prior.describe()                              # one row per term
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from frozendict import frozendict
from pyeki.gauss import Gaussian
from pyeki.linalg import DensePSD, PSDBlockDiag, PSDDiagonal, PSDLinOp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    Parameter,
    ParameterVector,
    Support,
    probe_points,
)
from sipnet_calibration.validation import as_bounded_integer, truncated

__all__ = [
    "Prior",
    "PriorFunction",
    "PriorTerm",
    "check_prior_term_is_valid",
    "independent_over_dim",
    "iid_over_dim",
    "log_normal",
    "log_normal_from_interval",
    "log_normal_from_samples",
    "logit_normal",
    "logit_normal_from_interval",
    "logit_normal_from_samples",
    "softmax_normal",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

#: A prior over a whole natural value, built for the vector at hand: called as
#: ``f(dim_index, site_table)`` and returning a TFP distribution.
type PriorFunction = Callable[..., tfd.Distribution]


# ── the prior ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, repr=False)
class Prior:
    """A prior over a parameter vector, one term per parameter.

    Parameters
    ----------
    parameter_vector:
        The vector the prior is over.
    terms:
        ``{parameter name: PriorTerm}``, one for every parameter.

    Raises
    ------
    TypeError
        If a key is not a parameter name, a term is not a
        :class:`PriorTerm`, or a parameter with a dim is given a bare
        distribution.
    KeyError
        If a key names no parameter of the vector.
    ValueError
        If a parameter has no term, or a term fails a check of
        :func:`check_prior_term_is_valid`; the message names the term.
    """

    parameter_vector: ParameterVector
    terms: Mapping[str, PriorTerm]
    _built: Mapping[str, _BuiltTerm] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", frozendict(self.terms))
        check_terms_cover_the_parameters(self.terms, self.parameter_vector)
        # With a given graph the terms are built in its topological order;
        # without one, every order is.
        built = {
            p.name: _BuiltTerm.build(p, self.terms[p.name], self.parameter_vector)
            for p in self.parameter_vector.parameters
        }
        object.__setattr__(self, "_built", frozendict(built))

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, name: str) -> PriorTerm:
        """The term of the parameter called *name*."""
        check_term_is_held(name, self)
        return self.terms[name]

    def __repr__(self) -> str:
        return f"Prior(D={self.parameter_vector.dimension}, terms={list(self.terms)})"

    def describe(self) -> pd.DataFrame:
        """One row per term, indexed by ``parameters``: ``prior``, ``given``,
        ``evaluated_by`` (``"base density"`` or ``"change of variables"``),
        ``declared_gaussian`` and ``provenance``."""
        rows = [
            {
                "parameters": name,
                "prior": _term_name(self.terms[name], built.distribution),
                "given": "",
                "evaluated_by": built.evaluated_by,
                "declared_gaussian": built.declared is not None,
                "provenance": self.terms[name].provenance,
            }
            for name, built in self._built.items()
        ]
        return pd.DataFrame(rows).set_index("parameters")

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Any) -> Prior:
        """``Prior(vector.select(**selectors), kept terms)``: the prior of a
        smaller vector, each term rebuilt on its dim labels.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`ParameterVector.select` and :class:`Prior`.
        """
        vector = self.parameter_vector.select(**selectors)
        return Prior(vector, {name: self.terms[name] for name in vector.parameter_names})

    # ── evaluation ────────────────────────────────────────────────────────────

    def sample(self, key: Array, n: int) -> Array:
        """``n`` draws of theta from the prior, ``(n, D)``.

        Each term draws with ``jax.random.fold_in(key, crc32(name))``, so its
        draws depend on its name and not on where it is declared, and adding
        or reordering other terms leaves them unchanged.

        Raises
        ------
        TypeError
            If *n* is not an integer.
        ValueError
            If *n* is negative, or a draw maps to a non-finite theta, which a
            prior with mass on its support's boundary gives in ``float64``;
            the message names the term.
        """
        n = as_bounded_integer(n, minimum=0, message_name="n")
        pieces = []
        for name, built in self._built.items():
            theta = built.sample_theta(_term_key(key, name), n)
            check_draws_map_to_finite_theta(name, theta)
            pieces.append(theta.reshape((n, -1)))
        return jnp.concatenate(pieces, axis=-1)

    def log_prob(self, theta: Any) -> Array:
        """:math:`\\log \\pi_\\theta(\\theta)`, ``(..., D) -> (...)``.

        Each term contributes, at its :math:`\\theta_{B}`,

        - by **base density**, when its distribution is
          ``TransformedDistribution(base, b)`` with ``b`` equal to the
          parameter's bijector at the probe points:
          :math:`\\log \\mathrm{base}(\\theta_B)`, exactly;
        - by **change of variables** otherwise:
          :math:`\\log \\pi_b(T(\\theta_B)) + \\sum \\log J(\\theta_B)`, with
          :math:`\\log J` from
          :meth:`~sipnet_calibration.parameter_vector.Support.log_jacobian`.

        Traceable under ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        theta = jnp.asarray(theta, dtype=jnp.float64)
        check_theta_ends_in_the_dimension(theta.shape, self.parameter_vector.dimension)
        lead = theta.shape[:-1]
        total = jnp.zeros(lead, dtype=jnp.float64)
        for built in self._built.values():
            total = total + built.log_prob(theta[..., built.positions].reshape(lead + built.shape))
        return total

    def gaussian(self, *, key: Array | None = None, n_moment_samples: int = 0) -> Gaussian:
        """The prior as a Gaussian over theta, ``pyeki.gauss.Gaussian``.

        The covariance is a ``PSDBlockDiag`` with one block per parameter, in
        theta's order. A term whose Gaussian is declared and honored gets its
        exact :math:`(m, C)`; any other is moment-matched from
        :math:`M` = *n_moment_samples* draws :math:`\\theta^{(i)}` of it,

        .. math::

            \\hat m = \\frac{1}{M} \\sum_i \\theta^{(i)}, \\qquad
            \\hat C = \\frac{1}{M - 1} \\sum_i (\\theta^{(i)} - \\hat m)
                     (\\theta^{(i)} - \\hat m)^\\top,

        each with ``jax.random.fold_in(key, crc32(name))``.

        Raises
        ------
        NotImplementedError
            If a term needs moment matching and no *key* was given.
        ValueError
            If a moment-matched block has at least *n_moment_samples*
            entries: :math:`\\hat C` then has rank at most :math:`M - 1` and
            is singular.
        """
        means, blocks = [], []
        for name, built in self._built.items():
            if built.declared is not None:
                mean, block = built.declared
            else:
                check_moment_matching_is_possible(name, built.size, key, n_moment_samples)
                mean, block = built.moment_matched(_term_key(key, name), n_moment_samples)
            means.append(jnp.asarray(mean, dtype=jnp.float64))
            blocks.append(block)
        return Gaussian(mean=jnp.concatenate(means), cov=PSDBlockDiag(tuple(blocks)))


# ── its pieces ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class PriorTerm:
    """One parameter's prior, and where it came from.

    Parameters
    ----------
    distribution:
        A TFP distribution over the parameter's whole natural value, or a
        :data:`PriorFunction` that builds one; the module docstring says
        which a parameter takes.
    provenance:
        Where the prior came from, with its citation; a placeholder says it
        is one.
    """

    distribution: tfd.Distribution | PriorFunction
    _: KW_ONLY
    provenance: str

    def __post_init__(self) -> None:
        check_provenance_is_given(self.provenance)


# ── priors over a dim ─────────────────────────────────────────────────────────


def iid_over_dim(distribution: tfd.Distribution) -> PriorFunction:
    """Every dim label independently distributed as *distribution*:

    .. math::

        \\pi(x) = \\prod_{\\ell=1}^{n} \\pi_0(x_\\ell).

    A ``TransformedDistribution(base, b)`` (a family builder's, say) is
    built as ``TransformedDistribution(Sample(base, n), b)``, with the
    bijector outside, so that :meth:`Prior.log_prob` evaluates it by its base
    density; any other distribution as ``Sample(distribution, n)``. Over a
    family builder's distribution with Gaussian base :math:`(m_0, C_0)` it
    declares :math:`(\\mathbf 1_n \\otimes m_0,\\ I_n \\otimes C_0)`.

    Parameters
    ----------
    distribution:
        The prior of one dim label's value, batch shape ``()``.

    Returns
    -------
    PriorFunction
    """
    return _OverDim(
        per_dim_label=lambda dim_index: distribution,
        repeated=True,
        name=f"iid {_distribution_name(distribution)}",
    )


def independent_over_dim(
    distribution_family: Callable[..., tfd.Distribution], /, **arguments: Any
) -> PriorFunction:
    """Independent across dim labels, each dim label's distribution made by
    *distribution_family* from its own arguments:

    .. math::

        \\pi(x) = \\prod_{\\ell=1}^{n} \\pi_{\\mathrm{family}}\\big(x_\\ell;\\ a_\\ell\\big).

    Parameters
    ----------
    distribution_family:
        Builds a distribution from keyword arguments, such as
        :func:`log_normal`, and gives it batch shape ``(n,)`` when each
        argument has one entry per dim label.
    **arguments:
        Each either one value shared by every dim label, passed as it is, or
        a ``Mapping`` or ``pd.Series`` keyed by dim label, read at the
        vector's dim labels (a superset is allowed) and passed as an array
        with one leading entry per dim label. A transformed family is built
        with its bijector outside, as for :func:`iid_over_dim`, and a family
        builder's declares its Gaussian, per dim label.

    Returns
    -------
    PriorFunction

    Raises
    ------
    KeyError
        When called, for a dim label a keyed argument lacks.
    ValueError
        When called, if the family's distribution is not a batch of one per
        dim label, as when no argument is keyed (:func:`iid_over_dim` is that
        prior).
    """

    def per_dim_label(dim_index: pd.Index) -> tfd.Distribution:
        aligned = {
            name: _aligned_argument(name, value, dim_index) for name, value in arguments.items()
        }
        distribution = distribution_family(**aligned)
        check_family_is_one_per_dim_label(distribution, len(dim_index))
        return distribution

    return _OverDim(
        per_dim_label=per_dim_label,
        repeated=False,
        name=f"independent {getattr(distribution_family, '__name__', 'family')}",
    )


# ── the family builders ───────────────────────────────────────────────────────


def log_normal(*, median: Any, geometric_sd: Any) -> tfd.LogNormal:
    """The log-normal of the given median and geometric standard deviation:

    .. math::

        \\log x \\sim \\mathcal N\\big(\\log \\mathrm{median},\\ (\\log \\mathrm{geometric\\_sd})^2\\big),

    so its central 95% interval is
    :math:`\\mathrm{median} \\cdot \\mathrm{geometric\\_sd}^{\\pm 1.96}`.

    Parameters
    ----------
    median:
        Positive; one value per dim label gives a batch.
    geometric_sd:
        Above 1.

    Raises
    ------
    ValueError
        If *median* is not positive or *geometric_sd* not above 1.
    """
    median = _positive_array("log_normal median", median)
    geometric_sd = _positive_array("log_normal geometric_sd", geometric_sd)
    check_geometric_sd_exceeds_one(geometric_sd)
    return tfd.LogNormal(loc=jnp.log(median), scale=jnp.log(geometric_sd))


def log_normal_from_interval(*, lower: Any, upper: Any, mass: float = 0.95) -> tfd.LogNormal:
    """The log-normal whose central *mass* interval is ``[lower, upper]``:

    .. math::

        \\mu = \\tfrac12 (\\log l + \\log u), \\qquad
        \\sigma = \\frac{\\log u - \\log l}{2 z}, \\qquad
        z = \\Phi^{-1}\\big(\\tfrac12 + \\tfrac{\\mathrm{mass}}{2}\\big).

    Raises
    ------
    ValueError
        If an end is not positive, ``upper <= lower``, or *mass* is not in
        ``(0, 1)``.
    """
    lower = _positive_array("log_normal_from_interval lower", lower)
    upper = _positive_array("log_normal_from_interval upper", upper)
    loc, scale = _normal_from_interval(jnp.log(lower), jnp.log(upper), mass)
    return tfd.LogNormal(loc=loc, scale=scale)


def log_normal_from_samples(samples: Any) -> tfd.LogNormal:
    """The maximum-likelihood log-normal of positive samples:
    :math:`\\mu = \\overline{\\log x}`, :math:`\\sigma` the standard deviation
    of :math:`\\log x`.

    Raises
    ------
    ValueError
        If fewer than two samples are given, one is missing or not positive,
        or all are equal.
    """
    logs = jnp.log(_samples_in_support("log_normal_from_samples", samples, lambda v: v > 0))
    return tfd.LogNormal(loc=jnp.mean(logs), scale=_positive_std(logs, "log_normal_from_samples"))


def logit_normal(
    *, median: Any, logit_sd: Any, support: Support = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on the open interval *support*, :math:`(a, b)`:

    .. math::

        \\operatorname{logit}\\frac{x - a}{b - a} \\sim
            \\mathcal N\\Big(\\operatorname{logit}\\frac{\\mathrm{median} - a}{b - a},\\
            \\mathrm{logit\\_sd}^2\\Big).

    On :math:`(0, 1)` this is ``tfd.LogitNormal``; on :math:`(a, b)`,
    ``TransformedDistribution(Normal, Sigmoid(low=a, high=b))``. A
    ``logit_sd`` near 1.7 is close to flat.

    Raises
    ------
    ValueError
        If *median* is outside *support*, *logit_sd* is not positive, or
        *support* is not an interval.
    """
    check_support_is_an_interval(support)
    fraction = _interval_fraction("logit_normal median", median, support)
    logit_sd = _positive_array("logit_normal logit_sd", logit_sd)
    return _logit_normal_on(support, _logit(fraction), logit_sd)


def logit_normal_from_interval(
    *, lower: Any, upper: Any, mass: float = 0.95, support: Support = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on *support* whose central *mass* interval is
    ``[lower, upper]``: :func:`log_normal_from_interval`'s formulas on the
    logit scale of :math:`(x - a)/(b - a)`.

    Raises
    ------
    ValueError
        If an end is outside *support*, ``upper <= lower``, or *mass* is not
        in ``(0, 1)``.
    """
    check_support_is_an_interval(support)
    lower = _interval_fraction("logit_normal_from_interval lower", lower, support)
    upper = _interval_fraction("logit_normal_from_interval upper", upper, support)
    loc, scale = _normal_from_interval(_logit(lower), _logit(upper), mass)
    return _logit_normal_on(support, loc, scale)


def logit_normal_from_samples(samples: Any, *, support: Support = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The maximum-likelihood logit-normal on *support* of samples inside
    it, on the logit scale of :math:`(x - a)/(b - a)`.

    Raises
    ------
    ValueError
        As :func:`log_normal_from_samples`, for samples outside *support*.
    """
    check_support_is_an_interval(support)
    fractions = (
        _samples_in_support(
            "logit_normal_from_samples", samples, lambda v: (v > support.low) & (v < support.high)
        )
        - support.low
    ) / (support.high - support.low)
    logits = _logit(fractions)
    return _logit_normal_on(support, jnp.mean(logits), _positive_std(logits, "logit_normal_from_samples"))


def softmax_normal(*, center: Any, logit_sd: Any) -> tfd.TransformedDistribution:
    """A Gaussian on :math:`\\mathbb{R}^{k-1}` pushed through
    ``SoftmaxCentered`` onto the open simplex:

    .. math::

        t_i = \\log\\frac{x_i}{x_k} \\sim \\mathcal N\\Big(\\log\\frac{c_i}{c_k},\\
            \\mathrm{logit\\_sd}_i^2\\Big), \\quad i < k,

    independently, with :math:`c` the *center*, so the base's mean maps to
    *center*.

    Parameters
    ----------
    center:
        ``(k,)`` positive fractions summing to 1, ``k >= 2``, or ``(n, k)``
        for one per dim label.
    logit_sd:
        A scalar, or one value per unconstrained number (``k - 1``).

    Raises
    ------
    ValueError
        For a center that is not a point of the simplex, or a *logit_sd* of
        another shape or not positive.

    Notes
    -----
    Aitchison's name for the family is the logistic-normal; the logit-normal
    is its ``k = 2`` case, hence softmax-normal here.
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    check_center_is_on_the_simplex(center)
    logit_sd = _positive_array("softmax_normal logit_sd", logit_sd)
    loc = jnp.log(center[..., :-1] / center[..., -1:])
    check_logit_sd_fits_the_center(logit_sd, loc)
    scale = jnp.broadcast_to(logit_sd, loc.shape)
    return tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), tfb.SoftmaxCentered()
    )


# ── the declared Gaussian and term evaluation ─────────────────────────────────


@dataclass(frozen=True, eq=False)
class _OverDim:
    """A :data:`PriorFunction` over a dim that declares its Gaussian when
    each dim label's distribution is a family builder's.

    ``per_dim_label(dim_index)`` gives one dim label's distribution when
    *repeated*, else every dim label's as a batch of ``n``.
    """

    per_dim_label: Callable[[pd.Index], tfd.Distribution]
    repeated: bool
    name: str

    def __call__(self, dim_index: pd.Index | None, site_table: pd.DataFrame) -> tfd.Distribution:
        check_prior_over_a_dim_has_a_dim(dim_index, self.name)
        distribution = self.per_dim_label(dim_index)
        n = len(dim_index)
        if type(distribution) in _CARRIES_ITS_BIJECTOR:
            base = distribution.distribution
            base = tfd.Sample(base, n) if self.repeated else tfd.Independent(base, 1)
            return tfd.TransformedDistribution(base, distribution.bijector)
        return tfd.Sample(distribution, n) if self.repeated else tfd.Independent(distribution, 1)

    def unconstrained_gaussian(
        self, dim_index: pd.Index, site_table: pd.DataFrame
    ) -> tuple[Array, PSDLinOp, tfb.Bijector] | None:
        """The declared :math:`(m, C)` over the flattened ``(n, e)`` value, and
        the bijector it assumes; ``None`` outside the family builders'."""
        family = _family_gaussian(self.per_dim_label(dim_index))
        if family is None:
            return None
        loc, scale, bijector = family
        if self.repeated:
            n = len(dim_index)
            loc = jnp.broadcast_to(loc, (n, *loc.shape))
            scale = jnp.broadcast_to(scale, (n, *scale.shape))
        return jnp.ravel(loc), PSDDiagonal(jnp.ravel(scale) ** 2), bijector


@dataclass(frozen=True, eq=False)
class _BuiltTerm:
    """A term built for its parameter on the vector at hand."""

    name: str
    parameter: Parameter
    distribution: tfd.Distribution
    positions: np.ndarray
    shape: tuple[int, ...]
    by_base_density: bool
    declared: tuple[Array, PSDLinOp] | None

    @classmethod
    def build(cls, parameter: Parameter, term: PriorTerm, vector: ParameterVector) -> _BuiltTerm:
        check_term_is_a_prior_term(parameter.name, term)
        distribution = _distribution_for(parameter, term, vector)
        check_term_shape(parameter.name, distribution, vector.value_shape(parameter.name))
        shape = vector.unconstrained_shape(parameter.name)
        probes = jnp.asarray(probe_points(shape, unconstrained_size=parameter.unconstrained_size))
        by_base_density = _pushes_through(distribution, parameter.bijector, probes)
        built = cls(
            name=parameter.name,
            parameter=parameter,
            distribution=distribution,
            positions=vector.positions(parameter_name=parameter.name),
            shape=shape,
            by_base_density=by_base_density,
            declared=None,
        )
        check_prior_term_is_valid(built)
        declared = _declaration(term, parameter, vector, probes)
        if declared is not None:
            check_declaration_agrees_with_log_prob(parameter.name, declared, built, probes)
            object.__setattr__(built, "declared", declared)
        return built

    @property
    def evaluated_by(self) -> str:
        return "base density" if self.by_base_density else "change of variables"

    @property
    def size(self) -> int:
        return len(self.positions)

    def log_prob(self, theta: Array) -> Array:
        """This term's contribution at its theta, ``(..., *shape) -> (...)``."""
        if self.by_base_density:
            return self.distribution.distribution.log_prob(theta)
        support = self.parameter.support
        log_jacobian = support.log_jacobian(self.parameter.bijector, theta)
        density = self.distribution.log_prob(self.parameter.bijector.forward(theta))
        extra_axes = tuple(range(theta.ndim - len(self.shape), log_jacobian.ndim))
        return density + log_jacobian.sum(axis=extra_axes)

    def sample_theta(self, key: Array, n: int) -> Array:
        """``n`` draws of this term's theta, ``(n, *shape)``."""
        if self.by_base_density:
            return self.distribution.distribution.sample(n, seed=key)
        return self.parameter.bijector.inverse(self.distribution.sample(n, seed=key))

    def moment_matched(self, key: Array, n_moment_samples: int) -> tuple[Array, PSDLinOp]:
        draws = self.sample_theta(key, n_moment_samples).reshape((n_moment_samples, -1))
        mean = draws.mean(axis=0)
        centered = draws - mean
        return mean, DensePSD(centered.T @ centered / (n_moment_samples - 1))


def _declaration(
    term: PriorTerm, parameter: Parameter, vector: ParameterVector, probes: Array
) -> tuple[Array, PSDLinOp] | None:
    """The term's declared Gaussian, when it declares one and the parameter's
    bijector agrees with the one it assumes."""
    source = term.distribution
    if isinstance(source, _OverDim):
        declared = source.unconstrained_gaussian(
            vector.dim_index(parameter.dim) if parameter.dim else None, vector.site_table
        )
    elif isinstance(source, tfd.Distribution) and parameter.dim is None:
        family = _family_gaussian(source)
        declared = None if family is None else (
            jnp.ravel(family[0]), PSDDiagonal(jnp.ravel(family[1]) ** 2), family[2]
        )
    else:
        declared = None
    if declared is None:
        return None
    mean, covariance, assumed = declared
    if not _bijectors_agree(assumed, parameter.bijector, probes):
        return None
    return mean, covariance


def _family_gaussian(distribution: tfd.Distribution) -> tuple[Array, Array, tfb.Bijector] | None:
    """A family builder's distribution as its base's mean and standard
    deviations, broadcast to ``(*batch, e?)``, and its bijector; ``None`` for
    any other distribution. Recognized by exact class."""
    kind = type(distribution)
    if kind is tfd.Normal:
        found = distribution.loc, distribution.scale, tfb.Identity()
    elif kind in (tfd.LogNormal, tfd.LogitNormal):
        base = distribution.distribution
        found = base.loc, base.scale, distribution.bijector
    elif kind is tfd.TransformedDistribution and type(distribution.distribution) is tfd.Normal and type(
        distribution.bijector
    ) is tfb.Sigmoid:
        base = distribution.distribution
        found = base.loc, base.scale, distribution.bijector
    elif kind is tfd.TransformedDistribution and type(
        distribution.distribution
    ) is tfd.MultivariateNormalDiag and type(distribution.bijector) is tfb.SoftmaxCentered:
        base = distribution.distribution
        found = base.mean(), base.stddev(), distribution.bijector
    else:
        return None
    loc, scale, bijector = found
    loc, scale = jnp.broadcast_arrays(jnp.asarray(loc), jnp.asarray(scale))
    return loc, scale, bijector


def _pushes_through(distribution: tfd.Distribution, bijector: tfb.Bijector, probes: Array) -> bool:
    """Whether *distribution* is a pushforward through *bijector*: an exact
    ``TransformedDistribution``, ``LogNormal`` or ``LogitNormal`` whose own
    bijector agrees with it at the probe points."""
    return type(distribution) in _CARRIES_ITS_BIJECTOR and _bijectors_agree(
        distribution.bijector, bijector, probes
    )


def _bijectors_agree(first: tfb.Bijector, second: tfb.Bijector, probes: Array) -> bool:
    """Whether two bijectors map the probe points alike. Images are compared,
    never bijectors: ``Sigmoid()`` and ``Sigmoid(0, 1)`` compare unequal."""
    return bool(
        np.allclose(first.forward(probes), second.forward(jnp.array(probes)), rtol=1e-10, atol=0.0)
    )


# ── private helpers ───────────────────────────────────────────────────────────

#: TFP's classes whose ``.distribution`` and ``.bijector`` are a base in theta
#: and a map from it. Subclasses such as ``MultivariateNormalTriL`` carry an
#: internal reparameterization instead, so only the exact classes count.
_CARRIES_ITS_BIJECTOR = (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)

#: The number of draws of the draw-based support check, and the size of each
#: batch of them, which bounds its memory for a term over many dim labels.
_SUPPORT_DRAWS, _SUPPORT_DRAW_BATCH = 10_000, 1_000

#: The seed of the draw-based support check.
_SUPPORT_SEED = 20260927

#: The declaration check's tolerance: :math:`|\\log q - \\log p| \\le
#: 10^{-10} |\\log p| + 10^{-12} D_b`.
_DECLARATION_RELATIVE_TOLERANCE, _DECLARATION_ABSOLUTE_TOLERANCE = 1e-10, 1e-12


def _term_key(key: Array, name: str) -> Array:
    return jax.random.fold_in(key, zlib.crc32(name.encode()))


def _distribution_for(parameter: Parameter, term: PriorTerm, vector: ParameterVector) -> tfd.Distribution:
    """The term's distribution for the vector: called when a function."""
    source = term.distribution
    if isinstance(source, tfd.Distribution):
        check_bare_distribution_has_no_dim(parameter)
        return source
    dim_index = vector.dim_index(parameter.dim) if parameter.dim else None
    return source(dim_index, vector.site_table)


def _aligned_argument(name: str, value: Any, dim_index: pd.Index) -> Any:
    """One argument of :func:`independent_over_dim`: a keyed one as one entry
    per dim label, a shared one as it is."""
    if not isinstance(value, (Mapping, pd.Series)):
        return value
    check_argument_covers_the_dim_labels(name, value, dim_index)
    return jnp.asarray(np.asarray([value[label] for label in dim_index], dtype=np.float64))


def _logit_normal_on(support: Support, loc: Array, scale: Array) -> tfd.Distribution:
    if (support.low, support.high) == (0.0, 1.0):
        return tfd.LogitNormal(loc=loc, scale=scale)
    return tfd.TransformedDistribution(tfd.Normal(loc, scale), support.bijector())


def _interval_fraction(what: str, value: Any, support: Support) -> Array:
    """``(value - a) / (b - a)``, checked inside :math:`(0, 1)`."""
    array = jnp.asarray(value, dtype=jnp.float64)
    fraction = (array - support.low) / (support.high - support.low)
    check_values_are_inside(what, fraction, value, support)
    return fraction


def _positive_array(what: str, value: Any) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    check_values_are_positive(what, array, value)
    return array


def _logit(p: Array) -> Array:
    return jnp.log(p) - jnp.log1p(-p)


def _normal_from_interval(lower: Array, upper: Array, mass: float) -> tuple[Array, Array]:
    """The Normal whose central *mass* interval is ``[lower, upper]``."""
    check_interval_is_valid(lower, upper, mass)
    z = tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)).quantile(jnp.float64(0.5 + mass / 2))
    return (lower + upper) / 2.0, (upper - lower) / (2.0 * z)


def _samples_in_support(what: str, samples: Any, in_support: Callable[[Array], Array]) -> Array:
    array = jnp.asarray(samples, dtype=jnp.float64).ravel()
    check_samples_are_usable(what, array, in_support)
    return array


def _positive_std(values: Array, what: str) -> Array:
    std = jnp.std(values)
    check_samples_vary(what, std)
    return std


def _term_name(term: PriorTerm, distribution: tfd.Distribution) -> str:
    source = term.distribution
    return source.name if isinstance(source, _OverDim) else _distribution_name(distribution)


def _distribution_name(distribution: tfd.Distribution | None) -> str:
    kind = type(distribution)
    if kind is tfd.LogNormal:
        return "log-normal"
    if kind is tfd.LogitNormal:
        return "logit-normal"
    if kind is tfd.TransformedDistribution:
        base = distribution.distribution
        inner = base.distribution if type(base) in (tfd.Sample, tfd.Independent) else base
        if isinstance(distribution.bijector, tfb.SoftmaxCentered):
            return "softmax-normal" if type(inner) is tfd.MultivariateNormalDiag else "softmax-transformed"
        return f"{type(inner).__name__} through {distribution.bijector.name}"
    return kind.__name__


def _probe_log_prob_is_finite(distribution: tfd.Distribution, parameter: Parameter, probes: Array) -> bool:
    """Whether the density is finite at the image of every probe point that
    lies inside the support; one the bijector rounds onto the boundary
    (``IteratedSigmoidCentered`` at 20) says nothing of the prior's support."""
    values = parameter.bijector.forward(probes)
    inside = parameter.support.contains(values)
    inside = inside.reshape((len(probes), -1)).all(axis=-1)
    return bool(jnp.all(jnp.isfinite(distribution.log_prob(values)) | ~inside))


def _default_bijector_images_lie_in(distribution: tfd.Distribution, support: Support) -> bool | None:
    """Whether the distribution's own default event-space bijector maps the
    probe points into the closure of *support*; ``None`` when it has none."""
    try:
        bijector = distribution.experimental_default_event_space_bijector()
    except NotImplementedError:
        return None
    if bijector is None:
        return None
    shape = tuple(bijector.inverse_event_shape(distribution.event_shape))
    probes = jnp.asarray(probe_points(shape, unconstrained_size=shape[-1] if shape else 1))
    # The closure: a bijector may round onto the boundary at the outer probes,
    # which is float64, not a wrong support.
    return bool(jnp.all(support.contains(bijector.forward(probes), closure=True)))


def _draws_lie_in_the_support(distribution: tfd.Distribution, parameter: Parameter) -> bool | None:
    """Whether fixed-seed draws lie in the support and map to finite theta;
    ``None`` when the distribution cannot be sampled."""
    key = jax.random.key(_SUPPORT_SEED)
    for batch in range(_SUPPORT_DRAWS // _SUPPORT_DRAW_BATCH):
        try:
            draws = distribution.sample(_SUPPORT_DRAW_BATCH, seed=jax.random.fold_in(key, batch))
        except NotImplementedError:
            return None
        theta = parameter.bijector.inverse(draws)
        if not (
            bool(jnp.all(parameter.support.contains(draws)))
            and bool(jnp.all(jnp.isfinite(theta)))
        ):
            return False
    return True


# ── checks ────────────────────────────────────────────────────────────────────


def check_prior_term_is_valid(built: _BuiltTerm) -> None:
    """A term's prior has the support its parameter declares, and on the
    simplex a density :meth:`Prior.log_prob` can evaluate."""
    check_simplex_density_is_a_dirichlet(built)
    check_declared_support_lies_in_the_priors(built)
    check_priors_support_lies_in_the_declared(built)


def check_terms_cover_the_parameters(terms: Mapping[Any, Any], vector: ParameterVector) -> None:
    """The terms are keyed by the vector's parameter names, one each."""
    for key in terms:
        if not isinstance(key, str):
            raise TypeError(
                f"a prior term is keyed by a parameter name, got {key!r}; joint terms over "
                "several parameters are not supported yet."
            )
        if key not in vector:
            raise KeyError(
                f"a prior term names {key!r}, which is no parameter of the vector; name one of "
                f"{truncated(list(vector.parameter_names))}."
            )
    missing = [name for name in vector.parameter_names if name not in terms]
    if missing:
        raise ValueError(f"the parameter(s) {truncated(missing)} have no prior term; give each one.")


def check_term_is_held(name: Any, prior: Prior) -> None:
    if name not in prior.terms:
        raise KeyError(f"the prior has no term {name!r}; name one of {truncated(list(prior.terms))}.")


def check_term_is_a_prior_term(name: str, term: Any) -> None:
    """A term is a :class:`PriorTerm`, which carries its provenance."""
    if not isinstance(term, PriorTerm):
        raise TypeError(
            f"the prior of {name!r} is a {type(term).__name__}; wrap it as "
            "PriorTerm(distribution, provenance=...)."
        )


def check_provenance_is_given(provenance: Any) -> None:
    """A prior says where it came from."""
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "a prior term needs a provenance: where the prior came from, or that it is a "
            "placeholder."
        )


def check_bare_distribution_has_no_dim(parameter: Parameter) -> None:
    """A parameter with a dim takes a prior function, since its prior depends
    on the dim labels present."""
    if parameter.dim is not None:
        raise TypeError(
            f"parameter {parameter.name!r} varies over {parameter.dim!r}, so its prior is built "
            "for the dim labels present; wrap the distribution as iid_over_dim(distribution) "
            "or independent_over_dim(family, ...)."
        )


def check_prior_over_a_dim_has_a_dim(dim_index: Any, name: str) -> None:
    """A prior over a dim is given to a parameter with a dim."""
    if dim_index is None:
        raise TypeError(
            f"{name} is a prior over a dim, given to a parameter without one; give that "
            "parameter the distribution itself."
        )


def check_term_shape(name: str, distribution: Any, value_shape: tuple[int, ...]) -> None:
    """A term's distribution is ``float64`` over the parameter's whole value:
    event shape the value shape, batch shape ``()``."""
    if not isinstance(distribution, tfd.Distribution):
        raise TypeError(
            f"the prior of {name!r} built a {type(distribution).__name__}, not a TFP distribution."
        )
    event, batch = tuple(distribution.event_shape), tuple(distribution.batch_shape)
    if event != value_shape or batch != () or distribution.dtype != jnp.float64:
        raise ValueError(
            f"the prior of {name!r} has event shape {event}, batch shape {batch} and dtype "
            f"{distribution.dtype}, but must be a float64 distribution over the parameter's "
            f"whole value: event shape {value_shape}, batch shape (). A batch of priors, one "
            "per dim label, is iid_over_dim or independent_over_dim."
        )


def check_simplex_density_is_a_dirichlet(built: _BuiltTerm) -> None:
    """On the simplex, a term evaluated by change of variables is a
    ``Dirichlet``, whose density is against the first ``k - 1`` coordinates,
    as the Jacobian is."""
    if built.parameter.support.kind != "simplex" or built.by_base_density:
        return
    inner = built.distribution
    if type(inner) in (tfd.Sample, tfd.Independent):
        inner = inner.distribution
    if type(inner) is not tfd.Dirichlet:
        raise ValueError(
            f"the prior of {built.name!r} is a density on the simplex other than a Dirichlet, "
            "whose reference measure is unknown; write it as a pushforward through the "
            "parameter's bijector, TransformedDistribution(base, parameter.bijector)."
        )


def check_declared_support_lies_in_the_priors(built: _BuiltTerm) -> None:
    """The prior's density is finite at the image of every probe point, so
    the declared support lies in the prior's."""
    probes = jnp.asarray(probe_points(built.shape, unconstrained_size=built.parameter.unconstrained_size))
    if not _probe_log_prob_is_finite(built.distribution, built.parameter, probes):
        raise ValueError(
            f"the prior of {built.name!r} has no density at some values of its declared "
            f"support {built.parameter.support.name!r}; its own support is smaller. Declare "
            "the support the prior has, or choose a prior over the whole support."
        )


def check_priors_support_lies_in_the_declared(built: _BuiltTerm) -> None:
    """The prior puts no mass outside the declared support, by its own
    default bijector's images and by draws that map to finite theta."""
    support = built.parameter.support
    by_bijector = _default_bijector_images_lie_in(built.distribution, support)
    by_draws = _draws_lie_in_the_support(built.distribution, built.parameter)
    if by_bijector is None and by_draws is None:
        raise ValueError(
            f"the prior of {built.name!r} has neither a default event-space bijector nor a "
            "sampler, so its support cannot be checked; give a distribution TFP can sample."
        )
    if by_bijector is False or by_draws is False:
        raise ValueError(
            f"the prior of {built.name!r} puts mass outside its declared support "
            f"{support.name!r}, or on its boundary, where theta is not finite; declare the "
            "support the prior has, or choose a prior inside it."
        )


def check_declaration_agrees_with_log_prob(
    name: str, declared: tuple[Array, PSDLinOp], built: _BuiltTerm, probes: Array
) -> None:
    """A declared Gaussian's log density equals the term's at every probe
    point, :math:`|\\log q - \\log p| \\le 10^{-10} |\\log p| + 10^{-12} D_b`."""
    mean, covariance = declared
    flat = probes.reshape((len(probes), -1))
    declared_log_density = Gaussian(mean=mean, cov=covariance).log_density(flat)
    log_density = built.log_prob(probes)
    tolerance = (
        _DECLARATION_RELATIVE_TOLERANCE * jnp.abs(log_density)
        + _DECLARATION_ABSOLUTE_TOLERANCE * flat.shape[-1]
    )
    if not bool(jnp.all(jnp.abs(declared_log_density - log_density) <= tolerance)):
        raise ValueError(
            f"the Gaussian declared for {name!r} disagrees with its log density at the probe "
            "points; this is a defect in the prior builder, not in the inputs."
        )


def check_draws_map_to_finite_theta(name: str, theta: Array) -> None:
    """Every draw maps to a finite theta."""
    if not bool(jnp.all(jnp.isfinite(theta))):
        raise ValueError(
            f"the prior of {name!r} drew a value on its support's boundary, whose theta is not "
            "finite; choose a prior with less mass at the boundary."
        )


def check_moment_matching_is_possible(
    name: str, size: int, key: Array | None, n_moment_samples: int
) -> None:
    """A moment-matched block has a key, and more draws than entries, since
    its estimate is otherwise singular."""
    if key is None:
        raise NotImplementedError(
            f"the prior of {name!r} declares no Gaussian, so it is moment-matched from draws; "
            "pass key= and n_moment_samples=."
        )
    if size >= n_moment_samples:
        raise ValueError(
            f"the prior of {name!r} has {size} entries and would be moment-matched from "
            f"{n_moment_samples} draws, whose covariance has rank at most "
            f"{max(n_moment_samples - 1, 0)} and is singular; draw more than {size}, or write "
            "the prior as a family builder's distribution, which declares its Gaussian."
        )


def check_theta_ends_in_the_dimension(shape: tuple[int, ...], dimension: int) -> None:
    """Theta's last axis has ``D`` entries."""
    if not shape or shape[-1] != dimension:
        raise ValueError(
            f"theta must end in the vector's dimension {dimension}, got shape {shape}; pass "
            "(..., D)."
        )


def check_argument_covers_the_dim_labels(name: str, value: Mapping[Any, Any], dim_index: pd.Index) -> None:
    """A keyed argument has a value for every dim label."""
    missing = [label for label in dim_index if label not in value]
    if missing:
        raise KeyError(
            f"independent_over_dim argument {name!r} has no value for dim label(s) "
            f"{truncated(missing)}; key it by every dim label."
        )


def check_family_is_one_per_dim_label(distribution: Any, n_dim_labels: int) -> None:
    """A family's distribution is a batch of one per dim label, which the dim
    labels would otherwise be misaligned with."""
    batch = tuple(getattr(distribution, "batch_shape", ()))
    if batch != (n_dim_labels,):
        raise ValueError(
            f"independent_over_dim built a distribution of batch shape {batch} for "
            f"{n_dim_labels} dim labels; key at least one argument by dim label, or use "
            "iid_over_dim for one distribution shared by every dim label."
        )


def check_geometric_sd_exceeds_one(geometric_sd: Array) -> None:
    if not bool(jnp.all(geometric_sd > 1.0)):
        raise ValueError("log_normal: geometric_sd must exceed 1, since it multiplies.")


def check_values_are_positive(what: str, array: Array, value: Any) -> None:
    if not bool(jnp.all(jnp.isfinite(array)) and jnp.all(array > 0)):
        raise ValueError(f"{what} must be finite and positive; got {value!r}.")


def check_values_are_inside(what: str, fraction: Array, value: Any, support: Support) -> None:
    if not bool(jnp.all((fraction > 0) & (fraction < 1))):
        raise ValueError(f"{what} must lie inside {support.name}; got {value!r}.")


def check_support_is_an_interval(support: Any) -> None:
    if not isinstance(support, Support) or support.kind != "interval":
        raise ValueError(
            f"a logit-normal is on an open interval, got support {support!r}; use "
            "OPEN_UNIT_INTERVAL or OpenInterval(low, high)."
        )


def check_interval_is_valid(lower: Array, upper: Array, mass: float) -> None:
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must lie in (0, 1); got {mass}.")
    if not bool(jnp.all(upper > lower)):
        raise ValueError("upper must exceed lower.")


def check_samples_are_usable(what: str, array: Array, in_support: Callable[[Array], Array]) -> None:
    if array.size < 2:
        raise ValueError(f"{what}: need at least two samples; got {array.size}.")
    bad = ~(jnp.isfinite(array) & in_support(array))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{what}: {int(bad.sum())} of {array.size} samples are missing or outside the "
            "support; exclude and count them before fitting."
        )


def check_samples_vary(what: str, std: Array) -> None:
    if not bool(std > 0):
        raise ValueError(f"{what}: the samples are all equal; no scale can be fitted.")


def check_center_is_on_the_simplex(center: Array) -> None:
    if center.ndim not in (1, 2) or center.shape[-1] < 2:
        raise ValueError("softmax_normal: center must be (k,) or (n, k) with k >= 2.")
    if not bool(jnp.all(center > 0)) or not bool(jnp.allclose(center.sum(axis=-1), 1.0, atol=1e-8)):
        raise ValueError("softmax_normal: center must be positive fractions summing to 1.")


def check_logit_sd_fits_the_center(logit_sd: Array, loc: Array) -> None:
    if logit_sd.shape not in ((), loc.shape[-1:]):
        raise ValueError(
            "softmax_normal: logit_sd must be a scalar or one value per unconstrained number, "
            f"shape {loc.shape[-1:]}; got shape {logit_sd.shape}."
        )

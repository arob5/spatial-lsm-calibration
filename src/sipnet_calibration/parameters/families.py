"""The family builders: one value's distribution, a pushforward of a
Gaussian through its support's default bijector.

Where this sits
---------------
::

    parameters.support                 (the supports and their bijectors)
      -> parameters.families           (one value's distribution)
      -> parameters.prior_functions    (repeated or varied over a block)
      -> parameters.prior.PriorTerm    (one factor of the prior)

The families know nothing of a prior, a vector or labels: each returns a
TFP distribution of one value, of TFP batch shape ``()`` unless an argument
has one entry per label, as :func:`~sipnet_calibration.parameters.prior_functions.independent_over_dim`
reads it.

The families
------------
Each is the pushforward :math:`x = T(t)` of a Gaussian :math:`t`, :math:`T`
the support's default bijector:

- :func:`log_normal`, :func:`log_normal_from_interval`,
  :func:`log_normal_from_samples`: on the positive reals, :math:`T = \\exp`;
- :func:`logit_normal`, :func:`logit_normal_from_interval`,
  :func:`logit_normal_from_samples`: on a finite interval :math:`(a, b)`,
  :math:`T` the scaled sigmoid;
- :func:`softmax_normal`: on the simplex, :math:`T` ``SoftmaxCentered``.

So a prior term over a parameter with that default bijector is evaluated by
its base density, with no Jacobian.

Notes
-----
A family's arguments are checked when they are concrete. Inside a prior
function given others, which the prior traces and vmaps over draws, they
are traced and have no value to check, so the value checks are skipped
there. The prior builds each term at concrete ancestral draws too, where
they do run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters.support import OPEN_UNIT_INTERVAL, Interval, Support, bijector_for

__all__ = [
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


def log_normal(*, median: Any, geometric_sd: Any) -> tfd.LogNormal:
    """The log-normal of the given median and geometric standard deviation:

    .. math::

        \\log x \\sim \\mathcal N\\big(\\log \\mathrm{median},\\ (\\log \\mathrm{geometric\\_sd})^2\\big),

    so its central 95% interval is
    :math:`\\mathrm{median} \\cdot \\mathrm{geometric\\_sd}^{\\pm 1.96}`.

    Parameters
    ----------
    median:
        Positive; one value per label gives a batch.
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


def logit_normal(*, median: Any, logit_sd: Any, support: Interval = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The logit-normal on the finite interval *support*, :math:`(a, b)`:

    .. math::

        \\operatorname{logit}\\frac{x - a}{b - a} \\sim
            \\mathcal N\\Big(\\operatorname{logit}\\frac{\\mathrm{median} - a}{b - a},\\
            \\mathrm{logit\\_sd}^2\\Big).

    On :math:`(0, 1)` this is ``tfd.LogitNormal``; on :math:`(a, b)`,
    ``TransformedDistribution(Normal, Sigmoid(low=a, high=b))``. A
    ``logit_sd`` near 1.7 is close to flat. Its mass is on the interior
    whichever ends *support* closes.

    Raises
    ------
    TypeError
        If *support* is not an :class:`Interval`.
    ValueError
        If *median* is outside the interior of *support*, *logit_sd* is not
        positive, or an end of *support* is infinite.
    """
    check_support_is_a_finite_interval(support)
    fraction = _interval_fraction("logit_normal median", median, support)
    logit_sd = _positive_array("logit_normal logit_sd", logit_sd)
    return _logit_normal_on(support, _logit(fraction), logit_sd)


def logit_normal_from_interval(
    *, lower: Any, upper: Any, mass: float = 0.95, support: Interval = OPEN_UNIT_INTERVAL
) -> tfd.Distribution:
    """The logit-normal on *support* whose central *mass* interval is
    ``[lower, upper]``: :func:`log_normal_from_interval`'s formulas on the
    logit scale of :math:`(x - a)/(b - a)`.

    Raises
    ------
    TypeError, ValueError
        As :func:`logit_normal`, for an end outside *support*, ``upper <=
        lower``, or *mass* not in ``(0, 1)``.
    """
    check_support_is_a_finite_interval(support)
    lower = _interval_fraction("logit_normal_from_interval lower", lower, support)
    upper = _interval_fraction("logit_normal_from_interval upper", upper, support)
    loc, scale = _normal_from_interval(_logit(lower), _logit(upper), mass)
    return _logit_normal_on(support, loc, scale)


def logit_normal_from_samples(samples: Any, *, support: Interval = OPEN_UNIT_INTERVAL) -> tfd.Distribution:
    """The maximum-likelihood logit-normal on *support*, :math:`(a, b)`, of
    samples inside it: with :math:`t = \\operatorname{logit}((x - a)/(b - a))`,
    :math:`\\mu = \\bar t` and :math:`\\sigma` the standard deviation of
    :math:`t`.

    Raises
    ------
    TypeError, ValueError
        As :func:`log_normal_from_samples`, for samples outside the interior
        of *support*, and as :func:`logit_normal` for *support*.
    """
    check_support_is_a_finite_interval(support)
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
        for one per label.
    logit_sd:
        A scalar; one value per unconstrained number, ``(k - 1,)``; or, with
        an ``(n, k)`` center, one value per label, ``(n,)``, or one per
        label and unconstrained number, ``(n, k - 1)``.

    Raises
    ------
    ValueError
        For a center that is not a point of the simplex, or a *logit_sd* of
        another shape, of an ``(n,)`` shape that could be read either way
        (``n = k - 1``), or not positive.

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
    if loc.ndim == 2 and logit_sd.shape == loc.shape[:1]:
        logit_sd = logit_sd[:, None]  # one per label, across its numbers
    scale = jnp.broadcast_to(logit_sd, loc.shape)
    return tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), tfb.SoftmaxCentered()
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _logit_normal_on(support: Interval, loc: Array, scale: Array) -> tfd.Distribution:
    if (support.low, support.high) == (0.0, 1.0):
        return tfd.LogitNormal(loc=loc, scale=scale)
    return tfd.TransformedDistribution(tfd.Normal(loc, scale), bijector_for(support))


def _interval_fraction(what: str, value: Any, support: Interval) -> Array:
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
    check_interval_ends_and_mass_are_valid(lower, upper, mass)
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


def _is_traced(*arrays: Any) -> bool:
    """Whether any of *arrays* is a JAX tracer, whose values a check cannot
    read."""
    return any(isinstance(array, jax.core.Tracer) for array in arrays)


# ── checks ────────────────────────────────────────────────────────────────────


def check_geometric_sd_exceeds_one(geometric_sd: Array) -> None:
    """A geometric standard deviation exceeds 1, since its log is the scale."""
    if _is_traced(geometric_sd):
        return
    if not bool(jnp.all(geometric_sd > 1.0)):
        raise ValueError("log_normal: geometric_sd is at most 1, and it multiplies; give a value above 1.")


def check_values_are_positive(what: str, array: Array, value: Any) -> None:
    """A family builder's positive argument is finite and positive."""
    if _is_traced(array):
        return
    if not bool(jnp.all(jnp.isfinite(array)) and jnp.all(array > 0)):
        raise ValueError(f"{what} is {value!r}, which is not finite and positive; give a positive value.")


def check_values_are_inside(what: str, fraction: Array, value: Any, support: Support) -> None:
    """A logit-normal's median or end lies inside the interior of its support."""
    if _is_traced(fraction):
        return
    if not bool(jnp.all((fraction > 0) & (fraction < 1))):
        raise ValueError(f"{what} is {value!r}, outside the interior of {support.name}; give a value inside it.")


def check_support_is_a_finite_interval(support: Any) -> None:
    """A logit-normal's support is an interval with finite ends."""
    if not isinstance(support, Interval):
        raise TypeError(f"support must be an Interval, got {type(support).__name__}; use Interval(low, high).")
    if math.isinf(support.low) or math.isinf(support.high):
        raise ValueError(
            f"a logit-normal is on a finite interval, got support {support.name}; use log_normal on "
            "a half-line."
        )


def check_interval_ends_and_mass_are_valid(lower: Array, upper: Array, mass: float) -> None:
    """An interval's mass is in (0, 1) and its upper end exceeds its lower."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass is {mass}, outside (0, 1); give the central mass as a fraction.")
    if _is_traced(lower, upper):
        return
    if not bool(jnp.all(upper > lower)):
        raise ValueError("upper does not exceed lower; give the interval's ends in order.")


def check_samples_are_usable(what: str, array: Array, in_support: Callable[[Array], Array]) -> None:
    """Samples to fit are at least two, finite and inside the support."""
    if array.size < 2:
        raise ValueError(f"{what}: {array.size} sample(s) cannot be fitted; give at least two.")
    if _is_traced(array):
        return
    bad = ~(jnp.isfinite(array) & in_support(array))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{what}: {int(bad.sum())} of {array.size} samples are missing or outside the "
            "support; exclude and count them before fitting."
        )


def check_samples_vary(what: str, std: Array) -> None:
    """Samples to fit are not all equal, or no scale can be fitted."""
    if _is_traced(std):
        return
    if not bool(std > 0):
        raise ValueError(f"{what}: the samples are all equal, so no scale can be fitted; give samples that vary.")


def check_center_is_on_the_simplex(center: Array) -> None:
    """A softmax-normal's center is a point of the simplex, per label or shared."""
    if center.ndim not in (1, 2) or center.shape[-1] < 2:
        raise ValueError(f"softmax_normal: center has shape {center.shape}; give (k,) or (n, k) with k >= 2.")
    if _is_traced(center):
        return
    if not bool(jnp.all(center > 0)) or not bool(jnp.allclose(center.sum(axis=-1), 1.0, atol=1e-8)):
        raise ValueError("softmax_normal: center is not a point of the simplex; give positive fractions summing to 1.")


def check_logit_sd_fits_the_center(logit_sd: Array, loc: Array) -> None:
    """A ``logit_sd`` has a shape that reads one way against the center: an
    ``(n,)`` one equal to ``(k - 1,)`` would be laid along the wrong axis."""
    per_number, per_label = loc.shape[-1:], loc.shape[:-1]
    allowed = {(), per_number} | ({per_label, loc.shape} if loc.ndim == 2 else set())
    if logit_sd.shape not in allowed:
        raise ValueError(
            f"softmax_normal: logit_sd has shape {logit_sd.shape}, which fits the center of shape "
            f"{(*loc.shape[:-1], loc.shape[-1] + 1)} as none of {sorted(allowed)}; give a scalar, one "
            "value per unconstrained number, or, for a center per label, one per label or one per "
            "label and number."
        )
    if loc.ndim == 2 and per_label == per_number and logit_sd.shape == per_number:
        raise ValueError(
            f"softmax_normal: logit_sd of shape {logit_sd.shape} could be one value per label "
            "or one per unconstrained number, since there are as many of each; give it as "
            f"{loc.shape}."
        )

"""Supports: the sets values lie in, and their default bijections.

A :class:`Support` is a set of values: the set a component's values lie in.
Each support says whether a value is in it (``contains``), gives its closure,
and has a default bijection onto its interior. The supports implemented are
:class:`Interval`, a set of numbers, :class:`Simplex`, a set of vectors, and
:class:`PositiveDefinite`, a set of matrices.

A component's transform :math:`T` maps unconstrained space onto the
interior of its support; its default is the support's entry in
:data:`DEFAULT_BIJECTORS`, found by :func:`bijector_for`:

=============================== ===================================================
support                         default :math:`T(\\theta)`
=============================== ===================================================
:math:`(-\\infty, \\infty)`       :math:`\\theta` (``Identity``)
:math:`(0, \\infty)`             :math:`e^{\\theta}` (``Exp``)
:math:`(a, \\infty)`             :math:`a + e^{\\theta}` (``Shift(a)`` after ``Exp``)
:math:`(-\\infty, b)`            :math:`b - e^{\\theta}` (``Shift(b)`` after ``Scale(-1)`` after ``Exp``)
:math:`(0, 1)`                  :math:`\\sigma(\\theta) = 1 / (1 + e^{-\\theta})` (``Sigmoid()``)
:math:`(a, b)`                  :math:`a + (b - a)\\,\\sigma(\\theta)` (``Sigmoid(low=a, high=b)``)
the simplex, :math:`k` numbers  :math:`x_i = e^{\\theta_i} / (1 + \\sum_{j<k} e^{\\theta_j})` for
                                :math:`i < k`, :math:`x_k = 1 / (1 + \\sum_{j<k} e^{\\theta_j})`
                                (``SoftmaxCentered``, the inverse of the additive
                                log-ratio :math:`\\theta_i = \\log(x_i / x_k)`)
the positive-definite           :math:`L L^\\top`, with :math:`L` lower triangular, filled
:math:`p \\times p` matrices     from :math:`\\theta \\in \\mathbb R^{p(p+1)/2}` and its
                                diagonal exponentiated (``CholeskyOuterProduct`` after
                                ``FillScaleTriL(diag_bijector=Exp(), diag_shift=None)``)
=============================== ===================================================

A closed end of an interval changes nothing in this table: each bijector
maps onto the interior, and a density puts no mass on an endpoint.

Functions and classes
---------------------
:class:`Support`, :class:`Interval`, :class:`Simplex`, :class:`PositiveDefinite`
    The sets: ``contains``, ``closure``, ``event_ndims``, ``name``.
:data:`REAL`, :data:`POSITIVE`, :data:`NON_NEGATIVE`, :data:`OPEN_UNIT_INTERVAL`, :data:`UNIT_INTERVAL`, :data:`SIMPLEX`, :data:`POSITIVE_DEFINITE`
    The supports in common use.
:data:`DEFAULT_BIJECTORS`, :func:`bijector_for`
    The default transform of each type of support.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

__all__ = [
    "DEFAULT_BIJECTORS",
    "NON_NEGATIVE",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "POSITIVE_DEFINITE",
    "REAL",
    "SIMPLEX",
    "UNIT_INTERVAL",
    "Interval",
    "PositiveDefinite",
    "Simplex",
    "Support",
    "bijector_for",
    "check_interval_is_valid",
]

tfb = tfp.bijectors

Array = jax.Array


class Support(ABC):
    """A set of values.

    Attributes
    ----------
    event_ndims : int
        The rank of one element of the set: 0 for a set of numbers, 1 for a
        set of vectors such as the simplex, 2 for a set of matrices.
    name : str
        ``"real"``, ``"(0, inf)"``, ``"[0, 1]"``, ``"simplex"``,
        ``"positive definite"``, ...
    """

    event_ndims: int

    @property
    @abstractmethod
    def name(self) -> str:
        """The set, as tables and messages show it."""

    @abstractmethod
    def contains(self, values: Any) -> Array:
        """Whether each element of *values*, its last ``event_ndims`` axes,
        lies in the set; a non-finite element never does.

        Returns
        -------
        jax.Array
            ``bool``, of shape ``values.shape[: values.ndim - event_ndims]``.
        """

    @abstractmethod
    def closure(self) -> Support:
        """This set together with its boundary.

        Notes
        -----
        The probe checks take it, since float64 rounds a correct bijector's
        image onto the boundary at extreme points of unconstrained space.
        """


@dataclass(frozen=True)
class Interval(Support):
    """An interval of the real line, each end open or closed.

    Parameters
    ----------
    low, high:
        The ends, ``low < high``; either may be infinite.
    low_closed, high_closed:
        Whether each end belongs to the interval. An infinite end never
        does.

    Raises
    ------
    TypeError
        If an end is not a real number, or a closed flag not a boolean.
    ValueError
        If an end is NaN, ``low >= high``, or an infinite end is closed.
    """

    low: float = -math.inf
    high: float = math.inf
    low_closed: bool = False
    high_closed: bool = False

    event_ndims = 0

    def __post_init__(self) -> None:
        check_interval_is_valid(self)
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))

    @property
    def name(self) -> str:
        if math.isinf(self.low) and math.isinf(self.high):
            return "real"
        left = "[" if self.low_closed else "("
        right = "]" if self.high_closed else ")"
        return f"{left}{self.low:g}, {self.high:g}{right}"

    def contains(self, values: Any) -> Array:
        values = jnp.asarray(values, dtype=jnp.float64)
        above = values >= self.low if self.low_closed else values > self.low
        below = values <= self.high if self.high_closed else values < self.high
        return jnp.isfinite(values) & above & below

    def closure(self) -> Interval:
        return Interval(
            self.low, self.high, low_closed=math.isfinite(self.low), high_closed=math.isfinite(self.high)
        )


@dataclass(frozen=True)
class Simplex(Support):
    """The probability simplex over the last axis: numbers that are positive
    (non-negative when *closed*) and sum to 1, at least two of them.

    Parameters
    ----------
    closed:
        Whether values with a zero coordinate are included.
    """

    closed: bool = False

    event_ndims = 1

    @property
    def name(self) -> str:
        return "closed simplex" if self.closed else "simplex"

    def contains(self, values: Any) -> Array:
        """Every number finite and positive (non-negative when closed), and
        their sum within ``1e-10`` of 1, over the last axis: ``(..., k) ->
        (...)``."""
        values = jnp.asarray(values, dtype=jnp.float64)
        positive = values >= 0.0 if self.closed else values > 0.0
        inside = jnp.all(jnp.isfinite(values) & positive, axis=-1)
        return inside & (jnp.abs(values.sum(axis=-1) - 1.0) <= _SIMPLEX_SUM_TOLERANCE)

    def closure(self) -> Simplex:
        return Simplex(closed=True)


@dataclass(frozen=True)
class PositiveDefinite(Support):
    """The symmetric positive-definite matrices over the last two axes
    (positive semi-definite when *closed*).

    A density on the set is against Lebesgue measure on the
    :math:`p(p+1)/2` entries of the lower triangle. Theta's entries follow
    ``tfb.FillTriangular``'s packing of the Cholesky factor, its diagonal on
    the log scale.

    Parameters
    ----------
    closed:
        Whether singular positive semi-definite matrices are included.
    """

    closed: bool = False

    event_ndims = 2

    @property
    def name(self) -> str:
        return "positive semi-definite" if self.closed else "positive definite"

    def contains(self, values: Any) -> Array:
        """Whether each ``(..., p, p)`` matrix is finite, symmetric to a
        relative ``1e-10``, and has a Cholesky factor (when closed, has no
        eigenvalue below ``-1e-10`` times its largest magnitude): ``(..., p,
        p) -> (...)``."""
        values = jnp.asarray(values, dtype=jnp.float64)
        finite = jnp.all(jnp.isfinite(values), axis=(-2, -1))
        safe = jnp.where(jnp.isfinite(values), values, 0.0)
        scale = jnp.max(jnp.abs(safe), axis=(-2, -1))
        asymmetry = jnp.max(jnp.abs(safe - jnp.swapaxes(safe, -1, -2)), axis=(-2, -1))
        symmetric = asymmetry <= _SYMMETRY_TOLERANCE * scale
        symmetrized = (safe + jnp.swapaxes(safe, -1, -2)) / 2.0
        if self.closed:
            eigenvalues = jnp.linalg.eigvalsh(symmetrized)
            definite = eigenvalues[..., 0] >= -_SYMMETRY_TOLERANCE * jnp.max(jnp.abs(eigenvalues), axis=-1)
        else:
            definite = jnp.all(jnp.isfinite(jnp.linalg.cholesky(symmetrized)), axis=(-2, -1)) & (scale > 0.0)
        return finite & symmetric & definite

    def closure(self) -> PositiveDefinite:
        return PositiveDefinite(closed=True)


def bijector_for(
    support: Support, bijectors: Mapping[type, Callable[[Support], tfb.Bijector]] | None = None
) -> tfb.Bijector:
    """The bijector *bijectors* registers for *support*'s type, or for the
    nearest of its base classes that has one.

    Parameters
    ----------
    support:
        The support.
    bijectors:
        ``{type of support: factory}``, each factory taking the support and
        returning a bijector onto its interior; :data:`DEFAULT_BIJECTORS`
        when ``None``.

    Raises
    ------
    KeyError
        If neither *support*'s type nor a base class has an entry.
    """
    registry = DEFAULT_BIJECTORS if bijectors is None else bijectors
    for kind in type(support).__mro__:
        if kind in registry:
            return registry[kind](support)
    raise KeyError(
        f"no bijector is registered for a {type(support).__name__} support; register a factory "
        "for its type, or give the component a bijector."
    )


# ── helpers ───────────────────────────────────────────────────────────────────

#: How far a simplex value's sum may be from 1: float64 rounding of a
#: ``SoftmaxCentered`` image is about ``1e-16`` per number.
_SIMPLEX_SUM_TOLERANCE = 1e-10

#: How far a positive-definite value may be from symmetric, relative to its
#: largest entry: ``CholeskyOuterProduct`` rounds ``L L^T`` symmetrically, so
#: anything beyond rounding is a value that is not a covariance.
_SYMMETRY_TOLERANCE = 1e-10


def _interval_bijector(support: Interval) -> tfb.Bijector:
    """The default bijector onto an interval's interior, as the module's
    table gives it."""
    low, high = support.low, support.high
    if math.isinf(low) and math.isinf(high):
        return tfb.Identity()
    if math.isinf(high):
        return tfb.Exp() if low == 0.0 else tfb.Chain([tfb.Shift(jnp.float64(low)), tfb.Exp()])
    if math.isinf(low):
        return tfb.Chain([tfb.Shift(jnp.float64(high)), tfb.Scale(jnp.float64(-1.0)), tfb.Exp()])
    if (low, high) == (0.0, 1.0):
        return tfb.Sigmoid()
    return tfb.Sigmoid(low=jnp.float64(low), high=jnp.float64(high))


def _simplex_bijector(support: Simplex) -> tfb.Bijector:
    return tfb.SoftmaxCentered()


def _positive_definite_bijector(support: PositiveDefinite) -> tfb.Bijector:
    return tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)])


# ── checks ────────────────────────────────────────────────────────────────────


def check_interval_is_valid(interval: Interval) -> None:
    """An interval's ends are real numbers with ``low < high``, and an
    infinite end is open: a closed one would claim infinity as a value."""
    for end in (interval.low, interval.high):
        if isinstance(end, (bool, np.bool_)) or not isinstance(end, (int, float, np.integer, np.floating)):
            raise TypeError(f"an interval's ends are real numbers, got {end!r}; pass floats.")
    for flag in (interval.low_closed, interval.high_closed):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"an interval's low_closed and high_closed are booleans, got {flag!r}.")
    if math.isnan(interval.low) or math.isnan(interval.high) or not interval.low < interval.high:
        raise ValueError(
            f"an interval needs low < high, got ({interval.low}, {interval.high}); give its ends in "
            "order."
        )
    if (interval.low_closed and math.isinf(interval.low)) or (interval.high_closed and math.isinf(interval.high)):
        raise ValueError(
            f"the interval ({interval.low}, {interval.high}) closes an infinite end, which no value "
            "reaches; leave that end open."
        )


# ── the supports and the registry ─────────────────────────────────────────────
# Last, since building an Interval runs its check, defined above.

#: The real line, per number; its bijector is the identity.
REAL = Interval()

#: :math:`(0, \infty)`; its bijector is :math:`\exp`.
POSITIVE = Interval(0.0, math.inf)

#: :math:`[0, \infty)`, as pySIPNET's ``ParameterDomain.NON_NEGATIVE`` is.
NON_NEGATIVE = Interval(0.0, math.inf, low_closed=True)

#: :math:`(0, 1)`, as pySIPNET's ``ParameterDomain.OPEN_UNIT_INTERVAL`` is;
#: its bijector is ``tfb.Sigmoid()``.
OPEN_UNIT_INTERVAL = Interval(0.0, 1.0)

#: :math:`[0, 1]`, as pySIPNET's ``ParameterDomain.UNIT_INTERVAL`` is.
UNIT_INTERVAL = Interval(0.0, 1.0, low_closed=True, high_closed=True)

#: The open simplex: positive numbers summing to 1, over the last axis; its
#: bijector is ``tfb.SoftmaxCentered()``.
SIMPLEX = Simplex()

#: The symmetric positive-definite matrices, over the last two axes; its
#: bijector is the Cholesky factor with a log diagonal.
POSITIVE_DEFINITE = PositiveDefinite()

#: The default bijector of each type of support, as a factory from the
#: support to a bijector onto its interior (the module's table).
DEFAULT_BIJECTORS: Mapping[type, Callable[[Any], tfb.Bijector]] = frozendict(
    {Interval: _interval_bijector, Simplex: _simplex_bijector, PositiveDefinite: _positive_definite_bijector}
)

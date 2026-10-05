"""Laws: the concrete distributions a factor evaluates to, one draw at a
time.

Where this sits
---------------
::

    probability.support                       (the supports and their bijectors)
      -> probability.laws                     (Law, as_law, pushforward)
      -> probability.families, builders       (laws of one value, and over a block)
      -> probability.parts.FactorSpec         (a conditional law over components)

A **law** is what the layer requires of a distribution: TFP's ``log_prob``
and ``sample``, with ``log_prob`` against the reference measure of its
event's support (Lebesgue measure per number on an interval, on the first
:math:`k - 1` coordinates on the simplex, on the lower triangle for a
positive-definite matrix). TFP distributions are laws as they are; a law
from another package is an object that implements :class:`Law` itself.

Functions and classes
---------------------
:class:`Law`
    The protocol.
:func:`as_law`
    A distribution as a law, or a ``TypeError``.
:func:`pushforward`
    The law of :math:`T(u)`, :math:`u` drawn from a base law, through a
    support's bijector or a given one.
:func:`is_law`
    Whether an object is a law, not a class or a function making one.
:func:`distribution_name`
    A short name for a law, for a description.
:class:`GaussianLaw`
    A Gaussian over a block, holding a structured covariance: what a
    :class:`~sipnet_calibration.probability.parts.GaussianSpec` evaluates
    to, and what :func:`as_law` makes of a ``pyeki.gauss.Gaussian``.
:data:`CARRIES_ITS_BIJECTOR`
    The TFP classes whose ``.distribution`` and ``.bijector`` a model reads.

Notes
-----
Adapters for EnsKit's ``Gaussian`` and numpyro's distributions, which
:func:`as_law` will recognize by class, come with the foreign-law PR (P9);
until then they are given as objects implementing :class:`Law`. Today's
``pyeki.gauss.Gaussian``, which EnsKit's replaces, is adapted already, as
a :class:`GaussianLaw`.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import _linalg
from sipnet_calibration.probability.support import Support, bijector_for

__all__ = [
    "CARRIES_ITS_BIJECTOR",
    "GaussianLaw",
    "Law",
    "as_law",
    "distribution_name",
    "is_law",
    "pushforward",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

#: TFP's classes whose ``.distribution`` and ``.bijector`` are a base in theta
#: and a map from it. Subclasses such as ``MultivariateNormalTriL`` and
#: :class:`~sipnet_calibration.probability.families.InverseWishart` carry an
#: internal reparameterization instead, so only the exact classes count.
CARRIES_ITS_BIJECTOR = (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)


@runtime_checkable
class Law(Protocol):
    """What the layer requires of a factor's law: TFP's ``log_prob`` and
    ``sample``, with ``log_prob`` against the reference measure of the
    event's support. TFP distributions satisfy it; others go through
    :func:`as_law`."""

    def log_prob(self, value: Array) -> Array: ...

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array: ...


def as_law(distribution: Any) -> Law:
    """*distribution* as a :class:`Law`.

    A TFP distribution is returned unchanged, and so is any other object
    that implements :class:`Law` (a callable ``log_prob`` and ``sample``).
    A ``pyeki.gauss.Gaussian`` becomes a :class:`GaussianLaw` over its
    ``(n,)`` vector.

    Raises
    ------
    TypeError
        If *distribution* is none of these.
    """
    if isinstance(distribution, _linalg.Gaussian):
        return GaussianLaw(distribution.mean, distribution.cov)
    check_distribution_is_a_law(distribution)
    return distribution


def is_law(distribution: Any) -> bool:
    """Whether *distribution* is a law: a TFP distribution, or an instance
    with a callable ``log_prob`` and ``sample``. A class is not one, even a
    TFP distribution class, whose methods are callable on the class too."""
    if isinstance(distribution, type):
        return False
    if isinstance(distribution, tfd.Distribution):
        return True
    return callable(getattr(distribution, "log_prob", None)) and callable(getattr(distribution, "sample", None))


def pushforward(base: Any, *, support: Support | None = None, bijector: tfb.Bijector | None = None) -> Law:
    """The law of :math:`T(u)`, :math:`u \\sim` *base*, with :math:`T` the
    *bijector*, or ``bijector_for(support)``; exactly one of the two.

    A factor whose law is a pushforward through its components' own
    bijectors is evaluated by the base density, exactly, with no Jacobian.

    Parameters
    ----------
    base:
        A TFP distribution: the law of :math:`u`, unconstrained.
    support, bijector:
        Keyword-only. Where :math:`T` comes from.

    Returns
    -------
    Law
        ``tfd.TransformedDistribution(base, T)``, of that exact class.

    Raises
    ------
    TypeError
        If both or neither of *support* and *bijector* are given, *base* is
        not a TFP distribution, or *bijector* is not a TFP bijector.
    KeyError
        If *support*'s type has no default bijector.
    """
    check_one_of_support_and_bijector_is_given(support, bijector)
    check_base_is_a_tfp_distribution(base)
    if bijector is None:
        bijector = bijector_for(support)
    check_bijector_is_a_tfp_bijector(bijector)
    return tfd.TransformedDistribution(base, bijector)


def distribution_name(distribution: Any) -> str:
    """A short name for a law, for a description: the family builders'
    names, or the class and its bijector."""
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


class GaussianLaw:
    """:math:`\\mathcal N(m, \\Sigma)` over a block, its entries in C
    order: a :class:`Law` holding a structured covariance.

    Parameters
    ----------
    mean : ArrayLike
        Positional-only. :math:`m`, of the block's shape.
    covariance : PSDLinOp
        Positional-only. :math:`\\Sigma` over the block's ``n`` entries: one
        of pyEKI's positive-definite operators (``pyeki.linalg.PSDLinOp``),
        which must support ``whiten``, ``logdet`` and ``factor``.

    Attributes
    ----------
    mean : jax.Array
        Of the block's shape.
    covariance : PSDLinOp
    event_shape : tuple of int
        The block's shape.
    gaussian : pyeki.gauss.Gaussian
        The same law over the flattened block, as pyEKI holds one.

    Raises
    ------
    TypeError
        If *covariance* is not a ``PSDLinOp``.
    ValueError
        If *covariance* is not ``(n, n)`` over the block's ``n`` entries.

    Notes
    -----
    A covariance that is not positive definite gives a density that is not
    finite, not an error: its operator's precondition is not checked when
    it is built (:mod:`~sipnet_calibration.probability.covariance`).
    """

    __slots__ = ("mean", "covariance", "gaussian")

    def __init__(self, mean: Any, covariance: Any, /) -> None:
        mean = jnp.asarray(mean, dtype=jnp.float64)
        check_covariance_is_an_operator(covariance)
        check_covariance_is_over_the_block(covariance, mean.shape)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "gaussian", _linalg.Gaussian(mean.reshape((-1,)), covariance))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a GaussianLaw is frozen; build another rather than setting {name!r}.")

    def __repr__(self) -> str:
        return f"GaussianLaw(event_shape={self.event_shape}, covariance={type(self.covariance).__name__})"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.mean.shape)

    @property
    def dtype(self) -> Any:
        return jnp.float64

    def log_prob(self, value: Any) -> Array:
        """The log density at *value*, ``(..., *block) -> (...)``,

        .. math::

            -\\tfrac12 \\big(n \\log 2\\pi + \\log\\det\\Sigma
                + \\lVert W (x - m) \\rVert^2\\big),

        :math:`W` a whitener of :math:`\\Sigma` (``Gaussian.log_density``)."""
        value = jnp.asarray(value, dtype=jnp.float64)
        lead = value.shape[: value.ndim - self.mean.ndim]
        return self.gaussian.log_density(value.reshape((*lead, -1)))

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws :math:`m + L z`, :math:`z \\sim \\mathcal N(0, I)` and
        :math:`L` the covariance's factor, ``(*sample_shape, *block)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed)
        sample_shape = tuple(sample_shape) if isinstance(sample_shape, (tuple, list)) else (int(sample_shape),)
        factor = self.covariance.factor()
        noise = jax.random.normal(seed, (*sample_shape, factor.shape[1]), dtype=jnp.float64)
        draws = self.mean.reshape((-1,)) + factor.matvec(noise)
        return draws.reshape((*sample_shape, *self.mean.shape))


# ── checks ────────────────────────────────────────────────────────────────────


def check_distribution_is_a_law(distribution: Any) -> None:
    """A law is a TFP distribution, or implements :class:`Law`."""
    if not is_law(distribution):
        raise TypeError(
            f"a {type(distribution).__name__} is not a law; give a TFP distribution, or an object "
            "with log_prob(value) and sample(sample_shape, seed=key)."
        )


def check_one_of_support_and_bijector_is_given(support: Any, bijector: Any) -> None:
    """A pushforward names its map once: a support, whose bijector it takes,
    or a bijector."""
    if (support is None) == (bijector is None):
        raise TypeError(
            "pushforward takes exactly one of support= and bijector=; give the component's support to "
            "push through its own bijector."
        )


def check_base_is_a_tfp_distribution(base: Any) -> None:
    """A pushforward's base is a TFP distribution, the one kind of law it
    can wrap until the foreign-law adapters exist."""
    if not isinstance(base, tfd.Distribution):
        raise TypeError(
            f"pushforward's base is a {type(base).__name__}; give a TFP distribution of the "
            "unconstrained values."
        )


def check_bijector_is_a_tfp_bijector(bijector: Any) -> None:
    """A pushforward's map is a TFP bijector."""
    if not isinstance(bijector, tfb.Bijector):
        raise TypeError(f"pushforward's bijector is a {type(bijector).__name__}; give a TFP bijector.")


def check_covariance_is_an_operator(covariance: Any) -> None:
    """A Gaussian law's covariance is a positive-definite operator."""
    if not isinstance(covariance, _linalg.PSDLinOp):
        raise TypeError(
            f"a GaussianLaw's covariance is a {type(covariance).__name__}; give a positive-definite operator, "
            "such as one a covariance spec builds."
        )


def check_covariance_is_over_the_block(covariance: Any, shape: tuple[int, ...]) -> None:
    """A Gaussian law's covariance is over its block's entries."""
    size = math.prod(shape)
    if tuple(covariance.shape) != (size, size):
        raise ValueError(
            f"a GaussianLaw's covariance is {tuple(covariance.shape)}, but its mean of shape {tuple(shape)} has "
            f"{size} entries; give a ({size}, {size}) covariance."
        )


def check_seed_is_given(seed: Any) -> None:
    """A draw is made from a key."""
    if seed is None:
        raise TypeError("a GaussianLaw draws from a key; give seed=jax.random.key(...).")

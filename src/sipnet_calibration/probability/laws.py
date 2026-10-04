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
:func:`distribution_name`
    A short name for a law, for a description.

Notes
-----
Adapters for EnsKit's ``Gaussian`` and numpyro's distributions, which
:func:`as_law` will recognize by class, come with the foreign-law PR (P9);
until then they are given as objects implementing :class:`Law`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability.support import Support, bijector_for

__all__ = [
    "CARRIES_ITS_BIJECTOR",
    "Law",
    "as_law",
    "distribution_name",
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

    Raises
    ------
    TypeError
        If *distribution* is neither.
    """
    check_distribution_is_a_law(distribution)
    return distribution


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


# ── checks ────────────────────────────────────────────────────────────────────


def check_distribution_is_a_law(distribution: Any) -> None:
    """A law is a TFP distribution, or implements :class:`Law`."""
    if isinstance(distribution, tfd.Distribution):
        return
    if not (callable(getattr(distribution, "log_prob", None)) and callable(getattr(distribution, "sample", None))):
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

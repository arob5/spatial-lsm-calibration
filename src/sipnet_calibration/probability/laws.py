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
positive-definite matrix). TFP distributions are laws as they are. Two
other kinds are adapted by :func:`as_law`, which a model applies to every
law a factor is given or builds: EnsKit's ``Gaussian`` becomes a
:class:`GaussianLaw`, and a numpyro distribution, GPJax's
``GaussianDistribution`` among them, a :class:`NumpyroLaw`. A law from any
other package is an object that implements :class:`Law` itself.

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
    to, and what :func:`as_law` makes of an EnsKit ``Gaussian`` of one
    block.
:class:`NumpyroLaw`
    A numpyro distribution as a law.
:class:`PushforwardLaw`
    What :func:`pushforward` makes of a base from another package.
:data:`CARRIES_ITS_BIJECTOR`
    The TFP classes whose ``.distribution`` and ``.bijector`` a model reads.

Notes
-----
numpyro is not a dependency: a numpyro distribution is recognized by the
names of the classes in its MRO, so the layer never imports numpyro, and a
model holds one only when its author has numpyro installed. EnsKit's
``Gaussian`` comes through the private ``_linalg`` shim, which is where
:func:`as_law` finds the class.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import _linalg, _numpyro
from sipnet_calibration.probability.support import Support, bijector_for

__all__ = [
    "CARRIES_ITS_BIJECTOR",
    "GaussianLaw",
    "Law",
    "NumpyroLaw",
    "PushforwardLaw",
    "as_law",
    "distribution_name",
    "is_law",
    "pushforward",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

# The name of the one block of the EnsKit Gaussian a GaussianLaw holds.
_BLOCK_NAME = "block"

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

    A TFP distribution is returned unchanged. Two others are adapted, found
    by class: an ``enskit.distribution.Gaussian`` of one block becomes a
    :class:`GaussianLaw` over its ``(n,)`` vector, with the block's
    covariance, and a numpyro distribution (an instance of
    ``numpyro.distributions.Distribution``, GPJax's ``GaussianDistribution``
    among them) a :class:`NumpyroLaw`. Any other object that implements
    :class:`Law`, a callable ``log_prob`` and ``sample``, is returned
    unchanged.

    Raises
    ------
    TypeError
        If *distribution* is none of these.
    ValueError
        If an EnsKit ``Gaussian`` has more than one block, or its block has
        no independent term.

    Notes
    -----
    A numpyro distribution is tested for before the protocol: its
    ``log_prob`` and ``sample`` are callable, but its ``sample`` takes the
    key first.
    """
    if isinstance(distribution, tfd.Distribution):
        return distribution
    if isinstance(distribution, _linalg.Gaussian):
        check_gaussian_has_one_block(distribution)
        (name,) = distribution.names
        check_gaussian_block_has_an_independent_term(distribution, name)
        return GaussianLaw(distribution.mean(name), distribution.cov(name))
    if _numpyro.is_numpyro_distribution(distribution):
        return NumpyroLaw(distribution)
    check_distribution_is_a_law(distribution)
    return distribution


def is_law(distribution: Any) -> bool:
    """Whether *distribution* is a law, or one :func:`as_law` adapts: a TFP
    distribution, an EnsKit ``Gaussian``, a numpyro distribution, or an
    instance with a callable ``log_prob`` and ``sample``. A class is not
    one, even a TFP distribution class, whose methods are callable on the
    class too."""
    if isinstance(distribution, type):
        return False
    if isinstance(distribution, (tfd.Distribution, _linalg.Gaussian)) or _numpyro.is_numpyro_distribution(distribution):
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
        The law of :math:`u`, unconstrained: a TFP distribution, or any law
        :func:`as_law` adapts, such as a numpyro distribution.
    support, bijector:
        Keyword-only. Where :math:`T` comes from.

    Returns
    -------
    Law
        For a TFP *base*, ``tfd.TransformedDistribution(base, T)``, of that
        exact class; for any other, a :class:`PushforwardLaw` of the adapted
        base.

    Raises
    ------
    TypeError
        If both or neither of *support* and *bijector* are given, *base* is
        not a law, a base that is not TFP's has no ``event_shape``, or
        *bijector* is not a TFP bijector.
    KeyError
        If *support*'s type has no default bijector.
    """
    check_one_of_support_and_bijector_is_given(support, bijector)
    check_base_is_a_law(base)
    if bijector is None:
        bijector = bijector_for(support)
    check_bijector_is_a_tfp_bijector(bijector)
    if isinstance(base, tfd.Distribution):
        return tfd.TransformedDistribution(base, bijector)
    base = as_law(base)
    check_base_has_an_event_shape(base)
    return PushforwardLaw(base, bijector)


def distribution_name(distribution: Any) -> str:
    """A short name for a law, for a description: the family builders'
    names, or the class and its bijector."""
    kind = type(distribution)
    if kind is NumpyroLaw:
        return f"numpyro {type(distribution.distribution).__name__}"
    if kind is PushforwardLaw:
        return f"{distribution_name(distribution.distribution)} through {distribution.bijector.name}"
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
        of EnsKit's positive-definite operators (``enskit.linalg.PSDLinOp``),
        which must support ``whiten``, ``logdet`` and ``factor``.

    Attributes
    ----------
    mean : jax.Array
        Of the block's shape.
    covariance : PSDLinOp
    event_shape : tuple of int
        The block's shape.
    gaussian : enskit.distribution.Gaussian
        The same law over the flattened block, as EnsKit holds one: one
        block, whose independent term is :math:`\\Sigma`.

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
        gaussian = _linalg.Gaussian({_BLOCK_NAME: mean.reshape((-1,))}, block_covs={_BLOCK_NAME: covariance})
        object.__setattr__(self, "gaussian", gaussian)

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
        return self.gaussian.log_density({_BLOCK_NAME: value.reshape((*lead, -1))})

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws :math:`m + L z`, :math:`z \\sim \\mathcal N(0, I)` and
        :math:`L` the covariance's factor, ``(*sample_shape, *block)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed, what="GaussianLaw")
        sample_shape = _as_sample_shape(sample_shape)
        factor = self.covariance.factor()
        noise = jax.random.normal(seed, (*sample_shape, factor.shape[1]), dtype=jnp.float64)
        draws = self.mean.reshape((-1,)) + factor.matvec(noise)
        return draws.reshape((*sample_shape, *self.mean.shape))


class NumpyroLaw:
    """A numpyro distribution as a :class:`Law`.

    Parameters
    ----------
    distribution
        Positional-only. An instance of
        ``numpyro.distributions.Distribution``, such as GPJax's
        ``GaussianDistribution``.

    Attributes
    ----------
    distribution
        The numpyro distribution.
    event_shape, batch_shape : tuple of int
        Its own.
    dtype
        The dtype of its draws, or of a floating parameter that is not
        ``float64``.

    Raises
    ------
    TypeError
        If *distribution* is not a numpyro distribution.

    Notes
    -----
    numpyro's ``sample(key, sample_shape)`` becomes ``sample(sample_shape,
    seed=key)``. ``log_prob`` is vmapped over the axes in front of the
    distribution's batch and event, on which a model batches its values and
    which some numpyro distributions, GPJax's among them, do not broadcast.
    """

    __slots__ = ("distribution",)

    def __init__(self, distribution: Any, /) -> None:
        check_distribution_is_numpyro(distribution)
        object.__setattr__(self, "distribution", distribution)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a NumpyroLaw is frozen; build another rather than setting {name!r}.")

    def __repr__(self) -> str:
        return f"NumpyroLaw({type(self.distribution).__name__}, event_shape={self.event_shape})"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.distribution.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.distribution.batch_shape)

    @property
    def dtype(self) -> Any:
        # A numpyro draw is float64 under x64 whatever its parameters, so a
        # float32 parameter, which the density is computed in, is read off the
        # leaves.
        for leaf in jax.tree_util.tree_leaves(self.distribution):
            dtype = getattr(leaf, "dtype", None)
            if dtype is not None and jnp.issubdtype(dtype, jnp.floating) and dtype != jnp.float64:
                return dtype
        return jax.eval_shape(self.distribution.sample, jax.random.key(0)).dtype

    def log_prob(self, value: Any) -> Array:
        """The distribution's ``log_prob`` at *value*, ``(..., *batch,
        *event) -> (..., *batch)``."""
        value = jnp.asarray(value)
        ndim = len(self.batch_shape) + len(self.event_shape)
        if value.ndim <= ndim:
            return self.distribution.log_prob(value)
        lead = value.shape[: value.ndim - ndim]
        flat = value.reshape((-1, *value.shape[value.ndim - ndim :]))
        return jax.vmap(self.distribution.log_prob)(flat).reshape((*lead, *self.batch_shape))

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws, ``(*sample_shape, *batch, *event)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed, what="NumpyroLaw")
        return self.distribution.sample(seed, _as_sample_shape(sample_shape))


class PushforwardLaw:
    """The law of :math:`x = T(u)`, :math:`u` drawn from a law that is not
    TFP's: what :func:`pushforward` makes of such a base,

    .. math::

        \\log p_x(x) = \\log p_u\\big(T^{-1}(x)\\big)
            + \\log \\left|\\det \\frac{\\partial T^{-1}(x)}{\\partial x}\\right|,

    the log-determinant being TFP's ``inverse_log_det_jacobian`` summed over
    the base's event. A model evaluates a factor whose law is a pushforward
    through its components' own bijectors by the base density instead,
    exactly.

    Parameters
    ----------
    distribution : Law
        Positional-only. The law of :math:`u`, with an ``event_shape``.
    bijector : tfb.Bijector
        Positional-only. :math:`T`.

    Notes
    -----
    The constructor checks nothing; :func:`pushforward` is the checked way
    to make one.

    Attributes
    ----------
    distribution, bijector
        As given.
    event_shape : tuple of int
        :math:`T`'s image of the base's event shape.
    batch_shape : tuple of int
        The base's, ``()`` if it has none.
    dtype
        The base's, or that of its draws where it has none.
    """

    __slots__ = ("distribution", "bijector")

    def __init__(self, distribution: Law, bijector: tfb.Bijector, /) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "bijector", bijector)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a PushforwardLaw is frozen; build another rather than setting {name!r}.")

    def __repr__(self) -> str:
        return f"PushforwardLaw({distribution_name(self.distribution)}, {self.bijector.name})"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.bijector.forward_event_shape(self.distribution.event_shape))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.distribution, "batch_shape", ()))

    @property
    def dtype(self) -> Any:
        if hasattr(self.distribution, "dtype"):
            return self.distribution.dtype
        return jax.eval_shape(lambda key: self.distribution.sample((), seed=key), jax.random.key(0)).dtype

    def log_prob(self, value: Any) -> Array:
        """The log density at *value*, by change of variables, ``(...,
        *event) -> (...)``."""
        value = jnp.asarray(value, dtype=jnp.float64)
        # TFP's event_ndims is the rank of the value, T's output, which
        # differs from the base's where T reshapes (FillScaleTriL).
        log_jacobian = self.bijector.inverse_log_det_jacobian(value, event_ndims=len(self.event_shape))
        return self.distribution.log_prob(self.bijector.inverse(value)) + log_jacobian

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """:math:`T(u)` at draws of :math:`u`, ``(*sample_shape, *event)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed, what="PushforwardLaw")
        return self.bijector.forward(self.distribution.sample(_as_sample_shape(sample_shape), seed=seed))


def _as_sample_shape(sample_shape: Any) -> tuple[int, ...]:
    """A sample shape given as an integer or a sequence, as a tuple."""
    return tuple(sample_shape) if isinstance(sample_shape, (tuple, list)) else (int(sample_shape),)


# ── checks ────────────────────────────────────────────────────────────────────


def check_distribution_is_a_law(distribution: Any) -> None:
    """A law is a TFP distribution, one :func:`as_law` adapts, or an object
    implementing :class:`Law`."""
    if not is_law(distribution):
        raise TypeError(
            f"a {type(distribution).__name__} is not a law; give a TFP or numpyro distribution, an EnsKit "
            "Gaussian, or an object with log_prob(value) and sample(sample_shape, seed=key)."
        )


def check_one_of_support_and_bijector_is_given(support: Any, bijector: Any) -> None:
    """A pushforward names its map once: a support, whose bijector it takes,
    or a bijector."""
    if (support is None) == (bijector is None):
        raise TypeError(
            "pushforward takes exactly one of support= and bijector=; give the component's support to "
            "push through its own bijector."
        )


def check_base_is_a_law(base: Any) -> None:
    """A pushforward's base is a law, or one :func:`as_law` adapts."""
    if not is_law(base):
        raise TypeError(
            f"pushforward's base is a {type(base).__name__}; give a law of the unconstrained values, such as "
            "a TFP or numpyro distribution."
        )


def check_base_has_an_event_shape(base: Any) -> None:
    """A pushforward's base says its event shape, which the bijector's
    log-Jacobian is summed over."""
    if getattr(base, "event_shape", None) is None:
        raise TypeError(
            f"pushforward's base, a {type(base).__name__}, has no event_shape, which its bijector's "
            "log-Jacobian is summed over; give the law an event_shape attribute."
        )


def check_distribution_is_numpyro(distribution: Any) -> None:
    """A numpyro law adapts a numpyro distribution."""
    if not _numpyro.is_numpyro_distribution(distribution):
        raise TypeError(
            f"a NumpyroLaw adapts a numpyro distribution, not a {type(distribution).__name__}; give the "
            "distribution to as_law, which adapts what it recognizes."
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


def check_gaussian_has_one_block(gaussian: Any) -> None:
    """An EnsKit ``Gaussian`` adapted to a law is over one block."""
    if len(gaussian.names) != 1:
        raise ValueError(
            f"an EnsKit Gaussian adapted to a law is over one block, not {len(gaussian.names)} "
            f"({', '.join(map(repr, gaussian.names))}); give the marginal of one, gaussian.marginal(name)."
        )


def check_gaussian_block_has_an_independent_term(gaussian: Any, name: str) -> None:
    """An EnsKit ``Gaussian`` adapted to a law has an independent term on its
    block, without which its covariance is the low-rank :math:`F F^\\top`."""
    if gaussian.block_cov(name) is None:
        raise ValueError(
            f"the EnsKit Gaussian's block {name!r} has no independent term, so its covariance is the low-rank "
            "F F^T, with no density over the block; add one with gaussian.add_noise(...)."
        )


def check_seed_is_given(seed: Any, *, what: str) -> None:
    """A draw is made from a key."""
    if seed is None:
        raise TypeError(f"a {what} draws from a key; give seed=jax.random.key(...).")

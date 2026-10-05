"""The builders: a factor's law over a block of values, built for the
labels in use.

Where this sits
---------------
::

    probability.families               (one value's law)
      -> probability.builders          (a law over a block)
      -> probability.parts.FactorSpec  (one factor of a model)

What a builder is
-----------------
A :class:`Builder` is a factor's law for the labels in use: the model calls
it as ``builder(index_shape, **reads)``: the index shape is the number of
labels in use of each dim the factor's components are indexed by, which a
builder needs because, for labels independent and identically distributed,
nothing it reads has those dims; ``reads`` are the values it names. It returns a law of TFP batch
shape ``()`` over the factor's blocks. Each builder exposes what it wraps,
:attr:`Builder.law`, and what it reads, :attr:`Builder.reads`, so a model
reads its given components and constants off it by the keyword rule.

The builders
------------
:func:`iid_over_dim`
    One value's law, repeated over the block:
    :math:`p(x) = \\prod_\\ell p_0(x_\\ell)`.
:func:`independent_over_dim`
    Each label's law from its own arguments:
    :math:`p(x) = \\prod_\\ell p_{\\mathrm{family}}(x_\\ell; a_\\ell)`.
:func:`gaussian_copula`
    A joint law over scalar components indexed by nothing, its marginals
    kept and their unconstrained values correlated.

They wrap TFP laws only. A transformed law is built with its bijector
outside the product over labels, so a model evaluates it by its base
density.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
from frozendict import frozendict
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._keywords import function_reads
from sipnet_calibration.probability.laws import CARRIES_ITS_BIJECTOR, distribution_name

__all__ = [
    "Builder",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array


class Builder(ABC):
    """A factor's law for the labels in use, called by the model with the
    index shape first, ``builder(index_shape, **reads)``.

    Attributes
    ----------
    name : str
        A short name for a description.
    law : Any
        What it wraps: the law per label, the function or family that makes
        it, or a copula's marginals.
    reads : tuple of str
        The names it reads, by the keyword rule.
    """

    name: str

    @property
    @abstractmethod
    def law(self) -> Any: ...

    @property
    @abstractmethod
    def reads(self) -> tuple[str, ...]: ...

    @abstractmethod
    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution: ...


def iid_over_dim(distribution: tfd.Distribution | Callable[..., tfd.Distribution]) -> Builder:
    """Every label of the index dims independently and identically
    distributed: one value's law :math:`p_0`, repeated over the block,

    .. math::

        p(x) = \\prod_{\\ell} p_0(x_\\ell),

    over the labels :math:`\\ell` of the product of the component's index
    dims. A ``TransformedDistribution(base, b)`` (a family builder's, say)
    is built as ``TransformedDistribution(Sample(base, index_shape), b)``,
    with the bijector outside, so that a model evaluates it by its base
    density; any other law as ``Sample(distribution, index_shape)``.

    Parameters
    ----------
    distribution:
        The law of one value, of TFP batch shape ``()`` and event shape the
        component's ``shape``: a TFP distribution, or a function of what it
        reads returning one. What the function reads is the same at every
        label, such as a spread indexed by nothing:
        ``iid_over_dim(lambda spread: tfd.Normal(0.0, spread))``.

    Returns
    -------
    Builder
        Its :attr:`~Builder.law` is *distribution*; it reads nothing, or what
        the function reads.

    Raises
    ------
    TypeError
        When called for a component indexed by nothing, or, for a fixed law,
        given values it would ignore; when :attr:`~Builder.reads` is asked
        of a function breaking the keyword rule.
    """
    if isinstance(distribution, tfd.Distribution):
        name = f"iid {distribution_name(distribution)}"

        def fixed(index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
            check_fixed_law_reads_nothing(name, reads)
            return distribution

        return _OverDim(per_label=fixed, repeated=True, name=name, wrapped=distribution, reads_of=None)
    return _OverDim(
        per_label=lambda index_shape, **reads: distribution(**reads),
        repeated=True,
        name=f"iid {getattr(distribution, '__name__', 'function')}",
        wrapped=distribution,
        reads_of=distribution,
    )


def independent_over_dim(distribution_family: Callable[..., tfd.Distribution], /, **arguments: Any) -> Builder:
    """Independent across the labels of the index dims, each label's law
    made by *distribution_family* from its own arguments:

    .. math::

        p(x) = \\prod_{\\ell} p_{\\mathrm{family}}\\big(x_\\ell;\\ a_\\ell\\big).

    Parameters
    ----------
    distribution_family:
        Builds a law from keyword arguments, such as
        :func:`~sipnet_calibration.probability.families.log_normal`, giving
        it the TFP batch shape of the index when an argument has one entry
        per label.
    **arguments:
        Arguments shared by every label, passed as they are. An argument per
        label is a constant of the factor (``FactorSpec(constants=...)``),
        passed to the family by its name, its leading axes the component's
        index dims.

    Returns
    -------
    Builder
        Its :attr:`~Builder.law` is the family; it reads the family's
        parameters without defaults that *arguments* leaves unset. A
        transformed family is built with its bijector outside, as for
        :func:`iid_over_dim`.

    Raises
    ------
    TypeError
        When called for a component indexed by nothing; when
        :attr:`~Builder.reads` is asked of a family breaking the keyword rule.
    ValueError
        When called, if the family's law is not a batch of one per label, as
        when no argument is per label (:func:`iid_over_dim` is that law).
    """

    def per_label(index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        distribution = distribution_family(**arguments, **reads)
        check_family_is_one_per_label(distribution, index_shape)
        return distribution

    return _OverDim(
        per_label=per_label,
        repeated=False,
        name=f"independent {getattr(distribution_family, '__name__', 'family')}",
        wrapped=distribution_family,
        reads_of=distribution_family,
        bound=tuple(arguments),
    )


def gaussian_copula(marginals: Mapping[str, tfd.Distribution], *, correlation: Any) -> Builder:
    """The joint law of scalar components indexed by nothing whose
    marginals are *marginals* and whose unconstrained values are correlated
    Gaussians.

    Each marginal :math:`i` is a family builder's scalar law, a
    pushforward :math:`x_i = T_i(t_i)` of :math:`t_i \\sim \\mathcal
    N(\\mu_i, \\sigma_i^2)`. The copula keeps the marginals and correlates
    the :math:`t_i`:

    .. math::

        t \\sim \\mathcal N\\big(\\mu,\\ \\mathrm{diag}(\\sigma)\\, R\\,
            \\mathrm{diag}(\\sigma)\\big), \\qquad x_i = T_i(t_i),

    with :math:`R` the *correlation*, in the order of *marginals*. In theta
    this is that Gaussian exactly when every component's bijector is its
    marginal's :math:`T_i`, and a model then evaluates it by its base
    density.

    Parameters
    ----------
    marginals:
        ``{component name: law}``, each
        :func:`~sipnet_calibration.probability.families.log_normal`,
        :func:`~sipnet_calibration.probability.families.logit_normal` (on any
        finite interval), their ``_from_*`` forms, or a ``tfd.Normal``, with
        TFP batch shape ``()``. List them in the order of the factor's
        event, which is the order its base is laid out in.
    correlation:
        ``(m, m)``: symmetric, unit diagonal, positive definite.

    Returns
    -------
    Builder
        For components indexed by nothing, reading nothing; its draws are
        dicts keyed like *marginals*, and its :attr:`~Builder.law` is
        *marginals*.

    Raises
    ------
    ValueError
        If a marginal is not a scalar pushforward of a Gaussian, or
        *correlation* is not a correlation matrix of that size.
    """
    names = tuple(marginals)
    families = []
    for name in names:
        family = _marginal_gaussian(marginals[name])
        check_copula_marginal_is_a_scalar_gaussian_pushforward(name, marginals[name], family)
        families.append(family)
    correlation = jnp.asarray(correlation, dtype=jnp.float64)
    check_correlation_is_a_correlation_matrix(correlation, len(names))
    return _GaussianCopula(
        names=names,
        loc=jnp.stack([jnp.asarray(f[0], dtype=jnp.float64) for f in families]),
        scale=jnp.stack([jnp.asarray(f[1], dtype=jnp.float64) for f in families]),
        bijectors=tuple(f[2] for f in families),
        correlation=correlation,
        marginals=frozendict(marginals),
    )


# ── private: what the builders return ─────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _OverDim(Builder):
    """A law over the index dims.

    ``per_label(index_shape, **reads)`` gives one label's law when
    *repeated*, else every label's as a batch of the index shape. What it
    reads is ``reads_of``'s keywords less those *bound*; ``reads_of`` is
    ``None`` for a fixed law.
    """

    per_label: Callable[..., tfd.Distribution]
    repeated: bool
    name: str
    wrapped: Any
    reads_of: Callable[..., Any] | None
    bound: tuple[str, ...] = field(default=())

    @property
    def law(self) -> Any:
        return self.wrapped

    @cached_property
    def reads(self) -> tuple[str, ...]:
        # Read when a factor asks, so a builder made for the parameter
        # layer, which passes everything a prior term names, never meets it.
        if self.reads_of is None:
            return ()
        reads = function_reads(self.reads_of, message_name=f"{self.name}'s law")
        return tuple(name for name in reads if name not in self.bound)

    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        check_component_has_index_dims(index_shape, self.name)
        distribution = self.per_label(index_shape, **reads)
        if type(distribution) in CARRIES_ITS_BIJECTOR:
            base = distribution.distribution
            base = tfd.Sample(base, index_shape) if self.repeated else tfd.Independent(base, len(index_shape))
            return tfd.TransformedDistribution(base, distribution.bijector)
        if self.repeated:
            return tfd.Sample(distribution, index_shape)
        return tfd.Independent(distribution, len(index_shape))


@dataclass(frozen=True, eq=False)
class _GaussianCopula(Builder):
    """:func:`gaussian_copula`'s builder."""

    names: tuple[str, ...]
    loc: Array
    scale: Array
    bijectors: tuple[tfb.Bijector, ...]
    correlation: Array
    marginals: Mapping[str, tfd.Distribution]
    name: str = "gaussian copula"

    @property
    def law(self) -> Mapping[str, tfd.Distribution]:
        return self.marginals

    @property
    def reads(self) -> tuple[str, ...]:
        return ()

    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        check_components_have_no_index_dims(index_shape, self.name)
        check_fixed_law_reads_nothing(self.name, reads)
        to_values = tfb.Chain([
            tfb.JointMap({n: tfb.Chain([b, tfb.Reshape([], [1])]) for n, b in zip(self.names, self.bijectors)}),
            tfb.Restructure({n: i for i, n in enumerate(self.names)}),
            tfb.Split(len(self.names)),
        ])
        base = tfd.MultivariateNormalTriL(self.loc, jnp.linalg.cholesky(self._covariance))
        return tfd.TransformedDistribution(base, to_values)

    @property
    def _covariance(self) -> Array:
        return self.scale[:, None] * self.correlation * self.scale[None, :]


def _marginal_gaussian(distribution: tfd.Distribution) -> tuple[Array, Array, tfb.Bijector] | None:
    """A copula marginal, a pushforward of a Normal, as the Normal's mean and
    standard deviation, broadcast together, and its bijector; ``None`` for any
    other law. Recognized by exact class."""
    if type(distribution) is tfd.Normal:
        loc, scale, bijector = distribution.loc, distribution.scale, tfb.Identity()
    elif type(distribution) in (tfd.LogNormal, tfd.LogitNormal) or _is_pushforward(
        distribution, tfd.Normal, tfb.Sigmoid
    ):
        base = distribution.distribution
        loc, scale, bijector = base.loc, base.scale, distribution.bijector
    else:
        return None
    loc, scale = jnp.broadcast_arrays(jnp.asarray(loc), jnp.asarray(scale))
    return loc, scale, bijector


def _is_pushforward(distribution: tfd.Distribution, base_class: type, bijector_class: type) -> bool:
    """Whether *distribution* is exactly a ``TransformedDistribution`` of an
    exact *base_class* through an exact *bijector_class*."""
    return (
        type(distribution) is tfd.TransformedDistribution
        and type(distribution.distribution) is base_class
        and type(distribution.bijector) is bijector_class
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_component_has_index_dims(index_shape: tuple[int, ...], name: str) -> None:
    """A law over the index dims is given to a component indexed by some."""
    if not index_shape:
        raise TypeError(
            f"{name} is a law over a component's index dims, given to a component indexed by "
            "nothing; give that component the law itself."
        )


def check_fixed_law_reads_nothing(name: str, reads: Mapping[str, Any]) -> None:
    """A builder's fixed law is given nothing and reads no constant, which
    it would otherwise ignore."""
    if reads:
        raise TypeError(
            f"{name} is a fixed law, but it is given {sorted(reads)}, which it would ignore; write a "
            "function of them returning a law, or, for labels independent and identically distributed, "
            "iid_over_dim(lambda spread: tfd.Normal(0.0, spread))."
        )


def check_components_have_no_index_dims(index_shape: tuple[int, ...], name: str) -> None:
    """A law of components indexed by nothing is not given indexed ones,
    whose draws would otherwise be refused for their shape, far from the
    reason."""
    if index_shape:
        raise TypeError(
            f"{name} is a law of components indexed by nothing, given components of index shape "
            f"{index_shape}; give those components a law over their index dims."
        )


def check_family_is_one_per_label(distribution: Any, index_shape: tuple[int, ...]) -> None:
    """A family's law is a batch of one per label, which the labels would
    otherwise be misaligned with."""
    batch = tuple(getattr(distribution, "batch_shape", ()))
    if batch != tuple(index_shape):
        raise ValueError(
            f"independent_over_dim built a distribution of TFP batch shape {batch} for the index "
            f"shape {tuple(index_shape)}; give at least one argument per label, as a constant of "
            "the factor, or use iid_over_dim for one law shared by every label."
        )


def check_copula_marginal_is_a_scalar_gaussian_pushforward(
    name: str, marginal: Any, family: tuple[Array, Array, tfb.Bijector] | None
) -> None:
    """A copula's marginal is a family builder's scalar law, a pushforward
    of a Normal whose argument the copula correlates."""
    if family is None or tuple(marginal.batch_shape) != () or tuple(marginal.event_shape) != ():
        raise ValueError(
            f"gaussian_copula's marginal {name!r} is not a scalar pushforward of a Gaussian; give "
            "log_normal, logit_normal, their _from_* forms, or tfd.Normal, with TFP batch shape ()."
        )


def check_correlation_is_a_correlation_matrix(correlation: Array, n_marginals: int) -> None:
    """A copula's correlation is an ``(m, m)`` symmetric, unit-diagonal,
    positive definite matrix: TFP would otherwise read its lower triangle, or
    fail its Cholesky factor, without an error."""
    shape = (n_marginals, n_marginals)
    if (
        correlation.shape != shape
        or not bool(jnp.allclose(correlation, correlation.T, rtol=0.0, atol=1e-12))
        or not bool(jnp.allclose(jnp.diag(correlation), 1.0, rtol=0.0, atol=1e-12))
        or not bool(jnp.all(jnp.linalg.eigvalsh(correlation) > 0.0))
    ):
        raise ValueError(
            f"gaussian_copula's correlation must be a {shape} symmetric, positive definite matrix "
            f"with unit diagonal, got shape {correlation.shape}; give a correlation matrix in the "
            "order of the marginals."
        )

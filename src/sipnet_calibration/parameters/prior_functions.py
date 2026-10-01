"""The builders of prior functions: a prior term's distribution over its
parameters' blocks, built for the labels in use.

Where this sits
---------------
::

    parameters.families                (one value's distribution)
      -> parameters.prior_functions    (a distribution over a block)
      -> parameters.prior.PriorTerm    (one factor of the prior)

What a prior function is
------------------------
A :data:`PriorFunction` is called as ``f(**given, **constants)`` and
returns the distribution of a term's event, its parameters' blocks;
:class:`~sipnet_calibration.parameters.prior.PriorTerm` states what it
receives and returns. A user's function reads its block's size off what it
reads. The builders here also receive the index shape, the number of
labels in use of each dim the parameters are indexed by, from the prior,
since for labels independent and identically distributed nothing they read
carries it.

The builders
------------
:func:`iid_over_dim`
    One value's distribution, repeated over the block:
    :math:`\\pi(x) = \\prod_\\ell \\pi_0(x_\\ell)`.
:func:`independent_over_dim`
    Each label's distribution from its own arguments:
    :math:`\\pi(x) = \\prod_\\ell \\pi_{\\mathrm{family}}(x_\\ell; a_\\ell)`.
:func:`gaussian_copula`
    A joint term over scalar parameters indexed by nothing, its marginals
    kept and its unconstrained values correlated.

A transformed distribution is built with its bijector outside the product
over labels, so the prior evaluates it by its base density.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.parameters._distributions import (
    CARRIES_ITS_BIJECTOR,
    IndexShapedFunction,
    distribution_name,
)

__all__ = [
    "PriorFunction",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

#: A function building a term's distribution for the labels in use: one a
#: user writes is called as ``f(**given, **constants)``, as
#: :class:`~sipnet_calibration.parameters.prior.PriorTerm` says; one a builder
#: here returns is also passed the index shape, first.
type PriorFunction = Callable[..., tfd.Distribution]


def iid_over_dim(distribution: tfd.Distribution | Callable[..., tfd.Distribution]) -> PriorFunction:
    """Every label of the index dims independently and identically
    distributed: one value's distribution :math:`\\pi_0`, repeated over the
    block,

    .. math::

        \\pi(x) = \\prod_{\\ell} \\pi_0(x_\\ell),

    over the labels :math:`\\ell` of the product of the parameter's index
    dims. A ``TransformedDistribution(base, b)`` (a family builder's, say)
    is built as ``TransformedDistribution(Sample(base, index_shape), b)``,
    with the bijector outside, so that
    :meth:`~sipnet_calibration.parameters.prior.Prior.log_prob` evaluates it
    by its base density; any other distribution as ``Sample(distribution,
    index_shape)``.

    Parameters
    ----------
    distribution:
        The distribution of one value, of TFP batch shape ``()`` and event
        shape the parameter's ``shape``: a TFP distribution, or, for a term
        given others or reading constants, a function
        ``f(**given, **constants)`` returning one. What the function reads
        is the same at every label, such as a spread indexed by nothing,
        given as ``iid_over_dim(lambda spread: tfd.Normal(0.0, spread))``.

    Returns
    -------
    PriorFunction
        A builder's, which the prior passes the index shape.
    """
    if isinstance(distribution, tfd.Distribution):
        name = f"iid {distribution_name(distribution)}"

        def fixed(index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
            check_fixed_prior_reads_nothing(name, reads)
            return distribution

        return _OverDim(per_label=fixed, repeated=True, name=name)
    return _OverDim(
        per_label=lambda index_shape, **reads: distribution(**reads),
        repeated=True,
        name=f"iid {getattr(distribution, '__name__', 'function')}",
    )


def independent_over_dim(distribution_family: Callable[..., tfd.Distribution], /, **arguments: Any) -> PriorFunction:
    """Independent across the labels of the index dims, each label's
    distribution made by *distribution_family* from its own arguments:

    .. math::

        \\pi(x) = \\prod_{\\ell} \\pi_{\\mathrm{family}}\\big(x_\\ell;\\ a_\\ell\\big).

    Parameters
    ----------
    distribution_family:
        Builds a distribution from keyword arguments, such as
        :func:`~sipnet_calibration.parameters.families.log_normal`, giving
        it the TFP batch shape of the index when an argument has one entry
        per label.
    **arguments:
        Arguments shared by every label, passed as they are. An argument per
        label is a constant of the term (``PriorTerm(constants=...)``),
        passed to the family by its name, its leading axes the parameter's
        index dims.

    Returns
    -------
    PriorFunction
        A builder's, which the prior passes the index shape. A transformed
        family is built with its bijector outside, as for
        :func:`iid_over_dim`.

    Raises
    ------
    ValueError
        When called, if the family's distribution is not a batch of one per
        label, as when no argument is per label (:func:`iid_over_dim` is that
        prior).
    """

    def per_label(index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        distribution = distribution_family(**arguments, **reads)
        check_family_is_one_per_label(distribution, index_shape)
        return distribution

    return _OverDim(
        per_label=per_label,
        repeated=False,
        name=f"independent {getattr(distribution_family, '__name__', 'family')}",
    )


def gaussian_copula(marginals: Mapping[str, tfd.Distribution], *, correlation: Any) -> PriorFunction:
    """The joint prior of scalar parameters indexed by nothing whose
    marginals are *marginals* and whose unconstrained values are correlated
    Gaussians.

    Each marginal :math:`i` is a family builder's scalar distribution, a
    pushforward :math:`x_i = T_i(t_i)` of :math:`t_i \\sim \\mathcal
    N(\\mu_i, \\sigma_i^2)`. The copula keeps the marginals and correlates
    the :math:`t_i`:

    .. math::

        t \\sim \\mathcal N\\big(\\mu,\\ \\mathrm{diag}(\\sigma)\\, R\\,
            \\mathrm{diag}(\\sigma)\\big), \\qquad x_i = T_i(t_i),

    with :math:`R` the *correlation*, in the order of *marginals*. In theta
    this is that Gaussian exactly when every parameter's bijector is its
    marginal's :math:`T_i`, and the prior then evaluates it by its base
    density.

    Parameters
    ----------
    marginals:
        ``{parameter name: distribution}``, each
        :func:`~sipnet_calibration.parameters.families.log_normal`,
        :func:`~sipnet_calibration.parameters.families.logit_normal` (on any
        finite interval), their ``_from_*`` forms, or a ``tfd.Normal``, with
        TFP batch shape ``()``. List them in
        the order of the term's ``parameter_names``, which is the order its
        base is laid out in.
    correlation:
        ``(m, m)``: symmetric, unit diagonal, positive definite.

    Returns
    -------
    PriorFunction
        A builder's, for parameters indexed by nothing; its draws are dicts
        keyed like
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
    )


# ── private: what the builders return ─────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _OverDim(IndexShapedFunction):
    """A prior function over the index dims.

    ``per_label(index_shape, **reads)`` gives one label's distribution
    when *repeated*, else every label's as a batch of the index shape.
    """

    per_label: Callable[..., tfd.Distribution]
    repeated: bool
    name: str

    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        check_prior_over_a_dim_has_a_dim(index_shape, self.name)
        distribution = self.per_label(index_shape, **reads)
        if type(distribution) in CARRIES_ITS_BIJECTOR:
            base = distribution.distribution
            base = tfd.Sample(base, index_shape) if self.repeated else tfd.Independent(base, len(index_shape))
            return tfd.TransformedDistribution(base, distribution.bijector)
        if self.repeated:
            return tfd.Sample(distribution, index_shape)
        return tfd.Independent(distribution, len(index_shape))


@dataclass(frozen=True, eq=False)
class _GaussianCopula(IndexShapedFunction):
    """:func:`gaussian_copula`'s prior function."""

    names: tuple[str, ...]
    loc: Array
    scale: Array
    bijectors: tuple[tfb.Bijector, ...]
    correlation: Array
    name: str = "gaussian copula"

    def __call__(self, index_shape: tuple[int, ...], **reads: Any) -> tfd.Distribution:
        check_prior_without_a_dim_has_none(index_shape, self.name)
        check_fixed_prior_reads_nothing(self.name, reads)
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
    other distribution. Recognized by exact class."""
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


def check_prior_over_a_dim_has_a_dim(index_shape: tuple[int, ...], name: str) -> None:
    """A prior over the index dims is given to a parameter indexed by some."""
    if not index_shape:
        raise TypeError(
            f"{name} is a prior over a parameter's index dims, given to a parameter indexed by "
            "nothing; give that parameter the distribution itself."
        )


def check_fixed_prior_reads_nothing(name: str, reads: Mapping[str, Any]) -> None:
    """A builder's fixed prior is given nothing and reads no constant, which
    it would otherwise ignore."""
    if reads:
        raise TypeError(
            f"{name} is a fixed prior, but its term is given or reads {sorted(reads)}, which it would "
            "ignore; write a prior function of them, or, for labels independent and identically "
            "distributed, iid_over_dim(lambda spread: tfd.Normal(0.0, spread))."
        )


def check_prior_without_a_dim_has_none(index_shape: tuple[int, ...], name: str) -> None:
    """A prior of parameters indexed by nothing is not given indexed ones,
    whose draws would otherwise be refused for their shape, far from the
    reason."""
    if index_shape:
        raise TypeError(
            f"{name} is a prior of parameters indexed by nothing, given parameters of index shape "
            f"{index_shape}; give those parameters a prior over their index dims."
        )


def check_family_is_one_per_label(distribution: Any, index_shape: tuple[int, ...]) -> None:
    """A family's distribution is a batch of one per label, which the labels
    would otherwise be misaligned with."""
    batch = tuple(getattr(distribution, "batch_shape", ()))
    if batch != tuple(index_shape):
        raise ValueError(
            f"independent_over_dim built a distribution of TFP batch shape {batch} for the index "
            f"shape {tuple(index_shape)}; give at least one argument per label, as a constant of "
            "the term, or use iid_over_dim for one distribution shared by every label."
        )


def check_copula_marginal_is_a_scalar_gaussian_pushforward(
    name: str, marginal: Any, family: tuple[Array, Array, tfb.Bijector] | None
) -> None:
    """A copula's marginal is a family builder's scalar distribution, a
    pushforward of a Normal whose argument the copula correlates."""
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

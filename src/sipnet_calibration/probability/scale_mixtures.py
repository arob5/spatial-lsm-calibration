"""Gaussian scale mixtures: the laws a Gaussian factor becomes when the
variance scaling its covariance, or the covariance block it reads, is
integrated out under a conjugate prior.

Where this sits
---------------
::

    probability.covariance, labels
      -> probability.scale_mixtures      (StudentTSpec, MatrixStudentTSpec, their laws)
      -> probability.parts.FactorSpec    (a factor's law)
      -> probability.conjugacy           (marginalize makes them)

A spec here is a factor's law, as a
:class:`~sipnet_calibration.probability.parts.GaussianSpec` is: centered on
a mean component, over one component on ``REAL`` indexed by one dim at
most, with no element axes, so its entries are its labels in order. Bound
to the labels in use, it groups the entries; at each draw it builds its
law from the mean and what its covariance reads.

The laws
--------
A **Student-t**, :class:`StudentTSpec`, over groups :math:`s` of the
entries, independent across groups:

.. math::

    z_s \\sim t_{2a_s}\\big(m_s,\\ (b_s/a_s)\\,\\Sigma_s\\big), \\qquad
    \\log p(z_s) = \\log\\Gamma(a_s + \\tfrac{n_s}{2}) - \\log\\Gamma(a_s)
        + a_s \\log b_s - \\tfrac{n_s}{2}\\log 2\\pi - \\tfrac12 \\log|\\Sigma_s|
        - (a_s + \\tfrac{n_s}{2}) \\log(b_s + \\tfrac{q_s}{2}),

:math:`n_s` entries, :math:`q_s = r_s^\\top \\Sigma_s^{-1} r_s` and
:math:`r_s = z_s - m_s`. It is the law of :math:`z_s \\mid v_s \\sim
\\mathcal N(m_s, v_s \\Sigma_s)` with :math:`v_s \\sim \\mathrm{IG}(a_s,
b_s)` integrated out, which is how it is drawn. With no grouping the event
is one group.

A **matrix Student-t**, :class:`MatrixStudentTSpec`, over groups
:math:`i = 1, \\dots, n` whose entries map one to one onto the same
:math:`q` rows :math:`O` of a :math:`p \\times p` matrix:

.. math::

    \\log p(r_{1:n}) = -\\tfrac{nq}{2}\\log\\pi + \\log\\Gamma_q\\big(\\tfrac{\\nu'+n}{2}\\big)
        - \\log\\Gamma_q\\big(\\tfrac{\\nu'}{2}\\big) + \\tfrac{\\nu'}{2}\\log|\\Psi_{OO}|
        - \\tfrac{\\nu'+n}{2}\\log|\\Psi_{OO} + E|,

:math:`r_i` group :math:`i`'s residual in :math:`O`'s order,
:math:`E = \\sum_i r_i r_i^\\top` and :math:`\\nu' = \\nu - (p - q)`. It is
the law of :math:`r_i \\mid S \\sim \\mathcal N(0, S_{OO})`, independent,
with :math:`S \\sim \\mathcal W^{-1}_p(\\nu, \\Psi)` integrated out; then
:math:`S_{OO} \\sim \\mathcal W^{-1}_q(\\nu', \\Psi_{OO})`.

Classes
-------
:class:`StudentTSpec`, :class:`StudentTLaw`
    The Student-t, declared and at one draw.
:class:`MatrixStudentTSpec`, :class:`MatrixStudentTLaw`
    The matrix Student-t, declared and at one draw.

Notes
-----
The specs hold their prior's numbers as labeled arrays, read by label when
bound: a shape and scale per group's label, an inverse Wishart's scale per
row and column label. So the law at fewer labels is the marginal of the law
at more, and :math:`O` is found again at the labels in use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special
import numpy as np
import pandas as pd
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._validation import as_names, truncated
from sipnet_calibration.probability.covariance import (
    CovarianceSpec,
    _group_keys,
    _groups,
    _Scope,
    _slice_to_group,
    check_by_names_levels_of_the_event_dim,
    check_event_has_a_dim,
    check_group_is_contiguous,
    check_grouping_names_distinct_levels,
    check_terms_are_covariance_specs,
)
from sipnet_calibration.probability.labels import aligned_label_maps
from sipnet_calibration.probability.laws import check_seed_is_given

__all__ = [
    "MatrixStudentTLaw",
    "MatrixStudentTSpec",
    "StudentTLaw",
    "StudentTSpec",
    "inverse_wishart_log_prob",
    "sample_inverse_wishart",
]

tfd = tfp.distributions

Array = jax.Array


class StudentTSpec:
    """A Student-t law centered on a component, for a :class:`FactorSpec
    <sipnet_calibration.probability.parts.FactorSpec>`'s ``law``: each group
    :math:`s` of the event's entries, independently,
    :math:`t_{2a_s}(m_s, (b_s/a_s)\\Sigma_s)` (the module docstring).

    Parameters
    ----------
    mean : str
        Keyword-only. The component or input the event is centered on, of
        the event's layout and units.
    covariance : CovarianceSpec
        Keyword-only. :math:`\\Sigma_s`, over each group's entries (its
        scope).
    shape, scale : float or xr.DataArray
        Keyword-only. :math:`a_s` and :math:`b_s`, positive: one number for
        every group, or with *by* a DataArray on the dim named *by*, read at
        each group's label.
    by : str, optional
        Keyword-only. A level of the event's stacked dim: the entries
        sharing their label there form a group. Default: the event is one
        group.

    Attributes
    ----------
    mean : tuple of str
        The one mean.
    covariance : CovarianceSpec
    shape, scale : xr.DataArray
        0-d, or on *by*.
    by : str or None
    reads : tuple of str
        The mean, then what the covariance reads.

    Raises
    ------
    TypeError
        If *mean* or *by* is not a name, *covariance* not a covariance spec,
        or *shape* or *scale* neither a number nor a DataArray.
    ValueError
        If *shape* or *scale* is not positive and finite, or a DataArray of
        them is on another dim than *by*, or unlabeled; and, when bound, as
        a grouping is refused by
        :class:`~sipnet_calibration.probability.covariance.BlockDiagonalSpec`.
    KeyError
        When bound, if *shape* or *scale* lacks a group's label.
    """

    __slots__ = ("mean", "covariance", "shape", "scale", "by")

    mean: tuple[str, ...]
    covariance: CovarianceSpec
    shape: xr.DataArray
    scale: xr.DataArray
    by: str | None

    def __init__(
        self, *, mean: str, covariance: CovarianceSpec, shape: Any, scale: Any, by: str | None = None
    ) -> None:
        check_name_is_one_name(mean, what="StudentTSpec's mean")
        check_terms_are_covariance_specs([covariance], what="StudentTSpec's covariance")
        if by is not None:
            check_name_is_one_name(by, what="StudentTSpec's by")
        _set(self, "mean", (mean,))
        _set(self, "covariance", covariance)
        _set(self, "shape", _per_group(shape, by, what="StudentTSpec's shape"))
        _set(self, "scale", _per_group(scale, by, what="StudentTSpec's scale"))
        _set(self, "by", by)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a StudentTSpec is frozen; build another rather than setting {name!r}.")

    @property
    def reads(self) -> tuple[str, ...]:
        """The mean, then what the covariance reads."""
        return tuple(dict.fromkeys((*self.mean, *self.covariance.reads)))

    def __repr__(self) -> str:
        by = "" if self.by is None else f", by={self.by!r}"
        return f"StudentTSpec(mean={self.mean[0]!r}, covariance={self.covariance!r}{by})"

    def _at(self, scope: _Scope) -> _StudentTAtLabels:
        """This spec at the factor's labels: its groups, each with its
        covariance bound and its shape and scale."""
        if self.by is None:
            groups = [_StudentTGroup(
                positions=np.arange(scope.size, dtype=np.int64), by={}, label=None,
                covariance=self.covariance._at(scope), shape=float(self.shape), scale=float(self.scale),
            )]
            return _StudentTAtLabels(groups=tuple(groups), event_dim=scope.event_dim, read_dims=scope.read_dims)
        check_event_has_a_dim(scope, what="StudentTSpec grouped by a level")
        check_by_names_levels_of_the_event_dim((self.by,), scope)
        groups = []
        for key, positions in _groups(_group_keys((self.by,), scope)):
            check_group_is_contiguous(scope, (self.by,), key, positions)
            inner = scope.group(positions, by=(self.by,), key=key)
            groups.append(_StudentTGroup(
                positions=positions, by=inner.by_positions, label=key[0], covariance=self.covariance._at(inner),
                shape=_at_label(self.shape, self.by, key[0], what="shape", factor_name=scope.factor_name),
                scale=_at_label(self.scale, self.by, key[0], what="scale", factor_name=scope.factor_name),
            ))
        return _StudentTAtLabels(groups=tuple(groups), event_dim=scope.event_dim, read_dims=scope.read_dims)


class StudentTLaw:
    """:math:`\\prod_s t_{2a_s}(m_s, (b_s/a_s)\\Sigma_s)` over a block, its
    entries in C order: a :class:`~sipnet_calibration.probability.laws.Law`.

    Parameters
    ----------
    mean : ArrayLike
        Positional-only. :math:`m`, of the block's shape.
    groups : Sequence of (positions, covariance, shape, scale)
        Positional-only. Each group's entries (``int64``, into the flattened
        block; together every entry once), :math:`\\Sigma_s` as a
        positive-definite operator over them, :math:`a_s` and :math:`b_s`.

    Attributes
    ----------
    mean : jax.Array
    groups : tuple
    covariances : tuple of PSDLinOp
        Each group's :math:`\\Sigma_s`.
    event_shape : tuple of int
    """

    __slots__ = ("mean", "groups")

    def __init__(self, mean: Any, groups: Sequence[tuple[np.ndarray, Any, Any, Any]], /) -> None:
        object.__setattr__(self, "mean", jnp.asarray(mean, dtype=jnp.float64))
        object.__setattr__(self, "groups", tuple(groups))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a StudentTLaw is frozen; build another rather than setting {name!r}.")

    def __repr__(self) -> str:
        return f"StudentTLaw(event_shape={self.event_shape}, groups={len(self.groups)})"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.mean.shape)

    @property
    def dtype(self) -> Any:
        return jnp.float64

    @property
    def covariances(self) -> tuple[Any, ...]:
        return tuple(covariance for _, covariance, _, _ in self.groups)

    def log_prob(self, value: Any) -> Array:
        """The log density at *value*, ``(..., *block) -> (...)``, the module
        docstring's sum over groups."""
        value = jnp.asarray(value, dtype=jnp.float64)
        lead = value.shape[: value.ndim - self.mean.ndim]
        residual = value.reshape((*lead, -1)) - self.mean.reshape((-1,))
        total = jnp.zeros(lead, dtype=jnp.float64)
        for positions, covariance, shape, scale in self.groups:
            n = len(positions)
            whitened = covariance.whiten(residual[..., positions])
            quadratic = jnp.sum(whitened**2, axis=-1)
            total = total + (
                jax.scipy.special.gammaln(shape + n / 2.0) - jax.scipy.special.gammaln(shape) + shape * jnp.log(scale)
                - n / 2.0 * jnp.log(2.0 * jnp.pi) - covariance.logdet() / 2.0
                - (shape + n / 2.0) * jnp.log(scale + quadratic / 2.0)
            )
        return total

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws :math:`m_s + \\sqrt{v_s} L_s \\varepsilon`, :math:`v_s \\sim
        \\mathrm{IG}(a_s, b_s)`, :math:`\\varepsilon \\sim \\mathcal N(0, I)`
        and :math:`L_s` the group's covariance factor, ``(*sample_shape,
        *block)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed, what="StudentTLaw")
        sample_shape = tuple(sample_shape) if isinstance(sample_shape, (tuple, list)) else (int(sample_shape),)
        draws = jnp.broadcast_to(self.mean.reshape((-1,)), (*sample_shape, self.mean.size))
        for k, (positions, covariance, shape, scale) in enumerate(self.groups):
            variance_key, noise_key = jax.random.split(jax.random.fold_in(seed, k))
            variance = tfd.InverseGamma(jnp.float64(shape), jnp.float64(scale)).sample(sample_shape, seed=variance_key)
            factor = covariance.factor()
            noise = jax.random.normal(noise_key, (*sample_shape, factor.shape[1]), dtype=jnp.float64)
            draws = draws.at[..., positions].add(jnp.sqrt(variance)[..., None] * factor.matvec(noise))
        return draws.reshape((*sample_shape, *self.mean.shape))


class MatrixStudentTSpec:
    """A matrix Student-t law centered on a component, for a
    :class:`FactorSpec <sipnet_calibration.probability.parts.FactorSpec>`'s
    ``law``: each group's residual :math:`r_i \\sim \\mathcal N(0, S_{OO})`,
    independently, with :math:`S \\sim \\mathcal W^{-1}_p(\\nu, \\Psi)`
    integrated out (the module docstring).

    Parameters
    ----------
    mean : str
        Keyword-only. As :class:`StudentTSpec`'s.
    by : str or Sequence[str]
        Keyword-only. The grouping, as
        :class:`~sipnet_calibration.probability.covariance.BlockDiagonalSpec`'s.
    label_map : xr.DataArray
        Keyword-only. For each label of the event's dim, the row of
        :math:`S` its entry takes: on the event's dim, named for the row
        axis of *scale*.
    degrees_of_freedom : float
        Keyword-only. :math:`\\nu > p - 1`.
    scale : xr.DataArray
        Keyword-only. :math:`\\Psi`, ``(p, p)`` positive definite, on two
        axes holding the same labels, rows first.

    Attributes
    ----------
    mean : tuple of str
    by : tuple of str
    label_map, scale : xr.DataArray
    degrees_of_freedom : float
    reads : tuple of str
        The mean.

    Raises
    ------
    TypeError
        If *mean* is not a name, *by* not one or more, *label_map* or
        *scale* not a DataArray, or *degrees_of_freedom* not a number.
    ValueError
        If *scale* is not square on two axes of the same labels, or not
        positive definite; *degrees_of_freedom* is not above ``p - 1``;
        *label_map* is not one-dimensional or not named for the rows of
        *scale*; and, when bound, as a grouping is refused by
        ``BlockDiagonalSpec``, a group maps two entries to one row, or the
        groups map to different rows.
    KeyError
        When bound, if *label_map* lacks a label in use or maps one to a
        label *scale* lacks.
    """

    __slots__ = ("mean", "by", "label_map", "degrees_of_freedom", "scale")

    mean: tuple[str, ...]
    by: tuple[str, ...]
    label_map: xr.DataArray
    degrees_of_freedom: float
    scale: xr.DataArray

    def __init__(
        self, *, mean: str, by: str | Sequence[str], label_map: xr.DataArray, degrees_of_freedom: Any, scale: xr.DataArray
    ) -> None:
        check_name_is_one_name(mean, what="MatrixStudentTSpec's mean")
        by = (by,) if isinstance(by, str) else as_names(by, message_name="MatrixStudentTSpec's by")
        check_grouping_names_distinct_levels(by)
        check_is_a_data_array(label_map, what="MatrixStudentTSpec's label_map")
        check_is_a_data_array(scale, what="MatrixStudentTSpec's scale")
        check_degrees_of_freedom_is_a_number(degrees_of_freedom)
        check_scale_is_a_labeled_positive_definite_matrix(scale)
        check_degrees_of_freedom_exceed_the_dimension(float(degrees_of_freedom), scale.shape[0])
        check_label_map_names_the_rows(label_map, scale)
        _set(self, "mean", (mean,))
        _set(self, "by", by)
        _set(self, "label_map", _read_only(label_map))
        _set(self, "degrees_of_freedom", float(degrees_of_freedom))
        _set(self, "scale", _read_only(scale.astype(np.float64)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a MatrixStudentTSpec is frozen; build another rather than setting {name!r}.")

    @property
    def reads(self) -> tuple[str, ...]:
        """The mean."""
        return self.mean

    @property
    def rows(self) -> str:
        """The row axis of :math:`\\Psi`."""
        return str(self.scale.dims[0])

    def __repr__(self) -> str:
        by = self.by[0] if len(self.by) == 1 else list(self.by)
        return f"MatrixStudentTSpec(mean={self.mean[0]!r}, by={by!r}, rows={self.rows!r})"

    def _at(self, scope: _Scope) -> _MatrixStudentTAtLabels:
        """This spec at the factor's labels: each group's entries in the
        order of the rows :math:`O` they map to."""
        check_event_has_a_dim(scope, what="MatrixStudentTSpec")
        row_labels = self.scale.indexes[self.rows]
        (rows,) = aligned_label_maps(
            {"label_map": self.label_map}, {scope.event_dim: scope.labels},
            element_axes={self.rows: row_labels}, message_name=f"the matrix Student-t of {scope.factor_name!r}",
        ).values()
        groups = _groups(_group_keys(self.by, scope))
        observed = _shared_rows(scope.factor_name, groups, rows, row_labels)
        entries = np.stack([positions[np.argsort(rows[positions])] for _, positions in groups])
        p, q = len(row_labels), len(observed)
        scale = np.asarray(self.scale.values)[np.ix_(observed, observed)]
        return _MatrixStudentTAtLabels(
            entries=entries, degrees_of_freedom=self.degrees_of_freedom - (p - q), scale=jnp.asarray(scale),
        )


class MatrixStudentTLaw:
    """The matrix Student-t over a block (the module docstring): a
    :class:`~sipnet_calibration.probability.laws.Law`.

    Parameters
    ----------
    mean : ArrayLike
        Positional-only. :math:`m`, of the block's shape.
    entries : numpy.ndarray
        Positional-only. ``(n, q)`` ``int64``: each group's entries, into
        the flattened block, in the order of the rows :math:`O`.
    degrees_of_freedom : float
        Positional-only. :math:`\\nu'`.
    scale : ArrayLike
        Positional-only. :math:`\\Psi_{OO}`, ``(q, q)``.

    Attributes
    ----------
    mean, scale : jax.Array
    entries : numpy.ndarray
    degrees_of_freedom : float
    event_shape : tuple of int
    covariances : tuple
        Empty: :math:`\\Psi_{OO}` was checked when the spec was bound.
    """

    __slots__ = ("mean", "entries", "degrees_of_freedom", "scale")

    def __init__(self, mean: Any, entries: np.ndarray, degrees_of_freedom: float, scale: Any, /) -> None:
        object.__setattr__(self, "mean", jnp.asarray(mean, dtype=jnp.float64))
        object.__setattr__(self, "entries", np.asarray(entries, dtype=np.int64))
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "scale", jnp.asarray(scale, dtype=jnp.float64))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a MatrixStudentTLaw is frozen; build another rather than setting {name!r}.")

    def __repr__(self) -> str:
        n, q = self.entries.shape
        return f"MatrixStudentTLaw(event_shape={self.event_shape}, groups={n}, rows={q})"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.mean.shape)

    @property
    def dtype(self) -> Any:
        return jnp.float64

    @property
    def covariances(self) -> tuple[Any, ...]:
        return ()

    def log_prob(self, value: Any) -> Array:
        """The log density at *value*, ``(..., *block) -> (...)``."""
        value = jnp.asarray(value, dtype=jnp.float64)
        lead = value.shape[: value.ndim - self.mean.ndim]
        residual = value.reshape((*lead, -1)) - self.mean.reshape((-1,))
        grouped = residual[..., self.entries]
        scatter = jnp.einsum("...gi,...gj->...ij", grouped, grouped)
        n, q = self.entries.shape
        nu = self.degrees_of_freedom
        return (
            -n * q / 2.0 * jnp.log(jnp.pi)
            + jax.scipy.special.multigammaln((nu + n) / 2.0, q) - jax.scipy.special.multigammaln(nu / 2.0, q)
            + nu / 2.0 * _log_determinant(self.scale)
            - (nu + n) / 2.0 * _log_determinant(self.scale + scatter)
        )

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws :math:`S_{OO} \\sim \\mathcal W^{-1}_q(\\nu', \\Psi_{OO})`,
        then each group's residual :math:`r_i \\sim \\mathcal N(0, S_{OO})`,
        ``(*sample_shape, *block)``.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed, what="MatrixStudentTLaw")
        sample_shape = tuple(sample_shape) if isinstance(sample_shape, (tuple, list)) else (int(sample_shape),)
        matrix_key, noise_key = jax.random.split(seed)
        n, q = self.entries.shape
        covariance = sample_inverse_wishart(matrix_key, self.degrees_of_freedom, self.scale, sample_shape)
        lower = jnp.linalg.cholesky(covariance)
        noise = jax.random.normal(noise_key, (*sample_shape, n, q), dtype=jnp.float64)
        grouped = jnp.einsum("...ij,...gj->...gi", lower, noise)
        draws = jnp.broadcast_to(self.mean.reshape((-1,)), (*sample_shape, self.mean.size))
        draws = draws.at[..., self.entries].add(grouped)
        return draws.reshape((*sample_shape, *self.mean.shape))


def inverse_wishart_log_prob(matrix: Any, degrees_of_freedom: Any, scale: Any) -> Array:
    """:math:`\\log \\mathcal W^{-1}_p(S; \\nu, \\Psi)` against Lebesgue
    measure on the lower triangle, ``(..., p, p) -> (...)``,

    .. math::

        \\tfrac{\\nu}{2}\\log|\\Psi| - \\tfrac{\\nu p}{2}\\log 2 - \\log\\Gamma_p(\\tfrac{\\nu}{2})
            - \\tfrac{\\nu + p + 1}{2}\\log|S| - \\tfrac12\\operatorname{tr}(\\Psi S^{-1});

    traceable, with no check of its arguments, which
    :class:`~sipnet_calibration.probability.families.InverseWishart` makes."""
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    p = matrix.shape[-1]
    lower = jnp.linalg.cholesky(matrix)
    solved = jax.scipy.linalg.cho_solve((lower, True), jnp.broadcast_to(scale, matrix.shape))
    return (
        degrees_of_freedom / 2.0 * _log_determinant(scale) - degrees_of_freedom * p / 2.0 * jnp.log(2.0)
        - jax.scipy.special.multigammaln(degrees_of_freedom / 2.0, p)
        - (degrees_of_freedom + p + 1.0) / 2.0 * _log_determinant(matrix) - jnp.trace(solved, axis1=-2, axis2=-1) / 2.0
    )


def sample_inverse_wishart(key: Array, degrees_of_freedom: Any, scale: Any, sample_shape: tuple[int, ...] = ()) -> Array:
    """Draws of :math:`\\mathcal W^{-1}_p(\\nu, \\Psi)`, ``(*sample_shape,
    *batch, p, p)`` for *degrees_of_freedom* of shape ``batch`` and *scale*
    ``(*batch, p, p)``: the inverse of a Wishart draw of scale
    :math:`\\Psi^{-1}`. Traceable, with no check of its arguments."""
    scale = jnp.asarray(scale, dtype=jnp.float64)
    lower = jnp.linalg.cholesky(scale)
    eye = jnp.broadcast_to(jnp.eye(scale.shape[-1], dtype=jnp.float64), scale.shape)
    inverse = jax.scipy.linalg.cho_solve((lower, True), eye)
    inverse_lower = jnp.linalg.cholesky((inverse + jnp.swapaxes(inverse, -1, -2)) / 2.0)
    wishart = tfd.WishartTriL(df=jnp.asarray(degrees_of_freedom, dtype=jnp.float64), scale_tril=inverse_lower)
    draws = wishart.sample(sample_shape, seed=key)
    draw_lower = jnp.linalg.cholesky(draws)
    eye = jnp.broadcast_to(jnp.eye(scale.shape[-1], dtype=jnp.float64), draws.shape)
    inverted = jax.scipy.linalg.cho_solve((draw_lower, True), eye)
    return (inverted + jnp.swapaxes(inverted, -1, -2)) / 2.0


# ── private: a spec at the labels in use ──────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _StudentTGroup:
    positions: np.ndarray
    by: dict[str, int]
    label: Any
    covariance: Any
    shape: float
    scale: float


@dataclass(frozen=True, eq=False)
class _StudentTAtLabels:
    """A :class:`StudentTSpec` at the labels in use: builds its law from
    the mean and what the covariance reads, at one draw."""

    groups: tuple[_StudentTGroup, ...]
    event_dim: str | None
    read_dims: Any

    def law(self, mean: Array, reads: Any) -> StudentTLaw:
        return StudentTLaw(mean, [
            (group.positions, group.covariance.operator(self.reads_of_group(reads, group)), group.shape, group.scale)
            for group in self.groups
        ])

    def reads_of_group(self, reads: Any, group: _StudentTGroup) -> dict[str, Any]:
        """What a group's covariance reads, sliced to it."""
        return {
            name: _slice_to_group(value, self.read_dims.get(name, ()), self.event_dim, group.positions, group.by)
            for name, value in reads.items()
        }


@dataclass(frozen=True, eq=False)
class _MatrixStudentTAtLabels:
    """A :class:`MatrixStudentTSpec` at the labels in use."""

    entries: np.ndarray
    degrees_of_freedom: float
    scale: Array

    def law(self, mean: Array, reads: Any) -> MatrixStudentTLaw:
        return MatrixStudentTLaw(mean, self.entries, self.degrees_of_freedom, self.scale)


# ── helpers ───────────────────────────────────────────────────────────────────


def _set(spec: Any, name: str, value: Any) -> None:
    object.__setattr__(spec, name, value)


def _read_only(array: xr.DataArray) -> xr.DataArray:
    copy = array.copy(deep=True)
    copy.values.flags.writeable = False
    return copy


def _shared_rows(factor_name: str, groups: Sequence[tuple[Any, np.ndarray]], rows: np.ndarray, row_labels: pd.Index) -> np.ndarray:
    """The rows :math:`O` every group maps to, ascending, each group's
    entries checked to map to distinct rows and every group to the same
    ones.

    Raises
    ------
    ValueError
        If a group maps two entries to one row, or two groups map to
        different rows, naming the groups that differ from the first.
    """
    for key, positions in groups:
        check_group_maps_to_distinct_rows(factor_name, key, rows[positions], row_labels)
    first_key, first = groups[0]
    observed = np.sort(rows[first])
    differing = [key for key, positions in groups if not np.array_equal(np.sort(rows[positions]), observed)]
    check_groups_map_to_the_same_rows(factor_name, first_key, observed, differing, row_labels)
    return observed


def _log_determinant(matrix: Array) -> Array:
    """:math:`\\log|A|` of positive-definite matrices ``(..., p, p)``, from
    their Cholesky factors: ``NaN`` for one that is not positive definite."""
    lower = jnp.linalg.cholesky(matrix)
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(lower, axis1=-2, axis2=-1)), axis=-1)


def _per_group(value: Any, by: str | None, *, what: str) -> xr.DataArray:
    """A shape or scale as a 0-d DataArray, or one on *by*, positive and
    finite."""
    if isinstance(value, xr.DataArray):
        check_per_group_value_is_on_its_grouping(value, by, what=what)
        check_per_group_labels_are_unique(value, by, what=what)
        array = value.astype(np.float64)
    else:
        check_value_is_a_number(value, what=what)
        array = xr.DataArray(np.float64(np.asarray(value)))
    check_values_are_positive_and_finite(np.asarray(array.values), what=what)
    return _read_only(array)


def _at_label(value: xr.DataArray, dim: str, label: Any, *, what: str, factor_name: str) -> float:
    """A shape or scale at a group's label."""
    if value.ndim == 0:
        return float(value)
    index = value.indexes[dim]
    position = int(index.get_indexer([label])[0])
    check_group_label_has_a_value(factor_name, what, dim, label, position)
    return float(value.values[position])


# ── checks ────────────────────────────────────────────────────────────────────


def check_name_is_one_name(name: Any, *, what: str) -> None:
    """A mean or grouping level is named by one non-empty string."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"{what} is {name!r}; give one name.")


def check_is_a_data_array(value: Any, *, what: str) -> None:
    """A labeled argument is a DataArray."""
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"{what} is a {type(value).__name__}; give an xr.DataArray.")


def check_value_is_a_number(value: Any, *, what: str) -> None:
    """A shape or scale shared by every group is a real number: a scalar of
    an integer or float dtype, not a boolean."""
    kind = np.asarray(value).dtype.kind if not isinstance(value, (str, bytes)) else "U"
    if isinstance(value, bool) or kind not in "iuf" or np.ndim(value) != 0:
        raise TypeError(f"{what} is a {type(value).__name__}; give a number, or a DataArray on the grouping level.")


def check_per_group_value_is_on_its_grouping(value: xr.DataArray, by: str | None, *, what: str) -> None:
    """A shape or scale per group is labeled on the grouping level."""
    if value.ndim == 0:
        return
    if by is None or tuple(map(str, value.dims)) != (by,) or by not in value.indexes:
        raise ValueError(
            f"{what} is on {tuple(map(str, value.dims))}; give one number, or with by= a DataArray labeled on "
            "the grouping level alone."
        )


def check_per_group_labels_are_unique(value: xr.DataArray, by: str | None, *, what: str) -> None:
    """A shape or scale per group labels each group once."""
    if value.ndim and not value.indexes[by].is_unique:
        raise ValueError(f"{what} labels a group of {by!r} twice; give each label once.")


def check_values_are_positive_and_finite(values: np.ndarray, *, what: str) -> None:
    """A shape or scale is positive and finite."""
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError(f"{what} is not positive and finite everywhere; give positive numbers.")


def check_group_label_has_a_value(factor_name: str, what: str, dim: str, label: Any, position: int) -> None:
    """A group reads its shape and scale at its label."""
    if position < 0:
        raise KeyError(
            f"the Student-t of {factor_name!r} has no {what} at the group {label!r} of {dim!r}; give one for every "
            "group's label."
        )


def check_degrees_of_freedom_is_a_number(value: Any) -> None:
    """An inverse Wishart's degrees of freedom are a real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"MatrixStudentTSpec's degrees_of_freedom is a {type(value).__name__}; give a number.")


def check_scale_is_a_labeled_positive_definite_matrix(scale: xr.DataArray) -> None:
    """An inverse Wishart's scale is square on two labeled axes of the same
    labels, and positive definite."""
    dims = tuple(map(str, scale.dims))
    square = len(dims) == 2 and all(d in scale.indexes for d in dims) and scale.indexes[dims[0]].equals(
        pd.Index(scale.indexes[dims[-1]])
    )
    if not square:
        raise ValueError(
            f"MatrixStudentTSpec's scale is on {dims}; give a square matrix on two axes labeled alike, rows first."
        )
    values = np.asarray(scale.values, dtype=np.float64)
    if not (np.all(np.isfinite(values)) and np.allclose(values, values.T) and np.all(np.linalg.eigvalsh(values) > 0)):
        raise ValueError("MatrixStudentTSpec's scale is not a symmetric positive-definite matrix; give one.")


def check_degrees_of_freedom_exceed_the_dimension(degrees_of_freedom: float, p: int) -> None:
    """An inverse Wishart over ``p x p`` matrices has :math:`\\nu > p - 1`."""
    if not (math.isfinite(degrees_of_freedom) and degrees_of_freedom > p - 1):
        raise ValueError(
            f"MatrixStudentTSpec's degrees_of_freedom is {degrees_of_freedom}, but an inverse Wishart over {p} x {p} "
            f"matrices needs more than {p - 1}."
        )


def check_label_map_names_the_rows(label_map: xr.DataArray, scale: xr.DataArray) -> None:
    """The label map is one-dimensional, named for the scale's row axis."""
    if label_map.ndim != 1 or str(label_map.name) != str(scale.dims[0]):
        raise ValueError(
            f"MatrixStudentTSpec's label_map is named {label_map.name!r} on {tuple(map(str, label_map.dims))}; give "
            f"a one-dimensional DataArray on the event's dim, named for the rows of the scale, {str(scale.dims[0])!r}."
        )


def check_group_maps_to_distinct_rows(factor_name: str, key: Any, rows: np.ndarray, row_labels: pd.Index) -> None:
    """No two entries of a group map to one row, which would repeat it."""
    if len(np.unique(rows)) != len(rows):
        repeated = sorted({str(row_labels[r]) for r in rows if (rows == r).sum() > 1})
        raise ValueError(
            f"the group {key} of {factor_name!r} maps two entries to {truncated(repeated)}, which would repeat a "
            "row; group the entries so each maps to a row once."
        )


def check_groups_map_to_the_same_rows(
    factor_name: str, first_key: Any, observed: np.ndarray, differing: Sequence[Any], row_labels: pd.Index
) -> None:
    """Every group maps to the same rows, as the closed form needs: a group
    that maps to others, as when sites' records are ragged, has no
    conjugate form."""
    if differing:
        raise ValueError(
            f"the groups of {factor_name!r} map to different rows: {truncated([str(k) for k in differing])} differ "
            f"from {first_key}, which maps to {truncated([str(row_labels[r]) for r in observed])}; the inverse-Wishart "
            "closed form needs every group to map to the same rows, so sample the matrix jointly instead."
        )

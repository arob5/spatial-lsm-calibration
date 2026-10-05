"""Covariance specs: how a Gaussian factor's covariance is built from what
it reads, as structure over labels.

Where this sits
---------------
::

    probability.covariance              (DiagonalSpec, DenseSpec, ..., BlockDiagonalSpec)
      -> probability.parts.GaussianSpec (a Gaussian law: means and a covariance)
      -> probability.model.bind         (each spec at the labels in use)
      -> a structured operator per draw (pyEKI's, through probability._linalg)

A covariance spec is a declaration: it holds no labels and no numbers. It
says which entries of a factor's event form blocks and what each block is
built from. Bound to the labels in use, it builds one operator per draw of
what it reads, and the operator does the algebra (the solves, the
log-determinant, the factor a draw is made with); nothing here forms a
dense matrix over more than one block.

What a spec reads
-----------------
A spec's **scope** is the entries it covers: the factor's event, or one
group of a :class:`BlockDiagonalSpec`, ``n`` entries. A string argument
names a constant, a label map or a component the factor reads; a callable
receives the keywords it names (the keyword rule of
:mod:`~sipnet_calibration.probability.parts`). Inside a
:class:`BlockDiagonalSpec`, what a block reads arrives:

- on the event's dim: sliced to the group;
- on a dim named like the grouping level: at the group's label, that axis
  removed, such as a per-site variance under ``by="site"``; the dim is a
  dim of the coords in use (some component is indexed by it) or one of the
  factor's ``own_dims``, its labels then the constants' own;
- otherwise: whole.

The event is one component, indexed by one dim at most and with no element
axes, so its entries are its labels in order.

Classes
-------
:class:`CovarianceSpec`
    The base: ``reads``.
:class:`DiagonalSpec`, :class:`DenseSpec`
    A diagonal, or a full matrix, over the scope.
:class:`SumSpec`, :class:`ScaledSpec`
    A sum of covariances, and a positive multiple of one.
:class:`BlockDiagonalSpec`
    Independent groups of entries, each with its own block.
:class:`SubmatrixSpec`
    A matrix component's rows and columns at each entry's label.

Notes
-----
Each spec evaluates to one of pyEKI's operators: ``PSDDiagonal``,
``DensePSD``, ``PSDScaled`` and ``PSDBlockDiag``. A value precondition, a
positive variance or a positive-definite block, is not checked when an
operator is built: one that fails gives a density that is not finite, which
the model reads as no density. A covariance that depends on no parameter is
checked once, when the model is conditioned, and one that does at each
draw, whose sample is then invalid.

A block-diagonal grouping must keep each group's entries contiguous, so the
operator is the covariance in the event's order: true of an observation
dim grouped by ``site``, whose labels are sorted by site.

Usage
-----
::

    from sipnet_calibration.probability import BlockDiagonalSpec, DenseSpec, DiagonalSpec

    def block(time_since_epoch, standard_deviation):
        lag = jnp.abs(time_since_epoch[:, None] - time_since_epoch[None, :]) / 86_400.0
        return jnp.diag(standard_deviation**2) + 0.5**2 * jnp.exp(-lag / 30.0)

    BlockDiagonalSpec(DenseSpec(block), by="site")     # one dense block per site
    DiagonalSpec(lambda standard_deviation: standard_deviation**2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pandas as pd

from sipnet_calibration.probability import _linalg
from sipnet_calibration.probability._keywords import function_reads
from sipnet_calibration.probability._validation import as_names, truncated
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import Interval, PositiveDefinite

__all__ = [
    "BlockDiagonalSpec",
    "CovarianceSpec",
    "DenseSpec",
    "DiagonalSpec",
    "ScaledSpec",
    "SubmatrixSpec",
    "SumSpec",
]

Array = jax.Array


class CovarianceSpec(ABC):
    """How a covariance is built from what its factor reads.

    Its scope is the entries it covers: the factor's event, or one group of
    a :class:`BlockDiagonalSpec`. Bound to the labels in use, it builds a
    positive-definite operator over its scope at each draw. A string
    argument names a constant, label map or component the factor reads; a
    callable receives the keywords it names.

    Attributes
    ----------
    reads : tuple of str
        Every name it reads, in order of first appearance.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a {type(self).__name__} is frozen; build another rather than setting {name!r}.")

    @property
    @abstractmethod
    def reads(self) -> tuple[str, ...]:
        """Every name it reads."""

    @abstractmethod
    def _at(self, scope: _Scope) -> _Covariance:
        """This spec at a scope: what builds its operator at each draw.

        Raises
        ------
        ValueError
            If it cannot be built over the scope, naming the factor.
        """


class DiagonalSpec(CovarianceSpec):
    """:math:`\\Sigma = \\operatorname{diag}(v)`, a ``PSDDiagonal``.

    Parameters
    ----------
    variance : str or callable
        Positional-only. Each entry's variance, positive: the name of a
        value on the event's dim, or of a scalar, which every entry takes;
        or ``(**reads) -> (n,)``.

    Raises
    ------
    TypeError
        If *variance* is neither a string nor a function following the
        keyword rule.
    """

    __slots__ = ("variance", "_reads")

    variance: str | Callable[..., Array]

    def __init__(self, variance: str | Callable[..., Array], /) -> None:
        _set(self, "variance", variance)
        _set(self, "_reads", _reads_of(variance, what="DiagonalSpec's variance"))

    @property
    def reads(self) -> tuple[str, ...]:
        return self._reads

    def __repr__(self) -> str:
        return f"DiagonalSpec({_argument_name(self.variance)})"

    def _at(self, scope: _Scope) -> _Covariance:
        return _Diagonal(spec=self, size=scope.size, factor_name=scope.factor_name)


class DenseSpec(CovarianceSpec):
    """A full matrix over the scope, a ``DensePSD``.

    Parameters
    ----------
    matrix : str or callable
        Positional-only. The name of an ``(n, n)`` value, or
        ``(**reads) -> (n, n)``; symmetric positive definite. Its symmetric
        part is what is factored, as pyEKI's ``DensePSD`` does, so an
        asymmetric matrix is not refused.

    Raises
    ------
    TypeError
        As :class:`DiagonalSpec`.
    """

    __slots__ = ("matrix", "_reads")

    matrix: str | Callable[..., Array]

    def __init__(self, matrix: str | Callable[..., Array], /) -> None:
        _set(self, "matrix", matrix)
        _set(self, "_reads", _reads_of(matrix, what="DenseSpec's matrix"))

    @property
    def reads(self) -> tuple[str, ...]:
        return self._reads

    def __repr__(self) -> str:
        return f"DenseSpec({_argument_name(self.matrix)})"

    def _at(self, scope: _Scope) -> _Covariance:
        return _Dense(spec=self, size=scope.size, factor_name=scope.factor_name)


class SumSpec(CovarianceSpec):
    """:math:`\\Sigma = \\sum_i \\Sigma_i` over the scope.

    ================================== =================================
    terms                              result
    ================================== =================================
    diagonals                          ``PSDDiagonal``
    no block-diagonal term             ``DensePSD`` over the scope
    block-diagonal and diagonals       blockwise, the diagonals sliced
    block-diagonal, equal groupings    blockwise
    anything else                      refused
    ================================== =================================

    Blockwise, ``SumSpec(BlockDiagonalSpec(a, by=g), DiagonalSpec(v))`` is
    ``BlockDiagonalSpec(SumSpec(a, DiagonalSpec(v)), by=g)``.

    Parameters
    ----------
    *terms : CovarianceSpec
        Positional. At least two.

    Raises
    ------
    TypeError
        If a term is not a covariance spec.
    ValueError
        If there are fewer than two terms; and, when bound, for a
        combination the table refuses, which would need a dense matrix over
        the whole event, a block-diagonal term held inside another term
        (scaled, or in a nested sum) among them: write
        ``BlockDiagonalSpec(SumSpec(...), by=...)``.
    """

    __slots__ = ("terms",)

    terms: tuple[CovarianceSpec, ...]

    def __init__(self, *terms: CovarianceSpec) -> None:
        check_terms_are_covariance_specs(terms, what="SumSpec's terms")
        check_sum_has_two_terms(terms)
        _set(self, "terms", tuple(terms))

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for term in self.terms for name in term.reads))

    def __repr__(self) -> str:
        return f"SumSpec({', '.join(map(repr, self.terms))})"

    def _at(self, scope: _Scope) -> _Covariance:
        blockwise = [t for t in self.terms if isinstance(t, BlockDiagonalSpec)]
        check_sum_holds_no_hidden_block_diagonal(self, scope)
        if not blockwise:
            return _Sum(terms=tuple(t._at(scope) for t in self.terms))
        check_sum_is_blockwise(self, blockwise, scope)
        by = blockwise[0].by
        inner = [t.block if isinstance(t, BlockDiagonalSpec) else t for t in self.terms]
        return BlockDiagonalSpec(SumSpec(*inner), by=by)._at(scope)


class ScaledSpec(CovarianceSpec):
    """:math:`\\Sigma = s\\,\\Sigma_0`, :math:`s` positive, a ``PSDScaled``.

    Parameters
    ----------
    base : CovarianceSpec
        Positional-only. :math:`\\Sigma_0`; it may not read *scale*.
    scale : str
        Keyword-only. :math:`s`: a scalar component or constant, or inside a
        :class:`BlockDiagonalSpec` one indexed by the grouping level, of
        which each group reads its own.

    Raises
    ------
    TypeError
        If *base* is not a covariance spec or *scale* not a name.
    ValueError
        If *base* reads *scale*; and, when bound, if *scale* is not a scalar
        in its scope, or is a component whose support allows a value at or
        below zero.
    """

    __slots__ = ("base", "scale")

    base: CovarianceSpec
    scale: str

    def __init__(self, base: CovarianceSpec, /, *, scale: str) -> None:
        check_terms_are_covariance_specs([base], what="ScaledSpec's base")
        check_name_is_a_string(scale, what="ScaledSpec's scale")
        check_base_does_not_read_its_scale(base, scale)
        _set(self, "base", base)
        _set(self, "scale", scale)

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.base.reads, self.scale)))

    def __repr__(self) -> str:
        return f"ScaledSpec({self.base!r}, scale={self.scale!r})"

    def _at(self, scope: _Scope) -> _Covariance:
        check_scale_is_a_positive_scalar(self.scale, scope)
        return _Scaled(base=self.base._at(scope), scale=self.scale)


class BlockDiagonalSpec(CovarianceSpec):
    """Independent groups of entries, each with its own block: a
    ``PSDBlockDiag``.

    Entries sharing their labels at *by* form a group, the scope of
    *block*, in the order the groups first appear. What *block* reads
    arrives sliced to the group, at the group's label, or whole, as the
    module docstring says.

    Parameters
    ----------
    block : CovarianceSpec
        Positional-only.
    by : str or Sequence[str]
        Keyword-only. A level of the event's stacked dim, the dim itself
        (every entry its own group), or several levels.

    Raises
    ------
    TypeError
        If *block* is not a covariance spec, or *by* is not one or more
        names.
    ValueError
        If *by* is empty or names a level twice; when bound: if the event
        has no dim, *by* is neither its dim nor
        levels of it, a group's entries are not contiguous, or a group's
        label is not among the labels of the dim named like *by*, which
        something the block reads is on.
    """

    __slots__ = ("block", "by")

    block: CovarianceSpec
    by: tuple[str, ...]

    def __init__(self, block: CovarianceSpec, /, *, by: str | Sequence[str]) -> None:
        check_terms_are_covariance_specs([block], what="BlockDiagonalSpec's block")
        by = (by,) if isinstance(by, str) else as_names(by, message_name="BlockDiagonalSpec's by")
        check_grouping_names_distinct_levels(by)
        _set(self, "block", block)
        _set(self, "by", by)

    @property
    def reads(self) -> tuple[str, ...]:
        return self.block.reads

    def __repr__(self) -> str:
        by = self.by[0] if len(self.by) == 1 else list(self.by)
        return f"BlockDiagonalSpec({self.block!r}, by={by!r})"

    def _at(self, scope: _Scope) -> _Covariance:
        check_event_has_a_dim(scope, what="BlockDiagonalSpec")
        keys = _group_keys(self.by, scope)
        groups = []
        for key, positions in _groups(keys):
            check_group_is_contiguous(scope, self.by, key, positions)
            inner = scope.group(positions, by=self.by, key=key)
            groups.append(_Group(positions=positions, by=inner.by_positions, block=self.block._at(inner)))
        return _BlockDiagonal(groups=tuple(groups), event_dim=scope.event_dim, read_dims=scope.read_dims)


class SubmatrixSpec(CovarianceSpec):
    """:math:`\\Sigma_{ij} = S[\\mu_i, \\mu_j]`, a matrix component's rows
    and columns at each entry's label: a ``DensePSD``.

    Parameters
    ----------
    component : str
        Positional-only. A component on
        :data:`~sipnet_calibration.probability.support.POSITIVE_DEFINITE`,
        ``(p, p)``, indexed by nothing.
    label_map : str
        Keyword-only. A label map :math:`\\mu` the factor holds, from the
        event's dim to the component's first element axis.

    Raises
    ------
    TypeError
        If either is not a name.
    ValueError
        When bound: if *component* is not such a component, *label_map* is
        not such a label map, or two entries in the scope map to one label.
    """

    __slots__ = ("component", "label_map")

    component: str
    label_map: str

    def __init__(self, component: str, /, *, label_map: str) -> None:
        check_name_is_a_string(component, what="SubmatrixSpec's component")
        check_name_is_a_string(label_map, what="SubmatrixSpec's label_map")
        _set(self, "component", component)
        _set(self, "label_map", label_map)

    @property
    def reads(self) -> tuple[str, ...]:
        return (self.component, self.label_map)

    def __repr__(self) -> str:
        return f"SubmatrixSpec({self.component!r}, label_map={self.label_map!r})"

    def _at(self, scope: _Scope) -> _Covariance:
        check_submatrix_component_is_a_matrix(self, scope)
        check_submatrix_label_map_targets_its_rows(self, scope)
        positions = np.asarray(scope.fixed[self.label_map], dtype=np.int64)
        check_scope_maps_to_distinct_labels(self, scope, positions)
        return _Submatrix(component=self.component, positions=positions)


# ── private: a spec at the labels in use ──────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _Scope:
    """What a spec is bound over: the entries it covers and what it may read.

    ``read_dims`` gives the dims of each name the factor reads, in its
    arrays' axis order; ``fixed`` the concrete constants and label maps;
    ``specs`` the components'; ``label_map_targets`` each label map's
    target; ``coords`` the labels in use. ``labels`` are the event dim's
    labels in the scope, ``None`` for an event indexed by nothing."""

    factor_name: str
    size: int
    event_dim: str | None
    labels: pd.Index | None
    read_dims: Mapping[str, tuple[str, ...]]
    fixed: Mapping[str, Any]
    specs: Mapping[str, ArraySpec]
    label_map_targets: Mapping[str, str]
    coords: Mapping[str, pd.Index]
    by_positions: Mapping[str, int] = field(default_factory=dict)

    def group(self, positions: np.ndarray, *, by: tuple[str, ...], key: tuple[Any, ...]) -> _Scope:
        """The scope of one group: its entries, and each read on a dim
        named like the one grouping level at the group's label, that axis
        removed."""
        by_positions: dict[str, int] = {}
        if len(by) == 1 and by[0] != self.event_dim:
            (level,) = by
            readers = [n for n, dims in self.read_dims.items() if level in dims]
            if readers:
                check_group_label_is_a_label_of_its_dim(self, level, key[0], readers)
                by_positions[level] = int(self.coords[level].get_loc(key[0]))
        read_dims = {name: tuple(d for d in dims if d not in by_positions) for name, dims in self.read_dims.items()}
        fixed = {
            name: _slice_to_group(np.asarray(value), self.read_dims.get(name, ()), self.event_dim, positions, by_positions)
            for name, value in self.fixed.items()
        }
        return _Scope(
            factor_name=self.factor_name, size=len(positions), event_dim=self.event_dim,
            labels=self.labels[positions], read_dims=read_dims, fixed=fixed, specs=self.specs,
            label_map_targets=self.label_map_targets, coords=self.coords, by_positions=by_positions,
        )


class _Covariance(ABC):
    """A spec at its scope: builds its operator, or its dense matrix, from
    what it reads there."""

    @abstractmethod
    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        """The operator at one draw of what the factor reads, sliced to the
        scope."""

    @abstractmethod
    def matrix(self, reads: Mapping[str, Array]) -> Array:
        """The covariance as a dense ``(n, n)`` array, for a sum: a term of
        a sum may be only semi-definite, so it is not factored alone."""


@dataclass(frozen=True, eq=False)
class _Diagonal(_Covariance):
    spec: DiagonalSpec
    size: int
    factor_name: str

    def variance(self, reads: Mapping[str, Array]) -> Array:
        variance = jnp.asarray(_evaluate_argument(self.spec.variance, self.spec.reads, reads), dtype=jnp.float64)
        if variance.ndim == 0:
            variance = jnp.broadcast_to(variance, (self.size,))
        check_covariance_has_its_shape(self.factor_name, self.spec, tuple(variance.shape), (self.size,))
        return variance

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        return _linalg.PSDDiagonal(self.variance(reads))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        return jnp.diag(self.variance(reads))


@dataclass(frozen=True, eq=False)
class _Dense(_Covariance):
    spec: DenseSpec
    size: int
    factor_name: str

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        return _linalg.DensePSD(self.matrix(reads))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        matrix = jnp.asarray(_evaluate_argument(self.spec.matrix, self.spec.reads, reads), dtype=jnp.float64)
        check_covariance_has_its_shape(self.factor_name, self.spec, tuple(matrix.shape), (self.size, self.size))
        return matrix


@dataclass(frozen=True, eq=False)
class _Sum(_Covariance):
    terms: tuple[_Covariance, ...]

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        if all(isinstance(term, _Diagonal) for term in self.terms):
            return _linalg.PSDDiagonal(sum(term.variance(reads) for term in self.terms))
        return _linalg.DensePSD(self.matrix(reads))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        return sum(term.matrix(reads) for term in self.terms)


@dataclass(frozen=True, eq=False)
class _Scaled(_Covariance):
    base: _Covariance
    scale: str

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        return _linalg.PSDScaled(self.base.operator(reads), jnp.asarray(reads[self.scale], dtype=jnp.float64))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        return jnp.asarray(reads[self.scale], dtype=jnp.float64) * self.base.matrix(reads)


@dataclass(frozen=True, eq=False)
class _Group:
    positions: np.ndarray
    by: Mapping[str, int]
    block: _Covariance


@dataclass(frozen=True, eq=False)
class _BlockDiagonal(_Covariance):
    groups: tuple[_Group, ...]
    event_dim: str
    read_dims: Mapping[str, tuple[str, ...]]

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        return _linalg.PSDBlockDiag(tuple(group.block.operator(self._reads_of_group(reads, group)) for group in self.groups))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        return jax.scipy.linalg.block_diag(*(group.block.matrix(self._reads_of_group(reads, group)) for group in self.groups))

    def _reads_of_group(self, reads: Mapping[str, Array], group: _Group) -> dict[str, Array]:
        return {
            name: _slice_to_group(value, self.read_dims.get(name, ()), self.event_dim, group.positions, group.by)
            for name, value in reads.items()
        }


@dataclass(frozen=True, eq=False)
class _Submatrix(_Covariance):
    component: str
    positions: np.ndarray

    def operator(self, reads: Mapping[str, Array]) -> _linalg.PSDLinOp:
        return _linalg.DensePSD(self.matrix(reads))

    def matrix(self, reads: Mapping[str, Array]) -> Array:
        matrix = jnp.asarray(reads[self.component], dtype=jnp.float64)
        return matrix[np.ix_(self.positions, self.positions)]

# ── helpers ───────────────────────────────────────────────────────────────────


def _set(spec: Any, name: str, value: Any) -> None:
    object.__setattr__(spec, name, value)


def _reads_of(argument: Any, *, what: str) -> tuple[str, ...]:
    """What a string or callable argument reads: the name, or the keywords."""
    if isinstance(argument, str):
        return (argument,)
    check_argument_is_a_name_or_a_function(argument, what=what)
    return function_reads(argument, message_name=what)


def _argument_name(argument: Any) -> str:
    """A string argument quoted, or a function's name."""
    if isinstance(argument, str):
        return repr(argument)
    return getattr(argument, "__name__", type(argument).__name__)


def _evaluate_argument(argument: str | Callable[..., Array], names: Sequence[str], reads: Mapping[str, Array]) -> Array:
    """A string argument's value, or a function called with the *names* it
    reads."""
    if isinstance(argument, str):
        return reads[argument]
    return argument(**{name: reads[name] for name in names})


def _holds_a_block_diagonal(spec: CovarianceSpec) -> bool:
    """Whether *spec* is or holds a :class:`BlockDiagonalSpec`."""
    if isinstance(spec, BlockDiagonalSpec):
        return True
    if isinstance(spec, SumSpec):
        return any(_holds_a_block_diagonal(term) for term in spec.terms)
    if isinstance(spec, ScaledSpec):
        return _holds_a_block_diagonal(spec.base)
    return False


def _group_keys(by: tuple[str, ...], scope: _Scope) -> list[tuple[Any, ...]]:
    """Each entry's group: its label at the levels *by*, or its whole
    label when *by* is the event's dim."""
    labels = scope.labels
    if by == (scope.event_dim,):
        return [label if isinstance(label, tuple) else (label,) for label in labels]
    check_by_names_levels_of_the_event_dim(by, scope)
    return list(zip(*(labels.get_level_values(level) for level in by)))


def _groups(keys: Sequence[tuple[Any, ...]]) -> list[tuple[tuple[Any, ...], np.ndarray]]:
    """``(key, positions)`` of each group, in order of first appearance."""
    positions: dict[tuple[Any, ...], list[int]] = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)
    return [(key, np.asarray(p, dtype=np.int64)) for key, p in positions.items()]


def _slice_to_group(value: Any, dims: tuple[str, ...], event_dim: str | None, positions: np.ndarray, by: Mapping[str, int]) -> Any:
    """*value*, on *dims*, sliced to a group: taken at *positions* along the
    event's dim, and at its label's position along a dim named like the
    grouping level, that axis removed."""
    if not dims:
        return value
    index: list[Any] = []
    for dim in dims:
        if dim == event_dim:
            index.append(positions)
        elif dim in by:
            index.append(by[dim])
        else:
            index.append(slice(None))
    return value[tuple(index)]


# ── checks ────────────────────────────────────────────────────────────────────


def check_argument_is_a_name_or_a_function(argument: Any, *, what: str) -> None:
    """A covariance spec's argument is a name or a function."""
    if not callable(argument):
        raise TypeError(
            f"{what} is a {type(argument).__name__}; give the name of what it reads, or a function of the "
            "names it reads."
        )


def check_name_is_a_string(name: Any, *, what: str) -> None:
    """A name is a non-empty string."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"{what} is {name!r}; give the name of a component or constant the factor reads.")


def check_terms_are_covariance_specs(terms: Sequence[Any], *, what: str) -> None:
    """A covariance is built from covariance specs."""
    wrong = [type(t).__name__ for t in terms if not isinstance(t, CovarianceSpec)]
    if wrong:
        raise TypeError(f"{what} holds {truncated(wrong)}; give covariance specs, such as DiagonalSpec(...).")


def check_sum_has_two_terms(terms: Sequence[CovarianceSpec]) -> None:
    """A sum has at least two terms."""
    if len(terms) < 2:
        raise ValueError(f"a SumSpec has {len(terms)} term(s); give at least two, or the one term alone.")


def check_base_does_not_read_its_scale(base: CovarianceSpec, scale: str) -> None:
    """A scaled covariance's base does not read its scale, so the scale
    enters once, as a factor."""
    if scale in base.reads:
        raise ValueError(
            f"ScaledSpec's base reads its scale {scale!r}; a scaled covariance is s * base, so build the base "
            "without it."
        )


def check_sum_is_blockwise(spec: SumSpec, blockwise: Sequence[BlockDiagonalSpec], scope: _Scope) -> None:
    """A sum with a block-diagonal term is blockwise: every block-diagonal
    term grouped alike, the others diagonals, so no dense matrix spans
    groups."""
    groupings = {t.by for t in blockwise}
    others = [t for t in spec.terms if not isinstance(t, (BlockDiagonalSpec, DiagonalSpec))]
    if len(groupings) > 1 or others:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} is {spec!r}, a sum that would need a dense matrix over the "
            "whole event: block-diagonal terms are summed only with diagonals, or with block-diagonals of the "
            "same grouping. Write BlockDiagonalSpec(SumSpec(...), by=...)."
        )


def check_sum_holds_no_hidden_block_diagonal(spec: SumSpec, scope: _Scope) -> None:
    """A sum's terms hold a block-diagonal only as a term of their own,
    which is summed blockwise; one inside a scaled term or a nested sum
    would be summed as a dense matrix over the whole event."""
    hidden = [t for t in spec.terms if not isinstance(t, BlockDiagonalSpec) and _holds_a_block_diagonal(t)]
    if hidden:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} is {spec!r}, whose term(s) {truncated(hidden)} hold a "
            "block-diagonal inside, which would be summed as a dense matrix over the whole event; move the "
            "grouping outside, as BlockDiagonalSpec(SumSpec(...), by=...) or "
            "BlockDiagonalSpec(ScaledSpec(...), by=...)."
        )


def check_grouping_names_distinct_levels(by: tuple[str, ...]) -> None:
    """A grouping names at least one level, each once."""
    if not by or len(set(by)) != len(by):
        raise ValueError(
            f"BlockDiagonalSpec groups by {list(by)}; give the dim, or one or more distinct levels of it."
        )


def check_scale_is_a_positive_scalar(scale: str, scope: _Scope) -> None:
    """A scale is one number in its scope, and a component's support
    keeps it positive."""
    dims = scope.read_dims.get(scale)
    if dims:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} scales by {scale!r}, which is on {dims} in its scope; a "
            "scale is a scalar, or inside a BlockDiagonalSpec one indexed by the grouping level."
        )
    spec = scope.specs.get(scale)
    if spec is None:
        return
    support = spec.support
    if not (isinstance(support, Interval) and support.low >= 0 and not (support.low == 0 and support.low_closed)):
        raise ValueError(
            f"the covariance of {scope.factor_name!r} scales by {scale!r}, whose support {spec.support.name!r} "
            "is not positive; declare it on POSITIVE."
        )


def check_event_has_a_dim(scope: _Scope, *, what: str) -> None:
    """A grouping of entries needs the labels of the event's dim."""
    if scope.event_dim is None:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} is a {what}, but its event is indexed by nothing, so "
            "there are no labels to group by."
        )


def check_by_names_levels_of_the_event_dim(by: tuple[str, ...], scope: _Scope) -> None:
    """A grouping is the event's dim, or levels of it."""
    levels = [n for n in scope.labels.names if n is not None] if isinstance(scope.labels, pd.MultiIndex) else []
    unknown = [level for level in by if level not in levels]
    if unknown:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} groups by {list(by)}, but the event's dim "
            f"{scope.event_dim!r} has levels {levels}; group by the dim itself or by its levels."
        )


def check_group_is_contiguous(scope: _Scope, by: tuple[str, ...], key: tuple[Any, ...], positions: np.ndarray) -> None:
    """A group's entries are contiguous in the event, so the block-diagonal
    operator is the covariance in the event's order."""
    if positions[-1] - positions[0] + 1 != len(positions):
        raise ValueError(
            f"the covariance of {scope.factor_name!r} groups by {list(by)}, but the entries of the group {key} "
            f"are not contiguous in {scope.event_dim!r}; group by a level its labels are sorted by first, such "
            "as site."
        )


def check_group_label_is_a_label_of_its_dim(scope: _Scope, level: str, label: Any, readers: Sequence[str]) -> None:
    """Something a block reads on a dim named like the grouping level has
    the group's label, at which each group reads it."""
    labels = scope.coords.get(level)
    if labels is None or label not in labels:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} reads {truncated(readers)} on {level!r} at each group's "
            f"label, but the group {label!r} is not among the labels of {level!r}; bind {level!r} at every label "
            "the event's groups have."
        )


def check_covariance_has_its_shape(name: str, spec: CovarianceSpec, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """A diagonal is ``(n,)`` and a matrix ``(n, n)`` over the scope."""
    if shape != expected:
        raise ValueError(
            f"the covariance of {name!r}, {spec!r}, gives shape {shape} over a scope of {expected[0]} entries; "
            f"return {expected}."
        )


def check_submatrix_component_is_a_matrix(spec: SubmatrixSpec, scope: _Scope) -> None:
    """A submatrix is taken from a positive-definite component indexed by
    nothing."""
    component = scope.specs.get(spec.component)
    if component is None or not isinstance(component.support, PositiveDefinite) or component.indexed_by:
        raise ValueError(
            f"the covariance of {scope.factor_name!r} takes a submatrix of {spec.component!r}, which is not a "
            "component on POSITIVE_DEFINITE indexed by nothing; declare it so."
        )


def check_submatrix_label_map_targets_its_rows(spec: SubmatrixSpec, scope: _Scope) -> None:
    """A submatrix's label map is the factor's, from the event's dim to the
    matrix's first element axis."""
    rows = next(iter(scope.specs[spec.component].element_axes))
    target = scope.label_map_targets.get(spec.label_map)
    check_event_has_a_dim(scope, what="SubmatrixSpec")
    if target != rows or scope.read_dims.get(spec.label_map) != (scope.event_dim,):
        raise ValueError(
            f"the covariance of {scope.factor_name!r} reads {spec.label_map!r} as a label map from "
            f"{scope.event_dim!r} to {rows!r}, the rows of {spec.component!r}; give the factor such a label map."
        )


def check_scope_maps_to_distinct_labels(spec: SubmatrixSpec, scope: _Scope, positions: np.ndarray) -> None:
    """No two entries of a submatrix's scope map to one label, whose rows
    would repeat and make it singular."""
    if len(np.unique(positions)) != len(positions):
        rows = scope.specs[spec.component].element_axes[next(iter(scope.specs[spec.component].element_axes))]
        repeated = sorted({str(rows[p]) for p in positions if (positions == p).sum() > 1})
        raise ValueError(
            f"the covariance of {scope.factor_name!r} maps two entries of one scope to {truncated(repeated)} of "
            f"{spec.component!r}, which would repeat its rows; group the entries so each maps to a label once."
        )

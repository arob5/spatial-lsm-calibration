"""Derived parameters: quantities computed deterministically from the
parameters.

Where this sits
---------------
::

    parameters.vector.ParameterVector          (the parameters x)
      -> parameters.derived.DerivedParameters  (y = f(x), over that vector)
      -> parameters.prior.Prior                (a term may be given y)
      -> the labeled natural values, y's beside x's

What it reads
-------------
A :class:`~sipnet_calibration.parameters.vector.ParameterVector`, the labels
of any dims only derived parameters are indexed by, and each derived
parameter's constants and memberships, labeled ``xr.DataArray``\\ s as
:mod:`~sipnet_calibration.parameters.labels` defines them.

Data model
----------
A derived parameter has one value of its ``shape`` at each tuple of labels
of its ``indexed_by`` dims; together they are its **block**, of block shape
``(*index shape, *shape)``, as a parameter's are. Its values by parameter
are ``(*batch, *block shape)``. Its labeled form
(:meth:`DerivedParameters.values_to_dataset`) is the vector's: one
``float64`` variable per derived parameter on ``(*batch dims, *indexed_by,
*element axes)``, with ``units`` (omitted when ``None``), ``support`` (when
declared) and ``long_name`` (when set); missing never.

The model
---------
A **derived parameter** is a deterministic node of the statistical model,

.. math::

    y = f\\big(x_{g_1}, \\dots, x_{g_m};\\ c_1, \\dots;\\ \\mu_1, \\dots\\big),

computed from what it is **given**, parameters and other derived parameters
:math:`x_{g_i}`, with constants :math:`c_j` (a covariate per site, a
location) and memberships :math:`\\mu_k` (which PFT each site is). With
:math:`\\pi` the prior of the parameters, the distribution of :math:`y` is
the pushforward :math:`f_\\# \\pi`; it has no entries of theta and no prior
of its own. Its uses are non-centered hierarchies, regressions on
covariates, latent-factor or Gaussian-process fields, and indexing one
dim's values by another's, such as the location at each site that a
centered hierarchy's term is given.

:class:`DerivedParameters` holds the derived parameters over one vector, in
dependency order. It reads each constant and membership at the labels in
use before any call, so :math:`f` is a pure array function;
:class:`DerivedParameter` says what it receives and returns.

Functions and classes
---------------------
:class:`DerivedParameter`
    One node: a declaration of :math:`y` and its function.
:class:`DerivedParameters`
    The nodes over a vector, in dependency order: ``values``,
    ``values_to_dataset``, ``select``, ``parameters_behind``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.parameters._description import (
    as_shape,
    check_description_is_valid,
    labeled_form,
    resolved_element_labels,
)
from sipnet_calibration.parameters.labels import (
    aligned_constants,
    aligned_memberships,
    as_constants,
    as_coords,
    as_memberships,
)
from sipnet_calibration.parameters._probes import joint_probe_points
from sipnet_calibration.parameters._validation import (
    as_names,
    as_sequence,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.parameters.parameter import (
    check_shape_has_the_supports_event_axes,
)
from sipnet_calibration.parameters.support import Interval, Support
from sipnet_calibration.parameters.vector import (
    PARAMETER_LEVEL,
    ParameterVector,
    ValuesByParameter,
    check_batch_dims_name_the_leading_axes,
    check_labels_are_the_dims,
    check_labels_have_the_dims_type,
    check_selector_keeps_something,
    check_values_end_in_the_block_shape,
    check_values_share_a_batch_shape,
)

__all__ = [
    "DerivedParameter",
    "DerivedParameters",
    "check_derived_parameter_is_valid",
    "check_derived_parameters_are_valid",
]

Array = jax.Array


@dataclass(frozen=True, eq=False, kw_only=True)
class DerivedParameter:
    """A quantity computed deterministically from parameters and other
    derived parameters, :math:`y = f(x_g)`: like a parameter, one value at
    each tuple of labels of its dims, but with no entries of theta and no
    prior of its own.

    Parameters
    ----------
    name, units, shape, element_labels, indexed_by, long_name:
        As :class:`~sipnet_calibration.parameters.parameter.Parameter`'s,
        describing :math:`y`: *shape* is one value's, and its block is
        ``(*index shape, *shape)``, the index shape being the number of
        labels in use of each dim of *indexed_by*.
    support:
        A set every value of :math:`y` is declared to lie in, checked at the
        probe points; ``None`` when nothing is known. It documents
        :math:`y` and catches a wrong function early.
    given:
        The parameters and derived parameters :math:`y` is computed from;
        at least one.
    constants:
        ``{name: xr.DataArray}``: values *function* reads that are the same
        in every draw, such as a covariate per site or a longitude. Each is
        a constant as :mod:`~sipnet_calibration.parameters.labels` defines
        one, read at the labels of the collection's coords and of the
        element axes of :math:`y` and of what it is given, with
        *indexed_by* first, then :math:`y`'s element axes. Default none.
    memberships:
        ``{name: xr.DataArray}``: for each label of one dim, the label of
        another it belongs to, such as each site's PFT, a DataArray on
        ``site`` named ``"pft"``. Each is a membership as
        :mod:`~sipnet_calibration.parameters.labels` defines one, passed as
        ``int64`` positions, so ``x[pft_of_site]`` reads a PFT-level value at
        each site. Default none.
    function:
        ``function(**given, **constants, **memberships) -> y`` for **one
        draw**: each name in *given* its block, each constant and
        membership as read; it returns :math:`y`'s block. It is traced by
        JAX and vmapped over draws, so it is a pure function of its
        arguments.

    Raises
    ------
    TypeError, ValueError
        As :class:`~sipnet_calibration.parameters.parameter.Parameter`'s;
        for nothing given, a function that is not callable, a constant
        neither ``float64`` nor ``bool``, a membership not on one dim, or one
        keyword naming two things (a name given and a constant, say).

    Notes
    -----
    A function whose value at a label depends on other labels (a Gaussian
    process's non-centered field :math:`\\exp(m + L z)`, with :math:`L` the
    Cholesky factor of a kernel matrix over the sites) gives other values
    at the same labels after :meth:`DerivedParameters.select`, since
    :math:`L` is then over fewer points. The prior of :math:`y` is still its
    marginal, but a draw does not carry over between the two vectors.

    Standardize a covariate once, before it becomes a constant, never inside
    *function*: after a selection the constant holds fewer labels, and a
    standardization over them would change what the parameters mean.
    """

    name: str
    units: str | None
    support: Support | None = None
    shape: tuple[int, ...] = ()
    element_labels: Mapping[str, Sequence[str]] | None = None
    indexed_by: tuple[str, ...] = ()
    long_name: str | None = None
    given: Sequence[str]
    constants: Mapping[str, xr.DataArray] = field(default_factory=frozendict)
    memberships: Mapping[str, xr.DataArray] = field(default_factory=frozendict)
    function: Callable[..., Array]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", as_shape(self.shape, message_name=f"{self.name!r} shape"))
        object.__setattr__(self, "indexed_by", as_names(self.indexed_by, message_name=f"{self.name!r} indexed_by"))
        object.__setattr__(
            self, "element_labels", resolved_element_labels(self.name, self.shape, self.element_labels)
        )
        object.__setattr__(self, "given", as_names(self.given, message_name=f"{self.name!r} given"))
        object.__setattr__(self, "constants", as_constants(self.constants, message_name=f"{self.name!r} constants"))
        object.__setattr__(
            self, "memberships", as_memberships(self.memberships, message_name=f"{self.name!r} memberships")
        )
        check_derived_parameter_is_valid(self)

    def __repr__(self) -> str:
        indexed = f", indexed_by={self.indexed_by}" if self.indexed_by else ""
        return f"DerivedParameter(name={self.name!r}, given={list(self.given)}{indexed})"


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class DerivedParameters:
    """Derived parameters over one vector.

    Parameters
    ----------
    parameter_vector:
        The vector whose parameters they are computed from.
    derived_parameters:
        The :class:`DerivedParameter`\\ s, in any order; they are held and
        computed in dependency order.
    coords:
        Labels of the dims some derived parameter is indexed by that the
        vector lacks, and no other: a regression :math:`x_s =
        \\exp(\\beta^\\top w_s)` with :math:`\\beta` shared has ``site``
        here, since no parameter is on it. Default empty.

    Attributes
    ----------
    coords : frozendict of str to pandas.Index
        The vector's coords, then these: the labels every derived
        parameter's dims, constants and memberships are read at.
    derived_parameters : tuple of DerivedParameter
        In dependency order: each after the derived parameters it is given,
        otherwise in declaration order.
    derived_parameter_names : tuple of str
        Their names, in that order.

    Raises
    ------
    TypeError
        For a malformed argument.
    KeyError
        For a name given that is neither a parameter nor a derived
        parameter; a constant or membership missing a label of the coords;
        a membership whose value is not a label of its target dim.
    ValueError
        For a name given twice or shared with a parameter, a dim or an
        element axis; a cycle among the names given; a dim of
        ``indexed_by`` the coords lack; a coords dim the vector already has,
        or that no derived parameter uses; a value of the wrong shape at
        :math:`\\theta = 0`; or a value outside the declared support at the
        probe points.
    """

    parameter_vector: ParameterVector
    derived_parameters: Sequence[DerivedParameter]
    coords: Mapping[str, Sequence[Any]] = field(default_factory=frozendict)
    _own_coords: Mapping[str, pd.Index] = field(init=False)
    _aligned: Mapping[str, tuple[dict[str, Array], dict[str, np.ndarray]]] = field(init=False)

    def __post_init__(self) -> None:
        check_derived_parameters_are_derived_parameters(self.derived_parameters)
        object.__setattr__(self, "derived_parameters", tuple(self.derived_parameters))
        own = as_coords(self.coords)
        check_coords_are_new_dims(own, self.parameter_vector)
        check_coords_are_not_element_axes(own, self.parameter_vector)
        object.__setattr__(self, "_own_coords", own)
        object.__setattr__(self, "coords", frozendict({**self.parameter_vector.coords, **own}))
        check_derived_parameters_are_valid(self)
        by_name = {d.name: d for d in self.derived_parameters}
        object.__setattr__(self, "derived_parameters", tuple(by_name[n] for n in _dependency_order(self)))
        object.__setattr__(
            self,
            "_aligned",
            frozendict({
                d.name: (
                    aligned_constants(d.constants, self._labels_for(d), dim_order=(*d.indexed_by, *d.element_labels),
                                      message_name=f"{d.name!r} constants"),
                    aligned_memberships(d.memberships, self.coords, message_name=f"{d.name!r} memberships"),
                )
                for d in self.derived_parameters
            }),
        )
        for derived in self.derived_parameters:
            check_derived_parameter_has_its_block_shape(derived, self)
            if derived.support is not None:
                check_derived_parameter_lies_in_its_support(derived, self)

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, name: str) -> DerivedParameter:
        """The derived parameter called *name*; ``KeyError`` for any other."""
        check_derived_parameter_names_are_held([name], self)
        return self.derived_parameters[self.derived_parameter_names.index(name)]

    def __contains__(self, name: object) -> bool:
        try:
            return name in self.derived_parameter_names
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.derived_parameter_names)

    def __len__(self) -> int:
        return len(self.derived_parameters)

    def __repr__(self) -> str:
        return (
            f"DerivedParameters(derived_parameter_names={list(self.derived_parameter_names)}, "
            f"over={list(self.parameter_vector)})"
        )

    @property
    def derived_parameter_names(self) -> tuple[str, ...]:
        """The derived parameters' names, in dependency order."""
        return tuple(d.name for d in self.derived_parameters)

    def block_shape(self, name: str) -> tuple[int, ...]:
        """A derived parameter's block shape, ``(*index shape, *shape)``."""
        derived = self[name]
        return (*(len(self.coords[d]) for d in derived.indexed_by), *derived.shape)

    def parameters_behind(self, names: Sequence[str]) -> tuple[str, ...]:
        """The vector's parameters among *names* and those the derived
        parameters among them are computed from, directly or through
        others, in declaration order."""
        behind, pending = set(), list(names)
        while pending:
            name = pending.pop()
            if name in self.derived_parameter_names:
                pending.extend(self[name].given)
            else:
                behind.add(name)
        return tuple(n for n in self.parameter_vector.parameter_names if n in behind)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Sequence[Any]) -> DerivedParameters:
        """Over ``parameter_vector.select(**selectors)``, with the selectors
        on this collection's own coords applied to them too.

        It keeps each derived parameter whose names given are all kept,
        its constants and memberships read again at the kept labels. A dim of
        the vector that no kept parameter is indexed by, but a kept derived
        parameter is, becomes one of the collection's own coords.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`ParameterVector.select`; ``KeyError`` too if a kept
            membership points to a label the selection removed.
        """
        vector_selectors = {k: v for k, v in selectors.items() if k == PARAMETER_LEVEL or k in self.parameter_vector.coords}
        own = dict(self._own_coords)
        for key, value in selectors.items():
            if key in vector_selectors:
                continue
            check_selector_is_a_dim_of_the_coords(key, self)
            wanted = as_sequence(value, message_name=f"select {key}=")
            check_names_are_unique(wanted, message_name=f"select {key}=")
            check_selector_keeps_something(key, wanted)
            check_labels_have_the_dims_type(key, wanted, own[key])
            check_labels_are_the_dims(key, wanted, own[key])
            own[key] = own[key][own[key].isin(wanted)]
        vector = self.parameter_vector.select(**vector_selectors)
        available, kept = set(vector.parameter_names), []
        for derived in self.derived_parameters:
            if available.issuperset(derived.given):
                kept.append(derived)
                available.add(derived.name)
        used = {d for derived in kept for d in derived.indexed_by}
        # A dim of the vector that no kept parameter is on, but a kept derived
        # parameter is, becomes this collection's own, at the kept labels.
        for dim in used - set(vector.coords) - set(own):
            labels = self.parameter_vector.coords[dim]
            if dim in vector_selectors:
                labels = labels[labels.isin(list(vector_selectors[dim]))]
            own[dim] = labels
        return DerivedParameters(
            parameter_vector=vector,
            derived_parameters=[d for d in self.derived_parameters if d in kept],
            coords={d: labels for d, labels in own.items() if d in used},
        )

    # ── evaluation ────────────────────────────────────────────────────────────

    def values(
        self, values_by_parameter: Mapping[str, Any], *, derived_parameter_names: Sequence[str] | None = None
    ) -> ValuesByParameter:
        """The derived parameters, from the parameters' natural values.
        Traceable.

        Parameters
        ----------
        values_by_parameter:
            ``{name: (*batch, *block shape)}``, holding every parameter the
            requested derived parameters are computed from; derived values
            it already holds are used as they are.
        derived_parameter_names:
            The derived parameters wanted; those they are given are computed
            too. ``None``: every one.

        Returns
        -------
        ValuesByParameter
            ``{name: (*batch, *block shape)}``, ``float64``, for the wanted
            derived parameters and those they need, in dependency order.

        Raises
        ------
        KeyError
            For a name that is no derived parameter, or a parameter the
            values lack.
        ValueError
            If a value does not end in its block shape, or the batch shapes
            differ.
        """
        wanted = self._closure(derived_parameter_names)
        needed = self.parameters_behind(sorted(wanted))
        check_values_hold_the_parameters(values_by_parameter, needed)
        batch_ndim = self._batch_ndim_of(values_by_parameter, needed)
        values = dict(values_by_parameter)
        out = {}
        for derived in self.derived_parameters:
            if derived.name in wanted:
                if derived.name not in values:
                    values[derived.name] = self._evaluate(derived, values, batch_ndim=batch_ndim)
                out[derived.name] = values[derived.name]
        return out

    def values_to_dataset(self, values_by_parameter: Mapping[str, Any], *, batch_dims: Sequence[str] = ()) -> xr.Dataset:
        """The labeled form of the derived parameters *values_by_parameter*
        holds, laid out as the vector's (:mod:`~sipnet_calibration.parameters.vector`'s
        data model): one ``float64`` variable per derived parameter on
        ``(*batch_dims, *indexed_by, *element axes)``, with ``units``,
        ``support`` (when declared) and ``long_name`` (when set).

        Raises
        ------
        ValueError
            If a value does not end in its block shape, the batch shapes
            differ, or *batch_dims* does not name one dim per leading axis.
        """
        batch_dims = as_names(batch_dims, message_name="batch_dims")
        present = [name for name in self.derived_parameter_names if name in values_by_parameter]
        batches = []
        for name in present:
            shape = tuple(jnp.shape(values_by_parameter[name]))
            check_values_end_in_the_block_shape(name, shape, self.block_shape(name))
            batches.append((name, shape[: len(shape) - len(self.block_shape(name))]))
        if not present:
            return xr.Dataset()
        check_values_share_a_batch_shape(batches)
        batch = batches[0][1]
        check_batch_dims_name_the_leading_axes(batch_dims, batch)
        return labeled_form(
            [self[name] for name in present], values_by_parameter, self.coords,
            batch_dims=batch_dims, batch_shape=batch,
        )

    # ── supporting methods ────────────────────────────────────────────────────

    def _labels_for(self, derived: DerivedParameter) -> Mapping[str, pd.Index]:
        """The labels *derived*'s constants are read at: the coords, and the
        element axes of its own value and of what it is given."""
        given = [self.parameter_vector[n] if n in self.parameter_vector else self[n] for n in derived.given]
        element_axes = {axis: labels for p in (derived, *given) for axis, labels in p.element_labels.items()}
        return {**element_axes, **self.coords}

    def _evaluate(self, derived: DerivedParameter, values_by_parameter: Mapping[str, Any], *, batch_ndim: int) -> Array:
        """*derived* for every draw: *values_by_parameter* are ``{name:
        (*batch, *block shape)}`` over the leading *batch_ndim* axes, and its
        function is vmapped over ``*batch`` with its constants and
        memberships read at the labels in use."""
        constants, memberships = self._aligned[derived.name]
        given = {name: jnp.asarray(values_by_parameter[name], dtype=jnp.float64) for name in derived.given}
        batch = jnp.shape(given[derived.given[0]])[:batch_ndim]

        def one_draw(draw: Mapping[str, Array]) -> Array:
            return jnp.asarray(derived.function(**draw, **constants, **memberships), dtype=jnp.float64)

        if not batch:
            return one_draw(given)
        flat = {name: value.reshape((-1, *value.shape[batch_ndim:])) for name, value in given.items()}
        out = jax.vmap(one_draw)(flat)
        return out.reshape((*batch, *out.shape[1:]))

    def _closure(self, derived_parameter_names: Sequence[str] | None) -> set[str]:
        """The derived parameters *derived_parameter_names* need, themselves
        included; every one for ``None``."""
        if derived_parameter_names is None:
            return set(self.derived_parameter_names)
        wanted = set(as_names(derived_parameter_names, message_name="derived_parameter_names"))
        check_derived_parameter_names_are_held(wanted, self)
        pending = list(wanted)
        while pending:
            for name in self[pending.pop()].given:
                if name in self.derived_parameter_names and name not in wanted:
                    wanted.add(name)
                    pending.append(name)
        return wanted

    def _batch_ndim_of(self, values_by_parameter: Mapping[str, Any], needed: Sequence[str]) -> int:
        """The number of batch axes, read off the first parameter needed."""
        vector = self.parameter_vector
        batches = []
        for name in needed:
            shape = tuple(jnp.shape(values_by_parameter[name]))
            check_values_end_in_the_block_shape(name, shape, vector.block_shape(name))
            batches.append((name, shape[: len(shape) - len(vector.block_shape(name))]))
        check_values_share_a_batch_shape(batches)
        return len(batches[0][1])

    def _probe_values(self, derived: DerivedParameter) -> tuple[np.ndarray, Array]:
        """Theta at the joint probe points of the parameters *derived* is
        computed from (every other entry 0), and its values there."""
        vector = self.parameter_vector
        unconstrained = vector.unconstrained
        names = self.parameters_behind([derived.name])
        parts = joint_probe_points([
            (unconstrained.block_shape(n), math.prod(unconstrained[n].shape)) for n in names
        ])
        values = {n: jnp.zeros((len(parts[0]), *unconstrained.block_shape(n))) for n in unconstrained}
        values.update({n: jnp.asarray(points) for n, points in zip(names, parts)})
        theta = unconstrained.values_to_flat(values)
        natural = vector.flat_to_values(vector.to_natural(theta))
        return np.asarray(theta), self.values(natural, derived_parameter_names=[derived.name])[derived.name]


# ── helpers ───────────────────────────────────────────────────────────────────

#: The magnitude of theta beyond which a probe is an outer one, where a
#: derived value may round onto its support's boundary or overflow: every
#: probe but theta = 0 and +-3 * 1. A random direction of norm 10 already
#: underflows exp(m + tau z).
_OUTER_PROBE = 3.0


def _dependency_order(collection: DerivedParameters) -> tuple[str, ...]:
    """The derived parameters' names, each after those it is computed from,
    otherwise in declaration order; a cycle is refused by name."""
    derived = {d.name: d for d in collection.derived_parameters}
    order: list[str] = []
    state: dict[str, str] = {}

    def visit(name: str, path: list[str]) -> None:
        check_derived_parameters_are_acyclic(name, path, state)
        if state.get(name) == "done":
            return
        state[name] = "open"
        for parent in derived[name].given:
            if parent in derived:
                visit(parent, [*path, name])
        state[name] = "done"
        order.append(name)

    for name in derived:
        visit(name, [])
    return tuple(order)


def _lies_in_the_support_or_overflows(support: Support, values: Array) -> Array:
    """Whether each value lies in *support*'s closure, or is infinite at an
    end the support leaves unbounded, which is float64 overflow; NaN never
    does."""
    values = jnp.asarray(values, dtype=jnp.float64)
    inside = support.closure().contains(values)
    if isinstance(support, Interval):
        overflow = (jnp.isposinf(values) & np.isinf(support.high)) | (jnp.isneginf(values) & np.isinf(support.low))
        return inside | overflow
    return inside


# ── checks ────────────────────────────────────────────────────────────────────


def check_derived_parameter_is_valid(derived: DerivedParameter) -> None:
    """A derived parameter describes a value the labeled form can hold, is
    computed from something by a function, and reads each keyword once."""
    check_description_is_valid(
        name=derived.name,
        units=derived.units,
        element_labels=derived.element_labels,
        indexed_by=derived.indexed_by,
        long_name=derived.long_name,
        what="a derived parameter",
    )
    check_support_is_a_support_or_none(derived)
    if derived.support is not None:
        check_shape_has_the_supports_event_axes(derived.name, derived.shape, derived.support)
    check_derived_parameter_is_computed_from_something(derived)
    check_names_are_unique(derived.given, message_name=f"{derived.name!r} given")
    check_function_is_callable(derived)
    check_keywords_name_one_thing_each(derived)


def check_derived_parameters_are_valid(collection: DerivedParameters) -> None:
    """The derived parameters are named apart from everything else in the
    labeled form, are computed from what exists, and are on dims of the
    coords."""
    vector = collection.parameter_vector
    names = [d.name for d in collection.derived_parameters]
    check_names_are_unique(names, message_name="the derived parameter names")
    check_derived_names_are_free(names, collection)
    check_derived_element_axes_are_free(collection)
    for derived in collection.derived_parameters:
        check_given_names_exist(derived, names, vector)
        check_derived_dims_are_in_the_coords(derived, collection.coords)
    check_own_coords_are_used(collection)


def check_derived_parameters_are_derived_parameters(derived_parameters: Any) -> None:
    """The derived parameters are a sequence of :class:`DerivedParameter`."""
    items = as_sequence(derived_parameters, message_name="derived_parameters")
    wrong = [type(d).__name__ for d in items if not isinstance(d, DerivedParameter)]
    if wrong:
        raise TypeError(f"derived_parameters must be DerivedParameters, got {truncated(wrong)}.")


def check_support_is_a_support_or_none(derived: DerivedParameter) -> None:
    """A declared support is a :class:`Support`."""
    if derived.support is not None and not isinstance(derived.support, Support):
        raise TypeError(f"derived parameter {derived.name!r} has support {derived.support!r}; give a Support or None.")


def check_derived_parameter_is_computed_from_something(derived: DerivedParameter) -> None:
    """A derived parameter is given something, whose draws give its values
    their batch shape; a value fixed across draws is a constant."""
    if not derived.given:
        raise ValueError(
            f"derived parameter {derived.name!r} is computed from nothing; a value fixed across draws "
            "is a constant, not a derived parameter."
        )


def check_function_is_callable(derived: DerivedParameter) -> None:
    """A derived parameter's function is callable."""
    if not callable(derived.function):
        raise TypeError(f"derived parameter {derived.name!r} has a function {derived.function!r} that is not callable.")


def check_keywords_name_one_thing_each(derived: DerivedParameter) -> None:
    """No keyword the function receives names two things, one of which it
    would never see."""
    keywords = [*derived.given, *derived.constants, *derived.memberships]
    repeated = sorted({k for k in keywords if keywords.count(k) > 1})
    if repeated:
        raise ValueError(
            f"derived parameter {derived.name!r} receives {repeated} as two of a name given, a "
            "constant and a membership; name them apart."
        )


def check_coords_are_new_dims(own: Mapping[str, pd.Index], vector: ParameterVector) -> None:
    """A collection's own coords are of dims the vector lacks, whose labels
    the vector already gives otherwise."""
    shared = [d for d in own if d in vector.coords]
    if shared:
        raise ValueError(
            f"DerivedParameters(coords=) gives {shared}, which the vector already has; give only the "
            "dims derived parameters use and the vector lacks."
        )


def check_coords_are_not_element_axes(own: Mapping[str, pd.Index], vector: ParameterVector) -> None:
    """A collection's own dims are named like no element axis of a
    parameter, with which the labeled values and constants would confuse
    them."""
    axes = {axis for p in vector.parameters for axis in p.element_labels}
    clashing = [d for d in own if d in axes]
    if clashing:
        raise ValueError(
            f"DerivedParameters(coords=) gives {clashing}, which are element axes of parameters; "
            "name the dims apart."
        )


def check_derived_names_are_free(names: Sequence[str], collection: DerivedParameters) -> None:
    """No derived parameter is named like a parameter, a dim or an element
    axis, with which the labeled values would confuse it."""
    vector = collection.parameter_vector
    taken = {*vector.parameter_names, *collection.coords}
    taken |= {axis for p in vector.parameters for axis in p.element_labels}
    clashing = [n for n in names if n in taken]
    if clashing:
        raise ValueError(
            f"the derived parameters {clashing} are named like a parameter, a dim or an element axis; "
            "values are read by name, so rename them."
        )


def check_derived_element_axes_are_free(collection: DerivedParameters) -> None:
    """A derived parameter's element axis is named like no dim, parameter or
    derived parameter, and one it shares with another value has the same
    labels, since the labeled values would otherwise misalign them."""
    vector = collection.parameter_vector
    taken = {*collection.coords, *vector.parameter_names, *(d.name for d in collection.derived_parameters)}
    axes = {axis: labels for p in vector.parameters for axis, labels in p.element_labels.items()}
    for derived in collection.derived_parameters:
        for axis, labels in derived.element_labels.items():
            if axis in taken:
                raise ValueError(
                    f"derived parameter {derived.name!r} has an element axis {axis!r} named like a dim, a "
                    "parameter or a derived parameter; name its axes for what they index."
                )
            if axis in axes and not axes[axis].equals(labels):
                raise ValueError(
                    f"derived parameter {derived.name!r} names an element axis {axis!r} that another value "
                    "has with different labels, which the labeled values would misalign; name the axes apart."
                )
            axes[axis] = labels


def check_given_names_exist(derived: DerivedParameter, names: Sequence[str], vector: ParameterVector) -> None:
    """A derived parameter is given parameters and derived parameters that
    exist."""
    missing = [n for n in derived.given if n not in vector and n not in names]
    if missing:
        raise KeyError(
            f"derived parameter {derived.name!r} is given {missing}, which are neither parameters nor "
            "derived parameters; name ones that are."
        )


def check_derived_dims_are_in_the_coords(derived: DerivedParameter, coords: Mapping[str, pd.Index]) -> None:
    """Every dim a derived parameter is indexed by has labels, the vector's
    or the collection's own."""
    missing = [d for d in derived.indexed_by if d not in coords]
    if missing:
        raise ValueError(
            f"derived parameter {derived.name!r} is indexed by {missing}, which neither the vector nor "
            "DerivedParameters(coords=) give labels for; give them."
        )


def check_own_coords_are_used(collection: DerivedParameters) -> None:
    """Every one of a collection's own coords indexes some derived
    parameter, as the vector's coords index some parameter."""
    used = {d for derived in collection.derived_parameters for d in derived.indexed_by}
    unused = [d for d in collection._own_coords if d not in used]
    if unused:
        raise ValueError(f"DerivedParameters(coords=) gives {unused}, which no derived parameter is indexed by; drop them.")


def check_derived_parameters_are_acyclic(name: str, path: Sequence[str], state: Mapping[str, str]) -> None:
    """What the derived parameters are given forms no cycle, which no order
    of computation could satisfy."""
    if state.get(name) == "open":
        cycle = [*path[path.index(name):], name]
        raise ValueError(f"the derived parameters form a cycle, {' -> '.join(cycle)}; break it.")


def check_derived_parameter_names_are_held(names: Sequence[str] | set[str], collection: DerivedParameters) -> None:
    """Every name is one of the collection's derived parameters."""
    unknown = [n for n in names if n not in collection.derived_parameter_names]
    if unknown:
        raise KeyError(
            f"there is no derived parameter {truncated(sorted(unknown))}; name one of "
            f"{list(collection.derived_parameter_names)}."
        )


def check_values_hold_the_parameters(values_by_parameter: Mapping[str, Any], needed: Sequence[str]) -> None:
    """The values hold every parameter the wanted derived parameters are
    computed from, directly or through others."""
    missing = [n for n in needed if n not in values_by_parameter]
    if missing:
        raise KeyError(f"the derived parameters are computed from {missing}, which the values lack; give them.")


def check_selector_is_a_dim_of_the_coords(key: str, collection: DerivedParameters) -> None:
    """A selector names the parameters, a dim of the vector, or one of the
    collection's own dims."""
    if key not in collection._own_coords:
        raise KeyError(
            f"there is no dim {key!r}; select with parameter= or one of the dims {list(collection.coords)}."
        )


def check_derived_parameter_has_its_block_shape(derived: DerivedParameter, collection: DerivedParameters) -> None:
    """A derived parameter's function returns its block shape at
    :math:`\\theta = 0`, which would otherwise broadcast or misalign
    against the labels."""
    vector = collection.parameter_vector
    natural = vector.flat_to_values(vector.to_natural(jnp.zeros((1, vector.unconstrained.size))))
    value = collection.values(natural, derived_parameter_names=[derived.name])[derived.name]
    shape, expected = tuple(jnp.shape(value))[1:], collection.block_shape(derived.name)
    if shape != expected:
        raise ValueError(
            f"derived parameter {derived.name!r} computes a value of shape {shape} for one draw, but "
            f"its indexed_by and shape give {expected}; return (*index shape, *shape)."
        )


def check_derived_parameter_lies_in_its_support(derived: DerivedParameter, collection: DerivedParameters) -> None:
    """A derived parameter's values at the probe points lie in the support
    it declares; at the outer probes its closure passes, as does infinity at
    an unbounded end, which float64 underflow and overflow reach."""
    theta, values = collection._probe_values(derived)
    n_probes = len(theta)
    support = derived.support
    inside = support.contains(values).reshape((n_probes, -1)).all(axis=-1)
    lenient = _lies_in_the_support_or_overflows(support, values).reshape((n_probes, -1)).all(axis=-1)
    outer = np.abs(theta).max(axis=-1) > _OUTER_PROBE
    if not bool(jnp.all(jnp.where(outer, lenient, inside))):
        raise ValueError(
            f"derived parameter {derived.name!r} takes values outside its declared support "
            f"{support.name!r} at the probe points; declare the support its values have, or none."
        )

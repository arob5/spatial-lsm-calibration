"""Derived parameters: values computed deterministically from the
parameters.

A **derived parameter** is a deterministic node of the statistical model,

.. math::

    y = f\\big(x_{a_1}, \\dots, x_{a_m};\\ c_1, \\dots;\\ g_1, \\dots\\big),

computed from parameters and other derived parameters :math:`x_{a_i}` (its
``parameter_names``), labeled constants :math:`c_j` (a covariate per site,
a location) and memberships :math:`g_k` (which PFT each site is). With
:math:`\\pi` the prior of the parameters, the distribution of :math:`y` is
the pushforward :math:`f_\\# \\pi`; it has no entries of theta and no prior
of its own. Its uses are non-centered hierarchies, regressions on
covariates, and latent-factor or Gaussian-process fields. A formula that is
how SIPNET wants a value expressed (a unit reference, a formula of
``sipnet.c``) is a ``Compute`` rule of the SIPNET parameter map instead.

:class:`DerivedParameters` holds the derived parameters over one vector. It
reads each constant and membership at the coords' labels before any call,
so :math:`f` is a pure array function: it receives one draw's value of each
parameter name, of its block shape, each constant of the lengths of its
dims, and each membership as ``int64`` positions, and returns
:math:`y` of block shape ``(*[len(coords[d]) for d in indexed_by], *shape)``.

Functions and classes
---------------------
:class:`DerivedParameter`
    One node: its description and its function.
:class:`DerivedParameters`
    The nodes over a vector, in dependency order: ``values``,
    ``values_to_dataset``, ``select``.
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
    resolved_element_labels,
)
from sipnet_calibration.parameters._labels import (
    aligned_constants,
    aligned_memberships,
    as_constants,
    as_coords,
    as_memberships,
)
from sipnet_calibration.parameters._validation import as_names, as_sequence, check_names_are_unique, truncated
from sipnet_calibration.parameters.parameter import check_shape_has_the_supports_event_axes
from sipnet_calibration.parameters.support import Interval, Support, joint_probe_points
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
    """A value computed deterministically from parameters and other
    derived parameters: :math:`y = f(x)`.

    Parameters
    ----------
    name, units, shape, element_labels, indexed_by, long_name:
        As :class:`~sipnet_calibration.parameters.parameter.Parameter`'s,
        describing :math:`y`.
    support:
        A set every value of :math:`y` is declared to lie in, checked at the
        probe points; ``None`` when nothing is known. It documents
        :math:`y` and catches a wrong function early; a SIPNET rule's
        requirements on :math:`y` are checked on its values either way.
    parameter_names:
        The parameters and derived parameters :math:`y` is computed from;
        at least one.
    constants:
        ``{name: xr.DataArray}``: values *function* reads that are the same
        in every draw, each on dims of the collection's coords, keyed by
        label (a covariate on ``site``, a longitude). They are passed at the
        coords' labels, in their order, so they may hold more labels.
        ``float64``, or ``bool``, which stays boolean.
    memberships:
        ``{name: xr.DataArray}``: for each label of one dim of the coords,
        the label of another dim it belongs to, as a one-dimensional
        DataArray named for the other dim (``site_dims.labels("pft")`` is
        on ``site`` and named ``"pft"``). Each is passed as ``int64``
        positions into the other dim's labels, so ``x[pft_of_site]`` reads a
        PFT-level value at each site.
    function:
        ``function(**parameters, **constants, **memberships) -> y`` for
        **one draw**: each parameter name's value of its block shape, each
        constant of the lengths of its dims, each membership of the length
        of its dim; :math:`y` of block shape ``(*[len(coords[d]) for d in
        indexed_by], *shape)``. Traceable by JAX.

    Raises
    ------
    TypeError, ValueError
        As :class:`~sipnet_calibration.parameters.parameter.Parameter`'s;
        for no parameter names, a function that is not callable, a constant
        neither ``float64`` nor ``bool``, a membership not on one dim, or
        one keyword naming two things (a parameter and a constant, say).

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
    parameter_names: Sequence[str]
    constants: Mapping[str, xr.DataArray] = field(default_factory=frozendict)
    memberships: Mapping[str, xr.DataArray] = field(default_factory=frozendict)
    function: Callable[..., Array]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", as_shape(self.shape, message_name=f"{self.name!r} shape"))
        object.__setattr__(self, "indexed_by", as_names(self.indexed_by, message_name=f"{self.name!r} indexed_by"))
        object.__setattr__(
            self, "element_labels", resolved_element_labels(self.name, self.shape, self.element_labels)
        )
        object.__setattr__(
            self, "parameter_names", as_names(self.parameter_names, message_name=f"{self.name!r} parameter_names")
        )
        object.__setattr__(self, "constants", as_constants(self.constants, message_name=f"{self.name!r} constants"))
        object.__setattr__(
            self, "memberships", as_memberships(self.memberships, message_name=f"{self.name!r} memberships")
        )
        check_derived_parameter_is_valid(self)

    def __call__(
        self,
        values_by_parameter: Mapping[str, Any],
        constants: Mapping[str, Any],
        memberships: Mapping[str, Any],
        *,
        batch_ndim: int,
    ) -> Array:
        """:math:`y` for every draw: *values_by_parameter* are ``{name:
        (*batch, *block shape)}`` over the leading *batch_ndim* axes, and
        *constants* and *memberships* already read at the coords' labels.
        The function is vmapped over ``*batch``. :class:`DerivedParameters`
        does the reading, so this is how it evaluates one node, not how a
        caller should."""
        inputs = {name: jnp.asarray(values_by_parameter[name], dtype=jnp.float64) for name in self.parameter_names}
        batch = jnp.shape(inputs[self.parameter_names[0]])[:batch_ndim]

        def one_draw(draw: Mapping[str, Array]) -> Array:
            return jnp.asarray(self.function(**draw, **constants, **memberships), dtype=jnp.float64)

        if not batch:
            return one_draw(inputs)
        flat = {name: value.reshape((-1, *value.shape[batch_ndim:])) for name, value in inputs.items()}
        out = jax.vmap(one_draw)(flat)
        return out.reshape((*batch, *out.shape[1:]))

    def __repr__(self) -> str:
        indexed = f", indexed_by={self.indexed_by}" if self.indexed_by else ""
        return f"DerivedParameter(name={self.name!r}, parameter_names={list(self.parameter_names)}{indexed})"


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class DerivedParameters:
    """Derived parameters over one vector.

    Parameters
    ----------
    parameter_vector:
        The vector whose parameters they are computed from.
    derived_parameters:
        The :class:`DerivedParameter`\\ s, in any order; they are computed in
        dependency order.
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
    names : tuple of str
        The derived parameters' names, in dependency order: each after the
        derived parameters it is computed from, otherwise in declaration
        order.

    Raises
    ------
    TypeError
        For a malformed argument.
    KeyError
        For a parameter name that is neither a parameter nor a derived
        parameter; a constant or membership missing a label of the coords;
        a membership whose value is not a label of its target dim.
    ValueError
        For a name given twice or shared with a parameter, a dim or an
        element axis; a cycle among the parameter names; a dim of
        ``indexed_by`` the coords lack; a coords dim the vector already has,
        or that no derived parameter uses; a value of the wrong shape at
        :math:`\\theta = 0`; or a value outside the declared support at the
        probe points.
    """

    parameter_vector: ParameterVector
    derived_parameters: Sequence[DerivedParameter]
    coords: Mapping[str, Sequence[Any]] = field(default_factory=frozendict)
    names: tuple[str, ...] = field(init=False)
    _own_coords: Mapping[str, pd.Index] = field(init=False)
    _aligned: Mapping[str, tuple[dict[str, Array], dict[str, np.ndarray]]] = field(init=False)

    def __post_init__(self) -> None:
        check_derived_parameters_are_derived_parameters(self.derived_parameters)
        object.__setattr__(self, "derived_parameters", tuple(self.derived_parameters))
        own = as_coords(self.coords)
        check_coords_are_new_dims(own, self.parameter_vector)
        object.__setattr__(self, "_own_coords", own)
        object.__setattr__(self, "coords", frozendict({**self.parameter_vector.coords, **own}))
        check_derived_parameters_are_valid(self)
        object.__setattr__(self, "names", _dependency_order(self))
        object.__setattr__(
            self,
            "_aligned",
            frozendict({
                d.name: (
                    aligned_constants(d.constants, self.coords, message_name=f"{d.name!r} constants"),
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
        return self.derived_parameters[[d.name for d in self.derived_parameters].index(name)]

    def __contains__(self, name: object) -> bool:
        try:
            return name in self.names
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.derived_parameters)

    def __repr__(self) -> str:
        return f"DerivedParameters(names={list(self.names)}, over={list(self.parameter_vector)})"

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
            if name in self.names:
                pending.extend(self[name].parameter_names)
            else:
                behind.add(name)
        return tuple(n for n in self.parameter_vector.parameter_names if n in behind)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Sequence[Any]) -> DerivedParameters:
        """Over ``parameter_vector.select(**selectors)``, with the selectors
        on this collection's own coords applied to them too.

        It keeps each derived parameter whose parameter names are all kept,
        its constants and memberships read again at the kept labels.

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
        for name in self.names:
            derived = self[name]
            if available.issuperset(derived.parameter_names):
                kept.append(derived)
                available.add(name)
        used = {d for derived in kept for d in derived.indexed_by}
        return DerivedParameters(
            parameter_vector=vector,
            derived_parameters=[d for d in self.derived_parameters if d in kept],
            coords={d: labels for d, labels in own.items() if d in used},
        )

    # ── evaluation ────────────────────────────────────────────────────────────

    def values(self, values_by_parameter: Mapping[str, Any], *, names: Sequence[str] | None = None) -> ValuesByParameter:
        """The derived parameters, from the parameters' natural values.
        Traceable.

        Parameters
        ----------
        values_by_parameter:
            ``{name: (*batch, *block shape)}``, holding every parameter the
            requested derived parameters are computed from; derived values
            it already holds are used as they are.
        names:
            The derived parameters wanted; those they are computed from are
            computed too. ``None``: every one.

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
        wanted = self._closure(names)
        needed = self.parameters_behind(sorted(wanted))
        check_values_hold_the_parameters(values_by_parameter, needed)
        batch_ndim = self._batch_ndim_of(values_by_parameter, needed)
        values = dict(values_by_parameter)
        out = {}
        for name in self.names:
            if name in wanted:
                if name not in values:
                    constants, memberships = self._aligned[name]
                    values[name] = self[name](values, constants, memberships, batch_ndim=batch_ndim)
                out[name] = values[name]
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
        present = [name for name in self.names if name in values_by_parameter]
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
        variables, coordinates = {}, {}
        for name in present:
            derived = self[name]
            coordinates.update({d: (d, np.asarray(self.coords[d])) for d in derived.indexed_by})
            coordinates.update({axis: (axis, np.asarray(labels)) for axis, labels in derived.element_labels.items()})
            attributes = {} if derived.support is None else {"support": derived.support.name}
            if derived.units is not None:
                attributes["units"] = derived.units
            if derived.long_name is not None:
                attributes["long_name"] = derived.long_name
            variables[name] = (
                (*batch_dims, *derived.indexed_by, *derived.element_labels),
                np.asarray(values_by_parameter[name], dtype=np.float64),
                attributes,
            )
        coordinates.update({d: (d, np.arange(n, dtype=np.int64)) for d, n in zip(batch_dims, batch)})
        return xr.Dataset(variables, coords=coordinates)

    # ── supporting methods ────────────────────────────────────────────────────

    def _closure(self, names: Sequence[str] | None) -> set[str]:
        """The derived parameters *names* need, themselves included; every
        one for ``None``."""
        if names is None:
            return set(self.names)
        wanted = set(as_names(names, message_name="names"))
        check_derived_parameter_names_are_held(wanted, self)
        pending = list(wanted)
        while pending:
            for name in self[pending.pop()].parameter_names:
                if name in self.names and name not in wanted:
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
        return np.asarray(theta), self.values(natural, names=[derived.name])[derived.name]


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
        check_parameter_names_are_acyclic(name, path, state)
        if state.get(name) == "done":
            return
        state[name] = "open"
        for parent in derived[name].parameter_names:
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
    check_names_are_unique(derived.parameter_names, message_name=f"{derived.name!r} parameter_names")
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
    for derived in collection.derived_parameters:
        check_parameter_names_exist(derived, names, vector)
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
    """A derived parameter has parameter names, whose draws give its values
    their batch shape; a value fixed across draws is a constant."""
    if not derived.parameter_names:
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
    keywords = [*derived.parameter_names, *derived.constants, *derived.memberships]
    repeated = sorted({k for k in keywords if keywords.count(k) > 1})
    if repeated:
        raise ValueError(
            f"derived parameter {derived.name!r} receives {repeated} as both a parameter and a "
            "constant or membership; name them apart."
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


def check_parameter_names_exist(derived: DerivedParameter, names: Sequence[str], vector: ParameterVector) -> None:
    """A derived parameter is computed from parameters and derived
    parameters that exist."""
    missing = [n for n in derived.parameter_names if n not in vector and n not in names]
    if missing:
        raise KeyError(
            f"derived parameter {derived.name!r} is computed from {missing}, which are neither "
            "parameters nor derived parameters; name ones that are."
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


def check_parameter_names_are_acyclic(name: str, path: Sequence[str], state: Mapping[str, str]) -> None:
    """The derived parameters' parameter names form no cycle, which no order
    of computation could satisfy."""
    if state.get(name) == "open":
        cycle = [*path[path.index(name):], name]
        raise ValueError(f"the derived parameters form a cycle, {' -> '.join(cycle)}; break it.")


def check_derived_parameter_names_are_held(names: Sequence[str] | set[str], collection: DerivedParameters) -> None:
    """Every name is one of the collection's derived parameters."""
    unknown = [n for n in names if n not in collection.names]
    if unknown:
        raise KeyError(f"there is no derived parameter {truncated(sorted(unknown))}; name one of {list(collection.names)}.")


def check_values_hold_the_parameters(values_by_parameter: Mapping[str, Any], needed: Sequence[str]) -> None:
    """The values hold every parameter the wanted derived parameters are
    computed from."""
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
    value = collection.values(natural, names=[derived.name])[derived.name]
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

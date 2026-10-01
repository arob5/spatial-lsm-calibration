"""The parameter vector: parameters laid out in one flat array, and its
three forms.

Where this sits
---------------
::

    parameters.parameter.Parameter              (one value's description)
      -> parameters.vector.ParameterVector      (the layout: Flat, values by parameter, labeled)
      -> parameters.derived, parameters.prior   (what is computed from it, what is believed)
      -> the adapter layer (site_dims, sipnet_parameter_map, forward), through the labeled form

It imports NumPy, pandas, xarray, JAX and TFP, and nothing of the package
outside ``parameters``.

What it reads
-------------
Nothing from disk: a :class:`ParameterVector` is built from
:class:`~sipnet_calibration.parameters.parameter.Parameter`\\ s and the
labels of the dims they are indexed by (``coords``).

Data model
----------
Parameter :math:`p` has one value's shape :math:`s_p` and is indexed by
dims :math:`d_{p,1}, \\dots, d_{p,m}`, whose labels are ``coords``. Its
**index shape** is :math:`(n_{p,1}, \\dots, n_{p,m})`, the number of labels
of each; a **block** is its value at one tuple of labels, and its **block
shape** is ``(*index shape, *s_p)``. The vector holds
:math:`\\sum_p \\prod(\\text{index shape}) \\prod(s_p)` numbers, its
``size``.

**The order.** Blocks are sorted lexicographically by the levels named in
``order``, a permutation of ``("parameter", *dims)``: a parameter by its
declaration position; a dim by its label's position in ``coords``, a block
of a parameter not indexed by that dim coming before every label. A block's
numbers follow contiguously, in C order of the value. The default order,
``("parameter", *dims)``, puts each parameter's blocks together, sorted by
its labels in the order of the dims of ``coords``; ``("site", "parameter",
...)`` is site-major.

**The forms**, in natural space or, for :attr:`ParameterVector.unconstrained`,
in unconstrained space:

======================= ======================== ============================================
form                    type                     shape
======================= ======================== ============================================
Flat                    ``jax.Array``            ``(*batch, size)``
values by parameter     :data:`ValuesByParameter` ``{name: (*batch, *block shape)}``
labeled                 :data:`ParameterDataset`  ``xr.Dataset``, below
======================= ======================== ============================================

The labeled form, checked by :func:`validate_parameter_dataset`:

============ ================================================================
variables    one ``float64`` variable per parameter, named for it, on
             ``(*batch dims, *indexed_by, *element axes)``
coordinates  each dim, its labels; each element axis, its (string) labels;
             each batch dim, ``int64`` ``0`` to ``n - 1``; no attributes
attributes   per variable: ``units`` (omitted when ``None``), ``support`` (its
             name) and ``long_name`` (when set)
missing      never
============ ================================================================

**The spaces.** With :math:`T_p` parameter :math:`p`'s bijector and
:math:`\\theta_p` its values in :attr:`~ParameterVector.unconstrained`, whose
parameters are each :meth:`Parameter.unconstrained`,

.. math::

    x_p = T_p(\\theta_p), \\qquad \\theta_p = T_p^{-1}(x_p),

applied to every block, each value of shape :math:`s_p` mapped from its
unconstrained shape :math:`u_p`. :meth:`~ParameterVector.to_natural` and
:meth:`~ParameterVector.to_unconstrained` are these maps between Flat forms;
theta, with ``D = vector.unconstrained.size`` entries, is laid out by
``vector.unconstrained``. Labeled natural values from theta are
``vector.flat_to_dataset(vector.to_natural(theta), batch_dims=("sample",))``.

Functions and classes
---------------------
:class:`ParameterVector`
    Identity (``index``, ``entry_names``, ``describe``), the six conversions
    ``<source>_to_<target>`` between Flat (``flat``), values by parameter
    (``values``) and the labeled form (``dataset``), the spaces
    (``unconstrained``, ``to_natural``, ``to_unconstrained``, ``contains``),
    and ``select``/``positions``.
:data:`ValuesByParameter`, :data:`ParameterDataset`
    The aliases, with :func:`validate_values_by_parameter` and
    :func:`validate_parameter_dataset`.
:func:`check_parameter_vectors_share_a_layout`
    Whether two vectors give theta one meaning.

Notes
-----
**Values by parameter are the traceable, structured form**: Flat is
traceable but has no structure, and a parameter's entries are not one
slice under a non-default order; the labeled form is structured but does
not run under ``jax.jit``, ``jax.grad`` or ``jax.vmap``. So every prior,
derived and rule function receives values by parameter, which is also what
``jax.flatten_util.ravel_pytree`` unravels to.

**A value is one unit**: its numbers are contiguous in Flat, so applying a
bijector is a reshape, and a selection never splits a value.

Usage
-----
::

    vector = ParameterVector(
        parameters=[
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                      element_labels={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}),
            Parameter(name="initial_soil_carbon", support=POSITIVE, units="kg m-2", indexed_by=("site",)),
        ],
        coords={"pft": ["boreal.coniferous", "temperate.deciduous"], "site": [620, 865, 1037]},
    )
    vector.size, vector.unconstrained.size        # 1 + 8 + 3 = 12, and D = 1 + 6 + 3 = 10
    theta = jnp.zeros((4, vector.unconstrained.size))
    values_by_parameter = vector.flat_to_values(vector.to_natural(theta))   # {"allocation": (4, 2, 4), ...}
    parameter_dataset = vector.values_to_dataset(values_by_parameter, batch_dims=("sample",))
    vector.to_unconstrained(vector.dataset_to_flat(parameter_dataset))      # theta again
    vector.unconstrained.positions(site=[865])    # theta's entries a run at site 865 reads
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.parameters.labels import as_coords
from sipnet_calibration.parameters._probes import bijectors_agree, probe_points
from sipnet_calibration.parameters._validation import (
    as_names,
    as_sequence,
    check_names_are_unique,
    truncated,
)
from sipnet_calibration.parameters.parameter import Parameter

__all__ = [
    "ELEMENT_LEVEL",
    "PARAMETER_LEVEL",
    "ParameterDataset",
    "ParameterVector",
    "ValuesByParameter",
    "check_parameter_vectors_share_a_layout",
    "validate_parameter_dataset",
    "validate_values_by_parameter",
]

Array = jax.Array

#: The name of the order's and the index's level of parameters, and of the
#: selector that keeps some parameters.
PARAMETER_LEVEL = "parameter"

#: The name of the index's level of elements.
ELEMENT_LEVEL = "element"


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ParameterVector:
    """Parameters laid out in one flat array: each parameter's value
    replicated over the labels of its dims, the values placed in a stated
    order.

    Parameters
    ----------
    parameters:
        The :class:`~sipnet_calibration.parameters.parameter.Parameter`\\ s,
        with distinct names, in declaration order.
    coords:
        ``{dim: labels}`` for every dim some parameter is indexed by, and no
        other. Labels unique, all integers or all strings; kept with their
        dtype.
    order:
        A permutation of ``("parameter", *coords)``; the module's data
        model says how it orders the blocks. Default
        ``("parameter", *coords)``.

    Raises
    ------
    TypeError
        For a malformed argument, or labels of a dim that are neither all
        integers nor all strings.
    KeyError
        For a dim a parameter is indexed by that *coords* lacks.
    ValueError
        For no parameter; a name given twice; a dim with no or repeated
        labels, or that no parameter uses; an order that is not a
        permutation; or names that would collide in the labeled form (a
        parameter named for a dim, an element axis named for a dim or a
        parameter, one element axis name with two sets of labels).
    """

    parameters: Sequence[Parameter]
    coords: Mapping[str, Sequence[Any]] = field(default_factory=frozendict)
    order: Sequence[str] | None = None

    def __post_init__(self) -> None:
        check_parameters_are_parameters(self.parameters)
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "coords", as_coords(self.coords))
        default_order = (PARAMETER_LEVEL, *self.coords)
        order = default_order if self.order is None else as_names(self.order, message_name="order")
        object.__setattr__(self, "order", order)
        check_parameter_vector_is_valid(self)

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, name: str) -> Parameter:
        """The parameter called *name*; ``KeyError`` for any other."""
        check_parameter_names_are_held([name], self)
        return self.parameters[self.parameter_names.index(name)]

    def __contains__(self, name: object) -> bool:
        try:
            return name in self.parameter_names
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.parameter_names)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self.parameter_names)

    def __len__(self) -> int:
        return len(self.parameters)

    def __repr__(self) -> str:
        coords = {dim: len(labels) for dim, labels in self.coords.items()}
        return (
            f"ParameterVector(size={self.size}, order={self.order}, "
            f"parameters={list(self.parameter_names)}, coords={coords})"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter names, in declaration order."""
        return tuple(p.name for p in self.parameters)

    @property
    def dims(self) -> tuple[str, ...]:
        """The dims, in ``coords`` order."""
        return tuple(self.coords)

    @property
    def size(self) -> int:
        """The number of entries of the Flat form."""
        return int(self._layout.entry_block.size)

    def index_shape(self, name: str) -> tuple[int, ...]:
        """A parameter's index shape, ``[len(coords[d]) for d in indexed_by]``:
        how many values it has along each dim it is indexed by."""
        return tuple(len(self.coords[d]) for d in self[name].indexed_by)

    def block_shape(self, name: str) -> tuple[int, ...]:
        """A parameter's block shape, ``(*index shape, *shape)``: all its
        values, one of ``shape`` at each tuple of labels of its dims."""
        return (*self.index_shape(name), *self[name].shape)

    @cached_property
    def index(self) -> pd.MultiIndex:
        """One row per entry, in Flat order, levels ``("parameter", *dims,
        "element")``: a dim's level is NA where the parameter is not indexed
        by it, and ``element`` is the element's label for a value of rank 1,
        the tuple of its labels for a higher rank, and NA for a scalar."""
        layout = self._layout
        parameter = np.asarray(self.parameter_names, dtype=object)[layout.entry_parameter]
        levels = [parameter]
        for j, dim in enumerate(self.dims):
            positions = layout.entry_label_positions[:, j]
            labels = self.coords[dim]
            if labels.dtype.kind in "iu":
                nullable = f"{'Int' if labels.dtype.kind == 'i' else 'UInt'}{8 * labels.dtype.itemsize}"
                level = pd.array(labels.to_numpy()[np.maximum(positions, 0)], dtype=nullable)
                level[positions < 0] = pd.NA
            else:
                level = np.empty(len(positions), dtype=object)
                level[:] = [labels[k] if k >= 0 else None for k in positions]
            levels.append(level)
        levels.append(self._element_level())
        return pd.MultiIndex.from_arrays(levels, names=[PARAMETER_LEVEL, *self.dims, ELEMENT_LEVEL])

    @cached_property
    def entry_names(self) -> tuple[str, ...]:
        """One display string per entry, ``name[label, ...][element]``, the
        labels in ``indexed_by`` order and the long name standing for the
        name where set."""
        out = []
        for row in self.index:
            name, labels, element = row[0], row[1:-1], row[-1]
            parameter = self[name]
            entry = parameter.long_name or name
            by_dim = dict(zip(self.dims, labels))
            shown = [str(by_dim[dim]) for dim in parameter.indexed_by]
            if shown:
                entry += f"[{', '.join(shown)}]"
            if parameter.shape:
                entry += f"[{', '.join(element) if isinstance(element, tuple) else element}]"
            out.append(entry)
        return tuple(out)

    def describe(self) -> pd.DataFrame:
        """One row per parameter, indexed by ``parameter``: ``indexed_by``,
        ``shape``, ``support``, ``bijector``, ``units``, ``entries`` (in
        this vector's Flat form) and ``unconstrained_shape``."""
        rows = [
            {
                PARAMETER_LEVEL: p.name,
                "indexed_by": ", ".join(p.indexed_by),
                "shape": p.shape,
                "support": p.support.name,
                "bijector": p.bijector.name,
                "units": p.units,
                "entries": math.prod(self.block_shape(p.name)),
                "unconstrained_shape": p.unconstrained_shape,
            }
            for p in self.parameters
        ]
        return pd.DataFrame(rows).set_index(PARAMETER_LEVEL)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Sequence[Any]) -> ParameterVector:
        """A smaller vector: some parameters, at some labels of some dims.

        Parameters
        ----------
        **selectors:
            ``parameter=[names]`` keeps those parameters; ``<dim>=[labels]``
            keeps those labels of one of the vector's dims. Each value is a
            sequence, in any order. A level not named is kept whole.

        Returns
        -------
        ParameterVector
            The kept parameters, in this vector's declaration order; coords
            restricted to the kept labels, in this vector's label order, and
            to the dims a kept parameter is indexed by; this vector's order.
            A parameter not indexed by a selected dim is kept whole, since
            it does not vary along it: ``select(site=[4977])`` is the vector
            a calibration at site 4977 reads.

        Raises
        ------
        TypeError
            If a selector is one label rather than a sequence (pass
            ``[4977]``), a set or a mapping, or holds labels of the wrong
            type (``"4977"`` for a site).
        KeyError
            For a keyword that is neither ``"parameter"`` nor a dim of the
            vector, or a label its level lacks.
        ValueError
            For a label given twice, or a selector keeping nothing.

        Examples
        --------
        >>> vector.select(parameter=["allocation", "initial_soil_carbon"])
        >>> vector.select(site=[620, 865], pft=["boreal.coniferous"])
        """
        names, labels = self._selection(selectors)
        kept = [p for p in self.parameters if p.name in names]
        dims = [d for d in self.dims if any(d in p.indexed_by for p in kept)]
        coords = {d: self.coords[d][self.coords[d].isin(labels[d])] if d in labels else self.coords[d] for d in dims}
        order = [level for level in self.order if level == PARAMETER_LEVEL or level in dims]
        return ParameterVector(parameters=kept, coords=coords, order=order)

    def positions(self, **selectors: Sequence[Any]) -> np.ndarray:
        """Where the entries :meth:`select` keeps sit in this vector's Flat
        form.

        Parameters
        ----------
        **selectors:
            As :meth:`select`.

        Returns
        -------
        numpy.ndarray
            ``int64``, ascending. A selection keeps this vector's order, so
            ``flat[..., vector.positions(**s)]`` is the Flat form of
            ``vector.select(**s)``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`select`.

        Examples
        --------
        Theta's entries for one site, the parameters not indexed by site
        included:

        >>> vector.unconstrained.positions(site=[4977])
        """
        names, labels = self._selection(selectors)
        layout = self._layout
        codes = [i for i, name in enumerate(self.parameter_names) if name in names]
        mask = np.isin(layout.entry_parameter, codes)
        for j, dim in enumerate(self.dims):
            if dim in labels:
                kept = np.flatnonzero(self.coords[dim].isin(labels[dim]))
                positions = layout.entry_label_positions[:, j]
                mask &= (positions < 0) | np.isin(positions, kept)
        return np.flatnonzero(mask).astype(np.int64)

    # ── representations ───────────────────────────────────────────────────────

    def flat_to_values(self, flat: Any) -> ValuesByParameter:
        """Flat to values by parameter, a layout operation. Traceable.

        Parameters
        ----------
        flat:
            ``(*batch, size)``.

        Returns
        -------
        ValuesByParameter
            ``{name: (*batch, *block shape)}``, ``float64``.

        Raises
        ------
        ValueError
            If the last axis of *flat* is not ``size`` long.
        """
        flat = jnp.asarray(flat, dtype=jnp.float64)
        check_flat_ends_in_the_size(flat.shape, self.size)
        batch = flat.shape[:-1]
        return {
            p.name: flat[..., self._layout.positions_by_parameter[p.name]].reshape((*batch, *self.block_shape(p.name)))
            for p in self.parameters
        }

    def values_to_flat(self, values_by_parameter: Mapping[str, Any]) -> Array:
        """Values by parameter to Flat, a layout operation. Traceable.

        Parameters
        ----------
        values_by_parameter:
            :data:`ValuesByParameter`: every parameter's values, of one
            batch shape; keys that are not parameters are ignored.

        Returns
        -------
        jax.Array
            ``(*batch, size)``, ``float64``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_values_by_parameter`.
        """
        validate_values_by_parameter(values_by_parameter, self)
        batch = self._batch_shape_of(values_by_parameter)
        parameter_major = jnp.concatenate(
            [jnp.asarray(values_by_parameter[p.name], dtype=jnp.float64).reshape((*batch, -1)) for p in self.parameters],
            axis=-1,
        )
        return parameter_major[..., self._layout.flat_from_parameter_major]

    def values_to_dataset(self, values_by_parameter: Mapping[str, Any], *, batch_dims: Sequence[str] = ()) -> ParameterDataset:
        """Values by parameter to the labeled form, a layout operation.

        Parameters
        ----------
        values_by_parameter:
            As :meth:`values_to_flat` takes them.
        batch_dims:
            The names of the batch's axes, one per leading axis, labeled
            ``0`` to ``n - 1``.

        Returns
        -------
        ParameterDataset
            As the module's data model has it.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_values_by_parameter`; ``ValueError`` too for
            a number of batch dims other than the leading axes', or a batch
            dim named like a dim, an element axis or a parameter.
        """
        validate_values_by_parameter(values_by_parameter, self)
        batch_dims = as_names(batch_dims, message_name="batch_dims")
        batch = self._batch_shape_of(values_by_parameter)
        check_batch_dims_name_the_leading_axes(batch_dims, batch)
        check_batch_dim_names_are_free(batch_dims, self)
        coordinates: dict[str, Any] = {d: (d, np.asarray(self.coords[d])) for d in self.dims}
        for p in self.parameters:
            coordinates.update({axis: (axis, np.asarray(labels)) for axis, labels in p.element_labels.items()})
        coordinates.update({d: (d, np.arange(n, dtype=np.int64)) for d, n in zip(batch_dims, batch)})
        variables = {
            p.name: (
                (*batch_dims, *p.indexed_by, *p.element_labels),
                np.asarray(values_by_parameter[p.name], dtype=np.float64),
                _variable_attributes(p),
            )
            for p in self.parameters
        }
        return xr.Dataset(variables, coords=coordinates)

    def dataset_to_values(self, parameter_dataset: ParameterDataset, *, batch_dims: Sequence[str] | None = None) -> ValuesByParameter:
        """The labeled form to values by parameter, a layout operation.

        Parameters
        ----------
        parameter_dataset:
            A :data:`ParameterDataset` of this vector, as
            :func:`validate_parameter_dataset` has it; each dim's and
            element axis's labels in any order, read by label. Batch labels
            are not read: rows keep their order.
        batch_dims:
            The order of the batch dims, needed when there are several.

        Returns
        -------
        ValuesByParameter
            ``{name: (*batch, *block shape)}``, ``float64``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_parameter_dataset`.
        """
        batch = validate_parameter_dataset(parameter_dataset, self, batch_dims=batch_dims)
        out = {}
        for p in self.parameters:
            own = (*p.indexed_by, *p.element_labels)
            variable = parameter_dataset[p.name]
            selection = {d: list(self.coords[d]) for d in p.indexed_by}
            selection.update({axis: list(labels) for axis, labels in p.element_labels.items()})
            variable = variable.sel(selection).transpose(*batch, *own)
            out[p.name] = jnp.asarray(np.asarray(variable.values, dtype=np.float64))
        return out

    def flat_to_dataset(self, flat: Any, *, batch_dims: Sequence[str] = ()) -> ParameterDataset:
        """``values_to_dataset(flat_to_values(flat), batch_dims=...)``."""
        return self.values_to_dataset(self.flat_to_values(flat), batch_dims=batch_dims)

    def dataset_to_flat(self, parameter_dataset: ParameterDataset, *, batch_dims: Sequence[str] | None = None) -> Array:
        """``values_to_flat(dataset_to_values(parameter_dataset, batch_dims=...))``."""
        return self.values_to_flat(self.dataset_to_values(parameter_dataset, batch_dims=batch_dims))

    # ── the spaces ────────────────────────────────────────────────────────────

    @cached_property
    def unconstrained(self) -> ParameterVector:
        """The vector of each parameter's
        :meth:`~sipnet_calibration.parameters.parameter.Parameter.unconstrained`,
        on the same coords and order: theta's layout."""
        return ParameterVector(
            parameters=[p.unconstrained() for p in self.parameters], coords=self.coords, order=self.order
        )

    def to_natural(self, theta: Any) -> Array:
        """Theta to the natural Flat form: :math:`x_p = T_p(\\theta_p)` for
        every parameter, block by block. Traceable.

        Parameters
        ----------
        theta:
            ``(*batch, D)``, laid out by :attr:`unconstrained`.

        Returns
        -------
        jax.Array
            ``(*batch, size)``.

        Raises
        ------
        ValueError
            If the last axis of *theta* is not ``D`` long.
        """
        unconstrained = self.unconstrained.flat_to_values(theta)
        return self.values_to_flat({p.name: p.bijector.forward(unconstrained[p.name]) for p in self.parameters})

    def to_unconstrained(self, natural_flat: Any) -> Array:
        """The natural Flat form to theta: :math:`\\theta_p = T_p^{-1}(x_p)`.
        A value outside its support maps to non-finite entries, which is not
        an error. Traceable.

        Parameters
        ----------
        natural_flat:
            ``(*batch, size)``.

        Returns
        -------
        jax.Array
            ``(*batch, D)``.

        Raises
        ------
        ValueError
            If the last axis of *natural_flat* is not ``size`` long.

        Notes
        -----
        A traced function cannot raise on values, so this one does not
        check them; a caller converting values it did not draw checks
        :meth:`contains` first.
        """
        natural = self.flat_to_values(natural_flat)
        return self.unconstrained.values_to_flat({p.name: p.bijector.inverse(natural[p.name]) for p in self.parameters})

    def contains(self, natural_flat: Any) -> Array:
        """Whether every value of each row has a theta: it lies in its
        parameter's support, and :math:`T_p^{-1}` of it is finite, so it is
        not on a closed end, which no theta reaches. ``(*batch, size) ->
        (*batch,)``. Traceable.

        A natural value without a theta, such as a hand-set initial value of
        0 for a positive parameter, becomes a non-finite theta without any
        error, so this is the check to make before :meth:`to_unconstrained`
        on values that were not drawn.

        Raises
        ------
        ValueError
            If the last axis of *natural_flat* is not ``size`` long.
        """
        natural = self.flat_to_values(natural_flat)
        batch = jnp.shape(natural_flat)[:-1]
        inside = []
        for p in self.parameters:
            in_support = p.support.contains(natural[p.name]).reshape((*batch, -1)).all(axis=-1)
            reached = jnp.isfinite(p.bijector.inverse(natural[p.name])).reshape((*batch, -1)).all(axis=-1)
            inside.append(in_support & reached)
        return jnp.all(jnp.stack(inside, axis=len(batch)), axis=len(batch))

    # ── supporting methods ────────────────────────────────────────────────────

    @cached_property
    def _layout(self) -> _Layout:
        return _Layout.from_vector(self)

    def _batch_shape_of(self, values_by_parameter: Mapping[str, Any]) -> tuple[int, ...]:
        first = self.parameters[0].name
        shape = tuple(jnp.shape(values_by_parameter[first]))
        return shape[: len(shape) - len(self.block_shape(first))]

    def _element_level(self) -> np.ndarray:
        """The index's element level, entry by entry."""
        per_parameter = []
        for p in self.parameters:
            axes = list(p.element_labels.values())
            if not axes:
                per_parameter.append([None])
            elif len(axes) == 1:
                per_parameter.append(list(axes[0]))
            else:
                per_parameter.append([tuple(labels) for labels in pd.MultiIndex.from_product(axes)])
        layout = self._layout
        level = np.empty(self.size, dtype=object)
        level[:] = [per_parameter[p][k] for p, k in zip(layout.entry_parameter, layout.entry_element)]
        return level

    def _selection(self, selectors: Mapping[str, Any]) -> tuple[set[str], dict[str, list[Any]]]:
        """The parameter names and the labels per dim a selection keeps."""
        names = set(self.parameter_names)
        labels: dict[str, list[Any]] = {}
        for key, value in selectors.items():
            check_selector_is_a_level(key, self)
            wanted = as_sequence(value, message_name=f"select {key}=")
            check_names_are_unique(wanted, message_name=f"select {key}=")
            check_selector_keeps_something(key, wanted)
            if key == PARAMETER_LEVEL:
                check_parameter_names_are_held(wanted, self)
                names = set(wanted)
            else:
                check_labels_have_the_dims_type(key, wanted, self.coords[key])
                check_labels_are_the_dims(key, wanted, self.coords[key])
                labels[key] = wanted
        return names, labels


# ── the aliases and their validators ──────────────────────────────────────────

#: Values by parameter, ``{name: (*batch, *block shape)}``, as the module's
#: data model has them; checked by :func:`validate_values_by_parameter`.
type ValuesByParameter = dict[str, Array]

#: The labeled form, as the module's data model has it; checked by
#: :func:`validate_parameter_dataset`.
type ParameterDataset = xr.Dataset


def validate_values_by_parameter(values_by_parameter: Any, parameter_vector: ParameterVector) -> None:
    """Check that *values_by_parameter* are :data:`ValuesByParameter` of the
    vector: a mapping holding every parameter's values, each ending in its
    block shape, all of one batch shape. Other keys are not read.

    Raises
    ------
    TypeError
        If it is not a mapping.
    KeyError
        If a parameter has no values.
    ValueError
        If a value does not end in its block shape, or the batch shapes
        differ.
    """
    check_values_are_a_mapping(values_by_parameter)
    check_parameter_names_have_values(values_by_parameter, parameter_vector)
    batches = []
    for p in parameter_vector.parameters:
        shape = tuple(jnp.shape(values_by_parameter[p.name]))
        expected = parameter_vector.block_shape(p.name)
        check_values_end_in_the_block_shape(p.name, shape, expected)
        batches.append((p.name, shape[: len(shape) - len(expected)]))
    check_values_share_a_batch_shape(batches)


def validate_parameter_dataset(
    parameter_dataset: Any, parameter_vector: ParameterVector, *, batch_dims: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Check that *parameter_dataset* is a :data:`ParameterDataset` of the
    vector, as the module's data model has it, and return its batch dims in
    order.

    Its variables are exactly the vector's parameters (take
    ``dataset[list(vector)]`` first); each is on exactly its own dims and
    the batch dims, in any order; each dim's and element axis's labels are
    the vector's as a set; and every value is finite. The batch dims are
    the dims left over, the same for every variable.

    Returns
    -------
    tuple of str
        The batch dims: *batch_dims*, or the one or none there is.

    Raises
    ------
    TypeError
        If it is not an ``xr.Dataset``.
    KeyError
        If a parameter has no variable.
    ValueError
        For another variable; a variable missing one of its dims; a dim
        without an index coordinate, or whose labels are not the vector's;
        variables with different batch dims, several batch dims without
        *batch_dims*, or *batch_dims* not those; a missing or non-finite
        value.
    """
    check_parameter_dataset_is_a_dataset(parameter_dataset)
    check_dataset_variables_are_the_parameters(parameter_dataset, parameter_vector)
    leftover = []
    for p in parameter_vector.parameters:
        variable = parameter_dataset[p.name]
        own = (*p.indexed_by, *p.element_labels)
        check_variable_has_its_dims(p.name, variable.dims, own)
        for dim in p.indexed_by:
            check_variable_labels_are_the_vectors(p.name, dim, variable, parameter_vector.coords[dim])
        for axis, labels in p.element_labels.items():
            check_variable_labels_are_the_vectors(p.name, axis, variable, labels)
        check_variable_is_finite(p.name, variable)
        leftover.append((p.name, tuple(d for d in variable.dims if d not in own)))
    check_variables_share_their_batch_dims(leftover)
    found = leftover[0][1]
    if batch_dims is not None:
        batch_dims = as_names(batch_dims, message_name="batch_dims")
        check_batch_dims_are_the_datasets(batch_dims, found)
        return batch_dims
    check_batch_dims_are_ordered(found)
    return found


# ── the layout ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _Layout:
    """Where every number sits in Flat, computed once per vector.

    ``positions_by_parameter[name]`` is ``(*index shape, value size)``:
    the Flat position of each number of each block, in C order.
    ``flat_from_parameter_major`` reorders the parameter-major concatenation
    of every parameter's values, each in C order, into Flat. Per entry of
    Flat: ``entry_parameter`` (a declaration position), ``entry_block`` (a
    block number), ``entry_label_positions`` (``(size, n_dims)``, a label's
    position in ``coords``, ``-1`` where not indexed) and ``entry_element``
    (the element's position in C order).
    """

    positions_by_parameter: Mapping[str, np.ndarray]
    flat_from_parameter_major: np.ndarray
    entry_parameter: np.ndarray
    entry_block: np.ndarray
    entry_label_positions: np.ndarray
    entry_element: np.ndarray

    @classmethod
    def from_vector(cls, vector: ParameterVector) -> _Layout:
        """The layout of *vector*'s Flat."""
        block_parameter, block_label_positions = _blocks_of(vector)
        value_size = np.asarray([math.prod(p.shape) for p in vector.parameters])[block_parameter]
        flat_order = _blocks_in_flat_order(vector, block_parameter, block_label_positions)
        block_start = np.empty_like(value_size)
        block_start[flat_order] = np.cumsum(value_size[flat_order]) - value_size[flat_order]
        positions_by_parameter = {
            p.name: _value_positions(block_start[block_parameter == i], math.prod(p.shape)).reshape(
                (*vector.index_shape(p.name), math.prod(p.shape))
            )
            for i, p in enumerate(vector.parameters)
        }
        parameter_major = np.concatenate([positions_by_parameter[p.name].ravel() for p in vector.parameters])
        entry_block = np.repeat(flat_order, value_size[flat_order])
        return cls(
            positions_by_parameter=frozendict(positions_by_parameter),
            flat_from_parameter_major=_inverse_permutation(parameter_major),
            entry_parameter=block_parameter[entry_block],
            entry_block=entry_block,
            entry_label_positions=block_label_positions[entry_block],
            entry_element=np.arange(entry_block.size) - block_start[entry_block],
        )


def _blocks_of(vector: ParameterVector) -> tuple[np.ndarray, np.ndarray]:
    """Every block of *vector*, parameter by parameter in declaration order
    and each parameter's blocks in C order over its index shape: the
    parameter's declaration position, ``(n_blocks,)``, and the block's label
    position along each of the vector's dims, ``(n_blocks, n_dims)``, ``-1``
    along a dim its parameter is not indexed by."""
    block_parameter, block_label_positions = [], []
    for i, p in enumerate(vector.parameters):
        index_shape = vector.index_shape(p.name)
        n_blocks = math.prod(index_shape)
        grid = np.indices(index_shape).reshape(len(index_shape), n_blocks)
        label_positions = np.full((n_blocks, len(vector.dims)), -1, dtype=np.int64)
        for axis, dim in enumerate(p.indexed_by):
            label_positions[:, vector.dims.index(dim)] = grid[axis]
        block_parameter.append(np.full(n_blocks, i, dtype=np.int64))
        block_label_positions.append(label_positions)
    return np.concatenate(block_parameter), np.concatenate(block_label_positions)


def _blocks_in_flat_order(
    vector: ParameterVector, block_parameter: np.ndarray, block_label_positions: np.ndarray
) -> np.ndarray:
    """The block numbers sorted by the vector's order, level by level, where
    ``-1`` (not indexed) sorts before every label."""
    keys = [
        block_parameter if level == PARAMETER_LEVEL else block_label_positions[:, vector.dims.index(level)]
        for level in vector.order
    ]
    # np.lexsort sorts by its last key first.
    return np.lexsort(keys[::-1])


def _value_positions(block_start: np.ndarray, value_size: int) -> np.ndarray:
    """The Flat positions of each block's numbers, ``(n_blocks, value_size)``:
    a value is contiguous, from its block's start."""
    return block_start[:, None] + np.arange(value_size)[None, :]


def _inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    """The permutation ``q`` with ``q[permutation[k]] = k``."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    return inverse


# ── helpers ───────────────────────────────────────────────────────────────────


def _variable_attributes(parameter: Parameter) -> dict[str, Any]:
    attributes: dict[str, Any] = {"support": parameter.support.name}
    if parameter.units is not None:
        attributes["units"] = parameter.units
    if parameter.long_name is not None:
        attributes["long_name"] = parameter.long_name
    return attributes


def _same_entries(first: pd.MultiIndex, second: pd.MultiIndex) -> bool:
    """Whether two indexes name the same entries in the same order, labels
    compared by value whatever their dtype, NA alike."""
    return len(first) == len(second) and first.to_frame(index=False).astype(str).equals(
        second.to_frame(index=False).astype(str)
    )


def _is_integer_label(label: Any) -> bool:
    return isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))


# ── checks ────────────────────────────────────────────────────────────────────


def check_parameter_vector_is_valid(parameter_vector: ParameterVector) -> None:
    """The parameters, coords and order make one vector whose labeled form
    is well defined."""
    check_vector_has_a_parameter(parameter_vector.parameters)
    check_names_are_unique(parameter_vector.parameter_names, message_name="the parameter names")
    for p in parameter_vector.parameters:
        check_parameter_dims_are_in_the_coords(p, parameter_vector.coords)
    check_every_dim_is_used(parameter_vector)
    check_order_is_a_permutation(parameter_vector.order, (PARAMETER_LEVEL, *parameter_vector.coords))
    check_names_do_not_collide(parameter_vector)
    check_theta_element_axes_agree(parameter_vector.parameters)


def check_parameter_vectors_share_a_layout(first: ParameterVector, second: ParameterVector) -> None:
    """Two vectors give theta one meaning: the same index, coords and order
    in both spaces, the same supports, and transforms that agree at the
    probe points. Labels are compared by value, so site ids of two integer
    dtypes are one layout."""
    differences = [
        ("index", lambda: _same_entries(first.index, second.index)
         and _same_entries(first.unconstrained.index, second.unconstrained.index)),
        ("coords", lambda: list(first.coords) == list(second.coords) and all(
            first.coords[d].tolist() == second.coords[d].tolist() for d in first.coords
        )),
        ("order", lambda: first.order == second.order),
        ("supports", lambda: all(p.support == q.support for p, q in zip(first.parameters, second.parameters))),
        ("transforms", lambda: all(
            bijectors_agree(p.bijector, q.bijector, probe_points(p.unconstrained_shape))
            for p, q in zip(first.parameters, second.parameters)
        )),
    ]
    for what, same in differences:
        if not same():
            raise ValueError(
                f"the two parameter vectors differ in their {what}, so theta means different things "
                "under them; pair a prior and a forward model built on the same vector, as "
                "prior.parameter_vector."
            )


def check_parameters_are_parameters(parameters: Any) -> None:
    """The parameters are a sequence of :class:`Parameter`."""
    items = as_sequence(parameters, message_name="parameters")
    wrong = [type(p).__name__ for p in items if not isinstance(p, Parameter)]
    if wrong:
        raise TypeError(f"parameters must be Parameters, got {truncated(wrong)}.")


def check_vector_has_a_parameter(parameters: tuple[Parameter, ...]) -> None:
    """A vector holds at least one parameter, since an empty one fails far from its cause."""
    if not parameters:
        raise ValueError("a parameter vector needs at least one parameter; give parameters=[...].")


def check_parameter_dims_are_in_the_coords(parameter: Parameter, coords: Mapping[str, pd.Index]) -> None:
    """Every dim a parameter is indexed by has labels in the coords."""
    missing = [d for d in parameter.indexed_by if d not in coords]
    if missing:
        raise KeyError(
            f"parameter {parameter.name!r} is indexed by {missing}, which coords lack; give their "
            f"labels as coords={{dim: labels}} (coords hold {list(coords)})."
        )


def check_every_dim_is_used(parameter_vector: ParameterVector) -> None:
    """Every dim of the coords indexes some parameter, since the vector's
    coords are exactly its parameters' dims."""
    used = {d for p in parameter_vector.parameters for d in p.indexed_by}
    unused = [d for d in parameter_vector.coords if d not in used]
    if unused:
        raise ValueError(
            f"coords give the dims {unused}, which no parameter is indexed by; drop them (a dim only "
            "derived parameters use is DerivedParameters(coords=))."
        )


def check_order_is_a_permutation(order: tuple[str, ...], levels: tuple[str, ...]) -> None:
    """The order names every level once: the parameters and each dim."""
    if sorted(order) != sorted(levels):
        raise ValueError(f"order must be a permutation of {levels}, got {order}.")


def check_names_do_not_collide(parameter_vector: ParameterVector) -> None:
    """No parameter is named for a dim, no element axis for a dim or a
    parameter, and an element axis name shared by two parameters has one
    set of labels, since the labeled form would otherwise misalign them."""
    dims = set(parameter_vector.coords)
    names = set(parameter_vector.parameter_names)
    axes: dict[str, pd.Index] = {}
    for p in parameter_vector.parameters:
        if p.name in dims:
            raise ValueError(f"parameter {p.name!r} is named like a dim; name it for what it is.")
        for axis, labels in p.element_labels.items():
            if axis in dims or axis in names:
                raise ValueError(
                    f"parameter {p.name!r} has an element axis {axis!r} named like a dim or a "
                    "parameter; name its axes for what they index."
                )
            if axis in axes and not axes[axis].equals(labels):
                raise ValueError(
                    f"two parameters name an element axis {axis!r} with different labels, which the "
                    "labeled form would misalign; name the axes apart."
                )
            axes[axis] = labels


def check_theta_element_axes_agree(parameters: Sequence[Parameter]) -> None:
    """An element axis two parameters share has one set of labels in theta's
    layout too, where a simplex drops its last label: a simplex and another
    value on one axis would otherwise build, and fail at the first use of
    theta."""
    axes: dict[str, pd.Index] = {}
    for parameter in parameters:
        for axis, labels in parameter.unconstrained().element_labels.items():
            if axis in axes and not axes[axis].equals(labels):
                raise ValueError(
                    f"the element axis {axis!r} has different labels in theta's layout for two parameters "
                    "(a simplex drops its last label); name their axes apart."
                )
            axes[axis] = labels


def check_parameter_names_are_held(names: Sequence[Any], parameter_vector: ParameterVector) -> None:
    """Every name is one of the vector's parameters."""
    for name in names:
        if name not in parameter_vector.parameter_names:
            raise KeyError(
                f"the vector has no parameter {name!r}; name one of "
                f"{truncated(list(parameter_vector.parameter_names))}."
            )


def check_selector_is_a_level(key: str, parameter_vector: ParameterVector) -> None:
    """A selector names the parameters or one of the vector's dims."""
    if key != PARAMETER_LEVEL and key not in parameter_vector.coords:
        raise KeyError(
            f"the vector has no dim {key!r}; select with parameter= or one of its dims "
            f"{list(parameter_vector.coords)}."
        )


def check_selector_keeps_something(key: str, wanted: Sequence[Any]) -> None:
    """A selector keeps at least one label, since an empty one would keep nothing."""
    if not wanted:
        raise ValueError(f"select {key}=[] keeps nothing; name at least one, or omit the selector.")


def check_labels_have_the_dims_type(dim: str, wanted: Sequence[Any], labels: pd.Index) -> None:
    """A selector's labels have the dim's type, so a site id is never
    matched by a string."""
    integers = labels.dtype.kind in "iu"
    wrong = [w for w in wanted if not (_is_integer_label(w) if integers else isinstance(w, str))]
    if wrong:
        kind = "integers" if integers else "strings"
        raise TypeError(f"the labels of {dim!r} are {kind}, got {truncated(wrong)}; pass {kind}.")


def check_labels_are_the_dims(dim: str, wanted: Sequence[Any], labels: pd.Index) -> None:
    """Every label asked for is one of the dim's."""
    unknown = [w for w in wanted if w not in set(labels)]
    if unknown:
        raise KeyError(f"{dim!r} has no label(s) {truncated(unknown)}; name some of {truncated(list(labels))}.")


def check_flat_ends_in_the_size(shape: tuple[int, ...], size: int) -> None:
    """A Flat form's last axis has one entry per number of the vector."""
    if not shape or shape[-1] != size:
        raise ValueError(f"a Flat form of this vector ends in {size} entries, got shape {shape}; pass (..., {size}).")


def check_values_are_a_mapping(values_by_parameter: Any) -> None:
    """Values by parameter are a mapping, which is how they are read."""
    if not isinstance(values_by_parameter, Mapping):
        raise TypeError(
            f"values by parameter are a mapping of parameter names to arrays, got "
            f"{type(values_by_parameter).__name__}."
        )


def check_parameter_names_have_values(values_by_parameter: Mapping[str, Any], parameter_vector: ParameterVector) -> None:
    """Every parameter has values."""
    missing = [name for name in parameter_vector.parameter_names if name not in values_by_parameter]
    if missing:
        raise KeyError(f"the values hold no {truncated(missing)}; give every parameter's values.")


def check_values_end_in_the_block_shape(name: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """A parameter's values end in its block shape, which would otherwise
    broadcast or misalign silently."""
    if len(shape) < len(expected) or shape[len(shape) - len(expected):] != expected:
        raise ValueError(
            f"the values of {name!r} have shape {shape}, which does not end in its block shape "
            f"{expected}, (*index shape, *shape)."
        )


def check_values_share_a_batch_shape(batches: Sequence[tuple[str, tuple[int, ...]]]) -> None:
    """Every parameter's values have one batch shape, one value per draw."""
    if len({batch for _, batch in batches}) > 1:
        raise ValueError(f"the values have different batch shapes {dict(batches)}; give every parameter the same draws.")


def check_batch_dims_name_the_leading_axes(batch_dims: tuple[str, ...], batch: tuple[int, ...]) -> None:
    """One batch dim is named per leading axis of the values."""
    if len(batch_dims) != len(batch):
        raise ValueError(
            f"the values have {len(batch)} leading axes {batch}, but batch_dims names "
            f"{len(batch_dims)}: {batch_dims}; name one per axis."
        )


def check_batch_dim_names_are_free(batch_dims: tuple[str, ...], parameter_vector: ParameterVector) -> None:
    """A batch dim is named like no dim, element axis or parameter, with
    which the labeled form would confuse it."""
    check_names_are_unique(batch_dims, message_name="batch_dims")
    taken = {*parameter_vector.coords, *parameter_vector.parameter_names}
    taken |= {axis for p in parameter_vector.parameters for axis in p.element_labels}
    clashing = [d for d in batch_dims if d in taken]
    if clashing:
        raise ValueError(
            f"batch_dims {clashing} are named like a dim, an element axis or a parameter of the "
            "vector; name a batch dim for what it indexes, such as 'sample'."
        )


def check_parameter_dataset_is_a_dataset(parameter_dataset: Any) -> None:
    """A labeled form is an ``xr.Dataset``."""
    if not isinstance(parameter_dataset, xr.Dataset):
        raise TypeError(
            f"a parameter dataset is an xarray Dataset, got {type(parameter_dataset).__name__}; "
            "build it with values_to_dataset."
        )


def check_dataset_variables_are_the_parameters(parameter_dataset: xr.Dataset, parameter_vector: ParameterVector) -> None:
    """The labeled form holds exactly the vector's parameters, so nothing
    is read by mistake and nothing is missed."""
    held = [str(name) for name in parameter_dataset.data_vars]
    missing = [name for name in parameter_vector.parameter_names if name not in held]
    if missing:
        raise KeyError(f"the parameter dataset has no variable {truncated(missing)}; give every parameter.")
    extra = [name for name in held if name not in parameter_vector.parameter_names]
    if extra:
        raise ValueError(
            f"the parameter dataset holds {truncated(extra)}, which are not parameters of the vector; "
            "take dataset[list(vector)] first."
        )


def check_variable_has_its_dims(name: str, dims: tuple[Any, ...], own: tuple[str, ...]) -> None:
    """A variable is on every dim and element axis of its parameter, which
    would otherwise be broadcast or dropped silently."""
    missing = [d for d in own if d not in dims]
    if missing:
        raise ValueError(
            f"the parameter dataset's {name!r} is on {tuple(dims)}, which lacks {missing}; the two "
            "vectors do not define the parameter alike."
        )


def check_variable_labels_are_the_vectors(name: str, dim: str, variable: xr.DataArray, labels: pd.Index) -> None:
    """A variable's labels along a dim or element axis are labeled, once
    each, and are the vector's as a set, since xarray would otherwise read
    by position or twice."""
    if dim not in variable.indexes:
        raise ValueError(
            f"the parameter dataset's {name!r} has no {dim!r} coordinate, so its values cannot be "
            f"matched to labels; give {dim!r} its labels."
        )
    listed = variable.indexes[dim].tolist()
    held = set(listed)
    if len(held) != len(listed) or held != set(labels.tolist()):
        raise ValueError(
            f"the parameter dataset's {name!r} has {dim!r} labels {truncated(sorted(held, key=str))}, "
            f"not the vector's {truncated(list(labels))}; select the vector's labels first."
        )


def check_variable_is_finite(name: str, variable: xr.DataArray) -> None:
    """Every value of a variable is finite, as theta's always are."""
    if not np.isfinite(np.asarray(variable.values, dtype=np.float64)).all():
        raise ValueError(
            f"the parameter dataset's {name!r} holds a missing or non-finite value; drop or fill "
            "those draws first."
        )


def check_variables_share_their_batch_dims(leftover: Sequence[tuple[str, tuple[Any, ...]]]) -> None:
    """Every variable has the same batch dims, the dims beyond its own."""
    if len({frozenset(dims) for _, dims in leftover}) > 1:
        raise ValueError(
            f"the parameter dataset's variables have different batch dims {dict(leftover)}; give "
            "every parameter the same draws."
        )


def check_batch_dims_are_the_datasets(batch_dims: tuple[str, ...], found: tuple[Any, ...]) -> None:
    """``batch_dims`` names exactly the dataset's batch dims."""
    if sorted(batch_dims) != sorted(map(str, found)):
        raise ValueError(f"batch_dims {batch_dims} are not the parameter dataset's batch dims {found}.")


def check_batch_dims_are_ordered(found: tuple[Any, ...]) -> None:
    """Several batch dims are put in order by ``batch_dims``, which nothing
    else fixes."""
    if len(found) > 1:
        raise ValueError(
            f"the parameter dataset has the batch dims {found}; fix their order with batch_dims=."
        )

"""Layout: named arrays laid out as one flat vector, and the forms values
take across module boundaries.

Where this sits
---------------
::

    probability.spec.ArraySpec      (what a component is)
    probability.labels              (the coords of its dims)
      -> probability.layout.Layout  (Flat, values by name, labeled values)
      -> a model's parameters (theta's layout) and observations (y's)

It imports NumPy, pandas, xarray, JAX and TFP, and nothing of the package
outside ``probability``.

What it reads
-------------
Nothing from disk: a :class:`Layout` is built from
:class:`~sipnet_calibration.probability.spec.ArraySpec`\\ s and the labels of
the dims they are indexed by (``coords``, as
:mod:`~sipnet_calibration.probability.labels` defines them).

Data model
----------
Component :math:`c` has one value's shape :math:`s_c` and is indexed by
dims :math:`d_{c,1}, \\dots, d_{c,m}`. Its **block** is its values at every
tuple of labels of its dims, of shape ``(*index shape, *s_c)``, the index
shape being the number of labels of each. The layout holds :math:`\\sum_c
\\prod(\\text{block shape})` numbers, its ``size``.

**The order** is ProbPipe's canonical one: components in declaration order,
each block in C order, so each component's entries are one contiguous
slice.

**The forms**, in natural space or, for :attr:`Layout.unconstrained`, in
unconstrained space:

=============== ======================== =========================================
form            type                     shape
=============== ======================== =========================================
Flat            ``jax.Array``            ``(*batch, size)``
values by name  :data:`ValuesByName`     ``{name: (*batch, *block shape)}``
labeled         :data:`LabeledValues`    ``{name: xr.DataArray}``, below
=============== ======================== =========================================

Labeled values, checked by :func:`validate_labeled_values`:

============ ================================================================
values       one ``float64`` DataArray per component, named for it, on
             ``(*batch dims, *indexed_by, *element axes)``
coordinates  each plain dim, its labels; a stacked dim, its ``MultiIndex``
             (its levels as coordinates on it); each element axis, its string
             labels; each batch dim, ``int64`` ``0`` to ``n - 1``
attributes   ``units`` (omitted when ``None``), ``support`` (its name)
missing      never
============ ================================================================

It is a dict rather than one Dataset because two stacked dims with a
``site`` level cannot share one: xarray would take their levels for one
coordinate. :func:`encode_labeled_values` makes the one Dataset netCDF can
hold, and :func:`decode_labeled_values` reverses it.

**The index** has one row per entry, its levels ``("component", *plain dims,
*levels of stacked dims, "element")``, a level named twice being one level:
``positions(site=[4977])`` finds site 4977 in every component, through a
``site`` dim or a stacked dim's ``site`` level.

**The spaces.** With :math:`T_c` component :math:`c`'s bijector,

.. math::

    x_c = T_c(\\theta_c), \\qquad \\theta_c = T_c^{-1}(x_c),

applied value by value; :meth:`~Layout.to_natural` and
:meth:`~Layout.to_unconstrained` are these maps between Flat forms, theta
laid out by ``layout.unconstrained``.

Functions and classes
---------------------
:class:`Layout`
    Identity (``index``, ``entry_names``, ``describe``), the six conversions
    ``<source>_to_<target>`` between Flat (``flat``), values by name
    (``values``) and labeled values (``labeled``), the spaces
    (``unconstrained``, ``to_natural``, ``to_unconstrained``, ``contains``),
    and ``select``/``positions``.
:data:`ValuesByName`, :data:`LabeledValues`
    The aliases, with :func:`validate_values_by_name` and
    :func:`validate_labeled_values`.
:func:`encode_labeled_values`, :func:`decode_labeled_values`
    Labeled values as one Dataset netCDF can hold, and back.

Notes
-----
A layout takes no ``order``: its order is the canonical one, the components
in declaration order and each block in C order over its dims.

Usage
-----
::

    layout = Layout(
        [
            ArraySpec("respiration_share", units="1", support=OPEN_UNIT_INTERVAL),
            ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=("pft",),
                      element_axes={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}),
            ArraySpec("initial_soil_carbon", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
        ],
        coords={"pft": ["boreal.coniferous", "temperate.deciduous"], "site": [620, 865, 1037]},
    )
    layout.size, layout.unconstrained.size     # 1 + 8 + 3 = 12, and D = 1 + 6 + 3 = 10
    theta = jnp.zeros((4, layout.unconstrained.size))
    values_by_name = layout.flat_to_values(layout.to_natural(theta))       # {"allocation": (4, 2, 4), ...}
    labeled = layout.values_to_labeled(values_by_name, batch_dims=("sample",))
    layout.to_unconstrained(layout.labeled_to_flat(labeled))              # theta again
    layout.unconstrained.positions(site=[865])  # theta's entries a run at site 865 reads
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.probability._validation import as_names, as_sequence, check_names_are_unique, truncated
from sipnet_calibration.probability.labels import as_coords, indexer, is_stacked, label_kind, label_kinds
from sipnet_calibration.probability.names import COMPONENT_LEVEL, ELEMENT_LEVEL, RESERVED_NAMES
from sipnet_calibration.probability.spec import ArraySpec

__all__ = [
    "STACKED_DIMS_ATTRIBUTE",
    "LabeledValues",
    "Layout",
    "ValuesByName",
    "check_batch_dims_name_the_leading_axes",
    "check_flat_ends_in_the_size",
    "check_values_end_in_the_block_shape",
    "check_values_share_a_batch_shape",
    "decode_labeled_values",
    "encode_labeled_values",
    "validate_labeled_values",
    "validate_values_by_name",
]

Array = jax.Array

#: The attribute :func:`encode_labeled_values` writes: a JSON object
#: ``{stacked dim: [level, ...]}``.
STACKED_DIMS_ATTRIBUTE = "stacked_dims"


class Layout:
    """Named arrays laid out in one flat vector: each component's values
    replicated over the labels of its dims, components in declaration order,
    each block in C order.

    Parameters
    ----------
    components : Sequence[ArraySpec]
        Positional-only. Distinct names, in declaration order.
    coords : Mapping[str, Sequence]
        Keyword-only. ``{dim: labels}`` for every dim some component is
        indexed by, and no other: a plain dim's labels unique integers or
        strings, a stacked dim's a ``pandas.MultiIndex``
        (:mod:`~sipnet_calibration.probability.labels`).

    Attributes
    ----------
    components : tuple of ArraySpec
    coords : frozendict of str to pandas.Index

    Raises
    ------
    TypeError
        For a malformed argument, or labels of a kind the coords refuse.
    KeyError
        For a dim a component is indexed by that *coords* lacks.
    ValueError
        For no component; a name given twice; a dim with no or repeated
        labels, or that no component uses; a component indexed by two dims
        that share a level; or names that would collide (a component, dim,
        level or element axis named alike where the index or labeled values
        would confuse them, one element axis name with two sets of labels,
        or a reserved name).
    """

    def __init__(self, components: Sequence[ArraySpec], /, *, coords: Mapping[str, Any] | None = None) -> None:
        check_components_are_array_specs(components)
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "coords", as_coords({} if coords is None else coords))
        check_layout_is_valid(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a Layout is frozen; build another rather than setting {name!r}.")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_layout_from_arguments, (self.components, dict(self.coords)))

    # ── identity ──────────────────────────────────────────────────────────────

    def __getitem__(self, name: str) -> ArraySpec:
        """The component called *name*; ``KeyError`` for any other."""
        check_component_names_are_held([name], self)
        return self.components[self.component_names.index(name)]

    def __contains__(self, name: object) -> bool:
        try:
            return name in self.component_names
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.component_names)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self.component_names)

    def __len__(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        coords = {dim: len(labels) for dim, labels in self.coords.items()}
        return f"Layout(size={self.size}, components={list(self.component_names)}, coords={coords})"

    @property
    def component_names(self) -> tuple[str, ...]:
        """The component names, in declaration order."""
        return tuple(c.name for c in self.components)

    @property
    def dims(self) -> tuple[str, ...]:
        """The dims, in ``coords`` order."""
        return tuple(self.coords)

    @cached_property
    def level_names(self) -> tuple[str, ...]:
        """The index's levels: ``("component", *plain dims, *levels of
        stacked dims, "element")``, each name once."""
        levels: dict[str, None] = {d: None for d, labels in self.coords.items() if not is_stacked(labels)}
        for labels in self.coords.values():
            if is_stacked(labels):
                levels.update(dict.fromkeys(labels.names))
        return (COMPONENT_LEVEL, *levels, ELEMENT_LEVEL)

    @property
    def size(self) -> int:
        """The number of entries of the Flat form."""
        return int(self._tables.offsets[-1])

    def index_shape(self, name: str) -> tuple[int, ...]:
        """A component's index shape, ``[len(coords[d]) for d in indexed_by]``."""
        return tuple(len(self.coords[d]) for d in self[name].indexed_by)

    def block_shape(self, name: str) -> tuple[int, ...]:
        """A component's block shape, ``(*index shape, *shape)``."""
        return (*self.index_shape(name), *self[name].shape)

    def slice_of(self, name: str) -> slice:
        """Where a component's entries sit in Flat: one contiguous slice."""
        i = self.component_names.index(self[name].name)
        return slice(int(self._tables.offsets[i]), int(self._tables.offsets[i + 1]))

    @cached_property
    def index(self) -> pd.MultiIndex:
        """One row per entry, in Flat order, levels :attr:`level_names`: a
        dim's or level's value is NA where the component is not indexed by
        it, and ``element`` is the element's label for a value of rank 1, the
        tuple of its labels for a higher rank, and NA for a scalar."""
        tables = self._tables
        component = np.asarray(self.component_names, dtype=object)[tables.entry_component]
        arrays = [component]
        for level in self.level_names[1:-1]:
            arrays.append(self._level_values(level))
        arrays.append(self._element_level())
        return pd.MultiIndex.from_arrays(arrays, names=list(self.level_names))

    @cached_property
    def entry_names(self) -> tuple[str, ...]:
        """One display string per entry, ``name[label, ...][element]``, the
        labels in ``indexed_by`` order, a stacked dim's levels joined by
        ``", "``."""
        tables = self._tables
        out = []
        for entry in range(self.size):
            spec = self.components[tables.entry_component[entry]]
            shown = []
            for dim in spec.indexed_by:
                label = self.coords[dim][tables.entry_label_positions[entry, self.dims.index(dim)]]
                shown.append(", ".join(map(_label_text, label)) if isinstance(label, tuple) else _label_text(label))
            text = spec.name + (f"[{', '.join(shown)}]" if shown else "")
            if spec.shape:
                labels = [axis[k] for axis, k in zip(spec.element_axes.values(), np.unravel_index(tables.entry_element[entry], spec.shape))]
                text += f"[{', '.join(labels)}]"
            out.append(text)
        return tuple(out)

    def describe(self) -> pd.DataFrame:
        """One row per component, indexed by ``component``: ``indexed_by``,
        ``shape``, ``support``, ``bijector``, ``units``, ``entries`` (in this
        layout's Flat form) and ``unconstrained_shape``."""
        rows = [
            {
                COMPONENT_LEVEL: c.name,
                "indexed_by": ", ".join(c.indexed_by),
                "shape": c.shape,
                "support": c.support.name,
                "bijector": c.bijector.name,
                "units": c.units,
                "entries": math.prod(self.block_shape(c.name)),
                "unconstrained_shape": c.unconstrained_shape,
            }
            for c in self.components
        ]
        return pd.DataFrame(rows).set_index(COMPONENT_LEVEL)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(self, **selectors: Sequence[Any]) -> Layout:
        """A smaller layout: some components, at some labels.

        Parameters
        ----------
        **selectors:
            ``component=[names]`` keeps those components; ``<dim>=[labels]``
            keeps those labels of a dim (a stacked dim's labels as tuples);
            ``<level>=[labels]`` keeps the labels of every stacked dim whose
            label has one of them at that level, and of a plain dim of that
            name. Each value is a sequence, in any order. A component not
            indexed by a selected dim is kept whole.

        Returns
        -------
        Layout
            The kept components, in declaration order; coords restricted to
            the kept labels, in this layout's order, and to the dims a kept
            component is indexed by.

        Raises
        ------
        TypeError
            If a selector is one label rather than a sequence, a set or a
            mapping, or holds labels of the wrong kind (``"4977"`` for a
            site).
        KeyError
            For a keyword that is neither ``"component"``, a dim nor a
            level, or a label it has nowhere.
        ValueError
            For a label given twice, a selector keeping nothing, or a
            selection leaving a kept component's dim without a label.
        """
        names, kept = self._selection(selectors)
        components = [c for c in self.components if c.name in names]
        dims = [d for d in self.dims if any(d in c.indexed_by for c in components)]
        coords = {}
        for dim in dims:
            labels = self.coords[dim]
            if dim in kept:
                labels = labels[np.sort(kept[dim])]
                check_selection_keeps_a_label(dim, labels)
                if is_stacked(labels):
                    labels = labels.remove_unused_levels()
            coords[dim] = labels
        return Layout(components, coords=coords)

    def positions(self, **selectors: Sequence[Any]) -> np.ndarray:
        """Where the entries :meth:`select` keeps sit in this layout's Flat
        form: ``int64``, ascending, so ``flat[..., layout.positions(**s)]``
        is the Flat form of ``layout.select(**s)`` whenever that exists.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :meth:`select`, except that a dim left without a label is
            not an error: its components contribute no entry.
        """
        names, kept = self._selection(selectors)
        tables = self._tables
        codes = [i for i, name in enumerate(self.component_names) if name in names]
        mask = np.isin(tables.entry_component, codes)
        for dim, positions in kept.items():
            on_dim = tables.entry_label_positions[:, self.dims.index(dim)]
            mask &= (on_dim < 0) | np.isin(on_dim, positions)
        return np.flatnonzero(mask).astype(np.int64)

    # ── representations ───────────────────────────────────────────────────────

    def flat_to_values(self, flat: Any) -> ValuesByName:
        """Flat to values by name, a layout operation. Traceable.

        Parameters
        ----------
        flat:
            ``(*batch, size)``.

        Returns
        -------
        ValuesByName
            ``{name: (*batch, *block shape)}``, ``float64``.

        Raises
        ------
        ValueError
            If the last axis of *flat* is not ``size`` long.
        """
        flat = jnp.asarray(flat, dtype=jnp.float64)
        check_flat_ends_in_the_size(flat.shape, self.size)
        batch = flat.shape[:-1]
        return {c.name: flat[..., self.slice_of(c.name)].reshape((*batch, *self.block_shape(c.name))) for c in self.components}

    def values_to_flat(self, values_by_name: Mapping[str, Any]) -> Array:
        """Values by name to Flat, a layout operation. Traceable.

        Parameters
        ----------
        values_by_name:
            :data:`ValuesByName`: every component's values, of one batch
            shape; keys that are not components are ignored.

        Returns
        -------
        jax.Array
            ``(*batch, size)``, ``float64``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_values_by_name`.
        """
        validate_values_by_name(values_by_name, self)
        batch = self._batch_shape_of(values_by_name)
        return jnp.concatenate(
            [
                jnp.asarray(values_by_name[c.name], dtype=jnp.float64).reshape((*batch, math.prod(self.block_shape(c.name))))
                for c in self.components
            ],
            axis=-1,
        )

    def values_to_labeled(self, values_by_name: Mapping[str, Any], *, batch_dims: Sequence[str] = ()) -> LabeledValues:
        """Values by name to labeled values, a layout operation.

        Parameters
        ----------
        values_by_name:
            As :meth:`values_to_flat` takes them.
        batch_dims:
            The names of the batch's axes, one per leading axis, labeled
            ``0`` to ``n - 1``.

        Returns
        -------
        LabeledValues
            As the module's data model has them.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_values_by_name`; ``ValueError`` too for a
            number of batch dims other than the leading axes', or a batch
            dim named like a component, dim, level or element axis.
        """
        validate_values_by_name(values_by_name, self)
        batch_dims = as_names(batch_dims, message_name="batch_dims")
        batch = self._batch_shape_of(values_by_name)
        check_batch_dims_name_the_leading_axes(batch_dims, batch)
        check_batch_dim_names_are_free(batch_dims, self)
        return {c.name: self._labeled_array(c, values_by_name[c.name], batch_dims, batch) for c in self.components}

    def labeled_to_values(self, labeled: LabeledValues, *, batch_dims: Sequence[str] | None = None) -> ValuesByName:
        """Labeled values to values by name, a layout operation.

        Parameters
        ----------
        labeled:
            :data:`LabeledValues` of this layout, as
            :func:`validate_labeled_values` has them; each dim's and element
            axis's labels in any order, read by label. Batch labels are not
            read: rows keep their order.
        batch_dims:
            The order of the batch dims, needed when there are several.

        Returns
        -------
        ValuesByName
            ``{name: (*batch, *block shape)}``, ``float64``.

        Raises
        ------
        TypeError, KeyError, ValueError
            As :func:`validate_labeled_values`.
        """
        batch = validate_labeled_values(labeled, self, batch_dims=batch_dims)
        out = {}
        for c in self.components:
            array = labeled[c.name]
            labels = {**{d: self.coords[d] for d in c.indexed_by}, **c.element_axes}
            array = array.isel({d: indexer(array.indexes[d], labels[d]) for d in labels}, drop=True)
            out[c.name] = jnp.asarray(np.asarray(array.transpose(*batch, *labels).values, dtype=np.float64))
        return out

    def flat_to_labeled(self, flat: Any, *, batch_dims: Sequence[str] = ()) -> LabeledValues:
        """``values_to_labeled(flat_to_values(flat), batch_dims=...)``."""
        return self.values_to_labeled(self.flat_to_values(flat), batch_dims=batch_dims)

    def labeled_to_flat(self, labeled: LabeledValues, *, batch_dims: Sequence[str] | None = None) -> Array:
        """``values_to_flat(labeled_to_values(labeled, batch_dims=...))``."""
        return self.values_to_flat(self.labeled_to_values(labeled, batch_dims=batch_dims))

    # ── the spaces ────────────────────────────────────────────────────────────

    @cached_property
    def unconstrained(self) -> Layout:
        """The layout of each component's
        :meth:`~sipnet_calibration.probability.spec.ArraySpec.unconstrained`,
        on the same coords: theta's layout."""
        return Layout([c.unconstrained() for c in self.components], coords=self.coords)

    def to_natural(self, theta: Any) -> Array:
        """Theta to the natural Flat form: :math:`x_c = T_c(\\theta_c)` for
        every component, value by value. Traceable.

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
        return self.values_to_flat({c.name: c.bijector.forward(unconstrained[c.name]) for c in self.components})

    def to_unconstrained(self, natural_flat: Any) -> Array:
        """The natural Flat form to theta: :math:`\\theta_c = T_c^{-1}(x_c)`.
        A value outside its support maps to non-finite entries, which is not
        an error; a caller converting values it did not draw checks
        :meth:`contains` first. Traceable.

        Raises
        ------
        ValueError
            If the last axis of *natural_flat* is not ``size`` long.
        """
        natural = self.flat_to_values(natural_flat)
        return self.unconstrained.values_to_flat({c.name: c.bijector.inverse(natural[c.name]) for c in self.components})

    def contains(self, natural_flat: Any) -> Array:
        """Whether every value of each row has a theta: it lies in its
        component's support, and :math:`T_c^{-1}` of it is finite, so it is
        not on a closed end, which no theta reaches. ``(*batch, size) ->
        (*batch,)``. Traceable.

        Raises
        ------
        ValueError
            If the last axis of *natural_flat* is not ``size`` long.
        """
        natural = self.flat_to_values(natural_flat)
        batch = jnp.shape(natural_flat)[:-1]
        inside = []
        for c in self.components:
            in_support = c.support.contains(natural[c.name]).reshape((*batch, -1)).all(axis=-1)
            reached = jnp.isfinite(c.bijector.inverse(natural[c.name])).reshape((*batch, -1)).all(axis=-1)
            inside.append(in_support & reached)
        return jnp.all(jnp.stack(inside, axis=len(batch)), axis=len(batch))

    # ── supporting methods ────────────────────────────────────────────────────

    @cached_property
    def _tables(self) -> _Tables:
        return _Tables.from_layout(self)

    def _batch_shape_of(self, values_by_name: Mapping[str, Any]) -> tuple[int, ...]:
        first = self.components[0].name
        shape = tuple(jnp.shape(values_by_name[first]))
        return shape[: len(shape) - len(self.block_shape(first))]

    def _labeled_array(self, spec: ArraySpec, values: Any, batch_dims: tuple[str, ...], batch: tuple[int, ...]) -> xr.DataArray:
        """One component's values as a labeled DataArray."""
        dims = (*batch_dims, *spec.indexed_by, *spec.element_axes)
        coordinates: dict[str, Any] = {d: np.arange(n, dtype=np.int64) for d, n in zip(batch_dims, batch)}
        coordinates.update({d: self.coords[d].values for d in spec.indexed_by if not is_stacked(self.coords[d])})
        coordinates.update({axis: labels.values for axis, labels in spec.element_axes.items()})
        array = xr.DataArray(np.asarray(values, dtype=np.float64), dims=dims, coords=coordinates, name=spec.name)
        for d in spec.indexed_by:
            if is_stacked(self.coords[d]):
                array = array.assign_coords(xr.Coordinates.from_pandas_multiindex(self.coords[d], d))
        array.attrs = {"support": spec.support.name, **({} if spec.units is None else {"units": spec.units})}
        return array

    def _level_values(self, level: str) -> Any:
        """The index's values at *level*, entry by entry: from a plain dim of
        that name, or a stacked dim's level, NA where neither indexes the
        entry's component."""
        tables = self._tables
        sources = []
        for j, (dim, labels) in enumerate(self.coords.items()):
            if is_stacked(labels) and level in labels.names:
                sources.append((j, labels.get_level_values(level)))
            elif not is_stacked(labels) and dim == level:
                sources.append((j, labels))
        out = _missing_level(self.size, [values for _, values in sources])
        for j, values in sources:
            positions = tables.entry_label_positions[:, j]
            held = positions >= 0
            out[held] = values.take(positions[held]).to_numpy()
        return out

    def _element_level(self) -> np.ndarray:
        """The index's element level, entry by entry."""
        tables = self._tables
        out = np.empty(self.size, dtype=object)
        for i, c in enumerate(self.components):
            axes = list(c.element_axes.values())
            if not axes:
                labels = np.array([None], dtype=object)
            elif len(axes) == 1:
                labels = np.asarray(axes[0], dtype=object)
            else:
                labels = np.empty(math.prod(c.shape), dtype=object)
                labels[:] = list(pd.MultiIndex.from_product(axes))
            entries = slice(int(tables.offsets[i]), int(tables.offsets[i + 1]))
            out[entries] = labels[tables.entry_element[entries]]
        return out

    def _selection(self, selectors: Mapping[str, Any]) -> tuple[set[str], dict[str, np.ndarray]]:
        """The component names a selection keeps, and the positions of the
        labels it keeps along each dim it restricts."""
        names = set(self.component_names)
        kept: dict[str, np.ndarray] = {}
        for key, value in selectors.items():
            check_selector_is_a_level(key, self)
            wanted = as_sequence(value, message_name=f"select {key}=")
            check_names_are_unique(wanted, message_name=f"select {key}=")
            check_selector_keeps_something(key, wanted)
            if key == COMPONENT_LEVEL:
                check_component_names_are_held(wanted, self)
                names = set(wanted)
                continue
            found = np.zeros(len(wanted), dtype=bool)
            targets = self._selector_targets(key)
            check_labels_have_the_kind(key, wanted, [labels for _, labels in targets])
            for dim, labels in targets:
                if labels is self.coords[dim]:
                    positions = indexer(labels, wanted)
                    found |= positions >= 0
                    on_dim = positions[positions >= 0]
                else:
                    # A level's labels, one per label of the dim, repeat:
                    # look each up among the labels wanted instead.
                    matches = indexer(pd.Index(list(wanted)), labels)
                    found[np.unique(matches[matches >= 0])] = True
                    on_dim = np.flatnonzero(matches >= 0)
                kept[dim] = on_dim if dim not in kept else np.intersect1d(kept[dim], on_dim)
            check_labels_are_held(key, wanted, found)
        return names, kept

    def _selector_targets(self, key: str) -> list[tuple[str, pd.Index]]:
        """What a selector *key* restricts: ``(dim, labels it is matched
        against)``, the dim's own labels or one level's labels entry by
        entry."""
        targets = []
        for dim, labels in self.coords.items():
            if dim == key:
                targets.append((dim, labels))
            elif is_stacked(labels) and key in labels.names:
                targets.append((dim, labels.get_level_values(key)))
        return targets


# ── the aliases and their validators ──────────────────────────────────────────

#: Values by name, ``{name: (*batch, *block shape)}``, as the module's data
#: model has them; checked by :func:`validate_values_by_name`. Traceable.
type ValuesByName = dict[str, Array]

#: Labeled values, ``{name: xr.DataArray}``, as the module's data model has
#: them; checked by :func:`validate_labeled_values`.
type LabeledValues = dict[str, xr.DataArray]


def validate_values_by_name(values_by_name: Any, layout: Layout) -> None:
    """Check that *values_by_name* are :data:`ValuesByName` of the layout: a
    mapping holding every component's values, each ending in its block
    shape, all of one batch shape. Other keys are not read.

    Raises
    ------
    TypeError
        If it is not a mapping.
    KeyError
        If a component has no values.
    ValueError
        If a value does not end in its block shape, or the batch shapes
        differ.
    """
    check_values_are_a_mapping(values_by_name)
    check_component_names_have_values(values_by_name, layout)
    batches = []
    for c in layout.components:
        shape = tuple(jnp.shape(values_by_name[c.name]))
        expected = layout.block_shape(c.name)
        check_values_end_in_the_block_shape(c.name, shape, expected)
        batches.append((c.name, shape[: len(shape) - len(expected)]))
    check_values_share_a_batch_shape(batches)


def validate_labeled_values(
    labeled: Any, layout: Layout, *, batch_dims: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Check that *labeled* are :data:`LabeledValues` of the layout, as the
    module's data model has them, and return their batch dims in order.

    Every component has a DataArray (other keys are not read), on its own
    dims and the batch dims, in any order, each dim's and element axis's
    labels the layout's as a set, a stacked dim's with its levels; every
    value is finite. The batch dims are the dims left over, the same for
    every component.

    Returns
    -------
    tuple of str
        The batch dims: *batch_dims*, or the one or none there is.

    Raises
    ------
    TypeError
        If it is not a mapping, or a value is not a DataArray.
    KeyError
        If a component has no value.
    ValueError
        For a DataArray missing one of its dims; a dim without an index
        coordinate, or whose labels or levels are not the layout's;
        different batch dims, several without *batch_dims*, or
        *batch_dims* not those; a missing or non-finite value.
    """
    check_values_are_a_mapping(labeled)
    check_component_names_have_values(labeled, layout)
    leftover = []
    for c in layout.components:
        array = labeled[c.name]
        check_value_is_a_data_array(c.name, array)
        labels = {**{d: layout.coords[d] for d in c.indexed_by}, **c.element_axes}
        check_array_has_its_dims(c.name, array.dims, tuple(labels))
        for dim, expected in labels.items():
            check_array_labels_are_the_layouts(c.name, dim, array, expected)
        check_array_is_finite(c.name, array)
        leftover.append((c.name, tuple(str(d) for d in array.dims if d not in labels)))
    check_arrays_share_their_batch_dims(leftover)
    found = leftover[0][1]
    check_arrays_share_their_batch_labels([labeled[c.name] for c in layout.components], found)
    if batch_dims is not None:
        batch_dims = as_names(batch_dims, message_name="batch_dims")
        check_batch_dims_are_the_arrays(batch_dims, found)
        return batch_dims
    check_batch_dims_are_ordered(found)
    return found


def encode_labeled_values(values: LabeledValues) -> xr.Dataset:
    """*values* as one Dataset netCDF can hold: each stacked dim's levels as
    plain coordinates named ``"<dim>__<level>"``, and the attribute
    :data:`STACKED_DIMS_ATTRIBUTE` recording them.

    Raises
    ------
    TypeError
        If *values* is not a mapping of DataArrays.
    ValueError
        If two values label one dim differently, or a coordinate name
        ``"<dim>__<level>"`` is taken.
    """
    check_values_are_a_mapping(values)
    for name, array in values.items():
        check_value_is_a_data_array(name, array)
    check_values_share_their_labels(list(values.values()))
    stacked: dict[str, list[str]] = {}
    encoded = {}
    for name, array in values.items():
        for dim in map(str, array.dims):
            index = array.indexes.get(dim)
            if is_stacked(index):
                stacked.setdefault(dim, list(index.names))
                array = _unstacked_coordinates(array, dim, index)
        encoded[name] = array
    dataset = xr.Dataset(encoded)
    dataset.attrs[STACKED_DIMS_ATTRIBUTE] = json.dumps(stacked)
    return dataset


def decode_labeled_values(dataset: xr.Dataset) -> LabeledValues:
    """The inverse of :func:`encode_labeled_values`: one DataArray per
    variable, each stacked dim's ``MultiIndex`` rebuilt from its level
    coordinates.

    Raises
    ------
    TypeError
        If *dataset* is not an ``xr.Dataset``.
    ValueError
        If *dataset* lacks the attribute :func:`encode_labeled_values`
        writes, or a level coordinate it records.
    """
    check_dataset_is_a_dataset(dataset)
    check_dataset_records_its_stacked_dims(dataset)
    stacked = json.loads(dataset.attrs[STACKED_DIMS_ATTRIBUTE])
    out = {}
    for name in dataset.data_vars:
        array = dataset[name]
        for dim, levels in stacked.items():
            if dim in array.dims:
                array = _restacked_coordinates(array, dim, levels)
        out[str(name)] = array
    return out


# ── the tables ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _Tables:
    """Where every entry sits, computed once per layout: ``offsets`` (each
    component's first entry, and ``size`` last); per entry,
    ``entry_component`` (a declaration position), ``entry_label_positions``
    (``(size, n_dims)``, the label's position along each dim, ``-1`` where
    not indexed) and ``entry_element`` (the element's position in its value,
    C order)."""

    offsets: np.ndarray
    entry_component: np.ndarray
    entry_label_positions: np.ndarray
    entry_element: np.ndarray

    @classmethod
    def from_layout(cls, layout: Layout) -> _Tables:
        sizes, components, positions, elements = [], [], [], []
        for i, c in enumerate(layout.components):
            index_shape = layout.index_shape(c.name)
            n_values, value_size = math.prod(index_shape), math.prod(c.shape)
            grid = np.indices(index_shape, dtype=np.int64).reshape(len(index_shape), n_values)
            label_positions = np.full((n_values * value_size, len(layout.dims)), -1, dtype=np.int64)
            for axis, dim in enumerate(c.indexed_by):
                label_positions[:, layout.dims.index(dim)] = np.repeat(grid[axis], value_size)
            sizes.append(n_values * value_size)
            components.append(np.full(n_values * value_size, i, dtype=np.int64))
            positions.append(label_positions)
            elements.append(np.tile(np.arange(value_size, dtype=np.int64), n_values))
        return cls(
            offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            entry_component=np.concatenate(components),
            entry_label_positions=np.concatenate(positions),
            entry_element=np.concatenate(elements),
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _layout_from_arguments(components: tuple[ArraySpec, ...], coords: Mapping[str, pd.Index]) -> Layout:
    """A layout from its arguments, for pickling."""
    return Layout(components, coords=coords)


def _label_text(label: Any) -> str:
    """A label as an entry name shows it: a time as its ISO date or time."""
    if isinstance(label, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(label)
        return stamp.isoformat() if stamp != stamp.normalize() else stamp.date().isoformat()
    return str(label)


def _missing_level(size: int, sources: Sequence[pd.Index]) -> Any:
    """An all-NA level of the index, of a dtype that holds *sources*' labels:
    a nullable integer, ``datetime64[ns]`` or ``object``."""
    kinds = {label_kind(values) for values in sources}
    if kinds == {"integer"}:
        dtypes = {values.dtype for values in sources}
        dtype = dtypes.pop() if len(dtypes) == 1 else np.dtype(np.int64)
        return pd.arrays.IntegerArray(np.zeros(size, dtype=dtype), np.ones(size, dtype=bool))
    if kinds == {"datetime"}:
        return np.full(size, np.datetime64("NaT"), dtype="datetime64[ns]")
    return np.full(size, None, dtype=object)


def _unstacked_coordinates(array: xr.DataArray, dim: str, index: pd.MultiIndex) -> xr.DataArray:
    """*array* with *dim*'s MultiIndex replaced by plain coordinates
    ``"<dim>__<level>"``."""
    check_encoded_names_are_free(array, dim, index)
    levels = {f"{dim}__{level}": (dim, index.get_level_values(level).to_numpy()) for level in index.names}
    array = array.drop_vars([dim, *index.names])
    return array.assign_coords(levels)


def _restacked_coordinates(array: xr.DataArray, dim: str, levels: Sequence[str]) -> xr.DataArray:
    """*array* with *dim*'s level coordinates made its MultiIndex again."""
    names = [f"{dim}__{level}" for level in levels]
    check_level_coordinates_are_present(array, dim, names)
    index = pd.MultiIndex.from_arrays([array[n].to_numpy() for n in names], names=list(levels))
    array = array.drop_vars(names)
    return array.assign_coords(xr.Coordinates.from_pandas_multiindex(index, dim))


# ── checks ────────────────────────────────────────────────────────────────────


def check_layout_is_valid(layout: Layout) -> None:
    """The components and coords make one layout whose index and labeled
    values are well defined."""
    check_layout_has_a_component(layout.components)
    check_names_are_unique(layout.component_names, message_name="the component names")
    for c in layout.components:
        check_component_dims_are_in_the_coords(c, layout.coords)
        check_component_dims_share_no_level(c, layout.coords)
    check_every_dim_is_used(layout)
    check_dims_and_levels_are_not_reserved(layout.coords)
    check_names_do_not_collide(layout)
    check_element_axes_agree(layout.components)
    check_element_axes_agree([c.unconstrained() for c in layout.components], theta=True)


def check_components_are_array_specs(components: Any) -> None:
    """The components are a sequence of :class:`ArraySpec`."""
    items = as_sequence(components, message_name="components")
    wrong = [type(c).__name__ for c in items if not isinstance(c, ArraySpec)]
    if wrong:
        raise TypeError(f"components must be ArraySpecs, got {truncated(wrong)}.")


def check_layout_has_a_component(components: tuple[ArraySpec, ...]) -> None:
    """A layout holds at least one component, since an empty one fails far
    from its cause."""
    if not components:
        raise ValueError("a layout needs at least one component; give [ArraySpec(...), ...].")


def check_component_dims_are_in_the_coords(spec: ArraySpec, coords: Mapping[str, pd.Index]) -> None:
    """Every dim a component is indexed by has labels in the coords."""
    missing = [d for d in spec.indexed_by if d not in coords]
    if missing:
        raise KeyError(
            f"component {spec.name!r} is indexed by {missing}, which coords lack; give their labels as "
            f"coords={{dim: labels}} (coords hold {list(coords)})."
        )


def check_component_dims_share_no_level(spec: ArraySpec, coords: Mapping[str, pd.Index]) -> None:
    """A component is not indexed by two dims with a level of one name (a
    ``site`` dim and a stacked dim with a ``site`` level), since the index
    holds one value per level."""
    seen: dict[str, str] = {}
    for dim in spec.indexed_by:
        labels = coords[dim]
        for level in (labels.names if is_stacked(labels) else [dim]):
            if level in seen:
                raise ValueError(
                    f"component {spec.name!r} is indexed by {seen[level]!r} and {dim!r}, which both have "
                    f"a level {level!r}; index it by one of them."
                )
            seen[level] = dim


def check_every_dim_is_used(layout: Layout) -> None:
    """Every dim of the coords indexes some component, since a layout's
    coords are exactly its components' dims."""
    used = {d for c in layout.components for d in c.indexed_by}
    unused = [d for d in layout.coords if d not in used]
    if unused:
        raise ValueError(f"coords give the dims {unused}, which no component is indexed by; drop them.")


def check_dims_and_levels_are_not_reserved(coords: Mapping[str, pd.Index]) -> None:
    """No dim or stacked dim's level takes a name the layer reserves, which
    the index or labeled values would confuse with their own."""
    names = [*coords, *(level for labels in coords.values() if is_stacked(labels) for level in labels.names)]
    reserved = [n for n in names if n in RESERVED_NAMES]
    if reserved:
        raise ValueError(f"coords name dims or levels {truncated(reserved)}, which the probability layer reserves.")


def check_names_do_not_collide(layout: Layout) -> None:
    """No component is named for a dim or level, and no element axis for a
    dim, level or component, since the index and labeled values would take
    them for one another."""
    dims = set(layout.coords)
    levels = set(layout.level_names[1:-1])
    names = set(layout.component_names)
    for c in layout.components:
        if c.name in dims | levels:
            raise ValueError(f"component {c.name!r} is named like a dim or level; name it for what it is.")
        for axis in c.element_axes:
            if axis in dims | levels | names:
                raise ValueError(
                    f"component {c.name!r} has an element axis {axis!r} named like a dim, a level or a "
                    "component; name its axes for what they index."
                )


def check_element_axes_agree(components: Sequence[ArraySpec], *, theta: bool = False) -> None:
    """An element axis name has one set of labels across the layout, in
    theta's layout too (where a simplex drops its last label), since the
    labeled values would otherwise misalign them."""
    axes: dict[str, pd.Index] = {}
    for spec in components:
        for axis, labels in spec.element_axes.items():
            if axis in axes and not axes[axis].equals(labels):
                where = " in theta's layout (a simplex drops its last label)" if theta else ""
                raise ValueError(
                    f"two components name an element axis {axis!r} with different labels{where}; name "
                    "the axes apart."
                )
            axes[axis] = labels


def check_component_names_are_held(names: Sequence[Any], layout: Layout) -> None:
    """Every name is one of the layout's components."""
    for name in names:
        if name not in layout.component_names:
            raise KeyError(
                f"the layout has no component {name!r}; name one of {truncated(list(layout.component_names))}."
            )


def check_selector_is_a_level(key: str, layout: Layout) -> None:
    """A selector names the components, a dim or a stacked dim's level."""
    if key != COMPONENT_LEVEL and key not in layout.coords and key not in layout.level_names[1:-1]:
        raise KeyError(
            f"the layout has no dim or level {key!r}; select with component= or one of "
            f"{truncated([*layout.coords, *layout.level_names[1:-1]])}."
        )


def check_selector_keeps_something(key: str, wanted: Sequence[Any]) -> None:
    """A selector keeps at least one label, since an empty one would keep nothing."""
    if not wanted:
        raise ValueError(f"select {key}=[] keeps nothing; name at least one, or omit the selector.")


def check_labels_have_the_kind(key: str, wanted: Sequence[Any], targets: Sequence[pd.Index]) -> None:
    """A selector's labels are of the kind of the labels they select, level
    by level for a stacked dim's tuples, so a site id is never matched by a
    string or a float."""
    given = label_kinds(pd.Index(list(wanted)))
    kinds = {label_kinds(labels) for labels in targets}
    if given not in kinds:
        shown = " or ".join(sorted(", ".join(k) for k in kinds))
        raise TypeError(f"the labels of {key!r} are {shown}, got {truncated(list(wanted))}; pass labels of that kind.")


def check_labels_are_held(key: str, wanted: Sequence[Any], found: np.ndarray) -> None:
    """Every label asked for is held somewhere the selector reaches."""
    unknown = [w for w, f in zip(wanted, found) if not f]
    if unknown:
        raise KeyError(f"{key!r} has no label(s) {truncated(unknown)}; select labels the layout holds.")


def check_selection_keeps_a_label(dim: str, labels: pd.Index) -> None:
    """A selection leaves a kept component's dim at least one label."""
    if len(labels) == 0:
        raise ValueError(
            f"the selection keeps no label of {dim!r}, which a kept component is indexed by; drop that "
            "component with component=[...]."
        )


def check_flat_ends_in_the_size(shape: tuple[int, ...], size: int) -> None:
    """A Flat form's last axis has one entry per number of the layout."""
    if not shape or shape[-1] != size:
        raise ValueError(f"a Flat form of this layout ends in {size} entries, got shape {shape}; pass (..., {size}).")


def check_values_are_a_mapping(values: Any) -> None:
    """Values by name and labeled values are a mapping, which is how they are read."""
    if not isinstance(values, Mapping):
        raise TypeError(f"values are a mapping of component names to arrays, got {type(values).__name__}.")


def check_component_names_have_values(values: Mapping[str, Any], layout: Layout) -> None:
    """Every component has values."""
    missing = [name for name in layout.component_names if name not in values]
    if missing:
        raise KeyError(f"the values hold no {truncated(missing)}; give every component's values.")


def check_values_end_in_the_block_shape(name: str, shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """A component's values end in its block shape, which would otherwise
    broadcast or misalign silently."""
    if len(shape) < len(expected) or shape[len(shape) - len(expected):] != expected:
        raise ValueError(
            f"the values of {name!r} have shape {shape}, which does not end in its block shape "
            f"{expected}, (*index shape, *shape)."
        )


def check_values_share_a_batch_shape(batches: Sequence[tuple[str, tuple[int, ...]]]) -> None:
    """Every component's values have one batch shape, one value per draw."""
    if len({batch for _, batch in batches}) > 1:
        raise ValueError(f"the values have different batch shapes {dict(batches)}; give every component the same draws.")


def check_batch_dims_name_the_leading_axes(batch_dims: tuple[str, ...], batch: tuple[int, ...]) -> None:
    """One batch dim is named per leading axis of the values."""
    if len(batch_dims) != len(batch):
        raise ValueError(
            f"the values have {len(batch)} leading axes {batch}, but batch_dims names "
            f"{len(batch_dims)}: {batch_dims}; name one per axis."
        )


def check_batch_dim_names_are_free(batch_dims: tuple[str, ...], layout: Layout) -> None:
    """A batch dim is named like no component, dim, level or element axis,
    with which the labeled values would confuse it."""
    check_names_are_unique(batch_dims, message_name="batch_dims")
    taken = {*layout.coords, *layout.level_names[1:-1], *layout.component_names}
    taken |= {axis for c in layout.components for axis in c.element_axes}
    clashing = [d for d in batch_dims if d in taken]
    if clashing:
        raise ValueError(
            f"batch_dims {clashing} are named like a component, dim, level or element axis of the layout; "
            "name a batch dim for what it indexes, such as 'sample'."
        )


def check_value_is_a_data_array(name: str, array: Any) -> None:
    """A labeled value is an ``xr.DataArray``."""
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"the labeled value of {name!r} is a {type(array).__name__}; give an xarray DataArray, as "
            "values_to_labeled makes."
        )


def check_array_has_its_dims(name: str, dims: tuple[Any, ...], own: tuple[str, ...]) -> None:
    """A labeled value is on every dim and element axis of its component,
    which would otherwise be broadcast or dropped silently."""
    missing = [d for d in own if d not in dims]
    if missing:
        raise ValueError(
            f"the labeled value of {name!r} is on {tuple(dims)}, which lacks {missing}; give it every dim and "
            "element axis of its component."
        )


def check_array_labels_are_the_layouts(name: str, dim: str, array: xr.DataArray, labels: pd.Index) -> None:
    """A labeled value's labels along a dim or element axis are labeled, once
    each, with the layout's levels, and are the layout's as a set, since
    xarray would otherwise read by position or twice."""
    if dim not in array.indexes:
        raise ValueError(f"the labeled value of {name!r} has no {dim!r} coordinate; give {dim!r} its labels.")
    held = array.indexes[dim]
    if is_stacked(held) != is_stacked(labels) or (is_stacked(held) and list(held.names) != list(labels.names)):
        raise ValueError(
            f"the labeled value of {name!r} labels {dim!r} with levels "
            f"{list(held.names) if is_stacked(held) else 'none'}, not the layout's "
            f"{list(labels.names) if is_stacked(labels) else 'none'}."
        )
    if len(held) != len(labels) or not held.is_unique or (indexer(held, labels) < 0).any():
        raise ValueError(
            f"the labeled value of {name!r} has {dim!r} labels {truncated(held.tolist())}, not the "
            f"layout's {truncated(labels.tolist())}; select the layout's labels first."
        )


def check_array_is_finite(name: str, array: xr.DataArray) -> None:
    """Every value is finite, as theta's always are."""
    if not np.isfinite(np.asarray(array.values, dtype=np.float64)).all():
        raise ValueError(f"the labeled value of {name!r} holds a missing or non-finite value; drop or fill those draws first.")


def check_arrays_share_their_batch_dims(leftover: Sequence[tuple[str, tuple[str, ...]]]) -> None:
    """Every component has the same batch dims, the dims beyond its own."""
    if len({frozenset(dims) for _, dims in leftover}) > 1:
        raise ValueError(f"the labeled values have different batch dims {dict(leftover)}; give every component the same draws.")


def check_arrays_share_their_batch_labels(arrays: Sequence[xr.DataArray], batch_dims: Sequence[str]) -> None:
    """Every component has as many draws along each batch dim, labeled
    alike in the same order where labeled, since the rows are read in
    order: draws in different orders would be paired wrongly."""
    for dim in batch_dims:
        sizes = {array.sizes[dim] for array in arrays}
        labeled = [array.indexes[dim] for array in arrays if dim in array.indexes]
        if len(sizes) > 1 or any(not labels.equals(labeled[0]) for labels in labeled):
            raise ValueError(
                f"the labeled values differ in their {dim!r} draws (sizes {sorted(sizes)}, or labels in "
                "another order); give every component the same draws, in one order."
            )


def check_batch_dims_are_the_arrays(batch_dims: tuple[str, ...], found: tuple[str, ...]) -> None:
    """``batch_dims`` names exactly the labeled values' batch dims."""
    if sorted(batch_dims) != sorted(found):
        raise ValueError(f"batch_dims {batch_dims} are not the labeled values' batch dims {found}; name those.")


def check_batch_dims_are_ordered(found: tuple[str, ...]) -> None:
    """Several batch dims are put in order by ``batch_dims``, which nothing
    else fixes."""
    if len(found) > 1:
        raise ValueError(f"the labeled values have the batch dims {found}; fix their order with batch_dims=.")


def check_values_share_their_labels(arrays: Sequence[xr.DataArray]) -> None:
    """Values that share a dim share its labels, in one order, so one
    Dataset holds them without reindexing any. Compared dim by dim, since
    ``xr.align`` would take a stacked dim's ``site`` level for the ``site``
    dim."""
    first: dict[str, tuple[str, pd.Index]] = {}
    for array in arrays:
        for dim in map(str, array.dims):
            if dim not in array.indexes:
                continue
            index = array.indexes[dim]
            name, held = first.setdefault(dim, (str(array.name), index))
            if not index.equals(held) or (is_stacked(index) and list(index.names) != list(held.names)):
                raise ValueError(
                    f"two labeled values label one dim differently ({name!r} and {array.name!r} on {dim!r}), "
                    "so one Dataset cannot hold them as they are; select one set of labels, in one order."
                )


def check_encoded_names_are_free(array: xr.DataArray, dim: str, index: pd.MultiIndex) -> None:
    """The coordinates ``"<dim>__<level>"`` an encoding adds are not taken."""
    taken = [f"{dim}__{level}" for level in index.names if f"{dim}__{level}" in array.coords]
    if taken:
        raise ValueError(f"the labeled value {array.name!r} already has coordinates {taken}, which encoding writes.")


def check_dataset_is_a_dataset(dataset: Any) -> None:
    """What is decoded is an ``xr.Dataset``."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"decode_labeled_values reads an xarray Dataset, got {type(dataset).__name__}.")


def check_dataset_records_its_stacked_dims(dataset: xr.Dataset) -> None:
    """A Dataset to decode carries the attribute the encoding writes."""
    if STACKED_DIMS_ATTRIBUTE not in dataset.attrs:
        raise ValueError(
            f"the Dataset has no {STACKED_DIMS_ATTRIBUTE!r} attribute, so it was not written by "
            "encode_labeled_values; encode labeled values with it before writing them."
        )


def check_level_coordinates_are_present(array: xr.DataArray, dim: str, names: Sequence[str]) -> None:
    """Every level coordinate an encoding recorded is present."""
    missing = [n for n in names if n not in array.coords]
    if missing:
        raise ValueError(
            f"the Dataset's {array.name!r} lacks the level coordinates {missing} of {dim!r}; decode a Dataset "
            "as encode_labeled_values wrote it."
        )

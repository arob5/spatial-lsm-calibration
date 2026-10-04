"""Labels: the coords a model's dims are read at, and the fixed labeled
values its functions read at them.

Where this sits
---------------
::

    probability.labels       (coords, constants, label maps)
      -> probability.layout  (the coords of the dims)
      -> the parts that read constants and label maps

The functions of a model are pure array functions: they receive arrays,
never labels. This module is where labeled values become those arrays, and
so where the rules for giving one are stated, once.

Data model
----------
**Coords** are ``{dim: pandas.Index}``, the labels of each dim, unique, at
least one, never missing:

- a plain dim's labels are all integers or all strings;
- a **stacked dim**'s labels are a ``pandas.MultiIndex`` whose levels are
  named (strings, unique, none named for the dim), each level all integers,
  all strings or all ``datetime64[ns]`` (another datetime resolution is
  converted). It is how a ragged set of labels becomes one dim, such as the
  ``(site, time)`` pairs an observation source observes.

A **constant** is an ``xr.DataArray`` of ``float64`` or ``bool``, the same in
every draw: a covariate per site, a prior median per site, a measurement
standard deviation per observation. Each of its dims is one of three kinds:

- a dim of the coords, stacked dims included, whose labels it is read at by
  label, in the coords' order: so it may hold more labels than are in use,
  such as every site of the pool;
- an element axis, read the same way at the axis's labels;
- its **own dim**, which the reader names (``own_dims``), passed whole in
  its own order: a site's ``(x, y)`` on a coordinate dim, for example.

It arrives transposed, the reader's dims first in the reader's order, then
its other dims in its own order, and a ``float64`` constant must be finite
at every label it is read at. A ``float64`` constant on ``site`` read for a
component indexed by ``site`` arrives as ``(S,)``; one on ``(pft,
allocation_part)`` for a simplex indexed by ``pft`` as ``(P, k)``.

A **label map** is a one-dimensional ``xr.DataArray`` on a dim of the coords,
named for its *target*, a dim of the coords or an element axis, whose values
are labels of the target: each site's PFT is a DataArray on ``site`` named
``"pft"`` holding PFT labels; each observation's year, one on an observation
dim named ``"year"`` holding the labels of a ``year`` element axis. It is
read at its dim's labels in use, and arrives as ``int64`` positions into the
target's labels, so ``x_pft[site_pft]`` reads each site's PFT value.

Missing is never allowed: a constant or label map lacking a label in use is
refused.

Functions
---------
:func:`as_coords`, :func:`as_constants`, :func:`as_label_maps`
    Coerce and check what a caller gives, before any labels are known.
:func:`aligned_constants`, :func:`aligned_label_maps`
    Read them at the labels in use, as arrays.
:func:`indexer`, :func:`label_kind`, :func:`is_stacked`
    The positions of labels in an index, by hashing; what kind of labels an
    index holds; whether a dim is stacked.

Notes
-----
Every lookup goes through :func:`indexer`, ``pandas.Index.get_indexer``, so
reading ``n`` labels from ``m`` is linear in ``n + m``; a label is matched by
value and type, so a site id is never matched by the string of its digits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.probability._validation import as_names, as_sequence, truncated

__all__ = [
    "aligned_constants",
    "aligned_label_maps",
    "as_constants",
    "as_coords",
    "as_label_maps",
    "check_labels_are_covered",
    "indexer",
    "is_stacked",
    "label_kind",
]


def as_coords(coords: Any) -> frozendict:
    """``{dim: pandas.Index named for the dim}``, a stacked dim's a
    ``MultiIndex`` with ``datetime64[ns]`` levels where they hold times.

    Raises
    ------
    TypeError
        If *coords* is not a mapping, a dim is not a string, labels are not
        a sequence, or a dim's or level's labels are of none of the allowed
        types.
    ValueError
        If a dim has no labels, repeats one or misses one, or a stacked
        dim's levels are unnamed, repeated or named for the dim.
    """
    check_coords_are_a_mapping(coords)
    out = {}
    for dim, labels in coords.items():
        check_dim_is_a_string(dim)
        index = _as_stacked_index(dim, labels) if isinstance(labels, pd.MultiIndex) else _as_plain_index(dim, labels)
        check_dim_has_a_label(dim, index)
        check_labels_are_unique(dim, index)
        out[dim] = index
    return frozendict(out)


def as_constants(constants: Any, *, message_name: str) -> frozendict:
    """``{name: xr.DataArray}``, each a read-only copy, after the checks
    every constant passes.

    Raises
    ------
    TypeError
        If *constants* is not a mapping, a value is not a DataArray, or its
        dtype is neither ``float64`` nor ``bool``.
    """
    check_mapping_of_data_arrays(constants, message_name=message_name)
    out = {}
    for name, constant in constants.items():
        check_constant_is_float64_or_bool(name, constant, message_name=message_name)
        out[name] = _read_only_copy(constant)
    return frozendict(out)


def as_label_maps(label_maps: Any, *, message_name: str) -> frozendict:
    """``{name: xr.DataArray}``, each a read-only copy, after the checks
    every label map passes.

    Raises
    ------
    TypeError
        If *label_maps* is not a mapping, or a value is not a DataArray.
    ValueError
        If a label map is not one-dimensional, or names no target.
    """
    check_mapping_of_data_arrays(label_maps, message_name=message_name)
    out = {}
    for name, label_map in label_maps.items():
        check_label_map_is_one_dimensional(name, label_map, message_name=message_name)
        out[name] = _read_only_copy(label_map)
    return frozendict(out)


def aligned_constants(
    constants: Mapping[str, xr.DataArray],
    labels_by_dim: Mapping[str, pd.Index],
    *,
    dim_order: Sequence[str],
    own_dims: Sequence[str] = (),
    message_name: str,
) -> dict[str, jax.Array]:
    """Each constant read at the labels of its dims, its dims put in
    *dim_order*: ``{name: array}``, its dtype kept.

    Parameters
    ----------
    constants:
        ``{name: xr.DataArray}``.
    labels_by_dim:
        The coords and the element axes in use, ``{dim: labels}``.
    dim_order:
        The reader's dims, in order: a constant's dims among them come
        first, in this order, then its others in its own order.
    own_dims:
        Dims of the constants that are neither in *labels_by_dim* nor read
        at labels, passed whole.
    message_name:
        What the constants are called in an error message.

    Raises
    ------
    ValueError
        If a constant is on a dim that is neither in *labels_by_dim* nor in
        *own_dims*, or one without labels; a stacked dim's levels differ
        from the coords'; or a ``float64`` constant is not finite at a label
        in use.
    KeyError
        If a constant lacks a label it is read at.
    """
    own_dims = as_names(own_dims, message_name="own_dims")
    out = {}
    for name, constant in constants.items():
        positions = {}
        for dim in map(str, constant.dims):
            if dim in own_dims:
                continue
            check_dim_is_in_the_coords(name, dim, labels_by_dim, message_name=message_name)
            check_dim_is_labeled(name, constant, dim, message_name=message_name)
            held = constant.indexes[dim]
            check_stacked_levels_agree(name, dim, held, labels_by_dim[dim], message_name=message_name)
            positions[dim] = indexer(held, labels_by_dim[dim])
            check_labels_are_covered(name, positions[dim], labels_by_dim[dim], message_name=message_name)
        selected = constant.isel(positions, drop=True) if positions else constant.copy()
        ordered = [d for d in dim_order if d in selected.dims]
        selected = selected.transpose(*ordered, *(d for d in selected.dims if d not in ordered))
        values = np.asarray(selected.values)
        check_constant_is_finite(name, values, message_name=message_name)
        out[name] = jnp.asarray(values)
    return out


def aligned_label_maps(
    label_maps: Mapping[str, xr.DataArray],
    coords: Mapping[str, pd.Index],
    *,
    element_axes: Mapping[str, pd.Index] | None = None,
    message_name: str,
) -> dict[str, np.ndarray]:
    """Each label map as ``int64`` positions: for each label of its dim, in
    the coords' order, the position of the label it maps to among its
    target's labels.

    Parameters
    ----------
    label_maps:
        ``{name: xr.DataArray}``.
    coords:
        The coords in use; each label map is on one of their dims.
    element_axes:
        The element axes in use, which a label map may target as it may a
        dim of the coords.
    message_name:
        What the label maps are called in an error message.

    Raises
    ------
    KeyError
        If a label map's dim is not a dim of the coords, its target is
        neither a dim of the coords nor an element axis, it lacks a label of
        its dim, or a value is not a label of its target.
    ValueError
        If a label map's dim is not labeled, or is its target.
    """
    targets = {**(element_axes or {}), **coords}
    out = {}
    for name, label_map in label_maps.items():
        (dim,) = (str(d) for d in label_map.dims)
        target = str(label_map.name)
        check_label_map_names_its_dim_and_target(name, dim, target, coords, targets, message_name=message_name)
        check_dim_is_labeled(name, label_map, dim, message_name=message_name)
        held = label_map.indexes[dim]
        check_stacked_levels_agree(name, dim, held, coords[dim], message_name=message_name)
        positions = indexer(held, coords[dim])
        check_labels_are_covered(name, positions, coords[dim], message_name=message_name)
        values = np.asarray(label_map.values)[positions]
        mapped = indexer(targets[target], values)
        check_label_map_values_are_labels(name, values, mapped, targets[target], message_name=message_name)
        out[name] = mapped.astype(np.int64)
    return out


def indexer(index: pd.Index, labels: Any) -> np.ndarray:
    """The position in *index* of each of *labels*, ``-1`` where it has none:
    ``int64``, by hashing, a label matched by value and kind (an integer
    label never matches a string, nor a boolean an integer)."""
    wanted = labels if isinstance(labels, pd.Index) else pd.Index(list(labels) if not isinstance(labels, np.ndarray) else labels)
    if len(wanted) and len(index) and label_kind(index) != label_kind(wanted):
        return np.full(len(wanted), -1, dtype=np.int64)
    return np.asarray(index.get_indexer(wanted), dtype=np.int64)


def label_kind(index: pd.Index) -> str:
    """What kind of labels *index* holds: ``"integer"``, ``"string"``,
    ``"datetime"``, ``"tuple"`` (a ``MultiIndex``), ``"empty"`` or
    ``"other"``, a mixture included."""
    if isinstance(index, pd.MultiIndex):
        return "tuple"
    if len(index) == 0:
        return "empty"
    if index.dtype.kind in "iu":
        return "integer"
    if index.dtype.kind == "M":
        return "datetime"
    inferred = pd.api.types.infer_dtype(index, skipna=False)
    return {"string": "string", "integer": "integer", "datetime64": "datetime", "datetime": "datetime"}.get(
        inferred, "other"
    )


def is_stacked(labels: pd.Index) -> bool:
    """Whether a dim's labels are a stacked dim's, a ``MultiIndex``."""
    return isinstance(labels, pd.MultiIndex)


# ── helpers ───────────────────────────────────────────────────────────────────


def _as_plain_index(dim: str, labels: Any) -> pd.Index:
    """A plain dim's labels as an index named for it, integers or strings,
    strings held as ``object``."""
    listed = as_sequence(labels, message_name=f"coords[{dim!r}]")
    if isinstance(labels, (pd.Index, np.ndarray)):
        index = pd.Index(labels, name=dim)
    else:
        index = pd.Index(list(listed), name=dim)
    check_labels_are_integers_or_strings(dim, index)
    return index.astype(object) if label_kind(index) == "string" else index


def _as_stacked_index(dim: str, labels: pd.MultiIndex) -> pd.MultiIndex:
    """A stacked dim's labels, its levels checked and times at ``[ns]``."""
    check_levels_are_named(dim, labels)
    check_labels_are_not_missing(dim, labels)
    levels = []
    for i, level in enumerate(labels.levels):
        check_level_labels_have_one_type(dim, labels.names[i], level)
        if level.dtype.kind == "M":
            level = level.astype("datetime64[ns]")
        elif label_kind(level) == "string" and level.dtype != object:
            level = level.astype(object)
        levels.append(level)
    return labels.set_levels(levels)


def _read_only_copy(array: xr.DataArray) -> xr.DataArray:
    """A deep copy whose data cannot be written."""
    copy = array.copy(deep=True)
    copy.values.setflags(write=False)
    return copy


# ── checks ────────────────────────────────────────────────────────────────────


def check_coords_are_a_mapping(coords: Any) -> None:
    """Coords are ``{dim: labels}``."""
    if not isinstance(coords, Mapping):
        raise TypeError(f"coords must be a mapping {{dim: labels}}, got {type(coords).__name__}.")


def check_dim_is_a_string(dim: Any) -> None:
    """A dim is named by a string."""
    if not isinstance(dim, str):
        raise TypeError(f"coords names a dim {dim!r}; dims are named by strings.")


def check_labels_are_integers_or_strings(dim: str, labels: pd.Index) -> None:
    """A plain dim's labels are all integers or all strings, since a
    selector's labels are compared with them by type."""
    if len(labels) and label_kind(labels) not in ("integer", "string"):
        raise TypeError(
            f"coords[{dim!r}] holds labels that are neither all integers nor all strings, such as "
            f"{truncated(labels.tolist())}; give site ids as integers and classes as strings, or a "
            "MultiIndex for a stacked dim."
        )


def check_levels_are_named(dim: str, labels: pd.MultiIndex) -> None:
    """A stacked dim's levels are named, once each, and none for the dim,
    since a level may be named wherever a dim may."""
    names = list(labels.names)
    if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names) or dim in names:
        raise ValueError(
            f"coords[{dim!r}] is a MultiIndex with levels named {names}; name every level, once, and "
            f"none {dim!r}, such as names=['site', 'time']."
        )


def check_labels_are_not_missing(dim: str, labels: pd.MultiIndex) -> None:
    """No label of a stacked dim is missing at any level."""
    if any((codes < 0).any() for codes in labels.codes):
        raise ValueError(f"coords[{dim!r}] holds a missing label at some level; drop or fill it first.")


def check_level_labels_have_one_type(dim: str, level_name: str, level: pd.Index) -> None:
    """A stacked dim's level holds all integers, all strings or all times."""
    if len(level) and label_kind(level) not in ("integer", "string", "datetime"):
        raise TypeError(
            f"coords[{dim!r}] level {level_name!r} holds labels such as {truncated(level.tolist())}, "
            "which are neither all integers, all strings nor all datetime64; convert them first."
        )


def check_dim_has_a_label(dim: str, labels: pd.Index) -> None:
    """A dim has at least one label, since a value on it would be empty."""
    if len(labels) == 0:
        raise ValueError(f"coords[{dim!r}] holds no label; give at least one, or drop the dim.")


def check_labels_are_unique(dim: str, labels: pd.Index) -> None:
    """A dim names each label once, so that a label finds one position."""
    if not labels.is_unique:
        repeated = labels[labels.duplicated()].unique().tolist()
        raise ValueError(f"coords[{dim!r}] names {truncated(repeated)} more than once; name each once.")


def check_mapping_of_data_arrays(mapping: Any, *, message_name: str) -> None:
    """Constants and label maps are ``{name: xr.DataArray}``, so they carry
    the labels they are read at."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{message_name} must be a mapping {{name: DataArray}}, got {type(mapping).__name__}.")
    for name, value in mapping.items():
        if not isinstance(name, str) or not isinstance(value, xr.DataArray):
            raise TypeError(
                f"{message_name}[{name!r}] is a {type(value).__name__}; give a DataArray keyed by "
                "label, such as pd.Series({...}).rename_axis('pft').to_xarray()."
            )


def check_constant_is_float64_or_bool(name: str, constant: xr.DataArray, *, message_name: str) -> None:
    """A constant is ``float64`` or ``bool``: a code or a string is no
    quantity until it is converted deliberately, and a boolean stays one."""
    if constant.dtype not in (np.float64, np.bool_):
        raise TypeError(
            f"{message_name}[{name!r}] is {constant.dtype}, neither float64 nor bool; convert it "
            "deliberately (a class code is not a quantity)."
        )


def check_label_map_is_one_dimensional(name: str, label_map: xr.DataArray, *, message_name: str) -> None:
    """A label map is on one dim and named for the dim or element axis its
    values are labels of."""
    if label_map.ndim != 1 or not isinstance(label_map.name, str):
        raise ValueError(
            f"{message_name}[{name!r}] is on {label_map.dims} and named {label_map.name!r}; a label "
            "map is on one dim and named for the dim or element axis its values label, such as each "
            "site's PFT on 'site', named 'pft'."
        )


def check_dim_is_in_the_coords(name: str, dim: str, labels_by_dim: Mapping[str, pd.Index], *, message_name: str) -> None:
    """A constant is on dims of the coords, element axes or its reader's own
    dims; any other dim would be read by position without saying so."""
    if dim not in labels_by_dim:
        raise ValueError(
            f"{message_name}[{name!r}] is on {dim!r}, which is neither a dim of the coords nor an "
            f"element axis ({truncated(list(labels_by_dim))}); give it on those dims, or declare "
            f"{dim!r} in own_dims= to pass it whole."
        )


def check_dim_is_labeled(name: str, array: xr.DataArray, dim: str, *, message_name: str) -> None:
    """A constant's or label map's dim is labeled, since xarray would
    otherwise read it by position."""
    if dim not in array.indexes:
        raise ValueError(f"{message_name}[{name!r}] has no {dim!r} coordinate; give {dim!r} its labels.")


def check_stacked_levels_agree(name: str, dim: str, held: pd.Index, labels: pd.Index, *, message_name: str) -> None:
    """A stacked dim is read at labels of the same levels, in the same
    order, as the coords'."""
    if is_stacked(held) != is_stacked(labels) or (is_stacked(held) and list(held.names) != list(labels.names)):
        expected = list(labels.names) if is_stacked(labels) else "a plain index"
        found = list(held.names) if is_stacked(held) else "a plain index"
        raise ValueError(
            f"{message_name}[{name!r}] labels {dim!r} with {found}, but the coords use {expected}; "
            "give it the coords' levels in their order."
        )


def check_labels_are_covered(name: str, positions: np.ndarray, labels: pd.Index, *, message_name: str) -> None:
    """A constant or label map has a value at every label it is read at."""
    missing = np.flatnonzero(positions < 0)
    if missing.size:
        raise KeyError(
            f"{message_name}[{name!r}] has no value at {labels.name!r} label(s) "
            f"{truncated(labels[missing[:10]].tolist())}"
            + (f" ({missing.size} in all)." if missing.size > 10 else ".")
        )


def check_constant_is_finite(name: str, values: np.ndarray, *, message_name: str) -> None:
    """A ``float64`` constant is finite at every label it is read at, since a
    missing value would spread into every draw without an error."""
    if values.dtype == np.float64 and not np.isfinite(values).all():
        raise ValueError(
            f"{message_name}[{name!r}] is not finite at some label in use; fill or drop it before "
            "binding."
        )


def check_label_map_names_its_dim_and_target(
    name: str,
    dim: str,
    target: str,
    coords: Mapping[str, pd.Index],
    targets: Mapping[str, pd.Index],
    *,
    message_name: str,
) -> None:
    """A label map is on a dim of the coords and maps it to another dim or
    an element axis."""
    if dim not in coords:
        raise KeyError(
            f"{message_name}[{name!r}] is on {dim!r}, which is not a dim of the coords "
            f"{truncated(list(coords))}."
        )
    if target not in targets:
        raise KeyError(
            f"{message_name}[{name!r}] maps {dim!r} to {target!r}, which is neither a dim of the "
            f"coords nor an element axis ({truncated(list(targets))})."
        )
    if dim == target:
        raise ValueError(f"{message_name}[{name!r}] maps {dim!r} to itself; a label map maps one dim to another.")


def check_label_map_values_are_labels(
    name: str, values: np.ndarray, mapped: np.ndarray, labels: pd.Index, *, message_name: str
) -> None:
    """Every value of a label map is a label of its target."""
    unknown = list(dict.fromkeys(values[mapped < 0].tolist()))
    if unknown:
        raise KeyError(
            f"{message_name}[{name!r}] holds {truncated(unknown)}, which are not labels of "
            f"{labels.name!r} ({truncated(labels.tolist())})."
        )

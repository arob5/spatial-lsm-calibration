"""Labels: the coords a vector's dims are read at, and the fixed labeled
values a derived parameter or a prior term reads at them.

Where this sits
---------------
::

    parameters.labels                (coords, constants, memberships)
      -> parameters.vector           (the coords of the dims)
      -> parameters.derived          (constants and memberships of a derived parameter)
      -> parameters.prior            (constants of a prior term)

The functions a derived parameter or a prior term computes with are pure
array functions: they receive arrays, never labels. This module is where
labeled values become those arrays, and so where the rules for giving one
are stated, once.

Data model
----------
**Coords** are ``{dim: pandas.Index}``: a dim's labels, all integers or all
strings, unique, at least one.

A **constant** is an ``xr.DataArray`` of ``float64`` or ``bool``: a value
that is the same in every draw, such as a covariate per site, a prior median
per site or a simplex center per PFT. Each of its dims is labeled (it has an
index coordinate), and is either a dim of the coords or an element axis
(which axes are allowed is the reader's to say). It is read:

- at the labels in use, in their order, by label: so it may hold more
  labels than are in use, such as every site of the pool, and it is read
  again when a selection changes them;
- with its dims transposed into the order of the values it is read for:
  their ``indexed_by`` dims first, in that order, then their element axes,
  then its other dims in its own order.

The function then receives an array of shape ``[len(labels[d]) for d in
dims]``, with the constant's dtype. A ``float64`` constant on ``site``, for
a parameter indexed by ``site``, arrives as ``(S,)``; one on ``(pft,
allocation_part)`` for a simplex indexed by ``pft`` as ``(P, k)``.

A **membership** is a one-dimensional ``xr.DataArray`` on one dim of the
coords, named for another (its *target*), whose values are labels of the
target: for each label of its dim, the target label it belongs to, such as
each site's PFT, a DataArray on ``site`` named ``"pft"`` holding PFT
labels. It is read at its dim's labels in use, and the function receives
``int64`` positions into the target's labels in use, so ``x[pft_of_site]``
reads a PFT-level value at each site. Only a derived parameter reads
memberships: indexing a random value by another dim is its job.

Missing is never allowed: a constant or membership lacking a label in use is
refused.

Functions
---------
:func:`as_coords`, :func:`as_constants`, :func:`as_memberships`
    Coerce and check what a caller gives, before any labels are known.
:func:`aligned_constants`, :func:`aligned_memberships`
    Read them at the labels in use, as arrays.
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

from sipnet_calibration.parameters._validation import (
    as_sequence,
    check_names_are_unique,
    truncated,
)

__all__ = [
    "aligned_constants",
    "aligned_memberships",
    "as_constants",
    "as_coords",
    "as_memberships",
]


def as_coords(coords: Any) -> frozendict:
    """``{dim: pd.Index of labels named for the dim}``, labels kept with
    their dtype.

    Raises
    ------
    TypeError
        If *coords* is not a mapping, a dim is not a string, labels are not
        a sequence, or they are neither all integers nor all strings.
    ValueError
        If a dim has no labels, or repeats one.
    """
    check_coords_are_a_mapping(coords)
    out = {}
    for dim, labels in coords.items():
        check_dim_is_a_string(dim)
        listed = as_sequence(labels, message_name=f"coords[{dim!r}]")
        index = pd.Index(labels if isinstance(labels, (pd.Index, np.ndarray)) else list(listed), name=dim)
        check_labels_are_integers_or_strings(dim, index)
        check_names_are_unique(list(index), message_name=f"coords[{dim!r}]")
        check_dim_has_a_label(dim, index)
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


def as_memberships(memberships: Any, *, message_name: str) -> frozendict:
    """``{name: xr.DataArray}``, each a read-only copy, after the checks
    every membership passes.

    Raises
    ------
    TypeError
        If *memberships* is not a mapping, or a value is not a DataArray.
    ValueError
        If a membership is not one-dimensional, or names no target dim.
    """
    check_mapping_of_data_arrays(memberships, message_name=message_name)
    out = {}
    for name, membership in memberships.items():
        check_membership_is_one_dimensional(name, membership, message_name=message_name)
        out[name] = _read_only_copy(membership)
    return frozendict(out)


def aligned_constants(
    constants: Mapping[str, xr.DataArray],
    labels_by_dim: Mapping[str, pd.Index],
    *,
    dim_order: Sequence[str],
    message_name: str,
) -> dict[str, jax.Array]:
    """Each constant read at the labels of its dims, its dims put in
    *dim_order*: ``{name: array}``, its dtype kept.

    Parameters
    ----------
    constants:
        ``{name: xr.DataArray}``.
    labels_by_dim:
        The coords and the element axes of the values in use, ``{dim:
        labels}``.
    dim_order:
        The dims of the values the constants are read for, in order: a
        constant's dims among them come first, in this order, then its
        others in its own order. A function receiving a constant then reads
        its axes in the values' order, whatever order the caller built it in.
    message_name:
        What the constants are called in an error message.

    Raises
    ------
    ValueError
        If a constant is on a dim *labels_by_dim* lacks, or a dim without
        labels.
    KeyError
        If a constant lacks a label it is read at.
    """
    out = {}
    for name, constant in constants.items():
        for dim in constant.dims:
            check_dim_is_in_the_coords(name, str(dim), labels_by_dim, message_name=message_name)
            check_dim_is_labeled(name, constant, str(dim), message_name=message_name)
            check_labels_are_covered(name, constant.indexes[dim], labels_by_dim[dim], message_name=message_name)
        selected = constant.sel({d: list(labels_by_dim[d]) for d in constant.dims})
        ordered = [d for d in dim_order if d in selected.dims]
        selected = selected.transpose(*ordered, *(d for d in selected.dims if d not in ordered))
        out[name] = jnp.asarray(np.asarray(selected.values))
    return out


def aligned_memberships(memberships: Mapping[str, xr.DataArray], coords: Mapping[str, pd.Index], *, message_name: str) -> dict[str, np.ndarray]:
    """Each membership as ``int64`` positions: for each label of its dim,
    in the coords' order, the position of the label it belongs to in the
    coords of the dim it is named for.

    Raises
    ------
    KeyError
        If a membership's dim or target is not a dim of the coords, it
        lacks a label of its dim, or a value is not a label of its target.
    ValueError
        If a membership's dim is not labeled, or is its target.
    """
    out = {}
    for name, membership in memberships.items():
        (dim,) = (str(d) for d in membership.dims)
        target = membership.name
        check_membership_names_two_dims_of_the_coords(name, dim, target, coords, message_name=message_name)
        check_dim_is_labeled(name, membership, dim, message_name=message_name)
        check_labels_are_covered(name, membership.indexes[dim], coords[dim], message_name=message_name)
        values = membership.sel({dim: list(coords[dim])}).values.tolist()
        check_membership_values_are_labels(name, values, coords[target], message_name=message_name)
        out[name] = coords[target].get_indexer(values).astype(np.int64)
    return out


# ── helpers ───────────────────────────────────────────────────────────────────


def _read_only_copy(array: xr.DataArray) -> xr.DataArray:
    """A deep copy whose data cannot be written."""
    copy = array.copy(deep=True)
    copy.values.setflags(write=False)
    return copy


def _is_integer_label(label: Any) -> bool:
    return isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))


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
    """A dim's labels are all integers or all strings, since a selector's
    labels are compared with them by type."""
    values = list(labels)
    if not (all(_is_integer_label(v) for v in values) or all(isinstance(v, str) for v in values)):
        raise TypeError(
            f"coords[{dim!r}] holds labels that are neither all integers nor all strings, such as "
            f"{truncated(values)}; give site ids as integers and classes as strings."
        )


def check_dim_has_a_label(dim: str, labels: pd.Index) -> None:
    """A dim has at least one label, since a value on it would be empty."""
    if len(labels) == 0:
        raise ValueError(f"coords[{dim!r}] holds no label; give at least one, or drop the dim.")


def check_mapping_of_data_arrays(mapping: Any, *, message_name: str) -> None:
    """Constants and memberships are ``{name: xr.DataArray}``, so they carry
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


def check_membership_is_one_dimensional(name: str, membership: xr.DataArray, *, message_name: str) -> None:
    """A membership is on one dim and named for the dim its values are
    labels of."""
    if membership.ndim != 1 or not isinstance(membership.name, str):
        raise ValueError(
            f"{message_name}[{name!r}] is on {membership.dims} and named {membership.name!r}; a "
            "membership is on one dim and named for the dim its values label, such as each "
            "site's PFT on 'site', named 'pft'."
        )


def check_dim_is_in_the_coords(name: str, dim: str, labels_by_dim: Mapping[str, pd.Index], *, message_name: str) -> None:
    """A constant is on dims of the coords or element axes, whose labels it
    is read at; any other dim would be read by position."""
    if dim not in labels_by_dim:
        raise ValueError(
            f"{message_name}[{name!r}] is on {dim!r}, which is neither a dim of the coords nor an "
            f"element axis ({list(labels_by_dim)}); give it on those dims."
        )


def check_dim_is_labeled(name: str, array: xr.DataArray, dim: str, *, message_name: str) -> None:
    """A constant's or membership's dim is labeled, since xarray would
    otherwise read it by position."""
    if dim not in array.indexes:
        raise ValueError(f"{message_name}[{name!r}] has no {dim!r} coordinate; give {dim!r} its labels.")


def check_labels_are_covered(name: str, held: pd.Index, labels: pd.Index, *, message_name: str) -> None:
    """A constant or membership has a value at every label it is read at."""
    missing = [label for label in labels if label not in set(held.tolist())]
    if missing:
        raise KeyError(f"{message_name}[{name!r}] has no value at {labels.name!r} label(s) {truncated(missing)}.")


def check_membership_names_two_dims_of_the_coords(
    name: str, dim: str, target: str, coords: Mapping[str, pd.Index], *, message_name: str
) -> None:
    """A membership maps the labels of one dim of the coords to those of
    another."""
    for which in (dim, target):
        if which not in coords:
            raise KeyError(
                f"{message_name}[{name!r}] maps {dim!r} to {target!r}, but {which!r} is not a dim of "
                f"the coords {list(coords)}."
            )
    if dim == target:
        raise ValueError(f"{message_name}[{name!r}] maps {dim!r} to itself; a membership maps one dim to another.")


def check_membership_values_are_labels(name: str, values: list[Any], labels: pd.Index, *, message_name: str) -> None:
    """Every value of a membership is a label of its target dim."""
    unknown = [v for v in dict.fromkeys(values) if v not in set(labels.tolist())]
    if unknown:
        raise KeyError(
            f"{message_name}[{name!r}] holds {truncated(unknown)}, which are not labels of "
            f"{labels.name!r} ({truncated(list(labels))})."
        )

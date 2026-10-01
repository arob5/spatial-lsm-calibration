"""What a parameter and a derived parameter both describe, and its checks:
a name, units, one value's shape and element labels, the dims it is indexed
by, and a long name. Private to the parameter layer.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from frozendict import frozendict

from sipnet_calibration.parameters._validation import (
    as_names,
    as_sequence,
    check_names_are_unique,
    truncated,
)

__all__ = [
    "as_shape",
    "check_description_is_valid",
    "resolved_element_labels",
]


def as_shape(shape: Any, *, message_name: str) -> tuple[int, ...]:
    """One value's shape as a tuple of positive integers.

    Raises
    ------
    TypeError
        If *shape* is not a sequence of integers.
    ValueError
        If an axis is not positive.
    """
    axes = as_sequence(shape, message_name=message_name)
    check_shape_is_positive_integers(axes, message_name=message_name)
    return tuple(int(n) for n in axes)


def resolved_element_labels(
    name: str, shape: tuple[int, ...], element_labels: Mapping[str, Sequence[str]] | None
) -> frozendict:
    """``{axis name: pd.Index of labels}``, one entry per axis of *shape*:
    as given, or ``<name>_axis_<i>`` labeled ``"0"`` to ``"n - 1"``.

    Raises
    ------
    TypeError
        If *element_labels* is not a mapping, an axis name is not a string,
        or a label is not a string.
    ValueError
        If the axes are not *shape*'s, or an axis repeats a label.
    """
    if element_labels is None:
        return frozendict(
            {f"{name}_axis_{i}": pd.Index([str(j) for j in range(n)], name=f"{name}_axis_{i}") for i, n in enumerate(shape)}
        )
    check_element_labels_are_a_mapping(name, element_labels)
    out = {}
    for axis_name, labels in element_labels.items():
        check_axis_name_is_a_string(name, axis_name)
        labels = as_names(labels, message_name=f"{name!r} element labels of {axis_name!r}")
        check_names_are_unique(labels, message_name=f"{name!r} element labels of {axis_name!r}")
        out[axis_name] = pd.Index(labels, name=axis_name)
    check_element_labels_fit_the_shape(name, shape, out)
    return frozendict(out)


# ── checks ────────────────────────────────────────────────────────────────────


def check_description_is_valid(
    *,
    name: Any,
    units: Any,
    element_labels: Mapping[str, pd.Index],
    indexed_by: tuple[str, ...],
    long_name: Any,
    what: str,
) -> None:
    """A name, units, element axes, indexed_by and long name describe one
    value that the labeled forms can hold."""
    check_name_is_an_identifier(name, what)
    check_units_are_a_string_or_none(name, units)
    check_names_are_unique(indexed_by, message_name=f"{name!r} indexed_by")
    check_axis_names_are_free(name, element_labels, indexed_by)
    check_long_name_is_a_string_or_none(name, long_name)


def check_name_is_an_identifier(name: Any, what: str) -> None:
    """A name is a Python identifier and not a keyword, since prior and
    derived functions receive values as keyword arguments named for it."""
    if not isinstance(name, str):
        raise TypeError(f"{what}'s name is a string, got {type(name).__name__} {name!r}.")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"{what} is named {name!r}, which is not a Python identifier, or is a keyword; values "
            "are passed as keyword arguments named for it, so name it like 'base_soil_respiration'."
        )


def check_units_are_a_string_or_none(name: str, units: Any) -> None:
    """Units are a string, or ``None`` for a value without physical units."""
    if units is not None and not isinstance(units, str):
        raise TypeError(f"{name!r} has units {units!r}; give a string, or None for none.")


def check_long_name_is_a_string_or_none(name: str, long_name: Any) -> None:
    """A long name is a string or ``None``."""
    if long_name is not None and not isinstance(long_name, str):
        raise TypeError(f"{name!r} has long_name {long_name!r}; give a string, or None.")


def check_shape_is_positive_integers(axes: Sequence[Any], *, message_name: str) -> None:
    """Every axis of a shape is a positive integer."""
    for n in axes:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise TypeError(f"{message_name} holds {n!r}; a shape is a sequence of integers, such as (4,).")
        if n < 1:
            raise ValueError(f"{message_name} holds the axis length {n}; every axis holds at least one number.")


def check_element_labels_are_a_mapping(name: str, element_labels: Any) -> None:
    """Element labels are ``{axis name: labels}``."""
    if not isinstance(element_labels, Mapping):
        raise TypeError(
            f"{name!r} element_labels is a {type(element_labels).__name__}; give a mapping "
            "{axis name: labels}, one entry per axis of the shape."
        )


def check_axis_name_is_a_string(name: str, axis_name: Any) -> None:
    """An element axis is named by a string, which names a dim of the labeled forms."""
    if not isinstance(axis_name, str):
        raise TypeError(f"{name!r} names an element axis {axis_name!r}; axis names are strings.")


def check_element_labels_fit_the_shape(name: str, shape: tuple[int, ...], labels: Mapping[str, pd.Index]) -> None:
    """There is one axis of labels per axis of the shape, each as long as
    its axis, since they become the labeled forms' coordinates."""
    lengths = tuple(len(v) for v in labels.values())
    if lengths != shape:
        raise ValueError(
            f"{name!r} has shape {shape}, but its element labels give axes of lengths {lengths}; "
            "give one axis of labels per axis of the shape, in order."
        )


def check_axis_names_are_free(name: str, element_labels: Mapping[str, pd.Index], indexed_by: tuple[str, ...]) -> None:
    """An element axis is named neither for the value nor for a dim it is
    indexed by, which the labeled forms would take for one another."""
    taken = [axis for axis in element_labels if axis == name or axis in indexed_by]
    if taken:
        raise ValueError(
            f"{name!r} names element axes {truncated(taken)} like itself or a dim it is indexed "
            "by; name its axes for what they index."
        )

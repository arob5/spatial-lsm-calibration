"""Argument coercion every module shares: check a value and convert it.

Each function here takes an argument as a caller passed it, refuses it if it
is not the thing asked for, and returns it in one canonical form, so that two
functions taking the same kind of argument accept and refuse the same values.

Contents
--------
:func:`as_site_ids`, :func:`as_site_id`
    Site ids as plain Python integers.
:func:`as_integer`, :func:`as_positive_integer`, :func:`as_bounded_integer`,
:func:`as_positive_integers`
    A count, a size, an index or a sequence of them as plain Python integers.
:func:`as_batched_flat`, :func:`is_one_vector`
    One vector or a batch of them, as a two-dimensional ``float64`` array;
    and whether one vector was given.
:func:`as_bbox`
    A ``(west, south, east, north)`` box as four floats.
:func:`as_sequence`, :func:`as_names`
    An ordered sequence of any items, and of names, as a tuple.
:func:`as_frozen_mapping`
    A mapping as a :class:`~sipnet_calibration.conventions.FrozenMapping`.
:func:`truncated`, :func:`range_summary`
    A list shortened for an error message, and the range of some values for
    a report.
The checks
    The ``check_*`` functions the coercers are written with, which other
    modules call where they check the same thing of an array:
    :func:`check_site_ids_are_unique`, :func:`check_site_ids_are_in_range`,
    :func:`check_integers_are_in_range`.

Every coercer takes the name the value goes by in a message, *message_name*,
as its last argument, a keyword, and raises by one rule: :class:`TypeError`
when the value is of the wrong type and :class:`ValueError` when it is of the
right type but a wrong value.

- **A sequence argument is a sequence, always.** Site ids, names and other
  sequences may be a list, a tuple, a generator (read once), an ordered view
  such as ``dict.keys()``, or an array-like (NumPy, JAX, xarray, pandas). One
  bare value or one string is refused, as is a ``set``, which has no order to
  keep, and a mapping, whose keys are passed as ``m.keys()`` or ``list(m)``
  when they are meant. Order is kept.
- **An integer is an integer.** A float is refused, even a whole one, since an
  integer that arrives as a float is usually the result of a mistake; cast it
  with ``int()`` where it is known to be whole. A boolean is refused too, and
  so is a missing value (``NaN``, ``pd.NA``).
- **A scalar may come wrapped.** A zero-dimensional array-like -- a NumPy
  scalar or 0-d array, a 0-d JAX array, a 0-d ``DataArray`` -- is read as the
  value it holds.

Notes
-----
This module depends only on :mod:`sipnet_calibration.conventions`, so any
module of the package can use it.

A JAX array given to :func:`as_batched_flat`, or a list of them, stays a JAX
array, so a function traced by ``jax.jit`` or ``jax.grad`` can coerce its
argument. JAX is looked up rather than imported here; importing the package
imports it anyway (see :mod:`sipnet_calibration`).

Usage
-----
::

    import numpy as np

    from sipnet_calibration.validation import as_positive_integer, as_site_ids

    sites = as_site_ids([1, 27, np.int32(4711)], message_name="sites")  # (1, 27, 4711)
    ncol = as_positive_integer(3, message_name="ncol")
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Iterable, KeysView, Mapping, Sequence, Set
from typing import Any

import numpy as np
import pandas as pd

from sipnet_calibration.conventions import SITE_DTYPE, FrozenMapping

__all__ = [
    "as_batched_flat",
    "as_bbox",
    "as_bounded_integer",
    "as_frozen_mapping",
    "as_integer",
    "as_names",
    "as_positive_integer",
    "as_positive_integers",
    "as_sequence",
    "as_site_id",
    "as_site_ids",
    "check_integers_are_in_range",
    "check_site_ids_are_in_range",
    "check_site_ids_are_unique",
    "is_one_vector",
    "range_summary",
    "truncated",
]

#: The largest site id: the largest value of :data:`SITE_DTYPE`.
_LARGEST_SITE_ID = int(np.iinfo(SITE_DTYPE).max)


# ── integers ──────────────────────────────────────────────────────────────────


def as_site_ids(values: Any, *, message_name: str) -> tuple[int, ...]:
    """Site ids as a tuple of plain integers, in the order given.

    Parameters
    ----------
    values:
        A sequence of site ids, each a Python or NumPy integer: a list, a
        tuple, a generator (read once), an ordered view such as
        ``dict.keys()``, or a one-dimensional array-like (NumPy, JAX, xarray,
        pandas).
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of int
        The ids, in the order given.

    Raises
    ------
    TypeError
        If *values* is one id, one string, a set, a mapping or not iterable,
        or an id is a boolean, a float, missing or not a number.
    ValueError
        If *values* is an array of more than one dimension, an id is not
        from 1 to the largest ``int32``, or a site is named more than once.
    """
    items = _sequence_items(values, what="site ids", example="[27]", message_name=message_name)
    site_ids = tuple(
        as_site_id(item, message_name=f"{message_name}[{i}]") for i, item in enumerate(items)
    )
    check_site_ids_are_unique(site_ids, message_name=message_name)
    return site_ids


def as_site_id(value: Any, *, message_name: str) -> int:
    """One site id as a plain integer.

    Parameters
    ----------
    value:
        A Python or NumPy integer, or a zero-dimensional array-like holding
        one.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The id.

    Raises
    ------
    TypeError
        If *value* is a boolean, a float or not a number.
    ValueError
        If *value* is not from 1 to the largest ``int32``.
    """
    site_id = as_integer(value, message_name=message_name)
    check_integer_is_in_range(
        site_id, minimum=1, maximum=_LARGEST_SITE_ID, what="a site id ", message_name=message_name
    )
    return site_id


def as_integer(value: Any, *, message_name: str) -> int:
    """An integer as a plain Python ``int``.

    Parameters
    ----------
    value:
        A Python or NumPy integer, or a zero-dimensional array-like holding
        one. A float is refused, even a whole one.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    TypeError
        If *value* is a boolean, a float or not an integer.
    """
    value = _scalar_of(value)
    check_value_is_an_integer(value, message_name=message_name)
    return int(value)


def as_bounded_integer(
    value: Any, *, minimum: int, maximum: int | None = None, message_name: str
) -> int:
    """An integer from *minimum* to *maximum*, as a plain Python ``int``.

    Parameters
    ----------
    value:
        As :func:`as_integer` takes it.
    minimum, maximum:
        The smallest and largest value accepted, inclusive; no upper bound
        when *maximum* is ``None``.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    TypeError
        If *value* is a boolean, a float or not an integer.
    ValueError
        If *value* is outside ``minimum..maximum``.
    """
    integer = as_integer(value, message_name=message_name)
    check_integer_is_in_range(integer, minimum=minimum, maximum=maximum, message_name=message_name)
    return integer


def as_positive_integer(value: Any, *, message_name: str) -> int:
    """A positive integer as a plain Python ``int``.

    Parameters
    ----------
    value:
        As :func:`as_integer` takes it, and at least 1.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    TypeError
        If *value* is a boolean, a float or not an integer.
    ValueError
        If *value* is less than 1.
    """
    return as_bounded_integer(value, minimum=1, message_name=message_name)


def as_positive_integers(values: Any, *, message_name: str) -> tuple[int, ...]:
    """A sequence of positive integers as a tuple of plain ``int``, in order.

    Parameters
    ----------
    values:
        A sequence, as :func:`as_site_ids` takes one, of values
        :func:`as_positive_integer` accepts. Repeats are kept.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of int
        The values, in the order given.

    Raises
    ------
    TypeError
        If *values* is one value, one string, a set, a mapping or not
        iterable, or an item is a boolean, a float, missing or not an
        integer.
    ValueError
        If *values* is an array of more than one dimension, or an item is
        less than 1.
    """
    items = _sequence_items(values, what="integers", example="[1, 2]", message_name=message_name)
    return tuple(
        as_positive_integer(item, message_name=f"{message_name}[{i}]")
        for i, item in enumerate(items)
    )


# ── arrays ────────────────────────────────────────────────────────────────────


def as_batched_flat(values: Any, dimension: int, *, message_name: str) -> Any:
    """One vector or a batch of vectors, as a two-dimensional ``float64`` batch.

    Parameters
    ----------
    values:
        One vector, shape ``(dimension,)``, or a batch of them, shape
        ``(n, dimension)``: an array-like of integers or floats, a JAX array,
        or a list of JAX arrays (tracers under ``jax.jit`` included).
    dimension:
        The number of entries each vector must have.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    array
        The values as ``float64``, shape ``(n, dimension)``, one vector its
        one row: a JAX array when *values* is or holds JAX arrays, else a
        NumPy array. A caller that needs to know whether one vector was
        given asks :func:`is_one_vector`.

    Raises
    ------
    TypeError
        If *values* is not a rectangular array of real numbers: ``None``, a
        string, a boolean or a complex number among them, or a ragged
        sequence.
    ValueError
        If *values* is not one- or two-dimensional, or does not have
        *dimension* entries per row.
    """
    array = _as_float64_array(values, message_name=message_name)
    check_flat_has_the_dimension(array.shape, dimension, message_name=message_name)
    return array[None, :] if array.ndim == 1 else array


def is_one_vector(values: Any) -> bool:
    """Whether Flat *values* is one vector, ``(D,)``, rather than a batch.

    Parameters
    ----------
    values:
        Flat as :func:`as_batched_flat` takes it, and has accepted.

    Returns
    -------
    bool
        ``True`` when *values* is one-dimensional.

    Notes
    -----
    ``np.ndim`` would read a list of JAX arrays by converting it to NumPy,
    which a list of tracers refuses under ``jax.jit``; such a list is read
    by JAX instead.
    """
    jax = sys.modules.get("jax")
    if jax is not None and _holds_jax_arrays(values, jax):
        return jax.numpy.asarray(values).ndim == 1
    return np.ndim(values) == 1


def as_bbox(bbox: Any, *, message_name: str) -> tuple[float, float, float, float]:
    """A ``(west, south, east, north)`` box, in degrees, as four floats.

    Parameters
    ----------
    bbox:
        Four numbers, west and east longitudes and south and north latitudes:
        a list, a tuple or a one-dimensional array-like. A box does not wrap
        the antimeridian, so west is at most east.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of float
        ``(west, south, east, north)``.

    Raises
    ------
    TypeError
        If *bbox* is not a sequence (a string, a mapping, a set and a
        generator are not), or a value is a boolean or not a number.
    ValueError
        If *bbox* does not hold four values, a value is not finite, west is
        east of east, or south is north of north.
    """
    check_bbox_is_a_sequence(bbox, message_name=message_name)
    values = tuple(_scalar_of(value) for value in _bbox_items(bbox))
    check_bbox_has_four_values(values, message_name=message_name)
    check_bbox_values_are_numbers(values, message_name=message_name)
    west, south, east, north = (float(value) for value in values)
    check_bbox_values_are_finite((west, south, east, north), message_name=message_name)
    check_bbox_west_is_not_east_of_east(west, east, message_name=message_name)
    check_bbox_south_is_not_north_of_north(south, north, message_name=message_name)
    return west, south, east, north


# ── names and mappings ────────────────────────────────────────────────────────


def as_sequence(values: Any, *, message_name: str) -> tuple[Any, ...]:
    """An ordered sequence of any items, as a tuple, in the order given.

    Parameters
    ----------
    values:
        A sequence, as :func:`as_site_ids` takes one: a list, a tuple, a
        generator, ``dict.keys()`` or a one-dimensional array-like. Its
        items may be of any type.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple
        The items, in the order given, an array-like's as plain Python
        values (``numpy.int64(1)`` as ``1``, ``numpy.str_("a")`` as ``"a"``).

    Raises
    ------
    TypeError
        If *values* is one string, one value, a set, a mapping or not
        iterable.
    ValueError
        If *values* is an array of more than one dimension.
    """
    example = _example_holding(values, fallback="[1, 2]", strings_only=False)
    items = _sequence_items(values, what="values", example=example, message_name=message_name)
    return tuple(_plain_item(item) for item in items)


def as_names(values: Any, *, message_name: str) -> tuple[str, ...]:
    """An ordered sequence of names as a tuple of strings.

    Parameters
    ----------
    values:
        A sequence of strings, as :func:`as_sequence` takes one.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of str
        The names, in the order given.

    Raises
    ------
    TypeError
        If *values* is one string, a set, a mapping or not iterable, or
        holds an item that is not a string.
    ValueError
        If *values* is an array of more than one dimension.
    """
    example = _example_holding(values, fallback="['a', 'b']", strings_only=True)
    items = _sequence_items(values, what="names", example=example, message_name=message_name)
    names = tuple(_plain_item(item) for item in items)
    check_names_are_strings(names, message_name=message_name)
    return names


def as_frozen_mapping(value: Any, *, message_name: str) -> FrozenMapping:
    """A mapping as a :class:`~sipnet_calibration.conventions.FrozenMapping`.

    Parameters
    ----------
    value:
        Any mapping; a ``FrozenMapping`` is returned as it is.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    FrozenMapping
        The same keys and values, in the same order, as a copy that cannot
        change.

    Raises
    ------
    TypeError
        If *value* is not a mapping.
    """
    if isinstance(value, FrozenMapping):
        return value
    check_value_is_a_mapping(value, message_name=message_name)
    return FrozenMapping(value)


# ── messages and reports ──────────────────────────────────────────────────────


def truncated(items: Iterable[Any], limit: int = 10) -> str:
    """*items* as a message shows them: a list of at most *limit*, then a count.

    Parameters
    ----------
    items:
        The items to show.
    limit:
        How many to show.

    Returns
    -------
    str
        ``"[1, 2, 3]"`` for *limit* items or fewer; past it, the first
        *limit* then a count, ``"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] and 5 more"``.
    """
    shown = list(items)
    if len(shown) <= limit:
        return repr(shown)
    return f"{shown[:limit]!r} and {len(shown) - limit} more"


def range_summary(values: Any) -> str:
    """The minimum, median, maximum and negative count of *values*, as a report shows them.

    Parameters
    ----------
    values:
        Finite numbers, any shape.

    Returns
    -------
    str
        Four columns, twelve characters wide, and dashes when *values* is
        empty, such as a variable absent everywhere.
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    if finite.size == 0:
        return f"{'-':<12s} {'-':<12s} {'-':<12s} -"
    return (
        f"{finite.min():<12.6g} {np.median(finite):<12.6g} {finite.max():<12.6g} "
        f"{int((finite < 0).sum())}"
    )


# ── supporting helpers ────────────────────────────────────────────────────────


def _scalar_of(value: Any) -> Any:
    """*value*, or the scalar a zero-dimensional array-like holds."""
    if hasattr(value, "__array__") and not isinstance(value, (str, bytes)) and np.ndim(value) == 0:
        return np.asarray(value).item()
    return value


def _is_array_like(value: Any) -> bool:
    """Whether *value* is an array-like (NumPy, JAX, xarray, pandas), not a list."""
    return hasattr(value, "__array__") and not isinstance(value, (str, bytes))


def _sequence_items(values: Any, *, what: str, example: str, message_name: str) -> list[Any]:
    """The items of a sequence argument, after the checks every one of them has."""
    check_sequence_is_not_a_string(values, what=what, example=example, message_name=message_name)
    check_sequence_is_not_a_set(values, message_name=message_name)
    check_sequence_is_not_a_mapping(values, message_name=message_name)
    if _is_array_like(values):
        array = _numpy_array_of(values)
        check_array_is_not_one_value(array, what=what, example=example, message_name=message_name)
        check_array_is_one_dimensional(array, message_name=message_name)
        return list(array)
    check_sequence_is_iterable(values, what=what, example=example, message_name=message_name)
    return list(values)


def _numpy_array_of(values: Any) -> np.ndarray:
    """An array-like as NumPy; a pandas extension array's missing values kept missing.

    ``np.asarray`` reads a nullable integer array holding ``pd.NA`` as floats,
    so a message would blame every value for being a float rather than the
    missing one for being missing; read as objects, each value is itself.
    """
    dtype = getattr(values, "dtype", None)
    if dtype is not None and not isinstance(dtype, np.dtype) and hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy(dtype=object))
    return np.asarray(values)


def _plain_item(item: Any) -> Any:
    """*item* as a plain Python value: a NumPy string as ``str``, a wrapped scalar unwrapped."""
    if isinstance(item, np.str_):
        return str(item)
    return _scalar_of(item)


def _example_holding(values: Any, *, fallback: str, strings_only: bool) -> str:
    """A message's example sequence: *values* in a list when it is one string or value."""
    item = _scalar_of(values)
    if isinstance(item, (str, np.str_)):
        return repr([str(item)])
    if not strings_only and isinstance(item, numbers.Number):
        return repr([item])
    return fallback


def _values_outside(values: Any, *, minimum: int, maximum: int | None) -> list[Any]:
    """The distinct values that are not integers from *minimum* to *maximum*, in order.

    A float is not an integer, even a whole one, and neither is ``NaN``; the
    values are listed once each, in the order they first appear.
    """
    wrong: dict[str, Any] = {}
    for value in np.asarray(values).ravel().tolist():
        is_integer = isinstance(value, numbers.Integral) and not isinstance(value, bool)
        if not is_integer or value < minimum or (maximum is not None and value > maximum):
            wrong.setdefault(repr(value), value)
    return list(wrong.values())


def _bbox_items(bbox: Any) -> list[Any]:
    """The values of a box, from a sequence or an array-like."""
    if _is_array_like(bbox):
        array = np.asarray(bbox)
        return list(array) if array.ndim == 1 else [array]
    return list(bbox)


def _holds_jax_arrays(values: Any, jax: Any) -> bool:
    """Whether *values* is a JAX array or a list or tuple holding one."""
    if isinstance(values, jax.Array):
        return True
    if isinstance(values, (list, tuple)):
        return any(_holds_jax_arrays(value, jax) for value in values)
    return False


def _numpy_array_or_none(values: Any) -> np.ndarray | None:
    """``np.asarray(values)``, or ``None`` when NumPy cannot read it as an array."""
    try:
        return np.asarray(values)
    except (TypeError, ValueError):
        return None


def _as_float64_array(values: Any, *, message_name: str) -> Any:
    """*values* as a ``float64`` array: JAX if it is or holds JAX arrays, else NumPy."""
    jax = sys.modules.get("jax")
    if jax is not None and _holds_jax_arrays(values, jax):
        others = [leaf for leaf in _list_leaves(values) if not isinstance(leaf, jax.Array)]
        if others:
            # Read by NumPy, the values beside the JAX arrays give the dtype, and so
            # the message, a list without JAX arrays would.
            check_array_holds_real_numbers(np.asarray(others).dtype, message_name=message_name)
        array = _jax_array_or_none(values, jax)
        check_values_form_an_array(array, message_name=message_name)
        check_array_holds_real_numbers(array.dtype, message_name=message_name)
        return array.astype(jax.numpy.float64)
    array = _numpy_array_or_none(values)
    check_values_form_an_array(array, message_name=message_name)
    check_array_holds_real_numbers(array.dtype, message_name=message_name)
    return array.astype(np.float64)


def _list_leaves(values: Any) -> list[Any]:
    """The items of nested lists and tuples, depth first; *values* itself if it is neither."""
    if isinstance(values, (list, tuple)):
        return [leaf for value in values for leaf in _list_leaves(value)]
    return [values]


def _jax_array_or_none(values: Any, jax: Any) -> Any:
    """``jnp.asarray(values)``, or ``None`` when JAX cannot read it as one array."""
    try:
        return jax.numpy.asarray(values)
    except (TypeError, ValueError):
        return None


# ── checks ────────────────────────────────────────────────────────────────────


def check_sequence_is_not_a_string(
    values: Any, *, what: str, example: str, message_name: str
) -> None:
    """A sequence argument is not one string, which would be read a character at a time."""
    if isinstance(values, (str, bytes, np.str_)):
        raise TypeError(
            f"{message_name} must be a sequence of {what}, got the one string {values!r}; "
            f"pass a sequence such as {example}."
        )


def check_sequence_is_not_a_set(values: Any, *, message_name: str) -> None:
    """A sequence argument is not a set, which has no order to keep."""
    if isinstance(values, Set) and not isinstance(values, KeysView):
        raise TypeError(
            f"{message_name} was given as a {type(values).__name__}, which has no order to "
            "keep; pass a list or a tuple."
        )


def check_sequence_is_not_a_mapping(values: Any, *, message_name: str) -> None:
    """A sequence argument is not a mapping, whose keys are passed as a sequence when meant."""
    if isinstance(values, Mapping):
        raise TypeError(
            f"{message_name} must be a sequence, got a {type(values).__name__}, which is a "
            "mapping; pass a list, or list(m) or m.keys() where its keys are meant."
        )


def check_sequence_is_iterable(values: Any, *, what: str, example: str, message_name: str) -> None:
    """A sequence argument is iterable, not one value."""
    if not isinstance(values, Iterable):
        raise TypeError(
            f"{message_name} must be a sequence of {what}, got {type(values).__name__} "
            f"{values!r}; pass a sequence such as {example}."
        )


def check_array_is_not_one_value(
    array: np.ndarray, *, what: str, example: str, message_name: str
) -> None:
    """An array-like sequence argument is not zero-dimensional, one value on its own."""
    if array.ndim == 0:
        raise TypeError(
            f"{message_name} must be a sequence of {what}, got the one value {array.item()!r}; "
            f"pass a sequence such as {example}."
        )


def check_array_is_one_dimensional(array: np.ndarray, *, message_name: str) -> None:
    """An array-like sequence argument is one-dimensional."""
    if array.ndim != 1:
        raise ValueError(
            f"{message_name} must be one-dimensional, got an array of shape {array.shape}; "
            "pass a flat sequence, or .ravel() it."
        )


def check_value_is_an_integer(value: Any, *, message_name: str) -> None:
    """A value is an integer: not a boolean, not a float, not a non-number."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{message_name} must be an integer, got the boolean {value!r}; pass an integer."
        )
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise TypeError(
            f"{message_name} must be an integer, got {float(value)!r}, which is missing or "
            "infinite; drop or fill such values first."
        )
    if isinstance(value, (float, np.floating)):
        raise TypeError(
            f"{message_name} must be an integer, got the float {value!r}; cast it with "
            "int() where it is known to be whole."
        )
    if value is pd.NA:
        raise TypeError(
            f"{message_name} must be an integer, got <NA>, a missing value; drop or fill "
            "missing values first."
        )
    if not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{message_name} must be an integer, got {type(value).__name__} {value!r}; "
            "pass an integer."
        )


def check_integer_is_in_range(
    value: int, *, minimum: int, maximum: int | None, what: str = "", message_name: str
) -> None:
    """An integer is from *minimum* to *maximum*, inclusive."""
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"at least {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        raise ValueError(
            f"{message_name} must be {what}{bounds}, got {value}; pass a value in that range."
        )


def check_integers_are_in_range(
    values: np.ndarray, *, minimum: int, maximum: int | None, message_name: str
) -> None:
    """Every one of some values is an integer from *minimum* to *maximum*, inclusive."""
    wrong = _values_outside(values, minimum=minimum, maximum=maximum)
    if wrong:
        bounds = f"at least {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        raise ValueError(
            f"{message_name} must be integers {bounds}, got {truncated(wrong)}; pass "
            "values in that range."
        )


def check_site_ids_are_in_range(site_ids: np.ndarray, *, message_name: str) -> None:
    """Every site id is an integer from 1 to the largest ``int32``, the site id's dtype."""
    wrong = _values_outside(site_ids, minimum=1, maximum=_LARGEST_SITE_ID)
    if wrong:
        raise ValueError(
            f"{message_name} must be integer site ids from 1 to {_LARGEST_SITE_ID}, got "
            f"{truncated(wrong)}; pass the site table's site_id values."
        )


def check_site_ids_are_unique(site_ids: Sequence[int], *, message_name: str) -> None:
    """No site id appears twice."""
    seen: set[int] = set()
    repeated = sorted({site for site in site_ids if site in seen or seen.add(site)})
    if repeated:
        raise ValueError(
            f"{message_name} names site(s) {truncated(repeated)} more than once; name each "
            "site once."
        )


def check_values_form_an_array(array: np.ndarray | None, *, message_name: str) -> None:
    """NumPy could read the values as one rectangular array."""
    if array is None:
        raise TypeError(
            f"{message_name} must be a rectangular array of numbers, got a ragged or "
            "unreadable sequence; pass an array of shape (n,) or (m, n)."
        )


def check_array_holds_real_numbers(dtype: Any, *, message_name: str) -> None:
    """An array's dtype is an integer or a float one, not a boolean, complex or object one."""
    if np.dtype(dtype).kind not in "iuf":
        raise TypeError(
            f"{message_name} must hold real numbers, got dtype {np.dtype(dtype)} (a None, a "
            "string, a boolean or a complex number among the values reads so); pass floats."
        )


def check_flat_has_the_dimension(
    shape: tuple[int, ...], dimension: int, *, message_name: str
) -> None:
    """Flat is one vector of *dimension* entries or a batch of them."""
    if len(shape) not in (1, 2) or shape[-1] != dimension:
        raise ValueError(
            f"{message_name} must be one vector of {dimension} entries or a batch of them, "
            f"shape ({dimension},) or (n, {dimension}), got shape {tuple(shape)}; pass Flat "
            "in that shape."
        )


def check_bbox_is_a_sequence(bbox: Any, *, message_name: str) -> None:
    """A box is a sequence or an array-like, whose order is its meaning."""
    if isinstance(bbox, (str, bytes, Mapping, Set)) or not (
        isinstance(bbox, Sequence) or _is_array_like(bbox)
    ):
        raise TypeError(
            f"{message_name} must be a sequence (west, south, east, north), got "
            f"{type(bbox).__name__} {bbox!r}; pass a tuple of four numbers."
        )


def check_bbox_has_four_values(values: tuple[Any, ...], *, message_name: str) -> None:
    """A box holds four values."""
    if len(values) != 4:
        raise ValueError(
            f"{message_name} must be (west, south, east, north), got {len(values)} "
            "value(s); pass four numbers."
        )


def check_bbox_values_are_numbers(values: tuple[Any, ...], *, message_name: str) -> None:
    """Every value of a box is a real number, and none a boolean."""
    wrong = [
        v for v in values if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)
    ]
    if wrong:
        raise TypeError(
            f"{message_name} values must be numbers, got {truncated(wrong)}; pass four "
            "longitudes and latitudes in degrees."
        )


def check_bbox_values_are_finite(values: tuple[float, ...], *, message_name: str) -> None:
    """Every value of a box is finite."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"{message_name} values must be finite, got {values}; pass the box's edges in "
            "degrees."
        )


def check_bbox_west_is_not_east_of_east(west: float, east: float, *, message_name: str) -> None:
    """A box's west edge is not east of its east edge."""
    if west > east:
        raise ValueError(
            f"{message_name} west {west} is east of east {east}; a box does not wrap the "
            "antimeridian, so give west <= east."
        )


def check_bbox_south_is_not_north_of_north(
    south: float, north: float, *, message_name: str
) -> None:
    """A box's south edge is not north of its north edge."""
    if south > north:
        raise ValueError(
            f"{message_name} south {south} is north of north {north}; give south <= north."
        )


def check_names_are_strings(names: tuple[Any, ...], *, message_name: str) -> None:
    """Every name is a string."""
    wrong = [name for name in names if not isinstance(name, str)]
    if wrong:
        raise TypeError(
            f"{message_name} must be strings, got {truncated(wrong)}; pass names as strings."
        )


def check_value_is_a_mapping(value: Any, *, message_name: str) -> None:
    """A value is a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{message_name} must be a mapping, got {type(value).__name__} {value!r}; pass "
            "a dict."
        )

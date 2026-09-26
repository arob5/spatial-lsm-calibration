"""Argument coercion every module shares: check a value and convert it.

Each function here takes an argument as a caller passed it, refuses it if it
is not the thing asked for, and returns it in one canonical form, so that two
functions taking the same kind of argument accept and refuse the same values.

Contents
--------
:func:`as_site_ids`, :func:`as_site_id`
    Site ids as plain Python integers.
:func:`as_integer`, :func:`as_positive_integer`
    A count or a size as a plain Python integer.
:func:`as_batched_flat`
    One vector or a batch of them, as a two-dimensional ``float64`` array.
:func:`as_bbox`
    A ``(west, south, east, north)`` box as four floats.
:func:`as_names`
    An ordered sequence of names as a tuple of strings.
:func:`truncated`
    A list shortened for an error message.

Every coercer takes the name the value goes by in a message, *message_name*,
as a keyword, and raises by one rule: :class:`TypeError` when the value is of
the wrong type -- a boolean where a number is wanted, or one string where a
sequence is -- and :class:`ValueError` when it is of the right type but a
wrong value.

Notes
-----
This module depends only on :mod:`sipnet_calibration.conventions`, so any
module of the package can use it.

A JAX array given to :func:`as_batched_flat` stays a JAX array, so a function
traced by ``jax.jit`` or ``jax.grad`` can coerce its argument; JAX is not
imported unless the caller has imported it already.

Usage
-----
::

    import numpy as np

    from sipnet_calibration.validation import as_positive_integer, as_site_ids

    sites = as_site_ids([1, 27.0, np.int32(4711)], message_name="sites")  # (1, 27, 4711)
    ncol = as_positive_integer(3, message_name="ncol")
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Any

import numpy as np

from sipnet_calibration.conventions import SITE_DTYPE

__all__ = [
    "as_batched_flat",
    "as_bbox",
    "as_integer",
    "as_names",
    "as_positive_integer",
    "as_site_id",
    "as_site_ids",
    "truncated",
]

#: The largest site id: the largest value of :data:`SITE_DTYPE`.
_LARGEST_SITE_ID = int(np.iinfo(SITE_DTYPE).max)


def as_site_ids(
    values: Iterable[Any] | Any,
    *,
    message_name: str = "sites",
    allow_one_id: bool = False,
) -> tuple[int, ...]:
    """Site ids as a tuple of plain integers, in the order given.

    Parameters
    ----------
    values:
        An iterable of site ids: Python or NumPy integers, or floats that are
        whole numbers, such as a table's ``site_id`` column read as float. It
        is read once, so a generator is accepted.
    message_name:
        What the argument is called in an error message.
    allow_one_id:
        Whether one site id on its own, rather than in a sequence, is
        accepted as a sequence of one.

    Returns
    -------
    tuple of int
        The ids, in the order given.

    Raises
    ------
    TypeError
        If *values* is a string or is not iterable, or an id is a boolean or
        not a number.
    ValueError
        If an id is not a whole number from 1 to the largest ``int32``, or
        *values* names a site more than once.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{message_name} must be a sequence of site ids, got the string {values!r}, "
            "which would be read one character per site; pass e.g. [27]."
        )
    if allow_one_id and _is_one_value(values):
        return (as_site_id(values, message_name=message_name),)
    if isinstance(values, np.ndarray) and values.ndim == 0:
        raise TypeError(
            f"{message_name} must be a sequence of site ids, got the one value {values!r}; "
            "pass e.g. [27]."
        )
    if not isinstance(values, Iterable):
        raise TypeError(
            f"{message_name} must be a sequence of site ids, got {type(values).__name__} "
            f"{values!r}; pass e.g. [1, 27]."
        )
    if isinstance(values, np.ndarray) and values.ndim > 1:
        raise ValueError(
            f"{message_name} must be one-dimensional, got an array of shape {values.shape}."
        )
    site_ids = tuple(as_site_id(value, message_name=f"each of {message_name}") for value in values)
    check_site_ids_are_unique(site_ids, message_name=message_name)
    return site_ids


def as_site_id(value: Any, *, message_name: str = "site") -> int:
    """One site id as a plain integer.

    Parameters
    ----------
    value:
        A Python or NumPy integer, or a float that is a whole number.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The id.

    Raises
    ------
    TypeError
        If *value* is a boolean or not a number.
    ValueError
        If *value* is not a whole number from 1 to the largest ``int32``.
    """
    value = _unwrapped_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{message_name} must be an integer, got {type(value).__name__} {value!r}; "
            "a site id is e.g. 27."
        )
    if not isinstance(value, numbers.Integral) and not (
        math.isfinite(value) and float(value).is_integer()
    ):
        raise ValueError(
            f"{message_name} must be a whole number, got {value!r}; int() would truncate "
            "it to a different site, so pass the integer site id."
        )
    site_id = int(value)
    if not 1 <= site_id <= _LARGEST_SITE_ID:
        raise ValueError(
            f"{message_name} must be a site id from 1 to {_LARGEST_SITE_ID}, got {site_id}."
        )
    return site_id


def as_integer(value: Any, *, message_name: str) -> int:
    """An integer as a plain Python ``int``.

    Parameters
    ----------
    value:
        A Python or NumPy integer. A float is refused, even a whole one, since
        a count that arrives as a float is usually the result of a mistake.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    TypeError
        If *value* is a boolean or not an integer.
    """
    value = _unwrapped_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{message_name} must be an integer, got {type(value).__name__} {value!r}."
        )
    return int(value)


def as_positive_integer(value: Any, *, message_name: str) -> int:
    """A positive integer as a plain Python ``int``.

    Parameters
    ----------
    value:
        A Python or NumPy integer of at least 1.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    int
        The value.

    Raises
    ------
    TypeError
        If *value* is a boolean or not an integer.
    ValueError
        If *value* is less than 1.
    """
    integer = as_integer(value, message_name=message_name)
    if integer < 1:
        raise ValueError(f"{message_name} must be a positive integer, got {integer}.")
    return integer


def as_batched_flat(
    values: Any,
    width: int,
    *,
    message_name: str,
    allow_no_rows: bool = True,
    allow_non_finite: bool = True,
) -> tuple[Any, bool]:
    """One vector or a batch of vectors, as a two-dimensional batch.

    Parameters
    ----------
    values:
        One vector, shape ``(width,)``, or a batch of them, shape
        ``(n, width)``: anything :func:`numpy.asarray` accepts, or a JAX
        array.
    width:
        The number of entries each vector must have.
    message_name:
        What the argument is called in an error message.
    allow_no_rows:
        Whether a batch of no vectors, shape ``(0, width)``, is accepted.
    allow_non_finite:
        Whether a ``NaN`` or an infinite entry is accepted. Only a concrete
        array is checked; a JAX array being traced cannot be.

    Returns
    -------
    batched_flat:
        The values as ``float64``, shape ``(n, width)``: a NumPy array, or a
        JAX array when *values* is one.
    was_one_vector:
        Whether *values* was one vector, ``(width,)``, now its one row.

    Raises
    ------
    TypeError
        If *values* cannot be read as an array of numbers.
    ValueError
        If *values* is not one- or two-dimensional; if it does not have
        *width* entries per row; if it has no rows and *allow_no_rows* is
        false; or if an entry is not finite and *allow_non_finite* is false.
    """
    array = _as_float64_array(values, message_name=message_name)
    if array.ndim not in (1, 2) or array.shape[-1] != width:
        raise ValueError(
            f"{message_name} must be one vector of {width} entries or a batch of them, "
            f"shape ({width},) or (n, {width}); got shape {tuple(array.shape)}."
        )
    was_one_vector = array.ndim == 1
    batched = array[None, :] if was_one_vector else array
    if not allow_no_rows and batched.shape[0] == 0:
        raise ValueError(
            f"{message_name} must be at least one vector of {width} entries; got shape "
            f"{tuple(batched.shape)}."
        )
    if not allow_non_finite and not _is_traced(batched) and not bool(np.isfinite(batched).all()):
        raise ValueError(
            f"{message_name} holds a non-finite value; every entry must be a finite number."
        )
    return batched, was_one_vector


def as_bbox(bbox: Any, *, message_name: str = "bbox") -> tuple[float, float, float, float]:
    """A ``(west, south, east, north)`` box, in degrees, as four floats.

    Parameters
    ----------
    bbox:
        Four numbers, west and east longitudes and south and north latitudes.
        A box does not wrap the antimeridian, so west is at most east.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of float
        ``(west, south, east, north)``.

    Raises
    ------
    TypeError
        If *bbox* is not a sequence, or a value is a boolean or not a number.
    ValueError
        If *bbox* does not hold four values, a value is not finite, west is
        east of east, or south is north of north.
    """
    if isinstance(bbox, (str, bytes, Mapping)) or not isinstance(bbox, Iterable):
        raise TypeError(
            f"{message_name} must be (west, south, east, north), got "
            f"{type(bbox).__name__} {bbox!r}."
        )
    values = tuple(_unwrapped_scalar(value) for value in bbox)
    if len(values) != 4:
        raise ValueError(
            f"{message_name} must be (west, south, east, north), got {len(values)} "
            f"value(s): {bbox!r}."
        )
    wrong = [v for v in values if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)]
    if wrong:
        raise TypeError(f"{message_name} values must be numbers, got {bbox!r}.")
    west, south, east, north = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError(f"{message_name} values must be finite, got {bbox!r}.")
    if west > east:
        raise ValueError(
            f"{message_name} west {west} is east of east {east}; a box does not wrap the "
            "antimeridian, so give west <= east. Longitudes are negative throughout the "
            "site pool."
        )
    if south > north:
        raise ValueError(
            f"{message_name} south {south} is north of north {north}; give south <= north."
        )
    return west, south, east, north


def as_names(values: Any, *, message_name: str) -> tuple[str, ...]:
    """An ordered sequence of names as a tuple of strings.

    Parameters
    ----------
    values:
        A list, tuple or other ordered iterable of strings.
    message_name:
        What the argument is called in an error message.

    Returns
    -------
    tuple of str
        The names, in the order given.

    Raises
    ------
    TypeError
        If *values* is one string, a set (which has no order to keep) or not
        iterable, or holds an item that is not a string.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{message_name} must be a sequence of names, got the one string {values!r}; "
            f"pass [{values!r}]."
        )
    if isinstance(values, Set):
        raise TypeError(
            f"{message_name} was given as a {type(values).__name__}, which has no order "
            "to keep; pass a list or a tuple."
        )
    if not isinstance(values, Iterable):
        raise TypeError(
            f"{message_name} must be a sequence of names, got {type(values).__name__} "
            f"{values!r}."
        )
    names = tuple(values)
    wrong = [name for name in names if not isinstance(name, str)]
    if wrong:
        raise TypeError(f"{message_name} must be strings, got {truncated(wrong)}.")
    return names


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
        ``"[1, 2, 3]"``, or ``"[1, 2, ..., 10] and 5 more"`` past *limit*.
    """
    shown = list(items)
    if len(shown) <= limit:
        return repr(shown)
    return f"{shown[:limit]!r} and {len(shown) - limit} more"


# ── supporting helpers ────────────────────────────────────────────────────────


def _is_one_value(value: Any) -> bool:
    """Whether *value* is one number, rather than a sequence of them."""
    return isinstance(value, numbers.Number) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _unwrapped_scalar(value: Any) -> Any:
    """*value*, or the scalar a zero-dimensional NumPy array holds."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _as_float64_array(values: Any, *, message_name: str) -> Any:
    """*values* as a ``float64`` array: a JAX array if it is one, else NumPy."""
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(values, jax.Array):
        return jax.numpy.asarray(values, dtype=jax.numpy.float64)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{message_name} must be an array of numbers; {error}.") from None


def _is_traced(array: Any) -> bool:
    """Whether *array* is a JAX tracer, whose values are not known."""
    jax = sys.modules.get("jax")
    return jax is not None and isinstance(array, jax.core.Tracer)


# ── checks ────────────────────────────────────────────────────────────────────


def check_site_ids_are_unique(site_ids: Sequence[int], *, message_name: str) -> None:
    """No site id appears twice."""
    seen: set[int] = set()
    repeated = sorted({site for site in site_ids if site in seen or seen.add(site)})
    if repeated:
        raise ValueError(
            f"{message_name} names site(s) {truncated(repeated)} more than once; name each "
            "site once."
        )

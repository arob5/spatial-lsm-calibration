"""The argument coercion the probability layer needs, private to it.

The probability layer imports nothing of the package outside itself, so the
few coercions it shares with :mod:`sipnet_calibration.validation` are kept
here, raising by the same rules: a sequence argument is a sequence, never
one string, a set or a mapping; an integer is never a boolean or a float.
The parameter layer reads them through its re-export.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, KeysView, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import numpy as np

__all__ = [
    "as_count",
    "as_names",
    "as_sequence",
    "check_names_are_unique",
    "truncated",
]


def as_sequence(values: Any, *, message_name: str) -> tuple[Any, ...]:
    """An ordered sequence of any items as a tuple of plain Python values.

    Raises
    ------
    TypeError
        If *values* is one string, one value, a set, a mapping or not
        iterable.
    ValueError
        If *values* is an array of more than one dimension.
    """
    check_sequence_is_not_a_string(values, message_name=message_name)
    check_sequence_is_not_a_set_or_a_mapping(values, message_name=message_name)
    if hasattr(values, "__array__"):
        array = np.asarray(values)
        check_array_is_one_dimensional(array, message_name=message_name)
        items: Iterable[Any] = array.tolist()
    else:
        check_sequence_is_iterable(values, message_name=message_name)
        items = values
    return tuple(_plain_item(item) for item in items)


def as_names(values: Any, *, message_name: str) -> tuple[str, ...]:
    """An ordered sequence of names as a tuple of strings.

    Raises
    ------
    TypeError
        As :func:`as_sequence`, or for an item that is not a string.
    ValueError
        As :func:`as_sequence`.
    """
    names = as_sequence(values, message_name=message_name)
    check_names_are_strings(names, message_name=message_name)
    return names


def as_count(value: Any, *, message_name: str) -> int:
    """A non-negative integer as a plain Python ``int``.

    Raises
    ------
    TypeError
        If *value* is a boolean, a float or not an integer.
    ValueError
        If *value* is negative.
    """
    value = _plain_item(value)
    check_value_is_an_integer(value, message_name=message_name)
    check_count_is_not_negative(int(value), message_name=message_name)
    return int(value)


def truncated(items: Iterable[Any], limit: int = 10) -> str:
    """*items* as a message shows them: at most *limit*, then a count."""
    shown = list(items)
    if len(shown) <= limit:
        return repr(shown)
    return f"{shown[:limit]!r} and {len(shown) - limit} more"


# ── helpers ───────────────────────────────────────────────────────────────────


def _plain_item(item: Any) -> Any:
    """*item* as a plain Python value: a NumPy scalar or a 0-d array unwrapped."""
    if isinstance(item, np.str_):
        return str(item)
    if hasattr(item, "__array__") and not isinstance(item, (str, bytes)) and np.ndim(item) == 0:
        return np.asarray(item).item()
    return item


# ── checks ────────────────────────────────────────────────────────────────────


def check_sequence_is_not_a_string(values: Any, *, message_name: str) -> None:
    """A sequence argument is not one string, which would be read a character at a time."""
    if isinstance(values, (str, bytes, np.str_)):
        raise TypeError(
            f"{message_name} must be a sequence, got the one string {values!r}; pass "
            f"[{values!r}]."
        )


def check_sequence_is_not_a_set_or_a_mapping(values: Any, *, message_name: str) -> None:
    """A sequence argument is not a set, which has no order, nor a mapping."""
    if isinstance(values, AbstractSet) and not isinstance(values, KeysView):
        raise TypeError(
            f"{message_name} was given as a {type(values).__name__}, which has no order to "
            "keep; pass a list or a tuple."
        )
    if isinstance(values, Mapping):
        raise TypeError(
            f"{message_name} must be a sequence, got a {type(values).__name__}; pass a list, "
            "or m.keys() where its keys are meant."
        )


def check_sequence_is_iterable(values: Any, *, message_name: str) -> None:
    """A sequence argument is iterable, not one value."""
    if not isinstance(values, Iterable):
        raise TypeError(
            f"{message_name} must be a sequence, got {type(values).__name__} {values!r}; "
            f"pass [{values!r}]."
        )


def check_array_is_one_dimensional(array: np.ndarray, *, message_name: str) -> None:
    """An array-like sequence argument is one-dimensional: not one value, not a table."""
    if array.ndim == 0:
        raise TypeError(
            f"{message_name} must be a sequence, got the one value {array.item()!r}; pass "
            f"[{array.item()!r}]."
        )
    if array.ndim != 1:
        raise ValueError(
            f"{message_name} must be one-dimensional, got an array of shape {array.shape}; "
            "pass a flat sequence."
        )


def check_names_are_strings(names: Sequence[Any], *, message_name: str) -> None:
    """Every name is a string."""
    wrong = [name for name in names if not isinstance(name, str)]
    if wrong:
        raise TypeError(f"{message_name} must be strings, got {truncated(wrong)}; pass names as strings.")


def check_names_are_unique(names: Sequence[Any], *, message_name: str) -> None:
    """No name appears twice."""
    seen: set[Any] = set()
    repeated = [name for name in names if name in seen or seen.add(name)]
    if repeated:
        raise ValueError(f"{message_name} names {truncated(dict.fromkeys(repeated))} more than once; name each once.")


def check_value_is_an_integer(value: Any, *, message_name: str) -> None:
    """A value is an integer: not a boolean, not a float, not a non-number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{message_name} must be an integer, got {type(value).__name__} {value!r}; pass "
            "an integer, casting a whole float with int()."
        )


def check_count_is_not_negative(value: int, *, message_name: str) -> None:
    """A count is not negative."""
    if value < 0:
        raise ValueError(f"{message_name} is {value}, a count below zero; give zero or more.")

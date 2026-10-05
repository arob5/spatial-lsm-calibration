"""The keyword rule: what a function a user writes reads. Private to the
probability layer.

A function is called with the names of its parameters that have no
default, so what it reads is its signature. A function with ``*args``,
``**kwargs`` or a positional-only parameter is refused, since what it reads
cannot be read off its signature; a ``functools.partial``'s bound arguments
are defaults, so they are not read.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

__all__ = ["check_function_follows_the_keyword_rule", "function_reads"]

#: The kinds of parameter whose names a call cannot be read from.
_UNREADABLE_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
    inspect.Parameter.POSITIONAL_ONLY: "a positional-only parameter",
}


def function_reads(function: Callable[..., Any], *, message_name: str) -> tuple[str, ...]:
    """The names *function* reads: its parameters without defaults, in
    signature order.

    Raises
    ------
    TypeError
        If *function* is not callable, has no signature Python can read, or
        breaks the keyword rule.
    """
    check_function_follows_the_keyword_rule(function, message_name=message_name)
    return tuple(
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_function_follows_the_keyword_rule(function: Any, *, message_name: str) -> None:
    """A function is callable, has a signature, and takes every argument by
    a name the signature states."""
    if not callable(function):
        raise TypeError(f"{message_name} is a {type(function).__name__}, which is not callable.")
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        raise TypeError(
            f"{message_name} has no signature Python can read; wrap it in a def that names what it reads."
        ) from None
    unreadable = [
        f"{name!r} ({_UNREADABLE_KINDS[parameter.kind]})"
        for name, parameter in parameters.items()
        if parameter.kind in _UNREADABLE_KINDS
    ]
    if unreadable:
        raise TypeError(
            f"{message_name} takes {', '.join(unreadable)}, so what it reads cannot be read off its "
            "signature; name each value it reads as a keyword parameter, such as "
            "def law(soil_carbon_location, soil_carbon_spread)."
        )

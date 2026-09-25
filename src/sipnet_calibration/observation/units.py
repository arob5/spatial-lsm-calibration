"""Arithmetic on model arrays that carries pySIPNET's ``units``, ``constituent``
and ``kind`` through, so an observation operator's result describes itself.

xarray drops attributes in arithmetic, and a prediction that arrives at
:meth:`~sipnet_calibration.observation.ObservationVector.predict` without
``units`` cannot be converted into the observation's units, only refused. The
five functions here do the arithmetic an operator needs and write the
attributes that are true of the result. Conversion itself is pySIPNET's:
:func:`pysipnet.units.convert_dataarray_units` reads the attributes these
write.

Provided:

* ``multiply(a, b)``, ``divide(a, b)``, ``add(a, b)``, ``subtract(a, b)`` --
  the operation, with ``units`` combined by symbol, ``constituent`` and
  ``kind`` by the rules below.
* ``step_length(array, units="d")`` -- pySIPNET's ``time_step_length`` as a
  float array with units, so a per-step total divides into a rate.

The rules
---------
An operand is a ``DataArray`` carrying ``units`` (and, optionally,
``constituent`` and ``kind``), or a plain number, which is dimensionless with
no constituent and no kind. A SIPNET parameter array from a SIPNET table
carries ``units`` and ``constituent`` and no ``kind``; a model variable
carries all three.

* **Units** combine by symbol: ``g m-2`` times ``d-1`` is ``g m-2 d-1``,
  and ``g m-2`` divided by ``g m-2`` is ``1``. Nothing is rescaled, so
  ``cm`` divided by ``m`` is ``cm m-1``, which pySIPNET converts to ``1``
  on request. ``add`` and ``subtract`` require the same unit string.
* **Constituent**: in a product, at most one operand names one and the result
  takes it; in a quotient, the numerator's is kept when the denominator has
  none, and it cancels when both name the same one (leaf carbon over leaf
  carbon per area is an area ratio, not carbon). A denominator with a
  constituent the numerator lacks, or two different constituents, is refused.
  ``add`` and ``subtract`` require the same constituent.
* **Kind**: in a product or quotient, at most one operand has a kind and the
  result keeps it, with two exceptions matching pySIPNET's kinds: a
  ``timestep_total`` divided by a time, or multiplied by a per-time, is a
  ``daily_rate``; a ``daily_rate`` multiplied by a time, or divided by a
  per-time, is a ``timestep_total``. (The kind is named ``daily_rate``
  whatever the time unit; the units say which.) Any other combination of a
  kinded operand with a time or a per-time is refused, because it would
  change what the value is over a step without a kind to say so: a pool
  times a turnover rate is a flux SIPNET reports itself. Two kinded operands
  in a product or quotient are refused (a state times a flux is not
  something SIPNET reports). ``add`` and ``subtract`` require the same kind.

The algebra is deliberately not closed. An operator that needs more sets the
three attributes itself, and is checked the same way at the boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from pysipnet.units import unit_registry, validate_units
from pysipnet.variables import CELL_METHODS_FOR_KIND, TIME_REFERENCE_FOR_KIND, VariableKind

from sipnet_calibration.observation.alignment import LENGTH_COORD, TIME_DIM

__all__ = ["add", "divide", "multiply", "step_length", "subtract"]

_TOKEN = re.compile(r"^([A-Za-z]+)(-?\d+)?$")


def multiply(a: Any, b: Any) -> xr.DataArray:
    """``a * b`` with ``units``, ``constituent`` and ``kind`` carried through."""
    ua, ca, ka = _describe(a, "multiply")
    ub, cb, kb = _describe(b, "multiply")
    kind = _product_kind(ka, ua, kb, ub, "*")
    units = _combine_units(ua, ub, +1)
    constituent = _product_constituent(ca, cb)
    return _result(_value(a) * _value(b), units, constituent, kind, a, b, "*")


def divide(a: Any, b: Any) -> xr.DataArray:
    """``a / b`` with ``units``, ``constituent`` and ``kind`` carried through."""
    ua, ca, ka = _describe(a, "divide")
    ub, cb, kb = _describe(b, "divide")
    kind = _product_kind(ka, ua, kb, ub, "/")
    units = _combine_units(ua, ub, -1)
    constituent = _quotient_constituent(ca, cb)
    return _result(_value(a) / _value(b), units, constituent, kind, a, b, "/")


def add(a: Any, b: Any) -> xr.DataArray:
    """``a + b``; the operands must agree in units, constituent and kind."""
    units, constituent, kind = _same_description(a, b, "add")
    return _result(_value(a) + _value(b), units, constituent, kind, a, b, "+")


def subtract(a: Any, b: Any) -> xr.DataArray:
    """``a - b``; the operands must agree in units, constituent and kind."""
    units, constituent, kind = _same_description(a, b, "subtract")
    return _result(_value(a) - _value(b), units, constituent, kind, a, b, "-")


def step_length(array: xr.DataArray, units: str = "d") -> xr.DataArray:
    """The length of each of *array*'s timesteps, as a float array with units.

    Reads pySIPNET's ``time_step_length`` coordinate and returns it on the
    same ``time`` coordinate in *units* (``"d"``, ``"h"`` or ``"s"``), so that
    ``divide(total, step_length(total))`` is a rate. A missing length (the
    ``NaT`` padding a stack of unequal records carries) is ``NaN``.
    """
    if LENGTH_COORD not in array.coords:
        raise ValueError(
            f"{array.name!r} carries no {LENGTH_COORD!r} coordinate; only model output "
            "from pySIPNET declares its step lengths."
        )
    per_unit = {"d": 86_400e9, "h": 3_600e9, "s": 1e9}
    if units not in per_unit:
        raise ValueError(f"units must be one of {sorted(per_unit)}, got {units!r}.")
    lengths = array[LENGTH_COORD].values.astype("timedelta64[ns]")
    nanoseconds = np.where(np.isnat(lengths), np.nan, lengths.astype("float64"))
    return xr.DataArray(
        nanoseconds / per_unit[units],
        dims=TIME_DIM,
        coords={TIME_DIM: array[TIME_DIM]},
        name="time_step_length",
        attrs={"units": units, "long_name": "Length of the timestep"},
    )


# ── supporting helpers ────────────────────────────────────────────────────────


def _value(operand: Any) -> Any:
    return operand if isinstance(operand, xr.DataArray) else float(operand)


def _describe(operand: Any, what: str) -> tuple[str, str, VariableKind | None]:
    """The ``(units, constituent, kind)`` an operand declares."""
    if isinstance(operand, xr.DataArray):
        units = operand.attrs.get("units")
        if not isinstance(units, str):
            raise ValueError(
                f"{what}: {operand.name!r} carries no 'units' attribute, so the result's "
                "units would be unknown. Take the operand from pySIPNET's output or a "
                "SIPNET table, or set attrs['units'] first."
            )
        validate_units(units)
        constituent = operand.attrs.get("constituent", "") or ""
        kind_value = operand.attrs.get("kind")
        kind = VariableKind(kind_value) if kind_value is not None else None
        return units, constituent, kind
    if isinstance(operand, (bool, np.bool_)) or not isinstance(operand, (int, float, np.number)):
        raise TypeError(
            f"{what}: an operand must be a DataArray with units or a plain number, not "
            f"{type(operand).__name__}."
        )
    return "1", "", None


def _tokens(units: str) -> dict[str, int]:
    exponents: dict[str, int] = {}
    for token in units.split():
        if token == "1":
            continue
        match = _TOKEN.match(token)
        if match is None:
            raise ValueError(f"cannot combine the unit token {token!r} of {units!r}.")
        symbol, exponent = match.group(1), int(match.group(2) or 1)
        exponents[symbol] = exponents.get(symbol, 0) + exponent
    return exponents


def _combine_units(ua: str, ub: str, sign: int) -> str:
    exponents = _tokens(ua)
    for symbol, exponent in _tokens(ub).items():
        exponents[symbol] = exponents.get(symbol, 0) + sign * exponent
    parts = [
        symbol if exponent == 1 else f"{symbol}{exponent}"
        for symbol, exponent in exponents.items()
        if exponent != 0
    ]
    units = " ".join(parts) if parts else "1"
    validate_units(units)
    return units


def _product_constituent(ca: str, cb: str) -> str:
    if ca and cb:
        raise ValueError(
            f"cannot multiply a quantity of {ca!r} by a quantity of {cb!r}; at most one "
            "operand of a product names a constituent."
        )
    return ca or cb


def _quotient_constituent(ca: str, cb: str) -> str:
    if not cb:
        return ca
    if ca == cb:
        return ""
    if not ca:
        raise ValueError(
            f"cannot divide a quantity with no constituent by a quantity of {cb!r}; the "
            "result would be per unit of a substance the numerator does not measure."
        )
    raise ValueError(
        f"cannot divide a quantity of {ca!r} by a quantity of {cb!r}; convert one of "
        "them with pysipnet.units.convert_dataarray_units first."
    )


def _time_power(units: str) -> int | None:
    """``1`` for a time, ``-1`` for a per-time, ``0`` for no time dimension, else ``None``."""
    dims = dict(unit_registry.Quantity(1.0, units).dimensionality)
    if not dims:
        return 0
    if set(dims) == {"[time]"} and dims["[time]"] in (1, -1):
        return int(dims["[time]"])
    return None if "[time]" in dims else 0


def _product_kind(
    ka: VariableKind | None, ua: str, kb: VariableKind | None, ub: str, op: str
) -> VariableKind | None:
    if ka is not None and kb is not None:
        raise ValueError(
            f"cannot combine two model variables of kinds {ka.value!r} and {kb.value!r} "
            f"with {op!r}; only one operand of a product or quotient may be a model "
            "variable. Combine a variable with a parameter, a constant or step_length()."
        )
    if ka is None and kb is None:
        return None
    if kb is not None and op == "/":
        raise ValueError(
            f"cannot divide by a model variable of kind {kb.value!r}; a per-step total or "
            "a pool in the denominator has no kind SIPNET names. Divide the variable by "
            "the other operand instead, or divide a total by step_length() first."
        )
    kind, other_units = (ka, ub) if ka is not None else (kb, ua)
    # How the other operand's time dimension enters the result: op "*" adds
    # its power, op "/" (variable in the numerator) subtracts it.
    power = _time_power(other_units)
    if power is None:
        raise ValueError(
            f"cannot combine a {kind.value!r} variable with {other_units!r}, whose time "
            "dimension is neither a time nor a per-time; the result's kind is undefined."
        )
    effect = power if op == "*" else -power
    if effect == 0:
        return kind
    if kind is VariableKind.TIMESTEP_TOTAL and effect == -1:
        return VariableKind.DAILY_RATE
    if kind is VariableKind.DAILY_RATE and effect == 1:
        return VariableKind.TIMESTEP_TOTAL
    raise ValueError(
        f"cannot {'multiply' if op == '*' else 'divide'} a {kind.value!r} variable "
        f"{'by' if op == '/' else 'with'} {other_units!r}: that changes what the value is "
        "over a step, and no pySIPNET kind names the result. Only a total per time (a "
        "rate) and a rate times a time (a total) are defined."
    )


def _same_description(a: Any, b: Any, what: str) -> tuple[str, str, VariableKind | None]:
    ua, ca, ka = _describe(a, what)
    ub, cb, kb = _describe(b, what)
    if ua != ub or ca != cb or ka != kb:
        raise ValueError(
            f"{what} needs operands that agree in units, constituent and kind; got "
            f"({ua!r}, {ca!r}, {ka.value if ka else None!r}) and "
            f"({ub!r}, {cb!r}, {kb.value if kb else None!r}). Convert one of them first."
        )
    return ua, ca, ka


def _name(operand: Any) -> str:
    if isinstance(operand, xr.DataArray):
        return str(operand.name) if operand.name is not None else "array"
    return repr(operand)


def _result(
    values: xr.DataArray,
    units: str,
    constituent: str,
    kind: VariableKind | None,
    a: Any,
    b: Any,
    op: str,
) -> xr.DataArray:
    attrs: dict[str, Any] = {"units": units}
    if constituent:
        attrs["constituent"] = constituent
    if kind is not None:
        attrs["kind"] = kind.value
        attrs["time_reference"] = TIME_REFERENCE_FOR_KIND[kind]
        cell_methods = CELL_METHODS_FOR_KIND[kind]
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods
        source = a if isinstance(a, xr.DataArray) and a.attrs.get("kind") else b
        if isinstance(source, xr.DataArray) and "sign_convention" in source.attrs:
            attrs["sign_convention"] = source.attrs["sign_convention"]
    attrs["long_name"] = f"{_name(a)} {op} {_name(b)}"
    attrs["derivation"] = f"{_name(a)} {op} {_name(b)}"
    result = values.copy()
    result.attrs = attrs
    if result.name is None or op in "*/+-":
        result.name = None
    return result

"""Observation-space operations shared by the likelihood and the plots.

This module exists so that temporal aggregation and observation indexing each
have exactly one implementation. It is imported by both the observation
operator H and the plotting layer, so a predictive-check figure cannot silently
disagree with what the likelihood consumed.

Provided:

* ``sipnet_time_index(year, day_of_year, hours_since_midnight, *,
  timestep_hours) -> DatetimeIndex`` -- SIPNET output and ``.clim`` drivers
  carry ``year``, ``day`` and ``time`` columns, not a datetime index. The
  ``time`` column drifts (issue #9) and is used only to identify a row's slot
  within its day, never as the timestamp.
* ``aggregate_time(field, freq, *, how=None) -> DataArray`` -- combine a
  canonical field's timesteps into coarser ones.

Planned (issue #6):

* ``obs_index(sites, variables, times) -> pd.MultiIndex`` -- the
  ``(site, variable, time)`` labeling of the flat observation vector. The same
  object must be used to flatten observations into ``y`` and to unstack EKI's
  ``(J, N)`` predictions back into canonical fields, or the two will mislabel
  relative to each other. (Note: this was originally expected to come from an
  ``index`` layer in pyEKI. That layer does not exist -- it is this module's job.)

How a method is chosen
----------------------
**Which methods mean anything is a property of the variable; which of them is
wanted is the caller's.** pySIPNET settles the first half: every model and
driver variable has a ``kind``, and :data:`RESAMPLING_METHODS_FOR_KIND` says
what may be done with it. A total over a step adds; a pool at the end of a step
does not, because adding end-of-step values counts the same stock once per
step; a rate or a step mean averages, weighted by step length, because SIPNET's
steps are not all the same length. :func:`aggregate_time` refuses a method the
kind does not admit, in pySIPNET's own words.

For the second half it supplies a default, which pySIPNET's ``resample``
deliberately does not: the one method that leaves the variable the kind it
already is (:data:`DEFAULT_METHOD_FOR_KIND`, derived from pySIPNET's
``RESAMPLED_KIND``). A total sums, a step mean or a rate means, a pool takes
its last value, a running total takes its last value. ``how=`` overrides it,
and taking the time-weighted mean of a pool -- a different quantity, and a
different kind -- is exactly what it is for.

SIPNET's ``net_ecosystem_exchange`` is a per-timestep total, so 3-hourly to
daily is a **sum**; a mean is wrong by a factor of 8 and looks entirely
plausible. That is the error the default exists to make impossible to reach by
omission.

Aggregation is a verb the caller applies, never a plotter keyword::

    plot_time_series(aggregate_time(nee, "1D"))   # yes
    plot_time_series(nee, temporal_agg="1D")      # no

That keeps a real subtlety at the call site: quantile-of-daily-mean is not
daily-mean-of-quantile, and which one is wanted is a modeling choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.dataset import assemble_time_coords
from pysipnet.resample import STEP_LENGTH_RESAMPLED
from pysipnet.resample import resample as pysipnet_resample
from pysipnet.variables import (
    CELL_METHODS_FOR_KIND,
    RESAMPLED_KIND,
    RESAMPLING_METHODS_FOR_KIND,
    TIME_REFERENCE_FOR_KIND,
    VariableKind,
    resolve_climate_variable,
    resolve_output_variable,
)

__all__ = [
    "DEFAULT_METHOD_FOR_KIND",
    "LENGTH_COORD",
    "RESAMPLING_METHODS",
    "STALE_ON_A_COARSER_STEP",
    "START_COORD",
    "TIME_DIM",
    "aggregate_time",
    "sipnet_time_index",
]

#: The dimension timesteps are combined along.
TIME_DIM = "time"

#: pySIPNET's coordinates for the interval a row covers. ``time`` is its end,
#: :data:`START_COORD` its start and :data:`LENGTH_COORD` its declared
#: duration. A field carrying them is aggregated exactly as pySIPNET's own
#: ``resample`` would; one that does not gets calendar cell edges and, for a
#: mean, equal weights.
START_COORD = "time_step_start"
LENGTH_COORD = "time_step_length"

#: The ways consecutive steps may be combined, as pySIPNET names them.
RESAMPLING_METHODS: tuple[str, ...] = ("sum", "mean", "last")


def _kind_preserving_methods() -> dict[VariableKind, str]:
    """The one method per kind that leaves a variable the kind it already is.

    Read off pySIPNET's ``RESAMPLED_KIND`` rather than written down, so a
    change there is either carried through or raises here.
    """
    defaults: dict[VariableKind, str] = {}
    for (kind, method), resulting in RESAMPLED_KIND.items():
        if resulting != kind:
            continue
        if kind in defaults:
            raise RuntimeError(
                f"pySIPNET admits both {defaults[kind]!r} and {method!r} as "
                f"kind-preserving for {kind.value!r}, so there is no one "
                "default; aggregate_time must be given an explicit rule."
            )
        defaults[kind] = method
    return defaults


#: What :func:`aggregate_time` does when ``how`` is not given.
DEFAULT_METHOD_FOR_KIND: dict[VariableKind, str] = _kind_preserving_methods()


def sipnet_time_index(
    year, day_of_year, hours_since_midnight, *, timestep_hours: float = 3.0
) -> pd.DatetimeIndex:
    """Timestamps for rows labeled the way SIPNET labels them.

    SIPNET's climate files and its output give each row a year, an integer day
    of year with 1 being January 1, and a fractional hour of the day -- the
    columns SIPNET calls ``year``, ``day`` and ``time``. This builds the
    nominal timestamp of each row as::

        year-01-01  +  (day_of_year - 1) days  +  slot * timestep_hours

    where ``slot = floor(hours_since_midnight / timestep_hours)`` is the row's
    position within its day. The hour value itself is used for nothing else.

    Parameters
    ----------
    year, day_of_year, hours_since_midnight:
        Array-likes of equal length: SIPNET's ``year``, ``day`` and ``time``
        columns under clearer names. ``year`` and ``day_of_year`` are integers
        (or floats that are whole numbers); ``hours_since_midnight`` is hours
        since midnight of that day.
    timestep_hours:
        Length of one row's timestep in hours; must be finite and divide 24
        into a whole number of steps. The default is the 3-hourly drivers. A
        daily file passes ``24.0``.

    Returns
    -------
    pandas.DatetimeIndex
        Naive timestamps, ``datetime64[ns]``, one per row, in row order. The
        index says nothing about the clock the labels are on or whether a
        label marks an interval's start or end; the caller records that, as
        :mod:`sipnet_calibration.drivers` does in the ``time`` attributes.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional or differ in length;
        ``timestep_hours`` is not a finite number dividing 24; a ``year`` or
        ``day_of_year`` is not numeric or not a whole number; a
        ``day_of_year`` is outside ``1..366``, or is 366 in a non-leap year;
        an ``hours_since_midnight`` is not finite or is outside ``[0, 24)``;
        or the resulting index is not strictly increasing. SIPNET writes rows
        in order, so the last condition is what a label drifting by one full
        step or more turns into: the row lands in the next row's slot and the
        two collide.

    Notes
    -----
    ``floor`` rather than ``round`` because the ``.clim`` ``time`` column
    drifts late by up to two hours within a year (issue #9): the row for
    nominal hour 21 is labeled ``23.00`` on 31 December, and rounding would
    put it in a ninth slot. The drift is always non-negative and always below
    one step, so ``floor`` identifies the slot in every row. SIPNET copies the
    same column verbatim into its output, which is why this lives here rather
    than in a driver-specific module.

    A drift of less than one step is invisible to a single label by design;
    :mod:`sipnet_calibration.drivers` asserts the drift model of the source
    separately, on whole files.

    The slot division is exact for the timesteps SIPNET is run at (3, 1, 0.5
    and 24 hours). A step such as ``0.1`` divides 24 but is not exactly
    representable, so a label sitting exactly on a slot boundary can floor
    into the slot below.

    Nothing about the result depends on an interval convention. The nominal
    label ``slot * timestep_hours`` is what the source wrote, and
    ``resample`` on it groups a day's rows exactly as SIPNET's own ``day``
    column does.
    """
    year = np.asarray(year)
    day = np.asarray(day_of_year)
    hours = np.asarray(hours_since_midnight, dtype=np.float64)
    if not (year.shape == day.shape == hours.shape) or year.ndim != 1:
        raise ValueError(
            "year, day_of_year and hours_since_midnight must be one-dimensional "
            f"and the same length; got shapes {year.shape}, {day.shape}, {hours.shape}"
        )

    try:
        timestep_hours = float(timestep_hours)
    except (TypeError, ValueError) as error:
        raise ValueError(f"timestep_hours must be a number, got {timestep_hours!r}") from error
    if not (np.isfinite(timestep_hours) and timestep_hours > 0):
        raise ValueError(f"timestep_hours must divide 24, got {timestep_hours!r}")
    steps_per_day = 24.0 / timestep_hours
    if abs(steps_per_day - round(steps_per_day)) > 1e-9:
        raise ValueError(f"timestep_hours must divide 24, got {timestep_hours!r}")
    steps_per_day = int(round(steps_per_day))

    year = _whole_numbers(year, name="year")
    day = _whole_numbers(day, name="day_of_year")

    if year.size == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    if np.any(day < 1) or np.any(day > 366):
        bad = day[(day < 1) | (day > 366)]
        raise ValueError(f"day_of_year must be within 1..366, found {bad[:5].tolist()}")
    is_leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    if np.any((day == 366) & ~is_leap):
        bad_years = np.unique(year[(day == 366) & ~is_leap])
        raise ValueError(f"day_of_year 366 in non-leap year(s) {bad_years[:5].tolist()}")

    if np.any(~np.isfinite(hours)) or np.any(hours < 0) or np.any(hours >= 24):
        bad = hours[~((hours >= 0) & (hours < 24))]
        raise ValueError(
            f"hours_since_midnight must lie within [0, 24), found {bad[:5].tolist()}"
        )

    slot = np.floor(hours / timestep_hours).astype(np.int64)
    # An hour just below 24 with a step that divides 24 always floors below
    # steps_per_day; the guard is against floating-point noise at the edge.
    slot = np.minimum(slot, steps_per_day - 1)

    year_start = pd.to_datetime(pd.Series(year), format="%Y").to_numpy()
    offset = (day - 1).astype("timedelta64[D]") + (
        (slot * timestep_hours * 3600.0).round().astype(np.int64).astype("timedelta64[s]")
    )
    index = pd.DatetimeIndex(year_start + offset).as_unit("ns")

    if not index.is_monotonic_increasing or index.has_duplicates:
        where = int(np.flatnonzero(np.diff(index.asi8) <= 0)[0]) + 1
        raise ValueError(
            "the timestamps are not strictly increasing: row "
            f"{where} ({index[where]}) does not follow row {where - 1} "
            f"({index[where - 1]}). Rows out of order, or a time label that "
            "drifted into the next slot."
        )
    return index


def aggregate_time(
    field: xr.DataArray, freq: str, *, how: str | None = None
) -> xr.DataArray:
    """Combine a field's timesteps into coarser ones.

    Parameters
    ----------
    field:
        A canonical field with a ``time`` dimension; any other dimensions are
        carried through untouched. A field produced by
        :func:`sipnet_calibration.fields.from_sipnet_output` or by
        :func:`sipnet_calibration.drivers.driver_fields` carries the ``kind``
        attribute this reads; so does any array taken from a pySIPNET Dataset.
    freq:
        A pandas offset alias for the coarser step: ``"1D"``, ``"7D"``,
        ``"MS"`` for calendar months, ``"YS"`` for calendar years.
    how:
        ``"sum"``, ``"mean"`` or ``"last"``. Omit it to take the method from
        the variable's kind, which is what makes a figure and a likelihood
        agree by default; pass it to ask for something else, such as the
        time-weighted mean of a pool.

    Returns
    -------
    xarray.DataArray
        The same name, dimensions and non-time coordinates, on a coarser
        ``time``. ``kind``, ``time_reference`` and ``cell_methods`` describe
        what the values now are, and a ``resampling`` attribute says how they
        were made. Cells no step falls in are dropped.

        A field carrying :data:`START_COORD` and :data:`LENGTH_COORD` keeps
        them, covering the span the cell's steps actually covered rather than
        the calendar cell: the start is the earliest step start, the ``time``
        the latest step end, and the length the sum of the declared lengths, so
        a cell the record only partly fills can be told from a full one by
        comparing the two.

        ``time`` keeps the attributes that are still true of it and loses
        :data:`STALE_ON_A_COARSER_STEP`, which describe the step it had before.

    Raises
    ------
    ValueError
        If *field* has no ``time`` dimension; if *how* is not one of
        :data:`RESAMPLING_METHODS`; if the variable's kind does not admit
        *how*, with pySIPNET's own explanation and the methods that would
        work; if *how* is omitted and the variable's kind cannot be
        determined; or if a mean is asked for on unequal steps that carry no
        :data:`LENGTH_COORD` to weight by.

    Notes
    -----
    Cells are right-closed and right-labeled, as pySIPNET's ``resample`` makes
    them, because ``time`` is the **end** of a step: a step ending at midnight
    belongs to the day that ended, so ``"1D"`` cells are ``(00:00, 24:00]`` and
    a daily record resamples to itself.

    A mean is weighted by the step length, which matters wherever the steps are
    not all equal -- SIPNET's own Niwot record alternates day and night steps
    between 0.29 and 0.63 days. A field with no :data:`LENGTH_COORD` is
    weighted equally, and refused if its steps are not equally spaced, rather
    than quietly averaging steps of different lengths.

    A cell holding a ``NaN`` is ``NaN``, for every method. Aggregating half a
    day of a gappy record into a number that looks like a whole day is how a
    gap stops being visible.
    """
    if TIME_DIM not in field.dims:
        raise ValueError(
            f"aggregate_time needs a {TIME_DIM!r} dimension; {field.name!r} has "
            f"dims {tuple(str(d) for d in field.dims)}."
        )

    _check_interval_coords_are_one_dimensional(field)
    kind = _variable_kind(field)
    method = _method_for(field, kind, how)
    weights = _step_weights(field) if method == "mean" else None

    bare = field.drop_vars(
        [str(name) for name in field.coords if _is_on_time(field, name) and name != TIME_DIM]
    )
    values = _combine(bare, freq, method, weights)
    keep = _nonempty_cells(field, freq)
    values = values.isel({TIME_DIM: keep})

    result = values.rename(field.name) if field.name is not None else values
    result = result.assign_coords(_aggregated_time_coords(field, freq, keep))
    result[TIME_DIM].attrs = _without_stale_interval_attrs(result[TIME_DIM].attrs)
    weighted = method == "mean" and LENGTH_COORD in field.coords
    result.attrs = _aggregated_attrs(field.attrs, kind, method, freq, weighted)
    return result


# ── supporting helpers ────────────────────────────────────────────────────────


def _whole_numbers(values: np.ndarray, *, name: str) -> np.ndarray:
    """*values* as ``int64``, raising if any is not a whole number."""
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    if values.dtype.kind == "f":
        if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must hold whole numbers")
        return values.astype(np.int64)
    if values.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, got booleans")
    try:
        return _whole_numbers(values.astype(np.float64), name=name)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _check_interval_coords_are_one_dimensional(field: xr.DataArray) -> None:
    """The interval coordinates must describe the whole field, not one slice of it.

    :func:`sipnet_calibration.fields.stack_sipnet_outputs` gives them a ``site``
    or ``member`` dimension when the runs it stacked ran over different time
    axes, and a coarser step then has no single span or length.
    """
    offenders = [
        name
        for name in (START_COORD, LENGTH_COORD)
        if name in field.coords and field[name].dims != (TIME_DIM,)
    ]
    if offenders:
        raise ValueError(
            f"{field.name!r} has {offenders} on dims "
            f"{[tuple(str(d) for d in field[n].dims) for n in offenders]} rather "
            f"than on {TIME_DIM!r} alone, which happens when runs on different "
            "time axes are stacked together. Select one site, or drop those "
            "coordinates to aggregate on calendar cells with equal weights."
        )


def _variable_kind(field: xr.DataArray) -> VariableKind | None:
    """The field's pySIPNET kind, from its attributes or the registries, or ``None``."""
    declared = field.attrs.get("kind")
    if declared is not None:
        try:
            return VariableKind(declared)
        except ValueError as error:
            raise ValueError(
                f"{field.name!r} declares kind={declared!r}, which is not one of "
                f"{[k.value for k in VariableKind]}."
            ) from error
    if field.name is None:
        return None
    for resolve in (resolve_output_variable, resolve_climate_variable):
        try:
            return resolve(str(field.name)).kind
        except KeyError:
            continue
    return None


def _method_for(field: xr.DataArray, kind: VariableKind | None, how: str | None) -> str:
    """The resampling method to apply, checked against what the kind admits."""
    if how is None:
        if kind is None:
            raise ValueError(
                f"{field.name!r} carries no 'kind' attribute and is not a SIPNET "
                "output or climate variable, so there is no rule to take the "
                "aggregation from. Pass how='sum', 'mean' or 'last'."
            )
        if kind not in DEFAULT_METHOD_FOR_KIND:
            raise ValueError(
                f"{field.name!r} is of kind {kind.value!r}, which no method leaves "
                "unchanged, so there is no default. Pass how= explicitly."
            )
        return DEFAULT_METHOD_FOR_KIND[kind]

    if how not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method {how!r} for {field.name!r}; choose from "
            f"{list(RESAMPLING_METHODS)}."
        )
    if kind is not None and how not in RESAMPLING_METHODS_FOR_KIND[kind]:
        _refuse(field.name, kind, how)
    return how


def _refuse(name: Any, kind: VariableKind, method: str) -> None:
    """Raise pySIPNET's own explanation of why *method* is meaningless for *kind*.

    Obtained by putting the pair to ``pysipnet.resample.resample``, on two rows
    that exist only to be refused, rather than by restating a reason that would
    then be this project's to keep in step with pySIPNET's.
    """
    label = str(name) if name is not None else "the field"
    try:
        pysipnet_resample(_refusal_probe(label, kind), "1D", how=method)
    except ValueError:
        raise
    raise ValueError(
        f"Cannot aggregate {label!r} with {method!r}: it is of kind "
        f"{kind.value!r}, which admits only "
        f"{sorted(RESAMPLING_METHODS_FOR_KIND[kind])}."
    )


def _refusal_probe(name: str, kind: VariableKind) -> xr.Dataset:
    """Two rows with pySIPNET's time layout, holding one variable of *kind*."""
    start = np.array(["2000-01-01T00:00", "2000-01-01T12:00"], dtype="datetime64[ns]")
    length = np.array([12, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    coords = assemble_time_coords(
        start=start,
        end=start + length,
        length=length,
        attributes_for=lambda _name: {},
        length_source="a probe, to obtain pySIPNET's refusal",
    )
    return xr.Dataset({name: (TIME_DIM, np.zeros(2), {"kind": kind.value})}, coords=coords)


def _is_on_time(field: xr.DataArray, name: Any) -> bool:
    """Whether the coordinate *name* varies along ``time``."""
    return TIME_DIM in field[name].dims


def _step_mask(field: xr.DataArray) -> np.ndarray:
    """Which timestamps are steps of this field rather than alignment padding.

    Selecting one site out of a stack of runs on different time axes leaves the
    union of those axes, so the timestamps outside that site's own record have
    no interval and must not be counted into a cell or weighted into a mean.
    """
    if START_COORD not in field.coords:
        return np.ones(field.sizes[TIME_DIM], dtype=bool)
    return ~np.isnat(field[START_COORD].values)


def _step_days(field: xr.DataArray) -> np.ndarray:
    """Declared step lengths in days, zero where there is no step."""
    nanoseconds = field[LENGTH_COORD].values.astype("timedelta64[ns]").astype("int64")
    # A NaT casts to the int64 sentinel, which would be a wildly negative
    # length rather than a missing one.
    return np.where(_step_mask(field), nanoseconds / 86_400e9, 0.0)


def _step_weights(field: xr.DataArray) -> xr.DataArray:
    """Step lengths in days, for a length-weighted mean."""
    if LENGTH_COORD in field.coords:
        return _on_time(field, _step_days(field))

    spacing = np.diff(field[TIME_DIM].values.astype("datetime64[ns]").astype("int64"))
    if spacing.size and (spacing != spacing[0]).any():
        raise ValueError(
            f"{field.name!r} has no {LENGTH_COORD!r} coordinate and its steps are "
            "not all the same length, so a mean over them has no defined "
            "weighting. Attach the step lengths, or aggregate a field that "
            "carries them."
        )
    return xr.DataArray(
        np.ones(field.sizes[TIME_DIM]), dims=TIME_DIM, coords={TIME_DIM: field[TIME_DIM]}
    )


def _grouped(obj: xr.DataArray, freq: str) -> Any:
    """Cells of *freq*, right-closed and right-labeled because ``time`` is a step end."""
    return obj.resample({TIME_DIM: freq}, closed="right", label="right")


def _combine(
    bare: xr.DataArray, freq: str, method: str, weights: xr.DataArray | None
) -> xr.DataArray:
    """*bare* reduced onto cells of *freq* by *method*."""
    if method == "sum":
        return _grouped(bare, freq).sum(skipna=False)
    if method == "last":
        return _grouped(bare, freq).last(skipna=False)
    assert weights is not None
    return _grouped(bare * weights, freq).sum(skipna=False) / _grouped(weights, freq).sum()


def _nonempty_cells(field: xr.DataArray, freq: str) -> np.ndarray:
    """Which cells of *freq* at least one step falls in.

    Resampling produces a cell for every period the record spans, including the
    ones no step lands in; an empty cell sums to zero, which is not a
    measurement.
    """
    steps = _on_time(field, _step_mask(field).astype(float))
    return np.asarray(_grouped(steps, freq).sum().values > 0)


def _aggregated_time_coords(
    field: xr.DataArray, freq: str, keep: np.ndarray
) -> dict[str, xr.DataArray]:
    """The ``time`` coordinates of the coarser field.

    A field carrying pySIPNET's interval coordinates gets them back, describing
    the span its steps covered and built by pySIPNET's own
    ``assemble_time_coords`` so the wording cannot drift. One that does not
    keeps its own ``time`` attributes on the calendar cell edges. Either way
    the caller drops :data:`STALE_ON_A_COARSER_STEP`.
    """
    if START_COORD not in field.coords or LENGTH_COORD not in field.coords:
        return {}

    mask = _step_mask(field)
    not_a_time = np.datetime64("NaT")
    # skipna, because a cell holding a padded timestamp beside real steps would
    # otherwise take NaT for its whole span; xarray does not skip it by default
    # on a datetime reduction.
    start = _grouped(_on_time(field, field[START_COORD].values), freq).min(skipna=True)
    end = _grouped(
        _on_time(field, np.where(mask, field[TIME_DIM].values, not_a_time)), freq
    ).max(skipna=True)
    length = _grouped(_on_time(field, _step_days(field)), freq).sum()

    built = assemble_time_coords(
        start=start.values[keep].astype("datetime64[ns]"),
        end=end.values[keep].astype("datetime64[ns]"),
        length=np.rint(length.values[keep] * 86_400e9)
        .astype("int64")
        .view("timedelta64[ns]"),
        attributes_for=lambda name: dict(field[name].attrs) if name in field.coords else {},
        length_source=STEP_LENGTH_RESAMPLED,
    )
    return {
        name: xr.DataArray(values, dims=TIME_DIM, attrs=_without_stale_interval_attrs(attrs))
        for name, (_dims, values, attrs) in built.items()
        if name in (TIME_DIM, START_COORD, LENGTH_COORD)
    }


#: ``time`` attributes that describe the *source's* step and are false of a
#: coarser one. ``bounds`` names pySIPNET's two-dimensional ``time_bounds``
#: variable, which a field cannot carry (see
#: :mod:`sipnet_calibration.fields`); ``time_label_note`` is where
#: :mod:`sipnet_calibration.drivers` writes the width of one driver interval.
#: What a label marks is unchanged -- cells are labeled at their end, as the
#: steps were -- so ``time_label`` itself stays.
STALE_ON_A_COARSER_STEP: tuple[str, ...] = ("bounds", "time_label_note")


def _without_stale_interval_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """*attrs* less the ones that describe the interval before it was coarsened."""
    return {key: value for key, value in attrs.items() if key not in STALE_ON_A_COARSER_STEP}


def _on_time(field: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    """*values*, one per timestep, as a resamplable array on the field's ``time``."""
    return xr.DataArray(values, dims=TIME_DIM, coords={TIME_DIM: field[TIME_DIM]})


def _aggregated_attrs(
    attrs: Mapping[str, Any],
    kind: VariableKind | None,
    method: str,
    freq: str,
    weighted: bool,
) -> dict[str, Any]:
    """The variable's attributes, rewritten to describe what the values now are."""
    out = dict(attrs)
    if kind is not None:
        new_kind = RESAMPLED_KIND[(kind, method)]
        out["kind"] = new_kind.value
        out["time_reference"] = TIME_REFERENCE_FOR_KIND[new_kind]
        cell_methods = CELL_METHODS_FOR_KIND[new_kind]
        if cell_methods is None:
            out.pop("cell_methods", None)
        else:
            out["cell_methods"] = cell_methods
    weighting = f", weighted by {LENGTH_COORD}" if weighted else ""
    of_kind = f" of {kind.value} values" if kind is not None else ""
    out["resampling"] = f"{method}{of_kind} over {freq}{weighting}"
    # The source file's printf precision no longer describes a combined value.
    out.pop("output_decimals", None)
    return out

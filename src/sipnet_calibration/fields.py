"""The array form that the plotting and inference layers read.

Data reaches this project in as many shapes as it has sources. SIPNET writes
columnar output, the meteorological drivers are text files, the initial
conditions are per-site netCDF, the observations are a csv and a set of R
objects, and the calibration returns flat blocks that carry no record of space
or time. This module defines the single form all of them are converted into,
and holds the adapters that do the converting. Everything downstream -- the
plots and the observation operator -- reads that form and nothing else, so
adding a source costs one adapter rather than a change in every consumer.

Where this sits
---------------
The model-output adapters here read what pySIPNET produces, in memory; no
script writes a processed file for them::

    SIPNETRunner / SIPNETModel  ->  SIPNETResult.outputs (SIPNETOutput)

    one run    -> from_sipnet_output()   -> dict[str, DataArray] on (time,)
               -> label_run()            -> Dataset on (time,), labeled
    many runs  -> stack_sipnet_outputs() -> dict[str, DataArray]
                                              on (member, site, time)
               -> stack_model_outputs()  -> Dataset on (member, site, time)

with ``data/processed/sites/sites.csv`` joined on for ``lon``/``lat``, read
through :func:`sipnet_calibration.sites.load_sites`. The dependency runs one
way: this module reads pySIPNET and the site table, and nothing in either
knows about it.

What it reads
-------------
:class:`pysipnet.result.SIPNETResult` or :class:`pysipnet.output.SIPNETOutput`
    One SIPNET run. Only the variables asked for are read from it, so a
    file-backed output holds one column per variable rather than the whole
    frame.
:func:`sipnet_calibration.sites.load_sites`
    The site table, for the ``lon``/``lat`` of a site id. Read only when a
    ``site`` label is given.

The form
--------
A **field** is an ``xarray.DataArray`` holding one variable, with

* dimensions drawn from ``member``, ``site`` and ``time``, in any combination;
* ``lon`` and ``lat`` as non-dimension coordinates on ``site``, whenever
  ``site`` is a dimension;
* ``units`` and ``long_name`` in ``attrs``;
* a ``name`` that is the variable's processed name.

Which dimensions are present depends on the quantity. A single deterministic
run is ``(time,)``, an initial condition ensemble is ``(member, site)``, and
an ensemble of runs over sites is ``(member, site, time)``. Calibration
parameters are ``(member, site)`` for an ensemble and ``(site,)`` for one
value: :meth:`sipnet_calibration.parameter_vector.ParameterVector.fields`
returns one per scalar component as a ``Dataset``, the one exception to the
rule below, since every one of them shares those dims. Their names are
``<parameter>`` or ``<parameter>.<component>``, the calibration vector's own,
not registry names.

One array holds one variable, and variables are not combined into a
``Dataset``: they do not share a time axis, model output being on SIPNET's
steps, the constraints annual, dated or static by product, and the initial
conditions static. A group
of variables is a ``dict[str, DataArray]``, which is what the multi-variable
adapters return and what the readers in
:mod:`sipnet_calibration.drivers` and :mod:`sipnet_calibration.constraints`
already produce.

Identifiers
-----------
``site``
    The handed-down integer site id, 1 to 8000, as ``int32``. It is a shared
    key with collaborators' files and is never renumbered; a spatially
    meaningful ordering, where one is wanted, is added as a separate
    coordinate. Only 185 of the sites are Ameriflux sites, so an
    Ameriflux-keyed identifier cannot address the pool: ``ameriflux_site_id``
    is a non-dimension coordinate on ``site``, missing for the rest. Plant
    functional type is not site metadata and is not carried here; which site
    labels to use is an experimental choice, and they live in their own product
    under ``data/processed/site_labels/``.
``member``
    A 0-based ensemble index as ``int16``, meaningful only within the source
    it came from. Whether member *i* of one source corresponds to member *i*
    of another is not established, and xarray aligns on the integer label
    without complaint, so any arithmetic across two sources needs that settled
    first.
``time``
    Timestamps, whose meaning is the source's and is recorded in the
    coordinate's attributes rather than assumed. Model output and the drivers
    carry pySIPNET's axis, the end of each step (see below); each constraint
    product carries its source's own label, with CF ``time_bounds`` where the
    support is documented.

Model output
------------
:func:`from_sipnet_output` is a thin wrapper over ``SIPNETOutput.select``. It
adds the identifiers above and nothing else: **every name, unit, kind and
description is pySIPNET's**, carried through as attributes and never restated
here. pySIPNET's registry names are already
``lower_case_with_underscores`` (``net_ecosystem_exchange``, ``wood_carbon``),
so they are the processed names; aliases (``"nee"``) are accepted on the way
in and resolved to them.

Each field keeps three of pySIPNET's time coordinates, :data:`TIME_COORDS`,
the same three a driver field from
:func:`sipnet_calibration.drivers.driver_fields` keeps -- a run's output and
its drivers are on one axis:

===================== ===================================================
``time``              end of the timestep, the CF upper bound
``time_step_start``   start of the timestep, the CF lower bound
``time_step_length``  its duration, as ``timedelta64``
===================== ===================================================

so the interval a value covers is ``[time_step_start, time]``. That is the
pair pySIPNET writes as its CF ``time_bounds`` variable, which a DataArray
cannot carry: ``time_bounds`` is two-dimensional on ``(time, bounds)`` and
``bounds`` is not a field dimension. ``time``'s ``bounds`` attribute is
dropped for the same reason, rather than left pointing at a variable that is
not there. pySIPNET's ``year``/``day_of_year``/``hour_of_day`` row labels are
dropped too; ``time_step_start`` is the same instant.

:func:`sipnet_calibration.observation.time_alignment.aggregate_time` needs
``time_step_length`` for a length-weighted mean, which is why it is kept
rather than recomputed.

Functions
---------
:func:`from_sipnet_output`
    One run's chosen variables as fields, optionally labeled with a
    site and a member.
:func:`label_run`
    One run's output ``Dataset`` labeled with its site and member: the
    ``model_output`` the observation operators read.
:func:`stack_sipnet_outputs`
    Many runs, each labeled ``(site, member)``, stacked into
    ``(member, site, time)`` fields.
:func:`stack_model_outputs`
    Many runs' output ``Dataset`` objects, keyed by ``(site, member)``,
    stacked into one ``Dataset`` on ``(member, site, time)``.
:func:`resolve_output_variable_names`
    Requested output variable names as pySIPNET registry names, in order,
    without repeats.
:func:`field_label`
    How a field is called in a message: its name, or else its derivation.
:func:`site_lookup`
    The site table keyed on ``site_id``, for a caller adapting run after run.
``validate_field``
    Not written yet (issue #6).

The drivers, the constraints and the initial conditions have readers of their
own that already produce the form above
(:func:`sipnet_calibration.drivers.driver_fields`,
:func:`sipnet_calibration.constraints.constraint_fields`,
:func:`sipnet_calibration.initial_conditions.initial_condition_fields`), and a
block of predictions is unstacked by ``ObservationVector.fields``.

Notes
-----
**Why no wrapper class.** The three operations this project performs on a
field are ``.quantile(dim="member")``, ``.resample(time=...)`` and
``.sel(site=...)``, all of which xarray already has. A class would spend the
project re-exporting them, and every plotter would have to unwrap it. The
convention plus a validator is the whole design.

**Why the adapter selects.** ``SIPNETOutput.xarray`` reads and caches every
column SIPNET wrote. Across an ensemble that is every member's full output held
at once, where one variable per member is what the caller asked for, so the
adapter goes through ``select`` and never touches ``.xarray`` or ``.pandas``.

**Why a missing run is not an error.** The caller supplies the mapping, so it
already knows which ``(site, member)`` pairs it left out; those cells read
``NaN``. This differs from :func:`sipnet_calibration.drivers.load_drivers`,
which discovers absence on disk and therefore has to report it.

A ``(J, N)`` block of predictions is unstacked by
:meth:`sipnet_calibration.observation.ObservationVector.fields`, which owns the
``(site, product, time)`` index the block was flattened with, so the two
cannot mislabel against each other. The traps of the observation and
initial-condition sources are in ``CLAUDE.md``'s Data section, where they
apply to the readers that already exist as well.

Usage
-----
One run, no site pool involved::

    from sipnet_calibration.fields import from_sipnet_output

    fields = from_sipnet_output(result, ["nee", "wood_carbon"])
    fields["net_ecosystem_exchange"].dims          # ('time',)
    fields["net_ecosystem_exchange"].attrs["kind"] # 'timestep_total'

An ensemble over sites and members, keyed by the pair each run stands for::

    from sipnet_calibration.fields import stack_sipnet_outputs
    from sipnet_calibration.observation.time_alignment import aggregate_time
    from sipnet_calibration.plotting import plot_time_series

    runs = {(1, 0): first, (1, 1): second, (27, 0): third, (27, 1): fourth}
    nee = stack_sipnet_outputs(runs, "nee")["net_ecosystem_exchange"]
    nee.dims                                       # ('member', 'site', 'time')

    plot_time_series(aggregate_time(nee, "1D").sel(site=1))

Adapting run after run, with the site table read once::

    from sipnet_calibration.fields import site_lookup
    from sipnet_calibration.sites import load_sites

    table = site_lookup(load_sites())
    for site, member, run in ensemble:
        fields = from_sipnet_output(run, "nee", site=site, member=member, sites=table)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pysipnet.dataset import TIME_DIMENSION

from sipnet_calibration.sites import load_sites

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pysipnet.output import SIPNETOutput
    from pysipnet.result import SIPNETResult

__all__ = [
    "FIELD_DIMS",
    "MEMBER_DIM",
    "SITE_DIM",
    "TIME_COORDS",
    "TIME_DIM",
    "field_label",
    "from_sipnet_output",
    "label_run",
    "resolve_output_variable_names",
    "site_lookup",
    "stack_model_outputs",
    "stack_sipnet_outputs",
]

MEMBER_DIM = "member"
SITE_DIM = "site"
TIME_DIM = TIME_DIMENSION

#: The dimensions a field may have, in the order they are written.
FIELD_DIMS: tuple[str, ...] = (MEMBER_DIM, SITE_DIM, TIME_DIM)

#: pySIPNET's time coordinates, which a model field and a driver field both
#: keep. ``time`` is the end of the step and ``time_step_start`` its start, so
#: the two are the CF bounds pair.
TIME_COORDS: tuple[str, ...] = (TIME_DIM, "time_step_start", "time_step_length")

_LON_ATTRS = {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}
_LAT_ATTRS = {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}
_SITE_ATTRS = {
    "long_name": "Model site identifier",
    "comment": "The handed-down 1-8000 identifier; never renumbered.",
}
_MEMBER_ATTRS = {
    "long_name": "Ensemble member",
    "comment": "0-based, meaningful only within this source.",
}

#: The scalar coordinates :func:`label_run` adds.
_IDENTITY_COORDS: tuple[str, ...] = (SITE_DIM, MEMBER_DIM, "lon", "lat")


def site_lookup(sites: pd.DataFrame) -> pd.DataFrame:
    """The site table keyed on ``site_id``, so looking a site up is not a scan.

    :func:`sipnet_calibration.sites.load_sites` returns a table, not a lookup,
    and one adapter call per run over the whole pool would otherwise search
    8000 rows every time. Idempotent, so passing the result back in costs
    nothing; ``site_id`` stays a column as well as the index.
    """
    if sites.index.name == "site_id":
        return sites
    return sites.set_index("site_id", drop=False)


def label_run(
    dataset: xr.Dataset,
    *,
    site: int | None = None,
    member: int | None = None,
    site_table: pd.DataFrame | None = None,
) -> xr.Dataset:
    """A run's output dataset labeled with the site and member it was.

    pySIPNET's ``SIPNETOutput.select(names)`` gives one run's variables as a
    CF Dataset on ``(time,)``; what it cannot know is which site of the pool
    and which ensemble member the run was. This adds those as scalar
    coordinates, ``site`` with its ``lon``/``lat`` from the site table and
    ``member``, and changes nothing else: every variable, coordinate
    (``time_bounds`` included) and attribute is pySIPNET's. The result is the
    ``model_output`` the observation operators read.

    Parameters
    ----------
    dataset:
        The run's variables, from ``result.outputs.select(names)`` or
        ``result.outputs[[...]]``.
    site, member:
        As for :func:`from_sipnet_output`.
    site_table:
        The site table, as :func:`sipnet_calibration.sites.load_sites` returns
        it, read from disk when omitted and a *site* is given; pass it when
        labeling many runs so it is read once.

    Returns
    -------
    xarray.Dataset
        *dataset* with a scalar ``site`` (``int32``) and its scalar
        ``lon``/``lat`` when *site* is given, and a scalar ``member``
        (``int16``) when *member* is given; *dataset* itself when neither is.

    Raises
    ------
    TypeError
        If *dataset* is not an ``xr.Dataset``.
    ValueError
        If *dataset* has no ``time`` rows, which is what a failed run leaves;
        if *site* or *member* is not a whole number in range; or if the site
        table lists a site twice.
    KeyError
        If *site* is not in the site table.
    FileNotFoundError
        If *site* is given, *site_table* is not, and the site table is absent.
    """
    check_is_a_dataset(dataset)
    check_run_has_rows(dataset)
    labels = _identity_coords(site=site, member=member, sites=site_table)
    return dataset.assign_coords(labels) if labels else dataset


def from_sipnet_output(
    output: SIPNETResult | SIPNETOutput,
    output_variable_names: str | Sequence[str],
    *,
    site: int | None = None,
    member: int | None = None,
    sites: pd.DataFrame | None = None,
) -> dict[str, xr.DataArray]:
    """The named variables of one SIPNET run, as fields.

    Parameters
    ----------
    output:
        A :class:`pysipnet.result.SIPNETResult` or the
        :class:`pysipnet.output.SIPNETOutput` inside one.
    output_variable_names:
        One name, or a sequence of them. pySIPNET's names and aliases are both
        accepted (``"nee"``, ``"NEE"`` and ``"net_ecosystem_exchange"`` are the
        same variable); the keys of the result are always the registry name.
        Repeats are dropped and the requested order is kept.
    site:
        The 1-8000 site identifier this run is for, or ``None`` when the run is
        not a site of the pool. When given, it becomes a scalar ``site``
        coordinate and ``lon``/``lat`` are looked up beside it.
    member:
        The 0-based ensemble index this run is, or ``None``. When given, it
        becomes a scalar ``member`` coordinate.
    sites:
        The site table to look ``site`` up in, as
        :func:`sipnet_calibration.sites.load_sites` returns it. Read from disk
        when omitted and a *site* is given; pass it when adapting many runs so
        the table is read once.

    Returns
    -------
    dict
        Keyed by pySIPNET registry name, in the order requested. Each value is
        a ``DataArray`` with dimension ``time``, the ``time`` coordinates of
        :data:`TIME_COORDS`, scalar ``site``/``lon``/``lat`` and
        ``member`` coordinates for the labels that were given, and pySIPNET's
        variable attributes unchanged.

    Raises
    ------
    KeyError
        If a variable is not a SIPNET output variable or alias, or if *site* is
        not in the site table.
    ValueError
        If *output_variable_names* is empty, unordered, or not a sequence of names; if the
        run wrote no rows, which is what a failed run leaves; or if *member* or
        *site* is not a whole number in range.
    TypeError
        If *output* is neither a ``SIPNETResult`` nor a ``SIPNETOutput``.
    FileNotFoundError
        If *site* is given, *sites* is not, and the site table is absent.

    Notes
    -----
    Only the columns named are read from a file-backed output; see this
    module's Notes for why ``.xarray`` and ``.pandas`` are never touched.
    """
    source = _output_of(output)
    names = resolve_output_variable_names(output_variable_names)
    dataset = label_run(source.select(names), site=site, member=member, site_table=sites)
    dataset = _with_field_coords(dataset)
    return {name: dataset[name] for name in names}


def stack_sipnet_outputs(
    runs: Mapping[tuple[int, int], SIPNETResult | SIPNETOutput],
    output_variable_names: str | Sequence[str],
    *,
    sites: pd.DataFrame | None = None,
) -> dict[str, xr.DataArray]:
    """Many SIPNET runs, labeled by site and member, as ``(member, site, time)`` fields.

    Parameters
    ----------
    runs:
        A mapping from ``(site, member)`` to the run for that pair.
        ``site`` is the 1-8000 identifier and ``member`` the 0-based ensemble
        index. The pairs need not form a full rectangle; a pair left out reads
        as ``NaN``.
    output_variable_names:
        As for :func:`from_sipnet_output`.
    sites:
        The site table. Read once from disk when omitted.

    Returns
    -------
    dict
        Keyed by pySIPNET registry name, in the order requested. Each value is
        a ``DataArray`` with dims ``(member, site, time)``, ascending in
        ``member`` and ``site``, with ``lon``/``lat`` on ``site`` and the
        ``time`` coordinates of :data:`TIME_COORDS`.

    Raises
    ------
    TypeError
        If *runs* is not a mapping.
    ValueError
        If *runs* is empty, or a key is not a pair of integers.
    KeyError
        If a variable or a site identifier is unknown.

    Notes
    -----
    Runs whose time axes differ are aligned by an outer join, so a site
    covering a shorter record is ``NaN`` outside it. Where every run shares one
    axis -- the usual case, one driver period across the site pool --
    ``time_step_start`` and ``time_step_length`` stay one-dimensional on
    ``time``; where they do not, xarray gives them the dimensions over which
    they differ.

    Each run is read and reduced to the variables asked for before the next is
    touched, so what is held is one column per run and variable, never a run's
    whole frame.
    """
    check_runs_are_a_nonempty_mapping(runs)
    names = resolve_output_variable_names(output_variable_names)
    table = site_lookup(sites if sites is not None else load_sites())
    datasets = {_run_key(key): _output_of(run).select(names) for key, run in runs.items()}
    stacked = stack_model_outputs(datasets, site_table=table)
    return {name: stacked[name] for name in names}


def stack_model_outputs(
    runs: Mapping[tuple[int, int], xr.Dataset],
    *,
    site_table: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Many runs' output Datasets, keyed by site and member, as one Dataset.

    Parameters
    ----------
    runs:
        A mapping from ``(site, member)`` to that run's output ``Dataset``,
        from ``result.outputs.select(names)`` or :func:`label_run`, every run
        carrying the same variables. ``site`` is the 1-8000 identifier and
        ``member`` the 0-based ensemble index. A run already labeled by
        :func:`label_run` must carry the labels of its key. The pairs need
        not form a full rectangle; a pair left out reads as ``NaN``.
    site_table:
        The site table, for ``lon``/``lat``. Read once from disk when omitted.

    Returns
    -------
    xarray.Dataset
        The runs' variables on ``(member, site, time)``, ascending in
        ``member`` (``int16``) and ``site`` (``int32``), with ``lon``/``lat``
        (``float64``, CF attributes) on ``site`` and the ``time`` coordinates
        of :data:`TIME_COORDS`. The variables' and the first run's dataset
        attributes are pySIPNET's. Every other coordinate -- ``time_bounds``,
        which a field cannot carry, and SIPNET's
        ``year``/``day_of_year``/``hour_of_day`` row labels -- is dropped, as
        is the ``bounds`` attribute of ``time``, so each variable of the
        result is a field.

    Raises
    ------
    TypeError
        If *runs* is not a mapping, or a value is not an ``xr.Dataset``.
    ValueError
        If *runs* is empty; if a key is not a pair of integers in range; if a
        run has no ``time`` rows; or if a run's own ``site`` or ``member``
        label disagrees with its key.
    KeyError
        If a site identifier is not in the site table.

    Notes
    -----
    Runs whose time axes differ are aligned by an outer join, so a site
    covering a shorter record is ``NaN`` outside it. Where every run shares one
    axis ``time_step_start`` and ``time_step_length`` stay one-dimensional on
    ``time``; where they do not, xarray gives them the dimensions over which
    they differ.
    """
    check_runs_are_a_nonempty_mapping(runs)
    table = site_lookup(site_table if site_table is not None else load_sites())
    by_member: dict[int, list[xr.Dataset]] = {}
    for key in sorted(runs, key=_run_key):
        site_id, member_id = _run_key(key)
        dataset = runs[key]
        check_is_a_dataset(dataset)
        check_run_has_rows(dataset)
        check_run_labels_match_the_key(dataset, site_id, member_id)
        labeled = _with_field_coords(dataset.drop_vars(_IDENTITY_COORDS, errors="ignore"))
        labeled = labeled.assign_coords(
            {
                SITE_DIM: xr.DataArray(np.int32(site_id), attrs=dict(_SITE_ATTRS)),
                MEMBER_DIM: xr.DataArray(np.int16(member_id), attrs=dict(_MEMBER_ATTRS)),
            }
        )
        by_member.setdefault(member_id, []).append(labeled)
    stacked = _concat(
        [_concat(per_site, SITE_DIM) for _, per_site in sorted(by_member.items())], MEMBER_DIM
    )
    stacked = stacked.transpose(*FIELD_DIMS, ...)
    return stacked.assign_coords(_site_locations(stacked[SITE_DIM].values, table))


def resolve_output_variable_names(output_variable_names: str | Sequence[str]) -> list[str]:
    """Requested output variable names as pySIPNET registry names.

    Parameters
    ----------
    output_variable_names:
        One name, or an ordered sequence of them. pySIPNET's registry names,
        aliases and SIPNET's own column names are all accepted (``"nee"``,
        ``"NEE"`` and ``"net_ecosystem_exchange"`` are the same variable).

    Returns
    -------
    list of str
        The registry names, in the order requested, each once.

    Raises
    ------
    ValueError
        If *output_variable_names* is empty, is a set (which has no order to
        keep), or is neither a name nor an iterable of names.
    TypeError
        If an item is not a string.
    KeyError
        If a name is not a pySIPNET output variable or alias.
    """
    from pysipnet.variables import resolve_output_variable

    if isinstance(output_variable_names, str):
        requested = [output_variable_names]
    elif isinstance(output_variable_names, (set, frozenset)):
        raise ValueError(
            f"output_variable_names was given as a {type(output_variable_names).__name__}, "
            "which has no order to keep. Pass a list or a tuple."
        )
    else:
        try:
            requested = list(output_variable_names)
        except TypeError:
            raise ValueError(
                "output_variable_names must be a name or a sequence of names, got "
                f"{output_variable_names!r}."
            ) from None
    if not requested:
        raise ValueError(
            "No variables were asked for. Name at least one SIPNET output "
            "variable, e.g. output_variable_names=['nee']."
        )
    names: list[str] = []
    for item in requested:
        if not isinstance(item, str):
            raise TypeError(f"Variable names must be strings, got {item!r}.")
        name = resolve_output_variable(item).name
        if name not in names:
            names.append(name)
    return names


def field_label(field: xr.DataArray, default: str = "the field") -> str:
    """How a field is called in a message: its name, or else its derivation.

    Parameters
    ----------
    field:
        The field to name.
    default:
        What to call it when it has neither a name nor a ``derivation``.

    Returns
    -------
    str
        ``repr`` of the field's name; else ``repr`` of its ``derivation``
        attribute, which a result of :mod:`pysipnet.arithmetic` carries in
        place of a name; else *default*, as given.
    """
    if field.name is not None:
        return repr(field.name)
    derivation = field.attrs.get("derivation")
    return repr(derivation) if derivation else default


# ── supporting helpers ────────────────────────────────────────────────────────


def _output_of(output: SIPNETResult | SIPNETOutput) -> SIPNETOutput:
    """The :class:`SIPNETOutput` of a run, given either it or the result holding it."""
    resolved = getattr(output, "outputs", output)
    if hasattr(resolved, "select"):
        return resolved
    if resolved is output:
        raise TypeError(
            "Expected a pysipnet SIPNETResult or SIPNETOutput, got "
            f"{type(output).__name__}, which has neither .outputs nor .select."
        )
    raise TypeError(
        f"Expected a pysipnet SIPNETResult, got {type(output).__name__} whose "
        f".outputs is {type(resolved).__name__} rather than a SIPNETOutput."
    )


def _identity_coords(
    *, site: int | None, member: int | None, sites: pd.DataFrame | None
) -> dict[str, xr.DataArray]:
    """Scalar ``site``/``lon``/``lat`` and ``member`` coordinates for the labels given."""
    coords: dict[str, xr.DataArray] = {}
    if member is not None:
        coords[MEMBER_DIM] = xr.DataArray(
            _bounded_integer(member, name="member", dtype=np.int16, minimum=0),
            attrs=dict(_MEMBER_ATTRS),
        )
    if site is not None:
        site_id = _bounded_integer(site, name="site", dtype=np.int32, minimum=1)
        lon, lat = _site_location(int(site_id), sites)
        coords[SITE_DIM] = xr.DataArray(site_id, attrs=dict(_SITE_ATTRS))
        coords["lon"] = xr.DataArray(lon, attrs=dict(_LON_ATTRS))
        coords["lat"] = xr.DataArray(lat, attrs=dict(_LAT_ATTRS))
    return coords


def _bounded_integer(value: Any, *, name: str, dtype: type, minimum: int) -> Any:
    """*value* as *dtype*, raising if it is not a whole number at least *minimum*."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got the boolean {value!r}.")
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from error
    if as_int != value:
        raise ValueError(f"{name} must be a whole number, got {value!r}.")
    if as_int < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {as_int}.")
    info = np.iinfo(dtype)
    if as_int > info.max:
        raise ValueError(f"{name} {as_int} does not fit in {dtype.__name__}.")
    return dtype(as_int)


def _site_locations(site_ids: np.ndarray, sites: pd.DataFrame) -> dict[str, xr.DataArray]:
    """``lon``/``lat`` on the ``site`` dimension, in *site_ids* order.

    Assigned after stacking rather than left to :func:`xarray.concat`, which
    promotes a scalar coordinate to the concatenated dimension only when the
    values it is given differ -- so a one-site or one-member stack would
    otherwise keep ``lon``/``lat`` scalar and break the convention.
    """
    table = site_lookup(sites)
    located = [_site_location(int(site_id), table) for site_id in site_ids]
    lon = np.asarray([value[0] for value in located], dtype=np.float64)
    lat = np.asarray([value[1] for value in located], dtype=np.float64)
    return {
        "lon": xr.DataArray(lon, dims=SITE_DIM, attrs=dict(_LON_ATTRS)),
        "lat": xr.DataArray(lat, dims=SITE_DIM, attrs=dict(_LAT_ATTRS)),
    }


def _site_location(site: int, sites: pd.DataFrame | None) -> tuple[np.float64, np.float64]:
    """The ``lon``/``lat`` of *site* in the site table, read from disk if not supplied."""
    table = site_lookup(sites if sites is not None else load_sites())
    if table.index.has_duplicates:
        repeated = sorted(set(table.index[table.index.duplicated()].tolist()))
        raise ValueError(
            f"The site table lists site(s) {repeated[:10]} more than once; a site has one "
            "row. load_sites() never produces this."
        )
    try:
        row = table.loc[site]
    except KeyError:
        ids = table.index
        raise KeyError(
            f"Site {site} is not in the site table, which holds "
            f"{len(table)} sites from {int(ids.min())} to {int(ids.max())}. "
            "Site identifiers are the handed-down 1-8000 ids and are never "
            "renumbered."
        ) from None
    return np.float64(row["lon"]), np.float64(row["lat"])


def _run_key(key: Any) -> tuple[int, int]:
    """*key* as a ``(site, member)`` pair of plain integers."""
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(
            f"Every key of runs must be a (site, member) pair, got {key!r}."
        )
    site, member = key
    return (
        int(_bounded_integer(site, name="site", dtype=np.int32, minimum=1)),
        int(_bounded_integer(member, name="member", dtype=np.int16, minimum=0)),
    )


def _with_field_coords(dataset: xr.Dataset) -> xr.Dataset:
    """*dataset* with only the coordinates a field keeps.

    Those are :data:`TIME_COORDS` and the identity coordinates. ``time``'s
    ``bounds`` attribute goes too, since the ``time_bounds`` variable it names
    is one of the coordinates dropped.
    """
    keep = {*TIME_COORDS, *_IDENTITY_COORDS}
    # The copy gives this dataset its own variables, so rewriting an attribute
    # below leaves the caller's dataset untouched.
    dataset = dataset.drop_vars([str(c) for c in dataset.coords if str(c) not in keep]).copy()
    # A DataArray cannot carry time_bounds -- its 'bounds' dimension is not a
    # field dimension -- so the attribute naming it would dangle.
    dataset[TIME_DIM].attrs = {
        key: value for key, value in dataset[TIME_DIM].attrs.items() if key != "bounds"
    }
    return dataset


def _concat(objects: list[Any], dim: str) -> Any:
    """Concatenate along *dim*, promoting the scalar *dim* coordinate each carries.

    ``coords="different"`` is what gives the interval coordinates a ``site`` or
    ``member`` dimension when the runs disagree about them, rather than
    refusing; it and ``data_vars`` are passed explicitly because xarray's
    defaults for them are changing.
    """
    if len(objects) == 1:
        return objects[0].expand_dims(dim)
    options: dict[str, Any] = {"data_vars": "all"} if isinstance(objects[0], xr.Dataset) else {}
    return xr.concat(
        objects,
        dim=dim,
        join="outer",
        coords="different",
        compat="equals",
        combine_attrs="override",
        **options,
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_is_a_dataset(dataset: Any) -> None:
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"expected a run's output as an xarray Dataset, from "
            f"result.outputs.select(names), got {type(dataset).__name__}."
        )


def check_run_has_rows(dataset: xr.Dataset) -> None:
    if TIME_DIM not in dataset.coords or dataset.sizes.get(TIME_DIM, 0) == 0:
        raise ValueError(
            "This SIPNET output has no rows, so there is nothing to put on a time axis. "
            "That usually means the run failed; check result.provenance.success and its "
            "stderr. Stacking an ensemble hits this on the first member that did not run."
        )


def check_runs_are_a_nonempty_mapping(runs: Any) -> None:
    if not isinstance(runs, Mapping):
        raise TypeError(
            f"runs must be a mapping from (site, member) to a run, not {type(runs).__name__}."
        )
    if not runs:
        raise ValueError("runs is empty; there is nothing to stack.")


def check_run_labels_match_the_key(dataset: xr.Dataset, site: int, member: int) -> None:
    for name, expected in ((SITE_DIM, site), (MEMBER_DIM, member)):
        if name not in dataset.coords:
            continue
        labels = np.asarray(dataset[name].values).ravel().tolist()
        if labels != [expected]:
            raise ValueError(
                f"the run keyed (site={site}, member={member}) is labeled {name}={labels}; "
                "key each run by the site and member it was."
            )

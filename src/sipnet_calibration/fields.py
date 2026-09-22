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
    many runs  -> stack_sipnet_outputs() -> dict[str, DataArray]
                                              on (member, site, time)

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
A **canonical field** is an ``xarray.DataArray`` holding one variable, with

* dimensions drawn from ``member``, ``site`` and ``time``, in any combination;
* ``lon`` and ``lat`` as non-dimension coordinates on ``site``, whenever
  ``site`` is a dimension;
* ``units`` and ``long_name`` in ``attrs``;
* a ``name`` that is the variable's processed name.

Which dimensions are present depends on the quantity. A single deterministic
run is ``(time,)``, an initial condition ensemble is ``(member, site)``, the
gap-filled NEE observations are ``(member, site, time)``, and a calibrated
per-site parameter is ``(member, site)``.

One array holds one variable, and variables are not combined into a
``Dataset``: they do not share a time axis, NEE being 3-hourly, the constraints
annual, dated or static by product, and the initial conditions static. A group
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
    functional type is not site metadata and is not carried here; a labeling
    is an experimental choice and lives in its own product under
    ``data/processed/labelings/``.
``member``
    A 0-based ensemble index as ``int16``, meaningful only within the source
    it came from. Whether member *i* of one source corresponds to member *i*
    of another is not established, and xarray aligns on the integer label
    without complaint, so any arithmetic across two sources needs that settled
    first.
``time``
    Timestamps, whose meaning is the source's and is recorded in the
    coordinate's attributes rather than assumed: the drivers label the end of
    each interval, and each constraint product carries its source's own label,
    with CF ``time_bounds`` where the support is documented.

Model output
------------
:func:`from_sipnet_output` is a thin wrapper over ``SIPNETOutput.select``. It
adds the identifiers above and nothing else: **every name, unit, kind and
description is pySIPNET's**, carried through as attributes and never restated
here. pySIPNET's registry names are already
``lower_case_with_underscores`` (``net_ecosystem_exchange``, ``wood_carbon``),
so they are the processed names; aliases (``"nee"``) are accepted on the way
in and resolved to them.

Each field keeps three of pySIPNET's time coordinates:

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

:func:`sipnet_calibration.obs_ops.aggregate_time` needs
``time_step_length`` for a length-weighted mean, which is why it is kept
rather than recomputed.

Functions
---------
:func:`from_sipnet_output`
    One run's chosen variables as canonical fields, optionally labeled with a
    site and a member.
:func:`stack_sipnet_outputs`
    Many runs, each labeled ``(site, member)``, stacked into
    ``(member, site, time)`` fields.
:func:`site_lookup`
    The site table keyed on ``site_id``, for a caller adapting run after run.
``from_clim``, ``from_nee_store``, ``from_eki_predictions``
    Not written yet. The drivers and the constraints have readers of their own
    that already produce the form above
    (:func:`sipnet_calibration.drivers.driver_fields`,
    :func:`sipnet_calibration.constraints.constraint_fields`,
    :func:`sipnet_calibration.initial_conditions.initial_condition_fields`),
    so it is the NEE observations and the calibration output that are still
    owed. ``validate_field`` is owed with them (issue #6).

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

**The ERA5 drivers put a day's last step just past midnight.** SIPNET copies
its climate file's ``time`` column into its output verbatim, and in the ERA5
``.clim`` files that column drifts late within a year -- by seconds a step,
which is issue #9 seen from close up. pySIPNET builds its axis from that
column, snapping each step's end onto the next step's start, so a day's eighth
step ends a few seconds after midnight and a ``"1D"`` aggregation puts it in
the next day. Each *interior* daily cell is still eight consecutive steps
covering twenty-four hours; it is the window that is a step later than the one
SIPNET's own ``day`` column marks. The cells at the two ends of a record hold
whatever is left over, which for an extensive variable is a fraction of a day
reported in the units of a whole one. What that costs a daily comparison against an
observation has not been settled. It is not something an adapter can decide:
rebuilding the axis from
:func:`sipnet_calibration.obs_ops.sipnet_time_index`, which floors the hour
onto its slot, would align the cells with SIPNET's days and put this project's
labels at odds with pySIPNET's for the same run.

One trap belongs to an adapter still to be written here.
``from_eki_predictions`` will unstack a ``(J, N)`` block with the
``(site, variable, time)`` index from ``obs_ops.obs_index``, and it must be the
same index the observation operator used to build the observation vector, or
the predictions come back mislabeled against the observations they are
compared with. The traps of the observation and initial-condition sources are
in ``CLAUDE.md``'s Data section, where they apply to the readers that already
exist as well.

Usage
-----
One run, no site pool involved::

    from sipnet_calibration.fields import from_sipnet_output

    fields = from_sipnet_output(result, ["nee", "wood_carbon"])
    fields["net_ecosystem_exchange"].dims          # ('time',)
    fields["net_ecosystem_exchange"].attrs["kind"] # 'timestep_total'

An ensemble over sites and members, keyed by the pair each run stands for::

    from sipnet_calibration.fields import stack_sipnet_outputs
    from sipnet_calibration.obs_ops import aggregate_time
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

from sipnet_calibration.sites import load_sites

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pysipnet.output import SIPNETOutput
    from pysipnet.result import SIPNETResult

__all__ = [
    "CANONICAL_DIMS",
    "MEMBER_DIM",
    "MODEL_TIME_COORDS",
    "SITE_DIM",
    "TIME_DIM",
    "from_sipnet_output",
    "site_lookup",
    "stack_sipnet_outputs",
]

MEMBER_DIM = "member"
SITE_DIM = "site"
TIME_DIM = "time"

#: The dimensions a canonical field may have, in canonical order.
CANONICAL_DIMS: tuple[str, ...] = (MEMBER_DIM, SITE_DIM, TIME_DIM)

#: pySIPNET time coordinates a model field keeps. ``time`` is the end of the
#: step and ``time_step_start`` its start, so the two are the CF bounds pair.
MODEL_TIME_COORDS: tuple[str, ...] = (TIME_DIM, "time_step_start", "time_step_length")

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


def from_sipnet_output(
    output: SIPNETResult | SIPNETOutput,
    variables: str | Sequence[str],
    *,
    site: int | None = None,
    member: int | None = None,
    sites: pd.DataFrame | None = None,
) -> dict[str, xr.DataArray]:
    """The named variables of one SIPNET run, as canonical fields.

    Parameters
    ----------
    output:
        A :class:`pysipnet.result.SIPNETResult` or the
        :class:`pysipnet.output.SIPNETOutput` inside one.
    variables:
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
        :data:`MODEL_TIME_COORDS`, scalar ``site``/``lon``/``lat`` and
        ``member`` coordinates for the labels that were given, and pySIPNET's
        variable attributes unchanged.

    Raises
    ------
    KeyError
        If a variable is not a SIPNET output variable or alias, or if *site* is
        not in the site table.
    ValueError
        If *variables* is empty, unordered, or not a sequence of names; if the
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
    names = _resolved_names(variables)
    dataset = source.select(names)
    if TIME_DIM not in dataset.coords or dataset.sizes.get(TIME_DIM, 0) == 0:
        raise ValueError(
            "This SIPNET output has no rows, so there is nothing to put on a "
            "time axis. That usually means the run failed; check "
            "result.provenance.success and its stderr. Stacking an ensemble "
            "hits this on the first member that did not run."
        )
    # A DataArray cannot carry time_bounds -- its 'bounds' dimension is not a
    # field dimension -- so the attribute naming it would dangle.
    dataset[TIME_DIM].attrs = {
        key: value for key, value in dataset[TIME_DIM].attrs.items() if key != "bounds"
    }
    labels = _identity_coords(site=site, member=member, sites=sites)

    fields: dict[str, xr.DataArray] = {}
    for name in names:
        field = dataset[name]
        drop = [str(c) for c in field.coords if str(c) not in MODEL_TIME_COORDS]
        if drop:
            field = field.drop_vars(drop)
        if labels:
            field = field.assign_coords(labels)
        fields[name] = field
    return fields


def stack_sipnet_outputs(
    runs: Mapping[tuple[int, int], SIPNETResult | SIPNETOutput],
    variables: str | Sequence[str],
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
    variables:
        As for :func:`from_sipnet_output`.
    sites:
        The site table. Read once from disk when omitted.

    Returns
    -------
    dict
        Keyed by pySIPNET registry name, in the order requested. Each value is
        a ``DataArray`` with dims ``(member, site, time)``, ascending in
        ``member`` and ``site``, with ``lon``/``lat`` on ``site`` and the
        ``time`` coordinates of :data:`MODEL_TIME_COORDS`.

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
    if not isinstance(runs, Mapping):
        raise TypeError(
            f"runs must be a mapping from (site, member) to a run, not {type(runs).__name__}."
        )
    if not runs:
        raise ValueError("runs is empty; there is nothing to stack.")

    names = _resolved_names(variables)
    table = site_lookup(sites if sites is not None else load_sites())
    keys = sorted(_run_key(key) for key in runs)

    # {variable: {member: [field per site]}}, filled one run at a time.
    collected: dict[str, dict[int, list[xr.DataArray]]] = {name: {} for name in names}
    for site_id, member_id in keys:
        fields = from_sipnet_output(
            runs[(site_id, member_id)], names, site=site_id, member=member_id, sites=table
        )
        for name, field in fields.items():
            collected[name].setdefault(member_id, []).append(field)

    stacked: dict[str, xr.DataArray] = {}
    for name in names:
        by_member = [
            _concat(fields, SITE_DIM) for _, fields in sorted(collected[name].items())
        ]
        field = _concat(by_member, MEMBER_DIM).transpose(*CANONICAL_DIMS)
        stacked[name] = field.assign_coords(_site_locations(field[SITE_DIM].values, table))
    return stacked


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


def _resolved_names(variables: str | Sequence[str]) -> list[str]:
    """Requested variables as pySIPNET registry names, in order, without repeats."""
    from pysipnet.variables import resolve_output_variable

    if isinstance(variables, str):
        requested = [variables]
    elif isinstance(variables, (set, frozenset)):
        raise ValueError(
            f"variables was given as a {type(variables).__name__}, which has no "
            "order to keep. Pass a list or a tuple."
        )
    else:
        try:
            requested = list(variables)
        except TypeError:
            raise ValueError(
                f"variables must be a name or a sequence of names, got {variables!r}."
            ) from None
    if not requested:
        raise ValueError(
            "No variables were asked for. Name at least one SIPNET output "
            "variable, e.g. variables=['nee']."
        )
    names: list[str] = []
    for item in requested:
        if not isinstance(item, str):
            raise TypeError(f"Variable names must be strings, got {item!r}.")
        name = resolve_output_variable(item).name
        if name not in names:
            names.append(name)
    return names


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


def _concat(fields: list[xr.DataArray], dim: str) -> xr.DataArray:
    """Concatenate along *dim*, promoting the scalar *dim* coordinate each field carries.

    ``coords="different"`` is what gives the interval coordinates a ``site`` or
    ``member`` dimension when the runs disagree about them, rather than
    refusing; it is passed explicitly because xarray's default for it is
    changing.
    """
    if len(fields) == 1:
        return fields[0].expand_dims(dim)
    return xr.concat(
        fields,
        dim=dim,
        join="outer",
        coords="different",
        compat="equals",
        combine_attrs="override",
    )

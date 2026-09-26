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
``xarray.Dataset``
    One run's chosen variables, as ``SIPNETOutput.select(names)`` returns
    them: what :func:`label_run` labels and :func:`stack_model_outputs`
    stacks.
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
returns one per scalar component. Their names are ``<parameter>`` or
``<parameter>.<component>``, the calibration vector's own, not registry
names.

One array holds one variable. Variables that share one grid are held together
as an ``xarray.Dataset``: one run's output from :func:`label_run`, a stack of
runs from :func:`stack_model_outputs`, and the calibration parameters' fields.
Variables that do not share one are a ``dict[str, DataArray]`` keyed by name,
as the constraints are, being annual, dated or static by product. The field
adapters here return such a dict too, one field per variable asked for, as the
readers in :mod:`sipnet_calibration.drivers` and
:mod:`sipnet_calibration.constraints` do.

Identifiers
-----------
``site``
    The integer site id of the site table in use, as ``int32``. Ids of a pool
    shared with collaborators' files are never renumbered; a spatially
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

Each field keeps three of pySIPNET's time coordinates, ``time``,
``time_step_start`` and ``time_step_length``
(:data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`), the same three a
driver field from
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
``bounds`` is not a field dimension. ``time``'s ``bounds`` attribute
(:data:`~sipnet_calibration.conventions.STALE_TIME_ATTRIBUTE_NAMES`) is
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
:func:`coordinate_labels`, :func:`missing_labels`
    A coordinate's labels as a list, and the labels a field's coordinate
    lacks, in the order asked for.
:func:`without_stale_time_attributes`
    ``time`` attributes less
    :data:`~sipnet_calibration.conventions.STALE_TIME_ATTRIBUTE_NAMES`.
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
    nee = stack_sipnet_outputs(runs, ["nee"])["net_ecosystem_exchange"]
    nee.dims                                       # ('member', 'site', 'time')

    plot_time_series(aggregate_time(nee, "1D").sel(site=1))

Adapting run after run, with the site table read once::

    from sipnet_calibration.sites import load_sites, site_lookup

    table = site_lookup(load_sites())
    for site, member, run in ensemble:
        fields = from_sipnet_output(run, ["nee"], site=site, member=member, sites=table)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.variables import (
    resolve_output_variable_names as resolve_sipnet_output_variable_names,
)

from sipnet_calibration.conventions import (
    LAT,
    LON,
    SITE,
    SITE_ATTRIBUTES,
    SITE_DTYPE,
    STALE_TIME_ATTRIBUTE_NAMES,
    TIME,
    TIME_COORD_NAMES,
)
from sipnet_calibration.sites import (
    load_sites,
    site_coordinates,
    site_locations,
    site_lookup,
)
from sipnet_calibration.validation import as_bounded_integer, as_names, as_site_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pysipnet.output import SIPNETOutput
    from pysipnet.result import SIPNETResult

__all__ = [
    "FIELD_DIMS",
    "MEMBER_DIM",
    "coordinate_labels",
    "field_label",
    "from_sipnet_output",
    "label_run",
    "missing_labels",
    "resolve_output_variable_names",
    "stack_model_outputs",
    "stack_sipnet_outputs",
    "without_stale_time_attributes",
]

MEMBER_DIM = "member"

#: The dimensions a field may have, in the order they are written.
FIELD_DIMS: tuple[str, ...] = (MEMBER_DIM, SITE, TIME)


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
        If *dataset* is not an ``xr.Dataset``, *site* or *member* is a
        boolean, a float or not an integer, or *site_table* is not a
        ``DataFrame``.
    ValueError
        If *dataset* has no ``time`` rows, which is what a failed run leaves;
        if *site* or *member* is out of range; or if the site table lists a
        site twice or has no ``lon`` and ``lat`` columns.
    KeyError
        If *site* is not in the site table.
    FileNotFoundError
        If *site* is given, *site_table* is not, and the site table is absent.
    """
    check_is_a_dataset(dataset)
    check_run_has_rows(dataset)
    labels = _identity_coords(site=site, member=member, site_table=site_table)
    return dataset.assign_coords(labels) if labels else dataset


def from_sipnet_output(
    output: SIPNETResult | SIPNETOutput,
    output_variable_names: Sequence[str],
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
        A sequence of names. pySIPNET's names and aliases are both accepted
        (``"nee"``, ``"NEE"`` and ``"net_ecosystem_exchange"`` are the same
        variable); the keys of the result are always the registry name.
        Repeats are dropped and the requested order is kept.
    site:
        The site id this run is for, or ``None`` when the run is not at a
        site of the site table. When given, it becomes a scalar ``site``
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
        :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`, scalar
        ``site``/``lon``/``lat`` and ``member`` coordinates for the labels
        that were given, and pySIPNET's variable attributes unchanged.

    Raises
    ------
    KeyError
        If a variable is not a SIPNET output variable or alias, or if *site* is
        not in the site table.
    ValueError
        If *output_variable_names* is empty; if the run wrote no rows, which
        is what a failed run leaves; if *member* or *site* is out of range;
        or if the site table lists a site twice or has no ``lon`` and ``lat``
        columns.
    TypeError
        If *output* is neither a ``SIPNETResult`` nor a ``SIPNETOutput``; if
        *output_variable_names* is one string, a set, is not iterable, or
        holds a name that is not a string; or if *site* or *member* is a
        boolean, a float or not an integer.
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
    output_variable_names: Sequence[str],
    *,
    sites: pd.DataFrame | None = None,
) -> dict[str, xr.DataArray]:
    """Many SIPNET runs, labeled by site and member, as ``(member, site, time)`` fields.

    Parameters
    ----------
    runs:
        A mapping from ``(site, member)`` to the run for that pair.
        ``site`` is the site id and ``member`` the 0-based ensemble
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
        ``time`` coordinates of
        :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`.

    Raises
    ------
    TypeError
        If *runs* is not a mapping; if a value is neither a ``SIPNETResult``
        nor a ``SIPNETOutput``; if *output_variable_names* is not an ordered
        sequence of names; if a key's site or member is a boolean, a float or
        not an integer; or if *sites* is not a ``DataFrame``.
    ValueError
        If *runs* is empty, or a key is not a pair of integers in range; and
        for any refusal of :func:`stack_model_outputs`.
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
    check_is_a_nonempty_mapping(runs, "runs")
    names = resolve_output_variable_names(output_variable_names)
    table = site_lookup(sites if sites is not None else load_sites())
    model_outputs = {_run_key(key): _output_of(run).select(names) for key, run in runs.items()}
    stacked = stack_model_outputs(model_outputs, site_table=table)
    return {name: stacked[name] for name in names}


def stack_model_outputs(
    model_outputs: Mapping[tuple[int, int], xr.Dataset],
    *,
    site_table: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Many runs' output Datasets, keyed by site and member, as one Dataset.

    Parameters
    ----------
    model_outputs:
        A mapping from ``(site, member)`` to that run's output ``Dataset``,
        from ``result.outputs.select(names)`` or :func:`label_run`, every run
        carrying the same variables with the same ``units``, ``constituent``
        and ``kind``. ``site`` is the site id and ``member`` the
        0-based ensemble index. A run already labeled by :func:`label_run`
        must carry the labels of its key. The pairs need not form a full
        rectangle; a pair left out reads as ``NaN``.
    site_table:
        The site table, for ``lon``/``lat``. Read once from disk when omitted.

    Returns
    -------
    xarray.Dataset
        The runs' variables on ``(member, site, time)``, ascending in
        ``member`` (``int16``) and ``site`` (``int32``), with ``lon``/``lat``
        (``float64``, CF attributes) on ``site`` and the ``time`` coordinates
        of :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`. The
        variables' and the first run's dataset attributes are pySIPNET's.
        Every other coordinate -- ``time_bounds``, which a field cannot
        carry, and SIPNET's
        ``year``/``day_of_year``/``hour_of_day`` row labels -- is dropped, as
        is the ``bounds`` attribute of ``time``, so each variable of the
        result is a field.

    Raises
    ------
    TypeError
        If *model_outputs* is not a mapping, or a value is not an
        ``xr.Dataset``; if a key's site or member is a boolean, a float or
        not an integer; or if *site_table* is not a ``DataFrame``.
    ValueError
        If *model_outputs* is empty; if a key is not a pair of integers in
        range; if a run has no ``time`` rows; if a run's own ``site`` or
        ``member`` label disagrees with its key; if two runs carry different
        variables, or describe one with different ``units``, ``constituent``
        or ``kind``; or if the site table lists a site twice or has no
        ``lon`` and ``lat`` columns.
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
    check_is_a_nonempty_mapping(model_outputs, "model_outputs")
    by_key = _checked_model_outputs(model_outputs)
    table = site_table if site_table is not None else load_sites()
    # Located before stacking, so a site the table lacks fails before the work;
    # the stack's sites are ascending, as these are.
    locations = site_locations(sorted({site_id for site_id, _ in by_key}), table)
    by_member: dict[int, list[xr.Dataset]] = {}
    for (site_id, member_id), dataset in by_key.items():
        labeled = _labeled_for_stacking(dataset, site_id, member_id)
        by_member.setdefault(member_id, []).append(labeled)
    per_member = [_stack_along(per_site, SITE) for _, per_site in sorted(by_member.items())]
    stacked = _stack_along(per_member, MEMBER_DIM).transpose(*FIELD_DIMS, ...)
    # lon/lat are assigned after stacking rather than left to xarray.concat,
    # which promotes a scalar coordinate to the concatenated dimension only
    # when the values it is given differ, so a one-site or one-member stack
    # would otherwise keep them scalar and break the convention.
    return stacked.assign_coords(locations)


def resolve_output_variable_names(output_variable_names: Iterable[str]) -> list[str]:
    """Requested output variable names as pySIPNET registry names.

    Parameters
    ----------
    output_variable_names:
        An ordered sequence of names. pySIPNET's registry names, aliases and
        SIPNET's own column names, the legacy ones included, are all accepted
        (``"nee"``, ``"NEE"`` and ``"net_ecosystem_exchange"`` are the same
        variable).

    Returns
    -------
    list of str
        The registry names, in the order requested, each once, as
        :func:`pysipnet.variables.resolve_output_variable_names` resolves them.

    Raises
    ------
    TypeError
        If *output_variable_names* is one string, a set (which has no order to
        keep) or not iterable, or if an item is not a string.
    ValueError
        If *output_variable_names* is empty.
    KeyError
        If a name is not a pySIPNET output variable or alias.
    """
    requested = list(as_names(output_variable_names, message_name="output_variable_names"))
    check_names_are_given(requested)
    return resolve_sipnet_output_variable_names(requested)


def field_label(field: xr.DataArray, default: str = "the field", *, quoted: bool = True) -> str:
    """How a field is called in a message: its name, or else its derivation.

    Parameters
    ----------
    field:
        The field to name.
    default:
        What to call it when it has neither a name nor a ``derivation``.
    quoted:
        Whether to give the name or derivation as its ``repr``, for a message
        that does not quote it itself.

    Returns
    -------
    str
        The field's name; else its ``derivation`` attribute, which a result
        of :mod:`pysipnet.arithmetic` carries in place of a name; quoted when
        *quoted*. Else *default*, as given.
    """
    label = field.name if field.name is not None else field.attrs.get("derivation")
    if label is None or label == "":
        return default
    return repr(label) if quoted else str(label)


def coordinate_labels(coordinate: xr.DataArray) -> list:
    """A coordinate's labels as a flat list, whether it is a dimension or a scalar."""
    return np.asarray(coordinate.values).ravel().tolist()


def missing_labels(field: xr.DataArray | xr.Dataset, dim: str, labels: Iterable[Any]) -> list:
    """The *labels* that *field*'s *dim* coordinate lacks, in the order given.

    Parameters
    ----------
    field:
        The array or Dataset to look in; *dim* must be one of its indexed
        dimensions.
    dim:
        The dimension whose coordinate is read.
    labels:
        The labels wanted.

    Returns
    -------
    list
        The labels of *labels* not on *field*'s *dim*, in *labels*' order.
    """
    wanted = pd.Index(list(labels))
    return wanted[~wanted.isin(field.indexes[dim])].tolist()


def without_stale_time_attributes(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """*attrs* of a ``time`` coordinate less its stale attributes.

    Those are :data:`~sipnet_calibration.conventions.STALE_TIME_ATTRIBUTE_NAMES`.
    """
    return {key: value for key, value in attrs.items() if key not in STALE_TIME_ATTRIBUTE_NAMES}


# ── supporting helpers ────────────────────────────────────────────────────────

_MEMBER_ATTRS = {
    "long_name": "Ensemble member",
    "comment": "0-based, meaningful only within this source.",
}

#: The scalar coordinates :func:`label_run` adds.
_IDENTITY_COORD_NAMES: tuple[str, ...] = (SITE, MEMBER_DIM, LON, LAT)

#: The attributes that say what a variable is, which every stacked run must
#: agree on.
_VARIABLE_IDENTITY_ATTRIBUTE_NAMES: tuple[str, ...] = ("units", "constituent", "kind")


def _output_of(output: SIPNETResult | SIPNETOutput) -> SIPNETOutput:
    """The :class:`SIPNETOutput` of a run, given either it or the result holding it."""
    resolved = getattr(output, "outputs", output)
    if hasattr(resolved, "select"):
        return resolved
    if resolved is output:
        raise TypeError(
            "expected a pysipnet SIPNETResult or SIPNETOutput, got "
            f"{type(output).__name__}, which has neither .outputs nor .select; pass "
            "the result SIPNETModel or SIPNETRunner returned, or its .outputs."
        )
    raise TypeError(
        f"expected a pysipnet SIPNETResult, got {type(output).__name__} whose "
        f".outputs is {type(resolved).__name__} rather than a SIPNETOutput; pass "
        "the result SIPNETModel or SIPNETRunner returned."
    )


def _identity_coords(
    *, site: int | None, member: int | None, site_table: pd.DataFrame | None
) -> dict[str, xr.DataArray]:
    """Scalar ``site``/``lon``/``lat`` and ``member`` coordinates for the labels given."""
    coords: dict[str, xr.DataArray] = {}
    if member is not None:
        coords[MEMBER_DIM] = xr.DataArray(np.int16(_as_member(member)), attrs=_MEMBER_ATTRS)
    if site is not None:
        site_id = as_site_id(site, message_name="site")
        table = site_table if site_table is not None else load_sites()
        located = site_coordinates([site_id], table)
        coords.update({name: coordinate.isel({SITE: 0}) for name, coordinate in located.items()})
    return coords


def _as_member(member: Any) -> int:
    """A member label as a plain ``int``, from 0 to the largest ``int16``."""
    return as_bounded_integer(
        member, minimum=0, maximum=int(np.iinfo(np.int16).max), message_name="member"
    )


def _run_key(key: Any) -> tuple[int, int]:
    """*key* as a ``(site, member)`` pair of plain integers."""
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError(
            f"every key of the runs must be a (site, member) pair, got {key!r}; key "
            "each run by the site and member it was."
        )
    site, member = key
    return as_site_id(site, message_name="site"), _as_member(member)


def _checked_model_outputs(
    model_outputs: Mapping[Any, Any],
) -> dict[tuple[int, int], xr.Dataset]:
    """*model_outputs* keyed by plain ``(site, member)``, ascending, each checked."""
    by_key: dict[tuple[int, int], xr.Dataset] = {}
    for key in sorted(model_outputs, key=_run_key):
        site_id, member_id = _run_key(key)
        dataset = model_outputs[key]
        check_is_a_dataset(dataset)
        check_run_has_rows(dataset)
        check_run_labels_match_the_key(dataset, site_id, member_id)
        by_key[(site_id, member_id)] = dataset
    check_model_outputs_carry_the_same_variables(by_key)
    return by_key


def _labeled_for_stacking(dataset: xr.Dataset, site_id: int, member_id: int) -> xr.Dataset:
    """*dataset* with only a field's coordinates and its scalar ``site`` and ``member``."""
    labeled = _with_field_coords(dataset.drop_vars(_IDENTITY_COORD_NAMES, errors="ignore"))
    return labeled.assign_coords(
        {
            SITE: xr.DataArray(SITE_DTYPE(site_id), attrs=SITE_ATTRIBUTES),
            MEMBER_DIM: xr.DataArray(np.int16(member_id), attrs=_MEMBER_ATTRS),
        }
    )


def _with_field_coords(dataset: xr.Dataset) -> xr.Dataset:
    """*dataset* with only the coordinates a field keeps.

    Those are :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES` and
    the identity coordinates. ``time``'s ``bounds`` attribute goes too, since
    the ``time_bounds`` variable it names is one of the coordinates dropped.
    """
    keep = {*TIME_COORD_NAMES, *_IDENTITY_COORD_NAMES}
    # The copy gives this dataset its own variables, so rewriting an attribute
    # below leaves the caller's dataset untouched.
    dataset = dataset.drop_vars([str(c) for c in dataset.coords if str(c) not in keep]).copy()
    dataset[TIME].attrs = without_stale_time_attributes(dataset[TIME].attrs)
    return dataset


def _stack_along(datasets: list[xr.Dataset], dim: str) -> xr.Dataset:
    """*datasets* stacked along a new *dim*, from the scalar *dim* coordinate each carries.

    ``coords="different"`` is what gives the interval coordinates a ``site`` or
    ``member`` dimension when the runs disagree about them, rather than
    refusing; it and ``data_vars`` are passed explicitly because xarray's
    defaults for them are changing.
    """
    if len(datasets) == 1:
        return datasets[0].expand_dims(dim)
    return xr.concat(
        datasets,
        dim=dim,
        data_vars="all",
        join="outer",
        coords="different",
        compat="equals",
        combine_attrs="override",
    )


def _variable_identities(dataset: xr.Dataset) -> dict[str, tuple[Any, ...]]:
    """Each variable's ``units``, ``constituent`` and ``kind``, by name."""
    return {
        str(name): tuple(variable.attrs.get(a) for a in _VARIABLE_IDENTITY_ATTRIBUTE_NAMES)
        for name, variable in dataset.data_vars.items()
    }


def _key_label(key: tuple[int, int]) -> str:
    """A ``(site, member)`` key as a message names it."""
    return f"(site={key[0]}, member={key[1]})"


# ── checks ────────────────────────────────────────────────────────────────────


def check_is_a_dataset(dataset: Any) -> None:
    """*dataset* is an ``xr.Dataset``."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"expected a run's output as an xarray Dataset, from "
            f"result.outputs.select(names), got {type(dataset).__name__}."
        )


def check_run_has_rows(dataset: xr.Dataset) -> None:
    """A run's output has at least one ``time`` row."""
    if TIME not in dataset.coords or dataset.sizes.get(TIME, 0) == 0:
        raise ValueError(
            "this SIPNET output has no rows, so there is nothing to put on a time axis. "
            "That usually means the run failed; check result.provenance.success and its "
            "stderr. Stacking an ensemble hits this on the first member that did not run."
        )


def check_names_are_given(names: list[str]) -> None:
    """At least one output variable is asked for."""
    if not names:
        raise ValueError(
            "no variables were asked for; name at least one SIPNET output "
            "variable, e.g. output_variable_names=['nee']."
        )


def check_is_a_nonempty_mapping(value: Any, message_name: str) -> None:
    """*value* is a mapping from ``(site, member)`` with at least one entry."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{message_name} must be a mapping from (site, member) to a run, not "
            f"{type(value).__name__}; key each run by the site and member it was."
        )
    if not value:
        raise ValueError(f"{message_name} is empty; there is nothing to stack.")


def check_run_labels_match_the_key(dataset: xr.Dataset, site: int, member: int) -> None:
    """A run's own ``site`` and ``member`` labels, where it has them, are its key's."""
    for name, expected in ((SITE, site), (MEMBER_DIM, member)):
        if name not in dataset.coords:
            continue
        labels = coordinate_labels(dataset[name])
        if labels != [expected]:
            raise ValueError(
                f"the run keyed (site={site}, member={member}) is labeled {name}={labels}; "
                "key each run by the site and member it was."
            )


def check_model_outputs_carry_the_same_variables(
    model_outputs: Mapping[tuple[int, int], xr.Dataset],
) -> None:
    """Every run carries the same variables, each with the same identity attributes.

    A run missing a variable would be filled with ``NaN`` by the stack, and
    the stack takes the first run's attributes, so a second run's different
    ``units`` would be relabeled silently.
    """
    (first_key, first), *rest = model_outputs.items()
    expected = _variable_identities(first)
    for key, dataset in rest:
        found = _variable_identities(dataset)
        if set(found) != set(expected):
            raise ValueError(
                f"the run keyed {_key_label(key)} carries the variables {sorted(found)} "
                f"and the run keyed {_key_label(first_key)} {sorted(expected)}; select "
                "the same names from every run, with result.outputs.select(names)."
            )
        for name in expected:
            if found[name] != expected[name]:
                described = dict(zip(_VARIABLE_IDENTITY_ATTRIBUTE_NAMES, found[name]))
                first_described = dict(zip(_VARIABLE_IDENTITY_ATTRIBUTE_NAMES, expected[name]))
                raise ValueError(
                    f"the run keyed {_key_label(key)} describes {name!r} as {described} "
                    f"and the run keyed {_key_label(first_key)} as {first_described}; "
                    "the stack carries one set of attributes, so convert the runs to "
                    "one before stacking them."
                )

"""The field contract, and the labeling and stacking of SIPNET runs.

Data reaches this project in as many shapes as it has sources. SIPNET writes
columnar output, the meteorological drivers are text files, the initial
conditions are per-site netCDF, the observations are a csv and a set of R
objects, and the calibration returns flat arrays that carry no record of space
or time. This module defines the single form all of them are converted into,
the validator that checks it, and the adapters that label SIPNET's runs with
it. Everything downstream -- the plots and the observation operator -- reads
that form and nothing else, so adding a source costs one adapter rather than a
change in every consumer.

Where this sits
---------------
The model-output adapters here read what pySIPNET produces, in memory; no
script writes a processed file for them::

    SIPNETRunner / SIPNETModel  ->  SIPNETResult.outputs (SIPNETOutput)

    one run    -> from_sipnet_output()   -> dict[str, DataArray] on (time,)
               -> label_run()            -> Dataset on (time,), labeled
    many runs  -> stack_sipnet_outputs() -> dict[str, DataArray]
                                              on (*batch, site, time)
               -> stack_model_outputs()  -> Dataset on (*batch, site, time)

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

The field contract
------------------
A **field** is an ``xarray.DataArray`` holding one variable, with

* **dims** ``(*batch, space, time)``: zero or more batch dims, then at most
  one spatial dim, then ``time`` if present. No other dim is allowed: a
  structural axis (variable, component, quantile, bounds, a PFT class) is
  split into a ``dict`` or ``Dataset`` of fields instead.
* **the spatial dim** one of ``site`` (site ids of the site table in use),
  ``point`` (arbitrary locations, integer labels) or a raster pair ``lat``,
  ``lon`` (or projected ``y``, ``x``); these names
  (:data:`~sipnet_calibration.conventions.SPATIAL_DIM_NAMES`) are never batch
  dims. ``site`` and ``point`` carry ``float64`` ``lon``/``lat`` coordinates on
  that dim; a scalar ``site`` coordinate marks a field of one site.
* **a batch dim** every other dim whose index coordinate holds integers, any
  distinct ones: an axis of independent replicates. A dim with string or float
  labels, or none, is refused, which is what keeps a ``variable``,
  ``quantile``, ``pft`` or ``bounds`` dim off a field.
* ``site`` ``int32`` site ids, unique; ``time`` naive ``datetime64``,
  strictly increasing, no ``NaT`` (any datetime64 unit; pandas and
  xarray make microseconds), with pySIPNET's timestep coordinates and
  an observation's window coordinates, where present, on ``time`` alone.
* ``units`` in ``attrs``, valid by pySIPNET's ``validate_units`` (a
  categorical field -- CF ``flag_values``, or string or boolean values --
  needs none); ``long_name``;
  ``constituent`` and ``kind`` where pySIPNET's apply.

:func:`validate_field` checks all of this. A **scalar** coordinate is not a
dim: a field whose batch dim was selected away with ``.isel(sample=k)`` has no
batch dim, and its scalar label is metadata.

**Two batch dims with the same name are the same index; different names are
different indices.** xarray aligns two ``sample`` dims by label and PyEns
zips them, while a ``sample`` and an ``initial_condition_member`` cross. The
batch dim made from batched Flat is named
:data:`~sipnet_calibration.conventions.SAMPLE` by default, with labels ``0``
to ``n_samples - 1`` in row order; a data source's own ensemble is named for
its source (``initial_condition_member``, ``driver_member``). Batch labels
are ``int64``
(:data:`~sipnet_calibration.conventions.BATCH_LABEL_DTYPE`).

Which dims are present depends on the quantity. A single deterministic run is
``(time,)``, an initial condition ensemble is
``(initial_condition_member, site)``, and runs over samples and sites are
``(sample, site, time)``. Calibration parameters are ``(sample, site)`` for a
batch and ``(site,)`` for one value:
:meth:`sipnet_calibration.parameter_vector.ParameterVector.fields` returns
one per scalar component. Their names are ``<parameter>`` or
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
a batch dim
    ``int64`` labels, meaningful only within the dim's own name. Whether
    member *i* of one data source corresponds to member *i* of another is not
    established, which is why the two carry different names.
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

so the interval a value covers is ``(time_step_start, time]``. Its two edges
are the pair pySIPNET writes as its CF ``time_bounds`` variable, which a DataArray
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
:func:`validate_field`
    Check that an array is a field, raising on the first rule it breaks.
:func:`batch_dims`
    A field's batch dims, in its dim order.
:func:`stack_batch_dims`, :func:`unstack_batch_dims`
    Several batch dims stacked into one, labeled ``0`` to ``n - 1`` with the
    original labels kept beside it, and back.
:func:`from_sipnet_output`
    One run's chosen variables as fields, optionally labeled with a site and
    batch labels.
:func:`label_run`
    One run's output ``Dataset`` labeled with its site and batch labels: the
    ``model_output`` the observation operators read.
:func:`stack_sipnet_outputs`
    Many runs, each keyed by its labels in ``key_dims`` order, stacked into
    ``(*batch, site, time)`` fields.
:func:`stack_model_outputs`
    Many runs' output ``Dataset`` objects, keyed the same way, stacked into
    one ``Dataset`` on ``(*batch, site, time)``.
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

The drivers, the constraints and the initial conditions have readers of their
own that already produce the form above
(:func:`sipnet_calibration.drivers.driver_fields`,
:func:`sipnet_calibration.constraints.constraint_fields`,
:func:`sipnet_calibration.initial_conditions.initial_condition_fields`), and a
batch of predictions is unstacked by ``ObservationVector.fields``.

Notes
-----
**Why no wrapper class.** The three operations this project performs on a
field are ``.quantile(dim="sample")``, ``.resample(time=...)`` and
``.sel(site=...)``, all of which xarray already has. A class would spend the
project re-exporting them, and every plotter would have to unwrap it. The
convention plus a validator is the whole design.

**Why a batch dim is known by its labels.** Every structural dim the
project's own idioms produce has labels that are not integers
(``groupby("pft")`` gives strings, ``.quantile`` floats, ``to_array``
strings) or no coordinate at all (``bounds``), and every batch dim the
project creates carries an integer coordinate. So the rule needs neither a
deny-list of names nor a marker attribute, which xarray drops in some
operations.

**Why Flat takes one batch dim.** pyEKI takes exactly ``(J, ·)``, and a
``(J, N)`` array cannot say which of several dims its rows came from, so a
field with several is reduced, or stacked with :func:`stack_batch_dims`,
before it is flattened.

**Why the adapter selects.** ``SIPNETOutput.xarray`` reads and caches every
column SIPNET wrote. Across an ensemble that is every run's full output held
at once, where one variable per run is what the caller asked for, so the
adapter goes through ``select`` and never touches ``.xarray`` or ``.pandas``.

**Why a missing run is not an error.** The caller supplies the mapping, so it
already knows which runs it left out; those cells read ``NaN``. This differs
from :func:`sipnet_calibration.drivers.load_drivers`, which discovers absence
on disk and therefore has to report it.

A ``(J, N)`` batch of predictions is unstacked by
:meth:`sipnet_calibration.observation.ObservationVector.fields`, which owns the
``(site, product, time)`` index the batch was flattened with, so the two
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

An ensemble over samples and sites, keyed by the ``(sample, site)`` each run
stands for::

    from sipnet_calibration.fields import stack_sipnet_outputs, validate_field
    from sipnet_calibration.observation.time_alignment import aggregate_time
    from sipnet_calibration.plotting import plot_time_series

    runs = {(0, 1): first, (1, 1): second, (0, 27): third, (1, 27): fourth}
    nee = stack_sipnet_outputs(runs, ["nee"])["net_ecosystem_exchange"]
    nee.dims                                       # ('sample', 'site', 'time')
    validate_field(nee)                            # None: it is a field

    plot_time_series(aggregate_time(nee, "1D").sel(site=1))

Two batch dims, stacked into one for Flat and back::

    from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

    wood.dims                                  # ('sample', 'initial_condition_member', 'site')
    stacked = stack_batch_dims(wood)           # ('sample', 'site'), sample 0..n-1
    stacked.coords["initial_condition_member_label"]   # the original labels, on sample
    unstack_batch_dims(stacked).dims           # ('sample', 'initial_condition_member', 'site')

Adapting run after run, with the site table read once::

    from sipnet_calibration.sites import load_sites, site_lookup

    table = site_lookup(load_sites())
    for site, sample, run in ensemble:
        fields = from_sipnet_output(
            run, ["nee"], site=site, batch={"sample": sample}, sites=table
        )
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.units import validate_units
from pysipnet.variables import (
    resolve_output_variable_names as resolve_sipnet_output_variable_names,
)

from sipnet_calibration.conventions import (
    BATCH_LABEL_DTYPE,
    DATA_SOURCE_MEMBER_ATTRIBUTES,
    DATA_SOURCE_MEMBER_NAMES,
    LAT,
    LON,
    POINT,
    SAMPLE,
    SAMPLE_ATTRIBUTES,
    SITE,
    SITE_ATTRIBUTES,
    SITE_DTYPE,
    SOURCE_INDEX,
    SPATIAL_DIM_NAMES,
    STALE_TIME_ATTRIBUTE_NAMES,
    TIME,
    TIME_COORD_NAMES,
    TIMESTEP_LENGTH,
    TIMESTEP_START,
    WINDOW_END,
    WINDOW_START,
    X,
    Y,
)
from sipnet_calibration.sites import (
    check_site_table_locates_the_sites,
    load_sites,
    site_coordinates,
    site_locations,
    site_lookup,
)
from sipnet_calibration.validation import as_batch_label, as_names, as_site_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pysipnet.output import SIPNETOutput
    from pysipnet.result import SIPNETResult

__all__ = [
    "STACKED_COMPANIONS_ATTRIBUTE",
    "STACKED_DIMS_ATTRIBUTE",
    "STACKED_LABEL_SUFFIX",
    "batch_coordinate",
    "batch_dims",
    "check_at_most_one_batch_dim",
    "check_batch_dim_name_is_not_reserved",
    "check_batch_labels_are_a_mapping",
    "check_dims_are_batch_spatial_or_time",
    "coordinate_labels",
    "field_label",
    "from_sipnet_output",
    "label_run",
    "missing_labels",
    "resolve_output_variable_names",
    "scalar_batch_labels",
    "stack_batch_dims",
    "stack_model_outputs",
    "stack_sipnet_outputs",
    "unstack_batch_dims",
    "validate_field",
    "without_stale_time_attributes",
]

#: The suffix of the coordinates :func:`stack_batch_dims` keeps each stacked
#: dim's labels in: stacking ``initial_condition_member`` keeps its labels as
#: ``initial_condition_member_label`` on the new dim, which is what
#: :func:`unstack_batch_dims` reads back.
STACKED_LABEL_SUFFIX = "_label"

#: The attribute of a stacked batch coordinate naming the dims stacked into
#: it, in their order, separated by spaces (``"sample driver_member"``).
#: :func:`unstack_batch_dims` reads only these names back.
STACKED_DIMS_ATTRIBUTE = "stacked_dims"

#: The attribute of a stacked batch coordinate naming each coordinate that was
#: on stacked dims alone, with those dims (``"source_index:driver_member"``,
#: space-separated, dims joined by commas), so that
#: :func:`unstack_batch_dims` puts it back on them.
STACKED_COMPANIONS_ATTRIBUTE = "stacked_companions"


# ── the field contract ────────────────────────────────────────────────────────


def validate_field(field: Any, *, message_name: str | None = None) -> None:
    """Check that *field* is a field, raising on the first rule it breaks.

    Runs the checks of the field contract (this module's docstring) in
    order: :func:`check_field_is_a_dataarray`,
    :func:`check_field_dims_are_field_dims`,
    :func:`check_field_site_holds_site_ids`,
    :func:`check_field_locations_are_on_the_spatial_dim`,
    :func:`check_field_time_is_a_time_axis`,
    :func:`check_field_interval_coordinates_are_on_time` and
    :func:`check_field_units_are_valid`.

    Parameters
    ----------
    field:
        The array to check.
    message_name:
        What an error message calls it; its name, or else its derivation,
        when omitted.

    Raises
    ------
    TypeError
        If *field* is not an ``xr.DataArray``.
    ValueError
        Naming the first rule broken: a dim that is neither a batch dim, a
        spatial dim nor ``time``, or dims out of the ``(*batch, space, time)``
        order; more than one spatial dim; a dim without an index coordinate;
        repeated batch labels; ``site`` labels that are not unique ``int32``
        site ids; ``lon``/``lat`` missing from, or not ``float64`` on, a
        ``site`` or ``point`` dim; ``time`` labels that are not naive
        ``datetime64``, strictly increasing and free of ``NaT``; a
        timestep or window coordinate on a dim other than ``time``; or
        ``units`` missing or refused by pySIPNET's ``validate_units``.
    """
    check_field_is_a_dataarray(field, message_name)
    name = message_name if message_name is not None else field_label(field)
    check_field_dims_are_field_dims(field, name)
    check_field_site_holds_site_ids(field, name)
    check_field_locations_are_on_the_spatial_dim(field, name)
    check_field_time_is_a_time_axis(field, name)
    check_field_interval_coordinates_are_on_time(field, name)
    check_field_units_are_valid(field, name)


def batch_dims(field: xr.DataArray | xr.Dataset) -> tuple[str, ...]:
    """The batch dims of a field, or of a Dataset of fields, in dim order.

    A batch dim is a dim other than the spatial names
    (:data:`~sipnet_calibration.conventions.SPATIAL_DIM_NAMES`) and ``time``
    whose index coordinate holds integers. A dim that is none of these --
    one with string or float labels, or none -- is not returned; it is not
    allowed on a field, and :func:`validate_field` refuses it.

    Parameters
    ----------
    field:
        A field, or an ``xr.Dataset`` whose variables are fields, such as a
        model output (whose ``bounds`` dim, having no coordinate, is not a
        batch dim).

    Returns
    -------
    tuple of str
        The batch dims, in the order *field* has them. A scalar coordinate is
        on no dim and is never one.
    """
    return tuple(str(dim) for dim in field.dims if _is_batch_dim(field, str(dim)))


def scalar_batch_labels(field: xr.DataArray | xr.Dataset) -> tuple[str, ...]:
    """The scalar coordinates of *field* that are batch labels, in coordinate order.

    A batch label left by selecting a batch dim away, as ``.isel(sample=k)``
    leaves ``sample``: a zero-dimensional coordinate holding an integer,
    whose name is not a spatial name
    (:data:`~sipnet_calibration.conventions.SPATIAL_DIM_NAMES`, so neither
    ``site`` nor ``lon``/``lat``), not ``time`` and not
    :data:`~sipnet_calibration.conventions.SOURCE_INDEX`, the file index
    beside a data source's member.

    Parameters
    ----------
    field:
        A field, or a Dataset of fields.

    Returns
    -------
    tuple of str
        The names, in the order *field* holds its coordinates.
    """
    return tuple(
        str(name)
        for name, coordinate in field.coords.items()
        if coordinate.ndim == 0
        and coordinate.dtype.kind in "iu"
        and name not in SPATIAL_DIM_NAMES
        and name not in (TIME, SOURCE_INDEX)
    )


def batch_coordinate(dim: str, labels: Any) -> xr.DataArray:
    """A batch dim's coordinate: ``int64`` labels with the attributes of its dim.

    Parameters
    ----------
    dim:
        The batch dim's name.
    labels:
        One integer, for a scalar label, or a sequence of integers.

    Returns
    -------
    xarray.DataArray
        ``int64``, zero-dimensional for one label and on ``(dim,)`` for a
        sequence, with
        :data:`~sipnet_calibration.conventions.SAMPLE_ATTRIBUTES` for
        ``sample``,
        :data:`~sipnet_calibration.conventions.DATA_SOURCE_MEMBER_ATTRIBUTES`
        for a data source's member dim
        (:data:`~sipnet_calibration.conventions.DATA_SOURCE_MEMBER_NAMES`),
        and none otherwise.

    Raises
    ------
    TypeError
        If *dim* is not a string, or a label is a boolean, a float or not an
        integer.
    ValueError
        If *dim* is a reserved name, or a label does not fit ``int64``.
    """
    check_batch_dim_name_is_not_reserved(dim, message_name="dim")
    attributes = _batch_coordinate_attributes(dim)
    if np.ndim(labels) == 0:
        label = as_batch_label(labels, message_name=dim)
        return xr.DataArray(BATCH_LABEL_DTYPE(label), attrs=attributes)
    values = [as_batch_label(label, message_name=dim) for label in np.asarray(labels).tolist()]
    return xr.DataArray(np.asarray(values, dtype=BATCH_LABEL_DTYPE), dims=(dim,), attrs=attributes)


def stack_batch_dims(field: xr.DataArray, *, into: str) -> xr.DataArray:
    """A field's batch dims stacked into one new batch dim, so it can be flattened.

    Parameters
    ----------
    field:
        A field with at least one batch dim.
    into:
        The name of the stacked dim, first in the result's dims. It is a new
        index, so it takes a new name: none of the dims stacked, no
        coordinate of *field*, no ``<dim>_label`` name the stack creates,
        and no reserved name (:func:`check_batch_dim_name_is_not_reserved`).

    Returns
    -------
    xarray.DataArray
        *field* on ``(into, space, time)``: *into* is labeled ``0`` to
        ``n - 1`` (``int64``) in C order over the batch dims as *field* has
        them, the last varying fastest, and its coordinate records them in
        :data:`STACKED_DIMS_ATTRIBUTE`. Each stacked dim's labels are kept,
        with their attributes, as a non-dim coordinate on *into* named
        ``<dim>_label`` (:data:`STACKED_LABEL_SUFFIX`); a coordinate that was
        on stacked dims alone, such as ``source_index``, is on *into* too and
        recorded in :data:`STACKED_COMPANIONS_ATTRIBUTE`. That is what
        :func:`unstack_batch_dims` reverses it by.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray`` or *into* is not a string.
    ValueError
        If *field* is not a field (:func:`validate_field`) or has no batch
        dim; if *into* is a reserved name, one of the dims stacked, a
        coordinate of *field*, or one of the ``<dim>_label`` names the stack
        creates; or if *field* carries one of those names already.

    Notes
    -----
    Stacking into one of the stacked dims' names would reuse that name for a
    different index, and xarray and the operators align on names: a stack of
    ``(sample, driver_member)`` labeled ``sample`` ``0..n-1`` would then be
    read at the wrong rows of a SIPNET table on theta's ``sample``.
    """
    validate_field(field)
    dims = batch_dims(field)
    check_field_has_a_batch_dim(dims, field_label(field))
    label_names = {dim: f"{dim}{STACKED_LABEL_SUFFIX}" for dim in dims}
    check_stack_names_are_free(field, into, dims, tuple(label_names.values()))
    companions = _companion_coordinates(field, dims)
    rest = [str(d) for d in field.dims if d not in dims]
    stacked = field.transpose(*dims, *rest).stack({into: list(dims)}, create_index=False)
    stacked = stacked.rename(label_names)
    coordinate = batch_coordinate(into, np.arange(stacked.sizes[into]))
    coordinate.attrs[STACKED_DIMS_ATTRIBUTE] = " ".join(dims)
    if companions:
        coordinate.attrs[STACKED_COMPANIONS_ATTRIBUTE] = " ".join(
            f"{name}:{','.join(on)}" for name, on in companions.items()
        )
    return stacked.assign_coords({into: coordinate}).transpose(into, *rest)


def unstack_batch_dims(
    field: xr.DataArray, *, labels_from: xr.DataArray | None = None
) -> xr.DataArray:
    """The inverse of :func:`stack_batch_dims`: the stacked dim unstacked.

    Parameters
    ----------
    field:
        A field with one batch dim stacked by :func:`stack_batch_dims`, its
        coordinate recording the stacked dims and carrying their
        ``<dim>_label`` coordinates; or, with *labels_from*, a field on the
        same batch dim without them, such as one of the arrays a vector's
        ``fields(flat_values, batch_dim=<the stacked dim>)`` makes from Flat.
    labels_from:
        The stacked field, or its stacked coordinate (``stacked[into]``), to
        copy the record and the label and companion coordinates from, when
        *field* lacks them. Its rows must be *field*'s: the same labels in
        the same order.

    Returns
    -------
    xarray.DataArray
        *field* on the stacked dims, in the order they were stacked, then its
        spatial dim and ``time``. Each dim's labels are in the order they
        first appear along the stacked dim, which is their original order
        when every row is there; a combination of labels the stacked dim
        lacks is ``NaN``. The label coordinates' attributes and the
        companion coordinates are restored.

    Raises
    ------
    TypeError
        If *field* or *labels_from* is not a ``DataArray``.
    ValueError
        If *field* is not a field; if no batch dim, or more than one,
        records a stack (:data:`STACKED_DIMS_ATTRIBUTE`, which some xarray
        operations drop); if a ``<dim>_label`` coordinate it names is
        missing; if two rows carry the same labels; or if *labels_from* is
        on another batch dim or other rows.

    Notes
    -----
    A Dataset of stacked fields is unstacked variable by variable,
    ``stacked_dataset.map(unstack_batch_dims)``, and a ``dict`` by a
    comprehension; with *labels_from*,
    ``{name: unstack_batch_dims(array, labels_from=stacked[name]) for name,
    array in made.items()}``. A coordinate on a stacked dim and another dim
    at once comes back on every stacked dim.
    """
    validate_field(field)
    if labels_from is not None:
        field = _with_stack_record_from(field, labels_from)
    stacked_dim = _stacked_dim(field)
    record = field[stacked_dim].attrs
    originals = str(record[STACKED_DIMS_ATTRIBUTE]).split()
    label_names = [f"{dim}{STACKED_LABEL_SUFFIX}" for dim in originals]
    check_stack_labels_are_present(field, stacked_dim, label_names)
    check_stacked_rows_are_distinct(field, label_names)
    companions = _recorded_companions(record)
    saved = {name: field[name] for name in [*label_names, *companions] if name in field.coords}
    rest = [str(d) for d in field.dims if d != stacked_dim]
    if len(originals) == 1:
        unstacked = field.swap_dims({stacked_dim: label_names[0]}).drop_vars(stacked_dim)
        unstacked = unstacked.rename({label_names[0]: originals[0]})
    else:
        bare = field.drop_vars([stacked_dim, *companions])
        unstacked = bare.set_index({stacked_dim: label_names}).unstack(stacked_dim)
        unstacked = unstacked.rename(dict(zip(label_names, originals)))
        unstacked = unstacked.reindex(
            {dim: pd.unique(saved[label].values) for dim, label in zip(originals, label_names)}
        )
    for dim, label in zip(originals, label_names):
        unstacked[dim].attrs = dict(saved[label].attrs)
    unstacked = unstacked.transpose(*originals, *rest)
    for name, on in companions.items():
        unstacked = unstacked.assign_coords({name: _companion_on(saved, name, on, unstacked)})
    return unstacked


# ── labeling and stacking runs ────────────────────────────────────────────────


def label_run(
    dataset: xr.Dataset,
    *,
    site: int | None = None,
    batch: Mapping[str, int] | None = None,
    site_table: pd.DataFrame | None = None,
) -> xr.Dataset:
    """A run's output dataset labeled with the site and batch labels it was.

    pySIPNET's ``SIPNETOutput.select(names)`` gives one run's variables as a
    CF Dataset on ``(time,)``; what it cannot know is which site of the pool
    and which sample (or member of a data source) the run was. This adds
    those as scalar coordinates, ``site`` with its ``lon``/``lat`` from the
    site table and one per batch label, and changes nothing else: every
    variable, coordinate (``time_bounds`` included) and attribute is
    pySIPNET's. The result is the ``model_output`` the observation operators
    read.

    Parameters
    ----------
    dataset:
        The run's variables, from ``result.outputs.select(names)`` or
        ``result.outputs[[...]]``.
    site:
        The site id this run is for, or ``None`` when the run is not at a
        site of the site table.
    batch:
        ``{batch dim: label}`` for each batch dim the run stands at, such as
        ``{"sample": 3}``, or ``None``. Each label is an integer that fits
        ``int64``; each dim name a string that is not reserved
        (:func:`check_batch_dim_name_is_not_reserved`) and is no variable,
        dim or coordinate of *dataset*.
    site_table:
        The site table, as :func:`sipnet_calibration.sites.load_sites` returns
        it, read from disk when omitted and a *site* is given; pass it when
        labeling many runs so it is read once.

    Returns
    -------
    xarray.Dataset
        *dataset* with a scalar ``site`` (``int32``) and its scalar
        ``lon``/``lat`` when *site* is given, and a scalar ``int64``
        coordinate per entry of *batch*; *dataset* itself when neither is.

    Raises
    ------
    TypeError
        If *dataset* is not an ``xr.Dataset``; *site* or a batch label is a
        boolean, a float or not an integer; *batch* is not a mapping or a
        dim name is not a string; or *site_table* is not a ``DataFrame``.
    ValueError
        If *dataset* has no ``time`` rows, which is what a failed run leaves;
        if *site* is out of range, or a batch label does not fit ``int64``; if
        a batch dim name is reserved or is a variable, dim or coordinate of
        *dataset* (``time_step_length``, ``time_bounds``, ``bounds``, a
        variable's name); or if the site table lists a site twice or has no
        ``lon`` and ``lat`` columns.
    KeyError
        If *site* is not in the site table.
    FileNotFoundError
        If *site* is given, *site_table* is not, and the site table is absent.
    """
    check_is_a_dataset(dataset)
    check_run_has_rows(dataset)
    if batch is not None:
        check_batch_labels_are_a_mapping(batch)
        for dim in batch:
            check_batch_dim_name_is_not_reserved(dim, message_name="batch")
            check_batch_name_is_not_the_model_outputs(dataset, dim)
    labels = _run_label_coords(site=site, batch=batch, site_table=site_table)
    return dataset.assign_coords(labels) if labels else dataset


def from_sipnet_output(
    output: SIPNETResult | SIPNETOutput,
    output_variable_names: Sequence[str],
    *,
    site: int | None = None,
    batch: Mapping[str, int] | None = None,
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
    batch:
        ``{batch dim: label}`` for the run, as :func:`label_run` takes it; each
        becomes a scalar coordinate.
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
        ``site``/``lon``/``lat`` and batch-label coordinates for the labels
        that were given, and pySIPNET's variable attributes unchanged.

    Raises
    ------
    KeyError
        If a variable is not a SIPNET output variable or alias, or if *site* is
        not in the site table.
    ValueError
        If *output_variable_names* is empty; if the run wrote no rows, which
        is what a failed run leaves; if *site* is out of range or a batch
        label does not fit ``int64``; if a batch dim name is reserved or a
        variable, dim or coordinate of the run's output; or if the site table
        lists a site twice or has no ``lon`` and ``lat`` columns.
    TypeError
        If *output* is neither a ``SIPNETResult`` nor a ``SIPNETOutput``; if
        *output_variable_names* is one string, a set, is not iterable, or
        holds a name that is not a string; or if *site* or a batch label is a
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
    dataset = label_run(source.select(names), site=site, batch=batch, site_table=sites)
    dataset = _with_field_coords(dataset, tuple(batch or ()))
    return {name: dataset[name] for name in names}


def stack_sipnet_outputs(
    runs: Mapping[tuple[int, ...], SIPNETResult | SIPNETOutput],
    output_variable_names: Sequence[str],
    *,
    key_dims: Sequence[str] = (SAMPLE, SITE),
    sites: pd.DataFrame | None = None,
) -> dict[str, xr.DataArray]:
    """Many SIPNET runs, keyed by their labels, as ``(*batch, site, time)`` fields.

    Parameters
    ----------
    runs:
        A mapping from each run's labels, a tuple in *key_dims* order, to the
        run. With the default *key_dims*, ``(sample, site)``. The keys need not
        form a full rectangle; a combination left out reads as ``NaN``.
    output_variable_names:
        As for :func:`from_sipnet_output`.
    key_dims:
        What each position of a key labels, as for
        :func:`stack_model_outputs`.
    sites:
        The site table. Read once from disk when omitted.

    Returns
    -------
    dict
        Keyed by pySIPNET registry name, in the order requested. Each value is
        a ``DataArray`` with dims ``(*batch, site, time)``, the batch dims in
        *key_dims* order, ascending in every label, with ``lon``/``lat`` on
        ``site`` and the ``time`` coordinates of
        :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`.

    Raises
    ------
    TypeError
        If *runs* is not a mapping; if a value is neither a ``SIPNETResult``
        nor a ``SIPNETOutput``; if *output_variable_names* or *key_dims* is not
        an ordered sequence of names; if a key is not a tuple, or a label in
        one is a boolean, a float or not an integer; or if *sites* is not a
        ``DataFrame``.
    ValueError
        If *runs* is empty, or a key does not hold one label per key dim; and
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
    they differ, and the result is no longer a field until one site is
    selected.

    Each run is read and reduced to the variables asked for before the next is
    touched, so what is held is one column per run and variable, never a run's
    whole frame.
    """
    check_is_a_nonempty_mapping(runs, "runs")
    names = resolve_output_variable_names(output_variable_names)
    dims = _as_key_dims(key_dims)
    table = site_lookup(sites if sites is not None else load_sites())
    model_outputs = {
        _run_key(key, dims): _output_of(run).select(names) for key, run in runs.items()
    }
    stacked = stack_model_outputs(model_outputs, key_dims=dims, site_table=table)
    return {name: stacked[name] for name in names}


def stack_model_outputs(
    model_outputs: Mapping[tuple[int, ...], xr.Dataset],
    *,
    key_dims: Sequence[str] = (SAMPLE, SITE),
    site_table: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Many runs' output Datasets, keyed by their labels, as one Dataset.

    Parameters
    ----------
    model_outputs:
        A mapping from each run's labels, a tuple in *key_dims* order, to that
        run's output ``Dataset``, from ``result.outputs.select(names)`` or
        :func:`label_run`, every run carrying the same variables with the same
        ``units``, ``constituent`` and ``kind``. A run already labeled by
        :func:`label_run` must carry the labels of its key. The keys need not
        form a full rectangle; a combination left out reads as ``NaN``.
    key_dims:
        What each position of a key labels: ``site`` once, and a batch dim
        name for every other position, such as ``("sample", "site")`` or
        ``("sample", "driver_member", "site")``. ``("site",)`` stacks one run
        per site.
    site_table:
        The site table, for ``lon``/``lat``. Read once from disk when omitted.

    Returns
    -------
    xarray.Dataset
        The runs' variables on ``(*batch, site, time)``, the batch dims in
        *key_dims* order, ascending in each batch dim (``int64``, with the
        attributes :func:`batch_coordinate` gives its name) and in
        ``site`` (``int32``), with ``lon``/``lat`` (``float64``, CF
        attributes) on ``site`` and the ``time`` coordinates of
        :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`. The
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
        ``xr.Dataset``; if *key_dims* is not an ordered sequence of names; if
        a key is not a tuple, or a label in one is a boolean, a float or not
        an integer; or if *site_table* is not a ``DataFrame``.
    ValueError
        If *model_outputs* is empty; if *key_dims* does not name ``site``
        exactly once, repeats a name, or names a reserved name or a
        variable, dim or coordinate of a run; if a key does not hold one
        label per key dim, a site id is out of range or a batch label does
        not fit ``int64``; if a run has no ``time`` rows; if a run's own label
        disagrees with its key, or it carries a batch label *key_dims* does
        not name; if two runs carry different variables, or
        describe one with different ``units``, ``constituent`` or ``kind``; or
        if the site table lists a site twice or has no ``lon`` and ``lat``
        columns.
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
    dims = _as_key_dims(key_dims)
    by_key = _model_outputs_by_key(model_outputs, dims)
    table = site_table if site_table is not None else load_sites()
    site_position = dims.index(SITE)
    # Checked before stacking, so a site the table lacks fails before the work.
    check_site_table_locates_the_sites(table, sorted({key[site_position] for key in by_key}))
    batch = tuple(d for d in dims if d != SITE)
    # Nested outermost first, site innermost: each level concatenates the
    # level below along its dim, so a combination left out is filled by the
    # outer join.
    order = (*batch, SITE)
    entries = [
        (tuple(key[dims.index(d)] for d in order), _labeled_for_stacking(dataset, dims, key))
        for key, dataset in by_key.items()
    ]
    stacked = _stacked_along_nested(entries, order).transpose(*batch, SITE, TIME, ...)
    # lon/lat are assigned after stacking rather than left to xarray.concat,
    # which promotes a scalar coordinate to the concatenated dimension only
    # when the values it is given differ, so a one-site or one-sample stack
    # would otherwise keep them scalar and break the convention. They are
    # looked up for the stack's own sites, in its order, since they assign
    # by position.
    return stacked.assign_coords(site_locations(stacked[SITE].values.tolist(), table))


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

#: The scalar location coordinates :func:`label_run` adds beside ``site``.
_LOCATION_COORD_NAMES: tuple[str, ...] = (SITE, LON, LAT)

#: The attributes that say what a variable is, which every stacked run must
#: agree on.
_VARIABLE_IDENTITY_ATTRIBUTE_NAMES: tuple[str, ...] = ("units", "constituent", "kind")

#: The sets of spatial dims a field may have: none, one of ``site`` and
#: ``point``, or a raster pair.
_SPATIAL_DIM_SETS: tuple[frozenset[str], ...] = (
    frozenset(),
    frozenset({SITE}),
    frozenset({POINT}),
    frozenset({LAT, LON}),
    frozenset({Y, X}),
)

#: The coordinates that describe an interval on ``time``, and so must be on
#: ``time`` alone: pySIPNET's timestep coordinates and an observation's window.
_INTERVAL_COORD_NAMES: tuple[str, ...] = (TIMESTEP_START, TIMESTEP_LENGTH, WINDOW_START, WINDOW_END)


def _is_batch_dim(field: xr.DataArray | xr.Dataset, dim: str) -> bool:
    """Whether *dim* of *field* is a batch dim: not reserved, integer labels."""
    if dim in SPATIAL_DIM_NAMES or dim == TIME or dim not in field.indexes:
        return False
    return field.indexes[dim].dtype.kind in "iu"


def _is_categorical(field: xr.DataArray) -> bool:
    """Whether *field* holds classes: CF flags, strings, bytes or booleans."""
    if {"flag_values", "flag_meanings"} & set(field.attrs) or field.dtype.kind in "USb":
        return True
    if field.dtype.kind == "O":
        return all(isinstance(value, str) for value in np.asarray(field.values).ravel())
    return False


def _dim_rank(field: xr.DataArray, dim: str) -> int | None:
    """0 for a batch dim, 1 for a spatial dim, 2 for ``time``, ``None`` for any other."""
    if dim == TIME:
        return 2
    if dim in SPATIAL_DIM_NAMES:
        return 1
    return 0 if _is_batch_dim(field, dim) else None


def _batch_coordinate_attributes(dim: str) -> dict[str, Any]:
    """The attributes a batch coordinate named *dim* carries."""
    if dim == SAMPLE:
        return dict(SAMPLE_ATTRIBUTES)
    if dim in DATA_SOURCE_MEMBER_NAMES:
        return dict(DATA_SOURCE_MEMBER_ATTRIBUTES)
    return {}


def _companion_coordinates(field: xr.DataArray, dims: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """The non-index coordinates of *field* on stacked dims alone, with those dims."""
    return {
        str(name): tuple(str(d) for d in coordinate.dims)
        for name, coordinate in field.coords.items()
        if name not in field.dims and coordinate.dims and set(coordinate.dims) <= set(dims)
    }


def _recorded_companions(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """The companion coordinates a stacked coordinate's attributes record."""
    entries = str(record.get(STACKED_COMPANIONS_ATTRIBUTE, "")).split()
    return {
        name: tuple(on.split(",")) for name, on in (entry.split(":", 1) for entry in entries)
    }


def _companion_on(
    saved: Mapping[str, xr.DataArray],
    name: str,
    on: tuple[str, ...],
    unstacked: xr.DataArray,
) -> xr.DataArray:
    """A companion coordinate put back on its own dims, read row by row from the stack."""
    values = saved[name]
    labels = [saved[f"{dim}{STACKED_LABEL_SUFFIX}"].values for dim in on]
    by_labels = pd.Series(values.values, index=pd.MultiIndex.from_arrays(labels, names=on))
    by_labels = by_labels[~by_labels.index.duplicated()]
    wanted = pd.MultiIndex.from_product([unstacked.indexes[dim] for dim in on], names=on)
    restored = by_labels.reindex(wanted).to_numpy().reshape([unstacked.sizes[dim] for dim in on])
    if not pd.isna(restored).any():
        restored = restored.astype(values.dtype)
    return xr.DataArray(restored, dims=on, attrs=dict(values.attrs))


def _with_stack_record_from(field: xr.DataArray, labels_from: Any) -> xr.DataArray:
    """*field* with the stack record and the label coordinates of *labels_from*."""
    check_labels_from_is_a_dataarray(labels_from)
    source_dim = (
        str(labels_from.name) if labels_from.name in labels_from.dims else _stacked_dim(labels_from)
    )
    check_labels_from_is_on_the_field_dim(field, labels_from, source_dim)
    source = labels_from[source_dim]
    copied = {
        str(name): (source_dim, coordinate.values, dict(coordinate.attrs))
        for name, coordinate in source.coords.items()
        if name != source_dim and coordinate.dims == (source_dim,)
    }
    coordinate = field[source_dim].copy()
    coordinate.attrs.update(
        {
            key: source.attrs[key]
            for key in (STACKED_DIMS_ATTRIBUTE, STACKED_COMPANIONS_ATTRIBUTE)
            if key in source.attrs
        }
    )
    return field.assign_coords({source_dim: coordinate, **copied})


def _stacked_dim(field: xr.DataArray) -> str:
    """The one batch dim of *field* whose coordinate records a stack."""
    recording = [
        dim for dim in batch_dims(field) if STACKED_DIMS_ATTRIBUTE in field[dim].attrs
    ]
    check_one_batch_dim_is_stacked(field, recording, field_label(field))
    return recording[0]


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


def _run_label_coords(
    *, site: int | None, batch: Mapping[str, Any] | None, site_table: pd.DataFrame | None
) -> dict[str, xr.DataArray]:
    """Scalar ``site``/``lon``/``lat`` and batch-label coordinates for the labels given."""
    coords: dict[str, xr.DataArray] = {}
    if batch is not None:
        for dim, label in batch.items():
            coords[dim] = batch_coordinate(dim, as_batch_label(label, message_name=f"batch[{dim!r}]"))
    if site is not None:
        site_id = as_site_id(site, message_name="site")
        table = site_table if site_table is not None else load_sites()
        located = site_coordinates([site_id], table)
        coords.update({name: coordinate.isel({SITE: 0}) for name, coordinate in located.items()})
    return coords


def _as_key_dims(key_dims: Any) -> tuple[str, ...]:
    """*key_dims* as a tuple of names, ``site`` once and batch dim names otherwise."""
    dims = as_names(key_dims, message_name="key_dims")
    check_key_dims_name_the_site_once(dims)
    for dim in dims:
        if dim != SITE:
            check_batch_dim_name_is_not_reserved(dim, message_name="key_dims")
    return dims


def _run_key(key: Any, key_dims: tuple[str, ...]) -> tuple[int, ...]:
    """*key* as a tuple of plain integers, one per key dim."""
    check_key_has_one_label_per_key_dim(key, key_dims)
    return tuple(
        as_site_id(label, message_name=SITE)
        if dim == SITE
        else as_batch_label(label, message_name=dim)
        for dim, label in zip(key_dims, key)
    )


def _model_outputs_by_key(
    model_outputs: Mapping[Any, Any], key_dims: tuple[str, ...]
) -> dict[tuple[int, ...], xr.Dataset]:
    """*model_outputs* keyed by plain-integer tuples, ascending, each checked."""
    by_key: dict[tuple[int, ...], xr.Dataset] = {}
    for key in sorted(model_outputs, key=lambda k: _run_key(k, key_dims)):
        labels = _run_key(key, key_dims)
        dataset = model_outputs[key]
        check_is_a_dataset(dataset)
        check_run_has_rows(dataset)
        for dim in key_dims:
            if dim != SITE:
                check_batch_name_is_not_the_model_outputs(dataset, dim)
        check_run_labels_match_the_key(dataset, key_dims, labels)
        check_run_batch_labels_are_key_dims(dataset, key_dims, labels)
        by_key[labels] = dataset
    check_model_outputs_carry_the_same_variables(by_key, key_dims)
    return by_key


def _labeled_for_stacking(
    dataset: xr.Dataset, key_dims: tuple[str, ...], labels: tuple[int, ...]
) -> xr.Dataset:
    """*dataset* with only a field's coordinates and a scalar coordinate per key dim."""
    dropped = dataset.drop_vars([*_LOCATION_COORD_NAMES, *key_dims], errors="ignore")
    labeled = _with_field_coords(dropped, ())
    coords = {
        dim: (
            xr.DataArray(SITE_DTYPE(label), attrs=SITE_ATTRIBUTES)
            if dim == SITE
            else batch_coordinate(dim, label)
        )
        for dim, label in zip(key_dims, labels)
    }
    return labeled.assign_coords(coords)


def _with_field_coords(dataset: xr.Dataset, batch_names: tuple[str, ...]) -> xr.Dataset:
    """*dataset* with only the coordinates a field keeps.

    Those are :data:`~sipnet_calibration.conventions.TIME_COORD_NAMES`, the
    location coordinates and the batch labels named. ``time``'s ``bounds``
    attribute goes too, since the ``time_bounds`` variable it names is one of
    the coordinates dropped.
    """
    keep = {*TIME_COORD_NAMES, *_LOCATION_COORD_NAMES, *batch_names}
    # The copy gives this dataset its own variables, so rewriting an attribute
    # below leaves the caller's dataset untouched.
    dataset = dataset.drop_vars([str(c) for c in dataset.coords if str(c) not in keep]).copy()
    dataset[TIME].attrs = without_stale_time_attributes(dataset[TIME].attrs)
    return dataset


def _stacked_along_nested(
    entries: list[tuple[tuple[int, ...], xr.Dataset]], dims: tuple[str, ...]
) -> xr.Dataset:
    """*entries* stacked along *dims*, the first outermost.

    Each entry is ``(labels, dataset)``, the labels in *dims* order and each
    dataset carrying them as scalar coordinates. Groups are stacked in
    ascending label order.
    """
    if len(dims) == 1:
        return _stack_along([dataset for _, dataset in sorted(entries, key=lambda e: e[0])], dims[0])
    groups: dict[int, list[tuple[tuple[int, ...], xr.Dataset]]] = {}
    for labels, dataset in entries:
        groups.setdefault(labels[0], []).append((labels[1:], dataset))
    inner = [_stacked_along_nested(group, dims[1:]) for _, group in sorted(groups.items())]
    return _stack_along(inner, dims[0])


def _stack_along(datasets: list[xr.Dataset], dim: str) -> xr.Dataset:
    """*datasets* stacked along a new *dim*, from the scalar *dim* coordinate each carries.

    ``coords="different"`` is what gives the interval coordinates a ``site``
    or batch dimension when the runs disagree about them, rather than
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


def _key_label(key_dims: tuple[str, ...], key: tuple[int, ...]) -> str:
    """A run's key as a message names it, ``(sample=3, site=27)``."""
    return "(" + ", ".join(f"{dim}={label}" for dim, label in zip(key_dims, key)) + ")"


# ── checks ────────────────────────────────────────────────────────────────────


def check_field_is_a_dataarray(field: Any, message_name: str | None = None) -> None:
    """*field* is an ``xr.DataArray``."""
    if not isinstance(field, xr.DataArray):
        prefix = f"{message_name}: " if message_name else ""
        raise TypeError(
            f"{prefix}a field is an xarray DataArray, got {type(field).__name__}; "
            "pass one variable, such as dataset[name]."
        )


def check_field_dims_are_field_dims(field: xr.DataArray, message_name: str) -> None:
    """Every dim is a batch dim, a spatial dim or ``time``, in that order, one space."""
    check_dims_are_batch_spatial_or_time(field, message_name=message_name)
    dims = [str(d) for d in field.dims]
    ranks = [_dim_rank(field, d) for d in dims]
    if POINT in dims and field.indexes[POINT].dtype.kind not in "iu":
        raise ValueError(
            f"{message_name}: point labels are integers, got {field.indexes[POINT].dtype}; "
            "label the points 0 to n - 1 and keep any names as a coordinate on point."
        )
    spatial = frozenset(d for d in dims if d in SPATIAL_DIM_NAMES)
    if spatial not in _SPATIAL_DIM_SETS:
        raise ValueError(
            f"{message_name}: spatial dims {sorted(spatial)}; a field has at most one: "
            "site, point, or a raster pair (lat, lon) or (y, x). Select the others away."
        )
    if ranks != sorted(ranks):
        raise ValueError(
            f"{message_name}: dims {tuple(dims)} are not in the order (*batch, space, "
            "time); transpose it, with .transpose(*batch_dims, space, 'time')."
        )
    for dim in batch_dims(field):
        if field.indexes[dim].has_duplicates:
            raise ValueError(
                f"{message_name}: the batch dim {dim!r} repeats a label; batch labels are "
                "distinct integers."
            )


def check_dims_are_batch_spatial_or_time(field: xr.DataArray, *, message_name: str) -> None:
    """Every dim of *field* is labeled and is a batch dim, a spatial dim or ``time``.

    The part of the field contract that holds in any dim order, which the
    vectors' ``flat`` and :func:`sipnet_calibration.parameter_vector.sipnet_overrides`
    apply to what they read.
    """
    dims = [str(d) for d in field.dims]
    unindexed = [d for d in dims if d not in field.indexes]
    if unindexed:
        raise ValueError(
            f"{message_name}: dim(s) {unindexed} carry no coordinate; a field labels every "
            "dim (a batch dim with integers), so a structural axis such as bounds is "
            "not a dim of a field. Label it, or split it into a dict of fields."
        )
    structural = [d for d in dims if _dim_rank(field, d) is None]
    if structural:
        dtypes = {d: str(field.indexes[d].dtype) for d in structural}
        raise ValueError(
            f"{message_name}: dim(s) {structural} are neither a batch dim (a dim whose "
            f"coordinate holds integers), a spatial dim nor time; they are labeled "
            f"{dtypes}. A structural axis (variable, quantile, a PFT class) is never a "
            "dim of a field: select it away, or split it into a dict of fields."
        )


def check_field_site_holds_site_ids(field: xr.DataArray, message_name: str) -> None:
    """A ``site`` coordinate, a dim or a scalar, holds unique ``int32`` site ids."""
    if SITE not in field.coords:
        return
    site = field[SITE]
    if site.dtype != SITE_DTYPE:
        raise ValueError(
            f"{message_name}: site ids are {np.dtype(SITE_DTYPE)}, got {site.dtype}; cast "
            "the coordinate with .assign_coords(site=field.site.astype('int32'))."
        )
    if SITE in field.dims and field.indexes[SITE].has_duplicates:
        raise ValueError(
            f"{message_name}: the site coordinate repeats a site id; a field names each "
            "site once."
        )


def check_field_locations_are_on_the_spatial_dim(field: xr.DataArray, message_name: str) -> None:
    """``lon``/``lat`` are ``float64`` on a ``site`` or ``point`` dim, or scalars beside one.

    A ``site`` or ``point`` dim carries them on it alone; a scalar ``site`` or
    ``point`` carries them as scalars; with neither, they may only be
    scalars (or, on a projected raster, on its ``y``/``x`` dims), never on a
    batch dim or ``time``.
    """
    for dim in (SITE, POINT):
        if dim in field.dims:
            for name in (LON, LAT):
                check_field_location_is_on_the_dim(field, name, dim, message_name)
            return
    for dim in (SITE, POINT):
        if dim in field.coords:
            for name in (LON, LAT):
                check_field_location_is_a_scalar(field, name, dim, message_name)
            return
    raster = {str(d) for d in field.dims} & {X, Y}
    for name in (LON, LAT):
        if name in field.coords and name not in field.dims:
            coordinate = field[name]
            if not set(map(str, coordinate.dims)) <= raster:
                raise ValueError(
                    f"{message_name}: {name!r} is on {coordinate.dims}; a location is on the "
                    "spatial dim, so select a batch dim away before placing it elsewhere, "
                    "or drop the coordinate."
                )


def check_field_location_is_on_the_dim(
    field: xr.DataArray, name: str, dim: str, message_name: str
) -> None:
    """*name* (``lon`` or ``lat``) is a ``float64`` coordinate on *dim* alone."""
    if name not in field.coords:
        raise ValueError(
            f"{message_name}: the {dim} dim carries no {name!r} coordinate; locate "
            f"the {dim}s, with sites.site_coordinates for site ids."
        )
    coordinate = field[name]
    if coordinate.dims != (dim,) or coordinate.dtype != np.float64:
        raise ValueError(
            f"{message_name}: {name!r} must be float64 on ({dim},), got "
            f"{coordinate.dtype} on {coordinate.dims}."
        )


def check_field_location_is_a_scalar(
    field: xr.DataArray, name: str, dim: str, message_name: str
) -> None:
    """Beside a scalar *dim* coordinate, *name* (``lon`` or ``lat``) is a ``float64`` scalar."""
    if name not in field.coords:
        raise ValueError(
            f"{message_name}: a scalar {dim} carries a scalar {name!r} coordinate, and this "
            f"field has none; keep it when selecting one {dim} (.isel and .sel do), or "
            "locate it with sites.site_coordinates."
        )
    coordinate = field[name]
    if coordinate.dims != () or coordinate.dtype != np.float64:
        raise ValueError(
            f"{message_name}: {name!r} must be a float64 scalar beside the scalar {dim}, got "
            f"{coordinate.dtype} on {coordinate.dims}."
        )


def check_field_time_is_a_time_axis(field: xr.DataArray, message_name: str) -> None:
    """``time`` is naive ``datetime64``, strictly increasing, with no ``NaT``."""
    if TIME not in field.dims:
        return
    time = field[TIME]
    # Any datetime64 unit: pandas 3 and xarray make microseconds by default,
    # and numpy compares labels across units, so the unit carries no meaning.
    # A time zone aware axis has a pandas dtype, not a numpy one.
    if not isinstance(time.dtype, np.dtype) or time.dtype.kind != "M":
        raise ValueError(
            f"{message_name}: time labels must be naive datetime64, got {time.dtype}; "
            "convert them to the model's clock and drop any time zone."
        )
    values = time.values
    if np.isnat(values).any():
        raise ValueError(f"{message_name}: the time coordinate holds NaT; drop those labels.")
    if values.size > 1 and not (np.diff(values) > np.timedelta64(0, "ns")).all():
        raise ValueError(
            f"{message_name}: time labels are not strictly increasing; sort them with "
            ".sortby('time') and combine any repeated label."
        )


def check_field_interval_coordinates_are_on_time(field: xr.DataArray, message_name: str) -> None:
    """pySIPNET's timestep coordinates and the window coordinates are on ``time`` alone.

    On ``(time,)`` when ``time`` is a dim, and scalars when it is not, as
    ``.isel(time=k)`` leaves them.
    """
    expected = (TIME,) if TIME in field.dims else ()
    for name in _INTERVAL_COORD_NAMES:
        if name in field.coords and field[name].dims != expected:
            raise ValueError(
                f"{message_name}: {name!r} is on {field[name].dims}, not on {expected} alone; "
                "an interval belongs to one time label, and runs on different time axes "
                "give it more dims when stacked, so select one site (or one batch label) "
                "first."
            )


def check_field_units_are_valid(field: xr.DataArray, message_name: str) -> None:
    """``units`` is present and valid by pySIPNET, unless the field is categorical.

    A categorical field has CF ``flag_values`` or ``flag_meanings``, or values
    that are strings (``U``, ``S``, or an object array holding only strings),
    bytes or booleans; it has classes, not units.
    """
    if _is_categorical(field):
        return
    units = field.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            f"{message_name}: a field carries its units in attrs['units']; set them, "
            "'1' for a dimensionless quantity."
        )
    try:
        validate_units(units)
    except ValueError as error:
        raise ValueError(f"{message_name}: {error}") from error


def check_field_has_a_batch_dim(dims: tuple[str, ...], message_name: str) -> None:
    """A field to stack has at least one batch dim."""
    if not dims:
        raise ValueError(f"{message_name}: the field has no batch dim to stack.")


def check_stack_names_are_free(
    field: xr.DataArray, into: Any, dims: tuple[str, ...], label_names: tuple[str, ...]
) -> None:
    """The stacked dim's name is new, and its label coordinates' names are free on *field*."""
    check_batch_dim_name_is_not_reserved(into, message_name="into")
    if into in dims:
        raise ValueError(
            f"into={into!r} is one of the dims stacked ({list(dims)}); a stacked dim is a new "
            "index, labeled 0 to n - 1, so it takes a new name, such as 'run'."
        )
    if into in label_names:
        raise ValueError(
            f"into={into!r} is one of the names the stacked labels would take "
            f"({list(label_names)}); stack into another name, such as 'run'."
        )
    if into in field.coords:
        raise ValueError(
            f"into={into!r} is a coordinate of the field already; drop it, or stack into "
            "another name."
        )
    taken = [name for name in label_names if name in field.coords]
    if taken:
        raise ValueError(
            f"the field carries {taken} already, the names the stacked labels would take; "
            "unstack it first, or drop them."
        )


def check_one_batch_dim_is_stacked(
    field: xr.DataArray, recording: Sequence[str], message_name: str
) -> None:
    """Exactly one batch dim's coordinate records a stack (:data:`STACKED_DIMS_ATTRIBUTE`)."""
    if len(recording) == 1:
        return
    if recording:
        raise ValueError(
            f"{message_name}: the batch dims {list(recording)} each record a stack; unstack "
            "one at a time, selecting the others away."
        )
    labeled = [
        str(name) for name in field.coords if str(name).endswith(STACKED_LABEL_SUFFIX)
    ]
    hint = (
        f" It carries {labeled}, so it was stacked and an operation dropped the "
        f"{STACKED_DIMS_ATTRIBUTE!r} attribute of the stacked coordinate; pass "
        "labels_from=<the stacked field>, or restack the original."
        if labeled
        else " Pass labels_from=<the stacked field> for a field a vector's fields() made "
        "from Flat."
    )
    raise ValueError(
        f"{message_name}: no batch dim records a stack in {STACKED_DIMS_ATTRIBUTE!r}, as "
        f"stack_batch_dims leaves it.{hint}"
    )


def check_stack_labels_are_present(
    field: xr.DataArray, stacked_dim: str, label_names: Sequence[str]
) -> None:
    """Every ``<dim>_label`` coordinate the stack records is on the stacked dim."""
    missing = [
        name for name in label_names
        if name not in field.coords or field[name].dims != (stacked_dim,)
    ]
    if missing:
        raise ValueError(
            f"the stacked dim {stacked_dim!r} records {list(label_names)} but lacks "
            f"{missing}; pass labels_from=<the stacked field> to copy them over."
        )


def check_stacked_rows_are_distinct(field: xr.DataArray, label_names: Sequence[str]) -> None:
    """No two rows of the stacked dim carry the same labels, which one cell cannot hold."""
    rows = pd.MultiIndex.from_arrays([field[name].values for name in label_names])
    if rows.has_duplicates:
        raise ValueError(
            f"two rows of the stack carry the same labels {rows[rows.duplicated()][0]} of "
            f"{list(label_names)}; drop the repeats before unstacking."
        )


def check_labels_from_is_a_dataarray(labels_from: Any) -> None:
    """*labels_from* is a stacked field or its stacked coordinate."""
    if not isinstance(labels_from, xr.DataArray):
        raise TypeError(
            f"labels_from must be the stacked field or its stacked coordinate, a DataArray, "
            f"got {type(labels_from).__name__}; for a Dataset or dict pass one array at a time."
        )


def check_labels_from_is_on_the_field_dim(
    field: xr.DataArray, labels_from: xr.DataArray, source_dim: str
) -> None:
    """*field* has *labels_from*'s stacked dim, with the same labels in the same order."""
    if source_dim not in field.dims:
        raise ValueError(
            f"labels_from is stacked on {source_dim!r}, which the field lacks (it has "
            f"{list(field.dims)}); make the field with batch_dim={source_dim!r}."
        )
    if not np.array_equal(field[source_dim].values, labels_from[source_dim].values):
        raise ValueError(
            f"the field's {source_dim} labels are not the rows of labels_from; unstack the "
            "field made from the same rows as the stacked one."
        )


def check_at_most_one_batch_dim(dims: Sequence[str], *, message_name: str) -> None:
    """Something to flatten has at most one batch dim, since Flat has one row axis."""
    if len(dims) > 1:
        raise ValueError(
            f"{message_name} carry the batch dims {list(dims)}, and Flat has one row axis; "
            "reduce all but one, or stack them into a new dim with "
            "fields.stack_batch_dims(field, into='run'): for a Dataset, "
            "dataset.map(lambda field: stack_batch_dims(field, into='run')), and for a "
            "dict, one call per entry."
        )


def check_batch_dim_name_is_not_reserved(name: Any, *, message_name: str) -> None:
    """*name* can name a batch dim: a string, not a spatial name, ``time`` or ``source_index``."""
    if not isinstance(name, str):
        raise TypeError(f"{message_name}: a batch dim name is a string, got {type(name).__name__}.")
    if name in SPATIAL_DIM_NAMES or name in (TIME, SOURCE_INDEX) or not name:
        raise ValueError(
            f"{message_name}: {name!r} cannot name a batch dim; {list(SPATIAL_DIM_NAMES)}, "
            f"{TIME!r} and {SOURCE_INDEX!r} are reserved. Name it for what it indexes, such "
            "as 'sample'."
        )


def check_batch_labels_are_a_mapping(batch: Any) -> None:
    """*batch* is a mapping from batch dim to label."""
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"batch must be a mapping from batch dim to label, such as {{'sample': 3}}, "
            f"got {type(batch).__name__}."
        )


def check_key_dims_name_the_site_once(key_dims: tuple[str, ...]) -> None:
    """*key_dims* names ``site`` exactly once and no name twice."""
    if key_dims.count(SITE) != 1 or len(set(key_dims)) != len(key_dims):
        raise ValueError(
            f"key_dims must name 'site' once and each batch dim once, got {list(key_dims)}; "
            "for example ('sample', 'site')."
        )


def check_key_has_one_label_per_key_dim(key: Any, key_dims: tuple[str, ...]) -> None:
    """A run's key is a tuple of one label per key dim."""
    if not isinstance(key, tuple):
        raise TypeError(
            f"every key of the runs must be a tuple of labels in key_dims order "
            f"{tuple(key_dims)}, got {type(key).__name__} {key!r}; key each run by the "
            "labels it was, such as (3, 27)."
        )
    if len(key) != len(key_dims):
        raise ValueError(
            f"every key of the runs must be a tuple of labels in key_dims order "
            f"{tuple(key_dims)}, got {key!r}; key each run by the labels it was."
        )


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
            "stderr. Stacking an ensemble hits this on the first run that did not complete."
        )


def check_names_are_given(names: list[str]) -> None:
    """At least one output variable is asked for."""
    if not names:
        raise ValueError(
            "no variables were asked for; name at least one SIPNET output "
            "variable, e.g. output_variable_names=['nee']."
        )


def check_is_a_nonempty_mapping(value: Any, message_name: str) -> None:
    """*value* is a mapping from run keys with at least one entry."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{message_name} must be a mapping from each run's labels to the run, not "
            f"{type(value).__name__}; key each run by its labels in key_dims order."
        )
    if not value:
        raise ValueError(f"{message_name} is empty; there is nothing to stack.")


def check_run_labels_match_the_key(
    dataset: xr.Dataset, key_dims: tuple[str, ...], key: tuple[int, ...]
) -> None:
    """A run's own labels, where it carries them, are its key's."""
    for name, expected in zip(key_dims, key):
        if name not in dataset.coords:
            continue
        labels = coordinate_labels(dataset[name])
        if labels != [expected]:
            raise ValueError(
                f"the run keyed {_key_label(key_dims, key)} is labeled {name}={labels}; "
                "key each run by the labels it was."
            )


def check_batch_name_is_not_the_model_outputs(dataset: xr.Dataset, name: str) -> None:
    """A batch name is no variable, dim or coordinate of the run, bar its own label."""
    own = {*scalar_batch_labels(dataset), SITE}
    what = (
        "a variable"
        if name in dataset.data_vars
        else "a dim"
        if name in dataset.dims
        else "a coordinate"
        if name in dataset.coords and name not in own
        else None
    )
    if what is not None:
        raise ValueError(
            f"batch dim {name!r} is {what} of the model output; labeling a run with it "
            "would replace that. Name the batch dim for what it indexes, such as 'sample'."
        )


def check_run_batch_labels_are_key_dims(
    dataset: xr.Dataset, key_dims: tuple[str, ...], key: tuple[int, ...]
) -> None:
    """Every batch label a run carries is one of the key dims, so the stack keeps it."""
    dropped = [name for name in scalar_batch_labels(dataset) if name not in key_dims]
    if dropped:
        labels = {name: coordinate_labels(dataset[name])[0] for name in dropped}
        raise ValueError(
            f"the run keyed {_key_label(key_dims, key)} is labeled {labels}, and key_dims "
            f"{list(key_dims)} does not name it, so the stack would lose it; add it to "
            "key_dims and to the key, or drop the label."
        )


def check_model_outputs_carry_the_same_variables(
    model_outputs: Mapping[tuple[int, ...], xr.Dataset], key_dims: tuple[str, ...]
) -> None:
    """Every run carries the same variables, each with the same identity attributes.

    A run missing a variable would be filled with ``NaN`` by the stack, and
    the stack takes the first run's attributes, so a second run's different
    ``units`` would be relabeled silently.
    """
    (first_key, first), *rest = model_outputs.items()
    expected = _variable_identities(first)
    first_label = _key_label(key_dims, first_key)
    for key, dataset in rest:
        found = _variable_identities(dataset)
        label = _key_label(key_dims, key)
        if set(found) != set(expected):
            raise ValueError(
                f"the run keyed {label} carries the variables {sorted(found)} "
                f"and the run keyed {first_label} {sorted(expected)}; select "
                "the same names from every run, with result.outputs.select(names)."
            )
        for name in expected:
            if found[name] != expected[name]:
                described = dict(zip(_VARIABLE_IDENTITY_ATTRIBUTE_NAMES, found[name]))
                first_described = dict(zip(_VARIABLE_IDENTITY_ATTRIBUTE_NAMES, expected[name]))
                raise ValueError(
                    f"the run keyed {label} describes {name!r} as {described} "
                    f"and the run keyed {first_label} as {first_described}; "
                    "the stack carries one set of attributes, so convert the runs to "
                    "one before stacking them."
                )

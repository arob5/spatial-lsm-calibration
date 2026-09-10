"""The array form that the plotting and inference layers read.

Data reaches this project in as many shapes as it has sources. SIPNET writes
columnar output, the meteorological drivers are text files, the initial
conditions are per-site netCDF, the observations are a csv and a set of R
objects, and the calibration returns flat blocks that carry no record of space
or time. This module defines the single form all of them are converted into,
and holds the adapters that do the converting. Everything downstream -- the
plots and the observation operator -- reads that form and nothing else, so
adding a source costs one adapter rather than a change in every consumer.

The form
--------
A **canonical field** is an ``xarray.DataArray`` holding one variable, with

* dimensions drawn from ``member``, ``site`` and ``time``, in any combination;
* ``lon`` and ``lat`` as non-dimension coordinates on ``site``, whenever
  ``site`` is a dimension;
* ``units`` and ``long_name`` in ``attrs``;
* a ``name`` that is a key of the ``VARIABLES`` registry.

Which dimensions are present depends on the quantity. A single deterministic
run is ``(time,)``, an initial-condition ensemble is ``(member, site)``, the
gap-filled NEE observations are ``(member, site, time)``, and a calibrated
per-site parameter is ``(member, site)``.

One array holds one variable, and variables are not combined into a
``Dataset``: they do not share a time axis, NEE being 3-hourly, the biomass
and leaf area constraints annual, and the initial conditions static. A group
of variables is a ``dict[str, DataArray]``, which is what the multi-variable
adapters return and what the readers in
:mod:`sipnet_calibration.drivers` and :mod:`sipnet_calibration.constraints`
already produce.

Identifiers
-----------
``site``
    The handed-down integer site id, 1 to 8000. It is a shared key with
    collaborators' files and is never renumbered; a spatially meaningful
    ordering, where one is wanted, is added as a separate coordinate. Only 185
    of the sites are Ameriflux sites, so an Ameriflux-keyed identifier cannot
    address the pool: ``ameriflux_site_id`` is a non-dimension coordinate on
    ``site``, missing for the rest. Plant functional type is not site
    metadata and is not carried here; a labeling is an experimental choice and
    lives in its own product under ``data/processed/labelings/``.
``member``
    A 0-based ensemble index, meaningful only within the source it came from.
    Whether member *i* of one source corresponds to member *i* of another is
    not established, and xarray aligns on the integer label without
    complaint, so any arithmetic across two sources needs that settled first.
``time``
    Timestamps, whose meaning is the source's and is recorded in the
    coordinate's attributes rather than assumed: the drivers label the end of
    each interval, and the annual constraints carry a nominal bookkeeping date
    rather than an observation date.

Functions
---------
:func:`validate_field`
    Check an array against the form above and raise on the first property that
    does not hold.
:func:`from_sipnet_result`, :func:`from_clim`, :func:`from_ic_store`,
:func:`from_nee_store`, :func:`from_eki_predictions`
    One adapter per source. They are also where unit conversion happens: each
    variable has one canonical unit in the registry, adapters convert into it,
    and nothing downstream reconciles units.

Notes
-----
Nothing in this module is implemented yet. The drivers and the annual
constraints already have readers of their own that produce the form described
above, so it is the remaining sources -- SIPNET output, the initial
conditions, the NEE observations and the calibration output -- that this
module is still owed for. The ``VARIABLES`` registry is issue #6.

Three traps are worth knowing before writing an adapter:

* The NEE csv carries an ``ens_mean`` column. It is a derived mean, not a
  member, and admitting it to the ``member`` dimension corrupts every quantile
  taken afterwards.
* The initial-condition netCDFs must be opened with ``decode_times=False``.
  Their time units are an unsubstituted template,
  ``"days since [year]-01-01 00:00:00 UTC"``, which no calendar library can
  parse; ``cftime`` does not help. The dimension is length one and carries no
  information (issue #3).
* :func:`from_eki_predictions` unstacks a ``(J, N)`` block with the
  ``(site, variable, time)`` index from
  :func:`sipnet_calibration.obs_ops.obs_index`. It must be the same index the
  observation operator used to build the observation vector, or the
  predictions come back mislabeled against the observations they are compared
  with.
"""

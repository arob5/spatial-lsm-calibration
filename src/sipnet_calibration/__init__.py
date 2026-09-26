"""Scalable Bayesian calibration of the SIPNET land model over many sites.

The package re-exports nothing: import from the module that owns a name,
``from sipnet_calibration.sites import load_sites``.

Modules
-------
Foundations, which everything else may import:

:mod:`~sipnet_calibration.conventions`
    Every name, coordinate attribute and setting two modules share: the dims
    ``site``, ``time`` and ``sample``, the reserved spatial names, pySIPNET's
    timestep coordinates, the window edges, ``site_id``, the CF attributes of
    ``site``/``lon``/``lat``, and where ``data/`` is
    (:func:`~sipnet_calibration.conventions.data_root`).
:mod:`~sipnet_calibration.validation`
    Argument coercion, ``as_<thing>(value, *, message_name)``: site ids,
    integers, Flat vectors and batches, boxes, names, frozen mappings.
:mod:`~sipnet_calibration.io`
    Writing a file safely through a ``.partial`` path, and the md5 and
    timestamp its provenance records.

The site table and the fields:

:mod:`~sipnet_calibration.sites`
    The site table: its schema, :func:`~sipnet_calibration.sites.load_sites`,
    selection, the lookups every module locates a site with, and
    :data:`~sipnet_calibration.sites.SITE_GRID`.
:mod:`~sipnet_calibration.projection`
    The display projection, over PROJ.
:mod:`~sipnet_calibration.fields`
    The field convention, and labeling and stacking SIPNET runs.

The data sources, each a spec, a reader, a builder, a loader and a field view:

:mod:`~sipnet_calibration.constraints`
    The five constraint data sources.
:mod:`~sipnet_calibration.initial_conditions`
    PEcAn's initial condition ensemble, and its conversion to SIPNET's.
:mod:`~sipnet_calibration.drivers`
    The ERA5 meteorological drivers, read from the raw ``.clim`` files.
:mod:`~sipnet_calibration.site_labels`
    Site labels, such as plant functional type, one data source per file.

The inverse problem:

:mod:`~sipnet_calibration.parameter_vector`
    The calibration vector: priors, bijectors and the maps to SIPNET
    parameters.
:mod:`~sipnet_calibration.observation`
    The observation vector, the observation operators and the time
    alignment they are written with.
:mod:`~sipnet_calibration.forward`
    The forward model, unconstrained parameters to predictions, run through
    PyEns.
:mod:`~sipnet_calibration.compute`
    The SCC's PyEns backend preset.

:mod:`~sipnet_calibration.plotting`
    Figures of fields: series, maps and grids of either.

Dependencies
------------
The dependency runs one way, from the foundations up::

    conventions  <-  validation, io  <-  sites  <-  fields
        <-  constraints, initial_conditions, drivers, site_labels,
            parameter_vector, observation
        <-  forward  <-  experiments

:mod:`~sipnet_calibration.projection` depends on the foundations only, and
:mod:`~sipnet_calibration.plotting` on the foundations, the site table and the
projection; nothing outside plotting imports it.
:mod:`~sipnet_calibration.compute` depends on nothing here.

Notes
-----
**The one import-time side effect.** Importing the package, and so any of its
modules, turns on JAX's 64-bit mode (``jax_enable_x64``), so every JAX array
the package makes is ``float64``, as pyEKI requires and an MCMC baseline
that never imports pyEKI still gets. The setting is per process: workers of
a process pool that compute with JAX need ``JAX_ENABLE_X64=1`` in their
environment.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

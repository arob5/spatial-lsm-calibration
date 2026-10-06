"""Scalable Bayesian calibration of the SIPNET land model over many sites.

The package re-exports nothing: import from the module that owns a name,
``from sipnet_calibration.sites import load_sites``.

Modules
-------
Foundations, which everything else may import:

:mod:`~sipnet_calibration.conventions`
    Every name, coordinate attribute and setting two modules share: the dims
    ``site``, ``time`` and ``sample``, the reserved spatial names, pySIPNET's
    timestep coordinates, the window edges, ``site_id``, ``source_index``,
    the attributes of ``site``/``lon``/``lat``/``sample`` and of a data
    source's member dim, the batch-label dtype, where ``data/`` is
    (:func:`~sipnet_calibration.conventions.data_root`), and the read-only
    copies of xarray data a frozen class keeps
    (:class:`~sipnet_calibration.conventions.ReadOnlyCopies`).
:mod:`~sipnet_calibration.validation`
    Argument coercion, ``as_<thing>(value, *, message_name)``: site ids,
    integers, batched Flat, boxes, names, frozen mappings; and the checks
    they are written with.
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
    The field contract and the forms of the model's input and output
    (``Field``, ``ModelOutput``, ``SIPNETParameterFields`` and their
    validators), batch dims and their stacking, and labeling and stacking
    SIPNET runs.

The data sources, each a spec, a reader, a builder, a loader and a field view:

:mod:`~sipnet_calibration.constraints`
    The five constraint data sources.
:mod:`~sipnet_calibration.initial_conditions`
    PEcAn's initial condition ensemble, and its conversion to SIPNET's.
:mod:`~sipnet_calibration.net_ecosystem_exchange`
    Observed net ecosystem exchange: AmeriFlux's towers, their pool sites and
    clocks, and one processed file per series.
:mod:`~sipnet_calibration.drivers`
    The ERA5 meteorological drivers, read from the raw ``.clim`` files.
:mod:`~sipnet_calibration.site_labels`
    Site labels, such as plant functional type, one data source per file.

The inverse problem:

:mod:`~sipnet_calibration.probability`
    The probability layer, independent of the rest of the package: the
    declaration of a component (``ArraySpec``) and its support, laws and
    their families, factors, deterministics and simulators joined into a
    model, bound at coords and conditioned into a posterior, the layout of
    named arrays with its three forms, the Gaussian observation model and
    the conjugate rules.
:mod:`~sipnet_calibration.site_dims`
    The sites and the dims they define: coords for a model, constants and
    label maps for its functions, and values read at the sites.
:mod:`~sipnet_calibration.sipnet_parameter_map`
    How the values at a site become SIPNET parameters: rules, fixed values
    and external inputs.
:mod:`~sipnet_calibration.calibration`
    A calibration's record, from its posterior and SIPNET parameter map,
    and an example.
:mod:`~sipnet_calibration.observation`
    The observation vector, the observation operators and the time
    alignment they are written with; its ``model`` module, which the
    package does not import, makes the sources' components for the
    probability layer.
:mod:`~sipnet_calibration.forward`
    The forward model, labeled values to predictions, run through PyEns:
    SIPNET's runs and the simulator a model reads them through.
:mod:`~sipnet_calibration.compute`
    The SCC's PyEns backend preset.

Sampling the posterior:

:mod:`~sipnet_calibration.smc`
    Tempered sequential Monte Carlo from a base density to the posterior,
    and importance sampling as its one-step case, over any prior and
    batched log likelihood; it knows nothing of SIPNET.
:mod:`~sipnet_calibration.inference`
    A posterior as each algorithm reads it: EKI's forward map, y, noise
    covariance and initial ensemble; ``smc``'s tempering problem; an MCMC
    sampler's log density and starting points.

:mod:`~sipnet_calibration.plotting`
    Figures of fields: series, maps and grids of either.

Dependencies
------------
The dependency runs one way, from the foundations up::

    probability  (imports nothing of the package)
    probability, smc, validation  <-  inference

    conventions  <-  validation  <-  sites
        <-  fields, site_labels
        <-  constraints, drivers, initial_conditions,
            net_ecosystem_exchange, site_dims
        <-  observation, sipnet_parameter_map
        <-  calibration, forward  <-  experiments

:mod:`~sipnet_calibration.probability` imports nothing of the package outside
itself, which ``tests/test_package.py`` enforces; the adapter layer
(``site_dims``, ``sipnet_parameter_map``, ``forward``) imports it, and the
seam between them is the labeled values.
:mod:`~sipnet_calibration.io` depends on nothing here, and the data sources
and :mod:`~sipnet_calibration.projection` write through it; ``site_dims``
reads ``site_labels``' column name and the site table through ``sites``.
``constraints``, ``drivers``, ``initial_conditions``,
``net_ecosystem_exchange`` and ``observation`` depend on ``fields`` (the
field contract, its batch dims, the window coordinates of the two
observation data sources, and the SIPNET parameter fields alias and
validator, which is why neither ``initial_conditions`` nor ``observation``
imports the probability layer); ``observation.model`` alone of the observation
package imports it. ``sipnet_parameter_map`` depends on ``probability``,
``site_dims``, ``fields`` and ``initial_conditions``, and is the one that
imports pySIPNET's parameter specs; ``calibration`` on it and the
probability layer; and ``forward`` on ``fields``, ``observation``,
``probability``, ``site_dims`` and ``sipnet_parameter_map``.
:mod:`~sipnet_calibration.projection` depends on ``validation`` and ``io``
only, and :mod:`~sipnet_calibration.plotting` on ``conventions``,
``validation``, ``fields``, the site table and the projection; nothing outside
plotting imports it. :mod:`~sipnet_calibration.compute` depends on nothing here,
and :mod:`~sipnet_calibration.smc` on ``io`` and ``validation`` only.
:mod:`~sipnet_calibration.inference` depends on ``probability``, ``smc`` and
``validation`` only, so it reads a posterior and nothing of SIPNET, and
imports no algorithm package.

Notes
-----
**The one import-time side effect.** Importing the package, and so any of its
modules, turns on JAX's 64-bit mode (``jax_enable_x64``), so every JAX array
the package makes is ``float64``, as EnsKit requires and an MCMC baseline
that never imports EnsKit still gets. The setting is per process. A worker of
a process pool that imports the package -- as unpickling any of its objects
does -- gets it too; only a worker that computes with JAX without importing
the package needs ``JAX_ENABLE_X64=1`` in its environment.
"""

import jax as _jax

__all__: list[str] = []

_jax.config.update("jax_enable_x64", True)

"""The parameter vector: which SIPNET parameters are calibrated, under what
prior, over which sites, and the three forms its values take.

Where this sits
---------------
pySIPNET owns the physical SIPNET parameters, their units and their domains.
pyEKI takes a Gaussian prior on unconstrained ``R^D``, an initial ``(J, D)``
ensemble drawn from it, and a callable forward model, and leaves transforms,
constraints and priors to the caller. PyEns runs an ensemble declared as
``Grid``\\ s over ``Axis``\\ es, built from a SIPNET table by
``pyens.xarray.fields_from_dataset``. This module is what sits between
them, and the dependency runs one way::

    pysipnet.parameters.model.PARAMETER_SPECS   (names, domains, units)
    pyeki.gauss / pyeki.linalg                  (Gaussian, PSDBlockDiag, ...)
    tensorflow_probability.substrates.jax       (distributions, bijectors)
      -> this module
      -> experiments/<task>/config.py           (builds a ParameterVector)

What it reads
-------------
Nothing from disk. A :class:`ParameterVector` is built over sites and site
labels the caller has already loaded:

``sites``
    Plain site ids, or a site table from
    :func:`sipnet_calibration.sites.select_sites` in ascending ``site_id``
    order, whose ``lon``/``lat`` are then carried onto every dataset the
    vector produces.
``site_labels``
    For each site-labels name used as a ``varies_by``, a site-labels product
    from :func:`sipnet_calibration.site_labels.load_site_labels`, a pandas
    categorical, or one label per site as a plain sequence.

The object
----------
A :class:`ParameterVector` is a named random vector ``theta`` in ``R^D``. It
is built from :class:`CalibrationParameter`\\ s, each with a prior in natural
space, a rule for how it varies over the sites (``varies_by``: shared, one
copy per site, or one copy per class of a site-labels product) and a
:class:`SIPNETMap` saying which SIPNET parameters it sets; and from
:class:`FixedParameter`\\ s, SIPNET parameters held at a value. For each
calibration parameter ``c``::

    theta_c  --T_c-->  vartheta_c  --M_c(., fixed)-->  {SIPNET parameter: value}

``T_c`` is the prior's bijector from unconstrained to natural space and is
invertible. ``M_c`` is the calibration parameter's SIPNET map and in general
is not, because it folds in fixed SIPNET parameters.

Vocabulary
----------
calibration parameter
    One named piece of ``theta``, ``photosynthesis``; a scalar or a short
    vector per copy.
SIPNET parameter
    One of pySIPNET's flat parameter names, ``max_photosynthesis_rate``: the
    keyword ``SIPNETModel(**overrides)`` takes.
natural space, unconstrained space
    Natural space is where a prior is named and a person reads a value: a
    rate in yr-1, a fraction, a point on the simplex. Unconstrained space is
    ``R^D``, where a sampler moves.
group
    One copy of a calibration parameter: the one copy of a shared parameter,
    one per site, or one per class of a site-labels product.
component, element
    A component is one entry of a copy's natural-space value; an element is
    one entry of its unconstrained value. They differ only for the simplex,
    whose four components have three elements.
site labels
    A site-labels product assigns one class to every site. A vector names the
    products it uses in ``site_labels={site_labels_name: ...}``, and a
    calibration parameter's ``varies_by`` names one of them.
column label
    The name of one of the ``D`` columns of ``theta``, from
    :attr:`Layout.column_labels`.

Data model
----------
A value of the vector, or an ensemble of ``J`` of them, has three
representations. Every public function says which one it takes and returns.

**Flat.** A float64 ``jax.Array``, ``(D,)`` for one value and ``(J, D)`` for
an ensemble, always in unconstrained space: a natural-space flat vector is
not defined, since the simplex has one more component than it has elements.
Columns follow :class:`Layout`: calibration parameters in declaration order;
within one, groups in group order; within a group, elements in order. Flat is
what :meth:`ParameterVector.sample` returns and what
:meth:`ParameterVector.log_prior` and pyEKI consume. NaN is never produced
and :meth:`ParameterVector.flat` refuses it.

**Fields.** An ``xarray.Dataset`` of fields (see
:mod:`sipnet_calibration.fields`):

============ ========================================================
dims         a batch dim for a batch only, ``sample`` unless
             ``batch_dim=`` names it otherwise (``int64``: 0 to J-1
             when built from Flat); ``site`` (``int32`` site ids,
             ascending)
variables    one float64 variable per scalar component in natural
             space, or per element in unconstrained space, on
             ``(sample, site)`` or ``(site,)``: ``<parameter>`` when
             there is one (``initial_soil_carbon``),
             ``<parameter>.<component>`` or ``<parameter>.<element>``
             when there are several (``allocation.leaf_allocation``,
             ``allocation.alr(leaf_allocation:coarse_root_allocation)``)
variable     ``parameter`` (the calibration parameter's name),
attributes   ``component`` (the component or element label, only when
             there are several), ``varies_by``
             (``"shared"``, ``"site"`` or a site-labels name),
             ``space``, ``long_name``, ``units`` (the component's in
             natural space, ``"1"`` in unconstrained space) and
             ``sipnet_parameters`` (what the calibration parameter
             writes, comma-separated)
coordinates  ``site``; ``lon``/``lat`` (float64) on ``site`` when the
             vector was built from a site table; one coordinate per
             site-labels name on ``site``, holding the class
attributes   ``representation = "calibration_parameters"``;
             ``space = "natural"`` or ``"unconstrained"``, mandatory
missing      impossible: a shared copy is repeated at every site and a
             per-class copy at every site of the class
============ ========================================================

``fields.filter_by_attrs(parameter="allocation")`` selects a whole
calibration parameter.

**SIPNET table.** An ``xarray.Dataset`` with the same dims and coordinates as
Fields, and one float64 variable per SIPNET parameter the vector sets,
calibrated and fixed alike, keyed on the pySIPNET name, in
``PARAMETER_SPECS`` order. Built from Fields, it keeps their batch dim and
its labels, so a subset of a batch keeps its samples' identity. Each variable carries ``units``, ``sipnet_name``,
``constituent`` where pySIPNET declares one, and ``source``
(``"parameter <name>"`` or ``"fixed"``); the dataset carries
``representation = "sipnet_parameters"``. A SIPNET parameter the vector
neither calibrates nor fixes is absent, never NaN:
:attr:`ParameterVector.unset_sipnet_parameter_names` lists them, and the run's
base parameter set supplies them.

Conversions
-----------
===================== ================== ====================================== ==========
from                  to                 call                                   lossless
===================== ================== ====================================== ==========
Flat                  Fields             ``vector.fields(theta, space=)``       yes
Fields (either space) Flat               ``vector.flat(fields)``                yes [1]_
Flat or Fields        SIPNET table       ``vector.sipnet_table(x)``             no
SIPNET table          one run's kwargs   :func:`sipnet_overrides`               selection
SIPNET table          PyEns grids        ``pyens.xarray.fields_from_dataset``   values
===================== ================== ====================================== ==========

:meth:`~ParameterVector.flat` validates: the ``space`` attribute, the
variables and sites the vector needs (others are ignored, so a larger
vector's Fields project onto a smaller one), finiteness, and that each
group's value agrees across its sites, and that each natural value is in the
image of its bijector (inside the prior's support; simplex components summing
to 1). Nothing converts a SIPNET table back.

.. [1] Up to float64 rounding, which near a logit bound grows: the inverse
   of a fraction within about ``1e-12`` of 0 or 1 loses digits, and one that
   rounds to exactly 0 or 1 is refused.

Functions and classes
---------------------
Prior helpers, each returning a TFP distribution in natural space whose
``.distribution`` is Gaussian and whose ``.bijector`` is ``T_c``:
:func:`log_normal`, :func:`log_normal_from_interval`,
:func:`log_normal_from_samples`, :func:`logit_normal`,
:func:`logit_normal_from_interval`, :func:`logit_normal_from_samples`,
:func:`softmax_normal`, :func:`product_transformed_gaussian_prior`.

SIPNET maps: the :class:`SIPNETMap` protocol, :class:`Identity`,
:class:`SimplexMap`, :class:`PhotosynthesisMap`, and the instances
:data:`ALLOCATION` and :data:`PHOTOSYNTHESIS`.

:class:`CalibrationParameter`, :class:`FixedParameter`, :class:`Layout`,
:class:`ParameterVector`; :func:`sipnet_overrides`;
:func:`example_parameter_vector`, the worked example and test fixture.

Constants: :data:`REQUIRED_SIPNET_PARAMETERS`; :data:`NATURAL`,
:data:`UNCONSTRAINED` and :data:`SPACES`, the values of ``space``;
:data:`FIELDS_REPRESENTATION` and :data:`SIPNET_TABLE_REPRESENTATION`, the
``representation`` attribute of each dataset.

Notes
-----
**One prior per calibration parameter, stored in natural space.** TFP's
``LogNormal``, ``LogitNormal`` and every ``TransformedDistribution`` expose
``.distribution`` (the unconstrained base) and ``.bijector``, so the density
of ``theta_c``, its samples and its moments are all read off the one object.
TFP does not push moments through a non-affine bijector, which is why
:meth:`ParameterVector.gaussian_prior` consults the base.

**Two prior shapes.** A prior is either *independent copies*, batch shape
``()`` (one prior shared by every group) or ``(n_groups,)`` (one per group)
with the event of one copy, or *joint over groups*, batch shape ``()`` and
event ``(n_groups,)``, one distribution over every copy of a scalar
calibration parameter. The second is how a Gaussian process over per-site
copies enters; which shape a prior has is read off its event shape against
the number of components its SIPNET map names. A joint prior for a
vector-valued calibration parameter is not supported.

**Parameter-major layout.** Each calibration parameter's copies form one
contiguous block of ``theta``, which is the block a covariance operator
wants: :meth:`ParameterVector.gaussian_prior` builds one block per
calibration parameter, and a spatial covariance over the per-site copies of
one of them replaces that block and nothing else.

**Groups of a site-labels product.** With a product from
:func:`~sipnet_calibration.site_labels.load_site_labels`, a calibration
parameter's groups are the product's declared classes that some site of the
vector carries, in the product's order. A per-class prior or fixed value may
be written over every declared class and is restricted to those present, so
a vector over the whole pool and one over a handful of sites are written the
same way. With a plain sequence the groups are its sorted distinct labels and
coverage must be exact.

**The SIPNET table holds arrays, not ``SIPNETParameters``.** Building
``J x S`` validated Pydantic models would dominate a run that is otherwise a
subprocess. ``SIPNETModel`` validates each run before invoking the binary,
and the checks a vector runs when it is built (the ``check_*`` helpers at
the bottom of this module) are what keep a prior draw from reaching that
failure.

**What crosses a process boundary.** A ``ParameterVector`` holds live TFP
objects. Everything the vector itself holds pickles -- its mappings are
:class:`~sipnet_calibration.conventions.FrozenMapping`\\ s and its arrays
plain NumPy -- so a vector whose priors are ``LogNormal`` or ``LogitNormal``
round-trips through ``pickle`` and ``copy.deepcopy``. A prior built from
``TransformedDistribution`` or ``Blockwise`` (the simplex, the product
prior, and so :func:`example_parameter_vector`) pickles but fails
``pickle.loads`` with this TFP build, and so does a vector holding one. Ship
the SIPNET table, or the ``Grid``\\ s of plain floats
``pyens.xarray.fields_from_dataset`` makes from it, never the vector.

The vector computes in float64: importing the package sets
``jax_enable_x64``, as ``import pyeki`` does, so an MCMC baseline that never
imports pyEKI still gets it (see :mod:`sipnet_calibration`). The setting is
per process, and a worker that imports the package, as unpickling anything
of it does, gets it too; only a worker that computes with JAX without
importing the package needs ``JAX_ENABLE_X64=1`` in its environment.

Usage
-----
Build a vector for three sites of the reanalysis's three-PFT site labels,
look at it, subset it, draw from it, and take the draws to SIPNET::

    import jax
    from pyens import EnsembleSpec
    from pyens.xarray import fields_from_dataset
    from sipnet_calibration.parameter_vector import (
        ALLOCATION, PHOTOSYNTHESIS, CalibrationParameter, FixedParameter,
        ParameterVector, log_normal, log_normal_from_interval, logit_normal,
        product_transformed_gaussian_prior, sipnet_overrides, softmax_normal,
    )
    from sipnet_calibration.site_labels import load_site_labels
    from sipnet_calibration.sites import load_sites, select_sites

    # Build it for a site set and a site-labels product.
    sites = select_sites(load_sites(), ids=(620, 865, 1037))  # DataFrame: site_id, lon, lat, ...
    pft = load_site_labels("reanalysis_3pft")                  # DataFrame: site_id, label
    vector = ParameterVector(                                  # ParameterVector, D = 13
        parameters=(
            CalibrationParameter(
                name="photosynthesis",
                prior=product_transformed_gaussian_prior(
                    capacity=log_normal(median=115.0, geometric_sd=1.75),
                    respiration_share=logit_normal(median=0.18, logit_sd=0.35),
                ),
                sipnet_map=PHOTOSYNTHESIS, provenance="...",
            ),
            CalibrationParameter(
                name="allocation",
                prior=softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5),
                sipnet_map=ALLOCATION, varies_by="pft", provenance="...",
            ),
            CalibrationParameter(
                name="base_soil_respiration",
                prior=log_normal_from_interval(lower=0.004, upper=0.020),
                sipnet_map="base_soil_respiration_rate", varies_by="pft",
                provenance="...",
            ),
            CalibrationParameter(
                name="initial_soil_carbon",
                prior=log_normal(median=[30_000.0] * 3, geometric_sd=2.0),
                sipnet_map="soil_carbon", varies_by="site", provenance="...",
            ),
        ),
        fixed=(
            FixedParameter(
                name="daily_mean_photosynthesis_fraction", value=0.76, provenance="...",
            ),
            FixedParameter(                    # keyed on all three classes; the
                name="leaf_carbon_fraction",   # grassland value is dropped, since
                varies_by="pft",               # no site here is grassland
                value={
                    "boreal.coniferous": 0.506,
                    "temperate.deciduous.HPDA": 0.466,
                    "semiarid.grassland_HPDA": 0.483,
                },
                provenance="...",
            ),
        ),
        sites=sites,
        site_labels={"pft": pft},
    )

    # Print it.
    print(vector)
    # ParameterVector  D = 13  |  3 sites  |  site labels: pft {boreal.coniferous, temperate.deciduous.HPDA}
    #   parameter              varies by  groups  size  prior                             -> SIPNET parameters
    #   photosynthesis         shared          1     2  product of transformed Gaussians  max_photosynthesis_rate, foliar_respiration_fraction
    #   allocation             pft             2     3  softmax-normal                    leaf_allocation, wood_allocation, fine_root_allocation
    #   base_soil_respiration  pft             2     1  log-normal                        base_soil_respiration_rate
    #   initial_soil_carbon    site            3     1  log-normal                        soil_carbon
    #   fixed: daily_mean_photosynthesis_fraction (shared), leaf_carbon_fraction (pft)
    #   unset: 39 required SIPNET parameters, taken from the run's base parameter set
    vector.dimension                                   # int: 13
    vector.layout.column_labels[:2]                    # ('photosynthesis[log(capacity)]', ...)
    vector["allocation"].components                    # ('leaf_allocation', ..., 'coarse_root_allocation')
    vector.describe()                                  # DataFrame: one row per column of theta
    vector.site_table                                  # DataFrame: site_id, lon, lat, pft

    # Subset it.
    conifer = vector.select(labels={"pft": ["boreal.coniferous"]})         # ParameterVector, D = 8
    two = vector.select(parameters=("allocation", "initial_soil_carbon"))  # ParameterVector, D = 9

    # Sample it and evaluate its density.
    theta = vector.sample(jax.random.key(0), n=50)     # Flat: jax.Array (50, 13), unconstrained
    vector.log_prior(theta)                            # jax.Array (50,)
    gaussian = vector.gaussian_prior()                 # pyeki.gauss.Gaussian over Flat

    # Look at the prior draws, one calibration parameter at a time.
    fields = vector.fields(theta)                      # Fields: Dataset (sample, site), natural space
    fields.filter_by_attrs(parameter="allocation")     # Fields: the four allocation components
    soil = fields["initial_soil_carbon"]               # DataArray (sample, site); lon, lat, pft on site
    fields["allocation.coarse_root_allocation"].sel(site=865)    # DataArray (sample,): the conifer copy
    fields["allocation.leaf_allocation"].groupby("pft").first()  # DataArray (sample, pft): one per class
    soil.quantile([0.05, 0.5, 0.95], dim="sample")     # DataArray (quantile, site)

    # Convert to SIPNET parameters, for one run and for a PyEns spec.
    table = vector.sipnet_table(theta)                 # SIPNET table: Dataset (sample, site), fixed included
    kwargs = sipnet_overrides(table, batch={"sample": 3}, site=865)  # dict[str, float]: model(**kwargs)
    grids = fields_from_dataset(table)                 # dict[str, Grid] along [sample, site]
    spec = EnsembleSpec(inputs=grids)                  # 150 runs; add a climate Grid on the site axis

    # Back from a pyEKI result, and across vectors.
    posterior = vector.fields(eki.state.ensemble)      # Fields, natural space; eki: a pyeki EKIResult
    theta_again = vector.flat(fields)                  # Flat (50, 13): theta again, to 1e-10
    conifer.flat(fields)                               # Flat (50, 8): the same draws, projected
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from pyeki.gauss import Gaussian
from pyeki.linalg import DensePSD, PSDBlockDiag, PSDDiagonal, PSDLinOp
from pysipnet.parameters.base import ParameterDomain, ParameterSpec
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from tensorflow_probability.substrates import jax as tfp

# SITE, the site dimension's name, is also the reserved ``varies_by`` value
# meaning one copy per site. Calibration parameter names match NAME_PATTERN.
from sipnet_calibration.conventions import (
    BATCH_LABEL_DTYPE,
    DATA_SOURCE_MEMBER_NAMES,
    LAT,
    LAT_ATTRIBUTES,
    LON,
    LON_ATTRIBUTES,
    NAME_PATTERN,
    POINT,
    SAMPLE,
    SITE,
    SITE_DTYPE,
    SITE_ID,
    SOURCE_INDEX,
    FrozenMapping,
    X,
    Y,
)
from sipnet_calibration.fields import (
    batch_coordinate,
    batch_dims,
    check_at_most_one_batch_dim,
    check_batch_dim_name_is_not_reserved,
    check_batch_labels_are_a_mapping,
    check_dims_are_batch_spatial_or_time,
)
from sipnet_calibration.site_labels import LABEL_COLUMN
from sipnet_calibration.sites import (
    check_site_table_has_locations,
    check_site_table_is_keyed_on_site_ids,
    site_lookup,
)
from sipnet_calibration.validation import (
    as_batched_flat,
    as_frozen_mapping,
    as_names,
    as_sequence,
    as_batch_label,
    as_site_id,
    as_site_ids,
    is_one_vector,
)

__all__ = [
    "ALLOCATION",
    "DOMAIN_CHECK_CORNERS",
    "FIELDS_REPRESENTATION",
    "NATURAL",
    "PHOTOSYNTHESIS",
    "REQUIRED_SIPNET_PARAMETERS",
    "RESERVED_PARAMETER_NAMES",
    "RESERVED_SITE_LABELS_NAMES",
    "SHARED",
    "SIPNET_TABLE_REPRESENTATION",
    "SPACES",
    "UNCONSTRAINED",
    "CalibrationParameter",
    "FixedParameter",
    "Identity",
    "Layout",
    "ParameterVector",
    "PhotosynthesisMap",
    "SIPNETMap",
    "SimplexMap",
    "example_parameter_vector",
    "log_normal",
    "log_normal_from_interval",
    "log_normal_from_samples",
    "logit_normal",
    "logit_normal_from_interval",
    "logit_normal_from_samples",
    "product_transformed_gaussian_prior",
    "sipnet_overrides",
    "softmax_normal",
]

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jax.Array

# ── prior helpers ─────────────────────────────────────────────────────────────


def log_normal(*, median: Any, geometric_sd: Any) -> tfd.LogNormal:
    """Log-normal with the given median and geometric standard deviation.

    ``log x ~ Normal(log median, log geometric_sd)``. The central 95%
    interval is ``median * geometric_sd ** (-1.96, 1.96)``, so "about 0.01,
    within a factor of 3" is ``log_normal(median=0.01, geometric_sd=3)``.

    Parameters
    ----------
    median:
        Positive scalar, or an array of one median per group.
    geometric_sd:
        Greater than 1; scalar or one per group.

    Examples
    --------
    >>> prior = log_normal(median=0.01, geometric_sd=2.0)
    >>> round(float(prior.quantile(0.5)), 6)
    0.01
    """
    median = _positive_array("log_normal median", median)
    geometric_sd = _positive_array("log_normal geometric_sd", geometric_sd)
    if not bool(jnp.all(geometric_sd > 1.0)):
        raise ValueError("log_normal: geometric_sd must exceed 1 (it multiplies).")
    return tfd.LogNormal(loc=jnp.log(median), scale=jnp.log(geometric_sd))


def log_normal_from_interval(*, lower: Any, upper: Any, mass: float = 0.95) -> tfd.LogNormal:
    """The log-normal whose central *mass* interval is ``[lower, upper]``.

    Two quantiles determine the two parameters exactly: the median is the
    geometric midpoint and the log-scale sd is set so that the interval
    carries *mass*.

    Examples
    --------
    >>> prior = log_normal_from_interval(lower=0.004, upper=0.020)
    >>> round(float(prior.quantile(0.975)), 3)
    0.02
    """
    lower = _positive_array("log_normal_from_interval lower", lower)
    upper = _positive_array("log_normal_from_interval upper", upper)
    loc, scale = _normal_from_interval(jnp.log(lower), jnp.log(upper), mass)
    return tfd.LogNormal(loc=loc, scale=scale)


def log_normal_from_samples(x: Any) -> tfd.LogNormal:
    """Maximum-likelihood log-normal fit to strictly positive samples.

    Refuses NaN or non-positive values rather than dropping them, so the
    caller counts what it excludes.
    """
    logs = jnp.log(_samples_in_support("log_normal_from_samples", x, lambda v: v > 0))
    return tfd.LogNormal(loc=jnp.mean(logs), scale=_positive_std(logs, "log_normal_from_samples"))


def logit_normal(*, median: Any, logit_sd: Any) -> tfd.LogitNormal:
    """Logit-normal on ``(0, 1)`` with the given median and logit-scale sd.

    ``logit x ~ Normal(logit median, logit_sd)``. ``logit_sd`` around 1.7 is
    close to flat on ``(0, 1)``.

    Notes
    -----
    The spread is stated on the logit scale because this family has no
    closed-form natural-space moments, so no natural-space spread statistic
    would be exact.
    """
    median = _unit_interval_array("logit_normal median", median)
    logit_sd = _positive_array("logit_normal logit_sd", logit_sd)
    return tfd.LogitNormal(loc=_logit(median), scale=logit_sd)


def logit_normal_from_interval(
    *, lower: Any, upper: Any, mass: float = 0.95
) -> tfd.LogitNormal:
    """The logit-normal whose central *mass* interval is ``[lower, upper]``."""
    lower = _unit_interval_array("logit_normal_from_interval lower", lower)
    upper = _unit_interval_array("logit_normal_from_interval upper", upper)
    loc, scale = _normal_from_interval(_logit(lower), _logit(upper), mass)
    return tfd.LogitNormal(loc=loc, scale=scale)


def logit_normal_from_samples(x: Any) -> tfd.LogitNormal:
    """Maximum-likelihood logit-normal fit to samples in ``(0, 1)``.

    Refuses NaN and values outside the open interval rather than dropping
    them.
    """
    logits = _logit(
        _samples_in_support("logit_normal_from_samples", x, lambda v: (v > 0) & (v < 1))
    )
    return tfd.LogitNormal(
        loc=jnp.mean(logits), scale=_positive_std(logits, "logit_normal_from_samples")
    )


def softmax_normal(*, center: Any, logit_sd: Any) -> tfd.TransformedDistribution:
    """A Normal on ``R^(k-1)`` pushed through ``SoftmaxCentered`` onto the
    interior of the ``k``-simplex.

    Parameters
    ----------
    center:
        Length-``k`` vector of positive fractions summing to 1, or
        ``(n_groups, k)`` for one center per group. The base is a
        ``MultivariateNormalDiag`` located at the additive log-ratios
        ``log(center[:-1] / center[-1])``, so ``bijector.forward(base.loc)``
        is *center*.
    logit_sd:
        Scale of the base: a scalar, or one value per unconstrained element
        (``k - 1`` of them). Any other shape is refused.

    Returns
    -------
    tfd.TransformedDistribution
        Event shape ``(k,)``; every draw has ``k`` strictly positive
        components summing to 1.

    Notes
    -----
    Aitchison's name for this family is the logistic-normal; the logit-normal
    is its ``k = 2`` case. It is called softmax-normal here to keep the two
    apart.
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    if center.ndim not in (1, 2) or center.shape[-1] < 2:
        raise ValueError("softmax_normal: center must be (k,) or (n_groups, k) with k >= 2.")
    if not bool(jnp.all(center > 0)):
        raise ValueError("softmax_normal: every component of center must be positive.")
    if not bool(jnp.allclose(center.sum(axis=-1), 1.0, atol=1e-8)):
        raise ValueError("softmax_normal: center must sum to 1 along its last axis.")
    logit_sd = _positive_array("softmax_normal logit_sd", logit_sd)
    loc = jnp.log(center[..., :-1] / center[..., -1:])
    if logit_sd.shape not in ((), loc.shape[-1:]):
        raise ValueError(
            "softmax_normal: logit_sd must be a scalar or one value per unconstrained "
            f"element, shape {loc.shape[-1:]}; got shape {logit_sd.shape}."
        )
    scale = jnp.broadcast_to(logit_sd, loc.shape)
    return tfd.TransformedDistribution(
        tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale), tfb.SoftmaxCentered()
    )


def product_transformed_gaussian_prior(
    **components: tfd.TransformedDistribution,
) -> tfd.TransformedDistribution:
    """The product of independent scalar priors as one vector-valued prior.

    Each component must be a ``TransformedDistribution`` with a scalar
    ``Normal`` base — every scalar helper in this module returns one — because
    the product is realized as a single ``MultivariateNormalDiag`` base under
    a ``Blockwise`` of the components' bijectors, which gives the
    calibration parameter one unconstrained Gaussian density with analytic
    moments. A component with any other base, a vector event, or a batch
    shape is refused.

    Keyword order is component order; it must match the ``components`` of
    the calibration parameter's :class:`SIPNETMap`.

    Examples
    --------
    >>> prior = product_transformed_gaussian_prior(
    ...     capacity=log_normal(median=115.0, geometric_sd=1.75),
    ...     respiration_share=logit_normal(median=0.18, logit_sd=0.35),
    ... )
    >>> prior.event_shape
    TensorShape([2])
    """
    if len(components) < 2:
        raise ValueError("product_transformed_gaussian_prior: give at least two components.")
    locs, scales, bijectors = [], [], []
    for name, component in components.items():
        base = getattr(component, "distribution", None)
        if not isinstance(component, tfd.TransformedDistribution) or not isinstance(
            base, tfd.Normal
        ):
            raise TypeError(
                f"product_transformed_gaussian_prior: component {name!r} must be a "
                "TransformedDistribution with a Normal base, e.g. from log_normal() or "
                f"logit_normal(); got {type(component).__name__}."
            )
        if component.event_shape != () or component.batch_shape != ():
            raise ValueError(
                f"product_transformed_gaussian_prior: component {name!r} must be a scalar "
                f"with no batch shape; got event {component.event_shape}, batch "
                f"{component.batch_shape}."
            )
        locs.append(base.loc)
        scales.append(base.scale)
        bijectors.append(component.bijector)
    base = tfd.MultivariateNormalDiag(loc=jnp.stack(locs), scale_diag=jnp.stack(scales))
    return tfd.TransformedDistribution(base, tfb.Blockwise(bijectors, block_sizes=[1] * len(locs)))


# ── SIPNET maps ───────────────────────────────────────────────────────────────


@runtime_checkable
class SIPNETMap(Protocol):
    """``M_c``: a calibration parameter's natural-space value to SIPNET
    parameter values.

    Attributes
    ----------
    writes:
        SIPNET parameters this SIPNET map sets.
    reads:
        Fixed SIPNET parameters it needs; ``()`` for most.
    components:
        Names of the ``k`` components of the natural-space value, in order.
    component_units:
        UDUNITS units of each component, in the same order.
    """

    writes: tuple[str, ...]
    reads: tuple[str, ...]
    components: tuple[str, ...]
    component_units: tuple[str, ...]

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        """*natural* has shape ``(..., k)``; each returned array has shape
        ``(...)``. Values in *fixed* broadcast against ``(...)``."""


@dataclass(frozen=True)
class Identity:
    """One scalar component to one SIPNET parameter; what a SIPNET parameter
    name passed as ``sipnet_map`` stands for."""

    parameter: str

    @property
    def writes(self) -> tuple[str, ...]:
        return (self.parameter,)

    @property
    def reads(self) -> tuple[str, ...]:
        return ()

    @property
    def components(self) -> tuple[str, ...]:
        return (self.parameter,)

    @property
    def component_units(self) -> tuple[str, ...]:
        return (_FLAT_SPECS[self.parameter].units,)

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        return {self.parameter: natural[..., 0]}


@dataclass(frozen=True)
class SimplexMap:
    """A point on the ``k``-simplex to ``k - 1`` SIPNET parameters.

    The first ``k - 1`` components are written; the last is a residual SIPNET
    recomputes itself and has no parameter for. :data:`ALLOCATION` says why
    the residual is nonetheless part of the calibration parameter.

    Parameters
    ----------
    writes:
        SIPNET parameters for the first ``k - 1`` components, in order.
    residual:
        Name of the last component; it names a Fields variable and is
        written to no SIPNET parameter.
    """

    writes: tuple[str, ...]
    residual: str

    @property
    def reads(self) -> tuple[str, ...]:
        return ()

    @property
    def components(self) -> tuple[str, ...]:
        return (*self.writes, self.residual)

    @property
    def component_units(self) -> tuple[str, ...]:
        return ("1",) * len(self.components)

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        return {name: natural[..., i] for i, name in enumerate(self.writes)}


@dataclass(frozen=True)
class PhotosynthesisMap:
    """The identifiable photosynthesis pair, replacing four parameters by two.

    ``aMax``, ``aMaxFrac``, ``baseFolRespFrac`` and ``cFracLeaf`` enter SIPNET
    only through two quantities (``sipnet.c:614, 617, 633``), leaving a
    two-dimensional exactly flat direction. With ``aMaxFrac`` and
    ``cFracLeaf`` fixed, the calibration parameter is

    .. math::

        P = \\frac{\\texttt{aMax}\\,(\\texttt{aMaxFrac} + \\texttt{baseFolRespFrac})}
                  {\\texttt{cFracLeaf}},
        \\qquad
        \\rho = \\frac{\\texttt{baseFolRespFrac}}
                      {\\texttt{aMaxFrac} + \\texttt{baseFolRespFrac}},

    and this SIPNET map inverts it. With ``S = aMaxFrac + baseFolRespFrac``,
    ``rho = baseFolRespFrac / S`` gives ``S = aMaxFrac / (1 - rho)``, so

    .. math::

        \\texttt{baseFolRespFrac} = \\frac{\\rho\\,\\texttt{aMaxFrac}}{1 - \\rho},
        \\qquad
        \\texttt{aMax} = \\frac{P\\,\\texttt{cFracLeaf}\\,(1 - \\rho)}{\\texttt{aMaxFrac}}.

    ``components = ("capacity", "respiration_share")``, in that order.
    ``capacity`` is nmol CO2 per gram of leaf carbon per second: its unit
    string is ``max_photosynthesis_rate``'s, whose gram is of leaf dry
    mass, since ``cFracLeaf`` converts one to the other and has unit
    ``"1"``. ``respiration_share`` is dimensionless.

    Notes
    -----
    Both outputs are positive for every ``P > 0``, ``rho in (0, 1)`` and
    ``aMaxFrac in (0, 1)``, so a log-normal on ``P`` and a logit-normal on
    ``rho`` produce values pySIPNET accepts, up to float64: the sigmoid
    saturates to exactly 1 beyond ``logit rho`` of about 37, dozens of prior
    standard deviations out. ``P`` reads as canopy assimilation capacity per
    unit leaf carbon and ``rho`` as the share of it spent on basal foliar
    respiration. ``phi`` is reserved for hyperparameters in this project,
    hence ``rho`` for the share.
    """

    writes: tuple[str, ...] = ("max_photosynthesis_rate", "foliar_respiration_fraction")
    reads: tuple[str, ...] = ("daily_mean_photosynthesis_fraction", "leaf_carbon_fraction")
    components: tuple[str, ...] = ("capacity", "respiration_share")

    @property
    def component_units(self) -> tuple[str, ...]:
        return (_FLAT_SPECS["max_photosynthesis_rate"].units, "1")

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        capacity, share = natural[..., 0], natural[..., 1]
        a_max_frac = fixed["daily_mean_photosynthesis_fraction"]
        c_frac_leaf = fixed["leaf_carbon_fraction"]
        return {
            "max_photosynthesis_rate": capacity * c_frac_leaf * (1.0 - share) / a_max_frac,
            "foliar_respiration_fraction": share * a_max_frac / (1.0 - share),
        }


ALLOCATION = SimplexMap(
    writes=("leaf_allocation", "wood_allocation", "fine_root_allocation"),
    residual="coarse_root_allocation",
)
"""The allocation simplex: the 4-vector (leaf, wood, fine root, coarse root)
to ``leaf_allocation``, ``wood_allocation`` and ``fine_root_allocation``.

Notes
-----
Coarse root is not written because SIPNET has no parameter for it: it
recomputes exactly ``1 - (leaf + wood + fine root)`` at
``sipnet.c:1113-1115`` and calls ``exit()`` if that is not positive
(``sipnet.c:1117-1122``; pySIPNET's ``SIPNETParameters`` validator enforces
the same ``sum < 1`` first). Keeping coarse root in the calibration parameter
is what puts every draw strictly inside the simplex, so no ``theta`` the
prior can produce reaches that exit.
"""

PHOTOSYNTHESIS = PhotosynthesisMap()
"""``(P, rho)`` to ``aMax`` and ``baseFolRespFrac``, reading the fixed
``aMaxFrac`` and ``cFracLeaf``. See :class:`PhotosynthesisMap`."""


# ── the specs ─────────────────────────────────────────────────────────────────

SHARED = "shared"
"""Group label, and ``varies_by`` attribute value, of a calibration parameter
that does not vary."""

RESERVED_SITE_LABELS_NAMES = frozenset(
    {SHARED, SITE, SAMPLE, POINT, LON, LAT, X, Y, SITE_ID, SOURCE_INDEX, *DATA_SOURCE_MEMBER_NAMES}
)
"""Names a site-labels product cannot take in a vector, because they are
dimension or coordinate names already, or reserved for one. It may not be
named like a calibration parameter or a SIPNET parameter either."""

RESERVED_PARAMETER_NAMES = frozenset(
    {SITE, SAMPLE, POINT, LON, LAT, X, Y, SOURCE_INDEX, *DATA_SOURCE_MEMBER_NAMES}
)
"""Names a calibration parameter cannot take, because a Fields variable of
that name would collide with a coordinate, or with a dimension name reserved
for one."""

REQUIRED_SIPNET_PARAMETERS: tuple[str, ...] = tuple(
    name
    for group in SIPNETParameters.model_fields.values()
    for name, field_info in group.annotation.model_fields.items()
    if field_info.is_required()
)
"""The SIPNET parameters pySIPNET requires a value for, in declaration order:
every parameter without a default. Flag-dependent and zero-defaulted
parameters are not required and are not listed."""

DOMAIN_CHECK_CORNERS = (-12.0, 0.0, 12.0)
"""Corners of the unconstrained cube at which
:func:`check_sipnet_map_image_is_in_domain` evaluates a calibration
parameter. +-12 spans ten orders of magnitude on a log scale and reaches
``1 - 6e-6`` on a logit scale while staying inside float64; the check is
about support, not plausibility."""


@dataclass(frozen=True, eq=False, kw_only=True)
class CalibrationParameter:
    """One named piece of ``theta``: a prior in natural space, how it varies
    over sites, and how it reaches SIPNET.

    Parameters
    ----------
    name:
        ``lower_case_with_underscores``, unique within a vector. Names the
        calibration parameter, not a SIPNET parameter: ``photosynthesis``,
        not ``max_photosynthesis_rate``.
    prior:
        A TFP distribution in natural space, normally from a prior helper,
        whose ``.bijector`` is ``T_c`` and whose ``.distribution`` is the
        density of ``theta_c``. Either *independent copies*: batch shape
        ``()`` for one prior shared by every group or ``(n_groups,)`` for one
        per group, and the event of one copy, ``()`` or ``(k,)``. Or, for a
        scalar calibration parameter, *joint over groups*: batch shape ``()``
        and event ``(n_groups,)``, one distribution over every copy, with an
        elementwise bijector. Any TFP distribution is accepted with its
        default event-space bijector; one without analytic unconstrained
        moments is then moment matched by
        :meth:`ParameterVector.gaussian_prior` when that is given ``key=``
        and ``n_moment_samples=``.
    sipnet_map:
        A :class:`SIPNETMap`, or a SIPNET parameter name as shorthand for
        :class:`Identity`.
    varies_by:
        ``None`` (shared), ``"site"``, or a site-labels name the
        :class:`ParameterVector` supplies.
    provenance:
        Where the prior came from, in words, with the citation. A placeholder
        says it is one.

    Raises
    ------
    TypeError
        If *prior* is not a TFP distribution or *sipnet_map* is neither a
        :class:`SIPNETMap` nor a SIPNET parameter name.
    ValueError
        For a malformed or reserved name, an empty provenance, SIPNET
        parameters that do not exist, a prior that is not ``float64``, of the
        wrong batch or event rank, or without a default event-space bijector,
        or a SIPNET map whose ``components`` do not match the prior's event.

    Examples
    --------
    >>> soil = CalibrationParameter(
    ...     name="base_soil_respiration",
    ...     prior=log_normal_from_interval(lower=0.004, upper=0.020),
    ...     sipnet_map="base_soil_respiration_rate",
    ...     varies_by="pft",
    ...     provenance="BETY som_respiration_rate posterior, 2.5-97.5% quantiles.",
    ... )
    >>> soil.size, soil.element_labels
    (1, ('log(base_soil_respiration_rate)',))
    """

    name: str
    prior: tfd.Distribution
    sipnet_map: SIPNETMap | str
    varies_by: str | None = None
    provenance: str

    def __post_init__(self) -> None:
        if isinstance(self.sipnet_map, str):
            object.__setattr__(self, "sipnet_map", Identity(self.sipnet_map))
        check_parameter_name(self.name)
        check_provenance_is_given(self.name, self.provenance)
        check_prior_is_a_distribution(self)
        check_sipnet_map_is_a_sipnet_map(self)
        check_sipnet_parameters_exist(
            (*self.sipnet_map.writes, *self.sipnet_map.reads),
            f"calibration parameter {self.name}",
        )
        check_prior_is_float64(self)
        check_prior_event_rank(self)
        check_prior_batch_rank(self)
        check_prior_has_a_bijector(self)
        check_components_match_prior(self)

    @property
    def bijector(self) -> tfb.Bijector | None:
        """``T_c``, unconstrained to natural space: the prior's own bijector
        when it is a ``TransformedDistribution`` (or ``LogNormal``,
        ``LogitNormal``), else its default event-space bijector; ``None``
        when it has neither."""
        if _carries_its_bijector(self.prior):
            return self.prior.bijector
        try:
            return self.prior.experimental_default_event_space_bijector()
        except NotImplementedError:
            return None

    @property
    def unconstrained_prior(self) -> tfd.Distribution:
        """The density of ``theta_c``."""
        if _carries_its_bijector(self.prior):
            return self.prior.distribution
        if isinstance(self.bijector, tfb.Identity):
            return self.prior
        return tfd.TransformedDistribution(self.prior, tfb.Invert(self.bijector))

    @property
    def components(self) -> tuple[str, ...]:
        """Natural-space component names, from the SIPNET map."""
        return tuple(self.sipnet_map.components)

    @property
    def component_units(self) -> tuple[str, ...]:
        """Units of each natural-space component, from the SIPNET map."""
        return tuple(self.sipnet_map.component_units)

    @property
    def natural_size(self) -> int:
        """``k``, the number of natural-space components of one copy."""
        return len(self.components)

    @property
    def is_scalar(self) -> bool:
        """Whether one copy's natural value is a scalar (``k == 1``)."""
        return self.natural_size == 1

    @property
    def is_joint(self) -> bool:
        """Whether the prior is one distribution over every copy."""
        return len(tuple(self.prior.event_shape)) == (1 if self.is_scalar else 2)

    @property
    def joint_groups(self) -> int | None:
        """The number of copies a joint prior is over; ``None`` otherwise."""
        return int(self.prior.event_shape[0]) if self.is_joint else None

    @property
    def size(self) -> int:
        """The number of columns of ``theta`` per group."""
        if self.is_scalar:
            return 1
        shape = self.bijector.inverse_event_shape(self.prior.event_shape)
        return int(shape[-1])

    @property
    def element_labels(self) -> tuple[str, ...]:
        """Unconstrained element names: ``log(x)``, ``logit(x)``,
        ``alr(a:residual)``, ``logit(x in (low, high))``, derived from the
        bijector. None contains ``/``, so every Fields variable name is a
        legal netCDF name."""
        return _unconstrained_labels(self.bijector, self.components)

    @property
    def distribution_name(self) -> str:
        """A short name for :meth:`ParameterVector.describe`."""
        return _distribution_name(self.prior)

    @property
    def has_analytic_moments(self) -> bool:
        """Whether ``unconstrained_prior`` answers ``mean()`` and
        ``variance()``/``covariance()`` without sampling."""
        try:
            self.unconstrained_prior.mean()
            if self.is_scalar and not self.is_joint:
                self.unconstrained_prior.variance()
            else:
                self.unconstrained_prior.covariance()
        except NotImplementedError:
            return False
        return True


@dataclass(frozen=True, eq=False, kw_only=True)
class FixedParameter:
    """A SIPNET parameter held at a value.

    Parameters
    ----------
    name:
        SIPNET parameter name, checked against ``PARAMETER_SPECS``.
    value:
        A float when *varies_by* is ``None``; otherwise a mapping from group
        to float, covering every group of the vector. Each value is checked
        against the SIPNET parameter's ``ParameterDomain``, so a fixed value
        can never be one pySIPNET refuses.
    varies_by:
        As for :class:`CalibrationParameter`.
    provenance:
        Where the value came from. A ``template.param`` value says so and
        says why nothing better covers it.

    Examples
    --------
    >>> FixedParameter(
    ...     name="vapor_pressure_deficit_exponent", value=2.0,
    ...     provenance="Braswell et al. (2005) fix the exponent at 2.",
    ... ).value
    2.0

    Notes
    -----
    A mapping *value* is stored as a
    :class:`~sipnet_calibration.conventions.FrozenMapping`, so it cannot be
    changed after the checks and the parameter pickles. Compared and hashed
    by identity (``eq=False``), as :class:`CalibrationParameter` is.
    """

    name: str
    value: float | Mapping[Any, float]
    varies_by: str | None = None
    provenance: str

    def __post_init__(self) -> None:
        check_sipnet_parameters_exist((self.name,), f"fixed parameter {self.name}")
        check_provenance_is_given(self.name, self.provenance)
        check_fixed_value_shape(self)
        if isinstance(self.value, Mapping):
            object.__setattr__(
                self, "value", as_frozen_mapping(self.value, message_name="value")
            )
        check_fixed_values_are_numbers(self)
        for value in self.values():
            check_fixed_value_is_in_domain(self.name, value)

    def values(self) -> tuple[float, ...]:
        """Every value, in mapping order (one value when shared)."""
        if isinstance(self.value, Mapping):
            return tuple(float(v) for v in self.value.values())
        return (float(self.value),)


# ── the layout ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class Layout:
    """Where each calibration parameter lives in Flat ``theta``, by name.

    Calibration parameters in declaration order; within one, groups in group
    order; within a group, elements in order. Built by
    :class:`ParameterVector`; consumers call :meth:`unpack`, :meth:`pack` and
    :meth:`index` rather than computing offsets.

    Attributes
    ----------
    parameters:
        Calibration parameter names in layout order.
    sizes:
        Columns per group, per calibration parameter.
    groups:
        Group labels per calibration parameter, in the order they occupy
        ``theta``.
    dims:
        The group dimension per calibration parameter: ``"shared"``,
        ``"site"``, or a site-labels name.
    element_labels:
        Unconstrained element labels per calibration parameter.

    Examples
    --------
    With ``vector`` a :class:`ParameterVector` and ``theta`` from
    ``vector.sample``::

        layout = vector.layout
        layout.dimension == theta.shape[-1]                # True
        parts = layout.unpack(theta)                       # {name: (..., n_groups, size)}
        layout.pack(parts)                                 # theta again
        layout.index("allocation", group="conifer")        # array([2, 3, 4])
    """

    parameters: tuple[str, ...]
    sizes: Mapping[str, int]
    groups: Mapping[str, tuple[Any, ...]]
    dims: Mapping[str, str]
    element_labels: Mapping[str, tuple[str, ...]]

    @cached_property
    def slices(self) -> dict[str, slice]:
        """The contiguous slice of ``theta`` each calibration parameter owns."""
        out, start = {}, 0
        for name in self.parameters:
            width = len(self.groups[name]) * self.sizes[name]
            out[name] = slice(start, start + width)
            start += width
        return out

    @property
    def dimension(self) -> int:
        """``D``."""
        return sum(len(self.groups[n]) * self.sizes[n] for n in self.parameters)

    @cached_property
    def column_labels(self) -> tuple[str, ...]:
        """``D`` strings: ``"c"``, ``"c[group]"``, ``"c[element]"`` or
        ``"c[group][element]"`` as the calibration parameter needs."""
        out = []
        for name in self.parameters:
            groups, elements = self.groups[name], self.element_labels[name]
            for group in groups:
                for element in elements:
                    label = name
                    if len(groups) > 1:
                        label += f"[{group}]"
                    if len(elements) > 1:
                        label += f"[{element}]"
                    out.append(label)
        return tuple(out)

    def slice(self, parameter: str) -> slice:
        """The columns of ``theta`` a calibration parameter owns."""
        return self.slices[self._known(parameter)]

    def index(self, parameter: str, group: Any = None, element: str | None = None) -> np.ndarray:
        """Column indices of a calibration parameter, narrowed by group and
        element label."""
        name = self._known(parameter)
        n_groups, size = len(self.groups[name]), self.sizes[name]
        idx = np.arange(self.slices[name].start, self.slices[name].stop).reshape(n_groups, size)
        if group is not None:
            idx = idx[[_position(self.groups[name], group, f"group of {name}")]]
        if element is not None:
            idx = idx[:, [_position(self.element_labels[name], element, f"element of {name}")]]
        return idx.ravel()

    def unpack(self, theta: Array) -> dict[str, Array]:
        """Flat ``(..., D)`` to ``{calibration parameter: (..., n_groups, size)}``."""
        theta = _as_theta(theta, self.dimension)
        lead = theta.shape[:-1]
        return {
            name: theta[..., self.slices[name]].reshape(
                (*lead, len(self.groups[name]), self.sizes[name])
            )
            for name in self.parameters
        }

    def pack(self, parts: Mapping[str, Array]) -> Array:
        """Inverse of :meth:`unpack`; refuses a missing or extra calibration
        parameter."""
        if set(parts) != set(self.parameters):
            raise ValueError(
                "Layout.pack: expected exactly the calibration parameters "
                f"{list(self.parameters)}, got {sorted(parts)}."
            )
        pieces = []
        for name in self.parameters:
            part = jnp.asarray(parts[name], dtype=jnp.float64)
            expected = (len(self.groups[name]), self.sizes[name])
            if part.shape[-2:] != expected:
                raise ValueError(
                    f"Layout.pack: {name} must end in (n_groups, size) = {expected}, "
                    f"got shape {part.shape}."
                )
            pieces.append(part.reshape((*part.shape[:-2], expected[0] * expected[1])))
        try:
            return jnp.concatenate(pieces, axis=-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Layout.pack: the parts' leading dimensions disagree; every part must "
                f"share one leading shape. Got {[tuple(p.shape[:-1]) for p in pieces]}."
            ) from error

    def _known(self, parameter: str) -> str:
        if parameter not in self.slices:
            raise KeyError(
                f"Layout: no calibration parameter {parameter!r}; have {list(self.parameters)}."
            )
        return parameter


# ── the parameter vector ──────────────────────────────────────────────────────

NATURAL = "natural"
"""The ``space`` of a Fields dataset holding natural-space components."""

UNCONSTRAINED = "unconstrained"
"""The ``space`` of a Fields dataset holding unconstrained elements."""

SPACES = (NATURAL, UNCONSTRAINED)
"""The values ``space`` takes."""

FIELDS_REPRESENTATION = "calibration_parameters"
"""The ``representation`` attribute of a Fields dataset."""

SIPNET_TABLE_REPRESENTATION = "sipnet_parameters"
"""The ``representation`` attribute of a SIPNET table."""


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ParameterVector:
    """A named random vector: which SIPNET parameters are calibrated, as what
    calibration parameters, under what prior, over which sites, with what
    held fixed.

    Its values have three representations, which the module docstring
    defines in full. *Flat* is a float64 ``(D,)`` or ``(J, D)`` array in
    unconstrained space, what :meth:`sample` returns and :meth:`log_prior`
    and pyEKI consume. *Fields* is an ``xarray.Dataset`` of fields
    on ``(sample, site)``, from :meth:`fields` and back through
    :meth:`flat`. The *SIPNET table* is an ``xarray.Dataset`` of SIPNET
    parameters on the same dims, from :meth:`sipnet_table`, which
    :func:`sipnet_overrides` and ``pyens.xarray.fields_from_dataset`` read.

    Parameters
    ----------
    parameters:
        The :class:`CalibrationParameter`\\ s, in the order they occupy
        ``theta``.
    fixed:
        The SIPNET parameters held at a value. Every SIPNET parameter any
        SIPNET map ``reads`` must appear here.
    sites:
        The sites the vector is defined over: site ids, ascending, or a site
        table with a ``site_id`` column in ascending order, such as
        :func:`sipnet_calibration.sites.select_sites` returns, whose
        ``lon``/``lat`` are then carried onto Fields and the SIPNET table.
        The ids are the site table's; ids of a shared pool are never
        renumbered.
    site_labels:
        ``{site_labels_name: labels}`` for every ``varies_by`` other than
        ``None`` and ``"site"``: a site-labels product with ``site_id`` and
        ``label`` columns (:data:`~sipnet_calibration.site_labels.LABEL_COLUMN`),
        such as
        :func:`sipnet_calibration.site_labels.load_site_labels` returns, which
        must label every site here; a pandas categorical; or one label per
        site, in site order.
        The module Notes say how each decides the groups.
    require_complete:
        Refuse a vector that leaves any :data:`REQUIRED_SIPNET_PARAMETERS`
        neither calibrated nor fixed. Off by default, so a partial vector can
        run on top of a base parameter set.

    Attributes
    ----------
    dimension : int
        ``D``.
    layout : Layout
        Which columns of ``theta`` belong to which calibration parameter,
        group and element.
    parameter_names : tuple[str, ...]
        Calibration parameter names in layout order.
    sites : tuple[int, ...]
        The site ids the vector covers, in the order of its site axis.
    site_labels : Mapping[str, tuple]
        The class of every site, per site-labels name, in site order.
    site_table : pandas.DataFrame
        ``site_id``, ``lon``/``lat`` when known, and one column per
        site-labels name.
    sipnet_parameter_names, unset_sipnet_parameter_names : tuple[str, ...]
        What the vector sets, calibrated and fixed, and the required SIPNET
        parameters it leaves to a run's base parameter set, which is then
        part of the calibration's specification.

    Raises
    ------
    ValueError, TypeError
        From the ``check_*`` helpers: a repeated or reserved name, two
        writers of one SIPNET parameter, a SIPNET map reading a parameter
        nobody fixed, site labels missing, of the wrong length, not covering
        every site or colliding with another name, a prior whose shape does
        not match its groups, a fixed value missing a group, a calibration
        parameter whose transform can leave a SIPNET parameter's domain, or
        *require_complete* set with a required parameter unset.

    Examples
    --------
    >>> vector = example_parameter_vector(sites=(1, 27), pft=("deciduous", "conifer"))
    >>> vector.dimension
    13
    >>> theta = vector.sample(jax.random.key(0), n=4)
    >>> table = vector.sipnet_table(theta)                  # (sample, site)
    >>> sorted(sipnet_overrides(table, batch={"sample": 0}, site=27))[:2]
    ['base_soil_respiration_rate', 'daily_mean_photosynthesis_fraction']
    """

    parameters: tuple[CalibrationParameter, ...]
    fixed: tuple[FixedParameter, ...] = ()
    sites: Sequence[int] | pd.DataFrame
    site_labels: Mapping[str, Any] = field(default_factory=dict)
    require_complete: bool = False
    _declared_classes: Mapping[str, tuple[Any, ...]] = field(init=False, default=None)
    _lon_lat: tuple[np.ndarray, np.ndarray] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        sites, lon_lat = _normalized_sites(self.sites)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "_lon_lat", lon_lat)
        check_vector_has_a_site(self.sites)
        check_sites_are_ascending(self.sites)
        labels, declared = {}, {}
        for name, value in dict(self.site_labels).items():
            labels[name], declared[name] = _normalized_site_labels(name, value, self.sites)
        object.__setattr__(self, "site_labels", FrozenMapping(labels))
        object.__setattr__(self, "_declared_classes", FrozenMapping(declared))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "fixed", tuple(self.fixed))
        check_parameters_have_their_types(self.parameters, self.fixed)
        check_parameter_names_are_unique(self.parameters)
        check_site_labels_cover_sites(self)
        object.__setattr__(
            self, "parameters", tuple(self._prior_restricted_to_groups(p) for p in self.parameters)
        )
        object.__setattr__(
            self, "fixed", tuple(self._fixed_restricted_to_groups(f) for f in self.fixed)
        )
        check_each_sipnet_parameter_has_one_writer(self.parameters, self.fixed)
        check_reads_are_fixed(self.parameters, self.fixed)
        for parameter in self.parameters:
            check_prior_shape_matches_groups(parameter, self.n_groups(parameter.varies_by))
        for parameter in self.fixed:
            check_fixed_value_covers_groups(parameter, self.group_labels(parameter.varies_by))
        for parameter in self.parameters:
            check_sipnet_map_image_is_in_domain(self, parameter)
        if self.require_complete:
            check_every_required_parameter_is_set(self)

    # -- structure -----------------------------------------------------------

    def __getitem__(self, name: str) -> CalibrationParameter:
        """The calibration parameter called *name*."""
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"no calibration parameter {name!r}; have {list(self.parameter_names)}.")

    def __repr__(self) -> str:
        """One line per calibration parameter, then the fixed and unset
        SIPNET parameters."""
        return _summary(self)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Calibration parameter names in layout order."""
        return tuple(p.name for p in self.parameters)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        """Every SIPNET parameter this vector sets, calibrated and fixed, in
        pySIPNET's declaration order."""
        written = {n for p in self.parameters for n in p.sipnet_map.writes}
        written |= {f.name for f in self.fixed}
        return tuple(n for n in _FLAT_SPECS if n in written)

    @property
    def unset_sipnet_parameter_names(self) -> tuple[str, ...]:
        """The :data:`REQUIRED_SIPNET_PARAMETERS` this vector leaves to the
        base parameter set."""
        written = set(self.sipnet_parameter_names)
        return tuple(n for n in REQUIRED_SIPNET_PARAMETERS if n not in written)

    @property
    def dimension(self) -> int:
        """``D``."""
        return self.layout.dimension

    @cached_property
    def layout(self) -> Layout:
        """Where each calibration parameter lives in ``theta``; see
        :class:`Layout`."""
        return Layout(
            parameters=self.parameter_names,
            sizes={p.name: p.size for p in self.parameters},
            groups={p.name: self.group_labels(p.varies_by) for p in self.parameters},
            dims={p.name: p.varies_by or SHARED for p in self.parameters},
            element_labels={p.name: p.element_labels for p in self.parameters},
        )

    @property
    def site_table(self) -> pd.DataFrame:
        """``site_id`` (``int32``), ``lon``/``lat`` when known, and one
        categorical column per site-labels name, one row per site."""
        frame = pd.DataFrame({SITE_ID: np.asarray(self.sites, dtype=SITE_DTYPE)})
        if self._lon_lat is not None:
            frame[LON], frame[LAT] = self._lon_lat
        for name, labels in self.site_labels.items():
            frame[name] = pd.Categorical(labels, categories=self.group_labels(name))
        return frame

    def group_labels(self, varies_by: str | None) -> tuple[Any, ...]:
        """The groups of a ``varies_by`` value, in the order they occupy
        ``theta``: ``("shared",)``, the sites, or the classes of a
        site-labels product that some site here carries."""
        if varies_by is None:
            return (SHARED,)
        if varies_by == SITE:
            return self.sites
        present = set(self.site_labels[varies_by])
        return tuple(c for c in self._declared_classes[varies_by] if c in present)

    def n_groups(self, varies_by: str | None) -> int:
        """The number of copies a calibration parameter with this
        ``varies_by`` has."""
        return len(self.group_labels(varies_by))

    def sites_with(self, site_labels_name: str, label: Any) -> tuple[int, ...]:
        """The sites carrying *label* under the site-labels product
        *site_labels_name*: ``()`` for a declared class no site here carries,
        ``KeyError`` for a label that is not a declared class."""
        if site_labels_name not in self.site_labels:
            raise KeyError(
                f"no site labels {site_labels_name!r}; have {sorted(self.site_labels)}."
            )
        if label not in self._declared_classes[site_labels_name]:
            raise KeyError(
                f"{label!r} is not a class of site labels {site_labels_name!r}; have "
                f"{list(self._declared_classes[site_labels_name])}."
            )
        labels = self.site_labels[site_labels_name]
        return tuple(s for s, lab in zip(self.sites, labels, strict=True) if lab == label)

    def select(
        self,
        *,
        parameters: Sequence[str] | None = None,
        sites: Sequence[int] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> ParameterVector:
        """A smaller vector: some of the calibration parameters, over some of
        the sites.

        Parameters
        ----------
        parameters:
            Calibration parameter names to keep, a sequence; ``None`` keeps
            all. They keep this vector's layout order. Fixed parameters are
            always kept.
        sites:
            Site ids to keep, a sequence, each one of :attr:`sites` and named
            once; ``None`` keeps all. The result keeps this vector's site
            order.
        labels:
            ``{site_labels_name: classes}``, each a sequence of class names:
            keeps the sites carrying one of those classes. Composes with
            *sites* by intersection.

        Returns
        -------
        ParameterVector
            With its own layout and dimension. A per-site prior is sliced to
            the kept sites (a joint one marginalized, exactly); a per-class
            prior, and a per-class or per-site fixed value, to the groups the
            kept sites still have. ``lon``/``lat`` and the site labels are
            carried.

        Raises
        ------
        TypeError
            If *parameters*, *sites* or a *labels* value is one value, a
            string or a set; if a name or class is not a string; or if a site
            id is a boolean, a float or not a number.
        KeyError
            For an unknown calibration parameter, site, site-labels name or
            class.
        ValueError
            If a site is named twice or is not a site id, *sites* is a
            two-dimensional array, or no site remains.

        Notes
        -----
        The smaller vector's Flat values are not this one's. To move an
        ensemble across, go through Fields, where the correspondence is by
        site and by variable: ``small.flat(big.fields(theta))``.
        """
        names = self._selected_parameter_names(parameters)
        kept = self._selected_sites(sites, labels)
        positions = np.asarray([self.sites.index(s) for s in kept])
        site_labels = {
            name: pd.Categorical([labels_[i] for i in positions], categories=self.group_labels(name))
            for name, labels_ in self.site_labels.items()
        }
        return ParameterVector(
            parameters=tuple(self._prior_restricted_to_sites(self[n], positions) for n in names),
            fixed=tuple(self._fixed_restricted_to_sites(f, kept) for f in self.fixed),
            sites=self._sites_argument(positions),
            site_labels=site_labels,
            require_complete=self.require_complete,
        )

    def describe(self) -> pd.DataFrame:
        """One row per column of ``theta``.

        Columns: ``parameter``, ``group``, ``element``,
        ``sipnet_parameters``, ``distribution``, ``theta_mean``,
        ``theta_sd``, ``natural_median``, ``natural_2.5``, ``natural_97.5``,
        ``theta_moments`` (``"analytic"`` or ``"monte_carlo"``) and
        ``provenance``. Natural quantiles are NaN for a vector-valued or
        joint prior, where a marginal quantile would misrepresent a simplex
        or is not available from TFP.
        """
        rows = []
        for parameter in self.parameters:
            groups = self.group_labels(parameter.varies_by)
            moments = self._describe_moments(parameter)
            quantiles = self._describe_quantiles(parameter)
            for g, group in enumerate(groups):
                for e, element in enumerate(parameter.element_labels):
                    rows.append(
                        {
                            "parameter": parameter.name,
                            "group": group,
                            "element": element,
                            "sipnet_parameters": ", ".join(parameter.sipnet_map.writes),
                            "distribution": parameter.distribution_name,
                            "theta_mean": moments[0][g, e],
                            "theta_sd": moments[1][g, e],
                            "natural_median": quantiles[0][g],
                            "natural_2.5": quantiles[1][g],
                            "natural_97.5": quantiles[2][g],
                            "theta_moments": (
                                "analytic" if parameter.has_analytic_moments else "monte_carlo"
                            ),
                            "provenance": parameter.provenance,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.index.name = "column"
        return frame

    # -- the prior, on Flat --------------------------------------------------

    def sample(self, key: Array, n: int) -> Array:
        """``n`` prior draws as Flat, ``(n, D)``.

        One key split per calibration parameter, so, with JAX's default
        partitionable key splitting, appending one to a vector leaves the
        earlier ones' draws unchanged.
        """
        parts = {}
        for parameter, subkey in zip(
            self.parameters, jax.random.split(key, len(self.parameters)), strict=True
        ):
            if parameter.is_joint:
                draw = parameter.unconstrained_prior.sample((n,), seed=subkey)
            else:
                draw = self._broadcast_unconstrained(parameter).sample((n,), seed=subkey)
            n_groups = self.n_groups(parameter.varies_by)
            parts[parameter.name] = jnp.reshape(draw, (n, n_groups, parameter.size))
        return self.layout.pack(parts)

    def log_prior(self, theta: Array) -> Array:
        """Log prior density of Flat *theta*: ``(J,)`` for ``(J, D)``, a
        scalar for ``(D,)``.

        Notes
        -----
        The Jacobian of ``T_c`` is included by construction: the summand is
        ``prior.distribution.log_prob(theta_c)``, which equals
        ``prior.log_prob(T_c(theta_c)) + T_c.forward_log_det_jacobian(theta_c)``.
        The finite-difference test asserts that identity. ``jit``-able and
        differentiable; the entry point an MCMC baseline uses.
        """
        parts = self.layout.unpack(theta)
        total = jnp.zeros(parts[self.parameters[0].name].shape[:-2], dtype=jnp.float64)
        for parameter in self.parameters:
            part = parts[parameter.name]
            if parameter.is_joint:
                total = total + parameter.unconstrained_prior.log_prob(part[..., 0])
                continue
            if parameter.is_scalar:
                part = part[..., 0]
            prior = self._broadcast_unconstrained(parameter)
            total = total + prior.log_prob(part).sum(axis=-1)
        return total

    def gaussian_prior(
        self, *, key: Array | None = None, n_moment_samples: int = 0
    ) -> Gaussian:
        """The prior over Flat as a ``pyeki.gauss.Gaussian``.

        Parameters
        ----------
        key, n_moment_samples:
            Used only for a calibration parameter without analytic
            unconstrained moments (a hand-built prior that is not a
            ``TransformedDistribution``): it is moment matched from
            *n_moment_samples* draws (at least 2) with its own split of
            *key*. Left at their defaults, such a calibration parameter
            raises instead.

        Returns
        -------
        pyeki.gauss.Gaussian
            Mean of length ``D``. Covariance a ``PSDBlockDiag`` with one
            block per calibration parameter in layout order: a
            ``PSDDiagonal`` over groups for a scalar one, a ``PSDBlockDiag``
            of one ``DensePSD`` per group for a vector-valued one, and one
            ``DensePSD`` over every group for a joint prior.

        Raises
        ------
        NotImplementedError
            If a calibration parameter lacks analytic moments and no *key*
            was given.

        Notes
        -----
        EKI consults the prior once, through ``EKIState.from_prior``, so this
        object only has to be sampled. For every prior the helpers build the
        match is exact, because the prior is Gaussian in ``theta`` by
        construction.
        """
        means, blocks = [], []
        keys = (
            jax.random.split(key, len(self.parameters))
            if key is not None
            else (None,) * len(self.parameters)
        )
        for parameter, subkey in zip(self.parameters, keys, strict=True):
            mean, block = self._moment_block(parameter, subkey, n_moment_samples)
            means.append(mean)
            blocks.append(block)
        return Gaussian(mean=jnp.concatenate(means), cov=PSDBlockDiag(tuple(blocks)))

    # -- conversions ---------------------------------------------------------

    def fields(
        self, theta: Array, *, space: str = NATURAL, batch_dim: str = SAMPLE
    ) -> xr.Dataset:
        """Flat to Fields.

        Parameters
        ----------
        theta:
            Flat, ``(D,)`` or ``(J, D)``.
        space:
            ``"natural"`` applies each calibration parameter's bijector, and
            the variables are natural-space components; ``"unconstrained"``
            leaves ``theta`` as it is, and the variables are unconstrained
            elements.
        batch_dim:
            The name of the batch dim a ``(J, D)`` *theta* is given, labeled
            ``0`` to ``J - 1`` in row order. It may not be a spatial name,
            ``time``, a site-labels name or a calibration parameter name.

        Returns
        -------
        xarray.Dataset
            Fields with ``attrs["space"]`` set to *space*: one variable per
            scalar component (or element) on ``(batch_dim, site)``, or
            ``(site,)`` for one value, shared and per-class copies repeated
            at every site that reads them.
        """
        check_space_is_known(space)
        check_batch_dim_name_is_not_taken(self, batch_dim)
        theta = _as_theta(theta, self.dimension)
        blocks = self._natural_blocks(theta) if space == NATURAL else self.layout.unpack(theta)
        dims = (batch_dim, SITE) if theta.ndim == 2 else (SITE,)
        variables = {}
        for parameter in self.parameters:
            on_sites = np.asarray(self._group_values_onto_sites(parameter, blocks[parameter.name]))
            labels = _space_labels(parameter, space)
            for i, variable in enumerate(_fields_variable_names(parameter, space)):
                attributes = _fields_attributes(parameter, labels, i, space)
                variables[variable] = (dims, on_sites[..., i], attributes)
        attributes = {"representation": FIELDS_REPRESENTATION, "space": space}
        coords = self._coordinates(theta, batch_dim)
        return xr.Dataset(variables, coords=coords, attrs=attributes)

    def flat(self, fields: xr.Dataset) -> Array:
        """Fields to Flat: the inverse of :meth:`fields`, in either space.

        Reads ``fields.attrs["space"]`` and applies the inverse bijector when
        it is natural. Rows follow the order of the dataset's batch dim,
        whatever it is named; Flat has no batch labels, and
        :meth:`sipnet_table` is where Fields' labels are kept. A scalar batch
        coordinate, left by ``.isel(sample=k)``, is not a batch dim: such
        Fields give one vector. Takes exactly the variables and sites this
        vector needs and ignores any others, so Fields from a larger vector
        project onto this one.

        Returns
        -------
        jax.Array
            Flat, ``(J, D)`` when *fields* has a batch dim and ``(D,)``
            otherwise.

        Raises
        ------
        ValueError
            If ``attrs["space"]`` is missing or unknown; if *fields* has more
            than one batch dim (stack them first, with
            :func:`sipnet_calibration.fields.stack_batch_dims`); if a needed
            variable or site is absent, or a variable is not on the dataset's
            ``(*batch, site)``; if a value is not finite; or if a group's
            value differs between two of its sites, which no Flat vector can
            represent.
        """
        space = check_fields_space_is_given(fields)
        if SITE in fields.coords and fields[SITE].ndim == 0:
            fields = fields.expand_dims(SITE)
        check_fields_hold_the_sites(fields, self.sites)
        fields = fields.sel({SITE: list(self.sites)})
        batch = batch_dims(fields)
        check_at_most_one_batch_dim(batch, message_name="Fields")
        dims = (*batch, SITE)
        parts = {}
        for parameter in self.parameters:
            names = _fields_variable_names(parameter, space)
            check_fields_hold_the_variables(fields, names, parameter.name)
            for name in names:
                check_dims_are_batch_spatial_or_time(fields[name], message_name=repr(name))
            check_fields_variables_are_in_the_space(fields, names, space)
            on_sites = np.stack([_variable_values(fields[n], dims) for n in names], axis=-1)
            check_fields_values_are_finite(on_sites, parameter.name)
            groups = self._site_group_index(parameter.varies_by)
            first = np.asarray(
                [np.flatnonzero(groups == g)[0] for g in range(self.n_groups(parameter.varies_by))]
            )
            block = on_sites[..., first, :]
            check_group_values_agree_across_sites(self, parameter, on_sites, block[..., groups, :])
            block = jnp.asarray(block, dtype=jnp.float64)
            if space == NATURAL:
                unconstrained = self._to_unconstrained(parameter, block)
                check_natural_values_are_in_the_support(parameter, block, unconstrained)
                block = unconstrained
            parts[parameter.name] = block
        return self.layout.pack(parts)

    def sipnet_table(self, x: Array | xr.Dataset, *, batch_dim: str = SAMPLE) -> xr.Dataset:
        """Flat or Fields to the SIPNET table.

        Parameters
        ----------
        x:
            Flat, ``(D,)`` or ``(J, D)``, whose rows are labeled 0 to
            ``J - 1`` on *batch_dim*; or Fields in either space, whose batch
            dim, its name and its labels, is kept.
        batch_dim:
            The name of the batch dim a ``(J, D)`` Flat is given; ignored for
            Fields, which keep their own. It may not be a spatial name,
            ``time``, a site-labels name or a calibration parameter name.

        Returns
        -------
        xarray.Dataset
            The SIPNET table: one float64 variable per SIPNET parameter this
            vector sets (:attr:`sipnet_parameter_names`), calibrated and
            fixed alike, on ``(batch_dim, site)`` or ``(site,)``, each carrying
            ``units``, ``sipnet_name``, ``source`` and, where pySIPNET
            declares one, ``constituent``. :attr:`unset_sipnet_parameter_names`
            are absent and take the base parameter set's values at the run.

        Notes
        -----
        Not invertible: the SIPNET maps are many-to-one once the fixed
        parameters are folded in.
        """
        labels = None
        if isinstance(x, xr.Dataset):
            theta = self.flat(x)
            batch = batch_dims(x)
            if batch:
                batch_dim, labels = batch[0], x[batch[0]].values
        else:
            check_batch_dim_name_is_not_taken(self, batch_dim)
            theta = _as_theta(x, self.dimension)
        natural = self._natural_blocks(theta)
        lead = theta.shape[:-1]
        columns: dict[str, tuple[Array, str]] = {}
        for parameter in self.parameters:
            on_sites = self._group_values_onto_sites(parameter, natural[parameter.name])
            for name, values in parameter.sipnet_map(on_sites, self._fixed_table).items():
                columns[name] = (values, f"parameter {parameter.name}")
        for parameter in self.fixed:
            values = jnp.broadcast_to(self._fixed_table[parameter.name], (*lead, len(self.sites)))
            columns[parameter.name] = (values, "fixed")
        dims = (batch_dim, SITE) if theta.ndim == 2 else (SITE,)
        variables = {
            name: (dims, np.asarray(values, dtype=np.float64), _sipnet_attributes(name, source))
            for name, (values, source) in sorted(columns.items(), key=lambda kv: _SPEC_ORDER[kv[0]])
        }
        attributes = {"representation": SIPNET_TABLE_REPRESENTATION}
        coords = self._coordinates(theta, batch_dim, batch_labels=labels)
        return xr.Dataset(variables, coords=coords, attrs=attributes)

    # -- private: groups and sites -------------------------------------------

    def _prior_restricted_to_groups(self, parameter: CalibrationParameter) -> CalibrationParameter:
        """A per-class prior written over every declared class, sliced to the
        classes some site here carries."""
        if parameter.varies_by in (None, SITE):
            return parameter
        groups = self.group_labels(parameter.varies_by)
        declared = self._declared_classes[parameter.varies_by]
        if len(groups) == len(declared):
            return parameter
        positions = np.asarray([declared.index(g) for g in groups])
        if parameter.is_joint and parameter.joint_groups == len(declared):
            return _restricted(parameter, _joint_marginal(parameter, positions), positions)
        if not parameter.is_joint and tuple(parameter.prior.batch_shape) == (len(declared),):
            return _restricted(parameter, _prior_for_groups(parameter, positions), positions)
        return parameter

    def _fixed_restricted_to_groups(self, parameter: FixedParameter) -> FixedParameter:
        """A per-class fixed value keyed on declared classes, restricted to
        the ones present; left alone when a key is not a declared class, so
        the coverage check names it."""
        if not isinstance(parameter.value, Mapping) or parameter.varies_by in (None, SITE):
            return parameter
        # Declared classes no site carries are dropped; a key that is not a
        # declared class is kept, so the coverage check names it.
        declared = set(self._declared_classes[parameter.varies_by])
        groups = set(self.group_labels(parameter.varies_by))
        kept = {k: v for k, v in parameter.value.items() if k in groups or k not in declared}
        if len(kept) == len(parameter.value):
            return parameter
        return dataclasses.replace(parameter, value=kept)

    def _prior_restricted_to_sites(
        self, parameter: CalibrationParameter, positions: np.ndarray
    ) -> CalibrationParameter:
        if parameter.varies_by != SITE or len(positions) == len(self.sites):
            return parameter
        if parameter.is_joint:
            return _restricted(parameter, _joint_marginal(parameter, positions), positions)
        if tuple(parameter.prior.batch_shape) == (len(self.sites),):
            return _restricted(parameter, _prior_for_groups(parameter, positions), positions)
        return parameter

    def _fixed_restricted_to_sites(
        self, parameter: FixedParameter, kept: Sequence[int]
    ) -> FixedParameter:
        if parameter.varies_by != SITE or not isinstance(parameter.value, Mapping):
            return parameter
        return dataclasses.replace(parameter, value={s: parameter.value[s] for s in kept})

    def _selected_parameter_names(self, parameters: Sequence[str] | None) -> tuple[str, ...]:
        if parameters is None:
            return self.parameter_names
        wanted = as_names(parameters, message_name="parameters")
        unknown = [n for n in wanted if n not in self.parameter_names]
        if unknown:
            raise KeyError(
                f"select: no calibration parameters {unknown}; have {list(self.parameter_names)}."
            )
        return tuple(n for n in self.parameter_names if n in wanted)

    def _selected_sites(
        self, sites: Sequence[int] | None, labels: Mapping[str, Any] | None
    ) -> list[int]:
        kept = list(self.sites)
        if sites is not None:
            requested = set(as_site_ids(sites, message_name="sites"))
            unknown = sorted(requested - set(self.sites))
            if unknown:
                raise KeyError(f"select: sites {unknown} are not in this vector.")
            kept = [s for s in kept if s in requested]
        for name, wanted in (labels or {}).items():
            if name not in self.site_labels:
                raise KeyError(f"select: no site labels {name!r}; have {sorted(self.site_labels)}.")
            wanted = set(as_sequence(wanted, message_name=f"labels[{name!r}]"))
            undeclared = sorted(map(str, wanted - set(self._declared_classes[name])))
            if undeclared:
                raise KeyError(
                    f"select: {undeclared} are not classes of site labels {name!r}; have "
                    f"{list(self._declared_classes[name])}."
                )
            carrying = {s for s, lab in zip(self.sites, self.site_labels[name]) if lab in wanted}
            kept = [s for s in kept if s in carrying]
        if not kept:
            raise ValueError("select: no site of this vector satisfies every condition given.")
        return kept

    def _sites_argument(self, positions: np.ndarray) -> Sequence[int] | pd.DataFrame:
        ids = [self.sites[i] for i in positions]
        if self._lon_lat is None:
            return tuple(ids)
        lon, lat = self._lon_lat
        return pd.DataFrame({SITE_ID: ids, LON: lon[positions], LAT: lat[positions]})

    def _coordinates(
        self, theta: Array, batch_dim: str, batch_labels: np.ndarray | None = None
    ) -> dict[str, Any]:
        """The coordinates Fields and the SIPNET table share; *batch_labels*
        labels the batch dim, 0 to J-1 when ``None``."""
        coords: dict[str, Any] = {SITE: np.asarray(self.sites, dtype=SITE_DTYPE)}
        if self._lon_lat is not None:
            # Copies, so the Dataset handed out is writable and a write to it
            # never reaches the vector's own read-only arrays.
            lon, lat = self._lon_lat
            coords[LON] = (SITE, lon.copy(), LON_ATTRIBUTES)
            coords[LAT] = (SITE, lat.copy(), LAT_ATTRIBUTES)
        for name, labels in self.site_labels.items():
            coords[name] = (SITE, list(labels))
        if theta.ndim == 2:
            values = (
                _batch_labels(theta.shape[0])
                if batch_labels is None
                else _as_batch_labels(batch_labels)
            )
            coords[batch_dim] = batch_coordinate(batch_dim, values)
        return coords

    @cached_property
    def _fixed_table(self) -> dict[str, Array]:
        """Every fixed parameter as an ``(S,)`` array on the site axis."""
        table = {}
        for parameter in self.fixed:
            groups = self.group_labels(parameter.varies_by)
            if isinstance(parameter.value, Mapping):
                per_group = jnp.asarray([parameter.value[g] for g in groups], dtype=jnp.float64)
            else:
                per_group = jnp.asarray([parameter.value], dtype=jnp.float64)
            table[parameter.name] = per_group[self._site_group_index(parameter.varies_by)]
        return table

    def _site_group_index(self, varies_by: str | None) -> np.ndarray:
        """For each site, the index of the group it reads, ``(S,)``."""
        if varies_by is None:
            return np.zeros(len(self.sites), dtype=int)
        if varies_by == SITE:
            return np.arange(len(self.sites))
        position = {group: i for i, group in enumerate(self.group_labels(varies_by))}
        return np.asarray([position[label] for label in self.site_labels[varies_by]])

    def _group_values_onto_sites(self, parameter: CalibrationParameter, block: Array) -> Array:
        """Each group's value copied to the sites that read it:
        ``(..., n_groups, k)`` to ``(..., S, k)``."""
        return block[..., self._site_group_index(parameter.varies_by), :]

    # -- private: spaces and moments -----------------------------------------

    def _natural_blocks(self, theta: Array) -> dict[str, Array]:
        """Flat ``(..., D)`` to ``{calibration parameter: (..., n_groups, k)}``
        in natural space."""
        parts = self.layout.unpack(theta)
        return {p.name: self._to_natural(p, parts[p.name]) for p in self.parameters}

    @staticmethod
    def _to_natural(parameter: CalibrationParameter, block: Array) -> Array:
        # A scalar calibration parameter's bijector sees (..., n_groups), so a
        # bijector with one parameter per group lines up with its groups.
        if parameter.is_scalar:
            return parameter.bijector.forward(block[..., 0])[..., None]
        return parameter.bijector.forward(block)

    @staticmethod
    def _to_unconstrained(parameter: CalibrationParameter, block: Array) -> Array:
        if parameter.is_scalar:
            return parameter.bijector.inverse(block[..., 0])[..., None]
        return parameter.bijector.inverse(block)

    def _broadcast_unconstrained(self, parameter: CalibrationParameter) -> tfd.Distribution:
        """An independent-copies unconstrained prior with batch shape
        ``(n_groups,)``."""
        prior = parameter.unconstrained_prior
        n_groups = self.n_groups(parameter.varies_by)
        if tuple(prior.batch_shape) == (n_groups,):
            return prior
        return tfd.BatchBroadcast(prior, to_shape=(n_groups,))

    def _moment_block(
        self, parameter: CalibrationParameter, key: Array | None, n_moment_samples: int
    ) -> tuple[Array, PSDLinOp]:
        prior = (
            parameter.unconstrained_prior
            if parameter.is_joint
            else self._broadcast_unconstrained(parameter)
        )
        independent_scalar = parameter.is_scalar and not parameter.is_joint
        if parameter.has_analytic_moments:
            mean = prior.mean()
            spread = prior.variance() if independent_scalar else prior.covariance()
        elif key is not None:
            mean, spread = _monte_carlo_moments(prior, key, n_moment_samples, independent_scalar)
        else:
            raise NotImplementedError(
                f"calibration parameter {parameter.name!r}: its prior has no analytic "
                "unconstrained moments. Build it with a prior helper (log_normal, "
                "logit_normal, ...) or pass key= and n_moment_samples= to moment match by "
                "Monte Carlo."
            )
        mean = jnp.asarray(mean, dtype=jnp.float64)
        spread = jnp.asarray(spread, dtype=jnp.float64)
        check_prior_spread_is_positive(parameter, spread)
        if parameter.is_joint:
            return jnp.ravel(mean), DensePSD.from_matrix(spread)
        if independent_scalar:
            return jnp.ravel(mean), PSDDiagonal(jnp.ravel(spread))
        blocks = tuple(DensePSD.from_matrix(spread[g]) for g in range(spread.shape[0]))
        return jnp.ravel(mean), PSDBlockDiag(blocks)

    def _describe_moments(self, parameter: CalibrationParameter) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.n_groups(parameter.varies_by), parameter.size)
        if not parameter.has_analytic_moments:
            return np.full(shape, np.nan), np.full(shape, np.nan)
        prior = (
            parameter.unconstrained_prior
            if parameter.is_joint
            else self._broadcast_unconstrained(parameter)
        )
        return (
            np.asarray(prior.mean()).reshape(shape),
            np.asarray(prior.stddev()).reshape(shape),
        )

    def _describe_quantiles(self, parameter: CalibrationParameter) -> tuple[np.ndarray, ...]:
        n_groups = self.n_groups(parameter.varies_by)
        blank = np.full(n_groups, np.nan)
        if not parameter.is_scalar or parameter.is_joint:
            return blank, blank, blank
        try:
            return tuple(
                np.broadcast_to(np.asarray(parameter.prior.quantile(q)), (n_groups,))
                for q in (0.5, 0.025, 0.975)
            )
        except NotImplementedError:
            return blank, blank, blank


def sipnet_overrides(
    table: xr.Dataset, *, site: int, batch: Mapping[str, int] | None = None
) -> dict[str, float]:
    """One run's keyword arguments for ``SIPNETModel``, from a SIPNET table.

    Only the SIPNET parameters the table holds are returned; the model's base
    parameter set supplies the rest.

    Parameters
    ----------
    table:
        A SIPNET table, from :meth:`ParameterVector.sipnet_table`.
    site:
        The site id.
    batch:
        ``{batch dim: label}`` for every batch dim of the table, such as
        ``{"sample": 3}``: ``0`` to ``J - 1`` for a table built from Flat, or
        the Fields' own labels for one built from Fields. Required when the
        table has a batch dim, and a dim the table lacks is refused.

    Returns
    -------
    dict[str, float]
        ``{SIPNET parameter: value}``, Python floats.

    Raises
    ------
    TypeError
        If *site* or a batch label is a boolean, a float or not an integer,
        or *batch* is not a mapping.
    ValueError
        If a table variable has a dim that is neither a batch dim (integer
        labels), ``site`` nor ``time``; or if *batch* does not name exactly
        the table's batch dims.
    KeyError
        If *site*, or a batch label, is not in the table.

    Examples
    --------
    With ``model`` a ``SIPNETModel`` and ``table`` a SIPNET table::

        model(**sipnet_overrides(table, batch={"sample": 3}, site=27))
    """
    site_id = as_site_id(site, message_name="site")
    requested = _requested_batch_labels(batch)
    for name, variable in table.data_vars.items():
        check_dims_are_batch_spatial_or_time(variable, message_name=repr(str(name)))
    selected = table.sel({SITE: site_id})
    check_batch_labels_name_the_table_batch_dims(batch_dims(selected), requested)
    for dim, label in requested.items():
        check_batch_label_is_in_the_table(table, dim, label)
    if requested:
        selected = selected.sel(requested)
    return {str(name): float(value) for name, value in selected.data_vars.items()}


def _requested_batch_labels(batch: Any) -> dict[str, int]:
    """*batch* as ``{dim: label}`` with plain-integer labels, or empty for ``None``."""
    if batch is None:
        return {}
    check_batch_labels_are_a_mapping(batch)
    return {
        dim: as_batch_label(label, message_name=f"batch[{dim!r}]") for dim, label in batch.items()
    }


# ── the example ───────────────────────────────────────────────────────────────


def example_parameter_vector(
    sites: Sequence[int] | pd.DataFrame, *, pft: Sequence[str] | pd.DataFrame
) -> ParameterVector:
    """A small example vector with a shared, a per-class and a per-site
    calibration parameter of each kind of prior. **Not the calibration
    vector, and not a reviewed prior.**

    This module's worked example and test fixture. Each center either traces
    to the BETY reanalysis trait posteriors or another named source, or says
    it is a placeholder, and every provenance string says what the value is
    not. Nothing comes from ``template.param``. The vector is partial:
    :attr:`ParameterVector.unset_sipnet_parameter_names` lists what a run takes
    from its base parameter set.

    Parameters
    ----------
    sites:
        Site ids, ascending, or a site table in ascending ``site_id`` order.
    pft:
        One PFT label per site, or a site-labels product; named ``"pft"`` in
        the vector.

    Returns
    -------
    ParameterVector
        With calibration parameters ``photosynthesis`` (shared, size 2,
        :data:`PHOTOSYNTHESIS`), ``allocation`` (by ``"pft"``, size 3,
        :data:`ALLOCATION`), ``base_soil_respiration`` (by ``"pft"``,
        log-normal), ``leaf_fall_fraction`` (shared, logit-normal) and
        ``initial_soil_carbon`` (by ``"site"``, log-normal, one prior per
        site); fixed ``daily_mean_photosynthesis_fraction`` (shared),
        ``leaf_carbon_fraction`` (by ``"pft"``) and
        ``vapor_pressure_deficit_exponent`` (shared).
    """
    n_sites = len(sites)
    labels = _declared_classes_of("pft", pft)
    fixture = "Example fixture, not a reviewed prior. "

    # Temperate-deciduous BETY medians and 2.5-97.5% quantiles, with aMaxFrac
    # 0.76 and cFracLeaf 0.466 held fixed:
    # P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf and
    # rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac).
    a_max_frac, c_frac_leaf = 0.76, 0.466
    a_max, fol_resp = 58.0, 0.17
    fol_resp_lower, fol_resp_upper = 0.10, 0.39
    photosynthesis = CalibrationParameter(
        name="photosynthesis",
        prior=product_transformed_gaussian_prior(
            capacity=log_normal(
                median=a_max * (a_max_frac + fol_resp) / c_frac_leaf, geometric_sd=1.75
            ),
            respiration_share=logit_normal_from_interval(
                lower=fol_resp_lower / (a_max_frac + fol_resp_lower),
                upper=fol_resp_upper / (a_max_frac + fol_resp_upper),
            ),
        ),
        sipnet_map=PHOTOSYNTHESIS,
        provenance=fixture
        + "Capacity median from the temperate-deciduous BETY posterior medians aMax 58 "
        "nmol g-1 s-1 and baseFolRespFrac 0.17, with aMaxFrac 0.76 and cFracLeaf 0.466 "
        "(BETY leafC). Geometric sd 1.75 is about twice, on the log scale, the 1.32 the "
        "BETY aMax 2.5-97.5% range 28-83 implies: a meta-analysis prior is to be "
        "widened, by a factor not yet decided. Respiration share interval from the "
        "baseFolRespFrac 2.5-97.5% range 0.10-0.39 at fixed aMaxFrac. Deciduous values "
        "applied to every PFT here.",
    )
    allocation = CalibrationParameter(
        name="allocation",
        prior=softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5),
        sipnet_map=ALLOCATION,
        varies_by="pft",
        provenance=fixture
        + "Center is the temperate-deciduous BETY allocation posterior medians, leaf "
        "0.18, wood 0.40, fine root 0.07, coarse root the 0.35 remainder; logit sd 0.5 "
        "is a placeholder. One copy per PFT, all with this prior.",
    )
    base_soil_respiration = CalibrationParameter(
        name="base_soil_respiration",
        prior=log_normal_from_interval(lower=0.004, upper=0.020),
        sipnet_map="base_soil_respiration_rate",
        varies_by="pft",
        provenance=fixture
        + "BETY som_respiration_rate posterior, 2.5-97.5% quantiles 0.004-0.020 yr-1; "
        "BETY has one prior for every PFT, so every PFT copy uses this interval.",
    )
    leaf_fall_fraction = CalibrationParameter(
        name="leaf_fall_fraction",
        prior=logit_normal(median=0.5, logit_sd=1.7),
        sipnet_map="leaf_off_fall_fraction",
        provenance=fixture
        + "No elicited value: a near-flat logit-normal on (0, 1), median 0.5, logit sd "
        "1.7. A placeholder until a prior is fitted to BETY fracLeafFall.",
    )
    initial_soil_carbon = CalibrationParameter(
        name="initial_soil_carbon",
        prior=log_normal(median=np.full(n_sites, 30_000.0), geometric_sd=2.0),
        sipnet_map="soil_carbon",
        varies_by="site",
        provenance=fixture
        + "One prior per site, identical here: median 30 kg C m-2 (30000 g m-2 in "
        "pySIPNET's units) is the center of the 12-75 kg C m-2 that the ISCN-derived "
        "initial soil carbon spans at the first test sites; geometric sd 2 spans "
        "roughly 7.7-117 at 95%. The real per-site priors are fitted to the initial "
        "condition ensemble; this is a placeholder until they are.",
    )
    fixed = (
        FixedParameter(
            name="daily_mean_photosynthesis_fraction",
            value=a_max_frac,
            provenance=fixture
            + "0.76, within the BETY per-PFT posterior median range 0.75-0.86. Fixed "
            "because aMaxFrac spans one of the two exactly degenerate photosynthesis "
            "directions (sipnet.c:614, 617, 633).",
        ),
        FixedParameter(
            name="leaf_carbon_fraction",
            value={label: c_frac_leaf for label in labels},
            varies_by="pft",
            provenance=fixture
            + "BETY leafC posterior median for temperate deciduous, 0.466, applied to "
            "every PFT label here; BETY gives boreal conifer 0.506 and grassland 0.483. "
            "Fixed for the same reason as aMaxFrac.",
        ),
        FixedParameter(
            name="vapor_pressure_deficit_exponent",
            value=2.0,
            provenance="Braswell et al. (2005) fix the exponent at 2; the BETY draws of "
            "1.0-2.9 would add a near-degeneracy with dVpdSlope for nothing.",
        ),
    )
    return ParameterVector(
        parameters=(
            photosynthesis,
            allocation,
            base_soil_respiration,
            leaf_fall_fraction,
            initial_soil_carbon,
        ),
        fixed=fixed,
        sites=sites,
        site_labels={"pft": pft},
    )



# ── supporting helpers ────────────────────────────────────────────────────────

_FLAT_SPECS: Mapping[str, ParameterSpec] = FrozenMapping(
    {path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()}
)
_SPEC_ORDER: Mapping[str, int] = FrozenMapping({name: i for i, name in enumerate(_FLAT_SPECS)})


def _as_theta(theta: Any, dimension: int) -> Array:
    """*theta* as a JAX ``float64`` array, ``(D,)`` or ``(J, D)`` as given."""
    batched = as_batched_flat(theta, dimension, message_name="theta")
    return jnp.asarray(batched[0] if is_one_vector(theta) else batched)


def _positive_array(what: str, value: Any) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(array)) and jnp.all(array > 0)):
        raise ValueError(f"{what} must be finite and positive; got {value!r}.")
    return array


def _unit_interval_array(what: str, value: Any) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if not bool(jnp.all((array > 0) & (array < 1))):
        raise ValueError(f"{what} must lie strictly inside (0, 1); got {value!r}.")
    return array


def _logit(p: Array) -> Array:
    return jnp.log(p) - jnp.log1p(-p)


def _normal_from_interval(lower: Array, upper: Array, mass: float) -> tuple[Array, Array]:
    """The Normal whose central *mass* interval is ``[lower, upper]``."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must lie in (0, 1); got {mass}.")
    if not bool(jnp.all(upper > lower)):
        raise ValueError("upper must exceed lower.")
    z = tfd.Normal(jnp.float64(0.0), jnp.float64(1.0)).quantile(jnp.float64(0.5 + mass / 2))
    return (lower + upper) / 2.0, (upper - lower) / (2.0 * z)


def _samples_in_support(what: str, x: Any, in_support: Callable[[Array], Array]) -> Array:
    array = jnp.asarray(x, dtype=jnp.float64).ravel()
    if array.size < 2:
        raise ValueError(f"{what}: need at least two samples; got {array.size}.")
    bad = ~(jnp.isfinite(array) & in_support(array))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{what}: {int(bad.sum())} of {array.size} samples are NaN or outside the "
            "support. Exclude and count them before fitting; this function does not."
        )
    return array


def _positive_std(values: Array, what: str) -> Array:
    std = jnp.std(values)
    if not bool(std > 0):
        raise ValueError(f"{what}: samples are all equal; no scale can be fitted.")
    return std


def _as_labels(name: str, labels: Any) -> tuple[Any, ...]:
    """Site labels as a tuple of one label per site; refuses a string or a mapping."""
    if isinstance(labels, (str, bytes, Mapping)):
        raise TypeError(
            f"site labels {name!r} must be a sequence of one label per site, not "
            f"{type(labels).__name__}."
        )
    if hasattr(labels, "tolist"):  # numpy, jax, pandas
        labels = labels.tolist()
    return tuple(labels)


def _position(labels: Sequence[Any], label: Any, what: str) -> int:
    try:
        return list(labels).index(label)
    except ValueError:
        raise KeyError(f"{label!r} is not a {what}; have {list(labels)}.") from None


def _unconstrained_labels(bijector: tfb.Bijector, components: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(bijector, tfb.SoftmaxCentered):
        return tuple(f"alr({c}:{components[-1]})" for c in components[:-1])
    if isinstance(bijector, tfb.Blockwise) and len(bijector.bijectors) == len(components):
        return tuple(
            _scalar_unconstrained_label(b, c) for b, c in zip(bijector.bijectors, components)
        )
    if len(components) == 1:
        return (_scalar_unconstrained_label(bijector, components[0]),)
    return tuple(f"{type(bijector).__name__}^-1({c})" for c in components)


def _scalar_unconstrained_label(bijector: tfb.Bijector, component: str) -> str:
    if isinstance(bijector, tfb.Exp):
        return f"log({component})"
    if isinstance(bijector, tfb.Sigmoid):
        if bijector.low is None:
            return f"logit({component})"
        low, high = float(bijector.low), float(bijector.high)
        return f"logit({component} in ({low:g}, {high:g}))"
    if isinstance(bijector, tfb.Identity):
        return component
    return f"{type(bijector).__name__}^-1({component})"


def _distribution_name(prior: tfd.Distribution) -> str:
    if isinstance(prior, tfd.LogNormal):
        return "log-normal"
    if isinstance(prior, tfd.LogitNormal):
        return "logit-normal"
    if type(prior) is tfd.TransformedDistribution:
        if isinstance(prior.bijector, tfb.SoftmaxCentered):
            return "softmax-normal"
        if isinstance(prior.bijector, tfb.Blockwise):
            return "product of transformed Gaussians"
        return f"{type(prior.distribution).__name__} through {type(prior.bijector).__name__}"
    return type(prior).__name__


def _sipnet_attributes(name: str, source: str) -> dict[str, str]:
    spec = _FLAT_SPECS[name]
    attrs = {"units": spec.units, "sipnet_name": spec.sipnet_name, "source": source}
    if spec.constituent:
        attrs["constituent"] = spec.constituent
    return attrs


def _in_domain(domain: ParameterDomain, values: Array) -> bool:
    values = jnp.asarray(values)
    finite = jnp.all(jnp.isfinite(values))
    if domain is ParameterDomain.REAL:
        inside = True
    elif domain is ParameterDomain.POSITIVE:
        inside = jnp.all(values > 0)
    elif domain is ParameterDomain.NON_NEGATIVE:
        inside = jnp.all(values >= 0)
    elif domain is ParameterDomain.UNIT_INTERVAL:
        inside = jnp.all((values >= 0) & (values <= 1))
    elif domain is ParameterDomain.OPEN_UNIT_INTERVAL:
        inside = jnp.all((values > 0) & (values < 1))
    else:  # pragma: no cover - a new pySIPNET domain
        raise NotImplementedError(f"no domain predicate for {domain!r}.")
    return bool(finite & inside)


def _normalized_sites(sites: Any) -> tuple[tuple[int, ...], tuple[np.ndarray, np.ndarray] | None]:
    """Site ids, and ``lon``/``lat`` when *sites* is a site table that has
    them."""
    if not isinstance(sites, pd.DataFrame):
        return as_site_ids(sites, message_name="sites"), None
    check_site_table_is_keyed_on_site_ids(sites)
    table = site_lookup(sites)
    ids = as_site_ids(table.index, message_name="the site table's site_id")
    check_site_table_is_in_site_order(ids)
    if {LON, LAT} & set(table.columns):
        check_site_table_has_locations(table)
        check_site_table_positions_are_finite(table)
        return ids, (_read_only_copy(table[LON]), _read_only_copy(table[LAT]))
    return ids, None


def _read_only_copy(column: pd.Series) -> np.ndarray:
    """*column* as a ``float64`` array of its own, which cannot be written."""
    array = column.to_numpy(np.float64, copy=True)
    array.flags.writeable = False
    return array


def _normalized_site_labels(
    name: str, value: Any, sites: tuple[int, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """One site-labels argument as (the class of each site, in site order;
    the declared classes, in their order)."""
    if isinstance(value, pd.DataFrame):
        check_site_labels_product_has_columns(name, value)
        indexed = site_lookup(value)[LABEL_COLUMN]
        check_site_labels_product_covers_sites(name, indexed, sites)
        labels = tuple(indexed.loc[list(sites)].tolist())
        check_site_labels_are_present(name, labels)
        return labels, _declared_classes_of(name, value)
    if isinstance(value, pd.Categorical) or isinstance(
        getattr(value, "dtype", None), pd.CategoricalDtype
    ):
        categorical = pd.Categorical(value)
        check_site_labels_are_declared(name, categorical)
        return tuple(categorical.tolist()), tuple(categorical.categories.tolist())
    labels = _as_labels(name, value)
    check_site_labels_are_present(name, labels)
    return labels, _sorted_classes(name, labels)


def _declared_classes_of(name: str, value: Any) -> tuple[Any, ...]:
    """The classes a site-labels argument declares: a product's categories,
    else its sorted distinct labels."""
    if isinstance(value, pd.DataFrame):
        check_site_labels_product_has_columns(name, value)
        value = value[LABEL_COLUMN]
    if isinstance(value, pd.Categorical) or isinstance(
        getattr(value, "dtype", None), pd.CategoricalDtype
    ):
        return tuple(pd.Categorical(value).categories.tolist())
    labels = _as_labels(name, value)
    return _sorted_classes(name, tuple(label for label in labels if not pd.isna(label)))


def _sorted_classes(name: str, labels: Sequence[Any]) -> tuple[Any, ...]:
    """The distinct labels in sorted order, the order plain labels give."""
    try:
        return tuple(sorted(set(labels)))
    except TypeError:
        raise ValueError(
            f"site labels {name!r} mix types that cannot be ordered, so their classes have "
            "no group order; pass a site-labels product or a pandas categorical, whose "
            "categories give the order."
        ) from None


def _carries_its_bijector(prior: tfd.Distribution) -> bool:
    """Whether *prior* is built as a base and a bijector this module reads.

    TFP implements several distributions (``MultivariateNormalTriL``,
    ``Weibull``, ``Gumbel``, ...) as ``TransformedDistribution`` subclasses
    over an internal reparameterization, which is not an unconstrained space,
    so only the exact class and the two helpers' families count.
    """
    return type(prior) in (tfd.TransformedDistribution, tfd.LogNormal, tfd.LogitNormal)


def _restricted(
    parameter: CalibrationParameter, prior: tfd.Distribution, positions: np.ndarray
) -> CalibrationParameter:
    """*parameter* with *prior*, its restriction to the groups at
    *positions*, after checking the restriction is one."""
    restricted = dataclasses.replace(parameter, prior=prior)
    check_restriction_keeps_the_kept_groups(parameter, restricted, positions)
    return restricted


def _prior_for_groups(parameter: CalibrationParameter, positions: np.ndarray) -> tfd.Distribution:
    """An independent-copies prior, one group per entry of its TFP batch
    shape, restricted to the groups at *positions*."""
    prior = parameter.prior
    if type(prior) in (tfd.LogNormal, tfd.LogitNormal):
        batch = tuple(prior.batch_shape)
        return type(prior)(
            loc=jnp.broadcast_to(prior.loc, batch)[positions],
            scale=jnp.broadcast_to(prior.scale, batch)[positions],
        )
    if type(prior) is tfd.TransformedDistribution and type(
        prior.distribution
    ) is tfd.MultivariateNormalDiag:
        # TFP's own slicing of a MultivariateNormalDiag fails for two or more
        # indices, so the softmax-normal base is rebuilt from its moments.
        base = prior.distribution
        rebuilt = tfd.MultivariateNormalDiag(
            loc=base.mean()[positions], scale_diag=base.stddev()[positions]
        )
        return tfd.TransformedDistribution(rebuilt, prior.bijector)
    try:
        return prior[positions]
    except Exception as error:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: TFP cannot restrict its "
            f"{type(prior).__name__} prior to fewer groups; give its TFP batch shape one "
            "entry per group the vector has."
        ) from error


def _joint_marginal(parameter: CalibrationParameter, positions: np.ndarray) -> tfd.Distribution:
    """A joint prior restricted to the groups at *positions*: the Gaussian
    marginal of its base, exactly, under the same bijector."""
    if not parameter.has_analytic_moments:
        raise NotImplementedError(
            f"calibration parameter {parameter.name!r}: a joint prior without analytic "
            "unconstrained moments cannot be restricted to fewer groups."
        )
    base = parameter.unconstrained_prior
    mean = jnp.asarray(base.mean(), dtype=jnp.float64)[positions]
    covariance = jnp.asarray(base.covariance(), dtype=jnp.float64)[positions][:, positions]
    marginal = tfd.MultivariateNormalTriL(loc=mean, scale_tril=jnp.linalg.cholesky(covariance))
    return tfd.TransformedDistribution(marginal, parameter.bijector)


def _monte_carlo_moments(
    prior: tfd.Distribution, key: Array, n_moment_samples: int, independent_scalar: bool
) -> tuple[Array, Array]:
    """Mean and variance (independent scalar) or covariance from draws."""
    if n_moment_samples < 2:
        raise ValueError(
            "n_moment_samples must be at least 2 for Monte Carlo moment matching; "
            f"got {n_moment_samples}."
        )
    draws = prior.sample((n_moment_samples,), seed=key)
    mean = draws.mean(axis=0)
    if independent_scalar:
        return mean, draws.var(axis=0, ddof=1)
    centered = draws - mean
    subscripts = "ni,nj->ij" if draws.ndim == 2 else "ngi,ngj->gij"
    return mean, jnp.einsum(subscripts, centered, centered) / (n_moment_samples - 1)


def _batch_labels(n_samples: int) -> np.ndarray:
    """The labels of a batch made from Flat: ``0`` to ``n_samples - 1``, ``int64``."""
    return np.arange(n_samples, dtype=BATCH_LABEL_DTYPE)


def _as_batch_labels(values: Any) -> np.ndarray:
    """Batch labels carried from Fields, as ``int64``."""
    check_batch_labels_are_distinct_integers(values)
    return np.asarray(values).astype(BATCH_LABEL_DTYPE)


def _space_labels(parameter: CalibrationParameter, space: str) -> tuple[str, ...]:
    """Component names in natural space, element labels in unconstrained."""
    return parameter.components if space == NATURAL else parameter.element_labels


def _fields_variable_names(parameter: CalibrationParameter, space: str) -> tuple[str, ...]:
    """The Fields variables a calibration parameter occupies in *space*."""
    labels = _space_labels(parameter, space)
    if len(labels) == 1:
        return (parameter.name,)
    return tuple(f"{parameter.name}.{label}" for label in labels)


def _fields_attributes(
    parameter: CalibrationParameter, labels: tuple[str, ...], index: int, space: str
) -> dict[str, str]:
    label = labels[index]
    attributes = {"parameter": parameter.name}
    if len(labels) > 1:
        attributes["component"] = label
    attributes.update(
        varies_by=parameter.varies_by or SHARED,
        space=space,
        long_name=f"{parameter.name}: {label}",
        units=parameter.component_units[index] if space == NATURAL else "1",
        sipnet_parameters=", ".join(parameter.sipnet_map.writes),
    )
    return attributes


def _variable_values(array: xr.DataArray, dims: tuple[str, ...]) -> np.ndarray:
    check_fields_variable_dims(array, dims)
    return np.asarray(array.transpose(*dims).values, dtype=np.float64)


def _summary(vector: ParameterVector) -> str:
    """The text of ``repr(vector)``."""
    site_count = len(vector.sites)
    labels = "; ".join(
        f"{name} {{{', '.join(map(str, vector.group_labels(name)))}}}"
        for name in vector.site_labels
    )
    lines = [
        f"ParameterVector  D = {vector.dimension}  |  {site_count} site"
        f"{'' if site_count == 1 else 's'}  |  site labels: {labels or 'none'}"
    ]
    rows = [("parameter", "varies by", "groups", "size", "prior", "-> SIPNET parameters")]
    for p in vector.parameters:
        prior = p.distribution_name + (", joint over groups" if p.is_joint else "")
        rows.append(
            (
                p.name,
                p.varies_by or SHARED,
                str(vector.n_groups(p.varies_by)),
                str(p.size),
                prior,
                ", ".join(p.sipnet_map.writes),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        cells = [
            cell.rjust(width) if i in (2, 3) else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  " + "  ".join(cells) + "  " + row[-1])
    fixed = ", ".join(f"{f.name} ({f.varies_by or SHARED})" for f in vector.fixed)
    lines.append(f"  fixed: {fixed or 'none'}")
    unset = len(vector.unset_sipnet_parameter_names)
    lines.append(
        f"  unset: {unset} required SIPNET parameters, taken from the run's base parameter set"
        if unset
        else "  unset: none; every required SIPNET parameter is calibrated or fixed"
    )
    return "\n".join(lines)


# ── checks ────────────────────────────────────────────────────────────────────


def check_parameter_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"calibration parameter name {name!r} is not lower_case_with_underscores; see "
            "NAME_PATTERN."
        )
    if name in RESERVED_PARAMETER_NAMES:
        raise ValueError(
            f"calibration parameter name {name!r} is reserved: a Fields variable of that name "
            f"would collide with a coordinate ({sorted(RESERVED_PARAMETER_NAMES)})."
        )


def check_provenance_is_given(name: str, provenance: str) -> None:
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            f"{name}: provenance is empty. Say where the value came from; a placeholder "
            "says it is one."
        )


def check_sipnet_parameters_exist(names: Sequence[str], context: str) -> None:
    unknown = [n for n in names if n not in _FLAT_SPECS]
    if unknown:
        raise ValueError(
            f"{context}: {unknown} are not pySIPNET parameter names. Use the flat parameter "
            "names of pysipnet.parameters.model.PARAMETER_SPECS, e.g. "
            "'max_photosynthesis_rate', not 'aMax'."
        )


def check_prior_is_a_distribution(parameter: CalibrationParameter) -> None:
    if not isinstance(parameter.prior, tfd.Distribution):
        raise TypeError(
            f"calibration parameter {parameter.name!r}: prior must be a TFP distribution "
            f"(tensorflow_probability.substrates.jax.distributions); got "
            f"{type(parameter.prior).__name__}."
        )


def check_sipnet_map_is_a_sipnet_map(parameter: CalibrationParameter) -> None:
    if not isinstance(parameter.sipnet_map, SIPNETMap):
        raise TypeError(
            f"calibration parameter {parameter.name!r}: sipnet_map must be a SIPNETMap "
            "(with writes, reads, components, component_units and __call__) or a SIPNET "
            f"parameter name; got {type(parameter.sipnet_map).__name__}."
        )


def check_prior_is_float64(parameter: CalibrationParameter) -> None:
    dtype = parameter.prior.dtype
    try:
        is_float64 = jnp.dtype(dtype) == jnp.dtype(jnp.float64)
    except TypeError:  # a structured dtype, as a joint distribution has
        is_float64 = False
    if not is_float64:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: prior dtype is {dtype}, not float64. "
            "TFP builds float32 from Python floats and lists; pass jnp.float64 values or use "
            "the prior helpers."
        )


def check_prior_event_rank(parameter: CalibrationParameter) -> None:
    event = tuple(parameter.prior.event_shape)
    k = parameter.natural_size
    if k == 1 and len(event) > 1:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: prior event shape {event} has rank "
            "above 1; a scalar calibration parameter's prior has event () for independent "
            "copies or (n_groups,) for one joint prior over its groups."
        )
    if k > 1 and len(event) == 2:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: prior event shape {event} would be a "
            "joint prior over the groups of a vector-valued calibration parameter, which is "
            f"not supported; give each copy event ({k},)."
        )
    if len(event) > 2:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: prior event shape {event} has rank "
            "above 1; one copy is a scalar or a vector."
        )


def check_prior_has_a_bijector(parameter: CalibrationParameter) -> None:
    if parameter.bijector is None:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: {type(parameter.prior).__name__} has "
            "no default event-space bijector. Give the prior as a TransformedDistribution "
            "with an explicit bijector, e.g. TransformedDistribution(prior, Exp())."
        )


def check_prior_batch_rank(parameter: CalibrationParameter) -> None:
    batch = tuple(parameter.prior.batch_shape)
    if parameter.is_joint and batch:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: a joint prior over groups has batch "
            f"shape (), not {batch}."
        )
    if len(batch) > 1:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: prior batch shape {batch} has rank "
            "above 1; it must be () or (n_groups,)."
        )


def check_components_match_prior(parameter: CalibrationParameter) -> None:
    event = tuple(parameter.prior.event_shape)
    if parameter.is_joint:
        unconstrained = tuple(parameter.bijector.inverse_event_shape(parameter.prior.event_shape))
        if unconstrained != event:
            raise ValueError(
                f"calibration parameter {parameter.name!r}: a joint prior over groups needs an "
                f"elementwise bijector, but {type(parameter.bijector).__name__} changes the "
                f"event shape from {event} to {unconstrained}. If the prior is one "
                f"vector-valued copy, its SIPNET map must name {event[0]} components."
            )
        return
    natural = 1 if event == () else event[0]
    if natural != parameter.natural_size:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: the SIPNET map names "
            f"{parameter.natural_size} components {parameter.components} but the prior's "
            f"natural value has {natural}. Match the prior (product_transformed_gaussian_prior "
            "keyword order, softmax_normal center length) to the SIPNET map."
        )


def check_fixed_value_shape(parameter: FixedParameter) -> None:
    if parameter.varies_by is None and isinstance(parameter.value, Mapping):
        raise ValueError(
            f"fixed parameter {parameter.name!r}: a shared value must be a float, not a "
            "mapping; set varies_by to key it on groups."
        )
    if parameter.varies_by is not None and not isinstance(parameter.value, Mapping):
        raise ValueError(
            f"fixed parameter {parameter.name!r} varies by {parameter.varies_by!r}; its value "
            "must be a mapping from every group to a float."
        )


def check_fixed_value_is_in_domain(name: str, value: float) -> None:
    domain = _FLAT_SPECS[name].domain
    if not _in_domain(domain, jnp.asarray(value, dtype=jnp.float64)):
        raise ValueError(
            f"fixed parameter {name!r} = {value!r} is outside its pySIPNET domain "
            f"{domain.value!r}; pySIPNET would refuse it."
        )


def check_fixed_value_covers_groups(parameter: FixedParameter, groups: tuple[Any, ...]) -> None:
    if not isinstance(parameter.value, Mapping):
        return
    missing = [g for g in groups if g not in parameter.value]
    extra = [g for g in parameter.value if g not in groups]
    if missing or extra:
        raise ValueError(
            f"fixed parameter {parameter.name!r} by {parameter.varies_by!r}: values must be "
            f"keyed on exactly the groups {list(groups)}; missing {missing}, extra {extra}."
        )


def check_vector_has_a_site(sites: tuple[int, ...]) -> None:
    """The vector has at least one site."""
    if not sites:
        raise ValueError("a ParameterVector needs at least one site.")


def check_sites_are_ascending(sites: tuple[int, ...]) -> None:
    """The vector's sites are strictly ascending, so each is listed once."""
    if list(sites) != sorted(set(sites)):
        raise ValueError("sites must be strictly ascending with no repeats.")


def check_site_labels_product_has_columns(name: str, frame: pd.DataFrame) -> None:
    missing = sorted({SITE_ID, LABEL_COLUMN} - set(frame.columns))
    if missing:
        raise ValueError(
            f"site labels {name!r}: a site-labels product needs {SITE_ID!r} and "
            f"{LABEL_COLUMN!r} columns, as sipnet_calibration.site_labels.load_site_labels "
            f"returns; missing {missing}."
        )


def check_site_labels_product_covers_sites(
    name: str, labels: pd.Series, sites: tuple[int, ...]
) -> None:
    if not labels.index.is_unique:
        raise ValueError(f"site labels {name!r}: the product repeats a site_id.")
    missing = [s for s in sites if s not in labels.index]
    if missing:
        raise ValueError(
            f"site labels {name!r} do not label sites {missing}; a site-labels product must "
            "label every site of the vector."
        )


def check_site_labels_are_declared(name: str, labels: pd.Categorical) -> None:
    if (labels.codes < 0).any():
        raise ValueError(
            f"site labels {name!r}: some sites carry no declared class (a missing value, or "
            "a label outside the categorical's categories)."
        )


def check_parameter_names_are_unique(parameters: tuple[CalibrationParameter, ...]) -> None:
    names = [p.name for p in parameters]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"calibration parameter names repeat: {repeated}.")
    if not names:
        raise ValueError("a ParameterVector needs at least one calibration parameter.")


def check_site_labels_cover_sites(vector: ParameterVector) -> None:
    reserved = RESERVED_SITE_LABELS_NAMES & set(vector.site_labels)
    if reserved:
        raise ValueError(f"site-labels names {sorted(reserved)} are reserved dimension names.")
    malformed = sorted(n for n in vector.site_labels if not NAME_PATTERN.match(str(n)))
    if malformed:
        raise ValueError(
            f"site-labels names {malformed} are not lower_case_with_underscores; each becomes a "
            "coordinate of the datasets this module builds."
        )
    for name, labels in vector.site_labels.items():
        if len(labels) != len(vector.sites):
            raise ValueError(
                f"site labels {name!r} have {len(labels)} labels for {len(vector.sites)} "
                "sites; give one label per site, in site order."
            )
    taken = set(vector.parameter_names) | set(_FLAT_SPECS)
    colliding = sorted(taken & set(vector.site_labels))
    if colliding:
        raise ValueError(
            f"site-labels names {colliding} collide with a calibration parameter or SIPNET "
            "parameter name; both become variables or coordinates of the datasets this "
            "module builds."
        )
    used = {p.varies_by for p in vector.parameters} | {f.varies_by for f in vector.fixed}
    missing = sorted(v for v in used if v not in (None, SITE) and v not in vector.site_labels)
    if missing:
        raise ValueError(
            f"varies_by names {missing} have no site labels; pass site_labels={{name: labels}}."
        )


def check_each_sipnet_parameter_has_one_writer(
    parameters: tuple[CalibrationParameter, ...], fixed: tuple[FixedParameter, ...]
) -> None:
    writers: dict[str, list[str]] = {}
    for parameter in parameters:
        for name in parameter.sipnet_map.writes:
            writers.setdefault(name, []).append(f"calibration parameter {parameter.name}")
    for parameter in fixed:
        writers.setdefault(parameter.name, []).append("fixed")
    clashes = {name: who for name, who in writers.items() if len(who) > 1}
    if clashes:
        raise ValueError(
            f"SIPNET parameters set more than once: {clashes}. Each SIPNET parameter has "
            "exactly one writer, a calibration parameter or a fixed value."
        )


def check_reads_are_fixed(
    parameters: tuple[CalibrationParameter, ...], fixed: tuple[FixedParameter, ...]
) -> None:
    fixed_names = {f.name for f in fixed}
    for parameter in parameters:
        missing = [n for n in parameter.sipnet_map.reads if n not in fixed_names]
        if missing:
            raise ValueError(
                f"calibration parameter {parameter.name!r} reads {missing}, which are not "
                "fixed. Add a FixedParameter for each."
            )


def check_prior_shape_matches_groups(parameter: CalibrationParameter, n_groups: int) -> None:
    if parameter.is_joint:
        if parameter.joint_groups != n_groups:
            raise ValueError(
                f"calibration parameter {parameter.name!r} varies by {parameter.varies_by!r} "
                f"with {n_groups} groups, but its joint prior is over {parameter.joint_groups}."
            )
        return
    batch = tuple(parameter.prior.batch_shape)
    if batch not in ((), (n_groups,)):
        raise ValueError(
            f"calibration parameter {parameter.name!r} varies by {parameter.varies_by!r} with "
            f"{n_groups} groups, but its prior has batch shape {batch}; use () for one prior "
            f"shared by every group or ({n_groups},) for one per group."
        )


def check_prior_spread_is_positive(parameter: CalibrationParameter, spread: Array) -> None:
    diagonal = spread if spread.ndim == 1 else jnp.diagonal(spread, axis1=-2, axis2=-1)
    if not bool(jnp.all(jnp.isfinite(spread)) and jnp.all(diagonal > 0)):
        raise ValueError(
            f"calibration parameter {parameter.name!r}: its unconstrained prior has a zero, "
            "negative or non-finite variance, which pyEKI cannot whiten. Give every element a "
            "positive spread (geometric_sd above 1, logit_sd above 0)."
        )


def check_every_required_parameter_is_set(vector: ParameterVector) -> None:
    unset = vector.unset_sipnet_parameter_names
    if unset:
        raise ValueError(
            f"require_complete: {len(unset)} required SIPNET parameters are neither "
            f"calibrated nor fixed: {list(unset)}. Add a CalibrationParameter or a "
            "FixedParameter for each, or build with require_complete=False to take them from "
            "the base set."
        )


def check_sipnet_map_image_is_in_domain(
    vector: ParameterVector, parameter: CalibrationParameter
) -> None:
    """Evaluate ``M_c(T_c(theta))`` at the corners of the unconstrained cube
    (:data:`DOMAIN_CHECK_CORNERS`) for every group, and test each SIPNET
    parameter written against its ``ParameterDomain``.

    Exhaustive for the shipped SIPNET maps and bijectors, whose outputs are
    monotone in each unconstrained element, so an excursion shows at a
    corner. A hand-written SIPNET map that is not monotone can leave a domain
    between the corners and is not caught here.
    """
    n_groups = vector.n_groups(parameter.varies_by)
    corners = jnp.asarray(
        list(itertools.product(DOMAIN_CHECK_CORNERS, repeat=parameter.size)), dtype=jnp.float64
    )
    theta_c = jnp.broadcast_to(corners[:, None, :], (corners.shape[0], n_groups, parameter.size))
    natural = ParameterVector._to_natural(parameter, theta_c)
    on_sites = vector._group_values_onto_sites(parameter, natural)
    written = parameter.sipnet_map(on_sites, vector._fixed_table)
    check_sipnet_map_returns_what_it_writes(parameter, written)
    for name, values in written.items():
        domain = _FLAT_SPECS[name].domain
        if not _in_domain(domain, values):
            raise ValueError(
                f"calibration parameter {parameter.name!r} can write {name!r} outside its "
                f"pySIPNET domain {domain.value!r} (checked at theta in {DOMAIN_CHECK_CORNERS}). "
                "Use a prior whose bijector maps onto the domain: log_normal for a positive "
                "parameter, logit_normal for a fraction, softmax_normal for a simplex."
            )


def check_space_is_known(space: str) -> None:
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}; got {space!r}.")


def check_fields_space_is_given(fields: xr.Dataset) -> str:
    space = fields.attrs.get("space")
    if space not in SPACES:
        raise ValueError(
            f"Fields must carry attrs['space'] in {SPACES}; got {space!r}. Build Fields with "
            "ParameterVector.fields, or set the attribute to say which space the values are in."
        )
    return space


def check_fields_hold_the_sites(fields: xr.Dataset, sites: tuple[int, ...]) -> None:
    if SITE not in fields.coords:
        raise ValueError("Fields must have a 'site' coordinate.")
    present = set(np.atleast_1d(np.asarray(fields[SITE].values)).tolist())
    missing = [s for s in sites if s not in present]
    if missing:
        raise ValueError(f"Fields lack sites {missing} of this vector.")


def check_fields_hold_the_variables(
    fields: xr.Dataset, names: tuple[str, ...], parameter: str
) -> None:
    missing = [n for n in names if n not in fields.data_vars]
    if missing:
        raise ValueError(
            f"Fields lack variables {missing} of calibration parameter {parameter!r}; in "
            f"{fields.attrs['space']} space it occupies {list(names)}."
        )


def check_fields_variable_dims(array: xr.DataArray, dims: tuple[str, ...]) -> None:
    if set(array.dims) != set(dims):
        raise ValueError(
            f"Fields variable {array.name!r} has dims {array.dims}; every variable must be on "
            f"{dims}."
        )


def check_fields_values_are_finite(values: np.ndarray, parameter: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(
            f"Fields values of calibration parameter {parameter!r} are not all finite; no "
            "Flat vector holds NaN or infinity."
        )


def check_group_values_agree_across_sites(
    vector: ParameterVector,
    parameter: CalibrationParameter,
    on_sites: np.ndarray,
    expected: np.ndarray,
) -> None:
    if np.array_equal(on_sites, expected):
        return
    site_axis = on_sites.ndim - 2
    other_axes = tuple(i for i in range(on_sites.ndim) if i != site_axis)
    differing = np.flatnonzero(np.any(on_sites != expected, axis=other_axes))
    index = vector._site_group_index(parameter.varies_by)
    group = index[differing[0]]
    members = [vector.sites[i] for i in np.flatnonzero(index == group)]
    raise ValueError(
        f"calibration parameter {parameter.name!r} varies by "
        f"{parameter.varies_by or SHARED!r}, but its values differ between the sites of group "
        f"{vector.group_labels(parameter.varies_by)[group]!r} ({members}). Fields must hold one "
        "value per group, repeated at every site of the group."
    )


def check_parameters_have_their_types(
    parameters: tuple[Any, ...], fixed: tuple[Any, ...]
) -> None:
    wrong = [type(p).__name__ for p in parameters if not isinstance(p, CalibrationParameter)]
    if wrong:
        raise TypeError(f"parameters= takes CalibrationParameters; got {wrong}.")
    wrong = [type(f).__name__ for f in fixed if not isinstance(f, FixedParameter)]
    if wrong:
        raise TypeError(f"fixed= takes FixedParameters; got {wrong}.")


def check_fixed_values_are_numbers(parameter: FixedParameter) -> None:
    values = parameter.value.values() if isinstance(parameter.value, Mapping) else [parameter.value]
    wrong = [v for v in values if isinstance(v, bool) or not isinstance(v, (int, float, np.number))]
    if wrong:
        raise TypeError(f"fixed parameter {parameter.name!r}: values must be numbers; got {wrong}.")


def check_site_table_is_in_site_order(ids: tuple[int, ...]) -> None:
    """A site table given as ``sites=`` lists its sites ascending, each once."""
    if list(ids) != sorted(set(ids)):
        raise ValueError(
            "a site table passed as sites= must be in ascending site_id order with no repeats, "
            "because a per-site prior and site labels given as a plain sequence are read in "
            "site order; sort it with .sort_values('site_id')."
        )


def check_site_table_positions_are_finite(table: pd.DataFrame) -> None:
    """A site table given as ``sites=`` has a finite ``lon`` and ``lat`` for every site."""
    if not np.isfinite(table[[LON, LAT]].to_numpy(np.float64)).all():
        raise ValueError(
            "a site table passed as sites= has missing or non-finite lon/lat; give every "
            "site its coordinates, or pass a table without lon and lat."
        )


def check_site_labels_are_present(name: str, labels: Sequence[Any]) -> None:
    missing = [i for i, label in enumerate(labels) if pd.isna(label)]
    if missing:
        raise ValueError(
            f"site labels {name!r} give no class for {len(missing)} of the vector's sites "
            f"(positions {missing[:5]}); every site needs a class."
        )


def check_sipnet_map_returns_what_it_writes(
    parameter: CalibrationParameter, written: Mapping[str, Any]
) -> None:
    declared = set(parameter.sipnet_map.writes)
    if set(written) != declared:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: its SIPNET map declares writes "
            f"{sorted(declared)} but returns {sorted(written)}; they must be the same."
        )


def check_restriction_keeps_the_kept_groups(
    parameter: CalibrationParameter, restricted: CalibrationParameter, positions: np.ndarray
) -> None:
    """The restricted prior's bijector and base agree with the original's at
    the kept groups; a bijector with a parameter per group, which TFP cannot
    always slice, fails here rather than silently mapping the wrong groups."""
    n_groups = parameter.joint_groups or int(parameter.prior.batch_shape[0])
    probe = jnp.linspace(-1.5, 1.5, n_groups * parameter.size).reshape(n_groups, parameter.size)
    try:
        expected = ParameterVector._to_natural(parameter, probe)[positions]
        actual = ParameterVector._to_natural(restricted, probe[positions])
        agrees = actual.shape == expected.shape and bool(jnp.allclose(actual, expected, rtol=1e-12))
        if agrees and parameter.has_analytic_moments and restricted.has_analytic_moments:
            full = jnp.broadcast_to(
                parameter.unconstrained_prior.mean(),
                (n_groups, *parameter.unconstrained_prior.event_shape[parameter.is_joint:]),
            )
            agrees = bool(jnp.allclose(restricted.unconstrained_prior.mean(), full[positions]))
    except Exception:  # a shape error from a bijector parameterized per group
        agrees = False
    if not agrees:
        raise ValueError(
            f"calibration parameter {parameter.name!r}: its prior cannot be restricted to the "
            "groups the vector has, most likely because its bijector has a parameter per "
            "group; give it one copy per group the vector has."
        )


def check_fields_variables_are_in_the_space(
    fields: xr.Dataset, names: tuple[str, ...], space: str
) -> None:
    other = [n for n in names if fields[n].attrs.get("space", space) != space]
    if other:
        raise ValueError(
            f"Fields variables {other} say they are not in {space} space, the dataset's "
            "attrs['space']; a dataset holds one space."
        )


def check_natural_values_are_in_the_support(
    parameter: CalibrationParameter, natural: Array, unconstrained: Array
) -> None:
    # A fresh copy, since TFP bijectors cache forward/inverse pairs and would
    # hand the input back unchanged.
    back = ParameterVector._to_natural(parameter, jnp.array(unconstrained, copy=True))
    inside = bool(jnp.all(jnp.isfinite(unconstrained))) and bool(
        jnp.allclose(back, natural, rtol=1e-9, atol=1e-12)
    )
    if not inside:
        raise ValueError(
            f"Fields values of calibration parameter {parameter.name!r} are outside the image "
            "of its bijector: a value outside its prior's support (a negative rate, a "
            "fraction at or beyond 0 or 1) or simplex components that do not sum to 1. No "
            "Flat vector maps to them."
        )


def check_batch_labels_are_distinct_integers(values: Any) -> None:
    """Fields' batch labels are distinct integers."""
    labels = np.asarray(values)
    integral = labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
    if not (integral and labels.size > 0 and len(set(labels.tolist())) == labels.size):
        raise ValueError(
            "Fields batch labels must be distinct integers; got "
            f"{labels.tolist()[:10]} ({labels.dtype})."
        )


def check_batch_dim_name_is_not_taken(vector: ParameterVector, batch_dim: Any) -> None:
    """*batch_dim* can name a batch dim of this vector's Fields and SIPNET table.

    Not a reserved name
    (:func:`sipnet_calibration.fields.check_batch_dim_name_is_not_reserved`),
    not a site-labels name (a coordinate on ``site``), and not a name a
    variable of either takes: a SIPNET parameter name, a Fields variable name
    in either space, or a calibration parameter name.
    """
    check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
    taken = {**dict.fromkeys(vector.site_labels, "a site-labels name")}
    taken.update(dict.fromkeys(vector.sipnet_parameter_names, "a SIPNET parameter name"))
    for parameter in vector.parameters:
        for space in SPACES:
            taken.update(
                dict.fromkeys(_fields_variable_names(parameter, space), "a Fields variable name")
            )
    taken.update(dict.fromkeys(vector.parameter_names, "a calibration parameter name"))
    if batch_dim in taken or batch_dim in (SHARED, SITE_ID):
        what = taken.get(batch_dim, "a reserved name")
        raise ValueError(
            f"batch_dim={batch_dim!r} is {what} of this vector; name the batch dim otherwise, "
            "such as 'sample'."
        )


def check_batch_labels_name_the_table_batch_dims(
    table_batch_dims: tuple[str, ...], requested: Mapping[str, Any]
) -> None:
    """*batch* names exactly the table's batch dims."""
    missing = [d for d in table_batch_dims if d not in requested]
    extra = [d for d in requested if d not in table_batch_dims]
    if missing or extra:
        raise ValueError(
            f"the table has the batch dims {list(table_batch_dims)} and batch= names "
            f"{list(requested)}; pass batch={{dim: label}} for each of the table's batch "
            "dims and no other."
        )


def check_batch_label_is_in_the_table(table: xr.Dataset, dim: str, label: Any) -> None:
    """The label asked for is one of the table's labels on *dim*."""
    labels = np.asarray(table[dim].values).tolist()
    if label not in labels:
        raise KeyError(
            f"{dim} {label!r} is not one of the table's {dim} labels "
            f"({labels[:5]}{', ...' if len(labels) > 5 else ''})."
        )

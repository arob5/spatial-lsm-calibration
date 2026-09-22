"""The calibration vector: coordinates, priors, bijectors and the map to
pySIPNET parameters.

Overview
--------
pySIPNET owns the physical SIPNET parameters, their units and their domains.
pyEKI takes a Gaussian prior over an unconstrained ``(J, D)`` ensemble and a
callable forward model, and leaves transforms, constraints and priors to the
caller. This module is that caller. A :class:`Parameterization` says which
SIPNET parameters are calibrated, in what coordinates, under what prior, with
what held fixed, and converts between the sampler's view of the vector and
pySIPNET's::

    theta_c  --T_c-->  vartheta_c  --M_c(., fixed)-->  {SIPNET parameter: value}

where ``theta`` is the calibration vector in unconstrained coordinates,
``T_c`` is coordinate ``c``'s TFP bijector, ``vartheta_c`` its value on the
natural scale (the scale its prior is named on), and ``M_c`` its
:class:`CoordToParamMap`. ``T_c`` is invertible; ``M_c`` in general is not.

The dependency runs one way::

    pysipnet.parameters.model.PARAMETER_SPECS   (names, domains, units)
    pyeki.gauss / pyeki.linalg                  (Gaussian, PSDBlockDiag, ...)
    tensorflow_probability.substrates.jax       (distributions, bijectors)
      -> this module
      -> experiments/<task>/config.py           (names a Parameterization)

Nothing here reads a data product. The design is
``logs/2026-09-21_Parameterization Layer Design.md`` in the vault.

Vocabulary
----------
``theta``
    The calibration vector, ``(D,)``, or an ensemble ``(J, D)``.
coordinate
    One named piece of ``theta``: a scalar (the log of a rate) or a short
    vector (the allocation simplex). Coordinate ``c`` occupies one contiguous
    slice ``theta_c`` that reshapes to ``(n_groups, size)``.
natural scale
    Where a coordinate's prior is named and a person reads it: a rate in
    yr-1, a fraction, a point on the simplex. Its value is ``vartheta_c``.
SIPNET parameter
    One of pySIPNET's flat parameter names, ``max_photosynthesis_rate``; the
    keyword ``SIPNETModel(**overrides)`` takes. Never "field".
group
    A coordinate *varies by* a labeling of the sites and takes one value per
    group of it: one when shared, one per site, one per PFT, one per
    land-cover class.

Data model
----------
**The layout of theta.** Coordinates in declaration order; within a
coordinate, groups in group order; within a group, elements in order. Each
coordinate owns one contiguous slice; ``D`` is the sum over coordinates of
``n_groups * size``. The rule is implemented once, in :class:`Layout`, and
every consumer goes through ``Layout.unpack``/``pack``/``index`` rather than
computing offsets.

A coordinate's ``varies_by`` decides its groups:

=================== ============================================ ===============
``varies_by``       groups                                       copies
=================== ============================================ ===============
``None``            one, labeled ``"shared"``                    1
``"site"``          the sites themselves                         ``S``
a labeling name     the sorted distinct labels of that labeling  its label count
=================== ============================================ ===============

**The override table.** :meth:`Parameterization.to_pysipnet_parameters`
returns an ``xarray.Dataset`` on ``(member, site)`` (or ``(site,)`` for one
member) with one ``float64`` variable per SIPNET parameter the
parameterization sets, calibrated and fixed alike, keyed on the flat pySIPNET
name. Shared and per-label values are broadcast and gathered onto the site
axis, so a caller reads one run's overrides without knowing about groups.
Each variable carries ``units``, ``sipnet_name`` and ``source``
(``"coordinate <name>"`` or ``"fixed"``); the ``site`` coordinate carries
every labeling as a non-dimension coordinate.

**The coordinates table.** :meth:`Parameterization.coordinates_table` returns
``dict[str, xarray.DataArray]``, one per coordinate, with dims drawn from
``member`` (when ``theta`` is an ensemble), the coordinate's ``varies_by``
name (when it varies), and ``element`` (when it has several). It is a dict
rather than a Dataset because ``element`` differs in length and labels from
one coordinate to the next.

Functions and classes
---------------------
Prior helpers, each returning a TFP distribution on the natural scale whose
``.distribution`` is Gaussian and whose ``.bijector`` is ``T_c``:
:func:`log_normal`, :func:`log_normal_from_interval`,
:func:`log_normal_from_samples`, :func:`logit_normal`,
:func:`logit_normal_from_interval`, :func:`logit_normal_from_samples`,
:func:`softmax_normal`, :func:`product_transformed_gaussian_prior`.

:class:`CoordToParamMap`, with :class:`Identity`, :class:`SimplexMap` and
:class:`PhotosynthesisMap`, and the two instances :data:`ALLOCATION` and
:data:`PHOTOSYNTHESIS`.

:class:`Coordinate`, :class:`FixedParameter`, :class:`Layout`,
:class:`Parameterization`, :func:`pysipnet_overrides`, and
:func:`example_parameterization`, the worked example and test fixture.

Notes
-----
Why one prior per coordinate, stored on the natural scale. TFP's
``LogNormal``, ``LogitNormal`` and every ``TransformedDistribution`` expose
``.distribution`` (the unconstrained base) and ``.bijector``, so the density
of ``theta_c``, its samples and its moments are all read off the one object.
What TFP does not do is push moments through a non-affine bijector
(``TransformedDistribution(...).mean()`` raises ``NotImplementedError``),
which is why the base is what :meth:`Parameterization.to_eki_gaussian_prior`
consults, and why :func:`product_transformed_gaussian_prior` insists on Normal
bases.

Why the layout is coordinate-major. Each coordinate's copies form one
contiguous block, which is the block a covariance operator wants:
``to_eki_gaussian_prior`` builds one ``PSDLinOp`` per coordinate. A spatial
GP over the per-site copies of one coordinate replaces that one block and
leaves everything else alone.

Why the override table holds arrays, not ``SIPNETParameters``. Building
``J x S`` validated Pydantic models per ensemble would dominate a run that is
otherwise a subprocess. Validation happens at the run, where
``SIPNETModel`` raises before invoking the binary; the import-time checks at
the bottom of this module are what make that failure unreachable from a
prior draw.

Importing this module sets ``jax_enable_x64``, as ``import pyeki`` does, so
an MCMC baseline that never imports pyEKI still computes in float64. The
setting is per process: workers of a process pool need ``JAX_ENABLE_X64=1``
in their environment.

Usage
-----
Build the example, draw from its prior, and run one member at one site::

    import jax
    from pysipnet import SIPNETModel
    from sipnet_calibration.parameterization import (
        example_parameterization, pysipnet_overrides,
    )

    p = example_parameterization(sites=(1, 27), pft=("deciduous", "conifer"))
    theta = p.sample(jax.random.key(0), n=50)          # (50, D)
    table = p.to_pysipnet_parameters(theta)             # Dataset (member, site)
    kwargs = pysipnet_overrides(table, member=3, site=27)
    result = model(**kwargs)                            # model: a SIPNETModel

Read the vector, and hand it to pyEKI or an MCMC sampler::

    p.describe()                                        # one row per column
    p.coordinates_table(theta, scale="natural")["allocation"].sel(pft="conifer")
    gaussian = p.to_eki_gaussian_prior()                # pyeki.gauss.Gaussian
    log_density = jax.jit(p.log_prior)                  # theta -> (J,)
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from pyeki.gauss import Gaussian  # noqa: E402
from pyeki.linalg import DensePSD, PSDBlockDiag, PSDDiagonal, PSDLinOp  # noqa: E402
from pysipnet.parameters.base import ParameterDomain, ParameterSpec  # noqa: E402
from pysipnet.parameters.model import PARAMETER_SPECS  # noqa: E402
from tensorflow_probability.substrates import jax as tfp  # noqa: E402

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
    >>> float(prior.quantile(0.5))
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
    carries *mass*. This is the constructor for a 2.5 / 97.5% table.

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
    caller counts what it excludes (a negative trait draw is not "handled",
    it is reported).
    """
    logs = jnp.log(_samples_in_support("log_normal_from_samples", x, lambda v: v > 0))
    return tfd.LogNormal(loc=jnp.mean(logs), scale=_positive_std(logs, "log_normal_from_samples"))


def logit_normal(*, median: Any, logit_sd: Any) -> tfd.LogitNormal:
    """Logit-normal on ``(0, 1)`` with the given median and logit-scale sd.

    ``logit x ~ Normal(logit median, logit_sd)``. There is no natural-scale
    spread statistic for this family that is exact and checkable, so the
    spread is stated on the logit scale; ``logit_sd`` around 1.7 is close to
    flat on ``(0, 1)``.
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
        Scale of the base, scalar or per element.

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
    a ``Blockwise`` of the components' bijectors. That is what gives the
    coordinate one unconstrained Gaussian density with analytic moments. A
    component with any other base, a vector event, or a batch shape is
    refused.

    Keyword order is element order; it must match the ``components`` of the
    :class:`CoordToParamMap` the coordinate uses.

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


# ── coordinate-to-parameter maps ──────────────────────────────────────────────


@runtime_checkable
class CoordToParamMap(Protocol):
    """``M_c``: a coordinate's natural-scale value to SIPNET parameter values.

    Attributes
    ----------
    writes:
        SIPNET parameters this map sets.
    reads:
        Fixed SIPNET parameters it needs; ``()`` for most maps.
    components:
        Names of the ``k`` elements of the natural-scale value, in order.
    """

    writes: tuple[str, ...]
    reads: tuple[str, ...]
    components: tuple[str, ...]

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        """*natural* has shape ``(..., k)``; each returned array has shape
        ``(...)``. Values in *fixed* broadcast against ``(...)``."""


@dataclass(frozen=True)
class Identity:
    """One scalar coordinate to one SIPNET parameter. The default map."""

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

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        return {self.parameter: natural[..., 0]}


@dataclass(frozen=True)
class SimplexMap:
    """A point on the ``k``-simplex to ``k - 1`` SIPNET parameters.

    The first ``k - 1`` components are written; the last is the residual
    SIPNET computes itself and has no parameter for. Keeping the residual in
    the coordinate is what makes every component strictly positive, so the
    residual SIPNET recomputes is positive for every ``theta`` (see
    :data:`ALLOCATION`).

    Parameters
    ----------
    writes:
        SIPNET parameters for the first ``k - 1`` components, in order.
    residual:
        Name of the last component, for labels only.
    """

    writes: tuple[str, ...]
    residual: str

    @property
    def reads(self) -> tuple[str, ...]:
        return ()

    @property
    def components(self) -> tuple[str, ...]:
        return (*self.writes, self.residual)

    def __call__(self, natural: Array, fixed: Mapping[str, Array]) -> dict[str, Array]:
        return {name: natural[..., i] for i, name in enumerate(self.writes)}


@dataclass(frozen=True)
class PhotosynthesisMap:
    """The identifiable photosynthesis pair, replacing four parameters by two.

    ``aMax``, ``aMaxFrac``, ``baseFolRespFrac`` and ``cFracLeaf`` enter SIPNET
    only through two quantities (``sipnet.c:614, 617, 633``; "SIPNET
    Parameters" section 3.1 in the vault), leaving a two-dimensional exactly
    flat direction. With ``aMaxFrac`` and ``cFracLeaf`` fixed, the coordinate
    is

    .. math::

        P = \\frac{\\texttt{aMax}\\,(\\texttt{aMaxFrac} + \\texttt{baseFolRespFrac})}
                  {\\texttt{cFracLeaf}},
        \\qquad
        \\rho = \\frac{\\texttt{baseFolRespFrac}}
                      {\\texttt{aMaxFrac} + \\texttt{baseFolRespFrac}},

    and this map inverts it. With ``S = aMaxFrac + baseFolRespFrac``,
    ``rho = baseFolRespFrac / S`` gives ``S = aMaxFrac / (1 - rho)``, so

    .. math::

        \\texttt{baseFolRespFrac} = \\frac{\\rho\\,\\texttt{aMaxFrac}}{1 - \\rho},
        \\qquad
        \\texttt{aMax} = \\frac{P\\,\\texttt{cFracLeaf}\\,(1 - \\rho)}{\\texttt{aMaxFrac}}.

    ``components = ("capacity", "respiration_share")``, in that order.

    Notes
    -----
    Both outputs are positive for every ``P > 0``, ``rho in (0, 1)`` and
    ``aMaxFrac in (0, 1)``, so a log-normal on ``P`` and a logit-normal on
    ``rho`` can never produce a value pySIPNET refuses. ``P`` reads as canopy
    assimilation capacity per unit leaf carbon and ``rho`` as the share of it
    spent on basal foliar respiration. The vault's parameters note writes the
    pair as ``(P, phi)``; ``phi`` is reserved for hyperparameters in this
    project, hence ``rho``.
    """

    writes: tuple[str, ...] = ("max_photosynthesis_rate", "foliar_respiration_fraction")
    reads: tuple[str, ...] = ("daily_mean_photosynthesis_fraction", "leaf_carbon_fraction")
    components: tuple[str, ...] = ("capacity", "respiration_share")

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
"""The allocation simplex.

The natural value is the 4-vector (leaf, wood, fine root, coarse root). The
first three are written; coarse root is not, because SIPNET has no parameter
for it and recomputes exactly ``1 - (leaf + wood + fine root)`` at
``sipnet.c:1113-1115``, calling ``exit()`` if that is not positive
(``sipnet.c:1117-1122``; pySIPNET's ``SIPNETParameters`` validator enforces
the same ``sum < 1`` first). With the coordinate on the simplex, every
component is strictly positive and that exit is unreachable, which three
independent fractions cannot promise. For EKI a dead member is a missing
column, not a low-likelihood one.
"""

PHOTOSYNTHESIS = PhotosynthesisMap()
"""``(P, rho)`` to ``aMax`` and ``baseFolRespFrac``, reading the fixed
``aMaxFrac`` and ``cFracLeaf``. See :class:`PhotosynthesisMap`."""


# ── the specs ─────────────────────────────────────────────────────────────────

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
"""Coordinate names: ``lower_case_with_underscores``."""

SHARED = "shared"
"""Group label and dimension name of a coordinate that does not vary."""

SITE = "site"
"""The reserved ``varies_by`` value meaning one copy per site."""

RESERVED_LABELING_NAMES = frozenset({SHARED, SITE, "member", "element"})
"""Names a labeling cannot take, because they are dimension names already."""


@dataclass(frozen=True, eq=False, kw_only=True)
class Coordinate:
    """One named piece of ``theta``: a prior on its natural scale, and how it
    reaches SIPNET.

    Parameters
    ----------
    name:
        ``lower_case_with_underscores``, unique in a registry. Names the
        coordinate, not a SIPNET parameter: ``photosynthesis``, not
        ``max_photosynthesis_rate``.
    prior:
        A ``tfd.TransformedDistribution`` on the natural scale, normally from
        a prior helper. Its ``.bijector`` is ``T_c`` and its ``.distribution``
        the density of ``theta_c``. Batch shape ``()`` for one prior shared by
        every group, or ``(n_groups,)`` for one per group. Any other TFP
        distribution is accepted with its default event-space bijector; it
        then has no analytic moments (see
        :meth:`Parameterization.to_eki_gaussian_prior`).
    coord_to_param:
        A :class:`CoordToParamMap`, or a SIPNET parameter name as shorthand
        for :class:`Identity`.
    varies_by:
        ``None`` (shared), ``"site"``, or the name of a labeling the
        :class:`Parameterization` supplies.
    provenance:
        Where the prior came from, in words, with the citation. A placeholder
        says it is one.

    Raises
    ------
    ValueError
        For a malformed name, an empty provenance, a map whose SIPNET
        parameters do not exist, a prior batch shape of rank above 1, or a
        map whose ``components`` do not match the prior's event size.

    Examples
    --------
    >>> soil = Coordinate(
    ...     name="base_soil_respiration",
    ...     prior=log_normal_from_interval(lower=0.004, upper=0.020),
    ...     coord_to_param="base_soil_respiration_rate",
    ...     varies_by="pft",
    ...     provenance="BETY som_respiration_rate 2.5-97.5% (readiness report 5.4).",
    ... )
    >>> soil.size, soil.element_labels
    (1, ('log(base_soil_respiration_rate)',))
    """

    name: str
    prior: tfd.Distribution
    coord_to_param: CoordToParamMap | str
    varies_by: str | None = None
    provenance: str

    def __post_init__(self) -> None:
        if isinstance(self.coord_to_param, str):
            object.__setattr__(self, "coord_to_param", Identity(self.coord_to_param))
        check_coordinate_name(self.name)
        check_provenance_is_given(self.name, self.provenance)
        check_sipnet_parameters_exist(
            (*self.coord_to_param.writes, *self.coord_to_param.reads), f"coordinate {self.name}"
        )
        check_prior_batch_rank(self)
        check_components_match_prior(self)

    @property
    def bijector(self) -> tfb.Bijector:
        """``T_c``, unconstrained to natural scale."""
        if isinstance(self.prior, tfd.TransformedDistribution):
            return self.prior.bijector
        return self.prior.experimental_default_event_space_bijector()

    @property
    def unconstrained_prior(self) -> tfd.Distribution:
        """The density of ``theta_c``."""
        if isinstance(self.prior, tfd.TransformedDistribution):
            return self.prior.distribution
        return tfd.TransformedDistribution(self.prior, tfb.Invert(self.bijector))

    @property
    def is_scalar(self) -> bool:
        """Whether the natural value is a scalar (event shape ``()``)."""
        return tuple(self.prior.event_shape) == ()

    @property
    def natural_size(self) -> int:
        """``k``, the number of natural-scale components."""
        return 1 if self.is_scalar else int(self.prior.event_shape[0])

    @property
    def size(self) -> int:
        """The number of columns of ``theta`` per group."""
        if self.is_scalar:
            return 1
        shape = self.bijector.inverse_event_shape(self.prior.event_shape)
        return int(shape[0])

    @property
    def components(self) -> tuple[str, ...]:
        """Natural-scale element names, from the map."""
        return tuple(self.coord_to_param.components)

    @property
    def element_labels(self) -> tuple[str, ...]:
        """Unconstrained element names: ``log(x)``, ``logit(x)``,
        ``alr(a/residual)``, derived from the bijector."""
        return _unconstrained_labels(self.bijector, self.components)

    @property
    def distribution_name(self) -> str:
        """A short name for :meth:`Parameterization.describe`."""
        return _distribution_name(self.prior)

    @property
    def has_analytic_moments(self) -> bool:
        """Whether ``unconstrained_prior`` answers ``mean()`` and
        ``variance()``/``covariance()`` without sampling."""
        try:
            self.unconstrained_prior.mean()
            if self.is_scalar:
                self.unconstrained_prior.variance()
            else:
                self.unconstrained_prior.covariance()
        except NotImplementedError:
            return False
        return True


@dataclass(frozen=True, kw_only=True)
class FixedParameter:
    """A SIPNET parameter held at a value.

    Parameters
    ----------
    name:
        SIPNET parameter name, checked against ``PARAMETER_SPECS``.
    value:
        A float when *varies_by* is ``None``; otherwise a mapping from every
        group label to a float. Each value is checked against the parameter's
        ``ParameterDomain``, so a fixed value can never be one pySIPNET
        refuses.
    varies_by:
        As for :class:`Coordinate`.
    provenance:
        Where the value came from. A ``template.param`` value says so and
        says why nothing better covers it.

    Examples
    --------
    >>> FixedParameter(
    ...     name="vapor_pressure_deficit_exponent", value=2.0,
    ...     provenance="Braswell et al.; readiness report 5.2.",
    ... ).value
    2.0
    """

    name: str
    value: float | Mapping[Any, float]
    varies_by: str | None = None
    provenance: str

    def __post_init__(self) -> None:
        check_sipnet_parameters_exist((self.name,), f"fixed parameter {self.name}")
        check_provenance_is_given(self.name, self.provenance)
        check_fixed_value_shape(self)
        for value in self.values():
            check_fixed_value_is_in_domain(self.name, value)

    def values(self) -> tuple[float, ...]:
        """Every value, in mapping order (one value when shared)."""
        if isinstance(self.value, Mapping):
            return tuple(float(v) for v in self.value.values())
        return (float(self.value),)


# ── the layout ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Layout:
    """Where each coordinate lives in ``theta``, by name.

    Coordinates in declaration order; within a coordinate, groups in group
    order; within a group, elements in order. Built by
    :class:`Parameterization`; consumers call :meth:`unpack`, :meth:`pack`
    and :meth:`index` rather than computing offsets.

    Attributes
    ----------
    coordinates:
        Coordinate names in layout order.
    sizes:
        Columns per group, per coordinate.
    groups:
        Group labels per coordinate, in the order they occupy ``theta``.
    dims:
        The group dimension name per coordinate: ``"shared"``, ``"site"``, or
        a labeling name.
    element_labels:
        Unconstrained element labels per coordinate.

    Examples
    --------
    >>> layout = p.layout                                   # p: a Parameterization
    >>> layout.dimension == theta.shape[-1]
    True
    >>> parts = layout.unpack(theta)                        # {name: (..., n_groups, size)}
    >>> bool(jnp.allclose(layout.pack(parts), theta))
    True
    >>> layout.index("allocation", group="conifer")         # three columns
    """

    coordinates: tuple[str, ...]
    sizes: Mapping[str, int]
    groups: Mapping[str, tuple[Any, ...]]
    dims: Mapping[str, str]
    element_labels: Mapping[str, tuple[str, ...]]

    @cached_property
    def slices(self) -> dict[str, slice]:
        """The contiguous slice of ``theta`` each coordinate owns."""
        out, start = {}, 0
        for name in self.coordinates:
            width = len(self.groups[name]) * self.sizes[name]
            out[name] = slice(start, start + width)
            start += width
        return out

    @property
    def dimension(self) -> int:
        """``D``."""
        return sum(len(self.groups[n]) * self.sizes[n] for n in self.coordinates)

    @cached_property
    def labels(self) -> tuple[str, ...]:
        """``D`` strings: ``"c"``, ``"c[group]"``, ``"c[element]"`` or
        ``"c[group][element]"`` as the coordinate needs."""
        out = []
        for name in self.coordinates:
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

    def slice(self, coordinate: str) -> slice:
        """The columns of ``theta`` a coordinate owns."""
        return self.slices[self._known(coordinate)]

    def index(self, coordinate: str, group: Any = None, element: str | None = None) -> np.ndarray:
        """Column indices of a coordinate, narrowed by group and element label."""
        name = self._known(coordinate)
        n_groups, size = len(self.groups[name]), self.sizes[name]
        idx = np.arange(self.slices[name].start, self.slices[name].stop).reshape(n_groups, size)
        if group is not None:
            idx = idx[[_position(self.groups[name], group, f"group of {name}")]]
        if element is not None:
            idx = idx[:, [_position(self.element_labels[name], element, f"element of {name}")]]
        return idx.ravel()

    def unpack(self, theta: Array) -> dict[str, Array]:
        """``(..., D)`` to ``{coordinate: (..., n_groups, size)}``."""
        theta = _as_theta(theta, self.dimension)
        lead = theta.shape[:-1]
        return {
            name: theta[..., self.slices[name]].reshape(
                (*lead, len(self.groups[name]), self.sizes[name])
            )
            for name in self.coordinates
        }

    def pack(self, parts: Mapping[str, Array]) -> Array:
        """Inverse of :meth:`unpack`; refuses a missing or extra coordinate."""
        if set(parts) != set(self.coordinates):
            raise ValueError(
                f"Layout.pack: expected exactly the coordinates {list(self.coordinates)}, "
                f"got {sorted(parts)}."
            )
        pieces = []
        for name in self.coordinates:
            part = jnp.asarray(parts[name], dtype=jnp.float64)
            expected = (len(self.groups[name]), self.sizes[name])
            if part.shape[-2:] != expected:
                raise ValueError(
                    f"Layout.pack: {name} must end in (n_groups, size) = {expected}, "
                    f"got shape {part.shape}."
                )
            pieces.append(part.reshape((*part.shape[:-2], expected[0] * expected[1])))
        return jnp.concatenate(pieces, axis=-1)

    def _known(self, coordinate: str) -> str:
        if coordinate not in self.slices:
            raise KeyError(f"Layout: no coordinate {coordinate!r}; have {list(self.coordinates)}.")
        return coordinate


# ── the parameterization ──────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class Parameterization:
    """A calibration vector: coordinates, fixed parameters, sites, labelings.

    The one object an experiment's ``config.py`` names. Everything about the
    vector's order is behind :attr:`layout`; everything about its prior is in
    the coordinates.

    Parameters
    ----------
    coordinates:
        In the order they occupy ``theta``.
    fixed:
        The parameters held at a value. Every name in any map's ``reads``
        must appear here.
    sites:
        Site ids, ascending; the project's 1-8000 ids, never renumbered.
    labelings:
        ``{name: one label per site}`` for every ``varies_by`` used other
        than ``None`` and ``"site"``.

    Attributes
    ----------
    dimension : int
        ``D``.
    layout : Layout

    Raises
    ------
    ValueError
        If a coordinate name repeats, two writers set one SIPNET parameter, a
        map reads a parameter nobody fixed, a labeling is missing or the wrong
        length, a prior's batch shape disagrees with its group count, a fixed
        value misses a group, or a coordinate's transform can leave a SIPNET
        parameter's domain.

    Examples
    --------
    >>> p = example_parameterization(sites=(1, 27), pft=("deciduous", "conifer"))
    >>> p.dimension
    13
    >>> theta = p.sample(jax.random.key(0), n=4)
    >>> table = p.to_pysipnet_parameters(theta)            # (member, site)
    >>> sorted(pysipnet_overrides(table, member=0, site=27))[:2]
    ['base_soil_respiration_rate', 'daily_mean_photosynthesis_fraction']
    """

    coordinates: tuple[Coordinate, ...]
    fixed: tuple[FixedParameter, ...] = ()
    sites: tuple[int, ...]
    labelings: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "fixed", tuple(self.fixed))
        object.__setattr__(self, "sites", tuple(int(s) for s in self.sites))
        object.__setattr__(
            self, "labelings", {k: tuple(v) for k, v in dict(self.labelings).items()}
        )
        check_sites_are_ascending(self.sites)
        check_coordinate_names_are_unique(self.coordinates)
        check_labelings_cover_sites(self)
        check_each_sipnet_parameter_has_one_writer(self.coordinates, self.fixed)
        check_reads_are_fixed(self.coordinates, self.fixed)
        for coordinate in self.coordinates:
            check_prior_batch_matches_groups(coordinate, self.n_groups(coordinate.varies_by))
        for parameter in self.fixed:
            check_fixed_value_covers_groups(parameter, self.group_labels(parameter.varies_by))
        for coordinate in self.coordinates:
            check_map_image_is_in_domain(self, coordinate)

    # -- structure -----------------------------------------------------------

    @property
    def dimension(self) -> int:
        """``D``."""
        return self.layout.dimension

    @cached_property
    def layout(self) -> Layout:
        return Layout(
            coordinates=tuple(c.name for c in self.coordinates),
            sizes={c.name: c.size for c in self.coordinates},
            groups={c.name: self.group_labels(c.varies_by) for c in self.coordinates},
            dims={c.name: c.varies_by or SHARED for c in self.coordinates},
            element_labels={c.name: c.element_labels for c in self.coordinates},
        )

    def group_labels(self, varies_by: str | None) -> tuple[Any, ...]:
        """The groups of a ``varies_by`` value, in the order they occupy
        ``theta``: ``("shared",)``, the sites, or a labeling's sorted labels."""
        if varies_by is None:
            return (SHARED,)
        if varies_by == SITE:
            return self.sites
        return tuple(sorted(set(self.labelings[varies_by])))

    def n_groups(self, varies_by: str | None) -> int:
        return len(self.group_labels(varies_by))

    def sites_with(self, labeling: str, label: Any) -> tuple[int, ...]:
        """The sites carrying *label* under *labeling*."""
        if labeling not in self.labelings:
            raise KeyError(f"no labeling {labeling!r}; have {sorted(self.labelings)}.")
        return tuple(s for s, lab in zip(self.sites, self.labelings[labeling], strict=True) if lab == label)

    def coordinate(self, name: str) -> Coordinate:
        """The coordinate called *name*."""
        for coordinate in self.coordinates:
            if coordinate.name == name:
                return coordinate
        raise KeyError(f"no coordinate {name!r}; have {list(self.layout.coordinates)}.")

    # -- core, on jax arrays, jit-able ---------------------------------------

    def constrain(self, theta: Array) -> dict[str, Array]:
        """``theta`` ``(..., D)`` to ``{coordinate: natural value}``, each of
        shape ``(..., n_groups, k)``."""
        parts = self.layout.unpack(theta)
        return {c.name: c.bijector.forward(parts[c.name]) for c in self.coordinates}

    def unconstrain(self, natural: Mapping[str, Array]) -> Array:
        """Inverse of :meth:`constrain`: ``{coordinate: (..., n_groups, k)}``
        to ``theta`` ``(..., D)``."""
        parts = {}
        for coordinate in self.coordinates:
            value = jnp.asarray(natural[coordinate.name], dtype=jnp.float64)
            parts[coordinate.name] = coordinate.bijector.inverse(value)
        return self.layout.pack(parts)

    def log_prior(self, theta: Array) -> Array:
        """Log prior density of ``theta``: ``(J,)`` for ``(J, D)``, a scalar
        for ``(D,)``.

        Notes
        -----
        The Jacobian of ``T_c`` is included by construction: the summand is
        ``prior.distribution.log_prob(theta_c)``, which equals
        ``prior.log_prob(T_c(theta_c)) + T_c.forward_log_det_jacobian(theta_c)``.
        The finite-difference test asserts that identity. ``jit``-able and
        differentiable; the entry point an MCMC baseline uses.
        """
        parts = self.layout.unpack(theta)
        total = jnp.zeros(parts[self.coordinates[0].name].shape[:-2], dtype=jnp.float64)
        for coordinate in self.coordinates:
            part = parts[coordinate.name]
            if coordinate.is_scalar:
                part = part[..., 0]
            total = total + self._broadcast_unconstrained(coordinate).log_prob(part).sum(axis=-1)
        return total

    def sample(self, key: Array, n: int) -> Array:
        """``n`` prior draws, ``(n, D)``.

        One key split per coordinate, so appending a coordinate to a registry
        leaves the earlier coordinates' draws unchanged.
        """
        parts = {}
        for coordinate, subkey in zip(
            self.coordinates, jax.random.split(key, len(self.coordinates)), strict=True
        ):
            draw = self._broadcast_unconstrained(coordinate).sample((n,), seed=subkey)
            n_groups = self.n_groups(coordinate.varies_by)
            parts[coordinate.name] = jnp.reshape(draw, (n, n_groups, coordinate.size))
        return self.layout.pack(parts)

    def to_eki_gaussian_prior(
        self, *, key: Array | None = None, n_moment_samples: int = 0
    ) -> Gaussian:
        """The prior as a ``pyeki.gauss.Gaussian`` over ``theta``.

        Parameters
        ----------
        key, n_moment_samples:
            Used only for a coordinate without analytic unconstrained moments
            (a hand-built prior that is not a ``TransformedDistribution``):
            that coordinate is moment matched from *n_moment_samples* draws.
            Left at their defaults, such a coordinate raises instead.

        Returns
        -------
        pyeki.gauss.Gaussian
            Mean of length ``D``. Covariance a ``PSDBlockDiag`` with one block
            per coordinate in layout order: ``PSDDiagonal`` over groups for a
            scalar coordinate, a ``PSDBlockDiag`` of one ``DensePSD`` per group
            otherwise.

        Raises
        ------
        NotImplementedError
            If a coordinate lacks analytic moments and no *key* was given.

        Notes
        -----
        EKI consults the prior once, through ``EKIState.from_prior``, so this
        object only has to be sampled. One block per coordinate is the seam
        for a spatial or hierarchical prior: replace the block of one
        ``varies_by="site"`` coordinate with a Matern operator and nothing
        else moves. For every coordinate the prior helpers build the match is
        exact, because the prior is Gaussian in ``theta`` by construction.
        """
        means, blocks = [], []
        for coordinate in self.coordinates:
            mean, block = self._moment_block(coordinate, key, n_moment_samples)
            means.append(mean)
            blocks.append(block)
        return Gaussian(mean=jnp.concatenate(means), cov=PSDBlockDiag(tuple(blocks)))

    # -- labeled views, for people and run loops -----------------------------

    def coordinates_table(
        self, theta: Array, *, scale: str = "unconstrained"
    ) -> dict[str, xr.DataArray]:
        """``theta`` as labeled arrays, one per coordinate.

        Parameters
        ----------
        theta:
            ``(J, D)`` or ``(D,)``.
        scale:
            ``"unconstrained"`` for ``theta_c`` itself, ``"natural"`` for
            ``T_c(theta_c)``.

        Returns
        -------
        dict[str, xarray.DataArray]
            Dims per array, each present only when needed: ``member`` for an
            ensemble; the coordinate's ``varies_by`` name (``site``, ``pft``,
            ...) with the group labels as its coordinate; ``element`` with the
            element labels — unconstrained (``log(...)``, ``alr(...)``) or
            natural (``leaf``, ``wood``, ...) as *scale* says.

        Examples
        --------
        >>> t = p.coordinates_table(theta, scale="natural")
        >>> t["allocation"].dims
        ('member', 'pft', 'element')
        >>> t["allocation"].sel(pft="conifer", element="coarse_root")   # the residual
        """
        if scale not in ("unconstrained", "natural"):
            raise ValueError(f"scale must be 'unconstrained' or 'natural', got {scale!r}.")
        theta = _as_theta(theta, self.dimension)
        parts = self.layout.unpack(theta) if scale == "unconstrained" else self.constrain(theta)
        out = {}
        for coordinate in self.coordinates:
            values = np.asarray(parts[coordinate.name])
            dim = self.layout.dims[coordinate.name]
            elements = (
                coordinate.element_labels if scale == "unconstrained" else coordinate.components
            )
            dims = (*(("member",) if theta.ndim == 2 else ()), dim, "element")
            coords: dict[str, Any] = {"element": list(elements)}
            if dim != SHARED:
                coords[dim] = list(self.group_labels(coordinate.varies_by))
            if dim == SITE:
                coords.update({k: (SITE, list(v)) for k, v in self.labelings.items()})
            array = xr.DataArray(values, dims=dims, coords=coords, name=coordinate.name)
            if dim == SHARED:
                array = array.isel({SHARED: 0}, drop=True)
            if len(elements) == 1:
                array = array.isel(element=0, drop=True)
            out[coordinate.name] = array
        return out

    def to_pysipnet_parameters(self, theta: Array) -> xr.Dataset:
        """The override table: every SIPNET parameter this parameterization
        sets, on ``(member, site)`` or ``(site,)``.

        Parameters
        ----------
        theta:
            ``(J, D)`` or ``(D,)``.

        Returns
        -------
        xarray.Dataset
            One ``float64`` variable per SIPNET parameter, calibrated and
            fixed alike, keyed on the flat pySIPNET name. Shared and per-label
            values are broadcast and gathered onto the site axis. Each
            variable carries ``units``, ``sipnet_name`` and ``source``; the
            ``site`` coordinate carries every labeling.

        Examples
        --------
        >>> table = p.to_pysipnet_parameters(theta)
        >>> table["leaf_allocation"].sel(site=27)          # (member,)
        >>> table.sel(site=list(p.sites_with("pft", "deciduous")))
        """
        theta = _as_theta(theta, self.dimension)
        natural = self.constrain(theta)
        lead = theta.shape[:-1]
        columns: dict[str, tuple[Array, str]] = {}
        for coordinate in self.coordinates:
            on_sites = self._onto_sites(coordinate, natural[coordinate.name])
            for name, values in coordinate.coord_to_param(on_sites, self._fixed_table).items():
                columns[name] = (values, f"coordinate {coordinate.name}")
        for parameter in self.fixed:
            values = jnp.broadcast_to(self._fixed_table[parameter.name], (*lead, len(self.sites)))
            columns[parameter.name] = (values, "fixed")

        dims = (*(("member",) if theta.ndim == 2 else ()), SITE)
        coords: dict[str, Any] = {SITE: list(self.sites)}
        coords.update({k: (SITE, list(v)) for k, v in self.labelings.items()})
        variables = {
            name: (dims, np.asarray(values, dtype=np.float64), _parameter_attrs(name, source))
            for name, (values, source) in sorted(columns.items(), key=lambda kv: _spec_order(kv[0]))
        }
        return xr.Dataset(variables, coords=coords)

    def for_site(self, table: Mapping[str, xr.DataArray], site: int) -> dict[str, xr.DataArray]:
        """Select one site from a :meth:`coordinates_table`.

        A site reads the shared copy of a shared coordinate, its own copy of
        a per-site one, and its label's copy of a per-label one, so this is a
        resolution through the site's labels, not a ``.sel``. On an override
        table use ``table.sel(site=site)`` directly.
        """
        if site not in self.sites:
            raise KeyError(f"site {site} is not one of this parameterization's sites.")
        position = self.sites.index(site)
        out = {}
        for name, array in table.items():
            dim = self.layout.dims[name]
            if dim == SHARED:
                out[name] = array
            elif dim == SITE:
                out[name] = array.sel({SITE: site})
            else:
                out[name] = array.sel({dim: self.labelings[dim][position]})
        return out

    def describe(self) -> pd.DataFrame:
        """One row per column of ``theta``.

        Columns: ``coordinate``, ``group``, ``element``,
        ``sipnet_parameters``, ``distribution``, ``theta_mean``,
        ``theta_sd``, ``natural_median``, ``natural_2.5``, ``natural_97.5``,
        ``theta_moments`` (``"analytic"`` or ``"monte_carlo"``) and
        ``provenance``. Natural quantiles are blank for a multivariate natural
        scale, where a marginal quantile would misrepresent a simplex.
        """
        rows = []
        for coordinate in self.coordinates:
            groups = self.group_labels(coordinate.varies_by)
            moments = self._describe_moments(coordinate)
            quantiles = self._describe_quantiles(coordinate)
            for g, group in enumerate(groups):
                for e, element in enumerate(coordinate.element_labels):
                    rows.append(
                        {
                            "coordinate": coordinate.name,
                            "group": group,
                            "element": element,
                            "sipnet_parameters": ", ".join(coordinate.coord_to_param.writes),
                            "distribution": coordinate.distribution_name,
                            "theta_mean": moments[0][g, e],
                            "theta_sd": moments[1][g, e],
                            "natural_median": quantiles[0][g],
                            "natural_2.5": quantiles[1][g],
                            "natural_97.5": quantiles[2][g],
                            "theta_moments": (
                                "analytic" if coordinate.has_analytic_moments else "monte_carlo"
                            ),
                            "provenance": coordinate.provenance,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.index.name = "index"
        return frame

    # -- private -------------------------------------------------------------

    @cached_property
    def _fixed_table(self) -> dict[str, Array]:
        """Every fixed parameter as an ``(S,)`` array on the site axis."""
        table = {}
        for parameter in self.fixed:
            labels = self.group_labels(parameter.varies_by)
            if isinstance(parameter.value, Mapping):
                per_group = jnp.asarray([parameter.value[g] for g in labels], dtype=jnp.float64)
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
        labels = self.group_labels(varies_by)
        position = {label: i for i, label in enumerate(labels)}
        return np.asarray([position[label] for label in self.labelings[varies_by]])

    def _onto_sites(self, coordinate: Coordinate, natural: Array) -> Array:
        """``(..., n_groups, k)`` gathered to ``(..., S, k)``."""
        return natural[..., self._site_group_index(coordinate.varies_by), :]

    def _broadcast_unconstrained(self, coordinate: Coordinate) -> tfd.Distribution:
        """The unconstrained prior with batch shape ``(n_groups,)``."""
        prior = coordinate.unconstrained_prior
        n_groups = self.n_groups(coordinate.varies_by)
        if tuple(prior.batch_shape) == (n_groups,):
            return prior
        return tfd.BatchBroadcast(prior, to_shape=(n_groups,))

    def _moment_block(
        self, coordinate: Coordinate, key: Array | None, n_moment_samples: int
    ) -> tuple[Array, PSDLinOp]:
        prior = self._broadcast_unconstrained(coordinate)
        if coordinate.has_analytic_moments:
            mean = prior.mean()
            spread = prior.variance() if coordinate.is_scalar else prior.covariance()
        elif key is not None and n_moment_samples > 1:
            draws = prior.sample((n_moment_samples,), seed=key)
            mean = draws.mean(axis=0)
            if coordinate.is_scalar:
                spread = draws.var(axis=0, ddof=1)
            else:
                centered = draws - mean
                spread = jnp.einsum("ngi,ngj->gij", centered, centered) / (n_moment_samples - 1)
        else:
            raise NotImplementedError(
                f"coordinate {coordinate.name!r}: its prior has no analytic unconstrained "
                "moments. Build it with a prior helper (log_normal, logit_normal, ...) or "
                "pass key= and n_moment_samples= to moment match by Monte Carlo."
            )
        if coordinate.is_scalar:
            return jnp.ravel(mean), PSDDiagonal(jnp.ravel(spread))
        blocks = tuple(DensePSD.from_matrix(spread[g]) for g in range(spread.shape[0]))
        return jnp.ravel(mean), PSDBlockDiag(blocks)

    def _describe_moments(self, coordinate: Coordinate) -> tuple[np.ndarray, np.ndarray]:
        n_groups = self.n_groups(coordinate.varies_by)
        shape = (n_groups, coordinate.size)
        if not coordinate.has_analytic_moments:
            return np.full(shape, np.nan), np.full(shape, np.nan)
        prior = self._broadcast_unconstrained(coordinate)
        return (
            np.asarray(prior.mean()).reshape(shape),
            np.asarray(prior.stddev()).reshape(shape),
        )

    def _describe_quantiles(self, coordinate: Coordinate) -> tuple[np.ndarray, ...]:
        n_groups = self.n_groups(coordinate.varies_by)
        blank = np.full(n_groups, np.nan)
        if not coordinate.is_scalar:
            return blank, blank, blank
        try:
            return tuple(
                np.broadcast_to(np.asarray(coordinate.prior.quantile(q)), (n_groups,))
                for q in (0.5, 0.025, 0.975)
            )
        except NotImplementedError:
            return blank, blank, blank


def pysipnet_overrides(
    table: xr.Dataset, *, site: int, member: int | None = None
) -> dict[str, float]:
    """One run's keyword arguments for ``SIPNETModel`` from an override table.

    Parameters
    ----------
    table:
        From :meth:`Parameterization.to_pysipnet_parameters`.
    site:
        The site id.
    member:
        The ensemble member's position; required when the table has a
        ``member`` dim.

    Examples
    --------
    >>> model(**pysipnet_overrides(table, member=3, site=27))
    """
    selected = table.sel({SITE: site})
    if "member" in selected.dims:
        if member is None:
            raise ValueError("the table has a member dim; pass member=.")
        selected = selected.isel(member=member)
    elif member is not None:
        raise ValueError("the table has no member dim; do not pass member=.")
    return {str(name): float(value) for name, value in selected.data_vars.items()}


# ── the example ───────────────────────────────────────────────────────────────


def example_parameterization(sites: Sequence[int], *, pft: Sequence[str]) -> Parameterization:
    """A small registry exercising every code path. **Not the calibration
    vector, and not a reviewed prior.**

    This module's worked example and test fixture, so that tests, later PRs
    and an experiment can import one reference registry. Every center traces
    to a table in the 2026-09-21 readiness report (sections 5.2 and 5.4) and
    every provenance string says what the value is not. Nothing comes from
    ``template.param``.

    Parameters
    ----------
    sites:
        Site ids, ascending.
    pft:
        One PFT label per site.

    Coordinates
    -----------
    ``photosynthesis``           shared, size 2, :data:`PHOTOSYNTHESIS`
    ``allocation``               by ``"pft"``, size 3, :data:`ALLOCATION`
    ``base_soil_respiration``    by ``"pft"``, log-normal rate
    ``leaf_fall_fraction``       shared, logit-normal fraction
    ``initial_soil_carbon``      by ``"site"``, log-normal, one prior per site

    Fixed: ``daily_mean_photosynthesis_fraction`` (shared),
    ``leaf_carbon_fraction`` (by ``"pft"``), ``vapor_pressure_deficit_exponent``
    (shared).
    """
    sites = tuple(int(s) for s in sites)
    pft = tuple(pft)
    labels = tuple(sorted(set(pft)))
    fixture = "Example fixture, not a reviewed prior. "

    # Temperate-deciduous row of readiness report table 5.4 (BETY medians and
    # 2.5-97.5% quantiles), with aMaxFrac 0.76 and cFracLeaf 0.466 from its
    # table 5.2: P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf and
    # rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac).
    a_max_frac, c_frac_leaf = 0.76, 0.466
    a_max, fol_resp = 58.0, 0.17
    fol_resp_lower, fol_resp_upper = 0.10, 0.39
    photosynthesis = Coordinate(
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
        coord_to_param=PHOTOSYNTHESIS,
        provenance=fixture
        + "Capacity median from the temperate-deciduous BETY medians aMax 58, "
        "baseFolRespFrac 0.17 (readiness report 5.4) with aMaxFrac 0.76 and cFracLeaf "
        "0.466 (5.2); geometric sd 1.75 approximates the aMax 2.5-97.5% ratio 83/28. "
        "Respiration share interval from the baseFolRespFrac 2.5-97.5% 0.10-0.39 at "
        "fixed aMaxFrac. Deciduous values applied to every PFT here.",
    )
    allocation = Coordinate(
        name="allocation",
        prior=softmax_normal(center=(0.18, 0.40, 0.07, 0.35), logit_sd=0.5),
        coord_to_param=ALLOCATION,
        varies_by="pft",
        provenance=fixture
        + "Center is the temperate-deciduous BETY allocation leaf 0.18, wood 0.40, fine "
        "root 0.07 (readiness report 5.4), coarse root the 0.35 remainder; logit sd 0.5 "
        "is a placeholder. One copy per PFT, all with this prior.",
    )
    base_soil_respiration = Coordinate(
        name="base_soil_respiration",
        prior=log_normal_from_interval(lower=0.004, upper=0.020),
        coord_to_param="base_soil_respiration_rate",
        varies_by="pft",
        provenance=fixture
        + "BETY som_respiration_rate 2.5-97.5% 0.004-0.020 yr-1 (readiness report 5.4), "
        "one BETY prior for every PFT; the same interval used for every PFT copy here.",
    )
    leaf_fall_fraction = Coordinate(
        name="leaf_fall_fraction",
        prior=logit_normal(median=0.5, logit_sd=1.7),
        coord_to_param="leaf_off_fall_fraction",
        provenance=fixture
        + "No elicited value: a near-flat logit-normal on (0, 1), median 0.5, logit sd "
        "1.7. A placeholder until PR D fits BETY fracLeafFall.",
    )
    initial_soil_carbon = Coordinate(
        name="initial_soil_carbon",
        prior=log_normal(median=np.full(len(sites), 30_000.0), geometric_sd=2.0),
        coord_to_param="soil_carbon",
        varies_by="site",
        provenance=fixture
        + "One prior per site, identical here: median 30 kg C m-2 (30000 g m-2 in "
        "pySIPNET's units) is the center of the ISCN 12-75 kg C m-2 range at the test "
        "sites (readiness report 5.1); geometric sd 2 spans roughly 7.7-117 at 95%. The "
        "real per-site priors are fitted to the IC ensemble in a later PR.",
    )
    fixed = (
        FixedParameter(
            name="daily_mean_photosynthesis_fraction",
            value=a_max_frac,
            provenance=fixture
            + "0.76, within the BETY per-PFT median range 0.75-0.86 of readiness report "
            "5.2. Fixed because it is one of the two exactly degenerate photosynthesis "
            "directions (SIPNET Parameters 3.1).",
        ),
        FixedParameter(
            name="leaf_carbon_fraction",
            value={label: c_frac_leaf for label in labels},
            varies_by="pft",
            provenance=fixture
            + "BETY leafC for temperate deciduous, 0.466 (readiness report 5.2), applied "
            "to every PFT label here; the real registry gives boreal conifer 0.506 and "
            "grassland 0.483. Fixed for the same reason as aMaxFrac.",
        ),
        FixedParameter(
            name="vapor_pressure_deficit_exponent",
            value=2.0,
            provenance="Braswell et al. fix the exponent at 2; BETY draws of 1.0-2.9 would "
            "add a near-degeneracy with dVpdSlope for nothing (readiness report 5.2).",
        ),
    )
    return Parameterization(
        coordinates=(
            photosynthesis,
            allocation,
            base_soil_respiration,
            leaf_fall_fraction,
            initial_soil_carbon,
        ),
        fixed=fixed,
        sites=sites,
        labelings={"pft": pft},
    )


# ── supporting helpers ────────────────────────────────────────────────────────

_FLAT_SPECS: dict[str, ParameterSpec] = {
    path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()
}
_SPEC_ORDER: dict[str, int] = {name: i for i, name in enumerate(_FLAT_SPECS)}

#: Corners of the unconstrained cube at which :func:`check_map_image_is_in_domain`
#: evaluates a coordinate. +-12 spans nine orders of magnitude on a log scale
#: and reaches 1 - 6e-6 on a logit scale while staying inside float64; the
#: check is about support, not plausibility.
DOMAIN_CHECK_CORNERS = (-12.0, 0.0, 12.0)


def _as_theta(theta: Any, dimension: int) -> Array:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    if theta.ndim not in (1, 2) or theta.shape[-1] != dimension:
        raise ValueError(f"theta must be (D,) or (J, D) with D = {dimension}; got shape {theta.shape}.")
    return theta


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
    z = tfd.Normal(0.0, 1.0).quantile(jnp.float64(0.5 + mass / 2))
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


def _position(labels: Sequence[Any], label: Any, what: str) -> int:
    try:
        return list(labels).index(label)
    except ValueError:
        raise KeyError(f"{label!r} is not a {what}; have {list(labels)}.") from None


def _unconstrained_labels(bijector: tfb.Bijector, components: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(bijector, tfb.SoftmaxCentered):
        return tuple(f"alr({c}/{components[-1]})" for c in components[:-1])
    if isinstance(bijector, tfb.Blockwise):
        return tuple(
            _scalar_unconstrained_label(b, c)
            for b, c in zip(bijector.bijectors, components, strict=True)
        )
    if len(components) == 1:
        return (_scalar_unconstrained_label(bijector, components[0]),)
    return tuple(f"{type(bijector).__name__}^-1({c})" for c in components)


def _scalar_unconstrained_label(bijector: tfb.Bijector, component: str) -> str:
    if isinstance(bijector, tfb.Exp):
        return f"log({component})"
    if isinstance(bijector, tfb.Sigmoid):
        return f"logit({component})"
    if isinstance(bijector, tfb.Identity):
        return component
    return f"{type(bijector).__name__}^-1({component})"


def _distribution_name(prior: tfd.Distribution) -> str:
    if isinstance(prior, tfd.LogNormal):
        return "log-normal"
    if isinstance(prior, tfd.LogitNormal):
        return "logit-normal"
    if isinstance(prior, tfd.TransformedDistribution):
        if isinstance(prior.bijector, tfb.SoftmaxCentered):
            return "softmax-normal"
        if isinstance(prior.bijector, tfb.Blockwise):
            return "product of transformed Gaussians"
        return f"{type(prior.distribution).__name__} through {type(prior.bijector).__name__}"
    return type(prior).__name__


def _parameter_attrs(name: str, source: str) -> dict[str, str]:
    spec = _FLAT_SPECS[name]
    attrs = {"units": spec.units, "sipnet_name": spec.sipnet_name, "source": source}
    if spec.constituent:
        attrs["constituent"] = spec.constituent
    return attrs


def _spec_order(name: str) -> int:
    return _SPEC_ORDER[name]


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


# ── checks ────────────────────────────────────────────────────────────────────


def check_coordinate_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"coordinate name {name!r} is not lower_case_with_underscores; see NAME_PATTERN."
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
            f"{context}: {unknown} are not pySIPNET parameter names. Use the flat field "
            "names of pysipnet.parameters.model.PARAMETER_SPECS, e.g. "
            "'max_photosynthesis_rate', not 'aMax'."
        )


def check_prior_batch_rank(coordinate: Coordinate) -> None:
    rank = len(tuple(coordinate.prior.batch_shape))
    if rank > 1:
        raise ValueError(
            f"coordinate {coordinate.name!r}: prior batch shape {coordinate.prior.batch_shape} "
            "has rank above 1; it must be () or (n_groups,)."
        )


def check_components_match_prior(coordinate: Coordinate) -> None:
    n = len(coordinate.components)
    if n != coordinate.natural_size:
        raise ValueError(
            f"coordinate {coordinate.name!r}: the map names {n} components "
            f"{coordinate.components} but the prior's natural value has "
            f"{coordinate.natural_size}. Match the prior (product_transformed_gaussian_prior "
            "keyword order, softmax_normal center length) to the map."
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
            "must be a mapping from every group label to a float."
        )


def check_fixed_value_is_in_domain(name: str, value: float) -> None:
    domain = _FLAT_SPECS[name].domain
    if not _in_domain(domain, jnp.asarray(value, dtype=jnp.float64)):
        raise ValueError(
            f"fixed parameter {name!r} = {value!r} is outside its pySIPNET domain "
            f"{domain.value!r}; pySIPNET would refuse it."
        )


def check_fixed_value_covers_groups(parameter: FixedParameter, labels: tuple[Any, ...]) -> None:
    if not isinstance(parameter.value, Mapping):
        return
    missing = [g for g in labels if g not in parameter.value]
    extra = [g for g in parameter.value if g not in labels]
    if missing or extra:
        raise ValueError(
            f"fixed parameter {parameter.name!r} by {parameter.varies_by!r}: values must be "
            f"keyed on exactly the groups {list(labels)}; missing {missing}, extra {extra}."
        )


def check_sites_are_ascending(sites: tuple[int, ...]) -> None:
    if not sites:
        raise ValueError("a Parameterization needs at least one site.")
    if list(sites) != sorted(set(sites)):
        raise ValueError("sites must be strictly ascending with no repeats.")


def check_coordinate_names_are_unique(coordinates: tuple[Coordinate, ...]) -> None:
    names = [c.name for c in coordinates]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"coordinate names repeat: {repeated}.")
    if not names:
        raise ValueError("a Parameterization needs at least one coordinate.")


def check_labelings_cover_sites(parameterization: Parameterization) -> None:
    reserved = RESERVED_LABELING_NAMES & set(parameterization.labelings)
    if reserved:
        raise ValueError(f"labeling names {sorted(reserved)} are reserved dimension names.")
    for name, labels in parameterization.labelings.items():
        if len(labels) != len(parameterization.sites):
            raise ValueError(
                f"labeling {name!r} has {len(labels)} labels for {len(parameterization.sites)} "
                "sites; give one label per site, in site order."
            )
    used = {c.varies_by for c in parameterization.coordinates} | {
        f.varies_by for f in parameterization.fixed
    }
    missing = sorted(v for v in used if v not in (None, SITE) and v not in parameterization.labelings)
    if missing:
        raise ValueError(
            f"varies_by names {missing} have no labeling; pass labelings={{name: labels}}."
        )


def check_each_sipnet_parameter_has_one_writer(
    coordinates: tuple[Coordinate, ...], fixed: tuple[FixedParameter, ...]
) -> None:
    writers: dict[str, list[str]] = {}
    for coordinate in coordinates:
        for name in coordinate.coord_to_param.writes:
            writers.setdefault(name, []).append(f"coordinate {coordinate.name}")
    for parameter in fixed:
        writers.setdefault(parameter.name, []).append("fixed")
    clashes = {name: who for name, who in writers.items() if len(who) > 1}
    if clashes:
        raise ValueError(
            f"SIPNET parameters set more than once: {clashes}. Each parameter has exactly "
            "one writer, a coordinate or a fixed value."
        )


def check_reads_are_fixed(
    coordinates: tuple[Coordinate, ...], fixed: tuple[FixedParameter, ...]
) -> None:
    fixed_names = {f.name for f in fixed}
    for coordinate in coordinates:
        missing = [n for n in coordinate.coord_to_param.reads if n not in fixed_names]
        if missing:
            raise ValueError(
                f"coordinate {coordinate.name!r} reads {missing}, which are not fixed. Add a "
                "FixedParameter for each."
            )


def check_prior_batch_matches_groups(coordinate: Coordinate, n_groups: int) -> None:
    batch = tuple(coordinate.prior.batch_shape)
    if batch not in ((), (n_groups,)):
        raise ValueError(
            f"coordinate {coordinate.name!r} varies by {coordinate.varies_by!r} with "
            f"{n_groups} groups, but its prior has batch shape {batch}; use () for one "
            f"prior shared by every group or ({n_groups},) for one per group."
        )


def check_map_image_is_in_domain(parameterization: Parameterization, coordinate: Coordinate) -> None:
    """Evaluate ``M_c(T_c(theta))`` at the corners of the unconstrained cube
    (:data:`DOMAIN_CHECK_CORNERS`) for every group, and test each SIPNET
    parameter written against its ``ParameterDomain``."""
    n_groups = parameterization.n_groups(coordinate.varies_by)
    corners = jnp.asarray(
        list(itertools.product(DOMAIN_CHECK_CORNERS, repeat=coordinate.size)), dtype=jnp.float64
    )
    theta_c = jnp.broadcast_to(corners[:, None, :], (corners.shape[0], n_groups, coordinate.size))
    natural = coordinate.bijector.forward(theta_c)
    on_sites = parameterization._onto_sites(coordinate, natural)
    for name, values in coordinate.coord_to_param(on_sites, parameterization._fixed_table).items():
        domain = _FLAT_SPECS[name].domain
        if not _in_domain(domain, values):
            raise ValueError(
                f"coordinate {coordinate.name!r} can write {name!r} outside its pySIPNET "
                f"domain {domain.value!r} (checked at theta in {DOMAIN_CHECK_CORNERS}). Use a "
                "prior whose bijector maps onto the domain: log_normal for a positive "
                "parameter, logit_normal for a fraction, softmax_normal for a simplex."
            )

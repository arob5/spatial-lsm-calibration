"""The SIPNET parameter map: how a value reaches SIPNET.

Where this sits
---------------
::

    parameter_vector.ParameterVector, theta        (calibrated values, by name)
    external inputs                                (propagated values, by name)
      -> sipnet_parameter_map.SIPNETParameterMap   (rules and fixed values)
      -> SIPNET parameter fields                   (fields.SIPNETParameterFields)
      -> forward.ForwardModel, one SIPNET run each

Of the three calibration objects it alone reads pySIPNET's parameter
specs, which own every SIPNET parameter's name, units and domain.

What it reads
-------------
A :class:`~sipnet_calibration.parameter_vector.ParameterVector` and theta,
and optionally :data:`ExternalInputs`: uncertain values a rule reads that are
propagated rather than calibrated.

The map
-------
For run :math:`s`, the SIPNET parameters are

.. math::

    \\psi_s = M\\big(\\{x_p^{(s)}\\}, u_s, c_s\\big),

with :math:`x_p^{(s)}` the parameters at the run's site
(:meth:`~sipnet_calibration.parameter_vector.ParameterVector.at_sites`),
:math:`u_s` its external inputs and :math:`c_s` the :class:`Fixed` values.
:math:`M` is a list of :class:`SIPNETRule`\\ s, applied in order. A rule
reads values by name, whether a parameter's or an external input's, and
declares what it requires of each (:class:`ValueRequirement`); it may read
SIPNET parameters that are fixed or written by an earlier rule. Each SIPNET
parameter has one writer.

**External inputs** (:data:`ExternalInputs`, checked by
:func:`validate_external_inputs`) are an ``xr.Dataset`` of ``float64``
variables, each with ``units`` (``"1"`` or a UDUNITS string pySIPNET
accepts), on any of:

============================ ===================================== ==============
dim                          meaning                               pairs with
                                                                   theta by
============================ ===================================== ==============
``site`` (optional)          one value per site, selected by       zip on
                             label; without it, every site         ``site``
the batch dim (``sample``)   one value per row of theta, labeled   zip
                             ``0`` to ``J - 1``
any other batch dim          an ensemble of the input's values     cross
(integer labels)
============================ ===================================== ==============

Variables sharing a crossed dim vary together; different crossed dims cross.

**The result** is SIPNET parameter fields
(:data:`~sipnet_calibration.fields.SIPNETParameterFields`), one ``float64``
variable per SIPNET parameter written, in ``PARAMETER_SPECS`` order, each on
the dims of what it was computed from, in the order ``(batch dim, crossed
dims, site)``: a variable fed by theta alone is on ``(batch dim, site)``, one
fed also by a crossed input on ``(batch dim, crossed dim, site)``, a fixed
value on ``(site,)``. Each carries pySIPNET's
``ParameterSpec.xarray_attributes()`` and ``set_by``.

Functions and classes
---------------------
:class:`SIPNETParameterMap`
    ``sipnet_parameter_fields``, ``out_of_domain``, ``describe``.
:class:`Fixed`
    A SIPNET parameter held at a value, shared or per dim label.
:class:`SIPNETRule`, :class:`ValueRequirement`, :class:`Bounds`
    The rule protocol, and what a rule requires of a value it reads.
:class:`Copy`, :class:`CopySimplex`, :class:`ComputePhotosynthesisRates`, :class:`ComputeInitialConditions`
    The rules.
:data:`ExternalInputs`, :func:`validate_external_inputs`
    The alias and its validator.
:func:`check_sipnet_parameter_map_fits`, :func:`check_sipnet_parameter_map_is_in_domain_at_the_corners`
    The checks against a vector and external inputs.
:class:`SIPNETParametersOutOfDomainError`
    Raised by the forward model for SIPNET parameters outside pySIPNET's
    domains.

Notes
-----
**A rule is elementwise over leading dims and sites**: each output entry
depends only on the input entries at the same leading position and site.
That is what lets the map broadcast theta's values against external inputs
on other dims and run each rule once.

Usage
-----
::

    vector, prior, _ = example_calibration(site_table, pft)   # sipnet_calibration.calibration
    theta = prior.sample(jax.random.key(0), 50)
    sipnet_map = SIPNETParameterMap(
        rules=[
            ComputePhotosynthesisRates(capacity_value_name="photosynthetic_capacity",
                                       respiration_share_value_name="respiration_share"),
            CopySimplex(value_name="allocation", sipnet_parameter_names=(
                "leaf_allocation", "wood_allocation", "fine_root_allocation")),
            Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon"),
        ],
        fixed=[
            Fixed(sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=0.76,
                  provenance="..."),
            Fixed(sipnet_parameter_name="leaf_carbon_fraction", dim="pft", provenance="...",
                  value={"boreal.coniferous": 0.506, "temperate.deciduous.HPDA": 0.466}),
        ],
    )
    check_sipnet_parameter_map_fits(sipnet_map, vector)
    sipnet_parameter_fields = sipnet_map.sipnet_parameter_fields(vector, theta)
    sipnet_map.out_of_domain(sipnet_parameter_fields)       # empty DataFrame: all in domain
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import inf
from typing import Any, ClassVar, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from pysipnet.parameters.base import ParameterDomain, ParameterSpec
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from pysipnet.units import conversion_factor, validate_units

from sipnet_calibration.conventions import NON_BATCH_DIM_NAMES, SAMPLE, SITE
from sipnet_calibration.fields import (
    SIPNETParameterFields,
    batch_coordinate,
    check_batch_dim_name_is_not_reserved,
    check_sipnet_parameter_name_is_a_flat_name,
)
from sipnet_calibration.initial_conditions.specs import resolve_initial_condition
from sipnet_calibration.parameter_vector import Parameter, ParameterVector, Support
from sipnet_calibration.sites import site_coordinates
from sipnet_calibration.validation import as_batched_flat, as_frozen_mapping, is_one_vector, truncated

__all__ = [
    "DOMAIN_CHECK_CORNERS",
    "INITIAL_STATE_NAMES",
    "REQUIRED_SIPNET_PARAMETER_NAMES",
    "Bounds",
    "ComputeInitialConditions",
    "ComputePhotosynthesisRates",
    "Copy",
    "CopySimplex",
    "ExternalInputs",
    "Fixed",
    "SIPNETParameterMap",
    "SIPNETParametersOutOfDomainError",
    "SIPNETRule",
    "ValueRequirement",
    "check_sipnet_parameter_map_fits",
    "check_sipnet_parameter_map_is_in_domain_at_the_corners",
    "check_sipnet_parameter_map_is_valid",
    "validate_external_inputs",
]

Array = jax.Array


# ── the map ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class SIPNETParameterMap:
    """The map :math:`M` from values to SIPNET parameters: rules applied in
    order, and fixed values.

    Parameters
    ----------
    rules:
        The :class:`SIPNETRule`\\ s, in the order they run.
    fixed:
        The :class:`Fixed` SIPNET parameters.

    Raises
    ------
    KeyError
        If a SIPNET parameter written or read is not pySIPNET's.
    ValueError
        If a SIPNET parameter has two writers, a rule reads a SIPNET
        parameter neither fixed nor written by an earlier rule, or a fixed
        value is outside its domain.
    """

    rules: Sequence[SIPNETRule]
    fixed: Sequence[Fixed] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "fixed", tuple(self.fixed))
        check_sipnet_parameter_map_is_valid(self)

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        """Every SIPNET parameter the rules and fixed values write, in
        ``PARAMETER_SPECS`` order."""
        written = set(self._writers)
        return tuple(name for name in _FLAT_SPECS if name in written)

    @property
    def values_read(self) -> Mapping[str, tuple[ValueRequirement, ...]]:
        """The values the rules read, each with every requirement a rule
        places on it, in rule order."""
        requirements: dict[str, tuple[ValueRequirement, ...]] = {}
        for rule in self.rules:
            for name, requirement in rule.values_read.items():
                requirements[name] = (*requirements.get(name, ()), requirement)
        return frozendict(requirements)

    def crossed_dims(self, external_inputs: ExternalInputs | None, *, batch_dim: str = SAMPLE) -> tuple[str, ...]:
        """The batch dims of the external inputs the rules read, other than
        *batch_dim*, in the order they first appear: the dims crossed with
        theta's rows."""
        if external_inputs is None:
            return ()
        dims: dict[str, None] = {}
        for name, variable in external_inputs.data_vars.items():
            if name in self.values_read:
                dims.update(dict.fromkeys(str(d) for d in variable.dims if d not in (SITE, batch_dim)))
        return tuple(dims)

    @property
    def unset_sipnet_parameter_names(self) -> tuple[str, ...]:
        """The :data:`REQUIRED_SIPNET_PARAMETER_NAMES` this map writes none of;
        a run's base parameter set supplies them."""
        written = set(self._writers)
        return tuple(name for name in REQUIRED_SIPNET_PARAMETER_NAMES if name not in written)

    def describe(self) -> pd.DataFrame:
        """One row per SIPNET parameter written, indexed by
        ``sipnet_parameter``: ``set_by``, ``values_read`` and ``provenance``."""
        rows = []
        for name in self.sipnet_parameter_names_written:
            writer = self._writers[name]
            if isinstance(writer, Fixed):
                rows.append({"sipnet_parameter": name, "set_by": "fixed", "values_read": "",
                             "provenance": writer.provenance})
            else:
                rows.append({"sipnet_parameter": name, "set_by": _set_by(writer),
                             "values_read": ", ".join(writer.values_read), "provenance": ""})
        return pd.DataFrame(rows).set_index("sipnet_parameter")

    # ── evaluation ────────────────────────────────────────────────────────────

    def sipnet_parameter_fields(
        self,
        parameter_vector: ParameterVector,
        theta: Any,
        *,
        external_inputs: ExternalInputs | None = None,
        batch_dim: str = SAMPLE,
    ) -> SIPNETParameterFields:
        """Theta, and any external inputs, to SIPNET parameter fields.

        Each rule reads its values aligned by dim name and broadcast to the
        union of their dims, and writes each SIPNET parameter on that union.
        The values may lie outside pySIPNET's domains; :meth:`out_of_domain`
        says where.

        Parameters
        ----------
        parameter_vector:
            The vector theta is of.
        theta:
            ``(D,)``, or ``(J, D)`` labeled ``0`` to ``J - 1`` on *batch_dim*.
        external_inputs:
            :data:`ExternalInputs`, or ``None``.
        batch_dim:
            The name of theta's batch dim.

        Returns
        -------
        SIPNETParameterFields
            As the module's docstring describes.

        Raises
        ------
        TypeError, ValueError
            If *theta* is not ``(D,)`` or ``(J, D)``, or *external_inputs*
            are not :data:`ExternalInputs` for *theta*.
        KeyError
            If a value a rule reads is neither a parameter nor an external
            input, or an external input lacks a site of the vector.
        """
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        batched = jnp.asarray(as_batched_flat(theta, parameter_vector.dimension, message_name="theta"))
        one = is_one_vector(theta)
        if external_inputs is not None:
            validate_external_inputs(external_inputs, batch_dim=batch_dim)
            check_external_inputs_are_for_theta(external_inputs, None if one else len(batched), batch_dim)
        natural_values = parameter_vector.to_natural(batched[0] if one else batched)
        at_sites = parameter_vector.at_sites(natural_values)
        theta_dims = (SITE,) if one else (batch_dim, SITE)
        values: dict[str, _Labeled] = {name: _Labeled(theta_dims, array) for name, array in at_sites.items()}
        order = (batch_dim, *self.crossed_dims(external_inputs, batch_dim=batch_dim), SITE)
        if external_inputs is not None:
            read = external_inputs[[name for name in external_inputs.data_vars if name in self.values_read]]
            values.update(_external_values(read, parameter_vector, order))
        sizes = _dim_sizes(values)
        site_table = parameter_vector.site_table
        written: dict[str, tuple[_Labeled, str]] = {
            fixed.sipnet_parameter_name: (_Labeled((SITE,), fixed.at_sites(parameter_vector)), "fixed")
            for fixed in self.fixed
        }
        external_names = () if external_inputs is None else tuple(map(str, external_inputs.data_vars))
        for rule in self.rules:
            for name in rule.values_read:
                check_value_is_held_once(name, parameter_vector, external_names)
            read = {name: values[name] for name in rule.values_read}
            read_sipnet = {name: written[name][0] for name in rule.sipnet_parameter_names_read}
            dims = _union(order, [*read.values(), *read_sipnet.values()])
            output = rule(
                {n: v.broadcast(dims, sizes) for n, v in read.items()},
                {n: v.broadcast(dims, sizes) for n, v in read_sipnet.items()},
                site_table,
            )
            for name, array in output.items():
                written[name] = (_Labeled(dims, array), _set_by(rule))
        return self._dataset(parameter_vector, written, batched, one, order[:-1], external_inputs)

    def out_of_domain(self, sipnet_parameter_fields: SIPNETParameterFields) -> pd.DataFrame:
        """Where SIPNET parameter fields lie outside pySIPNET's domains.

        Returns
        -------
        pandas.DataFrame
            One row per value outside its SIPNET parameter's
            ``ParameterDomain`` (non-finite values included): a column per
            dim of the fields (missing for a variable not on it), then
            ``sipnet_parameter`` and ``value``. Empty when every value is in
            its domain.
        """
        dims = list(sipnet_parameter_fields.dims)
        rows = []
        for name, variable in sipnet_parameter_fields.data_vars.items():
            values = np.asarray(variable.values, dtype=np.float64)
            outside = ~_FLAT_SPECS[str(name)].domain.contains(values) | ~np.isfinite(values)
            labels = [variable[dim].values for dim in variable.dims]
            for position in np.argwhere(outside):
                row = {dim: labels[axis][i] for axis, (dim, i) in enumerate(zip(variable.dims, position))}
                rows.append({**row, "sipnet_parameter": str(name), "value": values[tuple(position)]})
        return pd.DataFrame(rows, columns=[*dims, "sipnet_parameter", "value"])

    # ── supporting methods ────────────────────────────────────────────────────

    @property
    def _writers(self) -> dict[str, SIPNETRule | Fixed]:
        writers: dict[str, SIPNETRule | Fixed] = {f.sipnet_parameter_name: f for f in self.fixed}
        for rule in self.rules:
            writers.update(dict.fromkeys(rule.sipnet_parameter_names_written, rule))
        return writers

    def _dataset(
        self,
        parameter_vector: ParameterVector,
        written: Mapping[str, tuple[_Labeled, str]],
        batched: Array,
        one: bool,
        batch_dims: tuple[str, ...],
        external_inputs: xr.Dataset | None,
    ) -> xr.Dataset:
        """The SIPNET parameter fields; *batch_dims* are theta's batch dim,
        then the crossed dims."""
        batch_dim, *crossed = batch_dims
        coordinates: dict[str, Any] = dict(site_coordinates(parameter_vector.sites, parameter_vector.site_table))
        if not one:
            coordinates[batch_dim] = batch_coordinate(batch_dim, np.arange(len(batched)))
        for dim in crossed:
            coordinates[dim] = batch_coordinate(dim, external_inputs[dim].values)
        variables = {
            name: (
                labeled.dims,
                np.asarray(labeled.array, dtype=np.float64),
                {**_FLAT_SPECS[name].xarray_attributes(), "set_by": set_by},
            )
            for name, (labeled, set_by) in sorted(written.items(), key=lambda kv: _SPEC_ORDER[kv[0]])
        }
        return xr.Dataset(variables, coords=coordinates)


# ── fixed values ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class Fixed:
    """A SIPNET parameter held at a value.

    Parameters
    ----------
    sipnet_parameter_name:
        pySIPNET's flat name.
    value:
        A number; or, with *dim*, ``{dim label: number}`` covering the
        vector's dim labels (a superset is allowed).
    dim:
        ``"site"``, a site-labels name of the vector, or ``None``.
    provenance:
        Where the value came from.

    Raises
    ------
    TypeError
        If *value* is not a number or a mapping of numbers, or a mapping is
        given without *dim*.
    ValueError
        If *provenance* is empty.
    """

    sipnet_parameter_name: str
    value: float | Mapping[Any, float]
    dim: str | None = None
    provenance: str

    def __post_init__(self) -> None:
        if isinstance(self.value, Mapping):
            object.__setattr__(self, "value", as_frozen_mapping(self.value, message_name="value"))
        check_fixed_is_valid(self)

    def values(self) -> tuple[float, ...]:
        """Every value, in mapping order."""
        if isinstance(self.value, Mapping):
            return tuple(float(v) for v in self.value.values())
        return (float(self.value),)

    def at_sites(self, parameter_vector: ParameterVector) -> Array:
        """The value at every site of the vector, ``(S,)``."""
        if self.dim is None:
            return jnp.full(parameter_vector.n_sites, float(self.value))
        labels = (
            parameter_vector.sites
            if self.dim == SITE
            else parameter_vector.site_table[self.dim].astype(str).tolist()
        )
        return jnp.asarray([float(self.value[label]) for label in labels])


# ── rules ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class SIPNETRule(Protocol):
    """A rule of the map: values read by name to SIPNET parameters.

    Attributes
    ----------
    values_read:
        ``{name: ValueRequirement}``: parameters or external inputs.
    sipnet_parameter_names_read:
        SIPNET parameters it reads, each fixed or written by an earlier rule.
    sipnet_parameter_names_written:
        SIPNET parameters it writes.

    A rule may also declare ``dim_label_arguments``, ``{dim name: mapping
    keyed by dim label}``, whose keys the fit check holds to cover the
    vector's dim labels.

    The call takes ``values_at_sites`` (``{name: (..., S)}``, or ``(..., S,
    k)`` for a vector), ``sipnet_parameter_values`` (``{name: (..., S)}``)
    and the vector's site table, all broadcast to one leading shape, and
    returns ``{SIPNET name written: (..., S)}``. It must be elementwise over
    the leading dims and sites, and traceable by JAX.
    """

    values_read: Mapping[str, ValueRequirement]
    sipnet_parameter_names_read: tuple[str, ...]
    sipnet_parameter_names_written: tuple[str, ...]

    def __call__(
        self,
        values_at_sites: Mapping[str, Array],
        sipnet_parameter_values: Mapping[str, Array],
        site_table: pd.DataFrame,
    ) -> dict[str, Array]: ...


@dataclass(frozen=True)
class ValueRequirement:
    """What a rule requires of a value it reads.

    Parameters
    ----------
    units:
        UDUNITS units the value must be in, compared by a conversion factor
        of exactly 1 (``pysipnet.units.conversion_factor``); ``None`` for no
        physical units, which reads as ``"1"``.
    bounds:
        The :class:`Bounds` every number of the value lies in, or ``None``.
    natural_size:
        ``k``, the numbers per site.
    """

    units: str | None
    bounds: Bounds | None = None
    natural_size: int = 1


@dataclass(frozen=True)
class Bounds:
    """An interval a value must lie in; either end may be closed or infinite.

    Parameters
    ----------
    low, high:
        The ends.
    low_closed, high_closed:
        Whether each end belongs to the interval.
    """

    low: float = -inf
    high: float = inf
    low_closed: bool = False
    high_closed: bool = False

    @classmethod
    def from_sipnet_domain(cls, domain: ParameterDomain) -> Bounds:
        """The bounds of a pySIPNET ``ParameterDomain``: ``REAL``
        :math:`(-\\infty, \\infty)`, ``POSITIVE`` :math:`(0, \\infty)`,
        ``NON_NEGATIVE`` :math:`[0, \\infty)`, ``UNIT_INTERVAL`` :math:`[0, 1]`,
        ``OPEN_UNIT_INTERVAL`` :math:`(0, 1)`."""
        return _DOMAIN_BOUNDS[domain]

    def contains(self, values: Any) -> Array:
        """Whether each value lies within the bounds, elementwise."""
        values = jnp.asarray(values, dtype=jnp.float64)
        above = values >= self.low if self.low_closed else values > self.low
        below = values <= self.high if self.high_closed else values < self.high
        return above & below

    def contains_support(self, support: Support) -> bool:
        """Whether the open *support* lies within the bounds, every entry of
        a simplex being in :math:`(0, 1)`."""
        low, high = {
            "real": (-inf, inf),
            "positive": (0.0, inf),
            "interval": (support.low, support.high),
            "simplex": (0.0, 1.0),
        }[support.kind]
        return self.low <= low and high <= self.high


@dataclass(frozen=True, kw_only=True)
class Copy:
    """The identity: :math:`\\psi = x`, one scalar value to one SIPNET
    parameter, which requires the value in its units and domain."""

    value_name: str
    sipnet_parameter_name: str
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        return (self.sipnet_parameter_name,)

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        spec = _spec(self.sipnet_parameter_name)
        return {self.value_name: ValueRequirement(spec.units, Bounds.from_sipnet_domain(spec.domain))}

    def __call__(self, values_at_sites, sipnet_parameter_values, site_table) -> dict[str, Array]:
        return {self.sipnet_parameter_name: values_at_sites[self.value_name]}


@dataclass(frozen=True, kw_only=True)
class CopySimplex:
    """A point :math:`x` of the ``k``-simplex to ``k - 1`` SIPNET parameters:
    :math:`\\psi_i = x_i` for :math:`i < k`.

    The last number is not written: SIPNET recomputes it as
    :math:`1 - \\sum_{i<k} x_i` (``sipnet.c:1113-1115``) and exits if that
    is negative (``sipnet.c:1117-1122``). Reading the whole simplex is
    what keeps every draw strictly inside it.
    """

    value_name: str
    sipnet_parameter_names: tuple[str, ...]
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        return tuple(self.sipnet_parameter_names)

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        bounds = _intersection(
            [Bounds.from_sipnet_domain(_spec(n).domain) for n in self.sipnet_parameter_names]
        )
        return {self.value_name: ValueRequirement("1", bounds, len(self.sipnet_parameter_names) + 1)}

    def __call__(self, values_at_sites, sipnet_parameter_values, site_table) -> dict[str, Array]:
        simplex = values_at_sites[self.value_name]
        return {name: simplex[..., i] for i, name in enumerate(self.sipnet_parameter_names)}


@dataclass(frozen=True, kw_only=True)
class ComputePhotosynthesisRates:
    """SIPNET's photosynthesis pair from its identifiable combination.

    ``aMax`` (:math:`A`), ``aMaxFrac`` (:math:`f`), ``baseFolRespFrac``
    (:math:`r`) and ``cFracLeaf`` (:math:`c`) enter SIPNET only through
    :math:`P = A (f + r) / c` and :math:`\\rho = r / (f + r)`
    (``sipnet.c:614, 617, 633``). With :math:`f` and :math:`c` read as SIPNET
    parameters, this rule inverts them:

    .. math::

        r = \\frac{\\rho f}{1 - \\rho}, \\qquad A = \\frac{P c (1 - \\rho)}{f},

    from the capacity :math:`P` (``nmol g-1 s-1``, per gram of leaf carbon)
    and the respiration share :math:`\\rho` (``1``).
    """

    capacity_value_name: str
    respiration_share_value_name: str
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = (
        "daily_mean_photosynthesis_fraction",
        "leaf_carbon_fraction",
    )
    sipnet_parameter_names_written: ClassVar[tuple[str, ...]] = (
        "max_photosynthesis_rate",
        "foliar_respiration_fraction",
    )

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        return {
            self.capacity_value_name: ValueRequirement(
                _spec("max_photosynthesis_rate").units, Bounds(0.0, inf)
            ),
            self.respiration_share_value_name: ValueRequirement("1", Bounds(0.0, 1.0)),
        }

    def __call__(self, values_at_sites, sipnet_parameter_values, site_table) -> dict[str, Array]:
        capacity = values_at_sites[self.capacity_value_name]
        share = values_at_sites[self.respiration_share_value_name]
        fraction = sipnet_parameter_values["daily_mean_photosynthesis_fraction"]
        leaf_carbon = sipnet_parameter_values["leaf_carbon_fraction"]
        return {
            "max_photosynthesis_rate": capacity * leaf_carbon * (1.0 - share) / fraction,
            "foliar_respiration_fraction": share * fraction / (1.0 - share),
        }


#: The initial state a :class:`ComputeInitialConditions` reads, in order: soil
#: organic carbon, wood carbon and leaf carbon (``kg m-2`` of carbon), and
#: surface soil moisture (percent of saturation), in the initial conditions'
#: processed file's names and units.
INITIAL_STATE_NAMES: tuple[str, ...] = (
    "initial_soil_organic_carbon",
    "initial_wood_carbon",
    "initial_leaf_carbon",
    "initial_soil_moisture_saturation",
)


@dataclass(frozen=True, kw_only=True)
class ComputeInitialConditions:
    """SIPNET's initial state from a site's state values.

    With soil organic carbon :math:`C_s`, wood carbon :math:`C_w`, leaf
    carbon :math:`C_l` (``kg m-2``), soil moisture :math:`m` (percent of
    saturation), and the SIPNET parameters ``fine_root_fraction``
    :math:`f`, ``coarse_root_fraction`` :math:`g` and
    ``leaf_carbon_per_area`` :math:`\\lambda` (``g m-2``):

    .. math::

        \\mathtt{soil\\_carbon} = 1000\\, C_s, \\quad
        \\mathtt{total\\_wood\\_carbon} = \\frac{1000\\, C_w}{1 - f - g}, \\quad
        \\mathtt{leaf\\_area\\_index} = \\begin{cases} 0 & \\text{deciduous} \\\\
            1000\\, C_l / \\lambda & \\text{otherwise} \\end{cases}, \\quad
        \\mathtt{soil\\_wetness\\_fraction} = m / 100,

    as :func:`sipnet_calibration.initial_conditions.to_sipnet_initial_conditions`
    documents. A deciduous site starts with no leaves, since runs start
    outside leaf-on.

    Parameters
    ----------
    deciduous:
        ``{dim label: bool}`` over *deciduous_dim*'s dim labels (a superset
        is allowed).
    deciduous_dim:
        The site-labels name a site's deciduousness is read from.
    state_value_names:
        The names the four state values are read by, in the order of
        :data:`INITIAL_STATE_NAMES`.

    Raises
    ------
    TypeError
        If a value of *deciduous* is not a boolean.
    """

    deciduous: Mapping[str, bool]
    deciduous_dim: str = "pft"
    state_value_names: tuple[str, ...] = INITIAL_STATE_NAMES
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = (
        "fine_root_fraction",
        "coarse_root_fraction",
        "leaf_carbon_per_area",
    )
    sipnet_parameter_names_written: ClassVar[tuple[str, ...]] = (
        "soil_carbon",
        "total_wood_carbon",
        "leaf_area_index",
        "soil_wetness_fraction",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "deciduous", as_frozen_mapping(self.deciduous, message_name="deciduous"))
        object.__setattr__(self, "state_value_names", tuple(self.state_value_names))
        check_deciduous_values_are_booleans(self.deciduous)

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        carbon, saturation = Bounds(0.0, inf, low_closed=True), Bounds(0.0, 100.0, True, True)
        return {
            name: ValueRequirement(
                resolve_initial_condition(state).units,
                saturation if state == "initial_soil_moisture_saturation" else carbon,
            )
            for name, state in zip(self.state_value_names, INITIAL_STATE_NAMES, strict=True)
        }

    @property
    def dim_label_arguments(self) -> Mapping[str, Mapping[str, bool]]:
        return {self.deciduous_dim: self.deciduous}

    def __call__(self, values_at_sites, sipnet_parameter_values, site_table) -> dict[str, Array]:
        soil, wood, leaf, moisture = (values_at_sites[name] for name in self.state_value_names)
        fine = sipnet_parameter_values["fine_root_fraction"]
        coarse = sipnet_parameter_values["coarse_root_fraction"]
        leaf_carbon_per_area = sipnet_parameter_values["leaf_carbon_per_area"]
        deciduous = np.asarray(
            [self.deciduous[label] for label in site_table[self.deciduous_dim].astype(str)], dtype=bool
        )
        return {
            "soil_carbon": _GRAMS_PER_KILOGRAM * soil,
            "total_wood_carbon": _GRAMS_PER_KILOGRAM * wood / (1.0 - fine - coarse),
            "leaf_area_index": jnp.where(deciduous, 0.0, _GRAMS_PER_KILOGRAM * leaf / leaf_carbon_per_area),
            "soil_wetness_fraction": moisture / _PERCENT,
        }


# ── external inputs ───────────────────────────────────────────────────────────

#: Uncertain values a rule reads that are propagated, not calibrated, as the
#: module's docstring has them; checked by :func:`validate_external_inputs`.
type ExternalInputs = xr.Dataset


def validate_external_inputs(external_inputs: Any, *, batch_dim: str = SAMPLE) -> None:
    """Check that *external_inputs* are :data:`ExternalInputs`, as the
    module's docstring has them, *batch_dim* being theta's batch dim.

    Raises
    ------
    TypeError
        If it is not an ``xr.Dataset``.
    ValueError
        If a variable is not ``float64`` or lacks valid ``units``, or a dim
        is neither ``site`` nor a batch dim with integer labels.
    """
    check_external_inputs_are_a_dataset(external_inputs)
    for name, variable in external_inputs.data_vars.items():
        check_external_input_is_float64_with_units(str(name), variable)
    for dim in external_inputs.dims:
        check_external_input_dim_is_site_or_a_batch_dim(str(dim), external_inputs)


# ── constants and the domain error ────────────────────────────────────────────

#: The SIPNET parameters pySIPNET requires a value for, in declaration order:
#: every parameter without a default.
REQUIRED_SIPNET_PARAMETER_NAMES: tuple[str, ...] = tuple(
    name
    for group in SIPNETParameters.model_fields.values()
    for name, field_info in group.annotation.model_fields.items()
    if field_info.is_required()
)

#: The values of each unconstrained number at which
#: :func:`check_sipnet_parameter_map_is_in_domain_at_the_corners` evaluates
#: the map: ``+-12`` spans ten orders of magnitude on a log scale and reaches
#: ``1 - 6e-6`` on a logit scale, while staying inside float64.
DOMAIN_CHECK_CORNERS: tuple[float, ...] = (-12.0, 0.0, 12.0)


class SIPNETParametersOutOfDomainError(ValueError):
    """SIPNET parameters outside pySIPNET's domains: the prior and the map
    put mass where SIPNET is undefined."""


# ── private helpers ───────────────────────────────────────────────────────────

_FLAT_SPECS: Mapping[str, ParameterSpec] = frozendict(
    {path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()}
)
_SPEC_ORDER: Mapping[str, int] = frozendict({name: i for i, name in enumerate(_FLAT_SPECS)})

_DOMAIN_BOUNDS: Mapping[ParameterDomain, Bounds] = frozendict(
    {
        ParameterDomain.REAL: Bounds(),
        ParameterDomain.POSITIVE: Bounds(0.0, inf),
        ParameterDomain.NON_NEGATIVE: Bounds(0.0, inf, low_closed=True),
        ParameterDomain.UNIT_INTERVAL: Bounds(0.0, 1.0, True, True),
        ParameterDomain.OPEN_UNIT_INTERVAL: Bounds(0.0, 1.0),
    }
)

#: Grams in a kilogram, and percent in one.
_GRAMS_PER_KILOGRAM, _PERCENT = 1000.0, 100.0


@dataclass(frozen=True)
class _Labeled:
    """An array whose leading axes are *dims*, and whose further axes (a
    vector's natural numbers) are not dims."""

    dims: tuple[str, ...]
    array: Array

    def broadcast(self, dims: tuple[str, ...], sizes: Mapping[str, int]) -> Array:
        """The array on *dims*, a superset of its own in the same order."""
        array = jnp.asarray(self.array)
        trailing = array.shape[len(self.dims):]
        for axis, dim in enumerate(dims):
            if dim not in self.dims:
                array = jnp.expand_dims(array, axis)
        return jnp.broadcast_to(array, tuple(sizes[d] for d in dims) + trailing)


def _spec(sipnet_parameter_name: str) -> ParameterSpec:
    return _FLAT_SPECS[sipnet_parameter_name]


def _set_by(rule: Any) -> str:
    return f"rule {type(rule).__name__}"


def _intersection(bounds: Sequence[Bounds]) -> Bounds:
    """The tightest bounds within every one of *bounds*."""
    low = max(b.low for b in bounds)
    high = min(b.high for b in bounds)
    return Bounds(
        low,
        high,
        all(b.low_closed for b in bounds if b.low == low),
        all(b.high_closed for b in bounds if b.high == high),
    )


def _union(order: tuple[str, ...], labeled: Sequence[_Labeled]) -> tuple[str, ...]:
    """The dims of *labeled*, in *order*; always ``site``, since every value
    is read at the sites."""
    present = {SITE, *(dim for value in labeled for dim in value.dims)}
    return tuple(dim for dim in order if dim in present)


def _dim_sizes(values: Mapping[str, _Labeled]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for value in values.values():
        sizes.update(zip(value.dims, jnp.shape(value.array)))
    return sizes


def _external_values(
    external_inputs: xr.Dataset, parameter_vector: ParameterVector, order: tuple[str, ...]
) -> dict[str, _Labeled]:
    """Each external input on the vector's sites, its dims in *order*."""
    out = {}
    for name, variable in external_inputs.data_vars.items():
        if SITE in variable.dims:
            check_external_input_covers_the_sites(str(name), variable, parameter_vector.sites)
            variable = variable.sel({SITE: list(parameter_vector.sites)})
        variable = variable.transpose(*[d for d in order if d in variable.dims])
        out[str(name)] = _Labeled(tuple(str(d) for d in variable.dims), jnp.asarray(variable.values))
    return out


def _corner_theta(parameter_vector: ParameterVector) -> np.ndarray:
    """Rows of theta: each parameter at every corner of its unconstrained
    numbers (:data:`DOMAIN_CHECK_CORNERS`, the same for each dim label), the
    others at 0."""
    rows = []
    for parameter in parameter_vector.parameters:
        positions = parameter_vector.positions(parameter_name=parameter.name)
        for corner in itertools.product(DOMAIN_CHECK_CORNERS, repeat=parameter.unconstrained_size):
            row = np.zeros(parameter_vector.dimension)
            row[positions] = np.tile(corner, len(positions) // parameter.unconstrained_size)
            rows.append(row)
    return np.asarray(rows)


def _units_match(units: str | None, required: str | None) -> bool:
    """Whether *units* convert to *required* by exactly 1; ``None`` reads as
    ``"1"``."""
    try:
        return conversion_factor(units=units or "1", to_units=required or "1") == 1.0
    except (ValueError, KeyError):
        return False


# ── checks ────────────────────────────────────────────────────────────────────


def check_sipnet_parameter_map_is_valid(sipnet_parameter_map: SIPNETParameterMap) -> None:
    """The rules and fixed values make one map pySIPNET can take."""
    writers: dict[str, str] = {}
    for fixed in sipnet_parameter_map.fixed:
        check_sipnet_parameter_name_is_a_flat_name(fixed.sipnet_parameter_name, "a fixed value's")
        check_sipnet_parameter_has_one_writer(fixed.sipnet_parameter_name, "a fixed value", writers)
        check_fixed_values_are_in_the_domain(fixed)
    for rule in sipnet_parameter_map.rules:
        for name in rule.sipnet_parameter_names_read:
            check_sipnet_parameter_name_is_a_flat_name(name, f"{_set_by(rule)} reads, whose names")
            check_sipnet_parameter_read_is_set_earlier(name, rule, writers)
        for name in rule.sipnet_parameter_names_written:
            check_sipnet_parameter_name_is_a_flat_name(name, f"{_set_by(rule)} writes, whose names")
            check_sipnet_parameter_has_one_writer(name, _set_by(rule), writers)


def check_sipnet_parameter_map_fits(
    sipnet_parameter_map: SIPNETParameterMap,
    parameter_vector: ParameterVector,
    external_inputs: ExternalInputs | None = None,
) -> None:
    """The map reads what the vector and external inputs hold, as its rules
    require, and its per-dim-label values cover the vector's dim labels."""
    external_names = () if external_inputs is None else tuple(map(str, external_inputs.data_vars))
    check_external_inputs_share_no_name_with_the_parameters(external_names, parameter_vector)
    for name, requirements in sipnet_parameter_map.values_read.items():
        check_value_is_held_once(name, parameter_vector, external_names)
        for requirement in requirements:
            if name in parameter_vector:
                check_parameter_meets_the_requirement(parameter_vector[name], requirement)
            else:
                check_external_input_meets_the_requirement(name, external_inputs[name], requirement)
    for fixed in sipnet_parameter_map.fixed:
        if fixed.dim is not None:
            check_keys_cover_the_dim_labels(
                fixed.value, fixed.dim, parameter_vector, f"fixed {fixed.sipnet_parameter_name!r}"
            )
    for rule in sipnet_parameter_map.rules:
        for dim, mapping in getattr(rule, "dim_label_arguments", {}).items():
            check_keys_cover_the_dim_labels(mapping, dim, parameter_vector, _set_by(rule))


def check_sipnet_parameter_map_is_in_domain_at_the_corners(
    sipnet_parameter_map: SIPNETParameterMap,
    parameter_vector: ParameterVector,
    external_inputs: ExternalInputs | None = None,
    *,
    batch_dim: str = SAMPLE,
) -> None:
    """The map writes SIPNET parameters in pySIPNET's domains with each
    parameter at every corner of :data:`DOMAIN_CHECK_CORNERS`, the others at
    0: an early warning, complete only for maps monotone in each number of
    theta."""
    corners = _corner_theta(parameter_vector)
    inputs = external_inputs
    if inputs is not None and batch_dim in inputs.dims:
        # Rows of theta here are corners, not samples: take the inputs of one.
        inputs = inputs.isel({batch_dim: 0}, drop=True)
    fields = sipnet_parameter_map.sipnet_parameter_fields(
        parameter_vector, corners, external_inputs=inputs, batch_dim=batch_dim
    )
    outside = sipnet_parameter_map.out_of_domain(fields)
    if not outside.empty:
        names = sorted(set(outside["sipnet_parameter"]))
        raise ValueError(
            f"the map can write {truncated(names)} outside pySIPNET's domains, at a corner of "
            f"theta in {DOMAIN_CHECK_CORNERS}; give the parameter a support or prior whose "
            "values the rule maps into the domain."
        )


def check_fixed_is_valid(fixed: Fixed) -> None:
    """A fixed value is a number, or with a dim a mapping of numbers, and
    says where it came from."""
    if isinstance(fixed.value, Mapping) != (fixed.dim is not None):
        raise TypeError(
            f"fixed {fixed.sipnet_parameter_name!r}: a value per dim label is a mapping with "
            "dim=, and a shared value a number without it."
        )
    for value in (fixed.value.values() if isinstance(fixed.value, Mapping) else [fixed.value]):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"fixed {fixed.sipnet_parameter_name!r} holds {value!r}, which is not a number; "
                "give numbers."
            )
    if not isinstance(fixed.provenance, str) or not fixed.provenance.strip():
        raise ValueError(
            f"fixed {fixed.sipnet_parameter_name!r} has no provenance; say where the value came from."
        )


def check_fixed_values_are_in_the_domain(fixed: Fixed) -> None:
    """Every fixed value lies in its SIPNET parameter's domain."""
    domain = _spec(fixed.sipnet_parameter_name).domain
    values = np.asarray(fixed.values(), dtype=np.float64)
    if not (domain.contains(values) & np.isfinite(values)).all():
        raise ValueError(
            f"fixed {fixed.sipnet_parameter_name!r} holds a value outside pySIPNET's domain "
            f"{domain.value!r}; fix it inside the domain."
        )


def check_sipnet_parameter_has_one_writer(name: str, writer: str, writers: dict[str, str]) -> None:
    """Each SIPNET parameter is written once, by a rule or a fixed value."""
    if name in writers:
        raise ValueError(
            f"{name!r} is written by both {writers[name]} and {writer}; keep one writer."
        )
    writers[name] = writer


def check_sipnet_parameter_read_is_set_earlier(name: str, rule: Any, writers: Mapping[str, str]) -> None:
    """A SIPNET parameter a rule reads is fixed or written by an earlier rule."""
    if name not in writers:
        raise ValueError(
            f"{_set_by(rule)} reads {name!r}, which is neither fixed nor written by an earlier "
            "rule; fix it, or move the rule that writes it before this one."
        )


def check_external_inputs_share_no_name_with_the_parameters(
    external_names: Sequence[str], parameter_vector: ParameterVector
) -> None:
    """No external input is named like a parameter, since values are read by
    name."""
    shared = [name for name in external_names if name in parameter_vector]
    if shared:
        raise ValueError(
            f"the external inputs {truncated(shared)} are named like parameters of the vector; "
            "values are read by name, so rename them."
        )


def check_value_is_held_once(name: str, parameter_vector: ParameterVector, external_names: Sequence[str]) -> None:
    """A value a rule reads is exactly one parameter or external input."""
    held = (name in parameter_vector) + (name in external_names)
    if held == 0:
        raise KeyError(
            f"the map reads {name!r}, which is neither a parameter nor an external input; add it "
            "to one of them, or read a value that exists."
        )
    if held == 2:
        raise ValueError(
            f"the map reads {name!r}, which is both a parameter and an external input; rename one."
        )


def check_parameter_meets_the_requirement(parameter: Parameter, requirement: ValueRequirement) -> None:
    """A parameter read by a rule is in the units, natural size and bounds
    the rule requires, its whole support lying within the bounds."""
    if not _units_match(parameter.units, requirement.units):
        raise ValueError(
            f"parameter {parameter.name!r} is in {parameter.units!r}, but the rule reading it "
            f"requires {requirement.units!r}; give the parameter those units."
        )
    if parameter.natural_size != requirement.natural_size:
        raise ValueError(
            f"parameter {parameter.name!r} has {parameter.natural_size} natural numbers, but the "
            f"rule reading it requires {requirement.natural_size}."
        )
    if requirement.bounds is not None and not requirement.bounds.contains_support(parameter.support):
        raise ValueError(
            f"parameter {parameter.name!r} has support {parameter.support.name!r}, which reaches "
            f"outside what the rule reading it requires ({requirement.bounds}); give it a support "
            "within those bounds."
        )


def check_external_input_meets_the_requirement(
    name: str, variable: xr.DataArray, requirement: ValueRequirement
) -> None:
    """An external input read by a rule is in the units the rule requires,
    and every value lies within its bounds."""
    if requirement.natural_size != 1:
        raise ValueError(
            f"external input {name!r} is read by a rule requiring {requirement.natural_size} "
            "numbers per site, and an external input holds one."
        )
    if not _units_match(variable.attrs.get("units"), requirement.units):
        raise ValueError(
            f"external input {name!r} is in {variable.attrs.get('units')!r}, but the rule reading "
            f"it requires {requirement.units!r}; convert it first."
        )
    if requirement.bounds is not None:
        inside = np.asarray(requirement.bounds.contains(variable.values))
        if not inside.all():
            where = variable.where(xr.DataArray(~inside, dims=variable.dims), drop=True)
            labels = {str(d): truncated(where[d].values.tolist()) for d in where.dims if d in where.coords}
            raise ValueError(
                f"external input {name!r} holds values outside {requirement.bounds}, at {labels}; "
                "choose or clip the members before they enter."
            )


def check_keys_cover_the_dim_labels(
    mapping: Mapping[Any, Any], dim: str, parameter_vector: ParameterVector, what: str
) -> None:
    """A mapping per dim label covers every dim label of the vector."""
    labels = list(parameter_vector.dim_index(dim))
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise KeyError(
            f"{what} has no value for {dim} dim label(s) {truncated(missing)}; key it by every "
            "dim label of the vector."
        )


def check_deciduous_values_are_booleans(deciduous: Mapping[Any, Any]) -> None:
    """Deciduousness is a boolean per dim label: a number cast to bool would
    make every non-zero value, NaN included, deciduous and zero its leaves."""
    bad = [key for key, value in deciduous.items() if not isinstance(value, (bool, np.bool_))]
    if bad:
        raise TypeError(f"deciduous values must be booleans; {truncated(bad)} are not.")


def check_external_inputs_are_a_dataset(external_inputs: Any) -> None:
    """External inputs are an ``xr.Dataset``, which is how they are read."""
    if not isinstance(external_inputs, xr.Dataset):
        raise TypeError(
            f"external inputs are an xarray Dataset, got {type(external_inputs).__name__}; pass one."
        )


def check_external_input_is_float64_with_units(name: str, variable: xr.DataArray) -> None:
    """An external input is ``float64`` and carries valid ``units``, which the
    fit check compares with what a rule requires."""
    units = variable.attrs.get("units")
    if variable.dtype != np.float64:
        raise TypeError(f"external input {name!r} is {variable.dtype}; convert it to float64.")
    if not isinstance(units, str):
        raise ValueError(f"external input {name!r} has no units attribute; set one, '1' if dimensionless.")
    try:
        validate_units(units)
    except ValueError as error:
        raise ValueError(f"external input {name!r} has units pySIPNET refuses ({error}); correct them.") from None


def check_external_input_dim_is_site_or_a_batch_dim(dim: str, external_inputs: xr.Dataset) -> None:
    """An external input's dim is ``site`` or a batch dim with integer
    labels, since any other would be crossed with theta silently."""
    if dim == SITE:
        return
    labels = external_inputs.indexes.get(dim)
    if dim in NON_BATCH_DIM_NAMES or labels is None or labels.dtype.kind not in "iu":
        raise ValueError(
            f"external inputs are on {dim!r}, which is neither 'site' nor a batch dim with "
            "integer labels; select or reduce it first."
        )


def check_external_inputs_are_for_theta(external_inputs: xr.Dataset, n_rows: int | None, batch_dim: str) -> None:
    """External inputs on theta's batch dim are labeled ``0`` to ``J - 1``,
    one per row, with which they zip."""
    if batch_dim not in external_inputs.dims:
        return
    labels = external_inputs[batch_dim].values.tolist()
    if n_rows is None or labels != list(range(n_rows)):
        raise ValueError(
            f"external inputs on {batch_dim!r} are labeled {truncated(labels)}, but theta has "
            f"{'no rows' if n_rows is None else n_rows}; label them 0 to J - 1, one per row of "
            "theta, or name their dim otherwise to cross them with theta."
        )


def check_external_input_covers_the_sites(name: str, variable: xr.DataArray, sites: Sequence[int]) -> None:
    """An external input on ``site`` holds every site of the vector."""
    held = set(variable.indexes[SITE].tolist())
    missing = [site for site in sites if site not in held]
    if missing:
        raise KeyError(f"external input {name!r} has no value for site(s) {truncated(missing)}.")

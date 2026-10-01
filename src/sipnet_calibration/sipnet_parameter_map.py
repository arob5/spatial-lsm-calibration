"""The SIPNET parameter map: how the values at a site become SIPNET's
parameters at that site.

Where this sits
---------------
::

    labeled natural values (parameters, derived parameters)    the parameter layer's seam
    external inputs                                            propagated values, by name
      -> sipnet_parameter_map.SIPNETParameterMap (with a site_dims.SiteDims)
      -> SIPNET parameter fields (fields.SIPNETParameterFields)
      -> forward.ForwardModel, one SIPNET run each

Of the calibration's objects it alone reads pySIPNET's parameter specs,
which own every SIPNET parameter's name, units and domain. It knows nothing
of theta, priors, or which values are calibrated: it is a function of
labeled values, so the same map evaluates a posterior mean, a sensitivity
grid or BETY's medians as readily as a draw.

What it reads
-------------
One labeled Dataset of the values its rules read, by name: the parameters'
and derived parameters' labeled forms
(:mod:`sipnet_calibration.parameters.vector`'s data model) merged with any
:data:`ExternalInputs`, uncertain values that are propagated rather than
calibrated; and a :class:`~sipnet_calibration.site_dims.SiteDims`, which
reads them at the sites.

The map
-------
For the run at site :math:`s`, the SIPNET parameters are

.. math::

    \\psi_s = M\\big(v^{(s)}, c^{(s)}\\big),

with :math:`v^{(s)}` every value read at the site
(:meth:`SiteDims.at_sites`) and :math:`c^{(s)}` the rules' constants and
the :class:`Fixed` values there. :math:`M` is a list of
:class:`SIPNETRule`\\ s, applied in order. A rule reads values by name,
declaring what it requires of each (:class:`ValueRequirement`), reads
constants, and may read SIPNET parameters that are fixed or written by an
earlier rule. Each SIPNET parameter has one writer.

**The rule contract**: every SIPNET parameter a rule writes depends on
everything the rule reads, so a formula whose outputs read different inputs
is written as several rules (:func:`photosynthesis_rules`,
:func:`initial_condition_rules`). It makes
:meth:`SIPNETParameterMap.dependencies` exact, and with it which SIPNET
parameters a calibration varies
(:meth:`SIPNETParameterMap.sipnet_parameter_names_depending_on`).

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

**The result** is SIPNET parameter fields
(:data:`~sipnet_calibration.fields.SIPNETParameterFields`), one ``float64``
variable per SIPNET parameter written, in ``PARAMETER_SPECS`` order, each on
the batch dims of what it was computed from, in the order they first appear
among the values, then ``site``: a variable fed by theta's values is on
``(sample, site)``, one fed also by a crossed input on ``(sample, crossed
dim, site)``, a fixed value on ``(site,)``. Each carries pySIPNET's
``ParameterSpec.xarray_attributes()`` and ``set_by``.

**What is checked when.** Before anything runs,
:func:`check_sipnet_parameter_map_fits` checks that every value read exists,
in the units and shape its rule requires, and that every constant and fixed
value covers the labels the sites carry. Domains are checked on values
only, by :meth:`SIPNETParameterMap.out_of_domain`: each rule input against
its requirement's domain, each SIPNET parameter against pySIPNET's.

Functions and classes
---------------------
:class:`SIPNETParameterMap`
    ``sipnet_parameter_fields``, ``out_of_domain``, ``dependencies``,
    ``sipnet_parameter_names_depending_on``, ``describe``.
:class:`SIPNETRule`, :class:`ValueRequirement`, :func:`support_from_sipnet_domain`
    The rule protocol, what a rule requires of a value it reads, and
    pySIPNET's domains as supports.
:class:`Copy`, :class:`CopySimplex`, :class:`Compute`
    The rules; :func:`photosynthesis_rules` and :func:`initial_condition_rules`
    make the ``Compute`` rules of two derivations.
:class:`Fixed`
    A SIPNET parameter held at a value, shared or per label.
:data:`ExternalInputs`, :func:`validate_external_inputs`
    The alias and its validator.
:func:`check_sipnet_parameter_map_fits`
    The static check against the values' descriptions.
:class:`SIPNETParametersOutOfDomainError`
    Raised by the forward model for values outside their domains.

Notes
-----
**A rule is elementwise over leading dims and sites**: each output entry
depends only on the input entries at the same leading position and site.
That is what lets the map broadcast values on different batch dims and run
each rule once.

Usage
-----
::

    sipnet_map = SIPNETParameterMap(
        rules=[
            *photosynthesis_rules(capacity_value_name="photosynthetic_capacity",
                                  respiration_share_value_name="respiration_share"),
            CopySimplex(value_name="allocation", sipnet_parameter_names=(
                "leaf_allocation", "wood_allocation", "fine_root_allocation")),
            Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon"),
        ],
        fixed=[
            Fixed(sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=0.76,
                  provenance="..."),
            Fixed(sipnet_parameter_name="leaf_carbon_fraction", provenance="...",
                  value=pd.Series({"boreal.coniferous": 0.506, "temperate.deciduous": 0.466})
                  .rename_axis("pft").to_xarray()),
        ],
    )
    values = vector.flat_to_dataset(vector.to_natural(theta), batch_dims=("sample",))
    sipnet_parameter_fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    sipnet_map.out_of_domain(sipnet_parameter_fields, values, site_dims=site_dims)   # empty: in domain
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
    check_sipnet_parameter_name_is_a_flat_name,
)
from sipnet_calibration.initial_conditions.specs import resolve_initial_condition
from sipnet_calibration.parameters import (
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    UNIT_INTERVAL,
    DerivedParameter,
    Interval,
    Parameter,
    Simplex,
    Support,
)
from sipnet_calibration.site_dims import SiteDims
from sipnet_calibration.sites import site_coordinates
from sipnet_calibration.validation import as_frozen_mapping, as_names, truncated

__all__ = [
    "INITIAL_STATE_NAMES",
    "REQUIRED_SIPNET_PARAMETER_NAMES",
    "Compute",
    "Copy",
    "CopySimplex",
    "ExternalInputs",
    "Fixed",
    "SIPNETParameterMap",
    "SIPNETParametersOutOfDomainError",
    "SIPNETRule",
    "ValueRequirement",
    "check_sipnet_parameter_map_fits",
    "check_sipnet_parameter_map_is_valid",
    "initial_condition_rules",
    "photosynthesis_rules",
    "support_from_sipnet_domain",
    "validate_external_inputs",
]

Array = jax.Array


# ── the map ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False, kw_only=True)
class SIPNETParameterMap:
    """How values at a site become SIPNET parameters at that site: rules
    applied in order, and fixed values.

    Every SIPNET parameter a rule writes depends on everything the rule
    reads, the module's rule contract; a formula whose outputs read
    different inputs is written as several rules.

    Parameters
    ----------
    rules:
        The :class:`SIPNETRule`\\ s, in the order they run.
    fixed:
        The :class:`Fixed` SIPNET parameters.

    Raises
    ------
    KeyError
        For a SIPNET parameter name pySIPNET lacks.
    ValueError
        For a SIPNET parameter with two writers, a SIPNET parameter read
        before any rule or fixed value sets it, or a fixed value outside its
        domain.
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

    @property
    def unset_sipnet_parameter_names(self) -> tuple[str, ...]:
        """The :data:`REQUIRED_SIPNET_PARAMETER_NAMES` this map writes none of;
        a run's base parameter set supplies them."""
        written = set(self._writers)
        return tuple(name for name in REQUIRED_SIPNET_PARAMETER_NAMES if name not in written)

    def dependencies(self) -> frozendict:
        """For each SIPNET parameter written, the value names it depends on:
        its rule's ``values_read``, and, through each SIPNET parameter the
        rule reads, that parameter's dependencies. Empty for a fixed value,
        or a rule reading only constants and fixed values."""
        out: dict[str, frozenset[str]] = {fixed.sipnet_parameter_name: frozenset() for fixed in self.fixed}
        for rule in self.rules:
            names = set(rule.values_read)
            for read in rule.sipnet_parameter_names_read:
                names |= out[read]
            for written in rule.sipnet_parameter_names_written:
                out[written] = frozenset(names)
        return frozendict({name: out[name] for name in self.sipnet_parameter_names_written})

    def sipnet_parameter_names_depending_on(self, value_names: Sequence[str]) -> tuple[str, ...]:
        """The SIPNET parameters that depend on any of *value_names*, in
        ``PARAMETER_SPECS`` order: with a vector's parameter and derived
        parameter names, the SIPNET parameters a calibration varies.

        Raises
        ------
        TypeError
            If *value_names* is one string rather than a sequence.
        """
        wanted = set(as_names(value_names, message_name="value_names"))
        return tuple(name for name, depends in self.dependencies().items() if depends & wanted)

    def describe(self) -> pd.DataFrame:
        """One row per SIPNET parameter written, indexed by
        ``sipnet_parameter``: ``set_by``, ``values_read`` (its own rule's,
        comma-separated), ``depends_on`` (:meth:`dependencies`) and
        ``provenance`` (a fixed value's or a ``Compute`` rule's)."""
        dependencies = self.dependencies()
        rows = []
        for name in self.sipnet_parameter_names_written:
            writer = self._writers[name]
            is_fixed = isinstance(writer, Fixed)
            rows.append({
                "sipnet_parameter": name,
                "set_by": "fixed" if is_fixed else _set_by(writer),
                "values_read": "" if is_fixed else ", ".join(writer.values_read),
                "depends_on": ", ".join(sorted(dependencies[name])),
                "provenance": getattr(writer, "provenance", ""),
            })
        return pd.DataFrame(rows).set_index("sipnet_parameter")

    # ── evaluation ────────────────────────────────────────────────────────────

    def sipnet_parameter_fields(self, values: xr.Dataset, *, site_dims: SiteDims) -> SIPNETParameterFields:
        """Values to SIPNET parameter fields.

        Every value a rule reads, the rules' constants and the fixed values
        are read at the sites (:meth:`SiteDims.at_sites`); each rule then
        reads its inputs broadcast to the union of their batch dims (one
        name zips, two names cross) and writes each SIPNET parameter on that
        union. The values may lie outside their domains; :meth:`out_of_domain`
        says where.

        Parameters
        ----------
        values:
            Every value a rule reads, ``float64``, on any of: the dims of
            ``site_dims.coords``, batch dims (integer labels), and element
            axes (string labels). Variables no rule reads (a
            hyperparameter, say) are ignored.
        site_dims:
            The sites, which the fields are over.

        Returns
        -------
        SIPNETParameterFields
            As the module's docstring describes.

        Raises
        ------
        TypeError
            If *values* is not an ``xr.Dataset``.
        KeyError
            For a value a rule reads that *values* lacks, or a label a site
            carries that a value lacks.
        ValueError
            For a value whose element axes are not its requirement's shape.
        """
        check_values_are_a_dataset(values)
        read = self._values_at_sites(values, site_dims)
        order = _batch_dims_in_order(read.values())
        labeled = {
            **{name: _LabeledAtSites.of(variable, order) for name, variable in read.items()},
            **{name: _LabeledAtSites.of(variable, order) for name, variable in self._constants_at_sites(site_dims).items()},
        }
        written: dict[str, tuple[_LabeledAtSites, str]] = {
            fixed.sipnet_parameter_name: (_LabeledAtSites.of(fixed.at_sites(site_dims), order), "fixed")
            for fixed in self.fixed
        }
        for rule in self.rules:
            inputs = [labeled[n] for n in (*rule.values_read, *rule.constants)]
            inputs += [written[n][0] for n in rule.sipnet_parameter_names_read]
            dims = _union(order, inputs)
            sizes = {d: n for item in inputs for d, n in zip(item.batch_dims, item.batch_shape)}
            output = rule(
                {n: labeled[n].broadcast(dims, sizes) for n in (*rule.values_read, *rule.constants)},
                {n: written[n][0].broadcast(dims, sizes) for n in rule.sipnet_parameter_names_read},
            )
            labels = {d: item.labels[d] for item in inputs for d in item.batch_dims}
            for name, array in output.items():
                written[name] = (_LabeledAtSites(dims, np.asarray(array, dtype=np.float64), labels), _set_by(rule))
        return _sipnet_parameter_fields(written, site_dims)

    def out_of_domain(
        self, sipnet_parameter_fields: SIPNETParameterFields, values: xr.Dataset, *, site_dims: SiteDims
    ) -> pd.DataFrame:
        """Every value outside its domain: a rule input outside its
        requirement's domain, or a SIPNET parameter outside pySIPNET's.

        Parameters
        ----------
        sipnet_parameter_fields:
            The fields :meth:`sipnet_parameter_fields` made from *values*.
        values, site_dims:
            As :meth:`sipnet_parameter_fields` takes them.

        Returns
        -------
        pandas.DataFrame
            One row per value outside its domain (a non-finite one
            included), once for requirements alike: a column per batch dim and ``site`` (missing for a
            value not on it), then ``sipnet_parameter`` or ``value_name``
            (the other missing), and ``value`` (``NaN`` for a value of rank
            1 or more, whose element the row does not name). Empty when
            every value is in its domain.
        """
        rows = []
        for name, variable in sipnet_parameter_fields.data_vars.items():
            array = np.asarray(variable.values, dtype=np.float64)
            outside = ~_FLAT_SPECS[str(name)].domain.contains(array)
            rows += _rows_outside(variable, outside, array, {"sipnet_parameter": str(name)})
        read = self._values_at_sites(values, site_dims)
        for name, requirements in self.values_read.items():
            for requirement in dict.fromkeys(requirements):
                if requirement.domain is not None:
                    rows += _rows_outside_the_requirement(name, read[name], requirement)
        dims = list(dict.fromkeys([d for row in rows for d in row if d not in _REPORT_COLUMNS]))
        return pd.DataFrame(rows, columns=[*dims, *_REPORT_COLUMNS])

    # ── supporting methods ────────────────────────────────────────────────────

    @property
    def _writers(self) -> dict[str, SIPNETRule | Fixed]:
        writers: dict[str, SIPNETRule | Fixed] = {f.sipnet_parameter_name: f for f in self.fixed}
        for rule in self.rules:
            writers.update(dict.fromkeys(rule.sipnet_parameter_names_written, rule))
        return writers

    def _values_at_sites(self, values: xr.Dataset, site_dims: SiteDims) -> dict[str, xr.DataArray]:
        """Every value a rule reads, at the sites, in the order of *values*,
        checked for its shape."""
        check_values_hold_what_the_rules_read(list(self.values_read), values)
        names = [str(name) for name in values.data_vars if name in self.values_read]
        at_sites = site_dims.at_sites(values[names])
        for name, requirements in self.values_read.items():
            variable = at_sites[name]
            for requirement in requirements:
                check_value_has_the_required_shape(name, variable, requirement.shape)
        return {name: at_sites[name] for name in names}

    def _constants_at_sites(self, site_dims: SiteDims) -> dict[str, xr.DataArray]:
        """Every rule's constants, at the sites."""
        constants = {name: constant for rule in self.rules for name, constant in rule.constants.items()}
        if not constants:
            return {}
        at_sites = site_dims.at_sites(xr.Dataset({name: c.rename(None) for name, c in constants.items()}))
        return {name: at_sites[name] for name in constants}


# ── what a rule requires, and fixed values ────────────────────────────────────


@dataclass(frozen=True)
class ValueRequirement:
    """What a rule requires of a value it reads.

    Parameters
    ----------
    units:
        The units the rule's arithmetic assumes. The value's units must
        convert to them by a factor of exactly 1
        (``pysipnet.units.conversion_factor``), so ``"g/m2"`` meets
        ``"g m-2"`` and ``"kg m-2"`` does not: the rule never converts.
        ``None`` means no physical units, read as ``"1"``. Checked before
        anything runs.
    domain:
        The set the rule's formula is defined on for this value, checked on
        the values: at the corners of theta when the forward model is
        built, and at every evaluation (:meth:`SIPNETParameterMap.out_of_domain`).
        ``None``: no requirement.
    shape:
        One site's value's shape: ``()`` for a scalar, ``(k,)`` for a
        vector. Checked before anything runs.
    """

    units: str | None
    domain: Support | None = None
    shape: tuple[int, ...] = ()


def support_from_sipnet_domain(domain: ParameterDomain) -> Support:
    """pySIPNET's domain as a support: ``REAL`` the real line, ``POSITIVE``
    :math:`(0, \\infty)`, ``NON_NEGATIVE`` :math:`[0, \\infty)`,
    ``UNIT_INTERVAL`` :math:`[0, 1]`, ``OPEN_UNIT_INTERVAL`` :math:`(0, 1)`.

    Raises
    ------
    KeyError
        For a domain pySIPNET added that this table lacks.
    """
    return _SIPNET_DOMAIN_SUPPORTS[domain]


@dataclass(frozen=True, eq=False, kw_only=True)
class Fixed:
    """A SIPNET parameter held at a value.

    Parameters
    ----------
    sipnet_parameter_name:
        pySIPNET's flat name.
    value:
        A number; or a numeric ``xr.DataArray`` keyed by label, on ``site``
        or a site-labels dim (``pd.Series({...}).rename_axis("pft").to_xarray()``),
        covering the labels the sites carry (a superset is allowed).
    provenance:
        Where the value came from.

    Raises
    ------
    TypeError
        If *value* is neither a number nor a numeric DataArray.
    ValueError
        If *provenance* is empty.
    """

    sipnet_parameter_name: str
    value: float | xr.DataArray
    provenance: str

    def __post_init__(self) -> None:
        check_fixed_is_valid(self)
        if isinstance(self.value, xr.DataArray):
            value = self.value.astype(np.float64).copy(deep=True)
            value.values.setflags(write=False)
            object.__setattr__(self, "value", value)

    def values(self) -> np.ndarray:
        """Every value, flat, ``float64``."""
        if isinstance(self.value, xr.DataArray):
            return np.asarray(self.value.values, dtype=np.float64).ravel()
        return np.asarray([float(self.value)])

    def at_sites(self, site_dims: SiteDims) -> xr.DataArray:
        """The value at every site, on ``site``.

        Raises
        ------
        KeyError
            If a per-label value lacks a label some site carries.
        """
        value = self.value if isinstance(self.value, xr.DataArray) else xr.DataArray(float(self.value))
        return site_dims.at_sites(xr.Dataset({self.sipnet_parameter_name: value.rename(None)}))[self.sipnet_parameter_name]


# ── rules ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class SIPNETRule(Protocol):
    """A rule of the map: values read by name to SIPNET parameters.

    Attributes
    ----------
    values_read:
        ``{name: ValueRequirement}``: the values it reads (parameters,
        derived parameters, external inputs), each with what it requires.
    constants:
        ``{name: xr.DataArray}``: constants it reads, which the map reads at
        the sites and passes beside the values; empty when none.
    sipnet_parameter_names_read:
        SIPNET parameters it reads, each fixed or written by an earlier rule.
    sipnet_parameter_names_written:
        SIPNET parameters it writes, each depending on everything it reads.

    The call takes ``values`` (``{name: (..., S, *shape)}`` for each value
    and constant) and ``sipnet_parameter_values`` (``{name: (..., S)}``), all
    of one leading shape, and returns ``{SIPNET parameter written: (...,
    S)}``. It is elementwise over the leading axes and sites, and traceable
    by JAX.
    """

    values_read: Mapping[str, ValueRequirement]
    constants: Mapping[str, xr.DataArray]
    sipnet_parameter_names_read: tuple[str, ...]
    sipnet_parameter_names_written: tuple[str, ...]

    def __call__(self, values: Mapping[str, Array], sipnet_parameter_values: Mapping[str, Array]) -> dict[str, Array]: ...


@dataclass(frozen=True, kw_only=True)
class Copy:
    """The identity, :math:`\\psi = x`: one scalar value to one SIPNET
    parameter, which requires the value in its units and domain."""

    value_name: str
    sipnet_parameter_name: str
    constants: ClassVar[Mapping[str, xr.DataArray]] = frozendict()
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        return (self.sipnet_parameter_name,)

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        spec = _spec(self.sipnet_parameter_name)
        return {self.value_name: ValueRequirement(spec.units, support_from_sipnet_domain(spec.domain))}

    def __call__(self, values, sipnet_parameter_values) -> dict[str, Array]:
        return {self.sipnet_parameter_name: values[self.value_name]}


@dataclass(frozen=True, kw_only=True)
class CopySimplex:
    """A point :math:`x` of the ``k``-simplex to ``k - 1`` SIPNET
    parameters: :math:`\\psi_i = x_i` for :math:`i < k`.

    The last number is not written: SIPNET recomputes it as
    :math:`1 - \\sum_{i<k} x_i` (``sipnet.c:1113-1115``) and exits if that
    is negative (``sipnet.c:1117-1122``). Reading the whole simplex is what
    keeps every draw inside it; the requirement's domain is the closed
    simplex. Each SIPNET parameter written depends on the one value read.
    """

    value_name: str
    sipnet_parameter_names: tuple[str, ...]
    constants: ClassVar[Mapping[str, xr.DataArray]] = frozendict()
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        return tuple(self.sipnet_parameter_names)

    @property
    def values_read(self) -> Mapping[str, ValueRequirement]:
        return {self.value_name: ValueRequirement("1", Simplex(closed=True), (len(self.sipnet_parameter_names) + 1,))}

    def __call__(self, values, sipnet_parameter_values) -> dict[str, Array]:
        simplex = values[self.value_name]
        return {name: simplex[..., i] for i, name in enumerate(self.sipnet_parameter_names)}


@dataclass(frozen=True, eq=False, kw_only=True)
class Compute:
    """One SIPNET parameter as a formula of values, constants and SIPNET
    parameters: :math:`\\psi = f(v_1, \\dots, v_m, c_1, \\dots, \\phi_1, \\dots)`.

    Parameters
    ----------
    sipnet_parameter_name:
        pySIPNET's flat name of the SIPNET parameter written.
    function:
        Called as ``function(**values, **constants, **sipnet_parameter_values)``,
        keywords named as below. Each array has one leading shape ``(...,
        S)``, with a value's shape after it. The result is ``(..., S)``,
        **in the SIPNET parameter's units** (``ParameterSpec.units``): the
        map cannot check the arithmetic, but requiring the inputs' units
        makes the inputs checkable, and the domain checks check the result.
        Elementwise over the leading axes and sites; traceable by JAX.
    values_read:
        ``{name: ValueRequirement}``: the values it reads, each with what
        the function assumes.
    constants:
        ``{name: xr.DataArray}``: constants it reads, a scalar or keyed by
        label on ``site`` or a site-labels dim; a boolean one stays boolean.
    sipnet_parameter_names_read:
        SIPNET parameters it reads, each fixed or written by an earlier rule.
    provenance:
        Where the formula comes from: a line of ``sipnet.c``, a paper.

    Raises
    ------
    TypeError
        If *function* is not callable, or a constant is not a DataArray.
    ValueError
        If one keyword names two things (a value and a SIPNET parameter of
        one name, as ``soil_respiration_q10`` is both: rename in a wrapper),
        or *provenance* is empty.
    """

    sipnet_parameter_name: str
    function: Callable[..., Array]
    values_read: Mapping[str, ValueRequirement]
    constants: Mapping[str, xr.DataArray] = field(default_factory=frozendict)
    sipnet_parameter_names_read: tuple[str, ...] = ()
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values_read", as_frozen_mapping(self.values_read, message_name="values_read"))
        object.__setattr__(self, "constants", as_frozen_mapping(self.constants, message_name="constants"))
        object.__setattr__(
            self,
            "sipnet_parameter_names_read",
            as_names(self.sipnet_parameter_names_read, message_name="sipnet_parameter_names_read"),
        )
        check_compute_is_valid(self)

    @property
    def sipnet_parameter_names_written(self) -> tuple[str, ...]:
        return (self.sipnet_parameter_name,)

    def __call__(self, values, sipnet_parameter_values) -> dict[str, Array]:
        arguments = {n: values[n] for n in (*self.values_read, *self.constants)}
        arguments |= {n: sipnet_parameter_values[n] for n in self.sipnet_parameter_names_read}
        return {self.sipnet_parameter_name: jnp.asarray(self.function(**arguments), dtype=jnp.float64)}


def photosynthesis_rules(*, capacity_value_name: str, respiration_share_value_name: str) -> list[Compute]:
    """SIPNET's photosynthesis pair from its identifiable combination, as
    two ``Compute`` rules.

    ``aMax`` (:math:`A`), ``aMaxFrac`` (:math:`f`), ``baseFolRespFrac``
    (:math:`r`) and ``cFracLeaf`` (:math:`c`) enter SIPNET only through
    :math:`P = A (f + r) / c` and :math:`\\rho = r / (f + r)`
    (``sipnet.c:614, 617, 633``). With :math:`f` and :math:`c` read as
    SIPNET parameters, the rules invert them:

    .. math::

        A = \\frac{P c (1 - \\rho)}{f}, \\qquad r = \\frac{\\rho f}{1 - \\rho},

    from the capacity :math:`P` (``nmol g-1 s-1``, per gram of leaf carbon)
    and the respiration share :math:`\\rho` (``1``). They are two rules
    since :math:`r` reads neither :math:`P` nor :math:`c`.

    Returns
    -------
    list of Compute
        ``max_photosynthesis_rate``, then ``foliar_respiration_fraction``.
    """
    capacity = ValueRequirement(_spec("max_photosynthesis_rate").units, POSITIVE)
    share = ValueRequirement("1", OPEN_UNIT_INTERVAL)
    provenance = "P = aMax (aMaxFrac + baseFolRespFrac) / cFracLeaf, rho = baseFolRespFrac / (aMaxFrac + baseFolRespFrac): sipnet.c:614, 617, 633"
    return [
        Compute(
            sipnet_parameter_name="max_photosynthesis_rate",
            values_read={capacity_value_name: capacity, respiration_share_value_name: share},
            sipnet_parameter_names_read=("daily_mean_photosynthesis_fraction", "leaf_carbon_fraction"),
            function=_MaxPhotosynthesisRate(capacity_value_name, respiration_share_value_name),
            provenance=f"aMax = P cFracLeaf (1 - rho) / aMaxFrac, from {provenance}",
        ),
        Compute(
            sipnet_parameter_name="foliar_respiration_fraction",
            values_read={respiration_share_value_name: share},
            sipnet_parameter_names_read=("daily_mean_photosynthesis_fraction",),
            function=_FoliarRespirationFraction(respiration_share_value_name),
            provenance=f"baseFolRespFrac = rho aMaxFrac / (1 - rho), from {provenance}",
        ),
    ]


#: The initial state :func:`initial_condition_rules` reads, in order: soil
#: organic carbon, wood carbon and leaf carbon (``kg m-2`` of carbon), and
#: surface soil moisture (percent of saturation), in the initial conditions'
#: processed file's names and units.
INITIAL_STATE_NAMES: tuple[str, ...] = (
    "initial_soil_organic_carbon",
    "initial_wood_carbon",
    "initial_leaf_carbon",
    "initial_soil_moisture_saturation",
)


def initial_condition_rules(
    *,
    deciduous: Mapping[str, bool] | xr.DataArray,
    deciduous_dim: str = "pft",
    state_value_names: Sequence[str] = INITIAL_STATE_NAMES,
) -> list[Compute]:
    """SIPNET's initial state from a site's state values, as four
    ``Compute`` rules.

    With soil organic carbon :math:`C_s`, wood carbon :math:`C_w`, leaf
    carbon :math:`C_l` (``kg m-2``), soil moisture :math:`m` (percent of
    saturation), and the SIPNET parameters ``fine_root_fraction`` :math:`f`,
    ``coarse_root_fraction`` :math:`g` and ``leaf_carbon_per_area``
    :math:`\\lambda` (``g m-2``):

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
        Whether each class of *deciduous_dim* is deciduous: ``{label:
        bool}``, or a boolean DataArray on *deciduous_dim*; a superset of
        the classes is allowed.
    deciduous_dim:
        The site-labels name a site's deciduousness is read from.
    state_value_names:
        The names the four state values are read by, in the order of
        :data:`INITIAL_STATE_NAMES`.

    Returns
    -------
    list of Compute
        ``soil_carbon``, ``total_wood_carbon``, ``leaf_area_index``,
        ``soil_wetness_fraction``.

    Raises
    ------
    TypeError
        If a deciduousness is not a boolean: ``NaN`` or ``2.0`` cast to a
        boolean is ``True``.
    ValueError
        If *state_value_names* does not name four values.
    """
    if isinstance(deciduous, xr.DataArray):
        check_deciduous_values_are_booleans(dict(zip(deciduous[deciduous.dims[0]].values.tolist(), deciduous.values.tolist())))
        mask = deciduous.copy()
    else:
        deciduous = as_frozen_mapping(deciduous, message_name="deciduous")
        check_deciduous_values_are_booleans(deciduous)
        mask = xr.DataArray(np.asarray(list(deciduous.values()), dtype=bool), dims=deciduous_dim,
                            coords={deciduous_dim: list(deciduous)})
    state_value_names = as_names(state_value_names, message_name="state_value_names")
    check_state_value_names_are_four(state_value_names)
    soil, wood, leaf, moisture = state_value_names
    carbon = {state: ValueRequirement(resolve_initial_condition(state).units, NON_NEGATIVE) for state in INITIAL_STATE_NAMES[:3]}
    saturation = ValueRequirement(
        resolve_initial_condition("initial_soil_moisture_saturation").units,
        Interval(0.0, 100.0, low_closed=True, high_closed=True),
    )
    source = "to_sipnet_initial_conditions (sipnet_calibration.initial_conditions)"
    return [
        Compute(sipnet_parameter_name="soil_carbon", values_read={soil: carbon["initial_soil_organic_carbon"]},
                function=_Scaled(soil, _GRAMS_PER_KILOGRAM), provenance=f"1000 C_s, {source}"),
        Compute(sipnet_parameter_name="total_wood_carbon", values_read={wood: carbon["initial_wood_carbon"]},
                sipnet_parameter_names_read=("fine_root_fraction", "coarse_root_fraction"),
                function=_TotalWoodCarbon(wood), provenance=f"1000 C_w / (1 - fineRootFrac - coarseRootFrac), {source}"),
        Compute(sipnet_parameter_name="leaf_area_index", values_read={leaf: carbon["initial_leaf_carbon"]},
                constants={"deciduous": mask}, sipnet_parameter_names_read=("leaf_carbon_per_area",),
                function=_LeafAreaIndex(leaf),
                provenance=f"1000 C_l / leafCSpWt, 0 at a deciduous site, which starts leafless; {source}"),
        Compute(sipnet_parameter_name="soil_wetness_fraction", values_read={moisture: saturation},
                function=_Scaled(moisture, 1.0 / _PERCENT), provenance=f"m / 100, {source}"),
    ]


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
        If it is not an ``xr.Dataset``, or a variable is not ``float64``.
    ValueError
        If a variable lacks valid ``units``, or a dim is neither ``site``
        nor a batch dim with integer labels.
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


class SIPNETParametersOutOfDomainError(ValueError):
    """Values outside their domains: a rule input outside its requirement's,
    or a SIPNET parameter outside pySIPNET's. The prior and the map put mass
    where a rule or SIPNET is undefined."""


# ── private helpers ───────────────────────────────────────────────────────────

_FLAT_SPECS: Mapping[str, ParameterSpec] = frozendict(
    {path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()}
)
_SPEC_ORDER: Mapping[str, int] = frozendict({name: i for i, name in enumerate(_FLAT_SPECS)})

_SIPNET_DOMAIN_SUPPORTS: Mapping[ParameterDomain, Support] = frozendict(
    {
        ParameterDomain.REAL: REAL,
        ParameterDomain.POSITIVE: POSITIVE,
        ParameterDomain.NON_NEGATIVE: NON_NEGATIVE,
        ParameterDomain.UNIT_INTERVAL: UNIT_INTERVAL,
        ParameterDomain.OPEN_UNIT_INTERVAL: OPEN_UNIT_INTERVAL,
    }
)

#: The columns of :meth:`SIPNETParameterMap.out_of_domain` after the dims.
_REPORT_COLUMNS = ("sipnet_parameter", "value_name", "value")

#: Grams in a kilogram, and percent in one.
_GRAMS_PER_KILOGRAM, _PERCENT = 1000.0, 100.0


@dataclass(frozen=True)
class _LabeledAtSites:
    """A value at the sites: an array on ``(*batch_dims, site, *element
    axes)``, the batch dims' labels, and nothing else of xarray, so a rule
    receives arrays."""

    batch_dims: tuple[str, ...]
    array: np.ndarray
    labels: Mapping[str, np.ndarray]

    @classmethod
    def of(cls, variable: xr.DataArray, order: Sequence[str]) -> _LabeledAtSites:
        batch = tuple(d for d in order if d in variable.dims)
        elements = [d for d in variable.dims if d not in batch and d != SITE]
        variable = variable.transpose(*batch, SITE, *elements)
        labels = {d: np.asarray(variable[d].values) if d in variable.indexes else np.arange(variable.sizes[d]) for d in batch}
        return cls(batch, np.asarray(variable.values), labels)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.array.shape[: len(self.batch_dims)]

    def broadcast(self, dims: tuple[str, ...], sizes: Mapping[str, int]) -> Array:
        """The array on *dims* then ``site`` and its element axes, *dims* a
        superset of its batch dims in the same order."""
        array = self.array
        for axis, dim in enumerate(dims):
            if dim not in self.batch_dims:
                array = np.expand_dims(array, axis)
        target = (*(sizes[d] for d in dims), *array.shape[len(dims):])
        return jnp.asarray(np.broadcast_to(array, target))


@dataclass(frozen=True)
class _MaxPhotosynthesisRate:
    capacity_value_name: str
    respiration_share_value_name: str

    def __call__(self, daily_mean_photosynthesis_fraction, leaf_carbon_fraction, **values) -> Array:
        capacity = values[self.capacity_value_name]
        share = values[self.respiration_share_value_name]
        return capacity * leaf_carbon_fraction * (1.0 - share) / daily_mean_photosynthesis_fraction


@dataclass(frozen=True)
class _FoliarRespirationFraction:
    respiration_share_value_name: str

    def __call__(self, daily_mean_photosynthesis_fraction, **values) -> Array:
        share = values[self.respiration_share_value_name]
        return share * daily_mean_photosynthesis_fraction / (1.0 - share)


@dataclass(frozen=True)
class _Scaled:
    value_name: str
    factor: float

    def __call__(self, **values) -> Array:
        return self.factor * values[self.value_name]


@dataclass(frozen=True)
class _TotalWoodCarbon:
    value_name: str

    def __call__(self, fine_root_fraction, coarse_root_fraction, **values) -> Array:
        return _GRAMS_PER_KILOGRAM * values[self.value_name] / (1.0 - fine_root_fraction - coarse_root_fraction)


@dataclass(frozen=True)
class _LeafAreaIndex:
    value_name: str

    def __call__(self, deciduous, leaf_carbon_per_area, **values) -> Array:
        return jnp.where(deciduous, 0.0, _GRAMS_PER_KILOGRAM * values[self.value_name] / leaf_carbon_per_area)


def _spec(sipnet_parameter_name: str) -> ParameterSpec:
    return _FLAT_SPECS[sipnet_parameter_name]


def _set_by(rule: Any) -> str:
    return f"rule {type(rule).__name__}"


def _batch_dims_in_order(variables: Sequence[xr.DataArray] | Any) -> tuple[str, ...]:
    """The batch dims of the values, in the order they first appear: every
    dim but ``site`` and the element axes (string labels)."""
    order: dict[str, None] = {}
    for variable in variables:
        for dim in variable.dims:
            index = variable.indexes.get(dim)
            is_element = index is not None and len(index) > 0 and all(isinstance(v, str) for v in index)
            if dim != SITE and not is_element:
                order[str(dim)] = None
    return tuple(order)


def _union(order: tuple[str, ...], inputs: Sequence[_LabeledAtSites]) -> tuple[str, ...]:
    """The batch dims of *inputs*, in *order*. Values of one Dataset share
    one index per dim, so a batch dim's labels are the same in every input."""
    present = {d for item in inputs for d in item.batch_dims}
    return tuple(d for d in order if d in present)


def _sipnet_parameter_fields(written: Mapping[str, tuple[_LabeledAtSites, str]], site_dims: SiteDims) -> xr.Dataset:
    """The SIPNET parameter fields, in ``PARAMETER_SPECS`` order."""
    coordinates: dict[str, Any] = {}
    variables = {}
    for name, (labeled, set_by) in sorted(written.items(), key=lambda kv: _SPEC_ORDER[kv[0]]):
        for dim in labeled.batch_dims:
            coordinates[dim] = batch_coordinate(dim, labeled.labels[dim])
        variables[name] = (
            (*labeled.batch_dims, SITE),
            np.asarray(labeled.array, dtype=np.float64),
            {**_FLAT_SPECS[name].xarray_attributes(), "set_by": set_by},
        )
    dataset = xr.Dataset(variables, coords=coordinates)
    return dataset.assign_coords(site_coordinates(site_dims.sites, site_dims.site_table))


def _rows_outside(
    variable: xr.DataArray, outside: np.ndarray, values: np.ndarray | None, identity: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One row per position of *variable* where *outside* holds, labeled by
    its dims."""
    rows = []
    labels = [variable[dim].values if dim in variable.coords else np.arange(variable.sizes[dim]) for dim in variable.dims]
    for position in np.argwhere(outside):
        row = {str(dim): labels[axis][i].item() for axis, (dim, i) in enumerate(zip(variable.dims, position))}
        value = np.nan if values is None else float(values[tuple(position)])
        rows.append({**row, "sipnet_parameter": None, "value_name": None, **identity, "value": value})
    return rows


def _rows_outside_the_requirement(name: str, variable: xr.DataArray, requirement: ValueRequirement) -> list[dict[str, Any]]:
    """One row per batch position and site where a value read lies outside
    its requirement's domain; a value of rank 1 or more is outside when any
    of its numbers, or its whole vector, is."""
    array = np.asarray(variable.values, dtype=np.float64)
    inside = np.asarray(requirement.domain.contains(array))
    leading = variable.ndim - len(requirement.shape)
    if inside.ndim > leading:
        inside = inside.all(axis=tuple(range(leading, inside.ndim)))
    elements = variable.dims[leading:]
    at_positions = variable.isel({d: 0 for d in elements}, drop=True) if elements else variable
    return _rows_outside(at_positions, ~inside, None if elements else array, {"value_name": name})


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
        check_rule_is_a_rule(rule)
        for name in rule.sipnet_parameter_names_read:
            check_sipnet_parameter_name_is_a_flat_name(name, f"{_set_by(rule)} reads, whose names")
            check_sipnet_parameter_read_is_set_earlier(name, rule, writers)
        for name in rule.sipnet_parameter_names_written:
            check_sipnet_parameter_name_is_a_flat_name(name, f"{_set_by(rule)} writes, whose names")
            check_sipnet_parameter_has_one_writer(name, _set_by(rule), writers)


def check_sipnet_parameter_map_fits(
    sipnet_parameter_map: SIPNETParameterMap,
    descriptions: Mapping[str, Parameter | DerivedParameter],
    external_inputs: ExternalInputs | None = None,
    site_dims: SiteDims | None = None,
) -> None:
    """The map reads what the parameters, derived parameters and external
    inputs hold, in the units and shapes its rules require, and its
    constants and fixed values cover the labels the sites carry."""
    external_names = () if external_inputs is None else tuple(map(str, external_inputs.data_vars))
    check_external_inputs_share_no_name_with_the_values(external_names, descriptions)
    for name, requirements in sipnet_parameter_map.values_read.items():
        check_value_is_held(name, descriptions, external_names)
        for requirement in requirements:
            if name in descriptions:
                check_description_meets_the_requirement(descriptions[name], requirement)
            else:
                check_external_input_meets_the_requirement(name, external_inputs[name], requirement)
    if site_dims is not None:
        for rule in sipnet_parameter_map.rules:
            for name, constant in rule.constants.items():
                check_labels_cover_the_sites(constant, site_dims, f"{_set_by(rule)} constant {name!r}")
        for fixed in sipnet_parameter_map.fixed:
            if isinstance(fixed.value, xr.DataArray):
                check_labels_cover_the_sites(fixed.value, site_dims, f"fixed {fixed.sipnet_parameter_name!r}")


def check_rule_is_a_rule(rule: Any) -> None:
    """A rule has what the map reads of one: what it reads, its constants,
    what it writes, and a call."""
    if not isinstance(rule, SIPNETRule):
        raise TypeError(
            f"{type(rule).__name__} is not a SIPNET rule: a rule has values_read, constants, "
            "sipnet_parameter_names_read, sipnet_parameter_names_written and a call."
        )


def check_compute_is_valid(compute: Compute) -> None:
    """A ``Compute`` rule has a function, reads each keyword once, reads
    values by requirement and constants as DataArrays, and says where its
    formula came from."""
    if not callable(compute.function):
        raise TypeError(f"Compute {compute.sipnet_parameter_name!r} has a function that is not callable.")
    for name, requirement in compute.values_read.items():
        if not isinstance(requirement, ValueRequirement):
            raise TypeError(f"Compute {compute.sipnet_parameter_name!r} reads {name!r} without a ValueRequirement.")
    for name, constant in compute.constants.items():
        if not isinstance(constant, xr.DataArray):
            raise TypeError(
                f"Compute {compute.sipnet_parameter_name!r} constant {name!r} is a {type(constant).__name__}; "
                "give a DataArray, such as pd.Series({...}).rename_axis('pft').to_xarray()."
            )
    keywords = [*compute.values_read, *compute.constants, *compute.sipnet_parameter_names_read]
    repeated = sorted({k for k in keywords if keywords.count(k) > 1})
    if repeated:
        raise ValueError(
            f"Compute {compute.sipnet_parameter_name!r} receives {repeated} as two things (a value, a "
            "constant or a SIPNET parameter); rename in a wrapper."
        )
    if not isinstance(compute.provenance, str) or not compute.provenance.strip():
        raise ValueError(f"Compute {compute.sipnet_parameter_name!r} has no provenance; say where the formula comes from.")


def check_fixed_is_valid(fixed: Fixed) -> None:
    """A fixed value is a number or a numeric DataArray, and says where it
    came from."""
    value = fixed.value
    if isinstance(value, xr.DataArray):
        if value.dtype.kind not in "iuf" or value.dtype == np.bool_:
            raise TypeError(f"fixed {fixed.sipnet_parameter_name!r} holds {value.dtype} values; give numbers.")
    elif isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"fixed {fixed.sipnet_parameter_name!r} holds {value!r}, which is neither a number nor a numeric "
            "DataArray keyed by label."
        )
    if not isinstance(fixed.provenance, str) or not fixed.provenance.strip():
        raise ValueError(f"fixed {fixed.sipnet_parameter_name!r} has no provenance; say where the value came from.")


def check_fixed_values_are_in_the_domain(fixed: Fixed) -> None:
    """Every fixed value lies in its SIPNET parameter's domain."""
    domain = _spec(fixed.sipnet_parameter_name).domain
    if not domain.contains(fixed.values()).all():
        raise ValueError(
            f"fixed {fixed.sipnet_parameter_name!r} holds a value outside pySIPNET's domain "
            f"{domain.value!r}; fix it inside the domain."
        )


def check_sipnet_parameter_has_one_writer(name: str, writer: str, writers: dict[str, str]) -> None:
    """Each SIPNET parameter is written once, by a rule or a fixed value."""
    if name in writers:
        raise ValueError(f"{name!r} is written by both {writers[name]} and {writer}; keep one writer.")
    writers[name] = writer


def check_sipnet_parameter_read_is_set_earlier(name: str, rule: Any, writers: Mapping[str, str]) -> None:
    """A SIPNET parameter a rule reads is fixed or written by an earlier rule."""
    if name not in writers:
        raise ValueError(
            f"{_set_by(rule)} reads {name!r}, which is neither fixed nor written by an earlier rule; fix "
            "it, or move the rule that writes it before this one."
        )


def check_values_are_a_dataset(values: Any) -> None:
    """Values are a labeled ``xr.Dataset``, read by name."""
    if not isinstance(values, xr.Dataset):
        raise TypeError(f"the map reads values from an xarray Dataset, got {type(values).__name__}.")


def check_values_hold_what_the_rules_read(names: Sequence[str], values: xr.Dataset) -> None:
    """The values hold every value a rule reads."""
    missing = [name for name in names if name not in values.data_vars]
    if missing:
        raise KeyError(
            f"the map reads {truncated(missing)}, which the values lack; merge in the parameters', derived "
            "parameters' and external inputs' labeled values."
        )


def check_value_has_the_required_shape(name: str, variable: xr.DataArray, shape: tuple[int, ...]) -> None:
    """A value's element axes at a site have the shape its rule requires,
    which a rule would otherwise read across the wrong axes."""
    elements = tuple(variable.sizes[d] for d in variable.dims[variable.dims.index(SITE) + 1:])
    if elements != tuple(shape):
        raise ValueError(
            f"the value {name!r} has element axes of shape {elements} at a site, but the rule reading it "
            f"requires {tuple(shape)}."
        )


def check_external_inputs_share_no_name_with_the_values(
    external_names: Sequence[str], descriptions: Mapping[str, Any]
) -> None:
    """No external input is named like a parameter or derived parameter,
    since values are read by name."""
    shared = [name for name in external_names if name in descriptions]
    if shared:
        raise ValueError(
            f"the external inputs {truncated(shared)} are named like parameters or derived parameters; "
            "values are read by name, so rename them."
        )


def check_value_is_held(name: str, descriptions: Mapping[str, Any], external_names: Sequence[str]) -> None:
    """A value a rule reads is a parameter, a derived parameter or an
    external input."""
    if name not in descriptions and name not in external_names:
        raise KeyError(
            f"the map reads {name!r}, which is neither a parameter, a derived parameter nor an external "
            "input; add it to one of them, or read a value that exists."
        )


def check_description_meets_the_requirement(
    description: Parameter | DerivedParameter, requirement: ValueRequirement
) -> None:
    """A parameter or derived parameter read by a rule is in the units and
    shape the rule requires."""
    if not _units_match(description.units, requirement.units):
        raise ValueError(
            f"{description.name!r} is in {description.units!r}, but the rule reading it requires "
            f"{requirement.units!r}; give it those units."
        )
    if tuple(description.shape) != tuple(requirement.shape):
        raise ValueError(
            f"{description.name!r} has shape {description.shape}, but the rule reading it requires "
            f"{tuple(requirement.shape)}."
        )


def check_external_input_meets_the_requirement(name: str, variable: xr.DataArray, requirement: ValueRequirement) -> None:
    """An external input read by a rule is in the units the rule requires,
    and is one number per site, as the rule requires."""
    if requirement.shape != ():
        raise ValueError(
            f"external input {name!r} is read by a rule requiring a value of shape {requirement.shape} per "
            "site, and an external input holds one number."
        )
    if not _units_match(variable.attrs.get("units"), requirement.units):
        raise ValueError(
            f"external input {name!r} is in {variable.attrs.get('units')!r}, but the rule reading it "
            f"requires {requirement.units!r}; convert it first."
        )


def check_labels_cover_the_sites(array: xr.DataArray, site_dims: SiteDims, what: str) -> None:
    """A constant or fixed value is on dims of the site dims, labeled, with
    a value at every label the sites carry."""
    for dim in array.dims:
        if dim not in site_dims.coords:
            raise ValueError(f"{what} is on {dim!r}, which is not a dim of the site dims {list(site_dims.coords)}.")
        if dim not in array.indexes:
            raise ValueError(f"{what} has no {dim!r} coordinate; give {dim!r} its labels.")
        missing = [label for label in site_dims.coords[dim] if label not in set(array.indexes[dim].tolist())]
        if missing:
            raise KeyError(f"{what} has no value for {dim} label(s) {truncated(missing)} some site carries.")


def check_deciduous_values_are_booleans(deciduous: Mapping[Any, Any]) -> None:
    """Deciduousness is a boolean per class: a number cast to bool would
    make every non-zero value, NaN included, deciduous and zero its leaves."""
    bad = [key for key, value in deciduous.items() if not isinstance(value, (bool, np.bool_))]
    if bad:
        raise TypeError(f"deciduous values must be booleans; {truncated(bad)} are not.")


def check_state_value_names_are_four(state_value_names: Sequence[str]) -> None:
    """The initial state is read by four names, one per state value."""
    if len(state_value_names) != len(INITIAL_STATE_NAMES):
        raise ValueError(
            f"state_value_names names {len(state_value_names)} values; name the four of "
            f"{INITIAL_STATE_NAMES}, in that order."
        )


def check_external_inputs_are_a_dataset(external_inputs: Any) -> None:
    """External inputs are an ``xr.Dataset``, which is how they are read."""
    if not isinstance(external_inputs, xr.Dataset):
        raise TypeError(f"external inputs are an xarray Dataset, got {type(external_inputs).__name__}; pass one.")


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
            f"external inputs are on {dim!r}, which is neither 'site' nor a batch dim with integer labels; "
            "select or reduce it first."
        )

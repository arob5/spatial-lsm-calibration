"""What each net ecosystem exchange product is.

One :class:`NetEcosystemExchangeSpec` per product, and a product is one
**series**: a single estimate of net ecosystem exchange, from a single source,
at a single resolution. The spec says which source and raw file it comes from,
which columns carry its value, quality flag and uncertainties, and what the
quantity is. The processed netCDF stores a spec's fields as its variables'
attributes, so the file describes itself, and its reader checks it against the
same spec.

Contents
--------
:class:`NetEcosystemExchangeSpec`
    The record, one per product.
:data:`NET_ECOSYSTEM_EXCHANGE`, :data:`NET_ECOSYSTEM_EXCHANGE_NAMES`
    The registry and its product names, in order.
:data:`SOURCES`
    The sources a spec may name.
:func:`resolve_net_ecosystem_exchange`
    A spec by its product name.
:func:`describe`
    One spec as a paragraph, for a run log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from pysipnet.units import validate_units
from pysipnet.variables import VariableKind

from sipnet_calibration.net_ecosystem_exchange.names import resolve_resolution
from sipnet_calibration.net_ecosystem_exchange.source_files import SOURCE

__all__ = [
    "NAME_PATTERN",
    "NET_ECOSYSTEM_EXCHANGE",
    "NET_ECOSYSTEM_EXCHANGE_NAMES",
    "NetEcosystemExchangeSpec",
    "SOURCES",
    "describe",
    "resolve_net_ecosystem_exchange",
]


#: What a product name must look like: lower case words joined by underscores.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

#: The sources a spec may name, each with the source-file columns it can read.
SOURCES: dict[str, frozenset[str]] = {"ameriflux": frozenset(SOURCE.names)}


@dataclass(frozen=True)
class NetEcosystemExchangeSpec:
    """Everything the rest of the project needs to know about one NEE product.

    :meth:`xarray_attributes` is what the product stores on its ``value``;
    the ``*_attributes`` methods give its companions'.
    """

    name: str
    """Product name: the processed file's stem and the registry key; the raw
    file's stem followed by the series."""

    source: str
    """The source the product is read from, a key of :data:`SOURCES`."""

    resolution: str
    """``"half_hourly"`` or ``"hourly"``."""

    value_column: str
    """The source column carrying the value."""

    quality_column: str
    """The source column carrying its quality flag, or ``""``."""

    random_uncertainty_column: str
    """The source column carrying its random uncertainty, or ``""``."""

    joint_uncertainty_column: str
    """The source column carrying its joint uncertainty, or ``""``."""

    long_label: str
    """Plot-ready name without units."""

    description: str
    """What the series is and how the producer made it, with the citation."""

    product: str
    """The upstream data product."""

    sign_convention: str
    """Which direction is positive, and how that is known."""

    time_reference: str
    """What a value's time label means, in words."""

    units_provenance: str
    """Where the unit comes from and how firm it is."""

    units: str = "umol m-2 s-1"
    """UDUNITS unit string, physical only, validated by :mod:`pysipnet.units`."""

    constituent: str = "CO2"
    """Substance the unit refers to."""

    kind: VariableKind = VariableKind.TIMESTEP_MEAN
    """pySIPNET's kind: a mean rate over each step, so aggregation averages."""

    comment: str = ""
    """Anything else a reader must know; the CF ``comment`` attribute."""

    def __post_init__(self) -> None:
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"Name {self.name!r} is not lower_case_with_underscores.")
        validate_units(self.units)
        if self.source not in SOURCES:
            raise ValueError(f"{self.name!r}: source {self.source!r} is not one of {sorted(SOURCES)}")
        resolve_resolution(self.resolution)
        stem = self.raw_file.removesuffix(".nc")
        if not self.name.startswith(stem + "_"):
            raise ValueError(
                f"{self.name!r} does not start with its raw file's stem {stem!r} and a series"
            )
        for text in (self.long_label, self.description, self.product, self.sign_convention):
            if not text:
                raise ValueError(f"{self.name!r} needs a long_label, description, product and sign_convention.")
        if not self.time_reference or not self.units_provenance:
            raise ValueError(f"{self.name!r} needs time_reference and units_provenance.")
        readable = SOURCES[self.source]
        for column in self.source_columns:
            if column not in readable:
                raise ValueError(f"{self.name!r}: {column!r} is not a column the {self.source} source carries")

    @property
    def raw_file(self) -> str:
        """The raw file the product is built from."""
        return resolve_resolution(self.resolution).raw_file

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Every source column the product reads."""
        return tuple(
            column
            for column in (
                self.value_column,
                self.quality_column,
                self.random_uncertainty_column,
                self.joint_uncertainty_column,
            )
            if column
        )

    def xarray_attributes(self) -> dict[str, Any]:
        """Attributes for the product's ``value``.

        Keys follow the Climate and Forecast conventions where one exists
        (``units``, ``long_name``, ``comment``); the rest are spelled out.
        """
        attrs: dict[str, Any] = {
            "units": self.units,
            "long_name": self.long_label,
            "description": self.description,
            "product": self.product,
            "source_file": self.raw_file,
            "source_column": self.value_column,
            "kind": self.kind.value,
            "constituent": self.constituent,
            "sign_convention": self.sign_convention,
            "time_reference": self.time_reference,
            "units_provenance": self.units_provenance,
        }
        if self.comment:
            attrs["comment"] = self.comment
        return attrs

    def quality_flag_attributes(self) -> dict[str, Any]:
        """Attributes for the product's ``quality_flag``."""
        column = SOURCE.columns[self.quality_column]
        return {
            "long_name": f"{self.long_label}: quality flag",
            "flag_values": np.array(column.flag_values, dtype=np.int8),
            "flag_meanings": column.flag_meanings,
            "source_file": self.raw_file,
            "source_column": self.quality_column,
            "comment": "-1 where the source reports no flag.",
        }

    def random_uncertainty_attributes(self) -> dict[str, Any]:
        """Attributes for the product's ``random_uncertainty``."""
        return self._uncertainty_attributes(
            self.random_uncertainty_column,
            "random uncertainty",
            "ONEFlux's random uncertainty of the half-hour, estimated from measured data "
            "only; defined for measured steps.",
        )

    def joint_uncertainty_attributes(self) -> dict[str, Any]:
        """Attributes for the product's ``joint_uncertainty``."""
        return self._uncertainty_attributes(
            self.joint_uncertainty_column,
            "joint uncertainty",
            "ONEFlux's joint uncertainty: the random uncertainty combined with the u* "
            "filtering uncertainty, sqrt(RANDUNC^2 + ((NEE_84 - NEE_16) / 2)^2) over the "
            "u* threshold ensemble.",
        )

    def _uncertainty_attributes(self, column: str, what: str, description: str) -> dict[str, Any]:
        return {
            "units": self.units,
            "long_name": f"{self.long_label}: {what}",
            "description": description + f" From {SOURCE.documentation}.",
            "constituent": self.constituent,
            "source_file": self.raw_file,
            "source_column": column,
        }


_PRODUCT = (
    "AmeriFlux FLUXNET (ONEFlux) FULLSET, one CC-BY-4.0 dataset per site, each with its own "
    "DOI (the doi coordinate)"
)

_SIGN = (
    "positive is a flux from the ecosystem to the atmosphere, the micrometeorological "
    "convention. The FLUXNET2015 variable table does not state it; the values are positive "
    "at night and negative in summer daytime, so it is inferred from the data."
)

_TIME_REFERENCE = (
    "mean rate over the step [time_step_start, time], UTC; the source's local-standard-time "
    "stamps shifted by the tower's utc_offset"
)

_UNITS_PROVENANCE = (
    "The FLUXNET2015 FULLSET variable table gives umolCO2 m-2 s-1 for half-hourly and hourly "
    f"NEE ({SOURCE.documentation}); the files are ONEFlux output in that format, which is "
    "the documented source of the unit."
)

_VARIABLE = (
    "Net ecosystem exchange from ONEFlux, filtered with a friction velocity (u*) threshold "
    "estimated separately for each year (VUT), the reference estimate among ONEFlux's "
    "u*-threshold ensemble, chosen by model efficiency (NEE_VUT_REF). Steps that were not "
    "measured are gap-filled by marginal distribution sampling; quality_flag says which: 0 "
    "is measured, 1-3 good, medium and poor gap-fill."
)

_CONSTANT = (
    "Net ecosystem exchange from ONEFlux, filtered with one friction velocity (u*) threshold "
    "for all years (CUT), the 50th percentile of the threshold distribution "
    "(NEE_CUT_USTAR50). Steps that were not measured are gap-filled by marginal distribution "
    "sampling; quality_flag says which. Towers whose source file carries no CUT columns are "
    "absent from this product."
)


def _spec(resolution: str, series: str, estimate: str, label: str, description: str) -> NetEcosystemExchangeSpec:
    return NetEcosystemExchangeSpec(
        name=f"ameriflux_nee_{resolution}_{series}",
        source="ameriflux",
        resolution=resolution,
        value_column=f"NEE_{estimate}",
        quality_column=f"NEE_{estimate}_QC",
        random_uncertainty_column=f"NEE_{estimate}_RANDUNC",
        joint_uncertainty_column=f"NEE_{estimate}_JOINTUNC",
        long_label=label,
        description=description,
        product=_PRODUCT,
        sign_convention=_SIGN,
        time_reference=_TIME_REFERENCE,
        units_provenance=_UNITS_PROVENANCE,
    )


NET_ECOSYSTEM_EXCHANGE: tuple[NetEcosystemExchangeSpec, ...] = (
    _spec("half_hourly", "ustar_variable", "VUT_REF", "Net ecosystem exchange (variable u* threshold)", _VARIABLE),
    _spec("half_hourly", "ustar_constant", "CUT_USTAR50", "Net ecosystem exchange (constant u* threshold)", _CONSTANT),
    _spec("hourly", "ustar_variable", "VUT_REF", "Net ecosystem exchange (variable u* threshold)", _VARIABLE),
    _spec("hourly", "ustar_constant", "CUT_USTAR50", "Net ecosystem exchange (constant u* threshold)", _CONSTANT),
)

#: The product names, in registry order.
NET_ECOSYSTEM_EXCHANGE_NAMES: tuple[str, ...] = tuple(spec.name for spec in NET_ECOSYSTEM_EXCHANGE)


def resolve_net_ecosystem_exchange(name: str) -> NetEcosystemExchangeSpec:
    """The spec named *name*, or a ``KeyError`` listing the names that exist."""
    for spec in NET_ECOSYSTEM_EXCHANGE:
        if spec.name == name:
            return spec
    raise KeyError(
        f"No net ecosystem exchange product named {name!r}. Known: {list(NET_ECOSYSTEM_EXCHANGE_NAMES)}"
    )


def describe(spec: NetEcosystemExchangeSpec) -> str:
    """A spec as a paragraph, for ``--describe`` and the run log."""
    return "\n".join(
        [
            f"{spec.name}: {spec.long_label} ({spec.units} {spec.constituent}, {spec.kind.value}), "
            f"from {spec.product}.",
            f"  source     {spec.source}, {spec.raw_file}, columns {list(spec.source_columns)}",
            f"  what       {spec.description}",
            f"  time       {spec.time_reference}",
            f"  sign       {spec.sign_convention}",
            f"  units      {spec.units_provenance}",
        ]
    )

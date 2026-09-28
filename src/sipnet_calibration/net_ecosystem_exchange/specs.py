"""What each observed net ecosystem exchange series is.

One :class:`NetEcosystemExchangeSpec` per processed file, and a processed file
holds one **series**: a single estimate of net ecosystem exchange, from a
single source, at a single resolution. The spec says which source and raw file
it comes from, which columns carry its value, quality flag and uncertainties,
and what the quantity is. The processed netCDF stores a spec's fields as its
variables' attributes, so the file describes itself, and its reader checks it
against the same spec.

Contents
--------
:class:`NetEcosystemExchangeSpec`
    The record, one per series.
:data:`NET_ECOSYSTEM_EXCHANGE`, :data:`NET_ECOSYSTEM_EXCHANGE_NAMES`
    The registry and its names, in order.
:data:`SOURCES`
    The sources a spec may name.
:func:`resolve_net_ecosystem_exchange`
    A spec by its name.
:func:`describe`
    One spec as a paragraph, for a run log.
The checks
    One invariant each, grouped as
    :func:`check_net_ecosystem_exchange_spec_is_valid`, which every spec
    passes on construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from frozendict import frozendict
from pysipnet.units import validate_units
from pysipnet.variables import VariableKind

from sipnet_calibration.conventions import NAME_PATTERN
from sipnet_calibration.net_ecosystem_exchange.names import resolve_resolution
from sipnet_calibration.net_ecosystem_exchange.source_files import SOURCE
from sipnet_calibration.validation import truncated

__all__ = [
    "NET_ECOSYSTEM_EXCHANGE",
    "NET_ECOSYSTEM_EXCHANGE_NAMES",
    "NetEcosystemExchangeSpec",
    "SOURCES",
    "describe",
    "resolve_net_ecosystem_exchange",
]


#: The sources a spec may name, each with the source-file columns it can read.
SOURCES: frozendict[str, frozenset[str]] = frozendict({"ameriflux": frozenset(SOURCE.names)})


@dataclass(frozen=True)
class NetEcosystemExchangeSpec:
    """Everything the rest of the project needs to know about one NEE series.

    :meth:`xarray_attributes` is what the processed file stores on its
    ``value``; the ``*_attributes`` methods give its companions'.
    """

    name: str
    """The series' name: the processed file's stem and the registry key; the
    raw file's stem followed by the estimate."""

    source: str
    """The source the series is read from, a key of :data:`SOURCES`."""

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
    """What the series is and how the producer made it."""

    upstream_product: str
    """The producer's own product."""

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
        check_net_ecosystem_exchange_spec_is_valid(self)

    @property
    def raw_file(self) -> str:
        """The raw file the series is built from."""
        return resolve_resolution(self.resolution).raw_file

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Every source column the series reads."""
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
        """Attributes for the processed file's ``value``.

        Keys follow the Climate and Forecast conventions where one exists
        (``units``, ``long_name``, ``comment``); the rest are spelled out.
        """
        attrs: dict[str, Any] = {
            "units": self.units,
            "long_name": self.long_label,
            "description": self.description,
            "upstream_product": self.upstream_product,
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
        """Attributes for the processed file's ``quality_flag``."""
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
        """Attributes for the processed file's ``random_uncertainty``."""
        return self._uncertainty_attributes(
            self.random_uncertainty_column,
            "random uncertainty",
            "ONEFlux's random uncertainty of the step, estimated from measured data only "
            "and reported at gap-filled steps as well as measured ones.",
        )

    def joint_uncertainty_attributes(self) -> dict[str, Any]:
        """Attributes for the processed file's ``joint_uncertainty``."""
        return self._uncertainty_attributes(
            self.joint_uncertainty_column,
            "joint uncertainty",
            "ONEFlux's joint uncertainty: the random uncertainty combined with the u* "
            "filtering uncertainty, sqrt(RANDUNC^2 + ((NEE_84 - NEE_16) / 2)^2) over the "
            "u* threshold ensemble.",
        )

    def _uncertainty_attributes(self, column: str, what: str, description: str) -> dict[str, Any]:
        """The attributes of one uncertainty companion."""
        return {
            "units": self.units,
            "long_name": f"{self.long_label}: {what}",
            "description": description + f" From {SOURCE.documentation}.",
            "constituent": self.constituent,
            "source_file": self.raw_file,
            "source_column": column,
        }


def resolve_net_ecosystem_exchange(name: str) -> NetEcosystemExchangeSpec:
    """The spec named *name*, or a ``KeyError`` listing the names that exist."""
    check_net_ecosystem_exchange_is_registered(name)
    return next(spec for spec in NET_ECOSYSTEM_EXCHANGE if spec.name == name)


def describe(spec: NetEcosystemExchangeSpec) -> str:
    """A spec as a paragraph, for ``--describe`` and the run log."""
    return "\n".join(
        [
            f"{spec.name}: {spec.long_label} ({spec.units} {spec.constituent}, {spec.kind.value}), "
            f"from {spec.upstream_product}.",
            f"  source     {spec.source}, {spec.raw_file}, columns {list(spec.source_columns)}",
            f"  what       {spec.description}",
            f"  time       {spec.time_reference}",
            f"  sign       {spec.sign_convention}",
            f"  units      {spec.units_provenance}",
        ]
    )


# ── private helpers ───────────────────────────────────────────────────────────

_UPSTREAM_PRODUCT = (
    "AmeriFlux FLUXNET (ONEFlux) FULLSET, one CC-BY-4.0 dataset per site, each with its own "
    "DOI (the doi coordinate)"
)

_SIGN = (
    "positive is a flux from the ecosystem to the atmosphere, the micrometeorological "
    "convention. The FLUXNET2015 variable table does not state it; the values are positive "
    "at night and negative in summer daytime, so it is inferred from the data."
)

_TIME_REFERENCE = (
    "mean rate over the step (time_bounds[0], time], UTC; the source's local-standard-time "
    "stamps shifted by the tower's utc_offset"
)

_UNITS_PROVENANCE = (
    "The FLUXNET2015 FULLSET variable table gives umolCO2 m-2 s-1 for half-hourly and hourly "
    f"NEE ({SOURCE.documentation}), and the files are ONEFlux output in that format."
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
    "absent from this series."
)


def _spec(resolution: str, series: str, estimate: str, label: str, description: str) -> NetEcosystemExchangeSpec:
    """One AmeriFlux series' spec: *estimate*'s four columns at *resolution*."""
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
        upstream_product=_UPSTREAM_PRODUCT,
        sign_convention=_SIGN,
        time_reference=_TIME_REFERENCE,
        units_provenance=_UNITS_PROVENANCE,
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_net_ecosystem_exchange_spec_is_valid(spec: NetEcosystemExchangeSpec) -> None:
    """An NEE spec is complete and names a source, resolution and columns that exist."""
    check_net_ecosystem_exchange_name_is_a_processed_name(spec)
    validate_units(spec.units)
    check_net_ecosystem_exchange_source_is_known(spec)
    resolve_resolution(spec.resolution)
    check_net_ecosystem_exchange_name_starts_with_its_raw_file(spec)
    check_net_ecosystem_exchange_spec_is_described(spec)
    check_net_ecosystem_exchange_columns_are_the_sources(spec)


def check_net_ecosystem_exchange_is_registered(name: str) -> None:
    """An NEE series name is one of :data:`NET_ECOSYSTEM_EXCHANGE_NAMES`."""
    if name not in NET_ECOSYSTEM_EXCHANGE_NAMES:
        raise KeyError(
            f"no net ecosystem exchange series named {name!r}; pass one of "
            f"{truncated(NET_ECOSYSTEM_EXCHANGE_NAMES)}."
        )


def check_net_ecosystem_exchange_name_is_a_processed_name(spec: NetEcosystemExchangeSpec) -> None:
    """An NEE series' name is ``lower_case_with_underscores``."""
    if not NAME_PATTERN.match(spec.name):
        raise ValueError(f"series name {spec.name!r} is not lower_case_with_underscores; rename it.")


def check_net_ecosystem_exchange_source_is_known(spec: NetEcosystemExchangeSpec) -> None:
    """An NEE spec's source is one of :data:`SOURCES`."""
    if spec.source not in SOURCES:
        raise ValueError(
            f"series {spec.name!r}: source {spec.source!r} is not one of {truncated(SOURCES)}; "
            "add a reader and its columns first."
        )


def check_net_ecosystem_exchange_name_starts_with_its_raw_file(spec: NetEcosystemExchangeSpec) -> None:
    """An NEE series is named for its raw file's stem, then the estimate."""
    stem = spec.raw_file.removesuffix(".nc")
    if not spec.name.startswith(stem + "_"):
        raise ValueError(
            f"series {spec.name!r} does not start with its raw file's stem {stem!r} and an "
            "estimate; name it <raw file stem>_<estimate>."
        )


def check_net_ecosystem_exchange_spec_is_described(spec: NetEcosystemExchangeSpec) -> None:
    """An NEE spec says what the series is, where it is from and what its labels mean."""
    fields = (
        "long_label",
        "description",
        "upstream_product",
        "sign_convention",
        "time_reference",
        "units_provenance",
    )
    empty = [field for field in fields if not getattr(spec, field)]
    if empty:
        raise ValueError(f"series {spec.name!r} needs {truncated(empty)}; give each.")


def check_net_ecosystem_exchange_columns_are_the_sources(spec: NetEcosystemExchangeSpec) -> None:
    """Every column an NEE spec names is one its source carries."""
    readable = SOURCES[spec.source]
    for column in spec.source_columns:
        if column not in readable:
            raise ValueError(
                f"series {spec.name!r}: {column!r} is not a column the {spec.source} source "
                "carries; name one it does."
            )


# ── the registry ──────────────────────────────────────────────────────────────

# Last in the module, since building a spec runs the checks above.

NET_ECOSYSTEM_EXCHANGE: tuple[NetEcosystemExchangeSpec, ...] = (
    _spec("half_hourly", "ustar_variable", "VUT_REF", "Net ecosystem exchange (variable u* threshold)", _VARIABLE),
    _spec("half_hourly", "ustar_constant", "CUT_USTAR50", "Net ecosystem exchange (constant u* threshold)", _CONSTANT),
    _spec("hourly", "ustar_variable", "VUT_REF", "Net ecosystem exchange (variable u* threshold)", _VARIABLE),
    _spec("hourly", "ustar_constant", "CUT_USTAR50", "Net ecosystem exchange (constant u* threshold)", _CONSTANT),
)

#: The series' names, in registry order.
NET_ECOSYSTEM_EXCHANGE_NAMES: tuple[str, ...] = tuple(spec.name for spec in NET_ECOSYSTEM_EXCHANGE)

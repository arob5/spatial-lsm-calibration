"""One observation source: its observed values, and the operator that predicts them.

Where this sits
---------------
::

    scripts/ingest_constraints.py                data/processed/constraints/*.nc
      -> constraints.constraint_fields()         observed values, (site[, time])
      -> ObservationSource(...)                  this module: a source and its operator
      -> ObservationVector(observation_sources=[...])
                                                 observation.vector: the vector

An :class:`ObservationSource` is the piece an
:class:`~sipnet_calibration.observation.vector.ObservationVector` is built
from. It imports nothing from the data-source modules: it is built from an
array, and reads its coordinates and attributes.

What it reads
-------------
The observed values, a field (:mod:`sipnet_calibration.fields`), as
:func:`sipnet_calibration.constraints.constraint_fields` gives one per
constraint; and an
:class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
:data:`ObservedValues` (``xr.DataArray``) is a field on ``(site,)`` (a
static observation source) or ``(site, time)``, with no batch dim:

``site``
    ``int32`` site ids, unique, with ``float64`` ``lon``/``lat`` on ``site``
    (the field contract's).
``time``
    Naive ``datetime64`` labels, strictly increasing, no ``NaT``; where the
    values are attributed to windows, ``window_start``/``window_end`` on
    ``time``.
values
    Numeric, finite where observed, ``NaN`` where nothing was observed.
attributes
    ``units``, valid by pySIPNET, and ``constituent`` where the quantity has
    one; what a prediction is converted into.

A scalar coordinate, such as the ``sample=0`` that ``.isel(sample=0)`` of an
observation ensemble leaves, is metadata and is kept; a batch dim is refused,
since an observation ensemble is reduced to one value per ``(site[, time])``
by the experiment before it enters a vector.

Functions
---------
:func:`validate_observed_values`
    Check that an array is observed values, raising on the first rule it
    breaks.
:class:`ObservationSource`
    One observation source's name, observed values and operator; ``sites``,
    ``is_static``, ``n_observations`` and ``observations()`` (its
    observations as a table).

Notes
-----
**Sorting is a normalization, not a refusal.** An observation source sorts
its observed values by ``site`` and ``time`` before it checks them, since the
order of the labels carries nothing; the vector's order is its own
(site-major, sites ascending). It then keeps only the sites and time labels
holding an observation, so an operator reads the model nowhere else.

**Frozen values.** An observation source holds its own read-only copy of the
observed values it was given, loaded into memory, and ``observed_values``
hands out a read-only copy of that on every read, so neither a later write to
the caller's array nor one to what it reads can change the observations after
a vector's ``y`` and index were built from them.

Usage
-----
::

    from sipnet_calibration.constraints import constraint_fields
    from sipnet_calibration.observation import DEFAULT_OBS_OPS, ObservationSource

    # Two sites MODIS observes; sites 1 and 27 have no MODIS LAI.
    observed = constraint_fields(["modis_leaf_area_index"], sites=[3851, 3871])
    source = ObservationSource(
        observation_source_name="modis_leaf_area_index",
        observed_values=observed["modis_leaf_area_index"],
        operator=DEFAULT_OBS_OPS["modis_leaf_area_index"],
    )
    source.n_observations, source.sites
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import (
    DATA_SOURCE_MEMBER_NAMES,
    LAT,
    LON,
    SAMPLE,
    SITE,
    SOURCE_INDEX,
    TIME,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration.fields import (
    MODEL_OUTPUT_COORDINATE_NAMES,
    ReadOnlyCopies,
    batch_dims,
    message_name,
    read_only_copy,
    validate_field,
)

# The module, not its names: the operators read ObservedValues and
# validate_observed_values from here, and the package imports the operators
# first, so this module runs while they are still being defined and reads
# them only when a source is built.
from sipnet_calibration.observation import operators

__all__ = [
    "RESERVED_OBSERVATION_SOURCE_NAMES",
    "ObservationSource",
    "ObservedValues",
    "check_observation_source_is_valid",
    "validate_observed_values",
]

#: Names an observation source cannot take, because its arrays are named for
#: it and would then share a name with a dim or coordinate of theirs or of a
#: model output: ``sample``, ``site``, ``lon``, ``lat``, ``source_index``, the
#: data sources' member dims, the model output's coordinates (``time`` among
#: them) and the window edges.
RESERVED_OBSERVATION_SOURCE_NAMES: frozenset[str] = frozenset(
    {
        SAMPLE,
        SITE,
        LON,
        LAT,
        SOURCE_INDEX,
        *DATA_SOURCE_MEMBER_NAMES,
        *MODEL_OUTPUT_COORDINATE_NAMES,
        WINDOW_START,
        WINDOW_END,
    }
)

#: One observation source's observed values: a field on ``(site[, time])``
#: with no batch dim, ``NaN`` where nothing was observed; checked by
#: :func:`validate_observed_values`.
type ObservedValues = xr.DataArray


def validate_observed_values(observed_values: Any, *, message_name: str | None = None) -> None:
    """Check that *observed_values* are observed values, raising on the first rule.

    Runs :func:`sipnet_calibration.fields.validate_field`, then
    :func:`check_observed_values_have_no_batch_dim`,
    :func:`check_observed_values_are_on_site_and_time`,
    :func:`check_observed_values_are_numeric` and
    :func:`check_observed_values_are_finite_or_nan`, in that order.

    Parameters
    ----------
    observed_values:
        The array to check.
    message_name:
        What an error message calls it; its name, or else ``"the observed
        values"``, when omitted.

    Raises
    ------
    TypeError
        If *observed_values* is not an ``xr.DataArray``.
    ValueError
        If it is not a field (``int32`` ``site`` ids with ``lon``/``lat``,
        a ``time`` axis of naive datetimes strictly increasing and free of
        ``NaT``, valid ``units``, among the field contract's rules); if it
        has a batch dim; if its dims are not ``(site,)`` or ``(site, time)``;
        if its values are not numeric; or if one is infinite.
    """
    name = message_name
    if name is None:
        name = (
            _message_name(observed_values, "the observed values")
            if isinstance(observed_values, xr.DataArray)
            else "the observed values"
        )
    validate_field(observed_values, message_name=name)
    check_observed_values_have_no_batch_dim(observed_values, name)
    check_observed_values_are_on_site_and_time(observed_values, name)
    check_observed_values_are_numeric(observed_values, name)
    check_observed_values_are_finite_or_nan(observed_values, name)


@dataclass(frozen=True, eq=False, kw_only=True)
class ObservationSource:
    """One observation source, with the operator that predicts it.

    Parameters
    ----------
    observation_source_name:
        The observation source's name, a non-empty string; the observed
        values are renamed to it.
    observed_values:
        The observed values (:data:`ObservedValues`, the module docstring's
        data model). They are copied and loaded into memory, sorted by
        ``site`` and ``time``, trimmed to the sites and time labels holding
        at least one observation, and stored read-only.
    operator:
        The :class:`~sipnet_calibration.observation.operators.ObservationOperator`
        that predicts the observation source.

    Raises
    ------
    TypeError
        If *observation_source_name* is not a string, *observed_values* is
        not a ``DataArray``, or *operator* is not callable or declares its
        names other than as tuples of strings.
    ValueError
        If *observation_source_name* is empty or reserved
        (:data:`RESERVED_OBSERVATION_SOURCE_NAMES`); if *observed_values* are not
        observed values (:func:`validate_observed_values`), once sorted; or
        if the operator declares an alias.
    KeyError
        If the operator declares a name pySIPNET does not know.

    Notes
    -----
    Observation sources compare and hash by identity: two built from equal
    arrays are two observation sources, since an array has no single truth
    value to compare by.
    """

    observation_source_name: str
    # Every read is a read-only copy, so neither the values, the coordinates,
    # the attributes nor the coordinate bindings can be changed through it.
    observed_values: ObservedValues = ReadOnlyCopies()
    operator: operators.ObservationOperator

    def __post_init__(self) -> None:
        # The caller's array as given: a read through the field would make
        # the caller's own buffers read-only.
        values = _sort_by_site_and_time(vars(self)["_observed_values"])
        check_observation_source_is_valid(self.observation_source_name, values, self.operator)
        stored = _read_only_copy(_observed_labels_only(values), self.observation_source_name)
        object.__setattr__(self, "observed_values", stored)

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def sites(self) -> tuple[int, ...]:
        """The observed sites, ascending."""
        return tuple(int(s) for s in self.observed_values[SITE].values)

    @property
    def is_static(self) -> bool:
        """Whether the observed values document no time."""
        return TIME not in self.observed_values.dims

    @property
    def n_observations(self) -> int:
        """The number of observed (finite) values."""
        return int(self.observed_values.notnull().sum())

    def __repr__(self) -> str:
        return (
            f"ObservationSource({self.observation_source_name!r}, "
            f"{self.n_observations} observations, {self.operator!r})"
        )

    # ── representations ───────────────────────────────────────────────────────

    def observations(self) -> pd.DataFrame:
        """The observations as a table with ``site``, ``time`` and ``value`` columns."""
        values = self.observed_values
        mask = values.notnull()
        site_ids = values[SITE].values
        if not self.is_static:
            site_ids = site_ids[:, None]
        sites = np.broadcast_to(site_ids, values.shape)
        if self.is_static:
            times = np.full(values.shape, np.datetime64("NaT", "ns"))
        else:
            times = np.broadcast_to(values[TIME].values[None, :], values.shape)
        chosen = mask.values
        return pd.DataFrame(
            {
                SITE: sites[chosen].astype(np.int64),
                TIME: times[chosen].astype("datetime64[ns]"),
                "value": values.values[chosen].astype(np.float64),
            }
        )


# ── supporting helpers ────────────────────────────────────────────────────────

# For validate_observed_values, whose message_name argument shadows the
# function of that name.
_message_name = message_name


def _sort_by_site_and_time(values: Any) -> Any:
    """*values* on ``(site[, time])``, sorted by both; anything else as it is.

    What the checks refuse (not a ``DataArray``, no ``site`` dim or index)
    is passed through for them to refuse.
    """
    if not isinstance(values, xr.DataArray) or SITE not in values.indexes:
        return values
    values = values.transpose(..., *[d for d in (SITE, TIME) if d in values.dims])
    if not values.indexes[SITE].is_monotonic_increasing:
        values = values.sortby(SITE)
    if TIME in values.indexes and not values.indexes[TIME].is_monotonic_increasing:
        values = values.sortby(TIME)
    return values


def _observed_labels_only(values: xr.DataArray) -> xr.DataArray:
    """*values* without the sites and time labels that hold no observation."""
    observed = values.notnull()
    if TIME in values.dims:
        values = values.isel({SITE: np.flatnonzero(observed.any(TIME).values)})
        values = values.isel({TIME: np.flatnonzero(observed.any(SITE).values)})
        return values
    return values.isel({SITE: np.flatnonzero(observed.values)})


def _read_only_copy(values: xr.DataArray, observation_source_name: str) -> xr.DataArray:
    """A read-only, in-memory copy of *values*, named *observation_source_name*."""
    # load() computes a dask or lazily indexed copy in place, so the buffer made
    # read-only is the one the observation source keeps rather than a fresh one
    # per read.
    return read_only_copy(values.rename(observation_source_name).copy(deep=True).load())


# ── checks ────────────────────────────────────────────────────────────────────


def check_observation_source_is_valid(
    observation_source_name: Any, observed_values: Any, operator: Any
) -> None:
    """An observation source's name, observed values and operator are what it needs.

    Runs :func:`check_observation_source_name_is_a_nonempty_string`,
    :func:`check_observation_source_name_is_not_reserved`,
    :func:`validate_observed_values` and
    :func:`~sipnet_calibration.observation.operators.check_operator_declares_names`,
    in that order.
    """
    check_observation_source_name_is_a_nonempty_string(observation_source_name)
    check_observation_source_name_is_not_reserved(observation_source_name)
    validate_observed_values(observed_values, message_name=observation_source_name)
    operators.check_operator_declares_names(operator, observation_source_name)


def check_observation_source_name_is_a_nonempty_string(observation_source_name: Any) -> None:
    """The observation source's name is a non-empty string."""
    if not isinstance(observation_source_name, str):
        raise TypeError(
            "observation_source_name must be a string, got "
            f"{type(observation_source_name).__name__}; name the observation source as its "
            "constraint file is named, e.g. 'modis_leaf_area_index'."
        )
    if not observation_source_name:
        raise ValueError(
            "observation_source_name is empty; name the observation source as its "
            "constraint file is named, e.g. 'modis_leaf_area_index'."
        )


def check_observation_source_name_is_not_reserved(observation_source_name: str) -> None:
    """The source's name is none of :data:`RESERVED_OBSERVATION_SOURCE_NAMES`."""
    if observation_source_name in RESERVED_OBSERVATION_SOURCE_NAMES:
        raise ValueError(
            f"observation_source_name {observation_source_name!r} is reserved: the source's "
            "arrays are named for it, and would share the name of a dim or coordinate; name "
            "the observation source as its constraint file is named, e.g. "
            "'modis_leaf_area_index'."
        )


def check_observed_values_have_no_batch_dim(values: xr.DataArray, message_name: str) -> None:
    """Observed values carry no batch dim: one value per ``(site[, time])``."""
    batch = batch_dims(values)
    if batch:
        raise ValueError(
            f"{message_name}: observed values carry the batch dim(s) {list(batch)}. An "
            "observation ensemble is reduced to one value per (site[, time]) by the "
            "experiment before it enters the vector; reduce it, or select one label."
        )


def check_observed_values_are_on_site_and_time(values: xr.DataArray, message_name: str) -> None:
    """Observed values are on ``(site,)`` or ``(site, time)``, ``site`` a dim."""
    if tuple(values.dims) not in ((SITE,), (SITE, TIME)):
        raise ValueError(
            f"{message_name}: observed values are on (site,) or (site, time), got dims "
            f"{tuple(values.dims)}; give one site a site dim of length one with "
            ".expand_dims('site'), and select or reduce the other dimensions first."
        )


def check_observed_values_are_numeric(values: xr.DataArray, message_name: str) -> None:
    """Observed values are numbers."""
    if values.dtype.kind not in "iuf":
        raise ValueError(
            f"{message_name}: observed values must be numeric, got dtype {values.dtype}; "
            "convert them to float, with NaN where nothing was observed."
        )


def check_observed_values_are_finite_or_nan(values: xr.DataArray, message_name: str) -> None:
    """Every observed value is finite, or ``NaN`` where nothing was observed."""
    if np.isinf(values.values.astype(np.float64)).any():
        raise ValueError(
            f"{message_name}: an observed value is infinite; an observation is finite or "
            "NaN, so replace it with NaN or drop it."
        )

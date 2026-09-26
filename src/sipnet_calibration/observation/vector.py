"""The observation vector: the observations of an experiment in a fixed
order, with the operator that predicts each observation source, and the two
forms its values take.

Where this sits
---------------
::

    scripts/ingest_constraints.py                    data/processed/constraints/*.nc
      -> constraints.constraint_fields()             observed values, (site[, time])
      -> ObservationSource(observation_source_name=, observed_values=, operator=)
                                                     one per observation source
                                                     (observation.source)
      -> ObservationVector(observation_sources=[...]) the observations, in Flat order
           .y                                        what pyEKI compares against
           .predict(model_output, sipnet_parameter_fields=)
                                                     H applied, converted, checked
           .flat(fields) / .fields(flat_values)      Fields <-> Flat

The forward model flattens predictions with it into pyEKI's ``(J, N)``; the
inference layer reads ``y``, ``index`` and ``positions`` off it to build the
error model; a predictive-check figure unstacks a ``(J, N)`` batched Flat
with ``fields``. It imports nothing from the data-source modules: an
observation source is built from arrays, and reads their attributes.

What it reads
-------------
:class:`~sipnet_calibration.observation.source.ObservationSource`\ s, each
holding one observation source's observed values
(:data:`~sipnet_calibration.observation.source.ObservedValues`, a field on
``(site[, time])``) and its
:class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
**Fields**: ``dict[str, Field]`` keyed by observation source name, each a
``float64`` field on that source's own ``(site[, time])`` grid, with its
``site``, ``lon``/``lat`` and time coordinates, ``NaN`` where nothing is
observed; a ``(J, N)`` batch unstacks to the same with a leading batch dim,
``sample`` unless ``batch_dim=`` names it otherwise, an ``int64`` coordinate
labeled ``0`` to ``J - 1``. A source's grid holds only the sites and time
labels with at least one observation: the rest are dropped when its
observation source is built. A dict and not a Dataset, since each source is
on its own time grid.

**Flat**: ``y`` in ``R^N``, a ``float64`` ``jax.Array``, finite, one entry per
observation; predictions are ``(N,)`` or ``(J, N)`` ``jax.Array``\ s, where a
``NaN`` marks a failed run. Every method taking Flat accepts any array-like.

**The index**: a ``pandas.MultiIndex`` with levels
``(site, observation_source, time)`` -- ``site`` ``int64``,
``observation_source`` a string, ``time`` ``datetime64[ns]`` -- one row per
observation, in Flat order: **site-major**, sites ascending, then observation
sources in declaration order, then times ascending. A static source's
observations have ``time = NaT``. The order is a property of the vector;
consumers read it off ``index`` and ``positions`` and never assume it.

**The site table**: ``site_table``, one row per observed site, ascending:
``site_id`` (``int32``), ``lon`` and ``lat``, read off the observed values,
which every observation source must agree on.

Functions
---------
:class:`ObservationVector`
    A frozen, keyword-only dataclass. Its pieces are its observation
    sources: ``vector[name]``, ``name in vector``, ``iter(vector)`` (the
    names), ``len(vector)``, ``observation_source_names``,
    ``observation_sources``. ``sites``, ``site_table``, ``dimension``,
    ``index``, ``output_variable_names`` and ``sipnet_parameter_names_read``
    (the unions over the operators), ``observed_values_by_source`` (Fields)
    and ``y`` (Flat); ``select(observation_source_names=, sites=, time=)`` for
    a sub-vector and ``restrict_to_sites(sites)``, its intersecting form;
    ``positions(site=, observation_source_name=)`` for where a segment sits
    in Flat; ``flat(fields)`` and ``fields(flat_values)`` between the
    representations; ``predict(model_output, sipnet_parameter_fields=)`` for
    every operator applied, converted and checked; ``describe()`` for one
    row per observation source.

Notes
-----
**Site-major.** A likelihood that factorizes over sites, or an error
covariance block-diagonal by site, reads a site's observations as one
contiguous segment, and a worker producing one site's predictions produces
that segment. ``(site, observation_source)`` sub-segments are contiguous too.

**Selection never reorders.** ``select`` keeps the vector's order of sources
and sites whatever order they are asked in, and refuses a name or site the
vector does not hold, so a typo cannot silently shrink it;
``restrict_to_sites`` is the form for a caller holding a larger site set.

**No noise model here.** The observation sources' standard deviations, the
covariance and the likelihood are the inference layer's; this module gives it
``y``, ``index`` and ``positions``.

**Failed runs.** ``predict`` passes a ``NaN`` through where the model output
itself is ``NaN`` at that site and batch label (a failed run), and refuses one
anywhere else, so a coverage gap cannot masquerade as a failed run and be
repaired away.

**Only observed labels.** An observation source keeps only the sites and time
labels it observes, so an operator reads the model nowhere else: not at a site
the model output need not carry, and not at a label outside a shorter run.

Usage
-----
::

    from sipnet_calibration.constraints import constraint_fields
    from sipnet_calibration.observation import (
        DEFAULT_OBS_OPS, ObservationSource, ObservationVector, ReduceOverWindows,
    )

    names = ["modis_leaf_area_index", "landtrendr_aboveground_biomass"]
    observed = constraint_fields(names, sites=sites)
    vector = ObservationVector(observation_sources=[
        ObservationSource(
            observation_source_name="modis_leaf_area_index",
            observed_values=observed["modis_leaf_area_index"],
            operator=DEFAULT_OBS_OPS["modis_leaf_area_index"],
        ),
        ObservationSource(
            observation_source_name="landtrendr_aboveground_biomass",
            observed_values=observed["landtrendr_aboveground_biomass"],
            operator=ReduceOverWindows("wood_carbon", "mean"),  # the experiment's reading
        ),
    ]).select(time=slice("2012", "2024"))

    vector.dimension, vector.y.shape           # N, (N,)
    predicted_fields = vector.predict(model_output, sipnet_parameter_fields=sipnet_parameter_fields)
    predictions = vector.flat(predicted_fields)  # (J, N) for a (sample, site, time) output
    vector.fields(predictions)["modis_leaf_area_index"]  # back to (sample, site, time)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.units import convert_dataarray_units

from sipnet_calibration.conventions import (
    LAT,
    LON,
    SAMPLE,
    SITE,
    SITE_DTYPE,
    SITE_ID,
    TIME,
    FrozenMapping,
)
from sipnet_calibration.fields import (
    Field,
    ModelOutput,
    SIPNETParameterFields,
    batch_coordinate,
    in_field_layout,
    batch_dims,
    check_at_most_one_batch_dim,
    check_batch_dim_name_is_not_a_data_source_member,
    check_batch_dim_name_is_not_reserved,
    validate_field,
    missing_labels,
    scalar_batch_labels,
)
from sipnet_calibration.observation.operators import (
    check_model_output_carries_what_is_read,
    check_result_is_on_the_observation_grid,
)
from sipnet_calibration.observation.source import ObservationSource
from sipnet_calibration.validation import (
    as_batched_flat,
    as_names,
    as_sequence,
    as_site_id,
    as_site_ids,
    check_names_are_unique,
    check_the_restriction_keeps_a_site,
    is_one_vector,
    truncated,
)

__all__ = [
    "INDEX_LEVELS",
    "ObservationVector",
    "check_batch_dim_is_not_an_observation_source_name",
]

OBSERVATION_SOURCE = "observation_source"

#: The levels of :attr:`ObservationVector.index`, in order.
INDEX_LEVELS: tuple[str, ...] = (SITE, OBSERVATION_SOURCE, TIME)

#: The dim the vectorized read of the observations indexes along, one per
#: observation; internal to ``flat`` and never on a result.
_OBSERVATION_DIM = "__observation__"


@dataclass(frozen=True, eq=False, kw_only=True)
class ObservationVector:
    """The observations of several observation sources, in the module docstring's order.

    Parameters
    ----------
    observation_sources:
        One :class:`~sipnet_calibration.observation.source.ObservationSource`
        per observation source, in the order the sources take within each
        site's segment; stored as a tuple.

    Raises
    ------
    TypeError
        If *observation_sources* is not a sequence, or an entry is not an
        ``ObservationSource``.
    ValueError
        If there is no observation source, two share a name, one holds no
        observation, or two give one site different ``lon``/``lat``.

    Notes
    -----
    Compared and hashed by identity (``eq=False``), as every vector-like
    class of the package is: its observations are arrays, which have no
    single truth value to compare by. It pickles, which the forward model
    relies on to send a site's slice to a worker.
    """

    observation_sources: tuple[ObservationSource, ...]
    _by_name: Mapping[str, ObservationSource] = field(init=False, repr=False)
    _index: pd.MultiIndex = field(init=False, repr=False)
    _sites: tuple[int, ...] = field(init=False, repr=False)
    _locations: tuple[tuple[float, ...], tuple[float, ...]] = field(init=False, repr=False)
    _y: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observation_sources = tuple(
            as_sequence(self.observation_sources, message_name="observation_sources")
        )
        check_observation_vector_is_valid(observation_sources)
        object.__setattr__(self, "observation_sources", observation_sources)
        object.__setattr__(
            self,
            "_by_name",
            FrozenMapping({source.observation_source_name: source for source in observation_sources}),
        )
        index = _build_index(observation_sources)
        object.__setattr__(self, "_index", index)
        sites = tuple(int(s) for s in np.unique(index.get_level_values(SITE)))
        object.__setattr__(self, "_sites", sites)
        object.__setattr__(self, "_locations", _site_locations_of(observation_sources, sites))
        object.__setattr__(self, "_y", self.flat(self.observed_values_by_source))

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def observation_source_names(self) -> tuple[str, ...]:
        """The observation sources' names, in the vector's order."""
        return tuple(source.observation_source_name for source in self.observation_sources)

    @property
    def sites(self) -> tuple[int, ...]:
        """The sites with at least one observation, ascending."""
        return self._sites

    @property
    def site_table(self) -> pd.DataFrame:
        """The site table of :attr:`sites`: ``site_id`` (``int32``), ``lon`` and ``lat``.

        A new frame on every access, read off the observed values'
        locations.
        """
        lon, lat = self._locations
        return pd.DataFrame(
            {
                SITE_ID: np.asarray(self._sites, dtype=SITE_DTYPE),
                LON: np.asarray(lon, dtype=np.float64),
                LAT: np.asarray(lat, dtype=np.float64),
            }
        )

    @property
    def dimension(self) -> int:
        """``N``, the number of observations."""
        return len(self._index)

    @property
    def index(self) -> pd.MultiIndex:
        """One row per entry of Flat: levels :data:`INDEX_LEVELS`; a copy, so
        setting its names changes nothing of the vector."""
        return self._index.copy()

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        """The pySIPNET output variables the operators read, in declaration order."""
        return _union(
            source.operator.output_variable_names for source in self.observation_sources
        )

    @property
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        """The SIPNET parameters the operators read, in declaration order."""
        return _union(
            source.operator.sipnet_parameter_names_read for source in self.observation_sources
        )

    @property
    def observed_values_by_source(self) -> dict[str, Field]:
        """Fields: each observation source's observed values, by name, each a
        read-only copy (:attr:`ObservationSource.observed_values`)."""
        return {
            source.observation_source_name: source.observed_values
            for source in self.observation_sources
        }

    @property
    def y(self) -> jax.Array:
        """Flat: the observations, ``(N,)`` ``float64``, in :attr:`index` order."""
        return self._y

    def __getitem__(self, observation_source_name: str) -> ObservationSource:
        """The observation source called *observation_source_name*."""
        check_observation_source_names_are_held(
            [observation_source_name], self.observation_source_names
        )
        return self._by_name[observation_source_name]

    def __contains__(self, observation_source_name: object) -> bool:
        """Whether *observation_source_name* is an observation source of the vector."""
        return observation_source_name in self.observation_source_names

    def __iter__(self) -> Iterator[str]:
        """The observation source names, in the vector's order."""
        return iter(self.observation_source_names)

    def __reversed__(self) -> Iterator[str]:
        """The observation source names, in reverse order."""
        return reversed(self.observation_source_names)

    def __len__(self) -> int:
        """The number of observation sources."""
        return len(self.observation_sources)

    def __repr__(self) -> str:
        return (
            f"ObservationVector(N={self.dimension}, "
            f"observation_sources={list(self.observation_source_names)}, "
            f"sites={len(self.sites)})"
        )

    def describe(self) -> pd.DataFrame:
        """One row per observation source: counts, dates, units and operator."""
        rows = []
        for source in self.observation_sources:
            observations = source.observations()
            rows.append(
                {
                    OBSERVATION_SOURCE: source.observation_source_name,
                    "observations": len(observations),
                    "sites": int(observations[SITE].nunique()),
                    "first_time": None if source.is_static else observations[TIME].min(),
                    "last_time": None if source.is_static else observations[TIME].max(),
                    "units": source.observed_values.attrs["units"],
                    "constituent": source.observed_values.attrs.get("constituent", ""),
                    "operator": repr(source.operator),
                }
            )
        return pd.DataFrame(rows).set_index(OBSERVATION_SOURCE)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(
        self,
        *,
        observation_source_names: Sequence[str] | None = None,
        sites: Iterable[int] | None = None,
        time: slice | None = None,
    ) -> ObservationVector:
        """A sub-vector over some observation sources, sites and a time slice.

        Parameters
        ----------
        observation_source_names:
            The observation sources to keep, a sequence, each held by the
            vector and named once. The sub-vector keeps this vector's order,
            whatever the order asked for. Defaults to every observation
            source.
        sites:
            The sites to keep, a sequence of site ids, each one of
            :attr:`sites` and named once, read once. The sub-vector keeps
            them ascending. Defaults to every site;
            :meth:`restrict_to_sites` is the form that ignores a site the
            vector does not observe.
        time:
            A slice of ``time`` labels, such as ``slice("2012", "2024")``. It
            does not apply to a static observation source.

        Returns
        -------
        ObservationVector
            Every observation that survives all three filters. An observation
            source left with no observations is dropped, and each source's
            ``time`` keeps only the labels where a kept site is observed, so an
            operator run on the sub-vector reads the model only at those
            labels. Its Flat order is this vector's, restricted: a selection
            never reorders the entries it keeps.

        Raises
        ------
        TypeError
            If *time* is not a slice, or an end of it is not a time (a string
            or a datetime); if *observation_source_names* or *sites*
            is one value, a string or a set; or if a site id is a boolean, a
            float or not a number.
        KeyError
            If an observation source name is not held, or a site is not one
            of :attr:`sites`.
        ValueError
            If *time* has a step; if a name or a site is given twice, a site
            id is out of range, *sites* is a two-dimensional array, or the
            selection leaves no observation.

        Notes
        -----
        The sub-vector's Flat values are not this one's: move predictions
        across through Fields, where the correspondence is by label.
        """
        chosen = self.observation_sources
        if observation_source_names is not None:
            names = as_names(observation_source_names, message_name="observation_source_names")
            check_names_are_unique(names, message_name="observation_source_names")
            check_observation_source_names_are_held(names, self.observation_source_names)
            chosen = tuple(source for source in chosen if source.observation_source_name in names)
        wanted_sites = None
        if sites is not None:
            wanted_sites = list(as_site_ids(sites, message_name="sites"))
            check_sites_are_the_vectors(wanted_sites, self._sites)
        check_time_is_a_slice(time)
        kept = [_source_restricted_to(source, wanted_sites, time) for source in chosen]
        kept = [source for source in kept if source is not None]
        check_the_selection_keeps_an_observation(kept)
        return ObservationVector(observation_sources=kept)

    def restrict_to_sites(self, sites: Iterable[int]) -> ObservationVector:
        """The sub-vector over the vector's sites among *sites*; the others are ignored.

        The intersecting form of ``select(sites=)``: a site the vector does not
        observe is not an error here, which is what a caller holding a larger
        site set (a parameter vector's) wants.

        Parameters
        ----------
        sites:
            A sequence of site ids, each named once.

        Returns
        -------
        ObservationVector
            ``select(sites=[s for s in self.sites if s in sites])``.

        Raises
        ------
        TypeError, ValueError
            As ``select(sites=)`` raises them for *sites*; ``ValueError`` too
            if none of the vector's sites is among them.
        """
        wanted = set(as_site_ids(sites, message_name="sites"))
        kept = [site for site in self._sites if site in wanted]
        check_the_restriction_keeps_a_site(kept)
        return self.select(sites=kept)

    def positions(
        self, *, site: int | None = None, observation_source_name: str | None = None
    ) -> np.ndarray:
        """Where in Flat the observations of a site, an observation source, or both sit.

        Parameters
        ----------
        site:
            An integer site id, or ``None`` for every site.
        observation_source_name:
            An observation source the vector holds, or ``None`` for every
            observation source.

        Returns
        -------
        numpy.ndarray
            The ascending ``int64`` positions in Flat of the observations
            matching both, which may be none when the site is observed but not
            by that observation source.

        Raises
        ------
        TypeError
            If *site* is a boolean, a float, a string or not a number.
        ValueError
            If *site* is not a site id, from 1 to the largest ``int32``.
        KeyError
            If *site* is not one of :attr:`sites`, or
            *observation_source_name* is not held.
        """
        mask = np.ones(self.dimension, dtype=bool)
        if site is not None:
            site_id = as_site_id(site, message_name="site")
            check_sites_are_the_vectors([site_id], self._sites)
            mask &= self._index.get_level_values(SITE).values == site_id
        if observation_source_name is not None:
            check_observation_source_names_are_held(
                [observation_source_name], self.observation_source_names
            )
            labels = self._index.get_level_values(OBSERVATION_SOURCE).values
            mask &= labels == observation_source_name
        return np.flatnonzero(mask).astype(np.int64)

    # ── representations ───────────────────────────────────────────────────────

    def flat(self, fields: Mapping[str, Field]) -> jax.Array:
        """Fields to Flat: ``(N,)``, or ``(J, N)`` when the fields carry a batch dim.

        Parameters
        ----------
        fields:
            A mapping from every observation source name to an array on that
            source's ``site`` and ``time`` labels, with or without one batch
            dim (of any name, the same in every array), in any dimension
            order. An array may be larger than the observed values (a
            prediction over more sites or times); only the vector's
            observations are read, by label. A
            scalar batch coordinate is not a batch dim: such arrays give one
            vector.

        Returns
        -------
        jax.Array
            ``float64``, ``(N,)``, or ``(J, N)`` in the order of the fields'
            batch labels. Values are taken as they are: a ``NaN`` prediction
            stays ``NaN``.

        Raises
        ------
        TypeError
            If *fields* is not a mapping, or an entry is not a ``DataArray``.
        ValueError
            If an observation source is missing; if an array lacks an
            observed site or time label, or a ``site`` or ``time`` dimension
            the source has;
            if an array has a dim that is neither a batch dim (integer
            labels), a spatial dim nor ``time``; if an array has more than
            one batch dim (stack each into a new one first,
            ``{name: stack_batch_dims(array, new_batch_dim="run") for name, array in
            fields.items()}``, with
            :func:`sipnet_calibration.fields.stack_batch_dims`); or if some
            arrays carry a batch dim and others do not, they carry different
            ones, or they disagree on its labels.
        """
        check_fields_hold_the_observation_sources(fields, self.observation_source_names)
        arrays = {}
        for source in self.observation_sources:
            array = fields[source.observation_source_name]
            check_field_is_on_the_grid(array, source)
            laid_out = in_field_layout(array)
            if not isinstance(laid_out.data, np.ndarray):
                # A JAX-backed array, say: xarray indexes only NumPy by label.
                laid_out = laid_out.copy(data=np.asarray(laid_out.data))
            arrays[source.observation_source_name] = laid_out
        batch = _batch_dim_and_labels(arrays, self.observation_source_names)
        shape = (len(batch[1]), self.dimension) if batch is not None else (self.dimension,)
        out = np.full(shape, np.nan, dtype=np.float64)
        for source in self.observation_sources:
            array = arrays[source.observation_source_name]
            positions = self.positions(observation_source_name=source.observation_source_name)
            out[..., positions] = _read_observations(
                array,
                self._index[positions],
                source.is_static,
                None if batch is None else batch[0],
            )
        return jnp.asarray(out)

    def fields(self, flat_values: Any, *, batch_dim: str = SAMPLE) -> dict[str, Field]:
        """Flat to Fields: ``(N,)`` or ``(J, N)`` onto each observation source's grid.

        Parameters
        ----------
        flat_values:
            An array-like of shape ``(N,)`` or ``(J, N)``, in Flat order: a
            JAX or NumPy array, or a nested list.
        batch_dim:
            The name of the batch dim a ``(J, N)`` batch is given. It may not
            be a reserved name (a spatial name, ``time``, ``source_index``),
            a data source's member name (``driver_member``,
            ``initial_condition_member``), an observation source name, or a
            coordinate of an observation source's observed values other than
            a scalar batch label.

        Returns
        -------
        dict
            Observation source name to a ``float64`` array on the source's
            ``(site[, time])`` grid, with its coordinates and attributes and
            ``NaN`` where nothing is observed. A ``(J, N)`` batch gives each
            array a leading ``int64`` *batch_dim* labeled ``0`` to ``J - 1``.
            An observation source's scalar batch labels are not carried.

        Raises
        ------
        TypeError
            If *flat_values* is not a rectangular array of real numbers, or
            *batch_dim* is not a string.
        ValueError
            If *flat_values* is not one- or two-dimensional or does not have
            ``N`` entries per row, or *batch_dim* is a name it may not be.

        Notes
        -----
        An observation source's scalar batch labels
        (:func:`~sipnet_calibration.fields.scalar_batch_labels`, such as a
        ``sample=4`` left by selecting one sample of an ensemble) describe the
        observed input, not the batch: carried onto the arrays, they would
        say every row is sample 4, which the rows are not. They are dropped.
        """
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        check_batch_dim_name_is_not_a_data_source_member(batch_dim, message_name="batch_dim")
        check_batch_dim_is_not_an_observation_source_name(self.observation_sources, batch_dim)
        batched = np.asarray(
            as_batched_flat(flat_values, self.dimension, message_name="flat_values")
        )
        was_one_vector = is_one_vector(flat_values)
        out: dict[str, xr.DataArray] = {}
        for source in self.observation_sources:
            positions = self.positions(observation_source_name=source.observation_source_name)
            array = _unstacked(source, batched[:, positions], self._index[positions], batch_dim)
            out[source.observation_source_name] = (
                array.isel({batch_dim: 0}, drop=True) if was_one_vector else array
            )
        return out

    # ── evaluation ────────────────────────────────────────────────────────────

    def predict(
        self,
        model_output: ModelOutput,
        *,
        sipnet_parameter_fields: SIPNETParameterFields | None = None,
    ) -> dict[str, Field]:
        """Every observation source's operator applied, converted and checked.

        Parameters
        ----------
        model_output:
            The model output the operators read
            (:data:`~sipnet_calibration.fields.ModelOutput`): one run from
            :func:`sipnet_calibration.fields.to_model_output`, or a stack from
            :func:`sipnet_calibration.fields.stack_model_outputs`, carrying
            every variable in :attr:`output_variable_names` at every site the
            vector observes.
        sipnet_parameter_fields:
            The SIPNET parameter values the runs used, for the operators that
            read any (:attr:`sipnet_parameter_names_read`), as SIPNET
            parameter fields
            (:data:`~sipnet_calibration.fields.SIPNETParameterFields`):
            on ``(*batch, site)`` or ``(site,)``, or with no dim for one run.

        Returns
        -------
        dict
            Fields on each observation source's grid, in its units, with
            the model output's batch dims if any, first; pass the result to
            :meth:`flat` for a ``(J, N)`` batch. A ``NaN`` is allowed only
            where a variable the operators read is ``NaN`` at every timestep for
            that site and batch label (a failed run).

        Raises
        ------
        TypeError
            If *model_output* or *sipnet_parameter_fields* is not an
            ``xr.Dataset``, or an operator returns something other than a
            ``DataArray``.
        ValueError
            If *model_output* is not a model output, or lacks a variable that
            is read; if parameters are read and *sipnet_parameter_fields* is
            not given or are not SIPNET parameter fields; if a prediction is
            not a field on its observation source's grid or carries no
            ``units``; if pySIPNET's ``convert_dataarray_units`` refuses to
            convert a prediction into its source's units and constituent; or
            if a prediction is ``NaN`` at an observation where the run
            succeeded. An operator's own refusals pass through.
        """
        check_model_output_carries_what_is_read(
            model_output,
            output_variable_names=self.output_variable_names,
            sipnet_parameter_names_read=self.sipnet_parameter_names_read,
            sipnet_parameter_fields=sipnet_parameter_fields,
        )
        failed = _failed_runs(model_output, self.output_variable_names)
        return {
            source.observation_source_name: _predicted(
                source, model_output, sipnet_parameter_fields, failed
            )
            for source in self.observation_sources
        }


# ── supporting helpers ────────────────────────────────────────────────────────


def _union(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(chain.from_iterable(groups)))


def _source_restricted_to(
    source: ObservationSource, sites: list[int] | None, time: slice | None
) -> ObservationSource | None:
    """*source* restricted to *sites* and *time*, or ``None`` if no observation is left.

    Rebuilding the observation source drops the time labels no kept site is
    observed at, so an operator run on the selection never reads the model
    there.
    """
    values = source.observed_values
    if sites is not None:
        values = values.isel({SITE: np.flatnonzero(values[SITE].isin(sites).values)})
    if not source.is_static and time is not None:
        values = values.sel({TIME: time})
    if int(values.notnull().sum()) == 0:
        return None
    return ObservationSource(
        observation_source_name=source.observation_source_name,
        observed_values=values,
        operator=source.operator,
    )


def _site_locations_of(
    observation_sources: Sequence[ObservationSource], sites: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The ``lon`` and ``lat`` of each of *sites*, as tuples, from the observed values."""
    lon = np.full(len(sites), np.nan)
    lat = np.full(len(sites), np.nan)
    position = {site: k for k, site in enumerate(sites)}
    for source in observation_sources:
        values = source.observed_values
        for site, x, y in zip(values[SITE].values, values[LON].values, values[LAT].values):
            k = position.get(int(site))
            if k is not None:
                lon[k], lat[k] = x, y
    return tuple(lon.tolist()), tuple(lat.tolist())


def _build_index(observation_sources: Sequence[ObservationSource]) -> pd.MultiIndex:
    frames = []
    for order, source in enumerate(observation_sources):
        observations = source.observations()
        observations[OBSERVATION_SOURCE] = source.observation_source_name
        observations["_order"] = order
        frames.append(observations)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values([SITE, "_order", TIME], kind="stable", na_position="first")
    return pd.MultiIndex.from_arrays(
        [table[SITE].to_numpy(), table[OBSERVATION_SOURCE].to_numpy(), table[TIME].to_numpy()],
        names=INDEX_LEVELS,
    )


def _unstacked(
    source: ObservationSource,
    entries: np.ndarray,
    observation_index: pd.MultiIndex,
    batch_dim: str,
) -> xr.DataArray:
    """One observation source's entries of a ``(J, N)`` batch, shaped as its
    observed values with *batch_dim* first."""
    full = np.full((entries.shape[0], *source.observed_values.shape), np.nan, dtype=np.float64)
    site_positions, time_positions = _positions_in_observed_values(source, observation_index)
    if source.is_static:
        full[:, site_positions] = entries
    else:
        full[:, site_positions, time_positions] = entries
    # An observation source's scalar batch labels are metadata of the input: they
    # would contradict the rows, and one of the batch dim's name would
    # overwrite the batch coordinate made here.
    labels = {*scalar_batch_labels(source.observed_values), batch_dim}
    kept = {name: c for name, c in source.observed_values.coords.items() if name not in labels}
    return xr.DataArray(
        full,
        dims=(batch_dim, *source.observed_values.dims),
        coords={**kept, batch_dim: batch_coordinate(batch_dim, np.arange(entries.shape[0]))},
        attrs=dict(source.observed_values.attrs),
        name=source.observation_source_name,
    )


def _batch_dim_and_labels(
    fields: Mapping[str, xr.DataArray], observation_source_names: Sequence[str]
) -> tuple[str, np.ndarray] | None:
    """The one batch dim the fields share and its labels, or ``None`` if they have none."""
    arrays = {n: fields[n] for n in observation_source_names}
    by_source = {name: batch_dims(array) for name, array in arrays.items()}
    for name, dims in by_source.items():
        check_at_most_one_batch_dim(dims, message_name=f"the field {name!r}")
    with_batch = [n for n, dims in by_source.items() if dims]
    if not with_batch:
        return None
    check_fields_agree_on_the_batch_dim(fields, observation_source_names, with_batch)
    dim = by_source[with_batch[0]][0]
    return dim, fields[with_batch[0]][dim].values


def _read_observations(
    array: xr.DataArray, observation_index: pd.MultiIndex, is_static: bool, batch_dim: str | None
) -> np.ndarray:
    """*array* at the observations *observation_index* holds: ``(J, n)`` or ``(n,)``."""
    selectors: dict[str, Any] = {
        SITE: xr.DataArray(observation_index.get_level_values(SITE).values, dims=_OBSERVATION_DIM)
    }
    if not is_static:
        selectors[TIME] = xr.DataArray(
            observation_index.get_level_values(TIME).values, dims=_OBSERVATION_DIM
        )
    picked = array.sel(selectors)
    if batch_dim is not None:
        picked = picked.transpose(batch_dim, _OBSERVATION_DIM)
    return np.asarray(picked.values, dtype=np.float64)


def _positions_in_observed_values(
    source: ObservationSource, observation_index: pd.MultiIndex
) -> tuple[np.ndarray, np.ndarray]:
    """The (site, time) positions in *source*'s observed values of the observations
    *observation_index* holds."""
    indexes = source.observed_values.indexes
    site_positions = indexes[SITE].get_indexer(observation_index.get_level_values(SITE))
    if source.is_static:
        return site_positions, np.array([], dtype=np.int64)
    return site_positions, indexes[TIME].get_indexer(observation_index.get_level_values(TIME))


def _failed_runs(
    model_output: xr.Dataset, output_variable_names: Sequence[str]
) -> xr.DataArray | None:
    """Where a read variable is NaN at every timestep: a run that failed."""
    if TIME not in model_output.dims:
        return None
    failed = None
    for name in output_variable_names:
        this = model_output[name].isnull().all(TIME)
        failed = this if failed is None else (failed | this)
    return failed


def _predicted(
    source: ObservationSource,
    model_output: xr.Dataset,
    sipnet_parameter_fields: Any,
    failed: xr.DataArray | None,
) -> xr.DataArray:
    """A source's operator applied, checked, converted and checked again."""
    predicted = source.operator(
        model_output, source.observed_values, sipnet_parameter_fields=sipnet_parameter_fields
    )
    check_result_is_on_the_observation_grid(
        predicted, source.observed_values, model_output, source.observation_source_name
    )
    predicted = convert_dataarray_units(
        predicted,
        to_units=source.observed_values.attrs["units"],
        to_constituent=source.observed_values.attrs.get("constituent", "") or "",
    )
    predicted = in_field_layout(predicted)
    validate_field(predicted, message_name=f"{source.observation_source_name}: the prediction")
    check_prediction_is_finite_where_the_run_succeeded(predicted, source, failed)
    predicted.name = source.observation_source_name
    return predicted


# ── checks ────────────────────────────────────────────────────────────────────


def check_observation_vector_is_valid(observation_sources: Sequence[Any]) -> None:
    """The observation sources can make one vector.

    Runs :func:`check_there_is_an_observation_source`,
    :func:`check_entries_are_observation_sources`,
    :func:`~sipnet_calibration.validation.check_names_are_unique` on their names,
    :func:`check_every_observation_source_has_an_observation` and
    :func:`check_observation_sources_agree_on_site_locations`, in that order.
    """
    check_there_is_an_observation_source(observation_sources)
    check_entries_are_observation_sources(observation_sources)
    check_names_are_unique(
        [source.observation_source_name for source in observation_sources],
        message_name="observation_sources",
    )
    check_every_observation_source_has_an_observation(observation_sources)
    check_observation_sources_agree_on_site_locations(observation_sources)


def check_observation_sources_agree_on_site_locations(
    observation_sources: Sequence[ObservationSource],
) -> None:
    """Every observation source gives a site the same ``lon`` and ``lat``, exactly."""
    seen: dict[int, tuple[float, float, str]] = {}
    for source in observation_sources:
        values = source.observed_values
        for site, x, y in zip(values[SITE].values, values[LON].values, values[LAT].values):
            held = seen.setdefault(int(site), (float(x), float(y), source.observation_source_name))
            if held[:2] != (float(x), float(y)):
                raise ValueError(
                    f"site {int(site)} is at ({held[0]}, {held[1]}) in {held[2]!r} and at "
                    f"({float(x)}, {float(y)}) in {source.observation_source_name!r}; a site has "
                    "one location, so locate every observation source from one site table."
                )


def check_there_is_an_observation_source(observation_sources: Sequence[Any]) -> None:
    if not observation_sources:
        raise ValueError(
            "an ObservationVector needs at least one ObservationSource; pass one per "
            "observation source."
        )


def check_entries_are_observation_sources(observation_sources: Sequence[Any]) -> None:
    bad = [
        type(source).__name__
        for source in observation_sources
        if not isinstance(source, ObservationSource)
    ]
    if bad:
        raise TypeError(
            f"every entry must be an ObservationSource, got {bad}; wrap each source's "
            "observed values as ObservationSource(observation_source_name=..., "
            "observed_values=..., operator=...)."
        )


def check_every_observation_source_has_an_observation(
    observation_sources: Sequence[ObservationSource],
) -> None:
    empty = [
        source.observation_source_name
        for source in observation_sources
        if source.n_observations == 0
    ]
    if empty:
        raise ValueError(
            f"{empty} hold no observation; drop them or select sites that were observed."
        )


def check_the_selection_keeps_an_observation(kept: Sequence[ObservationSource]) -> None:
    if not kept:
        raise ValueError(
            "the selection leaves no observation; select sites, observation sources or a "
            "period the vector observes."
        )


def check_sites_are_the_vectors(sites: Sequence[int], held: Sequence[int]) -> None:
    """Every site asked for is observed by the vector."""
    unknown = [site for site in sites if site not in set(held)]
    if unknown:
        raise KeyError(
            f"the vector observes no site(s) {truncated(unknown)}; it observes "
            f"{truncated(list(held))}. Select from its sites, or use "
            "restrict_to_sites to keep the ones it observes."
        )


def check_observation_source_names_are_held(names: Sequence[str], held: Sequence[str]) -> None:
    unknown = [n for n in names if n not in held]
    if unknown:
        what = repr(unknown[0]) if len(unknown) == 1 else str(unknown)
        raise KeyError(f"no observation source {what}; the vector holds {list(held)}.")


def check_time_is_a_slice(time: Any) -> None:
    """``time=`` is ``None`` or a slice of times, with no step."""
    if time is None:
        return
    if not isinstance(time, slice):
        raise TypeError(
            f"time must be a slice such as slice('2012', '2024'), got {type(time).__name__}."
        )
    ends = [end for end in (time.start, time.stop) if end is not None]
    wrong = [end for end in ends if not isinstance(end, (str, np.datetime64, pd.Timestamp, datetime))]
    if wrong:
        raise TypeError(
            f"time slices by times, and got {wrong!r}; pass dates as strings or datetimes, "
            "such as slice('2012', '2024')."
        )
    if time.step is not None:
        raise ValueError(
            f"time must be a slice without a step, got step {time.step!r}; a step would keep "
            "every n-th time label, which selects by position, not by time."
        )


def check_fields_hold_the_observation_sources(
    fields: Any, observation_source_names: Sequence[str]
) -> None:
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"fields must be a mapping from observation source name to DataArray, got "
            f"{type(fields).__name__}; pass what predict or fields returns."
        )
    missing = [n for n in observation_source_names if n not in fields]
    if missing:
        raise ValueError(
            f"fields lack the observation source(s) {missing}; pass an array for every "
            "observation source the vector holds, or select the vector to the sources given."
        )


def check_batch_dim_is_not_an_observation_source_name(
    observation_sources: Sequence[ObservationSource], batch_dim: str
) -> None:
    """*batch_dim* names no observation source nor any coordinate of its values."""
    # A scalar batch label of that name is allowed: it is metadata of the
    # observed input, which fields() drops.
    for source in observation_sources:
        values = source.observed_values
        taken = set(map(str, values.coords)) - set(scalar_batch_labels(values))
        what = (
            "an observation source name"
            if batch_dim == source.observation_source_name
            else f"a coordinate of the observation source {source.observation_source_name!r}"
            if batch_dim in taken
            else None
        )
        if what is not None:
            other = "run" if batch_dim == SAMPLE else SAMPLE
            raise ValueError(
                f"batch_dim={batch_dim!r} is {what}; name the batch dim otherwise, such as "
                f"{other!r}."
            )


def check_fields_agree_on_the_batch_dim(
    fields: Mapping[str, xr.DataArray],
    observation_source_names: Sequence[str],
    with_batch: Sequence[str],
) -> None:
    """Every field carries the same batch dim, with the same labels in the same order."""
    if len(with_batch) != len(observation_source_names):
        raise ValueError(
            f"some fields carry a batch dim ({list(with_batch)}) and others do not; "
            "flatten predictions from one model output at a time."
        )
    first = with_batch[0]
    dim = batch_dims(fields[first])[0]
    labels = fields[first][dim].values
    for name in with_batch[1:]:
        other = batch_dims(fields[name])[0]
        if other != dim:
            raise ValueError(
                f"the fields carry different batch dims ({first!r} {dim!r}, {name!r} "
                f"{other!r}); flatten predictions from one model output at a time."
            )
        if not np.array_equal(fields[name][dim].values, labels):
            raise ValueError(
                f"the fields disagree on their {dim} labels ({first!r} and {name!r}); "
                "flatten predictions from one model output at a time."
            )


def check_field_is_on_the_grid(array: Any, source: ObservationSource) -> None:
    """An array to flatten is a field holding every observation of its source.

    Runs :func:`check_array_is_a_dataarray`, then, on the array laid out as a
    field (:func:`sipnet_calibration.fields.in_field_layout`, so any dim order
    and a scalar ``site`` are accepted),
    :func:`sipnet_calibration.fields.validate_field`,
    :func:`check_array_has_the_observed_sites` and
    :func:`check_array_has_the_observed_time_labels`.
    """
    name = source.observation_source_name
    check_array_is_a_dataarray(array, name)
    array = in_field_layout(array)
    validate_field(array, message_name=name)
    check_array_has_the_observed_sites(array, source)
    check_array_has_the_observed_time_labels(array, source)


def check_array_is_a_dataarray(array: Any, name: str) -> None:
    """An observation source's array is a ``DataArray``."""
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"{name}: expected a DataArray, got {type(array).__name__}; pass the field "
            "predict or fields returns for it."
        )


def check_array_has_the_observed_sites(array: xr.DataArray, source: ObservationSource) -> None:
    """The array has a ``site`` dim holding every site the source observes."""
    name = source.observation_source_name
    if SITE not in array.dims:
        raise ValueError(
            f"{name}: the array has no site; label it with the observation source's sites, "
            "as predict and fields do."
        )
    missing_sites = missing_labels(array, SITE, source.observed_values.indexes[SITE])
    if missing_sites:
        raise ValueError(
            f"{name}: the array lacks observed site(s) {missing_sites[:10]}; predict at "
            "every site the observation source holds, or select the vector to the sites "
            "given."
        )


def check_array_has_the_observed_time_labels(
    array: xr.DataArray, source: ObservationSource
) -> None:
    """For a dated source, the array has a ``time`` dim holding every observed label."""
    if source.is_static:
        return
    name = source.observation_source_name
    if TIME not in array.dims:
        raise ValueError(
            f"{name}: the array has no time dimension; read the model at the "
            "observation source's time labels."
        )
    observed_times = source.observed_values[TIME].values
    missing_times = observed_times[~np.isin(observed_times, array[TIME].values)]
    if missing_times.size:
        raise ValueError(
            f"{name}: the array lacks {missing_times.size} observed time label(s), the "
            f"first being {missing_times[0]}; read the model at the observation source's "
            "time labels, or select the vector to the period given."
        )


def check_prediction_is_finite_where_the_run_succeeded(
    predicted: xr.DataArray, source: ObservationSource, failed: xr.DataArray | None
) -> None:
    observed = source.observed_values.notnull()
    missing = predicted.isnull() & observed
    if failed is not None:
        if SITE in failed.dims:
            allowed = failed.sel({SITE: source.observed_values[SITE].values})
        else:
            allowed = failed
        missing = missing & ~allowed
    if bool(missing.any()):
        where = np.argwhere(missing.values)[0]
        raise ValueError(
            f"{source.observation_source_name}: the prediction is NaN at an observation (first "
            f"at position {where.tolist()} of dims {tuple(missing.dims)}) although the run "
            "succeeded there. A label outside the run, or a gap the operator produced, "
            "must be selected away, not passed on as a failed run."
        )

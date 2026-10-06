"""The observation vector: the observation sources of an experiment, with
the operator that predicts each, as components of a model.

Where this sits
---------------
::

    scripts/ingest_constraints.py                    data/processed/constraints/*.nc
      -> constraints.constraint_fields()             observed values, (site[, time])
      -> ObservationSource(observation_source_name=, observed_values=, operator=)
                                                     one per observation source
                                                     (observation.source)
      -> ObservationVector(observation_sources=[...])
           .predict(model_output, sipnet_parameter_fields=)
                                                     H applied, converted, checked
           .coords, .constants(), .observed_values_by_component()
                                                     the probability layer's view
           .to_fields(labeled)                       labeled values -> Fields
      -> observation.model                           the components themselves

A model of the probability layer is bound at its ``coords``, its noise
factors read its ``constants``, it is conditioned on
``observed_values_by_component()``, and ``to_fields`` turns its labeled
values back into fields. The forward model applies ``predict`` to each run
on the worker. The order of y belongs to the posterior conditioned on the
vector's observed values (``Posterior.gaussian_likelihood().y``), not to the
vector. It imports nothing from the data-source modules, and nothing of the
probability layer: an observation source is built from arrays, and reads
their attributes.

What it reads
-------------
:class:`~sipnet_calibration.observation.source.ObservationSource`\\ s, each
holding one observation source's observed values
(:data:`~sipnet_calibration.observation.source.ObservedValues`, a field on
``(site[, time])``) and its
:class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
**Fields**: ``dict[str, Field]`` keyed by observation source name, each a
``float64`` field on that source's own ``(site[, time])`` grid, with its
``site``, ``lon``/``lat`` and time coordinates, ``NaN`` where nothing is
observed, and any batch dims first. A source's grid holds only the sites
and time labels with at least one observation: the rest are dropped when its
observation source is built. A dict and not a Dataset, since each source is
on its own time grid.

**The site table**: ``site_table``, one row per observed site, ascending:
``site_id`` (``int32``), ``lon`` and ``lat``, read off the observed values,
which every observation source must agree on.

**As components of a model.** For each source ``k``, by name, never typed by
a user:

- its **observation dim**, ``"<k>_observation"``
  (:data:`OBSERVATION_DIM_SUFFIX`), a stacked dim whose labels are the
  source's ``observation_labels``: a ``MultiIndex`` of the ``(site, time)``
  pairs it observes, or ``(site,)`` for a static source, sorted by site,
  then time; ``coords`` holds one per source;
- its **observed component**, named ``k``, on that dim, whose values
  ``observed_values_by_component()`` gives;
- its **prediction**, ``"predicted_<k>"`` (:data:`PREDICTION_PREFIX`), the
  forward model's value of the observed quantity, on the same dim and in the
  same units;
- its **constants**, ``float64`` on that dim: :data:`OBSERVED`,
  :data:`STANDARD_DEVIATION` when the source has one, and for a dated source
  :data:`TIME_SINCE_EPOCH` and :data:`CALENDAR_YEAR`, and for one with
  windows :data:`WINDOW_LENGTH`; and its year label map (:data:`YEAR`).

The components themselves are :mod:`~sipnet_calibration.observation.model`'s.

Functions
---------
:class:`ObservationVector`
    The vector, whose pieces are its observation sources: ``select``,
    ``restrict_to_sites`` and ``with_observed_values``; ``coords``,
    ``observation_dim_name``, ``prediction_name``, ``constants``,
    ``year_label_map``, ``observed_values_by_component`` and ``to_fields``
    for a model; ``predict``, every operator applied, converted and checked.

Notes
-----
**By site, then time.** A likelihood that factorizes over sites, or an error
covariance block-diagonal by site, reads a site's observations of a source
as one contiguous segment of its observation dim, and a worker producing one
site's predictions produces that segment.

**Selection never reorders.** ``select`` keeps the vector's order of sources
and sites whatever order they are asked in, and refuses a name or site the
vector does not hold, so a typo cannot silently shrink it;
``restrict_to_sites`` is the form for a caller holding a larger site set.

**No noise model here.** The covariance and the likelihood are a noise
factor's; this module gives it the observation dims and the constants,
standard deviations included.

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
    observed = constraint_fields(names, sites=[3851, 3871])  # sites both observe
    vector = ObservationVector(observation_sources=[
        ObservationSource(
            observation_source_name="modis_leaf_area_index",
            observed_values=observed["modis_leaf_area_index"],
            operator=DEFAULT_OBS_OPS["modis_leaf_area_index"],
        ),
        ObservationSource(
            observation_source_name="landtrendr_aboveground_biomass",
            observed_values=observed["landtrendr_aboveground_biomass"],
            operator=ReduceOverWindows("wood_carbon", "mean"),  # the experiment's
        ),
    ]).select(time=slice("2012", "2024"))

    predicted_fields = vector.predict(
        model_output, sipnet_parameter_fields=sipnet_parameter_fields
    )                                               # one field per source

    # For a model of the probability layer:
    vector.coords                                   # {"modis_leaf_area_index_observation": MultiIndex, ...}
    vector.constants("modis_leaf_area_index")       # {"observed": ..., "time_since_epoch": ..., ...}
    vector.prediction_name("modis_leaf_area_index") # "predicted_modis_leaf_area_index"
    observed = vector.observed_values_by_component()   # what condition_on observes
    vector.to_fields(observed)["modis_leaf_area_index"]   # back to (site, time)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from frozendict import frozendict
from pysipnet.units import convert_dataarray_units

from sipnet_calibration.conventions import (
    LAT,
    LON,
    SITE,
    SITE_DTYPE,
    SITE_ID,
    TIME,
    WINDOW_END,
    WINDOW_START,
)
from sipnet_calibration.fields import (
    Field,
    ModelOutput,
    SIPNETParameterFields,
    batch_coordinate,
    in_field_layout,
    batch_dims,
    validate_field,
    scalar_batch_labels,
)
from sipnet_calibration.observation.operators import (
    check_model_output_carries_what_is_read,
    check_result_is_on_the_observation_grid,
)
from sipnet_calibration.observation.source import ObservationSource
from sipnet_calibration.validation import (
    as_names,
    as_sequence,
    as_site_ids,
    check_names_are_unique,
    check_sites_are_the_vectors,
    check_the_restriction_keeps_a_site,
    truncated,
)

__all__ = [
    "CALENDAR_YEAR",
    "OBSERVATION_DIM_SUFFIX",
    "OBSERVED",
    "PREDICTION_PREFIX",
    "STANDARD_DEVIATION",
    "TIME_SINCE_EPOCH",
    "WINDOW_LENGTH",
    "YEAR",
    "ObservationVector",
]

OBSERVATION_SOURCE = "observation_source"

#: An observation dim is named ``"<source>" + OBSERVATION_DIM_SUFFIX``.
OBSERVATION_DIM_SUFFIX = "_observation"

#: A prediction is named ``PREDICTION_PREFIX + "<source>"``.
PREDICTION_PREFIX = "predicted_"

# The constants of ObservationVector.constants, each on a source's observation dim.
#: The constant holding the observed values.
OBSERVED = "observed"
#: The constant holding their measurement-error standard deviations.
STANDARD_DEVIATION = "standard_deviation"
#: The constant holding each observation's time label, in seconds since
#: 1970-01-01 UTC.
TIME_SINCE_EPOCH = "time_since_epoch"
#: The constant holding each observation's calendar year, as a number.
CALENDAR_YEAR = "calendar_year"
#: The constant holding the length of each observation's window, in seconds.
WINDOW_LENGTH = "window_length"

#: The target of :meth:`ObservationVector.year_label_map`: an element axis
#: whose labels are years as strings.
YEAR = "year"

#: The origin of :data:`TIME_SINCE_EPOCH`.
_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")


@dataclass(frozen=True, eq=False, kw_only=True)
class ObservationVector:
    """The observations of several observation sources, as the module
    docstring describes them.

    Parameters
    ----------
    observation_sources:
        One :class:`~sipnet_calibration.observation.source.ObservationSource`
        per observation source, in the order of the vector's components;
        stored as a tuple.

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
    _sites: tuple[int, ...] = field(init=False, repr=False)
    _locations: tuple[tuple[float, ...], tuple[float, ...]] = field(init=False, repr=False)
    _coords: Mapping[str, pd.MultiIndex] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observation_sources = tuple(
            as_sequence(self.observation_sources, message_name="observation_sources")
        )
        check_observation_vector_is_valid(observation_sources)
        object.__setattr__(self, "observation_sources", observation_sources)
        object.__setattr__(
            self,
            "_by_name",
            frozendict({source.observation_source_name: source for source in observation_sources}),
        )
        sites = tuple(
            int(s)
            for s in np.unique(
                np.concatenate([s.observation_labels.get_level_values(SITE) for s in observation_sources])
            )
        )
        object.__setattr__(self, "_sites", sites)
        object.__setattr__(self, "_locations", _site_locations_of(observation_sources, sites))
        object.__setattr__(
            self,
            "_coords",
            frozendict({_observation_dim_name(s): s.observation_labels for s in observation_sources}),
        )

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
        """The site table of :attr:`sites`: ``site_id`` (``int32``), ``lon``, ``lat``.

        A new table on every access, read off the observed values'
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
            f"ObservationVector(observations={sum(len(s.observation_labels) for s in self.observation_sources)}, "
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
            labels. Its sources keep this vector's order, and each source's
            observations their order: a selection never reorders what it
            keeps.

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
            check_sites_are_the_vectors(wanted_sites, self._sites, message_name="the vector")
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
        check_the_restriction_keeps_a_site(kept, message_name="the vector")
        return self.select(sites=kept)

    def with_observed_values(self, values: Mapping[str, Any]) -> ObservationVector:
        """This vector with other observed values at the same observations,
        such as synthetic data for a twin experiment, so that the constants a
        noise factor reads (:data:`OBSERVED`) follow them.

        Parameters
        ----------
        values:
            ``{observation source name: observed values}``
            (:data:`~sipnet_calibration.observation.source.ObservedValues`),
            for some or all of the sources; each source keeps its operator
            and standard deviation, and a source not named keeps its values.

        Returns
        -------
        ObservationVector

        Raises
        ------
        TypeError
            If *values* is not a mapping, or a value is not a ``DataArray``.
        KeyError
            If a source is not in the vector.
        ValueError
            If a source's values are not observed values, observe other
            ``(site[, time])`` pairs than the source does, or differ from
            its values in another coordinate (a location, a window) or in
            units.
        """
        check_values_are_a_mapping(values, "values")
        check_observation_source_names_are_held(list(values), self.observation_source_names)
        sources = []
        for source in self.observation_sources:
            if source.observation_source_name not in values:
                sources.append(source)
                continue
            replaced = ObservationSource(
                observation_source_name=source.observation_source_name,
                observed_values=values[source.observation_source_name],
                operator=source.operator,
            )
            check_values_observe_the_same_labels(replaced, source)
            if source.standard_deviation is not None:
                replaced = ObservationSource(
                    observation_source_name=source.observation_source_name,
                    observed_values=replaced.observed_values,
                    operator=source.operator,
                    standard_deviation=source.standard_deviation,
                )
            sources.append(replaced)
        return ObservationVector(observation_sources=sources)

    # ── coordinates ───────────────────────────────────────────────────────────

    @property
    def coords(self) -> frozendict:
        """``{observation dim name: pandas.MultiIndex}``, one per source in
        the vector's order, its labels :attr:`ObservationSource.observation_labels
        <sipnet_calibration.observation.source.ObservationSource.observation_labels>`:
        what a model's components on the observations are bound at."""
        return frozendict({dim: labels.copy() for dim, labels in self._coords.items()})

    def observation_dim_name(self, observation_source_name: str) -> str:
        """The name of a source's observation dim,
        ``"<source>" + OBSERVATION_DIM_SUFFIX``.

        Raises
        ------
        KeyError
            If the source is not in the vector.
        """
        return _observation_dim_name(self[observation_source_name])

    def prediction_name(self, observation_source_name: str) -> str:
        """The name of a source's prediction, ``PREDICTION_PREFIX + "<source>"``:
        the forward model's value of the observed quantity, on the source's
        observation dim and in its units.

        Raises
        ------
        KeyError
            If the source is not in the vector.
        """
        return f"{PREDICTION_PREFIX}{self[observation_source_name].observation_source_name}"

    def constants(self, observation_source_name: str) -> dict[str, xr.DataArray]:
        """A source's constants, ``float64`` on its observation dim, in the
        order below: what a noise factor's functions read.

        - :data:`OBSERVED`, the observed values, with their attributes;
        - :data:`STANDARD_DEVIATION`, when the source has one, with its
          attributes;
        - for a dated source, :data:`TIME_SINCE_EPOCH`, each time label in
          seconds since 1970-01-01 UTC, and :data:`CALENDAR_YEAR`, its year;
        - for a source with windows, :data:`WINDOW_LENGTH`, each window's
          length in seconds.

        Raises
        ------
        KeyError
            If the source is not in the vector.
        """
        source = self[observation_source_name]
        values = {OBSERVED: (_at_observations(source, source.observed_values), source.observed_values.attrs)}
        if source.standard_deviation is not None:
            sd = source.standard_deviation
            values[STANDARD_DEVIATION] = (_at_observations(source, sd), sd.attrs)
        if not source.is_static:
            times = source.observation_labels.get_level_values(TIME).to_numpy("datetime64[ns]")
            values[TIME_SINCE_EPOCH] = (_seconds(times - _EPOCH), {"units": "s"})
            values[CALENDAR_YEAR] = (pd.DatetimeIndex(times).year.to_numpy(np.float64), {"units": "1"})
        if WINDOW_START in source.observed_values.coords:
            length = source.observed_values[WINDOW_END] - source.observed_values[WINDOW_START]
            values[WINDOW_LENGTH] = (_seconds(_at_observations(source, length, dtype=None)), {"units": "s"})
        return {
            name: _on_the_observation_dim(source, array, name=name, attrs=attrs)
            for name, (array, attrs) in values.items()
        }

    def year_label_map(self, observation_source_name: str) -> xr.DataArray:
        """A label map named :data:`YEAR` on the source's observation dim:
        each observation's calendar year, as a string label, such as
        ``"2012"``.

        Raises
        ------
        KeyError
            If the source is not in the vector.
        ValueError
            If the source is static.
        """
        source = self[observation_source_name]
        check_source_is_dated(source)
        years = source.observation_labels.get_level_values(TIME).year.astype(str).to_numpy(object)
        return _on_the_observation_dim(source, years, name=YEAR, attrs={})

    # ── representations ───────────────────────────────────────────────────────

    def observed_values_by_component(self) -> dict[str, xr.DataArray]:
        """``{observation source name: its observed values on its
        observation dim}``, ``float64`` with their attributes, in the vector's
        order: what ``condition_on`` observes, each source's observed
        component being named for it."""
        return {
            name: self.constants(name)[OBSERVED].rename(name) for name in self.observation_source_names
        }

    def to_fields(self, labeled: Mapping[str, xr.DataArray]) -> dict[str, Field]:
        """Labeled values on the vector's observation dims as fields.

        Parameters
        ----------
        labeled:
            ``{name: xr.DataArray}``, such as labeled values
            (:data:`~sipnet_calibration.probability.layout.LabeledValues`)
            holding an observed component, a prediction or a residual. Each
            array on one of the vector's observation dims, its labels some of
            that source's observations, becomes a field; its other dims must
            be batch dims (integer labels). An array on none of them is
            skipped.

        Returns
        -------
        dict
            ``{name: field}``, in *labeled*'s order: ``float64`` on
            ``(*batch, site[, time])``, the source's grid with its
            coordinates and attributes, ``NaN`` where the array has no value.

        Raises
        ------
        TypeError
            If *labeled* is not a mapping, or a value is not a ``DataArray``.
        ValueError
            If an array is on two observation dims, on a dim that is not a
            batch dim, labels its observation dim with other levels than
            the source's, or holds an observation twice.
        KeyError
            If an array holds a label the source does not observe.
        """
        check_values_are_a_mapping(labeled, "labeled")
        sources_by_dim = {_observation_dim_name(s): s for s in self.observation_sources}
        out: dict[str, xr.DataArray] = {}
        for name, array in labeled.items():
            check_array_is_a_dataarray(array, str(name))
            dims = [str(d) for d in array.dims if d in sources_by_dim]
            if not dims:
                continue
            check_array_is_on_one_observation_dim(str(name), dims)
            out[name] = _unstacked_from_observation_dim(sources_by_dim[dims[0]], array, dims[0], str(name))
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
            The SIPNET parameter values the runs used
            (:data:`~sipnet_calibration.fields.SIPNETParameterFields`), for
            the operators that read any (:attr:`sipnet_parameter_names_read`).

        Returns
        -------
        dict
            Fields on each observation source's grid, in its units, with
            the model output's batch dims if any, first. A ``NaN`` is allowed only
            where a variable the operators read is ``NaN`` at every timestep for
            that site and batch label (a failed run).

        Raises
        ------
        TypeError
            If an input is not an ``xr.Dataset``, or an operator returns
            something other than a ``DataArray``.
        ValueError
            If the inputs are not what the operators read
            (:func:`~sipnet_calibration.observation.operators.check_model_output_carries_what_is_read`),
            or a prediction is not on its source's grid, cannot be converted
            into its units, or is ``NaN`` where the run succeeded. An
            operator's own refusals pass through.
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
    arrays = [source.observed_values, source.standard_deviation]
    if sites is not None:
        positions = np.flatnonzero(source.observed_values[SITE].isin(sites).values)
        arrays = [None if a is None else a.isel({SITE: positions}) for a in arrays]
    if not source.is_static and time is not None:
        arrays = [None if a is None else a.sel({TIME: time}) for a in arrays]
    values, standard_deviation = arrays
    if int(values.notnull().sum()) == 0:
        return None
    return ObservationSource(
        observation_source_name=source.observation_source_name,
        observed_values=values,
        operator=source.operator,
        standard_deviation=standard_deviation,
    )


def _observation_dim_name(source: ObservationSource) -> str:
    return f"{source.observation_source_name}{OBSERVATION_DIM_SUFFIX}"


def _at_observations(source: ObservationSource, array: xr.DataArray, dtype: Any = np.float64) -> np.ndarray:
    """*array*, on or broadcast to *source*'s grid, at its observations, in
    the order of :attr:`ObservationSource.observation_labels`."""
    grid = source.observed_values
    values = array.broadcast_like(grid).transpose(*grid.dims).values
    return np.asarray(values, dtype=dtype)[grid.notnull().values]


def _seconds(durations: np.ndarray) -> np.ndarray:
    """``timedelta64`` durations in seconds, ``float64``."""
    return durations.astype("timedelta64[ns]").astype(np.int64) / 1e9


def _on_the_observation_dim(
    source: ObservationSource, values: np.ndarray, *, name: str, attrs: Mapping[str, Any]
) -> xr.DataArray:
    """*values*, one per observation, as a DataArray on *source*'s
    observation dim, labeled by its observations."""
    dim = _observation_dim_name(source)
    return xr.DataArray(
        values,
        dims=(dim,),
        coords=xr.Coordinates.from_pandas_multiindex(source.observation_labels, dim),
        name=name,
        attrs=dict(attrs),
    )


def _unstacked_from_observation_dim(
    source: ObservationSource, array: xr.DataArray, dim: str, name: str
) -> xr.DataArray:
    """*array*, on *source*'s observation dim and batch dims, as a field on
    the source's grid, ``NaN`` where it has no value."""
    batch = [str(d) for d in array.dims if d != dim]
    for batch_dim in batch:
        check_dim_is_a_batch_dim(name, array, batch_dim)
    labels = array.indexes.get(dim)
    check_observation_levels_are_the_sources(name, labels, source)
    check_observation_labels_are_unique(name, labels)
    grid = source.observed_values
    site_positions = grid.indexes[SITE].get_indexer(labels.get_level_values(SITE))
    positions = [site_positions]
    if not source.is_static:
        positions.append(grid.indexes[TIME].get_indexer(labels.get_level_values(TIME)))
    check_labels_are_observations(name, labels, positions, source)
    laid_out = array.transpose(*batch, dim)
    sizes = tuple(laid_out.sizes[d] for d in batch)
    full = np.full((*sizes, *grid.shape), np.nan, dtype=np.float64)
    full[(..., *positions)] = np.asarray(laid_out.values, dtype=np.float64)
    labels_kept = {*scalar_batch_labels(grid), *batch}
    kept = {key: c for key, c in grid.coords.items() if key not in labels_kept}
    batch_coords = {d: batch_coordinate(d, array.indexes[d].to_numpy()) for d in batch}
    field = xr.DataArray(
        full, dims=(*batch, *grid.dims), coords={**kept, **batch_coords}, attrs=dict(grid.attrs), name=name
    )
    validate_field(field, message_name=name)
    return field


def _site_locations_of(
    observation_sources: Sequence[ObservationSource], sites: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The ``lon`` and ``lat`` of each of *sites*, as tuples, from the sources."""
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
    """The observation sources can make one vector."""
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


def check_array_is_a_dataarray(array: Any, name: str) -> None:
    """An observation source's array is a ``DataArray``."""
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"{name}: expected a DataArray, got {type(array).__name__}; pass labeled values, one "
            "DataArray per component."
        )


def check_values_are_a_mapping(values: Any, message_name: str) -> None:
    """Values by source or by name are a mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(
            f"{message_name} must be a mapping {{name: DataArray}}, got {type(values).__name__}; pass a dict."
        )


def check_values_observe_the_same_labels(replaced: ObservationSource, source: ObservationSource) -> None:
    """Replacing a source's observed values keeps its observations, their
    coordinates (locations, windows) and units, so that what was bound at
    them, and the constants read from them, still apply."""
    name = source.observation_source_name
    if not replaced.observation_labels.equals(source.observation_labels):
        raise ValueError(
            f"{name}: the new observed values observe other (site, time) pairs than the source's "
            f"{source.n_observations}; give values at the same observations, NaN elsewhere."
        )
    new, old = replaced.observed_values, source.observed_values
    differing = [
        str(c) for c in sorted(set(new.coords) | set(old.coords), key=str)
        if c not in new.coords or c not in old.coords or not new[c].equals(old[c])
    ]
    differing += [a for a in ("units", "constituent") if new.attrs.get(a) != old.attrs.get(a)]
    if differing:
        raise ValueError(
            f"{name}: the new observed values differ from the source's in {differing}; give values "
            "with the source's coordinates and units, changing only the values."
        )


def check_source_is_dated(source: ObservationSource) -> None:
    """A source read by time has a time."""
    if source.is_static:
        raise ValueError(
            f"{source.observation_source_name} is static, so its observations have no year; ask for "
            "the years of a dated source."
        )


def check_array_is_on_one_observation_dim(name: str, dims: Sequence[str]) -> None:
    """An array to unstack is on one observation dim, one source's."""
    if len(dims) > 1:
        raise ValueError(
            f"{name} is on the observation dims {dims}; a source's values are on its own dim alone."
        )


def check_dim_is_a_batch_dim(name: str, array: xr.DataArray, dim: str) -> None:
    """An array's dim besides its observation dim is a batch dim: labeled
    by integers."""
    if dim not in batch_dims(array):
        raise ValueError(
            f"{name} is on {dim!r}, which is neither its observation dim nor a batch dim (integer "
            "labels); select or reduce it first."
        )


def check_observation_levels_are_the_sources(
    name: str, labels: pd.Index | None, source: ObservationSource
) -> None:
    """An array labels its observation dim with the source's levels."""
    expected = list(source.observation_labels.names)
    found = list(labels.names) if isinstance(labels, pd.MultiIndex) else None
    if found != expected:
        raise ValueError(
            f"{name} labels its observation dim with {found or 'a plain index, or none'}, the source "
            f"{source.observation_source_name!r} with {expected}; label it with the vector's coords."
        )


def check_observation_labels_are_unique(name: str, labels: pd.MultiIndex) -> None:
    """An array to unstack holds each observation once, so that one value is
    placed at each."""
    if not labels.is_unique:
        repeated = labels[labels.duplicated()].unique()
        raise ValueError(
            f"{name} holds the observation(s) {truncated([str(r) for r in repeated])} more than once; "
            "keep one value per observation."
        )


def check_labels_are_observations(
    name: str, labels: pd.MultiIndex, positions: Sequence[np.ndarray], source: ObservationSource
) -> None:
    """Every label of an array to unstack is an observation of its source."""
    grid = source.observed_values.notnull().values
    found = np.logical_and.reduce([p >= 0 for p in positions])
    observed = np.zeros(len(labels), dtype=bool)
    observed[found] = grid[tuple(p[found] for p in positions)]
    unknown = labels[~observed]
    if len(unknown):
        raise KeyError(
            f"{name} holds {truncated([str(u) for u in unknown])}, which are not observations of "
            f"{source.observation_source_name!r}; give values at its observations only."
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

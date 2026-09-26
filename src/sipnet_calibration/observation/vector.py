"""The observation vector: the observed cells of an experiment in a fixed
order, with the operator that predicts each product, and the two forms its
values take.

Where this sits
---------------
::

    scripts/ingest_constraints.py                    data/processed/constraints/*.nc
      -> constraints.constraint_fields()             observed arrays, (site[, time])
      -> Observation(product_name, values, operator) one per product
      -> ObservationVector([...])                    the cells, in Flat order
           .y                                        what pyEKI compares against
           .predict(model_output, sipnet_parameters=)
                                                     H applied, converted, checked
           .flat(fields) / .fields(flat_values)      Fields <-> Flat

The forward model flattens predictions with it into pyEKI's ``(J, N)``; the
inference layer reads ``y``, ``index`` and ``positions`` off it to build the
error model; a predictive-check figure unstacks a ``(J, N)`` block with
``fields``. It imports nothing from the product modules: an observation is
built from arrays, and reads their attributes.

What it reads
-------------
:attr:`Observation.values`
    A field ``(site[, time])`` with ``NaN`` where nothing was observed, finite
    elsewhere, integer ``site`` labels, naive datetime ``time`` labels, and
    ``units`` (and ``constituent``, where the quantity has one) in its
    attributes. An annual product's array also carries its ``time_bounds``
    as ``time_bounds_start``/``time_bounds_end``. A batch dim is refused: an
    observation ensemble is reduced to one value per cell by the experiment
    before it enters the vector. A scalar coordinate, such as one left by
    ``.isel(sample=0)``, is metadata and is kept.
:attr:`Observation.operator`
    An :class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
**Fields**: ``dict[str, xr.DataArray]`` keyed by product name, each ``float64``
on that product's own ``(site[, time])`` grid, ``NaN`` at unobserved cells; a
``(J, N)`` batch unstacks to the same with a leading batch dim, ``sample``
unless ``batch_dim=`` names it otherwise, an ``int64`` coordinate labeled
``0`` to ``J - 1``. A product's grid holds only
the sites and time labels with at least one observed cell: the rest are
dropped when its :class:`Observation` is built.

**Flat**: ``y`` in ``R^N``, ``float64``, finite, one entry per observed cell;
predictions are ``(N,)`` or ``(J, N)``, where a ``NaN`` marks a failed run.

**The index**: a ``pandas.MultiIndex`` with levels ``(site, product, time)``
-- ``site`` ``int64``, ``product`` a string, ``time`` ``datetime64[ns]`` --
one row per observed cell, in Flat order: **site-major**, sites ascending,
then products in declaration order, then times ascending. A static product's
cells have ``time = NaT``. The order is a property of the vector; consumers
read it off ``index`` and ``positions`` and never assume it.

Functions
---------
:class:`Observation`
    One product's values and operator; ``sites``, ``is_static``, ``n_cells``
    and ``cells()`` (the observed cells as a frame).
:class:`ObservationVector`
    ``product_names``, ``sites``, ``dimension``, ``index``,
    ``output_variable_names`` and ``sipnet_parameter_names`` (the union over
    the operators), ``observed_values`` (Fields) and ``y`` (Flat);
    ``select(product_names=, sites=, time=)`` for a sub-vector;
    ``positions(site=, product_name=)`` for where a block sits in Flat;
    ``flat(fields)`` and ``fields(flat_values)`` between the
    representations; ``predict(model_output, sipnet_parameters=)`` for every
    operator applied, converted and checked; ``describe()`` for one row per
    product.

Notes
-----
**Site-major.** A likelihood that factorizes over sites, or an error
covariance block-diagonal by site, reads a site's cells as one contiguous
block, and a worker producing one site's predictions produces that block.
``(site, product)`` sub-blocks are contiguous too.

**No noise model here.** The products' standard deviations, the covariance
and the likelihood are the inference layer's; this module gives it ``y``,
``index`` and ``positions``.

**Failed runs.** ``predict`` passes a ``NaN`` through where the model output
itself is ``NaN`` at that site and batch label (a failed run), and refuses one
anywhere else, so a coverage gap cannot masquerade as a failed run and be
repaired away.

**Only observed labels.** An observation keeps only the sites and time labels
it observes, so an operator reads the model nowhere else: not at a site the
model output need not carry, and not at a label outside a shorter run.

**Frozen values.** An observation holds its own read-only copy of the values
it was given, loaded into memory, so neither a later write to the caller's
array nor one through :attr:`ObservationVector.observed_values` can change the
cells after ``y`` and the index were built from them.

Usage
-----
::

    from sipnet_calibration.constraints import constraint_fields
    from sipnet_calibration.observation import (
        DEFAULT_OBS_OPS, Observation, ObservationVector, ReduceOverTimeBounds,
    )

    names = ["modis_leaf_area_index", "landtrendr_aboveground_biomass"]
    observed = constraint_fields(names, sites=sites)
    vector = ObservationVector([
        Observation("modis_leaf_area_index", observed["modis_leaf_area_index"],
                    DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        Observation("landtrendr_aboveground_biomass",
                    observed["landtrendr_aboveground_biomass"],
                    ReduceOverTimeBounds("wood_carbon", "mean")),  # the experiment's reading
    ]).select(time=slice("2012", "2024"))

    vector.dimension, vector.y.shape           # N, (N,)
    predicted_fields = vector.predict(model_output, sipnet_parameters=sipnet_table)
    predictions = vector.flat(predicted_fields)  # (J, N) for a (sample, site, time) output
    vector.fields(predictions)["modis_leaf_area_index"]  # back to (sample, site, time)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.units import convert_dataarray_units, validate_units

from sipnet_calibration.conventions import SAMPLE, SITE, TIME
from sipnet_calibration.fields import (
    batch_coordinate,
    batch_dims,
    check_at_most_one_batch_dim,
    check_batch_dim_name_is_not_reserved,
    check_dims_are_batch_spatial_or_time,
    missing_labels,
    scalar_batch_labels,
)
from sipnet_calibration.observation.operators import (
    ObservationOperator,
    check_model_output_carries_what_is_read,
    check_operator_declares_names,
    check_result_is_on_the_observation_grid,
)
from sipnet_calibration.validation import (
    as_batched_flat,
    as_names,
    as_site_id,
    as_site_ids,
    is_one_vector,
)

__all__ = ["INDEX_LEVELS", "Observation", "ObservationVector"]

PRODUCT = "product"

#: The levels of :attr:`ObservationVector.index`, in order.
INDEX_LEVELS: tuple[str, ...] = (SITE, PRODUCT, TIME)

#: The dim the vectorized read of the observations indexes along, one per
#: observation; internal to ``flat`` and never on a result.
_OBSERVATION_DIM = "__observation__"


@dataclass(frozen=True, eq=False)
class Observation:
    """One observed product, with the rule that predicts it.

    Parameters
    ----------
    product_name:
        The product's name, a non-empty string; the observation's values are
        renamed to it.
    values:
        The observed field, ``(site,)`` or ``(site, time)``, as described in
        the module docstring. It is copied and loaded into memory, ordered by
        ascending ``site`` and ``time``, trimmed to the sites and time labels
        holding at least one observed cell, and stored read-only.
    operator:
        The :class:`~sipnet_calibration.observation.operators.ObservationOperator`
        that predicts the product.

    Raises
    ------
    TypeError
        If *product_name* is not a string, *values* is not a ``DataArray``,
        or *operator* is not callable or declares its names other than as
        tuples of strings.
    ValueError
        If *product_name* is empty; if *values* has dims other than
        ``(site[, time])`` (a batch dim among them),
        non-integer or repeated ``site`` labels, non-numeric values, an
        infinite value, no ``units`` attribute or units pySIPNET's
        ``validate_units`` refuses (``'g C m-2'``, whose substance belongs in
        ``constituent``), or ``time`` labels that are not naive datetimes,
        repeat or hold ``NaT``; or if the operator declares an alias.
    KeyError
        If the operator declares a name pySIPNET does not know.

    Notes
    -----
    Observations compare and hash by identity: two built from equal arrays
    are two observations, since an array has no single truth value to
    compare by.
    """

    product_name: str
    values: xr.DataArray
    operator: ObservationOperator

    def __post_init__(self) -> None:
        check_product_name_is_a_nonempty_string(self.product_name)
        check_values_are_a_field(self.values, self.product_name)
        check_observed_values_are_finite_or_nan(self.values, self.product_name)
        check_values_have_units(self.values, self.product_name)
        check_operator_declares_names(self.operator, self.product_name)
        values = _observed_labels_only(_ordered(self.values))
        object.__setattr__(self, "values", _frozen(values, self.product_name))

    @property
    def sites(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.values[SITE].values)

    @property
    def is_static(self) -> bool:
        return TIME not in self.values.dims

    @property
    def n_cells(self) -> int:
        return int(self.values.notnull().sum())

    def __repr__(self) -> str:
        return f"Observation({self.product_name!r}, {self.n_cells} cells, {self.operator!r})"

    def cells(self) -> pd.DataFrame:
        """The observed cells as a frame with ``site``, ``time`` and ``value`` columns."""
        mask = self.values.notnull()
        site_labels = self.values[SITE].values
        if not self.is_static:
            site_labels = site_labels[:, None]
        sites = np.broadcast_to(site_labels, self.values.shape)
        if self.is_static:
            times = np.full(self.values.shape, np.datetime64("NaT", "ns"))
        else:
            times = np.broadcast_to(self.values[TIME].values[None, :], self.values.shape)
        chosen = mask.values
        return pd.DataFrame(
            {
                SITE: sites[chosen].astype(np.int64),
                TIME: times[chosen].astype("datetime64[ns]"),
                "value": self.values.values[chosen].astype(np.float64),
            }
        )


class ObservationVector:
    """The observed cells of several products in a fixed order; see the module docstring.

    Parameters
    ----------
    observations:
        One :class:`Observation` per product, in the order the products take
        within each site's block.

    Raises
    ------
    TypeError
        If an entry is not an :class:`Observation`.
    ValueError
        If there is no observation, two share a product name, or one holds no
        observed cell.
    """

    def __init__(self, observations: Sequence[Observation]) -> None:
        observations = tuple(observations)
        check_there_is_an_observation(observations)
        check_entries_are_observations(observations)
        check_product_names_are_unique(observations)
        check_every_observation_has_an_observed_cell(observations)
        self._observations = observations
        self._by_name = {observation.product_name: observation for observation in observations}
        self._index = _build_index(observations)
        self._sites = tuple(int(s) for s in np.unique(self._index.get_level_values(SITE)))
        self._y = self.flat(self.observed_values)

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self._observations

    @property
    def product_names(self) -> tuple[str, ...]:
        return tuple(observation.product_name for observation in self._observations)

    @property
    def sites(self) -> tuple[int, ...]:
        """The sites with at least one observed cell, ascending."""
        return self._sites

    @property
    def dimension(self) -> int:
        return len(self._index)

    @property
    def index(self) -> pd.MultiIndex:
        return self._index

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return _union(
            observation.operator.output_variable_names for observation in self._observations
        )

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return _union(
            observation.operator.sipnet_parameter_names for observation in self._observations
        )

    @property
    def observed_values(self) -> dict[str, xr.DataArray]:
        return {observation.product_name: observation.values for observation in self._observations}

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()

    def __getitem__(self, product_name: str) -> Observation:
        check_product_names_are_held([product_name], self.product_names)
        return self._by_name[product_name]

    def __repr__(self) -> str:
        return (
            f"ObservationVector(N={self.dimension}, products={list(self.product_names)}, "
            f"sites={len(self.sites)})"
        )

    def describe(self) -> pd.DataFrame:
        """One row per product: cells, sites, time span, units, operator."""
        rows = []
        for observation in self._observations:
            cells = observation.cells()
            rows.append(
                {
                    PRODUCT: observation.product_name,
                    "cells": len(cells),
                    "sites": int(cells[SITE].nunique()),
                    "first_time": None if observation.is_static else cells[TIME].min(),
                    "last_time": None if observation.is_static else cells[TIME].max(),
                    "units": observation.values.attrs["units"],
                    "constituent": observation.values.attrs.get("constituent", ""),
                    "operator": repr(observation.operator),
                }
            )
        return pd.DataFrame(rows).set_index(PRODUCT)

    # ── selection ─────────────────────────────────────────────────────────────

    def select(
        self,
        *,
        product_names: Sequence[str] | None = None,
        sites: Iterable[int] | None = None,
        time: slice | None = None,
    ) -> ObservationVector:
        """A sub-vector over some products, sites and a time slice.

        Parameters
        ----------
        product_names:
            The products to keep, a sequence, in the order the sub-vector
            takes them. Defaults to every product.
        sites:
            The sites to keep, a sequence of site ids, which is read once.
            Sites the vector does not observe are ignored. Defaults to every
            site.
        time:
            A slice of ``time`` labels, such as ``slice("2012", "2024")``. It
            does not apply to a static product.

        Returns
        -------
        ObservationVector
            Every observed cell that survives all three filters. A product left
            with no cells is dropped, and each product's ``time`` keeps only the
            labels where a kept site is observed, so an operator run on the
            sub-vector reads the model only at those labels.

        Raises
        ------
        TypeError
            If *time* is not a slice; if *product_names* or *sites* is one
            value, a string or a set; or if a site id is a boolean, a float or
            not a number.
        KeyError
            If a product name is not held.
        ValueError
            If a site id is out of range, a site is named twice, *sites* is a
            two-dimensional array, or the selection leaves no observed cell.
        """
        if product_names is not None:
            product_names = as_names(product_names, message_name="product_names")
        wanted_sites = None if sites is None else list(as_site_ids(sites, message_name="sites"))
        check_time_is_a_slice(time)
        chosen = self._observations
        if product_names is not None:
            check_product_names_are_held(product_names, self.product_names)
            chosen = tuple(self._by_name[n] for n in product_names)
        kept = [_selected(observation, wanted_sites, time) for observation in chosen]
        kept = [observation for observation in kept if observation is not None]
        check_the_selection_keeps_a_cell(kept)
        return ObservationVector(kept)

    def positions(self, *, site: int | None = None, product_name: str | None = None) -> np.ndarray:
        """Where the cells of a site, a product, or both sit in Flat order.

        Parameters
        ----------
        site:
            An integer site id, or ``None`` for every site.
        product_name:
            A product the vector holds, or ``None`` for every product.

        Returns
        -------
        numpy.ndarray
            The ascending ``int64`` positions in Flat of the cells matching
            both; empty for a site with no observed cell.

        Raises
        ------
        TypeError
            If *site* is a boolean, a float, a string or not a number.
        ValueError
            If *site* is not a site id, from 1 to the largest ``int32``.
        KeyError
            If *product_name* is not held.
        """
        mask = np.ones(self.dimension, dtype=bool)
        if site is not None:
            mask &= self._index.get_level_values(SITE).values == as_site_id(
                site, message_name="site"
            )
        if product_name is not None:
            check_product_names_are_held([product_name], self.product_names)
            mask &= self._index.get_level_values(PRODUCT).values == product_name
        return np.flatnonzero(mask)

    # ── representations ───────────────────────────────────────────────────────

    def flat(self, fields: Mapping[str, xr.DataArray]) -> np.ndarray:
        """Fields to Flat: ``(N,)``, or ``(J, N)`` when the fields carry a batch dim.

        Parameters
        ----------
        fields:
            A mapping from every product name to an array on that product's
            ``site`` and ``time`` labels, with or without one batch dim (of
            any name, the same in every array), in any dimension order. An
            array may be larger than the observation (a prediction over more
            sites or times); only the vector's cells are read, by label. A
            scalar batch coordinate is not a batch dim: such arrays give one
            vector.

        Returns
        -------
        numpy.ndarray
            ``float64``, ``(N,)``, or ``(J, N)`` in the order of the fields'
            batch labels. Values are taken as they are: a ``NaN`` prediction
            stays ``NaN``.

        Raises
        ------
        TypeError
            If *fields* is not a mapping, or an entry is not a ``DataArray``.
        ValueError
            If a product is missing; if an array lacks an observed site or
            time label, or a ``site`` or ``time`` dimension the product has;
            if an array has more than one batch dim (stack them first, with
            :func:`sipnet_calibration.fields.stack_batch_dims`); or if some
            arrays carry a batch dim and others do not, they carry different
            ones, or they disagree on its labels.
        """
        check_fields_hold_the_products(fields, self.product_names)
        batch = _batch_dim_and_labels(fields, self.product_names)
        shape = (len(batch[1]), self.dimension) if batch is not None else (self.dimension,)
        out = np.full(shape, np.nan, dtype=np.float64)
        for observation in self._observations:
            array = fields[observation.product_name]
            check_field_is_on_the_grid(array, observation)
            positions = self.positions(product_name=observation.product_name)
            out[..., positions] = _read_observations(
                array,
                self._index[positions],
                observation.is_static,
                None if batch is None else batch[0],
            )
        return out

    def fields(self, flat_values: Any, *, batch_dim: str = SAMPLE) -> dict[str, xr.DataArray]:
        """Flat to Fields: ``(N,)`` or ``(J, N)`` onto each product's grid.

        Parameters
        ----------
        flat_values:
            An array-like of shape ``(N,)`` or ``(J, N)``, in Flat order.
        batch_dim:
            The name of the batch dim a ``(J, N)`` batch is given. It may not
            be a reserved name (a spatial name, ``time``, ``source_index``),
            a product name, or a coordinate of an observation's values other
            than a scalar batch label, which the batch dim replaces.

        Returns
        -------
        dict
            Product name to a ``float64`` array on the observation's
            ``(site[, time])`` grid, with its coordinates and attributes and
            ``NaN`` at unobserved cells. A ``(J, N)`` batch gives each array a
            leading ``int64`` *batch_dim* labeled ``0`` to ``J - 1``; an
            observation's scalar label of that name is not carried.

        Raises
        ------
        TypeError
            If *flat_values* is not a rectangular array of real numbers, or
            *batch_dim* is not a string.
        ValueError
            If *flat_values* is not one- or two-dimensional or does not have
            ``N`` entries per row, or *batch_dim* is a name it may not be.
        """
        check_batch_dim_name_is_not_reserved(batch_dim, message_name="batch_dim")
        check_batch_dim_is_not_an_observation_name(self._observations, batch_dim)
        batched = np.asarray(
            as_batched_flat(flat_values, self.dimension, message_name="flat_values")
        )
        was_one_vector = is_one_vector(flat_values)
        out: dict[str, xr.DataArray] = {}
        for observation in self._observations:
            positions = self.positions(product_name=observation.product_name)
            array = _unstacked(
                observation, batched[:, positions], self._index[positions], batch_dim
            )
            out[observation.product_name] = (
                array.isel({batch_dim: 0}, drop=True) if was_one_vector else array
            )
        return out

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        model_output: xr.Dataset,
        *,
        sipnet_parameters: xr.Dataset | Mapping[str, Any] | None = None,
    ) -> dict[str, xr.DataArray]:
        """Every product's operator applied at its observation, converted and checked.

        Parameters
        ----------
        model_output:
            The labeled model output the operators read: one run from
            :func:`sipnet_calibration.fields.label_run`, or a stack from
            :func:`sipnet_calibration.fields.stack_model_outputs`, carrying
            every variable in :attr:`output_variable_names` at every site the
            vector observes.
        sipnet_parameters:
            The SIPNET parameter values the runs used, for the operators that
            read any (:attr:`sipnet_parameter_names`): a SIPNET table on
            ``(*batch, site)`` or ``(site,)``, or a mapping for one run.

        Returns
        -------
        dict
            Fields on each product's grid, in the observation's units, with
            the model output's batch dims if any, first; pass the result to
            :meth:`flat` for a ``(J, N)`` batch. A ``NaN`` is allowed only
            where a variable the operators read is ``NaN`` at every step for
            that site and batch label (a failed run).

        Raises
        ------
        TypeError
            If *model_output* is not an ``xr.Dataset``, or an operator returns
            something other than a ``DataArray``.
        ValueError
            If *model_output* lacks a variable that is read, or parameters are
            read and *sipnet_parameters* is not given; if a prediction is not
            on its observation's grid or carries no ``units``; if pySIPNET's
            ``convert_dataarray_units`` refuses to convert a prediction into
            its observation's units and constituent; or if a prediction is
            ``NaN`` at an observed cell where the run succeeded. An operator's
            own refusals pass through.
        """
        check_model_output_carries_what_is_read(
            model_output,
            output_variable_names=self.output_variable_names,
            sipnet_parameter_names=self.sipnet_parameter_names,
            sipnet_parameters=sipnet_parameters,
        )
        failed = _failed_runs(model_output, self.output_variable_names)
        return {
            observation.product_name: _predicted(observation, model_output, sipnet_parameters, failed)
            for observation in self._observations
        }


# ── supporting helpers ────────────────────────────────────────────────────────


def _ordered(values: xr.DataArray) -> xr.DataArray:
    dims = [d for d in (SITE, TIME) if d in values.dims]
    values = values.transpose(*dims)
    if not values.indexes[SITE].is_monotonic_increasing:
        values = values.sortby(SITE)
    if TIME in values.dims and not values.indexes[TIME].is_monotonic_increasing:
        values = values.sortby(TIME)
    return values


def _observed_labels_only(values: xr.DataArray) -> xr.DataArray:
    """*values* without the sites and time labels that hold no observed cell."""
    observed = values.notnull()
    if TIME in values.dims:
        values = values.isel({SITE: np.flatnonzero(observed.any(TIME).values)})
        values = values.isel({TIME: np.flatnonzero(observed.any(SITE).values)})
        return values
    return values.isel({SITE: np.flatnonzero(observed.values)})


def _frozen(values: xr.DataArray, product_name: str) -> xr.DataArray:
    """A read-only, in-memory copy of *values*, named *product_name*."""
    # load() computes a dask or lazily indexed copy in place, so the buffer made
    # read-only is the one the observation keeps rather than a fresh one per read.
    frozen = values.rename(product_name).copy(deep=True).load()
    frozen.values.setflags(write=False)
    return frozen


def _union(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(chain.from_iterable(groups)))


def _selected(
    observation: Observation, sites: list[int] | None, time: slice | None
) -> Observation | None:
    """*observation* restricted to *sites* and *time*, or ``None`` if no observed cell is left.

    Rebuilding the observation drops the time labels no kept site is observed
    at, so an operator run on the selection never reads the model there.
    """
    values = observation.values
    if sites is not None:
        values = values.isel({SITE: np.flatnonzero(values[SITE].isin(sites).values)})
    if not observation.is_static and time is not None:
        values = values.sel({TIME: time})
    if int(values.notnull().sum()) == 0:
        return None
    return Observation(observation.product_name, values, observation.operator)


def _build_index(observations: Sequence[Observation]) -> pd.MultiIndex:
    frames = []
    for order, observation in enumerate(observations):
        cells = observation.cells()
        cells[PRODUCT] = observation.product_name
        cells["_order"] = order
        frames.append(cells)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values([SITE, "_order", TIME], kind="stable", na_position="first")
    return pd.MultiIndex.from_arrays(
        [table[SITE].to_numpy(), table[PRODUCT].to_numpy(), table[TIME].to_numpy()],
        names=INDEX_LEVELS,
    )


def _unstacked(
    observation: Observation, columns: np.ndarray, cells: pd.MultiIndex, batch_dim: str
) -> xr.DataArray:
    """One product's columns of a ``(J, N)`` batch, on its grid with a leading *batch_dim*."""
    full = np.full((columns.shape[0], *observation.values.shape), np.nan, dtype=np.float64)
    rows, cols = _cell_positions(observation, cells)
    if observation.is_static:
        full[:, rows] = columns
    else:
        full[:, rows, cols] = columns
    # An observation's scalar label of the batch dim's name is metadata of the
    # input, and never overwrites the batch coordinate made here.
    kept = {name: c for name, c in observation.values.coords.items() if name != batch_dim}
    return xr.DataArray(
        full,
        dims=(batch_dim, *observation.values.dims),
        coords={**kept, batch_dim: batch_coordinate(batch_dim, np.arange(columns.shape[0]))},
        attrs=dict(observation.values.attrs),
        name=observation.product_name,
    )


def _batch_dim_and_labels(
    fields: Mapping[str, xr.DataArray], product_names: Sequence[str]
) -> tuple[str, np.ndarray] | None:
    """The one batch dim the fields share and its labels, or ``None`` if they have none."""
    arrays = {n: fields[n] for n in product_names if isinstance(fields[n], xr.DataArray)}
    for name, array in arrays.items():
        check_dims_are_batch_spatial_or_time(array, message_name=repr(name))
    by_product = {name: batch_dims(array) for name, array in arrays.items()}
    for name, dims in by_product.items():
        check_at_most_one_batch_dim(dims, message_name=f"the field {name!r}")
    with_batch = [n for n, dims in by_product.items() if dims]
    if not with_batch:
        return None
    check_fields_agree_on_the_batch_dim(fields, product_names, with_batch)
    dim = by_product[with_batch[0]][0]
    return dim, fields[with_batch[0]][dim].values


def _read_observations(
    array: xr.DataArray, cells: pd.MultiIndex, is_static: bool, batch_dim: str | None
) -> np.ndarray:
    """*array*'s values at the observations *cells* index, ``(J, n)`` or ``(n,)``."""
    selectors: dict[str, Any] = {
        SITE: xr.DataArray(cells.get_level_values(SITE).values, dims=_OBSERVATION_DIM)
    }
    if not is_static:
        selectors[TIME] = xr.DataArray(cells.get_level_values(TIME).values, dims=_OBSERVATION_DIM)
    picked = array.sel(selectors)
    if batch_dim is not None:
        picked = picked.transpose(batch_dim, _OBSERVATION_DIM)
    return np.asarray(picked.values, dtype=np.float64)


def _cell_positions(
    observation: Observation, cells: pd.MultiIndex
) -> tuple[np.ndarray, np.ndarray]:
    rows = observation.values.indexes[SITE].get_indexer(cells.get_level_values(SITE))
    if observation.is_static:
        return rows, np.array([], dtype=np.int64)
    return rows, observation.values.indexes[TIME].get_indexer(cells.get_level_values(TIME))


def _failed_runs(
    model_output: xr.Dataset, output_variable_names: Sequence[str]
) -> xr.DataArray | None:
    """Where a read variable is NaN at every step: a run that failed."""
    if TIME not in model_output.dims:
        return None
    failed = None
    for name in output_variable_names:
        this = model_output[name].isnull().all(TIME)
        failed = this if failed is None else (failed | this)
    return failed


def _predicted(
    observation: Observation,
    model_output: xr.Dataset,
    sipnet_parameters: Any,
    failed: xr.DataArray | None,
) -> xr.DataArray:
    """One product's operator applied, checked, converted and checked again."""
    predicted = observation.operator(
        model_output, observation.values, sipnet_parameters=sipnet_parameters
    )
    check_result_is_on_the_observation_grid(
        predicted, observation.values, model_output, observation.product_name
    )
    predicted = convert_dataarray_units(
        predicted,
        to_units=observation.values.attrs["units"],
        to_constituent=observation.values.attrs.get("constituent", "") or "",
    )
    predicted = _with_site_dimension(predicted)
    check_prediction_is_finite_where_the_run_succeeded(predicted, observation, failed)
    predicted.name = observation.product_name
    return predicted


def _with_site_dimension(predicted: xr.DataArray) -> xr.DataArray:
    """*predicted* with ``site`` as a dim, on ``(*batch, site[, time])``."""
    if SITE not in predicted.dims and SITE in predicted.coords:
        predicted = predicted.expand_dims(SITE)
    rest = [d for d in (SITE, TIME) if d in predicted.dims]
    others = [d for d in predicted.dims if d not in rest]
    return predicted.transpose(*others, *rest)


# ── checks ────────────────────────────────────────────────────────────────────


def check_product_name_is_a_nonempty_string(product_name: Any) -> None:
    if not isinstance(product_name, str):
        raise TypeError(
            f"product_name must be a string, got {type(product_name).__name__}; name the "
            "product as its constraint file is named, e.g. 'modis_leaf_area_index'."
        )
    if not product_name:
        raise ValueError(
            "product_name is empty; name the product as its constraint file is named, "
            "e.g. 'modis_leaf_area_index'."
        )


def check_values_are_a_field(values: Any, message_name: str) -> None:
    if not isinstance(values, xr.DataArray):
        raise TypeError(
            f"{message_name}: values must be a DataArray, got {type(values).__name__}; "
            "pass one product's array from constraint_fields."
        )
    batch = batch_dims(values)
    if batch:
        raise ValueError(
            f"{message_name}: values carry the batch dim(s) {list(batch)}. An observation "
            "ensemble is reduced to one value per cell by the experiment before it enters "
            "the vector; reduce it, or select one label."
        )
    extra = [d for d in values.dims if d not in (SITE, TIME)]
    if extra or SITE not in values.dims:
        raise ValueError(
            f"{message_name}: values must be on (site,) or (site, time), got dims "
            f"{tuple(values.dims)}; select or reduce the other dimensions first."
        )
    if SITE not in values.coords:
        raise ValueError(
            f"{message_name}: values carry no site coordinate; label the rows with the "
            "site ids of the site table."
        )
    if values[SITE].dtype.kind not in "iu":
        raise ValueError(
            f"{message_name}: site labels must be integers (the site ids), got dtype "
            f"{values[SITE].dtype}; cast them with .astype(int) if they are whole numbers."
        )
    if values.indexes[SITE].has_duplicates:
        raise ValueError(
            f"{message_name}: the site coordinate has duplicates; an observation names each "
            "site once, so combine the repeated rows."
        )
    if values.dtype.kind not in "iuf":
        raise ValueError(
            f"{message_name}: values must be numeric, got dtype {values.dtype}; convert them "
            "to float, with NaN where nothing was observed."
        )
    if TIME in values.dims:
        check_time_labels_are_naive_and_distinct(values, message_name)


def check_time_labels_are_naive_and_distinct(values: xr.DataArray, message_name: str) -> None:
    if TIME not in values.coords:
        raise ValueError(
            f"{message_name}: values have a time dimension but no time coordinate; label "
            "the columns with their timestamps."
        )
    if not pd.api.types.is_datetime64_dtype(values[TIME].dtype):
        raise ValueError(
            f"{message_name}: time labels must be naive datetime64, got dtype "
            f"{values[TIME].dtype}; a time-zone-aware or object coordinate is not "
            "one the model axis can be matched against. Convert it to the model's clock "
            "and drop the time zone."
        )
    if np.isnat(values[TIME].values).any():
        raise ValueError(
            f"{message_name}: the time coordinate holds NaT, which the index reserves for a "
            "static product; drop those labels."
        )
    if values.indexes[TIME].has_duplicates:
        raise ValueError(
            f"{message_name}: the time coordinate has duplicates; combine the repeated "
            "columns into one per label."
        )


def check_observed_values_are_finite_or_nan(values: xr.DataArray, message_name: str) -> None:
    if np.isinf(values.values.astype(np.float64)).any():
        raise ValueError(
            f"{message_name}: an observed value is infinite; an observation is finite or "
            "NaN, so replace it with NaN or drop it."
        )


def check_values_have_units(values: xr.DataArray, message_name: str) -> None:
    units = values.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            f"{message_name}: values carry no 'units' attribute, so a prediction cannot be "
            "converted to compare with them; set attrs['units'] to the source's units."
        )
    validate_units(units)


def check_there_is_an_observation(observations: Sequence[Any]) -> None:
    if not observations:
        raise ValueError(
            "an ObservationVector needs at least one Observation; pass one per product."
        )


def check_entries_are_observations(observations: Sequence[Any]) -> None:
    bad = [
        type(observation).__name__
        for observation in observations
        if not isinstance(observation, Observation)
    ]
    if bad:
        raise TypeError(
            f"every entry must be an Observation, got {bad}; wrap each product's array "
            "as Observation(product_name, values, operator)."
        )


def check_product_names_are_unique(observations: Sequence[Observation]) -> None:
    names = [observation.product_name for observation in observations]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f"product names must be unique; {repeated} repeat. Give each observation of "
            "the same quantity a name of its own."
        )


def check_every_observation_has_an_observed_cell(observations: Sequence[Observation]) -> None:
    empty = [observation.product_name for observation in observations if observation.n_cells == 0]
    if empty:
        raise ValueError(
            f"{empty} hold no observed cell; drop them or select sites that were observed."
        )


def check_the_selection_keeps_a_cell(kept: Sequence[Observation]) -> None:
    if not kept:
        raise ValueError(
            "the selection leaves no observed cell; select sites, products or a period "
            "the vector observes."
        )


def check_product_names_are_held(names: Sequence[str], held: Sequence[str]) -> None:
    unknown = [n for n in names if n not in held]
    if unknown:
        what = repr(unknown[0]) if len(unknown) == 1 else str(unknown)
        raise KeyError(f"no observation of {what}; the vector holds {list(held)}.")


def check_time_is_a_slice(time: Any) -> None:
    if time is not None and not isinstance(time, slice):
        raise TypeError(
            f"time must be a slice such as slice('2012', '2024'), got {type(time).__name__}."
        )


def check_fields_hold_the_products(fields: Any, product_names: Sequence[str]) -> None:
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"fields must be a mapping from product name to DataArray, got "
            f"{type(fields).__name__}; pass what predict or fields returns."
        )
    missing = [n for n in product_names if n not in fields]
    if missing:
        raise ValueError(
            f"fields lack the product(s) {missing}; pass an array for every product the "
            "vector holds, or select the vector to the products given."
        )


def check_batch_dim_is_not_an_observation_name(
    observations: Sequence[Observation], batch_dim: str
) -> None:
    """*batch_dim* names no product and no coordinate an observation's values carry.

    An observation's scalar batch label of that name is allowed: it is
    metadata, and the batch dim replaces it.
    """
    for observation in observations:
        values = observation.values
        taken = set(map(str, values.coords)) - set(scalar_batch_labels(values))
        what = (
            "a product name"
            if batch_dim == observation.product_name
            else f"a coordinate of the observation {observation.product_name!r}"
            if batch_dim in taken
            else None
        )
        if what is not None:
            raise ValueError(
                f"batch_dim={batch_dim!r} is {what}; name the batch dim otherwise, such as "
                "'sample'."
            )


def check_fields_agree_on_the_batch_dim(
    fields: Mapping[str, xr.DataArray], product_names: Sequence[str], with_batch: Sequence[str]
) -> None:
    """Every field carries the same batch dim, with the same labels in the same order."""
    if len(with_batch) != len(product_names):
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


def check_field_is_on_the_grid(array: Any, observation: Observation) -> None:
    name = observation.product_name
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"{name}: expected a DataArray, got {type(array).__name__}; pass the field "
            "predict or fields returns for it."
        )
    if SITE not in array.dims:
        raise ValueError(
            f"{name}: the array has no site dimension; give a one-site array a site "
            "dimension of length one with .expand_dims('site')."
        )
    missing_sites = missing_labels(array, SITE, observation.values.indexes[SITE])
    if missing_sites:
        raise ValueError(
            f"{name}: the array lacks observed site(s) {missing_sites[:10]}; predict at "
            "every site the observation holds, or select the vector to the sites given."
        )
    if not observation.is_static:
        if TIME not in array.dims:
            raise ValueError(
                f"{name}: the array has no time dimension; read the model at the "
                "observation's time labels."
            )
        observed_times = observation.values[TIME].values
        missing_times = observed_times[~np.isin(observed_times, array[TIME].values)]
        if missing_times.size:
            raise ValueError(
                f"{name}: the array lacks {missing_times.size} observed time label(s), the "
                f"first being {missing_times[0]}; read the model at the observation's time "
                "labels, or select the vector to the period given."
            )


def check_prediction_is_finite_where_the_run_succeeded(
    predicted: xr.DataArray, observation: Observation, failed: xr.DataArray | None
) -> None:
    observed = observation.values.notnull()
    missing = predicted.isnull() & observed
    if failed is not None:
        if SITE in failed.dims:
            allowed = failed.sel({SITE: observation.values[SITE].values})
        else:
            allowed = failed
        missing = missing & ~allowed
    if bool(missing.any()):
        where = np.argwhere(missing.values)[0]
        raise ValueError(
            f"{observation.product_name}: the prediction is NaN at an observed cell (first "
            f"at position {where.tolist()} of dims {tuple(missing.dims)}) although the run "
            "succeeded there. A label outside the run, or a gap the operator produced, "
            "must be selected away, not passed on as a failed run."
        )

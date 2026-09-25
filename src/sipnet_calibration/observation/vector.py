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
    as ``time_bounds_start``/``time_bounds_end``. A ``member`` dimension, or
    a scalar ``member`` coordinate, is refused: an observation ensemble is
    reduced to one value per cell by the experiment before it enters the
    vector.
:attr:`Observation.operator`
    An :class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
**Fields**: ``dict[str, xr.DataArray]`` keyed by product name, each ``float64``
on that product's own ``(site[, time])`` grid, ``NaN`` at unobserved cells; a
``(J, N)`` block unstacks to the same with a leading ``member`` dimension, an
``int16`` coordinate labeled ``0`` to ``J - 1``. A product's grid holds only
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
itself is ``NaN`` at that site and member (a failed run), and refuses one
anywhere else, so a coverage gap cannot masquerade as a failed member and be
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
    predictions = vector.flat(predicted_fields)  # (J, N) for a (member, site, time) output
    vector.fields(predictions)["modis_leaf_area_index"]  # back to (member, site, time)
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.units import convert_dataarray_units, validate_units

from sipnet_calibration.fields import missing_labels
from sipnet_calibration.observation.operators import (
    ObservationOperator,
    check_model_output_carries_what_is_read,
    check_operator_declares_names,
    check_result_is_on_the_observation_grid,
)

__all__ = ["INDEX_LEVELS", "Observation", "ObservationVector"]

SITE = "site"
MEMBER = "member"
TIME = "time"
PRODUCT = "product"

#: The levels of :attr:`ObservationVector.index`, in order.
INDEX_LEVELS: tuple[str, ...] = (SITE, PRODUCT, TIME)


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
        If *product_name* is empty; if *values* has a ``member`` dimension or
        a scalar ``member`` coordinate, or dims other than ``(site[, time])``,
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
        sites: Iterable[int] | int | None = None,
        time: slice | None = None,
    ) -> ObservationVector:
        """A sub-vector over some products, sites and a time slice.

        Parameters
        ----------
        product_names:
            The products to keep, in the order the sub-vector takes them; one
            name is accepted for a list of one. Defaults to every product.
        sites:
            The sites to keep, as integer site ids; one id is accepted, and
            any iterable, which is read once. Sites the vector does not
            observe are ignored. Defaults to every site.
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
            If *time* is not a slice, or a site id is a boolean, a string or
            not a number.
        KeyError
            If a product name is not held.
        ValueError
            If a site id is not a whole number, or the selection leaves no
            observed cell.
        """
        if isinstance(product_names, str):
            product_names = [product_names]
        wanted_sites = None if sites is None else _site_ids(sites)
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
            If *site* is a boolean, a string or not a number.
        ValueError
            If *site* is not a whole number.
        KeyError
            If *product_name* is not held.
        """
        mask = np.ones(self.dimension, dtype=bool)
        if site is not None:
            mask &= self._index.get_level_values(SITE).values == _site_id(site)
        if product_name is not None:
            check_product_names_are_held([product_name], self.product_names)
            mask &= self._index.get_level_values(PRODUCT).values == product_name
        return np.flatnonzero(mask)

    # ── representations ───────────────────────────────────────────────────────

    def flat(self, fields: Mapping[str, xr.DataArray]) -> np.ndarray:
        """Fields to Flat: ``(N,)``, or ``(J, N)`` when the fields carry ``member``.

        Parameters
        ----------
        fields:
            A mapping from every product name to an array on that product's
            ``site`` and ``time`` labels, with or without a ``member``
            dimension, in any dimension order. An array may be larger than the
            observation (a prediction over more sites or times); only the
            vector's cells are read, by label.

        Returns
        -------
        numpy.ndarray
            ``float64``, ``(N,)``, or ``(J, N)`` in the fields' ``member``
            order. Values are taken as they are: a ``NaN`` prediction stays
            ``NaN``.

        Raises
        ------
        TypeError
            If *fields* is not a mapping, or an entry is not a ``DataArray``.
        ValueError
            If a product is missing; if an array lacks an observed site or
            time label, or a ``site`` or ``time`` dimension the product has; or
            if some arrays carry ``member`` and others do not, or they disagree
            on its labels.
        """
        check_fields_hold_the_products(fields, self.product_names)
        members = _member_labels(fields, self.product_names)
        shape = (len(members), self.dimension) if members is not None else (self.dimension,)
        out = np.full(shape, np.nan, dtype=np.float64)
        for observation in self._observations:
            array = fields[observation.product_name]
            check_field_is_on_the_grid(array, observation)
            positions = self.positions(product_name=observation.product_name)
            out[..., positions] = _read_cells(
                array, self._index[positions], observation.is_static, members
            )
        return out

    def fields(self, flat_values: Any) -> dict[str, xr.DataArray]:
        """Flat to Fields: ``(N,)`` or ``(J, N)`` onto each product's grid.

        Parameters
        ----------
        flat_values:
            An array-like of shape ``(N,)`` or ``(J, N)``, in Flat order.

        Returns
        -------
        dict
            Product name to a ``float64`` array on the observation's
            ``(site[, time])`` grid, with its coordinates and attributes and
            ``NaN`` at unobserved cells. A ``(J, N)`` block gives each array a
            leading ``int16`` ``member`` dimension labeled ``0`` to ``J - 1``.

        Raises
        ------
        ValueError
            If *flat_values* is not one- or two-dimensional, does not have
            ``N`` entries per row, or has more rows than an ``int16`` can
            label from ``0``.
        """
        block, squeeze = _as_block(flat_values, self.dimension)
        out: dict[str, xr.DataArray] = {}
        for observation in self._observations:
            positions = self.positions(product_name=observation.product_name)
            array = _unstacked(observation, block[:, positions], self._index[positions])
            out[observation.product_name] = array.isel({MEMBER: 0}, drop=True) if squeeze else array
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
            ``(member, site)`` or ``(site,)``, or a mapping for one run.

        Returns
        -------
        dict
            Fields on each product's grid, in the observation's units, with
            the model output's ``member`` if any; pass the result to
            :meth:`flat` for a ``(J, N)`` block. A ``NaN`` is allowed only
            where a variable the operators read is ``NaN`` at every step for
            that site and member (a failed run).

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


def _site_ids(sites: Any) -> list[int]:
    """*sites*, one site id or an iterable of them, as plain integers."""
    if isinstance(sites, np.ndarray):
        items = list(sites.ravel())
    elif isinstance(sites, (str, bytes)) or not isinstance(sites, Iterable):
        items = [sites]
    else:
        items = list(sites)
    return [_site_id(site) for site in items]


def _site_id(site: Any) -> int:
    """One site id as a plain integer."""
    check_site_id_is_an_integer(site)
    return int(site)


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


def _as_block(flat_values: Any, dimension: int) -> tuple[np.ndarray, bool]:
    """*flat_values* as a ``(J, N)`` block, and whether it was ``(N,)``."""
    block = np.asarray(flat_values, dtype=np.float64)
    squeeze = block.ndim == 1
    if squeeze:
        block = block[None, :]
    check_block_is_members_by_cells(block, dimension)
    return block, squeeze


def _unstacked(
    observation: Observation, columns: np.ndarray, cells: pd.MultiIndex
) -> xr.DataArray:
    """One product's columns of a ``(J, N)`` block, on its grid with a ``member`` dim."""
    full = np.full((columns.shape[0], *observation.values.shape), np.nan, dtype=np.float64)
    rows, cols = _cell_positions(observation, cells)
    if observation.is_static:
        full[:, rows] = columns
    else:
        full[:, rows, cols] = columns
    return xr.DataArray(
        full,
        dims=(MEMBER, *observation.values.dims),
        coords={
            MEMBER: np.arange(columns.shape[0], dtype=np.int16),
            **observation.values.coords,
        },
        attrs=dict(observation.values.attrs),
        name=observation.product_name,
    )


def _member_labels(
    fields: Mapping[str, xr.DataArray], product_names: Sequence[str]
) -> np.ndarray | None:
    with_member = [n for n in product_names if MEMBER in fields[n].dims]
    if not with_member:
        return None
    check_fields_agree_on_members(fields, product_names, with_member)
    return fields[with_member[0]][MEMBER].values


def _read_cells(
    array: xr.DataArray, cells: pd.MultiIndex, is_static: bool, members: np.ndarray | None
) -> np.ndarray:
    selectors: dict[str, Any] = {
        SITE: xr.DataArray(cells.get_level_values(SITE).values, dims="cell")
    }
    if not is_static:
        selectors[TIME] = xr.DataArray(cells.get_level_values(TIME).values, dims="cell")
    picked = array.sel(selectors)
    if members is not None:
        picked = picked.transpose(MEMBER, "cell")
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
    if SITE not in predicted.dims and SITE in predicted.coords:
        predicted = predicted.expand_dims(SITE)
    dims = [d for d in (MEMBER, SITE, TIME) if d in predicted.dims]
    return predicted.transpose(*dims)


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
    if MEMBER in values.dims or MEMBER in values.coords:
        form = "dimension" if MEMBER in values.dims else "scalar coordinate"
        raise ValueError(
            f"{message_name}: values carry a member {form}. An observation ensemble is "
            "reduced to one value per cell by the experiment before it enters the "
            "vector; reduce it, or drop the coordinate with .drop_vars('member')."
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
            "1-8000 site ids."
        )
    if values[SITE].dtype.kind not in "iu":
        raise ValueError(
            f"{message_name}: site labels must be integers (the 1-8000 site ids), got dtype "
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


def check_site_id_is_an_integer(site: Any) -> None:
    if isinstance(site, (bool, np.bool_)) or not isinstance(site, numbers.Real):
        raise TypeError(
            f"a site id must be an integer (the 1-8000 site ids), got {site!r} of type "
            f"{type(site).__name__}; pass e.g. 27 or [1, 27]."
        )
    if not np.isfinite(site) or int(site) != site:
        raise ValueError(
            f"a site id must be a whole number (the 1-8000 site ids), got {site!r}; pass "
            "the integer id."
        )


def check_block_is_members_by_cells(block: np.ndarray, dimension: int) -> None:
    if block.ndim != 2:
        raise ValueError(
            f"flat_values must be (N,) or (J, N), got shape {block.shape}; flatten each "
            "member's predictions with ObservationVector.flat."
        )
    if block.shape[1] != dimension:
        raise ValueError(
            f"flat_values has {block.shape[1]} entries per row and the vector {dimension}; "
            "flatten with this vector, or select the vector the block was made with."
        )
    largest = np.iinfo(np.int16).max
    if block.shape[0] - 1 > largest:
        raise ValueError(
            f"a block of {block.shape[0]} members cannot be labeled: member is an int16 "
            f"coordinate from 0, so at most {largest + 1} members. Unstack it in parts."
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


def check_fields_agree_on_members(
    fields: Mapping[str, xr.DataArray], product_names: Sequence[str], with_member: Sequence[str]
) -> None:
    if len(with_member) != len(product_names):
        raise ValueError(
            f"some fields carry a member dimension ({list(with_member)}) and others do not; "
            "flatten predictions from one model output at a time."
        )
    labels = fields[with_member[0]][MEMBER].values
    for name in with_member[1:]:
        if not np.array_equal(fields[name][MEMBER].values, labels):
            raise ValueError(
                f"the fields disagree on their member labels ({with_member[0]!r} and "
                f"{name!r}); flatten predictions from one model output at a time."
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
            "must be selected away, not passed on as a failed member."
        )

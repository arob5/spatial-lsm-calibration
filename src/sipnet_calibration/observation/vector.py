"""The observation vector: the observed cells of an experiment in a fixed
order, with the operator that predicts each product, and the two forms its
values take.

Where this sits
---------------
::

    constraints.constraint_fields() / the NEE product   observed arrays, (site[, time])
      -> Observation(product_name, values, operator)     one per product
      -> ObservationVector([...])                        the cells, in Flat order
           .y                                            what pyEKI compares against
           .predict(model_output, sipnet_parameters=)    H applied, converted, checked
           .flat(fields) / .fields(values)               Fields <-> Flat

The ``ForwardModel`` flattens predictions with it into pyEKI's ``(J, N)``;
the inference layer reads ``y``, ``index`` and ``positions`` off it to build
the error model; a predictive-check figure unstacks a ``(J, N)`` block with
``fields``. It imports nothing from the product modules: an observation is
built from arrays, and reads their attributes.

What it reads
-------------
:class:`Observation.values`
    A field ``(site[, time])`` with ``NaN`` where nothing was observed, and
    ``units`` (and ``constituent``, where the quantity has one) in its
    attributes. An annual product's array also carries its ``time_bounds``
    as ``time_bounds_start``/``time_bounds_end``. A ``member`` dimension is
    refused: an observation ensemble is reduced to one value per cell by the
    experiment before it enters the vector.
:class:`Observation.operator`
    An :class:`~sipnet_calibration.observation.operators.ObservationOperator`.

Data model
----------
**Fields**: ``dict[str, xr.DataArray]`` keyed by product name, each on that
product's own ``(site[, time])`` grid, ``NaN`` at unobserved cells; a ``(J, N)``
block unstacks to the same with a leading ``member`` dimension.

**Flat**: ``y`` in ``R^N``, ``float64``, finite, one entry per observed cell;
predictions are ``(N,)`` or ``(J, N)``, where a ``NaN`` marks a failed run.

**The index**: a ``pandas.MultiIndex`` with levels ``(site, product, time)``,
one row per observed cell, in Flat order: **site-major**, sites ascending,
then products in declaration order, then times ascending. A static product's
cells have ``time = NaT``. The order is a property of the vector; consumers
read it off ``index`` and ``positions`` and never assume it.

Functions
---------
:class:`Observation`, :class:`ObservationVector`.

Notes
-----
**Site-major.** The hierarchical likelihood factorizes over sites and the
error covariance is block-diagonal by site and product, so a site's cells
being contiguous is what a block operator and R-localization consume, and a
worker producing one site's predictions produces one contiguous block.
``(site, product)`` sub-blocks are contiguous too.

**No noise model here.** The products' standard deviations, the covariance
and the likelihood are the inference layer's; this module gives it ``y``,
``index`` and ``positions``.

**Failed runs.** ``predict`` passes a ``NaN`` through where the model output
itself is ``NaN`` at that site and member (a failed run), and refuses one
anywhere else, so a coverage gap cannot masquerade as a failed member and be
repaired away.

Usage
-----
::

    from sipnet_calibration.constraints import constraint_fields
    from sipnet_calibration.observation import DEFAULT_OBS_OPS, Observation, ObservationVector, SelectTimestep

    observed = constraint_fields(["modis_leaf_area_index", "landtrendr_aboveground_biomass"], sites=sites)
    vector = ObservationVector([
        Observation("modis_leaf_area_index", observed["modis_leaf_area_index"],
                    DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        Observation("landtrendr_aboveground_biomass", observed["landtrendr_aboveground_biomass"],
                    SelectTimestep("wood_carbon")),
    ]).select(time=slice("2012", "2024"))

    vector.dimension, vector.y.shape                     # N, (N,)
    predictions = vector.predict(model_output, sipnet_parameters=table)   # Fields
    g = vector.flat(predictions)                         # (J, N) for a (member, site, time) model output
    vector.fields(g)["modis_leaf_area_index"]            # back to (member, site, time)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.parameters.model import resolve_parameter_name
from pysipnet.units import convert_dataarray_units, validate_units
from pysipnet.variables import resolve_output_variable

from sipnet_calibration.observation.operators import ObservationOperator

__all__ = ["INDEX_LEVELS", "Observation", "ObservationVector"]

SITE = "site"
MEMBER = "member"
TIME = "time"
PRODUCT = "product"

#: The levels of :attr:`ObservationVector.index`, in order.
INDEX_LEVELS: tuple[str, ...] = (SITE, PRODUCT, TIME)


@dataclass(frozen=True)
class Observation:
    """One observed product, with the rule that predicts it."""

    product_name: str
    values: xr.DataArray
    operator: ObservationOperator

    def __post_init__(self) -> None:
        check_product_name(self.product_name)
        check_values_are_a_field(self.product_name, self.values)
        check_values_have_units(self.product_name, self.values)
        check_operator_declares_names(self.product_name, self.operator)
        object.__setattr__(self, "values", _ordered(self.values))

    @property
    def sites(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.values[SITE].values)

    @property
    def is_static(self) -> bool:
        return TIME not in self.values.dims

    @property
    def n_cells(self) -> int:
        return int(self.values.notnull().sum())

    def cells(self) -> pd.DataFrame:
        """The observed cells as a frame with ``site``, ``time`` and ``value`` columns."""
        mask = self.values.notnull()
        sites = np.broadcast_to(self.values[SITE].values[:, None] if not self.is_static else self.values[SITE].values, self.values.shape)
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
    """The observed cells of several products in a fixed order; see the module docstring."""

    def __init__(self, observations: Sequence[Observation]) -> None:
        observations = tuple(observations)
        check_observations_are_observations(observations)
        check_product_names_are_unique(observations)
        check_there_are_observed_cells(observations)
        self._observations = observations
        self._by_name = {o.product_name: o for o in observations}
        self._index = _build_index(observations)
        self._y = _flatten_values(observations, self._index)

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self._observations

    @property
    def product_names(self) -> tuple[str, ...]:
        return tuple(o.product_name for o in self._observations)

    @property
    def sites(self) -> tuple[int, ...]:
        return tuple(sorted({s for o in self._observations for s in o.sites}))

    @property
    def dimension(self) -> int:
        return len(self._index)

    @property
    def index(self) -> pd.MultiIndex:
        return self._index

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return _union(o.operator.output_variable_names for o in self._observations)

    @property
    def sipnet_parameter_names(self) -> tuple[str, ...]:
        return _union(o.operator.sipnet_parameter_names for o in self._observations)

    @property
    def observed_values(self) -> dict[str, xr.DataArray]:
        return {o.product_name: o.values for o in self._observations}

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()

    def __getitem__(self, product_name: str) -> Observation:
        try:
            return self._by_name[product_name]
        except KeyError:
            raise KeyError(
                f"no observation of {product_name!r}; the vector holds {list(self.product_names)}."
            ) from None

    def __repr__(self) -> str:
        return (
            f"ObservationVector(N={self.dimension}, products={list(self.product_names)}, "
            f"sites={len(self.sites)})"
        )

    def describe(self) -> pd.DataFrame:
        """One row per product: cells, sites, time span, units, operator."""
        rows = []
        for o in self._observations:
            cells = o.cells()
            rows.append(
                {
                    PRODUCT: o.product_name,
                    "cells": len(cells),
                    "sites": int(cells[SITE].nunique()),
                    "first_time": None if o.is_static else cells[TIME].min(),
                    "last_time": None if o.is_static else cells[TIME].max(),
                    "units": o.values.attrs["units"],
                    "constituent": o.values.attrs.get("constituent", ""),
                    "operator": repr(o.operator),
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

        Selection keeps every observed cell that survives all three filters;
        a product left with no cells is dropped, and a vector left with none
        is refused. ``time`` does not apply to a static product.
        """
        chosen = self._observations
        if product_names is not None:
            unknown = [n for n in product_names if n not in self._by_name]
            if unknown:
                raise KeyError(f"no observation of {unknown}; the vector holds {list(self.product_names)}.")
            chosen = tuple(self._by_name[n] for n in product_names)
        kept: list[Observation] = []
        for o in chosen:
            values = o.values
            if sites is not None:
                wanted = [int(s) for s in sites]
                present = [s for s in wanted if s in set(o.sites)]
                if not present:
                    continue
                values = values.sel({SITE: present})
            if time is not None and not o.is_static:
                values = values.sel({TIME: time})
                if values.sizes[TIME] == 0:
                    continue
            if int(values.notnull().sum()) == 0:
                continue
            kept.append(Observation(o.product_name, values, o.operator))
        if not kept:
            raise ValueError("the selection leaves no observed cell.")
        return ObservationVector(kept)

    def positions(self, *, site: int | None = None, product_name: str | None = None) -> np.ndarray:
        """Where the cells of a site, a product, or both sit in Flat order."""
        mask = np.ones(self.dimension, dtype=bool)
        if site is not None:
            mask &= self._index.get_level_values(SITE).values == int(site)
        if product_name is not None:
            if product_name not in self._by_name:
                raise KeyError(f"no observation of {product_name!r}.")
            mask &= self._index.get_level_values(PRODUCT).values == product_name
        return np.flatnonzero(mask)

    # ── representations ───────────────────────────────────────────────────────

    def flat(self, fields: Mapping[str, xr.DataArray]) -> np.ndarray:
        """Fields to Flat: ``(N,)``, or ``(J, N)`` when the fields carry ``member``.

        Each product's array is read at the vector's cells by ``site`` and
        ``time`` label; it may be larger than the observation (a prediction
        over more sites or times) and is refused if a cell is missing from it.
        Values are taken as they are: a ``NaN`` prediction stays ``NaN``.
        """
        check_fields_hold_the_products(fields, self.product_names)
        members = _member_labels(fields, self.product_names)
        shape = (len(members), self.dimension) if members is not None else (self.dimension,)
        out = np.full(shape, np.nan, dtype=np.float64)
        for o in self._observations:
            array = fields[o.product_name]
            check_field_is_on_the_grid(o.product_name, array, o)
            positions = self.positions(product_name=o.product_name)
            cells = self._index[positions]
            values = _read_cells(array, cells, o.is_static, members)
            if members is not None:
                out[:, positions] = values
            else:
                out[positions] = values
        return out

    def fields(self, values: Any) -> dict[str, xr.DataArray]:
        """Flat to Fields: ``(N,)`` or ``(J, N)`` onto each product's grid.

        Unobserved cells read ``NaN``; a ``(J, N)`` block gives each array a
        leading ``member`` dimension labeled ``0`` to ``J - 1``. The arrays
        carry the observation's coordinates and attributes.
        """
        block = np.asarray(values, dtype=np.float64)
        if block.ndim == 1:
            block = block[None, :]
            squeeze = True
        elif block.ndim == 2:
            squeeze = False
        else:
            raise ValueError(f"values must be (N,) or (J, N), got shape {block.shape}.")
        if block.shape[1] != self.dimension:
            raise ValueError(
                f"values has {block.shape[1]} entries and the vector {self.dimension}."
            )
        out: dict[str, xr.DataArray] = {}
        for o in self._observations:
            positions = self.positions(product_name=o.product_name)
            cells = self._index[positions]
            full = np.full((block.shape[0], *o.values.shape), np.nan, dtype=np.float64)
            rows, cols = _cell_positions(o, cells)
            if o.is_static:
                full[:, rows] = block[:, positions]
            else:
                full[:, rows, cols] = block[:, positions]
            array = xr.DataArray(
                full,
                dims=(MEMBER, *o.values.dims),
                coords={MEMBER: np.arange(block.shape[0], dtype=np.int16), **o.values.coords},
                attrs=dict(o.values.attrs),
                name=o.product_name,
            )
            out[o.product_name] = array.isel({MEMBER: 0}, drop=True) if squeeze else array
        return out

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        model_output: xr.Dataset,
        *,
        sipnet_parameters: xr.Dataset | Mapping[str, Any] | None = None,
    ) -> dict[str, xr.DataArray]:
        """Every product's operator applied at its observation, converted and checked.

        Returns Fields on each product's grid, in the observation's units,
        with the model output's ``member`` if any; pass the result to
        :meth:`flat` for a ``(J, N)`` block. A ``NaN`` is allowed only where
        the model output is ``NaN`` for that site and member (a failed run).
        """
        check_model_output_is_a_dataset(model_output)
        check_model_output_carries(model_output, self.output_variable_names)
        check_parameters_are_given(self.sipnet_parameter_names, sipnet_parameters)
        failed = _failed_runs(model_output, self.output_variable_names)
        out: dict[str, xr.DataArray] = {}
        for o in self._observations:
            predicted = o.operator(model_output, o.values, sipnet_parameters=sipnet_parameters)
            check_prediction_is_an_array(o.product_name, predicted)
            predicted = convert_dataarray_units(
                predicted,
                to_units=o.values.attrs["units"],
                to_constituent=o.values.attrs.get("constituent", "") or "",
            )
            predicted = _with_site_dimension(predicted)
            check_prediction_is_on_the_grid(o.product_name, predicted, o, model_output)
            check_prediction_is_finite_where_the_run_succeeded(o.product_name, predicted, o, failed)
            predicted.name = o.product_name
            out[o.product_name] = predicted
        return out


# ── supporting helpers ────────────────────────────────────────────────────────


def _nanoseconds(values: Any) -> np.ndarray:
    """Datetimes as ``datetime64[ns]`` so labels of different resolutions compare."""
    return np.asarray(values).astype("datetime64[ns]")


def _ordered(values: xr.DataArray) -> xr.DataArray:
    dims = [d for d in (SITE, TIME) if d in values.dims]
    values = values.transpose(*dims)
    if not np.all(np.diff(values[SITE].values.astype(np.int64)) > 0):
        values = values.sortby(SITE)
    if TIME in values.dims and not values.indexes[TIME].is_monotonic_increasing:
        values = values.sortby(TIME)
    return values


def _union(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    seen: list[str] = []
    for group in groups:
        for name in group:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


def _build_index(observations: Sequence[Observation]) -> pd.MultiIndex:
    frames = []
    for order, o in enumerate(observations):
        cells = o.cells()
        cells[PRODUCT] = o.product_name
        cells["_order"] = order
        frames.append(cells)
    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values([SITE, "_order", TIME], kind="stable", na_position="first")
    return pd.MultiIndex.from_arrays(
        [table[SITE].to_numpy(), table[PRODUCT].to_numpy(), table[TIME].to_numpy()],
        names=INDEX_LEVELS,
    )


def _flatten_values(observations: Sequence[Observation], index: pd.MultiIndex) -> np.ndarray:
    out = np.empty(len(index), dtype=np.float64)
    products = index.get_level_values(PRODUCT).values
    for o in observations:
        positions = np.flatnonzero(products == o.product_name)
        cells = index[positions]
        out[positions] = _read_cells(o.values, cells, o.is_static, None)
    check_y_is_finite(out, index)
    return out


def _member_labels(fields: Mapping[str, xr.DataArray], names: Sequence[str]) -> np.ndarray | None:
    with_member = [n for n in names if MEMBER in fields[n].dims]
    if not with_member:
        return None
    if len(with_member) != len(names):
        raise ValueError(
            f"some fields carry a member dimension ({with_member}) and others do not; "
            "flatten predictions from one model output at a time."
        )
    labels = fields[with_member[0]][MEMBER].values
    for n in with_member[1:]:
        if not np.array_equal(fields[n][MEMBER].values, labels):
            raise ValueError("the fields disagree on their member labels.")
    return labels


def _read_cells(
    array: xr.DataArray, cells: pd.MultiIndex, is_static: bool, members: np.ndarray | None
) -> np.ndarray:
    site_labels = xr.DataArray(cells.get_level_values(SITE).values, dims="cell")
    selectors: dict[str, Any] = {SITE: site_labels}
    if not is_static:
        selectors[TIME] = xr.DataArray(_nanoseconds(cells.get_level_values(TIME).values), dims="cell")
    picked = array.sel(selectors)
    if members is not None:
        picked = picked.transpose(MEMBER, "cell")
    return np.asarray(picked.values, dtype=np.float64)


def _cell_positions(o: Observation, cells: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
    site_position = {int(s): i for i, s in enumerate(o.values[SITE].values)}
    rows = np.array([site_position[int(s)] for s in cells.get_level_values(SITE)], dtype=np.int64)
    if o.is_static:
        return rows, np.array([], dtype=np.int64)
    time_position = {int(t): i for i, t in enumerate(_nanoseconds(o.values[TIME].values).astype("int64"))}
    cols = np.array(
        [time_position[int(t)] for t in _nanoseconds(cells.get_level_values(TIME).values).astype("int64")],
        dtype=np.int64,
    )
    return rows, cols


def _failed_runs(model_output: xr.Dataset, names: Sequence[str]) -> xr.DataArray | None:
    """Where every step of every read variable is NaN: a run that failed."""
    if TIME not in model_output.dims:
        return None
    failed = None
    for name in names:
        this = model_output[name].isnull().all(TIME)
        failed = this if failed is None else (failed | this)
    return failed


def _with_site_dimension(predicted: xr.DataArray) -> xr.DataArray:
    if SITE not in predicted.dims and SITE in predicted.coords:
        predicted = predicted.expand_dims(SITE)
    dims = [d for d in (MEMBER, SITE, TIME) if d in predicted.dims]
    return predicted.transpose(*dims)


# ── checks ────────────────────────────────────────────────────────────────────


def check_product_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"product_name must be a non-empty string, got {name!r}.")


def check_values_are_a_field(name: str, values: Any) -> None:
    if not isinstance(values, xr.DataArray):
        raise TypeError(f"{name}: values must be a DataArray, got {type(values).__name__}.")
    extra = [d for d in values.dims if d not in (SITE, TIME)]
    if MEMBER in values.dims:
        raise ValueError(
            f"{name}: values carry a member dimension. An observation ensemble is reduced "
            "to one value per cell by the experiment before it enters the vector."
        )
    if extra or SITE not in values.dims:
        raise ValueError(
            f"{name}: values must be on (site,) or (site, time), got dims {tuple(values.dims)}."
        )
    if SITE not in values.coords:
        raise ValueError(f"{name}: values carry no site coordinate.")
    if len(set(values[SITE].values.tolist())) != values.sizes[SITE]:
        raise ValueError(f"{name}: the site coordinate has duplicates.")
    if TIME in values.dims and values.indexes[TIME].has_duplicates:
        raise ValueError(f"{name}: the time coordinate has duplicates.")


def check_values_have_units(name: str, values: xr.DataArray) -> None:
    units = values.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            f"{name}: values carry no 'units' attribute, so a prediction cannot be "
            "converted to compare with them."
        )
    validate_units(units)


def check_operator_declares_names(name: str, operator: Any) -> None:
    if not callable(operator):
        raise TypeError(f"{name}: operator must be callable, got {type(operator).__name__}.")
    for attribute in ("output_variable_names", "sipnet_parameter_names"):
        names = getattr(operator, attribute, None)
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"{name}: the operator must declare {attribute} as a tuple of names, got "
                f"{names!r}."
            )
    for variable in operator.output_variable_names:
        resolve_output_variable(variable)
    for parameter in operator.sipnet_parameter_names:
        resolve_parameter_name(parameter)


def check_observations_are_observations(observations: Sequence[Any]) -> None:
    if not observations:
        raise ValueError("an ObservationVector needs at least one Observation.")
    bad = [type(o).__name__ for o in observations if not isinstance(o, Observation)]
    if bad:
        raise TypeError(f"every entry must be an Observation, got {bad}.")


def check_product_names_are_unique(observations: Sequence[Observation]) -> None:
    names = [o.product_name for o in observations]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"product names must be unique; {repeated} repeat.")


def check_there_are_observed_cells(observations: Sequence[Observation]) -> None:
    empty = [o.product_name for o in observations if o.n_cells == 0]
    if empty:
        raise ValueError(f"{empty} hold no observed cell; drop them or select sites that were observed.")


def check_y_is_finite(y: np.ndarray, index: pd.MultiIndex) -> None:
    bad = ~np.isfinite(y)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} observed value(s) are not finite, the first at "
            f"{tuple(index[int(np.flatnonzero(bad)[0])])}."
        )


def check_fields_hold_the_products(fields: Any, names: Sequence[str]) -> None:
    if not isinstance(fields, Mapping):
        raise TypeError(f"fields must be a mapping from product name to DataArray, got {type(fields).__name__}.")
    missing = [n for n in names if n not in fields]
    if missing:
        raise ValueError(f"fields lack the product(s) {missing}.")


def check_field_is_on_the_grid(name: str, array: Any, o: Observation) -> None:
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"{name}: expected a DataArray, got {type(array).__name__}.")
    if SITE not in array.dims:
        raise ValueError(f"{name}: the array has no site dimension.")
    missing_sites = sorted(set(o.sites) - set(int(s) for s in array[SITE].values))
    if missing_sites:
        raise ValueError(f"{name}: the array lacks observed site(s) {missing_sites[:10]}.")
    if not o.is_static:
        if TIME not in array.dims:
            raise ValueError(f"{name}: the array has no time dimension.")
        observed_times = _nanoseconds(o.values[TIME].values[o.values.notnull().any(SITE).values])
        have = _nanoseconds(array[TIME].values)
        missing_times = observed_times[~np.isin(observed_times, have)]
        if missing_times.size:
            raise ValueError(
                f"{name}: the array lacks {missing_times.size} observed time label(s), the "
                f"first being {missing_times[0]}."
            )


def check_model_output_is_a_dataset(model_output: Any) -> None:
    if not isinstance(model_output, xr.Dataset):
        raise TypeError(
            f"model_output must be an xarray Dataset of pySIPNET variables, got "
            f"{type(model_output).__name__}; label a run with fields.label_run."
        )


def check_model_output_carries(model_output: xr.Dataset, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in model_output.data_vars]
    if missing:
        raise ValueError(
            f"the model output lacks {missing}, which the operators read; it has "
            f"{list(model_output.data_vars)[:10]}."
        )


def check_parameters_are_given(names: Sequence[str], sipnet_parameters: Any) -> None:
    if names and sipnet_parameters is None:
        raise ValueError(f"the operators read SIPNET parameters {list(names)}; pass sipnet_parameters=.")


def check_prediction_is_an_array(name: str, predicted: Any) -> None:
    if not isinstance(predicted, xr.DataArray):
        raise TypeError(f"{name}: the operator returned {type(predicted).__name__}, not a DataArray.")
    if not isinstance(predicted.attrs.get("units"), str):
        raise ValueError(
            f"{name}: the operator's result carries no 'units' attribute, so it cannot be "
            "converted into the observation's units. Write the operator with the "
            "observation.units verbs, or set attrs['units'] on its result."
        )


def check_prediction_is_on_the_grid(
    name: str, predicted: xr.DataArray, o: Observation, model_output: xr.Dataset
) -> None:
    if MEMBER in predicted.dims and MEMBER not in model_output.dims:
        raise ValueError(f"{name}: the prediction has a member dimension the model output lacks.")
    if predicted[SITE].values.tolist() != list(o.sites):
        raise ValueError(f"{name}: the prediction is not on the observation's sites, in order.")
    if o.is_static:
        if TIME in predicted.dims:
            raise ValueError(f"{name}: the prediction has a time dimension for a static observation.")
    elif TIME not in predicted.dims or not np.array_equal(
        _nanoseconds(predicted[TIME].values), _nanoseconds(o.values[TIME].values)
    ):
        raise ValueError(f"{name}: the prediction is not on the observation's time labels.")


def check_prediction_is_finite_where_the_run_succeeded(
    name: str, predicted: xr.DataArray, o: Observation, failed: xr.DataArray | None
) -> None:
    observed = o.values.notnull()
    missing = predicted.isnull() & observed
    if failed is not None:
        allowed = failed.sel({SITE: o.values[SITE].values}) if SITE in failed.dims else failed
        missing = missing & ~allowed
    if bool(missing.any()):
        where = np.argwhere(missing.values)[0]
        raise ValueError(
            f"{name}: the prediction is NaN at an observed cell (first at position "
            f"{where.tolist()} of dims {tuple(missing.dims)}) although the run succeeded "
            "there. A label outside the run, or a gap the operator produced, must be "
            "selected away, not passed on as a failed member."
        )

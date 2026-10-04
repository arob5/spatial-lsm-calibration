"""Tests for the probability layer's labels: coords with stacked dims,
constants read at the labels in use (own dims included), and label maps into
dims and element axes.

Every lookup is by label and by kind, never by position or by string; it
is linear in the labels, which a pool-sized case checks; and every check is
provoked once.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.probability.labels import (
    aligned_constants,
    aligned_label_maps,
    as_constants,
    as_coords,
    as_label_maps,
    indexer,
    label_kind,
)

SITES = np.array([620, 865, 1037], dtype=np.int32)
TIMES = pd.to_datetime(["2012-07-01", "2013-07-01"])
OBSERVATIONS = pd.MultiIndex.from_arrays(
    [[620, 620, 865], TIMES[[0, 1, 0]]], names=["site", "time"]
)
COORDS = as_coords({"site": SITES, "pft": ["boreal", "temperate"], "lai_observation": OBSERVATIONS})


def on_sites(values, sites=SITES, name=None) -> xr.DataArray:
    return xr.DataArray(np.asarray(values), dims="site", coords={"site": sites}, name=name)


def on_observations(values, index=OBSERVATIONS, name=None) -> xr.DataArray:
    coords = xr.Coordinates.from_pandas_multiindex(index, "lai_observation")
    return xr.DataArray(np.asarray(values), dims="lai_observation", coords=coords, name=name)


# ── coords ────────────────────────────────────────────────────────────────────


def test_coords_keep_their_kinds_and_are_named_for_their_dims():
    assert COORDS["site"].dtype == np.int32 and COORDS["site"].name == "site"
    assert COORDS["pft"].dtype == object and COORDS["pft"].tolist() == ["boreal", "temperate"]
    assert list(COORDS["lai_observation"].names) == ["site", "time"]


def test_stacked_times_are_held_at_nanoseconds():
    microseconds = pd.MultiIndex.from_arrays([[1, 2], TIMES.astype("datetime64[us]")], names=["site", "time"])
    held = as_coords({"obs": microseconds})["obs"]
    assert held.levels[1].dtype == "datetime64[ns]"
    assert held.get_level_values("time").tolist() == TIMES.tolist()


@pytest.mark.parametrize(
    ("coords", "error", "match"),
    [
        ([("site", [1])], TypeError, "mapping"),
        ({1: [1]}, TypeError, "strings"),
        ({"site": 4977}, TypeError, "sequence"),
        ({"site": "4977"}, TypeError, "one string"),
        ({"site": [1, "a"]}, TypeError, "neither all integers"),
        ({"site": [True, False]}, TypeError, "neither all integers"),
        ({"site": [1.0, 2.0]}, TypeError, "neither all integers"),
        ({"site": []}, ValueError, "no label"),
        ({"site": [1, 1]}, ValueError, "more than once"),
        ({"obs": pd.MultiIndex.from_arrays([[1], [2]])}, ValueError, "name every level"),
        ({"obs": pd.MultiIndex.from_arrays([[1], [2]], names=["a", "a"])}, ValueError, "name every level"),
        ({"obs": pd.MultiIndex.from_arrays([[1], [2]], names=["obs", "a"])}, ValueError, "name every level"),
        ({"obs": pd.MultiIndex.from_arrays([[1.5], [2]], names=["a", "b"])}, TypeError, "datetime64"),
        ({"obs": pd.MultiIndex.from_arrays([[1, 1], [2, 2]], names=["a", "b"])}, ValueError, "more than once"),
        ({"obs": pd.MultiIndex.from_arrays([[1, np.nan], ["x", "y"]], names=["a", "b"])}, ValueError, "missing"),
    ],
)
def test_coords_are_checked(coords, error, match):
    with pytest.raises(error, match=match):
        as_coords(coords)


# ── constants ─────────────────────────────────────────────────────────────────


def test_a_constant_is_read_at_the_labels_in_use_in_their_order():
    pool = on_sites([3.0, 1.0, 2.0, 9.0], sites=[1037, 620, 865, 4977])
    aligned = aligned_constants({"median": pool}, COORDS, dim_order=["site"], message_name="constants")
    assert aligned["median"].tolist() == [1.0, 2.0, 3.0]


def test_a_constant_on_a_stacked_dim_is_read_by_its_tuples():
    shuffled = OBSERVATIONS[[2, 0, 1]]
    constant = on_observations([30.0, 10.0, 20.0], index=shuffled)
    aligned = aligned_constants({"sd": constant}, COORDS, dim_order=["lai_observation"], message_name="constants")
    assert aligned["sd"].tolist() == [10.0, 20.0, 30.0]


def test_a_constant_arrives_in_the_readers_dim_order_then_its_own():
    labels_by_dim = {**COORDS, "part": pd.Index(["leaf", "wood"], name="part")}
    constant = xr.DataArray(
        np.arange(12.0).reshape(2, 3, 2),
        dims=("part", "site", "coordinate"),
        coords={"part": ["leaf", "wood"], "site": SITES},
    )
    aligned = aligned_constants({"c": constant}, labels_by_dim, dim_order=["site", "part"], own_dims=["coordinate"],
                                message_name="constants")["c"]
    assert aligned.shape == (3, 2, 2)
    assert np.array_equal(aligned, np.transpose(constant.values, (1, 0, 2)))


def test_an_own_dim_needs_no_labels_and_is_passed_whole():
    constant = xr.DataArray(np.ones((3, 2)), dims=("site", "xy"), coords={"site": SITES})
    aligned = aligned_constants({"c": constant}, COORDS, dim_order=["site"], own_dims=["xy"], message_name="constants")
    assert aligned["c"].shape == (3, 2)


def test_a_boolean_constant_keeps_its_dtype():
    aligned = aligned_constants({"flag": on_sites([True, False, True])}, COORDS, dim_order=["site"], message_name="c")
    assert aligned["flag"].dtype == bool


def test_constants_are_read_only_copies():
    original = on_sites([1.0, 2.0, 3.0])
    held = as_constants({"c": original}, message_name="constants")["c"]
    original.values[0] = 99.0
    assert held.values[0] == 1.0 and not held.values.flags.writeable


@pytest.mark.parametrize(
    ("constants", "error", "match"),
    [
        ([1.0], TypeError, "mapping"),
        ({"c": np.ones(3)}, TypeError, "DataArray"),
        ({"c": on_sites([1, 2, 3])}, TypeError, "neither float64 nor bool"),
    ],
)
def test_constants_are_checked_when_given(constants, error, match):
    with pytest.raises(error, match=match):
        as_constants(constants, message_name="constants")


@pytest.mark.parametrize(
    ("constant", "error", "match"),
    [
        (xr.DataArray(np.ones(3), dims="year"), ValueError, "own_dims"),
        (xr.DataArray(np.ones(3), dims="site"), ValueError, "no 'site' coordinate"),
        (on_sites([1.0, 2.0], sites=[620, 865]), KeyError, "1037"),
        (on_sites([1.0, 2.0, 3.0], sites=["620", "865", "1037"]), KeyError, "620"),
        (on_sites([1.0, np.nan, 3.0]), ValueError, "not finite"),
        (xr.DataArray(np.ones(3), dims="lai_observation", coords={"lai_observation": [1, 2, 3]}), ValueError, "levels"),
    ],
)
def test_constants_are_checked_when_read(constant, error, match):
    with pytest.raises(error, match=match):
        aligned_constants({"c": constant}, COORDS, dim_order=["site"], message_name="constants")


def test_a_constant_may_be_missing_away_from_the_labels_in_use():
    pool = on_sites([1.0, 2.0, 3.0, np.nan], sites=[620, 865, 1037, 4977])
    assert aligned_constants({"c": pool}, COORDS, dim_order=["site"], message_name="c")["c"].tolist() == [1.0, 2.0, 3.0]


# ── label maps ────────────────────────────────────────────────────────────────


def test_a_label_map_into_a_dim_arrives_as_positions():
    site_pft = on_sites(["temperate", "boreal", "boreal", "temperate"], sites=[620, 865, 1037, 4977], name="pft")
    aligned = aligned_label_maps(as_label_maps({"site_pft": site_pft}, message_name="label maps"), COORDS,
                                 message_name="label maps")
    assert aligned["site_pft"].tolist() == [1, 0, 0] and aligned["site_pft"].dtype == np.int64


def test_a_label_map_on_a_stacked_dim_into_an_element_axis():
    years = pd.Index(["2013", "2012"], name="year")
    year = on_observations(["2012", "2013", "2012"], name="year")
    aligned = aligned_label_maps({"year": year}, COORDS, element_axes={"year": years}, message_name="label maps")
    assert aligned["year"].tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    ("label_map", "error", "match"),
    [
        (xr.DataArray(np.ones((3, 1)), dims=("site", "x"), name="pft"), ValueError, "one dim"),
        (on_sites(["boreal"] * 3), ValueError, "named None"),
    ],
)
def test_label_maps_are_checked_when_given(label_map, error, match):
    with pytest.raises(error, match=match):
        as_label_maps({"m": label_map}, message_name="label maps")


@pytest.mark.parametrize(
    ("label_map", "error", "match"),
    [
        (xr.DataArray(["a"], dims="biome", coords={"biome": ["x"]}, name="pft"), KeyError, "not a dim of the coords"),
        (on_sites(["a", "b", "c"], name="biome"), KeyError, "neither a dim"),
        (on_sites([620, 865, 1037], name="site"), ValueError, "to itself"),
        (xr.DataArray(["boreal"] * 3, dims="site", name="pft"), ValueError, "no 'site' coordinate"),
        (on_sites(["boreal", "boreal"], sites=[620, 865], name="pft"), KeyError, "1037"),
        (on_sites(["boreal", "tundra", "boreal"], name="pft"), KeyError, "tundra"),
    ],
)
def test_label_maps_are_checked_when_read(label_map, error, match):
    with pytest.raises(error, match=match):
        aligned_label_maps({"m": label_map}, COORDS, message_name="label maps")


# ── lookups ───────────────────────────────────────────────────────────────────


def test_a_lookup_matches_by_kind():
    sites = pd.Index([1, 0, 2])
    assert indexer(sites, [0, 2]).tolist() == [1, 2]
    assert indexer(sites, ["0"]).tolist() == [-1]
    assert indexer(sites, [True]).tolist() == [-1]
    assert label_kind(pd.Index([1, "a"])) == "other"


def test_reading_a_pool_of_labels_is_linear():
    """80,000 labels in use read from 100,000 held, in a fraction of a
    second; the parameter layer's check took seconds at a fortieth of this."""
    rng = np.random.default_rng(0)
    held = rng.permutation(100_000)
    in_use = as_coords({"site": np.sort(rng.choice(100_000, 80_000, replace=False))})
    constant = on_sites(rng.standard_normal(100_000), sites=held)
    label_map = on_sites(np.where(held % 2 == 0, "even", "odd").astype(object), sites=held, name="parity")
    coords = {**in_use, "parity": pd.Index(["even", "odd"], name="parity", dtype=object)}
    start = time.perf_counter()
    aligned = aligned_constants({"c": constant}, coords, dim_order=["site"], message_name="c")["c"]
    positions = aligned_label_maps({"m": label_map}, coords, message_name="m")["m"]
    assert time.perf_counter() - start < 1.0
    assert np.array_equal(aligned, constant.sel(site=in_use["site"].values).values)
    assert np.array_equal(positions, in_use["site"].values % 2)


# ── the review's cases ────────────────────────────────────────────────────────


def test_a_missing_label_is_refused_in_a_plain_dim():
    """pandas 3 holds ["x", None] as strings, so the kind alone misses it."""
    with pytest.raises(ValueError, match="missing label"):
        as_coords({"pft": ["boreal", None]})


def test_integer_labels_held_as_objects_become_integers():
    coords = as_coords({
        "site": pd.Index([620, 865], dtype=object),
        "obs": pd.MultiIndex.from_arrays([pd.Index([1, 2], dtype=object), ["a", "b"]], names=["site", "part"]),
    })
    assert coords["site"].dtype == np.int64 and coords["obs"].levels[0].dtype == np.int64


def test_stacked_times_are_naive_and_within_nanoseconds():
    aware = pd.MultiIndex.from_arrays([[1], TIMES[:1].tz_localize("UTC")], names=["site", "time"])
    with pytest.raises(TypeError, match="time zone"):
        as_coords({"obs": aware})
    distant = pd.MultiIndex.from_arrays([[1], np.array(["3000-01-01"], dtype="datetime64[s]")], names=["site", "time"])
    with pytest.raises(ValueError, match="range"):
        as_coords({"obs": distant})


def test_an_own_dim_is_not_a_dim_read_at_labels():
    with pytest.raises(ValueError, match="drop them from own_dims"):
        aligned_constants({"c": on_sites([1.0, 2.0, 3.0])}, COORDS, dim_order=["site"], own_dims=["site"], message_name="c")


def test_a_constant_or_label_map_holds_each_label_once():
    sites = [620, 620, 865, 1037]
    with pytest.raises(ValueError, match="more than once"):
        aligned_constants({"c": on_sites([1.0, 2.0, 3.0, 4.0], sites=sites)}, COORDS, dim_order=["site"], message_name="c")
    with pytest.raises(ValueError, match="more than once"):
        aligned_label_maps({"m": on_sites(["boreal"] * 4, sites=sites, name="pft")}, COORDS, message_name="m")


def test_a_lookup_matches_a_stacked_dims_levels_by_kind():
    assert indexer(OBSERVATIONS, [(620.0, TIMES[0])]).tolist() == [-1]
    assert indexer(OBSERVATIONS, [(620, "2012-07-01")]).tolist() == [-1]
    assert indexer(OBSERVATIONS, [(620, TIMES[0])]).tolist() == [0]


def test_a_stacked_dim_is_read_at_the_coords_levels_in_their_order():
    swapped = on_observations([1.0, 2.0, 3.0], index=OBSERVATIONS.swaplevel())
    with pytest.raises(ValueError, match="levels in their order"):
        aligned_constants({"c": swapped}, COORDS, dim_order=["lai_observation"], message_name="c")
    plain = xr.DataArray(["2012"] * 3, dims="lai_observation", coords={"lai_observation": [0, 1, 2]}, name="year")
    with pytest.raises(ValueError, match="levels in their order"):
        aligned_label_maps({"m": plain}, COORDS, element_axes={"year": pd.Index(["2012"], name="year")}, message_name="m")


def test_label_maps_are_read_only_copies():
    original = on_sites(["boreal", "boreal", "temperate"], name="pft")
    held = as_label_maps({"m": original}, message_name="m")["m"]
    original.values[0] = "temperate"
    assert held.values[0] == "boreal" and not held.values.flags.writeable

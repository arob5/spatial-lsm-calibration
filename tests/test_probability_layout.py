"""Tests for Layout: named arrays as one flat vector, its three forms, its
spaces, its selection, and labeled values as one netCDF-ready Dataset.

A layout's entries are its components in declaration order, each block in
C order over its dims and element axes, in both spaces; it holds stacked
dims, whose levels merge with plain dims of their names in the index and in
selection. Every check is provoked once.
"""

from __future__ import annotations

import itertools
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.probability import names
from sipnet_calibration.probability.labels import aligned_constants, aligned_label_maps
from sipnet_calibration.probability.layout import (
    STACKED_DIMS_ATTRIBUTE,
    Layout,
    decode_labeled_values,
    encode_labeled_values,
    validate_labeled_values,
    validate_values_by_name,
)
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    POSITIVE_DEFINITE,
    SIMPLEX,
)

SITES = np.array([620, 865, 1037], dtype=np.int32)
PFT = ["boreal", "temperate"]
PARTS = ("leaf", "wood", "fine_root", "coarse_root")
YEARS = ("2012", "2013", "2014")
OBSERVATIONS = pd.MultiIndex.from_arrays(
    [[620, 620, 865, 1037], pd.to_datetime(["2012-07-01", "2013-07-01", "2012-07-01", "2014-07-01"])],
    names=["site", "time"],
)

SHARE = ArraySpec("share", support=OPEN_UNIT_INTERVAL, units="1")
ALLOCATION = ArraySpec("allocation", support=SIMPLEX, units="1", indexed_by=("pft",),
                       element_axes={"allocation_part": PARTS})
SOIL = ArraySpec("soil", support=POSITIVE, units="kg m-2", indexed_by=("site",))
RATE = ArraySpec("rate", support=POSITIVE, units="yr-1", indexed_by=("pft", "site"))
LOADING = ArraySpec("loading", units=None, element_axes={"row": 2, "column": 3})
SHARED = (SHARE, ALLOCATION, SOIL, RATE, LOADING)
COORDS = {"pft": PFT, "site": SITES}


@pytest.fixture(scope="module")
def layout() -> Layout:
    return Layout(SHARED, coords=COORDS)


@pytest.fixture(scope="module")
def stacked() -> Layout:
    """Parameters on a site dim, an observed component on a stacked dim with
    a site level, and a positive-definite value."""
    return Layout(
        [
            SOIL,
            ArraySpec("covariance", units="Mg2 ha-2", support=POSITIVE_DEFINITE,
                      element_axes={"year": YEARS, "other_year": YEARS}),
            ArraySpec("lai", units="1", indexed_by=("lai_observation",)),
        ],
        coords={"site": SITES, "lai_observation": OBSERVATIONS},
    )


def theta_of(layout: Layout, n: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, layout.unconstrained.size))


def expected_entries(specs, coords) -> list[tuple]:
    """``(component, *dim labels, element)`` per entry: the components in
    order, each block in C order over its dims, then its element axes; the
    element is a value's one label, or the tuple of its labels."""
    entries = []
    for spec in specs:
        dims = [list(coords[d]) for d in spec.indexed_by]
        elements = list(itertools.product(*[list(v) for v in spec.element_axes.values()]))
        for labels in itertools.product(*dims, elements):
            *dim_labels, element = labels
            element = () if not element else (str(element[0]) if len(element) == 1 else str(element))
            entries.append((spec.name, *[str(label) for label in dim_labels], *((element,) if element != () else ())))
    return entries


def entries_of(layout: Layout) -> list[tuple]:
    """``(component, *labels)`` per entry of *layout*'s index, its missing
    levels dropped and its labels as strings."""
    frame = layout.index.to_frame(index=False)
    return [tuple(str(v) for v in row if not pd.isna(v)) for row in frame.itertuples(index=False)]


# ── the canonical order ───────────────────────────────────────────────────────


@pytest.mark.parametrize("space", ["natural", "unconstrained"])
def test_the_entries_are_in_declaration_then_c_order(layout, space):
    mine = layout if space == "natural" else layout.unconstrained
    specs = SHARED if space == "natural" else [spec.unconstrained() for spec in SHARED]
    assert entries_of(mine) == expected_entries(specs, COORDS)
    assert mine.size == len(expected_entries(specs, COORDS))


def test_the_natural_values_are_each_blocks_bijector(layout):
    theta = theta_of(layout)
    natural = layout.to_natural(theta)
    unconstrained = layout.unconstrained.flat_to_values(theta)
    values = layout.flat_to_values(natural)
    for spec in SHARED:
        assert np.allclose(values[spec.name], spec.bijector.forward(unconstrained[spec.name]))
    assert np.allclose(layout.to_unconstrained(natural), theta)
    assert bool(np.all(layout.contains(natural)))
    labeled = layout.flat_to_labeled(natural, batch_dims=("sample",))
    for spec in SHARED:
        assert labeled[spec.name].dims == ("sample", *spec.indexed_by, *spec.element_axes)
        assert np.array_equal(labeled[spec.name].values, values[spec.name])


@pytest.mark.parametrize(
    "selectors",
    [{"site": [865]}, {"pft": ["temperate"], "site": [1037, 620]}, {"component": ["soil", "rate"]}],
)
def test_positions_are_the_entries_the_selection_names(layout, selectors):
    for mine in (layout, layout.unconstrained):
        frame = mine.index.to_frame(index=False)
        keep = np.ones(len(frame), dtype=bool)
        for level, labels in selectors.items():
            column = frame[level]
            keep &= column.isna().to_numpy() | column.astype(str).isin([str(v) for v in labels]).to_numpy()
        assert np.array_equal(mine.positions(**selectors), np.flatnonzero(keep))
        assert entries_of(mine.select(**selectors)) == [entries_of(mine)[i] for i in np.flatnonzero(keep)]


def test_a_block_is_in_c_order_whatever_the_coords_order():
    """Where a component's dims are not in the coords' order, a layout keeps
    the block's C order rather than the coords'."""
    rate = ArraySpec("rate", support=POSITIVE, units="yr-1", indexed_by=("site", "pft"))
    layout = Layout([rate], coords={"pft": PFT, "site": SITES})
    index = layout.index
    assert [(int(s), p) for s, p in zip(index.get_level_values("site"), index.get_level_values("pft"))] == [
        (s, p) for s in SITES.tolist() for p in PFT
    ]
    flat = jnp.arange(6.0)
    assert np.array_equal(layout.flat_to_values(flat)["rate"], flat.reshape(3, 2))


def test_a_component_is_one_contiguous_slice(layout):
    components = layout.index.get_level_values("component")
    for name in layout:
        where = np.flatnonzero(components == name)
        assert layout.slice_of(name) == slice(int(where[0]), int(where[-1]) + 1)


def test_the_conversions_are_traceable(layout):
    @jax.jit
    def round_trip(theta):
        return layout.to_unconstrained(layout.values_to_flat(layout.flat_to_values(layout.to_natural(theta))))

    theta = theta_of(layout)
    assert np.allclose(round_trip(theta), theta)


# ── stacked dims ──────────────────────────────────────────────────────────────


def test_a_stacked_dims_levels_merge_with_plain_dims_of_their_names(stacked):
    assert stacked.level_names == ("component", "site", "time", "element")
    index = stacked.index
    lai = index[index.get_level_values("component") == "lai"]
    assert [(int(s), t) for _, s, t, _ in lai] == list(zip(OBSERVATIONS.get_level_values("site"), OBSERVATIONS.get_level_values("time")))
    soil = index[index.get_level_values("component") == "soil"]
    assert [int(s) for s in soil.get_level_values("site")] == SITES.tolist()
    assert soil.get_level_values("time").isna().all()


def test_the_index_levels_keep_their_kinds(stacked):
    index = stacked.index
    assert index.get_level_values("site").dtype == "Int64"
    assert index.get_level_values("time").dtype == "datetime64[ns]"


def test_a_level_selects_in_every_component(stacked):
    positions = stacked.positions(site=[620])
    picked = stacked.index[positions]
    assert set(picked.get_level_values("component")) == {"soil", "covariance", "lai"}
    assert set(int(s) for s in picked.get_level_values("site").dropna()) == {620}
    assert len(picked[picked.get_level_values("component") == "lai"]) == 2


def test_a_selection_by_level_restricts_the_stacked_labels(stacked):
    kept = stacked.select(time=[pd.Timestamp("2012-07-01")])
    assert kept.coords["lai_observation"].tolist() == [OBSERVATIONS[0], OBSERVATIONS[2]]
    assert kept.coords["site"].tolist() == SITES.tolist()
    both = stacked.select(site=[620], time=[pd.Timestamp("2013-07-01")])
    assert both.coords["lai_observation"].tolist() == [OBSERVATIONS[1]]


def test_a_stacked_dim_selects_by_its_tuples(stacked):
    kept = stacked.select(lai_observation=[OBSERVATIONS[3], OBSERVATIONS[0]])
    assert kept.coords["lai_observation"].tolist() == [OBSERVATIONS[0], OBSERVATIONS[3]]
    picked = stacked.index[stacked.positions(lai_observation=[OBSERVATIONS[3], OBSERVATIONS[0]])]
    lai = picked[picked.get_level_values("component") == "lai"]
    assert [(int(s), t) for _, s, t, _ in lai] == [OBSERVATIONS[0], OBSERVATIONS[3]]


def test_selection_keeps_the_layouts_flat_form(stacked):
    """The positions of a selection are its Flat form in either space, so
    theta's selected entries map to the natural form's selected entries."""
    theta = theta_of(stacked)
    natural = stacked.to_natural(theta)
    for selectors in [{"site": [865]}, {"time": [pd.Timestamp("2012-07-01")]}, {"component": ["lai"]}]:
        kept = stacked.select(**selectors)
        theta_positions = stacked.unconstrained.positions(**selectors)
        natural_positions = stacked.positions(**selectors)
        assert kept.unconstrained.size == len(theta_positions) and kept.size == len(natural_positions)
        assert kept.index.equals(stacked.index[natural_positions])
        assert np.array_equal(kept.to_natural(theta[:, theta_positions]), natural[:, natural_positions])


def test_a_selection_leaving_a_dim_empty_is_refused_but_has_positions(stacked):
    with pytest.raises(ValueError, match="keeps no label of 'lai_observation'"):
        stacked.select(site=[1037], time=[pd.Timestamp("2012-07-01")])
    positions = stacked.positions(site=[1037], time=[pd.Timestamp("2012-07-01")])
    assert "lai" not in set(stacked.index[positions].get_level_values("component"))
    assert len(stacked.select(site=[1037], time=[pd.Timestamp("2012-07-01")], component=["soil"]).index) == 1


def test_a_positive_definite_value_has_its_cholesky_entries_in_theta(stacked):
    elements = stacked.unconstrained.index[stacked.unconstrained.slice_of("covariance")].get_level_values("element")
    assert len(elements) == 6 and all(e.startswith("L[") for e in elements)
    natural = stacked.flat_to_values(stacked.to_natural(theta_of(stacked)))["covariance"]
    assert natural.shape == (3, 3, 3) and bool(POSITIVE_DEFINITE.contains(natural).all())


# ── labeled values ────────────────────────────────────────────────────────────


def test_labeled_values_carry_their_labels_and_attributes(stacked):
    labeled = stacked.flat_to_labeled(stacked.to_natural(theta_of(stacked)), batch_dims=(names.SAMPLE,))
    lai = labeled["lai"]
    assert lai.dims == ("sample", "lai_observation") and lai.indexes["lai_observation"].equals(OBSERVATIONS)
    assert lai["sample"].dtype == np.int64 and lai["sample"].values.tolist() == [0, 1, 2]
    assert labeled["soil"].attrs == {"support": "(0, inf)", "units": "kg m-2"}
    assert labeled["covariance"].dims == ("sample", "year", "other_year")


def test_labeled_values_are_read_by_label(stacked):
    natural = stacked.to_natural(theta_of(stacked))
    labeled = stacked.flat_to_labeled(natural, batch_dims=("sample",))
    shuffled = {
        "soil": labeled["soil"].isel(site=[2, 0, 1]).transpose("site", "sample"),
        "covariance": labeled["covariance"].isel(year=[1, 2, 0]),
        "lai": labeled["lai"].isel(lai_observation=[3, 1, 0, 2]),
        "theta": xr.DataArray(np.zeros(3), dims="sample"),
    }
    assert np.array_equal(stacked.labeled_to_flat(shuffled), natural)


def test_several_batch_dims_are_ordered_by_name(layout):
    values = layout.flat_to_values(layout.to_natural(theta_of(layout, n=6).reshape(2, 3, -1)))
    labeled = layout.values_to_labeled(values, batch_dims=("chain", "draw"))
    with pytest.raises(ValueError, match="fix their order"):
        layout.labeled_to_values(labeled)
    back = layout.labeled_to_values(labeled, batch_dims=("chain", "draw"))
    assert all(np.array_equal(back[n], values[n]) for n in layout)


def test_labeled_values_round_trip_through_netcdf(stacked, tmp_path):
    natural = stacked.to_natural(theta_of(stacked))
    labeled = stacked.flat_to_labeled(natural, batch_dims=("sample",))
    dataset = encode_labeled_values(labeled)
    assert "lai_observation__site" in dataset.coords and "lai_observation__time" in dataset.coords
    path = tmp_path / "labeled.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        decoded = decode_labeled_values(opened.load())
    assert decoded["lai"].indexes["lai_observation"].equals(OBSERVATIONS)
    assert decoded["lai"].indexes["lai_observation"].levels[1].dtype == "datetime64[ns]"
    assert decoded["soil"].attrs["units"] == "kg m-2"
    assert np.array_equal(stacked.labeled_to_flat(decoded), natural)


def test_two_stacked_dims_with_a_site_level_share_one_encoding():
    other = pd.MultiIndex.from_arrays([[865, 1037], pd.to_datetime(["2012-01-01", "2012-01-02"])], names=["site", "time"])
    layout = Layout(
        [ArraySpec("lai", units="1", indexed_by=("lai_observation",)), ArraySpec("nee", units="1", indexed_by=("nee_observation",))],
        coords={"lai_observation": OBSERVATIONS, "nee_observation": other},
    )
    labeled = layout.flat_to_labeled(jnp.arange(6.0))
    decoded = decode_labeled_values(encode_labeled_values(labeled))
    assert decoded["nee"].indexes["nee_observation"].equals(other)
    assert np.array_equal(layout.labeled_to_flat(decoded), jnp.arange(6.0))


def test_encoding_refuses_one_dim_labeled_two_ways(layout):
    labeled = layout.flat_to_labeled(layout.to_natural(theta_of(layout)), batch_dims=("sample",))
    labeled["soil"] = labeled["soil"].assign_coords(site=[1, 2, 3])
    with pytest.raises(ValueError, match="label one dim differently"):
        encode_labeled_values(labeled)


def test_decoding_needs_the_encodings_attribute():
    with pytest.raises(ValueError, match=STACKED_DIMS_ATTRIBUTE):
        decode_labeled_values(xr.Dataset({"x": ("a", [1.0])}))
    with pytest.raises(TypeError, match="Dataset"):
        decode_labeled_values({"x": 1})
    broken = xr.Dataset({"x": ("obs", [1.0])}, attrs={STACKED_DIMS_ATTRIBUTE: '{"obs": ["site"]}'})
    with pytest.raises(ValueError, match="lacks the level coordinates"):
        decode_labeled_values(broken)


# ── scale ─────────────────────────────────────────────────────────────────────


def test_eighty_thousand_labels_bind_in_under_a_second():
    """The labels of a pool-sized model: 8000 sites, and 80,000 (site, time)
    observations with a constant and a label map on them."""
    rng = np.random.default_rng(1)
    sites = np.arange(1, 8001, dtype=np.int32)
    observations = pd.MultiIndex.from_arrays(
        [np.repeat(sites, 10), np.tile(pd.date_range("2012-01-01", periods=10, freq="365D").values, 8000)],
        names=["site", "time"],
    )
    standard_deviation = xr.DataArray(
        rng.uniform(0.1, 1.0, len(observations))[::-1], dims="nee_observation",
        coords=xr.Coordinates.from_pandas_multiindex(observations[::-1], "nee_observation"),
    )
    year = xr.DataArray(
        observations.get_level_values("time").year.astype(str).values.astype(object), dims="nee_observation",
        coords=xr.Coordinates.from_pandas_multiindex(observations, "nee_observation"), name="year",
    )
    start = time.perf_counter()
    layout = Layout(
        [
            ArraySpec("soil", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
            ArraySpec("nee", units="1", indexed_by=("nee_observation",)),
        ],
        coords={"site": sites, "nee_observation": observations},
    )
    years = pd.Index(sorted(set(year.values)), name="year", dtype=object)
    aligned_constants({"sd": standard_deviation}, layout.coords, dim_order=["nee_observation"], message_name="c")
    aligned_label_maps({"year": year}, layout.coords, element_axes={"year": years}, message_name="m")
    layout.positions(site=[4977])
    assert time.perf_counter() - start < 1.0
    assert layout.size == 88_000


# ── validators and checks ─────────────────────────────────────────────────────


def test_values_by_name_are_validated(layout):
    values = layout.flat_to_values(jnp.zeros((2, layout.size)))
    with pytest.raises(TypeError, match="mapping"):
        validate_values_by_name([1.0], layout)
    with pytest.raises(KeyError, match="soil"):
        validate_values_by_name({k: v for k, v in values.items() if k != "soil"}, layout)
    with pytest.raises(ValueError, match="block shape"):
        validate_values_by_name({**values, "soil": jnp.zeros((2, 4))}, layout)
    with pytest.raises(ValueError, match="different batch shapes"):
        validate_values_by_name({**values, "soil": jnp.zeros((3, 3))}, layout)


def test_labeled_values_are_validated(stacked):
    labeled = stacked.flat_to_labeled(jnp.zeros((2, stacked.size)) + 1.0, batch_dims=("sample",))
    cases = [
        ({**labeled, "soil": np.ones(3)}, TypeError, "DataArray"),
        ({**labeled, "soil": labeled["soil"].isel(site=0)}, ValueError, "lacks"),
        ({**labeled, "soil": labeled["soil"].drop_vars("site")}, ValueError, "no 'site' coordinate"),
        ({**labeled, "soil": labeled["soil"].isel(site=[0, 1])}, ValueError, "not the layout's"),
        ({**labeled, "lai": labeled["lai"].drop_vars(["lai_observation", "site", "time"])}, ValueError, "coordinate"),
        ({**labeled, "soil": labeled["soil"] * np.nan}, ValueError, "non-finite"),
        ({**labeled, "soil": labeled["soil"].rename(sample="draw")}, ValueError, "different batch dims"),
    ]
    for values, error, match in cases:
        with pytest.raises(error, match=match):
            validate_labeled_values(values, stacked)
    with pytest.raises(ValueError, match="not the labeled values' batch dims"):
        validate_labeled_values(labeled, stacked, batch_dims=("draw",))


def test_batch_dims_are_checked(layout):
    values = layout.flat_to_values(jnp.zeros((2, layout.size)))
    with pytest.raises(ValueError, match="name one per axis"):
        layout.values_to_labeled(values)
    with pytest.raises(ValueError, match="named like"):
        layout.values_to_labeled(values, batch_dims=("site",))
    with pytest.raises(ValueError, match="ends in"):
        layout.flat_to_values(jnp.zeros(3))


@pytest.mark.parametrize(
    ("components", "coords", "error", "match"),
    [
        ([], {}, ValueError, "at least one component"),
        ([object()], {}, TypeError, "ArraySpecs"),
        ([SOIL, SOIL], {"site": SITES}, ValueError, "more than once"),
        ([SOIL], {}, KeyError, "coords lack"),
        ([SHARE], {"site": SITES}, ValueError, "no component is indexed by"),
        ([ArraySpec("x", units=None, indexed_by=("site", "obs"))], {"site": SITES, "obs": OBSERVATIONS}, ValueError, "both have"),
        ([ArraySpec("x", units=None, indexed_by=("obs",))],
         {"obs": OBSERVATIONS.set_names(["site", "element"])}, ValueError, "reserves"),
        ([ArraySpec("site", units=None), ArraySpec("x", units=None, indexed_by=("obs",))], {"obs": OBSERVATIONS}, ValueError, "named like a dim or level"),
        ([ArraySpec("x", units=None, element_axes={"time": 2}), ArraySpec("y", units=None, indexed_by=("obs",))],
         {"obs": OBSERVATIONS}, ValueError, "element axis 'time'"),
        ([ArraySpec("x", units=None, element_axes={"part": 2}), ArraySpec("y", units=None, element_axes={"part": 3})],
         {}, ValueError, "different labels"),
        ([ArraySpec("x", units=None, element_axes={"part": PARTS[:3]}),
          ArraySpec("y", units="1", support=SIMPLEX, element_axes={"part": PARTS[:3]})], {}, ValueError, "theta's layout"),
    ],
)
def test_layouts_are_checked(components, coords, error, match):
    with pytest.raises(error, match=match):
        Layout(components, coords=coords)


@pytest.mark.parametrize(
    ("selectors", "error", "match"),
    [
        ({"biome": ["x"]}, KeyError, "no dim or level"),
        ({"component": ["nothing"]}, KeyError, "no component"),
        ({"site": 620}, TypeError, "sequence"),
        ({"site": ["620"]}, TypeError, "integer"),
        ({"site": [4977]}, KeyError, "4977"),
        ({"site": [620, 620]}, ValueError, "more than once"),
        ({"site": []}, ValueError, "keeps nothing"),
        ({"time": ["2012"]}, TypeError, "datetime"),
    ],
)
def test_selectors_are_checked(stacked, selectors, error, match):
    with pytest.raises(error, match=match):
        stacked.positions(**selectors)


def test_a_layout_is_frozen_pickles_and_describes_itself(stacked):
    with pytest.raises(AttributeError):
        stacked.coords = {}
    copy = pickle.loads(pickle.dumps(stacked))
    assert copy.index.equals(stacked.index) and copy.coords["lai_observation"].equals(OBSERVATIONS)
    described = stacked.describe()
    assert list(described.index) == ["soil", "covariance", "lai"]
    assert described.loc["covariance", "unconstrained_shape"] == (6,) and described.loc["lai", "entries"] == 4
    assert "soil" in stacked and "x" not in stacked and [] not in stacked and len(stacked) == 3
    assert list(reversed(stacked)) == ["lai", "covariance", "soil"]
    assert repr(stacked).startswith("Layout(size=16")



# ── the review's cases ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "times",
    [
        [pd.Timestamp("2012-07-01")],
        pd.to_datetime(["2012-07-01"]),
        np.array(["2012-07-01"], dtype="datetime64[ns]"),
        [np.datetime64("2012-07-01", "ns")],
        [np.datetime64("2012-07-01")],
    ],
)
def test_times_select_in_any_of_their_forms(stacked, times):
    picked = stacked.index[stacked.positions(time=times, component=["lai"])]
    assert [int(s) for s in picked.get_level_values("site")] == [620, 865]


def test_a_stacked_dims_tuples_are_checked_level_by_level(stacked):
    with pytest.raises(TypeError, match="integer, datetime"):
        stacked.positions(lai_observation=[(620.0, pd.Timestamp("2012-07-01"))])
    with pytest.raises(TypeError, match="integer, datetime"):
        stacked.positions(lai_observation=[(620, "2012-07-01")])


def test_labeled_values_share_their_draws_in_one_order(stacked):
    labeled = stacked.flat_to_labeled(stacked.to_natural(theta_of(stacked)), batch_dims=("sample",))
    with pytest.raises(ValueError, match="same draws, in one order"):
        stacked.labeled_to_flat({**labeled, "soil": labeled["soil"].isel(sample=[1, 0, 2])})
    with pytest.raises(ValueError, match="same draws, in one order"):
        stacked.labeled_to_flat({**labeled, "soil": labeled["soil"].isel(sample=[0, 1])})


def test_encoding_refuses_one_stacked_dim_ordered_two_ways():
    layout = Layout(
        [ArraySpec("lai", units="1", indexed_by=("obs",)), ArraySpec("residual", units="1", indexed_by=("obs",))],
        coords={"obs": OBSERVATIONS},
    )
    labeled = layout.flat_to_labeled(jnp.arange(8.0))
    reordered = {**labeled, "lai": labeled["lai"].isel(obs=[3, 2, 1, 0])}
    assert np.array_equal(layout.labeled_to_flat(reordered), jnp.arange(8.0))
    with pytest.raises(ValueError, match="label one dim differently"):
        encode_labeled_values(reordered)


def test_labeled_values_must_hold_exactly_the_layouts_labels(stacked):
    labeled = stacked.flat_to_labeled(stacked.to_natural(theta_of(stacked)), batch_dims=("sample",))
    wrong = labeled["soil"].assign_coords(site=[620, 865, 4977])
    wider = xr.concat([labeled["soil"], labeled["soil"].isel(site=[0]).assign_coords(site=[4977])], "site")
    swapped = labeled["lai"].copy()
    swapped = swapped.drop_vars(["lai_observation", "site", "time"]).assign_coords(
        xr.Coordinates.from_pandas_multiindex(OBSERVATIONS.swaplevel(), "lai_observation")
    )
    for values, match in [({**labeled, "soil": wrong}, "not the layout's"), ({**labeled, "soil": wider}, "not the layout's"),
                          ({**labeled, "lai": swapped}, "levels")]:
        with pytest.raises(ValueError, match=match):
            validate_labeled_values(values, stacked)


def test_a_level_selects_a_label_its_plain_dim_lacks():
    """Site 1037 is observed but not in the site dim: selecting it keeps its
    observation and refuses the site dim, which it leaves empty."""
    layout = Layout(
        [ArraySpec("soil", units="1", indexed_by=("site",)), ArraySpec("lai", units="1", indexed_by=("obs",))],
        coords={"site": [620, 865], "obs": OBSERVATIONS},
    )
    picked = layout.index[layout.positions(site=[1037])]
    assert list(picked.get_level_values("component")) == ["lai"]
    kept = layout.select(site=[1037], component=["lai"])
    assert kept.coords["obs"].tolist() == [OBSERVATIONS[3]]
    with pytest.raises(ValueError, match="keeps no label of 'site'"):
        layout.select(site=[1037])

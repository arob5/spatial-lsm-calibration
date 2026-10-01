"""Tests for the parameter vector: its layout, its three forms, its spaces
and its selection.

The order is lexicographic over the named levels with a missing label
first, a value is one contiguous unit in C order, and the default order is
parameter-major. The conversions are layout only and round-trip; the
labeled form is read strictly, by label. Selection keeps the vector's
order, so its positions are the selected vector's Flat form, in either
space. Every check is provoked once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.support import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
)
from sipnet_calibration.parameters.vector import (
    ParameterVector,
    check_parameter_vectors_share_a_layout,
    validate_parameter_dataset,
    validate_values_by_parameter,
)

SITES = np.array([620, 865, 1037], dtype=np.int32)
PFT = ["boreal", "temperate"]
PARTS = ("leaf", "wood", "fine_root", "coarse_root")

SHARE = Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1")
ALLOCATION = Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                       element_labels={"allocation_part": PARTS})
SOIL = Parameter(name="soil", support=POSITIVE, units="kg m-2", indexed_by=("site",), long_name="soil carbon")
RATE = Parameter(name="rate", support=POSITIVE, units="yr-1", indexed_by=("site", "pft"))
LOADING = Parameter(name="loading", units=None, shape=(2, 3))


def vector_of(*parameters, order=None, **coords) -> ParameterVector:
    coords = coords or {"pft": PFT, "site": SITES}
    used = {d for p in parameters for d in p.indexed_by}
    return ParameterVector(parameters=parameters, coords={d: v for d, v in coords.items() if d in used}, order=order)


@pytest.fixture(scope="module")
def vector() -> ParameterVector:
    return vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING)


@pytest.fixture(scope="module")
def site_major() -> ParameterVector:
    return vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING, order=("site", "parameter", "pft"))


def draws(vector: ParameterVector, n: int = 3, seed: int = 0) -> jax.Array:
    """Natural Flat values: theta of N(0, I) mapped to the natural space."""
    theta = np.random.default_rng(seed).standard_normal((n, vector.unconstrained.size))
    return vector.to_natural(theta)


# ── the layout ────────────────────────────────────────────────────────────────


def test_the_sizes_count_every_number(vector):
    # share 1, allocation 2 x 4, soil 3, rate 3 x 2, loading 6; theta drops one per simplex.
    assert vector.size == 1 + 8 + 3 + 6 + 6
    assert vector.unconstrained.size == 1 + 6 + 3 + 6 + 6
    assert vector.block_shape("rate") == (3, 2) and vector.block_shape("allocation") == (2, 4)


def test_the_default_order_is_parameter_major_in_c_order(vector):
    index = vector.index
    assert list(index.names) == ["parameter", "pft", "site", "element"]
    assert list(index.get_level_values("parameter").unique()) == ["share", "allocation", "soil", "rate", "loading"]
    rate = index[index.get_level_values("parameter") == "rate"]
    # rate is indexed by (site, pft), but its blocks sort by the order's dims, pft then site.
    assert [(p, int(s)) for _, p, s, _ in rate] == [(p, s) for p in PFT for s in SITES.tolist()]


def test_a_missing_label_sorts_before_every_label(site_major):
    index = site_major.index
    sites = index.get_level_values("site")
    # Everything not indexed by site first, then each site's blocks in turn.
    assert list(index.get_level_values("parameter")[sites.isna()].unique()) == ["share", "allocation", "loading"]
    by_site = index[~sites.isna()]
    assert [(n, int(s)) for n, _, s, _ in by_site][:3] == [("soil", 620), ("rate", 620), ("rate", 620)]
    assert list(dict.fromkeys(int(s) for s in by_site.get_level_values("site"))) == SITES.tolist()


def test_a_value_is_one_contiguous_unit(site_major):
    index = site_major.index
    loading = np.flatnonzero(index.get_level_values("parameter") == "loading")
    assert np.all(np.diff(loading) == 1)
    assert list(index[loading].get_level_values("element")) == [(a, b) for a in "01" for b in "012"]


def test_the_index_holds_na_where_a_parameter_is_not_indexed(vector):
    first = vector.index[0]
    assert first[0] == "share" and pd.isna(first[1]) and pd.isna(first[2]) and pd.isna(first[3])
    assert vector.index.get_level_values("site").dtype == "Int32"


def test_entry_names_show_labels_elements_and_long_names(vector):
    names = vector.unconstrained.entry_names
    assert names[0] == "logit(share)"
    assert names[1] == "alr(allocation)[boreal][leaf]"
    assert "log(soil carbon)[865]" in names
    assert "log(rate)[620, temperate]" in names
    assert names[-1] == "loading[1, 2]"


def test_describe_has_a_row_per_parameter(vector):
    table = vector.describe()
    assert list(table.index) == list(vector)
    assert table.loc["allocation", "unconstrained_shape"] == (3,)
    assert table.loc["rate", "entries"] == 6 and table.loc["rate", "indexed_by"] == "site, pft"


def test_the_vector_holds_its_parameters(vector):
    assert vector["soil"] is SOIL and "soil" in vector and "nothing" not in vector and ["soil"] not in vector
    assert next(reversed(vector)) == "loading" and len(vector) == 5
    assert repr(vector).startswith("ParameterVector(size=24")
    with pytest.raises(KeyError, match="no parameter 'nothing'"):
        vector["nothing"]


# ── the forms ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", ["vector", "site_major"])
def test_flat_and_values_round_trip(which, request):
    vector = request.getfixturevalue(which)
    flat = draws(vector)
    values = vector.flat_to_values(flat)
    assert {n: v.shape for n, v in values.items()} == {
        "share": (3,), "allocation": (3, 2, 4), "soil": (3, 3), "rate": (3, 3, 2), "loading": (3, 2, 3),
    }
    np.testing.assert_array_equal(vector.values_to_flat(values), flat)


def test_values_are_read_at_the_index(site_major):
    flat = jnp.arange(site_major.size, dtype=jnp.float64)
    values = site_major.flat_to_values(flat)
    index = site_major.index
    for position in np.asarray(values["rate"]).ravel().astype(int):
        assert index[position][0] == "rate"
    site = np.flatnonzero((index.get_level_values("parameter") == "rate") & (index.get_level_values("site") == 865))
    np.testing.assert_array_equal(np.asarray(values["rate"])[1], site.reshape(2))


def test_the_conversions_are_traceable(vector):
    theta = jnp.zeros((2, vector.unconstrained.size))
    jitted = jax.jit(lambda t: vector.values_to_flat(vector.flat_to_values(vector.to_natural(t))))
    np.testing.assert_allclose(jitted(theta), vector.to_natural(theta))
    gradient = jax.grad(lambda t: vector.flat_to_values(vector.to_natural(t))["soil"].sum())(theta[0])
    assert gradient.shape == theta[0].shape


def test_the_labeled_form_follows_the_data_model(vector):
    dataset = vector.flat_to_dataset(draws(vector), batch_dims=("sample",))
    assert dataset["allocation"].dims == ("sample", "pft", "allocation_part")
    assert dataset["rate"].dims == ("sample", "site", "pft")
    assert dataset["loading"].dims == ("sample", "loading_axis_0", "loading_axis_1")
    assert dataset["sample"].dtype == np.int64 and dataset["site"].dtype == np.int32
    assert list(dataset["allocation_part"].values) == list(PARTS)
    assert dataset["soil"].attrs == {"support": "(0, inf)", "units": "kg m-2", "long_name": "soil carbon"}
    assert dataset["loading"].attrs == {"support": "real"}
    assert all(not dataset[c].attrs for c in dataset.coords)


def test_the_labeled_form_round_trips_through_netcdf(vector, tmp_path):
    flat = draws(vector)
    dataset = vector.flat_to_dataset(flat, batch_dims=("sample",))
    dataset.to_netcdf(tmp_path / "values.nc", engine="h5netcdf")
    with xr.open_dataset(tmp_path / "values.nc", engine="h5netcdf") as stored:
        np.testing.assert_allclose(vector.dataset_to_flat(stored.load()), flat, rtol=1e-15)


def test_the_labeled_form_is_read_by_label_and_rows_keep_their_order(vector):
    flat = draws(vector)
    dataset = vector.flat_to_dataset(flat, batch_dims=("sample",))
    shuffled = dataset.isel(site=[2, 0, 1], pft=[1, 0], allocation_part=[3, 1, 0, 2]).transpose("pft", "sample", ...)
    shuffled = shuffled.assign_coords(sample=[1000, 1001, 1002])
    np.testing.assert_allclose(vector.dataset_to_flat(shuffled), flat)


def test_no_batch_dim_is_one_value(vector):
    flat = draws(vector)[0]
    dataset = vector.flat_to_dataset(flat)
    assert dataset["share"].dims == ()
    np.testing.assert_allclose(vector.dataset_to_flat(dataset), flat)


def test_several_batch_dims_are_ordered_by_name(vector):
    flat = draws(vector, n=6).reshape(2, 3, -1)
    dataset = vector.flat_to_dataset(flat, batch_dims=("chain", "draw"))
    with pytest.raises(ValueError, match="fix their order with batch_dims="):
        vector.dataset_to_flat(dataset)
    np.testing.assert_allclose(vector.dataset_to_flat(dataset, batch_dims=("chain", "draw")), flat)
    np.testing.assert_allclose(vector.dataset_to_flat(dataset, batch_dims=("draw", "chain")), flat.swapaxes(0, 1))
    with pytest.raises(ValueError, match="are not the parameter dataset's batch dims"):
        vector.dataset_to_flat(dataset, batch_dims=("chain",))


def test_the_labeled_form_is_read_strictly(vector):
    dataset = vector.flat_to_dataset(draws(vector), batch_dims=("sample",))
    with pytest.raises(TypeError, match="is an xarray Dataset"):
        vector.dataset_to_values(dataset.to_dataframe())
    with pytest.raises(KeyError, match="no variable \\['soil'\\]"):
        vector.dataset_to_values(dataset.drop_vars("soil"))
    with pytest.raises(ValueError, match="take dataset\\[list\\(vector\\)\\] first"):
        vector.dataset_to_values(dataset.assign(extra=dataset["share"]))
    with pytest.raises(ValueError, match="which lacks \\['site'\\]"):
        vector.dataset_to_values(dataset.assign(soil=dataset["soil"].isel(site=0, drop=True)))
    with pytest.raises(ValueError, match="not the vector's"):
        vector.dataset_to_values(dataset.isel(site=[0, 1]))
    with pytest.raises(ValueError, match="not the vector's"):
        vector.dataset_to_values(dataset.isel(site=[0, 1, 2, 2]))
    with pytest.raises(ValueError, match="has no 'site' coordinate"):
        vector.dataset_to_values(dataset.drop_vars("site"))
    with pytest.raises(ValueError, match="non-finite"):
        vector.dataset_to_values(dataset.assign(share=dataset["share"].where(dataset["sample"] > 0)))
    with pytest.raises(ValueError, match="different batch dims"):
        vector.dataset_to_values(dataset.assign(share=dataset["share"].rename(sample="draw")))
    assert validate_parameter_dataset(dataset, vector) == ("sample",)


def test_values_by_parameter_are_validated(vector):
    values = vector.flat_to_values(draws(vector))
    with pytest.raises(TypeError, match="are a mapping"):
        vector.values_to_flat([values["share"]])
    with pytest.raises(KeyError, match="hold no \\['soil'\\]"):
        vector.values_to_flat({n: v for n, v in values.items() if n != "soil"})
    with pytest.raises(ValueError, match="does not end in its block shape"):
        vector.values_to_flat({**values, "soil": values["soil"][..., :2]})
    with pytest.raises(ValueError, match="different batch shapes"):
        vector.values_to_flat({**values, "share": values["share"][:2]})
    validate_values_by_parameter({**values, "other": 1.0}, vector)


def test_batch_dims_name_the_leading_axes_and_nothing_else(vector):
    values = vector.flat_to_values(draws(vector))
    with pytest.raises(ValueError, match="name one per axis"):
        vector.values_to_dataset(values)
    with pytest.raises(ValueError, match="named like a dim, an element axis or a parameter"):
        vector.values_to_dataset(values, batch_dims=("site",))
    with pytest.raises(ValueError, match="named like a dim, an element axis or a parameter"):
        vector.values_to_dataset(values, batch_dims=("allocation_part",))


def test_flat_ends_in_the_size(vector):
    with pytest.raises(ValueError, match="ends in 24 entries"):
        vector.flat_to_values(jnp.zeros(23))
    with pytest.raises(ValueError, match="ends in 22 entries"):
        vector.to_natural(jnp.zeros(24))


# ── the spaces ────────────────────────────────────────────────────────────────


def test_the_spaces_round_trip_and_values_lie_in_their_supports(vector):
    theta = np.random.default_rng(1).standard_normal((5, vector.unconstrained.size))
    natural = vector.to_natural(theta)
    np.testing.assert_allclose(vector.to_unconstrained(natural), theta, atol=1e-12)
    assert bool(jnp.all(vector.contains(natural)))
    values = vector.flat_to_values(natural)
    np.testing.assert_allclose(np.asarray(values["allocation"]).sum(axis=-1), 1.0)
    assert bool(jnp.all(values["soil"] > 0))


def test_the_transform_is_applied_block_by_block(vector):
    theta = np.random.default_rng(2).standard_normal(vector.unconstrained.size)
    unconstrained = vector.unconstrained.flat_to_values(theta)
    natural = vector.flat_to_values(vector.to_natural(theta))
    np.testing.assert_allclose(natural["soil"], np.exp(unconstrained["soil"]))
    alr = np.asarray(unconstrained["allocation"])
    expected = np.concatenate([np.exp(alr), np.ones((2, 1))], axis=-1)
    np.testing.assert_allclose(natural["allocation"], expected / expected.sum(axis=-1, keepdims=True))


def test_a_value_outside_its_support_is_not_contained_and_maps_to_non_finite_theta(vector):
    natural = draws(vector, n=2)
    position = vector.positions(parameter=["soil"], site=[865])
    outside = natural.at[1, position].set(0.0)
    assert vector.contains(outside).tolist() == [True, False]
    assert not bool(jnp.all(jnp.isfinite(vector.to_unconstrained(outside)[1])))


# ── selection ─────────────────────────────────────────────────────────────────

SELECTIONS = [
    {"parameter": ["soil", "allocation"]},
    {"site": [1037, 620]},
    {"pft": ["temperate"]},
    {"site": [865], "pft": ["boreal"]},
    {"parameter": ["rate"], "site": [865]},
    {"parameter": ["share", "loading"]},
]


@pytest.mark.parametrize("which", ["vector", "site_major"])
@pytest.mark.parametrize("selectors", SELECTIONS)
def test_positions_are_the_selected_vectors_flat_form(which, selectors, request):
    vector = request.getfixturevalue(which)
    for space in (vector, vector.unconstrained):
        flat = jnp.asarray(np.random.default_rng(3).standard_normal((2, space.size)))
        selected = space.select(**selectors)
        values = selected.flat_to_values(flat[..., space.positions(**selectors)])
        whole = space.flat_to_values(flat)
        for name, value in values.items():
            parameter = space[name]
            expected = whole[name]
            for axis, dim in enumerate(parameter.indexed_by):
                kept = np.flatnonzero(space.coords[dim].isin(selected.coords[dim]))
                expected = jnp.take(expected, kept, axis=1 + axis)
            np.testing.assert_array_equal(value, expected)
    assert vector.select(**selectors).unconstrained.index.equals(vector.unconstrained.select(**selectors).index)


def test_select_keeps_order_and_parameters_it_does_not_split(vector):
    selected = vector.select(site=[1037, 620])
    assert list(selected) == list(vector)
    assert selected.coords["site"].tolist() == [620, 1037] and selected.coords["site"].dtype == np.int32
    assert selected["share"] is SHARE
    only = vector.select(parameter=["loading", "share"])
    assert list(only) == ["share", "loading"] and dict(only.coords) == {}
    assert vector.select(parameter=["soil"]).order == ("parameter", "site")


def test_selectors_are_refused_by_the_projects_rules(vector):
    with pytest.raises(TypeError, match="pass \\[865\\]"):
        vector.select(site=865)
    with pytest.raises(TypeError, match="no order to keep"):
        vector.select(site={865})
    with pytest.raises(TypeError, match="are integers"):
        vector.select(site=["865"])
    with pytest.raises(TypeError, match="are strings"):
        vector.select(pft=[0])
    with pytest.raises(KeyError, match="no dim 'biome'"):
        vector.select(biome=["x"])
    with pytest.raises(KeyError, match="no label"):
        vector.positions(site=[4977])
    with pytest.raises(KeyError, match="no parameter 'nothing'"):
        vector.select(parameter=["nothing"])
    with pytest.raises(ValueError, match="more than once"):
        vector.select(site=[865, 865])
    with pytest.raises(ValueError, match="keeps nothing"):
        vector.select(parameter=[])


# ── construction ──────────────────────────────────────────────────────────────


def test_coords_keep_their_labels_dtype_and_order():
    vector = vector_of(SOIL, site=np.array([27, 1], dtype=np.int32))
    assert vector.coords["site"].tolist() == [27, 1] and vector.coords["site"].name == "site"


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: ParameterVector(parameters=[]), ValueError, "at least one parameter"),
        (lambda: ParameterVector(parameters=[SHARE, SHARE]), ValueError, "more than once"),
        (lambda: ParameterVector(parameters=["share"]), TypeError, "must be Parameters"),
        (lambda: ParameterVector(parameters=[SOIL]), KeyError, "coords lack"),
        (lambda: ParameterVector(parameters=[SHARE], coords={"site": SITES}), ValueError, "no parameter is indexed by"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": [1, 1]}), ValueError, "more than once"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": []}), ValueError, "holds no label"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": [1, "2"]}), TypeError, "neither all integers"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": [True, False]}), TypeError, "neither all integers"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": 27}), TypeError, "must be a sequence"),
        (lambda: ParameterVector(parameters=[SOIL], coords=[("site", SITES)]), TypeError, "must be a mapping"),
        (lambda: ParameterVector(parameters=[SOIL], coords={"site": SITES}, order=("site",)), ValueError, "permutation"),
    ],
)
def test_a_malformed_vector_is_refused(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_names_that_would_collide_in_the_labeled_form_are_refused():
    with pytest.raises(ValueError, match="named like a dim"):
        vector_of(Parameter(name="site", units=None, indexed_by=("site",)))
    with pytest.raises(ValueError, match="named like a dim or a parameter"):
        vector_of(SOIL, Parameter(name="x", units=None, shape=(3,), element_labels={"site": ("a", "b", "c")}))
    with pytest.raises(ValueError, match="with different labels"):
        vector_of(
            Parameter(name="x", units=None, shape=(2,), element_labels={"part": ("a", "b")}),
            Parameter(name="y", units=None, shape=(2,), element_labels={"part": ("a", "c")}),
        )
    shared = vector_of(
        Parameter(name="x", units=None, shape=(2,), element_labels={"part": ("a", "b")}),
        Parameter(name="y", units=None, shape=(2,), element_labels={"part": ("a", "b")}),
    )
    assert shared.flat_to_dataset(jnp.zeros(4))["part"].values.tolist() == ["a", "b"]


# ── layouts ───────────────────────────────────────────────────────────────────


def test_vectors_built_alike_share_a_layout(vector):
    check_parameter_vectors_share_a_layout(vector, vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING))


@pytest.mark.parametrize(
    "other",
    [
        lambda: vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING, order=("site", "parameter", "pft")),
        lambda: vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING, pft=PFT[::-1], site=SITES),
        lambda: vector_of(SHARE, ALLOCATION, SOIL, RATE, LOADING, pft=PFT, site=SITES[:2]),
        lambda: vector_of(Parameter(name="share", support=REAL, units="1"), ALLOCATION, SOIL, RATE, LOADING),
    ],
)
def test_vectors_that_differ_do_not_share_a_layout(vector, other):
    with pytest.raises(ValueError, match="differ in their"):
        check_parameter_vectors_share_a_layout(vector, other())


def test_a_transform_that_differs_numerically_breaks_the_layout():
    from tensorflow_probability.substrates import jax as tfp

    softplus = Parameter(name="soil", support=POSITIVE, units="kg m-2", indexed_by=("site",),
                         long_name="soil carbon", bijector=tfp.bijectors.Softplus())
    with pytest.raises(ValueError, match="differ in their transforms"):
        check_parameter_vectors_share_a_layout(vector_of(SOIL), vector_of(softplus))


def test_a_simplex_and_another_value_on_one_axis_are_refused_when_built():
    with pytest.raises(ValueError, match="different labels in theta's layout"):
        vector_of(
            Parameter(name="x", support=SIMPLEX, units="1", shape=(2,), element_labels={"part": ("a", "b")}),
            Parameter(name="y", units=None, shape=(2,), element_labels={"part": ("a", "b")}),
        )


def test_site_ids_of_two_integer_dtypes_are_one_layout():
    sites = [1, 27]
    first = vector_of(SOIL, site=np.asarray(sites, dtype=np.int64))
    check_parameter_vectors_share_a_layout(first, vector_of(SOIL, site=np.asarray(sites, dtype=np.int32)))
    with pytest.raises(ValueError, match="differ in their"):
        check_parameter_vectors_share_a_layout(first, vector_of(SOIL, site=np.asarray(sites[::-1])))


def test_a_value_on_a_closed_end_has_no_theta():
    from sipnet_calibration.parameters.support import NON_NEGATIVE, UNIT_INTERVAL

    vector = vector_of(Parameter(name="share", support=UNIT_INTERVAL, units="1"),
                       Parameter(name="amount", support=NON_NEGATIVE, units="1"))
    natural = jnp.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 2.0]])
    assert vector.contains(natural).tolist() == [False, False, True]
    assert not bool(jnp.isfinite(vector.to_unconstrained(natural)[:2]).all())


def test_an_empty_batch_goes_to_flat_and_back():
    vector = vector_of(Parameter(name="scalar", units=None), Parameter(name="shares", support=SIMPLEX, units="1", shape=(3,)))
    theta = jnp.zeros((0, vector.unconstrained.size))
    assert vector.to_natural(theta).shape == (0, vector.size)
    assert vector.values_to_flat(vector.flat_to_values(jnp.zeros((0, vector.size)))).shape == (0, vector.size)

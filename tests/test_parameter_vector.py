"""Tests for the parameter vector.

The shapes of every form are checked against the module's table, theta
round trips through natural values and the labeled form, a smaller vector
reads a larger one's labeled form, and the per-site view is a collection of
fields. Each construction and entry-point check is provoked once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from conftest import site_table_of
from sipnet_calibration.fields import validate_field
from sipnet_calibration.parameter_vector import (
    NO_DIM,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    OpenInterval,
    Parameter,
    ParameterVector,
    Support,
    check_parameter_vectors_share_a_layout,
    probe_points,
    validate_natural_values,
    validate_parameter_dataset,
)

tfb = tfp.bijectors

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
ALLOCATION_NAMES = ("leaf", "wood", "fine_root", "coarse_root")


def parameters() -> list[Parameter]:
    return [
        Parameter(name="capacity", support=POSITIVE, units="nmol g-1 s-1"),
        Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1"),
        Parameter(
            name="allocation", support=SIMPLEX, units="1", dim="pft", natural_names=ALLOCATION_NAMES
        ),
        Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", dim="site"),
        Parameter(name="q10", support=OpenInterval(1.0, 5.0), units="1", dim="pft"),
        Parameter(name="offsets", support=REAL, units=None, natural_names=("a", "b")),
    ]


def site_table(**covariates: list[float]) -> pd.DataFrame:
    return site_table_of(*SITES).assign(**covariates)


@pytest.fixture(scope="module")
def vector() -> ParameterVector:
    return ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"pft": PFT})


@pytest.fixture(scope="module")
def theta(vector) -> jax.Array:
    return jax.random.normal(jax.random.key(0), (5, vector.dimension))


# ── shapes and the index ──────────────────────────────────────────────────────


def test_the_dimension_counts_every_entry(vector):
    # 1 + 1 + 2 PFTs x 3 + 3 sites + 2 PFTs + 2
    assert vector.dimension == 15
    assert len(vector.index) == len(vector.entry_names) == 15


@pytest.mark.parametrize(
    "name, value_shape, unconstrained_shape",
    [
        ("capacity", (), ()),
        ("offsets", (2,), (2,)),
        ("soil_carbon", (3,), (3,)),
        ("allocation", (2, 4), (2, 3)),
    ],
)
def test_shapes_follow_the_table(vector, theta, name, value_shape, unconstrained_shape):
    assert vector.value_shape(name) == value_shape
    assert vector.unconstrained_shape(name) == unconstrained_shape
    assert vector.to_natural(theta)[name].shape == (5, *value_shape)
    at_sites = vector.at_sites(vector.to_natural(theta))[name]
    assert at_sites.shape == (5, 3) + vector[name].natural_shape


def test_the_index_has_a_dim_level_and_no_dim_marks_undimensioned_entries(vector):
    first = vector.index[0]
    assert first == ("capacity", NO_DIM, NO_DIM, "log(capacity)")
    assert vector.index[vector.positions(parameter_name="soil_carbon")[1]] == (
        "soil_carbon",
        "site",
        27,
        "log(soil_carbon)",
    )


def test_theta_is_parameters_then_dim_labels_then_unconstrained_names(vector):
    allocation = vector.positions(parameter_name="allocation")
    labels = vector.index[allocation].get_level_values("dim_label")
    assert list(labels) == ["conifer"] * 3 + ["deciduous"] * 3
    assert vector.index[allocation[0]][3] == "alr(leaf:coarse_root)"


def test_positions_select_by_dim_and_dim_label(vector):
    assert list(vector.positions(dim="site", dim_label=27)) == [9]
    assert len(vector.positions(dim="pft", dim_label="conifer")) == 4
    assert list(vector.positions(parameter_name="q10", dim_label="deciduous")) == [12]


def test_a_site_dim_label_is_compared_as_a_site_id(vector):
    with pytest.raises(KeyError, match="not a dim label"):
        vector.positions(dim="site", dim_label="27")


def test_a_dim_label_needs_its_dim(vector):
    with pytest.raises(ValueError, match="needs dim= or parameter_name="):
        vector.positions(dim_label="conifer")


def test_no_dim_is_not_a_selector(vector):
    with pytest.raises(KeyError, match="NO_DIM"):
        vector.positions(dim=NO_DIM)


def test_an_unknown_selector_is_a_key_error(vector):
    with pytest.raises(KeyError, match="no entry of theta has dim"):
        vector.positions(dim="biome")
    with pytest.raises(KeyError, match="no parameter"):
        vector.positions(parameter_name="nothing")


def test_describe_has_one_row_per_parameter(vector):
    table = vector.describe()
    assert list(table.index) == list(vector.parameter_names)
    assert table.loc["allocation", "entries"] == 6
    assert table.loc["q10", "support"] == "(1, 5)"


# ── coordinates ───────────────────────────────────────────────────────────────


def test_theta_round_trips_through_natural_values(vector, theta):
    np.testing.assert_allclose(vector.to_unconstrained(vector.to_natural(theta)), theta, atol=1e-12)


def test_natural_values_lie_in_their_supports(vector, theta):
    natural_values = vector.to_natural(theta * 5)
    for parameter in vector.parameters:
        assert bool(jnp.all(parameter.support.contains(natural_values[parameter.name])))


def test_to_natural_is_traceable(vector, theta):
    jitted = jax.jit(lambda t: vector.to_natural(t)["q10"].sum())
    assert jnp.isfinite(jax.grad(jitted)(theta[0])).all()


def test_values_outside_the_support_map_to_non_finite_theta(vector, theta):
    natural_values = dict(vector.to_natural(theta))
    natural_values["capacity"] = -natural_values["capacity"]
    assert not bool(jnp.isfinite(vector.to_unconstrained(natural_values)).all())


def test_at_sites_reads_each_sites_dim_label(vector, theta):
    natural_values = vector.to_natural(theta)
    at_sites = vector.at_sites(natural_values)
    conifer = list(vector.dim_index("pft")).index("conifer")
    np.testing.assert_array_equal(at_sites["allocation"][:, 1], natural_values["allocation"][:, conifer])
    np.testing.assert_array_equal(at_sites["capacity"][:, 2], natural_values["capacity"])


def test_theta_must_end_in_the_dimension(vector):
    with pytest.raises(ValueError, match="must end in the vector's dimension"):
        vector.to_natural(jnp.zeros((2, 3)))


def test_natural_values_are_validated(vector, theta):
    natural_values = vector.to_natural(theta)
    with pytest.raises(TypeError, match="mapping"):
        validate_natural_values([1.0], vector)
    wrong = {**natural_values, "allocation": natural_values["allocation"][..., :3]}
    with pytest.raises(ValueError, match="does not end in"):
        validate_natural_values(wrong, vector)
    ragged = {**natural_values, "capacity": natural_values["capacity"][:2]}
    with pytest.raises(ValueError, match="different leading shapes"):
        validate_natural_values(ragged, vector)
    validate_natural_values(vector.at_sites(natural_values), vector, at_sites=True)


# ── the labeled form ──────────────────────────────────────────────────────────


def test_the_dataset_follows_the_data_model(vector, theta):
    dataset = vector.dataset(theta)
    assert set(dataset.dims) == {"sample", "pft", "site"}
    assert dataset["allocation.leaf"].dims == ("sample", "pft")
    assert dataset["capacity"].dims == ("sample",)
    assert dataset["soil_carbon"].attrs == {
        "parameter": "soil_carbon",
        "units": "g m-2",
        "support": "positive",
    }
    assert "units" not in dataset["offsets.a"].attrs
    assert dataset["offsets.a"].attrs["natural_name"] == "a"
    assert dataset["site"].dtype == np.int32 and "lon" in dataset.coords


def test_the_dim_index_is_the_datasets_index(vector, theta):
    dataset = vector.dataset(theta)
    assert dataset.indexes["pft"].equals(vector.dim_index("pft"))
    assert dataset.indexes["site"].equals(vector.dim_index("site"))


def test_theta_round_trips_through_the_dataset(vector, theta):
    np.testing.assert_allclose(vector.flat(vector.dataset(theta)), theta, atol=1e-12)
    one = vector.dataset(theta[0])
    assert "sample" not in one.dims
    np.testing.assert_allclose(vector.flat(one), theta[0], atol=1e-12)


def test_the_dataset_round_trips_through_netcdf(vector, theta, tmp_path):
    path = tmp_path / "theta.nc"
    vector.dataset(theta).to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        np.testing.assert_allclose(vector.flat(reloaded.load()), theta, atol=1e-12)


def test_a_smaller_vector_reads_a_larger_ones_dataset(vector, theta):
    conifer = vector.select(dim_labels={"pft": ["conifer"]}, parameter_names=["allocation", "soil_carbon"])
    projected = conifer.flat(vector.dataset(theta))
    np.testing.assert_allclose(
        projected[:, conifer.positions(parameter_name="allocation")],
        theta[:, vector.positions(parameter_name="allocation", dim_label="conifer")],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        projected[:, conifer.positions(parameter_name="soil_carbon")],
        theta[:, vector.positions(parameter_name="soil_carbon", dim_label=27)],
        atol=1e-12,
    )


def test_flat_refuses_a_value_outside_the_support(vector, theta):
    dataset = vector.dataset(theta)
    dataset["capacity"] = -dataset["capacity"]
    with pytest.raises(ValueError, match="outside its support"):
        vector.flat(dataset)


def test_flat_refuses_a_variable_on_other_dims(vector, theta):
    dataset = vector.dataset(theta)
    dataset["capacity"] = dataset["capacity"].expand_dims(pft=dataset["pft"].values).copy()
    with pytest.raises(ValueError, match="does not define the parameter alike|do not define"):
        vector.flat(dataset)


def test_the_dataset_is_validated(vector, theta):
    with pytest.raises(TypeError, match="xarray Dataset"):
        validate_parameter_dataset({"capacity": 1.0})
    dataset = vector.dataset(theta)
    dataset["capacity"] = dataset["capacity"].where(dataset["sample"] != 0)
    with pytest.raises(ValueError, match="missing or non-finite"):
        validate_parameter_dataset(dataset)
    crossed = vector.dataset(theta).expand_dims(run=[0, 1])
    with pytest.raises(ValueError, match="batch dims"):
        validate_parameter_dataset(crossed)


def test_site_fields_are_fields(vector, theta):
    site_fields = vector.site_fields(vector.dataset(theta))
    for name, variable in site_fields.data_vars.items():
        validate_field(variable, message_name=str(name))
        assert variable.dims == ("sample", "site")
    np.testing.assert_allclose(
        site_fields["allocation.wood"].values,
        np.asarray(vector.at_sites(vector.to_natural(theta))["allocation"][..., 1]),
    )


def test_a_batch_dim_may_not_take_a_vectors_name(vector, theta):
    with pytest.raises(ValueError, match="dim or variable of the vector's labeled form"):
        vector.dataset(theta, batch_dim="pft")
    with pytest.raises(ValueError, match="cannot name a batch dim"):
        vector.dataset(theta, batch_dim="site")


# ── site labels, site covariates and selection ────────────────────────────────


def test_dim_labels_are_the_classes_present_in_declared_order():
    table = pd.DataFrame(
        {
            "site_id": [1, 27, 4711, 9000],
            "label": pd.Categorical(
                ["temperate", "boreal", "temperate", "grass"],
                categories=["grass", "temperate", "boreal"],
            ),
        }
    )
    vector = ParameterVector(
        parameters=[Parameter(name="rate", support=POSITIVE, units="1", dim="pft")],
        site_table=site_table(),
        site_labels={"pft": table},
    )
    assert list(vector.dim_index("pft")) == ["temperate", "boreal"]


def test_site_labels_must_be_strings():
    with pytest.raises(TypeError, match="not strings"):
        ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"pft": [1, 2, 1]})


def test_site_labels_must_be_one_per_site():
    with pytest.raises(ValueError, match="one per site"):
        ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"pft": ["a"]})


def test_a_site_labels_table_must_label_every_site():
    table = pd.DataFrame({"site_id": [1, 27], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="no label for site"):
        ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"pft": table})
    with pytest.raises(ValueError, match="lack the column"):
        ParameterVector(
            parameters=parameters(), site_table=site_table(), site_labels={"pft": table[["site_id"]]}
        )


def test_named_site_covariates_are_kept_and_others_dropped():
    vector = ParameterVector(
        parameters=parameters(),
        site_table=site_table(temperature=[1.0, 2.0, 3.0], cluster=[4, 5, 6]),
        site_labels={"pft": PFT},
        site_covariate_names=["temperature"],
    )
    assert list(vector.site_table.columns) == ["site_id", "lon", "lat", "temperature", "pft"]
    kept = vector.select(sites=[27, 4711])
    assert list(kept.site_table["temperature"]) == [2.0, 3.0]


def test_a_named_site_covariate_must_be_a_finite_float64_column():
    table = site_table(cluster=[4, 5, 6], gap=[1.0, np.nan, 2.0])
    with pytest.raises(KeyError, match="not a column"):
        ParameterVector(parameters=parameters(), site_table=table, site_labels={"pft": PFT},
                        site_covariate_names=["temperature"])
    with pytest.raises(TypeError, match="not float64"):
        ParameterVector(parameters=parameters(), site_table=table, site_labels={"pft": PFT},
                        site_covariate_names=["cluster"])
    with pytest.raises(ValueError, match="not finite"):
        ParameterVector(parameters=parameters(), site_table=table, site_labels={"pft": PFT},
                        site_covariate_names=["gap"])


def test_the_site_table_must_be_ascending():
    with pytest.raises(ValueError, match="ascending"):
        ParameterVector(parameters=parameters(), site_table=site_table_of(27, 1), site_labels={"pft": PFT[:2]})


def test_select_keeps_order_and_recomputes_dim_indexes(vector):
    smaller = vector.select(parameter_names=["soil_carbon", "capacity"], sites=[1, 4711])
    assert smaller.parameter_names == ("capacity", "soil_carbon")
    assert smaller.sites == (1, 4711)
    assert list(smaller.select(parameter_names=["capacity"]).dim_index("pft")) == ["deciduous"]


def test_select_refuses_unknown_labels_and_empty_selections(vector):
    with pytest.raises(KeyError, match="not a dim label"):
        vector.select(dim_labels={"pft": ["grass"]})
    with pytest.raises(KeyError, match="select sites with sites="):
        vector.select(dim_labels={"site": [1]})
    with pytest.raises(ValueError, match="keeps no site"):
        vector.select(sites=[27], dim_labels={"pft": ["deciduous"]})


def test_dim_index_is_only_for_the_vectors_dims(vector):
    with pytest.raises(KeyError, match="no dim"):
        vector.dim_index("biome")


# ── construction checks ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["lambda", "class", "Rate", "two__underscores"])
def test_a_name_must_be_usable(name):
    with pytest.raises(ValueError, match="lower_case_with_underscores and not a Python keyword"):
        Parameter(name=name, support=REAL, units=None)


def test_a_name_may_not_be_reserved():
    with pytest.raises(ValueError, match="coordinate of the labeled form"):
        Parameter(name="sample", support=REAL, units=None)


def test_a_vector_needs_a_parameter():
    with pytest.raises(ValueError, match="at least one parameter"):
        ParameterVector(parameters=[], site_table=site_table())


def test_parameter_names_are_unique():
    with pytest.raises(ValueError, match="parameter names"):
        ParameterVector(
            parameters=[Parameter(name="rate", support=REAL, units=None)] * 2, site_table=site_table()
        )


def test_names_are_distinct_across_parameters_covariates_and_site_labels():
    with pytest.raises(ValueError, match="both a parameter and a site covariate"):
        ParameterVector(
            parameters=[Parameter(name="rate", support=REAL, units=None)],
            site_table=site_table(rate=[1.0, 2.0, 3.0]),
            site_covariate_names=["rate"],
        )
    with pytest.raises(ValueError, match="reserved name"):
        ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"time": PFT})


def test_a_dim_must_be_site_or_a_site_labels_name():
    with pytest.raises(ValueError, match="neither 'site' nor a site-labels name"):
        ParameterVector(
            parameters=[Parameter(name="rate", support=REAL, units=None, dim="biome")],
            site_table=site_table(),
        )


def test_the_simplex_needs_two_natural_names():
    with pytest.raises(ValueError, match="at least two"):
        Parameter(name="shares", support=SIMPLEX, units="1", natural_names=("only",))


def test_natural_names_are_unique_and_usable():
    with pytest.raises(ValueError, match="natural_names"):
        Parameter(name="pair", support=REAL, units=None, natural_names=("a", "a"))
    with pytest.raises(ValueError, match="natural name"):
        Parameter(name="pair", support=REAL, units=None, natural_names=("a", "B"))


@pytest.mark.parametrize(
    "kind, low, high",
    [("box", None, None), ("interval", 1.0, 1.0), ("interval", 0.0, np.inf), ("positive", 0.0, None)],
)
def test_a_support_is_checked(kind, low, high):
    with pytest.raises(ValueError, match="support"):
        Support(kind, low, high)


def test_a_custom_bijector_onto_the_support_is_accepted():
    parameter = Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Softplus())
    assert parameter.unconstrained_names == ("softplus_inverse(rate)",)
    simplex = Parameter(
        name="shares", support=SIMPLEX, units="1", natural_names=("a", "b", "c"),
        bijector=tfb.IteratedSigmoidCentered(),
    )
    assert simplex.unconstrained_size == 2


def test_a_custom_bijector_not_onto_the_support_is_refused():
    with pytest.raises(ValueError, match="does not map theta onto the support"):
        Parameter(name="share", support=OPEN_UNIT_INTERVAL, units="1", bijector=tfb.Exp())
    with pytest.raises(ValueError, match="does not map theta onto the support"):
        Parameter(name="rate", support=POSITIVE, units="1", bijector=tfb.Sigmoid())


def test_the_default_bijector_is_the_supports(vector):
    assert isinstance(vector["share"].bijector, tfb.Sigmoid)
    assert isinstance(vector["allocation"].bijector, tfb.SoftmaxCentered)
    assert float(vector["q10"].bijector.forward(jnp.float64(0.0))) == pytest.approx(3.0)


# ── supports ──────────────────────────────────────────────────────────────────


def test_the_elementwise_log_jacobian_matches_finite_differences():
    theta = jnp.linspace(-4.0, 4.0, 9)
    for support in (POSITIVE, OPEN_UNIT_INTERVAL, OpenInterval(-2.0, 3.0), REAL):
        transform = support.bijector()
        step = 1e-6
        derivative = (transform.forward(theta + step) - transform.forward(theta - step)) / (2 * step)
        np.testing.assert_allclose(
            support.log_jacobian(transform, theta), np.log(np.abs(derivative)), atol=1e-6
        )


def test_the_simplex_log_jacobian_is_against_the_first_k_minus_1_coordinates():
    # For the inverse additive log-ratio, |det d(x_1..x_{k-1})/d theta| is
    # the product of all k fractions.
    theta = jnp.asarray([[0.3, -1.2, 0.5], [2.0, 0.0, -3.0]])
    x = tfb.SoftmaxCentered().forward(theta)
    np.testing.assert_allclose(
        SIMPLEX.log_jacobian(tfb.SoftmaxCentered(), theta), jnp.log(x).sum(axis=-1), atol=1e-12
    )


def test_the_simplex_contains_positive_values_summing_to_one():
    assert bool(SIMPLEX.contains(jnp.asarray([0.2, 0.3, 0.5])))
    assert not bool(SIMPLEX.contains(jnp.asarray([0.2, 0.3, 0.6])))
    assert not bool(SIMPLEX.contains(jnp.asarray([0.0, 0.5, 0.5])))


def test_probe_points_cover_the_first_dim_labels_axes():
    points = probe_points((2, 3), unconstrained_size=3)
    assert points.shape == (1 + 6 + 3 * 4 + 4, 2, 3)
    assert (np.abs(points).max(axis=(1, 2)) <= 20.0 + 1e-12).all()


# ── layout ────────────────────────────────────────────────────────────────────


def test_vectors_built_alike_share_a_layout(vector):
    twin = ParameterVector(parameters=parameters(), site_table=site_table(), site_labels={"pft": PFT})
    check_parameter_vectors_share_a_layout(vector, twin)


@pytest.mark.parametrize(
    "change, what",
    [
        (lambda ps: ps[:-1], "index"),
        (lambda ps: [*ps[:3], Parameter(name="soil_carbon", support=POSITIVE, units="g m-2",
                                        dim="site", bijector=tfb.Softplus()), *ps[4:]], "index"),
    ],
)
def test_vectors_that_differ_do_not_share_a_layout(vector, change, what):
    other = ParameterVector(parameters=change(parameters()), site_table=site_table(), site_labels={"pft": PFT})
    with pytest.raises(ValueError, match=f"differ in their {what}"):
        check_parameter_vectors_share_a_layout(vector, other)


def test_a_transform_that_differs_numerically_breaks_the_layout():
    def with_shift(shift: float) -> ParameterVector:
        capacity = Parameter(
            name="capacity", support=POSITIVE, units="nmol g-1 s-1",
            bijector=tfb.Chain([tfb.Exp(), tfb.Shift(jnp.float64(shift))]),
        )
        return ParameterVector(
            parameters=[capacity, *parameters()[1:]], site_table=site_table(), site_labels={"pft": PFT}
        )

    with pytest.raises(ValueError, match="differ in their transforms"):
        check_parameter_vectors_share_a_layout(with_shift(1.0), with_shift(2.0))


def test_other_sites_break_the_layout(vector):
    with pytest.raises(ValueError, match="differ in their index"):
        check_parameter_vectors_share_a_layout(vector, vector.select(sites=[1, 27]))

"""Tests for derived parameters.

A derived parameter is a pure array function of parameters, constants and
memberships, which its collection reads at the coords' labels before the
call: a regression on a site covariate with shared coefficients, a
non-centered hierarchy within PFTs, and a PFT-level value around a biome
mean are the worked examples. They are computed in dependency order,
traceably, and a selection reads their constants again at the kept labels.
Every check is provoked once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.parameters.derived import DerivedParameter, DerivedParameters
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.support import POSITIVE, REAL, SIMPLEX, Interval
from sipnet_calibration.parameters.vector import ParameterVector

SITES = np.array([1, 27, 4711], dtype=np.int32)
PFT_OF_SITE = ["deciduous", "conifer", "deciduous"]
PFT = ["conifer", "deciduous"]
BIOME_OF_PFT = {"conifer": "boreal", "deciduous": "temperate"}
#: A standardized covariate, as one is standardized before it becomes a constant.
ELEVATION = np.array([-1.0, 1.5, 0.2])

INTERCEPT = Parameter(name="intercept", units=None)
SLOPE = Parameter(name="slope", units=None)
MEAN = Parameter(name="mean", units=None, indexed_by=("pft",))
SPREAD = Parameter(name="spread", support=POSITIVE, units=None)
STANDARDIZED = Parameter(name="standardized", units=None, indexed_by=("site",))
BIOME_MEAN = Parameter(name="biome_mean", units=None, indexed_by=("biome",))


def site_constant(values, name="elevation") -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims="site", coords={"site": SITES}, name=name)


def pft_of_site() -> xr.DataArray:
    return xr.DataArray(PFT_OF_SITE, dims="site", coords={"site": SITES}, name="pft")


def regression(**overrides) -> DerivedParameter:
    return DerivedParameter(**{
        "name": "log_rate", "units": None, "indexed_by": ("site",),
        "parameter_names": ("intercept", "slope"),
        "constants": {"elevation": site_constant(ELEVATION)},
        "function": lambda intercept, slope, elevation: intercept + slope * elevation,
        **overrides,
    })


def non_centered(**overrides) -> DerivedParameter:
    return DerivedParameter(**{
        "name": "soil_carbon", "units": "kg m-2", "support": POSITIVE, "indexed_by": ("site",),
        "parameter_names": ("mean", "spread", "standardized"),
        "memberships": {"pft_of_site": pft_of_site()},
        "function": lambda mean, spread, standardized, pft_of_site: jnp.exp(mean[pft_of_site] + spread * standardized),
        **overrides,
    })


@pytest.fixture(scope="module")
def regressed() -> DerivedParameters:
    vector = ParameterVector(parameters=[INTERCEPT, SLOPE])
    return DerivedParameters(parameter_vector=vector, derived_parameters=[regression()], coords={"site": SITES})


@pytest.fixture(scope="module")
def hierarchy() -> DerivedParameters:
    vector = ParameterVector(parameters=[MEAN, SPREAD, STANDARDIZED], coords={"pft": PFT, "site": SITES})
    return DerivedParameters(parameter_vector=vector, derived_parameters=[non_centered()])


def natural_values(collection: DerivedParameters, n: int = 4, seed: int = 0):
    vector = collection.parameter_vector
    theta = np.random.default_rng(seed).standard_normal((n, vector.unconstrained.size))
    return vector.flat_to_values(vector.to_natural(theta))


# ── the worked examples ───────────────────────────────────────────────────────


def test_a_regression_with_shared_coefficients_is_on_the_collections_own_site_dim(regressed):
    values = natural_values(regressed)
    log_rate = regressed.values(values)["log_rate"]
    assert log_rate.shape == (4, 3)
    expected = np.asarray(values["intercept"])[:, None] + np.asarray(values["slope"])[:, None] * ELEVATION
    np.testing.assert_allclose(log_rate, expected, rtol=1e-14)
    assert list(regressed.coords) == ["site"] and regressed.block_shape("log_rate") == (3,)


def test_a_non_centered_hierarchy_reads_each_sites_pft_through_its_membership(hierarchy):
    values = natural_values(hierarchy)
    soil = np.asarray(hierarchy.values(values)["soil_carbon"])
    mean, spread, z = (np.asarray(values[n]) for n in ("mean", "spread", "standardized"))
    position = [PFT.index(p) for p in PFT_OF_SITE]
    np.testing.assert_allclose(soil, np.exp(mean[:, position] + spread[:, None] * z), rtol=1e-14)


def test_a_membership_between_two_dims_reads_a_coarser_value_at_finer_labels():
    vector = ParameterVector(parameters=[BIOME_MEAN], coords={"biome": ["boreal", "temperate"]})
    biome_of_pft = pd.Series(BIOME_OF_PFT).rename_axis("pft").to_xarray().rename("biome")
    derived = DerivedParameters(
        parameter_vector=vector,
        coords={"pft": PFT},
        derived_parameters=[DerivedParameter(
            name="pft_mean", units=None, indexed_by=("pft",), parameter_names=("biome_mean",),
            memberships={"biome_of_pft": biome_of_pft},
            function=lambda biome_mean, biome_of_pft: biome_mean[biome_of_pft] + 1.0,
        )],
    )
    values = {"biome_mean": jnp.asarray([[10.0, 20.0]])}
    np.testing.assert_allclose(derived.values(values)["pft_mean"], [[11.0, 21.0]])


def test_a_constant_is_read_at_the_coords_labels_whatever_its_own_order():
    shuffled = site_constant(ELEVATION).isel(site=[2, 0, 1])
    extra = xr.DataArray(np.append(ELEVATION, 5.0), dims="site", coords={"site": [*SITES.tolist(), 9]})
    vector = ParameterVector(parameters=[INTERCEPT, SLOPE])
    for constant in (shuffled, extra):
        collection = DerivedParameters(
            parameter_vector=vector, coords={"site": SITES},
            derived_parameters=[regression(constants={"elevation": constant})],
        )
        values = {"intercept": jnp.zeros(1), "slope": jnp.ones(1)}
        np.testing.assert_allclose(collection.values(values)["log_rate"], [ELEVATION])


def test_a_boolean_constant_stays_boolean():
    deciduous = xr.DataArray([False, True], dims="pft", coords={"pft": PFT})
    vector = ParameterVector(parameters=[MEAN], coords={"pft": PFT})
    collection = DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="masked", units=None, indexed_by=("pft",), parameter_names=("mean",),
        constants={"deciduous": deciduous},
        function=lambda mean, deciduous: jnp.where(deciduous, 0.0, mean),
    )])
    np.testing.assert_allclose(collection.values({"mean": jnp.asarray([[3.0, 4.0]])})["masked"], [[3.0, 0.0]])


# ── evaluation ────────────────────────────────────────────────────────────────


def test_derived_parameters_are_computed_in_dependency_order():
    vector = ParameterVector(parameters=[INTERCEPT, SLOPE])
    rate = DerivedParameter(name="rate", units="yr-1", support=POSITIVE, indexed_by=("site",),
                            parameter_names=("log_rate",), function=lambda log_rate: jnp.exp(log_rate))
    collection = DerivedParameters(parameter_vector=vector, coords={"site": SITES},
                                   derived_parameters=[rate, regression()])
    assert collection.names == ("log_rate", "rate")
    values = collection.values(natural_values(collection), names=["rate"])
    assert list(values) == ["log_rate", "rate"]
    np.testing.assert_allclose(values["rate"], np.exp(values["log_rate"]))
    assert collection.parameters_behind(["rate"]) == ("intercept", "slope")


def test_values_are_traceable(hierarchy):
    vector = hierarchy.parameter_vector

    def total(theta):
        return hierarchy.values(vector.flat_to_values(vector.to_natural(theta)))["soil_carbon"].sum()

    theta = jnp.zeros(vector.unconstrained.size)
    np.testing.assert_allclose(jax.jit(total)(theta), total(theta))
    assert jax.grad(total)(theta).shape == theta.shape


def test_one_draw_has_no_batch(hierarchy):
    values = {name: value[0] for name, value in natural_values(hierarchy).items()}
    assert hierarchy.values(values)["soil_carbon"].shape == (3,)


def test_the_labeled_form_carries_the_derived_parameters(hierarchy):
    dataset = hierarchy.values_to_dataset(hierarchy.values(natural_values(hierarchy)), batch_dims=("sample",))
    assert dataset["soil_carbon"].dims == ("sample", "site")
    assert dataset["soil_carbon"].attrs == {"support": "(0, inf)", "units": "kg m-2"}
    assert dataset["site"].values.tolist() == SITES.tolist()
    assert hierarchy.values_to_dataset({}).data_vars.keys() == set()


def test_values_need_the_parameters_and_known_names(hierarchy):
    values = natural_values(hierarchy)
    with pytest.raises(KeyError, match="which the values lack"):
        hierarchy.values({n: v for n, v in values.items() if n != "spread"})
    with pytest.raises(KeyError, match="no derived parameter \\['nothing'\\]"):
        hierarchy.values(values, names=["nothing"])
    with pytest.raises(ValueError, match="different batch shapes"):
        hierarchy.values({**values, "spread": values["spread"][:2]})


# ── selection ─────────────────────────────────────────────────────────────────


def test_select_reads_the_constants_again_at_the_kept_labels(regressed):
    selected = regressed.select(site=[4711, 1])
    assert selected.coords["site"].tolist() == [1, 4711]
    values = {"intercept": jnp.zeros(1), "slope": jnp.ones(1)}
    np.testing.assert_allclose(selected.values(values)["log_rate"], [ELEVATION[[0, 2]]])


def test_select_keeps_the_derived_parameters_whose_inputs_are_kept(hierarchy):
    selected = hierarchy.select(site=[27])
    assert selected.names == ("soil_carbon",) and selected.parameter_vector.coords["site"].tolist() == [27]
    values = natural_values(selected)
    position = PFT.index("conifer")
    np.testing.assert_allclose(
        selected.values(values)["soil_carbon"][:, 0],
        np.exp(values["mean"][:, position] + values["spread"] * values["standardized"][:, 0]),
    )
    assert hierarchy.select(parameter=["mean", "spread"]).names == ()


def test_a_kept_membership_pointing_to_a_removed_label_is_refused(hierarchy):
    with pytest.raises(KeyError, match="not labels of 'pft'"):
        hierarchy.select(pft=["conifer"])


def test_a_selector_names_a_dim_somebody_has(regressed):
    with pytest.raises(KeyError, match="no dim 'pft'"):
        regressed.select(pft=["conifer"])
    with pytest.raises(TypeError, match="are integers"):
        regressed.select(site=["27"])


# ── the declared support ──────────────────────────────────────────────────────


def test_a_value_outside_its_declared_support_is_refused():
    with pytest.raises(ValueError, match="outside its declared support"):
        DerivedParameters(
            parameter_vector=ParameterVector(parameters=[INTERCEPT, SLOPE]), coords={"site": SITES},
            derived_parameters=[regression(support=POSITIVE)],
        )


def test_overflow_at_an_unbounded_end_passes_at_the_outer_probes():
    vector = ParameterVector(parameters=[Parameter(name="x", units=None)])
    huge = DerivedParameter(name="huge", units=None, support=POSITIVE, parameter_names=("x",),
                            function=lambda x: jnp.exp(40.0 * x))
    DerivedParameters(parameter_vector=vector, derived_parameters=[huge])


def test_the_boundary_passes_only_at_the_outer_probes():
    vector = ParameterVector(parameters=[Parameter(name="x", units=None)])
    flat = DerivedParameter(name="flat", units=None, support=POSITIVE, parameter_names=("x",),
                            function=lambda x: jnp.where(jnp.abs(x) > 5.0, 0.0, 1.0))
    DerivedParameters(parameter_vector=vector, derived_parameters=[flat])
    zero = DerivedParameter(name="zero", units=None, support=POSITIVE, parameter_names=("x",),
                            function=lambda x: 0.0 * x)
    with pytest.raises(ValueError, match="outside its declared support"):
        DerivedParameters(parameter_vector=vector, derived_parameters=[zero])


def test_a_nan_is_outside_every_support():
    vector = ParameterVector(parameters=[Parameter(name="x", units=None)])
    nan = DerivedParameter(name="nan", units=None, support=Interval(), parameter_names=("x",),
                           function=lambda x: jnp.where(jnp.abs(x) > 15.0, jnp.nan, x))
    with pytest.raises(ValueError, match="outside its declared support"):
        DerivedParameters(parameter_vector=vector, derived_parameters=[nan])


def test_a_simplex_valued_derived_parameter_is_checked_per_vector():
    vector = ParameterVector(parameters=[Parameter(name="x", units=None, shape=(2,))])
    softmax = DerivedParameter(name="shares", units="1", support=SIMPLEX, shape=(3,), parameter_names=("x",),
                               function=lambda x: jax.nn.softmax(jnp.concatenate([x, jnp.zeros(1)])))
    DerivedParameters(parameter_vector=vector, derived_parameters=[softmax])


# ── refusals ──────────────────────────────────────────────────────────────────


def test_a_derived_parameter_must_compute_its_block_shape():
    with pytest.raises(ValueError, match="computes a value of shape"):
        DerivedParameters(
            parameter_vector=ParameterVector(parameters=[INTERCEPT, SLOPE]), coords={"site": SITES},
            derived_parameters=[regression(function=lambda intercept, slope, elevation: intercept + slope)],
        )


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"parameter_names": ()}, ValueError, "computed from nothing"),
        ({"function": 3.0}, TypeError, "not callable"),
        ({"constants": {"intercept": site_constant(ELEVATION)}}, ValueError, "as both a parameter and a constant"),
        ({"constants": {"elevation": site_constant(ELEVATION).astype(np.int64)}}, TypeError, "neither float64 nor bool"),
        ({"constants": {"elevation": ELEVATION}}, TypeError, "give a DataArray"),
        ({"memberships": {"pft_of_site": pft_of_site().rename(None)}}, ValueError, "named for the dim"),
        ({"support": "positive"}, TypeError, "give a Support or None"),
        ({"name": "log rate"}, ValueError, "not a Python identifier"),
    ],
)
def test_a_malformed_derived_parameter_is_refused(arguments, error, match):
    with pytest.raises(error, match=match):
        regression(**arguments)


def test_a_constant_or_membership_covers_the_labels():
    vector = ParameterVector(parameters=[INTERCEPT, SLOPE])
    with pytest.raises(KeyError, match="no value at 'site' label"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES},
                          derived_parameters=[regression(constants={"elevation": site_constant(ELEVATION).isel(site=[0, 1])})])
    with pytest.raises(ValueError, match="which is not a dim of the coords"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES},
                          derived_parameters=[regression(constants={"elevation": site_constant(ELEVATION).rename(site="point")})])
    hierarchy_vector = ParameterVector(parameters=[MEAN, SPREAD, STANDARDIZED], coords={"pft": PFT, "site": SITES})
    wrong = xr.DataArray(["deciduous", "grass", "deciduous"], dims="site", coords={"site": SITES}, name="pft")
    with pytest.raises(KeyError, match="not labels of 'pft'"):
        DerivedParameters(parameter_vector=hierarchy_vector, derived_parameters=[non_centered(memberships={"pft_of_site": wrong})])
    itself = pft_of_site().rename("site")
    with pytest.raises(ValueError, match="to itself"):
        DerivedParameters(parameter_vector=hierarchy_vector, derived_parameters=[non_centered(memberships={"pft_of_site": itself})])
    unknown = pft_of_site().rename("biome")
    with pytest.raises(KeyError, match="'biome' is not a dim of the coords"):
        DerivedParameters(parameter_vector=hierarchy_vector, derived_parameters=[non_centered(memberships={"pft_of_site": unknown})])


def test_the_collection_is_refused_where_names_or_dims_do_not_fit():
    vector = ParameterVector(parameters=[INTERCEPT, SLOPE])
    with pytest.raises(KeyError, match="neither parameters nor derived parameters"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES},
                          derived_parameters=[regression(parameter_names=("intercept", "nothing"))])
    with pytest.raises(ValueError, match="named like a parameter"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES}, derived_parameters=[regression(name="slope")])
    with pytest.raises(ValueError, match="named like a parameter, a dim"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES}, derived_parameters=[regression(name="site")])
    with pytest.raises(ValueError, match="neither the vector nor"):
        DerivedParameters(parameter_vector=vector, derived_parameters=[regression()])
    with pytest.raises(ValueError, match="which no derived parameter is indexed by"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES, "pft": PFT}, derived_parameters=[regression()])
    with pytest.raises(ValueError, match="which the vector already has"):
        DerivedParameters(parameter_vector=ParameterVector(parameters=[STANDARDIZED], coords={"site": SITES}),
                          coords={"site": SITES}, derived_parameters=[])
    with pytest.raises(ValueError, match="more than once"):
        DerivedParameters(parameter_vector=vector, coords={"site": SITES}, derived_parameters=[regression(), regression()])
    with pytest.raises(TypeError, match="must be DerivedParameters"):
        DerivedParameters(parameter_vector=vector, derived_parameters=[INTERCEPT])


def test_a_cycle_is_refused_by_name():
    vector = ParameterVector(parameters=[INTERCEPT])
    first = DerivedParameter(name="a", units=None, parameter_names=("intercept", "b"), function=lambda intercept, b: b)
    second = DerivedParameter(name="b", units=None, parameter_names=("a",), function=lambda a: a)
    with pytest.raises(ValueError, match="cycle, a -> b -> a"):
        DerivedParameters(parameter_vector=vector, derived_parameters=[first, second])


def test_the_collection_holds_its_derived_parameters(hierarchy):
    assert "soil_carbon" in hierarchy and "mean" not in hierarchy and len(hierarchy) == 1
    assert list(hierarchy) == ["soil_carbon"] and hierarchy["soil_carbon"].units == "kg m-2"
    assert repr(hierarchy) == "DerivedParameters(names=['soil_carbon'], over=['mean', 'spread', 'standardized'])"
    with pytest.raises(KeyError, match="no derived parameter"):
        hierarchy["mean"]

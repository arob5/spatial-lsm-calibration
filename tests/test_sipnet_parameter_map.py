"""Tests for the SIPNET parameter map.

The map reproduces the stored reference fields (``conftest.EXAMPLE_REFERENCE``)
to 1e-12 from their natural values, and
``ComputeInitialConditions`` reproduces the initial-condition conversion.
External inputs broadcast by dim name: zip on ``site`` and the batch dim,
cross on any other. Every check is provoked once.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from pysipnet.parameters.base import ParameterDomain

from conftest import (
    EXAMPLE_REFERENCE_PFT,
    EXAMPLE_REFERENCE_SITES,
    example_reference,
    example_reference_natural_values,
    site_table_of,
)
from sipnet_calibration.calibration import example_calibration
from sipnet_calibration.fields import validate_sipnet_parameter_fields
from sipnet_calibration.initial_conditions import to_sipnet_initial_condition_fields
from sipnet_calibration.sites import site_coordinates
from sipnet_calibration.parameter_vector import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    OpenInterval,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.sipnet_parameter_map import (
    INITIAL_STATE_NAMES,
    REQUIRED_SIPNET_PARAMETER_NAMES,
    Bounds,
    ComputeInitialConditions,
    ComputePhotosynthesisRates,
    Copy,
    Fixed,
    SIPNETParameterMap,
    ValueRequirement,
    check_sipnet_parameter_map_fits,
    check_sipnet_parameter_map_is_in_domain_at_the_corners,
    validate_external_inputs,
)

SITES = EXAMPLE_REFERENCE_SITES
PFT = EXAMPLE_REFERENCE_PFT
ALLOCATION = ("leaf", "wood", "fine_root", "coarse_root")


def example_vector() -> ParameterVector:
    return example_calibration(site_table_of(*SITES), PFT)[0]


def example_map() -> SIPNETParameterMap:
    return example_calibration(site_table_of(*SITES), PFT)[2]


@pytest.fixture(scope="module")
def reference() -> xr.Dataset:
    return example_reference()


@pytest.fixture(scope="module")
def vector() -> ParameterVector:
    return example_vector()


@pytest.fixture(scope="module")
def sipnet_map() -> SIPNETParameterMap:
    return example_map()


@pytest.fixture(scope="module")
def theta(vector, reference):
    return vector.to_unconstrained(example_reference_natural_values(reference, vector))


# ── equivalence ───────────────────────────────────────────────────────────────


def test_the_map_reproduces_the_stored_reference_fields(vector, sipnet_map, theta, reference):
    fields = sipnet_map.sipnet_parameter_fields(vector, theta)
    sipnet_names = [name for name in reference.data_vars if not name.startswith("natural:")]
    assert set(fields.data_vars) == set(sipnet_names)
    for name in sipnet_names:
        np.testing.assert_allclose(
            fields[name].broadcast_like(reference[name]).transpose(*reference[name].dims),
            reference[name], rtol=1e-12, err_msg=name,
        )


def test_compute_initial_conditions_reproduces_the_conversion():
    members = [0, 3, 7]
    rng = np.random.default_rng(0)
    state = xr.Dataset(
        {
            "initial_soil_organic_carbon": (("initial_condition_member", "site"), rng.uniform(5, 40, (3, 3)), {"units": "kg m-2"}),
            "initial_wood_carbon": (("initial_condition_member", "site"), rng.uniform(1, 20, (3, 3)), {"units": "kg m-2"}),
            "initial_leaf_carbon": (("initial_condition_member", "site"), rng.uniform(0.1, 0.5, (3, 3)), {"units": "kg m-2"}),
            "initial_soil_moisture_saturation": (("initial_condition_member", "site"), rng.uniform(5, 95, (3, 3)), {"units": "percent"}),
        },
        coords=site_coordinates(SITES, site_table_of(*SITES)) | {"initial_condition_member": members},
    )
    vector = ParameterVector(
        parameters=[Parameter(name="offset", support=REAL, units=None)],
        site_table=site_table_of(*SITES),
        site_labels={"pft": PFT},
    )
    deciduous = {"deciduous": True, "conifer": False}
    sipnet_map = SIPNETParameterMap(
        rules=[ComputeInitialConditions(deciduous=deciduous)],
        fixed=[
            Fixed(sipnet_parameter_name="fine_root_fraction", value=0.2, provenance="t"),
            Fixed(sipnet_parameter_name="coarse_root_fraction", value=0.25, provenance="t"),
            Fixed(sipnet_parameter_name="leaf_carbon_per_area", value=45.0, provenance="t"),
        ],
    )
    check_sipnet_parameter_map_fits(sipnet_map, vector, state)
    fields = sipnet_map.sipnet_parameter_fields(vector, jnp.zeros((2, 1)), external_inputs=state)
    expected = to_sipnet_initial_condition_fields(
        state,
        leaf_carbon_per_area=45.0,
        fine_root_fraction=0.2,
        coarse_root_fraction=0.25,
        deciduous=xr.DataArray([deciduous[p] for p in PFT], dims="site", coords={"site": state["site"]}),
    )
    for name in ComputeInitialConditions.sipnet_parameter_names_written:
        assert fields[name].dims == ("initial_condition_member", "site")
        np.testing.assert_allclose(fields[name], expected[name].transpose(*fields[name].dims), rtol=1e-12)


# ── the fields ────────────────────────────────────────────────────────────────


def test_the_fields_are_sipnet_parameter_fields(vector, sipnet_map, theta):
    fields = sipnet_map.sipnet_parameter_fields(vector, theta)
    validate_sipnet_parameter_fields(fields)
    assert fields["soil_carbon"].dims == ("sample", "site")
    assert fields["vapor_pressure_deficit_exponent"].dims == ("site",)
    assert fields["soil_carbon"].attrs["set_by"] == "rule Copy"
    assert fields["leaf_carbon_fraction"].attrs["set_by"] == "fixed"
    assert list(fields.data_vars) == list(sipnet_map.sipnet_parameter_names_written)


def test_one_theta_gives_fields_without_a_batch_dim(vector, sipnet_map, theta):
    fields = sipnet_map.sipnet_parameter_fields(vector, theta[0])
    assert "sample" not in fields.dims


def crossed_inputs(members=(0, 1, 2)) -> xr.Dataset:
    values = np.linspace(10_000.0, 40_000.0, len(members) * len(SITES)).reshape(len(members), len(SITES))
    return xr.Dataset(
        {"initial_soil_carbon_input": (("initial_condition_member", "site"), values, {"units": "g m-2"})},
        coords={"initial_condition_member": list(members), "site": np.asarray(SITES, dtype=np.int32)},
    )


def crossing_map() -> SIPNETParameterMap:
    return SIPNETParameterMap(
        rules=[
            Copy(value_name="initial_soil_carbon_input", sipnet_parameter_name="soil_carbon"),
            Copy(value_name="base_soil_respiration", sipnet_parameter_name="base_soil_respiration_rate"),
            ComputePhotosynthesisRates(
                capacity_value_name="photosynthetic_capacity", respiration_share_value_name="respiration_share"
            ),
        ],
        fixed=[
            Fixed(sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=0.76, provenance="t"),
            Fixed(sipnet_parameter_name="leaf_carbon_fraction", value=0.466, provenance="t"),
        ],
    )


def test_a_crossed_input_writes_on_the_union_of_dims(vector, theta):
    sipnet_map = crossing_map()
    inputs = crossed_inputs()
    check_sipnet_parameter_map_fits(sipnet_map, vector, inputs)
    fields = sipnet_map.sipnet_parameter_fields(vector, theta, external_inputs=inputs)
    assert fields["soil_carbon"].dims == ("initial_condition_member", "site")
    assert fields["base_soil_respiration_rate"].dims == ("sample", "site")
    np.testing.assert_array_equal(fields["soil_carbon"], inputs["initial_soil_carbon_input"])
    validate_sipnet_parameter_fields(fields)


def test_a_rule_reading_theta_and_a_crossed_input_is_on_both():
    vector = example_vector()
    sipnet_map = SIPNETParameterMap(
        rules=[ComputePhotosynthesisRates(capacity_value_name="capacity_input",
                                          respiration_share_value_name="respiration_share")],
        fixed=[
            Fixed(sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=0.76, provenance="t"),
            Fixed(sipnet_parameter_name="leaf_carbon_fraction", value=0.466, provenance="t"),
        ],
    )
    inputs = xr.Dataset(
        {"capacity_input": (("initial_condition_member",), [100.0, 200.0], {"units": "nmol g-1 s-1"})},
        coords={"initial_condition_member": [0, 1]},
    )
    theta = jnp.zeros((4, vector.dimension))
    fields = sipnet_map.sipnet_parameter_fields(vector, theta, external_inputs=inputs)
    rate = fields["max_photosynthesis_rate"]
    assert rate.dims == ("sample", "initial_condition_member", "site")
    np.testing.assert_allclose(rate[:, 1] / rate[:, 0], 2.0)


def test_an_input_on_the_batch_dim_zips_with_theta_rows(vector, theta):
    inputs = xr.Dataset(
        {"initial_soil_carbon_input": (("sample",), np.arange(1.0, 9.0) * 1e4, {"units": "g m-2"})},
        coords={"sample": np.arange(8)},
    )
    fields = crossing_map().sipnet_parameter_fields(vector, theta, external_inputs=inputs)
    assert fields["soil_carbon"].dims == ("sample", "site")
    np.testing.assert_allclose(fields["soil_carbon"][:, 0], np.arange(1.0, 9.0) * 1e4)


def test_an_input_on_the_batch_dim_must_be_labeled_by_theta_rows(vector, theta):
    inputs = xr.Dataset(
        {"initial_soil_carbon_input": (("sample",), [1e4, 2e4], {"units": "g m-2"})},
        coords={"sample": [0, 1]},
    )
    with pytest.raises(ValueError, match="label them 0 to J - 1"):
        crossing_map().sipnet_parameter_fields(vector, theta, external_inputs=inputs)


# ── out of domain ─────────────────────────────────────────────────────────────


def test_out_of_domain_is_empty_in_domain_and_names_each_value_outside(vector, sipnet_map, theta):
    fields = sipnet_map.sipnet_parameter_fields(vector, theta).copy(deep=True)
    assert sipnet_map.out_of_domain(fields).empty
    fields["soil_carbon"].values[2, 1] = -1.0
    outside = sipnet_map.out_of_domain(fields)
    assert len(outside) == 1
    row = outside.iloc[0]
    assert (row["sample"], row["site"], row["sipnet_parameter"], row["value"]) == (2, 27, "soil_carbon", -1.0)


def test_the_corner_check_catches_a_map_outside_the_domain():
    vector = ParameterVector(
        parameters=[Parameter(name="offset", support=REAL, units="yr-1")],
        site_table=site_table_of(1),
    )
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="offset", sipnet_parameter_name="base_soil_respiration_rate")])
    with pytest.raises(ValueError, match="outside pySIPNET's domains"):
        check_sipnet_parameter_map_is_in_domain_at_the_corners(sipnet_map, vector)
    check_sipnet_parameter_map_is_in_domain_at_the_corners(example_map(), example_vector())


# ── describe and the constants ────────────────────────────────────────────────


def test_describe_names_each_writer(sipnet_map):
    table = sipnet_map.describe()
    assert table.loc["max_photosynthesis_rate", "set_by"] == "rule ComputePhotosynthesisRates"
    assert table.loc["max_photosynthesis_rate", "values_read"] == "photosynthetic_capacity, respiration_share"
    assert table.loc["leaf_carbon_fraction", "set_by"] == "fixed"


def test_unset_parameters_are_the_required_ones_not_written(sipnet_map):
    unset = set(sipnet_map.unset_sipnet_parameter_names)
    assert "soil_carbon" not in unset
    assert unset == set(REQUIRED_SIPNET_PARAMETER_NAMES) - set(sipnet_map.sipnet_parameter_names_written)


@pytest.mark.parametrize("domain", list(ParameterDomain))
def test_every_sipnet_domain_round_trips_through_bounds(domain):
    bounds = Bounds.from_sipnet_domain(domain)
    values = np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(bounds.contains(values)), domain.contains(values))


def test_bounds_contain_a_support_that_lies_within():
    assert Bounds(0.0, 1.0, True, True).contains_support(SIMPLEX)
    assert Bounds(0.0, np.inf).contains_support(POSITIVE)
    assert not Bounds(0.0, np.inf).contains_support(REAL)
    assert Bounds(1.0, 5.0).contains_support(OpenInterval(1.0, 5.0))
    assert not Bounds(1.0, 4.0).contains_support(OpenInterval(1.0, 5.0))


# ── checks ────────────────────────────────────────────────────────────────────


def test_a_sipnet_parameter_has_one_writer():
    with pytest.raises(ValueError, match="written by both"):
        SIPNETParameterMap(
            rules=[Copy(value_name="rate", sipnet_parameter_name="base_soil_respiration_rate")],
            fixed=[Fixed(sipnet_parameter_name="base_soil_respiration_rate", value=0.01, provenance="t")],
        )


def test_a_sipnet_parameter_read_is_set_earlier():
    with pytest.raises(ValueError, match="neither fixed nor written by an earlier rule"):
        SIPNETParameterMap(rules=[ComputePhotosynthesisRates(capacity_value_name="a", respiration_share_value_name="b")])


def test_names_are_pysipnets_flat_names():
    with pytest.raises(KeyError, match="not a pySIPNET parameter"):
        SIPNETParameterMap(rules=[Copy(value_name="rate", sipnet_parameter_name="not_a_parameter")])
    with pytest.raises(ValueError, match="an alias"):
        SIPNETParameterMap(fixed=[Fixed(sipnet_parameter_name="aMax", value=1.0, provenance="t")], rules=[])


def test_a_fixed_value_is_in_its_domain_and_well_formed():
    with pytest.raises(ValueError, match="outside pySIPNET's domain"):
        SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="soil_carbon", value=-1.0, provenance="t")])
    with pytest.raises(TypeError, match="mapping with dim="):
        Fixed(sipnet_parameter_name="soil_carbon", value={"a": 1.0}, provenance="t")
    with pytest.raises(TypeError, match="not a number"):
        Fixed(sipnet_parameter_name="soil_carbon", value=True, provenance="t")
    with pytest.raises(ValueError, match="no provenance"):
        Fixed(sipnet_parameter_name="soil_carbon", value=1.0, provenance="")


def test_the_fit_check_needs_each_value_once(vector):
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="missing", sipnet_parameter_name="soil_carbon")])
    with pytest.raises(KeyError, match="neither a parameter nor an external input"):
        check_sipnet_parameter_map_fits(sipnet_map, vector)
    both = SIPNETParameterMap(rules=[Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon")])
    inputs = crossed_inputs().rename({"initial_soil_carbon_input": "initial_soil_carbon"})
    with pytest.raises(ValueError, match="named like parameters"):
        check_sipnet_parameter_map_fits(both, vector, inputs)
    with pytest.raises(ValueError, match="both a parameter and an external input"):
        both.sipnet_parameter_fields(vector, jnp.zeros((1, vector.dimension)), external_inputs=inputs)


@pytest.mark.parametrize(
    "parameter, message",
    [
        (Parameter(name="value", support=POSITIVE, units="kg m-2"), "requires 'g m-2'"),
        (Parameter(name="value", support=REAL, units="g m-2"), "reaches outside"),
        (Parameter(name="value", support=POSITIVE, units="g m-2", natural_names=("a", "b")), "natural numbers"),
    ],
)
def test_a_parameter_must_meet_the_requirement(parameter, message):
    vector = ParameterVector(parameters=[parameter], site_table=site_table_of(*SITES))
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="value", sipnet_parameter_name="soil_carbon")])
    with pytest.raises(ValueError, match=message):
        check_sipnet_parameter_map_fits(sipnet_map, vector)


def test_units_are_compared_for_every_rule(vector):
    wrong = ParameterVector(
        parameters=[
            Parameter(name="photosynthetic_capacity", support=POSITIVE, units="umol g-1 s-1"),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="percent", dim="pft", natural_names=ALLOCATION),
        ],
        site_table=site_table_of(*SITES), site_labels={"pft": PFT},
    )
    photosynthesis = SIPNETParameterMap(rules=example_map().rules[:1], fixed=example_map().fixed)
    with pytest.raises(ValueError, match="requires 'nmol g-1 s-1'"):
        check_sipnet_parameter_map_fits(photosynthesis, wrong)
    simplex = SIPNETParameterMap(rules=example_map().rules[1:2])
    with pytest.raises(ValueError, match="requires '1'"):
        check_sipnet_parameter_map_fits(simplex, wrong)
    inputs = crossed_inputs()
    inputs["initial_soil_carbon_input"].attrs["units"] = "kg m-2"
    with pytest.raises(ValueError, match="convert it first"):
        check_sipnet_parameter_map_fits(crossing_map(), vector, inputs)
    state = xr.Dataset({name: (("site",), [1.0, 1.0, 1.0], {"units": "g m-2"}) for name in INITIAL_STATE_NAMES},
                       coords={"site": np.asarray(SITES, dtype=np.int32)})
    initial = SIPNETParameterMap(
        rules=[ComputeInitialConditions(deciduous={"deciduous": True, "conifer": False})],
        fixed=[Fixed(sipnet_parameter_name=n, value=0.2, provenance="t")
               for n in ("fine_root_fraction", "coarse_root_fraction", "leaf_carbon_per_area")],
    )
    with pytest.raises(ValueError, match="requires 'kg m-2'"):
        check_sipnet_parameter_map_fits(initial, vector, state)


def test_an_external_input_outside_its_bounds_is_refused_up_front(vector):
    inputs = crossed_inputs()
    inputs["initial_soil_carbon_input"][1, 2] = -5.0
    with pytest.raises(ValueError, match="outside Bounds.*initial_condition_member"):
        check_sipnet_parameter_map_fits(crossing_map(), vector, inputs)


def test_per_dim_label_values_cover_the_dim_labels(vector):
    short = SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="leaf_carbon_fraction", dim="pft",
                                                      value={"deciduous": 0.4}, provenance="t")])
    with pytest.raises(KeyError, match="no value for pft dim label"):
        check_sipnet_parameter_map_fits(short, vector)
    rule = SIPNETParameterMap(
        rules=[ComputeInitialConditions(deciduous={"deciduous": True})],
        fixed=[Fixed(sipnet_parameter_name=n, value=0.2, provenance="t")
               for n in ("fine_root_fraction", "coarse_root_fraction", "leaf_carbon_per_area")],
    )
    state = xr.Dataset({name: (("site",), [1.0, 1.0, 1.0], {"units": "kg m-2" if "saturation" not in name else "percent"})
                        for name in INITIAL_STATE_NAMES}, coords={"site": np.asarray(SITES, dtype=np.int32)})
    with pytest.raises(KeyError, match="ComputeInitialConditions has no value for pft"):
        check_sipnet_parameter_map_fits(rule, vector, state)


def test_the_deciduous_mask_is_read_from_the_site_table():
    rule = ComputeInitialConditions(deciduous={"deciduous": True, "conifer": False})
    values = {name: jnp.ones((1, 3)) for name in INITIAL_STATE_NAMES}
    sipnet = {n: jnp.full((1, 3), 0.2) for n in rule.sipnet_parameter_names_read}
    table = site_table_of(*SITES).assign(pft=list(PFT))
    lai = rule(values, sipnet, table)["leaf_area_index"]
    np.testing.assert_allclose(lai, [[0.0, 5000.0, 0.0]])
    with pytest.raises(TypeError, match="booleans"):
        ComputeInitialConditions(deciduous={"deciduous": 1})


def test_external_inputs_are_validated():
    with pytest.raises(TypeError, match="Dataset"):
        validate_external_inputs({"a": 1.0})
    with pytest.raises(ValueError, match="no units attribute"):
        validate_external_inputs(xr.Dataset({"a": ("site", [1.0])}, coords={"site": [1]}))
    with pytest.raises(ValueError, match="refuses"):
        validate_external_inputs(xr.Dataset({"a": ("site", [1.0], {"units": "g C m-2"})}, coords={"site": [1]}))
    with pytest.raises(ValueError, match="neither 'site' nor a batch dim"):
        validate_external_inputs(xr.Dataset({"a": ("pft", [1.0], {"units": "1"})}, coords={"pft": ["x"]}))


def test_an_external_input_must_be_float64():
    with pytest.raises(TypeError, match="convert it to float64"):
        validate_external_inputs(xr.Dataset({"a": ("site", np.asarray([1.0], dtype=np.float32), {"units": "1"})},
                                            coords={"site": [1]}))


def test_an_external_input_must_cover_the_sites(vector, theta):
    inputs = crossed_inputs().isel(site=[0, 1])
    with pytest.raises(KeyError, match="no value for site"):
        crossing_map().sipnet_parameter_fields(vector, theta, external_inputs=inputs)


def test_value_requirement_defaults():
    assert ValueRequirement("1") == ValueRequirement("1", None, 1)


def test_out_of_domain_reports_a_missing_value(vector, sipnet_map, theta):
    fields = sipnet_map.sipnet_parameter_fields(vector, theta).copy(deep=True)
    fields["soil_carbon"].values[0, 0] = np.nan
    outside = sipnet_map.out_of_domain(fields)
    assert len(outside) == 1 and np.isnan(outside.iloc[0]["value"])


@pytest.mark.parametrize("order", [1, -1])
def test_every_rule_reading_a_value_holds_it_to_its_requirement(order):
    vector = ParameterVector(parameters=[Parameter(name="amount", support=POSITIVE, units="g m-2")],
                             site_table=site_table_of(*SITES))
    rules = [Copy(value_name="amount", sipnet_parameter_name="soil_carbon"),
             Copy(value_name="amount", sipnet_parameter_name="base_soil_respiration_rate")][::order]
    sipnet_map = SIPNETParameterMap(rules=rules)
    assert len(sipnet_map.values_read["amount"]) == 2
    with pytest.raises(ValueError, match="requires 'yr-1'"):
        check_sipnet_parameter_map_fits(sipnet_map, vector)


def test_an_input_no_rule_reads_crosses_nothing(vector, theta):
    inputs = crossed_inputs().assign(
        unread=(("driver_member",), [1.0, 2.0], {"units": "1"})
    ).assign_coords(driver_member=[0, 1])
    sipnet_map = crossing_map()
    assert sipnet_map.crossed_dims(inputs) == ("initial_condition_member",)
    fields = sipnet_map.sipnet_parameter_fields(vector, theta, external_inputs=inputs)
    assert "driver_member" not in fields.dims

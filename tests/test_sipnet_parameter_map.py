"""Tests for the SIPNET parameter map.

The map is a function of labeled values at the sites: values read by name,
broadcast by dim name (zip on one name, cross on two), constants and fixed
values read at each site's labels. The initial-condition rules reproduce
the initial-condition conversion. The rule contract makes the dependencies
exact; domains are checked on values, in one report. Every check is
provoked once. The equivalence with the code before the parameter layer is
in ``test_equivalence.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.parameters.base import ParameterDomain

from conftest import site_table_of
from sipnet_calibration.fields import validate_sipnet_parameter_fields
from sipnet_calibration.initial_conditions import to_sipnet_initial_condition_fields
from sipnet_calibration.parameters import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    DerivedParameter,
    DerivedParameters,
    Parameter,
    ParameterVector,
)
from sipnet_calibration.sipnet_parameter_map import (
    INITIAL_STATE_NAMES,
    REQUIRED_SIPNET_PARAMETER_NAMES,
    Compute,
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
    ValueRequirement,
    check_sipnet_parameter_map_fits,
    initial_condition_rules,
    photosynthesis_rules,
    support_from_sipnet_domain,
    validate_external_inputs,
)
from sipnet_calibration.site_dims import SiteDims

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
ALLOCATION = ("leaf", "wood", "fine_root", "coarse_root")


def by_pft(values: dict) -> xr.DataArray:
    return pd.Series(values).rename_axis("pft").to_xarray()


@pytest.fixture(scope="module")
def site_dims() -> SiteDims:
    return SiteDims(site_table=site_table_of(*SITES), site_labels={"pft": PFT})


@pytest.fixture(scope="module")
def vector(site_dims) -> ParameterVector:
    return ParameterVector(
        parameters=[
            Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
            Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
            Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,), indexed_by=("pft",),
                      element_labels={"allocation_part": ALLOCATION}),
            Parameter(name="base_soil_respiration", support=POSITIVE, units="yr-1", indexed_by=("pft",)),
            Parameter(name="initial_soil_carbon", support=POSITIVE, units="g m-2", indexed_by=("site",)),
        ],
        coords={"pft": site_dims.coords["pft"], "site": site_dims.coords["site"]},
    )


def photosynthesis_fixed() -> list[Fixed]:
    return [
        Fixed(sipnet_parameter_name="daily_mean_photosynthesis_fraction", value=0.76, provenance="t"),
        Fixed(sipnet_parameter_name="leaf_carbon_fraction", value=by_pft({"conifer": 0.506, "deciduous": 0.466}),
              provenance="t"),
    ]


@pytest.fixture(scope="module")
def sipnet_map() -> SIPNETParameterMap:
    return SIPNETParameterMap(
        rules=[
            *photosynthesis_rules(capacity_value_name="photosynthetic_capacity",
                                  respiration_share_value_name="respiration_share"),
            CopySimplex(value_name="allocation", sipnet_parameter_names=(
                "leaf_allocation", "wood_allocation", "fine_root_allocation")),
            Copy(value_name="base_soil_respiration", sipnet_parameter_name="base_soil_respiration_rate"),
            Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon"),
        ],
        fixed=[*photosynthesis_fixed(),
               Fixed(sipnet_parameter_name="vapor_pressure_deficit_exponent", value=2.0, provenance="t")],
    )


@pytest.fixture(scope="module")
def values(vector) -> xr.Dataset:
    theta = jax.random.normal(jax.random.key(0), (8, vector.unconstrained.size))
    return vector.flat_to_dataset(vector.to_natural(theta), batch_dims=("sample",))


# ── the fields ────────────────────────────────────────────────────────────────


def test_the_fields_are_sipnet_parameter_fields(sipnet_map, values, site_dims, vector):
    check_sipnet_parameter_map_fits(sipnet_map, {p.name: p for p in vector.parameters}, site_dims=site_dims)
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    validate_sipnet_parameter_fields(fields)
    assert fields["soil_carbon"].dims == ("sample", "site")
    assert fields["vapor_pressure_deficit_exponent"].dims == ("site",)
    assert fields["soil_carbon"].attrs["set_by"] == "rule Copy"
    assert fields["max_photosynthesis_rate"].attrs["set_by"] == "rule Compute"
    assert fields["leaf_carbon_fraction"].attrs["set_by"] == "fixed"
    assert list(fields.data_vars) == list(sipnet_map.sipnet_parameter_names_written)


def test_the_rules_compute_what_they_say(sipnet_map, values, site_dims):
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    at_sites = site_dims.at_sites(values)
    capacity, share = at_sites["photosynthetic_capacity"].values, at_sites["respiration_share"].values
    leaf_carbon = np.asarray([0.466, 0.506, 0.466])
    np.testing.assert_allclose(fields["max_photosynthesis_rate"], capacity * leaf_carbon * (1 - share) / 0.76, rtol=1e-14)
    np.testing.assert_allclose(fields["foliar_respiration_fraction"], share * 0.76 / (1 - share), rtol=1e-14)
    np.testing.assert_allclose(fields["wood_allocation"], at_sites["allocation"][..., 1])
    np.testing.assert_allclose(fields["leaf_carbon_fraction"], leaf_carbon)


def test_values_without_a_batch_dim_give_fields_on_site(sipnet_map, values, site_dims):
    fields = sipnet_map.sipnet_parameter_fields(values.isel(sample=0, drop=True), site_dims=site_dims)
    assert fields["soil_carbon"].dims == ("site",) and "sample" not in fields.dims


def test_variables_no_rule_reads_are_ignored(sipnet_map, values, site_dims):
    extra = values.assign(hyperparameter=(("sample", "group"), np.ones((8, 2))))
    fields = sipnet_map.sipnet_parameter_fields(extra, site_dims=site_dims)
    assert "group" not in fields.dims


def crossed_inputs(members=(0, 1, 2)) -> xr.Dataset:
    soil = np.linspace(10_000.0, 40_000.0, len(members) * len(SITES)).reshape(len(members), len(SITES))
    return xr.Dataset(
        {"initial_soil_carbon_input": (("initial_condition_member", "site"), soil, {"units": "g m-2"})},
        coords={"initial_condition_member": list(members), "site": np.asarray(SITES, dtype=np.int32)},
    )


def crossing_map() -> SIPNETParameterMap:
    return SIPNETParameterMap(
        rules=[
            Copy(value_name="initial_soil_carbon_input", sipnet_parameter_name="soil_carbon"),
            Copy(value_name="base_soil_respiration", sipnet_parameter_name="base_soil_respiration_rate"),
            *photosynthesis_rules(capacity_value_name="photosynthetic_capacity",
                                  respiration_share_value_name="respiration_share"),
        ],
        fixed=photosynthesis_fixed(),
    )


def test_a_crossed_input_writes_on_the_union_of_dims(vector, values, site_dims):
    inputs = crossed_inputs()
    check_sipnet_parameter_map_fits(crossing_map(), {p.name: p for p in vector.parameters}, inputs, site_dims)
    fields = crossing_map().sipnet_parameter_fields(xr.merge([values, inputs]), site_dims=site_dims)
    assert fields["soil_carbon"].dims == ("initial_condition_member", "site")
    assert fields["base_soil_respiration_rate"].dims == ("sample", "site")
    np.testing.assert_array_equal(fields["soil_carbon"], inputs["initial_soil_carbon_input"])
    validate_sipnet_parameter_fields(fields)


def test_a_rule_reading_theta_and_a_crossed_input_is_on_both(values, site_dims):
    sipnet_map = SIPNETParameterMap(
        rules=photosynthesis_rules(capacity_value_name="capacity_input", respiration_share_value_name="respiration_share"),
        fixed=photosynthesis_fixed(),
    )
    inputs = xr.Dataset({"capacity_input": (("initial_condition_member",), [100.0, 200.0], {"units": "nmol g-1 s-1"})},
                        coords={"initial_condition_member": [0, 1]})
    fields = sipnet_map.sipnet_parameter_fields(xr.merge([values, inputs]), site_dims=site_dims)
    rate = fields["max_photosynthesis_rate"]
    assert rate.dims == ("sample", "initial_condition_member", "site")
    np.testing.assert_allclose(rate[:, 1] / rate[:, 0], 2.0)
    assert fields["foliar_respiration_fraction"].dims == ("sample", "site")


def test_an_input_on_the_batch_dim_zips_with_the_rows(values, site_dims):
    inputs = xr.Dataset({"initial_soil_carbon_input": (("sample",), np.arange(1.0, 9.0) * 1e4, {"units": "g m-2"})},
                        coords={"sample": np.arange(8)})
    fields = crossing_map().sipnet_parameter_fields(xr.merge([values, inputs]), site_dims=site_dims)
    assert fields["soil_carbon"].dims == ("sample", "site")
    np.testing.assert_allclose(fields["soil_carbon"][:, 0], np.arange(1.0, 9.0) * 1e4)


def test_an_input_no_rule_reads_crosses_nothing(values, site_dims):
    inputs = crossed_inputs().assign(unread=(("driver_member",), [1.0, 2.0], {"units": "1"})).assign_coords(driver_member=[0, 1])
    fields = crossing_map().sipnet_parameter_fields(xr.merge([values, inputs]), site_dims=site_dims)
    assert "driver_member" not in fields.dims


def test_values_are_read_by_name_and_must_be_there(sipnet_map, values, site_dims):
    with pytest.raises(KeyError, match="which the values lack"):
        sipnet_map.sipnet_parameter_fields(values.drop_vars("respiration_share"), site_dims=site_dims)
    with pytest.raises(TypeError, match="from an xarray Dataset"):
        sipnet_map.sipnet_parameter_fields(values["respiration_share"], site_dims=site_dims)
    with pytest.raises(KeyError, match="no value at the site label"):
        crossing_map().sipnet_parameter_fields(xr.merge([values, crossed_inputs().isel(site=[0, 1])], join="inner"),
                                               site_dims=site_dims)


def test_a_value_has_the_shape_its_rule_requires(sipnet_map, values, site_dims):
    shorter = values.isel(allocation_part=[0, 1, 2])
    with pytest.raises(ValueError, match="requires \\(4,\\)"):
        sipnet_map.sipnet_parameter_fields(shorter, site_dims=site_dims)


# ── the initial conditions and constants ──────────────────────────────────────

DECIDUOUS = {"deciduous": True, "conifer": False}
INITIAL_FIXED = [
    Fixed(sipnet_parameter_name="fine_root_fraction", value=0.2, provenance="t"),
    Fixed(sipnet_parameter_name="coarse_root_fraction", value=0.25, provenance="t"),
    Fixed(sipnet_parameter_name="leaf_carbon_per_area", value=45.0, provenance="t"),
]


def initial_state() -> xr.Dataset:
    rng = np.random.default_rng(0)
    variables = {
        name: (("initial_condition_member", "site"), rng.uniform(low, high, (3, 3)), {"units": units})
        for name, (low, high, units) in zip(INITIAL_STATE_NAMES, [(5, 40, "kg m-2"), (1, 20, "kg m-2"),
                                                                  (0.1, 0.5, "kg m-2"), (5, 95, "percent")])
    }
    return xr.Dataset(variables, coords={"initial_condition_member": [0, 3, 7], "site": np.asarray(SITES, dtype=np.int32)})


def test_the_initial_condition_rules_reproduce_the_conversion(site_dims):
    sipnet_map = SIPNETParameterMap(rules=initial_condition_rules(deciduous=DECIDUOUS), fixed=INITIAL_FIXED)
    state = initial_state()
    check_sipnet_parameter_map_fits(sipnet_map, {}, state, site_dims)
    fields = sipnet_map.sipnet_parameter_fields(state, site_dims=site_dims)
    expected = to_sipnet_initial_condition_fields(
        state.assign_coords(site_dims.at_sites(xr.Dataset()).coords),
        leaf_carbon_per_area=45.0, fine_root_fraction=0.2, coarse_root_fraction=0.25,
        deciduous=xr.DataArray([DECIDUOUS[p] for p in PFT], dims="site", coords={"site": state["site"]}),
    )
    for name in ("soil_carbon", "total_wood_carbon", "leaf_area_index", "soil_wetness_fraction"):
        assert fields[name].dims == ("initial_condition_member", "site")
        np.testing.assert_allclose(fields[name], expected[name].transpose(*fields[name].dims), rtol=1e-12)
    np.testing.assert_array_equal(fields["leaf_area_index"].values[:, [0, 2]], 0.0)


def test_the_deciduous_mask_may_be_a_boolean_data_array(site_dims):
    mask = xr.DataArray([True, False], dims="pft", coords={"pft": ["deciduous", "conifer"]})
    sipnet_map = SIPNETParameterMap(rules=initial_condition_rules(deciduous=mask), fixed=INITIAL_FIXED)
    fields = sipnet_map.sipnet_parameter_fields(initial_state(), site_dims=site_dims)
    np.testing.assert_array_equal(fields["leaf_area_index"].values[:, [0, 2]], 0.0)


def test_deciduousness_is_boolean():
    with pytest.raises(TypeError, match="booleans"):
        initial_condition_rules(deciduous={"deciduous": 1})
    with pytest.raises(TypeError, match="booleans"):
        initial_condition_rules(deciduous=xr.DataArray([1.0, np.nan], dims="pft", coords={"pft": ["a", "b"]}))
    with pytest.raises(ValueError, match="names 3 values"):
        initial_condition_rules(deciduous=DECIDUOUS, state_value_names=INITIAL_STATE_NAMES[:3])


def test_a_constant_reaches_its_rule_at_each_sites_label(site_dims):
    seen = []

    def record(soil, deciduous):
        seen.append(np.asarray(deciduous))
        return soil

    rule = Compute(sipnet_parameter_name="soil_carbon", values_read={"soil": ValueRequirement("g m-2")},
                   constants={"deciduous": by_pft(DECIDUOUS)}, function=record, provenance="t")
    values = xr.Dataset({"soil": (("sample", "site"), np.ones((2, 3)), {"units": "g m-2"})},
                        coords={"sample": [0, 1], "site": np.asarray(SITES, dtype=np.int32)})
    SIPNETParameterMap(rules=[rule]).sipnet_parameter_fields(values, site_dims=site_dims)
    assert seen[0].dtype == np.bool_ and seen[0].tolist() == [[True, False, True]] * 2


# ── dependencies and description ──────────────────────────────────────────────


def test_the_dependencies_follow_the_rule_contract(sipnet_map):
    dependencies = sipnet_map.dependencies()
    assert dependencies["max_photosynthesis_rate"] == {"photosynthetic_capacity", "respiration_share"}
    assert dependencies["foliar_respiration_fraction"] == {"respiration_share"}
    assert dependencies["leaf_carbon_fraction"] == frozenset()
    initial = SIPNETParameterMap(rules=initial_condition_rules(deciduous=DECIDUOUS), fixed=INITIAL_FIXED).dependencies()
    assert initial["leaf_area_index"] == {"initial_leaf_carbon"}
    assert initial["soil_carbon"] == {"initial_soil_organic_carbon"}


def test_a_dependency_passes_through_a_sipnet_parameter_read():
    chained = SIPNETParameterMap(rules=[
        Copy(value_name="rate", sipnet_parameter_name="base_soil_respiration_rate"),
        Compute(sipnet_parameter_name="base_wood_respiration_rate", values_read={},
                sipnet_parameter_names_read=("base_soil_respiration_rate",),
                function=lambda base_soil_respiration_rate: 2 * base_soil_respiration_rate, provenance="t"),
    ])
    assert chained.dependencies()["base_wood_respiration_rate"] == {"rate"}
    assert set(chained.sipnet_parameter_names_depending_on(["rate"])) == {"base_soil_respiration_rate", "base_wood_respiration_rate"}


def test_the_sipnet_parameters_a_calibration_varies(sipnet_map, vector):
    varied = sipnet_map.sipnet_parameter_names_depending_on(list(vector))
    assert set(varied) == set(sipnet_map.sipnet_parameter_names_written) - {
        "daily_mean_photosynthesis_fraction", "leaf_carbon_fraction", "vapor_pressure_deficit_exponent"}
    with pytest.raises(TypeError, match="must be a sequence"):
        sipnet_map.sipnet_parameter_names_depending_on("respiration_share")


def test_describe_names_each_writer_and_what_it_reads(sipnet_map):
    table = sipnet_map.describe()
    assert table.loc["max_photosynthesis_rate", "set_by"] == "rule Compute"
    assert table.loc["max_photosynthesis_rate", "values_read"] == "photosynthetic_capacity, respiration_share"
    assert table.loc["foliar_respiration_fraction", "values_read"] == "respiration_share"
    assert table.loc["foliar_respiration_fraction", "depends_on"] == "respiration_share"
    assert "sipnet.c:614, 617, 633" in table.loc["max_photosynthesis_rate", "provenance"]
    assert table.loc["leaf_carbon_fraction", "set_by"] == "fixed"
    assert table.loc["leaf_carbon_fraction", "provenance"] == "t"


def test_unset_parameters_are_the_required_ones_not_written(sipnet_map):
    unset = set(sipnet_map.unset_sipnet_parameter_names)
    assert "soil_carbon" not in unset
    assert unset == set(REQUIRED_SIPNET_PARAMETER_NAMES) - set(sipnet_map.sipnet_parameter_names_written)


# ── domains ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("domain", list(ParameterDomain))
def test_every_sipnet_domain_is_a_support(domain):
    values = np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0, np.inf, np.nan])
    np.testing.assert_array_equal(np.asarray(support_from_sipnet_domain(domain).contains(values)), domain.contains(values))


def test_out_of_domain_is_empty_in_domain_and_names_each_value_outside(sipnet_map, values, site_dims):
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    assert sipnet_map.out_of_domain(fields, values, site_dims=site_dims).empty
    broken = fields.copy(deep=True)
    broken["soil_carbon"].values[2, 1] = -1.0
    broken["wood_allocation"].values[0, 0] = np.nan
    outside = sipnet_map.out_of_domain(broken, values, site_dims=site_dims)
    assert len(outside) == 2
    row = outside.set_index("sipnet_parameter").loc["soil_carbon"]
    assert (row["sample"], row["site"], row["value"]) == (2, 27, -1.0) and pd.isna(row["value_name"])
    assert np.isnan(outside.set_index("sipnet_parameter").loc["wood_allocation", "value"])


def test_a_rule_input_outside_its_domain_is_reported(sipnet_map, values, site_dims):
    broken = values.copy(deep=True)
    broken["respiration_share"].values[3] = 1.5
    broken["allocation"].values[1, 0] = [0.5, 0.6, 0.1, -0.2]
    fields = sipnet_map.sipnet_parameter_fields(broken, site_dims=site_dims)
    outside = sipnet_map.out_of_domain(fields, broken, site_dims=site_dims)
    share = outside[outside["value_name"] == "respiration_share"]
    assert share["sample"].tolist() == [3, 3, 3] and share["value"].tolist() == [1.5] * 3
    allocation = outside[outside["value_name"] == "allocation"]
    # pft 0 is conifer, carried by site 27 alone; the vector is outside the simplex, so no one value is named.
    assert allocation[["sample", "site"]].values.tolist() == [[1, 27]] and np.isnan(allocation["value"].iloc[0])
    assert set(outside["sipnet_parameter"].dropna()) <= {"foliar_respiration_fraction", "max_photosynthesis_rate",
                                                         "fine_root_allocation", "wood_allocation", "leaf_allocation"}


# ── checks ────────────────────────────────────────────────────────────────────


def test_a_sipnet_parameter_has_one_writer():
    with pytest.raises(ValueError, match="written by both"):
        SIPNETParameterMap(
            rules=[Copy(value_name="rate", sipnet_parameter_name="base_soil_respiration_rate")],
            fixed=[Fixed(sipnet_parameter_name="base_soil_respiration_rate", value=0.01, provenance="t")],
        )


def test_a_sipnet_parameter_read_is_set_earlier():
    with pytest.raises(ValueError, match="neither fixed nor written by an earlier rule"):
        SIPNETParameterMap(rules=photosynthesis_rules(capacity_value_name="a", respiration_share_value_name="b"))


def test_names_are_pysipnets_flat_names():
    with pytest.raises(KeyError, match="not a pySIPNET parameter"):
        SIPNETParameterMap(rules=[Copy(value_name="rate", sipnet_parameter_name="not_a_parameter")])
    with pytest.raises(ValueError, match="an alias"):
        SIPNETParameterMap(fixed=[Fixed(sipnet_parameter_name="aMax", value=1.0, provenance="t")], rules=[])


def test_a_rule_is_a_rule():
    with pytest.raises(TypeError, match="is not a SIPNET rule"):
        SIPNETParameterMap(rules=[lambda values, sipnet: {}])


def test_a_fixed_value_is_in_its_domain_and_well_formed():
    with pytest.raises(ValueError, match="outside pySIPNET's domain"):
        SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="soil_carbon", value=-1.0, provenance="t")])
    with pytest.raises(ValueError, match="outside pySIPNET's domain"):
        SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="soil_carbon", provenance="t",
                                                  value=by_pft({"a": 1.0, "b": -1.0}))])
    with pytest.raises(TypeError, match="neither a number nor a numeric DataArray"):
        Fixed(sipnet_parameter_name="soil_carbon", value={"a": 1.0}, provenance="t")
    with pytest.raises(TypeError, match="neither a number"):
        Fixed(sipnet_parameter_name="soil_carbon", value=True, provenance="t")
    with pytest.raises(TypeError, match="give numbers"):
        Fixed(sipnet_parameter_name="soil_carbon", value=by_pft({"a": "x"}), provenance="t")
    with pytest.raises(ValueError, match="no provenance"):
        Fixed(sipnet_parameter_name="soil_carbon", value=1.0, provenance="")


def test_a_compute_rule_is_well_formed():
    arguments = {"sipnet_parameter_name": "soil_carbon", "values_read": {"soil": ValueRequirement("g m-2")},
                 "function": lambda soil: soil, "provenance": "t"}
    with pytest.raises(TypeError, match="not callable"):
        Compute(**{**arguments, "function": 1.0})
    with pytest.raises(ValueError, match="no provenance"):
        Compute(**{**arguments, "provenance": " "})
    with pytest.raises(ValueError, match="as two things"):
        Compute(**{**arguments, "values_read": {"soil_respiration_q10": ValueRequirement("1")},
                   "sipnet_parameter_names_read": ("soil_respiration_q10",)})
    with pytest.raises(TypeError, match="without a ValueRequirement"):
        Compute(**{**arguments, "values_read": {"soil": "g m-2"}})
    with pytest.raises(TypeError, match="give a DataArray"):
        Compute(**{**arguments, "constants": {"deciduous": DECIDUOUS}})


def test_value_requirement_defaults():
    assert ValueRequirement("1") == ValueRequirement("1", None, ())


def test_the_fit_check_needs_each_value(vector, site_dims):
    descriptions = {p.name: p for p in vector.parameters}
    missing = SIPNETParameterMap(rules=[Copy(value_name="missing", sipnet_parameter_name="soil_carbon")])
    with pytest.raises(KeyError, match="neither a parameter, a derived parameter nor an external input"):
        check_sipnet_parameter_map_fits(missing, descriptions)
    both = SIPNETParameterMap(rules=[Copy(value_name="initial_soil_carbon", sipnet_parameter_name="soil_carbon")])
    inputs = crossed_inputs().rename({"initial_soil_carbon_input": "initial_soil_carbon"})
    with pytest.raises(ValueError, match="named like parameters"):
        check_sipnet_parameter_map_fits(both, descriptions, inputs)


@pytest.mark.parametrize(
    "parameter, message",
    [
        (Parameter(name="value", support=POSITIVE, units="kg m-2"), "requires 'g m-2'"),
        (Parameter(name="value", support=POSITIVE, units="g m-2", shape=(2,)), "requires \\(\\)"),
    ],
)
def test_a_parameter_must_meet_the_requirement(parameter, message):
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="value", sipnet_parameter_name="soil_carbon")])
    with pytest.raises(ValueError, match=message):
        check_sipnet_parameter_map_fits(sipnet_map, {"value": parameter})


def test_a_derived_parameter_is_held_to_the_requirement_too():
    rate = DerivedParameter(name="respiration", units="d-1", parameter_names=("x",), function=lambda x: x)
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="respiration", sipnet_parameter_name="base_soil_respiration_rate")])
    with pytest.raises(ValueError, match="requires 'yr-1'"):
        check_sipnet_parameter_map_fits(sipnet_map, {"respiration": rate})


def test_units_are_compared_for_every_rule_and_input(vector, site_dims):
    descriptions = {p.name: p for p in vector.parameters}
    wrong = {**descriptions,
             "photosynthetic_capacity": Parameter(name="photosynthetic_capacity", support=POSITIVE, units="umol g-1 s-1")}
    photosynthesis = SIPNETParameterMap(rules=photosynthesis_rules(
        capacity_value_name="photosynthetic_capacity", respiration_share_value_name="respiration_share"),
        fixed=photosynthesis_fixed())
    with pytest.raises(ValueError, match="requires 'nmol g-1 s-1'"):
        check_sipnet_parameter_map_fits(photosynthesis, wrong)
    inputs = crossed_inputs()
    inputs["initial_soil_carbon_input"].attrs["units"] = "kg m-2"
    with pytest.raises(ValueError, match="convert it first"):
        check_sipnet_parameter_map_fits(crossing_map(), descriptions, inputs)
    initial = SIPNETParameterMap(rules=initial_condition_rules(deciduous=DECIDUOUS), fixed=INITIAL_FIXED)
    state = initial_state()
    state["initial_wood_carbon"].attrs["units"] = "g m-2"
    with pytest.raises(ValueError, match="requires 'kg m-2'"):
        check_sipnet_parameter_map_fits(initial, {}, state)


def test_an_external_input_is_one_number_per_site():
    simplex = SIPNETParameterMap(rules=[CopySimplex(value_name="shares", sipnet_parameter_names=(
        "leaf_allocation", "wood_allocation", "fine_root_allocation"))])
    inputs = xr.Dataset({"shares": ("site", [0.2, 0.2, 0.2], {"units": "1"})}, coords={"site": np.asarray(SITES)})
    with pytest.raises(ValueError, match="an external input holds one number"):
        check_sipnet_parameter_map_fits(simplex, {}, inputs)


def test_constants_and_fixed_values_cover_the_sites_labels(site_dims):
    short = SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="leaf_carbon_fraction",
                                                      value=by_pft({"deciduous": 0.4}), provenance="t")])
    with pytest.raises(KeyError, match="no value for pft label"):
        check_sipnet_parameter_map_fits(short, {}, site_dims=site_dims)
    rules = SIPNETParameterMap(rules=initial_condition_rules(deciduous={"deciduous": True}), fixed=INITIAL_FIXED)
    with pytest.raises(KeyError, match="rule Compute constant 'deciduous' has no value for pft"):
        check_sipnet_parameter_map_fits(rules, {}, initial_state(), site_dims)
    elsewhere = SIPNETParameterMap(rules=[], fixed=[Fixed(sipnet_parameter_name="leaf_carbon_fraction", provenance="t",
                                                          value=by_pft({"x": 0.4}).rename(pft="biome"))])
    with pytest.raises(ValueError, match="not a dim of the site dims"):
        check_sipnet_parameter_map_fits(elsewhere, {}, site_dims=site_dims)


@pytest.mark.parametrize("order", [1, -1])
def test_every_rule_reading_a_value_holds_it_to_its_requirement(order):
    rules = [Copy(value_name="amount", sipnet_parameter_name="soil_carbon"),
             Copy(value_name="amount", sipnet_parameter_name="base_soil_respiration_rate")][::order]
    sipnet_map = SIPNETParameterMap(rules=rules)
    assert len(sipnet_map.values_read["amount"]) == 2
    with pytest.raises(ValueError, match="requires 'yr-1'"):
        check_sipnet_parameter_map_fits(sipnet_map, {"amount": Parameter(name="amount", support=POSITIVE, units="g m-2")})


def test_external_inputs_are_validated():
    with pytest.raises(TypeError, match="Dataset"):
        validate_external_inputs({"a": 1.0})
    with pytest.raises(ValueError, match="no units attribute"):
        validate_external_inputs(xr.Dataset({"a": ("site", [1.0])}, coords={"site": [1]}))
    with pytest.raises(ValueError, match="refuses"):
        validate_external_inputs(xr.Dataset({"a": ("site", [1.0], {"units": "g C m-2"})}, coords={"site": [1]}))
    with pytest.raises(ValueError, match="neither 'site' nor a batch dim"):
        validate_external_inputs(xr.Dataset({"a": ("pft", [1.0], {"units": "1"})}, coords={"pft": ["x"]}))
    with pytest.raises(TypeError, match="convert it to float64"):
        validate_external_inputs(xr.Dataset({"a": ("site", np.asarray([1.0], dtype=np.float32), {"units": "1"})},
                                            coords={"site": [1]}))


# ── derived parameters ────────────────────────────────────────────────────────


def test_a_rule_reads_a_derived_parameter_by_name(site_dims):
    vector = ParameterVector(parameters=[Parameter(name="intercept", support=REAL, units=None),
                                         Parameter(name="slope", support=REAL, units="K-1")])
    derived = DerivedParameters(parameter_vector=vector, coords={"site": site_dims.coords["site"]}, derived_parameters=[
        DerivedParameter(name="respiration", units="yr-1", indexed_by=("site",), parameter_names=("intercept", "slope"),
                         constants={"anomaly": xr.DataArray([-1.0, 0.0, 2.0], dims="site", coords={"site": list(SITES)})},
                         function=lambda intercept, slope, anomaly: jnp.exp(intercept + slope * anomaly)),
    ])
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="respiration", sipnet_parameter_name="base_soil_respiration_rate")])
    check_sipnet_parameter_map_fits(sipnet_map, {"respiration": derived["respiration"]}, site_dims=site_dims)
    theta = jnp.asarray([[np.log(0.01), 0.5], [np.log(0.02), -0.1]])
    natural = vector.flat_to_values(vector.to_natural(theta))
    derived_values = derived.values(natural)
    values = xr.merge([vector.values_to_dataset(natural, batch_dims=("sample",)),
                       derived.values_to_dataset(derived_values, batch_dims=("sample",))])
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    np.testing.assert_allclose(fields["base_soil_respiration_rate"], derived_values["respiration"], rtol=1e-12)
    assert fields["base_soil_respiration_rate"].dims == ("sample", "site")

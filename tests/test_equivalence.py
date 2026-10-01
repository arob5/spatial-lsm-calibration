"""The parameter layer and its adapters against the code they replaced.

Two calibrations were evaluated by the code before the parameter layer at
fixed thetas, and stored (``tests/data/write_calibration_references.py``):
the example calibration, and a copy of the single-site Harvard Forest one,
whose four SIPNET translations are now ``Compute`` rules and whose initial
conditions and photosynthesis are rule factories. Built anew here, each
gives the same theta layout, prior log density, natural values and SIPNET
parameter fields, to 1e-12; a field is on fewer dims only where the rule
contract says it depends on fewer values. The older example reference
(``conftest.EXAMPLE_REFERENCE``) still maps to its stored fields from its
natural values.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from conftest import (
    EXAMPLE_REFERENCE_PFT,
    EXAMPLE_REFERENCE_SITES,
    example_reference,
    example_reference_natural_values,
    site_table_of,
)
from sipnet_calibration.calibration import example_calibration
from sipnet_calibration.parameters import (
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    Parameter,
    ParameterVector,
    Prior,
    PriorTerm,
    log_normal_from_interval,
    logit_normal_from_interval,
    softmax_normal,
)
from sipnet_calibration.sipnet_parameter_map import (
    Compute,
    Copy,
    CopySimplex,
    Fixed,
    SIPNETParameterMap,
    ValueRequirement,
    initial_condition_rules,
    photosynthesis_rules,
)
from sipnet_calibration.site_dims import SiteDims

DATA = Path(__file__).parent / "data"
TOLERANCE = 1e-12

# The single-site calibration's constants, as the reference script has them.
PHOTOSYNTHESIS_TEMPERATURE_RANGE = 24.0 - 0.041553
WOOD_RESPIRATION_Q10 = 1.80944
COARSE_ROOT_RESPIRATION_Q10 = 3.20614
DECIDUOUS_BY_SITE_LABEL = {
    "temperate.deciduous.HPDA": True,
    "boreal.coniferous": False,
    "semiarid.grassland_HPDA": False,
}
SINGLE_SITE_FIXED = {
    "daily_mean_photosynthesis_fraction": 0.860623,
    "leaf_carbon_fraction": 0.466075,
    "fine_root_fraction": 0.03,
    "coarse_root_fraction": 0.16,
    "leaf_carbon_per_area": 30.7092,
    "wood_respiration_q10": WOOD_RESPIRATION_Q10,
    "vapor_pressure_deficit_exponent": 2.0,
}


def reference(name: str) -> xr.Dataset:
    with xr.open_dataset(DATA / f"calibration_reference_{name}.nc", engine="h5netcdf") as dataset:
        return dataset.load()


def example():
    site_dims = SiteDims(site_table=site_table_of(1, 27, 4711), site_labels={"pft": ("deciduous", "conifer", "deciduous")})
    return (site_dims, *example_calibration(site_dims), None)


def single_site():
    """The single-site calibration in the parameter layer's terms."""
    site_dims = SiteDims(site_table=site_table_of(4977), site_labels={"pft": ["temperate.deciduous.HPDA"]})
    names = [
        ("photosynthetic_capacity", POSITIVE, "nmol g-1 s-1"), ("respiration_share", OPEN_UNIT_INTERVAL, "1"),
        ("optimum_photosynthesis_temperature", REAL, "degC"), ("half_saturation_light", POSITIVE, "mol m-2 d-1"),
        ("soil_water_holding_capacity", POSITIVE, "cm"), ("leaf_on_growth", POSITIVE, "g m-2"),
        ("leaf_on_growing_degree_days", POSITIVE, "K d"),
    ]
    parameters = [Parameter(name=n, support=s, units=u) for n, s, u in names]
    parameters.append(Parameter(name="allocation", support=SIMPLEX, units="1", shape=(4,),
                                element_labels={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}))
    parameters += [
        Parameter(name="wood_respiration_rate_at_10c", support=POSITIVE, units="yr-1"),
        Parameter(name="soil_respiration_flux_at_10c", support=POSITIVE, units="g m-2 yr-1"),
        Parameter(name="soil_respiration_q10", support=POSITIVE, units="1"),
        Parameter(name="initial_wood_carbon", support=POSITIVE, units="kg m-2"),
        Parameter(name="initial_soil_organic_carbon", support=POSITIVE, units="kg m-2"),
    ]
    vector = ParameterVector(parameters=parameters)

    def term(distribution):
        return PriorTerm(distribution, provenance="Reference fixture.")

    prior = Prior(vector, {
        "photosynthetic_capacity": term(log_normal_from_interval(lower=140.0, upper=450.0)),
        "respiration_share": term(logit_normal_from_interval(lower=0.04, upper=0.20)),
        "optimum_photosynthesis_temperature": term(tfp.distributions.Normal(jnp.float64(22.0), jnp.float64(2.5))),
        "half_saturation_light": term(log_normal_from_interval(lower=4.6, upper=26.3)),
        "soil_water_holding_capacity": term(log_normal_from_interval(lower=15.0, upper=150.0)),
        "leaf_on_growth": term(log_normal_from_interval(lower=50.0, upper=180.0)),
        "leaf_on_growing_degree_days": term(log_normal_from_interval(lower=500.0, upper=1100.0)),
        "allocation": term(softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]),
                                          logit_sd=jnp.array([0.25, 0.30, 0.30]))),
        "wood_respiration_rate_at_10c": term(log_normal_from_interval(lower=0.006, upper=0.04)),
        "soil_respiration_flux_at_10c": term(log_normal_from_interval(lower=200.0, upper=900.0)),
        "soil_respiration_q10": term(log_normal_from_interval(lower=1.3, upper=3.2)),
        "initial_wood_carbon": term(log_normal_from_interval(lower=5.0, upper=15.0)),
        "initial_soil_organic_carbon": term(log_normal_from_interval(lower=5.0, upper=60.0)),
    })
    copies = ("optimum_photosynthesis_temperature", "half_saturation_light", "soil_water_holding_capacity",
              "leaf_on_growth", "leaf_on_growing_degree_days", "soil_respiration_q10")
    rules = [
        *photosynthesis_rules(capacity_value_name="photosynthetic_capacity",
                              respiration_share_value_name="respiration_share"),
        *(Copy(value_name=name, sipnet_parameter_name=name) for name in copies),
        Compute(sipnet_parameter_name="min_photosynthesis_temperature",
                values_read={"optimum_photosynthesis_temperature": ValueRequirement("degC", REAL)},
                function=lambda optimum_photosynthesis_temperature:
                    optimum_photosynthesis_temperature - PHOTOSYNTHESIS_TEMPERATURE_RANGE,
                provenance="psnTMin = psnTOpt - the base set's range"),
        Compute(sipnet_parameter_name="base_wood_respiration_rate",
                values_read={"wood_respiration_rate_at_10c": ValueRequirement("yr-1", POSITIVE)},
                function=lambda wood_respiration_rate_at_10c: wood_respiration_rate_at_10c / WOOD_RESPIRATION_Q10,
                provenance="baseVegResp = r10 / vegRespQ10, sipnet.c:1066-1067"),
        Compute(sipnet_parameter_name="base_coarse_root_respiration_rate",
                values_read={"wood_respiration_rate_at_10c": ValueRequirement("yr-1", POSITIVE)},
                function=lambda wood_respiration_rate_at_10c: wood_respiration_rate_at_10c / COARSE_ROOT_RESPIRATION_Q10,
                provenance="baseCoarseRootResp = r10 / coarseRootQ10"),
        Compute(sipnet_parameter_name="base_soil_respiration_rate",
                values_read={"soil_respiration_flux_at_10c": ValueRequirement("g m-2 yr-1", POSITIVE),
                             "initial_soil_organic_carbon": ValueRequirement("kg m-2", NON_NEGATIVE),
                             "soil_respiration_q10": ValueRequirement("1", POSITIVE)},
                function=lambda soil_respiration_flux_at_10c, initial_soil_organic_carbon, soil_respiration_q10:
                    soil_respiration_flux_at_10c / (1000.0 * initial_soil_organic_carbon * soil_respiration_q10),
                provenance="F10 = k10 SOC0 referenced to 10 C: baseSoilResp = F10 / (1000 SOC0 Q10)"),
        CopySimplex(value_name="allocation",
                    sipnet_parameter_names=("leaf_allocation", "wood_allocation", "fine_root_allocation")),
        *initial_condition_rules(deciduous=DECIDUOUS_BY_SITE_LABEL),
    ]
    sipnet_map = SIPNETParameterMap(
        rules=rules,
        fixed=[Fixed(sipnet_parameter_name=n, value=v, provenance="Reference fixture.") for n, v in SINGLE_SITE_FIXED.items()],
    )
    external_inputs = xr.Dataset(
        {
            "initial_leaf_carbon": ("site", [0.19], {"units": "kg m-2"}),
            "initial_soil_moisture_saturation": ("site", [42.0], {"units": "percent"}),
        },
        coords={"site": np.array([4977], dtype=np.int32)},
    )
    return site_dims, vector, prior, sipnet_map, external_inputs


CALIBRATIONS = {"example": example, "single_site": single_site}


@pytest.fixture(scope="module", params=list(CALIBRATIONS))
def calibration(request):
    return request.param, reference(request.param), CALIBRATIONS[request.param]()


def labeled_values(vector, theta, external_inputs):
    values = vector.flat_to_dataset(vector.to_natural(theta), batch_dims=("sample",))
    return values if external_inputs is None else xr.merge([values, external_inputs])


def test_theta_has_the_same_layout(calibration):
    name, stored, (site_dims, vector, prior, sipnet_map, external_inputs) = calibration
    unconstrained = vector.unconstrained
    assert unconstrained.size == stored.sizes["entry"]
    index = unconstrained.index
    assert list(index.get_level_values("parameter")) == stored["entry_parameter"].values.tolist()
    for i, row in enumerate(index):
        parameter, labels, element = row[0], dict(zip(vector.dims, row[1:-1])), row[-1]
        dim = str(stored["entry_dim"].values[i])
        if dim:
            assert str(labels[dim]) == str(stored["entry_dim_label"].values[i]), i
        if unconstrained[parameter].shape:
            # The old unconstrained name of an element of the simplex was alr(<natural name>:<last>).
            assert stored["entry_unconstrained_name"].values[i].startswith(f"alr({element}:"), i


def test_the_prior_has_the_same_log_density(calibration):
    name, stored, (site_dims, vector, prior, sipnet_map, external_inputs) = calibration
    np.testing.assert_allclose(prior.log_prob(stored["theta"].values), stored["log_prob"].values, rtol=TOLERANCE)


def test_the_natural_values_are_the_same(calibration):
    name, stored, (site_dims, vector, prior, sipnet_map, external_inputs) = calibration
    values = vector.flat_to_dataset(vector.to_natural(stored["theta"].values), batch_dims=("sample",))
    compared = 0
    for variable in (v for v in stored.data_vars if v.startswith("natural:") and "." not in v):
        parameter = variable.removeprefix("natural:")
        if parameter in values:
            np.testing.assert_allclose(values[parameter].transpose(*stored[variable].dims), stored[variable],
                                       rtol=TOLERANCE, err_msg=parameter)
            compared += 1
    for part in ("leaf", "wood", "fine_root", "coarse_root"):
        variable = stored[f"natural:allocation.{part}"]
        new = values["allocation"].sel(allocation_part=part).transpose(*variable.dims)
        np.testing.assert_allclose(new, variable, rtol=TOLERANCE, err_msg=part)
    assert compared == len(vector) - 1


def test_the_sipnet_parameter_fields_are_the_same(calibration):
    name, stored, (site_dims, vector, prior, sipnet_map, external_inputs) = calibration
    values = labeled_values(vector, stored["theta"].values, external_inputs)
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    stored_names = {v.removeprefix("sipnet:") for v in stored.data_vars if v.startswith("sipnet:")}
    assert set(fields.data_vars) == stored_names
    narrowed = set()
    for field_name in stored_names:
        expected = stored[f"sipnet:{field_name}"]
        assert set(fields[field_name].dims) <= set(expected.dims), field_name
        if set(fields[field_name].dims) != set(expected.dims):
            narrowed.add(field_name)
        np.testing.assert_allclose(fields[field_name].broadcast_like(expected).transpose(*expected.dims), expected,
                                   rtol=TOLERANCE, err_msg=field_name)
    # The rule contract: the old rule wrote all four initial conditions on the
    # dims of everything it read; each now depends on its own value alone, so
    # the two read from external inputs on site are on site alone.
    assert narrowed == ({"leaf_area_index", "soil_wetness_fraction"} if name == "single_site" else set())
    assert sipnet_map.out_of_domain(fields, values, site_dims=site_dims).empty


def test_the_older_example_reference_maps_to_its_fields():
    stored = example_reference()
    site_dims = SiteDims(site_table=site_table_of(*EXAMPLE_REFERENCE_SITES),
                         site_labels={"pft": EXAMPLE_REFERENCE_PFT})
    vector, _, sipnet_map = example_calibration(site_dims)
    natural = vector.values_to_flat(example_reference_natural_values(stored, vector))
    values = vector.flat_to_dataset(natural, batch_dims=("sample",))
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    sipnet_names = [name for name in stored.data_vars if not name.startswith("natural:")]
    assert set(fields.data_vars) == set(sipnet_names)
    for name in sipnet_names:
        np.testing.assert_allclose(
            fields[name].broadcast_like(stored[name]).transpose(*stored[name].dims), stored[name],
            rtol=TOLERANCE, err_msg=name,
        )
    assert pd.Index(fields["site"].values).tolist() == list(EXAMPLE_REFERENCE_SITES)

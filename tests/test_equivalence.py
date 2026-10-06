"""The probability layer and its adapters against the code they replaced.

Two calibrations were evaluated by the code before the parameter layer at
fixed thetas, and stored by a script removed in R1
(``tests/data/write_calibration_references.py``, last at commit cf6b9af):
the example calibration, and a copy of the single-site Harvard Forest one.
Each is declared here again in the probability layer's terms and conditioned
on nothing; the single-site one twice, its four SIPNET translations once as
``Compute`` rules and once as deterministics the map copies, as the
reference script had them. Each gives the same theta layout, prior log
density, natural values and SIPNET parameter fields, to 1e-12; a field is on
fewer dims only where the rule contract says it depends on fewer values. The
older example reference (``conftest.EXAMPLE_REFERENCE``) still maps to its
stored fields from its natural values.
"""

from __future__ import annotations

import re
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
from sipnet_calibration.probability import (
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    condition_on,
    joint,
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
    check_sipnet_parameter_map_fits,
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
#: The values the single-site calibration's four SIPNET translations compute.
TRANSLATED_NAMES = ("min_photosynthesis_temperature", "base_wood_respiration_rate",
                    "base_coarse_root_respiration_rate", "base_soil_respiration_rate")


def reference(name: str) -> xr.Dataset:
    with xr.open_dataset(DATA / f"calibration_reference_{name}.nc", engine="h5netcdf") as dataset:
        return dataset.load()


def example():
    site_dims = SiteDims(site_table=site_table_of(1, 27, 4711), site_labels={"pft": ("deciduous", "conifer", "deciduous")})
    prior_factors, sipnet_map = example_calibration(site_dims)
    posterior = condition_on(joint(*prior_factors).bind(coords=site_dims.coords), {})
    return site_dims, posterior, sipnet_map, None


def single_site_factors() -> list[FactorSpec]:
    """The single-site calibration's prior, one factor per parameter."""

    def factor(name, units, support, law):
        return FactorSpec(ArraySpec(name, units=units, support=support), law=law, provenance="Reference fixture.")

    return [
        factor("photosynthetic_capacity", "nmol g-1 s-1", POSITIVE, log_normal_from_interval(lower=140.0, upper=450.0)),
        factor("respiration_share", "1", OPEN_UNIT_INTERVAL, logit_normal_from_interval(lower=0.04, upper=0.20)),
        factor("optimum_photosynthesis_temperature", "degC", REAL,
               tfp.distributions.Normal(jnp.float64(22.0), jnp.float64(2.5))),
        factor("half_saturation_light", "mol m-2 d-1", POSITIVE, log_normal_from_interval(lower=4.6, upper=26.3)),
        factor("soil_water_holding_capacity", "cm", POSITIVE, log_normal_from_interval(lower=15.0, upper=150.0)),
        factor("leaf_on_growth", "g m-2", POSITIVE, log_normal_from_interval(lower=50.0, upper=180.0)),
        factor("leaf_on_growing_degree_days", "K d", POSITIVE, log_normal_from_interval(lower=500.0, upper=1100.0)),
        FactorSpec(
            ArraySpec("allocation", units="1", support=SIMPLEX,
                      element_axes={"allocation_part": ("leaf", "wood", "fine_root", "coarse_root")}),
            law=softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]), logit_sd=jnp.array([0.25, 0.30, 0.30])),
            provenance="Reference fixture.",
        ),
        factor("wood_respiration_rate_at_10c", "yr-1", POSITIVE, log_normal_from_interval(lower=0.006, upper=0.04)),
        factor("soil_respiration_flux_at_10c", "g m-2 yr-1", POSITIVE, log_normal_from_interval(lower=200.0, upper=900.0)),
        factor("soil_respiration_q10", "1", POSITIVE, log_normal_from_interval(lower=1.3, upper=3.2)),
        factor("initial_wood_carbon", "kg m-2", POSITIVE, log_normal_from_interval(lower=5.0, upper=15.0)),
        factor("initial_soil_organic_carbon", "kg m-2", POSITIVE, log_normal_from_interval(lower=5.0, upper=60.0)),
    ]


def single_site_map(translations: list) -> SIPNETParameterMap:
    """The single-site map, with *translations* writing the four translated SIPNET parameters."""
    copies = ("optimum_photosynthesis_temperature", "half_saturation_light", "soil_water_holding_capacity",
              "leaf_on_growth", "leaf_on_growing_degree_days", "soil_respiration_q10")
    return SIPNETParameterMap(
        rules=[
            *photosynthesis_rules(capacity_value_name="photosynthetic_capacity",
                                  respiration_share_value_name="respiration_share"),
            *(Copy(value_name=name, sipnet_parameter_name=name) for name in copies),
            *translations,
            CopySimplex(value_name="allocation",
                        sipnet_parameter_names=("leaf_allocation", "wood_allocation", "fine_root_allocation")),
            *initial_condition_rules(deciduous=DECIDUOUS_BY_SITE_LABEL),
        ],
        fixed=[Fixed(sipnet_parameter_name=n, value=v, provenance="Reference fixture.") for n, v in SINGLE_SITE_FIXED.items()],
    )


def single_site_external_inputs() -> xr.Dataset:
    """The initial leaf carbon and soil moisture, on ``site``."""
    return xr.Dataset(
        {
            "initial_leaf_carbon": ("site", [0.19], {"units": "kg m-2"}),
            "initial_soil_moisture_saturation": ("site", [42.0], {"units": "percent"}),
        },
        coords={"site": np.array([4977], dtype=np.int32)},
    )


def single_site_site_dims() -> SiteDims:
    return SiteDims(site_table=site_table_of(4977), site_labels={"pft": ["temperate.deciduous.HPDA"]})


def single_site():
    """The single-site calibration, its four translations ``Compute`` rules."""
    translations = [
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
    ]
    posterior = condition_on(joint(*single_site_factors()).bind(coords={}), {})
    return single_site_site_dims(), posterior, single_site_map(translations), single_site_external_inputs()


def single_site_with_deterministics():
    """The single-site calibration, its four translations deterministics."""
    deterministics = [
        DeterministicSpec(ArraySpec("min_photosynthesis_temperature", units="degC"),
                          function=lambda optimum_photosynthesis_temperature:
                              optimum_photosynthesis_temperature - PHOTOSYNTHESIS_TEMPERATURE_RANGE),
        DeterministicSpec(ArraySpec("base_wood_respiration_rate", units="yr-1", support=POSITIVE),
                          function=lambda wood_respiration_rate_at_10c: wood_respiration_rate_at_10c / WOOD_RESPIRATION_Q10),
        DeterministicSpec(ArraySpec("base_coarse_root_respiration_rate", units="yr-1", support=POSITIVE),
                          function=lambda wood_respiration_rate_at_10c:
                              wood_respiration_rate_at_10c / COARSE_ROOT_RESPIRATION_Q10),
        DeterministicSpec(ArraySpec("base_soil_respiration_rate", units="yr-1", support=POSITIVE),
                          function=lambda soil_respiration_flux_at_10c, initial_soil_organic_carbon, soil_respiration_q10:
                              soil_respiration_flux_at_10c / (1000.0 * initial_soil_organic_carbon * soil_respiration_q10)),
    ]
    posterior = condition_on(joint(*single_site_factors(), *deterministics).bind(coords={}), {})
    translations = [Copy(value_name=name, sipnet_parameter_name=name) for name in TRANSLATED_NAMES]
    return single_site_site_dims(), posterior, single_site_map(translations), single_site_external_inputs()


#: Each calibration, and the reference it is held to.
CALIBRATIONS = {
    "example": ("example", example),
    "single_site": ("single_site", single_site),
    "single_site_with_deterministics": ("single_site", single_site_with_deterministics),
}


@pytest.fixture(scope="module", params=list(CALIBRATIONS))
def calibration(request):
    reference_name, build = CALIBRATIONS[request.param]
    return request.param, reference(reference_name), build()


def labeled_values(posterior, theta, external_inputs) -> dict[str, xr.DataArray]:
    """The labeled values at *theta*, with the external inputs beside them."""
    values = posterior.to_labeled(theta)
    return values if external_inputs is None else {**values, **dict(external_inputs.data_vars)}


def todays_entry_name(name: str) -> str:
    """A reference entry's name as theta's entries are named now: a simplex
    element was ``alr(<element>:<last element>)``, and is ``<element>``."""
    return re.sub(r"alr\((\w+):\w+\)", r"\1", name)


def test_theta_has_the_same_layout(calibration):
    name, stored, (site_dims, posterior, sipnet_map, external_inputs) = calibration
    unconstrained = posterior.parameters.unconstrained
    assert unconstrained.size == stored.sizes["entry"]
    assert list(unconstrained.entry_names) == [todays_entry_name(str(n)) for n in stored["entry"].values]
    index = unconstrained.index
    assert list(index.get_level_values("component")) == stored["entry_parameter"].values.tolist()
    for i in range(unconstrained.size):
        dim = str(stored["entry_dim"].values[i])
        if dim:
            assert str(index.get_level_values(dim)[i]) == str(stored["entry_dim_label"].values[i]), i


def test_the_prior_has_the_same_log_density(calibration):
    name, stored, (site_dims, posterior, sipnet_map, external_inputs) = calibration
    np.testing.assert_allclose(posterior.log_prior(stored["theta"].values), stored["log_prob"].values, rtol=TOLERANCE)


def test_the_natural_values_are_the_same(calibration):
    name, stored, (site_dims, posterior, sipnet_map, external_inputs) = calibration
    values = posterior.to_labeled(stored["theta"].values)
    stored_names = {v.removeprefix("natural:") for v in stored.data_vars if v.startswith("natural:") and "." not in v}
    compared = set()
    for value_name in stored_names & set(values):
        variable = stored[f"natural:{value_name}"]
        np.testing.assert_allclose(values[value_name].transpose(*variable.dims), variable,
                                   rtol=TOLERANCE, err_msg=value_name)
        compared.add(value_name)
    for part in ("leaf", "wood", "fine_root", "coarse_root"):
        variable = stored[f"natural:allocation.{part}"]
        new = values["allocation"].sel(allocation_part=part).transpose(*variable.dims)
        np.testing.assert_allclose(new, variable, rtol=TOLERANCE, err_msg=part)
    assert compared == set(values) - {"allocation", "theta"}
    # The Compute rules' values are not values of the model, so only the
    # deterministics' calibration has them; its SIPNET fields check the rest.
    assert stored_names - compared == (set(TRANSLATED_NAMES) if name == "single_site" else set())


def test_the_sipnet_parameter_fields_are_the_same(calibration):
    name, stored, (site_dims, posterior, sipnet_map, external_inputs) = calibration
    spec = posterior.model.spec
    descriptions = {n: spec.component_spec(n) for n in (*spec.component_names, *spec.input_names)}
    check_sipnet_parameter_map_fits(sipnet_map, descriptions, external_inputs, site_dims)
    values = labeled_values(posterior, stored["theta"].values, external_inputs)
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
    assert narrowed == ({"leaf_area_index", "soil_wetness_fraction"} if name.startswith("single_site") else set())
    assert sipnet_map.out_of_domain(fields, values, site_dims=site_dims).empty


def test_the_older_example_reference_maps_to_its_fields():
    stored = example_reference()
    site_dims = SiteDims(site_table=site_table_of(*EXAMPLE_REFERENCE_SITES),
                         site_labels={"pft": EXAMPLE_REFERENCE_PFT})
    prior_factors, sipnet_map = example_calibration(site_dims)
    posterior = condition_on(joint(*prior_factors).bind(coords=site_dims.coords), {})
    values_by_name = example_reference_natural_values(stored, site_dims.coords["pft"])
    values = posterior.parameters.values_to_labeled(values_by_name, batch_dims=("sample",))
    fields = sipnet_map.sipnet_parameter_fields(values, site_dims=site_dims)
    sipnet_names = [name for name in stored.data_vars if not name.startswith("natural:")]
    assert set(fields.data_vars) == set(sipnet_names)
    for name in sipnet_names:
        np.testing.assert_allclose(
            fields[name].broadcast_like(stored[name]).transpose(*stored[name].dims), stored[name],
            rtol=TOLERANCE, err_msg=name,
        )
    assert pd.Index(fields["site"].values).tolist() == list(EXAMPLE_REFERENCE_SITES)

"""Tests for the calibration record and the example calibration.

The record is two tables, one per component and input of the model and one
per SIPNET parameter with its role. The example's prior draws land in every
domain, assemble into a validated ``SIPNETParameters``, and run the bundled
Niwot fixture when a SIPNET binary is present.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from pysipnet import niwot_reference_parameters
from pysipnet.build import find_binary, missing_binary_message
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from tensorflow_probability.substrates import jax as tfp

from conftest import EXAMPLE_REFERENCE_PFT, EXAMPLE_REFERENCE_SITES, site_table_of
from sipnet_calibration.calibration import (
    ROLES,
    describe_calibration,
    example_calibration,
)
from sipnet_calibration.fields import sipnet_overrides
from sipnet_calibration.probability import (
    POSITIVE,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    Simulator,
    SimulatorOutput,
    condition_on,
    gaussian_copula,
    joint,
    log_normal,
    normal,
)
from sipnet_calibration.sipnet_parameter_map import (
    Compute,
    Copy,
    Fixed,
    SIPNETParameterMap,
    initial_condition_rules,
)
from sipnet_calibration.site_dims import SiteDims

tfd = tfp.distributions

SITE_DIMS = SiteDims(site_table=site_table_of(*EXAMPLE_REFERENCE_SITES), site_labels={"pft": EXAMPLE_REFERENCE_PFT})

#: The component table's columns, in order.
COMPONENT_COLUMNS = [
    "role", "part", "class", "indexed_by", "shape", "support", "units", "bijector",
    "law", "given", "provenance", "sipnet_parameter_names",
]


@pytest.fixture(scope="module")
def example():
    """The example calibration's posterior under no observations, and its map."""
    prior_factors, sipnet_map = example_calibration(SITE_DIMS)
    return condition_on(joint(*prior_factors).bind(coords=SITE_DIMS.coords), {}), sipnet_map


@pytest.fixture(scope="module")
def values(example):
    posterior, _ = example
    return posterior.to_labeled(posterior.sample_prior(jax.random.key(0), 200))


@pytest.fixture(scope="module")
def sipnet_parameter_fields(example, values):
    return example[1].sipnet_parameter_fields(values, site_dims=SITE_DIMS)


def test_describe_calibration_joins_the_descriptions(example):
    posterior, sipnet_map = example
    component_table, sipnet_parameter_table = describe_calibration(posterior, sipnet_map)
    assert list(component_table.index) == list(posterior.parameter_names)
    assert list(component_table.columns) == COMPONENT_COLUMNS
    row = component_table.loc["photosynthetic_capacity"]
    assert row["role"] == "parameter" and row["part"] == "photosynthetic_capacity" and row["class"] == "FactorSpec"
    assert row["support"] == "(0, inf)" and row["units"] == "nmol g-1 s-1" and row["bijector"] == "exp"
    assert row["law"] == "log-normal" and row["given"] == "" and row["provenance"].startswith("Example fixture")
    assert row["sipnet_parameter_names"] == "max_photosynthesis_rate"
    assert component_table.loc["respiration_share", "sipnet_parameter_names"] == (
        "max_photosynthesis_rate, foliar_respiration_fraction")
    assert component_table.loc["allocation", "indexed_by"] == "pft" and component_table.loc["allocation", "shape"] == (4,)
    assert component_table.loc["allocation", "law"] == "iid softmax-normal"
    assert list(sipnet_parameter_table.index) == list(sipnet_map.sipnet_parameter_names_written)
    assert list(sipnet_parameter_table.columns) == ["set_by", "role", "values_read", "depends_on", "provenance"]
    assert sipnet_parameter_table.loc["soil_carbon", "role"] == "calibrated"
    assert sipnet_parameter_table.loc["leaf_carbon_fraction", "role"] == "fixed"


def test_every_role_is_given_by_what_a_sipnet_parameter_depends_on():
    # Wood carbon is a parameter, soil and leaf carbon inputs of the model,
    # and moisture an external input.
    model = joint(
        FactorSpec(ArraySpec("initial_wood_carbon", units="kg m-2", support=POSITIVE),
                   law=log_normal(median=7.0, geometric_sd=1.3), provenance="test"),
        inputs=[ArraySpec("soil_input", units="kg m-2", support=POSITIVE),
                ArraySpec("leaf_input", units="kg m-2", support=POSITIVE)],
    ).bind(coords={}, inputs={"soil_input": 10.0, "leaf_input": 0.2})
    posterior = condition_on(model, {})
    sipnet_map = SIPNETParameterMap(
        rules=initial_condition_rules(deciduous={"deciduous": True, "conifer": False},
                                      state_value_names=("soil_input", "initial_wood_carbon", "leaf_input", "moisture_input")),
        fixed=[Fixed(sipnet_parameter_name=n, value=0.2, provenance="t")
               for n in ("fine_root_fraction", "coarse_root_fraction", "leaf_carbon_per_area")],
    )
    component_table, sipnet_parameter_table = describe_calibration(posterior, sipnet_map)
    roles = sipnet_parameter_table["role"].to_dict()
    assert roles["total_wood_carbon"] == "calibrated"
    assert roles["soil_carbon"] == roles["leaf_area_index"] == roles["soil_wetness_fraction"] == "propagated"
    assert roles["fine_root_fraction"] == "fixed"
    assert set(roles.values()) <= set(ROLES)
    assert list(component_table.index) == ["initial_wood_carbon", "soil_input", "leaf_input"]
    soil = component_table.loc["soil_input"]
    assert soil["role"] == "input" and soil["part"] == "" and soil["law"] == "" and soil["provenance"] == ""
    assert soil["sipnet_parameter_names"] == "soil_carbon"
    assert component_table.loc["leaf_input", "sipnet_parameter_names"] == "leaf_area_index"


def test_a_rule_of_constants_alone_is_constant():
    model = joint(FactorSpec(ArraySpec("x", units=None), law=normal(mean=0.0, standard_deviation=1.0), provenance="t"))
    sipnet_map = SIPNETParameterMap(rules=[Compute(
        sipnet_parameter_name="soil_carbon", values_read={}, constants={"level": xr.DataArray(1000.0)},
        function=lambda level: level, provenance="t")])
    posterior = condition_on(model.bind(coords={}), {})
    component_table, sipnet_parameter_table = describe_calibration(posterior, sipnet_map)
    assert sipnet_parameter_table.loc["soil_carbon", "role"] == "constant"
    assert component_table.loc["x", "sipnet_parameter_names"] == ""


def respiration_at_the_sites(intercept, slope, anomaly):
    return jnp.exp(intercept + slope * anomaly)


def test_describe_calibration_has_a_row_per_deterministic():
    normal_law = tfd.Normal(jnp.float64(0.0), jnp.float64(1.0))
    anomaly = xr.DataArray([0.0, 0.0, 0.0], dims="site", coords={"site": list(EXAMPLE_REFERENCE_SITES)})
    model = joint(
        FactorSpec([ArraySpec("intercept", units=None), ArraySpec("slope", units="K-1")],
                   law=gaussian_copula({"intercept": normal_law, "slope": normal_law},
                                       correlation=[[1.0, 0.2], [0.2, 1.0]]),
                   provenance="test"),
        DeterministicSpec(ArraySpec("respiration", units="yr-1", support=POSITIVE, indexed_by=("site",)),
                          function=respiration_at_the_sites, constants={"anomaly": anomaly}),
    ).bind(coords={"site": SITE_DIMS.coords["site"]})
    sipnet_map = SIPNETParameterMap(
        rules=[Copy(value_name="respiration", sipnet_parameter_name="base_soil_respiration_rate")]
    )
    component_table, sipnet_parameter_table = describe_calibration(condition_on(model, {}), sipnet_map)
    assert list(component_table.index) == ["intercept", "slope", "respiration"]
    assert component_table.loc["slope", "part"] == "intercept+slope"
    row = component_table.loc["respiration"]
    assert row["role"] == "computed" and row["part"] == "respiration" and row["class"] == "DeterministicSpec"
    assert row["law"] == "respiration_at_the_sites" and row["given"] == "intercept, slope" and row["provenance"] == ""
    assert row["indexed_by"] == "site"
    assert row["sipnet_parameter_names"] == "base_soil_respiration_rate"
    assert sipnet_parameter_table.loc["base_soil_respiration_rate", "role"] == "calibrated"
    # A parameter reaches what its deterministics reach.
    assert component_table.loc["intercept", "sipnet_parameter_names"] == "base_soil_respiration_rate"


class Doubled(Simulator):
    """``prediction = 2 x`` at each site."""

    @property
    def name(self):
        return "doubled"

    @property
    def given(self):
        return ("x",)

    @property
    def outputs(self):
        return (ArraySpec("prediction", units="1", indexed_by=("site",)),)

    def __call__(self, given_values):
        prediction = 2.0 * given_values["x"].transpose("sample", "site").values
        return SimulatorOutput(values={"prediction": prediction},
                               valid={"prediction": np.ones(prediction.shape[0], dtype=bool)})

    def at(self, coords, outputs):
        return self


def test_a_simulators_outputs_have_no_row_and_carry_what_reads_them():
    sites = SITE_DIMS.coords["site"]
    model = joint(
        FactorSpec(ArraySpec("x", units="1", indexed_by=("site",)),
                   law=lambda: tfd.Independent(tfd.Normal(jnp.zeros(len(sites)), 1.0), 1), provenance="t"),
        Doubled(),
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)),
                   law=lambda prediction: tfd.Independent(tfd.Normal(prediction, 1.0), 1), provenance="t"),
    ).bind(coords={"site": sites})
    posterior = condition_on(model, {"y": np.zeros(len(sites))})
    sipnet_map = SIPNETParameterMap(rules=[Copy(value_name="prediction", sipnet_parameter_name="leaf_off_fall_fraction")])
    component_table, sipnet_parameter_table = describe_calibration(posterior, sipnet_map)
    assert list(component_table.index) == ["x", "y"]
    assert component_table.loc["y", "role"] == "observed"
    assert component_table.loc["x", "sipnet_parameter_names"] == "leaf_off_fall_fraction"
    assert sipnet_parameter_table.loc["leaf_off_fall_fraction", "role"] == "calibrated"


def test_describe_calibration_takes_a_posterior(example):
    posterior, sipnet_map = example
    with pytest.raises(TypeError, match="takes the calibration's Posterior, got FactoredDistribution"):
        describe_calibration(posterior.model, sipnet_map)


def test_the_example_has_one_factor_per_parameter_with_its_provenance():
    prior_factors, _ = example_calibration(SITE_DIMS)
    assert [factor.name for factor in prior_factors] == [
        "photosynthetic_capacity", "respiration_share", "allocation", "base_soil_respiration",
        "leaf_fall_fraction", "initial_soil_carbon"]
    assert all(factor.provenance.startswith("Example fixture") for factor in prior_factors)


def test_the_example_needs_pft_labels():
    with pytest.raises(KeyError, match="pft"):
        example_calibration(SiteDims(site_table=site_table_of(*EXAMPLE_REFERENCE_SITES)))


def test_the_example_prior_lands_in_every_domain(example, values, sipnet_parameter_fields):
    assert example[1].out_of_domain(sipnet_parameter_fields, values, site_dims=SITE_DIMS).empty


def test_the_examples_factors_are_evaluated_by_their_base_densities(example):
    assert set(example[0].model.describe()["evaluated_by"]) == {"base density"}


def test_a_draw_assembles_into_validated_sipnet_parameters(sipnet_parameter_fields):
    base = niwot_reference_parameters().model_dump()
    group_of = {path.split(".", 1)[1]: path.split(".", 1)[0] for path in PARAMETER_SPECS}
    for sample in range(0, 200, 25):
        for site in EXAMPLE_REFERENCE_SITES:
            dump = {group: dict(values) for group, values in base.items()}
            overrides = sipnet_overrides(sipnet_parameter_fields, site=site, batch={"sample": sample})
            for name, value in overrides.items():
                dump[group_of[name]][name] = value
            parameters = SIPNETParameters.model_validate(dump)
            assert parameters.initial_conditions.soil_carbon == pytest.approx(overrides["soil_carbon"])


@pytest.mark.slow
def test_a_prior_draw_runs_the_niwot_fixture(sipnet_parameter_fields):
    from pysipnet import SIPNETModel, SIPNETRunner, niwot_reference_climate
    from pysipnet.parameters.model import ModelFlags

    if find_binary() is None:
        pytest.skip(missing_binary_message())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fixture has a few vpd <= 0 rows
        climate = niwot_reference_climate().head(8 * 30)
    model = SIPNETModel(SIPNETRunner(flags=ModelFlags.standard()), base_params=niwot_reference_parameters(),
                        base_climate=climate)
    result = model(**sipnet_overrides(sipnet_parameter_fields, site=27, batch={"sample": 0}))
    assert result.provenance.success, result.provenance.stderr
    assert result.outputs["net_ecosystem_exchange"].sizes["time"] == 8 * 30

"""Tests for the calibration record and the example calibration.

The record is two tables, one per parameter and one per SIPNET parameter
with its role. The example's prior draws land in every domain, assemble
into a validated ``SIPNETParameters``, and run the bundled Niwot fixture
when a SIPNET binary is present.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
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
from sipnet_calibration.parameters import (
    POSITIVE,
    REAL,
    DerivedParameter,
    DerivedParameters,
    Parameter,
    ParameterVector,
    Prior,
    PriorTerm,
    gaussian_copula,
)
from sipnet_calibration.sipnet_parameter_map import (
    Copy,
    Fixed,
    SIPNETParameterMap,
    initial_condition_rules,
)
from sipnet_calibration.site_dims import SiteDims

SITE_DIMS = SiteDims(site_table=site_table_of(*EXAMPLE_REFERENCE_SITES), site_labels={"pft": EXAMPLE_REFERENCE_PFT})


@pytest.fixture(scope="module")
def example():
    return example_calibration(SITE_DIMS)


@pytest.fixture(scope="module")
def values(example):
    vector, prior, _ = example
    return vector.flat_to_dataset(vector.to_natural(prior.sample(jax.random.key(0), 200)), batch_dims=("sample",))


@pytest.fixture(scope="module")
def sipnet_parameter_fields(example, values):
    return example[2].sipnet_parameter_fields(values, site_dims=SITE_DIMS)


def test_describe_calibration_joins_the_descriptions(example):
    parameter_table, sipnet_parameter_table = describe_calibration(*example)
    assert list(parameter_table.index) == list(example[0].parameter_names)
    row = parameter_table.loc["photosynthetic_capacity"]
    assert row["support"] == "(0, inf)" and row["units"] == "nmol g-1 s-1" and row["bijector"] == "exp"
    assert row["prior"] == "log-normal" and row["provenance"].startswith("Example fixture")
    assert row["sipnet_parameter_names"] == "max_photosynthesis_rate"
    assert parameter_table.loc["respiration_share", "sipnet_parameter_names"] == "max_photosynthesis_rate, foliar_respiration_fraction"
    assert parameter_table.loc["allocation", "indexed_by"] == "pft" and parameter_table.loc["allocation", "shape"] == (4,)
    assert parameter_table.loc["allocation", "prior"] == "iid softmax-normal"
    assert list(sipnet_parameter_table.index) == list(example[2].sipnet_parameter_names_written)
    assert sipnet_parameter_table.loc["soil_carbon", "role"] == "calibrated"
    assert sipnet_parameter_table.loc["leaf_carbon_fraction", "role"] == "fixed"


def test_every_role_is_given_by_what_a_sipnet_parameter_depends_on():
    vector = ParameterVector(parameters=[Parameter(name="initial_wood_carbon", support=POSITIVE, units="kg m-2")])
    prior = Prior(vector, [PriorTerm(parameter_names=("initial_wood_carbon",), distribution=
        tfp.distributions.LogNormal(jnp.float64(2.0), jnp.float64(0.3)), provenance="test")])
    sipnet_map = SIPNETParameterMap(
        rules=initial_condition_rules(deciduous={"deciduous": True, "conifer": False},
                                      state_value_names=("soil_input", "initial_wood_carbon", "leaf_input", "moisture_input")),
        fixed=[Fixed(sipnet_parameter_name=n, value=0.2, provenance="t")
               for n in ("fine_root_fraction", "coarse_root_fraction", "leaf_carbon_per_area")],
    )
    _, sipnet_parameter_table = describe_calibration(vector, prior, sipnet_map)
    roles = sipnet_parameter_table["role"].to_dict()
    assert roles["total_wood_carbon"] == "calibrated" and roles["leaf_area_index"] == "propagated"
    assert roles["fine_root_fraction"] == "fixed"
    constant = SIPNETParameterMap(rules=[*sipnet_map.rules[:1]], fixed=sipnet_map.fixed)
    assert describe_calibration(vector, prior, constant)[1].loc["soil_carbon", "role"] == "propagated"
    assert set(roles.values()) <= set(ROLES)


def test_a_rule_of_constants_alone_is_constant():
    from sipnet_calibration.sipnet_parameter_map import Compute

    vector = ParameterVector(parameters=[Parameter(name="x", units=None)])
    prior = Prior(vector, [PriorTerm(parameter_names=("x",), distribution=tfp.distributions.Normal(jnp.float64(0.0), jnp.float64(1.0)), provenance="t")])
    sipnet_map = SIPNETParameterMap(rules=[Compute(
        sipnet_parameter_name="soil_carbon", values_read={}, constants={"level": xr.DataArray(1000.0)},
        function=lambda level: level, provenance="t")])
    assert describe_calibration(vector, prior, sipnet_map)[1].loc["soil_carbon", "role"] == "constant"


def test_describe_calibration_has_a_row_per_derived_parameter_and_a_joint_terms_name():
    vector = ParameterVector(parameters=[
        Parameter(name="intercept", support=REAL, units=None),
        Parameter(name="slope", support=REAL, units="K-1"),
    ])
    derived = DerivedParameters(parameter_vector=vector, coords={"site": SITE_DIMS.coords["site"]}, derived_parameters=[
        DerivedParameter(name="respiration", units="yr-1", indexed_by=("site",), support=POSITIVE,
                         given=("intercept", "slope"),
                         constants={"anomaly": xr.DataArray([0.0, 0.0, 0.0], dims="site", coords={"site": list(EXAMPLE_REFERENCE_SITES)})},
                         function=lambda intercept, slope, anomaly: jnp.exp(intercept + slope * anomaly)),
    ])
    normal = tfp.distributions.Normal(jnp.float64(0.0), jnp.float64(1.0))
    prior = Prior(vector, [PriorTerm(parameter_names=("intercept", "slope"), distribution=
        gaussian_copula({"intercept": normal, "slope": normal}, correlation=[[1.0, 0.2], [0.2, 1.0]]),
        provenance="test",
    )], derived_parameters=derived)
    sipnet_map = SIPNETParameterMap(
        rules=[Copy(value_name="respiration", sipnet_parameter_name="base_soil_respiration_rate")]
    )
    table, sipnet_parameter_table = describe_calibration(vector, prior, sipnet_map)
    assert list(table.index) == ["intercept", "slope", "respiration"]
    assert table.loc["slope", "term"] == "intercept+slope"
    assert table.loc["slope", "prior"] == "gaussian copula"
    row = table.loc["respiration"]
    assert row["given"] == "intercept, slope" and row["prior"] == "" and row["bijector"] == ""
    assert row["sipnet_parameter_names"] == "base_soil_respiration_rate"
    assert sipnet_parameter_table.loc["base_soil_respiration_rate", "role"] == "calibrated"
    # A parameter reaches what its derived parameters reach.
    assert table.loc["intercept", "sipnet_parameter_names"] == "base_soil_respiration_rate"


def test_the_example_prior_lands_in_every_domain(example, values, sipnet_parameter_fields):
    assert example[2].out_of_domain(sipnet_parameter_fields, values, site_dims=SITE_DIMS).empty


def test_the_examples_terms_are_evaluated_by_their_base_densities(example):
    assert set(example[1].describe()["evaluated_by"]) == {"base density"}


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


def test_describe_calibration_needs_a_prior_over_the_vector(example):
    vector, _, sipnet_map = example
    other = example_calibration(SITE_DIMS)[1]
    smaller = other.select(parameter=["photosynthetic_capacity"])
    with pytest.raises(KeyError, match="the prior has no term for parameter 'respiration_share'"):
        describe_calibration(vector, smaller, sipnet_map)

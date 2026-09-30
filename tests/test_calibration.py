"""Tests for the calibration record and the example calibration.

The example's prior draws land in every pySIPNET domain, assemble into a
validated ``SIPNETParameters``, and run the bundled Niwot fixture when a
SIPNET binary is present.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest
from pysipnet import niwot_reference_parameters
from pysipnet.build import find_binary, missing_binary_message
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from tensorflow_probability.substrates import jax as tfp

from conftest import EXAMPLE_REFERENCE_PFT, EXAMPLE_REFERENCE_SITES, site_table_of
from sipnet_calibration.calibration import describe_calibration, example_calibration
from sipnet_calibration.fields import sipnet_overrides
from sipnet_calibration.parameter_vector import POSITIVE, REAL, DerivedParameter, Parameter, ParameterVector
from sipnet_calibration.prior import Prior, PriorTerm, gaussian_copula
from sipnet_calibration.sipnet_parameter_map import Copy, SIPNETParameterMap


@pytest.fixture(scope="module")
def example():
    return example_calibration(site_table_of(*EXAMPLE_REFERENCE_SITES), EXAMPLE_REFERENCE_PFT)


@pytest.fixture(scope="module")
def sipnet_parameter_fields(example):
    vector, prior, sipnet_map = example
    return sipnet_map.sipnet_parameter_fields(vector, prior.sample(jax.random.key(0), 200))


def test_describe_calibration_joins_the_three_descriptions(example):
    table = describe_calibration(*example)
    assert list(table.index) == list(example[0].parameter_names)
    row = table.loc["photosynthetic_capacity"]
    assert row["support"] == "positive" and row["units"] == "nmol g-1 s-1"
    assert row["prior"] == "log-normal" and row["provenance"].startswith("Example fixture")
    assert row["sipnet_parameters"] == "max_photosynthesis_rate, foliar_respiration_fraction"
    assert row["rules"] == "ComputePhotosynthesisRates"
    assert table.loc["allocation", "dim"] == "pft"
    assert table.loc["allocation", "prior"] == "iid softmax-normal"


def test_describe_calibration_has_a_row_per_derived_parameter_and_a_joint_terms_name():
    vector = ParameterVector(
        parameters=[
            Parameter(name="intercept", support=REAL, units=None),
            Parameter(name="slope", support=REAL, units="K-1"),
        ],
        derived_parameters=[
            DerivedParameter(
                name="respiration", units="yr-1", dim="site", support=POSITIVE,
                derived_from=("intercept", "slope"),
                compute=lambda dim_index, site_table, intercept, slope: jnp.exp(
                    intercept + slope * site_table["anomaly"].to_numpy()),
            )
        ],
        site_table=site_table_of(*EXAMPLE_REFERENCE_SITES).assign(anomaly=0.0),
        site_covariate_names=["anomaly"],
    )
    normal = tfp.distributions.Normal(jnp.float64(0.0), jnp.float64(1.0))
    prior = Prior(vector, {("intercept", "slope"): PriorTerm(
        gaussian_copula({"intercept": normal, "slope": normal}, correlation=[[1.0, 0.2], [0.2, 1.0]]),
        provenance="test",
    )})
    sipnet_map = SIPNETParameterMap(
        rules=[Copy(value_name="respiration", sipnet_parameter_name="base_soil_respiration_rate")]
    )
    table = describe_calibration(vector, prior, sipnet_map)
    assert list(table.index) == ["intercept", "slope", "respiration"]
    assert table.loc["slope", "term"] == "intercept+slope"
    assert table.loc["slope", "prior"] == "gaussian copula"
    derived = table.loc["respiration"]
    assert derived["derived_from"] == "intercept, slope" and derived["prior"] == ""
    assert derived["sipnet_parameters"] == "base_soil_respiration_rate" and derived["rules"] == "Copy"


def test_the_example_prior_lands_in_every_domain(example, sipnet_parameter_fields):
    assert example[2].out_of_domain(sipnet_parameter_fields).empty


def test_the_example_is_gaussian_in_theta(example):
    assert example[1].describe()["declared_gaussian"].all()
    assert example[1].gaussian().mean.shape == (example[0].dimension,)


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
    other = example_calibration(site_table_of(*EXAMPLE_REFERENCE_SITES), EXAMPLE_REFERENCE_PFT)[1]
    smaller = other.select(parameter_names=["photosynthetic_capacity"])
    with pytest.raises(KeyError, match="the prior has no term for parameter 'respiration_share'"):
        describe_calibration(vector, smaller, sipnet_map)

"""Tests for the parts of a model: factors, deterministics, the keyword rule
and the builders' declarations.

A part reads what its function's signature names without defaults; what
it holds and does not read, or reads of its own, is refused when it is
declared; and a builder says what it wraps and reads.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    POSITIVE,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    deterministic,
    factor,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
    log_normal,
    logit_normal,
    softmax_normal,
)

tfd = tfp.distributions

SOIL_CARBON = ArraySpec("soil_carbon", units="kg m-2", support=POSITIVE, indexed_by=("site",))
SPREAD = ArraySpec("spread", units="1", support=POSITIVE)
MEDIAN = xr.DataArray([1.0, 2.0], dims="site", coords={"site": [1, 2]})


def _law_of(soil_carbon_location, spread):
    return tfd.Independent(tfd.LogNormal(soil_carbon_location, spread), 1)


def test_a_factor_reads_the_keywords_its_law_names_without_defaults():
    def law(soil_carbon_location, spread, shift=0.0):
        return _law_of(soil_carbon_location + shift, spread)

    spec = FactorSpec(SOIL_CARBON, law=law, provenance="test")
    assert spec.given == ("soil_carbon_location", "spread") and spec.reads == spec.given
    assert spec.name == "soil_carbon" and spec.law_name == "law"


def test_constants_and_label_maps_are_read_but_not_given():
    def law(median, spread):
        return tfd.Independent(tfd.LogNormal(jnp.log(median), spread), 1)

    spec = FactorSpec(SOIL_CARBON, law=law, constants={"median": MEDIAN})
    assert spec.given == ("spread",) and spec.reads == ("median", "spread")


def test_a_partials_bound_arguments_are_defaults():
    spec = FactorSpec(SOIL_CARBON, law=functools.partial(_law_of, spread=0.5))
    assert spec.given == ("soil_carbon_location",)


@pytest.mark.parametrize(
    "law",
    [
        lambda *values: None,
        lambda **values: None,
        eval("lambda location, /, spread: None"),
    ],
    ids=["args", "kwargs", "positional-only"],
)
def test_the_keyword_rule_refuses_a_signature_it_cannot_read(law):
    with pytest.raises(TypeError, match="cannot be read off its signature"):
        FactorSpec(SOIL_CARBON, law=law)


def test_the_keyword_rule_refuses_a_deterministic_of_kwargs():
    with pytest.raises(TypeError, match="cannot be read off its signature"):
        DeterministicSpec(SOIL_CARBON, function=lambda **values: None)


def test_a_constant_the_law_never_reads_is_refused():
    with pytest.raises(ValueError, match="never reads"):
        FactorSpec(SOIL_CARBON, law=_law_of, constants={"median": MEDIAN})


def test_a_law_reading_its_own_event_is_refused():
    with pytest.raises(ValueError, match="which the factor declares"):
        FactorSpec(SOIL_CARBON, law=lambda soil_carbon: None)


def test_a_name_held_twice_is_refused():
    with pytest.raises(ValueError, match="more than once"):
        FactorSpec(SOIL_CARBON, law=lambda soil_carbon_location: None,
                   constants={"soil_carbon": MEDIAN})


def test_a_joint_factor_is_indexed_alike():
    with pytest.raises(ValueError, match="indexed alike"):
        FactorSpec([SOIL_CARBON, SPREAD], law=lambda: None)


@pytest.mark.parametrize(
    ("event", "arguments", "message"),
    [
        (SOIL_CARBON, {}, "iid_over_dim"),
        ([SPREAD, ArraySpec("other", units="1")], {}, "joint factor"),
    ],
)
def test_a_bare_law_is_for_one_component_indexed_by_nothing(event, arguments, message):
    with pytest.raises(TypeError, match=message):
        FactorSpec(event, law=tfd.Normal(0.0, 1.0), **arguments)


def test_a_bare_law_holding_a_constant_is_refused():
    with pytest.raises(TypeError, match="never reads|bare law"):
        FactorSpec(SPREAD, law=tfd.Normal(0.0, 1.0), constants={"median": MEDIAN})


@pytest.mark.parametrize(
    ("event", "law", "error"),
    [
        ("soil_carbon", lambda: None, TypeError),
        ([], lambda: None, ValueError),
        ([SOIL_CARBON, "spread"], lambda: None, TypeError),
        (SPREAD, 3.0, TypeError),
    ],
)
def test_a_factor_declares_array_specs_and_a_law(event, law, error):
    with pytest.raises(error):
        FactorSpec(event, law=law)


def test_a_provenance_is_a_sentence_or_absent():
    assert FactorSpec(SPREAD, law=log_normal(median=1.0, geometric_sd=2.0)).provenance is None
    with pytest.raises(TypeError, match="provenance"):
        FactorSpec(SPREAD, law=log_normal(median=1.0, geometric_sd=2.0), provenance="  ")


def test_parts_are_frozen():
    spec = FactorSpec(SPREAD, law=log_normal(median=1.0, geometric_sd=2.0))
    with pytest.raises(AttributeError, match="frozen"):
        spec.law = None
    computed = DeterministicSpec(SOIL_CARBON, function=lambda spread: spread)
    with pytest.raises(AttributeError, match="frozen"):
        computed.function = None


def test_a_deterministic_reads_a_component_and_none_of_its_outputs():
    with pytest.raises(ValueError, match="reads no component"):
        DeterministicSpec(SOIL_CARBON, function=lambda median: median, constants={"median": MEDIAN})
    with pytest.raises(ValueError, match="which it computes"):
        DeterministicSpec(SOIL_CARBON, function=lambda soil_carbon: soil_carbon)


def test_calling_a_deterministic_calls_its_function():
    computed = DeterministicSpec(SOIL_CARBON, function=lambda spread, median: spread * median, constants={"median": MEDIAN})
    assert computed.given == ("spread",)
    np.testing.assert_array_equal(computed(spread=2.0, median=np.array([1.0, 3.0])), [2.0, 6.0])


def test_the_decorators_make_parts():
    @factor(SOIL_CARBON, provenance="test")
    def soil_carbon(soil_carbon_location, spread):
        return _law_of(soil_carbon_location, spread)

    @deterministic(ArraySpec("soil_carbon_location", units="1", indexed_by=("site",)),
                   label_maps={"site_pft": xr.DataArray(["a", "b"], dims="site", coords={"site": [1, 2]}, name="pft")})
    def soil_carbon_location(pft_mean, site_pft):
        return pft_mean[site_pft]

    assert isinstance(soil_carbon, FactorSpec) and soil_carbon.provenance == "test"
    assert soil_carbon.given == ("soil_carbon_location", "spread")
    assert isinstance(soil_carbon_location, DeterministicSpec) and soil_carbon_location.given == ("pft_mean",)


def test_the_builders_say_what_they_wrap_and_read():
    fixed = log_normal(median=2.0, geometric_sd=1.5)
    assert iid_over_dim(fixed).law is fixed and iid_over_dim(fixed).reads == ()

    def per_label(spread):
        return tfd.Normal(0.0, spread)

    assert iid_over_dim(per_label).law is per_label and iid_over_dim(per_label).reads == ("spread",)
    independent = independent_over_dim(log_normal, geometric_sd=2.0)
    assert independent.law is log_normal and independent.reads == ("median",)
    assert independent_over_dim(softmax_normal, logit_sd=0.5).reads == ("center",)
    marginals = {"a": log_normal(median=1.0, geometric_sd=2.0), "b": logit_normal(median=0.5, logit_sd=1.0)}
    copula = gaussian_copula(marginals, correlation=np.eye(2))
    assert dict(copula.law) == marginals and copula.reads == ()


def test_a_factor_reads_what_its_builder_reads():
    spec = FactorSpec(SOIL_CARBON, law=independent_over_dim(log_normal, geometric_sd=2.0), constants={"median": MEDIAN})
    assert spec.given == () and spec.reads == ("median",) and spec.law_name == "independent log_normal"
    given = FactorSpec(SOIL_CARBON, law=iid_over_dim(lambda spread: tfd.LogNormal(0.0, spread)))
    assert given.given == ("spread",)


def test_a_builder_of_a_kwargs_function_is_refused_when_a_factor_reads_it():
    builder = iid_over_dim(lambda **values: tfd.Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="cannot be read off its signature"):
        FactorSpec(SOIL_CARBON, law=builder)

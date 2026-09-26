"""Tests for the parameter vector.

The prior helpers are checked against the quantities they claim to set
(median, interval, refusal of bad samples). The bijectors and the log prior
are checked by round trips and against a finite-difference Jacobian,
calibration parameter by calibration parameter, so the identity
``log_prior`` rests on is verified rather than trusted. Each construction
check is provoked once. The three value representations are checked against
their data models and each other: Fields and Flat round trip in both spaces,
``flat`` refuses what no Flat vector can be, and the SIPNET parameter fields agree with
Fields. The example vector is pushed through pySIPNET: prior draws land in
every domain, one draw assembles into a validated ``SIPNETParameters``, and
one draw runs the bundled Niwot fixture when the binary is present. The
module docstring's worked session is executed against the real site table
and site labels when they are present.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.build import find_binary, missing_binary_message
from pysipnet.parameters.base import ParameterDomain
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from tensorflow_probability.substrates import jax as tfp

from conftest import niwot_parameters, site_table_of
from sipnet_calibration import parameter_vector as module
from sipnet_calibration.conventions import LAT_ATTRIBUTES, LON_ATTRIBUTES
from sipnet_calibration.parameter_vector import (
    ALLOCATION,
    DOMAIN_CHECK_CORNERS,
    PHOTOSYNTHESIS,
    CalibrationParameter,
    FixedParameter,
    Identity,
    ParameterVector,
    PhotosynthesisMap,
    SimplexMap,
    example_parameter_vector,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    product_transformed_gaussian_prior,
    sipnet_overrides,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors




SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
FLAT_SPECS = {path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()}


@pytest.fixture(scope="module")
def example() -> ParameterVector:
    return example_parameter_vector(sites=SITES, pft=PFT)


@pytest.fixture(scope="module")
def theta(example) -> jax.Array:
    return example.sample(jax.random.key(0), n=8)


def in_domain(domain: ParameterDomain, values: np.ndarray) -> bool:
    return module._in_domain(domain, jnp.asarray(values))


# ── x64 ──────────────────────────────────────────────────────────────────────


def test_import_enables_float64():
    assert jax.config.jax_enable_x64
    assert log_normal(median=1.0, geometric_sd=2.0).sample(seed=jax.random.key(0)).dtype == jnp.float64


# ── prior helpers ────────────────────────────────────────────────────────────


def test_log_normal_median_and_geometric_sd_are_what_they_say():
    prior = log_normal(median=0.01, geometric_sd=3.0)
    assert float(prior.quantile(0.5)) == pytest.approx(0.01)
    z = float(tfd.Normal(0.0, 1.0).quantile(0.975))
    assert float(prior.quantile(0.975)) == pytest.approx(0.01 * 3.0**z)
    assert isinstance(prior.distribution, tfd.Normal)
    assert isinstance(prior.bijector, tfb.Exp)


def test_log_normal_refuses_non_multiplying_spread():
    with pytest.raises(ValueError, match="exceed 1"):
        log_normal(median=1.0, geometric_sd=0.5)
    with pytest.raises(ValueError, match="positive"):
        log_normal(median=-1.0, geometric_sd=2.0)


def test_log_normal_from_interval_hits_the_interval_exactly():
    prior = log_normal_from_interval(lower=0.004, upper=0.020)
    assert float(prior.quantile(0.025)) == pytest.approx(0.004)
    assert float(prior.quantile(0.975)) == pytest.approx(0.020)
    assert float(prior.quantile(0.5)) == pytest.approx(np.sqrt(0.004 * 0.020))


def test_log_normal_from_samples_fits_and_refuses_bad_values():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(-2.0, 0.5, size=20_000))
    prior = log_normal_from_samples(x)
    assert float(prior.distribution.loc) == pytest.approx(-2.0, abs=0.02)
    assert float(prior.distribution.scale) == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError, match="2 of 5 samples"):
        log_normal_from_samples([1.0, 2.0, -3.0, np.nan, 4.0])


def test_logit_normal_median_and_interval():
    prior = logit_normal(median=0.3, logit_sd=0.7)
    assert float(prior.quantile(0.5)) == pytest.approx(0.3)
    assert isinstance(prior, tfd.LogitNormal)
    prior = logit_normal_from_interval(lower=0.1, upper=0.4, mass=0.9)
    assert float(prior.quantile(0.05)) == pytest.approx(0.1)
    assert float(prior.quantile(0.95)) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="inside \\(0, 1\\)"):
        logit_normal(median=1.0, logit_sd=1.0)


def test_logit_normal_from_samples_refuses_the_boundary():
    with pytest.raises(ValueError, match="outside the support"):
        logit_normal_from_samples([0.2, 0.5, 1.0])
    rng = np.random.default_rng(1)
    x = 1 / (1 + np.exp(-rng.normal(-0.8, 0.4, size=20_000)))
    prior = logit_normal_from_samples(x)
    assert float(prior.distribution.loc) == pytest.approx(-0.8, abs=0.02)
    assert float(prior.distribution.scale) == pytest.approx(0.4, abs=0.02)


def test_softmax_normal_is_centered_and_on_the_simplex():
    center = np.array([0.18, 0.40, 0.07, 0.35])
    prior = softmax_normal(center=center, logit_sd=0.5)
    assert tuple(prior.event_shape) == (4,)
    np.testing.assert_allclose(prior.bijector.forward(prior.distribution.loc), center, rtol=1e-12)
    draws = np.asarray(prior.sample(1000, seed=jax.random.key(2)))
    assert (draws > 0).all()
    np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="sum to 1"):
        softmax_normal(center=[0.5, 0.6], logit_sd=1.0)


def test_softmax_normal_accepts_one_center_per_group():
    prior = softmax_normal(center=[[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]], logit_sd=0.5)
    assert tuple(prior.batch_shape) == (2,)
    assert tuple(prior.event_shape) == (3,)
    np.testing.assert_allclose(prior.distribution.stddev(), 0.5)
    per_element = softmax_normal(center=[0.25] * 4, logit_sd=[0.1, 0.2, 0.3])
    np.testing.assert_allclose(per_element.distribution.stddev(), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="one value per unconstrained element"):
        softmax_normal(center=[0.25] * 4, logit_sd=[0.1, 0.2])


def test_product_prior_is_independent_with_a_gaussian_base():
    a = log_normal(median=100.0, geometric_sd=1.5)
    b = logit_normal(median=0.2, logit_sd=0.4)
    prior = product_transformed_gaussian_prior(capacity=a, share=b)
    assert isinstance(prior.distribution, tfd.MultivariateNormalDiag)
    assert tuple(prior.event_shape) == (2,)
    x = jnp.array([120.0, 0.25])
    assert float(prior.log_prob(x)) == pytest.approx(float(a.log_prob(x[0]) + b.log_prob(x[1])))
    with pytest.raises(TypeError, match="Normal base"):
        product_transformed_gaussian_prior(a=a, b=tfd.Beta(2.0, 3.0))
    with pytest.raises(ValueError, match="at least two"):
        product_transformed_gaussian_prior(a=a)


# ── SIPNET maps ─────────────────────────────────────────────


def test_photosynthesis_map_inverts_the_identifiable_pair():
    a_max, a_max_frac, fol_resp, c_frac_leaf = 58.0, 0.76, 0.17, 0.466
    capacity = a_max * (a_max_frac + fol_resp) / c_frac_leaf
    share = fol_resp / (a_max_frac + fol_resp)
    out = PHOTOSYNTHESIS(
        jnp.array([capacity, share]),
        {
            "daily_mean_photosynthesis_fraction": jnp.float64(a_max_frac),
            "leaf_carbon_fraction": jnp.float64(c_frac_leaf),
        },
    )
    assert float(out["max_photosynthesis_rate"]) == pytest.approx(a_max)
    assert float(out["foliar_respiration_fraction"]) == pytest.approx(fol_resp)
    assert PHOTOSYNTHESIS.components == ("capacity", "respiration_share")
    assert set(PHOTOSYNTHESIS.reads) == set(PhotosynthesisMap().reads)


def test_simplex_map_writes_all_but_the_residual():
    out = ALLOCATION(jnp.array([0.1, 0.2, 0.3, 0.4]), {})
    assert set(out) == {"leaf_allocation", "wood_allocation", "fine_root_allocation"}
    assert ALLOCATION.components[-1] == "coarse_root_allocation"
    assert isinstance(ALLOCATION, SimplexMap) and ALLOCATION.reads == ()


def test_identity_map_shorthand():
    parameter = CalibrationParameter(
        name="w", prior=log_normal(median=1.0, geometric_sd=2.0),
        sipnet_map="wood_turnover_rate", provenance="test",
    )
    assert isinstance(parameter.sipnet_map, Identity)
    assert parameter.sipnet_map.writes == ("wood_turnover_rate",)
    assert parameter.element_labels == ("log(wood_turnover_rate)",)
    scaled = CalibrationParameter(
        name="t", prior=tfd.Uniform(jnp.float64(1.0), jnp.float64(5.0)),
        sipnet_map="optimum_photosynthesis_temperature", provenance="x",
    )
    assert scaled.element_labels == ("logit(optimum_photosynthesis_temperature in (1, 5))",)


# ── the specs and their checks ───────────────────────────────────────────────


def test_calibration_parameter_refuses_bad_name_parameter_and_provenance():
    prior = log_normal(median=1.0, geometric_sd=2.0)
    with pytest.raises(ValueError, match="lower_case_with_underscores"):
        CalibrationParameter(name="BadName", prior=prior, sipnet_map="wood_turnover_rate", provenance="x")
    with pytest.raises(ValueError, match="not pySIPNET parameter names"):
        CalibrationParameter(name="a", prior=prior, sipnet_map="aMax", provenance="x")
    with pytest.raises(ValueError, match="provenance is empty"):
        CalibrationParameter(name="a", prior=prior, sipnet_map="wood_turnover_rate", provenance="  ")


def test_calibration_parameter_refuses_component_count_mismatch():
    with pytest.raises(ValueError, match="names 4 components"):
        CalibrationParameter(
            name="a", prior=softmax_normal(center=[0.5, 0.5], logit_sd=1.0),
            sipnet_map=ALLOCATION, provenance="x",
        )


def test_fixed_parameter_checks_domain_and_shape():
    with pytest.raises(ValueError, match="outside its pySIPNET domain"):
        FixedParameter(name="leaf_carbon_fraction", value=1.5, provenance="x")
    with pytest.raises(ValueError, match="must be a float"):
        FixedParameter(name="leaf_carbon_fraction", value={"a": 0.4}, provenance="x")
    with pytest.raises(ValueError, match="must be a mapping"):
        FixedParameter(name="leaf_carbon_fraction", value=0.4, varies_by="pft", provenance="x")
    with pytest.raises(ValueError, match="not pySIPNET parameter names"):
        FixedParameter(name="cFracLeaf", value=0.4, provenance="x")


def build(parameters, fixed=(), sites=SITES, site_labels=None):
    return ParameterVector(
        parameters=parameters, fixed=fixed, sites=sites,
        site_labels={"pft": PFT} if site_labels is None else site_labels,
    )


def distinct_groups() -> ParameterVector:
    """A vector whose groups can be told apart: one prior per site and per
    PFT with distinct medians and spreads, and a per-PFT fixed value."""
    return ParameterVector(
        parameters=(
            CalibrationParameter(
                name="soil", prior=log_normal(median=[100.0, 200.0, 300.0], geometric_sd=[1.5, 2.0, 2.5]),
                sipnet_map="soil_carbon", varies_by="site", provenance="test",
            ),
            CalibrationParameter(
                name="turnover", prior=log_normal(median=[0.01, 0.02], geometric_sd=[1.2, 1.3]),
                sipnet_map="wood_turnover_rate", varies_by="pft", provenance="test",
            ),
            CalibrationParameter(
                name="allocation",
                prior=softmax_normal(center=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], logit_sd=[0.3, 0.6, 0.9]),
                sipnet_map=ALLOCATION, varies_by="pft", provenance="test",
            ),
        ),
        fixed=(FixedParameter(
            name="leaf_carbon_fraction", value={"a": 0.4, "b": 0.5}, varies_by="pft", provenance="test",
        ),),
        sites=SITES, site_labels={"pft": ("a", "b", "a")},   # site 27 is "b", the second group
    )


def rate(name="r", parameter="wood_turnover_rate", **kwargs):
    return CalibrationParameter(
        name=name, prior=log_normal(median=0.01, geometric_sd=2.0),
        sipnet_map=parameter, provenance="test", **kwargs,
    )


def test_parameter_vector_refuses_duplicate_names_and_writers():
    with pytest.raises(ValueError, match="names repeat"):
        build((rate(), rate()))
    with pytest.raises(ValueError, match="set more than once"):
        build((rate("a"), rate("b")))
    with pytest.raises(ValueError, match="set more than once"):
        build((rate(),), fixed=(FixedParameter(name="wood_turnover_rate", value=0.01, provenance="x"),))


def test_parameter_vector_refuses_unfixed_reads_and_missing_site_labels():
    pair = CalibrationParameter(
        name="p",
        prior=product_transformed_gaussian_prior(
            c=log_normal(median=100.0, geometric_sd=1.5), s=logit_normal(median=0.2, logit_sd=0.4)
        ),
        sipnet_map=PHOTOSYNTHESIS, provenance="x",
    )
    with pytest.raises(ValueError, match="which are not fixed"):
        build((pair,))
    with pytest.raises(ValueError, match="have no site labels"):
        build((rate(varies_by="landcover"),))
    with pytest.raises(ValueError, match="one label per site"):
        build((rate(varies_by="pft"),), site_labels={"pft": ("a", "b")})
    with pytest.raises(ValueError, match="reserved"):
        build((rate(),), site_labels={"sample": PFT})


def test_calibration_parameter_refuses_unusable_priors_and_sipnet_maps():
    prior = log_normal(median=1.0, geometric_sd=2.0)
    with pytest.raises(ValueError, match="rank above 1"):
        CalibrationParameter(
            name="a", prior=log_normal(median=np.ones((2, 2)), geometric_sd=2.0),
            sipnet_map="wood_turnover_rate", provenance="x",
        )
    with pytest.raises(ValueError, match="event shape .* rank above 1"):
        CalibrationParameter(
            name="a", prior=tfd.Independent(tfd.Normal(jnp.zeros((2, 3)), jnp.float64(1.0)), 2),
            sipnet_map="optimum_photosynthesis_temperature", provenance="x",
        )
    with pytest.raises(ValueError, match="not float64"):
        CalibrationParameter(name="a", prior=tfd.LogNormal(0.0, 1.0), sipnet_map="wood_turnover_rate", provenance="x")
    with pytest.raises(TypeError, match="must be a TFP distribution"):
        CalibrationParameter(name="a", prior=tfb.Exp(), sipnet_map="wood_turnover_rate", provenance="x")
    with pytest.raises(TypeError, match="must be a SIPNETMap"):
        CalibrationParameter(name="a", prior=prior, sipnet_map={"writes": ()}, provenance="x")
    mixture = tfd.MixtureSameFamily(
        tfd.Categorical(probs=jnp.array([0.5, 0.5])), tfd.LogNormal(jnp.array([0.0, 1.0]), jnp.float64(1.0))
    )
    with pytest.raises(ValueError, match="no default event-space bijector"):
        CalibrationParameter(name="a", prior=mixture, sipnet_map="soil_carbon", provenance="x")
    # ... whereas wrapping it in an explicit bijector is accepted.
    wrapped = CalibrationParameter(
        name="a", prior=tfd.TransformedDistribution(mixture, tfb.Identity()),
        sipnet_map="soil_carbon", provenance="x",
    )
    assert wrapped.bijector is not None


def test_parameter_vector_refuses_bad_sites_and_site_labels():
    with pytest.raises(TypeError, match="float"):
        build((rate(),), sites=(1.5, 27.0, 4711.0))
    with pytest.raises(TypeError, match="float"):
        build((rate(),), sites=np.array([1.0, 27.0, 4711.0]))
    with pytest.raises(TypeError, match="must be an integer"):
        build((rate(),), sites=("1", 27, 4711))
    assert build((rate(),), sites=np.array([1, 27, 4711])).sites == SITES
    assert build((rate(),), sites=jnp.array([1, 27, 4711])).sites == SITES
    assert build((rate(),), sites=xr.DataArray([1, 27, 4711], dims="site")).sites == SITES
    with pytest.raises(TypeError, match="sequence of one label per site"):
        build((rate(varies_by="pft"),), site_labels={"pft": "abc"})
    assert build((rate(varies_by="pft"),), site_labels={"pft": np.array(PFT)}).group_labels("pft") == ("conifer", "deciduous")
    with pytest.raises(ValueError, match="collide with a calibration parameter or SIPNET parameter"):
        build((rate(),), site_labels={"pft": PFT, "wood_turnover_rate": (1, 2, 3)})
    with pytest.raises(ValueError, match="collide"):
        build((rate(name="landcover"),), site_labels={"pft": PFT, "landcover": (1, 2, 3)})


def test_parameter_vector_refuses_wrong_prior_batch_and_fixed_coverage():
    per_site = CalibrationParameter(
        name="s", prior=log_normal(median=np.ones(2), geometric_sd=2.0),
        sipnet_map="soil_carbon", varies_by="site", provenance="x",
    )
    with pytest.raises(ValueError, match="batch shape \\(2,\\)"):
        build((per_site,))
    with pytest.raises(ValueError, match="missing \\['deciduous'\\]"):
        build(
            (rate(),),
            fixed=(FixedParameter(
                name="leaf_carbon_fraction", value={"conifer": 0.5}, varies_by="pft", provenance="x",
            ),),
        )
    with pytest.raises(ValueError, match="ascending"):
        build((rate(),), sites=(27, 1, 4711))


def test_parameter_vector_refuses_a_transform_that_leaves_the_domain():
    # A Normal is not a TransformedDistribution: its default bijector is the
    # identity, so theta = -12 writes a negative rate.
    normal = CalibrationParameter(
        name="n", prior=tfd.Normal(jnp.float64(0.01), jnp.float64(0.005)),
        sipnet_map="wood_turnover_rate", provenance="x",
    )
    with pytest.raises(ValueError, match="outside its pySIPNET domain 'positive'"):
        build((normal,))
    # Whereas the same Normal on a real-domain parameter is fine, and has no
    # analytic moments through the identity? It does: the base is itself.
    fine = CalibrationParameter(
        name="n", prior=tfd.Normal(jnp.float64(20.0), jnp.float64(5.0)),
        sipnet_map="optimum_photosynthesis_temperature", provenance="x",
    )
    p = build((fine,))
    assert p.dimension == 1


def test_domain_check_corners_are_far_but_finite():
    assert DOMAIN_CHECK_CORNERS == (-12.0, 0.0, 12.0)
    assert np.isfinite(np.exp(12.0)) and 0 < 1 / (1 + np.exp(12.0)) < 1e-5


def test_domain_check_looks_at_the_corners_not_only_the_center():
    # In domain at theta = 0 (the value 0.5), out of it at theta = +-12.
    inside_at_center = CalibrationParameter(
        name="n", prior=tfd.Normal(jnp.float64(0.5), jnp.float64(0.1)),
        sipnet_map="leaf_off_fall_fraction", provenance="x",
    )
    with pytest.raises(ValueError, match="outside its pySIPNET domain 'unit_interval'"):
        build((inside_at_center,))


def test_in_domain_predicates_at_the_boundaries():
    D = ParameterDomain
    assert in_domain(D.OPEN_UNIT_INTERVAL, np.array([0.5])) and not in_domain(D.OPEN_UNIT_INTERVAL, np.array([0.0]))
    assert not in_domain(D.OPEN_UNIT_INTERVAL, np.array([1.0]))
    assert in_domain(D.UNIT_INTERVAL, np.array([0.0, 1.0])) and not in_domain(D.UNIT_INTERVAL, np.array([1.0000001]))
    assert in_domain(D.NON_NEGATIVE, np.array([0.0])) and not in_domain(D.POSITIVE, np.array([0.0]))
    for domain in D:
        assert not in_domain(domain, np.array([np.nan]))
        assert not in_domain(domain, np.array([np.inf]))
    with pytest.raises(ValueError, match="finite and positive"):
        log_normal(median=np.inf, geometric_sd=2.0)
    with pytest.raises(ValueError, match="more than once"):
        build((rate(),), sites=(1, 1, 27), site_labels={"pft": PFT})


# ── layout ───────────────────────────────────────────────────────────────────


def test_layout_dimension_labels_and_slices(example):
    layout = example.layout
    assert layout.dimension == 2 + 3 * 2 + 1 * 2 + 1 + 1 * 3 == 14
    assert layout.parameters == (
        "photosynthesis", "allocation", "base_soil_respiration", "leaf_fall_fraction",
        "initial_soil_carbon",
    )
    assert layout.groups["allocation"] == ("conifer", "deciduous")
    assert layout.groups["initial_soil_carbon"] == SITES
    assert layout.dims == {
        "photosynthesis": "shared", "allocation": "pft", "base_soil_respiration": "pft",
        "leaf_fall_fraction": "shared", "initial_soil_carbon": "site",
    }
    assert layout.column_labels[0] == "photosynthesis[log(capacity)]"
    assert layout.column_labels[2] == "allocation[conifer][alr(leaf_allocation:coarse_root_allocation)]"
    assert layout.column_labels[-1] == "initial_soil_carbon[4711]"
    assert len(layout.column_labels) == 14
    deciduous = [layout.column_labels[i] for i in layout.index("allocation", group="deciduous")]
    assert deciduous == [
        f"allocation[deciduous][{e}]" for e in layout.element_labels["allocation"]
    ]
    assert all("[conifer]" in layout.column_labels[i] for i in layout.index("allocation", group="conifer"))
    stops = [layout.slice(c).stop for c in layout.parameters]
    assert stops == [2, 8, 10, 11, 14]


def test_layout_index_narrows_by_group_and_element(example):
    layout = example.layout
    np.testing.assert_array_equal(layout.index("allocation", group="deciduous"), [5, 6, 7])
    np.testing.assert_array_equal(
        layout.index("allocation", element="alr(wood_allocation:coarse_root_allocation)"), [3, 6]
    )
    np.testing.assert_array_equal(layout.index("initial_soil_carbon", group=27), [12])
    with pytest.raises(KeyError, match="not a group"):
        layout.index("allocation", group="grassland")
    with pytest.raises(KeyError, match="no calibration parameter"):
        layout.slice("nope")


def test_layout_pack_unpack_round_trip(example, theta):
    parts = example.layout.unpack(theta)
    assert parts["allocation"].shape == (8, 2, 3)
    assert parts["initial_soil_carbon"].shape == (8, 3, 1)
    np.testing.assert_array_equal(example.layout.pack(parts), theta)
    single = example.layout.unpack(theta[0])
    assert single["photosynthesis"].shape == (1, 2)
    with pytest.raises(ValueError, match="exactly the calibration parameters"):
        example.layout.pack({k: v for k, v in parts.items() if k != "allocation"})
    with pytest.raises(ValueError, match="theta must be"):
        example.layout.unpack(theta[:, :5])
    mismatched = dict(parts)
    mismatched["allocation"] = parts["allocation"][0]
    with pytest.raises(ValueError, match="leading dimensions disagree"):
        example.layout.pack(mismatched)


# ── bijectors and the log prior ──────────────────────────────────────────────


def test_fields_flat_round_trip_in_both_spaces(example, theta):
    natural = example.fields(theta)
    assert natural.attrs == {"representation": "calibration_parameters", "space": "natural"}
    allocation = natural.filter_by_attrs(parameter="allocation")
    assert list(allocation.data_vars) == [f"allocation.{c}" for c in ALLOCATION.components]
    np.testing.assert_allclose(allocation.to_array().sum("variable"), 1.0, atol=1e-12)
    assert (natural["initial_soil_carbon"] > 0).all()
    np.testing.assert_allclose(example.flat(natural), theta, rtol=1e-10, atol=1e-10)
    unconstrained = example.fields(theta, space="unconstrained")
    np.testing.assert_array_equal(example.flat(unconstrained), theta)
    one = example.fields(theta[0])
    assert dict(one.sizes) == {"site": 3}
    np.testing.assert_allclose(example.flat(one), theta[0], rtol=1e-10, atol=1e-10)
    # A site reads its own class's copy through xarray's own selection.
    conifer = natural["allocation.leaf_allocation"].sel(site=27)
    np.testing.assert_array_equal(conifer, natural["allocation.leaf_allocation"].isel(site=1))
    with pytest.raises(ValueError, match="space must be one of"):
        example.fields(theta, space="physical")


def finite_difference_log_det(forward, theta_c: np.ndarray, size: int, h: float = 1e-6) -> float:
    """``0.5 * log det(J^T J)`` for the central-difference Jacobian ``J`` of
    *forward*, the volume element of the map.

    For a scalar or square bijector this is ``log |det J|``. For
    ``SoftmaxCentered`` the Jacobian is ``k x (k - 1)`` and this is the
    volume element of the embedded simplex, which is the measure TFP's
    ``forward_log_det_jacobian`` uses; it differs from the "first ``k - 1``
    components" convention by the constant ``0.5 * log k``.
    """
    columns = []
    for i in range(size):
        step = np.zeros(size)
        step[i] = h
        columns.append((forward(theta_c + step) - forward(theta_c - step)) / (2 * h))
    jacobian = np.stack(columns, axis=-1)
    return 0.5 * float(np.linalg.slogdet(jacobian.T @ jacobian)[1])


def test_log_prior_includes_the_jacobian_parameter_by_parameter(example, theta):
    """``prior.distribution.log_prob(theta_c)`` equals the natural-space log
    density plus the finite-difference log Jacobian, for every calibration parameter
    and group, at a sampled point."""
    parts = example.layout.unpack(theta[0])
    for parameter in example.parameters:
        n_groups = example.n_groups(parameter.varies_by)
        for g in range(n_groups):
            theta_c = np.asarray(parts[parameter.name][g])
            natural_prior, unconstrained_prior = parameter.prior, parameter.unconstrained_prior
            if tuple(natural_prior.batch_shape) == (n_groups,):  # one prior per group
                natural_prior, unconstrained_prior = natural_prior[g], unconstrained_prior[g]

            def forward(t, parameter=parameter):
                x = parameter.bijector.forward(jnp.asarray(t if parameter.size > 1 else t[0]))
                return np.atleast_1d(np.asarray(x))

            natural = forward(theta_c)
            log_det = finite_difference_log_det(forward, theta_c, parameter.size)
            natural_lp = float(natural_prior.log_prob(natural if parameter.size > 1 else natural[0]))
            unconstrained_lp = float(
                unconstrained_prior.log_prob(jnp.asarray(theta_c if parameter.size > 1 else theta_c[0]))
            )
            assert unconstrained_lp == pytest.approx(natural_lp + log_det, rel=1e-6, abs=1e-6), (
                parameter.name
            )


def test_log_prior_and_unpack_take_a_list_of_tracers_under_jit(example, theta):
    """One vector given as a list of JAX scalars is read as one vector under jit."""
    one = theta[0]
    as_list = lambda a: [a[i] for i in range(example.dimension)]  # noqa: E731
    expected = example.log_prior(one)
    np.testing.assert_allclose(jax.jit(lambda a: example.log_prior(as_list(a)))(one), expected)
    assert jax.jit(lambda a: example.log_prior(as_list(a)))(one).shape == ()
    unpacked = jax.jit(lambda a: example.layout.unpack(as_list(a)))(one)
    assert all(part.ndim == 2 for part in unpacked.values())


def test_fields_take_a_list_of_jax_scalars_as_one_vector(example, theta):
    fields = example.fields([theta[0, i] for i in range(example.dimension)])
    assert "member" not in fields.dims


def test_log_prior_is_the_sum_over_parameters_and_jit_compiles(example, theta):
    parts = example.layout.unpack(theta)
    expected = np.zeros(8)
    for parameter in example.parameters:
        part = parts[parameter.name]
        if parameter.is_scalar:
            part = part[..., 0]
        expected += np.asarray(
            example._broadcast_unconstrained(parameter).log_prob(part).sum(axis=-1)
        )
    np.testing.assert_allclose(example.log_prior(theta), expected, rtol=1e-12)
    assert example.log_prior(theta).shape == (8,)
    assert example.log_prior(theta[0]).shape == ()
    np.testing.assert_allclose(jax.jit(example.log_prior)(theta), expected, rtol=1e-12)
    grad = jax.grad(example.log_prior)(theta[0])
    assert grad.shape == (14,) and np.isfinite(grad).all()


def test_sample_is_reproducible_and_per_parameter(example):
    a = example.sample(jax.random.key(3), n=4)
    b = example.sample(jax.random.key(3), n=4)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 14) and a.dtype == jnp.float64
    # Per-site draws are independent, not one value broadcast over sites.
    site_block = a[:, example.layout.slice("initial_soil_carbon")]
    assert not np.allclose(site_block[:, 0], site_block[:, 1])
    # Calibration parameters get their own keys: standardized draws are neither
    # identical nor correlated across them.
    gaussian = example.gaussian_prior()
    draws = np.asarray(example.sample(jax.random.key(7), n=20_000))
    standardized = (draws - np.asarray(gaussian.mean)) / np.sqrt(np.diag(np.asarray(gaussian.cov.to_dense())))
    correlation = np.corrcoef(standardized, rowvar=False)
    off_block = np.abs(correlation - np.eye(14))
    assert off_block.max() < 0.03
    assert not np.allclose(standardized[:, 10], standardized[:, 8])  # leaf_fall vs base_soil_respiration


# ── the Gaussian prior ───────────────────────────────────────────────────────────


def test_gaussian_prior_matches_the_prior_moments(example):
    gaussian = example.gaussian_prior()
    assert gaussian.mean.shape == (14,)
    assert gaussian.cov.shape == (14, 14)
    assert gaussian.cov.block_shapes == ((2, 2), (6, 6), (2, 2), (1, 1), (3, 3))
    draws = np.asarray(example.sample(jax.random.key(4), n=200_000))
    np.testing.assert_allclose(gaussian.mean, draws.mean(axis=0), atol=0.02)
    np.testing.assert_allclose(np.diag(gaussian.cov.to_dense()), draws.var(axis=0), rtol=0.03)
    # Independent calibration parameters: no cross-block covariance.
    dense = np.asarray(gaussian.cov.to_dense())
    assert np.all(dense[:2, 2:] == 0) and np.all(dense[8:, :8] == 0)
    # Exact where the prior is Gaussian in theta.
    ln = example["base_soil_respiration"].prior
    np.testing.assert_allclose(gaussian.mean[8:10], float(ln.distribution.loc))
    np.testing.assert_allclose(dense[8, 8], float(ln.distribution.scale) ** 2)


def test_gaussian_prior_equals_the_analytic_unconstrained_moments(example):
    gaussian = example.gaussian_prior()
    dense = np.asarray(gaussian.cov.to_dense())
    for parameter in example.parameters:
        prior = example._broadcast_unconstrained(parameter)
        sl = example.layout.slice(parameter.name)
        np.testing.assert_allclose(gaussian.mean[sl], np.ravel(prior.mean()), rtol=1e-12)
        if parameter.is_scalar:
            np.testing.assert_allclose(np.diag(dense)[sl], np.ravel(prior.variance()), rtol=1e-12)
        else:
            expected = np.asarray(prior.covariance())
            block = dense[sl, sl]
            for g in range(expected.shape[0]):
                k = parameter.size
                np.testing.assert_allclose(block[g * k:(g + 1) * k, g * k:(g + 1) * k], expected[g], rtol=1e-12)


def test_gaussian_prior_refuses_a_zero_spread():
    flat = CalibrationParameter(
        name="z", prior=tfd.LogNormal(jnp.float64(0.0), jnp.float64(0.0)),
        sipnet_map="wood_turnover_rate", provenance="x",
    )
    with pytest.raises(ValueError, match="zero, negative or non-finite variance"):
        build((flat,)).gaussian_prior()


def test_gaussian_prior_needs_a_key_for_non_analytic_moments():
    beta = CalibrationParameter(
        name="b", prior=tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
        sipnet_map="leaf_off_fall_fraction", provenance="x",
    )
    assert not beta.has_analytic_moments
    p = build((beta,))
    with pytest.raises(NotImplementedError, match="no analytic unconstrained moments"):
        p.gaussian_prior()
    with pytest.raises(ValueError, match="at least 2"):
        p.gaussian_prior(key=jax.random.key(0), n_moment_samples=1)
    gaussian = p.gaussian_prior(key=jax.random.key(0), n_moment_samples=50_000)
    draws = np.asarray(p.sample(jax.random.key(1), n=50_000))
    assert float(gaussian.mean[0]) == pytest.approx(draws.mean(), abs=0.02)
    assert gaussian.mean.dtype == jnp.float64
    assert p.describe()["theta_moments"].iloc[0] == "monte_carlo"
    # Two Monte Carlo calibration parameters get different keys, so their estimates differ.
    two = build((
        beta,
        CalibrationParameter(
            name="c", prior=tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
            sipnet_map="leaf_on_reallocation_fraction", provenance="x",
        ),
    ))
    mean = two.gaussian_prior(key=jax.random.key(0), n_moment_samples=50).mean
    assert float(mean[0]) != float(mean[1])


# ── SIPNET parameter fields and pySIPNET ───────────────────────────────────


def test_sipnet_parameter_fields_shape_names_and_attributes(example, theta):
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    assert isinstance(sipnet_parameter_fields, xr.Dataset)
    assert dict(sipnet_parameter_fields.sizes) == {"sample": 8, "site": 3}
    assert set(sipnet_parameter_fields.data_vars) == {
        "max_photosynthesis_rate", "foliar_respiration_fraction", "leaf_allocation",
        "wood_allocation", "fine_root_allocation", "base_soil_respiration_rate",
        "leaf_off_fall_fraction", "soil_carbon", "daily_mean_photosynthesis_fraction",
        "leaf_carbon_fraction", "vapor_pressure_deficit_exponent",
    }
    assert sipnet_parameter_fields["soil_carbon"].attrs == {
        "units": "g m-2", "sipnet_name": "soilInit", "set_by": "parameter initial_soil_carbon",
        "constituent": "C",
    }
    assert sipnet_parameter_fields["leaf_carbon_fraction"].attrs["set_by"] == "fixed"
    assert sipnet_parameter_fields.attrs == {"representation": "sipnet_parameter_fields"}
    assert list(sipnet_parameter_fields["pft"].values) == list(PFT)
    assert sipnet_parameter_fields["site"].dtype == np.int32 and sipnet_parameter_fields["sample"].dtype == np.int64
    # Shared and per-PFT values broadcast onto sites; the two deciduous sites agree.
    a = sipnet_parameter_fields["leaf_allocation"]
    np.testing.assert_array_equal(a.sel(site=1), a.sel(site=4711))
    assert not np.allclose(a.sel(site=1), a.sel(site=27))
    single = example.sipnet_parameter_fields(theta[0])
    assert dict(single.sizes) == {"site": 3}


def test_sipnet_parameter_fields_values_come_from_the_right_parameter_and_group(example, theta):
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    fields = example.fields(theta)
    for sample in (0, 5):
        for site in SITES:
            row = sipnet_parameter_fields.isel(sample=sample).sel(site=site)
            cell = fields.isel(sample=sample).sel(site=site)
            for sipnet, variable in [
                ("soil_carbon", "initial_soil_carbon"),
                ("leaf_off_fall_fraction", "leaf_fall_fraction"),
                ("base_soil_respiration_rate", "base_soil_respiration"),
                ("leaf_allocation", "allocation.leaf_allocation"),
                ("wood_allocation", "allocation.wood_allocation"),
                ("fine_root_allocation", "allocation.fine_root_allocation"),
            ]:
                assert float(row[sipnet]) == float(cell[variable]), (sipnet, site)
            capacity = float(cell["photosynthesis.capacity"])
            share = float(cell["photosynthesis.respiration_share"])
            assert float(row["max_photosynthesis_rate"]) == pytest.approx(capacity * 0.466 * (1 - share) / 0.76)
    # A Fields input gives the same SIPNET parameter fields as the Flat one.
    xr.testing.assert_allclose(example.sipnet_parameter_fields(fields), sipnet_parameter_fields, rtol=1e-12)


def test_distinct_groups_are_gathered_onto_the_right_sites():
    p = distinct_groups()
    gaussian = p.gaussian_prior()
    # The Gaussian mean is each prior's base location, so constraining it gives
    # the medians (and the softmax centers) group by group, in group order.
    np.testing.assert_allclose(gaussian.mean[p.layout.slice("soil")], np.log([100.0, 200.0, 300.0]))
    np.testing.assert_allclose(gaussian.mean[p.layout.slice("turnover")], np.log([0.01, 0.02]))
    dense = np.asarray(gaussian.cov.to_dense())
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("soil")], np.log([1.5, 2.0, 2.5]) ** 2)
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("turnover")], np.log([1.2, 1.3]) ** 2)
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("allocation")], np.tile([0.3, 0.6, 0.9], 2) ** 2)
    sipnet_parameter_fields = p.sipnet_parameter_fields(gaussian.mean)
    np.testing.assert_allclose(sipnet_parameter_fields["soil_carbon"].values, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(sipnet_parameter_fields["wood_turnover_rate"].values, [0.01, 0.02, 0.01])
    np.testing.assert_allclose(sipnet_parameter_fields["leaf_carbon_fraction"].values, [0.4, 0.5, 0.4])
    np.testing.assert_allclose(sipnet_parameter_fields["leaf_allocation"].values, [0.1, 0.4, 0.1], rtol=1e-12)
    np.testing.assert_allclose(sipnet_parameter_fields["fine_root_allocation"].values, [0.3, 0.2, 0.3], rtol=1e-12)
    frame = p.describe()
    soil = frame[frame["parameter"] == "soil"]
    np.testing.assert_allclose(soil["natural_median"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(soil["theta_sd"], np.log([1.5, 2.0, 2.5]))
    assert list(soil["group"]) == list(SITES)


def test_prior_draws_land_in_every_domain(example):
    sipnet_parameter_fields = example.sipnet_parameter_fields(example.sample(jax.random.key(5), n=2000))
    for name, values in sipnet_parameter_fields.data_vars.items():
        assert in_domain(FLAT_SPECS[name].domain, values.values), name
    triangle = sipnet_parameter_fields["leaf_allocation"] + sipnet_parameter_fields["wood_allocation"] + sipnet_parameter_fields["fine_root_allocation"]
    assert float(triangle.max()) < 1.0


def test_sipnet_overrides_gives_one_run_of_floats(example, theta):
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    kwargs = sipnet_overrides(sipnet_parameter_fields, batch={"sample": 2}, site=27)
    assert set(kwargs) == set(sipnet_parameter_fields.data_vars)
    assert all(type(v) is float for v in kwargs.values())
    assert kwargs["soil_carbon"] == float(sipnet_parameter_fields["soil_carbon"].isel(sample=2).sel(site=27))
    with pytest.raises(ValueError, match=r"have the batch dims \['sample'\] and batch= names \[\]"):
        sipnet_overrides(sipnet_parameter_fields, site=27)
    with pytest.raises(KeyError, match="not one of the SIPNET parameter fields' sample labels"):
        sipnet_overrides(sipnet_parameter_fields, batch={"sample": -1}, site=27)
    single = example.sipnet_parameter_fields(theta[0])
    assert sipnet_overrides(single, site=1)["leaf_carbon_fraction"] == 0.466
    with pytest.raises(ValueError, match=r"have the batch dims \[\] and batch= names \['sample'\]"):
        sipnet_overrides(single, batch={"sample": 0}, site=1)


def with_overrides(base: SIPNETParameters, overrides: dict[str, float]) -> SIPNETParameters:
    dump = base.model_dump()
    group_of = {path.split(".", 1)[1]: path.split(".", 1)[0] for path in PARAMETER_SPECS}
    for name, value in overrides.items():
        dump[group_of[name]][name] = value
    return SIPNETParameters.model_validate(dump)


def test_sipnet_parameter_fields_validates_through_sipnet_parameters(example, theta):
    base = niwot_parameters()
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    for sample in range(8):
        for site in SITES:
            params = with_overrides(base, sipnet_overrides(sipnet_parameter_fields, batch={"sample": sample}, site=site))
            assert params.photosynthesis.max_photosynthesis_rate > 0
            assert params.initial_conditions.soil_carbon == pytest.approx(
                float(sipnet_parameter_fields["soil_carbon"].isel(sample=sample).sel(site=site))
            )


@pytest.mark.slow
def test_a_prior_draw_runs_the_niwot_fixture(example, theta):
    from pysipnet import SIPNETModel, SIPNETRunner, niwot_reference_climate
    from pysipnet.parameters.model import ModelFlags

    if find_binary() is None:
        pytest.skip(missing_binary_message())
    runner = SIPNETRunner(flags=ModelFlags.standard())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fixture has a few vpd <= 0 rows
        climate = niwot_reference_climate().head(8 * 30)
    model = SIPNETModel(runner, base_params=niwot_parameters(), base_climate=climate)

    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    result = model(**sipnet_overrides(sipnet_parameter_fields, batch={"sample": 0}, site=27))
    assert result.provenance.success, result.provenance.stderr
    nee = result.outputs["net_ecosystem_exchange"]
    assert nee.sizes["time"] == 8 * 30 and bool(np.isfinite(nee.values).all())


# ── Fields ────────────────────────────────────────────────────────────


def test_fields_data_model(example, theta):
    natural = example.fields(theta)
    assert list(natural.data_vars) == [
        "photosynthesis.capacity", "photosynthesis.respiration_share",
        *(f"allocation.{c}" for c in ALLOCATION.components),
        "base_soil_respiration", "leaf_fall_fraction", "initial_soil_carbon",
    ]
    for variable in natural.data_vars.values():
        assert variable.dims == ("sample", "site") and variable.dtype == np.float64
    assert natural["site"].dtype == np.int32 and natural["sample"].dtype == np.int64
    assert list(natural["pft"].values) == list(PFT)
    assert "lon" not in natural.coords  # built from plain ids, so no positions
    assert natural["allocation.coarse_root_allocation"].attrs == {
        "parameter": "allocation", "component": "coarse_root_allocation", "varies_by": "pft",
        "space": "natural", "long_name": "allocation: coarse_root_allocation", "units": "1",
        "sipnet_parameters": "leaf_allocation, wood_allocation, fine_root_allocation",
    }
    soil = natural["initial_soil_carbon"].attrs
    assert soil["units"] == "g m-2" and soil["varies_by"] == "site" and "component" not in soil
    assert natural["photosynthesis.capacity"].attrs["units"] == "nmol g-1 s-1"
    assert natural["leaf_fall_fraction"].attrs["varies_by"] == "shared"
    unconstrained = example.fields(theta, space="unconstrained")
    assert list(unconstrained.data_vars)[:2] == ["photosynthesis.log(capacity)", "photosynthesis.logit(respiration_share)"]
    assert "allocation.alr(leaf_allocation:coarse_root_allocation)" in unconstrained
    assert unconstrained["initial_soil_carbon"].attrs["units"] == "1"
    np.testing.assert_array_equal(
        unconstrained.filter_by_attrs(parameter="allocation").sel(site=1).to_array().transpose("sample", "variable"),
        theta[:, 5:8],  # site 1 is deciduous, the second PFT group
    )
    # One copy per class, recovered with xarray's groupby.
    per_pft = natural["allocation.leaf_allocation"].groupby("pft").first()
    assert per_pft.dims == ("sample", "pft")


@pytest.mark.parametrize(
    "classes",
    [[1, 2, 1], np.array([1, 2, 1]), pd.Categorical([1, 2, 1])],
    ids=["list", "numpy", "categorical"],
)
def test_select_by_site_labels_takes_integer_classes(classes):
    vector = example_parameter_vector(sites=[1, 27, 4000], pft=classes)
    assert vector.select(labels={"pft": [1]}).sites == (1, 4000)
    assert vector.select(labels={"pft": np.array([2])}).sites == (27,)
    with pytest.raises(TypeError, match="one value 1"):
        vector.select(labels={"pft": np.int64(1)})


@pytest.mark.parametrize("site_id", [0, -3, 2**31], ids=["zero", "negative", "past int32"])
def test_a_site_table_whose_site_id_is_not_a_site_id_is_refused(site_id):
    table = site_table_of(1, 27)
    # Kept ascending, so the refusal is of the id itself, not of the order.
    table["site_id"] = np.array([site_id, 27] if site_id < 1 else [1, site_id], dtype=np.int64)
    with pytest.raises(ValueError, match=r"site_id\[\d\] must be a site id from 1 to 2147483647"):
        example_parameter_vector(sites=table, pft=PFT[:2])


def test_fields_carry_lon_lat_from_a_site_table():
    table = site_table_of(1, 27, 4711, lon=[-24.6, -78.6, -107.3], lat=[82.5, 80.6, 44.0])
    vector = example_parameter_vector(sites=table, pft=PFT)
    fields = vector.fields(vector.sample(jax.random.key(0), n=2))
    np.testing.assert_allclose(fields["lon"], [-24.6, -78.6, -107.3])
    assert fields["lat"].attrs == dict(LAT_ATTRIBUTES)
    assert fields["lon"].attrs == dict(LON_ATTRIBUTES)
    assert list(vector.site_table.columns) == ["site_id", "lon", "lat", "pft"]
    assert list(vector.site_table["pft"].cat.categories) == ["conifer", "deciduous"]
    np.testing.assert_allclose(vector.sipnet_parameter_fields(vector.sample(jax.random.key(1), n=2))["lat"], [82.5, 80.6, 44.0])
    small = vector.select(sites=(27, 4711))
    np.testing.assert_allclose(small.site_table["lon"], [-78.6, -107.3])
    np.testing.assert_allclose(small.site_table["lat"], [80.6, 44.0])
    # A table out of site order would misalign positional inputs, so it is refused.
    with pytest.raises(ValueError, match="ascending site_id order"):
        example_parameter_vector(sites=table.iloc[[2, 0, 1]], pft=PFT)
    with pytest.raises(ValueError, match=r"no \['lat'\] column"):
        example_parameter_vector(sites=table.drop(columns="lat"), pft=PFT)
    with pytest.raises(ValueError, match="non-finite lon/lat"):
        example_parameter_vector(sites=table.assign(lon=[np.nan, 0.0, 0.0]), pft=PFT)
    with pytest.raises(ValueError, match="no 'site_id' column or index"):
        example_parameter_vector(sites=table.drop(columns="site_id"), pft=PFT)
    with pytest.raises(TypeError, match="site_id must hold integers"):
        example_parameter_vector(sites=table.astype({"site_id": float}), pft=PFT)
    with pytest.raises(ValueError, match="more than once"):
        example_parameter_vector(sites=table.iloc[[0, 0, 1]], pft=PFT)
    keyed = example_parameter_vector(sites=table.set_index("site_id"), pft=PFT)
    assert keyed.sites == SITES
    only_ids = example_parameter_vector(sites=table[["site_id"]], pft=PFT)
    assert "lon" not in only_ids.site_table.columns


def test_flat_refuses_what_no_flat_vector_can_be(example, theta):
    fields = example.fields(theta)
    no_space = fields.copy()
    no_space.attrs = {}
    with pytest.raises(ValueError, match="attrs\\['space'\\]"):
        example.flat(no_space)
    with pytest.raises(ValueError, match="lack variables \\['initial_soil_carbon'\\]"):
        example.flat(fields.drop_vars("initial_soil_carbon"))
    with pytest.raises(ValueError, match="lack sites \\[4711\\]"):
        example.flat(fields.sel(site=[1, 27]))
    broken = fields.copy(deep=True)
    broken["leaf_fall_fraction"][0, 0] = np.nan
    with pytest.raises(ValueError, match="not all finite"):
        example.flat(broken)
    # Sites 1 and 4711 share the deciduous copy; making them disagree is refused.
    disagree = fields.copy(deep=True)
    disagree["base_soil_respiration"].loc[{"site": 4711}] = 0.5
    with pytest.raises(ValueError, match="differ between the sites of group 'deciduous' \\(\\[1, 4711\\]\\)"):
        example.flat(disagree)
    with pytest.raises(ValueError, match="must be on"):
        example.flat(fields.assign(initial_soil_carbon=fields["initial_soil_carbon"].isel(sample=0, drop=True)))
    # Extra variables and sites are ignored.
    extra = fields.assign(unrelated=fields["initial_soil_carbon"] * 2)
    np.testing.assert_allclose(example.flat(extra), theta, rtol=1e-10, atol=1e-10)


def test_unset_parameters_and_require_complete(example):
    from sipnet_calibration.parameter_vector import REQUIRED_SIPNET_PARAMETERS

    # Required means "pySIPNET has no default": flag-dependent parameters and
    # the zero-defaulted ones are not required.
    assert "snow_melt_rate" not in REQUIRED_SIPNET_PARAMETERS
    assert "litter_carbon" not in REQUIRED_SIPNET_PARAMETERS
    assert "max_photosynthesis_rate" in REQUIRED_SIPNET_PARAMETERS
    assert set(example.sipnet_parameter_names) == set(example.sipnet_parameter_fields(example.sample(jax.random.key(0), 1)).data_vars)
    assert set(example.unset_sipnet_parameter_names) == set(REQUIRED_SIPNET_PARAMETERS) - set(example.sipnet_parameter_names)
    assert "leaf_carbon_per_area" in example.unset_sipnet_parameter_names
    assert not set(example.unset_sipnet_parameter_names) & set(example.sipnet_parameter_names)
    with pytest.raises(ValueError, match="neither calibrated nor fixed: \\['total_wood_carbon'"):
        ParameterVector(
            parameters=example.parameters, fixed=example.fixed, sites=example.sites,
            site_labels=example.site_labels, require_complete=True,
        )
    # A vector that fixes everything it does not calibrate is complete.
    filled = tuple(
        FixedParameter(name=name, value=_in_domain_value(name), provenance="test")
        for name in example.unset_sipnet_parameter_names
    )
    complete = ParameterVector(
        parameters=example.parameters, fixed=example.fixed + filled, sites=example.sites,
        site_labels=example.site_labels, require_complete=True,
    )
    assert complete.unset_sipnet_parameter_names == ()


def _in_domain_value(name: str) -> float:
    return {
        ParameterDomain.REAL: 1.0, ParameterDomain.POSITIVE: 1.0, ParameterDomain.NON_NEGATIVE: 1.0,
        ParameterDomain.UNIT_INTERVAL: 0.5, ParameterDomain.OPEN_UNIT_INTERVAL: 0.5,
    }[FLAT_SPECS[name].domain]


def test_sites_with_and_parameter_lookup(example):
    assert example.sites_with("pft", "deciduous") == (1, 4711)
    with pytest.raises(KeyError, match="'grassland' is not a class of site labels 'pft'"):
        example.sites_with("pft", "grassland")
    assert example["allocation"].sipnet_map is ALLOCATION
    assert example.parameter_names == example.layout.parameters
    with pytest.raises(KeyError):
        example.sites_with("landcover", 1)
    with pytest.raises(KeyError, match="no calibration parameter 'nope'"):
        example["nope"]


def test_describe_has_one_row_per_column(example):
    frame = example.describe()
    assert len(frame) == 14
    assert list(frame["parameter"]) == [example.layout.column_labels[i].split("[")[0] for i in range(14)]
    assert (frame["provenance"].str.len() > 0).all()
    assert set(frame["distribution"]) == {
        "product of transformed Gaussians", "softmax-normal", "log-normal", "logit-normal",
    }
    assert (frame["theta_moments"] == "analytic").all()
    soil = frame[frame["parameter"] == "base_soil_respiration"].iloc[0]
    assert soil["natural_2.5"] == pytest.approx(0.004) and soil["natural_97.5"] == pytest.approx(0.020)
    ln = example["base_soil_respiration"].prior.distribution
    assert soil["theta_mean"] == pytest.approx(float(ln.loc)) and soil["theta_sd"] == pytest.approx(float(ln.scale))
    assert soil["sipnet_parameters"] == "base_soil_respiration_rate"
    assert np.isnan(frame[frame["parameter"] == "allocation"]["natural_median"]).all()
    assert frame[frame["parameter"] == "initial_soil_carbon"]["group"].tolist() == list(SITES)


# ── the summary and the module's usage session ───────────────────────────────


def test_repr_summarizes_parameters_groups_and_what_is_fixed(example):
    text = repr(example)
    lines = text.splitlines()
    assert lines[0] == "ParameterVector  D = 14  |  3 sites  |  site labels: pft {conifer, deciduous}"
    assert lines[1].split()[:5] == ["parameter", "varies", "by", "groups", "size"]
    rows = {line.split()[0]: line.split() for line in lines[2:7]}
    assert list(rows) == list(example.parameter_names)
    assert rows["allocation"][1:4] == ["pft", "2", "3"]
    assert rows["initial_soil_carbon"][1:4] == ["site", "3", "1"]
    assert lines[7].startswith("  fixed: daily_mean_photosynthesis_fraction (shared)")
    assert lines[8] == (
        f"  unset: {len(example.unset_sipnet_parameter_names)} required SIPNET parameters, taken "
        "from the run's base parameter set"
    )


def _usage_session() -> str:
    block = module.__doc__.split("Usage\n-----\n", 1)[1]
    return "\n".join(line[4:] for line in block.splitlines() if line.startswith("    "))


def test_the_module_usage_session_runs_and_prints_what_it_says():
    """The module docstring's worked session, executed against the real site
    table and site labels, with its printed summary compared line for line."""
    from sipnet_calibration.site_labels import site_labels_path
    from sipnet_calibration.sites import default_sites_path

    if not (default_sites_path().exists() and site_labels_path("reanalysis_3pft").exists()):
        pytest.skip("processed site table or site labels not available in this working copy")
    namespace: dict = {}

    class _EKIResult:  # stands in for a pyeki EKIResult
        @property
        def state(self):
            return type("State", (), {"ensemble": namespace["theta"]})

    namespace["eki"] = _EKIResult()
    code = _usage_session()
    exec(compile(code, "parameter_vector usage", "exec"), namespace)
    vector = namespace["vector"]
    after = code.split("print(vector)\n", 1)[1].splitlines()
    printed = [line[2:] for line in itertools.takewhile(lambda line: line.startswith("# "), after)]
    assert printed == repr(vector).splitlines()
    assert vector.dimension == 13
    assert namespace["conifer"].dimension == 8 and namespace["two"].dimension == 9
    assert namespace["spec"].n_runs == 150
    np.testing.assert_allclose(namespace["theta_again"], namespace["theta"], atol=1e-10)


# ── select ───────────────────────────────────────────────────────────────────


def test_select_by_parameters_keeps_layout_order_and_the_fixed(example):
    small = example.select(parameters=("initial_soil_carbon", "allocation"))
    assert small.parameter_names == ("allocation", "initial_soil_carbon")
    assert small.dimension == 3 * 2 + 3
    assert [f.name for f in small.fixed] == [f.name for f in example.fixed]
    assert small.select(parameters=["allocation"]).parameter_names == ("allocation",)
    with pytest.raises(TypeError, match="one string 'allocation'"):
        small.select(parameters="allocation")
    with pytest.raises(KeyError, match="no calibration parameters \\['nope'\\]"):
        example.select(parameters=("nope",))


def test_select_by_sites_slices_per_site_priors_and_moves_draws_across():
    vector = distinct_groups()
    theta = vector.sample(jax.random.key(0), n=6)
    small = vector.select(sites=(1, 4711))  # both "a": one PFT group remains
    assert small.sites == (1, 4711)
    assert small.group_labels("pft") == ("a",)
    assert small.dimension == 2 + 1 + 3
    np.testing.assert_allclose(
        small.gaussian_prior().mean[small.layout.slice("soil")], np.log([100.0, 300.0])
    )
    np.testing.assert_allclose(small.gaussian_prior().mean[small.layout.slice("turnover")], np.log([0.01]))
    assert small.fixed[0].value == {"a": 0.4}
    projected = small.flat(vector.fields(theta))
    np.testing.assert_array_equal(projected[:, small.layout.slice("soil")], theta[:, vector.layout.index("soil", group=1).tolist() + vector.layout.index("soil", group=4711).tolist()])
    np.testing.assert_allclose(
        small.fields(projected)["turnover"], vector.fields(theta)["turnover"].sel(site=[1, 4711])
    )


def test_select_by_labels_intersects_with_sites_and_refuses_the_unknown(example):
    deciduous = example.select(labels={"pft": ["deciduous"]})
    assert deciduous.sites == (1, 4711)
    assert deciduous.group_labels("pft") == ("deciduous",)
    assert example.select(labels={"pft": ["conifer", "deciduous"]}).sites == example.sites
    assert example.select(sites=(1, 27), labels={"pft": ["deciduous"]}).sites == (1,)
    with pytest.raises(TypeError, match="one string 'deciduous'"):
        example.select(labels={"pft": "deciduous"})
    with pytest.raises(KeyError, match="not classes of site labels 'pft'"):
        example.select(labels={"pft": ["grassland"]})
    with pytest.raises(KeyError, match="no site labels 'landcover'"):
        example.select(labels={"landcover": [1]})
    with pytest.raises(KeyError, match="sites \\[99\\] are not in this vector"):
        example.select(sites=(1, 99))
    with pytest.raises(ValueError, match="no site of this vector"):
        example.select(sites=(27,), labels={"pft": ["deciduous"]})


# ── site-labels products ─────────────────────────────────────────────────────


def _product(labels_by_site: dict[int, str], classes: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": np.asarray(list(labels_by_site), dtype=np.int32),
            "label": pd.Categorical(list(labels_by_site.values()), categories=classes),
        }
    )


def test_a_site_labels_product_gives_its_declared_classes_in_order():
    classes = ("needleleaf", "broadleaf", "grass")
    product = _product({1: "grass", 27: "broadleaf", 4711: "grass", 5000: "needleleaf"}, classes)
    turnover = CalibrationParameter(
        name="turnover", prior=log_normal(median=[0.01, 0.02, 0.03], geometric_sd=[1.2, 1.3, 1.4]),
        sipnet_map="wood_turnover_rate", varies_by="pft", provenance="test",
    )
    fixed = FixedParameter(
        name="leaf_carbon_fraction", varies_by="pft", provenance="test",
        value={"needleleaf": 0.50, "broadleaf": 0.46, "grass": 0.48},
    )
    vector = ParameterVector(
        parameters=(turnover,), fixed=(fixed,), sites=(1, 27, 4711), site_labels={"pft": product}
    )
    # Declared order, restricted to the classes present: needleleaf is dropped.
    assert vector.group_labels("pft") == ("broadleaf", "grass")
    np.testing.assert_allclose(vector.gaussian_prior().mean, np.log([0.02, 0.03]))
    assert vector.fixed[0].value == {"broadleaf": 0.46, "grass": 0.48}
    sipnet_parameter_fields = vector.sipnet_parameter_fields(vector.gaussian_prior().mean)
    np.testing.assert_allclose(sipnet_parameter_fields["leaf_carbon_fraction"], [0.48, 0.46, 0.48])
    np.testing.assert_allclose(sipnet_parameter_fields["wood_turnover_rate"], [0.03, 0.02, 0.03])
    with pytest.raises(ValueError, match="extra \\[.*'shrub'\\]"):
        ParameterVector(
            parameters=(turnover,),
            fixed=(dataclasses.replace(fixed, value={**fixed.value, "shrub": 0.4}),),
            sites=(1, 27, 4711), site_labels={"pft": product},
        )
    with pytest.raises(ValueError, match="do not label sites \\[99\\]"):
        ParameterVector(parameters=(turnover,), sites=(1, 99), site_labels={"pft": product})
    with pytest.raises(ValueError, match="needs 'site_id' and 'label' columns"):
        ParameterVector(
            parameters=(turnover,), sites=(1,), site_labels={"pft": product.rename(columns={"label": "pft"})}
        )


# ── joint priors over groups ─────────────────────────────────────────────────


COVARIANCE = jnp.asarray([[0.30, 0.10, 0.02], [0.10, 0.25, 0.08], [0.02, 0.08, 0.40]])
MEAN = jnp.log(jnp.asarray([100.0, 200.0, 300.0]))


def joint_soil(covariance=COVARIANCE, mean=MEAN) -> CalibrationParameter:
    base = tfd.MultivariateNormalTriL(loc=mean, scale_tril=jnp.linalg.cholesky(covariance))
    return CalibrationParameter(
        name="soil", prior=tfd.TransformedDistribution(base, tfb.Exp()),
        sipnet_map="soil_carbon", varies_by="site", provenance="test",
    )


def joint_vector() -> ParameterVector:
    return ParameterVector(parameters=(joint_soil(), rate(varies_by="pft")), sites=SITES, site_labels={"pft": PFT})


def test_a_joint_prior_is_read_off_its_event_shape():
    soil = joint_soil()
    assert soil.is_joint and soil.joint_groups == 3 and soil.size == 1
    assert not rate().is_joint
    assert "joint over groups" in repr(joint_vector())
    with pytest.raises(ValueError, match="joint prior is over 3"):
        ParameterVector(parameters=(joint_soil(),), sites=(1, 27))
    with pytest.raises(ValueError, match="needs an elementwise bijector"):
        CalibrationParameter(
            name="s", prior=softmax_normal(center=[0.5, 0.5], logit_sd=1.0),
            sipnet_map="soil_carbon", provenance="x",
        )
    two_by_four = tfd.TransformedDistribution(
        tfd.Independent(tfd.Normal(jnp.zeros((2, 3)), jnp.float64(1.0)), 2), tfb.SoftmaxCentered()
    )
    with pytest.raises(ValueError, match="vector-valued calibration parameter, which is not supported"):
        CalibrationParameter(name="a", prior=two_by_four, sipnet_map=ALLOCATION, provenance="x")


def test_a_joint_prior_samples_and_scores_as_one_distribution():
    vector = joint_vector()
    theta = vector.sample(jax.random.key(0), n=40_000)
    block = np.asarray(theta[:, vector.layout.slice("soil")])
    np.testing.assert_allclose(np.cov(block, rowvar=False), COVARIANCE, atol=0.01)
    one = theta[:4]
    natural = jnp.exp(one[:, vector.layout.slice("soil")])
    expected = joint_soil().prior.log_prob(natural) + one[:, vector.layout.slice("soil")].sum(axis=-1)
    expected = expected + vector._broadcast_unconstrained(vector["r"]).log_prob(one[:, vector.layout.slice("r")]).sum(axis=-1)
    np.testing.assert_allclose(vector.log_prior(one), expected, rtol=1e-10)
    np.testing.assert_allclose(jax.jit(vector.log_prior)(one), expected, rtol=1e-10)


def test_a_joint_prior_is_one_dense_block_of_the_gaussian_and_converts_like_any_other():
    vector = joint_vector()
    gaussian = vector.gaussian_prior()
    dense = np.asarray(gaussian.cov.to_dense())
    np.testing.assert_allclose(dense[:3, :3], COVARIANCE, rtol=1e-12)
    np.testing.assert_allclose(gaussian.mean[:3], MEAN, rtol=1e-12)
    assert np.all(dense[:3, 3:] == 0)
    theta = vector.sample(jax.random.key(1), n=5)
    fields = vector.fields(theta)
    np.testing.assert_allclose(fields["soil"], np.exp(theta[:, :3]), rtol=1e-12)
    np.testing.assert_allclose(vector.flat(fields), theta, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(vector.sipnet_parameter_fields(theta)["soil_carbon"], np.exp(theta[:, :3]), rtol=1e-12)
    frame = vector.describe()
    soil = frame[frame["parameter"] == "soil"]
    np.testing.assert_allclose(soil["theta_sd"], np.sqrt(np.diag(COVARIANCE)))
    assert soil["natural_median"].isna().all()


def test_select_takes_the_exact_marginal_of_a_joint_prior():
    small = joint_vector().select(sites=(1, 4711))
    gaussian = small.gaussian_prior()
    np.testing.assert_allclose(gaussian.mean[:2], np.asarray(MEAN)[[0, 2]], rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(gaussian.cov.to_dense())[:2, :2], np.asarray(COVARIANCE)[np.ix_([0, 2], [0, 2])], rtol=1e-12
    )


# ── SIPNET map units and reserved names ──────────────────────────────────────


def test_sipnet_maps_declare_component_units():
    assert Identity("soil_carbon").component_units == ("g m-2",)
    assert ALLOCATION.component_units == ("1",) * 4
    assert PHOTOSYNTHESIS.component_units == (FLAT_SPECS["max_photosynthesis_rate"].units, "1")
    with pytest.raises(ValueError, match="is reserved"):
        CalibrationParameter(
            name="lat", prior=log_normal(median=1.0, geometric_sd=2.0),
            sipnet_map="wood_turnover_rate", provenance="x",
        )


# ── regressions found in review ──────────────────────────────────────────────


def test_a_per_class_simplex_prior_is_restricted_to_the_classes_present():
    """TFP's own slicing of a batched softmax-normal fails for two or more
    kept indices; the restriction rebuilds it instead."""
    centers = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]]
    allocation = CalibrationParameter(
        name="allocation", prior=softmax_normal(center=centers, logit_sd=[0.3, 0.6, 0.9]),
        sipnet_map=ALLOCATION, varies_by="pft", provenance="test",
    )
    classes = ("a", "b", "c", "d")
    # Two of four declared classes present, not adjacent: a and c.
    product = _product({1: "a", 27: "c", 4711: "a"}, classes)
    four = CalibrationParameter(
        name="allocation",
        prior=softmax_normal(center=[*centers, [0.7, 0.1, 0.1, 0.1]], logit_sd=0.5),
        sipnet_map=ALLOCATION, varies_by="pft", provenance="test",
    )
    vector = ParameterVector(parameters=(four,), sites=SITES, site_labels={"pft": product})
    assert vector.group_labels("pft") == ("a", "c")
    natural = vector.fields(vector.gaussian_prior().mean)
    np.testing.assert_allclose(natural["allocation.leaf_allocation"], [0.1, 0.25, 0.1], rtol=1e-12)
    # The same through select: three per-site copies, two kept.
    per_site = dataclasses.replace(allocation, varies_by="site")
    small = ParameterVector(parameters=(per_site,), sites=SITES).select(sites=(1, 4711))
    np.testing.assert_allclose(
        small.fields(small.gaussian_prior().mean)["allocation.coarse_root_allocation"], [0.4, 0.25], rtol=1e-12
    )
    np.testing.assert_allclose(np.diag(small.gaussian_prior().cov.to_dense()), np.tile([0.3, 0.6, 0.9], 2) ** 2)


def test_a_scalar_bijector_with_a_parameter_per_group_lines_up_with_its_groups():
    medians = jnp.log(jnp.asarray([0.01, 0.1, 1.0]))
    prior = tfd.TransformedDistribution(
        tfd.Normal(jnp.zeros(3), jnp.full(3, 0.1)), tfb.Chain([tfb.Exp(), tfb.Shift(medians)])
    )
    soil = CalibrationParameter(name="turnover", prior=prior, sipnet_map="wood_turnover_rate", varies_by="site", provenance="test")
    vector = ParameterVector(parameters=(soil,), sites=SITES)
    at_median = vector.fields(jnp.zeros(3))
    np.testing.assert_allclose(at_median["turnover"], [0.01, 0.1, 1.0], rtol=1e-12)
    theta = vector.sample(jax.random.key(0), n=4)
    np.testing.assert_allclose(vector.flat(vector.fields(theta)), theta, rtol=1e-10, atol=1e-10)
    # Restricting it either slices the bijector correctly or refuses; never
    # maps a kept site through another site's shift.
    try:
        small = vector.select(sites=(4711,))
    except ValueError as error:
        assert "cannot be restricted" in str(error)
    else:
        np.testing.assert_allclose(small.fields(jnp.zeros(1))["turnover"], [1.0], rtol=1e-12)


def test_a_joint_prior_with_a_per_group_bijector_is_not_restricted_silently():
    base = tfd.MultivariateNormalTriL(loc=jnp.zeros(3), scale_tril=jnp.linalg.cholesky(COVARIANCE))
    shifted = tfd.TransformedDistribution(base, tfb.Chain([tfb.Exp(), tfb.Shift(jnp.log(jnp.asarray([0.01, 0.1, 1.0])))]))
    soil = CalibrationParameter(name="soil", prior=shifted, sipnet_map="soil_carbon", varies_by="site", provenance="test")
    vector = ParameterVector(parameters=(soil,), sites=SITES)
    np.testing.assert_allclose(vector.fields(jnp.zeros(3))["soil"], [0.01, 0.1, 1.0], rtol=1e-12)
    for kept in ((4711,), (1, 4711)):
        with pytest.raises(ValueError, match="cannot be restricted"):
            vector.select(sites=kept)


def test_tfp_distributions_that_subclass_transformed_distribution_use_their_default_bijector():
    raw = tfd.MultivariateNormalTriL(loc=jnp.asarray([20.0, 22.0, 24.0]), scale_tril=jnp.linalg.cholesky(COVARIANCE))
    temperature = CalibrationParameter(
        name="optimum", prior=raw, sipnet_map="optimum_photosynthesis_temperature", varies_by="site", provenance="test",
    )
    assert isinstance(temperature.bijector, tfb.Identity) and temperature.unconstrained_prior is raw
    assert temperature.has_analytic_moments
    vector = ParameterVector(parameters=(temperature,), sites=SITES)
    np.testing.assert_allclose(vector.fields(jnp.full(3, 5.0))["optimum"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(np.asarray(vector.gaussian_prior().cov.to_dense()), COVARIANCE, rtol=1e-12)
    np.testing.assert_allclose(vector.select(sites=(1, 4711)).gaussian_prior().mean, [20.0, 24.0])
    weibull = CalibrationParameter(
        name="w", prior=tfd.Weibull(jnp.float64(2.0), jnp.float64(0.01)),
        sipnet_map="wood_turnover_rate", provenance="test",
    )
    assert ParameterVector(parameters=(weibull,), sites=(1,)).dimension == 1


def test_flat_refuses_values_outside_the_image_of_the_bijector(example, theta):
    fields = example.fields(theta)
    for variable, value in [
        ("base_soil_respiration", -0.01),
        ("base_soil_respiration", 0.0),
        ("leaf_fall_fraction", 1.5),
        ("allocation.leaf_allocation", 0.9),  # the four no longer sum to 1
    ]:
        broken = fields.copy(deep=True)
        broken[variable][:] = value
        with pytest.raises(ValueError, match="outside the image of its bijector"):
            example.flat(broken)
        with pytest.raises(ValueError, match="outside the image of its bijector"):
            example.sipnet_parameter_fields(broken)


def test_flat_refuses_a_variable_from_the_other_space(example, theta):
    natural = example.fields(theta)
    unconstrained = example.fields(theta, space="unconstrained")
    mixed = natural.assign(initial_soil_carbon=unconstrained["initial_soil_carbon"])
    with pytest.raises(ValueError, match="not in natural space"):
        example.flat(mixed)


def test_flat_takes_fields_selected_to_one_site(example, theta):
    one_site = example.select(sites=(27,))
    np.testing.assert_allclose(
        one_site.flat(example.fields(theta).sel(site=27)),
        one_site.flat(example.fields(theta)), rtol=1e-12,
    )


def test_a_sipnet_map_must_return_exactly_what_it_writes():
    @dataclasses.dataclass(frozen=True)
    class Lying:
        writes: tuple = ("soil_carbon", "base_soil_respiration_rate")
        reads: tuple = ()
        components: tuple = ("soil_carbon",)
        component_units: tuple = ("g m-2",)

        def __call__(self, natural, fixed):
            return {"soil_carbon": natural[..., 0]}

    liar = CalibrationParameter(name="soil", prior=log_normal(median=1.0, geometric_sd=2.0), sipnet_map=Lying(), provenance="x")
    with pytest.raises(ValueError, match="declares writes .* but returns"):
        ParameterVector(parameters=(liar,), sites=(1,))


def test_site_labels_with_missing_or_unorderable_classes_are_refused():
    product = pd.DataFrame({"site_id": [1, 27, 4711], "label": ["a", None, "b"]})
    with pytest.raises(ValueError, match="give no class for 1"):
        ParameterVector(parameters=(rate(varies_by="pft"),), sites=SITES, site_labels={"pft": product})
    # A missing class at a site outside the vector does not matter.
    vector = ParameterVector(parameters=(rate(varies_by="pft"),), sites=(1, 4711), site_labels={"pft": product})
    assert vector.group_labels("pft") == ("a", "b")
    with pytest.raises(ValueError, match="cannot be ordered"):
        ParameterVector(parameters=(rate(varies_by="pft"),), sites=SITES, site_labels={"pft": ("a", 1, "a")})
    with pytest.raises(ValueError, match="not lower_case_with_underscores"):
        ParameterVector(parameters=(rate(),), sites=SITES, site_labels={"Plant.Type": PFT})


def test_a_fixed_value_with_an_undeclared_class_names_only_that_class():
    product = _product({1: "grass", 27: "broadleaf", 4711: "grass"}, ("needleleaf", "broadleaf", "grass"))
    fixed = FixedParameter(
        name="leaf_carbon_fraction", varies_by="pft", provenance="test",
        value={"needleleaf": 0.5, "broadleaf": 0.46, "grass": 0.48, "tundra": 0.4},
    )
    with pytest.raises(ValueError, match="missing \\[\\], extra \\['tundra'\\]"):
        ParameterVector(parameters=(rate(varies_by="pft"),), fixed=(fixed,), sites=SITES, site_labels={"pft": product})


def test_fixed_parameters_hold_numbers_and_their_own_mapping():
    values = {"a": 0.4, "b": 0.5}
    fixed = FixedParameter(name="leaf_carbon_fraction", varies_by="pft", value=values, provenance="test")
    values["a"] = 7.0
    assert fixed.value["a"] == 0.4
    with pytest.raises(TypeError):
        fixed.value["a"] = 7.0
    with pytest.raises(TypeError, match="must be numbers"):
        FixedParameter(name="leaf_carbon_fraction", value="0.4", provenance="test")
    with pytest.raises(TypeError, match="takes CalibrationParameters"):
        ParameterVector(parameters=(fixed,), sites=(1,))
    with pytest.raises(TypeError, match="takes FixedParameters"):
        ParameterVector(parameters=(rate(),), fixed=(rate(name="q"),), sites=(1,))


def test_select_takes_array_like_sites_and_classes(example):
    assert example.select(sites=np.array([27])).sites == (27,)
    assert example.select(sites=jnp.array([4711, 1])).sites == (1, 4711)
    theta = example.sample(jax.random.key(0), n=2)
    assert example.select(sites=example.fields(theta)["site"][:2]).sites == (1, 27)
    assert example.select(labels={"pft": np.array(["deciduous"])}).sites == (1, 4711)
    assert example.select(labels={"pft": pd.Index(["conifer", "deciduous"])}).sites == SITES


def test_select_refuses_what_is_not_a_sequence_of_distinct_site_ids(example):
    with pytest.raises(ValueError, match="more than once"):
        example.select(sites=(1, 1))
    for sites in (27, np.int64(27), (27.5,), (27.0,), (True,), {1, 27}, "27"):
        with pytest.raises(TypeError, match="sites"):
            example.select(sites=sites)


# ── coverage the mutation tests asked for ────────────────────────────────────


def test_fields_attributes_follow_the_space_and_the_component(example, theta):
    for variable in example.fields(theta, space="unconstrained").data_vars.values():
        assert variable.attrs["space"] == "unconstrained" and variable.attrs["units"] == "1"
    natural = example.fields(theta)
    assert natural["photosynthesis.respiration_share"].attrs["units"] == "1"
    assert natural["initial_soil_carbon"].attrs["long_name"] == "initial_soil_carbon: soil_carbon"
    assert natural["initial_soil_carbon"].attrs["sipnet_parameters"] == "soil_carbon"


def test_the_sipnet_parameter_fields_are_in_pysipnet_order_from_either_space(example, theta):
    order = [n for n in FLAT_SPECS if n in example.sipnet_parameter_names]
    assert list(example.sipnet_parameter_names) == order
    assert list(example.sipnet_parameter_fields(theta).data_vars) == order
    xr.testing.assert_allclose(
        example.sipnet_parameter_fields(example.fields(theta, space="unconstrained")), example.sipnet_parameter_fields(theta), rtol=1e-12
    )


def test_flat_reads_sites_by_label_and_ignores_extras(example, theta):
    fields = example.fields(theta)
    np.testing.assert_allclose(example.flat(fields.isel(site=[2, 0, 1])), theta, rtol=1e-10, atol=1e-10)
    extra = fields.isel(site=[0]).assign_coords(site=[9999])
    np.testing.assert_allclose(example.flat(xr.concat([fields, extra], dim="site")), theta, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="'site' coordinate"):
        example.flat(fields.drop_vars(["site", "pft"]))
    with pytest.raises(ValueError, match="carry no coordinate"):
        example.flat(fields.assign(initial_soil_carbon=fields["initial_soil_carbon"].expand_dims(time=2)))
    timed = fields["initial_soil_carbon"].expand_dims(time=pd.date_range("2012-01-01", periods=2))
    with pytest.raises(ValueError, match="must be on"):
        example.flat(fields.assign(initial_soil_carbon=timed))


def test_select_restricts_per_site_fixed_values_and_keeps_require_complete(example):
    vector = ParameterVector(
        parameters=(rate(),),
        fixed=(FixedParameter(name="leaf_carbon_fraction", varies_by="site", value={1: 0.4, 27: 0.45, 4711: 0.5}, provenance="t"),),
        sites=SITES,
    )
    assert dict(vector.select(sites=(27, 4711)).fixed[0].value) == {27: 0.45, 4711: 0.5}
    filled = tuple(
        FixedParameter(name=n, value=_in_domain_value(n), provenance="t") for n in example.unset_sipnet_parameter_names
    )
    complete = ParameterVector(
        parameters=example.parameters, fixed=example.fixed + filled, sites=example.sites,
        site_labels=example.site_labels, require_complete=True,
    )
    assert complete.select(sites=(1,)).require_complete
    assert repr(complete).splitlines()[-1] == "  unset: none; every required SIPNET parameter is calibrated or fixed"


def test_a_joint_prior_over_classes_is_restricted_to_its_marginal():
    base = tfd.MultivariateNormalTriL(loc=MEAN, scale_tril=jnp.linalg.cholesky(COVARIANCE))
    joint = CalibrationParameter(
        name="soil", prior=tfd.TransformedDistribution(base, tfb.Exp()),
        sipnet_map="soil_carbon", varies_by="pft", provenance="t",
    )
    product = _product({1: "a", 27: "c", 4711: "a"}, ("a", "b", "c"))
    vector = ParameterVector(parameters=(joint,), sites=SITES, site_labels={"pft": product})
    np.testing.assert_allclose(vector.gaussian_prior().mean, np.asarray(MEAN)[[0, 2]], rtol=1e-12)


def test_a_product_is_read_by_site_id_whatever_its_row_order():
    product = _product({4711: "grass", 27: "broadleaf", 1: "needleleaf"}, ("needleleaf", "broadleaf", "grass"))
    vector = ParameterVector(parameters=(rate(varies_by="pft"),), sites=SITES, site_labels={"pft": product})
    assert vector.site_labels["pft"] == ("needleleaf", "broadleaf", "grass")


def test_construction_refusals_the_suite_did_not_reach():
    with pytest.raises(ValueError, match="joint prior over groups has batch"):
        CalibrationParameter(
            name="s", prior=tfd.TransformedDistribution(
                tfd.MultivariateNormalDiag(loc=jnp.zeros((2, 3)), scale_diag=jnp.ones((2, 3))), tfb.Exp()
            ),
            sipnet_map="soil_carbon", varies_by="site", provenance="t",
        )
    with pytest.raises(ValueError, match="one copy is a scalar or a vector"):
        CalibrationParameter(
            name="a", prior=tfd.Independent(tfd.Normal(jnp.zeros((4, 2, 2)), jnp.float64(1.0)), 3),
            sipnet_map=ALLOCATION, provenance="t",
        )
    lognormals = CalibrationParameter(
        name="soil", prior=tfd.Independent(tfd.LogNormal(jnp.zeros(3), jnp.ones(3)), 1),
        sipnet_map="soil_carbon", varies_by="site", provenance="t",
    )
    assert not lognormals.has_analytic_moments
    with pytest.raises(NotImplementedError, match="cannot be restricted"):
        ParameterVector(parameters=(lognormals,), sites=SITES).select(sites=(1, 27))
    with pytest.raises(ValueError, match="at least one calibration parameter"):
        ParameterVector(parameters=(), sites=SITES)
    with pytest.raises(ValueError, match="at least one site"):
        ParameterVector(parameters=(rate(),), sites=())
    with pytest.raises(ValueError, match="repeats a site_id"):
        ParameterVector(
            parameters=(rate(varies_by="pft"),), sites=(1, 27),
            site_labels={"pft": pd.DataFrame({"site_id": [1, 1, 27], "label": ["a", "b", "a"]})},
        )
    with pytest.raises(ValueError, match="carry no declared class"):
        ParameterVector(
            parameters=(rate(varies_by="pft"),), sites=SITES,
            site_labels={"pft": pd.Categorical(["a", "zzz", "a"], categories=["a", "b"])},
        )
    with pytest.raises(ValueError, match="have no site labels"):
        ParameterVector(
            parameters=(rate(),), sites=SITES,
            fixed=(FixedParameter(name="leaf_carbon_fraction", varies_by="landcover", value={"x": 0.4}, provenance="t"),),
        )
    beyond_int16 = ParameterVector(parameters=(rate(),), sites=(1,)).fields(jnp.zeros((2**15 + 1, 1)))
    assert beyond_int16.sizes["sample"] == 2**15 + 1
    with pytest.raises(ValueError, match="is reserved"):
        rate(name="site")
    for name in ("lon", "lat", "site_id", "sample", "point", "x", "y"):
        with pytest.raises(ValueError, match="reserved"):
            ParameterVector(parameters=(rate(),), sites=SITES, site_labels={name: PFT})
    for name in ("sample", "point", "x", "y", "lon", "lat"):
        with pytest.raises(ValueError, match="is reserved"):
            rate(name=name)


@pytest.mark.parametrize("name", ["initial_condition_member", "driver_member", "source_index"])
def test_a_data_source_member_dim_name_is_reserved(name):
    """A parameter or site-labels product would collide where the IC conversion crosses them."""
    with pytest.raises(ValueError, match="is reserved"):
        rate(name=name)
    with pytest.raises(ValueError, match="reserved"):
        ParameterVector(parameters=(rate(),), sites=SITES, site_labels={name: PFT})


def test_repr_of_a_one_site_vector_with_nothing_fixed():
    lines = repr(ParameterVector(parameters=(rate(),), sites=(1,))).splitlines()
    assert lines[0] == "ParameterVector  D = 1  |  1 site  |  site labels: none"
    assert lines[-2] == "  fixed: none"


# ── batch identity and netCDF names ──────────────────────────────────────────


def test_sipnet_parameter_fields_from_fields_keep_the_samples_it_was_given(example, theta):
    subset = example.fields(theta).sel(sample=[1, 3])
    sipnet_parameter_fields = example.sipnet_parameter_fields(subset)
    assert sipnet_parameter_fields["sample"].values.tolist() == [1, 3] and sipnet_parameter_fields["sample"].dtype == np.int64
    full = example.sipnet_parameter_fields(theta)
    assert sipnet_overrides(sipnet_parameter_fields, batch={"sample": 3}, site=27) == pytest.approx(
        sipnet_overrides(full, batch={"sample": 3}, site=27)
    )
    with pytest.raises(KeyError, match="sample 0 is not one of the SIPNET parameter fields' sample labels"):
        sipnet_overrides(sipnet_parameter_fields, batch={"sample": 0}, site=27)
    from pyens.xarray import fields_from_dataset

    grids = fields_from_dataset(sipnet_parameter_fields)
    sample_axis, site_axis = grids["soil_carbon"].axes
    assert list(sample_axis.labels) == [1, 3]
    assert grids["soil_carbon"].value_at({sample_axis: 1, site_axis: 1}) == float(
        full["soil_carbon"].sel(sample=3, site=27)
    )
    with pytest.raises(ValueError, match="distinct integers"):
        example.sipnet_parameter_fields(example.fields(theta[:2]).assign_coords(sample=[0, 0]))


def test_labels_beyond_int16_are_kept(example, theta):
    fields = example.fields(theta[:2]).assign_coords(sample=[40000, -5])
    sipnet_parameter_fields = example.sipnet_parameter_fields(fields)
    assert sipnet_parameter_fields["sample"].values.tolist() == [40000, -5]


def test_a_batch_of_more_than_32768_samples_is_labeled(example):
    theta = np.zeros((32770, example.dimension))
    fields = example.fields(theta, space="unconstrained")
    assert fields.sizes["sample"] == 32770
    assert fields["sample"].values[-1] == 32769 and fields["sample"].dtype == np.int64


def test_the_batch_dim_may_be_named(example, theta):
    fields = example.fields(theta, batch_dim="draw")
    assert fields["initial_soil_carbon"].dims == ("draw", "site")
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta, batch_dim="draw")
    assert sipnet_parameter_fields["soil_carbon"].dims == ("draw", "site")
    np.testing.assert_allclose(example.flat(fields), theta, rtol=1e-10, atol=1e-10)
    # Fields keep their own batch dim through the SIPNET parameter fields.
    assert example.sipnet_parameter_fields(fields)["soil_carbon"].dims == ("draw", "site")


@pytest.mark.parametrize("name", ["pft", "initial_soil_carbon", "site", "time", "lat", "site_id"])
def test_a_batch_dim_may_not_take_a_reserved_or_taken_name(example, theta, name):
    with pytest.raises(ValueError, match="batch"):
        example.fields(theta, batch_dim=name)
    with pytest.raises(ValueError, match="batch"):
        example.sipnet_parameter_fields(theta, batch_dim=name)


@pytest.mark.parametrize(
    "name", ["time_step_start", "time_step_length", "year", "hour_of_day", "window_end"]
)
def test_a_batch_dim_may_not_take_a_model_output_or_window_coordinate_name(example, theta, name):
    """Fields on ``time_step_length`` were made, and validate_field refused them."""
    with pytest.raises(ValueError, match="cannot name a batch dim; it is a coordinate"):
        example.fields(theta, batch_dim=name)
    with pytest.raises(ValueError, match="cannot name a batch dim; it is a coordinate"):
        example.sipnet_parameter_fields(theta, batch_dim=name)


@pytest.mark.parametrize("name", ["driver_member", "initial_condition_member", "shared", "source_index"])
def test_a_batch_dim_may_not_take_a_data_source_member_or_vector_name(example, theta, name):
    """Theta's rows named ``driver_member`` were stamped as driver members."""
    with pytest.raises(ValueError, match="batch_dim"):
        example.fields(theta, batch_dim=name)
    with pytest.raises(ValueError, match="batch_dim"):
        example.sipnet_parameter_fields(theta, batch_dim=name)


def test_the_reserved_names_are_built_from_the_shared_constants():
    from sipnet_calibration.conventions import (
        DATA_SOURCE_MEMBER_NAMES,
        NON_BATCH_DIM_NAMES,
        SAMPLE,
        SITE_ID,
    )
    from sipnet_calibration.parameter_vector import (
        RESERVED_PARAMETER_NAMES,
        RESERVED_SITE_LABELS_NAMES,
        SHARED,
    )

    assert RESERVED_PARAMETER_NAMES == {SAMPLE, *NON_BATCH_DIM_NAMES, *DATA_SOURCE_MEMBER_NAMES}
    assert RESERVED_SITE_LABELS_NAMES == RESERVED_PARAMETER_NAMES | {SHARED, SITE_ID}


def test_the_fields_and_sipnet_parameter_fields_carry_the_attributes_of_their_batch_dim(example, theta):
    from sipnet_calibration.conventions import SAMPLE_ATTRIBUTES

    assert dict(example.fields(theta)["sample"].attrs) == dict(SAMPLE_ATTRIBUTES)
    assert dict(example.sipnet_parameter_fields(theta)["sample"].attrs) == dict(SAMPLE_ATTRIBUTES)
    assert dict(example.fields(theta, batch_dim="draw")["draw"].attrs) == {}


def test_sipnet_overrides_names_a_site_the_sipnet_parameter_fields_lack(example, theta):
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    with pytest.raises(KeyError, match="site 2 is not one of the SIPNET parameter fields' site"):
        sipnet_overrides(sipnet_parameter_fields, batch={"sample": 0}, site=2)


def test_a_scalar_batch_coordinate_gives_one_vector(example, theta):
    one = example.fields(theta).isel(sample=2)
    assert int(one["sample"]) == 2
    flat = example.flat(one)
    assert flat.shape == (example.dimension,)
    np.testing.assert_allclose(flat, theta[2], rtol=1e-10, atol=1e-10)


def _located_example():
    table = pd.DataFrame(
        {"site_id": list(SITES), "lon": [-24.6, -78.6, -107.3], "lat": [82.5, 80.6, 44.0]}
    )
    return example_parameter_vector(sites=table, pft=PFT)


def test_two_batch_dims_are_refused_until_stacked(theta):
    from sipnet_calibration.fields import stack_batch_dims

    located = _located_example()
    fields = located.fields(theta, space="unconstrained")
    crossed = fields.expand_dims(initial_condition_member=[0, 1]).transpose(
        "sample", "initial_condition_member", "site"
    )
    with pytest.raises(ValueError, match="stack_batch_dims") as refusal:
        located.flat(crossed)
    # The advice for a Dataset is the call that runs.
    assert "dataset.map(lambda field: stack_batch_dims(field, into='run'))" in str(refusal.value)
    stacked = crossed.map(lambda field: stack_batch_dims(field, into="run"))
    flat = located.flat(stacked)
    assert flat.shape == (2 * len(theta), located.dimension)
    np.testing.assert_allclose(flat[::2], theta, rtol=1e-10, atol=1e-10)


def test_the_flat_round_trip_of_a_stack_unstacks(theta):
    """Stacked Fields -> Flat -> fields(batch_dim=) -> unstacked, as documented."""
    from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

    located = _located_example()
    fields = located.fields(theta[:3], space="unconstrained")
    crossed = fields.expand_dims(initial_condition_member=[4, 1]).transpose(
        "sample", "initial_condition_member", "site"
    )
    stacked = crossed.map(lambda field: stack_batch_dims(field, into="run"))
    made = located.fields(located.flat(stacked), space="unconstrained", batch_dim="run")
    restored = made.map(
        lambda array: unstack_batch_dims(array, labels_from=stacked[array.name])
    )
    for name in crossed.data_vars:
        xr.testing.assert_allclose(restored[name], crossed[name])
        assert restored[name].dims == crossed[name].dims


@pytest.mark.parametrize(
    "name", ["soil_carbon", "leaf_carbon_fraction", "allocation.leaf_allocation", "allocation"]
)
def test_a_batch_dim_may_not_take_a_sipnet_parameter_or_fields_variable_name(
    example, theta, name
):
    variables = set(example.fields(theta).data_vars) | set(example.sipnet_parameter_fields(theta).data_vars)
    assert name in variables or name in example.parameter_names
    with pytest.raises(ValueError, match=f"batch_dim={name!r} is"):
        example.fields(theta, batch_dim=name)
    with pytest.raises(ValueError, match=f"batch_dim={name!r} is"):
        example.sipnet_parameter_fields(theta, batch_dim=name)


def test_a_dim_without_integer_labels_is_refused_in_the_fields_words(example, theta):
    fields = example.fields(theta)
    for broken in (fields.drop_vars("sample"), fields.assign_coords(sample=np.arange(8.0))):
        with pytest.raises(ValueError, match="neither a batch dim|carry no coordinate"):
            example.flat(broken)
        with pytest.raises(ValueError, match="neither a batch dim|carry no coordinate"):
            example.sipnet_parameter_fields(broken)
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    with pytest.raises(ValueError, match="carry no coordinate"):
        sipnet_overrides(sipnet_parameter_fields.drop_vars("sample"), batch={"sample": 0}, site=27)


def test_sipnet_overrides_takes_integer_labels_only(example, theta):
    sipnet_parameter_fields = example.sipnet_parameter_fields(theta)
    for label in (1.0, True):
        with pytest.raises(TypeError, match="integer"):
            sipnet_overrides(sipnet_parameter_fields, batch={"sample": label}, site=27)
    for site in (27.0, True):
        with pytest.raises(TypeError, match="integer"):
            sipnet_overrides(sipnet_parameter_fields, batch={"sample": 1}, site=site)
    with pytest.raises(TypeError, match="batch must be a mapping"):
        sipnet_overrides(sipnet_parameter_fields, batch=3, site=27)
    assert sipnet_overrides(sipnet_parameter_fields, batch={"sample": np.int64(1)}, site=np.int32(27)) == (
        sipnet_overrides(sipnet_parameter_fields, batch={"sample": 1}, site=27)
    )


def test_unconstrained_fields_write_to_netcdf_and_read_back(example, theta, tmp_path):
    unconstrained = example.fields(theta, space="unconstrained")
    assert not any("/" in name for name in unconstrained.data_vars)
    path = tmp_path / "unconstrained.nc"
    unconstrained.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as back:
        np.testing.assert_array_equal(example.flat(back.load()), theta)


# ── the sixteen-class site labels ─────────────────────────────────────────────


def test_a_vector_over_the_whole_pool_takes_its_groups_from_the_16class_labels():
    from sipnet_calibration.site_labels import load_site_labels, resolve_site_labels
    from sipnet_calibration.sites import load_sites

    try:
        sites, labels = load_sites(), load_site_labels("pft_16class")
    except FileNotFoundError as error:
        pytest.skip(f"processed site table or site labels not available in this working copy: {error}")
    classes = resolve_site_labels("pft_16class").labels
    vector = ParameterVector(
        parameters=(rate(varies_by="pft"),),
        fixed=(FixedParameter(
            name="leaf_carbon_fraction", value={c: 0.4 + 0.01 * i for i, c in enumerate(classes)},
            varies_by="pft", provenance="test",
        ),),
        sites=sites, site_labels={"pft": labels},
    )
    assert vector.group_labels("pft") == classes
    assert vector.dimension == len(classes)

    theta = vector.sample(jax.random.key(0), n=4)
    turnover = vector.fields(theta)["r"]
    by_site = labels.set_index("site_id")["label"].astype(str)
    assert (turnover["pft"].to_series().astype(str) == by_site.reindex(turnover["site"].values).values).all()
    # One draw per class, shared by every site of that class.
    grouped = turnover.groupby("pft")
    assert (grouped.max("site") == grouped.min("site")).all()

    wetland = vector.select(labels={"pft": ["Permanent_Wetlands"]})
    assert wetland.dimension == 1
    assert set(wetland.sites) == set(by_site.index[by_site == "Permanent_Wetlands"])


# ── the vector cannot change after its checks, and pickles ───────────────────


def _frozen_candidate(sites=SITES) -> ParameterVector:
    """A vector whose priors all round-trip through pickle, with a per-class
    fixed value, the mapping that once made the vector unpicklable."""
    return ParameterVector(
        parameters=(rate(varies_by="pft"),),
        fixed=(FixedParameter(
            name="leaf_carbon_fraction", value={"conifer": 0.4, "deciduous": 0.5},
            varies_by="pft", provenance="test",
        ),),
        sites=sites, site_labels={"pft": PFT},
    )


def test_a_vector_with_a_per_class_fixed_value_pickles_and_round_trips():
    import copy
    import pickle

    vector = _frozen_candidate()
    theta = vector.sample(jax.random.key(0), n=3)
    for restored in (pickle.loads(pickle.dumps(vector)), copy.deepcopy(vector)):
        assert restored.fields(theta).identical(vector.fields(theta))
        assert restored.sipnet_parameter_fields(theta).identical(vector.sipnet_parameter_fields(theta))
        assert dict(restored.fixed[0].value) == {"conifer": 0.4, "deciduous": 0.5}


def test_a_vector_with_a_transformed_distribution_prior_does_not_unpickle():
    """What the module Notes say: the limit is TFP's, not the vector's."""
    import pickle

    vector = example_parameter_vector(SITES, pft=PFT)
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(vector))


def test_site_labels_and_a_fixed_mapping_cannot_be_changed():
    vector = _frozen_candidate()
    with pytest.raises(TypeError):
        vector.site_labels["pft"] = ("conifer",) * 3
    with pytest.raises(TypeError):
        vector.fixed[0].value["conifer"] = 0.9
    assert vector.site_labels["pft"] == PFT


def test_the_vector_keeps_its_own_copy_of_the_site_tables_lon_lat():
    table = pd.DataFrame(
        {"site_id": list(SITES), "lon": [-24.6, -78.6, -107.3], "lat": [82.5, 80.6, 44.0]}
    )
    vector = _frozen_candidate(sites=table)
    table.loc[0, "lon"] = 0.0
    table["lat"] = np.zeros(3)
    np.testing.assert_array_equal(vector.site_table["lon"], [-24.6, -78.6, -107.3])
    np.testing.assert_array_equal(vector.site_table["lat"], [82.5, 80.6, 44.0])


def test_the_kept_lon_lat_are_read_only_and_what_fields_hands_out_is_writable():
    table = pd.DataFrame(
        {"site_id": list(SITES), "lon": [-24.6, -78.6, -107.3], "lat": [82.5, 80.6, 44.0]}
    )
    vector = _frozen_candidate(sites=table)
    lon, lat = vector._lon_lat
    assert not lon.flags.writeable and not lat.flags.writeable
    with pytest.raises(ValueError):
        lon[0] = 0.0
    fields = vector.fields(vector.sample(jax.random.key(0), n=2))
    fields["lon"].values[0] = 5.0
    np.testing.assert_array_equal(vector.site_table["lon"], [-24.6, -78.6, -107.3])


def test_site_labels_is_a_dict_to_pandas():
    vector = _frozen_candidate()
    assert pd.DataFrame(vector.site_labels).shape == (len(SITES), 1)


def test_a_per_class_fixed_parameter_compares_and_hashes_by_identity():
    first = _frozen_candidate().fixed[0]
    second = dataclasses.replace(first)
    assert first == first and first != second
    assert len({first, second}) == 2


@pytest.mark.parametrize("site", [27, 1])
def test_sipnet_overrides_refuses_sipnet_parameter_fields_selected_to_one_site_in_its_words(site):
    """SIPNET parameter fields with a scalar ``site`` raised a raw TypeError from ``in``."""
    sipnet_parameter_fields = xr.Dataset(
        {"soil_carbon": (("sample", "site"), np.ones((2, 2)))},
        coords={"sample": [0, 1], "site": np.array([1, 27], np.int32)},
    )
    with pytest.raises(ValueError, match="selected to site 27 alone"):
        sipnet_overrides(sipnet_parameter_fields.isel(site=1), site=site, batch={"sample": 1})


def test_sipnet_overrides_refuses_sipnet_parameter_fields_without_sites_in_its_words():
    sipnet_parameter_fields = xr.Dataset({"soil_carbon": (("sample",), np.ones(2))}, coords={"sample": [0, 1]})
    with pytest.raises(ValueError, match="have no site dim"):
        sipnet_overrides(sipnet_parameter_fields, site=1, batch={"sample": 1})

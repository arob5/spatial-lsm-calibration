"""Tests for the parameterization layer.

Four layers. The prior helpers are checked against the quantities they claim
to set (median, interval, refusal of bad samples). The bijectors and the log
prior are checked by round trips and against a finite-difference Jacobian,
coordinate by coordinate, so the identity ``log_prior`` rests on is verified
rather than trusted. The registry checks are each provoked once. Finally the
example registry is pushed through pySIPNET: 2000 prior draws land in every
domain, one draw assembles into a validated ``SIPNETParameters``, and one
draw runs the bundled Niwot fixture when the binary and fixture are present.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from pysipnet import niwot_reference_files
from pysipnet.parameters.base import ParameterDomain
from pysipnet.parameters.model import PARAMETER_SPECS, SIPNETParameters
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration import parameterization as module
from sipnet_calibration.parameterization import (
    ALLOCATION,
    DOMAIN_CHECK_CORNERS,
    PHOTOSYNTHESIS,
    Coordinate,
    FixedParameter,
    Identity,
    Parameterization,
    PhotosynthesisMap,
    SimplexMap,
    example_parameterization,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    product_transformed_gaussian_prior,
    pysipnet_overrides,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors



#: pySIPNET's Niwot Ridge reference inputs, which it ships inside the package
#: (its PR #40), so no source checkout is involved.
NIWOT = niwot_reference_files()

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
FLAT_SPECS = {path.split(".", 1)[1]: spec for path, spec in PARAMETER_SPECS.items()}


@pytest.fixture(scope="module")
def example() -> Parameterization:
    return example_parameterization(sites=SITES, pft=PFT)


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


# ── coordinate-to-parameter maps ─────────────────────────────────────────────


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
    coordinate = Coordinate(
        name="x", prior=log_normal(median=1.0, geometric_sd=2.0),
        coord_to_param="wood_turnover_rate", provenance="test",
    )
    assert isinstance(coordinate.coord_to_param, Identity)
    assert coordinate.coord_to_param.writes == ("wood_turnover_rate",)
    assert coordinate.element_labels == ("log(wood_turnover_rate)",)
    scaled = Coordinate(
        name="t", prior=tfd.Uniform(jnp.float64(1.0), jnp.float64(5.0)),
        coord_to_param="optimum_photosynthesis_temperature", provenance="x",
    )
    assert scaled.element_labels == ("logit((optimum_photosynthesis_temperature - 1)/(5 - 1))",)


# ── the specs and their checks ───────────────────────────────────────────────


def test_coordinate_refuses_bad_name_parameter_and_provenance():
    prior = log_normal(median=1.0, geometric_sd=2.0)
    with pytest.raises(ValueError, match="lower_case_with_underscores"):
        Coordinate(name="BadName", prior=prior, coord_to_param="wood_turnover_rate", provenance="x")
    with pytest.raises(ValueError, match="not pySIPNET parameter names"):
        Coordinate(name="a", prior=prior, coord_to_param="aMax", provenance="x")
    with pytest.raises(ValueError, match="provenance is empty"):
        Coordinate(name="a", prior=prior, coord_to_param="wood_turnover_rate", provenance="  ")


def test_coordinate_refuses_component_count_mismatch():
    with pytest.raises(ValueError, match="names 4 components"):
        Coordinate(
            name="a", prior=softmax_normal(center=[0.5, 0.5], logit_sd=1.0),
            coord_to_param=ALLOCATION, provenance="x",
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


def build(coordinates, fixed=(), sites=SITES, labelings=None):
    return Parameterization(
        coordinates=coordinates, fixed=fixed, sites=sites,
        labelings={"pft": PFT} if labelings is None else labelings,
    )


def distinct_groups() -> Parameterization:
    """A registry whose groups can be told apart: one prior per site and per
    PFT with distinct medians and spreads, and a per-PFT fixed value."""
    return Parameterization(
        coordinates=(
            Coordinate(
                name="soil", prior=log_normal(median=[100.0, 200.0, 300.0], geometric_sd=[1.5, 2.0, 2.5]),
                coord_to_param="soil_carbon", varies_by="site", provenance="test",
            ),
            Coordinate(
                name="turnover", prior=log_normal(median=[0.01, 0.02], geometric_sd=[1.2, 1.3]),
                coord_to_param="wood_turnover_rate", varies_by="pft", provenance="test",
            ),
            Coordinate(
                name="allocation",
                prior=softmax_normal(center=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], logit_sd=[0.3, 0.6, 0.9]),
                coord_to_param=ALLOCATION, varies_by="pft", provenance="test",
            ),
        ),
        fixed=(FixedParameter(
            name="leaf_carbon_fraction", value={"a": 0.4, "b": 0.5}, varies_by="pft", provenance="test",
        ),),
        sites=SITES, labelings={"pft": ("a", "b", "a")},   # site 27 is "b", the second group
    )


def rate(name="r", parameter="wood_turnover_rate", **kwargs):
    return Coordinate(
        name=name, prior=log_normal(median=0.01, geometric_sd=2.0),
        coord_to_param=parameter, provenance="test", **kwargs,
    )


def test_parameterization_refuses_duplicate_names_and_writers():
    with pytest.raises(ValueError, match="names repeat"):
        build((rate(), rate()))
    with pytest.raises(ValueError, match="set more than once"):
        build((rate("a"), rate("b")))
    with pytest.raises(ValueError, match="set more than once"):
        build((rate(),), fixed=(FixedParameter(name="wood_turnover_rate", value=0.01, provenance="x"),))


def test_parameterization_refuses_unfixed_reads_and_missing_labelings():
    pair = Coordinate(
        name="p",
        prior=product_transformed_gaussian_prior(
            c=log_normal(median=100.0, geometric_sd=1.5), s=logit_normal(median=0.2, logit_sd=0.4)
        ),
        coord_to_param=PHOTOSYNTHESIS, provenance="x",
    )
    with pytest.raises(ValueError, match="which are not fixed"):
        build((pair,))
    with pytest.raises(ValueError, match="have no labeling"):
        build((rate(varies_by="landcover"),))
    with pytest.raises(ValueError, match="one label per site"):
        build((rate(varies_by="pft"),), labelings={"pft": ("a", "b")})
    with pytest.raises(ValueError, match="reserved"):
        build((rate(),), labelings={"member": PFT})


def test_coordinate_refuses_unusable_priors_and_maps():
    prior = log_normal(median=1.0, geometric_sd=2.0)
    with pytest.raises(ValueError, match="rank above 1"):
        Coordinate(
            name="a", prior=log_normal(median=np.ones((2, 2)), geometric_sd=2.0),
            coord_to_param="wood_turnover_rate", provenance="x",
        )
    with pytest.raises(ValueError, match="event shape .* rank above 1"):
        Coordinate(
            name="a", prior=tfd.Independent(tfd.Normal(jnp.zeros((2, 3)), jnp.float64(1.0)), 2),
            coord_to_param="optimum_photosynthesis_temperature", provenance="x",
        )
    with pytest.raises(ValueError, match="not float64"):
        Coordinate(name="a", prior=tfd.LogNormal(0.0, 1.0), coord_to_param="wood_turnover_rate", provenance="x")
    with pytest.raises(TypeError, match="must be a TFP distribution"):
        Coordinate(name="a", prior=tfb.Exp(), coord_to_param="wood_turnover_rate", provenance="x")
    with pytest.raises(TypeError, match="must be a CoordToParamMap"):
        Coordinate(name="a", prior=prior, coord_to_param={"writes": ()}, provenance="x")
    mixture = tfd.MixtureSameFamily(
        tfd.Categorical(probs=jnp.array([0.5, 0.5])), tfd.LogNormal(jnp.array([0.0, 1.0]), jnp.float64(1.0))
    )
    with pytest.raises(ValueError, match="no default event-space bijector"):
        Coordinate(name="a", prior=mixture, coord_to_param="soil_carbon", provenance="x")
    # ... whereas wrapping it in an explicit bijector is accepted.
    wrapped = Coordinate(
        name="a", prior=tfd.TransformedDistribution(mixture, tfb.Identity()),
        coord_to_param="soil_carbon", provenance="x",
    )
    assert wrapped.bijector is not None


def test_parameterization_refuses_bad_sites_and_labelings():
    with pytest.raises(TypeError, match="site ids must be integers"):
        build((rate(),), sites=(1.5, 27.0, 4711.0))
    assert build((rate(),), sites=np.array([1.0, 27.0, 4711.0])).sites == SITES
    with pytest.raises(TypeError, match="sequence of one label per site"):
        build((rate(varies_by="pft"),), labelings={"pft": "abc"})
    assert build((rate(varies_by="pft"),), labelings={"pft": np.array(PFT)}).group_labels("pft") == ("conifer", "deciduous")
    with pytest.raises(ValueError, match="collide with a coordinate or SIPNET parameter"):
        build((rate(),), labelings={"pft": PFT, "wood_turnover_rate": (1, 2, 3)})
    with pytest.raises(ValueError, match="collide"):
        build((rate(name="landcover"),), labelings={"pft": PFT, "landcover": (1, 2, 3)})


def test_parameterization_refuses_wrong_prior_batch_and_fixed_coverage():
    per_site = Coordinate(
        name="s", prior=log_normal(median=np.ones(2), geometric_sd=2.0),
        coord_to_param="soil_carbon", varies_by="site", provenance="x",
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


def test_parameterization_refuses_a_transform_that_leaves_the_domain():
    # A Normal is not a TransformedDistribution: its default bijector is the
    # identity, so theta = -12 writes a negative rate.
    normal = Coordinate(
        name="n", prior=tfd.Normal(jnp.float64(0.01), jnp.float64(0.005)),
        coord_to_param="wood_turnover_rate", provenance="x",
    )
    with pytest.raises(ValueError, match="outside its pySIPNET domain 'positive'"):
        build((normal,))
    # Whereas the same Normal on a real-domain parameter is fine, and has no
    # analytic moments through the identity? It does: the base is itself.
    fine = Coordinate(
        name="n", prior=tfd.Normal(jnp.float64(20.0), jnp.float64(5.0)),
        coord_to_param="optimum_photosynthesis_temperature", provenance="x",
    )
    p = build((fine,))
    assert p.dimension == 1


def test_domain_check_corners_are_far_but_finite():
    assert DOMAIN_CHECK_CORNERS == (-12.0, 0.0, 12.0)
    assert np.isfinite(np.exp(12.0)) and 0 < 1 / (1 + np.exp(12.0)) < 1e-5


def test_domain_check_looks_at_the_corners_not_only_the_center():
    # In domain at theta = 0 (the value 0.5), out of it at theta = +-12.
    inside_at_center = Coordinate(
        name="n", prior=tfd.Normal(jnp.float64(0.5), jnp.float64(0.1)),
        coord_to_param="leaf_off_fall_fraction", provenance="x",
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
    with pytest.raises(ValueError, match="ascending"):
        build((rate(),), sites=(1, 1, 27), labelings={"pft": PFT})


# ── layout ───────────────────────────────────────────────────────────────────


def test_layout_dimension_labels_and_slices(example):
    layout = example.layout
    assert layout.dimension == 2 + 3 * 2 + 1 * 2 + 1 + 1 * 3 == 14
    assert layout.coordinates == (
        "photosynthesis", "allocation", "base_soil_respiration", "leaf_fall_fraction",
        "initial_soil_carbon",
    )
    assert layout.groups["allocation"] == ("conifer", "deciduous")
    assert layout.groups["initial_soil_carbon"] == SITES
    assert layout.dims == {
        "photosynthesis": "shared", "allocation": "pft", "base_soil_respiration": "pft",
        "leaf_fall_fraction": "shared", "initial_soil_carbon": "site",
    }
    assert layout.labels[0] == "photosynthesis[log(capacity)]"
    assert layout.labels[2] == "allocation[conifer][alr(leaf_allocation/coarse_root_allocation)]"
    assert layout.labels[-1] == "initial_soil_carbon[4711]"
    assert len(layout.labels) == 14
    deciduous = [layout.labels[i] for i in layout.index("allocation", group="deciduous")]
    assert deciduous == [
        f"allocation[deciduous][{e}]" for e in layout.element_labels["allocation"]
    ]
    assert all("[conifer]" in layout.labels[i] for i in layout.index("allocation", group="conifer"))
    stops = [layout.slice(c).stop for c in layout.coordinates]
    assert stops == [2, 8, 10, 11, 14]


def test_layout_index_narrows_by_group_and_element(example):
    layout = example.layout
    np.testing.assert_array_equal(layout.index("allocation", group="deciduous"), [5, 6, 7])
    np.testing.assert_array_equal(
        layout.index("allocation", element="alr(wood_allocation/coarse_root_allocation)"), [3, 6]
    )
    np.testing.assert_array_equal(layout.index("initial_soil_carbon", group=27), [12])
    with pytest.raises(KeyError, match="not a group"):
        layout.index("allocation", group="grassland")
    with pytest.raises(KeyError, match="no coordinate"):
        layout.slice("nope")


def test_layout_pack_unpack_round_trip(example, theta):
    parts = example.layout.unpack(theta)
    assert parts["allocation"].shape == (8, 2, 3)
    assert parts["initial_soil_carbon"].shape == (8, 3, 1)
    np.testing.assert_array_equal(example.layout.pack(parts), theta)
    single = example.layout.unpack(theta[0])
    assert single["photosynthesis"].shape == (1, 2)
    with pytest.raises(ValueError, match="exactly the coordinates"):
        example.layout.pack({k: v for k, v in parts.items() if k != "allocation"})
    with pytest.raises(ValueError, match="theta must be"):
        example.layout.unpack(theta[:, :5])
    mismatched = dict(parts)
    mismatched["allocation"] = parts["allocation"][0]
    with pytest.raises(ValueError, match="leading dimensions disagree"):
        example.layout.pack(mismatched)


# ── bijectors and the log prior ──────────────────────────────────────────────


def test_constrain_unconstrain_round_trip(example, theta):
    natural = example.constrain(theta)
    assert natural["allocation"].shape == (8, 2, 4)
    np.testing.assert_allclose(natural["allocation"].sum(axis=-1), 1.0, atol=1e-12)
    assert (natural["initial_soil_carbon"] > 0).all()
    np.testing.assert_allclose(example.unconstrain(natural), theta, rtol=1e-10, atol=1e-10)


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


def test_log_prior_includes_the_jacobian_coordinate_by_coordinate(example, theta):
    """``prior.distribution.log_prob(theta_c)`` equals the natural-scale log
    density plus the finite-difference log Jacobian, for every coordinate
    and group, at a sampled point."""
    parts = example.layout.unpack(theta[0])
    for coordinate in example.coordinates:
        n_groups = example.n_groups(coordinate.varies_by)
        for g in range(n_groups):
            theta_c = np.asarray(parts[coordinate.name][g])
            natural_prior, unconstrained_prior = coordinate.prior, coordinate.unconstrained_prior
            if tuple(natural_prior.batch_shape) == (n_groups,):  # one prior per group
                natural_prior, unconstrained_prior = natural_prior[g], unconstrained_prior[g]

            def forward(t, coordinate=coordinate):
                x = coordinate.bijector.forward(jnp.asarray(t if coordinate.size > 1 else t[0]))
                return np.atleast_1d(np.asarray(x))

            natural = forward(theta_c)
            log_det = finite_difference_log_det(forward, theta_c, coordinate.size)
            natural_lp = float(natural_prior.log_prob(natural if coordinate.size > 1 else natural[0]))
            unconstrained_lp = float(
                unconstrained_prior.log_prob(jnp.asarray(theta_c if coordinate.size > 1 else theta_c[0]))
            )
            assert unconstrained_lp == pytest.approx(natural_lp + log_det, rel=1e-6, abs=1e-6), (
                coordinate.name
            )


def test_log_prior_is_the_sum_over_coordinates_and_jit_compiles(example, theta):
    parts = example.layout.unpack(theta)
    expected = np.zeros(8)
    for coordinate in example.coordinates:
        part = parts[coordinate.name]
        if coordinate.is_scalar:
            part = part[..., 0]
        expected += np.asarray(
            example._broadcast_unconstrained(coordinate).log_prob(part).sum(axis=-1)
        )
    np.testing.assert_allclose(example.log_prior(theta), expected, rtol=1e-12)
    assert example.log_prior(theta).shape == (8,)
    assert example.log_prior(theta[0]).shape == ()
    np.testing.assert_allclose(jax.jit(example.log_prior)(theta), expected, rtol=1e-12)
    grad = jax.grad(example.log_prior)(theta[0])
    assert grad.shape == (14,) and np.isfinite(grad).all()


def test_sample_is_reproducible_and_per_coordinate(example):
    a = example.sample(jax.random.key(3), n=4)
    b = example.sample(jax.random.key(3), n=4)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 14) and a.dtype == jnp.float64
    # Per-site draws are independent, not one value broadcast over sites.
    site_block = a[:, example.layout.slice("initial_soil_carbon")]
    assert not np.allclose(site_block[:, 0], site_block[:, 1])
    # Coordinates get their own keys: standardized draws are neither identical
    # nor correlated across coordinates.
    gaussian = example.to_eki_gaussian_prior()
    draws = np.asarray(example.sample(jax.random.key(7), n=20_000))
    standardized = (draws - np.asarray(gaussian.mean)) / np.sqrt(np.diag(np.asarray(gaussian.cov.to_dense())))
    correlation = np.corrcoef(standardized, rowvar=False)
    off_block = np.abs(correlation - np.eye(14))
    assert off_block.max() < 0.03
    assert not np.allclose(standardized[:, 10], standardized[:, 8])  # leaf_fall vs base_soil_respiration


# ── the EKI export ───────────────────────────────────────────────────────────


def test_eki_gaussian_matches_the_prior_moments(example):
    gaussian = example.to_eki_gaussian_prior()
    assert gaussian.mean.shape == (14,)
    assert gaussian.cov.shape == (14, 14)
    assert gaussian.cov.block_shapes == ((2, 2), (6, 6), (2, 2), (1, 1), (3, 3))
    draws = np.asarray(example.sample(jax.random.key(4), n=200_000))
    np.testing.assert_allclose(gaussian.mean, draws.mean(axis=0), atol=0.02)
    np.testing.assert_allclose(np.diag(gaussian.cov.to_dense()), draws.var(axis=0), rtol=0.03)
    # Independent coordinates: no cross-block covariance.
    dense = np.asarray(gaussian.cov.to_dense())
    assert np.all(dense[:2, 2:] == 0) and np.all(dense[8:, :8] == 0)
    # Exact where the prior is Gaussian in theta.
    ln = example.coordinate("base_soil_respiration").prior
    np.testing.assert_allclose(gaussian.mean[8:10], float(ln.distribution.loc))
    np.testing.assert_allclose(dense[8, 8], float(ln.distribution.scale) ** 2)


def test_eki_gaussian_equals_the_analytic_unconstrained_moments(example):
    gaussian = example.to_eki_gaussian_prior()
    dense = np.asarray(gaussian.cov.to_dense())
    for coordinate in example.coordinates:
        prior = example._broadcast_unconstrained(coordinate)
        sl = example.layout.slice(coordinate.name)
        np.testing.assert_allclose(gaussian.mean[sl], np.ravel(prior.mean()), rtol=1e-12)
        if coordinate.is_scalar:
            np.testing.assert_allclose(np.diag(dense)[sl], np.ravel(prior.variance()), rtol=1e-12)
        else:
            expected = np.asarray(prior.covariance())
            block = dense[sl, sl]
            for g in range(expected.shape[0]):
                k = coordinate.size
                np.testing.assert_allclose(block[g * k:(g + 1) * k, g * k:(g + 1) * k], expected[g], rtol=1e-12)


def test_eki_gaussian_refuses_a_zero_spread():
    flat = Coordinate(
        name="z", prior=tfd.LogNormal(jnp.float64(0.0), jnp.float64(0.0)),
        coord_to_param="wood_turnover_rate", provenance="x",
    )
    with pytest.raises(ValueError, match="zero, negative or non-finite variance"):
        build((flat,)).to_eki_gaussian_prior()


def test_eki_gaussian_needs_a_key_for_non_analytic_moments():
    beta = Coordinate(
        name="b", prior=tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
        coord_to_param="leaf_off_fall_fraction", provenance="x",
    )
    assert not beta.has_analytic_moments
    p = build((beta,))
    with pytest.raises(NotImplementedError, match="no analytic unconstrained moments"):
        p.to_eki_gaussian_prior()
    with pytest.raises(ValueError, match="at least 2"):
        p.to_eki_gaussian_prior(key=jax.random.key(0), n_moment_samples=1)
    gaussian = p.to_eki_gaussian_prior(key=jax.random.key(0), n_moment_samples=50_000)
    draws = np.asarray(p.sample(jax.random.key(1), n=50_000))
    assert float(gaussian.mean[0]) == pytest.approx(draws.mean(), abs=0.02)
    assert gaussian.mean.dtype == jnp.float64
    assert p.describe()["theta_moments"].iloc[0] == "monte_carlo"
    # Two Monte Carlo coordinates get different keys, so their estimates differ.
    two = build((
        beta,
        Coordinate(
            name="c", prior=tfd.Beta(jnp.float64(2.0), jnp.float64(5.0)),
            coord_to_param="leaf_on_reallocation_fraction", provenance="x",
        ),
    ))
    mean = two.to_eki_gaussian_prior(key=jax.random.key(0), n_moment_samples=50).mean
    assert float(mean[0]) != float(mean[1])


# ── the override table and pySIPNET ──────────────────────────────────────────


def test_override_table_shape_names_and_attributes(example, theta):
    table = example.to_pysipnet_parameters(theta)
    assert isinstance(table, xr.Dataset)
    assert dict(table.sizes) == {"member": 8, "site": 3}
    assert set(table.data_vars) == {
        "max_photosynthesis_rate", "foliar_respiration_fraction", "leaf_allocation",
        "wood_allocation", "fine_root_allocation", "base_soil_respiration_rate",
        "leaf_off_fall_fraction", "soil_carbon", "daily_mean_photosynthesis_fraction",
        "leaf_carbon_fraction", "vapor_pressure_deficit_exponent",
    }
    assert table["soil_carbon"].attrs == {
        "units": "g m-2", "sipnet_name": "soilInit", "source": "coordinate initial_soil_carbon",
        "constituent": "C",
    }
    assert table["leaf_carbon_fraction"].attrs["source"] == "fixed"
    assert list(table["pft"].values) == list(PFT)
    # Shared and per-PFT values broadcast onto sites; the two deciduous sites agree.
    a = table["leaf_allocation"]
    np.testing.assert_array_equal(a.sel(site=1), a.sel(site=4711))
    assert not np.allclose(a.sel(site=1), a.sel(site=27))
    single = example.to_pysipnet_parameters(theta[0])
    assert dict(single.sizes) == {"site": 3}


def test_override_table_values_come_from_the_right_coordinate_and_group(example, theta):
    table = example.to_pysipnet_parameters(theta)
    natural = example.coordinates_table(theta, scale="natural")
    for member in (0, 5):
        for site, pft in zip(SITES, PFT, strict=True):
            row = table.isel(member=member).sel(site=site)
            assert float(row["soil_carbon"]) == float(natural["initial_soil_carbon"].isel(member=member).sel(site=site))
            assert float(row["leaf_off_fall_fraction"]) == float(natural["leaf_fall_fraction"].isel(member=member))
            assert float(row["base_soil_respiration_rate"]) == float(
                natural["base_soil_respiration"].isel(member=member).sel(pft=pft)
            )
            allocation = natural["allocation"].isel(member=member).sel(pft=pft)
            for name in ("leaf_allocation", "wood_allocation", "fine_root_allocation"):
                assert float(row[name]) == float(allocation.sel(element=name))
            photosynthesis = natural["photosynthesis"].isel(member=member)
            capacity, share = (float(photosynthesis.sel(element=e)) for e in PHOTOSYNTHESIS.components)
            assert float(row["max_photosynthesis_rate"]) == pytest.approx(capacity * 0.466 * (1 - share) / 0.76)


def test_distinct_groups_are_gathered_onto_the_right_sites():
    p = distinct_groups()
    gaussian = p.to_eki_gaussian_prior()
    # The Gaussian mean is each prior's base location, so constraining it gives
    # the medians (and the softmax centers) group by group, in group order.
    np.testing.assert_allclose(gaussian.mean[p.layout.slice("soil")], np.log([100.0, 200.0, 300.0]))
    np.testing.assert_allclose(gaussian.mean[p.layout.slice("turnover")], np.log([0.01, 0.02]))
    dense = np.asarray(gaussian.cov.to_dense())
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("soil")], np.log([1.5, 2.0, 2.5]) ** 2)
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("turnover")], np.log([1.2, 1.3]) ** 2)
    np.testing.assert_allclose(np.diag(dense)[p.layout.slice("allocation")], np.tile([0.3, 0.6, 0.9], 2) ** 2)
    table = p.to_pysipnet_parameters(gaussian.mean)
    np.testing.assert_allclose(table["soil_carbon"].values, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(table["wood_turnover_rate"].values, [0.01, 0.02, 0.01])
    np.testing.assert_allclose(table["leaf_carbon_fraction"].values, [0.4, 0.5, 0.4])
    np.testing.assert_allclose(table["leaf_allocation"].values, [0.1, 0.4, 0.1], rtol=1e-12)
    np.testing.assert_allclose(table["fine_root_allocation"].values, [0.3, 0.2, 0.3], rtol=1e-12)
    frame = p.describe()
    soil = frame[frame["coordinate"] == "soil"]
    np.testing.assert_allclose(soil["natural_median"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(soil["theta_sd"], np.log([1.5, 2.0, 2.5]))
    assert list(soil["group"]) == list(SITES)


def test_prior_draws_land_in_every_domain(example):
    table = example.to_pysipnet_parameters(example.sample(jax.random.key(5), n=2000))
    for name, values in table.data_vars.items():
        assert in_domain(FLAT_SPECS[name].domain, values.values), name
    triangle = table["leaf_allocation"] + table["wood_allocation"] + table["fine_root_allocation"]
    assert float(triangle.max()) < 1.0


def test_pysipnet_overrides_gives_one_run_of_floats(example, theta):
    table = example.to_pysipnet_parameters(theta)
    kwargs = pysipnet_overrides(table, member=2, site=27)
    assert set(kwargs) == set(table.data_vars)
    assert all(type(v) is float for v in kwargs.values())
    assert kwargs["soil_carbon"] == float(table["soil_carbon"].isel(member=2).sel(site=27))
    with pytest.raises(ValueError, match="pass member="):
        pysipnet_overrides(table, site=27)
    with pytest.raises(ValueError, match="position from 0"):
        pysipnet_overrides(table, member=-1, site=27)
    single = example.to_pysipnet_parameters(theta[0])
    assert pysipnet_overrides(single, site=1)["leaf_carbon_fraction"] == 0.466
    with pytest.raises(ValueError, match="no member dim"):
        pysipnet_overrides(single, member=0, site=1)


def niwot_parameters() -> SIPNETParameters:
    """The Niwot fixture's ``.param`` file as a ``SIPNETParameters``.

    Mirrors pySIPNET's own test helper: SIPNET names to flat field names
    through ``PYTHON_TO_SIPNET``, grouped into the sub-models.
    """
    from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file

    flat = read_param_file(NIWOT.param)
    groups: dict[str, dict[str, float]] = {}
    for python_path, sipnet_name in PYTHON_TO_SIPNET.items():
        if sipnet_name in flat:
            group, name = python_path.split(".", 1)
            groups.setdefault(group, {})[name] = flat[sipnet_name]
    return SIPNETParameters.model_validate(groups)


def with_overrides(base: SIPNETParameters, overrides: dict[str, float]) -> SIPNETParameters:
    dump = base.model_dump()
    group_of = {path.split(".", 1)[1]: path.split(".", 1)[0] for path in PARAMETER_SPECS}
    for name, value in overrides.items():
        dump[group_of[name]][name] = value
    return SIPNETParameters.model_validate(dump)


def test_override_table_validates_through_sipnet_parameters(example, theta):
    base = niwot_parameters()
    table = example.to_pysipnet_parameters(theta)
    for member in range(8):
        for site in SITES:
            params = with_overrides(base, pysipnet_overrides(table, member=member, site=site))
            assert params.photosynthesis.max_photosynthesis_rate > 0
            assert params.initial_conditions.soil_carbon == pytest.approx(
                float(table["soil_carbon"].isel(member=member).sel(site=site))
            )


@pytest.mark.slow
def test_a_prior_draw_runs_the_niwot_fixture(example, theta):
    from pysipnet import SIPNETModel, SIPNETRunner
    from pysipnet.climate import ClimateDrivers
    from pysipnet.io.clim_io import read_clim_file
    from pysipnet.parameters.model import ModelFlags

    from pysipnet.build import find_binary, missing_binary_message

    if find_binary() is None:
        pytest.skip(missing_binary_message())
    runner = SIPNETRunner(flags=ModelFlags.standard())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the fixture has a few vpd <= 0 rows
        full = read_clim_file(NIWOT.clim, n_columns=14)
    climate = ClimateDrivers.from_dataframe(full.pandas.head(8 * 30).copy(), n_columns=14)
    model = SIPNETModel(runner, base_params=niwot_parameters(), base_climate=climate)

    table = example.to_pysipnet_parameters(theta)
    result = model(**pysipnet_overrides(table, member=0, site=27))
    assert result.provenance.success, result.provenance.stderr
    nee = result.outputs["net_ecosystem_exchange"]
    assert nee.sizes["time"] == 8 * 30 and bool(np.isfinite(nee.values).all())


# ── labeled views ────────────────────────────────────────────────────────────


def test_coordinates_table_dims_and_labels(example, theta):
    t = example.coordinates_table(theta)
    assert t["photosynthesis"].dims == ("member", "element")
    assert list(t["photosynthesis"]["element"]) == ["log(capacity)", "logit(respiration_share)"]
    assert t["allocation"].dims == ("member", "pft", "element")
    assert t["base_soil_respiration"].dims == ("member", "pft")
    assert t["leaf_fall_fraction"].dims == ("member",)
    assert t["initial_soil_carbon"].dims == ("member", "site")
    assert list(t["initial_soil_carbon"]["pft"].values) == list(PFT)
    np.testing.assert_array_equal(
        t["allocation"].sel(pft="deciduous").values, theta[:, 5:8]
    )
    natural = example.coordinates_table(theta, scale="natural")
    assert list(natural["allocation"]["element"]) == list(ALLOCATION.components)
    np.testing.assert_allclose(natural["allocation"].sum("element"), 1.0)
    assert example.coordinates_table(theta[0])["allocation"].dims == ("pft", "element")
    with pytest.raises(ValueError, match="scale must be"):
        example.coordinates_table(theta, scale="physical")


def test_for_site_resolves_each_coordinate_through_the_labels(example, theta):
    t = example.coordinates_table(theta, scale="natural")
    site_27 = example.for_site(t, 27)  # conifer
    assert site_27["allocation"].dims == ("member", "element")
    np.testing.assert_array_equal(site_27["allocation"], t["allocation"].sel(pft="conifer"))
    np.testing.assert_array_equal(
        site_27["initial_soil_carbon"], t["initial_soil_carbon"].sel(site=27)
    )
    assert site_27["leaf_fall_fraction"] is t["leaf_fall_fraction"]
    with pytest.raises(KeyError, match="not one of"):
        example.for_site(t, 99)


def test_unset_parameters_and_require_complete(example):
    from sipnet_calibration.parameterization import REQUIRED_SIPNET_PARAMETERS

    # Required means "pySIPNET has no default": flag-dependent parameters and
    # the zero-defaulted ones are not required.
    assert "snow_melt_rate" not in REQUIRED_SIPNET_PARAMETERS
    assert "litter_carbon" not in REQUIRED_SIPNET_PARAMETERS
    assert "max_photosynthesis_rate" in REQUIRED_SIPNET_PARAMETERS
    assert set(example.sipnet_parameters) == set(example.to_pysipnet_parameters(example.sample(jax.random.key(0), 1)).data_vars)
    assert set(example.unset_sipnet_parameters) == set(REQUIRED_SIPNET_PARAMETERS) - set(example.sipnet_parameters)
    assert "leaf_carbon_per_area" in example.unset_sipnet_parameters
    assert not set(example.unset_sipnet_parameters) & set(example.sipnet_parameters)
    with pytest.raises(ValueError, match="neither calibrated nor fixed: \\['total_wood_carbon'"):
        Parameterization(
            coordinates=example.coordinates, fixed=example.fixed, sites=example.sites,
            labelings=example.labelings, require_complete=True,
        )
    # A vector that fixes everything it does not calibrate is complete.
    filled = tuple(
        FixedParameter(name=name, value=_in_domain_value(name), provenance="test")
        for name in example.unset_sipnet_parameters
    )
    complete = Parameterization(
        coordinates=example.coordinates, fixed=example.fixed + filled, sites=example.sites,
        labelings=example.labelings, require_complete=True,
    )
    assert complete.unset_sipnet_parameters == ()


def _in_domain_value(name: str) -> float:
    return {
        ParameterDomain.REAL: 1.0, ParameterDomain.POSITIVE: 1.0, ParameterDomain.NON_NEGATIVE: 1.0,
        ParameterDomain.UNIT_INTERVAL: 0.5, ParameterDomain.OPEN_UNIT_INTERVAL: 0.5,
    }[FLAT_SPECS[name].domain]


def test_sites_with_and_coordinate_lookup(example):
    assert example.sites_with("pft", "deciduous") == (1, 4711)
    assert example.sites_with("pft", "grassland") == ()
    assert example.coordinate("allocation").coord_to_param is ALLOCATION
    with pytest.raises(KeyError):
        example.sites_with("landcover", 1)
    with pytest.raises(KeyError):
        example.coordinate("nope")


def test_describe_has_one_row_per_column(example):
    frame = example.describe()
    assert len(frame) == 14
    assert list(frame["coordinate"]) == [example.layout.labels[i].split("[")[0] for i in range(14)]
    assert (frame["provenance"].str.len() > 0).all()
    assert set(frame["distribution"]) == {
        "product of transformed Gaussians", "softmax-normal", "log-normal", "logit-normal",
    }
    assert (frame["theta_moments"] == "analytic").all()
    soil = frame[frame["coordinate"] == "base_soil_respiration"].iloc[0]
    assert soil["natural_2.5"] == pytest.approx(0.004) and soil["natural_97.5"] == pytest.approx(0.020)
    ln = example.coordinate("base_soil_respiration").prior.distribution
    assert soil["theta_mean"] == pytest.approx(float(ln.loc)) and soil["theta_sd"] == pytest.approx(float(ln.scale))
    assert soil["sipnet_parameters"] == "base_soil_respiration_rate"
    assert np.isnan(frame[frame["coordinate"] == "allocation"]["natural_median"]).all()
    assert frame[frame["coordinate"] == "initial_soil_carbon"]["group"].tolist() == list(SITES)

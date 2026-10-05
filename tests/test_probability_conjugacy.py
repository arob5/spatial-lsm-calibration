"""Tests for the conjugate rules: a covariance scale or block integrated out
of a model, the Student-t laws that leaves, the full conditionals, and
``Posterior.theta_with``.

The scale rule, an inverse gamma on a covariance scale, is checked against
quadrature over the scale, its full conditional against the joint density
and a long random-walk Metropolis chain. The block rule, an inverse Wishart
on a covariance block,
is checked against Monte Carlo over the prior, including a block of which
the groups observe only some rows, and its full conditional against the
joint density and its closed-form moments.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.linalg
import scipy.special
import scipy.stats as st
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    INVERSE_GAMMA_SCALE,
    INVERSE_WISHART,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    ArraySpec,
    BlockDiagonalSpec,
    DenseSpec,
    DeterministicSpec,
    DiagonalSpec,
    FactorSpec,
    FullConditional,
    GaussianSpec,
    InverseWishartGivenRows,
    MatrixStudentTLaw,
    MatrixStudentTSpec,
    ScaledSpec,
    Simulator,
    SimulatorOutput,
    StudentTLaw,
    StudentTSpec,
    SubmatrixSpec,
    SumSpec,
    condition_on,
    iid_over_dim,
    independent_over_dim,
    inverse_gamma,
    inverse_wishart,
    joint,
    log_normal,
    normal,
)
from sipnet_calibration.probability.scale_mixtures import inverse_wishart_log_prob, sample_inverse_wishart

tfd = tfp.distributions

KEY = jax.random.key(8)
RNG = np.random.default_rng(20261005)

# ── the scale rule's model: three sites observing ragged times, a fourth none

LABELS = pd.MultiIndex.from_tuples(
    [(3, 0), (3, 1), (5, 0), (5, 1), (5, 2), (9, 0), (9, 1), (9, 2)], names=["site", "t"]
)
N = len(LABELS)
SITES = [3, 5, 9, 11]
VARIANCE = RNG.uniform(0.5, 2.0, N)
TIMES = np.array([0.0, 1.0, 0.0, 0.7, 1.9, 0.0, 1.2, 2.0])
OFFSETS = np.linspace(-1.0, 1.0, N)
Y = RNG.normal(size=N) + OFFSETS
SHAPE, SCALE = 3.0, 2.0


def _on_the_observations(values, labels=LABELS) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims=("obs",), coords={"obs": labels})


CONSTANTS = {"variance": _on_the_observations(VARIANCE), "t": _on_the_observations(TIMES)}


def correlated(variance, t):
    """A block: its variances on the diagonal, plus exponential correlation in time."""
    return jnp.diag(variance) + 0.4 * jnp.exp(-jnp.abs(t[:, None] - t[None, :]))


def _base(positions) -> np.ndarray:
    return np.asarray(correlated(VARIANCE[positions], TIMES[positions]))


def _site_positions(site) -> np.ndarray:
    return np.flatnonzero(LABELS.get_level_values("site") == site)


def _mu():
    return FactorSpec(ArraySpec("mu", units="1", support=REAL), law=normal(mean=0.0, standard_deviation=1.0),
                      provenance="The mean's prior.")


def _mean():
    return DeterministicSpec(ArraySpec("m", units="1", indexed_by=("obs",)), function=lambda mu, offset: mu + offset,
                             constants={"offset": _on_the_observations(OFFSETS)})


def _scale(per_site: bool, law=None):
    if law is None:
        law = iid_over_dim(inverse_gamma(shape=SHAPE, scale=SCALE)) if per_site else inverse_gamma(shape=SHAPE, scale=SCALE)
    return FactorSpec(ArraySpec("v", units="1", support=POSITIVE, indexed_by=("site",) if per_site else ()), law=law,
                      provenance="The scale's prior.")


def _y(covariance, *, constants=CONSTANTS):
    return FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                      law=GaussianSpec(mean="m", covariance=covariance), constants=constants,
                      provenance="The noise model.")


WHOLE = ScaledSpec(DenseSpec(correlated), scale="v")
BY_SITE = BlockDiagonalSpec(ScaledSpec(DenseSpec(correlated), scale="v"), by="site")


def _scale_model(covariance=BY_SITE, *, per_site=True, mean=None, scale=None, extra=(), constants=CONSTANTS, sites=SITES):
    spec = joint(_y(covariance, constants=constants), _mean() if mean is None else mean, _mu(),
                 _scale(per_site) if scale is None else scale, *extra)
    return spec.bind(coords={"obs": LABELS, "site": sites})



@functools.cache
def _per_site_model():
    """``_scale_model()``, bound once: binding checks each factor with
    thousands of draws, and the tests only read it."""
    return _scale_model()


@functools.cache
def _one_scale_model():
    """``_scale_model(per_site=False)``, bound once."""
    return _scale_model(per_site=False)

def _log_marginal_by_quadrature(y, m, covariance, shape=SHAPE, scale=SCALE) -> float:
    """log of the integral over v of IG(v; a, b) N(y; m, v R), in log v."""

    def log_integrand(u):
        v = np.exp(u)
        return st.invgamma.logpdf(v, shape, scale=scale) + st.multivariate_normal.logpdf(y, m, v * covariance) + u

    grid = np.linspace(-15.0, 15.0, 3001)
    peak = max(log_integrand(u) for u in grid)
    value, _ = scipy.integrate.quad(lambda u: np.exp(log_integrand(u) - peak), -30.0, 30.0, limit=400, epsabs=0.0, epsrel=1e-11)
    return peak + np.log(value)


def _m(mu) -> np.ndarray:
    return mu + OFFSETS


# ── the scale rule: the marginal ─────────────────────────────────────────────


def test_a_scale_per_site_integrates_to_a_student_t_per_site():
    marginal = _per_site_model().marginalize(["v"])
    law = marginal.law("y", given={"m": jnp.asarray(_m(0.3))})
    assert isinstance(law, StudentTLaw) and len(law.groups) == 3
    expected = sum(
        _log_marginal_by_quadrature(Y[p], _m(0.3)[p], _base(p)) for p in map(_site_positions, (3, 5, 9))
    )
    np.testing.assert_allclose(float(law.log_prob(Y)), expected, rtol=1e-9)


def test_a_scalar_scale_shared_by_the_sites_integrates_to_one_student_t():
    marginal = _scale_model(BY_SITE, per_site=False).marginalize(["v"])
    law = marginal.law("y", given={"m": jnp.asarray(_m(-0.4))})
    assert len(law.groups) == 1
    blocks = scipy.linalg.block_diag(*(_base(_site_positions(s)) for s in (3, 5, 9)))
    np.testing.assert_allclose(float(law.log_prob(Y)), _log_marginal_by_quadrature(Y, _m(-0.4), blocks), rtol=1e-9)


def test_a_scale_of_the_whole_covariance_integrates_to_one_student_t():
    marginal = _scale_model(WHOLE, per_site=False).marginalize(["v"])
    law = marginal.law("y", given={"m": jnp.asarray(_m(0.0))})
    dense = _base(np.arange(N))
    np.testing.assert_allclose(float(law.log_prob(Y)), _log_marginal_by_quadrature(Y, _m(0.0), dense), rtol=1e-9)


def test_the_marginal_models_density_is_the_prior_times_the_marginal():
    marginal = _per_site_model().marginalize(["v"])
    expected = st.norm.logpdf(0.3) + sum(
        _log_marginal_by_quadrature(Y[p], _m(0.3)[p], _base(p)) for p in map(_site_positions, (3, 5, 9))
    )
    np.testing.assert_allclose(float(marginal.log_prob({"mu": 0.3, "y": Y})), expected, rtol=1e-9)


def test_per_label_shapes_and_scales_are_read_at_each_groups_label():
    shapes = xr.DataArray([2.0, 3.0, 4.0, 5.0], dims=("site",), coords={"site": SITES})
    scales = xr.DataArray([1.0, 2.0, 0.5, 3.0], dims=("site",), coords={"site": SITES})
    prior = FactorSpec(ArraySpec("v", units="1", support=POSITIVE, indexed_by=("site",)),
                       law=independent_over_dim(inverse_gamma), constants={"shape": shapes, "scale": scales})
    law = _scale_model(scale=prior).marginalize(["v"]).law("y", given={"m": jnp.asarray(_m(0.0))})
    expected = sum(
        _log_marginal_by_quadrature(Y[p], _m(0.0)[p], _base(p), shape=a, scale=b)
        for p, a, b in zip(map(_site_positions, (3, 5, 9)), (2.0, 3.0, 4.0), (1.0, 2.0, 0.5))
    )
    np.testing.assert_allclose(float(law.log_prob(Y)), expected, rtol=1e-9)


def test_draws_of_the_student_t_have_its_mean_and_covariance():
    marginal = _scale_model(WHOLE, per_site=False).marginalize(["v"])
    law = marginal.law("y", given={"m": jnp.asarray(_m(0.5))})
    draws = np.asarray(law.sample(200_000, seed=KEY))
    np.testing.assert_allclose(draws.mean(axis=0), _m(0.5), atol=0.02)
    # A t of 2a degrees of freedom and scale (b / a) R has covariance b R / (a - 1).
    np.testing.assert_allclose(np.cov(draws.T), SCALE / (SHAPE - 1.0) * _base(np.arange(N)), atol=0.05)


def test_draws_of_a_student_t_per_site_are_uncorrelated_across_sites():
    law = _per_site_model().marginalize(["v"]).law("y", given={"m": jnp.zeros(N)})
    draws = np.asarray(law.sample(200_000, seed=KEY))
    expected = SCALE / (SHAPE - 1.0) * scipy.linalg.block_diag(*(_base(_site_positions(s)) for s in (3, 5, 9)))
    np.testing.assert_allclose(np.cov(draws.T), expected, atol=0.05)


def test_the_marginal_model_drops_the_scale_and_keeps_its_place_in_order():
    model = _per_site_model()
    marginal = model.marginalize(["v"])
    assert marginal.spec.component_names == ("y", "m", "mu")
    table = marginal.describe()
    assert table.loc["y", "law"] == "Student-t" and table.loc["y", "evaluated_by"] == "Student-t"
    (y,) = [p for p in marginal.spec.parts if p.name == "y"]
    assert isinstance(y.law, StudentTSpec) and y.law.by == "site"
    assert "The noise model." in y.provenance and "The scale's prior." in y.provenance
    assert "inverse gamma scale" in y.provenance


def test_a_marginal_selected_at_fewer_sites_is_the_selection_marginalized():
    marginal = _per_site_model().marginalize(["v"]).select(site=[3, 9])
    selected = _per_site_model().select(site=[3, 9]).marginalize(["v"])
    given = {"m": jnp.asarray(_m(0.2)[np.r_[_site_positions(3), _site_positions(9)]])}
    values = Y[np.r_[_site_positions(3), _site_positions(9)]]
    np.testing.assert_allclose(
        float(marginal.law("y", given=given).log_prob(values)), float(selected.law("y", given=given).log_prob(values)),
        rtol=1e-12,
    )


def test_conditioning_the_marginal_gives_a_likelihood_that_is_not_gaussian():
    posterior = condition_on(_per_site_model().marginalize(["v"]), {"y": Y})
    assert posterior.parameter_names == ("mu",)
    theta = jnp.asarray([[0.3]])
    law = posterior.model.law("y", given={"m": jnp.asarray(_m(0.3))})
    np.testing.assert_allclose(np.asarray(posterior.log_likelihood(theta)), [float(law.log_prob(Y))], rtol=1e-12)
    with pytest.raises(ValueError, match="not a GaussianSpec"):
        posterior.gaussian_likelihood()


def test_a_student_t_factor_can_be_a_parameter_drawn_and_scored():
    marginal = _per_site_model().marginalize(["v"])
    draws = marginal.sample(KEY, 5)
    assert draws["y"].shape == (5, N) and np.all(np.isfinite(np.asarray(draws["y"])))
    posterior = condition_on(marginal, {})
    theta = posterior.sample_prior(KEY, 3)
    assert np.all(np.isfinite(np.asarray(posterior.log_prior(theta))))


# ── the scale rule: what it refuses ──────────────────────────────────────────


def test_a_scale_whose_law_is_not_an_inverse_gamma_has_no_rule():
    model = _scale_model(scale=_scale(True, law=iid_over_dim(log_normal(median=1.0, geometric_sd=2.0))))
    with pytest.raises(ValueError, match="neither an inverse gamma"):
        model.marginalize(["v"])


def test_a_scale_whose_prior_reads_a_component_has_no_rule():
    prior = FactorSpec(ArraySpec("v", units="1", support=POSITIVE),
                       law=lambda mu: inverse_gamma(shape=SHAPE, scale=jnp.exp(mu)))
    with pytest.raises(ValueError, match="reads \\['mu'\\]"):
        _scale_model(BY_SITE, per_site=False, scale=prior).marginalize(["v"])


def test_a_scale_read_by_two_parts_has_no_rule():
    twice = DeterministicSpec(ArraySpec("w", units="1"), function=lambda v: 2.0 * v)
    with pytest.raises(ValueError, match="read by 2 parts"):
        _scale_model(BY_SITE, per_site=False, extra=(twice,)).marginalize(["v"])


def test_a_scale_read_by_a_law_that_is_not_gaussian_has_no_rule():
    z = FactorSpec(ArraySpec("z", units="1", support=REAL), law=lambda v: normal(mean=0.0, standard_deviation=jnp.sqrt(v)))
    spec = joint(z, _scale(False))
    with pytest.raises(ValueError, match="not a GaussianSpec"):
        spec.bind(coords={}).marginalize(["v"])


def test_a_scale_read_other_than_as_a_scale_has_no_rule():
    model = _scale_model(DiagonalSpec("v"), per_site=False, constants={})
    with pytest.raises(ValueError, match="the scale rule needs ScaledSpec"):
        model.marginalize(["v"])


def test_a_prior_reading_only_an_input_is_fixed_numbers():
    prior = FactorSpec(ArraySpec("v", units="1", support=POSITIVE), law=lambda b: inverse_gamma(shape=SHAPE, scale=b))
    spec = joint(_y(BY_SITE), _mean(), _mu(), prior, inputs=[ArraySpec("b", units="1", support=POSITIVE)])
    model = spec.bind(coords={"obs": LABELS, "site": SITES}, inputs={"b": SCALE})
    reference = _one_scale_model().marginalize(["v"]).law("y", given={"m": jnp.zeros(N)})
    law = model.marginalize(["v"]).law("y", given={"m": jnp.zeros(N)})
    np.testing.assert_allclose(float(law.log_prob(Y)), float(reference.log_prob(Y)), rtol=1e-12)


def test_a_component_whose_law_is_no_prior_of_a_rule_is_refused_by_its_law():
    with pytest.raises(ValueError, match="neither an inverse gamma"):
        _per_site_model().marginalize(["y"])


def test_a_scale_that_is_also_the_mean_has_no_rule():
    y = FactorSpec(ArraySpec("y", units="1", support=REAL),
                   law=GaussianSpec(mean="v", covariance=ScaledSpec(DiagonalSpec(lambda: jnp.ones(1)), scale="v")))
    model = joint(y, _scale(False)).bind(coords={})
    with pytest.raises(ValueError, match="centered on it"):
        model.marginalize(["v"])


def test_marginalizing_nothing_a_computed_component_or_an_unknown_one_is_refused():
    model = _per_site_model()
    with pytest.raises(ValueError, match="more than once"):
        model.marginalize(["v", "v"])
    with pytest.raises(ValueError, match="is empty"):
        model.marginalize([])
    with pytest.raises(ValueError, match="not by a factor of its own"):
        model.marginalize(["m"])
    with pytest.raises(KeyError):
        model.marginalize(["nope"])
    with pytest.raises(TypeError):
        model.marginalize("v")


# ── the scale rule: the full conditional ─────────────────────────────────────


def _evaluation(posterior, n=4):
    theta = posterior.sample_prior(jax.random.key(3), n)
    return posterior.evaluate(theta)


def test_the_scales_full_conditional_is_the_joint_density_in_the_scale():
    model = _per_site_model()
    posterior = condition_on(model, {"y": Y})
    conditional = posterior.full_conditional("v")
    assert isinstance(conditional, FullConditional) and conditional.rule == INVERSE_GAMMA_SCALE
    evaluation = _evaluation(posterior)
    mu = np.asarray(posterior.natural_values(evaluation.theta)["mu"])
    candidates = RNG.uniform(0.2, 3.0, size=(5, len(SITES)))
    for j in range(len(mu)):
        law = conditional.law(evaluation, j)
        joint_density = np.asarray([float(model.log_prob({"mu": mu[j], "v": v, "y": Y})) for v in candidates])
        conditional_density = np.asarray(law.log_prob(jnp.asarray(candidates)))
        np.testing.assert_allclose(np.diff(conditional_density), np.diff(joint_density), rtol=1e-9, atol=1e-9)


def test_the_scales_full_conditional_has_the_closed_form_parameters():
    posterior = condition_on(_per_site_model(), {"y": Y})
    evaluation = _evaluation(posterior, n=2)
    mu = float(posterior.natural_values(evaluation.theta)["mu"][1])
    law = posterior.full_conditional("v").law(evaluation, 1)
    residual = Y - _m(mu)
    shapes, scales = [], []
    for site in SITES:
        p = _site_positions(site)
        q = residual[p] @ np.linalg.solve(_base(p), residual[p]) if len(p) else 0.0
        shapes.append(SHAPE + len(p) / 2.0)
        scales.append(SCALE + q / 2.0)
    np.testing.assert_allclose(np.asarray(law.distribution.concentration), shapes, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(law.distribution.scale), scales, rtol=1e-10)


def _quadratic_forms(mu) -> dict[int, tuple[int, float]]:
    """Each observing site's entries and residual quadratic form at ``mu``."""
    residual = Y - _m(mu)
    out = {}
    for site in (3, 5, 9):
        p = _site_positions(site)
        out[site] = (len(p), float(residual[p] @ np.linalg.solve(_base(p), residual[p])))
    return out


def test_one_scale_shared_by_the_sites_sums_their_entries_and_forms():
    posterior = condition_on(_one_scale_model(), {"y": Y})
    evaluation = _evaluation(posterior, n=2)
    mu = float(posterior.natural_values(evaluation.theta)["mu"][0])
    law = posterior.full_conditional("v").law(evaluation, 0)
    forms = _quadratic_forms(mu).values()
    np.testing.assert_allclose(float(law.concentration), SHAPE + sum(n for n, _ in forms) / 2.0, rtol=1e-12)
    np.testing.assert_allclose(float(law.scale), SCALE + sum(q for _, q in forms) / 2.0, rtol=1e-10)


def test_each_sites_entries_and_form_go_to_its_label_whatever_the_coords_order():
    sites = [11, 9, 3, 5]
    posterior = condition_on(_scale_model(sites=sites), {"y": Y})
    evaluation = _evaluation(posterior, n=1)
    mu = float(posterior.natural_values(evaluation.theta)["mu"][0])
    law = posterior.full_conditional("v").law(evaluation, 0)
    forms = _quadratic_forms(mu)
    expected_shape = [SHAPE + forms[s][0] / 2.0 if s in forms else SHAPE for s in sites]
    expected_scale = [SCALE + forms[s][1] / 2.0 if s in forms else SCALE for s in sites]
    np.testing.assert_allclose(np.asarray(law.distribution.concentration), expected_shape, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(law.distribution.scale), expected_scale, rtol=1e-10)


def test_a_site_observing_nothing_keeps_its_prior():
    posterior = condition_on(_per_site_model(), {"y": Y})
    law = posterior.full_conditional("v").law(_evaluation(posterior), 0)
    assert float(law.distribution.concentration[3]) == SHAPE and float(law.distribution.scale[3]) == SCALE


def test_the_scales_full_conditional_matches_a_long_metropolis_chain():
    model = _scale_model(BY_SITE, per_site=False)
    posterior = condition_on(model, {"y": Y})
    evaluation = _evaluation(posterior, n=1)
    mu = float(posterior.natural_values(evaluation.theta)["mu"][0])
    draws = np.asarray(posterior.full_conditional("v").law(evaluation, 0).sample(200_000, seed=KEY))

    @jax.jit
    def log_target(u):
        return model.log_prob({"mu": jnp.float64(mu), "v": jnp.exp(u), "y": jnp.asarray(Y)}) + u

    def step(state, key):
        u, current = state
        proposal_key, accept_key = jax.random.split(key)
        proposed = u + 0.6 * jax.random.normal(proposal_key, dtype=jnp.float64)
        log_proposed = log_target(proposed)
        accept = jnp.log(jax.random.uniform(accept_key, dtype=jnp.float64)) < log_proposed - current
        u, current = jnp.where(accept, proposed, u), jnp.where(accept, log_proposed, current)
        return (u, current), jnp.exp(u)

    u0 = jnp.float64(0.0)
    _, chain = jax.lax.scan(step, (u0, log_target(u0)), jax.random.split(jax.random.key(5), 60_000))
    chain = np.asarray(chain)[5_000:]
    np.testing.assert_allclose(chain.mean(), draws.mean(), rtol=0.03)
    np.testing.assert_allclose(np.quantile(chain, [0.1, 0.5, 0.9]), np.quantile(draws, [0.1, 0.5, 0.9]), rtol=0.04)


def test_the_full_conditional_draws_one_value_per_sample():
    posterior = condition_on(_per_site_model(), {"y": Y})
    evaluation = _evaluation(posterior, n=6)
    draws = posterior.full_conditional("v").sample(KEY, evaluation)
    assert draws.shape == (6, len(SITES)) and np.all(np.asarray(draws) > 0)
    scalar = condition_on(_one_scale_model(), {"y": Y})
    assert scalar.full_conditional("v").sample(KEY, _evaluation(scalar, n=6)).shape == (6,)


class Shift(Simulator):
    """``m = mu + offsets``, failing where ``mu`` exceeds 2; counts its calls."""

    name = "shift"
    given = ("mu",)

    def __init__(self):
        self.calls = 0

    @property
    def outputs(self):
        return (ArraySpec("m", units="1", indexed_by=("obs",)),)

    def __call__(self, given_values):
        self.calls += 1
        mu = given_values["mu"].values
        return SimulatorOutput(values={"m": mu[:, None] + OFFSETS}, valid={"m": mu <= 2.0})

    def at(self, coords, outputs):
        return self


def test_a_full_conditional_reads_a_simulator_output_from_the_evaluation_and_is_nan_where_it_failed():
    posterior = condition_on(_scale_model(mean=Shift()), {"y": Y})
    theta = jnp.asarray([[0.5, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0]])
    evaluation = posterior.evaluate(theta)
    assert list(np.asarray(evaluation.simulator_valid)) == [True, False]
    draws = np.asarray(posterior.full_conditional("v").sample(KEY, evaluation))
    assert np.all(np.isfinite(draws[0])) and np.all(np.isnan(draws[1, :3]))
    reference = condition_on(_per_site_model(), {"y": Y}).full_conditional("v")
    reference_evaluation = condition_on(_per_site_model(), {"y": Y}).evaluate(theta[:1])
    np.testing.assert_allclose(
        np.asarray(posterior.full_conditional("v").law(evaluation, 0).distribution.scale),
        np.asarray(reference.law(reference_evaluation, 0).distribution.scale), rtol=1e-12,
    )


def test_a_full_conditional_is_a_parameters():
    posterior = condition_on(_per_site_model(), {"y": Y})
    with pytest.raises(ValueError, match="not a parameter"):
        posterior.full_conditional("y")
    with pytest.raises(KeyError):
        posterior.full_conditional("nope")
    with pytest.raises(ValueError, match="neither an inverse gamma"):
        posterior.full_conditional("mu")
    with pytest.raises(TypeError):
        posterior.full_conditional(["v"])


def test_a_full_conditional_refuses_another_posteriors_evaluation_and_a_bad_row():
    posterior = condition_on(_per_site_model(), {"y": Y})
    conditional = posterior.full_conditional("v")
    evaluation = _evaluation(posterior, n=2)
    with pytest.raises(TypeError):
        conditional.law("not an evaluation", 0)
    with pytest.raises(ValueError, match="rows 0 to 1"):
        conditional.law(evaluation, 2)
    with pytest.raises(TypeError):
        conditional.law(evaluation, 0.0)
    other = condition_on(_one_scale_model(), {"y": Y})
    with pytest.raises(ValueError, match="give a batch"):
        conditional.sample(KEY, _evaluation(other, n=2))


# ── theta_with ───────────────────────────────────────────────────────────────


def test_theta_with_sets_a_parameter_through_its_bijection():
    posterior = condition_on(_per_site_model(), {"y": Y})
    theta = posterior.sample_prior(KEY, 3)
    values = np.asarray([[0.5, 1.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0], [3.0, 0.1, 0.2, 0.3]])
    updated = posterior.theta_with(theta, {"v": values})
    natural = posterior.natural_values(updated)
    np.testing.assert_allclose(np.asarray(natural["v"]), values, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(natural["mu"]), np.asarray(posterior.natural_values(theta)["mu"]))


def test_theta_with_gives_every_row_one_block_and_is_traceable():
    posterior = condition_on(_per_site_model(), {"y": Y})
    theta = jnp.zeros((2, posterior.dimension))
    block = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    updated = jax.jit(lambda t, v: posterior.theta_with(t, {"v": v}))(theta, block)
    np.testing.assert_allclose(np.asarray(posterior.natural_values(updated)["v"]), np.tile(block, (2, 1)), rtol=1e-12)


def test_theta_with_refuses_a_batch_that_does_not_fit_and_a_boolean():
    posterior = condition_on(_per_site_model(), {"y": Y})
    theta = jnp.zeros((2, posterior.dimension))
    with pytest.raises(ValueError, match="does not broadcast"):
        posterior.theta_with(theta, {"v": jnp.ones((3, 4))})
    with pytest.raises(TypeError, match="not a number"):
        posterior.theta_with(theta, {"mu": True})


def test_theta_with_refuses_what_is_not_a_parameter_or_has_no_theta():
    posterior = condition_on(_per_site_model(), {"y": Y})
    theta = jnp.zeros((2, posterior.dimension))
    with pytest.raises(KeyError, match="not a parameter"):
        posterior.theta_with(theta, {"y": Y})
    with pytest.raises(ValueError, match="has no theta"):
        posterior.theta_with(theta, {"v": jnp.asarray([1.0, -1.0, 1.0, 1.0])})
    with pytest.raises(ValueError, match="block shape"):
        posterior.theta_with(theta, {"v": jnp.ones(3)})
    with pytest.raises(TypeError):
        posterior.theta_with(theta, [("v", 1.0)])


def test_a_gibbs_step_draws_the_scale_without_running_the_simulator_again():
    simulator = Shift()
    posterior = condition_on(_scale_model(mean=simulator), {"y": Y})
    theta = posterior.theta_with(posterior.sample_prior(KEY, 4), {"mu": jnp.asarray([-1.0, 0.0, 0.5, 1.0])})
    evaluation = posterior.evaluate(theta)
    calls = simulator.calls
    draw = posterior.full_conditional("v").sample(KEY, evaluation)
    assert simulator.calls == calls
    updated = posterior.theta_with(theta, {"v": draw})
    np.testing.assert_allclose(np.asarray(posterior.natural_values(updated)["v"]), np.asarray(draw), rtol=1e-12)


# ── the block rule's model: an inverse Wishart over three years ─────────────

YEARS = ("2012", "2013", "2014")
PSI = 3.0 * np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.8], [0.5, 0.8, 2.0]])
NU = 7.0


def _year_labels(observed) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([(site, year) for site, years in observed.items() for year in years],
                                     names=["site", "season"])


#: Every site observes 2012 and 2014, site 5 in the other order.
SOME_ROWS = _year_labels({3: ("2012", "2014"), 5: ("2014", "2012"), 9: ("2012", "2014")})
EVERY_ROW = _year_labels({3: YEARS, 5: YEARS, 9: YEARS})
RAGGED = _year_labels({3: YEARS, 5: ("2012", "2014"), 9: YEARS})


def _matrix_model(labels, covariance=None):
    year_of = xr.DataArray(np.asarray(labels.get_level_values("season"), dtype=object), dims=("obs",),
                           coords={"obs": labels}, name="year")
    n = len(labels)
    offsets = np.linspace(-0.5, 0.5, n)
    submatrix = BlockDiagonalSpec(SubmatrixSpec("S", label_map="year_of"), by="site")
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                   law=GaussianSpec(mean="m", covariance=submatrix if covariance is None else covariance),
                   label_maps={"year_of": year_of}),
        DeterministicSpec(ArraySpec("m", units="1", indexed_by=("obs",)), function=lambda mu: mu + jnp.asarray(offsets)),
        _mu(),
        FactorSpec(ArraySpec("S", units="1", support=POSITIVE_DEFINITE, element_axes={"year": YEARS, "other_year": YEARS}),
                   law=inverse_wishart(degrees_of_freedom=NU, scale=PSI), provenance="The block's prior."),
    )
    return spec.bind(coords={"obs": labels})


@functools.cache
def _block_model(name):
    """``_matrix_model`` at the labels called *name*, bound once."""
    return _matrix_model({"EVERY_ROW": EVERY_ROW, "SOME_ROWS": SOME_ROWS, "RAGGED": RAGGED}[name])


def _rows(labels):
    """Each site's entries, in its rows' order, and the rows."""
    years = np.asarray([YEARS.index(y) for y in labels.get_level_values("season")])
    sites = labels.get_level_values("site")
    groups = [np.flatnonzero(sites == s) for s in dict.fromkeys(sites)]
    groups = [g[np.argsort(years[g])] for g in groups]
    return groups, np.sort(years[groups[0]])


def _y_for(labels) -> np.ndarray:
    return np.random.default_rng(4).normal(size=len(labels)) * 2.0


def test_an_inverse_wishart_block_integrates_to_a_matrix_student_t():
    marginal = _block_model("EVERY_ROW").marginalize(["S"])
    law = marginal.law("y", given={"m": jnp.zeros(len(EVERY_ROW))})
    assert isinstance(law, MatrixStudentTLaw)
    y = _y_for(EVERY_ROW)
    groups, _ = _rows(EVERY_ROW)
    residuals = np.stack([y[g] for g in groups])
    draws = st.invwishart.rvs(df=NU, scale=PSI, size=400_000, random_state=1)
    log_likelihood = sum(_log_normal_rows(r, draws) for r in residuals)
    expected = scipy.special.logsumexp(log_likelihood) - np.log(len(draws))
    np.testing.assert_allclose(float(law.log_prob(y)), expected, atol=0.01)


def test_a_block_observed_on_some_rows_integrates_with_fewer_degrees_of_freedom():
    marginal = _block_model("SOME_ROWS").marginalize(["S"])
    law = marginal.law("y", given={"m": jnp.zeros(len(SOME_ROWS))})
    y = _y_for(SOME_ROWS)
    groups, rows = _rows(SOME_ROWS)
    residuals = np.stack([y[g] for g in groups])
    draws = st.invwishart.rvs(df=NU, scale=PSI, size=400_000, random_state=2)[:, rows][:, :, rows]
    log_likelihood = sum(_log_normal_rows(r, draws) for r in residuals)
    expected = scipy.special.logsumexp(log_likelihood) - np.log(len(draws))
    np.testing.assert_allclose(float(law.log_prob(y)), expected, atol=0.01)
    assert law.degrees_of_freedom == NU - 1.0


def _log_normal_rows(residual, covariances) -> np.ndarray:
    """log N(residual; 0, C) for each C."""
    lower = np.linalg.cholesky(covariances)
    solved = np.linalg.solve(lower, np.broadcast_to(residual, covariances.shape[:-1])[..., None])[..., 0]
    log_det = 2.0 * np.log(np.diagonal(lower, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (len(residual) * np.log(2 * np.pi) + log_det + (solved**2).sum(axis=-1))


def test_draws_of_the_matrix_student_t_have_the_blocks_prior_mean_as_covariance():
    law = _block_model("SOME_ROWS").marginalize(["S"]).law("y", given={"m": jnp.zeros(len(SOME_ROWS))})
    draws = np.asarray(law.sample(200_000, seed=KEY))
    groups, rows = _rows(SOME_ROWS)
    q, p = len(rows), len(YEARS)
    expected = PSI[np.ix_(rows, rows)] / (NU - (p - q) - q - 1.0)
    for group in groups:
        np.testing.assert_allclose(np.cov(draws[:, group].T), expected, rtol=0.05)


def test_groups_mapping_to_different_rows_are_refused_naming_them():
    with pytest.raises(ValueError, match=r"different rows: \['\(5,\)'\] differ"):
        _block_model("RAGGED").marginalize(["S"])
    posterior = condition_on(_block_model("RAGGED"), {"y": _y_for(RAGGED)})
    with pytest.raises(ValueError, match="different rows"):
        posterior.full_conditional("S")


def test_a_term_added_to_the_submatrix_is_refused():
    covariance = BlockDiagonalSpec(SumSpec(SubmatrixSpec("S", label_map="year_of"), DiagonalSpec(lambda: 0.1)), by="site")
    with pytest.raises(ValueError, match="nothing added"):
        _matrix_model(SOME_ROWS, covariance).marginalize(["S"])


def test_the_matrix_marginal_drops_the_block():
    marginal = _block_model("SOME_ROWS").marginalize(["S"])
    assert marginal.spec.component_names == ("y", "m", "mu")
    assert marginal.describe().loc["y", "law"] == "matrix Student-t"


def test_the_blocks_full_conditional_is_the_joint_density_in_the_block():
    for name, labels in (("EVERY_ROW", EVERY_ROW), ("SOME_ROWS", SOME_ROWS)):
        model = _block_model(name)
        y = _y_for(labels)
        posterior = condition_on(model, {"y": y})
        conditional = posterior.full_conditional("S")
        assert conditional.rule == INVERSE_WISHART
        evaluation = _evaluation(posterior, n=2)
        mu = np.asarray(posterior.natural_values(evaluation.theta)["mu"])
        candidates = st.invwishart.rvs(df=NU, scale=PSI, size=5, random_state=3)
        for j in range(2):
            law = conditional.law(evaluation, j)
            assert type(law) is InverseWishartGivenRows
            joint_density = np.asarray([float(model.log_prob({"mu": mu[j], "S": s, "y": y})) for s in candidates])
            conditional_density = np.asarray(law.log_prob(jnp.asarray(candidates)))
            np.testing.assert_allclose(np.diff(conditional_density), np.diff(joint_density), rtol=1e-9, atol=1e-8)


def _posterior_scatter(labels, posterior, evaluation, j):
    mu = float(posterior.natural_values(evaluation.theta)["mu"][j])
    residual = _y_for(labels) - (mu + np.linspace(-0.5, 0.5, len(labels)))
    groups, rows = _rows(labels)
    stacked = np.stack([residual[g] for g in groups])
    return stacked.T @ stacked, len(groups), rows


def test_the_blocks_full_conditional_draws_have_the_posterior_mean():
    posterior = condition_on(_block_model("EVERY_ROW"), {"y": _y_for(EVERY_ROW)})
    evaluation = _evaluation(posterior, n=1)
    law = posterior.full_conditional("S").law(evaluation, 0)
    draws = np.asarray(law.sample(100_000, seed=KEY))
    scatter, n, _ = _posterior_scatter(EVERY_ROW, posterior, evaluation, 0)
    expected = (PSI + scatter) / (NU + n - 3 - 1.0)
    np.testing.assert_allclose(draws.mean(axis=0), expected, rtol=0.03, atol=0.02)


def test_the_unobserved_rows_of_the_block_are_drawn_from_the_prior_given_the_observed():
    posterior = condition_on(_block_model("SOME_ROWS"), {"y": _y_for(SOME_ROWS)})
    evaluation = _evaluation(posterior, n=1)
    draws = np.asarray(posterior.full_conditional("S").law(evaluation, 0).sample(200_000, seed=KEY))
    scatter, n, rows = _posterior_scatter(SOME_ROWS, posterior, evaluation, 0)
    rest = np.setdiff1d(np.arange(3), rows)
    p, q = 3, len(rows)
    psi_oo, psi_or, psi_rr = PSI[np.ix_(rows, rows)], PSI[np.ix_(rows, rest)], PSI[np.ix_(rest, rest)]
    observed_mean = (psi_oo + scatter) / (NU - (p - q) + n - q - 1.0)
    coefficients = np.linalg.solve(psi_oo, psi_or)
    conditional_mean = (psi_rr - psi_or.T @ coefficients) / (NU - (p - q) - 1.0)
    rest_mean = (
        conditional_mean + coefficients.T @ observed_mean @ coefficients
        + np.trace(observed_mean @ np.linalg.inv(psi_oo)) * conditional_mean
    )
    np.testing.assert_allclose(draws[:, rows][:, :, rows].mean(axis=0), observed_mean, rtol=0.03)
    np.testing.assert_allclose(draws[:, rows][:, :, rest].mean(axis=0), observed_mean @ coefficients, rtol=0.05, atol=0.02)
    np.testing.assert_allclose(draws[:, rest][:, :, rest].mean(axis=0), rest_mean, rtol=0.05)


def test_the_blocks_full_conditional_is_nan_where_the_mean_was_not_computed():
    year_of = xr.DataArray(np.asarray(EVERY_ROW.get_level_values("season"), dtype=object), dims=("obs",),
                           coords={"obs": EVERY_ROW}, name="year")
    n = len(EVERY_ROW)

    class Mean(Simulator):
        name = "mean"
        given = ("mu",)
        outputs = (ArraySpec("m", units="1", indexed_by=("obs",)),)

        def __call__(self, given_values):
            mu = given_values["mu"].values
            return SimulatorOutput(values={"m": np.repeat(mu[:, None], n, axis=1)}, valid={"m": mu <= 2.0})

        def at(self, coords, outputs):
            return self

    spec = joint(
        FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                   law=GaussianSpec(mean="m", covariance=BlockDiagonalSpec(SubmatrixSpec("S", label_map="year_of"), by="site")),
                   label_maps={"year_of": year_of}),
        Mean(), _mu(),
        FactorSpec(ArraySpec("S", units="1", support=POSITIVE_DEFINITE, element_axes={"year": YEARS, "other_year": YEARS}),
                   law=inverse_wishart(degrees_of_freedom=NU, scale=PSI)),
    )
    posterior = condition_on(spec.bind(coords={"obs": EVERY_ROW}), {"y": _y_for(EVERY_ROW)})
    theta = posterior.theta_with(posterior.sample_prior(KEY, 2), {"mu": jnp.asarray([0.5, 3.0])})
    evaluation = posterior.evaluate(theta)
    conditional = posterior.full_conditional("S")
    assert np.isnan(float(conditional.law(evaluation, 1).log_prob(jnp.asarray(PSI))))
    assert np.isfinite(float(conditional.law(evaluation, 0).log_prob(jnp.asarray(PSI))))
    draws = np.asarray(conditional.sample(KEY, evaluation))
    assert np.all(np.isfinite(draws[0])) and np.all(np.isnan(draws[1]))


def test_the_blocks_full_conditional_draws_one_matrix_per_sample():
    for name, labels in (("EVERY_ROW", EVERY_ROW), ("SOME_ROWS", SOME_ROWS)):
        posterior = condition_on(_block_model(name), {"y": _y_for(labels)})
        draws = np.asarray(posterior.full_conditional("S").sample(KEY, _evaluation(posterior, n=5)))
        assert draws.shape == (5, 3, 3)
        assert np.all(np.linalg.eigvalsh(draws) > 0) and np.allclose(draws, np.swapaxes(draws, -1, -2))


# ── the laws and specs ───────────────────────────────────────────────────────


def test_the_inverse_wishart_density_is_scipys_and_the_families():
    matrices = st.invwishart.rvs(df=NU, scale=PSI, size=4, random_state=6)
    np.testing.assert_allclose(
        np.asarray(inverse_wishart_log_prob(matrices, NU, PSI)), st.invwishart.logpdf(matrices.transpose(1, 2, 0), df=NU, scale=PSI),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(inverse_wishart_log_prob(matrices, NU, PSI)), np.asarray(inverse_wishart(degrees_of_freedom=NU, scale=PSI).log_prob(matrices)),
        rtol=1e-10,
    )


def test_inverse_wishart_draws_have_its_mean():
    draws = np.asarray(sample_inverse_wishart(KEY, NU, PSI, (100_000,)))
    np.testing.assert_allclose(draws.mean(axis=0), PSI / (NU - 3 - 1.0), rtol=0.03)


def test_the_log_densities_are_nan_for_a_scale_that_is_not_positive_definite():
    indefinite = np.diag([1.0, -1.0, 1.0])
    assert np.isnan(float(inverse_wishart_log_prob(PSI, NU, indefinite)))
    law = MatrixStudentTLaw(jnp.zeros(3), np.array([[0, 1, 2]]), NU, indefinite)
    assert np.isnan(float(law.log_prob(jnp.zeros(3))))


def test_a_student_t_spec_takes_a_zero_dimensional_array_and_refuses_repeated_labels():
    spec = StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape=np.array(2.0), scale=jnp.float64(1.0))
    assert float(spec.shape) == 2.0 and float(spec.scale) == 1.0
    repeated = xr.DataArray([1.0, 2.0], dims=("site",), coords={"site": [3, 3]})
    with pytest.raises(ValueError, match="twice"):
        StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape=repeated, scale=1.0, by="site")


def test_a_student_t_spec_refuses_scales_that_are_not_positive_or_on_another_dim():
    with pytest.raises(ValueError, match="positive"):
        StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape=-1.0, scale=1.0)
    with pytest.raises(ValueError, match="grouping level"):
        StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape=xr.DataArray([1.0], dims=("year",), coords={"year": [1]}),
                     scale=1.0, by="site")
    with pytest.raises(TypeError):
        StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape="a", scale=1.0)
    with pytest.raises(TypeError):
        StudentTSpec(mean=["m"], covariance=DenseSpec(correlated), shape=1.0, scale=1.0)


def test_a_student_t_lacking_a_groups_shape_is_refused_when_bound():
    shape = xr.DataArray([2.0, 2.0], dims=("site",), coords={"site": [3, 5]})
    law = StudentTSpec(mean="m", covariance=DenseSpec(correlated), shape=shape, scale=1.0, by="site")
    spec = joint(FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)), law=law, constants=CONSTANTS),
                 _mean(), _mu())
    with pytest.raises(KeyError, match="no shape at the group 9"):
        spec.bind(coords={"obs": LABELS})


def test_a_matrix_student_t_spec_refuses_a_bad_scale_or_too_few_degrees_of_freedom():
    year_of = xr.DataArray(np.asarray(SOME_ROWS.get_level_values("season"), dtype=object), dims=("obs",),
                           coords={"obs": SOME_ROWS}, name="year")
    scale = xr.DataArray(PSI, dims=("year", "other_year"), coords={"year": list(YEARS), "other_year": list(YEARS)})
    with pytest.raises(ValueError, match="more than 2"):
        MatrixStudentTSpec(mean="m", by="site", label_map=year_of, degrees_of_freedom=2.0, scale=scale)
    with pytest.raises(ValueError, match="positive-definite"):
        MatrixStudentTSpec(mean="m", by="site", label_map=year_of, degrees_of_freedom=NU, scale=-scale)
    with pytest.raises(ValueError, match="named for the rows"):
        MatrixStudentTSpec(mean="m", by="site", label_map=year_of.rename("other_year"), degrees_of_freedom=NU, scale=scale)

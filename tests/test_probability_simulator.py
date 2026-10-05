"""Tests for the simulator seam: a part computed outside JAX for a batch of
samples, which may fail at some of them.

A toy simulator squares what it reads, fails at a sample whose value is
large, and records every call. The model runs it once per batch, never at
bind, never under a trace, and only for the outputs the likelihood (or a
prediction) needs; a failed sample truncates the posterior, scoring
``-inf``; and holding its outputs gives a traced density in the entries of
theta it does not read.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    POSITIVE,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    PosteriorEvaluation,
    Simulator,
    SimulatorOutput,
    condition_on,
    iid_over_dim,
    joint,
    log_normal,
)
from sipnet_calibration.probability._probes import CORNERS, corner_points

tfd = tfp.distributions

SITES = [10, 20]
Y = np.array([1.0, 2.0])
#: A sample at which any value of ``x`` reaches this fails.
FAILS_AT = 5.0
KEY = jax.random.key(7)


class Square(Simulator):
    """``m = x^2`` and ``m2 = x^2 + 1`` per site, failing where ``x`` reaches
    :data:`FAILS_AT`. Every call and every ``check_given`` is recorded."""

    def __init__(self, sites=SITES, outputs=("m", "m2"), log=None):
        self._sites = list(sites)
        self._outputs = tuple(outputs)
        self.log = {"calls": [], "checks": [], "at": []} if log is None else log

    @property
    def name(self):
        return "square"

    @property
    def given(self):
        return ("x",)

    @property
    def outputs(self):
        return tuple(ArraySpec(o, units="1", indexed_by=("site",)) for o in self._outputs)

    def __call__(self, given_values):
        x = given_values["x"]
        self.log["calls"].append((x.sizes["sample"], self._outputs))
        squared = x.transpose("sample", "site").values ** 2
        valid = (x.transpose("sample", "site").values < FAILS_AT).all(axis=1)
        values = {"m": squared, "m2": squared + 1.0}
        return SimulatorOutput(
            values={o: values[o] for o in self._outputs},
            valid={o: valid for o in self._outputs},
            record=("square", x.sizes["sample"]),
        )

    def at(self, coords, outputs):
        self.log["at"].append(tuple(outputs))
        if list(coords["site"]) == self._sites and tuple(outputs) == self._outputs:
            return self
        return Square(coords["site"], outputs, log=self.log)

    def check_given(self, given_specs, corner_values):
        self.log["checks"].append((dict(given_specs), corner_values))


def _x():
    return FactorSpec(ArraySpec("x", units="1", support=POSITIVE, indexed_by=("site",)),
                      law=iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))


def _sigma():
    return FactorSpec(ArraySpec("sigma", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))


def _normal_about(prediction):
    """A law ``N(prediction, sigma^2)`` per site, reading the prediction by name."""
    if prediction == "m":
        return lambda m, sigma: tfd.Independent(tfd.Normal(m, sigma), 1)
    return lambda m2, sigma: tfd.Independent(tfd.Normal(m2, sigma), 1)


def _model(simulator=None):
    simulator = Square() if simulator is None else simulator
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=_normal_about("m")),
        FactorSpec(ArraySpec("validation", units="1", indexed_by=("site",)), law=_normal_about("m2")),
        simulator,
        _x(),
        _sigma(),
    )
    return spec.bind(coords={"site": SITES}), simulator


def _theta(posterior, x, sigma):
    """Theta at natural ``x`` (J, S) and ``sigma`` (J,)."""
    return jnp.concatenate([jnp.log(jnp.asarray(x)), jnp.log(jnp.asarray(sigma))[:, None]], axis=1)


# ── declaring and binding ─────────────────────────────────────────────────────


def test_a_simulator_is_a_part_whose_outputs_are_components():
    model, simulator = _model()
    assert model.spec.component_names == ("y", "validation", "m", "m2", "x", "sigma")
    assert model.describe().loc["square", "class"] == "Square"
    assert model.block_shape("m") == (2,)


def test_binding_runs_no_simulator_and_asks_for_every_output():
    model, simulator = _model()
    assert simulator.log["calls"] == []
    assert simulator.log["at"] == [("m", "m2")]
    assert model.simulators["square"] is simulator


def test_a_simulator_reading_nothing_is_refused():
    class Empty(Square):
        @property
        def given(self):
            return ()

    with pytest.raises(ValueError, match="reads no component"):
        joint(Empty(), _x())


def test_a_simulator_reading_its_own_output_is_refused():
    class Loop(Square):
        @property
        def given(self):
            return ("m",)

    with pytest.raises(ValueError, match="which it computes"):
        joint(Loop(), _x())


def test_a_simulator_reading_an_undeclared_name_is_refused():
    with pytest.raises(ValueError, match="no part declares"):
        joint(Square(), _sigma())


def test_at_that_changes_what_is_read_is_refused():
    class Changes(Square):
        def at(self, coords, outputs):
            return Square(coords["site"], ("m",))

    with pytest.raises(ValueError, match="at\\(\\) restricts"):
        joint(Changes(), _x(), _sigma(),
              FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=_normal_about("m"))).bind(
            coords={"site": SITES})


def test_a_factor_downstream_is_checked_for_its_form_at_bind():
    def scalar(m, sigma):
        return tfd.Normal(m[0], sigma)

    spec = joint(FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=scalar), Square(), _x(), _sigma())
    with pytest.raises(ValueError, match="event shape"):
        spec.bind(coords={"site": SITES})


def test_an_initial_state_declared_both_calibrated_and_external_is_refused():
    """F7: a name is a component or an input, never both."""
    with pytest.raises(ValueError, match="x"):
        joint(Square(), _x(), inputs=[ArraySpec("x", units="1", support=POSITIVE, indexed_by=("site",))])


def test_a_simulator_may_read_an_input():
    """F7: an external initial state is an input, which the simulator reads like a component."""
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=_normal_about("m")),
        Square(), _sigma(),
        inputs=[ArraySpec("x", units="1", support=POSITIVE, indexed_by=("site",))],
    )
    model = spec.bind(coords={"site": SITES}, inputs={"x": np.array([2.0, 3.0])})
    posterior = condition_on(model, {"y": Y})
    evaluation = posterior.evaluate(jnp.zeros((3, 1)))
    np.testing.assert_array_equal(evaluation.values["m"], np.tile([4.0, 9.0], (3, 1)))


# ── sampling and the joint density ────────────────────────────────────────────


def test_sampling_runs_the_simulator_once_and_fails_to_nan():
    model, simulator = _model()
    draws = model.sample(KEY, 400)
    assert simulator.log["calls"] == [(400, ("m", "m2"))]
    failed = (np.asarray(draws["x"]) >= FAILS_AT).any(axis=1)
    assert failed.any() and not failed.all()
    for name in ("m", "m2", "y", "validation"):
        assert np.isnan(np.asarray(draws[name])[failed]).all()
        assert np.isfinite(np.asarray(draws[name])[~failed]).all()


def test_sampling_what_needs_no_simulator_runs_none():
    model, simulator = _model()
    model.sample(KEY, 10, component_names=["x", "sigma"])
    assert simulator.log["calls"] == []


def test_the_joint_density_runs_the_simulator_and_is_minus_infinity_where_it_failed():
    model, simulator = _model()
    x = np.array([[1.0, 2.0], [1.0, 6.0]])
    values = {"x": x, "sigma": np.array([0.5, 0.5]), "y": np.tile(Y, (2, 1)), "validation": np.zeros((2, 2))}
    log_prob = np.asarray(model.log_prob(values))
    assert len(simulator.log["calls"]) == 1
    assert np.isfinite(log_prob[0]) and log_prob[1] == -np.inf
    expected = (
        st.lognorm(s=np.log(2.0)).logpdf(x[0]).sum() + st.lognorm(s=np.log(2.0)).logpdf(0.5)
        + st.norm(x[0] ** 2, 0.5).logpdf(Y).sum() + st.norm(x[0] ** 2 + 1.0, 0.5).logpdf(0.0).sum()
    )
    np.testing.assert_allclose(log_prob[0], expected, rtol=1e-12)


def test_a_simulator_cannot_run_under_a_trace():
    model, _ = _model()
    values = {"x": np.ones((1, 2)), "sigma": np.ones(1), "y": np.ones((1, 2)), "validation": np.ones((1, 2))}
    with pytest.raises(ValueError, match="JAX trace"):
        jax.jit(model.log_prob)(values)


# ── conditioning ──────────────────────────────────────────────────────────────


def test_the_posterior_computes_only_the_outputs_the_likelihood_reads():
    model, simulator = _model()
    posterior = condition_on(model, {"y": Y})
    assert [o.name for o in posterior.simulators["square"].outputs] == ["m"]
    assert posterior.barren_names == ("validation",)
    posterior.evaluate(_theta(posterior, [[1.0, 2.0]], [1.0]))
    assert simulator.log["calls"][-1] == (1, ("m",))


def test_a_target_factor_reading_a_simulator_is_refused():
    def centered(m):
        return tfd.Independent(tfd.Normal(m, 1.0), 1)

    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=lambda latent: tfd.Independent(tfd.Normal(latent, 1.0), 1)),
        FactorSpec(ArraySpec("latent", units="1", indexed_by=("site",)), law=centered),
        Square(), _x(),
    )
    with pytest.raises(ValueError, match="target 'latent' has the simulator"):
        condition_on(spec.bind(coords={"site": SITES}), {"y": Y})


def test_a_constant_factor_reading_a_simulator_is_refused():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=lambda m: tfd.Independent(tfd.Normal(m, 1.0), 1)),
        FactorSpec(ArraySpec("z", units="1"), law=lambda sigma: tfd.Normal(0.0, sigma)),
        Square(), _sigma(),
        inputs=[ArraySpec("x", units="1", support=POSITIVE, indexed_by=("site",))],
    )
    model = spec.bind(coords={"site": SITES}, inputs={"x": np.array([2.0, 3.0])})
    with pytest.raises(ValueError, match="no target ancestor 'y'"):
        condition_on(model, {"y": Y, "z": 0.5})


def test_check_given_runs_once_at_the_corner_points():
    model, simulator = _model()
    posterior = condition_on(model, {"y": Y})
    ((given_specs, corner_values),) = simulator.log["checks"]
    assert given_specs == {"x": model.spec.component_spec("x")}
    # x's two labels share one value of one number: 3 corners; sigma: 3.
    assert corner_values["x"].sizes == {"sample": 6, "site": 2}
    np.testing.assert_allclose(np.log(corner_values["x"].values[:3, 0]), CORNERS)
    assert simulator.log["calls"] == []
    assert posterior.dimension == 3


def test_corner_points_match_the_forward_models():
    small = corner_points([np.array([[0], [1]]), np.array([[2]])], 3)
    assert small.shape == (6, 3)
    np.testing.assert_array_equal(small[:3, 0], CORNERS)
    np.testing.assert_array_equal(small[:3, 0], small[:3, 1])
    large = corner_points([np.arange(7)[None]], 7)
    assert large.shape == (17, 7)


# ── evaluation ────────────────────────────────────────────────────────────────


def test_evaluation_scores_a_failed_sample_minus_infinity():
    model, simulator = _model()
    posterior = condition_on(model, {"y": Y})
    x = np.array([[1.0, 2.0], [1.0, 6.0], [0.5, 0.5]])
    sigma = np.array([0.5, 0.5, 2.0])
    theta = _theta(posterior, x, sigma)
    evaluation = posterior.evaluate(theta)
    assert isinstance(evaluation, PosteriorEvaluation)
    assert simulator.log["calls"][-1] == (3, ("m",))
    assert evaluation.valid.tolist() == [True, False, True]
    assert evaluation.simulator_valid.tolist() == [True, False, True]
    expected = st.norm(x[[0, 2]] ** 2, sigma[[0, 2], None]).logpdf(Y).sum(axis=1)
    np.testing.assert_allclose(np.asarray(evaluation.log_likelihood)[[0, 2]], expected, rtol=1e-12)
    assert evaluation.log_likelihood[1] == -np.inf and evaluation.log_density[1] == -np.inf
    np.testing.assert_allclose(evaluation.log_prior, posterior.log_prior(theta), rtol=0, atol=0)
    np.testing.assert_array_equal(evaluation.log_density, evaluation.log_prior + evaluation.log_likelihood)
    assert np.isnan(np.asarray(evaluation.values["m"])[1]).all()
    assert evaluation.simulator_records == {"square": ("square", 3)}


def test_log_likelihood_and_log_density_evaluate_a_batch_or_one_sample():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    theta = _theta(posterior, [[1.0, 2.0], [0.5, 1.5]], [0.5, 1.0])
    evaluation = posterior.evaluate(theta)
    np.testing.assert_array_equal(posterior.log_likelihood(theta), evaluation.log_likelihood)
    np.testing.assert_array_equal(posterior.log_density(theta), evaluation.log_density)
    assert posterior.log_likelihood(theta[0]).shape == ()
    np.testing.assert_array_equal(posterior.log_density(theta[1]), evaluation.log_density[1])


def test_evaluation_refuses_a_theta_that_is_not_a_finite_batch():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    with pytest.raises(ValueError, match="non-finite"):
        posterior.evaluate(jnp.array([[0.0, np.nan, 0.0]]))
    with pytest.raises(ValueError, match="give a batch"):
        posterior.evaluate(jnp.zeros((2, 4)))
    with pytest.raises(ValueError, match="at least one row"):
        posterior.evaluate(jnp.zeros((0, 3)))


def test_holding_the_simulator_gives_a_traced_density_in_the_free_entries():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    np.testing.assert_array_equal(posterior.simulator_free_positions, [2])
    x = np.array([[1.0, 2.0], [1.0, 6.0]])
    theta = _theta(posterior, x, [0.5, 0.7])
    evaluation = posterior.evaluate(theta)
    density = posterior.log_density_given(evaluation)
    free = theta[:, posterior.simulator_free_positions]
    np.testing.assert_allclose(density(free)[0], evaluation.log_density[0], rtol=1e-14)
    assert density(free)[1] == -np.inf

    def closed_form(log_sigma):
        sigma = jnp.exp(log_sigma[0])
        prior = -0.5 * (log_sigma[0] / jnp.log(2.0)) ** 2
        return prior + tfd.Normal(jnp.asarray(x[0] ** 2), sigma).log_prob(Y).sum()

    gradient = jax.grad(lambda t: density(t)[0])(jnp.asarray(free))
    np.testing.assert_allclose(gradient[0], jax.grad(closed_form)(jnp.asarray(free[0])), rtol=1e-10)
    np.testing.assert_allclose(jax.jit(density)(free)[0], evaluation.log_density[0], rtol=1e-14)


def test_log_density_given_refuses_anything_but_an_evaluation():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    with pytest.raises(TypeError, match="PosteriorEvaluation"):
        posterior.log_density_given({"theta": jnp.zeros((1, 3))})


# ── predictions ───────────────────────────────────────────────────────────────


def test_predict_computes_the_outputs_only_barren_factors_read():
    model, simulator = _model()
    posterior = condition_on(model, {"y": Y})
    x = np.array([[1.0, 2.0], [1.0, 6.0]])
    theta = _theta(posterior, x, [1e-6, 1e-6])
    drawn, computed = posterior.predict(KEY, theta)
    assert simulator.log["calls"][-1] == (2, ("m2",))
    assert set(drawn) == {"validation"} and computed["validation"].tolist() == [True, False]
    np.testing.assert_allclose(drawn["validation"][0], x[0] ** 2 + 1.0, atol=1e-4)
    assert np.isnan(np.asarray(drawn["validation"][1])).all()
    again, _ = posterior.predict(KEY, theta)
    np.testing.assert_array_equal(again["validation"], drawn["validation"])


def test_replicate_draws_the_observed_components_given_each_sample():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    theta = _theta(posterior, [[1.0, 2.0]] * 3, [1e-6] * 3)
    drawn, computed = posterior.replicate(KEY, theta)
    assert set(drawn) == {"y"} and computed["y"].tolist() == [True] * 3
    np.testing.assert_allclose(drawn["y"], np.tile([1.0, 4.0], (3, 1)), atol=1e-4)


def test_simulator_inputs_are_what_the_simulator_receives():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    theta = _theta(posterior, [[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
    given = posterior.simulator_inputs(theta, "square")
    assert set(given) == {"x"} and given["x"].dims == ("sample", "site")
    np.testing.assert_allclose(given["x"].values, [[1.0, 2.0], [3.0, 4.0]], rtol=1e-14)
    with pytest.raises(KeyError, match="no simulator"):
        posterior.simulator_inputs(theta, "sipnet")


def test_a_deterministic_after_the_simulator_is_computed_and_masked():
    shifted = DeterministicSpec(ArraySpec("shifted", units="1", indexed_by=("site",)), function=lambda m: m + 1.0)
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)),
                   law=lambda shifted, sigma: tfd.Independent(tfd.Normal(shifted, sigma), 1)),
        shifted, Square(), _x(), _sigma(),
    )
    posterior = condition_on(spec.bind(coords={"site": SITES}), {"y": Y})
    evaluation = posterior.evaluate(_theta(posterior, [[1.0, 2.0], [6.0, 1.0]], [1.0, 1.0]))
    np.testing.assert_allclose(evaluation.values["shifted"][0], [2.0, 5.0])
    assert evaluation.valid.tolist() == [True, False]


# ── the output contract ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ("not an output", "not a SimulatorOutput"),
        (SimulatorOutput(values={}, valid={"m": np.ones(1, bool)}), "values for"),
        (SimulatorOutput(values={"m": np.ones((1, 3))}, valid={"m": np.ones(1, bool)}), "of shape"),
        (SimulatorOutput(values={"m": np.ones((1, 2))}, valid={"m": np.ones(1)}), "bool array"),
    ],
)
def test_a_simulator_output_that_is_not_whole_is_refused(output, match):
    class Broken(Square):
        def __call__(self, given_values):
            return output

        def at(self, coords, outputs):
            return Broken(coords["site"], outputs)

    model, _ = _model(Broken())
    posterior = condition_on(model, {"y": Y})
    with pytest.raises((TypeError, ValueError), match=match):
        posterior.evaluate(jnp.zeros((1, 3)))


# ── what review found ─────────────────────────────────────────────────────────


def test_a_deterministic_that_hides_a_failure_is_still_masked():
    """A failed output stays a failure through a deterministic that would turn NaN into numbers."""
    clean = DeterministicSpec(ArraySpec("clean", units="1", indexed_by=("site",)), function=lambda m: jnp.nan_to_num(m))
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)),
                   law=lambda clean, sigma: tfd.Independent(tfd.Normal(clean, sigma), 1)),
        clean, Square(), _x(), _sigma(),
    )
    model = spec.bind(coords={"site": SITES})
    x = np.array([[1.0, 2.0], [6.0, 1.0]])
    log_prob = np.asarray(model.log_prob({"x": x, "sigma": np.ones(2), "y": np.tile(Y, (2, 1))}))
    assert np.isfinite(log_prob[0]) and log_prob[1] == -np.inf
    posterior = condition_on(model, {"y": Y})
    evaluation = posterior.evaluate(_theta(posterior, x, [1.0, 1.0]))
    assert np.isnan(np.asarray(evaluation.values["clean"])[1]).all()
    assert evaluation.valid.tolist() == [True, False]


class HalfFails(Square):
    """``m`` fails where ``x`` reaches :data:`FAILS_AT`; ``m2`` never fails,
    and is infinite where ``x`` is below 0.1, though reported valid."""

    def __call__(self, given_values):
        output = super().__call__(given_values)
        x = given_values["x"].transpose("sample", "site").values
        values = dict(output.values)
        valid = dict(output.valid)
        if "m2" in values:
            values["m2"] = np.where(x < 0.1, np.inf, values["m2"])
            valid["m2"] = np.ones(len(x), dtype=bool)
        return SimulatorOutput(values=values, valid=valid)

    def at(self, coords, outputs):
        return HalfFails(coords["site"], outputs, log=self.log)


def _ignoring_nan(*names):
    """A law over ``y`` per site reading *names*, blind to a NaN in them."""
    def law(m, m2, sigma):
        return tfd.Independent(tfd.Normal(jnp.nan_to_num(m) + jnp.nan_to_num(m2, posinf=0.0), sigma), 1)
    return law


def test_a_part_reading_two_outputs_fails_where_either_failed():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)), law=_ignoring_nan()),
        HalfFails(), _x(), _sigma(),
    )
    model = spec.bind(coords={"site": SITES})
    x = np.array([[1.0, 2.0], [6.0, 1.0], [0.05, 1.0]])
    log_prob = np.asarray(model.log_prob({"x": x, "sigma": np.ones(3), "y": np.tile(Y, (3, 1))}))
    assert np.isfinite(log_prob[0]) and log_prob[1] == -np.inf and log_prob[2] == -np.inf
    posterior = condition_on(model, {"y": Y})
    evaluation = posterior.evaluate(_theta(posterior, x, [1.0, 1.0, 1.0]))
    assert evaluation.simulator_valid.tolist() == [True, False, False]
    assert evaluation.valid.tolist() == [True, False, False]
    assert np.asarray(evaluation.log_likelihood)[1:].tolist() == [-np.inf, -np.inf]
    drawn = model.sample(KEY, 400)
    failed = (np.asarray(drawn["x"]) >= FAILS_AT).any(axis=1) | (np.asarray(drawn["x"]) < 0.1).any(axis=1)
    assert failed.any()
    assert np.isnan(np.asarray(drawn["y"])[failed]).all() and np.isfinite(np.asarray(drawn["y"])[~failed]).all()


def test_an_evaluation_under_a_trace_is_refused_by_name():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    with pytest.raises(ValueError, match="theta is traced"):
        jax.jit(posterior.log_density)(jnp.zeros((2, 3)))


def test_no_draws_call_no_simulator():
    model, simulator = _model()
    draws = model.sample(KEY, 0)
    assert draws["m"].shape == (0, 2) and simulator.log["calls"] == []


def test_the_gradient_of_a_failed_row_is_zero_not_nan():
    model, _ = _model()
    posterior = condition_on(model, {"y": Y})
    evaluation = posterior.evaluate(_theta(posterior, [[1.0, 2.0], [1.0, 6.0]], [0.5, 0.5]))
    density = posterior.log_density_given(evaluation)
    free = evaluation.theta[:, posterior.simulator_free_positions]
    gradient = jax.grad(lambda t: jnp.where(jnp.isfinite(density(t)), density(t), 0.0).sum())(free)
    assert np.isfinite(np.asarray(gradient)).all() and gradient[1, 0] == 0.0


class Twice(Simulator):
    """``z = 2 m``, downstream of :class:`Square`, recording its checks."""

    def __init__(self, log):
        self.log = log

    name = property(lambda self: "twice")
    given = property(lambda self: ("m",))
    outputs = property(lambda self: (ArraySpec("z", units="1", indexed_by=("site",)),))

    def __call__(self, given_values):
        m = given_values["m"].transpose("sample", "site").values
        return SimulatorOutput(values={"z": 2 * m}, valid={"z": np.isfinite(m).all(axis=1)})

    def at(self, coords, outputs):
        return self

    def check_given(self, given_specs, corner_values):
        self.log.append("checked")


def test_a_chain_runs_no_simulator_when_conditioned_and_one_batch_when_evaluated():
    square, checks = Square(), []
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)),
                   law=lambda z, sigma: tfd.Independent(tfd.Normal(z, sigma), 1)),
        Twice(checks), square, _x(), _sigma(),
    )
    posterior = condition_on(spec.bind(coords={"site": SITES}), {"y": Y})
    assert square.log["calls"] == [] and checks == []
    evaluation = posterior.evaluate(_theta(posterior, [[1.0, 2.0], [1.0, 6.0]], [1.0, 1.0]))
    assert square.log["calls"] == [(2, ("m",))]
    np.testing.assert_allclose(evaluation.values["z"][0], [2.0, 8.0])
    assert evaluation.valid.tolist() == [True, False]


def test_an_entry_only_a_barren_simulator_output_depends_on_is_free():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("site",)),
                   law=lambda x, sigma: tfd.Independent(tfd.Normal(jnp.log(x), sigma), 1)),
        FactorSpec(ArraySpec("validation", units="1", indexed_by=("site",)), law=_normal_about("m2")),
        Square(), _x(), _sigma(),
    )
    posterior = condition_on(spec.bind(coords={"site": SITES}), {"y": Y})
    assert posterior.simulators == {}
    np.testing.assert_array_equal(posterior.simulator_free_positions, [0, 1, 2])

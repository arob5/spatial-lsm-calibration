"""Tests for Gaussian factors: the covariance specs, ``GaussianSpec``, the
Gaussian law, and the likelihood as ``y ~ N(G(theta), R)``.

The covariance specs are checked against the dense matrices they stand for,
on a stacked dim of three sites observing ragged times, and the densities
against SciPy. A covariance fixed by the held values is built once, when
the model is conditioned; one that reads a parameter is built per sample,
and keeps the likelihood from being written as a Gaussian until that
parameter is held.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.stats as st
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability import (
    NON_NEGATIVE,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    ArraySpec,
    BlockDiagonalSpec,
    DenseSpec,
    DeterministicSpec,
    DiagonalSpec,
    FactorSpec,
    GaussianLaw,
    GaussianLikelihood,
    GaussianSpec,
    ScaledSpec,
    Simulator,
    SimulatorOutput,
    SubmatrixSpec,
    SumSpec,
    as_law,
    condition_on,
    iid_over_dim,
    inverse_wishart,
    joint,
    log_normal,
    normal,
)
from sipnet_calibration.probability import _linalg

tfd = tfp.distributions

KEY = jax.random.key(11)
#: Three sites observing two, four and three times; the observation dim is
#: sorted by site, then year, then time, as an observation dim is.
LABELS = pd.MultiIndex.from_tuples(
    [(3, 2012, 0), (3, 2013, 1),
     (5, 2012, 0), (5, 2012, 1), (5, 2013, 2), (5, 2013, 3),
     (9, 2012, 0), (9, 2013, 1), (9, 2013, 2)],
    names=["site", "year", "t"],
)
N = len(LABELS)
SITES = [3, 5, 9]
RNG = np.random.default_rng(20261005)
VARIANCE = RNG.uniform(0.5, 2.0, N)
TIMES = np.array([0.0, 1.0, 0.0, 0.7, 1.9, 3.1, 0.0, 1.2, 2.0])
Y = RNG.normal(size=N)


def _on_the_observations(values) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims=("obs",), coords={"obs": LABELS})


CONSTANTS = {"variance": _on_the_observations(VARIANCE), "t": _on_the_observations(TIMES)}


def correlated(variance, t):
    """A block: its variances on the diagonal, plus exponential correlation in time."""
    return jnp.diag(variance) + 0.4 * jnp.exp(-jnp.abs(t[:, None] - t[None, :]))


def _correlated(positions) -> np.ndarray:
    return np.asarray(correlated(VARIANCE[positions], TIMES[positions]))


def _by_site(block) -> np.ndarray:
    """The block-diagonal matrix of *block* at each site's entries."""
    sites = LABELS.get_level_values("site")
    return scipy.linalg.block_diag(*(block(np.flatnonzero(sites == s)) for s in SITES))


def _mu():
    return FactorSpec(ArraySpec("mu", units="1", support=REAL), law=normal(mean=0.0, standard_deviation=1.0))


def _mean():
    return DeterministicSpec(ArraySpec("m", units="1", indexed_by=("obs",)), function=lambda mu: jnp.full((N,), mu))


def _y(covariance, *, constants=None, label_maps=None, units="1"):
    return FactorSpec(
        ArraySpec("y", units=units, support=REAL, indexed_by=("obs",)),
        law=GaussianSpec(mean="m", covariance=covariance),
        constants=CONSTANTS if constants is None else constants, label_maps=label_maps,
    )


def _model(covariance, *parts, constants=None, label_maps=None, coords=None):
    spec = joint(_y(covariance, constants=constants, label_maps=label_maps), _mean(), _mu(), *parts)
    return spec.bind(coords={"obs": LABELS, **(coords or {})})


def _dense(model, given=None) -> np.ndarray:
    law = model.law("y", given={"m": jnp.zeros(N), **(given or {})})
    return np.asarray(law.covariance.to_dense())


def _check_density(model, expected: np.ndarray, given=None) -> None:
    law = model.law("y", given={"m": jnp.zeros(N), **(given or {})})
    points = RNG.normal(size=(4, N))
    np.testing.assert_allclose(
        np.asarray(law.log_prob(points)), st.multivariate_normal(np.zeros(N), expected).logpdf(points), rtol=1e-12
    )


# ── the covariance specs ──────────────────────────────────────────────────────


def test_a_diagonal_is_its_variances():
    model = _model(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]})
    law = model.law("y", given={"m": jnp.zeros(N)})
    assert isinstance(law.covariance, _linalg.PSDDiagonal)
    np.testing.assert_allclose(_dense(model), np.diag(VARIANCE))
    _check_density(model, np.diag(VARIANCE))


def test_a_diagonal_of_a_scalar_gives_every_entry_its_variance():
    sigma2 = FactorSpec(ArraySpec("sigma2", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))
    model = _model(DiagonalSpec("sigma2"), sigma2, constants={})
    np.testing.assert_allclose(_dense(model, {"sigma2": 2.5}), 2.5 * np.eye(N))


def test_a_diagonal_from_a_function_reads_its_keywords():
    model = _model(DiagonalSpec(lambda variance: 2.0 * variance), constants={"variance": CONSTANTS["variance"]})
    np.testing.assert_allclose(_dense(model), np.diag(2.0 * VARIANCE))


def test_a_dense_block_per_site_is_the_block_diagonal_matrix():
    model = _model(BlockDiagonalSpec(DenseSpec(correlated), by="site"))
    law = model.law("y", given={"m": jnp.zeros(N)})
    assert isinstance(law.covariance, _linalg.PSDBlockDiag)
    assert [b.shape[0] for b in law.covariance.blocks] == [2, 4, 3]
    expected = _by_site(_correlated)
    np.testing.assert_allclose(_dense(model), expected, rtol=1e-12)
    _check_density(model, expected)


def test_a_dense_matrix_over_the_whole_event_is_one_block():
    model = _model(DenseSpec(correlated))
    np.testing.assert_allclose(_dense(model), _correlated(np.arange(N)), rtol=1e-12)


def test_a_sum_of_diagonals_is_a_diagonal():
    model = _model(SumSpec(DiagonalSpec("variance"), DiagonalSpec(lambda variance: 0.5 * variance)),
                   constants={"variance": CONSTANTS["variance"]})
    law = model.law("y", given={"m": jnp.zeros(N)})
    assert isinstance(law.covariance, _linalg.PSDDiagonal)
    np.testing.assert_allclose(_dense(model), np.diag(1.5 * VARIANCE))


def test_a_sum_with_a_dense_term_is_dense():
    model = _model(SumSpec(DenseSpec(correlated), DiagonalSpec("variance")))
    law = model.law("y", given={"m": jnp.zeros(N)})
    assert isinstance(law.covariance, _linalg.DensePSD)
    np.testing.assert_allclose(_dense(model), _correlated(np.arange(N)) + np.diag(VARIANCE), rtol=1e-12)


def test_a_block_diagonal_plus_a_diagonal_is_summed_blockwise():
    model = _model(SumSpec(BlockDiagonalSpec(DenseSpec(correlated), by="site"), DiagonalSpec("variance")))
    law = model.law("y", given={"m": jnp.zeros(N)})
    assert isinstance(law.covariance, _linalg.PSDBlockDiag)
    np.testing.assert_allclose(_dense(model), _by_site(_correlated) + np.diag(VARIANCE), rtol=1e-12)


def test_block_diagonals_of_one_grouping_are_summed_blockwise():
    model = _model(SumSpec(
        BlockDiagonalSpec(DenseSpec(correlated), by="site"),
        BlockDiagonalSpec(DenseSpec(lambda t: 0.1 * jnp.ones((t.shape[0], t.shape[0]))), by="site"),
    ))
    expected = _by_site(lambda p: _correlated(p) + 0.1 * np.ones((len(p), len(p))))
    np.testing.assert_allclose(_dense(model), expected, rtol=1e-12)


@pytest.mark.parametrize("terms", [
    (BlockDiagonalSpec(DenseSpec(correlated), by="site"), DenseSpec(correlated)),
    (BlockDiagonalSpec(DenseSpec(correlated), by="site"), BlockDiagonalSpec(DenseSpec(correlated), by="year")),
])
def test_a_sum_needing_a_dense_matrix_over_the_event_is_refused(terms):
    with pytest.raises(ValueError, match="dense matrix over the whole event"):
        _model(SumSpec(*terms))


def _scale():
    return FactorSpec(ArraySpec("s", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))


@pytest.mark.parametrize("hidden", [
    ScaledSpec(BlockDiagonalSpec(DenseSpec(correlated), by="site"), scale="s"),
    SumSpec(BlockDiagonalSpec(DenseSpec(correlated), by="site"), DiagonalSpec("variance")),
])
def test_a_block_diagonal_held_inside_a_sums_term_is_refused(hidden):
    with pytest.raises(ValueError, match="hold a block-diagonal inside"):
        _model(SumSpec(hidden, DiagonalSpec("variance")), _scale())


def test_a_scaled_term_of_a_dense_sum_is_scaled():
    model = _model(SumSpec(ScaledSpec(DenseSpec(correlated), scale="s"), DiagonalSpec("variance")), _scale())
    expected = 2.5 * _correlated(np.arange(N)) + np.diag(VARIANCE)
    np.testing.assert_allclose(_dense(model, {"s": 2.5}), expected, rtol=1e-12)
    _check_density(model, expected, {"s": 2.5})


def test_a_sum_has_two_terms():
    with pytest.raises(ValueError, match="at least two"):
        SumSpec(DiagonalSpec("variance"))


def test_a_scale_multiplies_the_covariance():
    scale = FactorSpec(ArraySpec("scale", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))
    model = _model(ScaledSpec(DiagonalSpec("variance"), scale="scale"), scale, constants={"variance": CONSTANTS["variance"]})
    law = model.law("y", given={"m": jnp.zeros(N), "scale": 3.0})
    assert isinstance(law.covariance, _linalg.PSDScaled)
    np.testing.assert_allclose(np.asarray(law.covariance.to_dense()), 3.0 * np.diag(VARIANCE))
    _check_density(model, 3.0 * np.diag(VARIANCE), {"scale": 3.0})


def test_a_scale_by_site_scales_each_sites_block_by_its_own():
    scale = FactorSpec(ArraySpec("scale", units="1", support=POSITIVE, indexed_by=("site",)),
                       law=iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))
    model = _model(BlockDiagonalSpec(ScaledSpec(DenseSpec(correlated), scale="scale"), by="site"), scale,
                   coords={"site": SITES})
    scales = np.array([0.5, 2.0, 4.0])
    sites = LABELS.get_level_values("site")
    expected = scipy.linalg.block_diag(*(
        scales[i] * _correlated(np.flatnonzero(sites == s)) for i, s in enumerate(SITES)
    ))
    np.testing.assert_allclose(_dense(model, {"scale": scales}), expected, rtol=1e-12)
    _check_density(model, expected, {"scale": scales})


def test_a_scale_that_is_not_one_number_in_its_scope_is_refused():
    scale = FactorSpec(ArraySpec("scale", units="1", support=POSITIVE, indexed_by=("site",)),
                       law=iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))
    with pytest.raises(ValueError, match="a scale is a scalar"):
        _model(ScaledSpec(DenseSpec(correlated), scale="scale"), scale, coords={"site": SITES})


def test_a_scale_whose_support_is_not_positive_is_refused():
    scale = FactorSpec(ArraySpec("scale", units="1", support=REAL), law=normal(mean=0.0, standard_deviation=1.0))
    with pytest.raises(ValueError, match="is not positive"):
        _model(ScaledSpec(DenseSpec(correlated), scale="scale"), scale)


def test_a_scale_that_may_be_zero_is_refused():
    scale = FactorSpec(ArraySpec("scale", units="1", support=NON_NEGATIVE), law=log_normal(median=1.0, geometric_sd=2.0))
    with pytest.raises(ValueError, match="is not positive"):
        _model(ScaledSpec(DenseSpec(correlated), scale="scale"), scale)


def test_a_scaled_base_may_not_read_its_scale():
    with pytest.raises(ValueError, match="reads its scale"):
        ScaledSpec(DiagonalSpec("scale"), scale="scale")


def test_grouping_by_the_dim_itself_gives_each_entry_its_own_block():
    for by in ("obs", ("site", "year", "t")):
        model = _model(BlockDiagonalSpec(DiagonalSpec("variance"), by=by), constants={"variance": CONSTANTS["variance"]})
        law = model.law("y", given={"m": jnp.zeros(N)})
        assert len(law.covariance.blocks) == N
        np.testing.assert_allclose(_dense(model), np.diag(VARIANCE))


def test_groupings_nest():
    model = _model(BlockDiagonalSpec(BlockDiagonalSpec(DenseSpec(correlated), by="year"), by="site"))
    years = LABELS.get_level_values("year")
    expected = scipy.linalg.block_diag(*(
        _correlated(np.flatnonzero((LABELS.get_level_values("site") == s) & (years == y)))
        for s in SITES for y in (2012, 2013)
    ))
    np.testing.assert_allclose(_dense(model), expected, rtol=1e-12)


def test_a_grouping_whose_groups_are_not_contiguous_is_refused():
    with pytest.raises(ValueError, match="not contiguous"):
        _model(BlockDiagonalSpec(DenseSpec(correlated), by="year"))


@pytest.mark.parametrize("by", [[], ["site", "site"]])
def test_a_grouping_names_distinct_levels(by):
    with pytest.raises(ValueError, match="one or more distinct levels"):
        BlockDiagonalSpec(DenseSpec(correlated), by=by)


def _per_site(values, sites=SITES) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims=("site",), coords={"site": sites})


def test_a_per_site_constant_on_an_own_dim_is_read_at_each_groups_label():
    """No component is indexed by site, so site is the factor's own dim,
    labeled by its constants, here in another order than the event's."""
    y = FactorSpec(
        ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
        law=GaussianSpec(mean="m", covariance=BlockDiagonalSpec(DiagonalSpec("site_variance"), by="site")),
        constants={"site_variance": _per_site([3.0, 1.0, 2.0], sites=[9, 3, 5])}, own_dims=["site"],
    )
    model = joint(y, _mean(), _mu()).bind(coords={"obs": LABELS})
    by_site = {9: 3.0, 3: 1.0, 5: 2.0}
    np.testing.assert_allclose(np.diag(_dense(model)), [by_site[s] for s in LABELS.get_level_values("site")])


def test_constants_on_an_own_dim_labeled_two_ways_are_refused():
    y = FactorSpec(
        ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
        law=GaussianSpec(mean="m", covariance=BlockDiagonalSpec(DiagonalSpec(lambda a, b: a * b), by="site")),
        constants={"a": _per_site([1.0, 2.0, 3.0]), "b": _per_site([1.0, 2.0, 3.0], sites=[9, 5, 3])},
        own_dims=["site"],
    )
    with pytest.raises(ValueError, match="with different labels"):
        joint(y, _mean(), _mu()).bind(coords={"obs": LABELS})


def test_a_group_split_by_one_entry_is_not_contiguous():
    labels = pd.MultiIndex.from_tuples([(1, 0), (2, 0), (1, 1)], names=["site", "t"])
    y = FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                   law=GaussianSpec(mean="mu3", covariance=BlockDiagonalSpec(DiagonalSpec(lambda: 1.0), by="site")))
    mean = DeterministicSpec(ArraySpec("mu3", units="1", indexed_by=("obs",)), function=lambda mu: jnp.full((3,), mu))
    with pytest.raises(ValueError, match="not contiguous"):
        joint(y, mean, _mu()).bind(coords={"obs": labels})


def test_a_grouping_by_what_is_not_a_level_is_refused():
    with pytest.raises(ValueError, match="has levels"):
        _model(BlockDiagonalSpec(DenseSpec(correlated), by="month"))


def test_a_grouping_of_an_event_indexed_by_nothing_is_refused():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1"), law=GaussianSpec(mean="mu", covariance=BlockDiagonalSpec(
            DiagonalSpec(lambda: jnp.ones(1)), by="site"))),
        _mu(),
    )
    with pytest.raises(ValueError, match="indexed by nothing"):
        spec.bind(coords={})


def test_a_group_whose_label_its_dim_lacks_is_refused():
    scale = FactorSpec(ArraySpec("scale", units="1", support=POSITIVE, indexed_by=("site",)),
                       law=iid_over_dim(log_normal(median=1.0, geometric_sd=2.0)))
    with pytest.raises(ValueError, match="is not among the labels of 'site'"):
        _model(BlockDiagonalSpec(ScaledSpec(DenseSpec(correlated), scale="scale"), by="site"), scale,
               coords={"site": [3, 5]})


def test_a_covariance_of_the_wrong_shape_is_refused_at_bind():
    with pytest.raises(ValueError, match=r"return \(9, 9\)"):
        _model(DenseSpec(lambda variance: jnp.diag(variance[:3])), constants={"variance": CONSTANTS["variance"]})


# ── a submatrix of a matrix component ─────────────────────────────────────────

YEARS = ("2012", "2013", "2014")
#: Each site observes some of the years, each at most once.
YEAR_LABELS = pd.MultiIndex.from_tuples(
    [(3, "2012"), (3, "2014"), (5, "2012"), (5, "2013"), (5, "2014"), (9, "2013")], names=["site", "season"],
)
S = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.8], [0.5, 0.8, 2.0]])


def _submatrix_model(by="site"):
    year_of = xr.DataArray(np.asarray(YEAR_LABELS.get_level_values("season"), dtype=object), dims=("obs",),
                           coords={"obs": YEAR_LABELS}, name="year")
    covariance = SubmatrixSpec("error_covariance", label_map="year_of")
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                   law=GaussianSpec(mean="m", covariance=covariance if by is None else BlockDiagonalSpec(covariance, by=by)),
                   label_maps={"year_of": year_of}),
        DeterministicSpec(ArraySpec("m", units="1", indexed_by=("obs",)), function=lambda mu: jnp.full((6,), mu)),
        _mu(),
        FactorSpec(ArraySpec("error_covariance", units="1", support=POSITIVE_DEFINITE,
                             element_axes={"year": YEARS, "other_year": YEARS}),
                   law=inverse_wishart(degrees_of_freedom=7.0, scale=3.0 * S)),
    )
    return spec.bind(coords={"obs": YEAR_LABELS})


def test_a_submatrix_takes_each_groups_rows_and_columns():
    model = _submatrix_model()
    law = model.law("y", given={"m": jnp.zeros(6), "error_covariance": S})
    rows = {"2012": 0, "2013": 1, "2014": 2}
    expected = scipy.linalg.block_diag(S[np.ix_([0, 2], [0, 2])], S[:, :], S[np.ix_([1], [1])])
    assert [rows[y] for y in YEAR_LABELS.get_level_values("season")] == [0, 2, 0, 1, 2, 1]
    np.testing.assert_allclose(np.asarray(law.covariance.to_dense()), expected, rtol=1e-12)


def test_a_submatrix_mapping_two_entries_to_one_label_is_refused():
    with pytest.raises(ValueError, match="would repeat its rows"):
        _submatrix_model(by=None)


def test_a_submatrix_of_what_is_not_a_matrix_component_is_refused():
    with pytest.raises(ValueError, match="POSITIVE_DEFINITE"):
        _model(SubmatrixSpec("mu", label_map="variance"), constants={"variance": CONSTANTS["variance"]})


# ── GaussianSpec ──────────────────────────────────────────────────────────────


def test_a_gaussian_is_centered_on_one_component():
    with pytest.raises(ValueError, match="several components is not supported"):
        GaussianSpec(mean=["a", "b"], covariance=DiagonalSpec("v"))


def test_a_gaussian_covariance_is_a_covariance_spec():
    with pytest.raises(TypeError, match="give a covariance spec"):
        GaussianSpec(mean="m", covariance=np.eye(2))


@pytest.mark.parametrize("event", [
    ArraySpec("y", units="1", support=POSITIVE, indexed_by=("obs",)),
    ArraySpec("y", units="1", support=REAL, indexed_by=("obs",), element_axes={"part": ("a", "b")}),
    ArraySpec("y", units="1", support=REAL, indexed_by=("obs", "site")),
])
def test_a_gaussian_event_is_one_vector_on_the_reals(event):
    with pytest.raises(ValueError, match="one component on REAL"):
        FactorSpec(event, law=GaussianSpec(mean="m", covariance=DiagonalSpec("v")), constants={"v": CONSTANTS["variance"]})


def test_a_gaussian_mean_that_is_a_constant_is_refused_at_joint():
    y = FactorSpec(ArraySpec("y", units="1", support=REAL, indexed_by=("obs",)),
                   law=GaussianSpec(mean="variance", covariance=DiagonalSpec("t")), constants=CONSTANTS)
    with pytest.raises(ValueError, match="not a component or input"):
        joint(y, _mu())


def test_a_gaussian_mean_of_another_shape_is_named_as_the_mean():
    mean = DeterministicSpec(ArraySpec("m", units="1", indexed_by=("obs",)), function=lambda mu: jnp.full((4,), mu))
    with pytest.raises(ValueError, match=r"centered on 'm' of shape \(4,\)"):
        joint(_y(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]}), mean, _mu()).bind(
            coords={"obs": LABELS})


def test_a_gaussian_mean_of_other_units_or_dims_is_refused_at_joint():
    with pytest.raises(ValueError, match="layout and units"):
        joint(_y(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]}, units="m"), _mean(), _mu())


def test_a_gaussian_factor_reads_its_mean_and_its_covariances_names():
    factor = _y(BlockDiagonalSpec(DenseSpec(correlated), by="site"))
    assert factor.given == ("m",)
    assert set(factor.reads) == {"m", "variance", "t"}
    assert factor.law_name == "Gaussian"


def test_the_description_says_a_gaussian_is_evaluated_as_one():
    model = _model(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]})
    assert model.describe().loc["y", "evaluated_by"] == "Gaussian"


# ── the Gaussian law ──────────────────────────────────────────────────────────


def test_draws_of_a_gaussian_factor_have_its_mean_and_covariance():
    model = _model(BlockDiagonalSpec(DenseSpec(correlated), by="site"))
    law = model.law("y", given={"m": jnp.arange(N, dtype=jnp.float64)})
    draws = np.asarray(law.sample((40_000,), seed=KEY))
    assert draws.shape == (40_000, N)
    np.testing.assert_allclose(draws.mean(axis=0), np.arange(N), atol=0.05)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), _by_site(_correlated), atol=0.06)


def test_ancestral_draws_of_a_gaussian_factor_follow_their_mean():
    model = _model(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]})
    draws = model.sample(KEY, 4000)
    residual = np.asarray(draws["y"] - draws["mu"][:, None])
    np.testing.assert_allclose(residual.var(axis=0), VARIANCE, rtol=0.1)


def test_an_enskit_gaussian_is_a_law():
    covariance = _linalg.DensePSD(jnp.asarray(_by_site(_correlated)))
    law = as_law(_linalg.Gaussian.independent(y=(jnp.zeros(N), covariance)))
    assert isinstance(law, GaussianLaw)
    np.testing.assert_allclose(
        np.asarray(law.log_prob(Y)), st.multivariate_normal(np.zeros(N), _by_site(_correlated)).logpdf(Y), rtol=1e-12
    )


def test_a_gaussian_law_over_a_matrix_block_is_over_its_entries_in_c_order():
    covariance = _by_site(_correlated)[:6, :6]
    law = GaussianLaw(jnp.zeros((2, 3)), _linalg.DensePSD(jnp.asarray(covariance)))
    points = RNG.normal(size=(4, 2, 3))
    expected = st.multivariate_normal(np.zeros(6), covariance).logpdf(points.reshape(4, 6))
    np.testing.assert_allclose(np.asarray(law.log_prob(points)), expected, rtol=1e-12)
    assert jnp.shape(law.log_prob(points[0])) == ()
    assert law.sample((5,), seed=KEY).shape == (5, 2, 3)


def test_a_gaussian_law_needs_an_operator_over_its_block_and_a_key():
    with pytest.raises(TypeError, match="positive-definite operator"):
        GaussianLaw(jnp.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match=r"give a \(3, 3\) covariance"):
        GaussianLaw(jnp.zeros(3), _linalg.PSDDiagonal(jnp.ones(2)))
    with pytest.raises(TypeError, match="draws from a key"):
        GaussianLaw(jnp.zeros(2), _linalg.PSDDiagonal(jnp.ones(2))).sample((3,))


# ── conditioning ──────────────────────────────────────────────────────────────


def test_the_log_likelihood_is_scipys():
    model = _model(BlockDiagonalSpec(DenseSpec(correlated), by="site"))
    posterior = condition_on(model, {"y": Y})
    theta = jnp.array([[0.0], [0.7], [-1.3]])
    expected = [st.multivariate_normal(np.full(N, mu), _by_site(_correlated)).logpdf(Y) for mu in (0.0, 0.7, -1.3)]
    np.testing.assert_allclose(np.asarray(posterior.log_likelihood(theta)), expected, rtol=1e-12)


def test_a_covariance_the_held_values_fix_is_built_once_when_conditioned():
    calls = []

    def counted(variance, t):
        calls.append(1)
        return correlated(variance, t)

    posterior = condition_on(_model(BlockDiagonalSpec(DenseSpec(counted), by="site")), {"y": Y})
    built = len(calls)
    assert posterior._likelihood[0].held_covariance is not None
    posterior.log_likelihood(jnp.zeros((5, 1)))
    posterior.evaluate(jnp.zeros((3, 1)))
    assert len(calls) == built


def test_a_covariance_with_a_zero_variance_is_refused_when_bound():
    zero = VARIANCE.copy()
    zero[4] = 0.0
    with pytest.raises(ValueError, match="not positive definite"):
        _model(DiagonalSpec("variance"), constants={"variance": _on_the_observations(zero)})


def test_a_covariance_the_observed_values_make_singular_is_refused_when_conditioned():
    v = FactorSpec(ArraySpec("v", units="1", support=POSITIVE), law=log_normal(median=2.0, geometric_sd=1.1))
    model = _model(DiagonalSpec(lambda variance, v: variance * (v - 1.0)), v, constants={"variance": CONSTANTS["variance"]})
    with pytest.raises(ValueError, match="not positive definite"):
        condition_on(model, {"y": Y, "v": 0.5})


def _with_a_variance_parameter():
    sigma2 = FactorSpec(ArraySpec("sigma2", units="1", support=POSITIVE), law=log_normal(median=1.0, geometric_sd=2.0))
    return _model(ScaledSpec(BlockDiagonalSpec(DenseSpec(correlated), by="site"), scale="sigma2"), sigma2)


def test_a_covariance_reading_a_parameter_is_built_per_sample():
    posterior = condition_on(_with_a_variance_parameter(), {"y": Y})
    assert posterior.parameter_names == ("mu", "sigma2")
    theta = jnp.array([[0.2, np.log(0.5)], [-0.4, np.log(3.0)]])
    expected = [
        st.multivariate_normal(np.full(N, mu), s * _by_site(_correlated)).logpdf(Y)
        for mu, s in ((0.2, 0.5), (-0.4, 3.0))
    ]
    np.testing.assert_allclose(np.asarray(posterior.log_likelihood(theta)), expected, rtol=1e-12)


def test_a_prior_putting_mass_on_covariances_that_are_not_positive_definite_binds():
    """A correlation drawn wide of (-1, 1) is checked per sample, not at the
    draws binding is made at."""
    rho = FactorSpec(ArraySpec("rho", units="1", support=REAL), law=normal(mean=0.0, standard_deviation=50.0))
    covariance = DenseSpec(lambda rho: (1.0 - rho) * jnp.eye(N) + rho * jnp.ones((N, N)))
    model = _model(covariance, rho, constants={})
    evaluation = condition_on(model, {"y": Y}).evaluate(jnp.array([[0.0, 0.3], [0.0, 2.0]]))
    assert evaluation.valid.tolist() == [True, False]


def test_a_sample_whose_covariance_is_not_positive_definite_is_invalid():
    v = FactorSpec(ArraySpec("v", units="1", support=POSITIVE), law=log_normal(median=4.0, geometric_sd=1.1))
    model = _model(DiagonalSpec(lambda variance, v: variance * (v - 1.0)), v, constants={"variance": CONSTANTS["variance"]})
    posterior = condition_on(model, {"y": Y})
    evaluation = posterior.evaluate(jnp.array([[0.0, np.log(2.0)], [0.0, np.log(0.5)]]))
    assert evaluation.valid.tolist() == [True, False]
    assert np.isneginf(np.asarray(evaluation.log_likelihood)[1])


# ── the likelihood as a Gaussian ──────────────────────────────────────────────


class Shift(Simulator):
    """``prediction = mu + offsets`` on the observations, failing where ``mu``
    exceeds 2."""

    def __init__(self, offsets):
        self.offsets = np.asarray(offsets)

    name = "shift"
    given = ("mu",)

    @property
    def outputs(self):
        return (ArraySpec("m", units="1", indexed_by=("obs",)),)

    def __call__(self, given_values):
        mu = given_values["mu"].values
        return SimulatorOutput(values={"m": mu[:, None] + self.offsets}, valid={"m": mu <= 2.0})

    def at(self, coords, outputs):
        return self


def _likelihood_model():
    covariance = BlockDiagonalSpec(DenseSpec(correlated), by="site")
    other = FactorSpec(ArraySpec("z", units="1", support=REAL, indexed_by=("site",)),
                       law=GaussianSpec(mean="site_mean", covariance=DiagonalSpec(lambda: 0.25 * jnp.ones(3))))
    site_mean = DeterministicSpec(ArraySpec("site_mean", units="1", indexed_by=("site",)),
                                  function=lambda mu: jnp.full((3,), 2.0 * mu))
    spec = joint(_y(covariance), Shift(np.linspace(-1.0, 1.0, N)), other, site_mean, _mu())
    return spec.bind(coords={"obs": LABELS, "site": SITES})


def test_the_gaussian_likelihood_holds_y_and_each_factors_covariance():
    posterior = condition_on(_likelihood_model(), {"y": Y, "z": [0.1, 0.2, 0.3]})
    likelihood = posterior.gaussian_likelihood()
    assert isinstance(likelihood, GaussianLikelihood)
    np.testing.assert_array_equal(np.asarray(likelihood.y), np.concatenate([Y, [0.1, 0.2, 0.3]]))
    assert isinstance(likelihood.noise_covariance, _linalg.PSDBlockDiag)
    first, second = likelihood.noise_covariance.blocks
    assert isinstance(first, _linalg.PSDBlockDiag) and isinstance(second, _linalg.PSDDiagonal)
    np.testing.assert_allclose(
        np.asarray(likelihood.noise_covariance.to_dense()),
        scipy.linalg.block_diag(_by_site(_correlated), 0.25 * np.eye(3)), rtol=1e-12,
    )


def test_the_gaussian_likelihoods_forward_map_is_the_means_and_scores_as_the_posterior():
    posterior = condition_on(_likelihood_model(), {"y": Y, "z": [0.1, 0.2, 0.3]})
    likelihood = posterior.gaussian_likelihood()
    theta = jnp.array([[0.5], [3.0], [-1.0]])
    predictions, valid, evaluation = likelihood.forward(theta)
    assert valid.tolist() == [True, False, True]
    assert np.isnan(np.asarray(predictions[1])).all()
    for row in (0, 2):
        mu = float(theta[row, 0])
        np.testing.assert_allclose(
            np.asarray(predictions[row]), np.concatenate([mu + np.linspace(-1.0, 1.0, N), np.full(3, 2.0 * mu)])
        )
    gaussian = _linalg.Gaussian.independent(y=(likelihood.y, likelihood.noise_covariance))
    expected = np.asarray(gaussian.log_density(y=predictions[jnp.array([0, 2])]))
    np.testing.assert_allclose(np.asarray(evaluation.log_likelihood)[[0, 2]], expected, rtol=1e-12)


def test_a_covariance_reading_a_parameter_keeps_the_likelihood_from_being_gaussian():
    posterior = condition_on(_with_a_variance_parameter(), {"y": Y})
    with pytest.raises(ValueError, match=r"depends on the parameter\(s\) \['sigma2'\].*observe them at values"):
        posterior.gaussian_likelihood()


def test_holding_the_parameter_makes_the_likelihood_gaussian():
    posterior = condition_on(_with_a_variance_parameter(), {"y": Y, "sigma2": 2.0})
    assert posterior.constant_names == ("sigma2",)
    np.testing.assert_allclose(
        np.asarray(posterior.gaussian_likelihood().noise_covariance.to_dense()), 2.0 * _by_site(_correlated), rtol=1e-12
    )


def test_a_covariance_reading_a_simulator_output_is_built_per_sample_and_not_held():
    spec = joint(_y(DiagonalSpec(lambda m: 0.1 + m**2), constants={}), Shift(np.zeros(N)), _mu())
    posterior = condition_on(spec.bind(coords={"obs": LABELS}), {"y": Y})
    theta = jnp.array([[0.5], [-1.0]])
    expected = [st.multivariate_normal(np.full(N, mu), (0.1 + mu**2) * np.eye(N)).logpdf(Y) for mu in (0.5, -1.0)]
    np.testing.assert_allclose(np.asarray(posterior.log_likelihood(theta)), expected, rtol=1e-12)
    with pytest.raises(ValueError, match=r"depends on the parameter\(s\) \['mu'\]"):
        posterior.gaussian_likelihood()


def test_a_likelihood_with_a_law_that_is_not_gaussian_is_refused():
    spec = joint(
        FactorSpec(ArraySpec("y", units="1", indexed_by=("obs",)), law=lambda m: tfd.Independent(tfd.Normal(m, 1.0), 1)),
        _mean(), _mu(),
    )
    posterior = condition_on(spec.bind(coords={"obs": LABELS}), {"y": Y})
    with pytest.raises(ValueError, match="not a GaussianSpec"):
        posterior.gaussian_likelihood()


def test_a_likelihood_of_nothing_observed_is_refused():
    posterior = condition_on(_model(DiagonalSpec("variance"), constants={"variance": CONSTANTS["variance"]}), {})
    with pytest.raises(ValueError, match="nothing observed depends on the parameters"):
        posterior.gaussian_likelihood()

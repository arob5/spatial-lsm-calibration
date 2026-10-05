"""Tests for ``observation.model.noise_factor``, and PR #69's noise model
written with it.

PR #69 (``feat/single-site-mcmc-vs-eki``, ``model/noise.py``) assembles its
one-site noise covariance ``R`` by hand: one dense block per source, from
the sources' standard deviations and its discrepancy terms, placed at Flat
positions it checks are contiguous. Its block builders are transcribed
here, with ``config.py``'s values, and the same ``R`` is declared with noise
factors and read off ``Posterior.gaussian_likelihood``. The four sources are
synthetic but shaped as PR #69's; a second test repeats the three
constraints' blocks on the local processed files at Harvard Forest, and is
skipped without them.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import xarray as xr

from conftest import dated_observed_values, static_observed_values, windowed_observed_values
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    ObservationSource,
    ObservationVector,
    ReduceOverRun,
    SelectTimestep,
)
from sipnet_calibration.observation.model import noise_factor, prediction_components
from sipnet_calibration.probability import (
    REAL,
    ArraySpec,
    BlockDiagonalSpec,
    DenseSpec,
    DeterministicSpec,
    DiagonalSpec,
    FactorSpec,
    GaussianSpec,
    condition_on,
    joint,
    normal,
)
from sipnet_calibration.probability import _linalg

#: Harvard Forest, PR #69's site.
SITE = 4977
DAY = 86_400.0
NEE, LAI, BIOMASS, SOIL = (
    "nee_night_centered", "modis_leaf_area_index", "landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon",
)
RNG = np.random.default_rng(69)

# ── PR #69's config.py values ─────────────────────────────────────────────────

NEE_SHORT_STANDARD_DEVIATION, NEE_SHORT_TIMESCALE = 0.603, 0.740
NEE_LONG_STANDARD_DEVIATION, NEE_LONG_TIMESCALE = 1.20, 57.2
LAI_STANDARD_DEVIATION_FLOOR = 0.66
LAI_DISCREPANCY_STANDARD_DEVIATION, LAI_DISCREPANCY_TIMESCALE = 0.5, 30.0
WOOD_CARBON_FRACTION, WOOD_CARBON_FRACTION_UNCERTAINTY = 0.48, 0.02
LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION = 5.0
SOIL_CARBON_DISCREPANCY_FRACTION = 0.25


# ── PR #69's block builders, transcribed (model/noise.py, model/discrepancy.py) ──


def _times_in_days(source: ObservationSource) -> np.ndarray:
    times = pd.DatetimeIndex(source.observation_labels.get_level_values("time"))
    return ((times - times[0]) / pd.Timedelta(days=1)).to_numpy()


def _measurement(source: ObservationSource) -> np.ndarray:
    return source.standard_deviation.stack(_entry=source.standard_deviation.dims).dropna("_entry").values


def _observed(source: ObservationSource) -> np.ndarray:
    return source.observed_values.stack(_entry=source.observed_values.dims).dropna("_entry").values


def _nee_block(source: ObservationSource) -> np.ndarray:
    times = _times_in_days(source)
    distance = np.abs(times[:, None] - times[None, :])
    discrepancy = (NEE_SHORT_STANDARD_DEVIATION**2 * np.exp(-distance / NEE_SHORT_TIMESCALE)
                   + NEE_LONG_STANDARD_DEVIATION**2 * np.exp(-distance / NEE_LONG_TIMESCALE))
    return np.diag(_measurement(source) ** 2) + discrepancy


def _leaf_area_index_block(source: ObservationSource) -> np.ndarray:
    floored = np.maximum(_measurement(source), LAI_STANDARD_DEVIATION_FLOOR)
    times = _times_in_days(source)
    years = pd.DatetimeIndex(source.observation_labels.get_level_values("time")).year.to_numpy()
    same_summer = years[:, None] == years[None, :]
    correlation = np.exp(-np.abs(times[:, None] - times[None, :]) / LAI_DISCREPANCY_TIMESCALE)
    return np.diag(floored**2) + LAI_DISCREPANCY_STANDARD_DEVIATION**2 * correlation * same_summer


def _landtrendr_block(source: ObservationSource) -> np.ndarray:
    values, measurement = _observed(source), _measurement(source)
    relative = WOOD_CARBON_FRACTION_UNCERTAINTY / WOOD_CARBON_FRACTION
    return (np.outer(measurement, measurement) + relative**2 * np.outer(values, values)
            + LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION**2 * np.eye(values.size))


def _soil_carbon_block(source: ObservationSource) -> np.ndarray:
    values = _observed(source)
    return np.diag(_measurement(source) ** 2 + (SOIL_CARBON_DISCREPANCY_FRACTION * values) ** 2)


HAND_BLOCKS = {NEE: _nee_block, LAI: _leaf_area_index_block, BIOMASS: _landtrendr_block, SOIL: _soil_carbon_block}


# ── the same blocks as covariance specs (the design's §10.1) ──────────────────


def nee_night_block(time_since_epoch, standard_deviation):
    lag = jnp.abs(time_since_epoch[:, None] - time_since_epoch[None, :]) / DAY
    discrepancy = (NEE_SHORT_STANDARD_DEVIATION**2 * jnp.exp(-lag / NEE_SHORT_TIMESCALE)
                   + NEE_LONG_STANDARD_DEVIATION**2 * jnp.exp(-lag / NEE_LONG_TIMESCALE))
    return jnp.diag(standard_deviation**2) + discrepancy


def leaf_area_index_block(time_since_epoch, calendar_year, standard_deviation):
    lag = jnp.abs(time_since_epoch[:, None] - time_since_epoch[None, :]) / DAY
    same_summer = calendar_year[:, None] == calendar_year[None, :]
    floored = jnp.maximum(standard_deviation, LAI_STANDARD_DEVIATION_FLOOR)
    return (jnp.diag(floored**2)
            + LAI_DISCREPANCY_STANDARD_DEVIATION**2 * jnp.exp(-lag / LAI_DISCREPANCY_TIMESCALE) * same_summer)


def landtrendr_block(standard_deviation, observed):
    relative = WOOD_CARBON_FRACTION_UNCERTAINTY / WOOD_CARBON_FRACTION
    return (jnp.outer(standard_deviation, standard_deviation) + relative**2 * jnp.outer(observed, observed)
            + LANDTRENDR_DISCREPANCY_STANDARD_DEVIATION**2 * jnp.eye(observed.shape[0]))


def soil_carbon_variance(standard_deviation, observed):
    return standard_deviation**2 + (SOIL_CARBON_DISCREPANCY_FRACTION * observed) ** 2


COVARIANCES = {
    NEE: BlockDiagonalSpec(DenseSpec(nee_night_block), by="site"),
    LAI: BlockDiagonalSpec(DenseSpec(leaf_area_index_block), by="site"),
    BIOMASS: BlockDiagonalSpec(DenseSpec(landtrendr_block), by="site"),
    SOIL: DiagonalSpec(soil_carbon_variance),
}


# ── the sources ───────────────────────────────────────────────────────────────


def _with_gaps(observed: xr.DataArray, fraction: float) -> xr.DataArray:
    return observed.where(RNG.uniform(size=observed.shape) > fraction)


def _source(name, observed, standard_deviation, operator) -> ObservationSource:
    return ObservationSource(observation_source_name=name, observed_values=observed,
                             standard_deviation=standard_deviation, operator=operator)


def _synthetic_sources(sites=(SITE,)) -> list[ObservationSource]:
    """Four sources shaped as PR #69's: night NEE window means over five
    weeks, LAI composites over two summers (some standard deviations zero or
    below the floor), LandTrendr's 2012-2017, and one soil carbon stock."""
    sites = list(sites)
    nee_times = pd.date_range("2013-06-01T12:00", periods=35, freq="D")
    nee = _with_gaps(windowed_observed_values(sites, nee_times, window_length="12h", units="umol m-2 s-1",
                                              constituent="CO2", name=NEE,
                                              values=RNG.normal(2.0, 1.0, (len(sites), 35))), 0.2)
    lai_times = pd.date_range("2012-06-01", "2012-09-30", freq="8D").append(
        pd.date_range("2013-06-01", "2013-09-30", freq="8D"))
    lai = _with_gaps(dated_observed_values(sites, lai_times, values=RNG.uniform(1, 5, (len(sites), len(lai_times))),
                                           name=LAI), 0.3)
    lai_sd = lai.copy(data=RNG.choice([0.0, 0.1, 0.4, 0.9, 1.3], size=lai.shape))
    biomass_times = pd.date_range("2012-01-01", periods=6, freq="YS")
    biomass = dated_observed_values(sites, biomass_times, values=RNG.uniform(150, 250, (len(sites), 6)),
                                    units="Mg ha-1", constituent="C", name=BIOMASS)
    soil = static_observed_values(sites, values=RNG.uniform(50, 150, len(sites)), name=SOIL)
    return [
        _source(NEE, nee, nee.copy(data=RNG.uniform(0.2, 1.5, nee.shape)), SelectTimestep("net_ecosystem_exchange")),
        _source(LAI, lai, lai_sd, DEFAULT_OBS_OPS[LAI]),
        _source(BIOMASS, biomass, biomass.copy(data=RNG.uniform(5, 30, biomass.shape)), SelectTimestep("wood_carbon")),
        _source(SOIL, soil, soil.copy(data=RNG.uniform(5, 20, soil.shape)), ReduceOverRun("soil_carbon", how="mean")),
    ]


def _posterior(vector: ObservationVector, covariances=COVARIANCES):
    """The noise factors, each source predicted as a constant offset, and a
    prior on the offset; conditioned on the observed values."""
    offset = FactorSpec(ArraySpec("offset", units="1", support=REAL), law=normal(mean=0.0, standard_deviation=10.0))
    predictions = [
        DeterministicSpec(spec, function=_constant_of_size(vector.coords[spec.indexed_by[0]].size))
        for spec in prediction_components(vector)
    ]
    factors = [noise_factor(vector, name, covariance=covariances[name], provenance="PR #69's noise model.")
               for name in vector.observation_source_names]
    model = joint(*factors, *predictions, offset).bind(coords=vector.coords)
    return condition_on(model, vector.observed_values_by_component())


def _constant_of_size(size):
    return lambda offset: jnp.full((size,), offset)


def _hand_assembled(vector: ObservationVector) -> list[np.ndarray]:
    return [HAND_BLOCKS[name](vector[name]) for name in vector.observation_source_names]


# ── PR #69's R ────────────────────────────────────────────────────────────────


def test_pr_69s_one_site_noise_covariance_equals_its_hand_assembly():
    vector = ObservationVector(observation_sources=_synthetic_sources())
    likelihood = _posterior(vector).gaussian_likelihood()
    blocks = _hand_assembled(vector)
    np.testing.assert_allclose(np.asarray(likelihood.noise_covariance.to_dense()), scipy.linalg.block_diag(*blocks),
                               rtol=1e-12, atol=1e-12)
    kinds = [type(block).__name__ for block in likelihood.noise_covariance.blocks]
    assert kinds == ["PSDBlockDiag", "PSDBlockDiag", "PSDBlockDiag", "PSDDiagonal"]


def test_pr_69s_likelihood_scores_as_its_hand_built_gaussian():
    """PR #69's calibration_likelihood is a Gaussian of y with R one
    DensePSD per source, built here as EnsKit's; its log density at the
    predictions is the posterior's log likelihood."""
    vector = ObservationVector(observation_sources=_synthetic_sources())
    posterior = _posterior(vector)
    by_hand = _linalg.Gaussian.independent(y=(posterior.y, _linalg.PSDBlockDiag(tuple(
        _linalg.DensePSD(jnp.asarray(block)) for block in _hand_assembled(vector)))))
    theta = jnp.array([[0.0], [1.5], [-3.0]])
    predictions, _, evaluation = posterior.gaussian_likelihood().forward(theta)
    np.testing.assert_allclose(np.asarray(evaluation.log_likelihood), np.asarray(by_hand.log_density(y=predictions)),
                               rtol=1e-11)


def test_the_noise_model_at_several_sites_has_one_block_per_site():
    """What PR #69 could not write: the same noise model at three sites."""
    sites = (4480, 4705, SITE)
    vector = ObservationVector(observation_sources=_synthetic_sources(sites))
    likelihood = _posterior(vector).gaussian_likelihood()
    nee = likelihood.noise_covariance.blocks[0]
    assert len(nee.blocks) == 3
    for site, block in zip(sites, nee.blocks):
        expected = _nee_block(vector.select(sites=[site])[NEE])
        np.testing.assert_allclose(np.asarray(block.to_dense()), expected, rtol=1e-12, atol=1e-12)


def test_pr_69s_constraint_blocks_on_the_local_data_equal_their_hand_assembly():
    constraints = pytest.importorskip("sipnet_calibration.constraints")
    names = [LAI, BIOMASS, SOIL]
    try:
        observed = constraints.constraint_fields(names, sites=[SITE])
        standard_deviations = constraints.constraint_standard_deviations(names, sites=[SITE])
    except FileNotFoundError as error:
        pytest.skip(f"constraints' processed files not available in this working copy: {error}")
    years = pd.DatetimeIndex(observed[BIOMASS]["time"].values).year
    in_period = (years >= 2012) & (years <= 2017)
    observed[BIOMASS] = observed[BIOMASS].isel(time=in_period)
    operators = {LAI: DEFAULT_OBS_OPS[LAI], BIOMASS: SelectTimestep("wood_carbon"),
                 SOIL: ReduceOverRun("soil_carbon", how="mean")}
    vector = ObservationVector(observation_sources=[
        _source(name, observed[name], standard_deviations[name], operators[name]) for name in names
    ])
    likelihood = _posterior(vector).gaussian_likelihood()
    np.testing.assert_allclose(np.asarray(likelihood.noise_covariance.to_dense()),
                               scipy.linalg.block_diag(*_hand_assembled(vector)), rtol=1e-12, atol=1e-12)


# ── noise_factor ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vector() -> ObservationVector:
    return ObservationVector(observation_sources=_synthetic_sources((4480, SITE)))


def test_a_noise_factor_centers_a_sources_observed_component_on_its_prediction(vector):
    factor = noise_factor(vector, LAI, covariance=COVARIANCES[LAI], provenance="test")
    assert isinstance(factor, FactorSpec) and isinstance(factor.law, GaussianSpec)
    (event,) = factor.event
    assert (event.name, event.indexed_by, event.units) == (LAI, (vector.observation_dim_name(LAI),), "m2 m-2")
    assert factor.law.mean == (vector.prediction_name(LAI),)
    assert factor.given == (vector.prediction_name(LAI),)
    assert factor.provenance == "test"


def test_a_noise_factor_holds_the_source_constants_its_covariance_reads(vector):
    assert set(noise_factor(vector, LAI, covariance=COVARIANCES[LAI]).constants) == {
        "time_since_epoch", "calendar_year", "standard_deviation"}
    assert set(noise_factor(vector, SOIL, covariance=COVARIANCES[SOIL]).constants) == {"standard_deviation", "observed"}


def test_a_noise_factor_takes_more_constants_and_label_maps(vector):
    floor = xr.DataArray(0.7)
    year = vector.year_label_map(LAI)
    factor = noise_factor(
        vector, LAI,
        covariance=DiagonalSpec(lambda standard_deviation, floor, year_of: jnp.maximum(standard_deviation, floor) ** 2
                                + 0.0 * year_of),
        constants={"floor": floor}, label_maps={"year_of": year},
    )
    assert set(factor.constants) == {"standard_deviation", "floor"} and set(factor.label_maps) == {"year_of"}


def test_a_noise_factor_constant_named_like_the_sources_is_refused(vector):
    with pytest.raises(ValueError, match="named like the source's own constants"):
        noise_factor(vector, SOIL, covariance=COVARIANCES[SOIL], constants={"observed": xr.DataArray(1.0)})


@pytest.mark.parametrize("names", [[LAI, SOIL], [LAI]])
def test_a_noise_factor_is_of_one_source_named_by_a_string(names):
    vector = ObservationVector(observation_sources=_synthetic_sources())
    with pytest.raises(TypeError, match="give one source's name"):
        noise_factor(vector, names, covariance=COVARIANCES[LAI])


def test_a_noise_factor_of_a_source_not_in_the_vector_is_refused(vector):
    with pytest.raises(KeyError):
        noise_factor(vector, "gedi_aboveground_biomass", covariance=COVARIANCES[LAI])


def test_a_noise_factor_reading_a_constant_its_source_lacks_is_refused(vector):
    with pytest.raises(ValueError, match=r"reads \['time_since_epoch'\], which the source does not have"):
        noise_factor(vector, SOIL, covariance=BlockDiagonalSpec(DenseSpec(nee_night_block), by="site"))
    bare = ObservationVector(observation_sources=[ObservationSource(
        observation_source_name=SOIL, observed_values=vector[SOIL].observed_values,
        operator=ReduceOverRun("soil_carbon", how="mean"))])
    with pytest.raises(ValueError, match="give the source a standard deviation"):
        noise_factor(bare, SOIL, covariance=COVARIANCES[SOIL])


def test_a_noise_factors_covariance_and_constants_are_of_their_types(vector):
    with pytest.raises(TypeError, match="give a covariance spec"):
        noise_factor(vector, SOIL, covariance="standard_deviation")
    with pytest.raises(TypeError, match=r"give \{name: DataArray\}"):
        noise_factor(vector, SOIL, covariance=COVARIANCES[SOIL], constants=[1])

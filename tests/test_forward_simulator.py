"""The SIPNET simulator: ``SIPNETRuns`` and ``SIPNETSimulator`` on the
stand-in SIPNET, through the probability layer.

The example calibration at sites 1 and 27, as ``test_forward`` and P1's
``forward_example`` reference have it, is declared as factors with a noise
factor per observation source and conditioned on the observed values. Its
predictions must be P1's, bit for bit; a failed run must invalidate only
the predictions of the sources observing its site; one pass of the runs
must serve several reductions; and the simulator is restricted, checked
and fed as the core asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import SequentialBackend
from pysipnet import niwot_reference_output
from tensorflow_probability.substrates import jax as tfp

from conftest import (
    BLOW_UP,
    REPOSITORY,
    scaled_niwot_model,
    site_table_of,
    two_source_observation_vector,
)
from sipnet_calibration.calibration import example_calibration
from sipnet_calibration.forward import SIPNETRuns, SIPNETRunsEvaluation, SIPNETSimulator
from sipnet_calibration.observation import ObservationVector
from sipnet_calibration.observation.model import noise_factor, observed_components
from sipnet_calibration.probability import (
    POSITIVE,
    REAL,
    ArraySpec,
    DiagonalSpec,
    FactorSpec,
    condition_on,
    iid_over_dim,
    joint,
    log_normal,
    normal,
)
from sipnet_calibration.sipnet_parameter_map import SIPNETParameterMap, ValueRequirement
from sipnet_calibration.site_dims import SiteDims

tfd = tfp.distributions

SITES = (1, 27)
PFT = ("temperate.deciduous", "boreal.coniferous")
SITE_TABLE = site_table_of(*SITES, lon=[-105.0, -70.0], lat=[40.0, 45.0])
SITE_DIMS = SiteDims(site_table=SITE_TABLE, site_labels={"pft": PFT})
REFERENCE = niwot_reference_output()
CLIMATE = {1: REFERENCE.climate, 27: REFERENCE.climate.head(40)}
FORWARD_REFERENCE = REPOSITORY / "tests" / "data" / "probability_references" / "forward_example.nc"
BIOMASS, LAI = "landtrendr_aboveground_biomass", "modis_leaf_area_index"
RATE_UNITS = "nmol g-1 s-1"
KEY = jax.random.key(5)


@dataclass(frozen=True, kw_only=True)
class CopyRate:
    """A rule copying a rate to ``max_photosynthesis_rate`` with no bounds,
    so a test can send the stand-in its failure bands."""

    constants: ClassVar[dict] = {}
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()
    sipnet_parameter_names_written: ClassVar[tuple[str, ...]] = ("max_photosynthesis_rate",)

    @property
    def values_read(self):
        return {"rate": ValueRequirement(RATE_UNITS)}

    def __call__(self, values, sipnet_parameter_values):
        return {"max_photosynthesis_rate": values["rate"]}


def _sipnet_map():
    return example_calibration(SITE_DIMS)[1]


def _runs(sipnet_map=None, **keywords):
    return SIPNETRuns(
        scaled_niwot_model(), sipnet_parameter_map=sipnet_map or _sipnet_map(), site_dims=SITE_DIMS,
        climate=CLIMATE, backend=SequentialBackend(), **keywords,
    )


def _noise_factors(observation_vector):
    """One noise factor per source, ``N(prediction, 10^2)`` per observation."""
    laws = {
        BIOMASS: lambda predicted_landtrendr_aboveground_biomass: tfd.Independent(
            tfd.Normal(predicted_landtrendr_aboveground_biomass, 10.0), 1),
        LAI: lambda predicted_modis_leaf_area_index: tfd.Independent(tfd.Normal(predicted_modis_leaf_area_index, 10.0), 1),
    }
    return [FactorSpec(spec, law=laws[spec.name], provenance="Test noise.") for spec in observed_components(observation_vector)]


def _model(observation_vector, simulator, factors=None, *, inputs=(), input_values=None):
    factors = list(example_calibration(SITE_DIMS)[0]) if factors is None else factors
    spec = joint(*_noise_factors(observation_vector), simulator, *factors, inputs=list(inputs))
    return spec.bind(coords={**SITE_DIMS.coords, **observation_vector.coords}, inputs=input_values)


def _posterior(observation_vector=None, simulator=None, **keywords):
    observation_vector = observation_vector or two_source_observation_vector(SITE_TABLE)
    simulator = simulator or SIPNETSimulator(_runs(), observation_vector=observation_vector)
    model = _model(observation_vector, simulator, **keywords)
    return condition_on(model, observation_vector.observed_values_by_component())


@pytest.fixture(scope="module")
def reference():
    with xr.open_dataset(FORWARD_REFERENCE, engine="h5netcdf") as dataset:
        return dataset.load()


@pytest.fixture(scope="module")
def posterior():
    return _posterior()


# ── parity with today's forward model ─────────────────────────────────────────


def test_theta_is_laid_out_as_the_forward_references(posterior, reference):
    index = posterior.parameters.unconstrained.index
    assert list(index.get_level_values("component")) == list(reference["entry_parameter"].values)


def test_predictions_are_todays_bit_for_bit(posterior, reference):
    evaluation = posterior.evaluate(reference["theta"].values)
    assert evaluation.simulator_valid.tolist() == reference["valid"].values.tolist()
    vector = posterior.model.simulators["sipnet"].observation_vector
    stored = pd.MultiIndex.from_arrays(
        [reference["observation_source"].values, reference["observation_site"].values,
         pd.DatetimeIndex(reference["observation_time"].values)],
        names=["observation_source", "site", "time"],
    )
    for source in vector.observation_source_names:
        labels = vector.coords[vector.observation_dim_name(source)]
        keys = pd.MultiIndex.from_arrays(
            [[source] * len(labels), labels.get_level_values("site"), labels.get_level_values("time")],
            names=stored.names,
        )
        expected = reference["predictions"].values[:, stored.get_indexer(keys)]
        np.testing.assert_array_equal(np.asarray(evaluation.values[vector.prediction_name(source)]), expected)


def test_the_gaussian_likelihoods_forward_map_is_todays_predictions_in_ys_order(reference):
    vector = two_source_observation_vector(SITE_TABLE)
    noise = [
        noise_factor(vector, name, covariance=DiagonalSpec(lambda observed: (0.1 * observed) ** 2 + 1.0),
                     provenance="Test noise.")
        for name in vector.observation_source_names
    ]
    simulator = SIPNETSimulator(_runs(), observation_vector=vector)
    model = joint(*noise, simulator, *list(example_calibration(SITE_DIMS)[0])).bind(coords={**SITE_DIMS.coords, **vector.coords})
    likelihood = condition_on(model, vector.observed_values_by_component()).gaussian_likelihood()
    predictions, valid, _ = likelihood.forward(reference["theta"].values)
    stored = pd.MultiIndex.from_arrays(
        [reference["observation_source"].values, reference["observation_site"].values,
         pd.DatetimeIndex(reference["observation_time"].values)],
    )
    in_y = pd.MultiIndex.from_tuples([
        (source, site, time)
        for source in vector.observation_source_names
        for site, time in vector.coords[vector.observation_dim_name(source)]
    ])
    assert valid.tolist() == reference["valid"].values.tolist()
    np.testing.assert_array_equal(np.asarray(predictions), reference["predictions"].values[:, stored.get_indexer(in_y)])
    observed = np.concatenate([np.asarray(v) for v in vector.observed_values_by_component().values()])
    np.testing.assert_allclose(np.asarray(likelihood.noise_covariance.diag()), (0.1 * observed) ** 2 + 1.0)


def test_the_record_is_the_runs_evaluation(posterior, reference):
    evaluation = posterior.evaluate(reference["theta"].values[:2])
    record = evaluation.simulator_records["sipnet"]
    assert isinstance(record, SIPNETRunsEvaluation)
    assert record.run_succeeded.dims == ("sample", "site")
    assert record.observation_vectors[0] is posterior.simulators["sipnet"].observation_vector


# ── failures ──────────────────────────────────────────────────────────────────


def _rate_posterior():
    """A posterior whose rate is a parameter per site, with biomass at both
    sites and LAI at site 27 alone."""
    both = two_source_observation_vector(SITE_TABLE)
    vector = ObservationVector(observation_sources=[
        both[BIOMASS], both.select(observation_source_names=[LAI], sites=[27])[LAI],
    ])
    example_map = _sipnet_map()
    rate_map = SIPNETParameterMap(rules=[CopyRate(), *example_map.rules[1:]], fixed=example_map.fixed)
    rate = FactorSpec(ArraySpec("rate", units=RATE_UNITS, support=POSITIVE, indexed_by=("site",)),
                      law=iid_over_dim(log_normal(median=100.0, geometric_sd=1.5)), provenance="Test rate.")
    simulator = SIPNETSimulator(_runs(rate_map, out_of_domain="fail_row"), observation_vector=vector)
    return _posterior(vector, simulator, factors=[rate, *list(example_calibration(SITE_DIMS)[0])[1:]]), vector


def test_a_failed_run_invalidates_only_the_predictions_of_its_site():
    posterior, vector = _rate_posterior()
    theta = np.asarray(posterior.sample_prior(KEY, 3))
    theta = jnp.asarray(theta).at[1, posterior.parameters.unconstrained.positions(component=["rate"], site=[1])].set(
        np.log(1.5 * BLOW_UP)
    )
    simulator = posterior.simulators["sipnet"]
    output = simulator(posterior.simulator_inputs(theta, "sipnet"))
    biomass, lai = vector.prediction_name(BIOMASS), vector.prediction_name(LAI)
    assert output.valid[biomass].tolist() == [True, False, True]
    assert output.valid[lai].tolist() == [True, True, True]
    assert np.isnan(output.values[biomass][1, :2]).all() and np.isfinite(output.values[lai][1]).all()
    assert output.record.failures["site"].tolist() == [1]

    evaluation = posterior.evaluate(theta)
    assert evaluation.valid.tolist() == [True, False, True]
    assert evaluation.log_likelihood[1] == -np.inf and np.isfinite(np.asarray(evaluation.log_likelihood)[[0, 2]]).all()
    assert np.isfinite(evaluation.log_prior).all()


# ── one pass, several reductions ──────────────────────────────────────────────


def test_one_pass_serves_two_vectors_and_daily_output(posterior, reference):
    runs = _runs()
    values = posterior.simulator_inputs(reference["theta"].values, "sipnet")
    vector = two_source_observation_vector(SITE_TABLE)
    biomass = vector.select(observation_source_names=[BIOMASS])
    lai = vector.select(observation_source_names=[LAI])
    together = runs.evaluate(values, observation_vectors=[biomass, lai], output_variable_names=["wood_carbon"], freq="1D")
    alone = [
        runs.evaluate(values, observation_vectors=[biomass]),
        runs.evaluate(values, observation_vectors=[lai]),
        runs.evaluate(values, output_variable_names=["wood_carbon"], freq="1D"),
    ]
    for k in (0, 1):
        assert list(together.predictions[k]) == list(alone[k].predictions[0])
        for name, values in together.predictions[k].items():
            np.testing.assert_array_equal(values, alone[k].predictions[0][name])
    xr.testing.assert_identical(together.model_output, alone[2].model_output)
    assert together.model_output.attrs["resampling_frequency"] == "1D"
    xr.testing.assert_identical(together.run_succeeded, alone[2].run_succeeded)


def test_the_runs_refuse_what_they_cannot_do(posterior, reference):
    runs = _runs()
    values = posterior.simulator_inputs(reference["theta"].values[:1], "sipnet")
    with pytest.raises(ValueError, match="nothing is asked for"):
        runs.evaluate(values)
    with pytest.raises(ValueError, match="freq= aggregates model output"):
        runs.evaluate(values, observation_vectors=[two_source_observation_vector(SITE_TABLE)], freq="1D")
    with pytest.raises(ValueError, match="no value the map reads is on the batch dim"):
        runs.evaluate({n: v.isel(sample=0, drop=True) for n, v in values.items()}, output_variable_names=["wood_carbon"])
    with pytest.raises(ValueError, match="label the samples 0 to J - 1"):
        runs.evaluate({n: v.assign_coords(sample=[4]) for n, v in values.items()}, output_variable_names=["wood_carbon"])
    with pytest.raises(KeyError, match="has no value for site"):
        runs.evaluate({n: v.sel(site=[1]) if "site" in v.dims else v for n, v in values.items()},
                      output_variable_names=["wood_carbon"])
    with pytest.raises(TypeError, match="labeled values"):
        runs.evaluate([1.0], output_variable_names=["wood_carbon"])


# ── restriction ───────────────────────────────────────────────────────────────


def test_the_simulator_runs_only_at_the_sites_it_observes():
    vector = two_source_observation_vector(SITE_TABLE).select(sites=[27])
    simulator = SIPNETSimulator(_runs(), observation_vector=vector)
    assert simulator.runs.sites == (27,)
    posterior = _posterior(vector, simulator)
    record = posterior.evaluate(jnp.zeros((1, posterior.dimension))).simulator_records["sipnet"]
    assert record.run_succeeded["site"].values.tolist() == [27]


def test_at_restricts_to_the_outputs_and_sites_asked_for():
    vector = two_source_observation_vector(SITE_TABLE)
    simulator = SIPNETSimulator(_runs(), observation_vector=vector)
    coords = {**SITE_DIMS.coords, **vector.coords}
    assert simulator.at(coords, [o.name for o in simulator.outputs]) is simulator
    fewer = simulator.at(coords, [vector.prediction_name(LAI)])
    assert fewer.observation_vector.observation_source_names == (LAI,)
    at_27 = vector.select(sites=[27])
    restricted = simulator.at({**coords, **at_27.coords}, [o.name for o in simulator.outputs])
    assert restricted.runs.sites == (27,) and restricted.observation_vector.sites == (27,)
    shuffled = vector.coords[vector.observation_dim_name(LAI)][::-1]
    with pytest.raises(ValueError, match="not the observation vector's observations"):
        simulator.at({**coords, vector.observation_dim_name(LAI): shuffled}, [vector.prediction_name(LAI)])
    with pytest.raises(ValueError, match="lack, so no value would be read"):
        simulator.at({**coords, "site": pd.Index([1])}, [o.name for o in simulator.outputs])


def test_a_model_selected_at_one_site_runs_one_site():
    vector = two_source_observation_vector(SITE_TABLE)
    model = _model(vector, SIPNETSimulator(_runs(), observation_vector=vector))
    one = model.select(site=[27])
    assert one.simulators["sipnet"].runs.sites == (27,)
    posterior = condition_on(one, one.simulators["sipnet"].observation_vector.observed_values_by_component())
    assert posterior.evaluate(jnp.zeros((1, posterior.dimension))).valid.tolist() == [True]


# ── what the simulator reads ──────────────────────────────────────────────────


def test_the_map_is_checked_against_the_declared_components():
    factors = list(example_calibration(SITE_DIMS)[0])
    soil = factors[-1]
    wrong_units = FactorSpec(ArraySpec("initial_soil_carbon", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
                             law=soil.law, provenance="Wrong units.")
    with pytest.raises(ValueError, match="initial_soil_carbon"):
        _posterior(factors=[*factors[:-1], wrong_units])


def test_the_map_is_checked_at_the_corners_of_the_target():
    factors = list(example_calibration(SITE_DIMS)[0])
    unbounded = FactorSpec(ArraySpec("leaf_fall_fraction", units="1", support=REAL),
                           law=normal(mean=0.5, standard_deviation=0.1), provenance="Unbounded.")
    with pytest.raises(ValueError, match="outside their domains, at a corner"):
        _posterior(factors=[*factors[:4], unbounded, factors[5]])
    vector = two_source_observation_vector(SITE_TABLE)
    lenient = SIPNETSimulator(_runs(out_of_domain="fail_row"), observation_vector=vector)
    _posterior(vector, lenient, factors=[*factors[:4], unbounded, factors[5]])


def test_an_external_initial_state_is_an_input_the_simulator_reads():
    """F7: the initial soil carbon held at each site's value, declared an input."""
    soil = xr.DataArray([20_000.0, 40_000.0], dims="site", coords={"site": list(SITES)})
    posterior = _posterior(
        factors=list(example_calibration(SITE_DIMS)[0])[:-1],
        inputs=[ArraySpec("initial_soil_carbon", units="g m-2", support=POSITIVE, indexed_by=("site",))],
        input_values={"initial_soil_carbon": soil},
    )
    assert "initial_soil_carbon" not in posterior.parameter_names
    record = posterior.evaluate(jnp.zeros((2, posterior.dimension))).simulator_records["sipnet"]
    np.testing.assert_array_equal(
        record.sipnet_parameter_fields["soil_carbon"].transpose("sample", "site").values, [[20_000.0, 40_000.0]] * 2
    )


def test_the_simulator_inputs_feed_a_daily_figure(posterior, reference):
    values = posterior.simulator_inputs(reference["theta"].values[:2], "sipnet")
    daily = _runs().evaluate(values, output_variable_names=["wood_carbon"], freq="1D").model_output
    assert daily["wood_carbon"].dims == ("sample", "site", "time")
    assert daily.sizes["sample"] == 2

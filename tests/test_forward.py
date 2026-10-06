"""The SIPNET runs: labeled values and external inputs to predictions and
model output through PyEns, on a stand-in and on SIPNET.

``SIPNETRuns.evaluate`` is fed the example calibration's labeled values at a
batch of theta, from its prior conditioned on nothing. Tested here: that
each run receives its own sample's and site's SIPNET parameters and drivers,
and its operators the run's own SIPNET parameters; how predictions are
placed per source on its observation dim; failures at the parameters
against failures of the machinery, across a process boundary too; values
outside their domains, refused or failing their rows; external inputs
zipping with the samples or crossing them by dim name, and the run index;
prior-predictive model output, aggregated on the worker with ``freq``; the
refusals; the batch dim's name; ``run_succeeded`` as a field; real SIPNET;
and the SCC backend preset. ``test_forward_simulator`` tests the runs as the
probability layer's simulator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import LocalBackend, SequentialBackend
from pysipnet import niwot_reference_output, niwot_reference_parameters
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import ModelFlags
from pysipnet.resample import STEP_LENGTH_RESAMPLED
from pysipnet.runner import SIPNETRunner

from conftest import (
    BLOW_UP,
    DIES_BAND,
    INVALID,
    NAN_BAND,
    SOIL_REFERENCE,
    TIMEOUT_BAND,
    ScaledNiwotRunner,
    located,
    scaled_niwot_model,
    site_table_of,
    with_parameter_value,
)
from sipnet_calibration.calibration import example_calibration
from sipnet_calibration.compute import scc_backend
from sipnet_calibration.fields import sipnet_overrides
from sipnet_calibration.forward import SIPNETRuns, SIPNETRunsEvaluation, SIPNETSimulator
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    ObservationSource,
    ObservationVector,
    ReduceOverRun,
    SelectTimestep,
    aggregate_time,
    extract_sipnet_parameter_at_coords,
    restrict_to_observed_sites,
    select_timestep_at,
)
from sipnet_calibration.probability import POSITIVE, ArraySpec, DeterministicSpec, condition_on, joint
from sipnet_calibration.sipnet_parameter_map import (
    Copy,
    SIPNETParameterMap,
    SIPNETParametersOutOfDomainError,
    ValueRequirement,
)
from sipnet_calibration.site_dims import SiteDims

SITES = (1, 27)
PFT = ("temperate.deciduous", "boreal.coniferous")
REFERENCE = niwot_reference_output()
REFERENCE_WOOD = REFERENCE.select(["wood_carbon"])["wood_carbon"]
SHORT_STEPS = 40  # site 27's drivers are cut to this many steps
LABELS = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30]])
SITE_TABLE = site_table_of(*SITES, lon=[-105.0, -70.0], lat=[40.0, 45.0])
SITE_DIMS = SiteDims(site_table=SITE_TABLE, site_labels={"pft": PFT})
RATE_UNITS = "nmol g-1 s-1"
BIOMASS, LAI = "landtrendr_aboveground_biomass", "modis_leaf_area_index"


class Foreign:
    """Another library's exception, named like a model failure but not one."""

    class TimeoutExpired(Exception):
        """Keyword-only, so it does not unpickle and PyEns sends a RemoteError."""

        def __init__(self, *, detail):
            super().__init__(detail)


class ForeignNiwotRunner(ScaledNiwotRunner):
    """The stand-in, with the machinery failure raised as :class:`Foreign.TimeoutExpired`."""

    def run(self, parameters, climate, *, events=None, **keywords):
        if float(parameters.dataarray("max_photosynthesis_rate")) > DIES_BAND:
            raise Foreign.TimeoutExpired(detail="the node died")
        return super().run(parameters, climate, events=events, **keywords)


#: What :class:`OwnLeafCarbonRunner` multiplies the run's leaf_carbon_per_area by.
OWN_LEAF_CARBON_FACTOR = 2.0


class OwnLeafCarbonRunner(ScaledNiwotRunner):
    """The stand-in, whose run used another leaf_carbon_per_area than it was given."""

    def run(self, parameters, climate, *, events=None, **keywords):
        result = super().run(parameters, climate, events=events, **keywords)
        leaf = float(parameters.dataarray("leaf_carbon_per_area"))
        return SimpleNamespace(
            outputs=result.outputs,
            parameters=with_parameter_value(parameters, "leaf_carbon_per_area", OWN_LEAF_CARBON_FACTOR * leaf),
        )


class NoParametersRunner(ScaledNiwotRunner):
    """The stand-in, whose result carries no parameters."""

    def run(self, parameters, climate, *, events=None, **keywords):
        return SimpleNamespace(outputs=super().run(parameters, climate).outputs)


@dataclass(frozen=True)
class ReadsSoilCarbon:
    """An operator whose prediction is the run's ``soil_carbon`` parameter itself."""

    output_variable_names = ("wood_carbon",)
    sipnet_parameter_names_read = ("soil_carbon",)

    def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
        wood = restrict_to_observed_sites(model_output["wood_carbon"], observed_values)
        picked = select_timestep_at(wood, observed_values["time"])
        soil = extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "soil_carbon", picked)
        result = picked * 0.0 + soil
        result.attrs = dict(soil.attrs)
        return result


@dataclass(frozen=True, kw_only=True)
class CopyRate:
    """A rule copying an external rate to ``max_photosynthesis_rate`` with no
    bounds, so a test can send the stand-in any rate, its failure bands and
    pySIPNET's refusals included."""

    value_name: str = "rate_input"
    constants: ClassVar[dict] = {}
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()
    sipnet_parameter_names_written: ClassVar[tuple[str, ...]] = ("max_photosynthesis_rate",)

    @property
    def values_read(self):
        return {self.value_name: ValueRequirement(RATE_UNITS)}

    def __call__(self, values, sipnet_parameter_values):
        return {"max_photosynthesis_rate": values[self.value_name]}


def fields_of(sipnet_map, values):
    """The SIPNET parameter fields at the labeled values, as the runs make them."""
    return sipnet_map.sipnet_parameter_fields(values, site_dims=SITE_DIMS)


def predicted_fields(evaluation, observation_vector, k=0):
    """The *k*-th vector's predictions as fields, keyed by source, on
    ``(*run index levels, site, time)``."""
    run_index = evaluation.run_index
    names = list(run_index.names)
    levels = [np.asarray(run_index.get_level_values(name).unique()) for name in names]
    labeled = {}
    for source in observation_vector.observation_source_names:
        dim = observation_vector.observation_dim_name(source)
        values = np.asarray(evaluation.predictions[k][observation_vector.prediction_name(source)])
        array = xr.DataArray(
            values.reshape(*(len(level) for level in levels), -1),
            dims=(*names, dim),
            coords=dict(zip(names, levels)),
        )
        labeled[source] = array.assign_coords(
            xr.Coordinates.from_pandas_multiindex(observation_vector.coords[dim], dim)
        )
    return observation_vector.to_fields(labeled)


def columns_at(observation_vector, source, site):
    """The columns of *source*'s predictions at *site*: its segment of the
    source's observation dim."""
    labels = observation_vector.coords[observation_vector.observation_dim_name(source)]
    return np.flatnonzero(np.asarray(labels.get_level_values("site")) == site)


def _expected_wood(sipnet_parameter_fields, sample, site, n_steps=None, **labels):
    """What the stand-in writes for wood carbon at this run, in Mg ha-1."""
    at = {"sample": sample, "site": site, **labels}
    fields = sipnet_parameter_fields
    rate = float(fields["max_photosynthesis_rate"].sel({k: v for k, v in at.items() if k in fields["max_photosynthesis_rate"].dims}))
    soil = float(fields["soil_carbon"].sel({k: v for k, v in at.items() if k in fields["soil_carbon"].dims}))
    wood = REFERENCE_WOOD if n_steps is None else REFERENCE_WOOD.isel(time=slice(0, n_steps))
    return select_timestep_at(wood, LABELS).values * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01


@pytest.fixture(scope="module")
def example():
    return example_calibration(SITE_DIMS)


@pytest.fixture(scope="module")
def prior_factors(example):
    return example[0]


@pytest.fixture(scope="module")
def sipnet_map(example):
    return example[1]


@pytest.fixture(scope="module")
def prior(prior_factors):
    """The example's prior over theta: its factors conditioned on nothing."""
    return condition_on(joint(*prior_factors).bind(coords=SITE_DIMS.coords), {})


@pytest.fixture(scope="module")
def rate_map(sipnet_map):
    """The example's map with the photosynthesis rule replaced by :class:`CopyRate`."""
    return SIPNETParameterMap(rules=[CopyRate(), *sipnet_map.rules[1:]], fixed=sipnet_map.fixed)


def rates(values: dict[tuple[int, int], float], n_samples: int = 3, default: float = 100.0) -> xr.Dataset:
    """An external rate on (sample, site): *default*, except at (sample, site position)."""
    table = np.full((n_samples, len(SITES)), default)
    for (sample, position), rate in values.items():
        table[sample, position] = rate
    return xr.Dataset(
        {"rate_input": (("sample", "site"), table, {"units": RATE_UNITS})},
        coords={"sample": np.arange(n_samples), "site": np.asarray(SITES, dtype=np.int32)},
    )


@pytest.fixture(scope="module")
def climate():
    """Site 1 on the full Niwot record, site 27 on a shorter one."""
    return {1: REFERENCE.climate, 27: REFERENCE.climate.head(SHORT_STEPS)}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The same drivers as files, for the process backends."""
    directory = tmp_path_factory.mktemp("two-sites")
    paths = {}
    for site, drivers in {1: REFERENCE.climate, 27: REFERENCE.climate.head(SHORT_STEPS)}.items():
        path = directory / f"site_{site}.clim"
        drivers.to_file(path)
        paths[site] = ClimateDrivers.from_path(path)
    return paths


@pytest.fixture(scope="module")
def observation_vector():
    wood = located(xr.DataArray(
        [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
        dims=("site", "time"),
        coords={"site": list(SITES), "time": LABELS},
        attrs={"units": "Mg ha-1", "constituent": "C"},
        name=BIOMASS,
    ), site_table=SITE_TABLE)
    lai = located(xr.DataArray(
        [[3.0, 2.0, np.nan], [np.nan, 1.0, 1.5]],
        dims=("site", "time"),
        coords={"site": list(SITES), "time": LABELS},
        attrs={"units": "m2 m-2"},
        name=LAI,
    ), site_table=SITE_TABLE)
    return ObservationVector(
        observation_sources=[
            ObservationSource(observation_source_name=BIOMASS, observed_values=wood,
                              operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name=LAI, observed_values=lai,
                              operator=DEFAULT_OBS_OPS[LAI]),
        ]
    )


def build(sipnet_map, climate, *, model=None, **keywords):
    keywords.setdefault("backend", SequentialBackend())
    keywords.setdefault("site_dims", SITE_DIMS)
    return SIPNETRuns(model or scaled_niwot_model(), sipnet_parameter_map=sipnet_map, climate=climate, **keywords)


@pytest.fixture
def runs(sipnet_map, climate):
    return build(sipnet_map, climate)


@pytest.fixture(scope="module")
def theta(prior):
    return np.asarray(prior.sample_prior(jax.random.key(3), 3))


@pytest.fixture(scope="module")
def values(prior, theta):
    """The labeled values at theta, on ``sample``."""
    return prior.to_labeled(theta)


class TestEvaluate:
    def test_the_right_parameters_and_drivers_reach_the_right_run(
        self, runs, sipnet_map, observation_vector, values
    ):
        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        assert isinstance(evaluation, SIPNETRunsEvaluation)
        fields = fields_of(sipnet_map, values)
        assert not np.allclose(fields["soil_carbon"].sel(site=1), fields["soil_carbon"].sel(site=27))
        predicted = predicted_fields(evaluation, observation_vector)[BIOMASS]
        for sample in range(3):
            for site in SITES:
                observed = observation_vector[BIOMASS].observed_values.sel(site=site).notnull().values
                np.testing.assert_allclose(
                    predicted.sel(sample=sample, site=site).values[observed],
                    _expected_wood(fields, sample, site)[observed],
                    rtol=1e-12,
                )
        assert evaluation.valid.all() and evaluation.failures.empty
        assert evaluation.model_output is None
        assert evaluation.observation_vectors == (observation_vector,)
        assert list(evaluation.run_index) == [0, 1, 2] and evaluation.run_index.name == "sample"

    def test_the_predictions_are_one_mapping_per_vector_on_its_observation_dims(
        self, runs, observation_vector, values
    ):
        lai = observation_vector.select(observation_source_names=[LAI])
        evaluation = runs.evaluate(values, observation_vectors=[observation_vector, lai])
        assert len(evaluation.predictions) == 2
        assert list(evaluation.predictions[0]) == [f"predicted_{BIOMASS}", f"predicted_{LAI}"]
        assert list(evaluation.predictions[1]) == [f"predicted_{LAI}"]
        for source in observation_vector.observation_source_names:
            labels = observation_vector.coords[observation_vector.observation_dim_name(source)]
            predicted = evaluation.predictions[0][observation_vector.prediction_name(source)]
            assert isinstance(predicted, jax.Array) and predicted.shape == (3, len(labels))
        np.testing.assert_array_equal(evaluation.predictions[1][f"predicted_{LAI}"],
                                      evaluation.predictions[0][f"predicted_{LAI}"])
        with pytest.raises(TypeError):
            evaluation.predictions[0][f"predicted_{LAI}"] = None

    def test_an_operator_reads_the_sipnet_parameters_the_run_used(self, sipnet_map, climate, values):
        observed = located(
            xr.DataArray(
                [[1.0, np.nan, 1.0], [1.0, 1.0, np.nan]],
                dims=("site", "time"),
                coords={"site": list(SITES), "time": LABELS},
                attrs={"units": "g m-2", "constituent": "C"},
                name="soil",
            ),
            site_table=SITE_TABLE,
        )
        observation_vector = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="soil", observed_values=observed, operator=ReadsSoilCarbon())
        ])
        evaluation = build(sipnet_map, climate).evaluate(values, observation_vectors=[observation_vector])
        predicted = predicted_fields(evaluation, observation_vector)["soil"]
        expected = fields_of(sipnet_map, values)["soil_carbon"]
        for sample in range(3):
            for site in SITES:
                row = predicted.sel(sample=sample, site=site).values
                np.testing.assert_allclose(row[np.isfinite(row)], float(expected.sel(sample=sample, site=site)))

    def test_the_lai_operator_reads_the_runs_own_leaf_carbon_per_area(self, runs, observation_vector, values):
        """The map leaves it to the base parameter set, which the run used."""
        assert "leaf_carbon_per_area" not in runs.sipnet_parameter_map.sipnet_parameter_names_written
        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        lai = predicted_fields(evaluation, observation_vector)[LAI]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        expected = leaf / float(runs.sipnet_model.base_params.dataarray("leaf_carbon_per_area"))
        observed = observation_vector[LAI].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(lai.sel(sample=0, site=1).values[observed], expected[observed], rtol=1e-12)

    def test_the_operators_read_the_runs_own_parameters_not_the_base_set(
        self, sipnet_map, climate, observation_vector, values
    ):
        runs = build(sipnet_map, climate, model=scaled_niwot_model(OwnLeafCarbonRunner))
        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        lai = predicted_fields(evaluation, observation_vector)[LAI]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        base = float(runs.sipnet_model.base_params.dataarray("leaf_carbon_per_area"))
        observed = observation_vector[LAI].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(
            lai.sel(sample=0, site=1).values[observed], (leaf / (OWN_LEAF_CARBON_FACTOR * base))[observed], rtol=1e-12
        )

    def test_a_read_parameter_the_base_set_leaves_unset_is_refused_before_anything_runs(
        self, runs, values
    ):
        @dataclass(frozen=True)
        class ReadsLeafWater:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names_read = ("leaf_water_pool_depth",)

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                return select_timestep_at(
                    restrict_to_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"]
                )

        wood = located(xr.DataArray(
            [[100.0, 110.0, 120.0]], dims=("site", "time"),
            coords={"site": [1], "time": LABELS}, attrs={"units": "g m-2", "constituent": "C"},
        ), site_table=SITE_TABLE)
        vector = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="wood", observed_values=wood, operator=ReadsLeafWater())
        ])
        with pytest.raises(ValueError, match="'leaf_water_pool_depth'.*leaves unset"):
            runs.evaluate(values, observation_vectors=[vector])

    def test_a_result_without_parameters_is_named_at_the_first_run(
        self, sipnet_map, climate, observation_vector, prior, theta
    ):
        runs = build(sipnet_map, climate, model=scaled_niwot_model(NoParametersRunner))
        with pytest.raises(RuntimeError, match="carries no SIPNETParameters as .parameters"):
            runs.evaluate(prior.to_labeled(theta[:1]), observation_vectors=[observation_vector])

    def test_an_observation_vector_over_fewer_sites_than_are_run(self, runs, observation_vector, values):
        one_site = observation_vector.select(sites=[1])
        evaluation = runs.evaluate(values, observation_vectors=[one_site])
        for source in one_site.observation_source_names:
            labels = one_site.coords[one_site.observation_dim_name(source)]
            predicted = evaluation.predictions[0][one_site.prediction_name(source)]
            assert predicted.shape == (3, len(labels)) and bool(jnp.isfinite(predicted).all())
        assert evaluation.run_succeeded.shape == (3, 2) and bool(evaluation.run_succeeded.all())

    def test_the_fields_that_were_run_are_located_from_the_site_dims_site_table(
        self, runs, observation_vector, values
    ):
        from sipnet_calibration.fields import validate_sipnet_parameter_fields

        fields = runs.evaluate(values, observation_vectors=[observation_vector]).sipnet_parameter_fields
        validate_sipnet_parameter_fields(fields)
        np.testing.assert_array_equal(fields["lon"].values, SITE_TABLE["lon"].values)

    def test_an_observation_source_with_a_site_it_never_observes(
        self, runs, observation_vector, prior, theta
    ):
        wood = observation_vector[BIOMASS].observed_values.copy()
        wood.loc[{"site": 27}] = np.nan
        sparse = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name=BIOMASS, observed_values=wood,
                              operator=SelectTimestep("wood_carbon"))
        ])
        assert sparse.sites == (1,)
        evaluation = runs.evaluate(prior.to_labeled(theta[:1]), observation_vectors=[sparse])
        predicted = evaluation.predictions[0][sparse.prediction_name(BIOMASS)]
        assert predicted.shape == (1, len(sparse.coords[sparse.observation_dim_name(BIOMASS)]))
        assert bool(jnp.isfinite(predicted).all())
        assert bool(evaluation.run_succeeded.sel(site=27).all())

    def test_output_variable_aliases_become_registry_names(self, runs, prior, theta):
        output = runs.evaluate(prior.to_labeled(theta[:1]), output_variable_names=("nee", "wood_carbon")).model_output
        assert list(output.data_vars) == ["net_ecosystem_exchange", "wood_carbon"]

    def test_a_site_slice_is_the_site_segment_of_each_observation_dim(self):
        """A run's predictions are placed in its site's segment, which needs
        each observation dim sorted by site."""
        sites = [1, 27, 40]
        times = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30, 50]])
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0, 130.0], [np.nan, 115.0, np.nan, np.nan], [90.0, 95.0, np.nan, 99.0]],
            dims=("site", "time"), coords={"site": sites, "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name=BIOMASS,
        ))
        lai = located(xr.DataArray(
            [[np.nan, 2.0, 2.5, np.nan], [1.0, np.nan, 1.5, 1.2], [np.nan] * 4],
            dims=("site", "time"), coords={"site": sites, "time": times},
            attrs={"units": "m2 m-2"}, name=LAI,
        ))
        soil = located(xr.DataArray(
            [np.nan, 5000.0, 4000.0], dims="site", coords={"site": sites},
            attrs={"units": "g m-2", "constituent": "C"}, name="soil",
        ))
        observation_vector = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name=BIOMASS, observed_values=wood,
                              operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name=LAI, observed_values=lai,
                              operator=DEFAULT_OBS_OPS[LAI]),
            ObservationSource(observation_source_name="soil", observed_values=soil,
                              operator=ReduceOverRun("soil_carbon", "mean")),
        ])
        for source in observation_vector.observation_source_names:
            dim = observation_vector.observation_dim_name(source)
            labels = observation_vector.coords[dim]
            for site in sites:
                columns = columns_at(observation_vector, source, site)
                one_site = observation_vector.select(sites=[site])
                if not len(columns):
                    assert dim not in one_site.coords
                    continue
                assert columns.tolist() == list(range(columns[0], columns[-1] + 1))
                assert one_site.coords[dim].equals(labels[columns])

    def test_an_observation_source_observed_beyond_a_shorter_sites_record(self, runs, sipnet_map, values):
        late = SHORT_STEPS + 10
        labels = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, late]])
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
            dims=("site", "time"), coords={"site": list(SITES), "time": labels},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name=BIOMASS,
        ), site_table=SITE_TABLE)
        observed = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name=BIOMASS, observed_values=wood,
                              operator=SelectTimestep("wood_carbon"))
        ])
        evaluation = runs.evaluate(values, observation_vectors=[observed])
        predicted = evaluation.predictions[0][observed.prediction_name(BIOMASS)]
        assert evaluation.valid.all() and bool(jnp.isfinite(predicted).all())
        fields = fields_of(sipnet_map, values)
        rate = fields["max_photosynthesis_rate"].sel(site=1).values
        soil = fields["soil_carbon"].sel(site=1).values
        expected = REFERENCE_WOOD.values[late] * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01
        np.testing.assert_allclose(predicted[:, columns_at(observed, BIOMASS, 1)[-1]], expected, rtol=1e-12)

    def test_a_run_at_a_site_no_observation_source_observes_returns_nothing(
        self, runs, sipnet_map, climate, observation_vector, values
    ):
        one_site = observation_vector.select(sites=[1])
        run = runs._plan(
            observation_vectors=(one_site,), read_variable_names=one_site.output_variable_names,
            model_output_variable_names=(), freq=None,
        ).run
        assert not run.model_output_variable_names
        overrides = sipnet_overrides(fields_of(sipnet_map, values), batch={"sample": 0}, site=27)
        output = run(climate=climate[27], site=27, site_observation_vectors=(None,), **overrides)
        assert output.model_output is None and output.predictions == (None,)


class TestFailures:
    def _evaluate_with(self, rate_map, climate, observation_vector, prior, theta, rate_values, n_samples=3, **keywords):
        model = keywords.pop("model", None)
        runs = build(rate_map, climate, model=model, **keywords)
        return runs.evaluate(
            prior.to_labeled(theta[:n_samples]), observation_vectors=[observation_vector],
            external_inputs=rates(rate_values, n_samples),
        )

    def test_a_run_failing_at_its_parameters_is_nan_in_its_entries_and_invalidates_its_row(
        self, rate_map, climate, observation_vector, prior, theta
    ):
        evaluation = self._evaluate_with(rate_map, climate, observation_vector, prior, theta, {(1, 0): 1.5 * BLOW_UP})
        assert evaluation.valid.tolist() == [True, False, True]
        for source in observation_vector.observation_source_names:
            predicted = np.asarray(evaluation.predictions[0][observation_vector.prediction_name(source)])
            assert np.isnan(predicted[1, columns_at(observation_vector, source, 1)]).all()
            assert np.isfinite(predicted[1, columns_at(observation_vector, source, 27)]).all()
            assert np.isfinite(predicted[[0, 2]]).all()
        assert not bool(evaluation.run_succeeded.sel(sample=1, site=1))
        assert bool(evaluation.run_succeeded.sel(sample=1, site=27))
        assert evaluation.failures["error"].tolist() == ["SIPNETRunError"]
        assert evaluation.failures["site"].tolist() == [1]
        assert "SIPNET blew up" in evaluation.failures["message"].iloc[0]

    def test_a_run_writing_nan_is_a_failure_at_its_parameters(
        self, rate_map, climate, observation_vector, prior, theta
    ):
        evaluation = self._evaluate_with(rate_map, climate, observation_vector, prior, theta, {(2, 1): 1.5 * NAN_BAND})
        assert evaluation.valid.tolist() == [True, True, False]
        assert evaluation.failures["error"].tolist() == ["ModelOutputNotFiniteError"]

    def test_the_machinery_failing_is_raised_with_what_was_collected(
        self, rate_map, climate, observation_vector, prior, theta
    ):
        with pytest.raises(RuntimeError, match="machinery") as raised:
            self._evaluate_with(
                rate_map, climate, observation_vector, prior, theta, {(2, 1): 1.5 * DIES_BAND, (0, 0): 1.5 * BLOW_UP}
            )
        partial = raised.value.evaluation
        assert partial.predictions == () and partial.model_output is None
        assert partial.failures["error"].tolist() == ["SIPNETRunError"]
        assert bool(partial.run_succeeded.sel(sample=1).all())
        assert not partial.valid.any()
        assert isinstance(partial.valid, jax.Array) and isinstance(partial.in_domain, jax.Array)

    def test_a_timeout_that_crosses_a_process_boundary_is_a_failure_at_its_parameters(
        self, rate_map, files, observation_vector, prior, theta
    ):
        """A TimeoutExpired built with keyword arguments does not unpickle; PyEns wraps it."""
        evaluation = self._evaluate_with(
            rate_map, files, observation_vector, prior, theta, {(0, 1): 1.5 * TIMEOUT_BAND}, n_samples=2,
            backend=LocalBackend(n_workers=1),
        )
        assert evaluation.valid.tolist() == [False, True]
        assert evaluation.failures["error"].tolist() == ["TimeoutExpired"]

    def test_an_infinite_prediction_leaves_the_row_valid_and_the_simulator_marks_it(
        self, runs, prior, theta
    ):
        """The runs' validity is the runs'; finiteness is the simulator's."""

        @dataclass(frozen=True)
        class Infinite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                out = select_timestep_at(model_output["wood_carbon"], observed_values["time"]) / 0.0
                out.attrs = {"units": "g m-2", "constituent": "C"}
                return out

        wood = located(xr.DataArray(
            [[100.0, 110.0, 120.0]], dims=("site", "time"), coords={"site": [1], "time": LABELS},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name=BIOMASS,
        ), site_table=SITE_TABLE)
        infinite = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name=BIOMASS, observed_values=wood, operator=Infinite())
        ])
        values = prior.to_labeled(theta[:1])
        evaluation = runs.evaluate(values, observation_vectors=[infinite])
        name = infinite.prediction_name(BIOMASS)
        assert bool(evaluation.run_succeeded.all()) and evaluation.valid.all()
        assert not bool(jnp.isfinite(evaluation.predictions[0][name]).any())
        output = SIPNETSimulator(runs, observation_vector=infinite)(values)
        assert output.valid[name].tolist() == [False]

    def test_a_machinery_failure_that_crosses_a_process_boundary_is_raised(
        self, rate_map, files, observation_vector, prior, theta
    ):
        with pytest.raises(RuntimeError, match="machinery"):
            self._evaluate_with(
                rate_map, files, observation_vector, prior, theta, {(0, 1): 1.5 * DIES_BAND}, n_samples=1,
                backend=LocalBackend(n_workers=1),
            )

    def test_an_unpicklable_exception_named_like_a_model_failure_is_the_machinery(
        self, rate_map, files, observation_vector, prior, theta
    ):
        """PyEns names it in full, which is not subprocess.TimeoutExpired."""
        with pytest.raises(RuntimeError, match="machinery") as raised:
            self._evaluate_with(
                rate_map, files, observation_vector, prior, theta, {(0, 1): 1.5 * DIES_BAND}, n_samples=1,
                backend=LocalBackend(n_workers=1), model=scaled_niwot_model(ForeignNiwotRunner),
            )
        assert raised.value.evaluation.failures.empty


class TestOutOfDomain:
    def test_values_outside_the_domain_are_refused_before_anything_runs(
        self, rate_map, climate, observation_vector, values
    ):
        with pytest.raises(SIPNETParametersOutOfDomainError, match="'sample': 1, 'site': 27.*nothing ran"):
            build(rate_map, climate).evaluate(
                values, observation_vectors=[observation_vector], external_inputs=rates({(1, 1): INVALID})
            )

    def test_fail_row_marks_the_row_out_of_the_domain_and_reports_the_fraction(
        self, rate_map, climate, observation_vector, values
    ):
        evaluation = build(rate_map, climate, out_of_domain="fail_row").evaluate(
            values, observation_vectors=[observation_vector], external_inputs=rates({(0, 1): 2 * INVALID})
        )
        assert evaluation.in_domain.tolist() == [False, True, True]
        assert evaluation.valid.tolist() == [False, True, True]
        assert evaluation.out_of_domain_fraction == pytest.approx(1 / 3)
        for predicted in evaluation.predictions[0].values():
            assert np.isnan(np.asarray(predicted)[0]).all()
        assert evaluation.failures["error"].tolist() == ["ValidationError"]  # pySIPNET refused the run too

    def test_out_of_domain_takes_two_values(self, sipnet_map, climate):
        with pytest.raises(ValueError, match="'raise' or 'fail_row'"):
            build(sipnet_map, climate, out_of_domain="ignore")

    def test_a_rule_input_outside_its_domain_is_refused_or_fails_its_row(self, soil_map, climate, prior, theta):
        negative = crossed_soil().assign(soil_input=lambda d: d["soil_input"] * xr.DataArray([1.0, -1.0], dims="site"))
        values = prior.to_labeled(theta[:1])
        with pytest.raises(SIPNETParametersOutOfDomainError, match="nothing ran"):
            build(soil_map, climate).evaluate(values, output_variable_names=("wood_carbon",), external_inputs=negative)
        evaluation = build(soil_map, climate, out_of_domain="fail_row").evaluate(
            values, output_variable_names=("wood_carbon",), external_inputs=negative
        )
        assert evaluation.valid.tolist() == [False, False] and evaluation.out_of_domain_fraction == 1.0


def crossed_soil(members=(0, 5), n_sites=len(SITES)) -> xr.Dataset:
    """Soil carbon crossed with the samples, on (initial_condition_member, site)."""
    values = SOIL_REFERENCE * (1.0 + np.arange(len(members) * n_sites).reshape(len(members), n_sites))
    return xr.Dataset(
        {"soil_input": (("initial_condition_member", "site"), values, {"units": "g m-2"})},
        coords={"initial_condition_member": list(members), "site": np.asarray(SITES, dtype=np.int32)},
    )


@pytest.fixture(scope="module")
def soil_map(sipnet_map):
    """The example's map with soil carbon read from an external input."""
    rules = [rule for rule in sipnet_map.rules if "soil_carbon" not in rule.sipnet_parameter_names_written]
    return SIPNETParameterMap(
        rules=[*rules, Copy(value_name="soil_input", sipnet_parameter_name="soil_carbon")], fixed=sipnet_map.fixed
    )


class TestExternalInputs:
    def test_a_crossed_input_runs_every_row_at_every_label(self, soil_map, climate, observation_vector, values):
        evaluation = build(soil_map, climate).evaluate(
            values, observation_vectors=[observation_vector], external_inputs=crossed_soil()
        )
        assert evaluation.run_index.names == ["sample", "initial_condition_member"]
        assert list(evaluation.run_index) == [(s, m) for s in range(3) for m in (0, 5)]
        for source in observation_vector.observation_source_names:
            labels = observation_vector.coords[observation_vector.observation_dim_name(source)]
            assert evaluation.predictions[0][observation_vector.prediction_name(source)].shape == (6, len(labels))
        assert evaluation.run_succeeded.dims == ("sample", "initial_condition_member", "site")
        assert evaluation.run_succeeded.size == 3 * 2 * 2 and bool(evaluation.run_succeeded.all())
        assert evaluation.valid.shape == (6,) and evaluation.valid.all()
        fields = evaluation.sipnet_parameter_fields
        assert fields["soil_carbon"].dims == ("initial_condition_member", "site")
        assert fields["max_photosynthesis_rate"].dims == ("sample", "site")

    def test_rows_follow_the_run_index_and_match_a_hand_computation(
        self, soil_map, climate, observation_vector, values
    ):
        evaluation = build(soil_map, climate).evaluate(
            values, observation_vectors=[observation_vector], external_inputs=crossed_soil()
        )
        fields = evaluation.sipnet_parameter_fields
        columns = columns_at(observation_vector, BIOMASS, 1)
        observed = observation_vector[BIOMASS].observed_values.sel(site=1).notnull().values
        predictions = np.asarray(evaluation.predictions[0][observation_vector.prediction_name(BIOMASS)]).reshape(3, 2, -1)
        for sample in range(3):
            for m, member in enumerate((0, 5)):
                expected = _expected_wood(fields, sample, 1, initial_condition_member=member)[observed]
                np.testing.assert_allclose(predictions[sample, m, columns], expected, rtol=1e-12)
        mean = predictions.mean(axis=1)
        by_hand = np.mean(
            [_expected_wood(fields, 0, 1, initial_condition_member=m)[observed] for m in (0, 5)], axis=0
        )
        np.testing.assert_allclose(mean[0, columns], by_hand, rtol=1e-12)

    def test_the_model_output_carries_the_crossed_dim(self, soil_map, climate, prior, theta):
        evaluation = build(soil_map, climate).evaluate(
            prior.to_labeled(theta[:2]), output_variable_names=("wood_carbon",), external_inputs=crossed_soil()
        )
        output = evaluation.model_output["wood_carbon"]
        assert output.dims == ("sample", "initial_condition_member", "site", "time")
        assert output["initial_condition_member"].values.tolist() == [0, 5]
        second = output.sel(sample=0, initial_condition_member=5, site=1).values
        first = output.sel(sample=0, initial_condition_member=0, site=1).values
        np.testing.assert_allclose(second, 3.0 * first)  # soil 3e4 against 1e4

    def test_a_failure_invalidates_its_combination_only(self, sipnet_map, climate, observation_vector, prior, theta):
        crossed_rates = xr.Dataset(
            {"rate_input": (("initial_condition_member",), [100.0, 1.5 * BLOW_UP], {"units": RATE_UNITS})},
            coords={"initial_condition_member": [0, 1]},
        )
        crossed = SIPNETParameterMap(rules=[CopyRate(), *sipnet_map.rules[1:]], fixed=sipnet_map.fixed)
        evaluation = build(crossed, climate).evaluate(
            prior.to_labeled(theta[:2]), observation_vectors=[observation_vector], external_inputs=crossed_rates
        )
        assert evaluation.valid.tolist() == [True, False, True, False]
        assert list(evaluation.failures.columns) == ["sample", "initial_condition_member", "site", "error", "message"]
        assert set(evaluation.failures["initial_condition_member"]) == {1}

    def test_unsorted_crossed_labels_keep_every_output_labeled_alike(self, sipnet_map, climate, prior, theta):
        crossed_rates = xr.Dataset(
            {"rate_input": (("initial_condition_member",), [1.5 * BLOW_UP, 100.0], {"units": RATE_UNITS})},
            coords={"initial_condition_member": [1, 0]},
        )
        crossed = SIPNETParameterMap(rules=[CopyRate(), *sipnet_map.rules[1:]], fixed=sipnet_map.fixed)
        evaluation = build(crossed, climate).evaluate(
            prior.to_labeled(theta[:2]), output_variable_names=("wood_carbon",), external_inputs=crossed_rates
        )
        assert list(evaluation.run_index) == [(0, 1), (0, 0), (1, 1), (1, 0)]
        assert evaluation.valid.tolist() == [False, True, False, True]
        assert not bool(evaluation.run_succeeded.sel(initial_condition_member=1).any())
        assert bool(evaluation.run_succeeded.sel(initial_condition_member=0).all())
        output = evaluation.model_output["wood_carbon"]
        assert output["initial_condition_member"].values.tolist() == [1, 0]
        assert bool(output.sel(initial_condition_member=1).isnull().all())

    def test_an_input_no_rule_reads_is_not_crossed(self, soil_map, climate, observation_vector, prior, theta):
        inputs = crossed_soil().assign(unread=(("driver_member",), [1.0, 2.0, 3.0], {"units": "1"})).assign_coords(
            driver_member=[0, 1, 2])
        evaluation = build(soil_map, climate).evaluate(
            prior.to_labeled(theta[:1]), observation_vectors=[observation_vector], external_inputs=inputs
        )
        assert evaluation.run_index.names == ["sample", "initial_condition_member"]
        labels = observation_vector.coords[observation_vector.observation_dim_name(BIOMASS)]
        assert evaluation.predictions[0][observation_vector.prediction_name(BIOMASS)].shape == (2, len(labels))

    def test_an_input_on_the_batch_dim_zips_with_the_samples(
        self, rate_map, climate, observation_vector, prior, theta
    ):
        runs = build(rate_map, climate)
        evaluation = runs.evaluate(
            prior.to_labeled(theta), observation_vectors=[observation_vector], external_inputs=rates({(1, 0): 50.0})
        )
        assert evaluation.run_index.name == "sample" and len(evaluation.run_index) == 3
        assert float(evaluation.sipnet_parameter_fields["max_photosynthesis_rate"].sel(sample=1, site=1)) == 50.0
        with pytest.raises(ValueError, match="give each the same number"):
            runs.evaluate(
                prior.to_labeled(theta[:2]), observation_vectors=[observation_vector],
                external_inputs=rates({(1, 0): 50.0}),
            )

    def test_an_input_named_like_a_value_given_is_refused(self, runs, values):
        inputs = crossed_soil().rename(soil_input="initial_soil_carbon")
        with pytest.raises(ValueError, match="named like values given"):
            runs.evaluate(values, output_variable_names=("wood_carbon",), external_inputs=inputs)

    def test_a_crossed_dim_may_not_take_a_model_output_name(self, soil_map, climate, values):
        inputs = crossed_soil().rename(initial_condition_member="wood_carbon")
        with pytest.raises(ValueError, match="'wood_carbon' is an output variable"):
            build(soil_map, climate).evaluate(values, output_variable_names=("wood_carbon",), external_inputs=inputs)


class TestPriorPredictive:
    def test_model_output_is_stacked_over_sample_and_site(self, runs, sipnet_map, values):
        evaluation = runs.evaluate(values, output_variable_names=("nee", "wood_carbon"))
        output = evaluation.model_output
        assert evaluation.predictions == ()
        assert set(output.data_vars) == {"net_ecosystem_exchange", "wood_carbon"}
        assert output["wood_carbon"].dims == ("sample", "site", "time")
        assert output["lon"].values.tolist() == [-105.0, -70.0]
        fields = fields_of(sipnet_map, values)
        for site, n_steps in ((1, REFERENCE_WOOD.sizes["time"]), (27, SHORT_STEPS)):
            wood = output["wood_carbon"].sel(sample=2, site=site).dropna("time")
            assert wood.sizes["time"] == n_steps
            rate = float(fields["max_photosynthesis_rate"].sel(sample=2, site=site))
            soil = float(fields["soil_carbon"].sel(sample=2, site=site))
            np.testing.assert_allclose(wood.values, REFERENCE_WOOD.values[:n_steps] * rate / 10.0 * soil / SOIL_REFERENCE)
        assert output["wood_carbon"].attrs["units"] == "g m-2"
        assert not {"time_bounds", "year", "day_of_year", "hour_of_day"} & set(output.coords)

    def test_freq_aggregates_on_the_worker_as_aggregate_time_does(self, runs, prior, theta):
        output = runs.evaluate(prior.to_labeled(theta[:1]), output_variable_names=("nee",), freq="1D").model_output
        expected = aggregate_time(REFERENCE.select(["net_ecosystem_exchange"])["net_ecosystem_exchange"], "1D")
        got = output["net_ecosystem_exchange"].sel(sample=0, site=1).dropna("time")
        np.testing.assert_allclose(got.values, expected.values)
        assert output["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"
        assert {"timestep_start", "timestep_length"} <= set(output.coords)
        np.testing.assert_array_equal(got["timestep_length"].values, expected["timestep_length"].values)

    def test_a_sample_failing_at_every_site_keeps_its_slot(self, rate_map, climate, values):
        evaluation = build(rate_map, climate).evaluate(
            values, output_variable_names=("nee",),
            external_inputs=rates({(1, 0): 1.5 * BLOW_UP, (1, 1): 1.5 * BLOW_UP}),
        )
        output = evaluation.model_output
        assert output["sample"].values.tolist() == [0, 1, 2] and output["sample"].dtype == np.int64
        assert bool(output["net_ecosystem_exchange"].sel(sample=1).isnull().all())
        assert evaluation.valid.tolist() == [True, False, True]
        assert len(evaluation.failures) == 2

    def test_every_run_failing_is_raised_with_what_was_collected(self, rate_map, climate, values):
        all_fail = {(s, p): 1.5 * BLOW_UP for s in range(3) for p in range(2)}
        with pytest.raises(RuntimeError, match="every run failed") as raised:
            build(rate_map, climate).evaluate(values, output_variable_names=("nee",), external_inputs=rates(all_fail))
        evaluation = raised.value.evaluation
        assert evaluation.model_output is None and not evaluation.valid.any()
        assert len(evaluation.failures) == 6

    def test_freq_keeps_the_runs_attributes_and_records_the_frequency(self, runs, prior, theta):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            output = runs.evaluate(
                prior.to_labeled(theta[:1]), output_variable_names=("nee", "wood_carbon"), freq="1D"
            ).model_output
        assert output.attrs["resampling_frequency"] == "1D"
        assert output.attrs["timestep_length_source"] == STEP_LENGTH_RESAMPLED
        assert output["net_ecosystem_exchange"].attrs["units"] == "g m-2"

    def test_a_site_where_every_run_failed_keeps_its_location(self, rate_map, climate, values):
        output = build(rate_map, climate).evaluate(
            values, output_variable_names=("nee",),
            external_inputs=rates({(s, 1): 1.5 * BLOW_UP for s in range(3)}),
        ).model_output
        assert not bool(output["net_ecosystem_exchange"].sel(site=27).notnull().any())
        assert output["lon"].values.tolist() == [-105.0, -70.0]


class TestRefusals:
    def test_a_site_without_drivers(self, sipnet_map):
        with pytest.raises(ValueError, match=r"no drivers for site\(s\) \[27\]"):
            build(sipnet_map, {1: REFERENCE.climate})

    def test_drivers_of_the_wrong_type(self, sipnet_map):
        with pytest.raises(TypeError, match="ClimateDrivers"):
            build(sipnet_map, {1: REFERENCE.climate, 27: "site_27.clim"})

    def test_an_observed_site_that_is_not_run(self, sipnet_map, climate, observation_vector, values):
        runs = build(sipnet_map, climate, site_dims=SITE_DIMS.select([1]))
        with pytest.raises(ValueError, match=r"observes site\(s\) \[27\]"):
            runs.evaluate(values, observation_vectors=[observation_vector])

    def test_memory_backed_drivers_under_a_process_backend(self, sipnet_map, climate):
        with pytest.raises(ValueError, match="held in memory"):
            build(sipnet_map, climate, backend=LocalBackend(n_workers=1))

    @pytest.mark.parametrize("name", ["time", "lon", "sample"])
    def test_an_external_input_with_a_reserved_name_or_the_batch_dims(self, runs, observation_vector, values, name):
        inputs = crossed_soil().rename({"soil_input": name})
        with pytest.raises(ValueError, match=f"external inputs \\['{name}'\\] are reserved names"):
            runs.evaluate(values, observation_vectors=[observation_vector], external_inputs=inputs)

    def test_nothing_asked_for(self, runs, values):
        with pytest.raises(ValueError, match="nothing is asked for"):
            runs.evaluate(values)

    def test_freq_with_an_observation_vector_alone(self, runs, observation_vector, values):
        with pytest.raises(ValueError, match="freq="):
            runs.evaluate(values, observation_vectors=[observation_vector], freq="1D")

    def test_a_flag_gated_output_variable(self, runs, values):
        with pytest.raises(ValueError, match="constant zero"):
            runs.evaluate(values, output_variable_names=("litter_carbon",))

    def test_a_freq_that_is_not_an_offset_alias(self, runs, values):
        with pytest.raises(ValueError, match="pandas offset alias"):
            runs.evaluate(values, output_variable_names=("nee",), freq="bogus")

    def test_freq_with_a_variable_no_method_keeps(self, runs, values):
        with pytest.raises(ValueError, match="no resampling method leaves unchanged"):
            runs.evaluate(values, output_variable_names=("year",), freq="1D")

    def test_a_map_reading_a_value_not_given(self, sipnet_map, climate, observation_vector, values):
        rules = [rule for rule in sipnet_map.rules if "soil_carbon" not in rule.sipnet_parameter_names_written]
        reads_missing = SIPNETParameterMap(
            rules=[*rules, Copy(value_name="missing", sipnet_parameter_name="soil_carbon")], fixed=sipnet_map.fixed,
        )
        with pytest.raises(KeyError, match=r"the map reads \['missing'\], which the values lack"):
            build(reads_missing, climate).evaluate(values, observation_vectors=[observation_vector])

    def test_not_a_sipnet_model_or_backend(self, sipnet_map, climate):
        with pytest.raises(TypeError, match="SIPNETModel"):
            SIPNETRuns(lambda **k: None, sipnet_parameter_map=sipnet_map, site_dims=SITE_DIMS, climate=climate,
                       backend=SequentialBackend())
        with pytest.raises(TypeError, match="Backend"):
            build(sipnet_map, climate, backend="local")


class TestRealSipnet:
    def test_two_samples_two_sites_under_a_process_backend(self, sipnet_map, observation_vector, files, prior, theta):
        from pysipnet.build import find_binary, missing_binary_message

        if find_binary() is None:
            pytest.skip(missing_binary_message())
        from sipnet_calibration.fields import to_model_output

        model = SIPNETModel(SIPNETRunner(flags=ModelFlags.standard(), timeout=120.0), base_params=niwot_reference_parameters())
        runs = SIPNETRuns(model, sipnet_parameter_map=sipnet_map, site_dims=SITE_DIMS, climate=files,
                          backend=LocalBackend(n_workers=2))
        evaluation = runs.evaluate(prior.to_labeled(theta[:2]), observation_vectors=[observation_vector])
        assert evaluation.valid.all()
        for predicted in evaluation.predictions[0].values():
            assert predicted.shape[0] == 2 and bool(jnp.isfinite(predicted).all())

        sample, site = 1, 27
        overrides = sipnet_overrides(evaluation.sipnet_parameter_fields, batch={"sample": sample}, site=site)
        run = model(climate=files[site], **overrides)
        direct = to_model_output(
            run.outputs.select(list(observation_vector.output_variable_names)), site=site, site_table=SITE_TABLE
        )
        one_site = observation_vector.select(sites=[site])
        run_parameters = xr.Dataset(
            {name: run.parameters.dataarray(name) for name in one_site.sipnet_parameter_names_read},
            coords={name: direct[name] for name in ("site", "lon", "lat")},
        )
        expected = one_site.predict(direct, sipnet_parameter_fields=run_parameters)
        predicted = predicted_fields(evaluation, observation_vector)
        for source in one_site.observation_source_names:
            direct_prediction = expected[source].sel(site=site)
            observed = one_site[source].observed_values.sel(site=site).notnull().values
            np.testing.assert_allclose(
                predicted[source].sel(sample=sample, site=site, time=direct_prediction["time"]).values[observed],
                direct_prediction.values[observed],
            )


class TestSccBackend:
    def test_carries_the_queue_and_the_environment(self, tmp_path):
        backend = scc_backend(
            walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-l mem_per_core=4G",)
        )
        assert "-P dietzelab" in backend.directives
        assert "-l buyin" in backend.directives
        assert "-v PYSIPNET_CACHE_DIR,PYSIPNET_BINARY,SIPNET_CALIBRATION_DATA" in backend.directives
        assert backend.directives[-1] == "-l mem_per_core=4G"
        assert backend.directives.index("-P dietzelab") < backend.directives.index(
            "-l mem_per_core=4G"
        )

    def test_refuses_overriding_the_queue(self, tmp_path):
        with pytest.raises(ValueError, match="override"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-P other",))
        with pytest.raises(TypeError, match="not one string"):
            scc_backend(
                walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives="-l mem_per_core=4G"
            )

    @pytest.mark.parametrize("directive", ["-P other", "  -P other", "-P\tother", "-Pother"])
    def test_refuses_naming_another_project(self, tmp_path, directive):
        with pytest.raises(ValueError, match="names a project"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=(directive,))

    @pytest.mark.parametrize(
        "directive", ["-l buyin=false", "-l mem_per_core=4G,buyin=false", "-lbuyin"]
    )
    def test_refuses_setting_buyin(self, tmp_path, directive):
        with pytest.raises(ValueError, match="sets the buyin resource"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=(directive,))

    @pytest.mark.parametrize(
        ("directive", "match"),
        [
            ("-l mem_per_core=4G -P other", "names a project"),
            ("-hard -l buyin=FALSE", "sets the buyin resource"),
            ("-pe omp 4 -l buyin=0", "sets the buyin resource"),
            ("-l BUYIN=0", "sets the buyin resource"),
            ("-soft -l mem_per_core=4G, Buyin", "sets the buyin resource"),
            ("-N 'unbalanced", "does not split into shell words"),
        ],
    )
    def test_reads_every_option_of_a_directive(self, tmp_path, directive, match):
        """PyEns writes each string as one #$ line, which can hold several options."""
        with pytest.raises(ValueError, match=match):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=(directive,))

    @pytest.mark.parametrize(
        "directive",
        ["-q other.q", "-v PYSIPNET_BINARY=/opt/sipnet", "-m ea -l mem_per_core=4G", "-p -10"],
    )
    def test_accepts_options_that_keep_the_queue(self, tmp_path, directive):
        backend = scc_backend(
            walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=(directive,)
        )
        assert backend.directives[-1] == directive

    def test_refuses_directives_that_are_not_strings(self, tmp_path):
        with pytest.raises(TypeError, match="directives must be a sequence of strings"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=None)
        with pytest.raises(TypeError, match="directives must be a sequence of strings"):
            scc_backend(
                walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives={"-l mem_per_core=4G"}
            )
        with pytest.raises(TypeError, match="every entry of directives"):
            scc_backend(
                walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-l h_rt=1:00:00", 3)
            )

    def test_passes_the_work_dir(self, tmp_path):
        assert (
            Path(scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2).work_dir) == tmp_path
        )


def test_an_evaluation_compares_and_hashes_by_identity():
    """A generated ``==`` would compare arrays and raise; identity cannot."""
    import dataclasses

    evaluation = SIPNETRunsEvaluation(
        sipnet_parameter_fields=xr.Dataset(),
        run_index=pd.Index([0, 1], name="sample"),
        model_output=None,
        predictions=(),
        observation_vectors=(),
        run_succeeded=xr.DataArray(np.ones((2, 1), dtype=bool), dims=("sample", "site")),
        failures=pd.DataFrame(),
        in_domain=jnp.ones(2, dtype=bool),
        valid=jnp.ones(2, dtype=bool),
    )
    copy = dataclasses.replace(evaluation)
    assert evaluation == evaluation and evaluation != copy
    assert len({evaluation, copy}) == 2


class TestTheBatchDimIsNamedOnce:
    """``batch_dim=`` names the values' dim, the SIPNET parameter fields',
    PyEns's axis, and every output's."""

    def test_every_output_carries_the_named_batch_dim(self, rate_map, climate, values):
        draws = rates({(1, 0): 1.5 * BLOW_UP}).rename(sample="draw")
        evaluation = build(rate_map, climate).evaluate(
            {name: array.rename(sample="draw") for name, array in values.items()},
            output_variable_names=("wood_carbon",), external_inputs=draws, batch_dim="draw",
        )
        assert evaluation.sipnet_parameter_fields["soil_carbon"].dims == ("draw", "site")
        assert evaluation.model_output["wood_carbon"].dims == ("draw", "site", "time")
        assert evaluation.run_succeeded.dims == ("draw", "site")
        assert list(evaluation.failures.columns) == ["draw", "site", "error", "message"]
        assert evaluation.failures["draw"].tolist() == [1]
        assert evaluation.valid.tolist() == [True, False, True]
        assert evaluation.run_index.name == "draw"

    def test_the_predictions_do_not_depend_on_its_name(self, runs, observation_vector, values):
        named = runs.evaluate(
            {name: array.rename(sample="draw") for name, array in values.items()},
            observation_vectors=[observation_vector], batch_dim="draw",
        )
        unnamed = runs.evaluate(values, observation_vectors=[observation_vector])
        assert list(named.predictions[0]) == list(unnamed.predictions[0])
        for name, predicted in named.predictions[0].items():
            np.testing.assert_allclose(predicted, unnamed.predictions[0][name])

    def test_a_reserved_name_is_refused(self, runs, observation_vector, values):
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            runs.evaluate(values, observation_vectors=[observation_vector], batch_dim="site")

    def test_an_output_variable_name_or_alias_is_refused(self, runs, values):
        with pytest.raises(ValueError, match="'wood_carbon' is an output variable"):
            runs.evaluate({name: array.rename(sample="wood_carbon") for name, array in values.items()},
                          output_variable_names=("wood_carbon",), batch_dim="wood_carbon")
        with pytest.raises(ValueError, match="'nee' is an alias of the output variable"):
            runs.evaluate({name: array.rename(sample="nee") for name, array in values.items()},
                          output_variable_names=("net_ecosystem_exchange",), batch_dim="nee")

    @pytest.mark.parametrize("name", ["timestep_length", "time_bounds", "bounds", "year"])
    def test_a_name_the_model_output_uses_is_refused(self, runs, values, name):
        with pytest.raises(ValueError, match=f"{name!r} cannot name a batch dim; it is a coordinate"):
            runs.evaluate(values, output_variable_names=("wood_carbon",), batch_dim=name)

    @pytest.mark.parametrize(
        "name", ["sipnet_model", "sipnet_parameter_map", "site_dims", "sites", "climate", "backend", "out_of_domain"]
    )
    def test_what_it_was_built_from_is_read_only(self, runs, name):
        with pytest.raises(AttributeError):
            setattr(runs, name, getattr(runs, name))

    def test_the_climate_it_hands_out_cannot_change_it(self, runs):
        with pytest.raises(TypeError):
            runs.climate[1] = None

    def test_the_arrays_are_jax(self, runs, observation_vector, values):
        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        for array in (*evaluation.predictions[0].values(), evaluation.valid, evaluation.in_domain):
            assert isinstance(array, jax.Array)

    def test_nothing_read_from_an_evaluation_changes_it(self, runs, values):
        evaluation = runs.evaluate(values, output_variable_names=("wood_carbon",))
        with pytest.raises(ValueError, match="read-only"):
            evaluation.run_succeeded.values[0, 0] = False
        with pytest.raises(ValueError, match="read-only"):
            evaluation.model_output["wood_carbon"].values[0, 0, 0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            evaluation.sipnet_parameter_fields["soil_carbon"].values[0, 0] = 0.0
        failures = evaluation.failures
        failures.loc[0] = [0, 1, "x", "y"]
        assert evaluation.failures.empty


class TestRunSucceededIsAField:
    def test_run_succeeded_validates_as_a_field(self, runs, observation_vector, values):
        from sipnet_calibration.fields import validate_field

        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        validate_field(evaluation.run_succeeded)
        assert evaluation.run_succeeded["site"].dtype == np.int32
        assert evaluation.run_succeeded["lon"].values.tolist() == [-105.0, -70.0]
        assert evaluation.run_succeeded["sample"].attrs["long_name"] == "Sample"

    def test_run_succeeded_is_mapped(self, runs, observation_vector, values):
        import matplotlib.pyplot as plt

        from sipnet_calibration.plotting import plot_map

        evaluation = runs.evaluate(values, observation_vectors=[observation_vector])
        figure, ax = plt.subplots()
        try:
            plot_map(evaluation.run_succeeded.isel(sample=0), ax=ax)
        finally:
            plt.close(figure)


class TestComposition:
    """What the runs read of the labeled values and the site dims."""

    def test_a_label_the_sites_carry_is_needed(self, runs, sipnet_map, climate, values):
        smaller = {name: array.sel(pft=["boreal.coniferous"]) if "pft" in array.dims else array
                   for name, array in values.items()}
        with pytest.raises(KeyError, match="no value at the pft label\\(s\\) \\['temperate.deciduous'\\]"):
            runs.evaluate(smaller, output_variable_names=("wood_carbon",))
        # Extra labels are read at no site.
        build(sipnet_map, climate, site_dims=SITE_DIMS.select([1])).evaluate(
            values, output_variable_names=("wood_carbon",)
        )

    def test_a_deterministic_value_reaches_the_map(self, prior_factors, sipnet_map, climate, theta):
        doubled_soil = DeterministicSpec(
            ArraySpec("soil_doubled", units="g m-2", support=POSITIVE, indexed_by=("site",)),
            function=lambda initial_soil_carbon: 2.0 * initial_soil_carbon,
        )
        prior = condition_on(joint(*prior_factors, doubled_soil).bind(coords=SITE_DIMS.coords), {})
        rules = [rule for rule in sipnet_map.rules if "soil_carbon" not in rule.sipnet_parameter_names_written]
        doubled = SIPNETParameterMap(rules=[*rules, Copy(value_name="soil_doubled", sipnet_parameter_name="soil_carbon")],
                                     fixed=sipnet_map.fixed)
        values = prior.to_labeled(theta)
        evaluation = build(doubled, climate).evaluate(values, output_variable_names=("wood_carbon",))
        np.testing.assert_allclose(
            evaluation.sipnet_parameter_fields["soil_carbon"].transpose("sample", "site"),
            2.0 * values["initial_soil_carbon"].transpose("sample", "site").values, rtol=1e-12,
        )

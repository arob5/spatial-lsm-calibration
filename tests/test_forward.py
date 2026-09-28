"""The forward model: theta and external inputs to predictions through PyEns,
on a stand-in and on SIPNET."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import LocalBackend, SequentialBackend
from pysipnet import niwot_reference_output
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
from sipnet_calibration.forward import ForwardEvaluation, ForwardModel
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
from sipnet_calibration.sipnet_parameter_map import (
    Copy,
    SIPNETParameterMap,
    SIPNETParametersOutOfDomainError,
    ValueRequirement,
)

SITES = (1, 27)
PFT = ("temperate.deciduous", "boreal.coniferous")
REFERENCE = niwot_reference_output()
REFERENCE_WOOD = REFERENCE.select(["wood_carbon"])["wood_carbon"]
SHORT_STEPS = 40  # site 27's drivers are cut to this many steps
LABELS = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30]])
SITE_TABLE = site_table_of(*SITES, lon=[-105.0, -70.0], lat=[40.0, 45.0])
RATE_UNITS = "nmol g-1 s-1"


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
    sipnet_parameter_names_read: ClassVar[tuple[str, ...]] = ()
    sipnet_parameter_names_written: ClassVar[tuple[str, ...]] = ("max_photosynthesis_rate",)

    @property
    def values_read(self):
        return {self.value_name: ValueRequirement(RATE_UNITS)}

    def __call__(self, values_at_sites, sipnet_parameter_values, site_table):
        return {"max_photosynthesis_rate": values_at_sites[self.value_name]}


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
    return example_calibration(SITE_TABLE, PFT)


@pytest.fixture(scope="module")
def parameter_vector(example):
    return example[0]


@pytest.fixture(scope="module")
def prior(example):
    return example[1]


@pytest.fixture(scope="module")
def sipnet_map(example):
    return example[2]


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
        name="landtrendr_aboveground_biomass",
    ), site_table=SITE_TABLE)
    lai = located(xr.DataArray(
        [[3.0, 2.0, np.nan], [np.nan, 1.0, 1.5]],
        dims=("site", "time"),
        coords={"site": list(SITES), "time": LABELS},
        attrs={"units": "m2 m-2"},
        name="modis_leaf_area_index",
    ), site_table=SITE_TABLE)
    return ObservationVector(
        observation_sources=[
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood,
                              operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai,
                              operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ]
    )


def build(parameter_vector, sipnet_map, climate, observation_vector=None, *, model=None, **keywords):
    keywords.setdefault("backend", SequentialBackend())
    return ForwardModel(
        model or scaled_niwot_model(), parameter_vector, sipnet_map, climate=climate,
        observation_vector=observation_vector, **keywords,
    )


@pytest.fixture
def forward(parameter_vector, sipnet_map, climate, observation_vector):
    return build(parameter_vector, sipnet_map, climate, observation_vector)


@pytest.fixture(scope="module")
def theta(prior):
    return np.asarray(prior.sample(jax.random.key(3), 3))


class TestEvaluate:
    def test_the_right_parameters_and_drivers_reach_the_right_run(
        self, forward, parameter_vector, sipnet_map, observation_vector, theta
    ):
        evaluation = forward.evaluate(theta)
        assert isinstance(evaluation, ForwardEvaluation)
        fields = sipnet_map.sipnet_parameter_fields(parameter_vector, theta)
        assert not np.allclose(fields["soil_carbon"].sel(site=1), fields["soil_carbon"].sel(site=27))
        predicted = evaluation.predicted_fields()["landtrendr_aboveground_biomass"]
        for sample in range(3):
            for site in SITES:
                observed = (
                    observation_vector["landtrendr_aboveground_biomass"].observed_values.sel(site=site).notnull().values
                )
                np.testing.assert_allclose(
                    predicted.sel(sample=sample, site=site).values[observed],
                    _expected_wood(fields, sample, site)[observed],
                    rtol=1e-12,
                )
        assert evaluation.valid.all() and evaluation.failures.empty
        assert evaluation.model_output is None
        assert list(evaluation.run_index) == [0, 1, 2] and evaluation.run_index.name == "sample"

    def test_an_operator_reads_the_sipnet_parameters_the_run_used(
        self, parameter_vector, sipnet_map, climate, theta
    ):
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
        forward = build(parameter_vector, sipnet_map, climate, observation_vector)
        predicted = observation_vector.fields(forward(theta))["soil"]
        expected = sipnet_map.sipnet_parameter_fields(parameter_vector, theta)["soil_carbon"]
        for sample in range(len(theta)):
            for site in SITES:
                row = predicted.sel(sample=sample, site=site).values
                np.testing.assert_allclose(row[np.isfinite(row)], float(expected.sel(sample=sample, site=site)))

    def test_a_single_theta_gives_one_row(self, forward, theta, observation_vector):
        assert forward(theta[0]).shape == (observation_vector.dimension,)
        assert forward(theta).shape == (3, observation_vector.dimension)
        assert forward(theta[:1]).shape == (1, observation_vector.dimension)
        assert forward.output_dimension == observation_vector.dimension
        assert forward.input_dimension == theta.shape[1]

    def test_the_lai_operator_reads_the_runs_own_leaf_carbon_per_area(self, forward, observation_vector, theta):
        """The map leaves it to the base parameter set, which the run used."""
        assert "leaf_carbon_per_area" not in forward.sipnet_parameter_map.sipnet_parameter_names_written
        lai = observation_vector.fields(forward(theta))["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        expected = leaf / float(forward.model.base_params.dataarray("leaf_carbon_per_area"))
        observed = observation_vector["modis_leaf_area_index"].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(lai.sel(sample=0, site=1).values[observed], expected[observed], rtol=1e-12)

    def test_the_operators_read_the_runs_own_parameters_not_the_base_set(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, sipnet_map, climate, observation_vector,
                        model=scaled_niwot_model(OwnLeafCarbonRunner))
        lai = observation_vector.fields(forward(theta))["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        base = float(forward.model.base_params.dataarray("leaf_carbon_per_area"))
        observed = observation_vector["modis_leaf_area_index"].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(
            lai.sel(sample=0, site=1).values[observed], (leaf / (OWN_LEAF_CARBON_FACTOR * base))[observed], rtol=1e-12
        )

    def test_a_read_parameter_the_base_set_leaves_unset_is_refused_when_built(
        self, parameter_vector, sipnet_map, climate
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
            build(parameter_vector, sipnet_map, climate, vector)

    def test_a_result_without_parameters_is_named_at_the_first_run(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, sipnet_map, climate, observation_vector,
                        model=scaled_niwot_model(NoParametersRunner))
        with pytest.raises(RuntimeError, match="carries no SIPNETParameters as .parameters"):
            forward.evaluate(theta[:1])

    def test_an_observation_vector_over_fewer_sites_than_are_run(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        one_site = observation_vector.select(sites=[1])
        evaluation = build(parameter_vector, sipnet_map, climate, one_site).evaluate(theta)
        assert evaluation.predictions.shape == (3, one_site.dimension)
        assert evaluation.run_succeeded.shape == (3, 2) and bool(evaluation.run_succeeded.all())
        assert np.isfinite(evaluation.predictions).all()

    def test_the_fields_that_were_run_are_located_from_the_vectors_site_table(self, forward, theta):
        from sipnet_calibration.fields import validate_sipnet_parameter_fields

        fields = forward.evaluate(theta).sipnet_parameter_fields
        validate_sipnet_parameter_fields(fields)
        np.testing.assert_array_equal(fields["lon"].values, SITE_TABLE["lon"].values)

    def test_an_observation_source_with_a_site_it_never_observes(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        wood = observation_vector["landtrendr_aboveground_biomass"].observed_values.copy()
        wood.loc[{"site": 27}] = np.nan
        sparse = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood,
                              operator=SelectTimestep("wood_carbon"))
        ])
        assert sparse.sites == (1,)
        evaluation = build(parameter_vector, sipnet_map, climate, sparse).evaluate(theta[:1])
        assert evaluation.predictions.shape == (1, sparse.dimension) and np.isfinite(evaluation.predictions).all()
        assert bool(evaluation.run_succeeded.sel(site=27).all())

    def test_output_variable_aliases_become_registry_names(self, parameter_vector, sipnet_map, climate):
        forward = build(parameter_vector, sipnet_map, climate, output_variable_names=("nee", "wood_carbon"))
        assert forward.output_variable_names == ("net_ecosystem_exchange", "wood_carbon")

    def test_a_site_slice_is_the_site_segment_of_the_whole_vector(self):
        """A run's segment is placed at positions(site=), which needs a site-major vector."""
        sites = [1, 27, 40]
        times = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30, 50]])
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0, 130.0], [np.nan, 115.0, np.nan, np.nan], [90.0, 95.0, np.nan, 99.0]],
            dims=("site", "time"), coords={"site": sites, "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
        ))
        lai = located(xr.DataArray(
            [[np.nan, 2.0, 2.5, np.nan], [1.0, np.nan, 1.5, 1.2], [np.nan] * 4],
            dims=("site", "time"), coords={"site": sites, "time": times},
            attrs={"units": "m2 m-2"}, name="modis_leaf_area_index",
        ))
        soil = located(xr.DataArray(
            [np.nan, 5000.0, 4000.0], dims="site", coords={"site": sites},
            attrs={"units": "g m-2", "constituent": "C"}, name="soil",
        ))
        observation_vector = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood,
                              operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai,
                              operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
            ObservationSource(observation_source_name="soil", observed_values=soil,
                              operator=ReduceOverRun("soil_carbon", "mean")),
        ])
        for site in sites:
            segment = observation_vector.index[observation_vector.positions(site=site)]
            assert segment.get_level_values("site").unique().tolist() == [site]
            assert observation_vector.select(sites=[site]).index.equals(segment)

    def test_an_observation_source_observed_beyond_a_shorter_sites_record(
        self, parameter_vector, sipnet_map, climate, theta
    ):
        late = SHORT_STEPS + 10
        labels = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, late]])
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
            dims=("site", "time"), coords={"site": list(SITES), "time": labels},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
        ), site_table=SITE_TABLE)
        observed = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood,
                              operator=SelectTimestep("wood_carbon"))
        ])
        evaluation = build(parameter_vector, sipnet_map, climate, observed).evaluate(theta)
        assert evaluation.valid.all() and np.isfinite(evaluation.predictions).all()
        fields = sipnet_map.sipnet_parameter_fields(parameter_vector, theta)
        rate = fields["max_photosynthesis_rate"].sel(site=1).values
        soil = fields["soil_carbon"].sel(site=1).values
        expected = REFERENCE_WOOD.values[late] * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01
        np.testing.assert_allclose(evaluation.predictions[:, observed.positions(site=1)[-1]], expected, rtol=1e-12)

    def test_a_run_at_a_site_no_observation_source_observes_returns_nothing(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, sipnet_map, climate, observation_vector.select(sites=[1]))
        assert not forward._run.returns_model_output
        overrides = sipnet_overrides(
            sipnet_map.sipnet_parameter_fields(parameter_vector, theta), batch={"sample": 0}, site=27
        )
        output = forward._run(climate=climate[27], site=27, site_observation_vector=None, **overrides)
        assert output.model_output is None and output.predictions is None


class TestFailures:
    def _forward_with(self, parameter_vector, rate_map, climate, observation_vector, values, n_samples=3, **keywords):
        return build(
            parameter_vector, rate_map, climate, observation_vector,
            external_inputs=rates(values, n_samples), **keywords,
        )

    def test_a_run_failing_at_its_parameters_is_a_nan_row(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = self._forward_with(parameter_vector, rate_map, climate, observation_vector, {(1, 0): 1.5 * BLOW_UP})
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, False, True]
        assert np.isnan(evaluation.predictions[1]).all()
        assert np.isfinite(np.asarray(evaluation.predictions)[[0, 2]]).all()
        assert not bool(evaluation.run_succeeded.sel(sample=1, site=1))
        assert bool(evaluation.run_succeeded.sel(sample=1, site=27))
        assert evaluation.failures["error"].tolist() == ["SIPNETRunError"]
        assert evaluation.failures["site"].tolist() == [1]
        assert "SIPNET blew up" in evaluation.failures["message"].iloc[0]

    def test_a_run_writing_nan_is_a_failure_at_its_parameters(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = self._forward_with(parameter_vector, rate_map, climate, observation_vector, {(2, 1): 1.5 * NAN_BAND})
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, True, False]
        assert evaluation.failures["error"].tolist() == ["ModelOutputNotFiniteError"]

    def test_the_machinery_failing_is_raised_with_what_was_collected(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, rate_map, climate, observation_vector, {(2, 1): 1.5 * DIES_BAND, (0, 0): 1.5 * BLOW_UP}
        )
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta)
        partial = raised.value.evaluation
        assert partial.predictions is None and partial.model_output is None
        assert partial.failures["error"].tolist() == ["SIPNETRunError"]
        assert bool(partial.run_succeeded.sel(sample=1).all())
        assert not partial.valid.any()
        assert isinstance(partial.theta, jax.Array) and isinstance(partial.valid, jax.Array)

    def test_a_timeout_that_crosses_a_process_boundary_is_a_nan_row(
        self, parameter_vector, rate_map, files, observation_vector, theta
    ):
        """A TimeoutExpired built with keyword arguments does not unpickle; PyEns wraps it."""
        forward = self._forward_with(
            parameter_vector, rate_map, files, observation_vector, {(0, 1): 1.5 * TIMEOUT_BAND}, n_samples=2,
            backend=LocalBackend(n_workers=1),
        )
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.valid.tolist() == [False, True]
        assert evaluation.failures["error"].tolist() == ["TimeoutExpired"]

    def test_an_infinite_prediction_is_invalid_though_the_run_succeeded(
        self, parameter_vector, sipnet_map, climate, theta
    ):
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
            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
        ), site_table=SITE_TABLE)
        infinite = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood,
                              operator=Infinite())
        ])
        evaluation = build(parameter_vector, sipnet_map, climate, infinite).evaluate(theta[:1])
        assert bool(evaluation.run_succeeded.all()) and not evaluation.valid.any()

    def test_a_machinery_failure_that_crosses_a_process_boundary_is_raised(
        self, parameter_vector, rate_map, files, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, rate_map, files, observation_vector, {(0, 1): 1.5 * DIES_BAND}, n_samples=1,
            backend=LocalBackend(n_workers=1),
        )
        with pytest.raises(RuntimeError, match="machinery"):
            forward.evaluate(theta[:1])

    def test_an_unpicklable_exception_named_like_a_model_failure_is_the_machinery(
        self, parameter_vector, rate_map, files, observation_vector, theta
    ):
        """PyEns names it in full, which is not subprocess.TimeoutExpired."""
        forward = self._forward_with(
            parameter_vector, rate_map, files, observation_vector, {(0, 1): 1.5 * DIES_BAND}, n_samples=1,
            backend=LocalBackend(n_workers=1), model=scaled_niwot_model(ForeignNiwotRunner),
        )
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta[:1])
        assert raised.value.evaluation.failures.empty


class TestOutOfDomain:
    def test_values_outside_the_domain_are_refused_before_anything_runs(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, rate_map, climate, observation_vector,
                        external_inputs=rates({(1, 1): INVALID}))
        with pytest.raises(SIPNETParametersOutOfDomainError, match="'sample': 1, 'site': 27.*nothing ran"):
            forward.evaluate(theta)

    def test_fail_row_marks_the_row_invalid_and_reports_the_fraction(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, rate_map, climate, observation_vector,
                        external_inputs=rates({(0, 1): 2 * INVALID}), out_of_domain="fail_row")
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [False, True, True]
        assert evaluation.out_of_domain_fraction == pytest.approx(1 / 3)
        assert np.isnan(evaluation.predictions[0]).all()
        assert evaluation.failures["error"].tolist() == ["ValidationError"]  # pySIPNET refused the run too

    def test_the_corner_check_refuses_a_map_outside_the_domain_when_built(self, climate, observation_vector):
        from sipnet_calibration.parameter_vector import REAL, Parameter, ParameterVector

        vector = ParameterVector(parameters=[Parameter(name="offset", support=REAL, units=RATE_UNITS)],
                                 site_table=SITE_TABLE)
        unbounded = SIPNETParameterMap(rules=[CopyRate(value_name="offset")])
        with pytest.raises(ValueError, match="outside pySIPNET's domains, at a corner"):
            build(vector, unbounded, climate, output_variable_names=("wood_carbon",))
        build(vector, unbounded, climate, output_variable_names=("wood_carbon",), out_of_domain="fail_row")

    def test_out_of_domain_takes_two_values(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="'raise' or 'fail_row'"):
            build(parameter_vector, sipnet_map, climate, observation_vector, out_of_domain="ignore")


def crossed_soil(members=(0, 5), n_sites=len(SITES)) -> xr.Dataset:
    """Soil carbon crossed with theta, on (initial_condition_member, site)."""
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
    def test_a_crossed_input_runs_every_row_at_every_label(
        self, parameter_vector, soil_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, soil_map, climate, observation_vector, external_inputs=crossed_soil())
        assert forward.crossed_dims == ("initial_condition_member",)
        assert forward.runs_per_sample == 4
        assert forward.describe()["crossed_dims"] == {"initial_condition_member": 2}
        evaluation = forward.evaluate(theta)
        assert evaluation.run_index.names == ["sample", "initial_condition_member"]
        assert list(evaluation.run_index) == [(s, m) for s in range(3) for m in (0, 5)]
        assert evaluation.predictions.shape == (6, observation_vector.dimension)
        assert evaluation.run_succeeded.dims == ("sample", "initial_condition_member", "site")
        assert evaluation.run_succeeded.size == 3 * 2 * 2 and bool(evaluation.run_succeeded.all())
        assert evaluation.valid.shape == (6,) and evaluation.valid.all()
        fields = evaluation.sipnet_parameter_fields
        assert fields["soil_carbon"].dims == ("initial_condition_member", "site")
        assert fields["max_photosynthesis_rate"].dims == ("sample", "site")

    def test_rows_follow_the_run_index_and_match_a_hand_computation(
        self, parameter_vector, soil_map, climate, observation_vector, theta
    ):
        evaluation = build(
            parameter_vector, soil_map, climate, observation_vector, external_inputs=crossed_soil()
        ).evaluate(theta)
        fields = evaluation.sipnet_parameter_fields
        wood = observation_vector["landtrendr_aboveground_biomass"]
        positions = observation_vector.positions(observation_source_name="landtrendr_aboveground_biomass", site=1)
        observed = wood.observed_values.sel(site=1).notnull().values
        predictions = np.asarray(evaluation.predictions).reshape(3, 2, -1)
        for sample in range(3):
            for m, member in enumerate((0, 5)):
                expected = _expected_wood(fields, sample, 1, initial_condition_member=member)[observed]
                np.testing.assert_allclose(predictions[sample, m, positions], expected, rtol=1e-12)
        mean = predictions.mean(axis=1)
        by_hand = np.mean(
            [_expected_wood(fields, 0, 1, initial_condition_member=m)[observed] for m in (0, 5)], axis=0
        )
        np.testing.assert_allclose(mean[0, positions], by_hand, rtol=1e-12)

    def test_predicted_fields_are_on_the_run_index_levels(
        self, parameter_vector, soil_map, climate, observation_vector, theta
    ):
        evaluation = build(
            parameter_vector, soil_map, climate, observation_vector, external_inputs=crossed_soil()
        ).evaluate(theta)
        predicted = evaluation.predicted_fields()["landtrendr_aboveground_biomass"]
        assert predicted.dims[:3] == ("sample", "initial_condition_member", "site")
        assert predicted["initial_condition_member"].values.tolist() == [0, 5]
        flat = np.asarray(evaluation.predictions)[1 * 2 + 1]
        np.testing.assert_allclose(
            predicted.sel(sample=1, initial_condition_member=5).values.ravel()[
                np.isfinite(predicted.sel(sample=1, initial_condition_member=5).values.ravel())
            ],
            flat[observation_vector.positions(observation_source_name="landtrendr_aboveground_biomass")],
        )

    def test_the_model_output_carries_the_crossed_dim(self, parameter_vector, soil_map, climate, theta):
        evaluation = build(
            parameter_vector, soil_map, climate, output_variable_names=("wood_carbon",), external_inputs=crossed_soil()
        ).evaluate(theta[:2])
        output = evaluation.model_output["wood_carbon"]
        assert output.dims == ("sample", "initial_condition_member", "site", "time")
        assert output["initial_condition_member"].values.tolist() == [0, 5]
        second = output.sel(sample=0, initial_condition_member=5, site=1).values
        first = output.sel(sample=0, initial_condition_member=0, site=1).values
        np.testing.assert_allclose(second, 3.0 * first)  # soil 3e4 against 1e4

    def test_a_failure_invalidates_its_combination_only(self, parameter_vector, sipnet_map, climate, observation_vector, theta):
        rules = [CopyRate(), *sipnet_map.rules[1:4]]
        crossed_rates = xr.Dataset(
            {"rate_input": (("initial_condition_member",), [100.0, 1.5 * BLOW_UP], {"units": RATE_UNITS})},
            coords={"initial_condition_member": [0, 1]},
        )
        crossed = SIPNETParameterMap(
            rules=[*rules, sipnet_map.rules[4]], fixed=sipnet_map.fixed
        )
        evaluation = build(parameter_vector, crossed, climate, observation_vector,
                           external_inputs=crossed_rates).evaluate(theta[:2])
        assert evaluation.valid.tolist() == [True, False, True, False]
        assert list(evaluation.failures.columns) == ["sample", "initial_condition_member", "site", "error", "message"]
        assert set(evaluation.failures["initial_condition_member"]) == {1}

    def test_unsorted_crossed_labels_keep_every_output_labeled_alike(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        crossed_rates = xr.Dataset(
            {"rate_input": (("initial_condition_member",), [1.5 * BLOW_UP, 100.0], {"units": RATE_UNITS})},
            coords={"initial_condition_member": [1, 0]},
        )
        crossed = SIPNETParameterMap(rules=[CopyRate(), *sipnet_map.rules[1:]], fixed=sipnet_map.fixed)
        evaluation = build(parameter_vector, crossed, climate, output_variable_names=("wood_carbon",),
                           external_inputs=crossed_rates).evaluate(theta[:2])
        assert list(evaluation.run_index) == [(0, 1), (0, 0), (1, 1), (1, 0)]
        assert evaluation.valid.tolist() == [False, True, False, True]
        assert not bool(evaluation.run_succeeded.sel(initial_condition_member=1).any())
        assert bool(evaluation.run_succeeded.sel(initial_condition_member=0).all())
        output = evaluation.model_output["wood_carbon"]
        assert output["initial_condition_member"].values.tolist() == [1, 0]
        assert bool(output.sel(initial_condition_member=1).isnull().all())

    def test_an_input_no_rule_reads_is_not_crossed(self, parameter_vector, soil_map, climate, observation_vector, theta):
        inputs = crossed_soil().assign(unread=(("driver_member",), [1.0, 2.0, 3.0], {"units": "1"})).assign_coords(
            driver_member=[0, 1, 2])
        forward = build(parameter_vector, soil_map, climate, observation_vector, external_inputs=inputs)
        assert forward.crossed_dims == ("initial_condition_member",) and forward.runs_per_sample == 4
        assert forward.evaluate(theta[:1]).predictions.shape == (2, observation_vector.dimension)

    def test_call_refuses_crossed_dims(self, parameter_vector, soil_map, climate, observation_vector, theta):
        forward = build(parameter_vector, soil_map, climate, observation_vector, external_inputs=crossed_soil())
        with pytest.raises(ValueError, match="call evaluate\\(theta\\) and reduce"):
            forward(theta)

    def test_an_input_on_the_batch_dim_zips_with_theta_and_fixes_its_rows(
        self, parameter_vector, rate_map, climate, observation_vector, theta
    ):
        forward = build(parameter_vector, rate_map, climate, observation_vector,
                        external_inputs=rates({(1, 0): 50.0}))
        assert forward.describe()["samples"] == 3 and forward.crossed_dims == ()
        assert forward(theta).shape == (3, observation_vector.dimension)
        with pytest.raises(ValueError, match="label them 0 to J - 1"):
            forward(theta[:2])

    def test_a_per_row_draw_runs_through_eki(self, parameter_vector, soil_map, climate, observation_vector, prior):
        """One soil draw frozen per member, labeled by theta's rows, under pyEKI's driver."""
        from pyeki.eki import EKIState, FixedSchedule, run
        from pyeki.linalg import PSDDiagonal

        n_members = 4
        draws = crossed_soil(members=tuple(range(n_members)))["soil_input"].rename(initial_condition_member="sample")
        forward = build(parameter_vector, soil_map, climate, observation_vector,
                        external_inputs=draws.to_dataset())
        state = EKIState(prior.sample(jax.random.key(1), n_members), 0.0, 0, jax.random.key(2))
        result = run(state, forward, observation_vector.y, PSDDiagonal(jax.numpy.ones(observation_vector.dimension)),
                     schedule=FixedSchedule((0.5, 0.5)))
        assert result.state.ensemble.shape == (n_members, parameter_vector.dimension)
        assert bool(jax.numpy.isfinite(result.state.ensemble).all())

    def test_an_input_named_like_a_parameter_is_refused(self, parameter_vector, sipnet_map, climate):
        inputs = crossed_soil().rename(soil_input="initial_soil_carbon")
        with pytest.raises(ValueError, match="named like parameters"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("wood_carbon",), external_inputs=inputs)

    def test_a_crossed_dim_may_not_take_a_model_output_name(self, parameter_vector, soil_map, climate):
        inputs = crossed_soil().rename(initial_condition_member="wood_carbon")
        with pytest.raises(ValueError, match="'wood_carbon' is an output variable"):
            build(parameter_vector, soil_map, climate, output_variable_names=("wood_carbon",), external_inputs=inputs)


class TestPriorPredictive:
    def test_model_output_is_stacked_over_sample_and_site(self, parameter_vector, sipnet_map, climate, theta):
        evaluation = build(parameter_vector, sipnet_map, climate, output_variable_names=("nee", "wood_carbon")).evaluate(theta)
        output = evaluation.model_output
        assert evaluation.predictions is None
        assert set(output.data_vars) == {"net_ecosystem_exchange", "wood_carbon"}
        assert output["wood_carbon"].dims == ("sample", "site", "time")
        assert output["lon"].values.tolist() == [-105.0, -70.0]
        fields = sipnet_map.sipnet_parameter_fields(parameter_vector, theta)
        for site, n_steps in ((1, REFERENCE_WOOD.sizes["time"]), (27, SHORT_STEPS)):
            wood = output["wood_carbon"].sel(sample=2, site=site).dropna("time")
            assert wood.sizes["time"] == n_steps
            rate = float(fields["max_photosynthesis_rate"].sel(sample=2, site=site))
            soil = float(fields["soil_carbon"].sel(sample=2, site=site))
            np.testing.assert_allclose(wood.values, REFERENCE_WOOD.values[:n_steps] * rate / 10.0 * soil / SOIL_REFERENCE)
        assert output["wood_carbon"].attrs["units"] == "g m-2"
        assert not {"time_bounds", "year", "day_of_year", "hour_of_day"} & set(output.coords)

    def test_freq_aggregates_on_the_worker_as_aggregate_time_does(self, parameter_vector, sipnet_map, climate, theta):
        output = build(parameter_vector, sipnet_map, climate, output_variable_names=("nee",), freq="1D").evaluate(
            theta[:1]).model_output
        expected = aggregate_time(REFERENCE.select(["net_ecosystem_exchange"])["net_ecosystem_exchange"], "1D")
        got = output["net_ecosystem_exchange"].sel(sample=0, site=1).dropna("time")
        np.testing.assert_allclose(got.values, expected.values)
        assert output["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"
        assert {"timestep_start", "timestep_length"} <= set(output.coords)
        np.testing.assert_array_equal(got["timestep_length"].values, expected["timestep_length"].values)

    def test_a_sample_failing_at_every_site_keeps_its_slot(self, parameter_vector, rate_map, climate, theta):
        evaluation = build(
            parameter_vector, rate_map, climate, output_variable_names=("nee",),
            external_inputs=rates({(1, 0): 1.5 * BLOW_UP, (1, 1): 1.5 * BLOW_UP}),
        ).evaluate(theta)
        output = evaluation.model_output
        assert output["sample"].values.tolist() == [0, 1, 2] and output["sample"].dtype == np.int64
        assert bool(output["net_ecosystem_exchange"].sel(sample=1).isnull().all())
        assert evaluation.valid.tolist() == [True, False, True]
        assert len(evaluation.failures) == 2

    def test_call_needs_an_observation_vector(self, parameter_vector, sipnet_map, climate, theta):
        with pytest.raises(ValueError, match="needs an observation vector"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("nee",))(theta)

    def test_every_run_failing_is_raised_with_what_was_collected(self, parameter_vector, rate_map, climate, theta):
        all_fail = {(s, p): 1.5 * BLOW_UP for s in range(3) for p in range(2)}
        forward = build(parameter_vector, rate_map, climate, output_variable_names=("nee",),
                        external_inputs=rates(all_fail))
        with pytest.raises(RuntimeError, match="every run failed") as raised:
            forward.evaluate(theta)
        evaluation = raised.value.evaluation
        assert evaluation.model_output is None and not evaluation.valid.any()
        assert len(evaluation.failures) == 6

    def test_freq_keeps_the_runs_attributes_and_records_the_frequency(self, parameter_vector, sipnet_map, climate, theta):
        forward = build(parameter_vector, sipnet_map, climate, output_variable_names=("nee", "wood_carbon"), freq="1D")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            output = forward.evaluate(theta[:1]).model_output
        assert output.attrs["resampling_frequency"] == "1D"
        assert output.attrs["timestep_length_source"] == STEP_LENGTH_RESAMPLED
        assert output["net_ecosystem_exchange"].attrs["units"] == "g m-2"

    def test_a_site_where_every_run_failed_keeps_its_location(self, parameter_vector, rate_map, climate, theta):
        output = build(
            parameter_vector, rate_map, climate, output_variable_names=("nee",),
            external_inputs=rates({(s, 1): 1.5 * BLOW_UP for s in range(3)}),
        ).evaluate(theta).model_output
        assert not bool(output["net_ecosystem_exchange"].sel(site=27).notnull().any())
        assert output["lon"].values.tolist() == [-105.0, -70.0]


class TestRefusals:
    def test_a_site_without_drivers(self, parameter_vector, sipnet_map, observation_vector):
        with pytest.raises(ValueError, match=r"no drivers for site\(s\) \[27\]"):
            build(parameter_vector, sipnet_map, {1: REFERENCE.climate}, observation_vector)

    def test_drivers_of_the_wrong_type(self, parameter_vector, sipnet_map, observation_vector):
        with pytest.raises(TypeError, match="ClimateDrivers"):
            build(parameter_vector, sipnet_map, {1: REFERENCE.climate, 27: "site_27.clim"}, observation_vector)

    def test_an_observed_site_that_is_not_run(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match=r"observes site\(s\) \[27\]"):
            build(parameter_vector.select(sites=[1]), sipnet_map, climate, observation_vector)

    def test_memory_backed_drivers_under_a_process_backend(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="held in memory"):
            build(parameter_vector, sipnet_map, climate, observation_vector, backend=LocalBackend(n_workers=1))

    def test_output_names_must_cover_the_operators(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="operators read"):
            build(parameter_vector, sipnet_map, climate, observation_vector, output_variable_names=("nee",))

    def test_neither_output_variable_names_nor_an_observation_vector(self, parameter_vector, sipnet_map, climate):
        with pytest.raises(ValueError, match="output_variable_names"):
            build(parameter_vector, sipnet_map, climate)

    def test_freq_with_an_observation_vector(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="freq="):
            build(parameter_vector, sipnet_map, climate, observation_vector, freq="1D")

    def test_a_flag_gated_output_variable(self, parameter_vector, sipnet_map, climate):
        with pytest.raises(ValueError, match="constant zero"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("litter_carbon",))

    def test_a_freq_that_is_not_an_offset_alias_is_refused_when_built(self, parameter_vector, sipnet_map, climate):
        with pytest.raises(ValueError, match="pandas offset alias"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("nee",), freq="bogus")

    def test_freq_with_a_variable_no_method_keeps(self, parameter_vector, sipnet_map, climate):
        with pytest.raises(ValueError, match="no resampling method leaves unchanged"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("year",), freq="1D")

    def test_a_map_that_does_not_fit_the_vector(self, parameter_vector, climate, observation_vector):
        reads_nothing_held = SIPNETParameterMap(rules=[Copy(value_name="missing", sipnet_parameter_name="soil_carbon")])
        with pytest.raises(KeyError, match="neither a parameter, a derived parameter nor an external input"):
            build(parameter_vector, reads_nothing_held, climate, observation_vector)

    def test_theta_of_the_wrong_shape_or_not_finite(self, forward, theta):
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:, :-1])
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:0])
        bad = theta.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            forward.evaluate(bad)

    def test_not_a_sipnet_model_or_backend(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(TypeError, match="SIPNETModel"):
            ForwardModel(lambda **k: None, parameter_vector, sipnet_map, climate=climate,
                         backend=SequentialBackend(), observation_vector=observation_vector)
        with pytest.raises(TypeError, match="Backend"):
            build(parameter_vector, sipnet_map, climate, observation_vector, backend="local")


class TestRealSipnet:
    def test_two_samples_two_sites_under_a_process_backend(
        self, parameter_vector, sipnet_map, observation_vector, files, theta
    ):
        from pysipnet.build import find_binary, missing_binary_message

        if find_binary() is None:
            pytest.skip(missing_binary_message())
        from conftest import niwot_parameters
        from sipnet_calibration.fields import to_model_output

        model = SIPNETModel(SIPNETRunner(flags=ModelFlags.standard(), timeout=120.0), base_params=niwot_parameters())
        forward = ForwardModel(model, parameter_vector, sipnet_map, climate=files,
                               backend=LocalBackend(n_workers=2), observation_vector=observation_vector)
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.predictions.shape == (2, observation_vector.dimension)
        assert np.isfinite(evaluation.predictions).all() and evaluation.valid.all()

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
        expected = one_site.flat(one_site.predict(direct, sipnet_parameter_fields=run_parameters))
        np.testing.assert_allclose(evaluation.predictions[sample, observation_vector.positions(site=site)], expected)


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

    evaluation = ForwardEvaluation(
        theta=np.zeros((2, 3)),
        sipnet_parameter_fields=xr.Dataset(),
        run_index=pd.Index([0, 1], name="sample"),
        model_output=None,
        predictions=np.zeros((2, 4)),
        run_succeeded=xr.DataArray(np.ones((2, 1), dtype=bool), dims=("sample", "site")),
        failures=pd.DataFrame(),
        valid=np.ones(2, dtype=bool),
    )
    copy = dataclasses.replace(evaluation)
    assert evaluation == evaluation and evaluation != copy
    assert len({evaluation, copy}) == 2


def test_predicted_fields_need_predictions(parameter_vector, sipnet_map, climate, theta):
    evaluation = build(parameter_vector, sipnet_map, climate, output_variable_names=("nee",)).evaluate(theta[:1])
    with pytest.raises(ValueError, match="no predictions"):
        evaluation.predicted_fields()


class TestTheBatchDimIsNamedOnce:
    """``batch_dim=`` names the SIPNET parameter fields' dim, PyEns's axis, and every output."""

    def test_every_output_carries_the_named_batch_dim(self, parameter_vector, rate_map, climate, theta):
        draws = rates({(1, 0): 1.5 * BLOW_UP}).rename(sample="draw")
        evaluation = build(
            parameter_vector, rate_map, climate, output_variable_names=("wood_carbon",),
            external_inputs=draws, batch_dim="draw",
        ).evaluate(theta)
        assert evaluation.sipnet_parameter_fields["soil_carbon"].dims == ("draw", "site")
        assert evaluation.model_output["wood_carbon"].dims == ("draw", "site", "time")
        assert evaluation.run_succeeded.dims == ("draw", "site")
        assert list(evaluation.failures.columns) == ["draw", "site", "error", "message"]
        assert evaluation.failures["draw"].tolist() == [1]
        assert evaluation.valid.tolist() == [True, False, True]
        assert evaluation.run_index.name == "draw"

    def test_the_predictions_do_not_depend_on_its_name(
        self, parameter_vector, sipnet_map, climate, observation_vector, theta
    ):
        named = build(parameter_vector, sipnet_map, climate, observation_vector, batch_dim="draw")
        np.testing.assert_allclose(
            named(theta), build(parameter_vector, sipnet_map, climate, observation_vector)(theta)
        )

    def test_a_reserved_name_is_refused(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            build(parameter_vector, sipnet_map, climate, observation_vector, batch_dim="site")

    @pytest.mark.parametrize("name", ["initial_soil_carbon", "pft", "allocation.leaf"])
    def test_a_name_of_the_vectors_labeled_form_is_refused(
        self, parameter_vector, sipnet_map, climate, observation_vector, name
    ):
        with pytest.raises(ValueError, match="dim or variable of the vector's labeled form"):
            build(parameter_vector, sipnet_map, climate, observation_vector, batch_dim=name)

    def test_an_output_variable_name_or_alias_is_refused(self, parameter_vector, sipnet_map, climate):
        with pytest.raises(ValueError, match="'wood_carbon' is an output variable"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("wood_carbon",), batch_dim="wood_carbon")
        with pytest.raises(ValueError, match="'nee' is an alias of the output variable"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("net_ecosystem_exchange",),
                  batch_dim="nee")

    @pytest.mark.parametrize("name", ["timestep_length", "time_bounds", "bounds", "year"])
    def test_a_name_the_model_output_uses_is_refused(self, parameter_vector, sipnet_map, climate, name):
        with pytest.raises(ValueError, match=f"{name!r} cannot name a batch dim; it is a coordinate"):
            build(parameter_vector, sipnet_map, climate, output_variable_names=("wood_carbon",), batch_dim=name)

    def test_an_observation_source_name_is_refused(self, parameter_vector, sipnet_map, climate, observation_vector):
        with pytest.raises(ValueError, match="is an observation source name"):
            build(parameter_vector, sipnet_map, climate, observation_vector,
                  batch_dim="landtrendr_aboveground_biomass")

    def test_the_batch_dim_is_read_only(self, forward):
        assert forward.batch_dim == "sample"
        with pytest.raises(AttributeError):
            forward.batch_dim = "draw"

    @pytest.mark.parametrize(
        "name",
        [
            "model", "parameter_vector", "sipnet_parameter_map", "external_inputs", "out_of_domain",
            "observation_vector", "backend", "freq", "climate", "sites", "output_variable_names",
            "crossed_dims",
        ],
    )
    def test_what_it_was_built_from_is_read_only(self, forward, name):
        with pytest.raises(AttributeError):
            setattr(forward, name, getattr(forward, name))

    def test_the_climate_and_inputs_it_hands_out_cannot_change_it(
        self, parameter_vector, soil_map, climate, observation_vector
    ):
        forward = build(parameter_vector, soil_map, climate, observation_vector, external_inputs=crossed_soil())
        with pytest.raises(TypeError):
            forward.climate[1] = None
        inputs = forward.external_inputs
        inputs["soil_input"].values[0, 0] = -1.0
        assert float(forward.external_inputs["soil_input"][0, 0]) > 0

    def test_flat_is_jax(self, forward, theta):
        evaluation = forward.evaluate(theta.tolist())
        for array in (evaluation.theta, evaluation.predictions, evaluation.valid):
            assert isinstance(array, jax.Array)
        assert isinstance(forward(jax.numpy.asarray(theta[0])), jax.Array)

    def test_nothing_read_from_an_evaluation_changes_it(self, parameter_vector, sipnet_map, climate, theta):
        evaluation = build(parameter_vector, sipnet_map, climate, output_variable_names=("wood_carbon",)).evaluate(theta)
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
    def test_run_succeeded_validates_as_a_field(self, forward, theta):
        from sipnet_calibration.fields import validate_field

        evaluation = forward.evaluate(theta)
        validate_field(evaluation.run_succeeded)
        assert evaluation.run_succeeded["site"].dtype == np.int32
        assert evaluation.run_succeeded["lon"].values.tolist() == [-105.0, -70.0]
        assert evaluation.run_succeeded["sample"].attrs["long_name"] == "Sample"

    def test_run_succeeded_is_mapped(self, forward, theta):
        import matplotlib.pyplot as plt

        from sipnet_calibration.plotting import plot_map

        evaluation = forward.evaluate(theta)
        figure, ax = plt.subplots()
        try:
            plot_map(evaluation.run_succeeded.isel(sample=0), ax=ax)
        finally:
            plt.close(figure)

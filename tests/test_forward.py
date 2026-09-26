"""The forward model: theta to predictions through PyEns, on a stand-in and on SIPNET."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import LocalBackend, SequentialBackend
from pysipnet import niwot_reference_output
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import ModelFlags, parameter_dataarray
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
from sipnet_calibration.compute import scc_backend
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
from sipnet_calibration.parameter_vector import example_parameter_vector

SITES = (1, 27)
REFERENCE = niwot_reference_output()
REFERENCE_WOOD = REFERENCE.select(["wood_carbon"])["wood_carbon"]
SHORT_STEPS = 40  # site 27's drivers are cut to this many steps
LABELS = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30]])
SITE_TABLE = site_table_of(*SITES, lon=[-105.0, -70.0], lat=[40.0, 45.0])


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
    """The stand-in, whose run used another leaf_carbon_per_area than it was given.

    So the run's own ``SIPNETResult.parameters`` differ from the base set with
    the overrides applied, and an operator reading the base set would be
    caught.
    """

    def run(self, parameters, climate, *, events=None, **keywords):
        result = super().run(parameters, climate, events=events, **keywords)
        leaf = float(parameters.dataarray("leaf_carbon_per_area"))
        return SimpleNamespace(
            outputs=result.outputs,
            parameters=with_parameter_value(
                parameters, "leaf_carbon_per_area", OWN_LEAF_CARBON_FACTOR * leaf
            ),
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


def _expected_wood(sipnet_parameter_fields, sample, site, n_steps=None):
    """What the stand-in writes for wood carbon at this sample and site, in Mg ha-1."""
    rate = float(sipnet_parameter_fields["max_photosynthesis_rate"].sel(sample=sample, site=site))
    soil = float(sipnet_parameter_fields["soil_carbon"].sel(sample=sample, site=site))
    wood = REFERENCE_WOOD if n_steps is None else REFERENCE_WOOD.isel(time=slice(0, n_steps))
    return select_timestep_at(wood, LABELS).values * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01


@pytest.fixture(scope="module")
def parameter_vector():
    return example_parameter_vector(sites=SITES, pft=("temperate.deciduous", "boreal.coniferous"))


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
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ]
    )


@pytest.fixture
def forward(parameter_vector, climate, observation_vector):
    return ForwardModel(
        scaled_niwot_model(),
        parameter_vector,
        climate=climate,
        backend=SequentialBackend(),
        observation_vector=observation_vector,
        site_table=SITE_TABLE,
    )


@pytest.fixture(scope="module")
def theta(parameter_vector):
    return np.asarray(parameter_vector.sample(jax.random.key(3), 3))


def _hooked(parameter_vector, rates):
    """A to_sipnet_parameter_fields hook setting max_photosynthesis_rate at (sample, site position)."""

    def hook(theta):
        sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
        values = sipnet_parameter_fields["max_photosynthesis_rate"].values.copy()
        for (sample, site_position), rate in rates.items():
            if sample < values.shape[0]:  # the init-time probe has one sample
                values[sample, site_position] = rate
        rate = sipnet_parameter_fields["max_photosynthesis_rate"].copy(data=values)
        return sipnet_parameter_fields.assign(max_photosynthesis_rate=rate)

    return hook


class TestEvaluate:
    def test_the_right_parameters_and_drivers_reach_the_right_run(
        self, forward, parameter_vector, observation_vector, theta
    ):
        evaluation = forward.evaluate(theta)
        assert isinstance(evaluation, ForwardEvaluation)
        sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
        assert not np.allclose(sipnet_parameter_fields["soil_carbon"].sel(site=1), sipnet_parameter_fields["soil_carbon"].sel(site=27))
        fields = observation_vector.fields(evaluation.predictions)["landtrendr_aboveground_biomass"]
        for sample in range(3):
            for site in SITES:
                observed = (
                    observation_vector["landtrendr_aboveground_biomass"]
                    .observed_values.sel(site=site)
                    .notnull()
                    .values
                )
                expected = _expected_wood(sipnet_parameter_fields, sample, site)
                np.testing.assert_allclose(
                    fields.sel(sample=sample, site=site).values[observed],
                    expected[observed],
                    rtol=1e-12,
                )
        assert evaluation.valid.all() and evaluation.failures.empty
        assert evaluation.model_output is None

    def test_an_operator_reads_the_sipnet_parameters_the_run_used(
        self, parameter_vector, climate, theta
    ):
        """The worker's SIPNET parameter fields are the run's own, per sample and site."""
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
        observation_vector = ObservationVector(
            observation_sources=[
                ObservationSource(
                    observation_source_name="soil",
                    observed_values=observed,
                    operator=ReadsSoilCarbon(),
                )
            ]
        )
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=observation_vector,
            site_table=SITE_TABLE,
        )
        predicted = observation_vector.fields(forward(theta))["soil"]
        expected = parameter_vector.sipnet_parameter_fields(theta)["soil_carbon"]
        for sample in range(len(theta)):
            for site in SITES:
                row = predicted.sel(sample=sample, site=site).values
                np.testing.assert_allclose(
                    row[np.isfinite(row)], float(expected.sel(sample=sample, site=site))
                )

    def test_a_single_theta_gives_one_row(self, forward, theta, observation_vector):
        assert forward(theta[0]).shape == (observation_vector.dimension,)
        assert forward(theta).shape == (3, observation_vector.dimension)
        assert forward(theta[:1]).shape == (1, observation_vector.dimension)
        assert forward.output_dimension == observation_vector.dimension
        assert forward.input_dimension == theta.shape[1]

    def test_the_lai_operator_reads_the_runs_own_leaf_carbon_per_area(
        self, forward, observation_vector, theta
    ):
        """The vector leaves it to the base parameter set, which the run used."""
        assert forward.sipnet_parameter_names_written == forward.parameter_vector.sipnet_parameter_names_written
        assert "leaf_carbon_per_area" not in forward.sipnet_parameter_names_written
        predictions = forward(theta)
        lai = observation_vector.fields(predictions)["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        base = float(forward.model.base_params.dataarray("leaf_carbon_per_area"))
        expected = leaf / base
        observed = observation_vector["modis_leaf_area_index"].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(
            lai.sel(sample=0, site=1).values[observed], expected[observed], rtol=1e-12
        )

    def test_the_operators_read_the_runs_own_parameters_not_the_base_set(
        self, parameter_vector, climate, observation_vector, theta
    ):
        """A stand-in whose run used another leaf_carbon_per_area than base plus overrides."""
        forward = ForwardModel(
            scaled_niwot_model(OwnLeafCarbonRunner), parameter_vector, climate=climate,
            backend=SequentialBackend(), observation_vector=observation_vector,
            site_table=SITE_TABLE,
        )
        lai = observation_vector.fields(forward(theta))["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        base = float(forward.model.base_params.dataarray("leaf_carbon_per_area"))
        expected = leaf / (OWN_LEAF_CARBON_FACTOR * base)
        observed = observation_vector["modis_leaf_area_index"].observed_values.sel(site=1).notnull().values
        np.testing.assert_allclose(
            lai.sel(sample=0, site=1).values[observed], expected[observed], rtol=1e-12
        )

    def test_a_read_parameter_the_base_set_leaves_unset_is_refused_when_built(
        self, parameter_vector, climate
    ):
        """It was refused only after every run of the batch had run."""

        @dataclass(frozen=True)
        class ReadsLeafWater:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names_read = ("leaf_water_pool_depth",)

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                return select_timestep_at(
                    restrict_to_observed_sites(model_output["wood_carbon"], observed_values),
                    observed_values["time"],
                )

        wood = located(xr.DataArray(
            [[100.0, 110.0, 120.0]], dims=("site", "time"),
            coords={"site": [1], "time": LABELS}, attrs={"units": "g m-2", "constituent": "C"},
        ), site_table=SITE_TABLE)
        vector = ObservationVector(observation_sources=[
            ObservationSource(observation_source_name="wood", observed_values=wood, operator=ReadsLeafWater())
        ])
        with pytest.raises(ValueError, match="'leaf_water_pool_depth'.*leaves unset"):
            ForwardModel(
                scaled_niwot_model(), parameter_vector, climate=climate,
                backend=SequentialBackend(), observation_vector=vector, site_table=SITE_TABLE,
            )

    def test_a_result_without_parameters_is_named_at_the_first_run(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = ForwardModel(
            scaled_niwot_model(NoParametersRunner), parameter_vector, climate=climate,
            backend=SequentialBackend(), observation_vector=observation_vector,
            site_table=SITE_TABLE,
        )
        with pytest.raises(RuntimeError, match="carries no SIPNETParameters as .parameters"):
            forward.evaluate(theta[:1])

    def test_an_observation_vector_over_fewer_sites_than_are_run(
        self, parameter_vector, climate, observation_vector, theta
    ):
        one_site = observation_vector.select(sites=[1])
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=one_site,
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.predictions.shape == (3, one_site.dimension)
        assert evaluation.run_succeeded.shape == (3, 2) and bool(evaluation.run_succeeded.all())
        assert np.isfinite(evaluation.predictions).all()

    def test_a_bare_site_id_vectors_sipnet_parameter_fields_are_located_from_the_site_table(
        self, parameter_vector, climate, observation_vector, theta
    ):
        """The strict validator refuses them unlocated; the model has the locations."""
        from sipnet_calibration.fields import validate_sipnet_parameter_fields

        assert "lon" not in parameter_vector.sipnet_parameter_fields(theta).coords
        forward = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate,
            backend=SequentialBackend(), observation_vector=observation_vector,
            site_table=SITE_TABLE,
        )
        fields = forward.evaluate(theta).sipnet_parameter_fields
        validate_sipnet_parameter_fields(fields)
        np.testing.assert_array_equal(fields["lon"].values, SITE_TABLE["lon"].values)

    def test_an_observation_source_with_a_site_it_never_observes(
        self, parameter_vector, climate, observation_vector, theta
    ):
        """The ObservationSource drops the site, so its run is made and reduced to nothing."""
        wood = observation_vector["landtrendr_aboveground_biomass"].observed_values.copy()
        wood.loc[{"site": 27}] = np.nan
        sparse = ObservationVector(
            observation_sources=[ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=SelectTimestep("wood_carbon"))]
        )
        assert 27 not in sparse.sites and sparse.sites == (1,)
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=sparse,
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta[:1])
        assert (
            evaluation.predictions.shape == (1, sparse.dimension)
            and np.isfinite(evaluation.predictions).all()
        )
        assert bool(evaluation.run_succeeded.sel(site=27).all())

    def test_output_variable_aliases_become_registry_names(self, parameter_vector, climate):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee", "wood_carbon"),
            site_table=SITE_TABLE,
        )
        assert forward.output_variable_names == ("net_ecosystem_exchange", "wood_carbon")

    def test_a_site_slice_is_the_site_segment_of_the_whole_vector(self):
        """A run's segment is placed at positions(site=), which needs a site-major vector."""
        sites = [1, 27, 40]
        times = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30, 50]])
        wood = located(xr.DataArray(
            [
                [100.0, np.nan, 120.0, 130.0],
                [np.nan, 115.0, np.nan, np.nan],
                [90.0, 95.0, np.nan, 99.0],
            ],
            dims=("site", "time"),
            coords={"site": sites, "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"},
            name="landtrendr_aboveground_biomass",
        ))
        lai = located(xr.DataArray(
            [[np.nan, 2.0, 2.5, np.nan], [1.0, np.nan, 1.5, 1.2], [np.nan] * 4],
            dims=("site", "time"),
            coords={"site": sites, "time": times},
            attrs={"units": "m2 m-2"},
            name="modis_leaf_area_index",
        ))
        soil = located(xr.DataArray(
            [np.nan, 5000.0, 4000.0],
            dims="site",
            coords={"site": sites},
            attrs={"units": "g m-2", "constituent": "C"},
            name="soil",
        ))
        observation_vector = ObservationVector(
            observation_sources=[
                ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=SelectTimestep("wood_carbon")),
                ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
                ObservationSource(observation_source_name="soil", observed_values=soil, operator=ReduceOverRun("soil_carbon", "mean")),
            ]
        )
        assert observation_vector.sites == tuple(sites)
        for site in sites:
            segment = observation_vector.index[observation_vector.positions(site=site)]
            assert segment.get_level_values("site").unique().tolist() == [site]
            assert observation_vector.select(sites=[site]).index.equals(segment)

    def test_an_observation_source_observed_beyond_a_shorter_sites_record(
        self, parameter_vector, climate, theta
    ):
        """Site 27's drivers end before a label only site 1 is observed at."""
        late = SHORT_STEPS + 10
        labels = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, late]])
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
            dims=("site", "time"),
            coords={"site": list(SITES), "time": labels},
            attrs={"units": "Mg ha-1", "constituent": "C"},
            name="landtrendr_aboveground_biomass",
        ), site_table=SITE_TABLE)
        observed = ObservationVector(
            observation_sources=[ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=SelectTimestep("wood_carbon"))]
        )
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=observed,
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.all() and np.isfinite(evaluation.predictions).all()
        sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
        rate = sipnet_parameter_fields["max_photosynthesis_rate"].sel(site=1).values
        soil = sipnet_parameter_fields["soil_carbon"].sel(site=1).values
        expected = REFERENCE_WOOD.values[late] * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01
        np.testing.assert_allclose(
            evaluation.predictions[:, observed.positions(site=1)[-1]], expected, rtol=1e-12
        )

    def test_a_run_at_a_site_no_observation_source_observes_returns_nothing(
        self, parameter_vector, climate, observation_vector, theta
    ):
        from sipnet_calibration.parameter_vector import sipnet_overrides

        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=observation_vector.select(sites=[1]),
            site_table=SITE_TABLE,
        )
        assert not forward._run.returns_model_output
        overrides = sipnet_overrides(parameter_vector.sipnet_parameter_fields(theta), batch={"sample": 0}, site=27)
        output = forward._run(
            climate=climate[27], site=27, site_observation_vector=None, **overrides
        )
        assert output.model_output is None and output.predictions is None


class TestFailures:
    def _forward_with(self, parameter_vector, climate, observation_vector, rates, backend=None):
        return ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=backend or SequentialBackend(),
            observation_vector=observation_vector,
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(parameter_vector, rates),
        )

    def test_a_run_failing_at_its_parameters_is_a_nan_row(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, climate, observation_vector, {(1, 0): 1.5 * BLOW_UP}
        )
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
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, climate, observation_vector, {(2, 1): 1.5 * NAN_BAND}
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, True, False]
        assert evaluation.failures["error"].tolist() == ["ModelOutputNotFiniteError"]

    def test_the_machinery_failing_is_raised_with_what_was_collected(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, climate, observation_vector, {(2, 1): 1.5 * DIES_BAND, (0, 0): 1.5 * BLOW_UP}
        )
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta)
        partial = raised.value.evaluation
        assert partial.predictions is None and partial.model_output is None
        assert partial.failures["error"].tolist() == ["SIPNETRunError"]
        assert bool(partial.run_succeeded.sel(sample=1).all())
        assert not partial.valid.any()

    def test_a_timeout_that_crosses_a_process_boundary_is_a_nan_row(
        self, parameter_vector, files, observation_vector, theta
    ):
        """A TimeoutExpired built with keyword arguments does not unpickle; PyEns wraps it."""
        forward = self._forward_with(
            parameter_vector,
            files,
            observation_vector,
            {(0, 1): 1.5 * TIMEOUT_BAND},
            backend=LocalBackend(n_workers=1),
        )
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.valid.tolist() == [False, True]
        assert evaluation.failures["error"].tolist() == ["TimeoutExpired"]

    def test_an_infinite_prediction_is_invalid_though_the_run_succeeded(
        self, parameter_vector, climate, theta
    ):
        @dataclass(frozen=True)
        class Infinite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                out = select_timestep_at(model_output["wood_carbon"], observed_values["time"])
                out = out / 0.0
                out.attrs = {"units": "g m-2", "constituent": "C"}
                return out

        wood = located(xr.DataArray(
            [[100.0, 110.0, 120.0]],
            dims=("site", "time"),
            coords={"site": [1], "time": LABELS},
            attrs={"units": "Mg ha-1", "constituent": "C"},
            name="landtrendr_aboveground_biomass",
        ), site_table=SITE_TABLE)
        infinite = ObservationVector(
            observation_sources=[ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=Infinite())]
        )
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=infinite,
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta[:1])
        assert bool(evaluation.run_succeeded.all()) and not evaluation.valid.any()

    def test_a_refusal_by_pydantic_is_a_nan_row(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector, climate, observation_vector, {(0, 1): 2 * INVALID}
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [False, True, True]
        assert evaluation.failures["error"].tolist() == ["ValidationError"]

    def test_a_machinery_failure_that_crosses_a_process_boundary_is_raised(
        self, parameter_vector, files, observation_vector, theta
    ):
        forward = self._forward_with(
            parameter_vector,
            files,
            observation_vector,
            {(0, 1): 1.5 * DIES_BAND},
            backend=LocalBackend(n_workers=1),
        )
        with pytest.raises(RuntimeError, match="machinery"):
            forward.evaluate(theta[:1])

    def test_an_unpicklable_exception_named_like_a_model_failure_is_the_machinery(
        self, parameter_vector, files, observation_vector, theta
    ):
        """PyEns names it in full, which is not subprocess.TimeoutExpired."""
        forward = ForwardModel(
            scaled_niwot_model(ForeignNiwotRunner),
            parameter_vector,
            climate=files,
            backend=LocalBackend(n_workers=1),
            observation_vector=observation_vector,
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(parameter_vector, {(0, 1): 1.5 * DIES_BAND}),
        )
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta[:1])
        assert raised.value.evaluation.failures.empty


class TestPriorPredictive:
    def test_model_output_is_stacked_over_sample_and_site(self, parameter_vector, climate, theta):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee", "wood_carbon"),
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta)
        output = evaluation.model_output
        assert evaluation.predictions is None
        assert set(output.data_vars) == {"net_ecosystem_exchange", "wood_carbon"}
        assert output["wood_carbon"].dims == ("sample", "site", "time")
        assert output["lon"].values.tolist() == [-105.0, -70.0]
        sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
        for site, n_steps in ((1, REFERENCE_WOOD.sizes["time"]), (27, SHORT_STEPS)):
            wood = output["wood_carbon"].sel(sample=2, site=site).dropna("time")
            assert wood.sizes["time"] == n_steps  # the shorter drivers make a shorter run
            rate = float(sipnet_parameter_fields["max_photosynthesis_rate"].sel(sample=2, site=site))
            soil = float(sipnet_parameter_fields["soil_carbon"].sel(sample=2, site=site))
            np.testing.assert_allclose(
                wood.values, REFERENCE_WOOD.values[:n_steps] * rate / 10.0 * soil / SOIL_REFERENCE
            )
        assert output["wood_carbon"].attrs["units"] == "g m-2"
        assert not {"time_bounds", "year", "day_of_year", "hour_of_day"} & set(output.coords)

    def test_freq_aggregates_on_the_worker_as_aggregate_time_does(
        self, parameter_vector, climate, theta
    ):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            freq="1D",
            site_table=SITE_TABLE,
        )
        output = forward.evaluate(theta[:1]).model_output
        expected = aggregate_time(
            REFERENCE.select(["net_ecosystem_exchange"])["net_ecosystem_exchange"], "1D"
        )
        got = output["net_ecosystem_exchange"].sel(sample=0, site=1).dropna("time")
        np.testing.assert_allclose(got.values, expected.values)
        assert output["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"

    def test_a_sample_failing_at_every_site_keeps_its_slot(self, parameter_vector, climate, theta):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(
                parameter_vector, {(1, 0): 1.5 * BLOW_UP, (1, 1): 1.5 * BLOW_UP}
            ),
        )
        evaluation = forward.evaluate(theta)
        output = evaluation.model_output
        assert output["sample"].values.tolist() == [0, 1, 2]
        # The reindex that restores the slot keeps the labels int64.
        assert output["sample"].dtype == np.int64
        assert bool(output["net_ecosystem_exchange"].sel(sample=1).isnull().all())
        assert (
            int(output["net_ecosystem_exchange"].sel(sample=0, site=1).notnull().sum())
            == REFERENCE_WOOD.sizes["time"]
        )
        assert evaluation.valid.tolist() == [True, False, True]
        assert len(evaluation.failures) == 2

    def test_call_needs_an_observation_vector(self, parameter_vector, climate, theta):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            site_table=SITE_TABLE,
        )
        with pytest.raises(ValueError, match="needs an observation vector"):
            forward(theta)

    def test_freq_keeps_the_interval_coordinates(self, parameter_vector, climate, theta):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            freq="1D",
            site_table=SITE_TABLE,
        )
        output = forward.evaluate(theta[:1]).model_output
        assert {"timestep_start", "timestep_length"} <= set(output.coords)
        expected = aggregate_time(
            REFERENCE.select(["net_ecosystem_exchange"])["net_ecosystem_exchange"], "1D"
        )
        got = output.sel(sample=0, site=1).dropna("time")
        np.testing.assert_array_equal(
            got["timestep_length"].values, expected["timestep_length"].values
        )
        np.testing.assert_array_equal(
            got["timestep_start"].values, expected["timestep_start"].values
        )

    def test_every_run_failing_is_raised_with_what_was_collected(
        self, parameter_vector, climate, theta
    ):
        rates = {(sample, position): 1.5 * BLOW_UP for sample in range(3) for position in range(2)}
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(parameter_vector, rates),
        )
        with pytest.raises(RuntimeError, match="every run failed") as raised:
            forward.evaluate(theta)
        evaluation = raised.value.evaluation
        assert evaluation.model_output is None and evaluation.predictions is None
        assert not bool(evaluation.run_succeeded.any()) and not evaluation.valid.any()
        assert len(evaluation.failures) == 6

    def test_freq_keeps_the_runs_attributes_and_records_the_frequency(
        self, parameter_vector, climate, theta
    ):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee", "wood_carbon"),
            freq="1D",
            site_table=SITE_TABLE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            output = forward.evaluate(theta[:1]).model_output
        assert output.attrs["Conventions"] == "CF-1.11"
        assert output.attrs["resampling_frequency"] == "1D"
        assert output.attrs["timestep_length_source"] == STEP_LENGTH_RESAMPLED
        assert not {"units", "constituent", "sign_convention", "kind"} & set(output.attrs)
        assert output["net_ecosystem_exchange"].attrs["units"] == "g m-2"

    def test_a_site_where_every_run_failed_keeps_its_location(
        self, parameter_vector, climate, theta
    ):
        rates = {(sample, 1): 1.5 * BLOW_UP for sample in range(3)}
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(parameter_vector, rates),
        )
        output = forward.evaluate(theta).model_output
        assert not bool(output["net_ecosystem_exchange"].sel(site=27).notnull().any())
        assert output["lon"].values.tolist() == [-105.0, -70.0]
        assert output["lat"].values.tolist() == [40.0, 45.0]

    def test_the_vectors_own_locations_are_the_default_site_table(self, climate, theta):
        located = SITE_TABLE.assign(lon=[-1.0, -2.0], lat=[10.0, 20.0])
        parameter_vector = example_parameter_vector(
            site_table=located, pft=("temperate.deciduous", "boreal.coniferous")
        )
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("nee",),
        )
        output = forward.evaluate(theta[:1]).model_output
        assert output["lon"].values.tolist() == [-1.0, -2.0]
        assert output["lat"].values.tolist() == [10.0, 20.0]


class TestRefusals:
    def _build(self, parameter_vector, climate, observation_vector=None, **kwargs):
        kwargs.setdefault("backend", SequentialBackend())
        kwargs.setdefault("site_table", SITE_TABLE)
        return ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            observation_vector=observation_vector,
            **kwargs,
        )

    def test_a_site_without_drivers(self, parameter_vector, observation_vector):
        with pytest.raises(ValueError, match=r"no drivers for site\(s\) \[27\]"):
            self._build(parameter_vector, {1: REFERENCE.climate}, observation_vector)

    def test_drivers_of_the_wrong_type(self, parameter_vector, observation_vector):
        with pytest.raises(TypeError, match="ClimateDrivers"):
            self._build(
                parameter_vector, {1: REFERENCE.climate, 27: "site_27.clim"}, observation_vector
            )

    def test_an_observed_site_that_is_not_run(self, parameter_vector, climate, observation_vector):
        with pytest.raises(ValueError, match=r"observes site\(s\) \[27\]"):
            self._build(parameter_vector.select(sites=[1]), climate, observation_vector)

    def test_memory_backed_drivers_under_a_process_backend(
        self, parameter_vector, climate, observation_vector
    ):
        with pytest.raises(ValueError, match="held in memory"):
            self._build(
                parameter_vector, climate, observation_vector, backend=LocalBackend(n_workers=1)
            )

    def test_output_names_must_cover_the_operators(
        self, parameter_vector, climate, observation_vector
    ):
        with pytest.raises(ValueError, match="operators read"):
            self._build(
                parameter_vector, climate, observation_vector, output_variable_names=("nee",)
            )

    def test_neither_output_variable_names_nor_an_observation_vector(
        self, parameter_vector, climate
    ):
        with pytest.raises(ValueError, match="output_variable_names"):
            self._build(parameter_vector, climate)

    def test_freq_with_an_observation_vector(self, parameter_vector, climate, observation_vector):
        with pytest.raises(ValueError, match="freq="):
            self._build(parameter_vector, climate, observation_vector, freq="1D")

    def test_a_flag_gated_output_variable(self, parameter_vector, climate):
        with pytest.raises(ValueError, match="constant zero"):
            self._build(parameter_vector, climate, output_variable_names=("litter_carbon",))

    def test_a_site_table_without_locations_or_a_site(
        self, parameter_vector, climate, observation_vector
    ):
        with pytest.raises(ValueError, match="'lon' and 'lat'"):
            self._build(
                parameter_vector,
                climate,
                observation_vector,
                site_table=pd.DataFrame({"site_id": [1, 27]}),
            )
        with pytest.raises(KeyError, match=r"site\(s\) \[27\] are not in the site table"):
            self._build(
                parameter_vector, climate, observation_vector, site_table=SITE_TABLE.iloc[:1]
            )

    def test_a_site_table_without_lat(self, parameter_vector, climate, observation_vector):
        with pytest.raises(ValueError, match="'lon' and 'lat'"):
            self._build(
                parameter_vector,
                climate,
                observation_vector,
                site_table=SITE_TABLE.drop(columns="lat"),
            )

    def test_a_freq_that_is_not_an_offset_alias_is_refused_when_built(
        self, parameter_vector, climate
    ):
        with pytest.raises(ValueError, match="pandas offset alias"):
            self._build(parameter_vector, climate, output_variable_names=("nee",), freq="bogus")

    def test_freq_with_a_variable_no_method_keeps(self, parameter_vector, climate):
        with pytest.raises(ValueError, match="no resampling method leaves unchanged"):
            self._build(parameter_vector, climate, output_variable_names=("year",), freq="1D")

    def test_sipnet_parameter_fields_hook_that_changes_the_free_inputs(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._build(parameter_vector, climate, observation_vector)
        forward._to_sipnet_parameter_fields = lambda t: parameter_vector.sipnet_parameter_fields(t).drop_vars(
            "soil_carbon"
        )
        with pytest.raises(ValueError, match="free inputs are fixed"):
            forward.evaluate(theta)

    def test_sipnet_parameter_fields_hook_that_adds_a_parameter(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._build(parameter_vector, climate, observation_vector)
        forward._to_sipnet_parameter_fields = lambda t: parameter_vector.sipnet_parameter_fields(t).assign(
            leaf_carbon_per_area=lambda d: parameter_dataarray(
                "leaf_carbon_per_area", d["soil_carbon"] * 0 + 50.0
            )
        )
        with pytest.raises(ValueError, match="free inputs are fixed"):
            forward.evaluate(theta)

    def test_sipnet_parameter_fields_hook_that_names_a_parameter_by_its_alias(
        self, parameter_vector, climate, observation_vector
    ):
        """SIPNETModel refuses an alias, so every run would fail on the workers."""
        with pytest.raises(
            ValueError, match="'aMax', an alias of pySIPNET's 'max_photosynthesis_rate'"
        ):
            self._build(
                parameter_vector,
                climate,
                observation_vector,
                to_sipnet_parameter_fields=lambda t: parameter_vector.sipnet_parameter_fields(t).rename(
                    max_photosynthesis_rate="aMax"
                ),
            )

    def test_sipnet_parameter_fields_hook_whose_variables_are_not_on_the_batch_dim(
        self, parameter_vector, climate, observation_vector
    ):
        """With only a sample coordinate, PyEns would build one run per site, not J x S."""

        def site_only(theta):
            sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
            n_samples = sipnet_parameter_fields.sizes["sample"]
            return sipnet_parameter_fields.isel(sample=0, drop=True).assign_coords(sample=np.arange(n_samples))

        with pytest.raises(ValueError, match="on both sample and site"):
            self._build(parameter_vector, climate, observation_vector, to_sipnet_parameter_fields=site_only)

    def test_sipnet_parameter_fields_hook_that_sets_an_unknown_parameter(
        self, parameter_vector, climate, observation_vector
    ):
        with pytest.raises(KeyError, match="not a pySIPNET parameter"):
            self._build(
                parameter_vector,
                climate,
                observation_vector,
                to_sipnet_parameter_fields=lambda t: parameter_vector.sipnet_parameter_fields(t).assign(
                    not_a_parameter=lambda d: d["soil_carbon"]
                ),
            )

    def test_sipnet_parameter_fields_hook_may_reorder_the_fields(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = self._build(parameter_vector, climate, observation_vector)
        names = list(forward.sipnet_parameter_names_written)
        forward._to_sipnet_parameter_fields = lambda t: parameter_vector.sipnet_parameter_fields(t)[names[::-1]]
        assert forward(theta[:1]).shape == (1, observation_vector.dimension)

    def test_theta_of_the_wrong_shape_or_not_finite(self, forward, theta):
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:, :-1])
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:0])
        bad = theta.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            forward.evaluate(bad)

    def test_not_a_sipnet_model_or_backend(self, parameter_vector, climate, observation_vector):
        with pytest.raises(TypeError, match="SIPNETModel"):
            ForwardModel(
                lambda **k: None,
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
            )
        with pytest.raises(TypeError, match="Backend"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend="local",
                observation_vector=observation_vector,
            )

    def test_a_site_table_listing_a_site_twice(self, parameter_vector, climate, observation_vector):
        doubled = pd.concat([SITE_TABLE, SITE_TABLE.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match=r"lists site\(s\) \[1\] more than once"):
            self._build(parameter_vector, climate, observation_vector, site_table=doubled)

    @pytest.mark.parametrize(
        ("make_hook", "match"),
        [
            (
                lambda v: lambda t: v.sipnet_parameter_fields(t).isel(sample=slice(None, None, -1)),
                "sample labels must be the integers 0 to 2",
            ),
            (
                lambda v: (
                    lambda t: v.sipnet_parameter_fields(t).isel(sample=slice(0, max(1, np.shape(t)[0] - 1)))
                ),
                "sample labels must be the integers 0 to 2",
            ),
            (
                lambda v: lambda t: v.sipnet_parameter_fields(t).expand_dims(extra=[0, 1]),
                r"exactly \(sample, site\)",
            ),
            (lambda v: lambda t: v.sipnet_parameter_fields(t).isel(site=[1, 0]), "sites, in order"),
        ],
        ids=["samples reversed", "a sample dropped", "an extra dimension", "sites reordered"],
    )
    def test_sipnet_parameter_fields_hook_that_does_not_fit_the_batch(
        self, parameter_vector, climate, observation_vector, theta, make_hook, match
    ):
        with pytest.raises(ValueError, match=match):
            self._build(
                parameter_vector,
                climate,
                observation_vector,
                to_sipnet_parameter_fields=make_hook(parameter_vector),
            ).evaluate(theta)


class TestRealSipnet:
    def test_two_samples_two_sites_under_a_process_backend(
        self, parameter_vector, observation_vector, files, theta
    ):
        from pysipnet.build import find_binary, missing_binary_message

        if find_binary() is None:
            pytest.skip(missing_binary_message())
        from conftest import niwot_parameters
        from sipnet_calibration.fields import to_model_output
        from sipnet_calibration.parameter_vector import sipnet_overrides

        model = SIPNETModel(
            SIPNETRunner(flags=ModelFlags.standard(), timeout=120.0), base_params=niwot_parameters()
        )
        forward = ForwardModel(
            model,
            parameter_vector,
            climate=files,
            backend=LocalBackend(n_workers=2),
            observation_vector=observation_vector,
            site_table=SITE_TABLE,
        )
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.predictions.shape == (2, observation_vector.dimension)
        assert np.isfinite(evaluation.predictions).all() and evaluation.valid.all()

        # One run's predictions, recomputed by hand: the same run and the same operators, on the driver.
        sample, site = 1, 27
        overrides = sipnet_overrides(evaluation.sipnet_parameter_fields, batch={"sample": sample}, site=site)
        run = model(climate=files[site], **overrides)
        direct = to_model_output(
            run.outputs.select(list(observation_vector.output_variable_names)),
            site=site,
            site_table=SITE_TABLE,
        )
        one_site = observation_vector.select(sites=[site])
        run_parameters = xr.Dataset(
            {name: run.parameters.dataarray(name) for name in one_site.sipnet_parameter_names_read},
            coords={name: direct[name] for name in ("site", "lon", "lat")},
        )
        expected = one_site.flat(one_site.predict(direct, sipnet_parameter_fields=run_parameters))
        np.testing.assert_allclose(
            evaluation.predictions[sample, observation_vector.positions(site=site)], expected
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

    evaluation = ForwardEvaluation(
        theta=np.zeros((2, 3)),
        sipnet_parameter_fields=xr.Dataset(),
        model_output=None,
        predictions=np.zeros((2, 4)),
        run_succeeded=xr.DataArray(np.ones((2, 1), dtype=bool), dims=("sample", "site")),
        failures=pd.DataFrame(),
        valid=np.ones(2, dtype=bool),
    )
    copy = dataclasses.replace(evaluation)
    assert evaluation == evaluation and evaluation != copy
    assert len({evaluation, copy}) == 2


class TestTheBatchDimIsNamedOnce:
    """``batch_dim=`` names the SIPNET parameter fields' dim, PyEns's axis, and every output."""

    def test_every_output_carries_the_named_batch_dim(self, parameter_vector, climate, theta):
        def blow_up_one(theta):
            sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta, batch_dim="draw")
            values = sipnet_parameter_fields["max_photosynthesis_rate"].values.copy()
            if values.shape[0] > 1:
                values[1, 0] = 1.5 * BLOW_UP
            rate = sipnet_parameter_fields["max_photosynthesis_rate"].copy(data=values)
            return sipnet_parameter_fields.assign(max_photosynthesis_rate=rate)

        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            output_variable_names=("wood_carbon",),
            site_table=SITE_TABLE,
            to_sipnet_parameter_fields=blow_up_one,
            batch_dim="draw",
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.sipnet_parameter_fields["soil_carbon"].dims == ("draw", "site")
        assert evaluation.model_output["wood_carbon"].dims == ("draw", "site", "time")
        assert evaluation.model_output["draw"].values.tolist() == [0, 1, 2]
        assert evaluation.run_succeeded.dims == ("draw", "site")
        assert list(evaluation.failures.columns) == ["draw", "site", "error", "message"]
        assert evaluation.failures["draw"].tolist() == [1]
        assert evaluation.valid.tolist() == [True, False, True]

    def test_the_default_hook_takes_the_named_batch_dim(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = ForwardModel(
            scaled_niwot_model(),
            parameter_vector,
            climate=climate,
            backend=SequentialBackend(),
            observation_vector=observation_vector,
            site_table=SITE_TABLE,
            batch_dim="draw",
        )
        evaluation = forward.evaluate(theta)
        assert evaluation.sipnet_parameter_fields["soil_carbon"].dims == ("draw", "site")
        np.testing.assert_allclose(
            evaluation.predictions,
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                site_table=SITE_TABLE,
            )(theta),
        )

    def test_a_hook_whose_batch_dim_is_named_otherwise_is_refused(
        self, parameter_vector, climate, observation_vector
    ):
        with pytest.raises(ValueError, match="name the batch dim 'draw'"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                site_table=SITE_TABLE,
                batch_dim="draw",
                to_sipnet_parameter_fields=parameter_vector.sipnet_parameter_fields,
            )

    def test_a_reserved_name_is_refused(self, parameter_vector, climate, observation_vector):
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                batch_dim="site",
            )

    @pytest.mark.parametrize("name", ["soil_carbon", "pft", "allocation.leaf_allocation"])
    def test_a_name_the_vector_refuses_is_refused_up_front(
        self, parameter_vector, climate, observation_vector, name
    ):
        """Even with a custom hook, which the vector's check never ran for."""
        for hook in (None, parameter_vector.sipnet_parameter_fields):
            with pytest.raises(ValueError, match=f"batch_dim={name!r} is"):
                ForwardModel(
                    scaled_niwot_model(),
                    parameter_vector,
                    climate=climate,
                    backend=SequentialBackend(),
                    observation_vector=observation_vector,
                    site_table=SITE_TABLE,
                    batch_dim=name,
                    to_sipnet_parameter_fields=hook,
                )

    def test_an_output_variable_name_is_refused_up_front(self, parameter_vector, climate):
        with pytest.raises(ValueError, match="'wood_carbon' is an output variable"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                output_variable_names=("wood_carbon",),
                site_table=SITE_TABLE,
                batch_dim="wood_carbon",
            )

    def test_an_output_variable_alias_is_refused_up_front(self, parameter_vector, climate):
        """``nee`` was accepted, though the output selection resolves it to
        ``net_ecosystem_exchange``."""
        with pytest.raises(ValueError, match="'nee' is an alias of the output variable"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                output_variable_names=("net_ecosystem_exchange",),
                site_table=SITE_TABLE,
                batch_dim="nee",
            )

    @pytest.mark.parametrize(
        "name",
        ["timestep_length", "timestep_start", "time_bounds", "bounds", "year", "day_of_year"],
    )
    @pytest.mark.parametrize("freq", [None, "1D"])
    def test_a_name_the_model_output_uses_is_refused_up_front(
        self, parameter_vector, climate, name, freq
    ):
        """It was refused only at stacking, after every run had completed."""
        with pytest.raises(ValueError, match=f"{name!r} cannot name a batch dim; it is a coordinate"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                output_variable_names=("wood_carbon",),
                freq=freq,
                site_table=SITE_TABLE,
                batch_dim=name,
            )

    @pytest.mark.parametrize("name", ["driver_member", "initial_condition_member"])
    def test_a_data_source_member_name_is_refused(self, parameter_vector, climate, name):
        """Theta's rows named ``driver_member`` were stamped as driver members."""
        with pytest.raises(ValueError, match="a data source's member dim"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                output_variable_names=("wood_carbon",),
                site_table=SITE_TABLE,
                batch_dim=name,
            )

    def test_an_observation_source_name_is_refused_up_front(
        self, parameter_vector, climate, observation_vector
    ):
        """It was accepted, and the vector's fields() then refused the predictions."""
        with pytest.raises(ValueError, match="is an observation source name"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                site_table=SITE_TABLE,
                batch_dim="landtrendr_aboveground_biomass",
            )

    def test_the_batch_dim_is_read_only(self, forward):
        assert forward.batch_dim == "sample"
        with pytest.raises(AttributeError):
            forward.batch_dim = "draw"

    @pytest.mark.parametrize(
        "name",
        [
            "model", "parameter_vector", "observation_vector", "backend", "freq", "climate",
            "sites", "site_table", "output_variable_names", "sipnet_parameter_names_written",
        ],
    )
    def test_what_it_was_built_from_is_read_only(self, forward, name):
        with pytest.raises(AttributeError):
            setattr(forward, name, getattr(forward, name))

    def test_the_climate_and_site_table_it_hands_out_cannot_change_it(self, forward):
        with pytest.raises(TypeError):
            forward.climate[1] = None
        table = forward.site_table
        table.loc[table.index[0], "lon"] = 0.0
        assert forward.site_table["lon"].iloc[0] != 0.0

    def test_flat_is_jax(self, forward, theta):
        evaluation = forward.evaluate(theta.tolist())
        for array in (evaluation.theta, evaluation.predictions, evaluation.valid):
            assert isinstance(array, jax.Array)
        assert isinstance(forward(jax.numpy.asarray(theta[0])), jax.Array)

    def test_flat_is_jax_on_the_prior_predictive_path(self, parameter_vector, climate, theta):
        prior = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate, backend=SequentialBackend(),
            output_variable_names=("wood_carbon",), site_table=SITE_TABLE,
        )
        assert isinstance(prior.evaluate(theta).valid, jax.Array)

    def test_flat_is_jax_in_what_a_machinery_failure_carries(
        self, parameter_vector, climate, observation_vector, theta
    ):
        forward = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate, backend=SequentialBackend(),
            observation_vector=observation_vector, site_table=SITE_TABLE,
            to_sipnet_parameter_fields=_hooked(parameter_vector, {(1, 0): 1.5 * DIES_BAND}),
        )
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta)
        evaluation = raised.value.evaluation
        assert isinstance(evaluation.theta, jax.Array) and isinstance(evaluation.valid, jax.Array)

    def test_a_hook_that_writes_more_than_the_vector_runs(
        self, parameter_vector, climate, observation_vector, theta
    ):
        """The free inputs are the hook's, learned when built, not the vector's."""

        def hook(t):
            fields = parameter_vector.sipnet_parameter_fields(t)
            return fields.assign(
                leaf_carbon_per_area=parameter_dataarray(
                    "leaf_carbon_per_area", fields["soil_carbon"] * 0 + 50.0
                )
            )

        forward = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate, backend=SequentialBackend(),
            observation_vector=observation_vector, site_table=SITE_TABLE,
            to_sipnet_parameter_fields=hook,
        )
        assert "leaf_carbon_per_area" in forward.sipnet_parameter_names_written
        assert "leaf_carbon_per_area" not in parameter_vector.sipnet_parameter_names_written
        assert forward(theta).shape == (len(theta), observation_vector.dimension)

    def test_nothing_read_from_an_evaluation_changes_it(
        self, parameter_vector, climate, theta
    ):
        """run_succeeded's and model_output's values could be written, failures edited."""
        prior = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate,
            backend=SequentialBackend(), output_variable_names=("wood_carbon",),
            site_table=SITE_TABLE,
        )
        evaluation = prior.evaluate(theta)
        with pytest.raises(ValueError, match="read-only"):
            evaluation.run_succeeded.values[0, 0] = False
        with pytest.raises(ValueError, match="read-only"):
            evaluation.model_output["wood_carbon"].values[0, 0, 0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            evaluation.sipnet_parameter_fields["soil_carbon"].values[0, 0] = 0.0
        evaluation.failures.loc[0] = [0, 1, "x", "y"]
        assert evaluation.failures.empty
        evaluation.run_succeeded.attrs["long_name"] = "x"
        assert evaluation.run_succeeded.attrs["long_name"] == "Whether the run succeeded"
        with pytest.raises(TypeError):
            ForwardEvaluation(*[None] * 7)

    def test_what_the_hook_returns_stays_writeable_in_its_hands(
        self, parameter_vector, climate, observation_vector, theta
    ):
        """The evaluation froze the Dataset the hook returned, which the hook may keep."""
        returned = []

        def hook(t):
            returned.append(parameter_vector.sipnet_parameter_fields(t).copy(deep=True))
            return returned[-1]

        forward = ForwardModel(
            scaled_niwot_model(), parameter_vector, climate=climate, backend=SequentialBackend(),
            observation_vector=observation_vector, site_table=SITE_TABLE,
            to_sipnet_parameter_fields=hook,
        )
        evaluation = forward.evaluate(theta)
        evaluation.sipnet_parameter_fields
        returned[-1]["soil_carbon"].values[0, 0] = 0.0
        assert evaluation.sipnet_parameter_fields["soil_carbon"].values[0, 0] != 0.0

    def test_an_evaluation_built_from_the_callers_arrays_leaves_them_writeable(self):
        run_succeeded = xr.DataArray(np.ones((1, 1), dtype=bool), dims=("sample", "site"))
        failures = pd.DataFrame({"sample": [0]})
        evaluation = ForwardEvaluation(
            theta=jax.numpy.zeros((1, 1)), sipnet_parameter_fields=xr.Dataset(),
            model_output=None, predictions=None, run_succeeded=run_succeeded,
            failures=failures, valid=jax.numpy.ones(1, dtype=bool),
        )
        evaluation.run_succeeded, evaluation.failures
        run_succeeded.values[0, 0] = False
        assert bool(evaluation.run_succeeded.values[0, 0])

    def test_a_vector_located_by_another_site_table_is_refused(self, climate):
        """Its SIPNET parameter fields kept its own lon/lat, beside runs labeled from another."""
        located = example_parameter_vector(
            site_table=SITE_TABLE, pft=("temperate.deciduous", "boreal.coniferous")
        )
        elsewhere = site_table_of(*SITES, lon=[-100.0, -60.0], lat=[40.0, 45.0])
        with pytest.raises(ValueError, match="other lon values than the model's site table"):
            ForwardModel(
                scaled_niwot_model(), located, climate=climate, backend=SequentialBackend(),
                output_variable_names=("wood_carbon",), site_table=elsewhere,
            )
        ForwardModel(
            scaled_niwot_model(), located, climate=climate, backend=SequentialBackend(),
            output_variable_names=("wood_carbon",), site_table=SITE_TABLE,
        )

    def test_a_crossed_batch_is_refused_with_the_advice_to_give_theta_its_rows(
        self, parameter_vector, climate, observation_vector
    ):
        def crossed(theta):
            fields = parameter_vector.sipnet_parameter_fields(theta)
            return fields.expand_dims(initial_condition_member=np.arange(2)).transpose(
                "sample", "initial_condition_member", "site"
            )

        with pytest.raises(ValueError, match=r"one row per combination.*stack_batch_dims"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                site_table=SITE_TABLE,
                to_sipnet_parameter_fields=crossed,
            )

    def test_a_hook_labeling_rows_with_floats_is_refused(
        self, parameter_vector, climate, observation_vector
    ):
        def floats(theta):
            sipnet_parameter_fields = parameter_vector.sipnet_parameter_fields(theta)
            return sipnet_parameter_fields.assign_coords(sample=sipnet_parameter_fields["sample"].values.astype(float))

        with pytest.raises(ValueError, match="integer"):
            ForwardModel(
                scaled_niwot_model(),
                parameter_vector,
                climate=climate,
                backend=SequentialBackend(),
                observation_vector=observation_vector,
                site_table=SITE_TABLE,
                to_sipnet_parameter_fields=floats,
            )


class TestRunSucceededIsAField:
    def test_run_succeeded_validates_as_a_field(self, forward, theta):
        from sipnet_calibration.fields import validate_field

        evaluation = forward.evaluate(theta)
        validate_field(evaluation.run_succeeded)
        assert evaluation.run_succeeded["site"].dtype == np.int32
        assert evaluation.run_succeeded["lon"].values.tolist() == [-105.0, -70.0]
        assert evaluation.run_succeeded["sample"].attrs["long_name"] == "Sample"

    def test_run_succeeded_is_mapped(self, forward, theta):
        """It validated as a field and was refused by the map for want of units."""
        import matplotlib.pyplot as plt

        from sipnet_calibration.plotting import plot_map

        evaluation = forward.evaluate(theta)
        figure, ax = plt.subplots()
        try:
            plot_map(evaluation.run_succeeded.isel(sample=0), ax=ax)
        finally:
            plt.close(figure)

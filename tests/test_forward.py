"""The forward model: theta to predictions through PyEns, on a stand-in and on SIPNET."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import LocalBackend, SequentialBackend
from pysipnet import niwot_reference_climate, niwot_reference_output
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.output import SIPNETOutput
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunError, SIPNETRunner

from sipnet_calibration.compute import scc_backend
from sipnet_calibration.forward import ForwardEvaluation, ForwardModel, ModelOutputNotFinite
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    Observation,
    ObservationVector,
    SelectTimestep,
    aggregate_time,
    select_timestep_at,
)
from sipnet_calibration.parameter_vector import example_parameter_vector

SITES = (1, 27)
REFERENCE = niwot_reference_output()
REFERENCE_WOOD = REFERENCE.select(["wood_carbon"])["wood_carbon"]
SHORT_STEPS = 40                       # site 27's drivers are cut to this many steps
LABELS = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 30]])
SITE_TABLE = pd.DataFrame({"site_id": list(SITES), "lon": [-105.0, -70.0], "lat": [40.0, 45.0]})
SOIL_REFERENCE = 1.0e4

#: A run "fails at its parameters" past this rate, writes NaN in a band above it,
#: times out in a band above that, and "fails in the machinery" below zero.
BLOW_UP = 1e6
NAN_BAND = 2e6
TIMEOUT_BAND = 3e6


class ScaledNiwot(SIPNETModel):
    """A SIPNETModel whose run is the Niwot output scaled by two parameters.

    ``wood_carbon`` is multiplied by ``max_photosynthesis_rate / 10`` (which
    the example vector shares across sites) and by ``soil_carbon /
    SOIL_REFERENCE`` (which it varies by site), so which parameter values
    reached which run can be read off the result, site by site. The run is as
    long as its drivers, so which drivers reached which run shows too.
    Defined at module level so PyEns can pickle it.
    """

    def __call__(self, *, climate=None, events=None, **overrides):
        rate = float(overrides["max_photosynthesis_rate"])
        if rate < 0:
            raise RuntimeError("the node died")
        if rate > TIMEOUT_BAND:
            raise subprocess.TimeoutExpired(cmd="sipnet", timeout=0.001)
        if rate > BLOW_UP and rate <= NAN_BAND:
            raise SIPNETRunError("SIPNET blew up", returncode=1, stdout="", stderr="", workdir=Path("/tmp"))
        n = climate.n_timesteps
        frame = REFERENCE.pandas.iloc[:n].copy()
        frame["wood_carbon"] = frame["wood_carbon"] * (rate / 10.0) * (float(overrides["soil_carbon"]) / SOIL_REFERENCE)
        if rate > NAN_BAND:
            frame.loc[frame.index[-5:], "wood_carbon"] = np.nan
        return SimpleNamespace(outputs=SIPNETOutput.from_dataframe(frame, climate=climate))


def _stand_in():
    from conftest import niwot_parameters

    return ScaledNiwot(SIPNETRunner(flags=ModelFlags.standard(), verify_binary=False), base_params=niwot_parameters())


def _expected_wood(table, member, site, n_steps=None):
    """What the stand-in writes for wood carbon at this member and site, in Mg ha-1."""
    rate = float(table["max_photosynthesis_rate"].sel(member=member, site=site))
    soil = float(table["soil_carbon"].sel(member=member, site=site))
    wood = REFERENCE_WOOD if n_steps is None else REFERENCE_WOOD.isel(time=slice(0, n_steps))
    return select_timestep_at(wood, LABELS).values * (rate / 10.0) * (soil / SOIL_REFERENCE) * 0.01


@pytest.fixture(scope="module")
def vector():
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
def observations():
    wood = xr.DataArray(
        [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
        dims=("site", "time"), coords={"site": list(SITES), "time": LABELS},
        attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
    )
    lai = xr.DataArray(
        [[3.0, 2.0, np.nan], [np.nan, 1.0, 1.5]],
        dims=("site", "time"), coords={"site": list(SITES), "time": LABELS},
        attrs={"units": "m2 m-2"}, name="modis_leaf_area_index",
    )
    return ObservationVector([
        Observation("landtrendr_aboveground_biomass", wood, SelectTimestep("wood_carbon")),
        Observation("modis_leaf_area_index", lai, DEFAULT_OBS_OPS["modis_leaf_area_index"]),
    ])


@pytest.fixture
def forward(vector, climate, observations):
    return ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                        observation_vector=observations, site_table=SITE_TABLE)


@pytest.fixture(scope="module")
def theta(vector):
    return np.asarray(vector.sample(jax.random.key(3), 3))


def _hooked(vector, rates):
    """A to_sipnet_table hook setting max_photosynthesis_rate at (member, site position)."""

    def hook(theta):
        table = vector.sipnet_table(theta)
        values = table["max_photosynthesis_rate"].values.copy()
        for (member, site_position), rate in rates.items():
            if member < values.shape[0]:  # the init-time probe has one member
                values[member, site_position] = rate
        return table.assign(max_photosynthesis_rate=(("member", "site"), values))

    return hook


class TestEvaluate:
    def test_the_right_parameters_and_drivers_reach_the_right_run(self, forward, vector, observations, theta):
        evaluation = forward.evaluate(theta)
        assert isinstance(evaluation, ForwardEvaluation)
        table = vector.sipnet_table(theta)
        assert not np.allclose(table["soil_carbon"].sel(site=1), table["soil_carbon"].sel(site=27))
        fields = observations.fields(evaluation.predictions)["landtrendr_aboveground_biomass"]
        for member in range(3):
            for site in SITES:
                observed = observations["landtrendr_aboveground_biomass"].values.sel(site=site).notnull().values
                expected = _expected_wood(table, member, site)
                np.testing.assert_allclose(fields.sel(member=member, site=site).values[observed], expected[observed], rtol=1e-12)
        assert evaluation.valid.all() and evaluation.failures.empty
        assert evaluation.model_output is None

    def test_a_single_theta_gives_one_row(self, forward, theta, observations):
        assert forward(theta[0]).shape == (observations.dimension,)
        assert forward(theta).shape == (3, observations.dimension)
        assert forward(theta[:1]).shape == (1, observations.dimension)
        assert forward.output_dimension == observations.dimension
        assert forward.input_dimension == theta.shape[1]

    def test_the_lai_operator_reads_the_base_leaf_carbon_per_area(self, forward, observations, theta):
        assert forward.sipnet_parameter_names == forward.parameter_vector.sipnet_parameter_names
        assert set(forward._base_values) == {"leaf_carbon_per_area"}
        block = forward(theta)
        lai = observations.fields(block)["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        expected = leaf / forward._base_values["leaf_carbon_per_area"]
        observed = observations["modis_leaf_area_index"].values.sel(site=1).notnull().values
        np.testing.assert_allclose(lai.sel(member=0, site=1).values[observed], expected[observed], rtol=1e-12)

    def test_an_observation_vector_over_fewer_sites_than_are_run(self, vector, climate, observations, theta):
        one_site = observations.select(sites=[1])
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               observation_vector=one_site, site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta)
        assert evaluation.predictions.shape == (3, one_site.dimension)
        assert evaluation.run_succeeded.shape == (3, 2) and bool(evaluation.run_succeeded.all())
        assert np.isfinite(evaluation.predictions).all()

    def test_a_site_in_the_coordinate_with_no_observed_cell(self, vector, climate, observations, theta):
        """A product's array keeps every site in its coordinate, observed or not."""
        wood = observations["landtrendr_aboveground_biomass"].values.copy()
        wood.loc[{"site": 27}] = np.nan
        sparse = ObservationVector([Observation("landtrendr_aboveground_biomass", wood, SelectTimestep("wood_carbon"))])
        assert sparse.positions(site=27).size == 0 and sparse["landtrendr_aboveground_biomass"].values.sizes["site"] == 2
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               observation_vector=sparse, site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta[:1])
        assert evaluation.predictions.shape == (1, sparse.dimension) and np.isfinite(evaluation.predictions).all()

    def test_names_are_resolved(self, vector, climate):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee", "wood_carbon"), site_table=SITE_TABLE)
        assert forward.output_variable_names == ("net_ecosystem_exchange", "wood_carbon")


class TestFailures:
    def _forward_with(self, vector, climate, observations, rates, backend=None):
        return ForwardModel(_stand_in(), vector, climate=climate, backend=backend or SequentialBackend(),
                            observation_vector=observations, site_table=SITE_TABLE,
                            to_sipnet_table=_hooked(vector, rates))

    def test_a_run_failing_at_its_parameters_is_a_nan_row(self, vector, climate, observations, theta):
        forward = self._forward_with(vector, climate, observations, {(1, 0): 1.5 * BLOW_UP})
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, False, True]
        assert np.isnan(evaluation.predictions[1]).all()
        assert np.isfinite(evaluation.predictions[[0, 2]]).all()
        assert not bool(evaluation.run_succeeded.sel(member=1, site=1))
        assert bool(evaluation.run_succeeded.sel(member=1, site=27))
        assert evaluation.failures["error"].tolist() == ["SIPNETRunError"]
        assert evaluation.failures["site"].tolist() == [1]

    def test_a_run_writing_nan_is_a_failure_at_its_parameters(self, vector, climate, observations, theta):
        forward = self._forward_with(vector, climate, observations, {(2, 1): 1.5 * NAN_BAND})
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, True, False]
        assert evaluation.failures["error"].tolist() == ["ModelOutputNotFinite"]

    def test_the_machinery_failing_is_raised_with_what_was_collected(self, vector, climate, observations, theta):
        forward = self._forward_with(vector, climate, observations, {(2, 1): -1.0, (0, 0): 1.5 * BLOW_UP})
        with pytest.raises(RuntimeError, match="machinery") as raised:
            forward.evaluate(theta)
        partial = raised.value.evaluation
        assert partial.predictions is None and partial.model_output is None
        assert partial.failures["error"].tolist() == ["SIPNETRunError"]
        assert bool(partial.run_succeeded.sel(member=1).all())
        assert not partial.valid.any()

    def test_a_timeout_that_crosses_a_process_boundary_is_a_nan_row(self, vector, files, observations, theta):
        """A TimeoutExpired built with keyword arguments does not unpickle; PyEns wraps it."""
        forward = self._forward_with(vector, files, observations, {(0, 1): 1.5 * TIMEOUT_BAND},
                                     backend=LocalBackend(n_workers=1))
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.valid.tolist() == [False, True]
        assert evaluation.failures["error"].tolist()[0] in ("TimeoutExpired", "RemoteError")

    def test_an_infinite_prediction_is_invalid_though_the_run_succeeded(self, vector, climate, theta):
        @dataclass(frozen=True)
        class Infinite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                out = select_timestep_at(model_output["wood_carbon"], observed_values["time"])
                out = out / 0.0
                out.attrs = {"units": "g m-2", "constituent": "C"}
                return out

        wood = xr.DataArray([[100.0, 110.0, 120.0]], dims=("site", "time"), coords={"site": [1], "time": LABELS},
                            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass")
        infinite = ObservationVector([Observation("landtrendr_aboveground_biomass", wood, Infinite())])
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               observation_vector=infinite, site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta[:1])
        assert bool(evaluation.run_succeeded.all()) and not evaluation.valid.any()


class TestPriorPredictive:
    def test_model_output_is_stacked_over_member_and_site(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee", "wood_carbon"), site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta)
        output = evaluation.model_output
        assert evaluation.predictions is None
        assert set(output.data_vars) == {"net_ecosystem_exchange", "wood_carbon"}
        assert output["wood_carbon"].dims == ("member", "site", "time")
        assert output["lon"].values.tolist() == [-105.0, -70.0]
        table = vector.sipnet_table(theta)
        for site, n_steps in ((1, REFERENCE_WOOD.sizes["time"]), (27, SHORT_STEPS)):
            wood = output["wood_carbon"].sel(member=2, site=site).dropna("time")
            assert wood.sizes["time"] == n_steps                  # the shorter drivers make a shorter run
            rate = float(table["max_photosynthesis_rate"].sel(member=2, site=site))
            soil = float(table["soil_carbon"].sel(member=2, site=site))
            np.testing.assert_allclose(wood.values, REFERENCE_WOOD.values[:n_steps] * rate / 10.0 * soil / SOIL_REFERENCE)
        assert output["wood_carbon"].attrs["units"] == "g m-2"

    def test_freq_aggregates_on_the_worker_as_aggregate_time_does(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee",), freq="1D", site_table=SITE_TABLE)
        output = forward.evaluate(theta[:1]).model_output
        expected = aggregate_time(REFERENCE.select(["net_ecosystem_exchange"])["net_ecosystem_exchange"], "1D")
        got = output["net_ecosystem_exchange"].sel(member=0, site=1).dropna("time")
        np.testing.assert_allclose(got.values, expected.values)
        assert output["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"

    def test_a_member_failing_at_every_site_keeps_its_slot(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee",), site_table=SITE_TABLE,
                               to_sipnet_table=_hooked(vector, {(1, 0): 1.5 * BLOW_UP, (1, 1): 1.5 * BLOW_UP}))
        evaluation = forward.evaluate(theta)
        output = evaluation.model_output
        assert output["member"].values.tolist() == [0, 1, 2]
        assert bool(output["net_ecosystem_exchange"].sel(member=1).isnull().all())
        assert int(output["net_ecosystem_exchange"].sel(member=0, site=1).notnull().sum()) == REFERENCE_WOOD.sizes["time"]
        assert evaluation.valid.tolist() == [True, False, True]
        assert len(evaluation.failures) == 2

    def test_call_needs_an_observation_vector(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee",), site_table=SITE_TABLE)
        with pytest.raises(ValueError, match="needs an observation vector"):
            forward(theta)


class TestRefusals:
    def _build(self, vector, climate, observations=None, **kwargs):
        kwargs.setdefault("backend", SequentialBackend())
        kwargs.setdefault("site_table", SITE_TABLE)
        return ForwardModel(_stand_in(), vector, climate=climate, observation_vector=observations, **kwargs)

    def test_a_site_without_drivers(self, vector, observations):
        with pytest.raises(ValueError, match=r"no drivers for site\(s\) \[27\]"):
            self._build(vector, {1: REFERENCE.climate}, observations)

    def test_drivers_of_the_wrong_type(self, vector, observations):
        with pytest.raises(TypeError, match="ClimateDrivers"):
            self._build(vector, {1: REFERENCE.climate, 27: "site_27.clim"}, observations)

    def test_an_observed_site_that_is_not_run(self, vector, climate, observations):
        with pytest.raises(ValueError, match=r"observes site\(s\) \[27\]"):
            self._build(vector.select(sites=[1]), climate, observations)

    def test_memory_backed_drivers_under_a_process_backend(self, vector, climate, observations):
        with pytest.raises(ValueError, match="held in memory"):
            self._build(vector, climate, observations, backend=LocalBackend(n_workers=1))

    def test_output_names_must_cover_the_operators(self, vector, climate, observations):
        with pytest.raises(ValueError, match="operators read"):
            self._build(vector, climate, observations, output_variable_names=("nee",))

    def test_neither_names_nor_a_vector(self, vector, climate):
        with pytest.raises(ValueError, match="output_variable_names"):
            self._build(vector, climate)

    def test_freq_with_an_observation_vector(self, vector, climate, observations):
        with pytest.raises(ValueError, match="freq="):
            self._build(vector, climate, observations, freq="1D")

    def test_a_flag_gated_output_variable(self, vector, climate):
        with pytest.raises(ValueError, match="constant zero"):
            self._build(vector, climate, output_variable_names=("litter_carbon",))

    def test_a_site_table_without_locations_or_a_site(self, vector, climate, observations):
        with pytest.raises(ValueError, match="'lon' and 'lat'"):
            self._build(vector, climate, observations, site_table=pd.DataFrame({"site_id": [1, 27]}))
        with pytest.raises(ValueError, match=r"no row for site\(s\) \[27\]"):
            self._build(vector, climate, observations, site_table=SITE_TABLE.iloc[:1])

    def test_a_table_hook_that_changes_the_free_fields(self, vector, climate, observations, theta):
        forward = self._build(vector, climate, observations)
        forward._to_sipnet_table = lambda t: vector.sipnet_table(t).drop_vars("soil_carbon")
        with pytest.raises(ValueError, match="free fields are fixed"):
            forward.evaluate(theta)

    def test_a_table_hook_that_sets_an_unknown_parameter(self, vector, climate, observations):
        with pytest.raises(ValueError, match="not a pySIPNET parameter"):
            self._build(vector, climate, observations,
                        to_sipnet_table=lambda t: vector.sipnet_table(t).assign(not_a_parameter=lambda d: d["soil_carbon"]))

    def test_a_table_hook_may_reorder_the_fields(self, vector, climate, observations, theta):
        forward = self._build(vector, climate, observations)
        names = list(forward.sipnet_parameter_names)
        forward._to_sipnet_table = lambda t: vector.sipnet_table(t)[names[::-1]]
        assert forward(theta[:1]).shape == (1, observations.dimension)

    def test_the_wrong_theta(self, forward, theta):
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:, :-1])
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:0])
        bad = theta.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            forward.evaluate(bad)

    def test_not_a_sipnet_model_or_backend(self, vector, climate, observations):
        with pytest.raises(TypeError, match="SIPNETModel"):
            ForwardModel(lambda **k: None, vector, climate=climate, backend=SequentialBackend(),
                         observation_vector=observations)
        with pytest.raises(TypeError, match="Backend"):
            ForwardModel(_stand_in(), vector, climate=climate, backend="local", observation_vector=observations)


class TestRealSipnet:
    def test_two_members_two_sites_under_a_process_backend(self, vector, observations, files, theta):
        from pysipnet.build import find_binary, missing_binary_message

        if find_binary() is None:
            pytest.skip(missing_binary_message())
        from conftest import niwot_parameters
        from sipnet_calibration.fields import label_run
        from sipnet_calibration.parameter_vector import sipnet_overrides

        model = SIPNETModel(SIPNETRunner(flags=ModelFlags.standard(), timeout=120.0), base_params=niwot_parameters())
        forward = ForwardModel(model, vector, climate=files, backend=LocalBackend(n_workers=2),
                               observation_vector=observations, site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.predictions.shape == (2, observations.dimension)
        assert np.isfinite(evaluation.predictions).all() and evaluation.valid.all()

        # One cell, recomputed by hand: the same run and the same operators, on the driver.
        member, site = 1, 27
        overrides = sipnet_overrides(evaluation.sipnet_table, member=member, site=site)
        direct = label_run(model(climate=files[site], **overrides).outputs.select(list(observations.output_variable_names)),
                           site=site, site_table=SITE_TABLE)
        expected = observations.select(sites=[site]).predict(direct, sipnet_parameters={**forward._base_values, **overrides})
        got = observations.fields(evaluation.predictions)["landtrendr_aboveground_biomass"].sel(member=member, site=site)
        observed = observations["landtrendr_aboveground_biomass"].values.sel(site=site).notnull().values
        np.testing.assert_allclose(got.values[observed], expected["landtrendr_aboveground_biomass"].sel(site=site).values[observed])


class TestSccBackend:
    def test_carries_the_queue_and_the_environment(self, tmp_path):
        backend = scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-l mem_per_core=4G",))
        assert "-P dietzelab" in backend.directives
        assert "-l buyin" in backend.directives
        assert "-v PYSIPNET_CACHE_DIR,PYSIPNET_BINARY,SIPNET_CALIBRATION_DATA" in backend.directives
        assert backend.directives[-1] == "-l mem_per_core=4G"
        assert backend.directives.index("-P dietzelab") < backend.directives.index("-l mem_per_core=4G")

    def test_refuses_overriding_the_queue(self, tmp_path):
        with pytest.raises(ValueError, match="override"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-P other",))
        with pytest.raises(TypeError, match="not one string"):
            scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives="-l mem_per_core=4G")

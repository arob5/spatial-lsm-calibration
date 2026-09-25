"""The forward model: theta to predictions through PyEns, on a stand-in and on SIPNET."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyens import LocalBackend, SequentialBackend
from pysipnet import niwot_reference_climate, niwot_reference_output
from pysipnet.model import SIPNETModel
from pysipnet.output import SIPNETOutput
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunError, SIPNETRunner

from sipnet_calibration.compute import SCC_DIRECTIVES, scc_backend
from sipnet_calibration.forward import ForwardEvaluation, ForwardModel
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    Observation,
    ObservationVector,
    SelectTimestep,
    select_timestep_at,
)
from sipnet_calibration.parameter_vector import example_parameter_vector

SITES = (1, 27)
REFERENCE = niwot_reference_output()
REFERENCE_WOOD = REFERENCE.select(["wood_carbon"])["wood_carbon"]
LABELS = pd.DatetimeIndex(REFERENCE_WOOD["time"].values[[5, 20, 40]])
SITE_TABLE = pd.DataFrame({"site_id": list(SITES), "lon": [-105.0, -70.0], "lat": [40.0, 45.0]})

#: A run "fails at its parameters" past this rate, and "in the machinery" below zero.
BLOW_UP = 1e6


class ScaledNiwot(SIPNETModel):
    """A SIPNETModel whose run is the Niwot output with wood carbon scaled by a parameter.

    ``wood_carbon`` is multiplied by ``max_photosynthesis_rate / 10``, so which
    parameter value reached which run can be read off the result. Defined at
    module level so PyEns can pickle it.
    """

    def __call__(self, *, climate=None, events=None, **overrides):
        rate = float(overrides["max_photosynthesis_rate"])
        if rate > BLOW_UP:
            raise SIPNETRunError("SIPNET blew up", returncode=1, stdout="", stderr="", workdir=Path("/tmp"))
        if rate < 0:
            raise RuntimeError("the node died")
        frame = REFERENCE.pandas.copy()
        frame["wood_carbon"] = frame["wood_carbon"] * (rate / 10.0)
        return SimpleNamespace(outputs=SIPNETOutput.from_dataframe(frame, climate=REFERENCE.climate))


def _stand_in():
    from conftest import niwot_parameters

    return ScaledNiwot(SIPNETRunner(flags=ModelFlags.standard(), verify_binary=False), base_params=niwot_parameters())


@pytest.fixture(scope="module")
def vector():
    return example_parameter_vector(sites=SITES, pft=("temperate.deciduous", "boreal.coniferous"))


@pytest.fixture(scope="module")
def climate():
    return {site: niwot_reference_climate() for site in SITES}


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


class TestEvaluate:
    def test_the_right_parameter_reaches_the_right_run(self, forward, vector, observations, theta):
        evaluation = forward.evaluate(theta)
        assert isinstance(evaluation, ForwardEvaluation)
        table = vector.sipnet_table(theta)
        expected = select_timestep_at(REFERENCE_WOOD, LABELS).values * 0.01     # g m-2 -> Mg ha-1
        fields = observations.fields(evaluation.predictions)["landtrendr_aboveground_biomass"]
        for member in range(3):
            for site in SITES:
                rate = float(table["max_photosynthesis_rate"].sel(member=member, site=site))
                observed = observations["landtrendr_aboveground_biomass"].values.sel(site=site).notnull().values
                np.testing.assert_allclose(
                    fields.sel(member=member, site=site).values[observed], (expected * rate / 10.0)[observed], rtol=1e-12
                )
        assert evaluation.valid.all() and evaluation.failures.empty
        assert evaluation.model_output is None

    def test_a_single_theta_gives_one_row(self, forward, theta, observations):
        assert forward(theta[0]).shape == (observations.dimension,)
        assert forward(theta).shape == (3, observations.dimension)
        assert forward.n_predictions == observations.dimension
        assert forward.dimension == theta.shape[1]

    def test_the_lai_operator_reads_the_base_leaf_carbon_per_area(self, forward, observations, theta):
        assert forward.sipnet_parameter_names == forward.parameter_vector.sipnet_parameter_names
        assert "leaf_carbon_per_area" in forward._base_values
        block = forward(theta)
        lai = observations.fields(block)["modis_leaf_area_index"]
        leaf = select_timestep_at(REFERENCE.select(["leaf_carbon"])["leaf_carbon"], LABELS).values
        expected = leaf / forward._base_values["leaf_carbon_per_area"]
        observed = observations["modis_leaf_area_index"].values.sel(site=1).notnull().values
        np.testing.assert_allclose(lai.sel(member=0, site=1).values[observed], expected[observed], rtol=1e-12)


class TestFailures:
    def _forward_with(self, vector, climate, observations, rates):
        def hooked(theta):
            table = vector.sipnet_table(theta)
            values = table["max_photosynthesis_rate"].values.copy()
            for (member, site_position), rate in rates.items():
                if member < values.shape[0]:  # the init-time probe has one member
                    values[member, site_position] = rate
            return table.assign(max_photosynthesis_rate=(("member", "site"), values))

        return ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                            observation_vector=observations, site_table=SITE_TABLE, to_sipnet_table=hooked)

    def test_a_run_failing_at_its_parameters_is_a_nan_row(self, vector, climate, observations, theta):
        forward = self._forward_with(vector, climate, observations, {(1, 0): 2 * BLOW_UP})
        evaluation = forward.evaluate(theta)
        assert evaluation.valid.tolist() == [True, False, True]
        assert np.isnan(evaluation.predictions[1]).all()
        assert np.isfinite(evaluation.predictions[[0, 2]]).all()
        assert not bool(evaluation.run_succeeded.sel(member=1, site=1))
        assert bool(evaluation.run_succeeded.sel(member=1, site=27))
        assert evaluation.failures["error"].tolist() == ["SIPNETRunError"]
        assert evaluation.failures["site"].tolist() == [1]

    def test_the_machinery_failing_is_raised(self, vector, climate, observations, theta):
        forward = self._forward_with(vector, climate, observations, {(2, 1): -1.0})
        with pytest.raises(RuntimeError, match="machinery"):
            forward.evaluate(theta)


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
        rate = float(table["max_photosynthesis_rate"].sel(member=2, site=27))
        np.testing.assert_allclose(output["wood_carbon"].sel(member=2, site=27).values, REFERENCE_WOOD.values * rate / 10.0)
        assert output["wood_carbon"].attrs["units"] == "g m-2"

    def test_freq_aggregates_on_the_worker(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee",), freq="1D", site_table=SITE_TABLE)
        output = forward.evaluate(theta[:1]).model_output
        assert output.sizes["time"] < REFERENCE_WOOD.sizes["time"]
        assert output["net_ecosystem_exchange"].attrs["kind"] == "timestep_total"

    def test_call_needs_an_observation_vector(self, vector, climate, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               output_variable_names=("nee",), site_table=SITE_TABLE)
        with pytest.raises(ValueError, match="needs an observation vector"):
            forward(theta)


class TestRefusals:
    def test_a_site_without_drivers(self, vector, observations):
        with pytest.raises(ValueError, match=r"no drivers for site\(s\) \[27\]"):
            ForwardModel(_stand_in(), vector, climate={1: niwot_reference_climate()}, backend=SequentialBackend(),
                         observation_vector=observations, site_table=SITE_TABLE)

    def test_an_observed_site_that_is_not_run(self, vector, climate, observations):
        with pytest.raises(ValueError, match=r"observes site\(s\) \[27\]"):
            ForwardModel(_stand_in(), vector.select(sites=[1]), climate=climate, backend=SequentialBackend(),
                         observation_vector=observations, site_table=SITE_TABLE)

    def test_memory_backed_drivers_under_a_process_backend(self, vector, climate, observations):
        with pytest.raises(ValueError, match="held in memory"):
            ForwardModel(_stand_in(), vector, climate=climate, backend=LocalBackend(n_workers=1),
                         observation_vector=observations, site_table=SITE_TABLE)

    def test_output_names_must_cover_the_operators(self, vector, climate, observations):
        with pytest.raises(ValueError, match="operators read"):
            ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                         observation_vector=observations, output_variable_names=("nee",), site_table=SITE_TABLE)

    def test_neither_names_nor_a_vector(self, vector, climate):
        with pytest.raises(ValueError, match="output_variable_names"):
            ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(), site_table=SITE_TABLE)

    def test_a_table_hook_that_changes_the_free_fields(self, vector, climate, observations, theta):
        forward = ForwardModel(_stand_in(), vector, climate=climate, backend=SequentialBackend(),
                               observation_vector=observations, site_table=SITE_TABLE)
        forward._to_sipnet_table = lambda t: vector.sipnet_table(t).drop_vars("soil_carbon")
        with pytest.raises(ValueError, match="free fields are fixed"):
            forward.evaluate(theta)

    def test_the_wrong_theta_shape(self, forward, theta):
        with pytest.raises(ValueError, match="theta must be"):
            forward.evaluate(theta[:, :-1])

    def test_not_a_sipnet_model(self, vector, climate, observations):
        with pytest.raises(TypeError, match="SIPNETModel"):
            ForwardModel(lambda **k: None, vector, climate=climate, backend=SequentialBackend(),
                         observation_vector=observations)


class TestRealSipnet:
    @pytest.fixture(scope="class")
    def files(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("two-sites")
        from pysipnet.climate import ClimateDrivers

        paths = {}
        for site in SITES:
            path = directory / f"site_{site}.clim"
            niwot_reference_climate().to_file(path)
            paths[site] = ClimateDrivers.from_path(path)
        return paths

    def test_two_members_two_sites_under_a_process_backend(self, vector, observations, files, theta):
        from pysipnet.build import find_binary, missing_binary_message

        if find_binary() is None:
            pytest.skip(missing_binary_message())
        from conftest import niwot_parameters

        model = SIPNETModel(SIPNETRunner(flags=ModelFlags.standard(), timeout=120.0), base_params=niwot_parameters())
        forward = ForwardModel(model, vector, climate=files, backend=LocalBackend(n_workers=2),
                               observation_vector=observations, site_table=SITE_TABLE)
        evaluation = forward.evaluate(theta[:2])
        assert evaluation.predictions.shape == (2, observations.dimension)
        assert np.isfinite(evaluation.predictions).all()
        assert evaluation.valid.all()
        assert bool(evaluation.run_succeeded.all())


def test_the_scc_backend_carries_the_queue_and_the_environment(tmp_path):
    backend = scc_backend(walltime="00:10:00", work_dir=tmp_path, n_jobs=2, directives=("-l mem_per_core=4G",))
    assert backend.directives[: len(SCC_DIRECTIVES)] == SCC_DIRECTIVES
    assert any(d.startswith("-v PYSIPNET_CACHE_DIR") for d in backend.directives)
    assert backend.directives[-1] == "-l mem_per_core=4G"

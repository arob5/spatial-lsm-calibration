"""The observation vector: its index, its two representations, and ``predict``."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet import niwot_reference_output

from conftest import as_sipnet_parameter_fields, located, niwot_stack_of, one_run_sipnet_parameter_fields, site_table_of
from sipnet_calibration.fields import label_run
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    INDEX_LEVELS,
    ComputeLeafAreaIndex,
    ObservationSource,
    ObservationVector,
    ReduceOverRun,
    SelectTimestep,
    restrict_to_observed_sites,
    select_timestep_at,
)

VARIABLES = ["leaf_carbon", "wood_carbon", "soil_carbon"]


@pytest.fixture(scope="module")
def one_run():
    return label_run(
        niwot_reference_output().select(VARIABLES),
        site=1,
        batch={"sample": 0},
        site_table=site_table_of(1, lon=-105.0, lat=40.0, keyed=True),
    )


@pytest.fixture(scope="module")
def stack():
    """Two samples and two sites built from the Niwot run by known factors."""
    return niwot_stack_of(VARIABLES)


@pytest.fixture(scope="module")
def times(one_run):
    return pd.DatetimeIndex(one_run["time"].values[[5, 20, 40]])


@pytest.fixture
def lai(times):
    values = np.array([[3.0, np.nan, 2.5], [np.nan, 1.0, 2.0]])
    return located(xr.DataArray(values, dims=("site", "time"), coords={"site": [1, 2], "time": times}, attrs={"units": "m2 m-2"}, name="modis_leaf_area_index"))


@pytest.fixture
def wood(times):
    values = np.array([[100.0, 110.0, np.nan], [np.nan, np.nan, 120.0]])
    return located(xr.DataArray(values, dims=("site", "time"), coords={"site": [1, 2], "time": times}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass"))


@pytest.fixture
def soil():
    return located(xr.DataArray([5.0, np.nan], dims="site", coords={"site": [1, 2]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon"))


@pytest.fixture
def sipnet_parameter_fields():
    return as_sipnet_parameter_fields(xr.Dataset({"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0], [540.0, 270.0]])}, coords={"sample": [0, 1], "site": [1, 2]}))


@pytest.fixture
def vector(lai, wood, soil):
    return ObservationVector([
        ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=wood, operator=SelectTimestep("wood_carbon")),
        ObservationSource(observation_source_name="soilgrids_soil_organic_carbon", observed_values=soil, operator=ReduceOverRun("soil_carbon", "mean")),
    ])


class TestObservationSource:
    def test_refuses_a_batch_dimension(self, lai):
        for name in ("sample", "nee_member"):
            with pytest.raises(ValueError, match=rf"batch dim\(s\) \['{name}'\]"):
                ObservationSource(observation_source_name="x", observed_values=lai.expand_dims({name: [0]}), operator=SelectTimestep("wood_carbon"))

    def test_refuses_an_array_without_units(self, lai):
        bare = lai.copy()
        bare.attrs = {}
        with pytest.raises(ValueError, match="carries its units"):
            ObservationSource(observation_source_name="x", observed_values=bare, operator=SelectTimestep("wood_carbon"))

    def test_refuses_an_operator_without_declarations(self, lai):
        with pytest.raises(TypeError, match="must declare output_variable_names"):
            ObservationSource(observation_source_name="x", observed_values=lai, operator=lambda *a, **k: None)

    def test_orders_sites_and_times(self, lai):
        shuffled = lai.isel(site=[1, 0], time=[2, 0, 1])
        source = ObservationSource(observation_source_name="x", observed_values=shuffled, operator=SelectTimestep("wood_carbon"))
        assert source.sites == (1, 2)
        assert source.observed_values.indexes["time"].is_monotonic_increasing

    def test_is_built_by_keyword_only(self, lai):
        with pytest.raises(TypeError):
            ObservationSource("x", lai, SelectTimestep("wood_carbon"))

    def test_refuses_observed_values_that_are_not_fields(self, lai):
        plain = lai.drop_vars(["lon", "lat"])
        with pytest.raises(ValueError, match="carries no 'lon' coordinate"):
            ObservationSource(observation_source_name="x", observed_values=plain, operator=SelectTimestep("wood_carbon"))
        wide = lai.assign_coords(site=lai["site"].astype(np.int64))
        with pytest.raises(ValueError, match="site ids are int32"):
            ObservationSource(observation_source_name="x", observed_values=wide, operator=SelectTimestep("wood_carbon"))

    def test_refuses_one_site_as_a_scalar(self, lai):
        with pytest.raises(ValueError, match=r"on \(site,\) or \(site, time\)"):
            ObservationSource(observation_source_name="x", observed_values=lai.isel(site=0), operator=SelectTimestep("wood_carbon"))

    def test_validate_observed_values_is_the_check_it_applies(self, lai, soil):
        from sipnet_calibration.observation import validate_observed_values

        validate_observed_values(lai)
        validate_observed_values(soil)
        with pytest.raises(ValueError, match="batch dim"):
            validate_observed_values(lai.expand_dims(sample=[0]))
        with pytest.raises(ValueError, match="must be numeric"):
            validate_observed_values(lai.astype(bool))
        with pytest.raises(TypeError, match="DataArray"):
            validate_observed_values(lai.to_dataset())

    def test_observations_are_the_observed_values(self, lai):
        observations = ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon")).observations()
        assert len(observations) == 4
        assert observations["site"].tolist() == [1, 1, 2, 2]


class TestIndex:
    def test_is_site_major_then_observation_source_then_time(self, vector, times):
        rows = vector.index.tolist()
        assert vector.index.names == list(INDEX_LEVELS)
        assert [r[0] for r in rows] == [1, 1, 1, 1, 1, 2, 2, 2]
        assert [r[1] for r in rows[:5]] == ["modis_leaf_area_index"] * 2 + ["landtrendr_aboveground_biomass"] * 2 + ["soilgrids_soil_organic_carbon"]
        assert rows[0][2] == times[0] and rows[1][2] == times[2]
        assert pd.isna(rows[4][2])

    def test_dimension_and_y(self, vector):
        assert vector.dimension == 8
        assert vector.y.tolist() == [3.0, 2.5, 100.0, 110.0, 5.0, 1.0, 2.0, 120.0]

    def test_positions(self, vector):
        assert vector.positions(site=2).tolist() == [5, 6, 7]
        assert vector.positions(observation_source_name="soilgrids_soil_organic_carbon").tolist() == [4]
        assert vector.positions(site=1, observation_source_name="landtrendr_aboveground_biomass").tolist() == [2, 3]

    def test_declared_reads_are_the_union(self, vector):
        assert vector.output_variable_names == ("leaf_carbon", "wood_carbon", "soil_carbon")
        assert vector.sipnet_parameter_names_read == ("leaf_carbon_per_area",)

    def test_y_is_a_copy(self, vector):
        y = vector.y
        y[0] = -1.0
        assert vector.y[0] == 3.0

    def test_refuses_duplicate_observation_sources(self, lai):
        with pytest.raises(ValueError, match="unique"):
            ObservationVector([ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon"))] * 2)

    def test_refuses_an_observation_source_with_no_observations(self, lai):
        empty = lai.where(False)
        with pytest.raises(ValueError, match="hold no observation;"):
            ObservationVector([ObservationSource(observation_source_name="x", observed_values=empty, operator=SelectTimestep("wood_carbon"))])

    def test_describe(self, vector):
        table = vector.describe()
        assert table.loc["modis_leaf_area_index", "observations"] == 4
        assert pd.isna(table.loc["soilgrids_soil_organic_carbon", "first_time"])


class TestSelect:
    def test_by_sites(self, vector):
        sub = vector.select(sites=[2])
        assert sub.sites == (2,)
        assert sub.observation_source_names == ("modis_leaf_area_index", "landtrendr_aboveground_biomass")  # soil has no observation at site 2
        assert sub.y.tolist() == [1.0, 2.0, 120.0]

    def test_by_observation_sources(self, vector):
        sub = vector.select(observation_source_names=["landtrendr_aboveground_biomass"])
        assert sub.observation_source_names == ("landtrendr_aboveground_biomass",)
        assert sub.dimension == 3

    def test_by_time(self, vector, times):
        sub = vector.select(time=slice(times[0], times[1]))
        assert sub.dimension == 2 + 2 + 1  # lai at t0 (site 1) and t1 (site 2); wood at t0, t1; soil static
        assert "soilgrids_soil_organic_carbon" in sub.observation_source_names

    def test_an_empty_selection_is_refused(self, vector):
        with pytest.raises(ValueError, match="leaves no observation;"):
            vector.select(sites=[99])


class TestRepresentations:
    def test_fields_of_y_are_the_observed_values(self, vector, lai, wood, soil):
        fields = vector.fields(vector.y)
        for name, original in [("modis_leaf_area_index", lai), ("landtrendr_aboveground_biomass", wood), ("soilgrids_soil_organic_carbon", soil)]:
            # soil is observed at site 1 only, so its grid holds site 1 alone
            kept = original.sel(site=fields[name]["site"].values)
            np.testing.assert_array_equal(fields[name].values, kept.values)
            assert fields[name].attrs["units"] == original.attrs["units"]
        assert fields["soilgrids_soil_organic_carbon"]["site"].values.tolist() == [1]

    def test_flat_of_the_fields_is_y(self, vector):
        np.testing.assert_array_equal(vector.flat(vector.fields(vector.y)), vector.y)

    def test_a_batch_round_trips_with_a_sample_dimension(self, vector):
        batched_flat = np.arange(3 * vector.dimension, dtype=float).reshape(3, -1)
        fields = vector.fields(batched_flat)
        assert fields["modis_leaf_area_index"].dims == ("sample", "site", "time")
        np.testing.assert_array_equal(vector.flat(fields), batched_flat)

    def test_the_batch_dim_may_be_named(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        fields = vector.fields(batched_flat, batch_dim="draw")
        assert fields["modis_leaf_area_index"].dims == ("draw", "site", "time")
        np.testing.assert_array_equal(vector.flat(fields), batched_flat)
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            vector.fields(batched_flat, batch_dim="site")

    def test_a_scalar_batch_coordinate_gives_one_vector(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        one = {name: field.isel(sample=1) for name, field in vector.fields(batched_flat).items()}
        np.testing.assert_array_equal(vector.flat(one), batched_flat[1])

    def test_flat_accepts_a_larger_array(self, vector, lai):
        bigger = xr.full_like(lai.reindex(site=[1, 2, 3]), 7.0)
        bigger.attrs = lai.attrs
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"] = bigger
        flat = vector.flat(fields)
        assert flat[vector.positions(observation_source_name="modis_leaf_area_index")].tolist() == [7.0] * 4

    def test_flat_refuses_a_missing_observation(self, vector, lai):
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"] = lai.isel(time=[0, 1])
        with pytest.raises(ValueError, match="lacks .* time label"):
            vector.flat(fields)

    def test_flat_keeps_nan_predictions(self, vector):
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"][0, 0] = np.nan
        assert np.isnan(vector.flat(fields)[0])

    def test_fields_refuses_the_wrong_length(self, vector):
        with pytest.raises(ValueError, match="entries"):
            vector.fields(np.zeros(vector.dimension + 1))


class TestPredict:
    def test_a_stack_predicts_every_observation_source_in_its_units(self, vector, stack, sipnet_parameter_fields, times):
        predicted = vector.predict(stack, sipnet_parameter_fields=sipnet_parameter_fields)
        assert set(predicted) == set(vector.observation_source_names)
        assert predicted["landtrendr_aboveground_biomass"].attrs["units"] == "Mg ha-1"
        wood_g = select_timestep_at(stack["wood_carbon"], times)
        np.testing.assert_allclose(predicted["landtrendr_aboveground_biomass"].values, wood_g.values * 0.01)
        assert predicted["soilgrids_soil_organic_carbon"].dims == ("sample", "site")

    def test_flat_of_the_predictions_is_sample_by_observation(self, vector, stack, sipnet_parameter_fields):
        batched_flat = vector.flat(vector.predict(stack, sipnet_parameter_fields=sipnet_parameter_fields))
        assert batched_flat.shape == (2, vector.dimension)
        assert np.isfinite(batched_flat).all()
        # the second sample's pools are half the first's, and its leaf carbon per
        # area double
        wood = np.concatenate([vector.positions(observation_source_name=n) for n in ("landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon")])
        np.testing.assert_allclose(batched_flat[1, wood], 0.5 * batched_flat[0, wood])
        lai = vector.positions(observation_source_name="modis_leaf_area_index")
        np.testing.assert_allclose(batched_flat[1, lai], 0.25 * batched_flat[0, lai])

    def test_one_run_predicts_a_one_site_vector(self, vector, one_run):
        sub = vector.select(sites=[1])
        predicted = sub.predict(one_run, sipnet_parameter_fields=one_run_sipnet_parameter_fields(leaf_carbon_per_area=270.0))
        flat = sub.flat(predicted)
        assert flat.shape == (sub.dimension,)

    def test_a_failed_run_passes_through_as_nan(self, vector, stack, sipnet_parameter_fields):
        failed = stack.copy(deep=True)
        for name in VARIABLES:
            failed[name].loc[{"sample": 1, "site": 2}] = np.nan
        batched_flat = vector.flat(vector.predict(failed, sipnet_parameter_fields=sipnet_parameter_fields))
        assert np.isnan(batched_flat[1, vector.positions(site=2)]).all()
        assert np.isfinite(batched_flat[0]).all()
        assert np.isfinite(batched_flat[1, vector.positions(site=1)]).all()

    def test_a_gap_the_operator_produced_is_refused(self, lai, stack, sipnet_parameter_fields):
        @dataclass(frozen=True)
        class Gappy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                out = select_timestep_at(model_output["leaf_carbon"].sel(site=observed_values["site"].values), observed_values["time"])
                out[..., 0] = np.nan
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=Gappy())])
        with pytest.raises(ValueError, match="although the run succeeded"):
            vector.predict(stack)

    def test_a_wrong_dimension_is_refused_naming_both_units(self, lai, stack):
        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=SelectTimestep("leaf_carbon"))])
        with pytest.raises(ValueError, match="m2 m-2"):
            vector.predict(stack)

    def test_a_missing_parameter_is_refused(self, vector, stack):
        with pytest.raises(ValueError, match="pass sipnet_parameter_fields"):
            vector.predict(stack)

    def test_a_model_output_lacking_a_variable_is_refused(self, vector, stack, sipnet_parameter_fields):
        with pytest.raises(ValueError, match="lacks"):
            vector.predict(stack.drop_vars("soil_carbon"), sipnet_parameter_fields=sipnet_parameter_fields)

    def test_a_mapping_is_refused(self, vector, stack):
        with pytest.raises(TypeError, match="Dataset"):
            vector.predict({name: stack[name] for name in VARIABLES})


@pytest.fixture(scope="module")
def observed():
    """Three real constraints at three sites, where the processed files exist."""
    constraints = pytest.importorskip("sipnet_calibration.constraints")
    names = ["modis_leaf_area_index", "landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon"]
    try:
        return constraints.constraint_fields(names, sites=[3851, 3871, 3875])
    except FileNotFoundError as error:
        pytest.skip(str(error))


class TestRealConstraints:
    def test_the_vector_round_trips_the_ragged_constraints(self, observed):
        vector = ObservationVector([
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=observed["modis_leaf_area_index"], operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
            ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=observed["landtrendr_aboveground_biomass"], operator=SelectTimestep("wood_carbon")),
            ObservationSource(observation_source_name="soilgrids_soil_organic_carbon", observed_values=observed["soilgrids_soil_organic_carbon"], operator=ReduceOverRun("soil_carbon", "mean")),
        ])
        assert vector.dimension == sum(int(a.notnull().sum()) for a in observed.values())
        assert np.isfinite(vector.y).all()
        fields = vector.fields(vector.y)
        for name, array in observed.items():
            kept = array.sel(site=fields[name]["site"].values)
            if "time" in array.dims:
                kept = kept.sel(time=fields[name]["time"].values)
            np.testing.assert_array_equal(fields[name].values, kept.values)
            assert int(fields[name].notnull().sum()) == int(array.notnull().sum())
        np.testing.assert_array_equal(vector.flat(fields), vector.y)
        sites = vector.index.get_level_values("site").values
        assert (np.diff(sites) >= 0).all()


class TestTwinObservations:
    """Observations built from the model output are reproduced exactly."""

    def test_the_state_at_the_labels_in_the_observed_values_units(self, one_run, times):
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        at = ends.get_indexer(times)
        observed = located(xr.DataArray(
            wood.values[at][None, :] * 0.01,  # g m-2 -> Mg ha-1
            dims=("site", "time"), coords={"site": [1], "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
        ))
        vector = ObservationVector([ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=observed, operator=SelectTimestep("wood_carbon"))])
        predicted = vector.flat(vector.predict(one_run))
        np.testing.assert_allclose(predicted, vector.y, rtol=1e-12)

    def test_leaf_area_index_from_leaf_carbon_and_the_parameter(self, one_run, times):
        leaf = one_run["leaf_carbon"]
        at = pd.DatetimeIndex(leaf["time"].values).get_indexer(times)
        observed = located(xr.DataArray(
            (leaf.values[at] / 270.0)[None, :], dims=("site", "time"), coords={"site": [1], "time": times},
            attrs={"units": "m2 m-2"}, name="modis_leaf_area_index",
        ))
        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=observed, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"])])
        predicted = vector.flat(vector.predict(one_run, sipnet_parameter_fields=one_run_sipnet_parameter_fields(leaf_carbon_per_area=270.0)))
        np.testing.assert_allclose(predicted, vector.y, rtol=1e-12)
        wrong = vector.flat(vector.predict(one_run, sipnet_parameter_fields=one_run_sipnet_parameter_fields(leaf_carbon_per_area=135.0)))
        np.testing.assert_allclose(wrong, 2 * vector.y, rtol=1e-12)


class TestObservationInputsAndBatchedFlatShapes:
    def test_an_operator_on_the_wrong_time_labels_is_refused_by_predict(self, lai, stack, sipnet_parameter_fields):
        @dataclass(frozen=True)
        class OffGrid:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                out = select_timestep_at(model_output["leaf_carbon"].sel(site=observed_values["site"].values), observed_values["time"])
                out = out.assign_coords(time=out["time"].values + np.timedelta64(1, "h"))
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=OffGrid())])
        with pytest.raises(ValueError, match="time labels"):
            vector.predict(stack)

    def test_an_infinite_observation_is_refused(self, lai):
        lai[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon"))

    def test_a_nat_label_is_refused(self, lai):
        broken = lai.assign_coords(time=[lai["time"].values[0], np.datetime64("NaT"), lai["time"].values[2]])
        with pytest.raises(ValueError, match="NaT"):
            ObservationSource(observation_source_name="x", observed_values=broken, operator=SelectTimestep("wood_carbon"))

    def test_float_site_labels_are_refused(self, lai):
        with pytest.raises(ValueError, match="site ids are int32"):
            ObservationSource(observation_source_name="x", observed_values=lai.assign_coords(site=[1.0, 2.5]), operator=SelectTimestep("wood_carbon"))

    def test_an_aware_time_coordinate_is_refused(self, lai):
        aware = lai.assign_coords(time=pd.DatetimeIndex(lai["time"].values).tz_localize("UTC"))
        with pytest.raises(ValueError, match="naive datetime64"):
            ObservationSource(observation_source_name="x", observed_values=aware, operator=SelectTimestep("wood_carbon"))

    def test_mixed_batch_and_no_batch_fields_are_refused(self, vector):
        fields = vector.fields(np.zeros((2, vector.dimension)))
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].isel(sample=0, drop=True)
        with pytest.raises(ValueError, match="carry a batch dim"):
            vector.flat(fields)

    def test_fields_carrying_different_batch_dims_are_refused(self, vector):
        fields = vector.fields(np.zeros((2, vector.dimension)))
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].rename(sample="draw")
        with pytest.raises(ValueError, match="different batch dims"):
            vector.flat(fields)

    def test_fields_label_samples_from_zero(self, vector):
        fields = vector.fields(np.zeros((3, vector.dimension)))
        assert fields["modis_leaf_area_index"]["sample"].values.tolist() == [0, 1, 2]
        assert fields["modis_leaf_area_index"]["sample"].dtype == np.int64

    def test_select_takes_sequences_and_refuses_one_name_or_one_site(self, vector):
        assert vector.select(observation_source_names=["landtrendr_aboveground_biomass"]).dimension == 3
        assert vector.select(sites=[2]).sites == (2,)
        with pytest.raises(TypeError, match="one string"):
            vector.select(observation_source_names="landtrendr_aboveground_biomass")
        with pytest.raises(TypeError, match="sequence of site ids"):
            vector.select(sites=2)
        with pytest.raises(TypeError, match="slice"):
            vector.select(time="2012")

    def test_observed_values_are_named_for_the_observation_source(self, lai):
        source = ObservationSource(observation_source_name="x", observed_values=lai.rename(None), operator=SelectTimestep("wood_carbon"))
        assert source.observed_values.name == "x"


class TestSelectKeepsOnlyObservedLabels:
    """A sub-vector's operators read the model only where a kept site is observed."""

    def test_a_one_site_selection_drops_the_other_sites_labels(self, vector, lai, times):
        sub = vector.select(sites=[1])
        # lai is observed at site 1 on the first and last label only
        kept = sub["modis_leaf_area_index"].observed_values
        assert kept["time"].values.tolist() == [times[0].value, times[2].value]
        np.testing.assert_array_equal(sub.y, vector.y[vector.positions(site=1)])
        assert sub.index.equals(vector.index[vector.positions(site=1)])

    def test_a_one_site_slice_predicts_from_that_sites_shorter_run(self, one_run, times):
        wood = located(xr.DataArray(
            [[100.0, np.nan, np.nan], [np.nan, np.nan, 120.0]], dims=("site", "time"),
            coords={"site": [1, 2], "time": times}, attrs={"units": "Mg ha-1", "constituent": "C"},
        ))
        vector = ObservationVector([ObservationSource(observation_source_name="wood", observed_values=wood, operator=SelectTimestep("wood_carbon"))])
        short = one_run.isel(time=slice(0, 30))  # ends before the label only site 2 is observed at
        predicted = vector.select(sites=[1]).predict(short)
        assert predicted["wood"].sizes["time"] == 1

    def test_a_generator_of_sites_is_read_once_for_every_observation_source(self, vector):
        sub = vector.select(sites=(site for site in (1, 2)))
        assert sub.observation_source_names == vector.observation_source_names
        np.testing.assert_array_equal(sub.y, vector.y)

    def test_selecting_thousands_of_sites_keeps_every_chosen_one(self):
        n = 8000
        values = located(xr.DataArray(
            np.ones((n, 2)), dims=("site", "time"),
            coords={"site": np.arange(1, n + 1), "time": pd.date_range("2000-01-01", periods=2)},
            attrs={"units": "Mg ha-1", "constituent": "C"},
        ))
        vector = ObservationVector([ObservationSource(observation_source_name="wood", observed_values=values, operator=SelectTimestep("wood_carbon"))])
        sub = vector.select(sites=range(1, n + 1, 2))
        assert sub.sites == tuple(range(1, n + 1, 2))

    def test_an_unknown_observation_source_is_refused(self, vector):
        with pytest.raises(KeyError, match="no observation source 'nothing'"):
            vector.select(observation_source_names=["nothing"])

    def test_observation_sources_come_in_the_requested_order(self, vector):
        names = ("soilgrids_soil_organic_carbon", "modis_leaf_area_index")
        assert vector.select(observation_source_names=list(names)).observation_source_names == names


class TestObservationSourceHoldsItsOwnValues:
    def test_a_write_to_the_callers_array_does_not_reach_the_observation_source(self, lai):
        source = ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon"))
        lai[1, 0] = 5.0  # an unobserved element of the caller's array
        assert source.n_observations == 4
        assert np.isnan(source.observed_values.values[1, 0])

    def test_the_stored_values_are_read_only(self, vector):
        dimension = vector.dimension
        with pytest.raises(ValueError, match="read-only"):
            vector.observed_values_by_source["modis_leaf_area_index"][0, 1] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            vector["modis_leaf_area_index"].observed_values.values[0, 1] = 1.0
        assert vector["modis_leaf_area_index"].n_observations == 4 and vector.dimension == dimension


class TestVectorSites:
    def test_a_site_observed_nowhere_is_not_a_site_of_the_vector(self, soil):
        vector = ObservationVector([ObservationSource(observation_source_name="soil", observed_values=soil, operator=ReduceOverRun("soil_carbon", "mean"))])
        assert soil["site"].values.tolist() == [1, 2]
        assert vector.sites == (1,)

    def test_sites_are_ascending_across_observation_sources(self, times):
        first = located(xr.DataArray([[1.0], [2.0]], dims=("site", "time"), coords={"site": [27, 3], "time": times[:1]}, attrs={"units": "m2 m-2"}))
        second = located(xr.DataArray([4.0, 5.0], dims="site", coords={"site": [8, 1]}, attrs={"units": "Mg ha-1", "constituent": "C"}))
        vector = ObservationVector([
            ObservationSource(observation_source_name="a", observed_values=first, operator=SelectTimestep("leaf_carbon")),
            ObservationSource(observation_source_name="b", observed_values=second, operator=ReduceOverRun("soil_carbon", "mean")),
        ])
        assert vector.sites == (1, 3, 8, 27)


class TestObservationSourceRefusals:
    def test_duplicate_sites_are_refused(self, lai):
        with pytest.raises(ValueError, match="repeats a site id"):
            ObservationSource(observation_source_name="x", observed_values=lai.assign_coords(site=np.array([1, 1], dtype=np.int32)), operator=SelectTimestep("wood_carbon"))

    def test_duplicate_times_are_refused(self, lai):
        repeated = lai.assign_coords(time=[lai["time"].values[0]] * 2 + [lai["time"].values[2]])
        with pytest.raises(ValueError, match="not strictly increasing"):
            ObservationSource(observation_source_name="x", observed_values=repeated, operator=SelectTimestep("wood_carbon"))

    def test_an_operator_that_is_not_callable_is_refused(self, lai):
        with pytest.raises(TypeError, match="must be callable"):
            ObservationSource(observation_source_name="x", observed_values=lai, operator="wood_carbon")

    def test_an_alias_declared_by_an_operator_is_refused(self, lai):
        @dataclass(frozen=True)
        class Aliased:
            output_variable_names = ("plantWoodC",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                return model_output["wood_carbon"]

        with pytest.raises(ValueError, match="alias"):
            ObservationSource(observation_source_name="x", observed_values=lai, operator=Aliased())

    def test_declarations_as_a_list_are_refused(self, lai):
        @dataclass(frozen=True)
        class Listed:
            output_variable_names = ["wood_carbon"]
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                return model_output["wood_carbon"]

        with pytest.raises(TypeError, match="tuple of names"):
            ObservationSource(observation_source_name="x", observed_values=lai, operator=Listed())


class TestFlatRefusals:
    def test_a_field_missing_an_observed_site_is_refused(self, vector):
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].sel(site=[1])
        with pytest.raises(ValueError, match=r"lacks observed site\(s\) \[2\]"):
            vector.flat(fields)

    def test_a_missing_observation_source_is_refused(self, vector):
        fields = vector.fields(vector.y)
        del fields["soilgrids_soil_organic_carbon"]
        with pytest.raises(ValueError, match="lack the observation source"):
            vector.flat(fields)

    def test_batch_labels_that_disagree_are_refused(self, vector):
        fields = vector.fields(np.zeros((2, vector.dimension)))
        fields["soilgrids_soil_organic_carbon"] = fields["soilgrids_soil_organic_carbon"].assign_coords(sample=[5, 6])
        with pytest.raises(ValueError, match="disagree on their sample labels"):
            vector.flat(fields)

    def test_the_batch_dim_need_not_be_the_leading_dimension(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        fields = vector.fields(batched_flat)
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].transpose("site", "time", "sample")
        np.testing.assert_array_equal(vector.flat(fields), batched_flat)

    def test_labels_in_seconds_are_read_by_instant(self, lai, one_run):
        coarse = lai.assign_coords(time=lai["time"].values.astype("datetime64[s]"))
        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=coarse, operator=SelectTimestep("leaf_carbon"))])
        assert vector["modis_leaf_area_index"].observed_values["time"].dtype == np.dtype("datetime64[s]")
        np.testing.assert_array_equal(vector.flat(vector.fields(vector.y)), vector.y)
        wide = xr.full_like(lai, 7.0)  # nanosecond labels, as a prediction carries them
        fields = {"modis_leaf_area_index": wide}
        assert vector.flat(fields).tolist() == [7.0] * vector.dimension


class TestFailedRuns:
    def test_a_run_with_only_some_missing_steps_has_not_failed(self, lai, stack):
        padded = stack.copy(deep=True)
        tail = padded["time"].values[-10:]
        padded["leaf_carbon"].loc[{"sample": 1, "site": 2, "time": tail}] = np.nan

        @dataclass(frozen=True)
        class Gappy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                leaf = restrict_to_observed_sites(model_output["leaf_carbon"], observed_values)
                out = select_timestep_at(leaf, observed_values["time"])
                out.loc[{"sample": 1, "site": 2, "time": out["time"].values[-1]}] = np.nan
                out.attrs = {"units": "1"}
                return out

        with pytest.raises(ValueError, match="although the run succeeded"):
            ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=Gappy())]).predict(padded)

    def test_one_read_variable_missing_throughout_is_a_failed_run(self, vector, stack, sipnet_parameter_fields):
        failed = stack.copy(deep=True)
        failed["wood_carbon"].loc[{"sample": 1, "site": 2}] = np.nan  # leaf and soil carbon finite
        batched_flat = vector.flat(vector.predict(failed, sipnet_parameter_fields=sipnet_parameter_fields))
        wood = vector.positions(site=2, observation_source_name="landtrendr_aboveground_biomass")
        assert np.isnan(batched_flat[1, wood]).all()
        assert np.isfinite(batched_flat[0]).all()

    def test_the_failure_mask_is_matched_by_site_label(self, lai, stack, sipnet_parameter_fields):
        reordered = stack.isel(site=[1, 0]).copy(deep=True)
        reordered["leaf_carbon"].loc[{"sample": 1, "site": 2}] = np.nan
        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=ComputeLeafAreaIndex())])
        batched_flat = vector.flat(vector.predict(reordered, sipnet_parameter_fields=sipnet_parameter_fields))
        assert np.isnan(batched_flat[1, vector.positions(site=2)]).all()
        assert np.isfinite(batched_flat[1, vector.positions(site=1)]).all()


class TestPredictSharesTheOperatorChecks:
    def test_a_result_with_a_spurious_batch_dim_is_refused(self, lai, one_run):
        @dataclass(frozen=True)
        class Spurious:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                out = select_timestep_at(model_output["leaf_carbon"], observed_values["time"])
                out = out.drop_vars("sample").expand_dims(sample=[0, 1])
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=Spurious())]).select(sites=[1])
        with pytest.raises(ValueError, match=r"dim\(s\) \['sample'\] that neither the model output"):
            vector.predict(one_run)

    def test_a_result_that_is_not_an_array_is_a_type_error(self, lai, stack):
        @dataclass(frozen=True)
        class Numpy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                return np.zeros(3)

        with pytest.raises(TypeError, match="not a DataArray"):
            ObservationVector([ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=Numpy())]).predict(stack)


class TestObservationSourceKeepsOnlyObservedLabels:
    def test_unobserved_sites_and_labels_are_dropped_on_construction(self, times):
        values = located(xr.DataArray(
            [[100.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, 120.0]],
            dims=("site", "time"), coords={"site": [1, 5, 9], "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"},
        ))
        source = ObservationSource(observation_source_name="wood", observed_values=values, operator=SelectTimestep("wood_carbon"))
        assert source.sites == (1, 9)
        assert source.observed_values["time"].values.tolist() == [times[0].value, times[2].value]
        assert source.n_observations == 2

    def test_a_static_sources_unobserved_sites_are_dropped(self, soil):
        assert ObservationSource(observation_source_name="soil", observed_values=soil, operator=ReduceOverRun("soil_carbon", "mean")).sites == (1,)

    def test_predict_needs_no_model_output_at_an_unobserved_site(self, one_run, times):
        values = located(xr.DataArray(
            [[100.0, np.nan, 110.0], [np.nan, np.nan, np.nan]], dims=("site", "time"),
            coords={"site": [1, 5], "time": times}, attrs={"units": "Mg ha-1", "constituent": "C"},
        ))
        vector = ObservationVector([ObservationSource(observation_source_name="wood", observed_values=values, operator=SelectTimestep("wood_carbon"))])
        predicted = vector.predict(one_run)  # one run at site 1; site 5 is observed nowhere
        assert predicted["wood"]["site"].values.tolist() == [1]
        assert predicted["wood"]["time"].values.tolist() == [times[0].value, times[2].value]

    def test_a_real_constraint_at_unobserved_sites_predicts_from_the_observed_ones(self):
        constraints = pytest.importorskip("sipnet_calibration.constraints")
        try:
            wood = constraints.constraint_fields(["landtrendr_aboveground_biomass"], sites=[1, 27, 3851])
        except FileNotFoundError as error:
            pytest.skip(str(error))
        array = wood["landtrendr_aboveground_biomass"]
        source = ObservationSource(observation_source_name="wood", observed_values=array, operator=SelectTimestep("wood_carbon"))
        observed_sites = [int(s) for s in array["site"].values if bool(array.sel(site=s).notnull().any())]
        assert list(source.sites) == observed_sites
        assert bool(source.observed_values.notnull().any("site").all())


class TestObservationSourceKeepsAScalarBatchLabelAsMetadata:
    def test_a_scalar_batch_coordinate_is_accepted(self, lai):
        one = lai.expand_dims(nee_member=[4]).isel(nee_member=0)
        source = ObservationSource(observation_source_name="x", observed_values=one, operator=SelectTimestep("wood_carbon"))
        assert source.n_observations == 4
        assert int(source.observed_values["nee_member"]) == 4


class TestSiteIdsAreIntegers:
    @pytest.mark.parametrize("sites", [True, [1.7], [2.0], ["1"], "1", [None], {2}])
    def test_select_refuses_what_is_not_a_sequence_of_site_ids(self, vector, sites):
        with pytest.raises(TypeError, match="sites"):
            vector.select(sites=sites)

    def test_a_float_site_is_a_type_error_and_an_id_out_of_range_a_value_error(self, vector):
        with pytest.raises(TypeError, match="float"):
            vector.positions(site=27.9)
        with pytest.raises(TypeError, match="float"):
            vector.positions(site=2.0)
        with pytest.raises(TypeError):
            vector.positions(site=True)
        with pytest.raises(ValueError, match="from 1 to"):
            vector.select(sites=[0])

    @pytest.mark.parametrize(
        "sites",
        [
            np.array([2]),
            [np.int32(2)],
            jnp.array([2]),
            xr.DataArray([2], dims="site"),
            pd.Index([2]),
        ],
    )
    def test_integer_array_likes_are_accepted(self, vector, sites):
        assert vector.select(sites=sites).sites == (2,)

    def test_positions_accepts_a_numpy_integer(self, vector):
        assert vector.positions(site=np.int64(2)).tolist() == vector.positions(site=2).tolist()


class TestObservationSourceIdentity:
    def test_observation_sources_compare_and_hash_by_identity(self, lai):
        first = ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon"))
        second = ObservationSource(observation_source_name="x", observed_values=lai, operator=SelectTimestep("wood_carbon"))
        assert first == first and first != second
        assert len({first, second, first}) == 2


class TestObservationSourceLoadsLazyValues:
    def test_dask_backed_values_are_loaded_and_read_only(self, lai):
        pytest.importorskip("dask")
        source = ObservationSource(observation_source_name="x", observed_values=lai.chunk({"site": 1}), operator=SelectTimestep("wood_carbon"))
        assert isinstance(source.observed_values.variable._data, np.ndarray)
        with pytest.raises(ValueError, match="read-only"):
            source.observed_values.values[0, 0] = -1.0
        assert source.observed_values.values[0, 0] == 3.0

    def test_file_backed_values_are_loaded_and_read_only(self, lai, tmp_path):
        path = tmp_path / "lai.nc"
        lai.to_netcdf(path)
        with xr.open_dataarray(path) as lazy:
            source = ObservationSource(observation_source_name="x", observed_values=lazy, operator=SelectTimestep("wood_carbon"))
        with pytest.raises(ValueError, match="read-only"):
            source.observed_values.values[0, 0] = -1.0
        assert source.observed_values.values[0, 0] == 3.0


class TestBatchSizes:
    def test_more_samples_than_int16_labels_are_labeled(self, vector):
        fields = vector.fields(np.zeros((40000, vector.dimension)))
        assert int(fields["modis_leaf_area_index"]["sample"].values[-1]) == 39999
        assert fields["modis_leaf_area_index"]["sample"].dtype == np.int64


class TestPredictOverAnyBatchDim:
    """A model output's batch dims may carry any name, and several of them."""

    def test_a_batch_dim_not_named_sample_predicts(self, vector, stack, sipnet_parameter_fields):
        """Before PR 2 a batch dim not named ``member`` crashed in predict with a
        raw xarray transpose error; it is now carried through like any other."""
        renamed = stack.rename(sample="driver_member")
        predicted = vector.predict(renamed, sipnet_parameter_fields=sipnet_parameter_fields.rename(sample="driver_member"))
        assert predicted["modis_leaf_area_index"].dims == ("driver_member", "site", "time")
        batched_flat = vector.flat(predicted)
        expected = vector.flat(vector.predict(stack, sipnet_parameter_fields=sipnet_parameter_fields))
        np.testing.assert_allclose(batched_flat, expected)

    def test_two_batch_dims_predict_and_flatten_once_stacked(self, vector, stack, sipnet_parameter_fields):
        from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

        # Observations that are fields in full: int32 site ids with lon/lat, so
        # both the predictions and the unstacked batch pass validate_field.
        def located(values):
            where = stack[["lon", "lat"]].sel(site=values["site"].values)
            return values.assign_coords(
                site=values["site"].astype(np.int32),
                lon=("site", where["lon"].values),
                lat=("site", where["lat"].values),
            )

        vector = ObservationVector([
            ObservationSource(observation_source_name=o.observation_source_name, observed_values=located(o.observed_values), operator=o.operator) for o in vector.observation_sources
        ])

        crossed = xr.concat(
            [stack, stack * 2.0], dim=pd.Index([0, 1], name="initial_condition_member"), coords="minimal",
        )
        for name in crossed.data_vars:
            crossed[name].attrs = stack[name].attrs
        crossed = crossed.transpose("sample", "initial_condition_member", "site", "time")
        predicted = vector.predict(crossed, sipnet_parameter_fields=sipnet_parameter_fields)
        assert predicted["modis_leaf_area_index"].dims == (
            "sample", "initial_condition_member", "site", "time"
        )
        with pytest.raises(ValueError, match="stack_batch_dims"):
            vector.flat(predicted)
        stacked = {name: stack_batch_dims(field, into="run") for name, field in predicted.items()}
        batched_flat = vector.flat(stacked)
        assert batched_flat.shape == (4, vector.dimension)
        back = vector.fields(batched_flat, batch_dim="run")
        restored = unstack_batch_dims(
            back["modis_leaf_area_index"], labels_from=stacked["modis_leaf_area_index"]
        )
        observed = vector["modis_leaf_area_index"].observed_values.notnull()
        assert restored.dims == predicted["modis_leaf_area_index"].dims
        np.testing.assert_allclose(
            restored.where(observed).values,
            predicted["modis_leaf_area_index"].where(observed).values,
        )


class TestFailureMaskIsMatchedByLabel:
    def test_a_gap_at_the_only_observed_site_of_a_larger_stack_is_refused(self, stack, times):
        values = located(xr.DataArray(
            [[1.0, 2.0]], dims=("site", "time"), coords={"site": [2], "time": times[:2]},
            attrs={"units": "g m-2", "constituent": "C"},
        ))

        @dataclass(frozen=True)
        class Gappy:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names_read = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameter_fields=None):
                picked = restrict_to_observed_sites(model_output["wood_carbon"], observed_values)
                out = select_timestep_at(picked, observed_values["time"])
                out[..., -1] = np.nan
                return out

        with pytest.raises(ValueError, match="although the run succeeded"):
            ObservationVector([ObservationSource(observation_source_name="wood", observed_values=values, operator=Gappy())]).predict(stack)


class TestObservationSourceNamesAndUnits:
    def test_an_observation_source_name_that_is_not_a_string_is_a_type_error(self, lai):
        with pytest.raises(TypeError, match="observation_source_name must be a string"):
            ObservationSource(observation_source_name=3, observed_values=lai, operator=SelectTimestep("wood_carbon"))

    def test_an_empty_observation_source_name_is_a_value_error(self, lai):
        with pytest.raises(ValueError, match="observation_source_name is empty"):
            ObservationSource(observation_source_name="", observed_values=lai, operator=SelectTimestep("wood_carbon"))

    def test_a_substance_inside_the_units_is_refused_in_pysipnets_words(self, lai):
        with pytest.raises(ValueError, match="substance token"):
            ObservationSource(observation_source_name="x", observed_values=lai.assign_attrs(units="g C m-2"), operator=SelectTimestep("wood_carbon"))


class TestFieldsNeverLetAnObservationSourceCoordinateTakeTheBatchDim:
    def test_a_scalar_batch_label_on_an_observation_source_gives_way_to_the_batch_dim(self, soil, lai):
        """Before, the observation source's scalar ``sample=4`` overwrote the created batch
        coordinate, and flat then failed with a raw xarray error."""
        labeled = soil.assign_coords(sample=np.int64(4))
        vector = ObservationVector([
            ObservationSource(observation_source_name="soilgrids_soil_organic_carbon", observed_values=labeled, operator=ReduceOverRun("soil_carbon", "mean")),
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ])
        batched_flat = np.arange(3 * vector.dimension, dtype=float).reshape(3, -1)
        fields = vector.fields(batched_flat)
        soil_field = fields["soilgrids_soil_organic_carbon"]
        assert soil_field.dims == ("sample", "site")
        assert soil_field["sample"].values.tolist() == [0, 1, 2]
        np.testing.assert_array_equal(vector.flat(fields), batched_flat)

    @pytest.mark.parametrize("name", ["ameriflux_site_id", "modis_leaf_area_index"])
    def test_a_batch_dim_named_like_an_observation_coordinate_is_refused(self, lai, times, name):
        windowed = lai.assign_coords(
            window_start=("time", times - pd.Timedelta("1D")),
            ameriflux_site_id=("site", ["US-A", "US-B"]),
        )
        vector = ObservationVector([
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=windowed, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ])
        batched_flat = np.zeros((2, vector.dimension))
        with pytest.raises(ValueError, match=f"batch_dim={name!r} is"):
            vector.fields(batched_flat, batch_dim=name)

    def test_a_reserved_name_is_refused(self, vector):
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            vector.fields(np.zeros((2, vector.dimension)), batch_dim="source_index")

    @pytest.mark.parametrize(
        "name", ["window_start", "window_end", "time_step_length", "year"]
    )
    def test_a_model_output_or_window_coordinate_name_is_refused(self, vector, name):
        """Fields on ``time_step_length`` were made, and validate_field refused them."""
        with pytest.raises(ValueError, match="cannot name a batch dim; it is a coordinate"):
            vector.fields(np.zeros((2, vector.dimension)), batch_dim=name)

    @pytest.mark.parametrize("name", ["driver_member", "initial_condition_member"])
    def test_a_data_source_member_name_is_refused(self, vector, name):
        """Theta's rows named ``driver_member`` were stamped as driver members."""
        with pytest.raises(ValueError, match="a data source's member dim"):
            vector.fields(np.zeros((2, vector.dimension)), batch_dim=name)

    def test_an_observation_sources_scalar_batch_labels_are_not_carried(self, soil, lai):
        """An observation source's ``sample=4`` rode along and contradicted the rows."""
        from sipnet_calibration.fields import scalar_batch_labels

        labeled = soil.assign_coords(sample=np.int64(4), driver_member=np.int64(2))
        vector = ObservationVector([
            ObservationSource(observation_source_name="soilgrids_soil_organic_carbon", observed_values=labeled, operator=ReduceOverRun("soil_carbon", "mean")),
            ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai, operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        ])
        batched_flat = np.zeros((3, vector.dimension))
        for batch_dim in ("run", "sample"):
            made = vector.fields(batched_flat, batch_dim=batch_dim)["soilgrids_soil_organic_carbon"]
            assert scalar_batch_labels(made) == ()
            assert "driver_member" not in made.coords
        one = vector.fields(batched_flat[0])["soilgrids_soil_organic_carbon"]
        assert scalar_batch_labels(one) == ()

    def test_the_labels_from_recipe_runs_with_an_observation_sources_scalar_label(self, stack, times):
        """The documented recipe crashed on an observation source carrying ``sample=4``."""
        from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

        locations = {"lon": ("site", stack["lon"].values), "lat": ("site", stack["lat"].values)}
        wood = located(xr.DataArray(
            [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
            dims=("site", "time"),
            coords={"site": np.asarray([1, 2], np.int32), "time": times, **locations},
            attrs={"units": "g m-2", "constituent": "C"},
            name="wood",
        )).assign_coords(sample=np.int64(4))
        vector = ObservationVector([ObservationSource(observation_source_name="wood", observed_values=wood, operator=SelectTimestep("wood_carbon"))])
        crossed = stack.expand_dims(driver_member=[0, 3], axis=1)
        predicted = vector.predict(crossed)
        stacked = {name: stack_batch_dims(field, into="run") for name, field in predicted.items()}
        made = vector.fields(vector.flat(stacked), batch_dim="run")
        restored = unstack_batch_dims(made["wood"], labels_from=stacked["wood"])
        assert restored.dims == ("sample", "driver_member", "site", "time")
        observed = wood.notnull()
        np.testing.assert_allclose(
            restored.where(observed).values, predicted["wood"].where(observed).values
        )

    def test_the_fields_carry_the_attributes_of_their_batch_dim(self, vector):
        from sipnet_calibration.conventions import SAMPLE_ATTRIBUTES

        for field in vector.fields(np.zeros((2, vector.dimension))).values():
            assert dict(field["sample"].attrs) == dict(SAMPLE_ATTRIBUTES)
        for field in vector.fields(np.zeros((2, vector.dimension)), batch_dim="draw").values():
            assert dict(field["draw"].attrs) == {}


class TestFlatRefusesADimThatIsNotABatchDim:
    def test_a_dropped_or_float_batch_coordinate_is_refused_in_the_fields_words(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        fields = vector.fields(batched_flat)
        dropped = {name: field.drop_vars("sample") for name, field in fields.items()}
        with pytest.raises(ValueError, match="carry no coordinate"):
            vector.flat(dropped)
        floats = {name: field.assign_coords(sample=[0.0, 1.0]) for name, field in fields.items()}
        with pytest.raises(ValueError, match="neither a batch dim"):
            vector.flat(floats)

    def test_several_batch_dims_are_refused_with_advice_for_a_dict(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        crossed = {
            name: field.expand_dims(driver_member=[0, 1])
            for name, field in vector.fields(batched_flat).items()
        }
        with pytest.raises(ValueError, match="for a\\s+dict, one call per entry"):
            vector.flat(crossed)

    def test_the_advice_names_the_field_in_the_singular(self, vector):
        batched_flat = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        crossed = {
            name: field.expand_dims(driver_member=[0, 1])
            for name, field in vector.fields(batched_flat).items()
        }
        with pytest.raises(ValueError, match="'modis_leaf_area_index' carries the batch dims"):
            vector.flat(crossed)


class TestPredictChecksAOneSampleSIPNETParameterFieldsAgainstAStack:
    def test_sipnet_parameter_fields_for_one_sample_is_refused_against_a_stack(self, vector, stack, sipnet_parameter_fields):
        """Sample 3's parameter was applied, silently, to the stacked runs of others."""
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack.expand_dims(driver_member=[0, 1], axis=1)
        stacked = crossed.map(lambda variable: stack_batch_dims(variable, into="run"))
        three = xr.concat([sipnet_parameter_fields, sipnet_parameter_fields.isel(sample=[0]).assign_coords(sample=[3])], "sample")
        with pytest.raises(ValueError, match="for sample 3 alone"):
            vector.predict(stacked, sipnet_parameter_fields=three.sel(sample=3))

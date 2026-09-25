"""The observation vector: its index, its two representations, and ``predict``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet import niwot_reference_output

from sipnet_calibration.fields import label_run
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    INDEX_LEVELS,
    ComputeLeafAreaIndex,
    Observation,
    ObservationVector,
    ReduceOverRun,
    SelectTimestep,
    select_observed_sites,
    select_timestep_at,
)

VARIABLES = ["leaf_carbon", "wood_carbon", "soil_carbon"]


def _site_table(*sites):
    return pd.DataFrame({"site_id": list(sites), "lon": [-105.0] * len(sites), "lat": [40.0] * len(sites)}).set_index("site_id", drop=False)


@pytest.fixture(scope="module")
def one_run():
    return label_run(niwot_reference_output().select(VARIABLES), site=1, member=0, site_table=_site_table(1))


@pytest.fixture(scope="module")
def stack(one_run):
    base = one_run.drop_vars(["site", "lon", "lat", "member"])
    sites = xr.concat([base.assign_coords(site=1), (base * 1.5).assign_coords(site=2)], dim="site")
    members = xr.concat([sites.assign_coords(member=0), (sites * 0.5).assign_coords(member=1)], dim="member")
    for name in VARIABLES:
        members[name].attrs = one_run[name].attrs
    return members


@pytest.fixture(scope="module")
def times(one_run):
    return pd.DatetimeIndex(one_run["time"].values[[5, 20, 40]])


@pytest.fixture
def lai(times):
    values = np.array([[3.0, np.nan, 2.5], [np.nan, 1.0, 2.0]])
    return xr.DataArray(values, dims=("site", "time"), coords={"site": [1, 2], "time": times}, attrs={"units": "m2 m-2"}, name="modis_leaf_area_index")


@pytest.fixture
def wood(times):
    values = np.array([[100.0, 110.0, np.nan], [np.nan, np.nan, 120.0]])
    return xr.DataArray(values, dims=("site", "time"), coords={"site": [1, 2], "time": times}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass")


@pytest.fixture
def soil():
    return xr.DataArray([5.0, np.nan], dims="site", coords={"site": [1, 2]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")


@pytest.fixture
def table():
    return xr.Dataset({"leaf_carbon_per_area": (("member", "site"), [[270.0, 135.0], [540.0, 270.0]])}, coords={"member": [0, 1], "site": [1, 2]})


@pytest.fixture
def vector(lai, wood, soil):
    return ObservationVector([
        Observation("modis_leaf_area_index", lai, DEFAULT_OBS_OPS["modis_leaf_area_index"]),
        Observation("landtrendr_aboveground_biomass", wood, SelectTimestep("wood_carbon")),
        Observation("soilgrids_soil_organic_carbon", soil, ReduceOverRun("soil_carbon", "mean")),
    ])


class TestObservation:
    def test_refuses_a_member_dimension(self, lai):
        with pytest.raises(ValueError, match="member dimension"):
            Observation("x", lai.expand_dims(member=[0]), SelectTimestep("wood_carbon"))

    def test_refuses_an_array_without_units(self, lai):
        bare = lai.copy()
        bare.attrs = {}
        with pytest.raises(ValueError, match="no 'units'"):
            Observation("x", bare, SelectTimestep("wood_carbon"))

    def test_refuses_an_operator_without_declarations(self, lai):
        with pytest.raises(ValueError, match="must declare output_variable_names"):
            Observation("x", lai, lambda *a, **k: None)

    def test_orders_sites_and_times(self, lai):
        shuffled = lai.isel(site=[1, 0], time=[2, 0, 1])
        observation = Observation("x", shuffled, SelectTimestep("wood_carbon"))
        assert observation.sites == (1, 2)
        assert observation.values.indexes["time"].is_monotonic_increasing

    def test_cells_are_the_observed_ones(self, lai):
        cells = Observation("x", lai, SelectTimestep("wood_carbon")).cells()
        assert len(cells) == 4
        assert cells["site"].tolist() == [1, 1, 2, 2]


class TestIndex:
    def test_is_site_major_then_product_then_time(self, vector, times):
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
        assert vector.positions(product_name="soilgrids_soil_organic_carbon").tolist() == [4]
        assert vector.positions(site=1, product_name="landtrendr_aboveground_biomass").tolist() == [2, 3]

    def test_declared_reads_are_the_union(self, vector):
        assert vector.output_variable_names == ("leaf_carbon", "wood_carbon", "soil_carbon")
        assert vector.sipnet_parameter_names == ("leaf_carbon_per_area",)

    def test_y_is_a_copy(self, vector):
        y = vector.y
        y[0] = -1.0
        assert vector.y[0] == 3.0

    def test_refuses_duplicate_products(self, lai):
        with pytest.raises(ValueError, match="unique"):
            ObservationVector([Observation("x", lai, SelectTimestep("wood_carbon"))] * 2)

    def test_refuses_a_product_with_no_cells(self, lai):
        empty = lai.where(False)
        with pytest.raises(ValueError, match="no observed cell"):
            ObservationVector([Observation("x", empty, SelectTimestep("wood_carbon"))])

    def test_describe(self, vector):
        table = vector.describe()
        assert table.loc["modis_leaf_area_index", "cells"] == 4
        assert pd.isna(table.loc["soilgrids_soil_organic_carbon", "first_time"])


class TestSelect:
    def test_by_sites(self, vector):
        sub = vector.select(sites=[2])
        assert sub.sites == (2,)
        assert sub.product_names == ("modis_leaf_area_index", "landtrendr_aboveground_biomass")  # soil has no cell at site 2
        assert sub.y.tolist() == [1.0, 2.0, 120.0]

    def test_by_products(self, vector):
        sub = vector.select(product_names=["landtrendr_aboveground_biomass"])
        assert sub.product_names == ("landtrendr_aboveground_biomass",)
        assert sub.dimension == 3

    def test_by_time(self, vector, times):
        sub = vector.select(time=slice(times[0], times[1]))
        assert sub.dimension == 2 + 2 + 1  # lai at t0 (site 1) and t1 (site 2); wood at t0, t1; soil static
        assert "soilgrids_soil_organic_carbon" in sub.product_names

    def test_an_empty_selection_is_refused(self, vector):
        with pytest.raises(ValueError, match="no observed cell"):
            vector.select(sites=[99])


class TestRepresentations:
    def test_fields_of_y_are_the_observed_values(self, vector, lai, wood, soil):
        fields = vector.fields(vector.y)
        for name, original in [("modis_leaf_area_index", lai), ("landtrendr_aboveground_biomass", wood), ("soilgrids_soil_organic_carbon", soil)]:
            np.testing.assert_array_equal(fields[name].values, original.values)
            assert fields[name].attrs["units"] == original.attrs["units"]

    def test_flat_of_the_fields_is_y(self, vector):
        np.testing.assert_array_equal(vector.flat(vector.fields(vector.y)), vector.y)

    def test_a_block_round_trips_with_a_member_dimension(self, vector):
        block = np.arange(3 * vector.dimension, dtype=float).reshape(3, -1)
        fields = vector.fields(block)
        assert fields["modis_leaf_area_index"].dims == ("member", "site", "time")
        np.testing.assert_array_equal(vector.flat(fields), block)

    def test_flat_accepts_a_larger_array(self, vector, lai):
        bigger = xr.full_like(lai.reindex(site=[1, 2, 3]), 7.0)
        bigger.attrs = lai.attrs
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"] = bigger
        flat = vector.flat(fields)
        assert flat[vector.positions(product_name="modis_leaf_area_index")].tolist() == [7.0] * 4

    def test_flat_refuses_a_missing_cell(self, vector, lai):
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
    def test_a_stack_predicts_every_product_in_the_observations_units(self, vector, stack, table, times):
        predicted = vector.predict(stack, sipnet_parameters=table)
        assert set(predicted) == set(vector.product_names)
        assert predicted["landtrendr_aboveground_biomass"].attrs["units"] == "Mg ha-1"
        wood_g = select_timestep_at(stack["wood_carbon"], times)
        np.testing.assert_allclose(predicted["landtrendr_aboveground_biomass"].values, wood_g.values * 0.01)
        assert predicted["soilgrids_soil_organic_carbon"].dims == ("member", "site")

    def test_flat_of_the_predictions_is_member_by_cell(self, vector, stack, table):
        block = vector.flat(vector.predict(stack, sipnet_parameters=table))
        assert block.shape == (2, vector.dimension)
        assert np.isfinite(block).all()
        # the second member's pools are half the first's, and its leaf carbon per area double
        wood = np.concatenate([vector.positions(product_name=n) for n in ("landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon")])
        np.testing.assert_allclose(block[1, wood], 0.5 * block[0, wood])
        lai = vector.positions(product_name="modis_leaf_area_index")
        np.testing.assert_allclose(block[1, lai], 0.25 * block[0, lai])

    def test_one_run_predicts_a_one_site_vector(self, vector, one_run):
        sub = vector.select(sites=[1])
        predicted = sub.predict(one_run, sipnet_parameters={"leaf_carbon_per_area": 270.0})
        flat = sub.flat(predicted)
        assert flat.shape == (sub.dimension,)

    def test_a_failed_run_passes_through_as_nan(self, vector, stack, table):
        failed = stack.copy(deep=True)
        for name in VARIABLES:
            failed[name].loc[{"member": 1, "site": 2}] = np.nan
        block = vector.flat(vector.predict(failed, sipnet_parameters=table))
        assert np.isnan(block[1, vector.positions(site=2)]).all()
        assert np.isfinite(block[0]).all()
        assert np.isfinite(block[1, vector.positions(site=1)]).all()

    def test_a_gap_the_operator_produced_is_refused(self, lai, stack, table):
        @dataclass(frozen=True)
        class Gappy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                out = select_timestep_at(model_output["leaf_carbon"].sel(site=observed_values["site"].values), observed_values["time"])
                out[..., 0] = np.nan
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([Observation("modis_leaf_area_index", lai, Gappy())])
        with pytest.raises(ValueError, match="although the run succeeded"):
            vector.predict(stack)

    def test_a_wrong_dimension_is_refused_naming_both_units(self, lai, stack):
        vector = ObservationVector([Observation("modis_leaf_area_index", lai, SelectTimestep("leaf_carbon"))])
        with pytest.raises(ValueError, match="m2 m-2"):
            vector.predict(stack)

    def test_a_missing_parameter_is_refused(self, vector, stack):
        with pytest.raises(ValueError, match="pass sipnet_parameters"):
            vector.predict(stack)

    def test_a_model_output_lacking_a_variable_is_refused(self, vector, stack, table):
        with pytest.raises(ValueError, match="lacks"):
            vector.predict(stack.drop_vars("soil_carbon"), sipnet_parameters=table)

    def test_a_mapping_is_refused(self, vector, stack):
        with pytest.raises(TypeError, match="Dataset"):
            vector.predict({name: stack[name] for name in VARIABLES})


@pytest.fixture(scope="module")
def observed():
    """Three real constraint products at three sites, where the processed files exist."""
    constraints = pytest.importorskip("sipnet_calibration.constraints")
    names = ["modis_leaf_area_index", "landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon"]
    try:
        return constraints.constraint_fields(names, sites=[3851, 3871, 3875])
    except FileNotFoundError as error:
        pytest.skip(str(error))


class TestRealProducts:
    def test_the_vector_round_trips_the_ragged_products(self, observed):
        vector = ObservationVector([
            Observation("modis_leaf_area_index", observed["modis_leaf_area_index"], DEFAULT_OBS_OPS["modis_leaf_area_index"]),
            Observation("landtrendr_aboveground_biomass", observed["landtrendr_aboveground_biomass"], SelectTimestep("wood_carbon")),
            Observation("soilgrids_soil_organic_carbon", observed["soilgrids_soil_organic_carbon"], ReduceOverRun("soil_carbon", "mean")),
        ])
        assert vector.dimension == sum(int(a.notnull().sum()) for a in observed.values())
        assert np.isfinite(vector.y).all()
        fields = vector.fields(vector.y)
        for name, array in observed.items():
            np.testing.assert_array_equal(fields[name].values, array.values)
        np.testing.assert_array_equal(vector.flat(fields), vector.y)
        sites = vector.index.get_level_values("site").values
        assert (np.diff(sites) >= 0).all()


class TestTwinObservations:
    """Observations built from the model output are reproduced exactly."""

    def test_the_state_at_the_labels_in_the_observations_units(self, one_run, times):
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        at = ends.get_indexer(times)
        observed = xr.DataArray(
            wood.values[at][None, :] * 0.01,  # g m-2 -> Mg ha-1
            dims=("site", "time"), coords={"site": [1], "time": times},
            attrs={"units": "Mg ha-1", "constituent": "C"}, name="landtrendr_aboveground_biomass",
        )
        vector = ObservationVector([Observation("landtrendr_aboveground_biomass", observed, SelectTimestep("wood_carbon"))])
        predicted = vector.flat(vector.predict(one_run))
        np.testing.assert_allclose(predicted, vector.y, rtol=1e-12)

    def test_leaf_area_index_from_leaf_carbon_and_the_parameter(self, one_run, times):
        leaf = one_run["leaf_carbon"]
        at = pd.DatetimeIndex(leaf["time"].values).get_indexer(times)
        observed = xr.DataArray(
            (leaf.values[at] / 270.0)[None, :], dims=("site", "time"), coords={"site": [1], "time": times},
            attrs={"units": "m2 m-2"}, name="modis_leaf_area_index",
        )
        vector = ObservationVector([Observation("modis_leaf_area_index", observed, DEFAULT_OBS_OPS["modis_leaf_area_index"])])
        predicted = vector.flat(vector.predict(one_run, sipnet_parameters={"leaf_carbon_per_area": 270.0}))
        np.testing.assert_allclose(predicted, vector.y, rtol=1e-12)
        wrong = vector.flat(vector.predict(one_run, sipnet_parameters={"leaf_carbon_per_area": 135.0}))
        np.testing.assert_allclose(wrong, 2 * vector.y, rtol=1e-12)


class TestMoreRefusals:
    def test_an_operator_on_the_wrong_time_labels_is_refused_by_predict(self, lai, stack, table):
        @dataclass(frozen=True)
        class OffGrid:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                out = select_timestep_at(model_output["leaf_carbon"].sel(site=observed_values["site"].values), observed_values["time"])
                out = out.assign_coords(time=out["time"].values + np.timedelta64(1, "h"))
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([Observation("modis_leaf_area_index", lai, OffGrid())])
        with pytest.raises(ValueError, match="time labels"):
            vector.predict(stack)

    def test_an_infinite_observation_is_refused(self, lai):
        lai[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            Observation("x", lai, SelectTimestep("wood_carbon"))

    def test_a_nat_label_is_refused(self, lai):
        broken = lai.assign_coords(time=[lai["time"].values[0], np.datetime64("NaT"), lai["time"].values[2]])
        with pytest.raises(ValueError, match="NaT"):
            Observation("x", broken, SelectTimestep("wood_carbon"))

    def test_float_site_labels_are_refused(self, lai):
        with pytest.raises(ValueError, match="site labels must be integers"):
            Observation("x", lai.assign_coords(site=[1.0, 2.5]), SelectTimestep("wood_carbon"))

    def test_an_aware_time_coordinate_is_refused(self, lai):
        aware = lai.assign_coords(time=pd.DatetimeIndex(lai["time"].values).tz_localize("UTC"))
        with pytest.raises(ValueError, match="naive datetime64"):
            Observation("x", aware, SelectTimestep("wood_carbon"))

    def test_mixed_member_and_no_member_fields_are_refused(self, vector):
        fields = vector.fields(np.zeros((2, vector.dimension)))
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].isel(member=0, drop=True)
        with pytest.raises(ValueError, match="member dimension"):
            vector.flat(fields)

    def test_fields_label_members_from_zero(self, vector):
        fields = vector.fields(np.zeros((3, vector.dimension)))
        assert fields["modis_leaf_area_index"]["member"].values.tolist() == [0, 1, 2]
        assert fields["modis_leaf_area_index"]["member"].dtype == np.int16

    def test_too_many_members_are_refused(self, vector):
        with pytest.raises(ValueError, match="at most"):
            vector.fields(np.zeros((40000, vector.dimension)))

    def test_select_accepts_one_name_and_one_site(self, vector):
        assert vector.select(product_names="landtrendr_aboveground_biomass").dimension == 3
        assert vector.select(sites=2).sites == (2,)
        with pytest.raises(TypeError, match="slice"):
            vector.select(time="2012")

    def test_observation_values_are_named_for_the_product(self, lai):
        assert Observation("x", lai.rename(None), SelectTimestep("wood_carbon")).values.name == "x"


class TestSelectKeepsOnlyObservedLabels:
    """A sub-vector's operators read the model only where a kept site is observed."""

    def test_a_one_site_selection_drops_the_other_sites_labels(self, vector, lai, times):
        sub = vector.select(sites=[1])
        # lai is observed at site 1 on the first and last label only
        kept = sub["modis_leaf_area_index"].values
        assert kept["time"].values.tolist() == [times[0].value, times[2].value]
        np.testing.assert_array_equal(sub.y, vector.y[vector.positions(site=1)])
        assert sub.index.equals(vector.index[vector.positions(site=1)])

    def test_a_one_site_slice_predicts_from_that_sites_shorter_run(self, one_run, times):
        wood = xr.DataArray(
            [[100.0, np.nan, np.nan], [np.nan, np.nan, 120.0]], dims=("site", "time"),
            coords={"site": [1, 2], "time": times}, attrs={"units": "Mg ha-1", "constituent": "C"},
        )
        vector = ObservationVector([Observation("wood", wood, SelectTimestep("wood_carbon"))])
        short = one_run.isel(time=slice(0, 30))  # ends before the label only site 2 is observed at
        predicted = vector.select(sites=[1]).predict(short)
        assert predicted["wood"].sizes["time"] == 1

    def test_a_generator_of_sites_is_read_once_for_every_product(self, vector):
        sub = vector.select(sites=(site for site in (1, 2)))
        assert sub.product_names == vector.product_names
        np.testing.assert_array_equal(sub.y, vector.y)

    def test_selecting_thousands_of_sites_keeps_every_chosen_one(self):
        n = 8000
        values = xr.DataArray(
            np.ones((n, 2)), dims=("site", "time"),
            coords={"site": np.arange(1, n + 1), "time": pd.date_range("2000-01-01", periods=2)},
            attrs={"units": "Mg ha-1", "constituent": "C"},
        )
        vector = ObservationVector([Observation("wood", values, SelectTimestep("wood_carbon"))])
        sub = vector.select(sites=range(1, n + 1, 2))
        assert sub.sites == tuple(range(1, n + 1, 2))

    def test_an_unknown_product_is_refused(self, vector):
        with pytest.raises(KeyError, match="no observation of 'nothing'"):
            vector.select(product_names=["nothing"])

    def test_products_come_in_the_requested_order(self, vector):
        names = ("soilgrids_soil_organic_carbon", "modis_leaf_area_index")
        assert vector.select(product_names=list(names)).product_names == names


class TestObservationHoldsItsOwnValues:
    def test_a_write_to_the_callers_array_does_not_reach_the_observation(self, lai):
        observation = Observation("x", lai, SelectTimestep("wood_carbon"))
        lai[1, 0] = 5.0  # an unobserved cell of the caller's array
        assert observation.n_cells == 4
        assert np.isnan(observation.values.values[1, 0])

    def test_the_stored_values_are_read_only(self, vector):
        dimension = vector.dimension
        with pytest.raises(ValueError, match="read-only"):
            vector.observed_values["modis_leaf_area_index"][0, 1] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            vector["modis_leaf_area_index"].values.values[0, 1] = 1.0
        assert vector["modis_leaf_area_index"].n_cells == 4 and vector.dimension == dimension


class TestVectorSites:
    def test_a_site_observed_nowhere_is_not_a_site_of_the_vector(self, soil):
        vector = ObservationVector([Observation("soil", soil, ReduceOverRun("soil_carbon", "mean"))])
        assert soil["site"].values.tolist() == [1, 2]
        assert vector.sites == (1,)

    def test_sites_are_ascending_across_products(self, times):
        first = xr.DataArray([[1.0], [2.0]], dims=("site", "time"), coords={"site": [27, 3], "time": times[:1]}, attrs={"units": "m2 m-2"})
        second = xr.DataArray([4.0, 5.0], dims="site", coords={"site": [8, 1]}, attrs={"units": "Mg ha-1", "constituent": "C"})
        vector = ObservationVector([
            Observation("a", first, SelectTimestep("leaf_carbon")),
            Observation("b", second, ReduceOverRun("soil_carbon", "mean")),
        ])
        assert vector.sites == (1, 3, 8, 27)


class TestObservationRefusals:
    def test_duplicate_sites_are_refused(self, lai):
        with pytest.raises(ValueError, match="site coordinate has duplicates"):
            Observation("x", lai.assign_coords(site=[1, 1]), SelectTimestep("wood_carbon"))

    def test_duplicate_times_are_refused(self, lai):
        repeated = lai.assign_coords(time=[lai["time"].values[0]] * 2 + [lai["time"].values[2]])
        with pytest.raises(ValueError, match="time coordinate has duplicates"):
            Observation("x", repeated, SelectTimestep("wood_carbon"))

    def test_an_operator_that_is_not_callable_is_refused(self, lai):
        with pytest.raises(TypeError, match="must be callable"):
            Observation("x", lai, "wood_carbon")

    def test_an_alias_declared_by_an_operator_is_refused(self, lai):
        @dataclass(frozen=True)
        class Aliased:
            output_variable_names = ("plantWoodC",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return model_output["wood_carbon"]

        with pytest.raises(ValueError, match="alias"):
            Observation("x", lai, Aliased())

    def test_declarations_as_a_list_are_refused(self, lai):
        @dataclass(frozen=True)
        class Listed:
            output_variable_names = ["wood_carbon"]
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return model_output["wood_carbon"]

        with pytest.raises(ValueError, match="tuple of names"):
            Observation("x", lai, Listed())


class TestFlatRefusals:
    def test_a_field_missing_an_observed_site_is_refused(self, vector):
        fields = vector.fields(vector.y)
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].sel(site=[1])
        with pytest.raises(ValueError, match=r"lacks observed site\(s\) \[2\]"):
            vector.flat(fields)

    def test_a_missing_product_is_refused(self, vector):
        fields = vector.fields(vector.y)
        del fields["soilgrids_soil_organic_carbon"]
        with pytest.raises(ValueError, match="lack the product"):
            vector.flat(fields)

    def test_member_labels_that_disagree_are_refused(self, vector):
        fields = vector.fields(np.zeros((2, vector.dimension)))
        fields["soilgrids_soil_organic_carbon"] = fields["soilgrids_soil_organic_carbon"].assign_coords(member=[5, 6])
        with pytest.raises(ValueError, match="disagree on their member labels"):
            vector.flat(fields)

    def test_member_need_not_be_the_leading_dimension(self, vector):
        block = np.arange(2 * vector.dimension, dtype=float).reshape(2, -1)
        fields = vector.fields(block)
        fields["modis_leaf_area_index"] = fields["modis_leaf_area_index"].transpose("site", "time", "member")
        np.testing.assert_array_equal(vector.flat(fields), block)

    def test_labels_in_seconds_are_read_by_instant(self, lai, one_run):
        coarse = lai.assign_coords(time=lai["time"].values.astype("datetime64[s]"))
        vector = ObservationVector([Observation("modis_leaf_area_index", coarse, SelectTimestep("leaf_carbon"))])
        assert vector["modis_leaf_area_index"].values["time"].dtype == np.dtype("datetime64[s]")
        np.testing.assert_array_equal(vector.flat(vector.fields(vector.y)), vector.y)
        wide = xr.full_like(lai, 7.0)  # nanosecond labels, as a prediction carries them
        fields = {"modis_leaf_area_index": wide}
        assert vector.flat(fields).tolist() == [7.0] * vector.dimension


class TestFailedRuns:
    def test_a_run_with_only_some_missing_steps_has_not_failed(self, lai, stack):
        padded = stack.copy(deep=True)
        tail = padded["time"].values[-10:]
        padded["leaf_carbon"].loc[{"member": 1, "site": 2, "time": tail}] = np.nan

        @dataclass(frozen=True)
        class Gappy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                leaf = select_observed_sites(model_output["leaf_carbon"], observed_values)
                out = select_timestep_at(leaf, observed_values["time"])
                out.loc[{"member": 1, "site": 2, "time": out["time"].values[-1]}] = np.nan
                out.attrs = {"units": "1"}
                return out

        with pytest.raises(ValueError, match="although the run succeeded"):
            ObservationVector([Observation("modis_leaf_area_index", lai, Gappy())]).predict(padded)

    def test_one_read_variable_missing_throughout_is_a_failed_run(self, vector, stack, table):
        failed = stack.copy(deep=True)
        failed["wood_carbon"].loc[{"member": 1, "site": 2}] = np.nan  # leaf and soil carbon finite
        block = vector.flat(vector.predict(failed, sipnet_parameters=table))
        wood = vector.positions(site=2, product_name="landtrendr_aboveground_biomass")
        assert np.isnan(block[1, wood]).all()
        assert np.isfinite(block[0]).all()

    def test_the_failure_mask_is_matched_by_site_label(self, lai, stack, table):
        reordered = stack.isel(site=[1, 0]).copy(deep=True)
        reordered["leaf_carbon"].loc[{"member": 1, "site": 2}] = np.nan
        vector = ObservationVector([Observation("modis_leaf_area_index", lai, ComputeLeafAreaIndex())])
        block = vector.flat(vector.predict(reordered, sipnet_parameters=table))
        assert np.isnan(block[1, vector.positions(site=2)]).all()
        assert np.isfinite(block[1, vector.positions(site=1)]).all()


class TestPredictSharesTheOperatorChecks:
    def test_a_result_with_a_spurious_member_is_refused(self, lai, one_run):
        @dataclass(frozen=True)
        class Spurious:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                out = select_timestep_at(model_output["leaf_carbon"], observed_values["time"])
                out = out.drop_vars("member").expand_dims(member=[0, 1])
                out.attrs = {"units": "1"}
                return out

        vector = ObservationVector([Observation("modis_leaf_area_index", lai, Spurious())]).select(sites=[1])
        with pytest.raises(ValueError, match="member dimension the model output lacks"):
            vector.predict(one_run)

    def test_a_result_that_is_not_an_array_is_a_type_error(self, lai, stack):
        @dataclass(frozen=True)
        class Numpy:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return np.zeros(3)

        with pytest.raises(TypeError, match="not a DataArray"):
            ObservationVector([Observation("modis_leaf_area_index", lai, Numpy())]).predict(stack)

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
    Observation,
    ObservationVector,
    ReduceOverRun,
    SelectTimestep,
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


class TestRealProducts:
    @pytest.fixture(scope="class")
    def observed(self):
        constraints = pytest.importorskip("sipnet_calibration.constraints")
        names = ["modis_leaf_area_index", "landtrendr_aboveground_biomass", "soilgrids_soil_organic_carbon"]
        try:
            return constraints.constraint_fields(names, sites=[3851, 3871, 3875])
        except FileNotFoundError as error:
            pytest.skip(str(error))

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

"""The shipped observation operators and the operator contract."""

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
    ComputeLeafAreaIndex,
    ObservationOperator,
    ReduceOverRun,
    ReduceOverTimeBounds,
    SelectTimestep,
    check_operator,
    select_observed_sites,
    select_timestep_at,
    select_sipnet_parameter,
    sipnet_parameter_spec,
)
from sipnet_calibration.observation.alignment import TIME_BOUNDS_END, TIME_BOUNDS_START

VARIABLES = ["leaf_carbon", "wood_carbon", "soil_carbon", "net_ecosystem_exchange", "soil_wetness_fraction"]


def _site_table(*sites):
    return pd.DataFrame({"site_id": list(sites), "lon": [-105.0] * len(sites), "lat": [40.0] * len(sites)}).set_index("site_id", drop=False)


@pytest.fixture(scope="module")
def one_run():
    """One Niwot run labeled as site 1, member 0."""
    return label_run(niwot_reference_output().select(VARIABLES), site=1, member=0, site_table=_site_table(1))


@pytest.fixture(scope="module")
def stack(one_run):
    """Two sites and two members built from the Niwot run by known factors."""
    base = one_run.drop_vars(["site", "lon", "lat", "member"])
    sites = xr.concat([base.assign_coords(site=1), (base * 1.5).assign_coords(site=2)], dim="site")
    members = xr.concat([sites.assign_coords(member=0), (sites * 0.5).assign_coords(member=1)], dim="member")
    for name in VARIABLES:
        members[name].attrs = one_run[name].attrs
    return members.assign_coords(lon=("site", [-105.0, -104.0]), lat=("site", [40.0, 41.0]))


@pytest.fixture
def labels(one_run):
    return pd.DatetimeIndex(one_run["time"].values[[5, 20, 40]])


def _observed(sites, times, units="m2 m-2", constituent="", name="modis_leaf_area_index", bounds=False):
    values = np.ones((len(sites), len(times)))
    coords = {"site": list(sites), "time": pd.DatetimeIndex(times)}
    if bounds:
        coords[TIME_BOUNDS_START] = ("time", pd.DatetimeIndex(times) - pd.Timedelta("1D"))
        coords[TIME_BOUNDS_END] = ("time", pd.DatetimeIndex(times))
    attrs = {"units": units}
    if constituent:
        attrs["constituent"] = constituent
    return xr.DataArray(values, dims=("site", "time"), coords=coords, attrs=attrs, name=name)


class TestSelectTimestep:
    def test_reads_the_state_at_the_observed_labels(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass")
        predicted = SelectTimestep("wood_carbon")(one_run, observed)
        expected = select_timestep_at(one_run["wood_carbon"], labels)
        np.testing.assert_array_equal(predicted.values, expected.values)
        assert predicted.attrs["units"] == "g m-2" and predicted.attrs["constituent"] == "C"

    def test_resolves_an_alias_to_the_registry_name(self):
        assert SelectTimestep("nee").output_variable_names == ("net_ecosystem_exchange",)

    def test_refuses_an_unknown_variable(self):
        with pytest.raises(ValueError, match="not a pySIPNET output variable"):
            SelectTimestep("leaf_area_index")

    def test_declares_no_parameters(self):
        assert SelectTimestep("wood_carbon").sipnet_parameter_names == ()

    def test_satisfies_the_protocol(self):
        assert isinstance(SelectTimestep("wood_carbon"), ObservationOperator)

    def test_a_stack_is_read_pointwise(self, stack, labels):
        observed = _observed([1, 2], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass")
        predicted = check_operator(SelectTimestep("wood_carbon"), stack, observed)
        assert predicted.dims == ("member", "site", "time")
        np.testing.assert_allclose(predicted.sel(site=2, member=1).values, 0.75 * predicted.sel(site=1, member=0).values)


class TestReduceOverTimeBounds:
    def test_reduces_over_each_observations_own_bounds(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass", bounds=True)
        predicted = ReduceOverTimeBounds("wood_carbon", "mean")(one_run, observed)
        np.testing.assert_array_equal(predicted["time"].values, observed["time"].values)
        assert predicted.attrs["kind"] == "timestep_mean"
        assert np.isfinite(predicted.values).all()

    def test_refuses_an_observation_without_bounds(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C")
        with pytest.raises(ValueError, match="documents no interval"):
            ReduceOverTimeBounds("wood_carbon", "mean")(one_run, observed)

    def test_refuses_an_unknown_reduction(self):
        with pytest.raises(ValueError, match="how must be one of"):
            ReduceOverTimeBounds("wood_carbon", "median")


class TestReduceOverRun:
    def test_reduces_over_the_whole_record_for_a_static_observation(self, one_run):
        observed = xr.DataArray([1.0], dims="site", coords={"site": [1]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")
        predicted = ReduceOverRun("soil_carbon", "mean")(one_run, observed)
        assert "time" not in predicted.dims
        np.testing.assert_allclose(float(predicted), float(one_run["soil_carbon"].weighted(one_run["time_step_length"].astype("int64")).mean()), rtol=1e-12)

    def test_refuses_a_dated_observation(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C")
        with pytest.raises(ValueError, match="static observation"):
            ReduceOverRun("soil_carbon", "mean")(one_run, observed)

    def test_pointwise_over_a_stack(self, stack):
        observed = xr.DataArray([1.0, 1.0], dims="site", coords={"site": [1, 2]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")
        predicted = check_operator(ReduceOverRun("soil_carbon", "last"), stack, observed)
        assert predicted.dims == ("member", "site")


class TestComputeLeafAreaIndex:
    def test_is_the_default_for_modis(self):
        assert set(DEFAULT_OBS_OPS) == {"modis_leaf_area_index"}
        assert isinstance(DEFAULT_OBS_OPS["modis_leaf_area_index"], ComputeLeafAreaIndex)

    def test_declares_its_reads(self):
        op = ComputeLeafAreaIndex()
        assert op.output_variable_names == ("leaf_carbon",)
        assert op.sipnet_parameter_names == ("leaf_carbon_per_area",)

    def test_leaf_carbon_over_the_parameter_from_a_mapping(self, one_run, labels):
        observed = _observed([1], labels)
        predicted = ComputeLeafAreaIndex()(one_run, observed, sipnet_parameters={"leaf_carbon_per_area": 270.0})
        expected = select_timestep_at(one_run["leaf_carbon"], labels).values / 270.0
        np.testing.assert_allclose(predicted.values, expected)
        assert predicted.attrs["units"] == "1"
        assert "constituent" not in predicted.attrs

    def test_the_parameter_varies_by_member_and_site(self, stack, labels):
        observed = _observed([1, 2], labels)
        table = xr.Dataset({"leaf_carbon_per_area": (("member", "site"), [[270.0, 135.0], [540.0, 270.0]])}, coords={"member": [0, 1], "site": [1, 2]})
        predicted = check_operator(ComputeLeafAreaIndex(), stack, observed, sipnet_parameters=table)
        leaf = select_timestep_at(stack["leaf_carbon"], labels)
        np.testing.assert_allclose(predicted.sel(member=1, site=2).values, leaf.sel(member=1, site=2).values / 270.0)
        np.testing.assert_allclose(predicted.sel(member=0, site=2).values, leaf.sel(member=0, site=2).values / 135.0)

    def test_the_parameter_is_required(self, one_run, labels):
        with pytest.raises(ValueError, match="pass sipnet_parameters"):
            ComputeLeafAreaIndex()(one_run, _observed([1], labels))

    def test_a_table_lacking_the_parameter_is_refused(self, one_run, labels):
        table = xr.Dataset({"max_photosynthesis_rate": (("site",), [10.0])}, coords={"site": [1]})
        with pytest.raises(ValueError, match="no variable 'leaf_carbon_per_area'"):
            ComputeLeafAreaIndex()(one_run, _observed([1], labels), sipnet_parameters=table)


class TestSelectObservedSites:
    def test_a_missing_site_is_named(self, stack, labels):
        with pytest.raises(ValueError, match=r"no site\(s\) \[3\]"):
            select_observed_sites(stack["wood_carbon"], _observed([1, 3], labels))

    def test_a_single_run_must_be_the_observed_site(self, one_run, labels):
        with pytest.raises(ValueError, match="one run at site 1"):
            select_observed_sites(one_run["wood_carbon"], _observed([2], labels))

    def test_an_unlabeled_run_is_refused(self, labels):
        array = niwot_reference_output().select(["wood_carbon"])["wood_carbon"]
        with pytest.raises(ValueError, match="no 'site' coordinate"):
            select_observed_sites(array, _observed([1], labels))

    def test_selection_follows_the_observations_order(self, stack, labels):
        picked = select_observed_sites(stack["wood_carbon"], _observed([2, 1], labels))
        assert picked["site"].values.tolist() == [2, 1]


class TestSelectSipnetParameter:
    def test_carries_pysipnets_units(self):
        array = select_sipnet_parameter({"leaf_carbon_per_area": 270.0}, "leaf_carbon_per_area", xr.DataArray(0.0))
        assert array.attrs["units"] == "g m-2" and array.attrs["constituent"] == "C"
        assert float(array) == 270.0

    def test_resolves_an_alias(self):
        assert sipnet_parameter_spec("leafCSpWt").sipnet_name == "leafCSpWt"

    def test_selects_the_arrays_sites_and_members_from_a_table(self):
        table = xr.Dataset({"leaf_carbon_per_area": (("member", "site"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}, coords={"member": [0, 1], "site": [1, 2, 3]})
        model_variable = xr.DataArray(np.zeros((2, 2)), dims=("member", "site"), coords={"member": [0, 1], "site": [3, 1]})
        array = select_sipnet_parameter(table, "leaf_carbon_per_area", model_variable)
        np.testing.assert_array_equal(array.values, [[3.0, 1.0], [6.0, 4.0]])


class TestCheckOperator:
    def test_an_alias_in_the_declaration_is_refused(self, one_run, labels):
        @dataclass(frozen=True)
        class Aliased:
            output_variable_names = ("nee",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"])

        with pytest.raises(ValueError, match="alias"):
            check_operator(Aliased(), one_run, _observed([1], labels))

    def test_a_result_off_the_grid_is_refused(self, one_run, labels):
        @dataclass(frozen=True)
        class OffGrid:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"][:-1])

        with pytest.raises(ValueError, match="time labels"):
            check_operator(OffGrid(), one_run, _observed([1], labels))

    def test_a_result_without_units_is_refused(self, one_run, labels):
        @dataclass(frozen=True)
        class Unitless:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                out = select_timestep_at(model_output["wood_carbon"], observed_values["time"])
                out.attrs = {}
                return out

        with pytest.raises(ValueError, match="no 'units'"):
            check_operator(Unitless(), one_run, _observed([1], labels))

    def test_an_operator_that_mixes_sites_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class SiteMean:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(select_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "site" not in picked.dims:
                    return picked
                mixed = picked.mean("site").expand_dims(site=picked["site"].values).transpose(*picked.dims)
                mixed.attrs = picked.attrs
                return mixed

        with pytest.raises(ValueError, match="not pointwise in 'site'"):
            check_operator(SiteMean(), stack, _observed([1, 2], labels))


class TestSiteOrderAndCoverage:
    def test_an_operator_follows_an_observation_out_of_model_order(self, stack, labels):
        observed = _observed([2, 1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass")
        predicted = check_operator(SelectTimestep("wood_carbon"), stack, observed)
        assert predicted["site"].values.tolist() == [2, 1]
        np.testing.assert_allclose(predicted.sel(site=2, member=0).values, 1.5 * predicted.sel(site=1, member=0).values)

    def test_an_operator_returning_model_order_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class ModelOrder:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                sites = sorted(observed_values["site"].values.tolist())
                return select_timestep_at(model_output["wood_carbon"].sel(site=sites), observed_values["time"])

        with pytest.raises(ValueError, match="sites, in order"):
            check_operator(ModelOrder(), stack, _observed([2, 1], labels, units="Mg ha-1", constituent="C"))

    def test_check_operator_handles_a_run_over_more_sites_than_observed(self, stack, labels):
        wider = xr.concat([stack, (stack.isel(site=[1]) * 2).assign_coords(site=[3])], dim="site")
        for name in VARIABLES:
            wider[name].attrs = stack[name].attrs
        observed = _observed([1, 2], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass")
        predicted = check_operator(SelectTimestep("wood_carbon"), wider, observed)
        assert predicted["site"].values.tolist() == [1, 2]

    def test_check_operator_with_a_scalar_site_observation(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C").isel(site=0)
        predicted = check_operator(SelectTimestep("wood_carbon"), one_run, observed)
        assert int(predicted["site"]) == 1


class TestReduceOverTimeBoundsValues:
    def test_the_label_need_not_be_a_bound_edge(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass", bounds=True)
        shifted = observed.assign_coords(time=observed["time"].values - np.timedelta64(12, "h"))
        predicted = ReduceOverTimeBounds("wood_carbon", "mean")(one_run, shifted)
        np.testing.assert_array_equal(predicted["time"].values, shifted["time"].values)
        from sipnet_calibration.observation import reduce_windows, windows_from_time_bounds

        expected = reduce_windows(one_run["wood_carbon"], windows_from_time_bounds(shifted), "mean")
        np.testing.assert_array_equal(predicted.values, expected.values)

    def test_the_mean_over_a_window_is_the_length_weighted_mean(self, one_run, labels):
        observed = _observed([1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass", bounds=True)
        predicted = ReduceOverTimeBounds("wood_carbon", "mean")(one_run, observed)
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        lengths = wood["time_step_length"].values.astype("timedelta64[ns]").astype("float64")
        for k, label in enumerate(labels):
            inside = (ends > label - pd.Timedelta("1D")) & (ends <= label)
            expected = np.average(wood.values[inside], weights=lengths[inside])
            np.testing.assert_allclose(predicted.values[k], expected)


class TestParameterLookups:
    def test_a_table_missing_the_runs_members_is_named(self, stack, labels):
        table = xr.Dataset({"leaf_carbon_per_area": (("member", "site"), [[270.0, 135.0]])}, coords={"member": [0], "site": [1, 2]})
        with pytest.raises(ValueError, match=r"no member label\(s\) \[1\]"):
            ComputeLeafAreaIndex()(stack, _observed([1, 2], labels), sipnet_parameters=table)

    def test_an_alias_key_in_a_mapping_is_found(self, one_run, labels):
        predicted = ComputeLeafAreaIndex()(one_run, _observed([1], labels), sipnet_parameters={"leafCSpWt": 270.0})
        assert np.isfinite(predicted.values).all()

    def test_a_non_numeric_mapping_value_is_refused(self, one_run, labels):
        with pytest.raises(ValueError, match="must be a number"):
            ComputeLeafAreaIndex()(one_run, _observed([1], labels), sipnet_parameters={"leaf_carbon_per_area": "270"})

    def test_repeated_observed_sites_are_refused(self, stack, labels):
        with pytest.raises(ValueError, match="repeats a site"):
            select_observed_sites(stack["wood_carbon"], _observed([1, 1], labels))

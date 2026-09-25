"""The shipped observation operators and the operator contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet import niwot_reference_output

from sipnet_calibration.constraints import TIME_BOUNDS_END, TIME_BOUNDS_START
from sipnet_calibration.fields import label_run
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    ComputeLeafAreaIndex,
    ObservationOperator,
    ReduceOverRun,
    ReduceOverTimeBounds,
    SelectTimestep,
    check_operator,
    check_operator_declares_names,
    check_result_is_on_the_observation_grid,
    extract_sipnet_parameter_at_coords,
    select_observed_sites,
    select_timestep_at,
)

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
        # Give each label a bounds window of its own length, so a window read
        # from the wrong label's bounds would give a different value.
        observed = observed.assign_coords({TIME_BOUNDS_START: ("time", labels - pd.to_timedelta([1, 2, 3], unit="D"))})
        predicted = ReduceOverTimeBounds("wood_carbon", "last")(one_run, observed)
        np.testing.assert_array_equal(predicted["time"].values, observed["time"].values)
        assert predicted.attrs["kind"] == "timestep_end_state"
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        for k, (start, end) in enumerate(zip(observed[TIME_BOUNDS_START].values, labels)):
            inside = (ends > start) & (ends <= end)
            assert predicted.values[k] == wood.values[inside][-1]
            assert predicted["time_step_start"].values[k] == starts[inside].min()

    @staticmethod
    def _one_window(start, end):
        return xr.DataArray(
            [[5.0]], dims=("site", "time"),
            coords={"site": [1], "time": [pd.Timestamp(end)],
                    TIME_BOUNDS_START: ("time", [pd.Timestamp(start)]),
                    TIME_BOUNDS_END: ("time", [pd.Timestamp(end)])},
            attrs={"units": "g m-2", "constituent": "C"}, name="annual_total",
        )

    def test_a_window_the_run_covers_only_in_part_is_refused(self, one_run):
        # The Niwot record is November 1998; a calendar-1998 total is not its sum.
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverTimeBounds("net_ecosystem_exchange", "sum")(
                one_run, self._one_window("1998-01-01", "1999-01-01")
            )

    def test_a_window_inside_the_record_is_reduced(self, one_run):
        wood = one_run["wood_carbon"]
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        ends = pd.DatetimeIndex(wood["time"].values)
        predicted = ReduceOverTimeBounds("wood_carbon", "last")(
            one_run, self._one_window(starts[0], ends[-1])
        )
        assert predicted.values.ravel()[0] == wood.values[-1]

    def test_less_than_a_step_short_at_each_edge_is_allowed(self, one_run):
        wood = one_run["wood_carbon"]
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        ends = pd.DatetimeIndex(wood["time"].values)
        first, last = ends[0] - starts[0], ends[-1] - starts[-1]
        observed = self._one_window(starts[0] - first / 2, ends[-1] + last / 2)
        ReduceOverTimeBounds("wood_carbon", "last")(one_run, observed)
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverTimeBounds("wood_carbon", "last")(
                one_run, self._one_window(starts[0], ends[-1] + last)
            )
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverTimeBounds("wood_carbon", "last")(
                one_run, self._one_window(starts[0] - first, ends[-1])
            )

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


class TestExtractSipnetParameterAtCoords:
    def test_carries_pysipnets_units(self):
        array = extract_sipnet_parameter_at_coords({"leaf_carbon_per_area": 270.0}, "leaf_carbon_per_area", xr.DataArray(0.0))
        assert array.attrs["units"] == "g m-2" and array.attrs["constituent"] == "C"
        assert float(array) == 270.0

    def test_is_labeled_by_pysipnet(self):
        array = extract_sipnet_parameter_at_coords({"leaf_carbon_per_area": 270.0}, "leaf_carbon_per_area", xr.DataArray(0.0))
        assert array.name == "leaf_carbon_per_area"
        assert array.attrs["sipnet_name"] == "leafCSpWt" and "kind" not in array.attrs

    def test_a_mapping_key_may_be_any_name_of_the_parameter(self):
        array = extract_sipnet_parameter_at_coords({"leafCSpWt": 270.0}, "leaf_carbon_per_area", xr.DataArray(0.0))
        assert float(array) == 270.0

    def test_the_parameter_may_be_asked_for_by_an_alias(self):
        table = xr.Dataset({"leaf_carbon_per_area": (("site",), [1.0, 2.0])}, coords={"site": [1, 2]})
        array = extract_sipnet_parameter_at_coords(table, "leafCSpWt", xr.DataArray([0.0], dims="site", coords={"site": [2]}))
        assert array.name == "leaf_carbon_per_area" and array.values.tolist() == [2.0]

    def test_a_value_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="domain"):
            extract_sipnet_parameter_at_coords({"leaf_carbon_per_area": -1.0}, "leaf_carbon_per_area", xr.DataArray(0.0))

    def test_a_missing_entry_is_refused(self):
        with pytest.raises(ValueError, match="no entry"):
            extract_sipnet_parameter_at_coords({"soil_carbon": 1.0}, "leaf_carbon_per_area", xr.DataArray(0.0))

    def test_selects_the_arrays_sites_and_members_from_a_table(self):
        table = xr.Dataset({"leaf_carbon_per_area": (("member", "site"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}, coords={"member": [0, 1], "site": [1, 2, 3]})
        target_field = xr.DataArray(np.zeros((2, 2)), dims=("member", "site"), coords={"member": [0, 1], "site": [3, 1]})
        array = extract_sipnet_parameter_at_coords(table, "leaf_carbon_per_area", target_field)
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

    def test_a_parameter_alias_in_the_declaration_is_refused(self):
        @dataclass(frozen=True)
        class AliasedParameter:
            output_variable_names = ("leaf_carbon",)
            sipnet_parameter_names = ("leafCSpWt",)

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return model_output["leaf_carbon"]

        with pytest.raises(ValueError, match="flat name 'leaf_carbon_per_area'"):
            check_operator_declares_names(AliasedParameter())

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
        from sipnet_calibration.observation import (
            reduce_windows,
            windows_from_time_bounds,
        )

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

    def test_a_non_numeric_mapping_value_is_refused(self, one_run, labels):
        with pytest.raises(TypeError, match="must be a number"):
            ComputeLeafAreaIndex()(one_run, _observed([1], labels), sipnet_parameters={"leaf_carbon_per_area": "270"})

    def test_repeated_observed_sites_are_refused(self, stack, labels):
        with pytest.raises(ValueError, match="repeats a site"):
            select_observed_sites(stack["wood_carbon"], _observed([1, 1], labels))


class TestParameterLookupsAtScalarCoordinates:
    """A one-run model output carries site and member as scalars."""

    @pytest.fixture
    def table(self):
        return xr.Dataset(
            {"leaf_carbon_per_area": (("member", "site"), [[270.0, 135.0], [540.0, 90.0]])},
            coords={"member": [0, 1], "site": [1, 2]},
        )

    def test_a_scalar_site_and_member_select_one_value(self, one_run, table):
        array = extract_sipnet_parameter_at_coords(table, "leaf_carbon_per_area", one_run["leaf_carbon"])
        assert array.dims == () and float(array) == 270.0

    def test_one_run_is_predicted_on_its_own_grid_from_a_table(self, one_run, labels, table):
        predicted = check_operator(ComputeLeafAreaIndex(), one_run, _observed([1], labels), sipnet_parameters=table)
        assert predicted.dims == ("time",)
        expected = select_timestep_at(one_run["leaf_carbon"], labels).values / 270.0
        np.testing.assert_allclose(predicted.values, expected)

    def test_a_table_dimension_the_target_has_no_coordinate_for_is_refused(self, stack, table):
        unlabeled = stack["leaf_carbon"].isel(member=0, drop=True)  # site, but no member
        with pytest.raises(ValueError, match="carries no member coordinate"):
            extract_sipnet_parameter_at_coords(table, "leaf_carbon_per_area", unlabeled)

    def test_a_scalar_table_site_is_not_used_at_every_site(self, stack, table):
        with pytest.raises(ValueError, match="for site 1 alone"):
            extract_sipnet_parameter_at_coords(table.sel(site=1), "leaf_carbon_per_area", stack["leaf_carbon"])

    def test_a_scalar_table_site_at_its_own_site_is_used(self, one_run, table):
        array = extract_sipnet_parameter_at_coords(table.sel(site=1), "leaf_carbon_per_area", one_run["leaf_carbon"])
        assert array.dims == () and float(array) == 270.0

    def test_a_boolean_mapping_value_is_refused(self):
        with pytest.raises(TypeError, match="must be a number"):
            extract_sipnet_parameter_at_coords({"leaf_carbon_per_area": True}, "leaf_carbon_per_area", xr.DataArray(0.0))


class TestSelectTimestepOnAStaticObservation:
    def test_is_refused_pointing_at_reduce_over_run(self, one_run):
        static = xr.DataArray([1.0], dims="site", coords={"site": [1]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")
        with pytest.raises(ValueError, match="'soilgrids_soil_organic_carbon' has no 'time'.*ReduceOverRun"):
            SelectTimestep("soil_carbon")(one_run, static)
        with pytest.raises(ValueError, match="ReduceOverRun"):
            ComputeLeafAreaIndex()(one_run, static, sipnet_parameters={"leaf_carbon_per_area": 270.0})


class TestSelectObservedSitesOneRun:
    def test_one_run_cannot_serve_two_observed_sites(self, one_run, labels):
        with pytest.raises(ValueError, match=r"one run at site 1.*site\(s\) \[1, 2\]"):
            select_observed_sites(one_run["wood_carbon"], _observed([1, 2], labels))


class TestCheckOperatorContract:
    def test_an_operator_that_mixes_members_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class MemberMean:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(select_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "member" not in picked.dims:
                    return picked
                mixed = picked.mean("member").expand_dims(member=picked["member"].values).transpose(*picked.dims)
                mixed.attrs = picked.attrs
                return mixed

        with pytest.raises(ValueError, match="not pointwise in 'member'"):
            check_operator(MemberMean(), stack, _observed([1, 2], labels, units="Mg ha-1", constituent="C"))

    def test_declarations_as_a_list_are_refused(self, one_run, labels):
        @dataclass(frozen=True)
        class Listed:
            output_variable_names = ["wood_carbon"]
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"])

        with pytest.raises(TypeError, match="tuple of names"):
            check_operator(Listed(), one_run, _observed([1], labels))

    def test_an_unregistered_name_is_a_key_error(self, one_run, labels):
        @dataclass(frozen=True)
        class Unknown:
            output_variable_names = ("not_a_variable",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return model_output["wood_carbon"]

        with pytest.raises(KeyError, match="not a SIPNET output variable"):
            check_operator(Unknown(), one_run, _observed([1], labels))

    def test_a_model_output_lacking_the_variable_is_refused(self, one_run, labels):
        with pytest.raises(ValueError, match=r"lacks \['wood_carbon'\], which SelectTimestep read"):
            check_operator(SelectTimestep("wood_carbon"), one_run.drop_vars("wood_carbon"), _observed([1], labels))

    def test_parameters_read_and_not_given_are_refused(self, one_run, labels):
        with pytest.raises(ValueError, match="ComputeLeafAreaIndex read SIPNET parameters"):
            check_operator(ComputeLeafAreaIndex(), one_run, _observed([1], labels))

    def test_a_result_that_is_not_an_array_is_a_type_error(self, one_run, labels):
        @dataclass(frozen=True)
        class Numpy:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return np.zeros(3)

        with pytest.raises(TypeError, match="not a DataArray"):
            check_operator(Numpy(), one_run, _observed([1], labels))


class TestTheGridCheck:
    """``check_result_is_on_the_observation_grid``, shared by check_operator and predict."""

    @pytest.fixture
    def result(self, one_run, labels):
        return select_timestep_at(one_run["wood_carbon"], labels)

    def test_a_result_on_the_grid_passes_with_a_scalar_or_a_dimension_site(self, one_run, labels, result):
        check_result_is_on_the_observation_grid(result, _observed([1], labels), one_run, "x")
        check_result_is_on_the_observation_grid(result.expand_dims("site"), _observed([1], labels), one_run, "x")
        check_result_is_on_the_observation_grid(result, _observed([1], labels).isel(site=0), one_run, "x")

    def test_labels_are_compared_as_instants_across_units(self, one_run, labels, result):
        coarse = _observed([1], labels.as_unit("s"))
        assert coarse["time"].dtype == np.dtype("datetime64[s]")
        check_result_is_on_the_observation_grid(result, coarse, one_run, "x")

    def test_a_spurious_member_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="member dimension the model output lacks"):
            check_result_is_on_the_observation_grid(result.drop_vars("member").expand_dims(member=[0]), _observed([1], labels), one_run, "x")

    def test_a_scalar_site_result_at_the_wrong_site_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="not at the observation's site"):
            check_result_is_on_the_observation_grid(result.assign_coords(site=2), _observed([1], labels), one_run, "x")

    def test_a_result_with_no_site_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="carries no site"):
            check_result_is_on_the_observation_grid(result.drop_vars(["site", "lon", "lat"]), _observed([1], labels), one_run, "x")

    def test_a_time_dimension_for_a_static_observation_is_refused(self, one_run, result):
        static = xr.DataArray([1.0], dims="site", coords={"site": [1]}, attrs={"units": "Mg ha-1"})
        with pytest.raises(ValueError, match="time dimension for a static observation"):
            check_result_is_on_the_observation_grid(result, static, one_run, "x")

    def test_a_result_without_time_for_a_dated_observation_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="time labels"):
            check_result_is_on_the_observation_grid(result.isel(time=0), _observed([1], labels), one_run, "x")

    def test_the_label_prefixes_the_message(self, one_run, labels, result):
        with pytest.raises(ValueError, match="^my_product: "):
            check_result_is_on_the_observation_grid(result.assign_coords(site=2), _observed([1], labels), one_run, "my_product")

    def test_an_operator_returning_the_wrong_site_is_refused_by_check_operator(self, one_run, labels):
        @dataclass(frozen=True)
        class WrongSite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"]).assign_coords(site=2)

        with pytest.raises(ValueError, match="not at the observation's site"):
            check_operator(WrongSite(), one_run, _observed([1], labels).isel(site=0))


class TestPointwiseIsComparedByLabel:
    def test_a_pointwise_operator_that_always_returns_a_site_dimension_passes(self, stack, labels):
        """On one site's slice its result is (member, site=1, time), the stack's sliced (member, time)."""

        @dataclass(frozen=True)
        class AlwaysOnSite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(select_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "site" not in picked.dims:
                    picked = picked.expand_dims("site")
                return picked.transpose(*[d for d in ("member", "site", "time") if d in picked.dims])

        observed = _observed([1, 2], labels, units="Mg ha-1", constituent="C")
        assert check_operator(AlwaysOnSite(), stack, observed).dims == ("member", "site", "time")

    def test_a_pointwise_operator_returning_its_own_dim_order_passes(self, stack, labels):
        @dataclass(frozen=True)
        class Transposed:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(select_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                return picked.transpose("time", ...)

        observed = _observed([1, 2], labels, units="Mg ha-1", constituent="C")
        assert check_operator(Transposed(), stack, observed).dims[0] == "time"

    def test_an_operator_that_drops_the_member_dimension_is_not_pointwise(self, stack, labels):
        @dataclass(frozen=True)
        class FirstMember:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                wood = model_output["wood_carbon"]
                if "member" in wood.dims:
                    wood = wood.isel(member=0, drop=True)
                return select_timestep_at(select_observed_sites(wood, observed_values), observed_values["time"])

        with pytest.raises(ValueError, match="not pointwise in 'member'"):
            check_operator(FirstMember(), stack, _observed([1, 2], labels, units="Mg ha-1", constituent="C"))


class _Declares:
    def __init__(self, names):
        self.output_variable_names = names
        self.sipnet_parameter_names = ()

    def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
        raise AssertionError("not called")


class TestDeclarationsAndConstruction:
    def test_a_declaration_holding_a_non_string_is_a_type_error(self):
        with pytest.raises(TypeError, match="tuple of names"):
            check_operator_declares_names(_Declares((1,)))

    def test_the_message_name_prefixes_a_declaration_refusal(self):
        with pytest.raises(ValueError, match=r"^p: the operator declares 'nee'"):
            check_operator_declares_names(_Declares(("nee",)), "p")

    def test_reduce_over_run_refuses_an_unknown_reduction_when_built(self):
        with pytest.raises(ValueError, match="how must be one of .* for ReduceOverRun"):
            ReduceOverRun("soil_carbon", "median")

    def test_a_reduction_that_is_not_a_string_is_a_type_error(self):
        with pytest.raises(TypeError, match="how must be a string"):
            ReduceOverTimeBounds("wood_carbon", None)

    def test_units_that_are_not_a_string_are_refused(self, one_run, labels):
        result = select_timestep_at(one_run["wood_carbon"], labels)
        result.attrs["units"] = None
        with pytest.raises(ValueError, match="no 'units'"):
            check_result_is_on_the_observation_grid(result, _observed([1], labels), one_run, "x")


class TestTheOperatorsNameThemselves:
    def test_a_window_reduction_on_output_without_intervals_names_the_operator(self, one_run, labels):
        bare = one_run.drop_vars(["time_step_start", "time_step_length", "time_bounds"], errors="ignore")
        observed = _observed([1], labels, units="g m-2", constituent="C", name="annual", bounds=True)
        with pytest.raises(ValueError, match="^ReduceOverTimeBounds on 'annual' reads the interval"):
            ReduceOverTimeBounds("wood_carbon", "last")(bare, observed)

    def test_a_window_beyond_the_record_names_the_operator_and_observation(self, one_run):
        observed = TestReduceOverTimeBounds._one_window("1998-01-01", "1999-01-01")
        with pytest.raises(ValueError, match="^ReduceOverTimeBounds on 'annual_total': the window"):
            ReduceOverTimeBounds("net_ecosystem_exchange", "sum")(one_run, observed)


class TestScalarTableLabels:
    def test_a_scalar_table_member_serves_a_run_without_a_member(self, one_run):
        run = one_run.drop_vars("member")
        table = xr.Dataset(
            {"leaf_carbon_per_area": (("member", "site"), [[270.0, 135.0]])}, coords={"member": [0], "site": [1, 2]}
        ).isel(member=0)
        assert float(extract_sipnet_parameter_at_coords(table, "leaf_carbon_per_area", run["leaf_carbon"])) == 270.0

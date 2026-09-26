"""The shipped observation operators and the operator contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet import niwot_reference_output

from conftest import (
    dated_observed_values,
    located,
    niwot_stack_of,
    site_table_of,
    windowed_observed_values,
)
from sipnet_calibration.conventions import WINDOW_END, WINDOW_START
from sipnet_calibration.fields import label_run
from sipnet_calibration.observation import (
    DEFAULT_OBS_OPS,
    ComputeLeafAreaIndex,
    ObservationOperator,
    ReduceOverRun,
    ReduceOverWindows,
    SelectTimestep,
    check_operator,
    check_operator_declares_names,
    check_result_is_on_the_observation_grid,
    extract_sipnet_parameter_at_coords,
    restrict_to_observed_sites,
    select_timestep_at,
)

VARIABLES = ["leaf_carbon", "wood_carbon", "soil_carbon", "net_ecosystem_exchange", "soil_wetness_fraction"]


@pytest.fixture(scope="module")
def one_run():
    """One Niwot run labeled as sample 0 at site 1."""
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


@pytest.fixture
def labels(one_run):
    return pd.DatetimeIndex(one_run["time"].values[[5, 20, 40]])


class TestSelectTimestep:
    def test_reads_the_state_at_the_observed_labels(self, one_run, labels):
        observed = dated_observed_values(
            [1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
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
        observed = dated_observed_values(
            [1, 2], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        predicted = check_operator(SelectTimestep("wood_carbon"), stack, observed)
        assert predicted.dims == ("sample", "site", "time")
        np.testing.assert_allclose(predicted.sel(site=2, sample=1).values, 0.75 * predicted.sel(site=1, sample=0).values)


class TestReduceOverWindows:
    def test_reduces_over_each_observations_own_window(self, one_run, labels):
        observed = windowed_observed_values(
            [1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        # Give each label a window of its own length, so a window read from the
        # wrong label would give a different value.
        observed = observed.assign_coords(
            {WINDOW_START: ("time", labels - pd.to_timedelta([1, 2, 3], unit="D"))}
        )
        predicted = ReduceOverWindows("wood_carbon", "last")(one_run, observed)
        np.testing.assert_array_equal(predicted["time"].values, observed["time"].values)
        assert predicted.attrs["kind"] == "timestep_end_state"
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        for k, (start, end) in enumerate(zip(observed[WINDOW_START].values, labels)):
            inside = (ends > start) & (ends <= end)
            assert predicted.values[k] == wood.values[inside][-1]
            assert predicted["time_step_start"].values[k] == starts[inside].min()

    @staticmethod
    def _one_window(start, end):
        return located(xr.DataArray(
            [[5.0]], dims=("site", "time"),
            coords={"site": [1], "time": [pd.Timestamp(end)],
                    WINDOW_START: ("time", [pd.Timestamp(start)]),
                    WINDOW_END: ("time", [pd.Timestamp(end)])},
            attrs={"units": "g m-2", "constituent": "C"}, name="annual_total",
        ))

    def test_a_window_the_run_covers_only_in_part_is_refused(self, one_run):
        # The Niwot record is November 1998; a calendar-1998 total is not its sum.
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverWindows("net_ecosystem_exchange", "sum")(
                one_run, self._one_window("1998-01-01", "1999-01-01")
            )

    def test_a_window_inside_the_record_is_reduced(self, one_run):
        wood = one_run["wood_carbon"]
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        ends = pd.DatetimeIndex(wood["time"].values)
        predicted = ReduceOverWindows("wood_carbon", "last")(
            one_run, self._one_window(starts[0], ends[-1])
        )
        assert predicted.values.ravel()[0] == wood.values[-1]

    def test_less_than_a_step_short_at_each_edge_is_allowed(self, one_run):
        wood = one_run["wood_carbon"]
        starts = pd.DatetimeIndex(wood["time_step_start"].values)
        ends = pd.DatetimeIndex(wood["time"].values)
        first, last = ends[0] - starts[0], ends[-1] - starts[-1]
        observed = self._one_window(starts[0] - first / 2, ends[-1] + last / 2)
        ReduceOverWindows("wood_carbon", "last")(one_run, observed)
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverWindows("wood_carbon", "last")(
                one_run, self._one_window(starts[0], ends[-1] + last)
            )
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            ReduceOverWindows("wood_carbon", "last")(
                one_run, self._one_window(starts[0] - first, ends[-1])
            )

    def test_refuses_observed_values_without_windows(self, one_run, labels):
        observed = dated_observed_values([1], labels, units="Mg ha-1", constituent="C")
        with pytest.raises(ValueError, match="documents no interval"):
            ReduceOverWindows("wood_carbon", "mean")(one_run, observed)

    def test_refuses_an_unknown_reduction(self):
        with pytest.raises(ValueError, match="how must be one of"):
            ReduceOverWindows("wood_carbon", "median")


class TestReduceOverRun:
    def test_reduces_over_the_whole_record_for_static_observed_values(self, one_run):
        observed = xr.DataArray([1.0], dims="site", coords={"site": [1]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")
        predicted = ReduceOverRun("soil_carbon", "mean")(one_run, observed)
        assert "time" not in predicted.dims
        np.testing.assert_allclose(float(predicted), float(one_run["soil_carbon"].weighted(one_run["time_step_length"].astype("int64")).mean()), rtol=1e-12)

    def test_refuses_dated_observed_values(self, one_run, labels):
        observed = dated_observed_values([1], labels, units="Mg ha-1", constituent="C")
        with pytest.raises(ValueError, match="static observation"):
            ReduceOverRun("soil_carbon", "mean")(one_run, observed)

    def test_pointwise_over_a_stack(self, stack):
        observed = xr.DataArray([1.0, 1.0], dims="site", coords={"site": [1, 2]}, attrs={"units": "Mg ha-1", "constituent": "C"}, name="soilgrids_soil_organic_carbon")
        predicted = check_operator(ReduceOverRun("soil_carbon", "last"), stack, observed)
        assert predicted.dims == ("sample", "site")


class TestComputeLeafAreaIndex:
    def test_is_the_default_for_modis(self):
        assert set(DEFAULT_OBS_OPS) == {"modis_leaf_area_index"}
        assert isinstance(DEFAULT_OBS_OPS["modis_leaf_area_index"], ComputeLeafAreaIndex)

    def test_declares_its_reads(self):
        op = ComputeLeafAreaIndex()
        assert op.output_variable_names == ("leaf_carbon",)
        assert op.sipnet_parameter_names == ("leaf_carbon_per_area",)

    def test_leaf_carbon_over_the_parameter_from_a_mapping(self, one_run, labels):
        observed = dated_observed_values([1], labels)
        predicted = ComputeLeafAreaIndex()(one_run, observed, sipnet_parameters={"leaf_carbon_per_area": 270.0})
        expected = select_timestep_at(one_run["leaf_carbon"], labels).values / 270.0
        np.testing.assert_allclose(predicted.values, expected)
        assert predicted.attrs["units"] == "1"
        assert "constituent" not in predicted.attrs

    def test_the_parameter_varies_by_member_and_site(self, stack, labels):
        observed = dated_observed_values([1, 2], labels)
        sipnet_parameter_fields = xr.Dataset({"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0], [540.0, 270.0]])}, coords={"sample": [0, 1], "site": [1, 2]})
        predicted = check_operator(ComputeLeafAreaIndex(), stack, observed, sipnet_parameters=sipnet_parameter_fields)
        leaf = select_timestep_at(stack["leaf_carbon"], labels)
        np.testing.assert_allclose(predicted.sel(sample=1, site=2).values, leaf.sel(sample=1, site=2).values / 270.0)
        np.testing.assert_allclose(predicted.sel(sample=0, site=2).values, leaf.sel(sample=0, site=2).values / 135.0)

    def test_the_parameter_is_required(self, one_run, labels):
        with pytest.raises(ValueError, match="pass sipnet_parameters"):
            ComputeLeafAreaIndex()(one_run, dated_observed_values([1], labels))

    def test_sipnet_parameter_fields_lacking_the_parameter_is_refused(self, one_run, labels):
        sipnet_parameter_fields = xr.Dataset({"max_photosynthesis_rate": (("site",), [10.0])}, coords={"site": [1]})
        with pytest.raises(ValueError, match="no variable 'leaf_carbon_per_area'"):
            ComputeLeafAreaIndex()(one_run, dated_observed_values([1], labels), sipnet_parameters=sipnet_parameter_fields)


class TestRestrictToObservedSites:
    def test_a_missing_site_is_named(self, stack, labels):
        with pytest.raises(ValueError, match=r"no site\(s\) \[3\]"):
            restrict_to_observed_sites(stack["wood_carbon"], dated_observed_values([1, 3], labels))

    def test_a_single_run_must_be_the_observed_site(self, one_run, labels):
        with pytest.raises(ValueError, match="one run at site 1"):
            restrict_to_observed_sites(one_run["wood_carbon"], dated_observed_values([2], labels))

    def test_an_unlabeled_run_is_refused(self, labels):
        array = niwot_reference_output().select(["wood_carbon"])["wood_carbon"]
        with pytest.raises(ValueError, match="no 'site' coordinate"):
            restrict_to_observed_sites(array, dated_observed_values([1], labels))

    def test_selection_follows_the_observed_values_order(self, stack, labels):
        picked = restrict_to_observed_sites(stack["wood_carbon"], dated_observed_values([2, 1], labels))
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
        sipnet_parameter_fields = xr.Dataset({"leaf_carbon_per_area": (("site",), [1.0, 2.0])}, coords={"site": [1, 2]})
        array = extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leafCSpWt", xr.DataArray([0.0], dims="site", coords={"site": [2]}))
        assert array.name == "leaf_carbon_per_area" and array.values.tolist() == [2.0]

    def test_a_value_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="domain"):
            extract_sipnet_parameter_at_coords({"leaf_carbon_per_area": -1.0}, "leaf_carbon_per_area", xr.DataArray(0.0))

    def test_a_missing_entry_is_refused(self):
        with pytest.raises(ValueError, match="no entry"):
            extract_sipnet_parameter_at_coords({"soil_carbon": 1.0}, "leaf_carbon_per_area", xr.DataArray(0.0))

    def test_selects_the_arrays_sites_and_samples_from_sipnet_parameter_fields(self):
        sipnet_parameter_fields = xr.Dataset({"leaf_carbon_per_area": (("sample", "site"), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}, coords={"sample": [0, 1], "site": [1, 2, 3]})
        target_field = xr.DataArray(np.zeros((2, 2)), dims=("sample", "site"), coords={"sample": [0, 1], "site": [3, 1]})
        array = extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", target_field)
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
            check_operator(Aliased(), one_run, dated_observed_values([1], labels))

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
            check_operator(OffGrid(), one_run, dated_observed_values([1], labels))

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
            check_operator(Unitless(), one_run, dated_observed_values([1], labels))

    def test_an_operator_that_mixes_sites_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class SiteMean:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(restrict_to_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "site" not in picked.dims:
                    return picked
                mixed = picked.mean("site").expand_dims(site=picked["site"].values).transpose(*picked.dims)
                mixed.attrs = picked.attrs
                return mixed

        with pytest.raises(ValueError, match="not pointwise in 'site'"):
            check_operator(SiteMean(), stack, dated_observed_values([1, 2], labels))


class TestSiteOrderAndCoverage:
    def test_an_operator_follows_observed_values_out_of_model_order(self, stack, labels):
        observed = dated_observed_values(
            [2, 1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        predicted = check_operator(SelectTimestep("wood_carbon"), stack, observed)
        assert predicted["site"].values.tolist() == [2, 1]
        np.testing.assert_allclose(predicted.sel(site=2, sample=0).values, 1.5 * predicted.sel(site=1, sample=0).values)

    def test_an_operator_returning_model_order_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class ModelOrder:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                sites = sorted(observed_values["site"].values.tolist())
                return select_timestep_at(model_output["wood_carbon"].sel(site=sites), observed_values["time"])

        with pytest.raises(ValueError, match="sites, in order"):
            check_operator(
                ModelOrder(),
                stack,
                dated_observed_values([2, 1], labels, units="Mg ha-1", constituent="C"),
            )

    def test_check_operator_handles_a_run_over_more_sites_than_observed(self, stack, labels):
        third = (stack.isel(site=[1]) * 2).assign_coords(site=np.array([3], dtype=np.int32))
        wider = xr.concat([stack, third], dim="site")
        for name in VARIABLES:
            wider[name].attrs = stack[name].attrs
        observed = dated_observed_values(
            [1, 2], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        predicted = check_operator(SelectTimestep("wood_carbon"), wider, observed)
        assert predicted["site"].values.tolist() == [1, 2]

    def test_check_operator_with_scalar_site_observed_values(self, one_run, labels):
        observed = dated_observed_values([1], labels, units="Mg ha-1", constituent="C").isel(site=0)
        predicted = check_operator(SelectTimestep("wood_carbon"), one_run, observed)
        assert int(predicted["site"]) == 1


class TestReduceOverWindowsValues:
    def test_the_label_need_not_be_a_window_edge(self, one_run, labels):
        observed = windowed_observed_values(
            [1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        shifted = observed.assign_coords(time=observed["time"].values - np.timedelta64(12, "h"))
        predicted = ReduceOverWindows("wood_carbon", "mean")(one_run, shifted)
        np.testing.assert_array_equal(predicted["time"].values, shifted["time"].values)
        from sipnet_calibration.observation import (
            reduce_windows,
            windows_from_observed_values,
        )

        expected = reduce_windows(one_run["wood_carbon"], windows_from_observed_values(shifted), "mean")
        np.testing.assert_array_equal(predicted.values, expected.values)

    def test_the_mean_over_a_window_is_the_length_weighted_mean(self, one_run, labels):
        observed = windowed_observed_values(
            [1], labels, units="Mg ha-1", constituent="C", name="landtrendr_aboveground_biomass"
        )
        predicted = ReduceOverWindows("wood_carbon", "mean")(one_run, observed)
        wood = one_run["wood_carbon"]
        ends = pd.DatetimeIndex(wood["time"].values)
        lengths = wood["time_step_length"].values.astype("timedelta64[ns]").astype("float64")
        for k, label in enumerate(labels):
            inside = (ends > label - pd.Timedelta("1D")) & (ends <= label)
            expected = np.average(wood.values[inside], weights=lengths[inside])
            np.testing.assert_allclose(predicted.values[k], expected)


class TestParameterLookups:
    def test_sipnet_parameter_fields_missing_the_runs_members_is_named(self, stack, labels):
        sipnet_parameter_fields = xr.Dataset({"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0]])}, coords={"sample": [0], "site": [1, 2]})
        with pytest.raises(ValueError, match=r"no sample label\(s\) \[1\]"):
            ComputeLeafAreaIndex()(
                stack, dated_observed_values([1, 2], labels), sipnet_parameters=sipnet_parameter_fields
            )

    def test_a_non_numeric_mapping_value_is_refused(self, one_run, labels):
        with pytest.raises(TypeError, match="must be a number"):
            ComputeLeafAreaIndex()(
                one_run,
                dated_observed_values([1], labels),
                sipnet_parameters={"leaf_carbon_per_area": "270"},
            )

    def test_repeated_observed_sites_are_refused(self, stack, labels):
        with pytest.raises(ValueError, match="more than once"):
            restrict_to_observed_sites(stack["wood_carbon"], dated_observed_values([1, 1], labels))


class TestParameterLookupsAtScalarCoordinates:
    """A one-run model output carries site and its batch labels as scalars."""

    @pytest.fixture
    def sipnet_parameter_fields(self):
        return xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0], [540.0, 90.0]])},
            coords={"sample": [0, 1], "site": [1, 2]},
        )

    def test_a_scalar_site_and_sample_select_one_value(self, one_run, sipnet_parameter_fields):
        array = extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", one_run["leaf_carbon"])
        assert array.dims == () and float(array) == 270.0

    def test_one_run_is_predicted_from_sipnet_parameter_fields(self, one_run, labels, sipnet_parameter_fields):
        predicted = check_operator(
            ComputeLeafAreaIndex(), one_run, dated_observed_values([1], labels), sipnet_parameters=sipnet_parameter_fields
        )
        assert predicted.dims == ("time",)
        expected = select_timestep_at(one_run["leaf_carbon"], labels).values / 270.0
        np.testing.assert_allclose(predicted.values, expected)

    def test_sipnet_parameter_fields_dimension_the_target_has_no_coordinate_for_is_refused(self, stack, sipnet_parameter_fields):
        unlabeled = stack["leaf_carbon"].isel(sample=0, drop=True)  # site, but no sample
        with pytest.raises(ValueError, match="carries no sample coordinate"):
            extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", unlabeled)

    def test_scalar_sipnet_parameter_fields_site_is_not_used_at_every_site(self, stack, sipnet_parameter_fields):
        with pytest.raises(ValueError, match="for site 1 alone"):
            extract_sipnet_parameter_at_coords(sipnet_parameter_fields.sel(site=1), "leaf_carbon_per_area", stack["leaf_carbon"])

    def test_scalar_sipnet_parameter_fields_site_at_its_own_site_is_used(self, one_run, sipnet_parameter_fields):
        array = extract_sipnet_parameter_at_coords(sipnet_parameter_fields.sel(site=1), "leaf_carbon_per_area", one_run["leaf_carbon"])
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


class TestRestrictToObservedSitesOneRun:
    def test_one_run_cannot_serve_two_observed_sites(self, one_run, labels):
        with pytest.raises(ValueError, match=r"one run at site 1.*site\(s\) \[1, 2\]"):
            restrict_to_observed_sites(one_run["wood_carbon"], dated_observed_values([1, 2], labels))


class TestCheckOperatorContract:
    def test_an_operator_that_mixes_members_is_refused(self, stack, labels):
        @dataclass(frozen=True)
        class MemberMean:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(restrict_to_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "sample" not in picked.dims:
                    return picked
                mixed = picked.mean("sample").expand_dims(sample=picked["sample"].values).transpose(*picked.dims)
                mixed.attrs = picked.attrs
                return mixed

        with pytest.raises(ValueError, match="not pointwise in 'sample'"):
            check_operator(
                MemberMean(),
                stack,
                dated_observed_values([1, 2], labels, units="Mg ha-1", constituent="C"),
            )

    def test_declarations_as_a_list_are_refused(self, one_run, labels):
        @dataclass(frozen=True)
        class Listed:
            output_variable_names = ["wood_carbon"]
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"])

        with pytest.raises(TypeError, match="tuple of names"):
            check_operator(Listed(), one_run, dated_observed_values([1], labels))

    def test_an_unregistered_name_is_a_key_error(self, one_run, labels):
        @dataclass(frozen=True)
        class Unknown:
            output_variable_names = ("not_a_variable",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return model_output["wood_carbon"]

        with pytest.raises(KeyError, match="not a SIPNET output variable"):
            check_operator(Unknown(), one_run, dated_observed_values([1], labels))

    def test_a_model_output_lacking_the_variable_is_refused(self, one_run, labels):
        with pytest.raises(ValueError, match=r"lacks \['wood_carbon'\], which SelectTimestep read"):
            check_operator(
                SelectTimestep("wood_carbon"),
                one_run.drop_vars("wood_carbon"),
                dated_observed_values([1], labels),
            )

    def test_parameters_read_and_not_given_are_refused(self, one_run, labels):
        with pytest.raises(ValueError, match="ComputeLeafAreaIndex read SIPNET parameters"):
            check_operator(ComputeLeafAreaIndex(), one_run, dated_observed_values([1], labels))

    def test_a_result_that_is_not_an_array_is_a_type_error(self, one_run, labels):
        @dataclass(frozen=True)
        class Numpy:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return np.zeros(3)

        with pytest.raises(TypeError, match="not a DataArray"):
            check_operator(Numpy(), one_run, dated_observed_values([1], labels))


class TestTheGridCheck:
    """``check_result_is_on_the_observation_grid``, shared by check_operator and predict."""

    @pytest.fixture
    def result(self, one_run, labels):
        return select_timestep_at(one_run["wood_carbon"], labels)

    def test_a_result_on_the_grid_passes_with_a_scalar_or_a_dimension_site(self, one_run, labels, result):
        check_result_is_on_the_observation_grid(
            result, dated_observed_values([1], labels), one_run, "x"
        )
        check_result_is_on_the_observation_grid(
            result.expand_dims("site"), dated_observed_values([1], labels), one_run, "x"
        )
        check_result_is_on_the_observation_grid(
            result, dated_observed_values([1], labels).isel(site=0), one_run, "x"
        )

    def test_labels_are_compared_as_instants_across_units(self, one_run, labels, result):
        coarse = dated_observed_values([1], labels.as_unit("s"))
        assert coarse["time"].dtype == np.dtype("datetime64[s]")
        check_result_is_on_the_observation_grid(result, coarse, one_run, "x")

    def test_a_spurious_member_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match=r"dim\(s\) \['sample'\] that neither the model output"):
            check_result_is_on_the_observation_grid(
                result.drop_vars("sample").expand_dims(sample=[0]),
                dated_observed_values([1], labels),
                one_run,
                "x",
            )

    def test_a_scalar_site_result_at_the_wrong_site_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="not at the observed values' site"):
            check_result_is_on_the_observation_grid(
                result.assign_coords(site=2), dated_observed_values([1], labels), one_run, "x"
            )

    def test_a_result_with_no_site_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="carries no site"):
            check_result_is_on_the_observation_grid(
                result.drop_vars(["site", "lon", "lat"]),
                dated_observed_values([1], labels),
                one_run,
                "x",
            )

    def test_a_time_dimension_for_static_observed_values_is_refused(self, one_run, result):
        static = xr.DataArray([1.0], dims="site", coords={"site": [1]}, attrs={"units": "Mg ha-1"})
        with pytest.raises(ValueError, match="time dimension for static observed values"):
            check_result_is_on_the_observation_grid(result, static, one_run, "x")

    def test_a_result_without_time_for_dated_observed_values_is_refused(self, one_run, labels, result):
        with pytest.raises(ValueError, match="time labels"):
            check_result_is_on_the_observation_grid(
                result.isel(time=0), dated_observed_values([1], labels), one_run, "x"
            )

    def test_the_label_prefixes_the_message(self, one_run, labels, result):
        with pytest.raises(ValueError, match="^my_observation_source: "):
            check_result_is_on_the_observation_grid(
                result.assign_coords(site=2), dated_observed_values([1], labels), one_run, "my_observation_source"
            )

    def test_an_operator_returning_the_wrong_site_is_refused_by_check_operator(self, one_run, labels):
        @dataclass(frozen=True)
        class WrongSite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                return select_timestep_at(model_output["wood_carbon"], observed_values["time"]).assign_coords(site=2)

        with pytest.raises(ValueError, match="not at the observed values' site"):
            check_operator(WrongSite(), one_run, dated_observed_values([1], labels).isel(site=0))


class TestPointwiseIsComparedByLabel:
    def test_a_pointwise_operator_that_always_returns_a_site_dimension_passes(self, stack, labels):
        """On one site's slice its result is (sample, site=1, time), the stack's sliced (sample, time)."""

        @dataclass(frozen=True)
        class AlwaysOnSite:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(restrict_to_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                if "site" not in picked.dims:
                    picked = picked.expand_dims("site")
                return picked.transpose(*[d for d in ("sample", "site", "time") if d in picked.dims])

        observed = dated_observed_values([1, 2], labels, units="Mg ha-1", constituent="C")
        assert check_operator(AlwaysOnSite(), stack, observed).dims == ("sample", "site", "time")

    def test_a_pointwise_operator_returning_its_own_dim_order_passes(self, stack, labels):
        @dataclass(frozen=True)
        class Transposed:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(restrict_to_observed_sites(model_output["wood_carbon"], observed_values), observed_values["time"])
                return picked.transpose("time", ...)

        observed = dated_observed_values([1, 2], labels, units="Mg ha-1", constituent="C")
        assert check_operator(Transposed(), stack, observed).dims[0] == "time"

    def test_an_operator_that_drops_the_member_dimension_is_not_pointwise(self, stack, labels):
        @dataclass(frozen=True)
        class FirstMember:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                wood = model_output["wood_carbon"]
                if "sample" in wood.dims:
                    wood = wood.isel(sample=0, drop=True)
                return select_timestep_at(restrict_to_observed_sites(wood, observed_values), observed_values["time"])

        with pytest.raises(ValueError, match="not pointwise in 'sample'"):
            check_operator(
                FirstMember(),
                stack,
                dated_observed_values([1, 2], labels, units="Mg ha-1", constituent="C"),
            )


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
            ReduceOverWindows("wood_carbon", None)

    def test_units_that_are_not_a_string_are_refused(self, one_run, labels):
        result = select_timestep_at(one_run["wood_carbon"], labels)
        result.attrs["units"] = None
        with pytest.raises(ValueError, match="no 'units'"):
            check_result_is_on_the_observation_grid(
                result, dated_observed_values([1], labels), one_run, "x"
            )


class TestTheOperatorsNameThemselves:
    def test_a_window_reduction_on_output_without_intervals_names_the_operator(self, one_run, labels):
        bare = one_run.drop_vars(["time_step_start", "time_step_length", "time_bounds"], errors="ignore")
        observed = windowed_observed_values([1], labels, units="g m-2", constituent="C", name="annual")
        with pytest.raises(ValueError, match="^ReduceOverWindows on 'annual' reads the interval"):
            ReduceOverWindows("wood_carbon", "last")(bare, observed)

    def test_a_window_beyond_the_record_names_the_operator_and_observation_source(self, one_run):
        observed = TestReduceOverWindows._one_window("1998-01-01", "1999-01-01")
        with pytest.raises(ValueError, match="^ReduceOverWindows on 'annual_total': the window"):
            ReduceOverWindows("net_ecosystem_exchange", "sum")(one_run, observed)


class TestScalarSIPNETParameterFieldsLabels:
    def test_scalar_sipnet_parameter_fields_member_serves_a_run_without_a_member(self, one_run):
        run = one_run.drop_vars("sample")
        sipnet_parameter_fields = xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0]])}, coords={"sample": [0], "site": [1, 2]}
        ).isel(sample=0)
        assert float(extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", run["leaf_carbon"])) == 270.0


class TestEveryBatchDimOfSIPNETParameterFieldsIsSelectedOrRefused:
    """Regression for SIPNET parameter fields whose batch dim was not named ``member``: it
    used to be broadcast whole into one run, silently, and its labels were
    never selected."""

    @pytest.fixture
    def sipnet_parameter_fields(self):
        return xr.Dataset(
            {"leaf_carbon_per_area": (("driver_member", "site"), [[270.0, 135.0], [540.0, 90.0]])},
            coords={"driver_member": [5, 6], "site": [1, 2]},
        )

    def test_sipnet_parameter_fields_batch_dim_of_any_name_is_refused_against_a_run_without_it(self, one_run, sipnet_parameter_fields):
        with pytest.raises(ValueError, match="'driver_member'.*would broadcast"):
            extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", one_run["leaf_carbon"])

    def test_sipnet_parameter_fields_batch_dim_of_any_name_is_selected_at_the_runs_labels(self, one_run, sipnet_parameter_fields):
        run = one_run["leaf_carbon"].assign_coords(driver_member=6)
        array = extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", run)
        assert array.dims == () and float(array) == 540.0

    def test_labels_the_sipnet_parameter_fields_lack_are_refused_rather_than_joined_away(self, stack, sipnet_parameter_fields):
        target = stack["leaf_carbon"].rename(sample="driver_member")  # labels 0 and 1
        with pytest.raises(ValueError, match=r"no driver_member label\(s\) \[0, 1\]"):
            extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", target)

    def test_a_scalar_batch_label_of_any_name_must_agree(self, stack, sipnet_parameter_fields):
        target = stack["leaf_carbon"].rename(sample="driver_member").assign_coords(driver_member=[5, 6])
        with pytest.raises(ValueError, match="for driver_member 5 alone"):
            extract_sipnet_parameter_at_coords(
                sipnet_parameter_fields.sel(driver_member=5), "leaf_carbon_per_area", target
            )

    def test_an_operator_mixing_a_batch_dim_of_any_name_is_not_pointwise(self, stack, labels):
        @dataclass(frozen=True)
        class MeanOverDriverMembers:
            output_variable_names = ("wood_carbon",)
            sipnet_parameter_names = ()

            def __call__(self, model_output, observed_values, *, sipnet_parameters=None):
                picked = select_timestep_at(
                    restrict_to_observed_sites(model_output["wood_carbon"], observed_values),
                    observed_values["time"],
                )
                if "driver_member" not in picked.dims:
                    return picked
                mixed = picked.mean("driver_member").expand_dims(
                    driver_member=picked["driver_member"].values
                ).transpose(*picked.dims)
                mixed.attrs = picked.attrs
                return mixed

        with pytest.raises(ValueError, match="not pointwise in 'driver_member'"):
            check_operator(
                MeanOverDriverMembers(),
                stack.rename(sample="driver_member"),
                dated_observed_values([1, 2], labels, units="Mg ha-1", constituent="C"),
            )


class TestTwoBatchDimsPassThroughTheOperators:
    def test_sample_by_initial_condition_member_is_pointwise_in_both(self, stack, labels):
        crossed = xr.concat(
            [stack, stack * 2.0], dim=pd.Index([0, 1], name="initial_condition_member"), coords="minimal",
        )
        for name in crossed.data_vars:
            crossed[name].attrs = stack[name].attrs
        crossed = crossed.transpose("sample", "initial_condition_member", "site", "time")
        sipnet_parameter_fields = xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), [[270.0, 135.0], [540.0, 270.0]])},
            coords={"sample": [0, 1], "site": [1, 2]},
        )
        predicted = check_operator(
            ComputeLeafAreaIndex(), crossed, dated_observed_values([1, 2], labels), sipnet_parameters=sipnet_parameter_fields
        )
        assert predicted.dims == ("sample", "initial_condition_member", "site", "time")
        np.testing.assert_allclose(
            predicted.sel(initial_condition_member=1).values,
            2.0 * predicted.sel(initial_condition_member=0).values,
        )


class TestAStackedTargetIsNotReadAtSIPNETParameterFieldsLabels:
    """A stack of ``(sample, driver_member)`` labeled ``sample`` 0..n-1 is not
    theta's ``sample``; reading the SIPNET parameter fields at those labels picked wrong rows."""

    def test_a_target_stacked_over_sipnet_parameter_fields_dim_is_refused(self, stack):
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack["leaf_carbon"].expand_dims(driver_member=[0, 1], axis=1)
        relabeled = stack_batch_dims(crossed, into="run").rename(run="sample")
        sipnet_parameter_fields = xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), np.arange(8.0).reshape(4, 2) + 1)},
            coords={"sample": [0, 1, 2, 3], "site": [1, 2]},
        )
        with pytest.raises(ValueError, match="is a stack"):
            extract_sipnet_parameter_at_coords(sipnet_parameter_fields, "leaf_carbon_per_area", relabeled)

    def _stacked_leaf_carbon(self, stack):
        """Leaf carbon over (sample 0..1, driver_member 0..1) stacked into ``run``."""
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack["leaf_carbon"].expand_dims(driver_member=[0, 1], axis=1)
        return stack_batch_dims(crossed, into="run")

    def _table(self):
        return xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), [[20.0, 40.0], [30.0, 60.0], [99.0, 99.0]])},
            coords={"sample": [0, 1, 3], "site": [1, 2]},
        )

    def test_sipnet_parameter_fields_for_one_sample_is_refused_against_a_stack_of_others(self, stack):
        """Sample 3's parameter was applied, silently, to runs of samples 0 and 1."""
        with pytest.raises(ValueError, match="for sample 3 alone"):
            extract_sipnet_parameter_at_coords(
                self._table().sel(sample=3), "leaf_carbon_per_area", self._stacked_leaf_carbon(stack)
            )

    def test_sipnet_parameter_fields_for_one_sample_serves_a_stack_of_that_sample(self, stack):
        stacked = self._stacked_leaf_carbon(stack)
        of_sample_1 = stacked.isel(run=stacked["sample_label"].values == 1)
        array = extract_sipnet_parameter_at_coords(
            self._table().sel(sample=1), "leaf_carbon_per_area", of_sample_1
        )
        assert array.dims == ("site",) and array.values.tolist() == [30.0, 60.0]

    def test_the_operators_refuse_one_sample_sipnet_parameter_fields_against_a_stack(self, stack, labels):
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack.expand_dims(driver_member=[0, 1], axis=1)
        stacked = crossed.map(lambda variable: stack_batch_dims(variable, into="run"))
        with pytest.raises(ValueError, match="for sample 3 alone"):
            ComputeLeafAreaIndex()(
                stacked,
                dated_observed_values([1, 2], labels),
                sipnet_parameters=self._table().sel(sample=3),
            )

    def test_a_label_coordinate_on_the_dim_itself_is_not_a_stack(self, stack):
        """An unrelated ``sample_label`` on ``sample`` was taken for a stack of ``sample``."""
        named = stack["leaf_carbon"].assign_coords(sample_label=("sample", [7, 8]))
        array = extract_sipnet_parameter_at_coords(self._table(), "leaf_carbon_per_area", named)
        assert array.dims == ("sample", "site")
        assert array.values.tolist() == [[20.0, 40.0], [30.0, 60.0]]


class TestARestackedTargetIsReadAtItsLabels:
    """A stack of a stack records only the dim it stacked, and carries the
    first stack's ``sample_label`` on the new dim; one-sample SIPNET parameter fields were
    applied, silently, to runs of other samples."""

    def _restacked_leaf_carbon(self, stack):
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack["leaf_carbon"].expand_dims(driver_member=[0, 1], axis=1)
        return stack_batch_dims(stack_batch_dims(crossed, into="run"), into="run2")

    def _table(self):
        return xr.Dataset(
            {"leaf_carbon_per_area": (("sample", "site"), [[20.0, 40.0], [30.0, 60.0], [99.0, 99.0]])},
            coords={"sample": [0, 1, 3], "site": [1, 2]},
        )

    def test_sipnet_parameter_fields_for_one_sample_is_refused_against_a_restack_of_others(self, stack):
        with pytest.raises(ValueError, match="for sample 3 alone"):
            extract_sipnet_parameter_at_coords(
                self._table().sel(sample=3), "leaf_carbon_per_area", self._restacked_leaf_carbon(stack)
            )

    def test_sipnet_parameter_fields_on_a_dim_restacked_is_refused_as_a_stack(self, stack):
        with pytest.raises(ValueError, match="'run2' is a stack of \\['sample'\\]"):
            extract_sipnet_parameter_at_coords(
                self._table(), "leaf_carbon_per_area", self._restacked_leaf_carbon(stack)
            )

    def test_the_operators_refuse_one_sample_sipnet_parameter_fields_against_a_restack(self, stack, labels):
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack.expand_dims(driver_member=[0, 1], axis=1)
        restacked = crossed.map(
            lambda variable: stack_batch_dims(stack_batch_dims(variable, into="run"), into="run2")
        )
        with pytest.raises(ValueError, match="for sample 3 alone"):
            ComputeLeafAreaIndex()(
                restacked,
                dated_observed_values([1, 2], labels),
                sipnet_parameters=self._table().sel(sample=3),
            )

    def test_a_stack_whose_labels_were_dropped_is_refused_in_the_modules_words(self, stack):
        from sipnet_calibration.fields import stack_batch_dims

        crossed = stack["leaf_carbon"].expand_dims(driver_member=[0, 1], axis=1)
        unlabeled = stack_batch_dims(crossed, into="run").drop_vars("sample_label")
        with pytest.raises(ValueError, match="no 'sample_label'"):
            extract_sipnet_parameter_at_coords(
                self._table().sel(sample=3), "leaf_carbon_per_area", unlabeled
            )

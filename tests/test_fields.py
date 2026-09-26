"""Tests for the model-output adapters.

Everything here runs against **real SIPNET output**: pySIPNET's committed
golden baseline for the Niwot Ridge reference inputs, and, where the driver
file and a compiled binary are both present, a real run of those parameters on
this copy's 3-hourly site-1 drivers. A synthetic frame would make the adapter
agree with a fixture of this project's own writing, which is exactly what these
tests are supposed to rule out: the names, units and kinds are pySIPNET's, and
a change there should show up here.

The site table is the real one, so the ``lon``/``lat`` the adapter attaches are
checked against ``sipnet_calibration.sites.load_sites`` rather than against
constants copied out of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import site_table_of
from sipnet_calibration.conventions import TIME_COORD_NAMES
from sipnet_calibration.fields import (
    from_sipnet_output,
    stack_sipnet_outputs,
)
from sipnet_calibration.sites import site_lookup

#: The dims of a stack of runs over samples and sites.
FIELD_DIMS = ("sample", "site", "time")


def rescaled(output, factor: float):
    """A second run's worth of output, differing from *output* by a known factor.

    Real output kept real: only the values are scaled, so a stacked field's
    cells can be told apart by which run they came from.
    """
    from pysipnet.output import SIPNETOutput

    frame = output.pandas.copy()
    frame["net_ecosystem_exchange"] = frame["net_ecosystem_exchange"] * factor
    return SIPNETOutput.from_dataframe(frame, climate=output.climate, run_id=f"x{factor}")


class TestFromSipnetOutput:
    def test_returns_one_field_per_variable_under_pysipnets_names(self, niwot_output):
        fields = from_sipnet_output(niwot_output, ["nee", "wood_carbon"])
        assert list(fields) == ["net_ecosystem_exchange", "wood_carbon"]
        assert all(isinstance(field, xr.DataArray) for field in fields.values())

    def test_aliases_and_registry_names_are_the_same_variable(self, niwot_output):
        by_alias = from_sipnet_output(niwot_output, ["NEE"])["net_ecosystem_exchange"]
        by_name = from_sipnet_output(niwot_output, ["net_ecosystem_exchange"])[
            "net_ecosystem_exchange"
        ]
        xr.testing.assert_identical(by_alias, by_name)

    def test_the_requested_order_is_kept_and_repeats_dropped(self, niwot_output):
        fields = from_sipnet_output(niwot_output, ["wood_carbon", "nee", "NEE"])
        assert list(fields) == ["wood_carbon", "net_ecosystem_exchange"]

    def test_an_unlabeled_run_is_a_time_series(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        assert field.dims == ("time",)
        assert "site" not in field.coords and "sample" not in field.coords

    def test_pysipnets_attributes_survive_unchanged(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        expected = niwot_output["nee"].attrs
        assert field.attrs == expected
        # The three the rest of the project reads off them.
        assert field.attrs["units"] == expected["units"]
        assert field.attrs["kind"] == "timestep_total"
        assert field.attrs["long_name"]

    def test_the_time_axis_is_pysipnets_step_end_with_its_bounds_pair(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        assert set(field.coords) == {"time", "time_step_start", "time_step_length"}
        assert set(field.coords) == set(TIME_COORD_NAMES)
        assert field["time"].attrs["long_name"] == "End of timestep"
        starts = field["time_step_start"].values
        lengths = field["time_step_length"].values
        # [time_step_start, time] is the pair pySIPNET writes as time_bounds.
        assert (field["time"].values >= starts).all()
        assert ((field["time"].values - starts) <= lengths + np.timedelta64(60, "s")).all()

    def test_the_dangling_bounds_attribute_is_dropped(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        assert "bounds" in niwot_output.xarray["time"].attrs
        assert "bounds" not in field["time"].attrs

    def test_sipnets_own_row_labels_are_dropped(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        for label in ("year", "day_of_year", "hour_of_day"):
            assert label not in field.coords

    def test_a_site_label_brings_its_coordinates_from_the_site_table(
        self, niwot_output, real_site_table
    ):
        field = from_sipnet_output(niwot_output, ["nee"], site=27)["net_ecosystem_exchange"]
        row = real_site_table.loc[real_site_table["site_id"] == 27].iloc[0]
        assert int(field["site"]) == 27
        assert field["site"].dtype == np.int32
        assert float(field["lon"]) == pytest.approx(row["lon"])
        assert float(field["lat"]) == pytest.approx(row["lat"])
        assert field["lon"].attrs["units"] == "degrees_east"

    def test_a_batch_label_is_a_scalar_int64_coordinate(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"], batch={"sample": 0})[
            "net_ecosystem_exchange"
        ]
        assert int(field["sample"]) == 0
        assert field["sample"].dtype == np.int64
        assert field["sample"].dims == ()

    def test_a_result_and_the_output_inside_it_give_the_same_thing(self, site_1_result):
        from_result = from_sipnet_output(site_1_result, ["nee"])["net_ecosystem_exchange"]
        from_output = from_sipnet_output(site_1_result.outputs, ["nee"])[
            "net_ecosystem_exchange"
        ]
        xr.testing.assert_identical(from_result, from_output)

    def test_the_values_are_the_column_sipnet_wrote(self, niwot_output):
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        assert np.array_equal(
            field.values, niwot_output.pandas["net_ecosystem_exchange"].to_numpy()
        )

    def test_the_whole_frame_is_never_read(self, niwot_output, monkeypatch):
        """``select`` is the route, so an ensemble holds one column per run."""
        from pysipnet.output import SIPNETOutput

        def refuse(self):
            raise AssertionError("the adapter read the whole output")

        monkeypatch.setattr(SIPNETOutput, "xarray", property(refuse))
        monkeypatch.setattr(SIPNETOutput, "pandas", property(refuse))
        field = from_sipnet_output(niwot_output, ["nee"])["net_ecosystem_exchange"]
        assert field.sizes["time"] > 0

    def test_an_unknown_variable_names_the_nearest_matches(self, niwot_output):
        with pytest.raises(KeyError, match="not a SIPNET output variable"):
            from_sipnet_output(niwot_output, ["net_exchange"])

    def test_no_variables_is_refused(self, niwot_output):
        with pytest.raises(ValueError, match="no variables were asked for"):
            from_sipnet_output(niwot_output, [])

    def test_a_site_outside_the_table_is_refused(self, niwot_output, real_site_table):
        with pytest.raises(KeyError, match="not in the site table"):
            from_sipnet_output(niwot_output, ["nee"], site=99999, sites=real_site_table)

    def test_a_batch_label_may_be_any_integer(self, niwot_output):
        for label in (-1, 40000):
            field = from_sipnet_output(niwot_output, ["nee"], batch={"sample": label})[
                "net_ecosystem_exchange"
            ]
            assert int(field["sample"]) == label

    def test_a_reserved_name_cannot_be_a_batch_dim(self, niwot_output):
        for name in ("site", "time", "lat"):
            with pytest.raises(ValueError, match="cannot name a batch dim"):
                from_sipnet_output(niwot_output, ["nee"], batch={name: 0})

    def test_something_that_is_not_a_run_is_refused(self, real_site_table):
        with pytest.raises(TypeError, match="SIPNETResult or SIPNETOutput"):
            from_sipnet_output(object(), ["nee"])


@pytest.fixture(scope="session")
def runs(niwot_output):
    """Four runs: two samples by two sites, each a known multiple of the golden NEE."""
    return {
        (sample, site): rescaled(niwot_output, 1 + site / 100 + sample / 1000)
        for site in (1, 27)
        for sample in (0, 1)
    }


class TestStackSipnetOutputs:
    def test_the_dims_are_the_field_order_and_ascending(self, runs, real_site_table):
        field = stack_sipnet_outputs(runs, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        assert field.dims == FIELD_DIMS
        assert list(field["site"].values) == [1, 27]
        assert list(field["sample"].values) == [0, 1]
        assert field["sample"].dtype == np.int64

    def test_lon_and_lat_are_on_site(self, runs, real_site_table):
        field = stack_sipnet_outputs(runs, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        assert field["lon"].dims == ("site",)
        assert field["lat"].dims == ("site",)
        expected = real_site_table.set_index("site_id").loc[[1, 27]]
        assert field["lon"].values == pytest.approx(expected["lon"].to_numpy())

    def test_each_cell_holds_the_run_it_was_labeled_with(self, runs, real_site_table):
        field = stack_sipnet_outputs(runs, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        for (sample, site), run in runs.items():
            assert np.array_equal(
                field.sel(site=site, sample=sample).values,
                run.select(["nee"])["net_ecosystem_exchange"].values,
            )

    def test_one_run_still_has_both_sample_dims(self, niwot_output, real_site_table):
        field = stack_sipnet_outputs({(0, 1): niwot_output}, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        assert field.dims == FIELD_DIMS
        assert field.sizes["sample"] == 1 and field.sizes["site"] == 1
        assert field["lon"].dims == ("site",)

    def test_a_pair_that_was_not_supplied_is_missing(self, niwot_output, real_site_table):
        runs = {(0, 1): niwot_output, (1, 27): niwot_output}
        field = stack_sipnet_outputs(runs, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        assert field.sizes["sample"] == 2 and field.sizes["site"] == 2
        assert np.isnan(field.sel(site=27, sample=0).values).all()
        assert np.isfinite(field.sel(site=27, sample=1).values).all()

    def test_the_attributes_are_still_pysipnets(self, runs, real_site_table):
        field = stack_sipnet_outputs(runs, ["nee"], sites=real_site_table)[
            "net_ecosystem_exchange"
        ]
        assert field.attrs == runs[(0, 1)]["nee"].attrs

    def test_runs_on_different_time_axes_are_joined_outward(self, niwot_output, real_site_table):
        from pysipnet.output import SIPNETOutput

        short = SIPNETOutput.from_dataframe(
            niwot_output.pandas.iloc[:20].copy(), climate=niwot_output.climate.head(20)
        )
        field = stack_sipnet_outputs(
            {(0, 1): niwot_output, (0, 27): short}, ["nee"], sites=real_site_table
        )["net_ecosystem_exchange"]
        # The axes are unioned, so each site is finite over its own record and
        # missing outside it. They are not nested: a truncated run's last step
        # ends at its declared length, where the full run's snapped to the next
        # row's start, so the union is longer than either.
        assert field.sizes["time"] >= niwot_output.n_timesteps
        assert int(np.isfinite(field.sel(site=1)).sum()) == niwot_output.n_timesteps
        assert int(np.isfinite(field.sel(site=27)).sum()) == 20
        assert np.isnan(field.sel(site=27, sample=0).values[-1])
        # The interval coordinates then describe each site separately, which is
        # what aggregate_time refuses rather than guess a cell's length from.
        assert set(field["time_step_length"].dims) == {"site", "time"}

    def test_not_a_mapping_is_refused(self, niwot_output):
        with pytest.raises(TypeError, match="must be a mapping"):
            stack_sipnet_outputs([niwot_output], ["nee"])

    def test_an_empty_mapping_is_refused(self):
        with pytest.raises(ValueError, match="nothing to stack"):
            stack_sipnet_outputs({}, ["nee"])

    def test_a_key_that_is_not_a_tuple_is_refused(self, niwot_output, real_site_table):
        with pytest.raises(ValueError, match=r"tuple of labels in key_dims order"):
            stack_sipnet_outputs({1: niwot_output}, ["nee"], sites=real_site_table)


class TestTheAdaptersTakeAKeyedSiteTable:
    def test_the_adapter_accepts_either_form(self, niwot_output, real_site_table):
        plain = from_sipnet_output(niwot_output, ["nee"], site=27, sites=real_site_table)
        keyed = from_sipnet_output(
            niwot_output, ["nee"], site=27, sites=site_lookup(real_site_table)
        )
        xr.testing.assert_identical(
            plain["net_ecosystem_exchange"], keyed["net_ecosystem_exchange"]
        )


class TestFromSipnetOutputRefusesBadInput:
    def test_a_run_that_wrote_no_rows_says_so(self, niwot_output):
        """The shape a failed run leaves; pySIPNET gives back an empty Dataset."""
        from pysipnet.output import SIPNETOutput

        empty = SIPNETOutput.from_dataframe(niwot_output.pandas.iloc[0:0].copy())
        with pytest.raises(ValueError, match="no rows"):
            from_sipnet_output(empty, ["nee"])

    def test_a_float_site_is_refused_even_a_whole_one(self, niwot_output, real_site_table):
        for site in (1.5, 27.0):
            with pytest.raises(TypeError, match="float"):
                from_sipnet_output(niwot_output, ["nee"], site=site, sites=real_site_table)

    def test_a_boolean_is_not_an_identifier(self, niwot_output, real_site_table):
        with pytest.raises(TypeError, match="bool"):
            from_sipnet_output(niwot_output, ["nee"], site=True, sites=real_site_table)
        with pytest.raises(TypeError, match="boolean"):
            from_sipnet_output(niwot_output, ["nee"], batch={"sample": False})

    def test_a_float_batch_label_is_refused_as_a_type_error(self, niwot_output):
        for label in (float("inf"), 2.0):
            with pytest.raises(TypeError, match="must be an integer"):
                from_sipnet_output(niwot_output, ["nee"], batch={"sample": label})

    def test_batch_labels_that_are_not_a_mapping_are_refused(self, niwot_output):
        with pytest.raises(TypeError, match="mapping from batch dim to label"):
            from_sipnet_output(niwot_output, ["nee"], batch=3)

    def test_one_bare_name_is_refused_as_a_type_error(self, niwot_output):
        with pytest.raises(TypeError, match=r"one string 'nee'; pass a sequence such as \['nee'\]"):
            from_sipnet_output(niwot_output, "nee")

    def test_variables_given_as_none_is_refused_as_a_type_error(self, niwot_output):
        with pytest.raises(TypeError, match="must be a sequence of names"):
            from_sipnet_output(niwot_output, None)

    def test_an_unordered_container_is_refused_because_order_is_promised(self, niwot_output):
        with pytest.raises(TypeError, match="no\\s+order to keep"):
            from_sipnet_output(niwot_output, {"nee", "wood_carbon"})

    def test_a_result_whose_outputs_are_not_an_output_is_refused(self, niwot_output):
        class NotAResult:
            outputs = 42

        with pytest.raises(TypeError, match="rather than a SIPNETOutput"):
            from_sipnet_output(NotAResult(), ["nee"])

    def test_a_repeated_variable_is_read_once(self, niwot_output, monkeypatch):
        """``select`` must not be handed the same column twice."""
        from pysipnet.output import SIPNETOutput

        seen = []
        original = SIPNETOutput.select

        def record(self, variables, **kwargs):
            seen.append(list(variables))
            return original(self, variables, **kwargs)

        monkeypatch.setattr(SIPNETOutput, "select", record)
        from_sipnet_output(niwot_output, ["nee", "NEE", "net_ecosystem_exchange"])
        assert seen == [["net_ecosystem_exchange"]]

    def test_a_site_lookup_without_the_column_still_explains_a_missing_site(
        self, real_site_table, niwot_output
    ):
        keyed = real_site_table.set_index("site_id")
        with pytest.raises(KeyError, match="not in the site table"):
            from_sipnet_output(niwot_output, ["nee"], site=99999, sites=keyed)


class TestStackSipnetOutputsRefusesBadKeys:
    def test_a_key_of_the_wrong_arity_names_the_contract(self, niwot_output, real_site_table):
        with pytest.raises(ValueError, match=r"key_dims order \('sample', 'site'\)"):
            stack_sipnet_outputs({(1, 0, 7): niwot_output}, ["nee"], sites=real_site_table)

    def test_key_dims_must_name_the_site_once(self, niwot_output, real_site_table):
        for key_dims in (("sample",), ("site", "site"), ("sample", "sample", "site")):
            with pytest.raises(ValueError, match="must name 'site' once"):
                stack_sipnet_outputs(
                    {(0,): niwot_output}, ["nee"], key_dims=key_dims, sites=real_site_table
                )


class TestLabelRun:
    def _table(self):
        return site_table_of(1, 27, lon=[-105.0, -70.0], lat=[40.0, 45.0], keyed=True)

    def test_adds_the_labels_and_nothing_else(self, niwot_output):
        from sipnet_calibration.fields import label_run

        dataset = niwot_output.select(["wood_carbon"])
        labeled = label_run(dataset, site=27, batch={"sample": 3}, site_table=self._table())
        assert int(labeled["site"]) == 27 and int(labeled["sample"]) == 3
        assert float(labeled["lon"]) == -70.0 and float(labeled["lat"]) == 45.0
        assert labeled["site"].dtype == np.int32 and labeled["sample"].dtype == np.int64
        xr.testing.assert_identical(labeled.drop_vars(["site", "sample", "lon", "lat"]), dataset)

    def test_labels_every_batch_dim_given(self, niwot_output):
        from sipnet_calibration.fields import label_run

        dataset = niwot_output.select(["wood_carbon"])
        labeled = label_run(dataset, batch={"sample": 3, "driver_member": 7})
        assert int(labeled["sample"]) == 3 and int(labeled["driver_member"]) == 7

    def test_an_empty_run_is_refused(self, niwot_output):
        from sipnet_calibration.fields import label_run

        with pytest.raises(ValueError, match="no rows"):
            label_run(niwot_output.select(["wood_carbon"]).isel(time=slice(0, 0)), site=1, site_table=self._table())

    def test_a_site_table_with_a_repeated_site_is_refused(self, niwot_output):
        from sipnet_calibration.fields import label_run

        table = pd.concat([self._table(), self._table().iloc[[0]]])
        with pytest.raises(ValueError, match="more than once"):
            label_run(niwot_output.select(["wood_carbon"]), site=1, site_table=table)

    def test_no_labels_returns_the_dataset_unchanged(self, niwot_output):
        from sipnet_calibration.fields import label_run

        dataset = niwot_output.select(["wood_carbon"])
        xr.testing.assert_identical(label_run(dataset), dataset)

    def test_a_dataarray_is_refused(self, niwot_output):
        from sipnet_calibration.fields import label_run

        with pytest.raises(TypeError, match="Dataset"):
            label_run(niwot_output.select(["wood_carbon"])["wood_carbon"], site=1, site_table=self._table())


class TestStackModelOutputs:
    def _table(self):
        return site_table_of(1, 27, lon=[-105.0, -70.0], lat=[40.0, 45.0], keyed=True)

    def test_is_the_dataset_stack_sipnet_outputs_splits_into_fields(self, runs):
        from sipnet_calibration.fields import stack_model_outputs

        datasets = {key: run.select(["nee", "wood_carbon"]) for key, run in runs.items()}
        stacked = stack_model_outputs(datasets, site_table=self._table())
        fields = stack_sipnet_outputs(runs, ["nee", "wood_carbon"], sites=self._table())
        assert isinstance(stacked, xr.Dataset)
        assert stacked["net_ecosystem_exchange"].dims == FIELD_DIMS
        for name, field in fields.items():
            xr.testing.assert_identical(stacked[name], field)
        assert stacked["lon"].attrs["units"] == "degrees_east"
        assert set(stacked.coords) == {"sample", "site", "lon", "lat", *TIME_COORD_NAMES}
        assert "bounds" not in stacked["time"].attrs

    def test_a_key_that_is_not_a_pair_of_integers_is_refused(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["nee"])
        for key in ((0, 1.5), (0, 1.0), (0.0, 1)):
            with pytest.raises(TypeError, match="float"):
                stack_model_outputs({key: run}, site_table=self._table())
        for key in ((0, True), (False, 1)):
            with pytest.raises(TypeError, match="boolean"):
                stack_model_outputs({key: run}, site_table=self._table())
        stacked = stack_model_outputs({(np.int16(0), np.int64(27)): run}, site_table=self._table())
        assert stacked["site"].values.tolist() == [27]

    def test_batch_labels_may_be_any_distinct_integers(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["nee"])
        stacked = stack_model_outputs(
            {(-1, 1): run, (40000, 1): run}, site_table=self._table()
        )
        assert stacked["sample"].values.tolist() == [-1, 40000]
        assert stacked["sample"].dtype == np.int64

    def test_several_batch_dims_are_stacked_in_key_dims_order(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs, validate_field

        run = niwot_output.select(["nee"])
        model_outputs = {
            (sample, member, site): run * (1 + sample + 10 * member)
            for sample in (0, 1)
            for member in (3, 5)
            for site in (1, 27)
        }
        stacked = stack_model_outputs(
            model_outputs,
            key_dims=("sample", "driver_member", "site"),
            site_table=self._table(),
        )
        assert stacked["net_ecosystem_exchange"].dims == ("sample", "driver_member", "site", "time")
        assert stacked["driver_member"].values.tolist() == [3, 5]
        validate_field(stacked["net_ecosystem_exchange"])
        np.testing.assert_allclose(
            stacked["net_ecosystem_exchange"].sel(sample=1, driver_member=5, site=27).values,
            (run * 52)["net_ecosystem_exchange"].values,
        )

    def test_one_run_per_site_needs_no_batch_dim(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["nee"])
        stacked = stack_model_outputs(
            {(1,): run, (27,): run}, key_dims=("site",), site_table=self._table()
        )
        assert stacked["net_ecosystem_exchange"].dims == ("site", "time")

    def test_runs_labeled_by_label_run_are_stacked_and_left_unchanged(self, niwot_output):
        from sipnet_calibration.fields import label_run, stack_model_outputs

        labeled = {
            (0, site): label_run(niwot_output.select(["nee"]), site=site, site_table=self._table())
            for site in (1, 27)
        }
        before = {key: dataset.copy(deep=True) for key, dataset in labeled.items()}
        stacked = stack_model_outputs(labeled, site_table=self._table())
        assert stacked["site"].values.tolist() == [1, 27] and stacked["sample"].values.tolist() == [0]
        for key, dataset in labeled.items():
            xr.testing.assert_identical(dataset, before[key])

    def test_a_run_labeled_with_another_site_is_refused(self, niwot_output):
        from sipnet_calibration.fields import label_run, stack_model_outputs

        labeled = label_run(niwot_output.select(["nee"]), site=27, site_table=self._table())
        with pytest.raises(ValueError, match=r"keyed \(sample=0, site=1\) is labeled site=\[27\]"):
            stack_model_outputs({(0, 1): labeled}, site_table=self._table())

    def test_a_run_labeled_with_another_batch_label_is_refused(self, niwot_output):
        from sipnet_calibration.fields import label_run, stack_model_outputs

        labeled = label_run(niwot_output.select(["nee"]), batch={"sample": 4})
        with pytest.raises(ValueError, match=r"keyed \(sample=0, site=1\) is labeled sample=\[4\]"):
            stack_model_outputs({(0, 1): labeled}, site_table=self._table())

    def test_a_run_with_no_rows_is_refused(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        empty = niwot_output.select(["nee"]).isel(time=slice(0, 0))
        with pytest.raises(ValueError, match="no rows"):
            stack_model_outputs({(0, 1): empty}, site_table=self._table())

    def test_something_that_is_not_a_dataset_is_refused(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        with pytest.raises(TypeError, match="Dataset"):
            stack_model_outputs({(0, 1): niwot_output}, site_table=self._table())


class TestResolveOutputVariableNames:
    def test_resolves_aliases_in_order_without_repeats(self):
        from sipnet_calibration.fields import resolve_output_variable_names

        assert resolve_output_variable_names(["wood_carbon", "nee", "NEE"]) == ["wood_carbon", "net_ecosystem_exchange"]
        assert resolve_output_variable_names(["nee"]) == ["net_ecosystem_exchange"]

    def test_a_name_that_is_not_a_string_is_a_type_error(self):
        from sipnet_calibration.fields import resolve_output_variable_names

        with pytest.raises(TypeError, match="must be strings"):
            resolve_output_variable_names(["nee", 3])


class TestFieldLabel:
    def test_a_name_then_a_derivation_then_the_default(self):
        from sipnet_calibration.fields import field_label

        assert field_label(xr.DataArray(0.0, name="wood_carbon")) == "'wood_carbon'"
        assert field_label(xr.DataArray(0.0, attrs={"derivation": "a / b"})) == "'a / b'"
        assert field_label(xr.DataArray(0.0)) == "the field"
        assert field_label(xr.DataArray(0.0), "the observation") == "the observation"


def stack_model_outputs_of(niwot_output, sites=(1, 27), n_samples=2):
    """Niwot's wood carbon stacked over ``(sample, site)``, every run the same."""
    from sipnet_calibration.fields import stack_model_outputs

    run = niwot_output.select(["wood_carbon"])
    runs = {(sample, site): run for sample in range(n_samples) for site in sites}
    return stack_model_outputs(runs, site_table=_small_table(*sites))


def _small_table(*sites):
    return site_table_of(*sites, lon=[-105.0 + s for s in sites], lat=40.0)


class TestStackModelOutputsRefusesRunsThatDisagree:
    def test_runs_carrying_different_variables_are_refused(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        wood = niwot_output.select(["wood_carbon"])
        both = niwot_output.select(["wood_carbon", "nee"])
        for model_outputs in ({(0, 1): wood, (0, 27): both}, {(0, 1): both, (1, 1): wood}):
            with pytest.raises(ValueError, match="carries the variables"):
                stack_model_outputs(model_outputs, site_table=_small_table(1, 27))

    @pytest.mark.parametrize("attribute, value", [("units", "kg m-2"), ("constituent", "N"), ("kind", "timestep_mean")])
    def test_runs_describing_a_variable_differently_are_refused(self, niwot_output, attribute, value):
        from sipnet_calibration.fields import stack_model_outputs

        wood = niwot_output.select(["wood_carbon"])
        other = wood.copy()
        other["wood_carbon"].attrs = {**wood["wood_carbon"].attrs, attribute: value}
        with pytest.raises(ValueError, match=f"describes 'wood_carbon' as .*'{attribute}': '{value}'"):
            stack_model_outputs({(0, 1): wood, (0, 27): other}, site_table=_small_table(1, 27))

    def test_runs_given_out_of_order_are_stacked_ascending(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["wood_carbon"])
        stacked = stack_model_outputs({(0, 27): run * 2, (0, 1): run}, site_table=_small_table(1, 27))
        assert stacked["site"].values.tolist() == [1, 27]
        assert stacked["lon"].values.tolist() == [-104.0, -78.0]
        np.testing.assert_allclose(stacked["wood_carbon"].sel(sample=0, site=27).values, 2 * run["wood_carbon"].values)

    def test_each_site_carries_its_own_location_whatever_the_pairs(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["wood_carbon"])
        table = _small_table(1, 27, 500)
        for keys in (
            [(0, 500), (1, 27), (1, 1)],
            [(1, 27), (0, 1), (2, 500)],
            [(0, 500), (1, 1)],
        ):
            stacked = stack_model_outputs({key: run for key in keys}, site_table=table)
            expected = site_lookup(table).loc[stacked["site"].values, "lon"].to_numpy()
            np.testing.assert_array_equal(stacked["lon"].values, expected)

    def test_a_run_labeled_with_several_sites_is_refused(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot_output.select(["wood_carbon"]).expand_dims(site=[1, 27])
        with pytest.raises(ValueError, match=r"is labeled site=\[1, 27\]"):
            stack_model_outputs({(0, 1): run}, site_table=_small_table(1, 27))

    def test_something_that_is_not_a_mapping_is_named(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        with pytest.raises(TypeError, match="^model_outputs must be a mapping"):
            stack_model_outputs([niwot_output.select(["nee"])], site_table=_small_table(1))


class TestLabelRunChecksTheSiteTable:
    def test_label_run_applies_it(self, niwot_output):
        from sipnet_calibration.fields import label_run

        with pytest.raises(ValueError, match="'lat'"):
            label_run(niwot_output.select(["nee"]), site=1, site_table=_small_table(1).drop(columns="lat"))


class TestLabelHelpers:
    def test_missing_labels_keeps_the_order_asked_for(self):
        from sipnet_calibration.fields import missing_labels

        source = xr.DataArray(np.zeros(2), dims="site", coords={"site": [1, 2]})
        assert missing_labels(source, "site", [9, 3, 1]) == [9, 3]
        assert missing_labels(source, "site", [2, 1]) == []

    def test_coordinate_labels_of_a_dimension_and_a_scalar(self):
        from sipnet_calibration.fields import coordinate_labels

        assert coordinate_labels(xr.DataArray([0, 0], dims="site", coords={"site": [3, 1]})["site"]) == [3, 1]
        assert coordinate_labels(xr.Dataset(coords={"site": 5})["site"]) == [5]

    def test_field_label_unquoted(self):
        from sipnet_calibration.fields import field_label

        assert field_label(xr.DataArray(0.0, name="wood_carbon"), quoted=False) == "wood_carbon"
        assert field_label(xr.DataArray(0.0, attrs={"derivation": "a / b"}), quoted=False) == "a / b"
        assert field_label(xr.DataArray(0.0), "the observation", quoted=False) == "the observation"

    def test_without_stale_time_attributes(self):
        from sipnet_calibration.conventions import STALE_TIME_ATTRIBUTE_NAMES
        from sipnet_calibration.fields import without_stale_time_attributes

        assert STALE_TIME_ATTRIBUTE_NAMES == ("bounds",)
        assert without_stale_time_attributes({"bounds": "time_bounds", "axis": "T"}) == {"axis": "T"}

    def test_the_time_coordinates_are_the_named_constants(self):
        from sipnet_calibration.conventions import TIME, TIMESTEP_LENGTH, TIMESTEP_START

        assert TIME_COORD_NAMES == (TIME, TIMESTEP_START, TIMESTEP_LENGTH)
        assert (TIMESTEP_START, TIMESTEP_LENGTH) == ("time_step_start", "time_step_length")


class TestResolveOutputVariableNamesDelegates:
    def test_a_legacy_column_is_resolved_as_pysipnet_resolves_it(self):
        from pysipnet.variables import LEGACY_OUTPUT_COLUMNS

        from sipnet_calibration.fields import resolve_output_variable_names

        legacy, name = next(iter(LEGACY_OUTPUT_COLUMNS.items()))
        assert resolve_output_variable_names([legacy, name]) == [name]

    def test_a_set_and_a_non_iterable_are_type_errors(self):
        from sipnet_calibration.fields import resolve_output_variable_names

        with pytest.raises(TypeError, match="no order to keep"):
            resolve_output_variable_names({"nee"})
        with pytest.raises(TypeError, match="a sequence of names"):
            resolve_output_variable_names(3)
        with pytest.raises(ValueError, match="no variables were asked for"):
            resolve_output_variable_names(())


# ── the field contract ────────────────────────────────────────────────────────


def _field(dims, **kwargs):
    from conftest import make_field

    return make_field(dims, **kwargs)


class TestValidateField:
    def test_the_synthetic_fields_are_fields(self):
        from sipnet_calibration.fields import validate_field

        for dims in (("time",), ("site",), ("sample", "site", "time"), ("sample", "time")):
            validate_field(_field(dims))

    def test_an_integer_labeled_extra_dim_is_a_batch_dim(self):
        from sipnet_calibration.fields import batch_dims, validate_field

        field = _field(("chain", "draw", "site", "time"))
        validate_field(field)
        assert batch_dims(field) == ("chain", "draw")

    @pytest.mark.parametrize(
        "labels, dtype", [(["a", "b", "c"], object), ([0.05, 0.5, 0.95], float)]
    )
    def test_a_string_or_float_labeled_extra_dim_is_refused(self, labels, dtype):
        from sipnet_calibration.fields import batch_dims, validate_field

        field = _field(("quantile", "site")).assign_coords(quantile=np.asarray(labels, dtype))
        assert batch_dims(field) == ()
        with pytest.raises(ValueError, match=r"\['quantile'\] are neither a batch dim"):
            validate_field(field)

    def test_an_extra_dim_without_a_coordinate_is_refused(self):
        from sipnet_calibration.fields import batch_dims, validate_field

        field = _field(("sample", "site")).drop_vars("sample")
        assert batch_dims(field) == ()
        with pytest.raises(ValueError, match="carry no coordinate"):
            validate_field(field)

    def test_the_spatial_names_are_never_batch_dims(self):
        from sipnet_calibration.fields import batch_dims

        point = xr.DataArray(
            np.zeros((2, 3)),
            dims=("sample", "point"),
            coords={"sample": [0, 1], "point": [0, 1, 2]},
        )
        assert batch_dims(point) == ("sample",)

    def test_dims_out_of_order_are_refused(self):
        from sipnet_calibration.fields import validate_field

        with pytest.raises(ValueError, match=r"not in the order \(\*batch, space, time\)"):
            validate_field(_field(("sample", "site", "time")).transpose("site", "sample", "time"))

    def test_two_spatial_dims_are_refused(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("site",)).expand_dims(point=[0, 1])
        with pytest.raises(ValueError, match="at most one"):
            validate_field(field)

    def test_site_ids_must_be_unique_int32(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("site",))
        with pytest.raises(ValueError, match="site ids are int32, got int64"):
            validate_field(field.assign_coords(site=field["site"].astype(np.int64)))
        repeated = field.assign_coords(site=np.asarray([1, 1], np.int32))
        with pytest.raises(ValueError, match="repeats a site id"):
            validate_field(repeated)

    def test_a_site_dim_carries_float64_lon_and_lat(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("site",))
        with pytest.raises(ValueError, match="no 'lat' coordinate"):
            validate_field(field.drop_vars("lat"))
        with pytest.raises(ValueError, match="'lon' must be float64"):
            validate_field(field.assign_coords(lon=field["lon"].astype(np.float32)))

    def test_time_is_strictly_increasing_and_has_no_nat(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("time",), n_time=4)
        with pytest.raises(ValueError, match="not strictly increasing"):
            validate_field(field.isel(time=[0, 2, 1, 3]))
        times = field["time"].values.copy()
        times[1] = np.datetime64("NaT")
        with pytest.raises(ValueError, match="NaT"):
            validate_field(field.assign_coords(time=times))

    def test_the_interval_coordinates_are_on_time_alone(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("site", "time"), n_time=4)
        start = xr.DataArray(
            np.broadcast_to(field["time"].values, (2, 4)), dims=("site", "time")
        )
        with pytest.raises(ValueError, match="'time_step_start' is on"):
            validate_field(field.assign_coords(time_step_start=start))

    def test_units_are_required_and_checked_by_pysipnet(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("time",))
        no_units = field.copy()
        no_units.attrs = {}
        with pytest.raises(ValueError, match="attrs\\['units'\\]"):
            validate_field(no_units)
        substance = field.copy()
        substance.attrs = {"units": "g C m-2"}
        with pytest.raises(ValueError):
            validate_field(substance)
        categorical = field.copy()
        categorical.attrs = {"flag_values": [0, 1], "flag_meanings": "a b"}
        validate_field(categorical)

    def test_repeated_batch_labels_are_refused(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("sample", "time")).assign_coords(sample=[0, 0, 1])
        with pytest.raises(ValueError, match="repeats a label"):
            validate_field(field)

    def test_labels_need_not_start_at_zero_and_may_exceed_int16(self):
        from sipnet_calibration.fields import validate_field

        validate_field(_field(("sample", "time")).assign_coords(sample=[5, 40000, -2]))

    def test_a_scalar_batch_coordinate_is_metadata(self):
        from sipnet_calibration.fields import batch_dims, validate_field

        field = _field(("sample", "site", "time")).isel(sample=1)
        validate_field(field)
        assert batch_dims(field) == ()
        assert int(field["sample"]) == 1

    def test_something_that_is_not_a_dataarray_is_a_type_error(self):
        from sipnet_calibration.fields import validate_field

        with pytest.raises(TypeError, match="a field is an xarray DataArray"):
            validate_field(_field(("time",)).to_dataset())

    def test_the_interval_coordinates_are_scalars_at_one_time(self, niwot_output):
        """``.isel(time=k)`` leaves the timestep coordinates scalar, as a map needs."""
        from sipnet_calibration.fields import validate_field

        field = stack_model_outputs_of(niwot_output)["wood_carbon"].isel(sample=0)
        validate_field(field.isel(time=-1))
        validate_field(field.sel(time=field["time"].values[3]))
        windowed = _field(("site", "time"), n_time=3).assign_coords(
            time_bounds_start=("time", pd.date_range("2011-01-01", periods=3, freq="D")),
            time_bounds_end=("time", pd.date_range("2011-01-02", periods=3, freq="D")),
        )
        validate_field(windowed.isel(time=0))

    def test_a_scalar_interval_coordinate_beside_a_time_dim_is_refused(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("site", "time"), n_time=3).assign_coords(
            time_step_start=np.datetime64("2012-01-01")
        )
        with pytest.raises(ValueError, match="'time_step_start' is on \\(\\)"):
            validate_field(field)

    def test_a_time_zone_aware_time_is_refused_as_not_naive(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("time",), n_time=3)
        aware = field.assign_coords(time=pd.date_range("2012-01-01", periods=3, tz="UTC"))
        with pytest.raises(ValueError, match="naive datetime64"):
            validate_field(aware)

    def test_point_labels_are_integers(self):
        from sipnet_calibration.fields import validate_field

        point = xr.DataArray(
            np.zeros(2),
            dims="point",
            coords={"point": ["a", "b"], "lon": ("point", [0.0, 1.0]), "lat": ("point", [0.0, 1.0])},
            attrs={"units": "1"},
        )
        with pytest.raises(ValueError, match="point labels are integers"):
            validate_field(point)
        validate_field(point.assign_coords(point=[0, 1]))

    def test_lon_and_lat_are_on_the_spatial_dim_or_scalars_beside_a_scalar_site(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("sample", "site"))
        one_site = field.isel(site=0)
        validate_field(one_site)
        on_sample = one_site.assign_coords(lon=("sample", [1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="'lon' must be a float64 scalar"):
            validate_field(on_sample)
        with pytest.raises(ValueError, match="a scalar site carries a scalar 'lat'"):
            validate_field(one_site.drop_vars("lat"))

    def test_an_object_array_is_categorical_only_when_it_holds_strings(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("time",), n_time=3)
        floats = field.copy(data=np.asarray([1.0, 2.0, 3.0], dtype=object))
        floats.attrs = {}
        with pytest.raises(ValueError, match="attrs\\['units'\\]"):
            validate_field(floats)
        strings = field.copy(data=np.asarray(["a", "b", "c"], dtype=object))
        strings.attrs = {}
        validate_field(strings)

    def test_a_units_refusal_names_the_field(self):
        from sipnet_calibration.fields import validate_field

        field = _field(("time",))
        field.attrs = {"units": "g C m-2"}
        with pytest.raises(ValueError, match="^'air_temperature': "):
            validate_field(field)

    def test_batch_labels_of_any_integer_dtype_are_accepted(self):
        from sipnet_calibration.fields import validate_field

        for dtype in (np.int16, np.uint8, np.int64):
            field = _field(("sample", "time"))
            validate_field(field.assign_coords(sample=field["sample"].values.astype(dtype)))


class TestStackBatchDims:
    def _two_batch_dims(self):
        field = _field(("sample", "initial_condition_member", "site", "time"), n_time=4)
        return field.assign_coords(sample=[5, 9, 12])

    def test_stacks_into_one_labeled_zero_to_n(self):
        from sipnet_calibration.fields import stack_batch_dims, validate_field

        field = self._two_batch_dims()
        stacked = stack_batch_dims(field)
        validate_field(stacked)
        assert stacked.dims == ("sample", "site", "time")
        assert stacked["sample"].values.tolist() == list(range(9))
        assert stacked["sample"].dtype == np.int64
        assert stacked["sample_label"].values.tolist() == [5, 5, 5, 9, 9, 9, 12, 12, 12]
        assert stacked["initial_condition_member_label"].values.tolist() == [0, 1, 2] * 3
        np.testing.assert_array_equal(
            stacked.isel(sample=4).values,
            field.sel(sample=9, initial_condition_member=1).values,
        )

    def test_unstack_reverses_it(self):
        from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

        field = self._two_batch_dims()
        xr.testing.assert_identical(
            unstack_batch_dims(stack_batch_dims(field)).drop_attrs(deep=False),
            field.drop_attrs(deep=False),
        )

    def test_into_another_name(self):
        from sipnet_calibration.fields import stack_batch_dims, unstack_batch_dims

        stacked = stack_batch_dims(self._two_batch_dims(), into="run")
        assert stacked.dims == ("run", "site", "time")
        assert unstack_batch_dims(stacked).dims == (
            "sample", "initial_condition_member", "site", "time"
        )

    def test_a_field_without_a_batch_dim_or_a_bad_name_is_refused(self):
        from sipnet_calibration.fields import stack_batch_dims

        with pytest.raises(ValueError, match="no batch dim"):
            stack_batch_dims(_field(("site", "time")))
        with pytest.raises(ValueError, match="cannot name a batch dim"):
            stack_batch_dims(self._two_batch_dims(), into="site")

    def test_unstack_needs_the_labels_the_stack_kept(self):
        from sipnet_calibration.fields import unstack_batch_dims

        with pytest.raises(ValueError, match="0 batch dims carry"):
            unstack_batch_dims(_field(("sample", "site", "time")))


class TestPyEnsPairsSameNamesAndCrossesDifferentOnes:
    """The rule the field contract takes from PyEns: a name is an index."""

    def _table(self, dim, size):
        return xr.Dataset(
            {"x": ((dim, "site"), np.zeros((size, 2)))},
            coords={dim: np.arange(size, dtype=np.int64), "site": np.asarray([1, 27], np.int32)},
        )

    def test_two_dims_of_one_name_zip(self):
        from pyens import EnsembleSpec
        from pyens.xarray import fields_from_dataset

        parameters = fields_from_dataset(self._table("sample", 2))
        drivers = fields_from_dataset(self._table("sample", 2).rename(x="y"))
        assert EnsembleSpec(inputs={**parameters, **drivers}).n_runs == 2 * 2

    def test_two_dims_of_different_names_cross(self):
        from pyens import EnsembleSpec
        from pyens.xarray import fields_from_dataset

        parameters = fields_from_dataset(self._table("sample", 2))
        drivers = fields_from_dataset(self._table("driver_member", 3).rename(x="y"))
        assert EnsembleSpec(inputs={**parameters, **drivers}).n_runs == 2 * 3 * 2

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
import pytest
import xarray as xr

from sipnet_calibration.fields import (
    CANONICAL_DIMS,
    TIME_COORDS,
    from_sipnet_output,
    site_lookup,
    stack_sipnet_outputs,
)


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
        by_alias = from_sipnet_output(niwot_output, "NEE")["net_ecosystem_exchange"]
        by_name = from_sipnet_output(niwot_output, ["net_ecosystem_exchange"])[
            "net_ecosystem_exchange"
        ]
        xr.testing.assert_identical(by_alias, by_name)

    def test_the_requested_order_is_kept_and_repeats_dropped(self, niwot_output):
        fields = from_sipnet_output(niwot_output, ["wood_carbon", "nee", "NEE"])
        assert list(fields) == ["wood_carbon", "net_ecosystem_exchange"]

    def test_an_unlabeled_run_is_a_time_series(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert field.dims == ("time",)
        assert "site" not in field.coords and "member" not in field.coords

    def test_pysipnets_attributes_survive_unchanged(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        expected = niwot_output["nee"].attrs
        assert field.attrs == expected
        # The three the rest of the project reads off them.
        assert field.attrs["units"] == expected["units"]
        assert field.attrs["kind"] == "timestep_total"
        assert field.attrs["long_name"]

    def test_the_time_axis_is_pysipnets_step_end_with_its_bounds_pair(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert set(field.coords) == {"time", "time_step_start", "time_step_length"}
        assert set(field.coords) == set(TIME_COORDS)
        assert field["time"].attrs["long_name"] == "End of timestep"
        starts = field["time_step_start"].values
        lengths = field["time_step_length"].values
        # [time_step_start, time] is the pair pySIPNET writes as time_bounds.
        assert (field["time"].values >= starts).all()
        assert ((field["time"].values - starts) <= lengths + np.timedelta64(60, "s")).all()

    def test_the_dangling_bounds_attribute_is_dropped(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert "bounds" in niwot_output.xarray["time"].attrs
        assert "bounds" not in field["time"].attrs

    def test_sipnets_own_row_labels_are_dropped(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        for label in ("year", "day_of_year", "hour_of_day"):
            assert label not in field.coords

    def test_a_site_label_brings_its_coordinates_from_the_site_table(
        self, niwot_output, sites_table
    ):
        field = from_sipnet_output(niwot_output, "nee", site=27)["net_ecosystem_exchange"]
        row = sites_table.loc[sites_table["site_id"] == 27].iloc[0]
        assert int(field["site"]) == 27
        assert field["site"].dtype == np.int32
        assert float(field["lon"]) == pytest.approx(row["lon"])
        assert float(field["lat"]) == pytest.approx(row["lat"])
        assert field["lon"].attrs["units"] == "degrees_east"

    def test_a_member_label_is_a_zero_based_integer(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee", member=0)["net_ecosystem_exchange"]
        assert int(field["member"]) == 0
        assert field["member"].dtype == np.int16

    def test_a_result_and_the_output_inside_it_give_the_same_thing(self, site_1_result):
        from_result = from_sipnet_output(site_1_result, "nee")["net_ecosystem_exchange"]
        from_output = from_sipnet_output(site_1_result.outputs, "nee")[
            "net_ecosystem_exchange"
        ]
        xr.testing.assert_identical(from_result, from_output)

    def test_the_values_are_the_column_sipnet_wrote(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
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
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert field.sizes["time"] > 0

    def test_an_unknown_variable_names_the_nearest_matches(self, niwot_output):
        with pytest.raises(KeyError, match="not a SIPNET output variable"):
            from_sipnet_output(niwot_output, "net_exchange")

    def test_no_variables_is_refused(self, niwot_output):
        with pytest.raises(ValueError, match="No variables were asked for"):
            from_sipnet_output(niwot_output, [])

    def test_a_site_outside_the_table_is_refused(self, niwot_output, sites_table):
        with pytest.raises(KeyError, match="not in the site table"):
            from_sipnet_output(niwot_output, "nee", site=99999, sites=sites_table)

    def test_a_negative_member_is_refused(self, niwot_output):
        with pytest.raises(ValueError, match="member must be at least 0"):
            from_sipnet_output(niwot_output, "nee", member=-1)

    def test_something_that_is_not_a_run_is_refused(self, sites_table):
        with pytest.raises(TypeError, match="SIPNETResult or SIPNETOutput"):
            from_sipnet_output(object(), "nee")


@pytest.fixture(scope="session")
def runs(niwot_output):
    """Four runs: two sites by two members, each a known multiple of the golden NEE."""
    return {
        (site, member): rescaled(niwot_output, 1 + site / 100 + member / 1000)
        for site in (1, 27)
        for member in (0, 1)
    }


class TestStackSipnetOutputs:
    def test_the_dims_are_canonical_and_ascending(self, runs, sites_table):
        field = stack_sipnet_outputs(runs, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        assert field.dims == CANONICAL_DIMS
        assert list(field["site"].values) == [1, 27]
        assert list(field["member"].values) == [0, 1]

    def test_lon_and_lat_are_on_site(self, runs, sites_table):
        field = stack_sipnet_outputs(runs, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        assert field["lon"].dims == ("site",)
        assert field["lat"].dims == ("site",)
        expected = sites_table.set_index("site_id").loc[[1, 27]]
        assert field["lon"].values == pytest.approx(expected["lon"].to_numpy())

    def test_each_cell_holds_the_run_it_was_labeled_with(self, runs, sites_table):
        field = stack_sipnet_outputs(runs, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        for (site, member), run in runs.items():
            assert np.array_equal(
                field.sel(site=site, member=member).values,
                run.select(["nee"])["net_ecosystem_exchange"].values,
            )

    def test_one_run_still_has_both_sample_dims(self, niwot_output, sites_table):
        field = stack_sipnet_outputs({(1, 0): niwot_output}, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        assert field.dims == CANONICAL_DIMS
        assert field.sizes["member"] == 1 and field.sizes["site"] == 1
        assert field["lon"].dims == ("site",)

    def test_a_pair_that_was_not_supplied_is_missing(self, niwot_output, sites_table):
        runs = {(1, 0): niwot_output, (27, 1): niwot_output}
        field = stack_sipnet_outputs(runs, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        assert field.sizes["member"] == 2 and field.sizes["site"] == 2
        assert np.isnan(field.sel(site=27, member=0).values).all()
        assert np.isfinite(field.sel(site=27, member=1).values).all()

    def test_the_attributes_are_still_pysipnets(self, runs, sites_table):
        field = stack_sipnet_outputs(runs, "nee", sites=sites_table)[
            "net_ecosystem_exchange"
        ]
        assert field.attrs == runs[(1, 0)]["nee"].attrs

    def test_runs_on_different_time_axes_are_joined_outward(self, niwot_output, sites_table):
        from pysipnet.output import SIPNETOutput

        short = SIPNETOutput.from_dataframe(
            niwot_output.pandas.iloc[:20].copy(), climate=niwot_output.climate.head(20)
        )
        field = stack_sipnet_outputs(
            {(1, 0): niwot_output, (27, 0): short}, "nee", sites=sites_table
        )["net_ecosystem_exchange"]
        # The axes are unioned, so each site is finite over its own record and
        # missing outside it. They are not nested: a truncated run's last step
        # ends at its declared length, where the full run's snapped to the next
        # row's start, so the union is longer than either.
        assert field.sizes["time"] >= niwot_output.n_timesteps
        assert int(np.isfinite(field.sel(site=1)).sum()) == niwot_output.n_timesteps
        assert int(np.isfinite(field.sel(site=27)).sum()) == 20
        assert np.isnan(field.sel(site=27, member=0).values[-1])
        # The interval coordinates then describe each site separately, which is
        # what aggregate_time refuses rather than guess a cell's length from.
        assert set(field["time_step_length"].dims) == {"site", "time"}

    def test_not_a_mapping_is_refused(self, niwot_output):
        with pytest.raises(TypeError, match="must be a mapping"):
            stack_sipnet_outputs([niwot_output], "nee")

    def test_an_empty_mapping_is_refused(self):
        with pytest.raises(ValueError, match="nothing to stack"):
            stack_sipnet_outputs({}, "nee")

    def test_a_key_that_is_not_a_pair_is_refused(self, niwot_output, sites_table):
        with pytest.raises(ValueError, match=r"\(site, member\) pair"):
            stack_sipnet_outputs({1: niwot_output}, "nee", sites=sites_table)


class TestSiteLookup:
    def test_it_keys_the_table_without_losing_the_column(self, sites_table):
        keyed = site_lookup(sites_table)
        assert keyed.index.name == "site_id"
        assert "site_id" in keyed.columns
        assert float(keyed.loc[27, "lon"]) == pytest.approx(
            float(sites_table.loc[sites_table["site_id"] == 27, "lon"].iloc[0])
        )

    def test_it_is_idempotent_so_passing_it_back_costs_nothing(self, sites_table):
        keyed = site_lookup(sites_table)
        assert site_lookup(keyed) is keyed

    def test_the_adapter_accepts_either_form(self, niwot_output, sites_table):
        plain = from_sipnet_output(niwot_output, "nee", site=27, sites=sites_table)
        keyed = from_sipnet_output(
            niwot_output, "nee", site=27, sites=site_lookup(sites_table)
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
            from_sipnet_output(empty, "nee")

    def test_a_fractional_site_is_refused(self, niwot_output, sites_table):
        with pytest.raises(ValueError, match="whole number"):
            from_sipnet_output(niwot_output, "nee", site=1.5, sites=sites_table)

    def test_a_boolean_is_not_an_identifier(self, niwot_output, sites_table):
        with pytest.raises(ValueError, match="boolean"):
            from_sipnet_output(niwot_output, "nee", site=True, sites=sites_table)
        with pytest.raises(ValueError, match="boolean"):
            from_sipnet_output(niwot_output, "nee", member=False)

    def test_an_infinite_identifier_is_refused_as_a_value_error(self, niwot_output):
        with pytest.raises(ValueError, match="member must be an integer"):
            from_sipnet_output(niwot_output, "nee", member=float("inf"))

    def test_variables_given_as_none_is_refused_as_a_value_error(self, niwot_output):
        with pytest.raises(ValueError, match="must be a name or a sequence"):
            from_sipnet_output(niwot_output, None)

    def test_an_unordered_container_is_refused_because_order_is_promised(self, niwot_output):
        with pytest.raises(ValueError, match="no\\s+order to keep"):
            from_sipnet_output(niwot_output, {"nee", "wood_carbon"})

    def test_a_result_whose_outputs_are_not_an_output_is_refused(self, niwot_output):
        class NotAResult:
            outputs = 42

        with pytest.raises(TypeError, match="rather than a SIPNETOutput"):
            from_sipnet_output(NotAResult(), "nee")

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
        self, sites_table, niwot_output
    ):
        keyed = sites_table.set_index("site_id")
        with pytest.raises(KeyError, match="not in the site table"):
            from_sipnet_output(niwot_output, "nee", site=99999, sites=keyed)


class TestStackSipnetOutputsRefusesBadKeys:
    def test_a_key_of_the_wrong_arity_names_the_contract(self, niwot_output, sites_table):
        with pytest.raises(ValueError, match=r"\(site, member\) pair"):
            stack_sipnet_outputs({(1, 0, 7): niwot_output}, "nee", sites=sites_table)

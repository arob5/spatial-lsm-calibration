"""Tests for the observation-space operations.

``sipnet_time_index`` is checked against cases built by hand from SIPNET's
``year``/``day``/``time`` row labels, including the drifting ``time`` column the
``.clim`` drivers carry (issue #9), so that a future change cannot start
trusting that column's value.

``aggregate_time`` is checked against **real SIPNET output** and, where the
inputs are present, a real run on this copy's 3-hourly site-1 drivers; the
synthetic fields would only show that it agrees with a fixture this project
wrote. The load-bearing tests are the ones that compare it with
``pysipnet.resample.resample`` -- values, time coordinates, attributes and
refusal message alike -- because the two are meant to be the same operation,
differing only in that one supplies a default method and reduces over the
other dimensions a canonical field may have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.resample import resample as pysipnet_resample
from pysipnet.variables import RESAMPLED_KIND, RESAMPLING_METHODS_FOR_KIND, VariableKind

from conftest import SITE_1_DRIVERS

from sipnet_calibration.fields import from_sipnet_output, stack_sipnet_outputs
from sipnet_calibration import conventions
from sipnet_calibration.obs_ops import (
    DEFAULT_METHOD_FOR_KIND,
    LENGTH_COORD,
    STALE_ON_A_COARSER_STEP,
    WINDOW_REDUCTIONS,
    aggregate_time,
    aggregation_counts,
    reduce_windows,
    sipnet_time_index,
    window_counts,
)

#: The same file the run fixtures use, defined once in ``conftest``.
REAL_FILE = SITE_1_DRIVERS


def one_day(year: int, day_of_year: int, hours) -> pd.DatetimeIndex:
    n = len(hours)
    return sipnet_time_index([year] * n, [day_of_year] * n, hours)


class TestSipnetTimeIndex:
    def test_builds_the_nominal_grid_from_year_day_and_slot(self):
        index = one_day(2013, 1, [0, 3, 6, 9, 12, 15, 18, 21])
        expected = pd.date_range("2013-01-01", periods=8, freq="3h").as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)
        assert index.dtype == np.dtype("datetime64[ns]")

    def test_ignores_the_drift_in_the_time_column(self):
        """A label of 23.00 in the last slot of 31 December still maps to 21:00."""
        drifted = np.arange(8) * 3.000685 + 1.995  # what 31 December looks like
        drifted[-1] = 23.0
        index = one_day(2013, 365, drifted)
        expected = pd.date_range("2013-12-31", periods=8, freq="3h").as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)

    def test_leap_day_is_placed_correctly(self):
        assert one_day(2012, 60, [0.0])[0] == pd.Timestamp("2012-02-29")
        assert one_day(2013, 60, [0.0])[0] == pd.Timestamp("2013-03-01")
        assert one_day(2012, 366, [21.0])[0] == pd.Timestamp("2012-12-31 21:00")

    def test_century_rule_for_leap_years(self):
        assert one_day(2000, 366, [0.0])[0] == pd.Timestamp("2000-12-31")
        with pytest.raises(ValueError, match="day_of_year 366 in non-leap year"):
            one_day(2100, 366, [0.0])

    def test_rejects_two_dimensional_input(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            sipnet_time_index([[2013]], [[1]], [[0.0]])

    def test_spans_a_year_boundary_in_row_order(self):
        index = sipnet_time_index([2012, 2012, 2013, 2013], [366, 366, 1, 1], [18, 21, 0, 3])
        expected = pd.DatetimeIndex(
            ["2012-12-31 18:00", "2012-12-31 21:00", "2013-01-01 00:00", "2013-01-01 03:00"]
        ).as_unit("ns")
        pd.testing.assert_index_equal(index, expected, check_names=False)

    def test_accepts_whole_number_floats(self):
        index = sipnet_time_index([2013.0, 2013.0], [5.0, 5.0], [0.0, 3.0])
        assert index[0] == pd.Timestamp("2013-01-05")

    def test_other_timesteps(self):
        daily = sipnet_time_index([2013, 2013], [1, 2], [0.0, 0.0], timestep_hours=24.0)
        pd.testing.assert_index_equal(
            daily, pd.DatetimeIndex(["2013-01-01", "2013-01-02"]).as_unit("ns"), check_names=False
        )
        hourly = sipnet_time_index([2013] * 24, [1] * 24, np.arange(24) + 0.01, timestep_hours=1.0)
        pd.testing.assert_index_equal(
            hourly, pd.date_range("2013-01-01", periods=24, freq="1h").as_unit("ns"), check_names=False
        )

    def test_empty_input_gives_an_empty_index(self):
        index = sipnet_time_index([], [], [])
        assert len(index) == 0 and index.dtype == np.dtype("datetime64[ns]")

    def test_rejects_a_timestep_that_does_not_divide_24(self):
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=5.0)
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=0.0)
        with pytest.raises(ValueError, match="divide 24"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours=float("inf"))
        with pytest.raises(ValueError, match="must be a number"):
            sipnet_time_index([2013], [1], [0.0], timestep_hours="three")

    def test_rejects_unequal_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            sipnet_time_index([2013, 2013], [1], [0.0])

    def test_accepts_the_arguments_by_keyword(self):
        index = sipnet_time_index(year=[2013], day_of_year=[2], hours_since_midnight=[6.5])
        assert index[0] == pd.Timestamp("2013-01-02 06:00")

    def test_rejects_non_whole_year_or_day(self):
        with pytest.raises(ValueError, match="day_of_year must hold whole numbers"):
            sipnet_time_index([2013], [1.5], [0.0])
        with pytest.raises(ValueError, match="year must hold whole numbers"):
            sipnet_time_index([2013.5], [1], [0.0])

    def test_rejects_day_outside_the_year(self):
        with pytest.raises(ValueError, match="within 1..366"):
            sipnet_time_index([2013], [0], [0.0])
        with pytest.raises(ValueError, match="within 1..366"):
            sipnet_time_index([2013], [367], [0.0])

    def test_rejects_day_366_in_a_non_leap_year(self):
        with pytest.raises(ValueError, match="day_of_year 366 in non-leap year"):
            sipnet_time_index([2013], [366], [0.0])

    def test_rejects_a_time_outside_the_day(self):
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [24.0])
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [-0.5])
        with pytest.raises(ValueError, match=r"within \[0, 24\)"):
            sipnet_time_index([2013], [1], [np.nan])

    def test_rejects_a_label_that_drifted_into_the_next_slot(self):
        """Two consecutive rows whose labels fall in one slot collide, and are
        refused rather than silently reassigned. A single label cannot reveal
        a drift on its own: a skipped slot is accepted, which is why the
        drivers module asserts the drift model on whole files instead."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            sipnet_time_index([2013, 2013], [1, 1], [3.0, 3.1])
        skipped = sipnet_time_index([2013, 2013], [1, 1], [0.0, 6.0])
        assert list(skipped) == [pd.Timestamp("2013-01-01 00:00"), pd.Timestamp("2013-01-01 06:00")]

    def test_rejects_rows_out_of_order(self):
        with pytest.raises(ValueError, match="not strictly increasing"):
            sipnet_time_index([2013, 2013], [2, 1], [0.0, 0.0])

    @pytest.mark.skipif(not REAL_FILE.exists(), reason="the raw driver file is not present")
    def test_reproduces_the_real_files_grid(self):
        frame = pd.read_csv(REAL_FILE, sep=r"\s+", header=None, usecols=[1, 2, 3], names=["year", "day", "time"])
        index = sipnet_time_index(frame["year"], frame["day"], frame["time"])
        expected = pd.date_range("2012-01-01", "2024-12-31 21:00", freq="3h").as_unit("ns")
        assert len(index) == 37992
        pd.testing.assert_index_equal(index, expected, check_names=False)


class TestDefaultMethodForKind:
    def test_exactly_one_method_per_kind_preserves_it_which_is_why_there_is_a_default(self):
        """The uniqueness `_kind_preserving_methods` relies on, over pySIPNET's table."""
        for kind in VariableKind:
            preserving = [
                method
                for method in ("sum", "mean", "last")
                if RESAMPLED_KIND.get((kind, method)) == kind
            ]
            assert len(preserving) <= 1, (kind, preserving)
            assert DEFAULT_METHOD_FOR_KIND.get(kind) == (
                preserving[0] if preserving else None
            )
            if preserving:
                assert preserving[0] in RESAMPLING_METHODS_FOR_KIND[kind]

    def test_the_defaults_are_the_ones_the_project_depends_on(self):
        assert DEFAULT_METHOD_FOR_KIND[VariableKind.TIMESTEP_TOTAL] == "sum"
        assert DEFAULT_METHOD_FOR_KIND[VariableKind.TIMESTEP_MEAN] == "mean"
        assert DEFAULT_METHOD_FOR_KIND[VariableKind.DAILY_RATE] == "mean"
        assert DEFAULT_METHOD_FOR_KIND[VariableKind.TIMESTEP_END_STATE] == "last"
        assert DEFAULT_METHOD_FOR_KIND[VariableKind.CUMULATIVE] == "last"

    def test_a_kind_no_method_preserves_has_no_default(self):
        assert VariableKind.TIMESTEP_START_COORDINATE not in DEFAULT_METHOD_FOR_KIND


class TestAggregateTimeAgainstPysipnet:
    """The one implementation must not disagree with pySIPNET's own ``resample``."""

    def test_a_sum_matches_values_time_and_attributes(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        reference = pysipnet_resample(niwot_output.select(["nee"]), "1D", how="sum")[
            "net_ecosystem_exchange"
        ]
        assert np.allclose(daily.values, reference.values)
        assert np.array_equal(daily["time"].values, reference["time"].values)
        assert np.array_equal(
            daily["time_step_start"].values, reference["time_step_start"].values
        )
        assert np.array_equal(
            daily["time_step_length"].values, reference["time_step_length"].values
        )
        assert daily.attrs == reference.attrs

    def test_a_pool_taken_last_matches(self, niwot_output):
        field = from_sipnet_output(niwot_output, "wood_carbon")["wood_carbon"]
        reference = pysipnet_resample(niwot_output.select(["wood_carbon"]), "1D", how="last")[
            "wood_carbon"
        ]
        daily = aggregate_time(field, "1D")
        assert np.allclose(daily.values, reference.values)
        assert daily.attrs == reference.attrs

    def test_a_weighted_mean_matches(self, niwot_output):
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        reference = pysipnet_resample(niwot_output.select(["soil_water"]), "1D", how="mean")[
            "soil_water"
        ]
        daily = aggregate_time(field, "1D", how="mean")
        assert np.allclose(daily.values, reference.values)
        assert daily.attrs == reference.attrs

    def test_a_refusal_is_pysipnets_own_words(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        with pytest.raises(ValueError) as ours:
            aggregate_time(field, "1D", how="mean")
        with pytest.raises(ValueError) as theirs:
            pysipnet_resample(niwot_output.select(["nee"]), "1D", how="mean")
        assert str(ours.value) == str(theirs.value)
        assert "sum them" in str(ours.value)


class TestAggregateTimeChoosesTheMethod:
    def test_a_per_timestep_total_sums(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        assert daily.attrs["resampling"] == "sum of timestep_total values over 1D"
        assert daily.attrs["kind"] == "timestep_total"
        assert daily.attrs["cell_methods"] == "time: sum"

    def test_an_end_of_step_state_takes_its_last_value(self, niwot_output):
        field = from_sipnet_output(niwot_output, "wood_carbon")["wood_carbon"]
        daily = aggregate_time(field, "1D")
        assert daily.attrs["resampling"] == "last of timestep_end_state values over 1D"
        assert daily.attrs["kind"] == "timestep_end_state"
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        expected = raw.groupby(raw.index.ceil("D")).last()
        assert np.allclose(daily.values, expected.to_numpy())

    def test_a_running_total_takes_its_last_value(self, niwot_output):
        name = "cumulative_net_ecosystem_exchange"
        field = from_sipnet_output(niwot_output, name)[name]
        assert aggregate_time(field, "1D").attrs["kind"] == "cumulative"

    def test_the_mean_of_a_pool_is_asked_for_and_is_no_longer_a_pool(self, niwot_output):
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        averaged = aggregate_time(field, "1D", how="mean")
        assert averaged.attrs["kind"] == "timestep_mean"
        assert averaged.attrs["cell_methods"] == "time: mean"
        assert "weighted by time_step_length" in averaged.attrs["resampling"]

    def test_a_mean_is_weighted_by_the_step_length(self, niwot_output):
        """Niwot's steps alternate day and night, so the weighting is visible."""
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        days = field["time_step_length"].values.astype("timedelta64[s]").astype(float) / 86400
        assert days.min() < days.max()

        raw = pd.DataFrame(
            {"value": field.values, "weight": days},
            index=pd.DatetimeIndex(field["time"].values),
        )
        cells = raw.groupby(raw.index.ceil("D"))
        weighted = cells.apply(
            lambda g: float(np.sum(g["value"] * g["weight"]) / np.sum(g["weight"])),
            include_groups=False,
        )
        unweighted = cells["value"].mean()

        ours = aggregate_time(field, "1D", how="mean")
        assert np.allclose(ours.values, weighted.to_numpy())
        assert not np.allclose(ours.values, unweighted.to_numpy())

    def test_an_unknown_method_is_refused(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        with pytest.raises(ValueError, match="Unknown resampling method"):
            aggregate_time(field, "1D", how="median")

    def test_a_field_with_no_kind_must_be_told_how(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        anonymous = field.rename("observed_thing")
        anonymous.attrs = {"units": "g m-2", "long_name": "Something observed"}
        with pytest.raises(ValueError, match="carries no 'kind' attribute"):
            aggregate_time(anonymous, "1D")
        assert aggregate_time(anonymous, "1D", how="sum").sizes["time"] > 0

    def test_a_nonsense_kind_is_refused(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        field.attrs["kind"] = "something_else"
        with pytest.raises(ValueError, match="which is not one of"):
            aggregate_time(field, "1D")

    def test_a_field_without_time_is_refused(self, field_member_site):
        with pytest.raises(ValueError, match="needs a 'time' dimension"):
            aggregate_time(field_member_site, "1D", how="sum")


class TestAggregateTimeOnThreeHourlyOutput:
    """The eight-steps-a-day case, on this copy's real site-1 drivers."""

    def test_a_daily_total_is_the_sum_of_the_eight_three_hourly_values(self, site_1_result):
        field = from_sipnet_output(site_1_result, "nee", site=1)["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")

        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        cells = raw.groupby(raw.index.ceil("D"))
        assert np.allclose(daily.values, cells.sum().to_numpy())

        # Every interior cell holds eight steps. The two on the ends are short
        # because the .clim hour column drifts late (issue #9): each day's last
        # step end lands just past midnight, so the whole record sits a step
        # later than SIPNET's own day column would put it.
        counts = cells.size().to_numpy()
        assert len(counts) > 3
        assert (counts[1:-1] == 8).all()
        assert counts[0] < 8 and counts[-1] < 8

    def test_a_daily_pool_is_its_value_at_the_last_of_the_eight_steps(self, site_1_result):
        field = from_sipnet_output(site_1_result, "soil_water", site=1)["soil_water"]
        daily = aggregate_time(field, "1D")
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        assert np.allclose(daily.values, raw.groupby(raw.index.ceil("D")).last().to_numpy())

    def test_a_day_of_steps_covers_twenty_four_hours(self, site_1_result):
        field = from_sipnet_output(site_1_result, "nee", site=1)["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        lengths = daily["time_step_length"].values[1:-1]
        assert (lengths == np.timedelta64(24, "h")).all()

    def test_the_site_label_and_its_coordinates_survive(self, site_1_result):
        field = from_sipnet_output(site_1_result, "nee", site=1)["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        assert int(daily["site"]) == 1
        assert float(daily["lon"]) == pytest.approx(float(field["lon"]))


class TestAggregateTimeOnEnsembles:
    def test_a_stacked_field_aggregates_slice_by_slice(self, niwot_output, sites_table):
        runs = {(site, member): niwot_output for site in (1, 27) for member in (0, 1)}
        stacked = stack_sipnet_outputs(runs, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(stacked, "1D")
        assert daily.dims == ("member", "site", "time")
        one = aggregate_time(
            from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"], "1D"
        )
        for site in (1, 27):
            for member in (0, 1):
                assert np.allclose(daily.sel(site=site, member=member).values, one.values)

    def test_lon_and_lat_survive_on_site(self, niwot_output, sites_table):
        runs = {(site, 0): niwot_output for site in (1, 27)}
        daily = aggregate_time(
            stack_sipnet_outputs(runs, "nee")["net_ecosystem_exchange"], "1D"
        )
        assert daily["lon"].dims == ("site",)

    def test_per_site_interval_coordinates_are_refused(self, niwot_output, sites_table):
        from pysipnet.output import SIPNETOutput

        short = SIPNETOutput.from_dataframe(
            niwot_output.pandas.iloc[:20].copy(),
            time_step_length=niwot_output.time_step_length[:20],
        )
        stacked = stack_sipnet_outputs({(1, 0): niwot_output, (27, 0): short}, "nee")[
            "net_ecosystem_exchange"
        ]
        with pytest.raises(ValueError, match="different\ntime axes|different time axes"):
            aggregate_time(stacked, "1D")
        # Selecting one site is what the message says to do, and it works.
        assert aggregate_time(stacked.sel(site=1), "1D").sizes["time"] > 0


class TestAggregateTimeOnDrivers:
    def test_par_sums_and_temperature_means(self, real_driver_field):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        dataset = drivers.load_drivers([1], members=[1])
        fields = drivers.driver_fields(dataset)
        assert aggregate_time(fields["par"], "1D").attrs["kind"] == "timestep_total"
        assert (
            aggregate_time(fields["air_temperature"], "1D").attrs["kind"] == "timestep_mean"
        )

    def test_a_daily_par_total_is_its_eight_three_hourly_values(self, real_driver_field):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        par = drivers.driver_fields(drivers.load_drivers([1], members=[1]))["par"]
        one = par.isel(member=0, site=0)
        daily = aggregate_time(one, "1D")
        raw = pd.Series(one.values, index=pd.DatetimeIndex(one["time"].values))
        cells = raw.groupby(raw.index.ceil("D"))
        assert np.allclose(daily.values, cells.sum().to_numpy())
        assert (cells.size().to_numpy()[1:-1] == 8).all()

    def test_the_drivers_have_no_declared_step_lengths_and_are_weighted_equally(
        self, real_driver_field
    ):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        tair = drivers.driver_fields(drivers.load_drivers([1], members=[1]))[
            "air_temperature"
        ].isel(member=0, site=0)
        assert LENGTH_COORD not in tair.coords
        daily = aggregate_time(tair, "1D")
        assert "weighted by" not in daily.attrs["resampling"]
        raw = pd.Series(tair.values, index=pd.DatetimeIndex(tair["time"].values))
        assert np.allclose(daily.values, raw.groupby(raw.index.ceil("D")).mean().to_numpy())

    def test_unequal_steps_without_declared_lengths_refuse_a_mean(self, niwot_output):
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        bare = field.drop_vars([LENGTH_COORD])
        with pytest.raises(ValueError, match="not all the same length"):
            aggregate_time(bare, "1D", how="mean")


class TestAggregateTimeKeepsGapsAndCells:
    def test_a_cell_holding_a_gap_is_missing(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        values = field.values.copy()
        values[3] = np.nan
        gappy = field.copy(data=values)
        daily = aggregate_time(gappy, "1D")
        assert int(np.isnan(daily.values).sum()) == 1

    def test_cells_no_step_falls_in_are_dropped(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        ends = xr.concat(
            [field.isel(time=slice(0, 4)), field.isel(time=slice(-4, None))], "time"
        )
        spanned = aggregate_time(field, "1D").sizes["time"]
        daily = aggregate_time(ends, "1D")
        assert daily.sizes["time"] == 4
        assert daily.sizes["time"] < spanned
        assert np.isfinite(daily.values).all()

    def test_a_daily_record_resampled_to_days_is_itself(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        again = aggregate_time(daily, "1D")
        assert np.allclose(again.values, daily.values)
        assert np.array_equal(again["time"].values, daily["time"].values)


class TestAggregatedFieldsPlot:
    def test_plot_time_series_accepts_the_result_unchanged(self, ax, niwot_output):
        plotting = pytest.importorskip("sipnet_calibration.plotting")
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        returned = plotting.plot_time_series(daily, ax=ax)
        assert returned is ax
        assert len(ax.lines) == 1
        assert len(ax.lines[0].get_xydata()) == daily.sizes["time"]
        assert ax.get_ylabel() == plotting.axis_label(daily)
        # Spelled out as well, so the label cannot come from an empty field.
        assert ax.get_ylabel() == "Net ecosystem exchange (g m-2)"
        assert np.allclose(ax.lines[0].get_ydata(), daily.values)

    def test_an_aggregated_ensemble_fans(self, ax, niwot_output, sites_table):
        plotting = pytest.importorskip("sipnet_calibration.plotting")
        runs = {(1, member): niwot_output for member in (0, 1, 2)}
        stacked = stack_sipnet_outputs(runs, "nee")["net_ecosystem_exchange"]
        plotting.plot_time_series(aggregate_time(stacked, "1D").sel(site=1), ax=ax)
        assert len(ax.collections) == 2


class TestAggregatedTimeCoordinateDescribesItself:
    def test_the_model_axis_is_rebuilt_in_pysipnets_words(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        assert daily["time"].attrs["long_name"] == "End of timestep"
        assert daily["time"].attrs["standard_name"] == "time"
        assert daily["time_step_length"].attrs["source"].startswith("sum of the declared")

    def test_nothing_points_at_a_bounds_variable_that_cannot_be_carried(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert "bounds" in niwot_output.xarray["time"].attrs
        assert "bounds" not in aggregate_time(field, "1D")["time"].attrs

    def test_a_driver_axis_keeps_what_is_still_true_of_it(self, real_driver_field):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        tair = drivers.driver_fields(drivers.load_drivers([1], members=[1]))[
            "air_temperature"
        ]
        daily = aggregate_time(tair, "1D")
        # Cells are labeled at their end, as the steps were.
        assert daily["time"].attrs["time_label"] == tair["time"].attrs["time_label"]
        assert daily["time"].attrs["time_zone"] == tair["time"].attrs["time_zone"]
        # The note stating the width of one driver interval is not.
        assert "time_label_note" in tair["time"].attrs
        assert "time_label_note" not in daily["time"].attrs

    def test_the_dropped_set_is_the_documented_two(self):
        """A change detector: each is checked against a real source above."""
        assert set(STALE_ON_A_COARSER_STEP) == {"bounds", "time_label_note"}


class TestAggregateTimeDropsAlignmentPadding:
    """A timestamp that is not a step of the field must not reach a cell.

    Stacking runs of different lengths is enough to produce one: pySIPNET snaps
    each interior step's end onto the next step's start, but a truncated run's
    last end is its start plus the declared length, so the two records differ
    by one timestamp in the middle of the longer one.
    """

    @staticmethod
    def stacked(niwot_output):
        from pysipnet.output import SIPNETOutput

        short = SIPNETOutput.from_dataframe(
            niwot_output.pandas.iloc[:20].copy(),
            time_step_length=niwot_output.time_step_length[:20],
        )
        return stack_sipnet_outputs({(1, 0): short, (27, 0): niwot_output}, "nee")[
            "net_ecosystem_exchange"
        ]

    def test_a_padded_timestamp_does_not_empty_the_cell_it_falls_in(
        self, niwot_output, sites_table
    ):
        alone = aggregate_time(
            from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"], "1D"
        )
        together = aggregate_time(self.stacked(niwot_output).sel(site=27), "1D")
        assert not np.isnan(together.values).any()
        assert np.allclose(together.values, alone.values)
        assert np.array_equal(together["time"].values, alone["time"].values)

    def test_the_shorter_record_keeps_only_its_own_cells(self, niwot_output, sites_table):
        short_side = aggregate_time(self.stacked(niwot_output).sel(site=1), "1D")
        assert not np.isnan(short_side.values).any()
        assert short_side.sizes["time"] < 30

    def test_a_step_whose_length_is_missing_is_not_a_step(self):
        """A NaT length casts to the int64 sentinel, not to a missing value."""
        times = pd.date_range("2000-01-01T03:00", periods=8, freq="3h")
        lengths = np.full(8, 3, dtype="timedelta64[h]").astype("timedelta64[ns]")
        lengths[3] = np.timedelta64("NaT")
        field = xr.DataArray(
            np.arange(8.0),
            dims="time",
            coords={
                "time": times,
                "time_step_start": ("time", (times - pd.Timedelta("3h")).values),
                "time_step_length": ("time", lengths),
            },
            name="net_ecosystem_exchange",
            attrs={"kind": "timestep_total", "units": "g m-2", "long_name": "NEE"},
        )
        daily = aggregate_time(field, "1D")
        assert float(daily.sum()) == pytest.approx(28.0 - 3.0)
        assert (daily["time_step_length"].values > np.timedelta64(0)).all()
        assert float(daily["time_step_length"].sum() / np.timedelta64(1, "h")) == 21.0


class TestAggregateTimeRefusesUnusableTime:
    @staticmethod
    def nee(values, times):
        return xr.DataArray(
            np.asarray(values, dtype=float),
            dims="time",
            coords={"time": pd.DatetimeIndex(times)},
            name="net_ecosystem_exchange",
            attrs={"kind": "timestep_total", "units": "g m-2", "long_name": "NEE"},
        )

    def test_duplicate_timestamps_are_refused_rather_than_added_together(self):
        field = self.nee([1, 1, 2, 2], ["2000-01-01T12:00"] * 2 + ["2000-01-02T12:00"] * 2)
        with pytest.raises(ValueError, match="do not increase"):
            aggregate_time(field, "1D")

    def test_timestamps_out_of_order_are_refused(self):
        field = self.nee([1, 2], ["2000-01-02T12:00", "2000-01-01T12:00"])
        with pytest.raises(ValueError, match="do not increase"):
            aggregate_time(field, "1D")

    def test_an_empty_time_dimension_is_refused(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        with pytest.raises(ValueError, match="no timesteps left to aggregate"):
            aggregate_time(field.isel(time=slice(0, 0)), "1D")


class TestAggregateTimeLabelsCells:
    """Where a field carries no interval coordinates, the resample's own labels ship."""

    def test_a_daily_driver_cell_is_labeled_at_its_end(self, real_driver_field):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        par = drivers.driver_fields(drivers.load_drivers([1], members=[1]))["par"].isel(
            member=0, site=0
        )
        daily = aggregate_time(par, "1D")
        raw = pd.Series(par.values, index=pd.DatetimeIndex(par["time"].values))
        expected = raw.groupby(raw.index.ceil("D")).sum()
        assert np.array_equal(
            daily["time"].values, pd.DatetimeIndex(expected.index).to_numpy()
        )
        # The steps of the cell labeled d end after midnight of d-1 and at
        # midnight of d, which is what "interval_end" means one level coarser.
        assert daily["time"].attrs["time_label"] == "interval_end"

    def test_a_model_cell_is_labeled_at_the_last_step_end_it_holds(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        expected = raw.groupby(raw.index.ceil("D")).apply(lambda cell: cell.index.max())
        assert np.array_equal(daily["time"].values, pd.DatetimeIndex(expected).to_numpy())


class TestAggregateTimeKeepsEveryMethodNaNAware:
    """The docstring promises a cell holding a NaN is NaN for every method."""

    @staticmethod
    def with_a_gap(niwot_output, name):
        field = from_sipnet_output(niwot_output, name)[name]
        values = field.values.copy()
        values[3] = np.nan
        return field.copy(data=values)

    def test_a_summed_cell_holding_a_gap_is_missing(self, niwot_output):
        gappy = self.with_a_gap(niwot_output, "net_ecosystem_exchange")
        assert int(np.isnan(aggregate_time(gappy, "1D").values).sum()) == 1

    def test_a_pool_cell_holding_a_gap_is_missing(self, niwot_output):
        gappy = self.with_a_gap(niwot_output, "soil_water")
        assert int(np.isnan(aggregate_time(gappy, "1D").values).sum()) == 1

    def test_an_averaged_cell_holding_a_gap_is_missing(self, niwot_output):
        gappy = self.with_a_gap(niwot_output, "soil_water")
        assert int(np.isnan(aggregate_time(gappy, "1D", how="mean").values).sum()) == 1


class TestAggregateTimeKeepsTheVariablesIdentity:
    def test_the_name_survives_every_method(self, niwot_output):
        """A mean divides two arrays, which is where xarray drops the name."""
        total = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        assert aggregate_time(total, "1D").name == "net_ecosystem_exchange"
        pool = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        assert aggregate_time(pool, "1D").name == "soil_water"
        assert aggregate_time(pool, "1D", how="mean").name == "soil_water"

    def test_a_field_stripped_of_attributes_is_recognized_by_its_name(self, niwot_output):
        """The registry fallback, for a field whose attrs were lost in transit."""
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        stripped = field.copy()
        stripped.attrs = {}
        daily = aggregate_time(stripped, "1D")
        assert daily.attrs["kind"] == "timestep_total"
        assert np.allclose(daily.values, aggregate_time(field, "1D").values)

    def test_a_driver_name_resolves_through_the_climate_registry(self, real_driver_field):
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        par = drivers.driver_fields(drivers.load_drivers([1], members=[1]))["par"]
        stripped = par.copy()
        stripped.attrs = {}
        assert aggregate_time(stripped, "1D").attrs["kind"] == "timestep_total"


class TestAggregationCounts:
    """What tells a partial cell from a full one where nothing else does."""

    def test_counts_align_with_the_aggregate(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        counts = aggregation_counts(field, "1D")
        assert counts.sizes["time"] == aggregate_time(field, "1D").sizes["time"]
        assert counts.dtype == np.int64
        assert counts.attrs == {}

    def test_a_count_is_the_number_of_values_that_are_not_missing(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        expected = raw.groupby(raw.index.ceil("D")).count()
        assert np.array_equal(aggregation_counts(field, "1D").values, expected.to_numpy())

    def test_a_gap_is_not_counted(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        values = field.values.copy()
        values[0] = np.nan
        counts = aggregation_counts(field.copy(data=values), "1D")
        assert int(counts.values[0]) == int(aggregation_counts(field, "1D").values[0]) - 1

    def test_the_boundary_cells_of_a_driver_record_are_visibly_partial(
        self, real_driver_field
    ):
        """The case aggregate_time alone cannot report: no declared step lengths."""
        drivers = pytest.importorskip("sipnet_calibration.drivers")
        par = drivers.driver_fields(drivers.load_drivers([1], members=[1]))["par"].isel(
            member=0, site=0
        )
        counts = aggregation_counts(par, "1D")
        assert int(counts.values[0]) < 8
        assert (counts.values[1:-1] == 8).all()

    def test_a_field_with_no_kind_can_still_be_counted(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        stripped = field.rename("observed_thing")
        stripped.attrs = {}
        assert int(aggregation_counts(stripped, "1D").sum()) == field.sizes["time"]


class TestReduceWindows:
    @staticmethod
    def blocks(start="1998-11-01", periods=4, freq="10D", closed="right"):
        edges = pd.date_range(start, periods=periods, freq=freq)
        return pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed=closed)

    def test_a_window_sum_is_the_sum_of_the_rows_labeled_inside_it(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks()
        total = reduce_windows(field, windows, "sum")
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        for i, window in enumerate(windows):
            inside = raw[(raw.index > window.left) & (raw.index <= window.right)]
            assert float(total.values[i]) == pytest.approx(inside.sum())

    def test_every_reduction_matches_pandas(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks()
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        cut = pd.cut(raw.index, windows.astype(f"interval[datetime64[{raw.index.unit}], right]"))
        for how in WINDOW_REDUCTIONS:
            expected = raw.groupby(cut, observed=False).agg(how)
            ours = reduce_windows(field, windows, how)
            assert np.allclose(ours.values, expected.to_numpy()), how

    def test_the_closed_side_decides_a_label_on_a_shared_edge(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        stamps = pd.DatetimeIndex(field["time"].values)
        edges = pd.DatetimeIndex([stamps[0], stamps[2], stamps[4]])
        right = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")
        left = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="left")
        assert int(window_counts(field, right).values[0]) == 2
        assert int(window_counts(field, left).values[0]) == 2
        # The row labeled exactly on the middle edge falls on either side.
        assert float(reduce_windows(field, right, "sum").values[0]) == pytest.approx(
            float(field.values[1] + field.values[2])
        )
        assert float(reduce_windows(field, left, "sum").values[0]) == pytest.approx(
            float(field.values[0] + field.values[1])
        )

    def test_rows_outside_every_window_are_ignored(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks(periods=2)
        assert int(window_counts(field, windows).sum()) < field.sizes["time"]

    def test_a_window_holding_nothing_is_missing(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        far = pd.IntervalIndex.from_arrays(
            pd.DatetimeIndex(["2050-01-01"]), pd.DatetimeIndex(["2050-02-01"]), closed="right"
        )
        assert np.isnan(reduce_windows(field, far, "sum").values).all()
        assert int(window_counts(field, far).values[0]) == 0

    def test_min_count_masks_a_thin_window(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks()
        counts = window_counts(field, windows)
        assert not np.isnan(reduce_windows(field, windows, "sum", min_count=1).values).any()
        thin = reduce_windows(field, windows, "sum", min_count=int(counts.max()) + 1)
        assert np.isnan(thin.values).all()

    def test_the_default_labels_are_the_right_edges_marked_as_interval_ends(
        self, niwot_output
    ):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks()
        reduced = reduce_windows(field, windows, "sum")
        assert np.array_equal(
            reduced["time"].values, pd.DatetimeIndex(windows.right).to_numpy()
        )
        assert reduced["time"].attrs[conventions.TIME_LABEL_ATTR] == conventions.INTERVAL_END

    def test_given_labels_keep_their_own_attributes(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        windows = self.blocks()
        labels = xr.DataArray(
            pd.DatetimeIndex(["1998-11-05", "1998-11-15", "1998-11-25"]).to_numpy(),
            dims="time",
            attrs={"long_name": "Observation date", "comment": "the observation's own"},
        )
        reduced = reduce_windows(field, windows, "sum", labels=labels)
        assert reduced["time"].attrs["long_name"] == "Observation date"
        assert np.array_equal(reduced["time"].values, labels.values)

    def test_the_other_dimensions_and_their_coordinates_survive(self, niwot_output):
        runs = {(site, member): niwot_output for site in (1, 27) for member in (0, 1)}
        stacked = stack_sipnet_outputs(runs, "nee")["net_ecosystem_exchange"]
        reduced = reduce_windows(stacked, self.blocks(), "sum")
        assert reduced.dims == ("member", "site", "time")
        assert reduced["lon"].dims == ("site",)
        one = reduce_windows(
            from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"],
            self.blocks(),
            "sum",
        )
        assert np.allclose(reduced.sel(site=1, member=0).values, one.values)

    def test_the_variable_keeps_its_name_and_attributes_plus_the_reduction(
        self, niwot_output
    ):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        reduced = reduce_windows(field, self.blocks(), "mean")
        assert reduced.name == "net_ecosystem_exchange"
        assert reduced.attrs["units"] == field.attrs["units"]
        assert reduced.attrs["reduction"] == "mean"
        # Unlike aggregate_time, the kind is the observation's business.
        assert reduced.attrs["kind"] == field.attrs["kind"]

    def test_calendar_windows_agree_with_aggregate_time(self, niwot_output):
        """The two are the same operation where the windows are a frequency."""
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        stamps = pd.DatetimeIndex(field["time"].values)
        edges = pd.date_range(
            stamps.min().floor("D"), stamps.max().ceil("D") + pd.Timedelta("1D"), freq="1D"
        )
        windows = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")
        by_window = reduce_windows(field, windows, "sum")
        kept = ~np.isnan(by_window.values)
        assert np.allclose(by_window.values[kept], daily.values)


class TestWindowInputsAreChecked:
    @staticmethod
    def field(niwot_output):
        return from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]

    def test_windows_must_be_an_interval_index(self, niwot_output):
        with pytest.raises(ValueError, match="must be a pandas.IntervalIndex"):
            reduce_windows(self.field(niwot_output), pd.DatetimeIndex(["1998-11-01"]), "sum")

    def test_overlapping_windows_are_refused(self, niwot_output):
        windows = pd.IntervalIndex.from_arrays(
            pd.DatetimeIndex(["1998-11-01", "1998-11-05"]),
            pd.DatetimeIndex(["1998-11-10", "1998-11-20"]),
            closed="right",
        )
        with pytest.raises(ValueError, match="overlap"):
            reduce_windows(self.field(niwot_output), windows, "sum")

    def test_decreasing_windows_are_refused(self, niwot_output):
        windows = pd.IntervalIndex.from_arrays(
            pd.DatetimeIndex(["1998-11-20", "1998-11-01"]),
            pd.DatetimeIndex(["1998-11-25", "1998-11-05"]),
            closed="right",
        )
        with pytest.raises(ValueError, match="increasing order"):
            reduce_windows(self.field(niwot_output), windows, "sum")

    def test_empty_windows_are_refused(self, niwot_output):
        empty = pd.IntervalIndex.from_arrays(
            pd.DatetimeIndex([]), pd.DatetimeIndex([]), closed="right"
        )
        with pytest.raises(ValueError, match="nothing to reduce into"):
            reduce_windows(self.field(niwot_output), empty, "sum")

    def test_numeric_windows_are_refused(self, niwot_output):
        with pytest.raises(ValueError, match="intervals of datetimes"):
            reduce_windows(
                self.field(niwot_output), pd.IntervalIndex.from_breaks([0, 1, 2]), "sum"
            )

    def test_a_time_zone_mismatch_is_refused_rather_than_matching_nothing(
        self, niwot_output
    ):
        edges = pd.date_range("1998-11-01", periods=2, freq="10D", tz="UTC")
        windows = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")
        with pytest.raises(ValueError, match="time zone"):
            reduce_windows(self.field(niwot_output), windows, "sum")

    def test_an_unknown_reduction_is_refused(self, niwot_output):
        windows = TestReduceWindows.blocks()
        with pytest.raises(ValueError, match="how must be one of"):
            reduce_windows(self.field(niwot_output), windows, "median")

    def test_min_count_must_be_a_positive_integer(self, niwot_output):
        windows = TestReduceWindows.blocks()
        with pytest.raises(ValueError, match="at least 1"):
            reduce_windows(self.field(niwot_output), windows, "sum", min_count=0)
        with pytest.raises(ValueError, match="must be an integer"):
            reduce_windows(self.field(niwot_output), windows, "sum", min_count=1.5)

    def test_numeric_labels_are_refused(self, niwot_output):
        windows = TestReduceWindows.blocks()
        with pytest.raises(ValueError, match="must be timestamps"):
            reduce_windows(self.field(niwot_output), windows, "sum", labels=[0, 1, 2])

    def test_one_label_per_window(self, niwot_output):
        windows = TestReduceWindows.blocks()
        with pytest.raises(ValueError, match="one label per"):
            reduce_windows(
                self.field(niwot_output),
                windows,
                "sum",
                labels=pd.DatetimeIndex(["1998-11-05"]),
            )


class TestTimeAxisIsChecked:
    def test_a_dataset_is_refused_by_name(self, niwot_output):
        with pytest.raises(ValueError, match="one field at a time"):
            aggregate_time(niwot_output.select(["nee"]), "1D")

    def test_a_non_datetime_time_coordinate_is_refused(self):
        field = xr.DataArray(
            np.arange(4.0), dims="time", coords={"time": [0, 1, 2, 3]},
            name="net_ecosystem_exchange", attrs={"kind": "timestep_total"},
        )
        with pytest.raises(ValueError, match="needs datetimes"):
            aggregate_time(field, "1D")

    def test_a_missing_timestamp_is_refused(self):
        times = pd.DatetimeIndex(["2000-01-01", pd.NaT, "2000-01-03"])
        field = xr.DataArray(
            np.arange(3.0), dims="time", coords={"time": times},
            name="net_ecosystem_exchange", attrs={"kind": "timestep_total"},
        )
        with pytest.raises(ValueError, match="missing timestamp"):
            aggregate_time(field, "1D")

    def test_a_nonsense_frequency_is_refused(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        with pytest.raises(ValueError, match="offset alias"):
            aggregate_time(field, "banana")

    def test_upsampling_is_refused_rather_than_returning_mostly_missing(
        self, niwot_output
    ):
        daily = aggregate_time(
            from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"], "1D"
        )
        with pytest.raises(ValueError, match="interpolate rather than aggregate"):
            aggregate_time(daily, "1h")


class TestCountsAlignWithTheAggregateTheyDescribe:
    """A count that did not line up with its aggregate would be worse than none."""

    @staticmethod
    def gappy(niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        return xr.concat(
            [field.isel(time=slice(0, 4)), field.isel(time=slice(-4, None))], "time"
        )

    def test_the_empty_cells_dropped_from_one_are_dropped_from_the_other(
        self, niwot_output
    ):
        gappy = self.gappy(niwot_output)
        assert (
            aggregation_counts(gappy, "1D").sizes["time"]
            == aggregate_time(gappy, "1D").sizes["time"]
        )

    def test_the_cells_are_right_closed_as_aggregate_time_makes_them(self, niwot_output):
        """Not xarray's left-closed default, which would count a different day."""
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        right_closed = raw.groupby(raw.index.ceil("D")).count()
        assert np.array_equal(
            aggregation_counts(field, "1D").values, right_closed.to_numpy()
        )

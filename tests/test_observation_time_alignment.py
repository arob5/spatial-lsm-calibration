"""Tests for the observation-space operations.

``aggregate_time`` is checked against **real SIPNET output** and, where the
inputs are present, a real run on this copy's 3-hourly site-1 drivers; the
synthetic fields would only show that it agrees with a fixture this project
wrote. The load-bearing tests are the ones that compare it with
``pysipnet.resample.resample`` -- values, time coordinates, attributes and
refusal message alike -- because the two are meant to be the same operation,
differing only in that one supplies a default method and reduces over the
other dimensions a field may have.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import SITE_1_DRIVERS
from pysipnet.resample import resample
from pysipnet.variables import RESAMPLED_KIND, RESAMPLING_METHODS_FOR_KIND, VariableKind

from sipnet_calibration.drivers import driver_fields, read_driver_file
from sipnet_calibration.fields import (
    STALE_TIME_ATTRIBUTE_NAMES,
    TIME_STEP_LENGTH,
    from_sipnet_output,
    stack_sipnet_outputs,
)
from sipnet_calibration.observation.time_alignment import (
    DEFAULT_METHOD_FOR_KIND,
    aggregate_time,
    aggregation_counts,
)


def site_1_member_1(real_drivers, name):
    """One variable of the site-1, member-1 drivers, on ``time`` alone."""
    return driver_fields(real_drivers)[name].sel(site=1, source_member_index=1)


class TestDefaultMethodForKind:
    def test_exactly_one_method_per_kind_preserves_it_which_is_why_there_is_a_default(self):
        """The uniqueness `DEFAULT_METHOD_FOR_KIND` relies on, over pySIPNET's table."""
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
        reference = resample(niwot_output.select(["nee"]), "1D", how="sum")[
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
        reference = resample(niwot_output.select(["wood_carbon"]), "1D", how="last")[
            "wood_carbon"
        ]
        daily = aggregate_time(field, "1D")
        assert np.allclose(daily.values, reference.values)
        assert daily.attrs == reference.attrs

    def test_a_weighted_mean_matches(self, niwot_output):
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        reference = resample(niwot_output.select(["soil_water"]), "1D", how="mean")[
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
            resample(niwot_output.select(["nee"]), "1D", how="mean")
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
        with pytest.raises(ValueError, match="(?i)unknown resampling method"):
            aggregate_time(field, "1D", how="median")

    def test_a_field_with_no_kind_must_be_told_how(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        anonymous = field.rename("observed_thing")
        anonymous.attrs = {"units": "g m-2", "long_name": "Something observed"}
        with pytest.raises(ValueError, match="carries no 'kind' attribute"):
            aggregate_time(anonymous, "1D")
        # pySIPNET checks a method against the kind, so a field with its
        # interval coordinates needs one even when told how.
        with pytest.raises(ValueError, match="interval coordinates but no 'kind'"):
            aggregate_time(anonymous, "1D", how="sum")
        observed = anonymous.drop_vars(["time_step_start", "time_step_length"])
        assert aggregate_time(observed, "1D", how="sum").sizes["time"] > 0

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

        # Every cell holds eight steps, the first and last included: a day's
        # eighth step ends at midnight, which the right-closed cells put in the
        # day that ended.
        counts = cells.size().to_numpy()
        assert len(counts) > 3
        assert (counts == 8).all()

    def test_a_daily_pool_is_its_value_at_the_last_of_the_eight_steps(self, site_1_result):
        field = from_sipnet_output(site_1_result, "soil_water", site=1)["soil_water"]
        daily = aggregate_time(field, "1D")
        raw = pd.Series(field.values, index=pd.DatetimeIndex(field["time"].values))
        assert np.allclose(daily.values, raw.groupby(raw.index.ceil("D")).last().to_numpy())

    def test_a_day_of_steps_covers_twenty_four_hours(self, site_1_result):
        field = from_sipnet_output(site_1_result, "nee", site=1)["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        lengths = daily["time_step_length"].values
        assert (lengths == np.timedelta64(24, "h")).all()
        spans = daily["time"].values - daily["time_step_start"].values
        assert (spans == np.timedelta64(24, "h")).all()

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
            niwot_output.pandas.iloc[:20].copy(), climate=niwot_output.climate.head(20)
        )
        stacked = stack_sipnet_outputs({(1, 0): niwot_output, (27, 0): short}, "nee")[
            "net_ecosystem_exchange"
        ]
        with pytest.raises(ValueError, match="different\ntime axes|different time axes"):
            aggregate_time(stacked, "1D")
        # Selecting one site is what the message says to do, and it works.
        assert aggregate_time(stacked.sel(site=1), "1D").sizes["time"] > 0


class TestAggregateTimeOnDrivers:
    def test_totals_sum_and_means_mean(self, real_drivers):
        fields = driver_fields(real_drivers)
        daily_precipitation = aggregate_time(fields["precipitation"], "1D")
        assert daily_precipitation.attrs["kind"] == "timestep_total"
        assert (
            aggregate_time(fields["air_temperature"], "1D").attrs["kind"] == "timestep_mean"
        )

    def test_a_daily_total_is_its_eight_three_hourly_values(self, real_drivers):
        par = site_1_member_1(real_drivers, "photosynthetically_active_radiation")
        daily = aggregate_time(par, "1D")
        raw = pd.Series(par.values, index=pd.DatetimeIndex(par["time"].values))
        cells = raw.groupby(raw.index.ceil("D"))
        assert np.allclose(daily.values, cells.sum().to_numpy())
        assert (cells.size().to_numpy() == 8).all()

    def test_a_driver_mean_is_weighted_by_its_declared_step_lengths(self, real_drivers):
        tair = site_1_member_1(real_drivers, "air_temperature")
        assert TIME_STEP_LENGTH in tair.coords
        daily = aggregate_time(tair, "1D")
        assert "weighted by" in daily.attrs["resampling"]
        raw = pd.Series(tair.values, index=pd.DatetimeIndex(tair["time"].values))
        assert np.allclose(daily.values, raw.groupby(raw.index.ceil("D")).mean().to_numpy())

    def test_a_driver_field_aggregates_as_pysipnet_resamples_its_drivers(
        self, real_drivers, regular_drivers_root
    ):
        """Values, time coordinates and attributes, against pySIPNET's own
        ``resample`` of the Dataset it builds for the same file."""
        with warnings.catch_warnings():
            # The file's exact zeros of vpd, which pySIPNET flags on read.
            warnings.simplefilter("ignore")
            own = read_driver_file(regular_drivers_root / SITE_1_DRIVERS).xarray
        expected = resample(own, "1D", how={"precipitation": "sum"})["precipitation"]
        daily = aggregate_time(site_1_member_1(real_drivers, "precipitation"), "1D")
        np.testing.assert_allclose(daily.values, expected.values)
        for name in ("time", "time_step_start", "time_step_length"):
            np.testing.assert_array_equal(daily[name].values, expected[name].values)
        assert daily.attrs["kind"] == expected.attrs["kind"]
        assert daily.attrs["cell_methods"] == expected.attrs["cell_methods"]

    def test_unequal_steps_without_declared_lengths_refuse_a_mean(self, niwot_output):
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        bare = field.drop_vars([TIME_STEP_LENGTH])
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

    def test_a_driver_axis_is_rebuilt_in_pysipnets_words_too(self, real_drivers):
        tair = site_1_member_1(real_drivers, "air_temperature")
        daily = aggregate_time(tair, "1D")
        assert daily["time"].attrs["long_name"] == "End of timestep"
        assert daily["time"].attrs["time_zone"] == tair["time"].attrs["time_zone"]
        assert "bounds" not in daily["time"].attrs

    def test_the_dropped_set_is_the_documented_one(self):
        """A change detector: it is checked against a real source above."""
        assert set(STALE_TIME_ATTRIBUTE_NAMES) == {"bounds"}


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
            niwot_output.pandas.iloc[:20].copy(), climate=niwot_output.climate.head(20)
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

    @staticmethod
    def with_a_missing_length(value):
        """Eight 3-hourly steps whose fourth has a NaT length and holds *value*."""
        times = pd.date_range("2000-01-01T03:00", periods=8, freq="3h")
        lengths = np.full(8, 3, dtype="timedelta64[h]").astype("timedelta64[ns]")
        lengths[3] = np.timedelta64("NaT")
        values = np.arange(8.0)
        values[3] = value
        return xr.DataArray(
            values,
            dims="time",
            coords={
                "time": times,
                "time_step_start": ("time", (times - pd.Timedelta("3h")).values),
                "time_step_length": ("time", lengths),
            },
            name="net_ecosystem_exchange",
            attrs={"kind": "timestep_total", "units": "g m-2", "long_name": "NEE"},
        )

    def test_a_padding_row_is_not_a_step(self):
        """A NaT length casts to the int64 sentinel, not to a missing value."""
        daily = aggregate_time(self.with_a_missing_length(np.nan), "1D")
        assert float(daily.sum()) == pytest.approx(28.0 - 3.0)
        assert (daily["time_step_length"].values > np.timedelta64(0)).all()
        assert float(daily["time_step_length"].sum() / np.timedelta64(1, "h")) == 21.0

    def test_a_row_with_a_value_and_no_length_is_refused_not_dropped(self):
        with pytest.raises(ValueError, match="these are not padding"):
            aggregate_time(self.with_a_missing_length(3.0), "1D")
        with pytest.raises(ValueError, match="these are not padding"):
            aggregation_counts(self.with_a_missing_length(3.0), "1D")


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
    """A cell is labeled at the last step end it holds, as pySIPNET labels it."""

    def test_a_daily_driver_cell_is_labeled_at_midnight_ending_it(self, real_drivers):
        par = site_1_member_1(real_drivers, "photosynthetically_active_radiation")
        daily = aggregate_time(par, "1D")
        ends = pd.DatetimeIndex(daily["time"].values)
        assert (ends == ends.normalize()).all()
        np.testing.assert_array_equal(
            daily["time_step_start"].values, (ends - pd.Timedelta(days=1)).to_numpy()
        )

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

    def test_a_driver_name_resolves_through_the_climate_registry(self, real_drivers):
        par = site_1_member_1(real_drivers, "photosynthetically_active_radiation")
        stripped = par.copy()
        stripped.attrs = {}
        assert aggregate_time(stripped, "1D").attrs["kind"] == "timestep_total"


class TestAggregateTimeLastSeesAGapAnywhereInTheCell:
    """``last`` reads only a cell's last step, so a gap before it must still show."""

    @staticmethod
    def second_day(field):
        """The positions of the steps in the second calendar cell, of which there are several."""
        day = pd.DatetimeIndex(field["time"].values).ceil("D")
        steps = np.flatnonzero(day == day.unique()[1])
        assert steps.size >= 2
        return steps

    def test_a_gap_before_the_last_step_makes_the_cell_missing(self, niwot_output):
        field = from_sipnet_output(niwot_output, "wood_carbon")["wood_carbon"]
        clean = aggregate_time(field, "1D")
        gappy = field.copy(data=field.values.copy())
        gappy[self.second_day(field)[0]] = np.nan
        daily = aggregate_time(gappy, "1D")
        np.testing.assert_array_equal(daily["time"].values, clean["time"].values)
        assert np.isnan(daily.values[1])
        np.testing.assert_array_equal(np.delete(daily.values, 1), np.delete(clean.values, 1))

    def test_on_a_stack_only_the_gapped_site_is_missing(self, niwot_output):
        from sipnet_calibration.fields import stack_model_outputs

        table = pd.DataFrame({"site_id": [1, 2], "lon": [0.0, 1.0], "lat": [0.0, 1.0]})
        run = niwot_output.select(["wood_carbon"])
        stacked = stack_model_outputs({(1, 0): run, (2, 0): run}, site_table=table)["wood_carbon"]
        values = stacked.values.copy()
        values[0, 1, self.second_day(stacked)[0]] = np.nan  # member 0, site 2
        daily = aggregate_time(stacked.copy(data=values), "1D")
        assert np.isnan(daily.sel(site=2, member=0).values[1])
        assert np.isfinite(daily.sel(site=1, member=0).values).all()

    def test_the_stated_default_is_what_is_masked(self, niwot_output):
        """A pool's default is last, so omitting how is the case that matters."""
        field = from_sipnet_output(niwot_output, "soil_water")["soil_water"]
        gappy = field.copy(data=field.values.copy())
        gappy[self.second_day(field)[0]] = np.nan
        assert np.isnan(aggregate_time(gappy, "1D").values[1])


class TestAggregationCountsLabelCellsAsTheAggregateDoes:
    def test_masking_the_aggregate_by_its_counts_keeps_every_cell(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        daily = aggregate_time(field, "1D")
        counts = aggregation_counts(field, "1D")
        np.testing.assert_array_equal(counts["time"].values, daily["time"].values)
        kept = daily.where(counts >= 1)
        assert kept.sizes == daily.sizes
        np.testing.assert_array_equal(kept.values, daily.values)

    def test_counts_are_how_many_steps_each_cell_holds(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        day = pd.Series(1, index=pd.DatetimeIndex(field["time"].values).ceil("D"))
        counts = aggregation_counts(field, "1D")
        np.testing.assert_array_equal(counts.values, day.groupby(level=0).size().to_numpy())
        assert counts.dtype == np.int64 and counts.name is None and counts.attrs == {}

    def test_an_observation_field_is_counted_on_its_calendar_cells(self):
        observed = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0], dims="time",
            coords={"time": pd.date_range("2012-01-01 12:00", periods=4, freq="12h")},
            attrs={"units": "g m-2", "kind": "timestep_mean"}, name="x",
        )
        daily = aggregate_time(observed, "1D", how="mean")
        counts = aggregation_counts(observed, "1D")
        np.testing.assert_array_equal(counts["time"].values, daily["time"].values)
        assert counts.values.tolist() == [1, 2]  # (01-01, 01-02] holds 12:00 and a gap at 00:00


class TestAggregateTimeReadsAnAliasedName:
    def test_a_field_named_by_an_alias_and_without_a_kind_is_aggregated(self, niwot_output):
        field = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        aliased = field.rename("nee")
        aliased.attrs = {key: value for key, value in field.attrs.items() if key != "kind"}
        daily = aggregate_time(aliased, "1D")
        np.testing.assert_allclose(daily.values, aggregate_time(field, "1D").values)
        assert daily.name == "nee" and daily.attrs["kind"] == "timestep_total"

    def test_a_refusal_names_an_unnamed_fields_derivation(self, niwot_output):
        from pysipnet.arithmetic import divide_with_units, step_length

        nee = from_sipnet_output(niwot_output, "nee")["net_ecosystem_exchange"]
        rate = divide_with_units(nee, step_length(nee, "d"))
        assert rate.name is None and rate.attrs.get("derivation")
        with pytest.raises(ValueError) as raised:
            aggregate_time(rate, "1D", how="sum")
        assert rate.attrs["derivation"] in str(raised.value)


def _daily_observation(n=8, freq="12h"):
    """An observation field with no interval coordinates, which takes the calendar path."""
    return xr.DataArray(
        np.arange(1.0, n + 1), dims="time",
        coords={"time": pd.date_range("2012-01-01 12:00", periods=n, freq=freq)},
        attrs={"units": "g m-2", "kind": "timestep_mean"}, name="x",
    )


class TestAggregateTimeOnCalendarCells:
    def test_upsampling_is_refused(self):
        with pytest.raises(ValueError, match="interpolate"):
            aggregate_time(_daily_observation(freq="1D"), "1h", how="mean")

    def test_a_bad_frequency_is_refused_with_pandas_reason(self):
        with pytest.raises(ValueError, match="pandas offset alias"):
            aggregate_time(_daily_observation(), "bogus", how="mean")
        with pytest.raises(ValueError, match="'ME'"):
            aggregate_time(_daily_observation(), "M", how="mean")

    @pytest.mark.parametrize("how", ["last", "mean", "sum"])
    def test_a_cell_holding_a_nan_is_nan(self, how):
        observed = _daily_observation(freq="6h")
        observed.attrs["kind"] = {"last": "timestep_end_state", "mean": "timestep_mean", "sum": "timestep_total"}[how]
        observed[1] = np.nan  # 2012-01-01 18:00, not the last value of (01-01, 01-02]
        daily = aggregate_time(observed, "1D", how=how)
        assert np.isnan(daily.sel(time="2012-01-02").values).all()
        assert np.isfinite(daily.sel(time="2012-01-03").values).all()

    def test_the_attributes_say_what_the_values_now_are(self):
        daily = aggregate_time(_daily_observation(), "1D")
        assert daily.attrs["resampling"] == "mean of timestep_mean values over 1D"
        assert daily.attrs["kind"] == "timestep_mean" and "time_reference" in daily.attrs
        assert daily.attrs["units"] == "g m-2"

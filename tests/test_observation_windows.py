"""The window and instant readings in ``sipnet_calibration.observation.time_alignment``:
``reduce_windows``, ``select_timestep_at``, the window builders and the counts.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysipnet.arithmetic import divide_with_units, multiply_with_units, step_length

from conftest import site_table_of

from sipnet_calibration.constraints import TIME_BOUNDS_END, TIME_BOUNDS_START
from sipnet_calibration.conventions import TIMESTEP_LENGTH, TIMESTEP_START
from sipnet_calibration.observation.time_alignment import (
    SELECTED_STEP_COORD,
    WINDOW_REDUCTIONS,
    aggregate_time,
    aggregation_counts,
    check_run_spans_the_windows,
    reduce_windows,
    run_window,
    select_timestep_at,
    window_counts,
    windows_from_time_bounds,
)


@pytest.fixture(scope="module")
def niwot(niwot_output):
    """The Niwot variables these tests read, on pySIPNET's irregular axis."""
    return niwot_output.select(
        ["wood_carbon", "net_ecosystem_exchange", "soil_wetness_fraction", "cumulative_net_ecosystem_exchange"]
    )


def _daily_windows(array):
    """Right-closed calendar days covering the array's record."""
    start = pd.Timestamp(array[TIMESTEP_START].values.min()).floor("D")
    end = pd.Timestamp(array["time"].values.max()).ceil("D")
    edges = pd.date_range(start, end, freq="1D")
    return pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")


class TestReduceWindowsAgreesWithAggregateTime:
    @pytest.mark.parametrize("name, how", [("net_ecosystem_exchange", "sum"), ("wood_carbon", "last"), ("soil_wetness_fraction", "mean")])
    def test_daily_windows_reproduce_the_calendar_cells(self, niwot, name, how):
        array = niwot[name]
        by_window = reduce_windows(array, _daily_windows(array), how)
        by_cell = aggregate_time(array, "1D", how=how)
        np.testing.assert_allclose(by_window.dropna("time").values, by_cell.values)

    def test_the_mean_is_weighted_by_the_step_length(self, niwot):
        """The Niwot steps alternate 7 h and 10 h, so an unweighted mean would differ."""
        array = niwot["soil_wetness_fraction"]
        windows = _daily_windows(array)
        weighted = reduce_windows(array, windows, "mean")
        unweighted = array.groupby_bins("time", bins=windows.right.insert(0, windows.left[0]), labels=windows.right, right=True).mean()
        assert not np.allclose(weighted.dropna("time").values, unweighted.dropna("time_bins").values)
        expected = aggregate_time(array, "1D", how="mean")
        np.testing.assert_allclose(weighted.dropna("time").values, expected.values)

    def test_the_kind_is_rewritten_as_aggregate_time_rewrites_it(self, niwot):
        array = niwot["wood_carbon"]
        mean = reduce_windows(array, _daily_windows(array), "mean")
        assert mean.attrs["kind"] == "timestep_mean"
        last = reduce_windows(array, _daily_windows(array), "last")
        assert last.attrs["kind"] == "timestep_end_state"
        assert "reduction" in mean.attrs


class TestReduceWindowsRefusals:
    def test_a_state_may_not_be_summed(self, niwot):
        array = niwot["wood_carbon"]
        with pytest.raises(ValueError, match="pool|additive"):
            reduce_windows(array, _daily_windows(array), "sum")

    def test_a_total_may_not_take_an_extreme(self, niwot):
        array = niwot["net_ecosystem_exchange"]
        with pytest.raises(ValueError, match="only a level"):
            reduce_windows(array, _daily_windows(array), "max")

    def test_a_running_total_may_not_take_a_first(self, niwot):
        array = niwot["cumulative_net_ecosystem_exchange"]
        with pytest.raises(ValueError, match="only a level"):
            reduce_windows(array, _daily_windows(array), "first")

    def test_an_unknown_method_is_refused(self, niwot):
        array = niwot["wood_carbon"]
        with pytest.raises(ValueError, match="how must be one of"):
            reduce_windows(array, _daily_windows(array), "median")

    def test_the_methods_are_the_documented_six(self):
        assert set(WINDOW_REDUCTIONS) == {"sum", "mean", "last", "min", "max", "first"}

    def test_overlapping_windows_are_refused(self, niwot):
        array = niwot["wood_carbon"]
        t = pd.Timestamp(array["time"].values[0])
        windows = pd.IntervalIndex.from_arrays([t, t + pd.Timedelta("1D")], [t + pd.Timedelta("2D"), t + pd.Timedelta("3D")], closed="right")
        with pytest.raises(ValueError, match="overlap"):
            reduce_windows(array, windows, "last")

    def test_duplicate_timestamps_are_refused(self, niwot):
        array = niwot["wood_carbon"]
        doubled = xr.concat([array, array], dim="time")
        with pytest.raises(ValueError, match="do not increase|duplicates"):
            reduce_windows(doubled, _daily_windows(array), "last")

    def test_wrong_label_count_is_refused(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        with pytest.raises(ValueError, match="one label per window"):
            reduce_windows(array, windows, "last", labels=windows.right[:-1])


class TestReduceWindowsSemantics:
    def test_a_gap_makes_the_window_missing_for_every_method(self, niwot):
        array = niwot["wood_carbon"].copy()
        array[3] = np.nan
        windows = _daily_windows(array)
        gapped = int(windows.get_indexer([pd.Timestamp(array["time"].values[3])])[0])
        for how in ("mean", "last", "min", "max", "first"):
            reduced = reduce_windows(array, windows, how)
            assert np.isnan(reduced.values[gapped]), how
            assert np.isfinite(reduced.values[gapped + 3]), how

    def test_a_window_holding_no_step_is_missing(self, niwot):
        array = niwot["wood_carbon"]
        far = pd.Timestamp("2005-01-01")
        windows = pd.IntervalIndex.from_arrays([far], [far + pd.Timedelta("1D")], closed="right")
        reduced = reduce_windows(array, windows, "last")
        assert reduced.shape == (1,) and np.isnan(reduced.values[0])
        assert np.isnat(reduced[TIMESTEP_START].values[0])

    def test_a_step_belongs_to_the_window_its_end_falls_in(self, niwot):
        array = niwot["wood_carbon"]
        end = pd.Timestamp(array["time"].values[4])
        # a window whose right edge is exactly this step's end holds it; the next window does not
        windows = pd.IntervalIndex.from_arrays([end - pd.Timedelta("1h"), end], [end, end + pd.Timedelta("1h")], closed="right")
        reduced = reduce_windows(array, windows, "last")
        assert reduced.values[0] == array.values[4]
        assert np.isnan(reduced.values[1])

    def test_the_coverage_coordinates_describe_the_steps_combined(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        reduced = reduce_windows(array, windows, "last")
        full = reduced.isel(time=5)
        assert full[TIMESTEP_LENGTH].values == np.timedelta64(24, "h")
        last_end = pd.Timestamp(full[SELECTED_STEP_COORD].values)
        assert windows[5].left < last_end <= windows[5].right
        assert last_end == pd.Timestamp(array["time"].values[windows.get_indexer(pd.DatetimeIndex(array["time"].values)) == 5][-1])

    def test_the_observations_labels_and_attributes_are_kept(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        labels = xr.DataArray(windows.left, dims="time", attrs={"time_reference": "the key"})
        reduced = reduce_windows(array, windows, "last", labels=labels)
        np.testing.assert_array_equal(reduced["time"].values, windows.left.values)
        assert reduced["time"].attrs["time_reference"] == "the key"

    def test_other_dimensions_are_carried_through(self, niwot):
        array = niwot["wood_carbon"]
        stacked = xr.concat([array.assign_coords(site=1), (array * 2).assign_coords(site=2)], dim="site")
        stacked.attrs = array.attrs
        reduced = reduce_windows(stacked, _daily_windows(array), "last")
        assert reduced.dims == ("site", "time")
        np.testing.assert_allclose(reduced.sel(site=2).values, 2 * reduced.sel(site=1).values)


class TestSelectTimestepAt:
    def test_a_label_at_a_step_end_reads_that_step(self, niwot):
        array = niwot["wood_carbon"]
        ends = pd.DatetimeIndex(array["time"].values)
        picked = select_timestep_at(array, ends[[3, 10]])
        np.testing.assert_array_equal(picked.values, array.values[[3, 10]])
        np.testing.assert_array_equal(picked[SELECTED_STEP_COORD].values, ends[[3, 10]].values)

    def test_a_label_inside_a_step_reads_the_step_containing_it(self, niwot):
        array = niwot["wood_carbon"]
        starts = pd.DatetimeIndex(array[TIMESTEP_START].values)
        inside = starts[[3, 10]] + pd.Timedelta("1s")
        picked = select_timestep_at(array, inside)
        np.testing.assert_array_equal(picked.values, array.values[[3, 10]])
        np.testing.assert_array_equal(picked["time"].values, inside.values)

    def test_a_label_at_a_step_start_reads_the_previous_step(self, niwot):
        array = niwot["wood_carbon"]
        starts = pd.DatetimeIndex(array[TIMESTEP_START].values)
        picked = select_timestep_at(array, starts[[5]])
        assert picked.values[0] == array.values[4]

    def test_a_label_outside_the_record_is_refused(self, niwot):
        array = niwot["wood_carbon"]
        with pytest.raises(ValueError, match="fall in no timestep"):
            select_timestep_at(array, pd.DatetimeIndex(["1990-01-01"]))
        with pytest.raises(ValueError, match="fall in no timestep"):
            select_timestep_at(array, pd.DatetimeIndex([array["time"].values[-1] + np.timedelta64(1, "s")]))

    def test_a_total_has_no_value_at_an_instant(self, niwot):
        with pytest.raises(ValueError, match="no value at an instant"):
            select_timestep_at(niwot["net_ecosystem_exchange"], pd.DatetimeIndex(niwot["time"].values[[3]]))

    def test_an_unnamed_result_is_called_by_its_derivation(self, niwot):
        doubled = multiply_with_units(niwot["net_ecosystem_exchange"], 2.0)
        with pytest.raises(ValueError, match=r"'net_ecosystem_exchange \* 2\.0' is of kind"):
            select_timestep_at(doubled, pd.DatetimeIndex(niwot["time"].values[[3]]))

    def test_a_total_over_its_step_length_has_a_value_at_an_instant(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        rate = divide_with_units(nee, step_length(nee))
        starts = pd.DatetimeIndex(rate[TIMESTEP_START].values)
        picked = select_timestep_at(rate, starts[[5]] + pd.Timedelta("30min"))
        assert picked.attrs["kind"] == "daily_rate" and picked.attrs["units"] == "g m-2 d-1"
        assert picked.values[0] == pytest.approx(float(rate.values[5]), rel=1e-12)

    def test_a_running_total_has_no_value_at_an_instant(self, niwot):
        with pytest.raises(ValueError, match="no value at an instant"):
            select_timestep_at(niwot["cumulative_net_ecosystem_exchange"], pd.DatetimeIndex(niwot["time"].values[[3]]))

    def test_a_step_mean_reads_the_containing_step(self, niwot):
        array = niwot["soil_wetness_fraction"]
        starts = pd.DatetimeIndex(array[TIMESTEP_START].values)
        picked = select_timestep_at(array, starts[[7]] + pd.Timedelta("30min"))
        assert picked.values[0] == array.values[7]
        assert "contains the label" in picked.attrs["time_reference"]

    def test_an_array_without_interval_coordinates_is_refused(self):
        array = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.DatetimeIndex(["2000-01-01", "2000-01-02"])}, attrs={"kind": "timestep_end_state"})
        with pytest.raises(ValueError, match="carries no"):
            select_timestep_at(array, pd.DatetimeIndex(["2000-01-01"]))

    def test_labels_must_increase(self, niwot):
        array = niwot["wood_carbon"]
        ends = pd.DatetimeIndex(array["time"].values)
        with pytest.raises(ValueError, match="strictly increasing"):
            select_timestep_at(array, ends[[10, 3]])

    def test_a_dataarray_of_labels_keeps_its_attributes(self, niwot):
        array = niwot["wood_carbon"]
        labels = xr.DataArray(pd.DatetimeIndex(array["time"].values[[2, 9]]), dims="time", attrs={"time_reference": "the composite date"})
        picked = select_timestep_at(array, labels)
        assert picked["time"].attrs["time_reference"] == "the composite date"

    def test_other_dimensions_are_carried_through(self, niwot):
        array = niwot["wood_carbon"]
        stacked = xr.concat([array.assign_coords(site=1), (array * 2).assign_coords(site=2)], dim="site")
        stacked.attrs = array.attrs
        picked = select_timestep_at(stacked, pd.DatetimeIndex(array["time"].values[[2, 9]]))
        assert picked.dims == ("site", "time")
        np.testing.assert_allclose(picked.sel(site=2).values, 2 * picked.sel(site=1).values)


class TestWindowBuilders:
    def test_windows_from_time_bounds(self):
        observed = xr.DataArray(
            [[1.0, 2.0]],
            dims=("site", "time"),
            coords={
                "site": [1],
                "time": pd.DatetimeIndex(["2012-01-01", "2013-01-01"]),
                TIME_BOUNDS_START: ("time", pd.DatetimeIndex(["2012-01-01", "2013-01-01"])),
                TIME_BOUNDS_END: ("time", pd.DatetimeIndex(["2013-01-01", "2014-01-01"])),
            },
        )
        windows = windows_from_time_bounds(observed)
        assert windows.closed == "right"
        assert list(windows.left) == list(pd.DatetimeIndex(["2012-01-01", "2013-01-01"]))
        assert list(windows.right) == list(pd.DatetimeIndex(["2013-01-01", "2014-01-01"]))

    def test_an_array_without_bounds_is_refused(self):
        observed = xr.DataArray([[1.0]], dims=("site", "time"), coords={"site": [1], "time": pd.DatetimeIndex(["2012-07-15"])})
        with pytest.raises(ValueError, match="documents no interval"):
            windows_from_time_bounds(observed)

    def test_the_real_annual_product_carries_bounds(self):
        constraints = pytest.importorskip("sipnet_calibration.constraints")
        try:
            wood = constraints.constraint_fields("landtrendr_aboveground_biomass", sites=[3851])["landtrendr_aboveground_biomass"]
        except FileNotFoundError as error:
            pytest.skip(str(error))
        windows = windows_from_time_bounds(wood)
        assert len(windows) == wood.sizes["time"]
        assert all(w.right - w.left >= pd.Timedelta("365D") for w in windows)

    def test_run_window_spans_the_record(self, niwot):
        array = niwot["wood_carbon"]
        window = run_window(array)
        assert len(window) == 1
        assert window[0].left == pd.Timestamp(array[TIMESTEP_START].values[0])
        assert window[0].right == pd.Timestamp(array["time"].values[-1])
        reduced = reduce_windows(array, window, "mean")
        assert reduced.shape == (1,) and np.isfinite(reduced.values[0])


class TestCounts:
    def test_aggregation_counts_align_with_the_aggregate_by_position(self, niwot):
        array = niwot["wood_carbon"].copy()
        array[3] = np.nan
        counts = aggregation_counts(array, "1D")
        aggregate = aggregate_time(array, "1D")
        assert counts.shape == aggregate.shape
        assert counts.dtype == np.int64
        gapped = int(np.flatnonzero(np.isnan(aggregate.values))[0])
        assert counts.values[gapped] == 1  # two steps that day, one of them missing
        assert counts.values[gapped + 1] == 2
        assert counts.attrs == {}

    def test_window_counts_align_with_reduce_windows(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        counts = window_counts(array, windows)
        reduced = reduce_windows(array, windows, "last")
        assert counts.shape == reduced.shape
        assert (counts.values > 0).tolist() == np.isfinite(reduced.values).tolist()


class TestWindowReductionValues:
    """The three methods pySIPNET does not have, against hand computations."""

    @pytest.fixture
    def membership(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        return array, windows, windows.get_indexer(pd.DatetimeIndex(array["time"].values))

    @pytest.mark.parametrize("how, pick", [("first", lambda v: v[0]), ("min", np.min), ("max", np.max), ("last", lambda v: v[-1])])
    def test_each_window_is_the_reduction_of_its_own_steps(self, membership, how, pick):
        array, windows, member = membership
        reduced = reduce_windows(array, windows, how)
        for k in range(len(windows)):
            steps = array.values[member == k]
            if steps.size == 0:
                assert np.isnan(reduced.values[k])
            else:
                assert reduced.values[k] == pick(steps), (how, k)

    def test_first_and_last_differ_on_a_window_with_two_steps(self, membership):
        array, windows, member = membership
        first = reduce_windows(array, windows, "first")
        last = reduce_windows(array, windows, "last")
        two = np.flatnonzero(np.bincount(member[member >= 0], minlength=len(windows)) == 2)
        assert two.size and not np.allclose(first.values[two], last.values[two])

    def test_a_total_with_a_gap_sums_to_missing(self, niwot):
        array = niwot["net_ecosystem_exchange"].copy()
        array[3] = np.nan
        windows = _daily_windows(array)
        gapped = int(windows.get_indexer([pd.Timestamp(array["time"].values[3])])[0])
        reduced = reduce_windows(array, windows, "sum")
        assert np.isnan(reduced.values[gapped]) and np.isfinite(reduced.values[gapped + 2])

    def test_the_window_start_is_the_earliest_step_start(self, membership):
        array, windows, member = membership
        reduced = reduce_windows(array, windows, "last")
        starts = array[TIMESTEP_START].values
        for k in range(len(windows)):
            if (member == k).any():
                assert reduced[TIMESTEP_START].values[k] == starts[member == k].min()

    def test_window_counts_do_not_count_a_gap(self, niwot):
        array = niwot["wood_carbon"].copy()
        windows = _daily_windows(array)
        before = window_counts(array, windows)
        array[3] = np.nan
        after = window_counts(array, windows)
        gapped = int(windows.get_indexer([pd.Timestamp(array["time"].values[3])])[0])
        assert after.values[gapped] == before.values[gapped] - 1

    def test_run_window_holds_the_last_step(self, niwot):
        array = niwot["wood_carbon"]
        last = reduce_windows(array, run_window(array), "last")
        assert last.values[0] == array.values[-1]
        assert window_counts(array, run_window(array)).values[0] == array.sizes["time"]


class TestLabelClockAndFrequencyRefusals:
    def test_reduce_windows_labels_must_increase(self, niwot):
        array = niwot["wood_carbon"]
        windows = _daily_windows(array)
        with pytest.raises(ValueError, match="strictly increasing"):
            reduce_windows(array, windows, "last", labels=windows.right[::-1])

    def test_a_label_on_another_clock_is_refused(self, niwot):
        array = niwot["wood_carbon"]
        aware = pd.DatetimeIndex(array["time"].values[[3]]).tz_localize("UTC")
        with pytest.raises(ValueError, match="time zone"):
            select_timestep_at(array, aware)
        windows = pd.IntervalIndex.from_arrays(aware - pd.Timedelta("1D"), aware, closed="right")
        with pytest.raises(ValueError, match="time zone"):
            reduce_windows(array, windows, "last")

    def test_a_label_at_the_start_of_a_step_after_a_gap_is_outside(self, niwot):
        array = niwot["wood_carbon"]
        gapped = array.isel(time=[i for i in range(array.sizes["time"]) if i != 10])
        starts = pd.DatetimeIndex(array[TIMESTEP_START].values)
        ends = pd.DatetimeIndex(array["time"].values)
        # the label equals step 11's start and step 10's end; step 10 is gone, so nothing contains it
        with pytest.raises(ValueError, match="fall in no timestep"):
            select_timestep_at(gapped, starts[[11]])
        with pytest.raises(ValueError, match="fall in no timestep"):
            select_timestep_at(gapped, ends[[10]] - pd.Timedelta("1h"))

    def test_a_single_zero_dimensional_label(self, niwot):
        array = niwot["wood_carbon"]
        label = xr.DataArray(array["time"].values[7])
        picked = select_timestep_at(array, label)
        assert picked.shape == (1,) and picked.values[0] == array.values[7]

    def test_aggregate_time_refuses_a_bad_or_finer_frequency(self, niwot):
        array = niwot["wood_carbon"]
        with pytest.raises(ValueError, match="offset alias"):
            aggregate_time(array, "daily")
        with pytest.raises(ValueError, match="shorter than the shortest step"):
            aggregate_time(array, "1h")
        with pytest.raises(ValueError, match="offset alias"):
            aggregation_counts(array, 3)

    def test_a_finer_label_is_not_truncated_onto_the_axis(self, niwot):
        array = niwot["wood_carbon"]
        coarse = array.assign_coords(
            time=array["time"].values.astype("datetime64[us]"),
            time_step_start=("time", array[TIMESTEP_START].values.astype("datetime64[us]")),
        )
        end0 = pd.Timestamp(array["time"].values[0])
        picked = select_timestep_at(coarse, pd.DatetimeIndex([end0 + pd.Timedelta("500ns")]))
        assert picked.values[0] == array.values[1]


class TestPaddedStacks:
    """Two records of different length stacked on one axis leave NaT padding."""

    @pytest.fixture
    def padded(self, niwot):
        full = niwot["wood_carbon"].assign_coords(site=1)
        short = niwot["wood_carbon"].isel(time=slice(0, 40)).assign_coords(site=2)
        stacked = xr.concat([full, short], dim="site", join="outer", coords="different", compat="equals", combine_attrs="override")
        return stacked

    def test_reduce_windows_ignores_the_padding_of_a_selected_site(self, padded, niwot):
        array = niwot["wood_carbon"]
        short = padded.sel(site=2)
        assert np.isnat(short[TIMESTEP_START].values).any()
        windows = _daily_windows(array)
        reduced = reduce_windows(short, windows, "last")
        expected = reduce_windows(array.isel(time=slice(0, 40)), windows, "last")
        np.testing.assert_array_equal(reduced.values, expected.values)

    def test_select_timestep_at_ignores_the_padding(self, padded, niwot):
        array = niwot["wood_carbon"]
        short = padded.sel(site=2)
        labels = pd.DatetimeIndex(array["time"].values[[3, 20]])
        np.testing.assert_array_equal(select_timestep_at(short, labels).values, array.values[[3, 20]])
        with pytest.raises(ValueError, match="fall in no timestep"):
            select_timestep_at(short, pd.DatetimeIndex(array["time"].values[[50]]))


class TestBoundsOrientation:
    def test_the_bounds_coordinates_are_start_then_end(self):
        from sipnet_calibration.constraints import _time_bounds_coords

        times = pd.DatetimeIndex(["2012-01-01", "2013-01-01"])
        bounds = np.array([[t, t + pd.Timedelta("366D")] for t in times]).astype("datetime64[ns]")
        dataset = xr.Dataset(coords={"time": times, "time_bounds": (("time", "bounds"), bounds)})
        coords = _time_bounds_coords(dataset)
        np.testing.assert_array_equal(coords[TIME_BOUNDS_START].values, times.values.astype("datetime64[ns]"))
        assert (coords[TIME_BOUNDS_END].values > coords[TIME_BOUNDS_START].values).all()

    def test_reversed_bounds_are_refused(self):
        observed = xr.DataArray(
            [[1.0]],
            dims=("site", "time"),
            coords={
                "site": [1],
                "time": pd.DatetimeIndex(["2012-01-01"]),
                TIME_BOUNDS_START: ("time", pd.DatetimeIndex(["2013-01-01"])),
                TIME_BOUNDS_END: ("time", pd.DatetimeIndex(["2012-01-01"])),
            },
        )
        with pytest.raises(ValueError, match="must follow its start"):
            windows_from_time_bounds(observed)


class TestEveryFunctionChecksTheStack:
    """Runs on different time axes, stacked, have interval coordinates per site."""

    @pytest.fixture
    def per_site(self, niwot):
        full = niwot["wood_carbon"].assign_coords(site=1)
        short = niwot["wood_carbon"].isel(time=slice(0, 40)).assign_coords(site=2)
        stacked = xr.concat([full, short], dim="site", join="outer", coords="different", compat="equals", combine_attrs="override")
        assert stacked[TIMESTEP_START].dims == ("site", "time")
        return stacked

    def test_run_window_refuses_it(self, per_site):
        with pytest.raises(ValueError, match="different time axes"):
            run_window(per_site)

    def test_aggregation_counts_refuses_it(self, per_site):
        with pytest.raises(ValueError, match="different time axes"):
            aggregation_counts(per_site, "1D")

    def test_window_counts_refuses_it(self, per_site, niwot):
        with pytest.raises(ValueError, match="different time axes"):
            window_counts(per_site, _daily_windows(niwot["wood_carbon"]))

    def test_run_window_refuses_a_record_with_no_steps(self, niwot):
        with pytest.raises(ValueError, match="no timesteps left"):
            run_window(niwot["wood_carbon"].isel(time=slice(0, 0)))


class TestCombinedLengthsAreExact:
    """Step lengths are summed as integer nanoseconds, not through float days."""

    @pytest.fixture
    def year(self):
        n = 8760
        ends = pd.date_range("2001-01-01T01:00", periods=n, freq="1h").as_unit("ns")
        length = np.full(n, 3_600 * 10**9 + 1, dtype="int64").view("timedelta64[ns]")  # 1 h + 1 ns
        return xr.DataArray(
            np.ones(n), dims="time",
            coords={"time": ends, TIMESTEP_START: ("time", (ends - pd.Timedelta("1h")).values), TIMESTEP_LENGTH: ("time", length)},
            attrs={"kind": "timestep_total", "units": "g m-2"}, name="x",
        ), int(length.astype("int64").sum())

    def test_a_yearly_cell(self, year):
        field, exact = year
        assert int(aggregate_time(field, "YS")[TIMESTEP_LENGTH].values.astype("int64")[0]) == exact

    def test_a_run_window(self, year):
        field, exact = year
        reduced = reduce_windows(field, run_window(field), "sum")
        assert int(reduced[TIMESTEP_LENGTH].values.astype("int64")[0]) == exact


class TestWindowAndLabelRefusals:
    @staticmethod
    def bounded(start, end):
        return xr.DataArray(
            [[1.0]], dims=("site", "time"),
            coords={"site": [1], "time": pd.DatetimeIndex(["2012-01-01"]), TIME_BOUNDS_START: ("time", start), TIME_BOUNDS_END: ("time", end)},
            name="annual",
        )

    def test_a_missing_bound_is_refused(self):
        with pytest.raises(ValueError, match="'annual': a time bound is missing"):
            windows_from_time_bounds(self.bounded(pd.DatetimeIndex([pd.NaT]), pd.DatetimeIndex(["2013-01-01"])))

    def test_bounds_that_are_not_datetimes_are_refused(self):
        with pytest.raises(ValueError, match="must hold datetimes"):
            windows_from_time_bounds(self.bounded([2012], pd.DatetimeIndex(["2013-01-01"])))

    def test_decreasing_windows_are_refused(self, niwot):
        windows = _daily_windows(niwot["wood_carbon"])[::-1]
        with pytest.raises(ValueError, match="increasing order"):
            reduce_windows(niwot["wood_carbon"], windows, "last")

    def test_empty_windows_are_refused(self, niwot):
        with pytest.raises(ValueError, match="windows is empty"):
            reduce_windows(niwot["wood_carbon"], _daily_windows(niwot["wood_carbon"])[:0], "last")

    def test_a_window_with_a_missing_edge_is_refused(self, niwot):
        t = pd.Timestamp(niwot["time"].values[0])
        windows = pd.IntervalIndex.from_arrays(pd.DatetimeIndex([t, pd.NaT]), pd.DatetimeIndex([t + pd.Timedelta("1D"), pd.NaT]), closed="right")
        with pytest.raises(ValueError, match="missing edge"):
            reduce_windows(niwot["wood_carbon"], windows, "last")

    def test_numeric_windows_are_refused(self, niwot):
        with pytest.raises(ValueError, match="intervals of datetimes"):
            reduce_windows(niwot["wood_carbon"], pd.IntervalIndex.from_breaks([0, 1, 2]), "last")

    def test_integer_labels_are_refused(self, niwot):
        with pytest.raises(ValueError, match="must be timestamps"):
            select_timestep_at(niwot["wood_carbon"], np.array([1, 2]))

    def test_no_labels_are_refused(self, niwot):
        with pytest.raises(ValueError, match="no labels were given"):
            select_timestep_at(niwot["wood_carbon"], pd.DatetimeIndex([]))

    def test_a_missing_timestamp_on_the_axis_is_refused(self, niwot):
        array = niwot["wood_carbon"]
        broken = array.assign_coords(time=np.where(np.arange(array.sizes["time"]) == 3, np.datetime64("NaT"), array["time"].values))
        with pytest.raises(ValueError, match=r"missing timestamp \(NaT\)"):
            aggregate_time(broken, "1D")

    def test_a_zero_frequency_is_refused(self, niwot):
        with pytest.raises(ValueError, match="positive frequency"):
            aggregate_time(niwot["wood_carbon"], "0D")

    def test_a_dataset_is_a_type_error(self, niwot):
        with pytest.raises(TypeError, match="align one at a time"):
            aggregate_time(niwot, "1D")


class TestOwnCadenceAndExtremes:
    def test_aggregating_at_the_fields_own_cadence_is_the_field(self):
        daily = xr.DataArray(
            np.arange(10.0), dims="time", coords={"time": pd.date_range("2000-01-02", periods=10, freq="D")},
            attrs={"kind": "timestep_total"}, name="x",
        )
        again = aggregate_time(daily, "1D")
        np.testing.assert_array_equal(again.values, daily.values)
        np.testing.assert_array_equal(again["time"].values, daily["time"].values)

    def test_an_empty_calendar_cell_is_dropped_not_zero(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        day = pd.Timestamp(nee["time"].values[0]).floor("D") + pd.Timedelta("3D")
        inside = (nee["time"] > np.datetime64(day)) & (nee["time"] <= np.datetime64(day + pd.Timedelta("1D")))
        daily = aggregate_time(nee.isel(time=~inside.values), "1D")
        assert np.datetime64(day + pd.Timedelta("1D")) not in daily["time"].values
        assert daily.sizes["time"] == aggregate_time(nee, "1D").sizes["time"] - 1

    @pytest.mark.parametrize("how, method", [("min", "minimum"), ("max", "maximum"), ("first", "point")])
    def test_the_cell_methods_of_an_extreme_say_which(self, niwot, how, method):
        array = niwot["wood_carbon"]
        reduced = reduce_windows(array, _daily_windows(array), how)
        assert reduced.attrs["cell_methods"] == f"time: {method}"
        assert reduced.attrs["kind"] == "timestep_end_state"
        assert reduced.attrs["time_reference"] == f"the {how} of the timestep_end_state values over the window"


class TestWindowAttributesAreLiterallyTrue:
    @pytest.mark.parametrize("how", ["min", "max", "first"])
    def test_an_extreme_of_step_means_claims_no_cell_method(self, niwot, how):
        array = niwot["soil_wetness_fraction"]
        assert array.attrs["kind"] == "timestep_mean" and array.attrs["cell_methods"].startswith("time: mean")
        reduced = reduce_windows(array, _daily_windows(array), how)
        assert "cell_methods" not in reduced.attrs
        assert reduced.attrs["kind"] == "timestep_mean"
        assert reduced.attrs["time_reference"] == f"the {how} of the timestep_mean values over the window"

    @pytest.mark.parametrize("how", ["min", "max", "first"])
    def test_an_extreme_of_a_rate_claims_no_cell_method(self, niwot, how):
        nee = niwot["net_ecosystem_exchange"]
        rate = divide_with_units(nee, step_length(nee))
        assert rate.attrs["kind"] == "daily_rate"
        assert "cell_methods" not in reduce_windows(rate, _daily_windows(rate), how).attrs

    def test_an_unweighted_window_mean_does_not_claim_weights(self):
        observed = xr.DataArray(
            np.arange(1.0, 9.0), dims="time",
            coords={"time": pd.date_range("2012-01-01 12:00", periods=8, freq="12h")},
            attrs={"units": "g m-2", "kind": "timestep_mean"}, name="x",
        )
        windows = pd.IntervalIndex.from_arrays([pd.Timestamp("2012-01-01")], [pd.Timestamp("2012-01-03")], closed="right")
        assert "weighted" not in reduce_windows(observed, windows, "mean").attrs["reduction"]

    def test_a_weighted_window_mean_says_so(self, niwot):
        array = niwot["soil_wetness_fraction"]
        reduced = reduce_windows(array, _daily_windows(array), "mean")
        assert reduced.attrs["reduction"].endswith(f"weighted by {TIMESTEP_LENGTH}")


class TestValuedRowsWithoutAnIntervalAreRefused:
    """Padding holds no value; a row with a value and a NaT interval is not padding."""

    @pytest.fixture
    def valued(self, niwot):
        array = niwot["wood_carbon"]
        start = array[TIMESTEP_START].values.copy()
        start[5] = np.datetime64("NaT")
        return array.assign_coords({TIMESTEP_START: ("time", start, array[TIMESTEP_START].attrs)})

    def test_every_reader_refuses_it(self, valued, niwot):
        windows = _daily_windows(niwot["wood_carbon"])
        labels = pd.DatetimeIndex(niwot["time"].values[[10]])
        for read in (
            lambda: reduce_windows(valued, windows, "last"),
            lambda: window_counts(valued, windows),
            lambda: select_timestep_at(valued, labels),
            lambda: run_window(valued),
            lambda: check_run_spans_the_windows(valued, windows[1:3], "x"),
            lambda: aggregate_time(valued, "1D"),
        ):
            with pytest.raises(ValueError, match="not padding"):
                read()

    def test_the_same_row_without_a_value_is_dropped(self, valued):
        padding = valued.copy(data=valued.values.copy())
        padding[5] = np.nan
        dropped = padding.isel(time=[i for i in range(padding.sizes["time"]) if i != 5])
        window = run_window(dropped)
        assert run_window(padding).equals(window)
        np.testing.assert_array_equal(reduce_windows(padding, window, "last").values, reduce_windows(dropped, window, "last").values)


class TestCheckRunSpansTheWindows:
    def test_a_window_beyond_a_selected_sites_shorter_record_is_refused(self, niwot):
        from sipnet_calibration.fields import stack_model_outputs

        run = niwot[["wood_carbon"]]
        table = site_table_of(1, 2, lon=[0.0, 1.0], lat=[0.0, 1.0])
        stacked = stack_model_outputs({(1, 0): run, (2, 0): run.isel(time=slice(0, 40))}, site_table=table)
        one = stacked.sel(site=2, member=0)["wood_carbon"]
        assert np.isnat(one[TIMESTEP_START].values).any()
        start, end = pd.Timestamp(one[TIMESTEP_START].values[0]), pd.Timestamp(niwot["time"].values[50])
        windows = pd.IntervalIndex.from_arrays([start], [end], closed="right")
        with pytest.raises(ValueError, match="reaches beyond the model record"):
            check_run_spans_the_windows(one, windows, "x")
        check_run_spans_the_windows(one, pd.IntervalIndex.from_arrays([start], [pd.Timestamp(niwot["time"].values[39])], closed="right"), "x")

    def test_a_field_without_interval_coordinates_is_refused_naming_the_reader(self, niwot):
        bare = niwot["wood_carbon"].drop_vars([TIMESTEP_START, TIMESTEP_LENGTH])
        windows = pd.IntervalIndex.from_arrays([bare["time"].values[0]], [bare["time"].values[5]], closed="right")
        with pytest.raises(ValueError, match="^ReduceOverTimeBounds on 'w' reads the interval"):
            check_run_spans_the_windows(bare, windows, "ReduceOverTimeBounds on 'w'")

    def test_the_message_names_the_first_window_beyond_the_record(self, niwot):
        wood = niwot["wood_carbon"]
        end = pd.Timestamp(wood["time"].values[-1])
        windows = pd.IntervalIndex.from_arrays(
            [end, end + pd.Timedelta("1D")], [end + pd.Timedelta("1D"), end + pd.Timedelta("2D")], closed="right"
        )
        with pytest.raises(ValueError, match=re.escape(f"the window {windows[0]} reaches")):
            check_run_spans_the_windows(wood, windows, "x")

    def test_windows_that_are_not_an_interval_index_are_a_type_error(self, niwot):
        with pytest.raises(TypeError, match="pandas.IntervalIndex"):
            check_run_spans_the_windows(niwot["wood_carbon"], [(0, 1)], "x")


class TestWindowEdgesAndLabels:
    def test_a_left_closed_window_takes_the_step_ending_on_its_left_edge(self, niwot):
        array = niwot["wood_carbon"]
        end = pd.Timestamp(array["time"].values[4])
        windows = pd.IntervalIndex.from_arrays([end - pd.Timedelta("1h"), end], [end, end + pd.Timedelta("1h")], closed="left")
        reduced = reduce_windows(array, windows, "last")
        assert np.isnan(reduced.values[0]) and reduced.values[1] == array.values[4]

    def test_repeated_labels_are_refused(self, niwot):
        array = niwot["wood_carbon"]
        t = pd.DatetimeIndex(array["time"].values)
        windows = pd.IntervalIndex.from_arrays(t[[0, 10]], t[[10, 20]], closed="right")
        with pytest.raises(ValueError, match="strictly increasing"):
            reduce_windows(array, windows, "last", labels=pd.DatetimeIndex([t[10], t[10]]))

    def test_a_window_with_no_step_counts_zero_beside_one_that_has_steps(self, niwot):
        array = niwot["wood_carbon"]
        t = pd.DatetimeIndex(array["time"].values)
        far = pd.Timestamp("2005-01-01")
        windows = pd.IntervalIndex.from_arrays([t[0], far], [t[10], far + pd.Timedelta("1D")], closed="right")
        assert window_counts(array, windows).values.tolist() == [10, 0]

    def test_an_empty_window_is_refused(self, niwot):
        t = pd.Timestamp(niwot["time"].values[3])
        windows = pd.IntervalIndex.from_arrays([t], [t], closed="right")
        with pytest.raises(ValueError, match="holds no instant"):
            reduce_windows(niwot["wood_carbon"], windows, "last")

    def test_windows_given_as_a_list_are_a_type_error(self, niwot):
        with pytest.raises(TypeError, match="pandas.IntervalIndex"):
            reduce_windows(niwot["wood_carbon"], [(0, 1)], "last")

    def test_a_how_that_is_not_a_string_is_a_type_error(self, niwot):
        array = niwot["wood_carbon"]
        with pytest.raises(TypeError, match="how must be a string"):
            reduce_windows(array, _daily_windows(array), None)

    def test_overlapping_steps_are_refused_by_select_timestep_at(self, niwot):
        array = niwot["wood_carbon"]
        start = array[TIMESTEP_START].values.copy()
        start[5] = start[4]  # step 5 now also covers step 4's interval
        overlapping = array.assign_coords({TIMESTEP_START: ("time", start)})
        with pytest.raises(ValueError, match="overlap"):
            select_timestep_at(overlapping, pd.DatetimeIndex(array["time"].values[[5]]))

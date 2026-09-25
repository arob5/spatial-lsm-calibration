"""The window and instant readings in ``sipnet_calibration.observation.alignment``:
``reduce_windows``, ``select_timestep_at``, the window builders and the counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.observation.alignment import (
    LENGTH_COORD,
    SELECTED_STEP_COORD,
    START_COORD,
    TIME_BOUNDS_END,
    TIME_BOUNDS_START,
    WINDOW_REDUCTIONS,
    aggregate_time,
    aggregation_counts,
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
    start = pd.Timestamp(array[START_COORD].values.min()).floor("D")
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
        assert np.isnat(reduced[START_COORD].values[0])

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
        assert full[LENGTH_COORD].values == np.timedelta64(24, "h")
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
        starts = pd.DatetimeIndex(array[START_COORD].values)
        inside = starts[[3, 10]] + pd.Timedelta("1s")
        picked = select_timestep_at(array, inside)
        np.testing.assert_array_equal(picked.values, array.values[[3, 10]])
        np.testing.assert_array_equal(picked["time"].values, inside.values)

    def test_a_label_at_a_step_start_reads_the_previous_step(self, niwot):
        array = niwot["wood_carbon"]
        starts = pd.DatetimeIndex(array[START_COORD].values)
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

    def test_a_running_total_has_no_value_at_an_instant(self, niwot):
        with pytest.raises(ValueError, match="no value at an instant"):
            select_timestep_at(niwot["cumulative_net_ecosystem_exchange"], pd.DatetimeIndex(niwot["time"].values[[3]]))

    def test_a_step_mean_reads_the_containing_step(self, niwot):
        array = niwot["soil_wetness_fraction"]
        starts = pd.DatetimeIndex(array[START_COORD].values)
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
        assert window[0].left == pd.Timestamp(array[START_COORD].values[0])
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

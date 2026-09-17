"""Tests for the observation-space operations.

The ``sipnet_time_index`` cases are built by hand from SIPNET's
``year``/``day``/``time`` labeling, including the drifting ``time`` column the
``.clim`` drivers carry (issue #9), so that a future change cannot start
trusting that column's value.

The ``aggregate_time`` cases exist mostly to pin down two things that fail
silently. The first is sum versus mean: a per-timestep total aggregated with a
mean is wrong by the number of steps in the period, which for 3-hourly to
daily is a factor of eight, and nothing about the resulting figure looks
wrong. The second is an empty period, which ``.resample(...).sum()`` reports
as zero rather than as missing, so a day with no observations reads as zero
flux.

Every expected value is computed with ``numpy`` rather than with the function
under test, so that a test cannot agree with a bug by construction. The
synthetic fields carry the ``aggregation`` attribute pySIPNET would write for
the variable they stand for, from a table in this module; one test checks that
table against a dataset pySIPNET actually builds. The fixtures deliberately do
**not** all store their dimensions in the same order: a reduction that assumes
an order is a mutation that survives an entire suite of same-shaped fixtures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from enum import StrEnum

from sipnet_calibration.observation_operators import (
    AGGREGATION_ATTR,
    REDUCTIONS,
    aggregate_time,
    aggregation_counts,
    reduce_windows,
    sipnet_time_index,
    window_counts,
)
from sipnet_calibration.time_conventions import TIME_LABEL_ATTR, TimeLabel

REAL_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "drivers"
    / "ERA5_1_1"
    / "ERA5.1.2012-01-01.2024-12-31.clim"
)


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

    def test_rejects_a_year_or_day_that_is_not_numeric(self):
        """A boolean would otherwise pass as 0 or 1, and a string that is not
        a number reaches the conversion of last resort. A string that *is* a
        number is accepted, which is what a column read as text gives."""
        with pytest.raises(ValueError, match="year must be numeric"):
            sipnet_time_index(["twenty thirteen"], [1], [0.0])
        with pytest.raises(ValueError, match="must be numeric, got booleans"):
            sipnet_time_index([True], [1], [0.0])
        assert sipnet_time_index(["2013"], [1], [0.0])[0] == pd.Timestamp("2013-01-01")

    def test_rejects_a_non_finite_year_or_day(self):
        """``NaN`` is not a whole number, and floor-comparing it silently
        yields False, so it needs its own guard."""
        with pytest.raises(ValueError, match="year must hold whole numbers"):
            sipnet_time_index([np.nan], [1], [0.0])
        with pytest.raises(ValueError, match="day_of_year must hold whole numbers"):
            sipnet_time_index([2013], [np.inf], [0.0])

    def test_a_timestep_whose_seconds_are_not_exact_is_rounded(self):
        """A step of 4.8 h divides 24 into five whole steps but is not exactly
        representable, so the fourth slot computes as 51839.99999999999
        seconds. Truncating instead of rounding puts it one second early, at
        14:23:59. The steps SIPNET is actually run at are all exact, so this
        is what keeps the function right for any step that divides 24."""
        index = sipnet_time_index([2013], [1], [15.0], timestep_hours=4.8)
        assert index[0] == pd.Timestamp("2013-01-01 14:24:00")

    def test_a_half_hour_timestep_lands_on_exact_seconds(self):
        """0.5 is a step SIPNET is run at, and it is the smallest one whose
        slot offset is not already a whole number of seconds."""
        index = sipnet_time_index(
            [2013] * 4, [1] * 4, [0.0, 0.5, 1.0, 1.5], timestep_hours=0.5
        )
        pd.testing.assert_index_equal(
            index,
            pd.date_range("2013-01-01", periods=4, freq="30min").as_unit("ns"),
            check_names=False,
        )

    @pytest.mark.skipif(not REAL_FILE.exists(), reason="the raw driver file is not present")
    def test_reproduces_the_real_files_grid(self):
        frame = pd.read_csv(REAL_FILE, sep=r"\s+", header=None, usecols=[1, 2, 3], names=["year", "day", "time"])
        index = sipnet_time_index(frame["year"], frame["day"], frame["time"])
        expected = pd.date_range("2012-01-01", "2024-12-31 21:00", freq="3h").as_unit("ns")
        assert len(index) == 37992
        pd.testing.assert_index_equal(index, expected, check_names=False)


# ── aggregate_time ────────────────────────────────────────────────────────────

#: Steps per day in the 3-hourly grid the fixtures and the drivers are on.
STEPS_PER_DAY = 8

#: Values are offset by this much per position along each dimension, so that
#: every cell of a fixture is distinct and no two dimensions can be confused.
_DIMENSION_STRIDE = {"time": 1.0, "site": 100.0, "member": 10000.0}

#: The ``aggregation`` attribute a field of each variable would carry, as
#: pySIPNET writes it from the variable's kind (a step total sums, a step mean
#: or a state averages). A name absent here stands for a field that carries
#: no attribute, which is what an observation looks like.
AGGREGATION_FOR = {
    "par": "sum",
    "precipitation": "sum",
    "nee": "sum",
    "air_temperature": "mean",
    "vpd": "mean",
}


def three_hourly_field(
    name: str,
    *,
    dims: tuple[str, ...] = ("time",),
    n_days: int = 3,
    start: str = "2013-01-01",
) -> xr.DataArray:
    """A field of *name* on a 3-hourly grid, with a known pattern of values.

    Parameters
    ----------
    name:
        The field's name. Its ``aggregation`` attribute, if any, comes from
        :data:`AGGREGATION_FOR`.
    dims:
        The dimensions, in the order they are stored. ``site`` and ``member``
        get two entries each, and ``site`` carries ``lon``/``lat`` as
        non-dimension coordinates. Storing them in the order given, rather
        than in canonical order, is what makes an order-dependent reduction
        visible.
    n_days:
        Whole days of 3-hourly steps.
    start:
        Timestamp of the first row.

    Returns
    -------
    xarray.DataArray
        Values are distinct per cell and are *not* symmetric across a period,
        so a sum, a mean, a first and a last are four different numbers.
    """
    sizes = {"member": 2, "site": 2, "time": STEPS_PER_DAY * n_days}
    unknown = [dim for dim in dims if dim not in sizes]
    if unknown:
        raise ValueError(f"not canonical dims: {unknown}")

    shape = tuple(sizes[dim] for dim in dims)
    values = np.ones(shape)
    for position, dim in enumerate(dims):
        values = values + np.indices(shape)[position] * _DIMENSION_STRIDE[dim]

    coords: dict[str, object] = {}
    if "time" in dims:
        coords["time"] = pd.date_range(start, periods=sizes["time"], freq="3h")
    if "site" in dims:
        coords["site"] = np.array([1, 27], dtype=np.int32)
        coords["lon"] = ("site", np.array([-24.5625, -78.5625]))
        coords["lat"] = ("site", np.array([82.5458, 44.0654]))
    if "member" in dims:
        coords["member"] = np.arange(2, dtype=np.int16)

    attrs = {"units": "arbitrary", "long_name": f"Synthetic {name}"}
    if name in AGGREGATION_FOR:
        attrs[AGGREGATION_ATTR] = AGGREGATION_FOR[name]
    return xr.DataArray(values, dims=dims, coords=coords, name=name, attrs=attrs)


def daily_blocks(field: xr.DataArray) -> np.ndarray:
    """*field*'s values reshaped to ``(n_days, 8)``, for a ``(time,)`` field.

    Expected values are computed from this with ``numpy``, so that no test
    asks the function under test what the answer is.
    """
    values = field.values
    assert values.ndim == 1 and values.size % STEPS_PER_DAY == 0
    return values.reshape(-1, STEPS_PER_DAY)


class TestTheRuleComesFromTheAttribute:
    """The highest-value tests in the module: they protect the likelihood."""

    def test_a_per_timestep_total_sums(self):
        """``par`` is a total over the timestep, so its daily value is the sum
        of the day's eight rows. Compared against ``numpy``'s own sum over the
        same rows, never against ``.resample``."""
        par = three_hourly_field("par")
        daily = aggregate_time(par, "1D")
        np.testing.assert_allclose(daily.values, daily_blocks(par).sum(axis=1))

    def test_an_intensive_variable_means(self):
        """``air_temperature`` describes the timestep rather than accumulating
        over it, so its daily value is the mean."""
        tair = three_hourly_field("air_temperature")
        daily = aggregate_time(tair, "1D")
        np.testing.assert_allclose(daily.values, daily_blocks(tair).mean(axis=1))

    def test_the_sum_and_the_mean_differ_by_the_number_of_steps(self):
        """The same values under both rules, so the eight-fold error is
        asserted as a ratio and not only as two separate numbers. This is the
        test that fails if the two rules are ever swapped."""
        values = three_hourly_field("par").values
        summed = aggregate_time(three_hourly_field("par"), "1D")
        meaned = aggregate_time(three_hourly_field("air_temperature"), "1D")
        np.testing.assert_array_equal(
            three_hourly_field("air_temperature").values, values
        )
        np.testing.assert_allclose(summed.values, meaned.values * STEPS_PER_DAY)
        assert not np.allclose(summed.values, meaned.values)

    def test_nee_sums(self):
        """SIPNET's NEE is a per-timestep total, which is the case the rule
        exists for; a mean here is the error worth the most."""
        nee = three_hourly_field("nee")
        daily = aggregate_time(nee, "1D")
        np.testing.assert_allclose(daily.values, daily_blocks(nee).sum(axis=1))

    def test_precipitation_sums_and_vpd_means(self):
        """A second variable under each rule, so the first two tests cannot
        pass by a coincidence of one entry."""
        precipitation = three_hourly_field("precipitation")
        vpd = three_hourly_field("vpd")
        np.testing.assert_allclose(
            aggregate_time(precipitation, "1D").values,
            daily_blocks(precipitation).sum(axis=1),
        )
        np.testing.assert_allclose(
            aggregate_time(vpd, "1D").values, daily_blocks(vpd).mean(axis=1)
        )

    def test_the_attribute_is_the_rule_whatever_the_name_says(self):
        """A field named ``par`` but carrying ``aggregation="mean"`` is
        meaned. The name is not consulted: there is no table of names here,
        and a field's own description of itself is what it says it is."""
        par = three_hourly_field("par")
        par.attrs[AGGREGATION_ATTR] = "mean"
        daily = aggregate_time(par, "1D")
        np.testing.assert_allclose(daily.values, daily_blocks(par).mean(axis=1))
        assert daily.attrs["aggregation_applied"] == "mean"

    def test_a_field_without_the_attribute_raises(self):
        """Rather than falling back to a default rule, which is how the
        eight-fold error would re-enter. The message says to pass ``how``."""
        field = three_hourly_field("lai")
        assert AGGREGATION_ATTR not in field.attrs
        with pytest.raises(ValueError, match=r"carries no 'aggregation' attribute.*pass how="):
            aggregate_time(field, "1D")

    def test_arithmetic_between_disagreeing_fields_drops_the_attribute(self):
        """xarray keeps an attribute through arithmetic only when both
        operands agree on it. A total minus a total is still a total and keeps
        ``"sum"``; a total minus a step mean has no rule, and the result is
        refused. The message names the array as unnamed rather than reporting
        ``None``."""
        same_rule = three_hourly_field("nee") - three_hourly_field("par")
        assert same_rule.attrs[AGGREGATION_ATTR] == "sum"

        mixed = three_hourly_field("nee") - three_hourly_field("air_temperature")
        assert mixed.name is None and AGGREGATION_ATTR not in mixed.attrs
        with pytest.raises(ValueError, match="this unnamed array carries no"):
            aggregate_time(mixed, "1D")

    def test_the_value_pysipnet_writes_on_a_time_coordinate_raises(self):
        """``aggregation="none"`` is what pySIPNET writes on ``year``,
        ``day_of_year`` and ``hour_of_day``; it names no reduction."""
        year = three_hourly_field("year")
        year.attrs[AGGREGATION_ATTR] = "none"
        with pytest.raises(ValueError, match="cannot be aggregated"):
            aggregate_time(year, "1D")

    def test_an_attribute_naming_no_known_reduction_raises(self):
        field = three_hourly_field("par")
        field.attrs[AGGREGATION_ATTR] = "total"
        with pytest.raises(ValueError, match=r"aggregation='total', which is not one of"):
            aggregate_time(field, "1D")


class TestTheAttributeAsPySIPNETWritesIt:
    """The seam with pySIPNET: the attribute this module reads is the one its
    ``xarray_attributes()`` writes, with the values this module accepts."""

    @staticmethod
    def sipnet_dataset(n_days: int = 3) -> xr.Dataset:
        """A dataset built by pySIPNET's own converter from a synthetic frame
        with the columns SIPNET writes, so its attributes are pySIPNET's."""
        from pysipnet.output import output_dataframe_to_dataset

        n = STEPS_PER_DAY * n_days
        frame = pd.DataFrame(
            {
                "year": 2013,
                "day_of_year": np.repeat(np.arange(1, n_days + 1), STEPS_PER_DAY),
                "hour_of_day": np.tile(np.arange(0, 24, 3.0), n_days),
                "net_ecosystem_exchange": np.arange(1.0, n + 1),
                "soil_carbon": 1000.0 + np.arange(n),
                "soil_wetness_fraction": np.linspace(0.2, 0.8, n),
            }
        )
        return output_dataframe_to_dataset(frame)

    def test_a_flux_carries_sum_and_a_state_carries_mean(self):
        dataset = self.sipnet_dataset()
        assert dataset["net_ecosystem_exchange"].attrs[AGGREGATION_ATTR] == "sum"
        assert dataset["soil_carbon"].attrs[AGGREGATION_ATTR] == "mean"
        assert dataset["soil_wetness_fraction"].attrs[AGGREGATION_ATTR] == "mean"

    def test_pysipnet_nee_sums_to_daily_without_being_told(self):
        """The whole point of reading the attribute: a model field aggregates
        at the rule its kind implies, with no table in this repository."""
        nee = self.sipnet_dataset()["net_ecosystem_exchange"]
        daily = aggregate_time(nee, "1D")
        expected = nee.values.reshape(-1, STEPS_PER_DAY).sum(axis=1)
        np.testing.assert_allclose(daily.values, expected)
        assert daily.attrs["units"] == nee.attrs["units"] == "g m-2"
        assert daily.attrs["constituent"] == "C"

    def test_a_pysipnet_state_averages_to_daily(self):
        soil = self.sipnet_dataset()["soil_carbon"]
        daily = aggregate_time(soil, "1D")
        expected = soil.values.reshape(-1, STEPS_PER_DAY).mean(axis=1)
        np.testing.assert_allclose(daily.values, expected)

    def test_the_table_of_synthetic_rules_matches_pysipnet(self):
        """``AGGREGATION_FOR`` stands in for pySIPNET in the other tests; it
        must say what pySIPNET says for the variables both know."""
        from pysipnet.variables import resolve_climate_variable, resolve_output_variable

        assert AGGREGATION_FOR["nee"] == resolve_output_variable("nee").aggregation.value
        pairs = (
            ("par", "par"),
            ("precipitation", "precip"),
            ("air_temperature", "tair"),
            ("vpd", "vpd"),
        )
        for ours, theirs in pairs:
            assert AGGREGATION_FOR[ours] == resolve_climate_variable(theirs).aggregation.value

    def test_the_accepted_values_cover_every_rule_pysipnet_can_write(self):
        """Every pySIPNET aggregation but ``none`` is a reduction here, and
        ``none`` is refused by name rather than by falling through."""
        from pysipnet.variables import Aggregation

        for rule in Aggregation:
            if rule is Aggregation.NONE:
                continue
            assert rule.value in REDUCTIONS


class TestEmptyPeriods:
    def test_a_period_with_no_observations_is_missing_not_zero(self):
        """A whole day of ``NaN`` under the sum rule. ``.resample().sum()``
        returns 0 there, which reads as zero flux rather than as unobserved,
        and NEE is about 55% missing over site and time."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        daily = aggregate_time(par, "1D")
        assert np.isnan(daily.values[1])
        assert daily.values[1] != 0.0

    @pytest.mark.parametrize("how", ["sum", "mean", "last", "first"])
    def test_a_period_with_no_observations_is_missing_under_every_rule(self, how):
        """So the guard cannot be attached to the sum alone."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        assert np.isnan(aggregate_time(par, "1D", how=how).values[1])

    def test_a_period_that_is_only_partly_missing_uses_what_is_there(self):
        """The complement: the finite values are aggregated, so the guard does
        not throw away a period that has data."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY : STEPS_PER_DAY + 3] = np.nan
        daily = aggregate_time(par, "1D")
        expected = np.nansum(daily_blocks(par)[1])
        assert daily.values[1] == pytest.approx(expected)
        assert np.isfinite(daily.values[1])

    def test_the_missing_periods_are_exactly_where_the_data_is_missing(self):
        """Asserted as a pattern over several periods, not on one, so a guard
        that masked everything or nothing would fail."""
        par = three_hourly_field("par", n_days=4)
        par[STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        par[3 * STEPS_PER_DAY :] = np.nan
        daily = aggregate_time(par, "1D")
        np.testing.assert_array_equal(
            np.isnan(daily.values), [False, True, False, True]
        )


class TestEmptyPeriodsWithEveryDimensionPresent:
    """The guard on the shape it exists for.

    Every case in :class:`TestEmptyPeriods` is one-dimensional, and the
    variable the guard was written for is ``(member, site, time)`` NEE that is
    about 55% missing. A guard that masked only a one-dimensional field, or
    that broadcast wrongly against a ``member`` dimension, would pass all of
    those and fail here.
    """

    def test_only_the_missing_member_site_pair_goes_missing(self):
        """The day is blanked for one member at one site. Every other cell of
        that day must still be finite, and the blanked one must be NaN --
        checked cell by cell, so a guard that masked the whole day, or none of
        it, or the wrong pair, fails."""
        field = three_hourly_field("par", dims=("member", "site", "time"), n_days=3)
        values = field.values.copy()
        values[1, 1, STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        field = field.copy(data=values)

        daily = aggregate_time(field, "1D")
        expected_missing = np.zeros((2, 2, 3), dtype=bool)
        expected_missing[1, 1, 1] = True
        np.testing.assert_array_equal(np.isnan(daily.values), expected_missing)

        finite = values.reshape(2, 2, 3, STEPS_PER_DAY).sum(axis=-1)
        np.testing.assert_allclose(
            daily.values[~expected_missing], finite[~expected_missing]
        )

    def test_a_partial_day_at_one_pair_is_dropped_only_there(self):
        """``min_count`` with more than one dimension present: only the pair
        whose day is short goes missing."""
        field = three_hourly_field("par", dims=("member", "site", "time"), n_days=3)
        values = field.values.copy()
        values[0, 1, STEPS_PER_DAY + 3 : 2 * STEPS_PER_DAY] = np.nan
        field = field.copy(data=values)

        daily = aggregate_time(field, "1D", min_count=STEPS_PER_DAY)
        expected_missing = np.zeros((2, 2, 3), dtype=bool)
        expected_missing[0, 1, 1] = True
        np.testing.assert_array_equal(np.isnan(daily.values), expected_missing)

    def test_the_same_holds_when_time_is_not_the_last_dimension(self):
        """A ``(time, site)`` field, so a guard that indexed positionally
        rather than by dimension name masks the wrong cells."""
        field = three_hourly_field("par", dims=("time", "site"), n_days=3)
        values = field.values.copy()
        values[STEPS_PER_DAY : 2 * STEPS_PER_DAY, 0] = np.nan
        field = field.copy(data=values)

        daily = aggregate_time(field, "1D")
        expected_missing = np.zeros((3, 2), dtype=bool)
        expected_missing[1, 0] = True
        np.testing.assert_array_equal(np.isnan(daily.values), expected_missing)

    def test_counts_are_per_cell_not_per_period(self):
        """``aggregation_counts`` has to count within each member and site,
        not collapse them, or the completeness rule it exists to support is
        wrong wherever coverage differs across the pool."""
        field = three_hourly_field("par", dims=("member", "site", "time"), n_days=3)
        values = field.values.copy()
        values[1, 1, STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        values[0, 0, STEPS_PER_DAY : STEPS_PER_DAY + 3] = np.nan
        field = field.copy(data=values)

        counts = aggregation_counts(field, "1D")
        expected = np.isfinite(values.reshape(2, 2, 3, STEPS_PER_DAY)).sum(axis=-1)
        np.testing.assert_array_equal(counts.values, expected)
        assert counts.dims == ("member", "site", "time")


class TestFrequenciesOtherThanDaily:
    """Nothing above leaves the daily period, so a guard or a grouping that is
    right only at ``"1D"`` -- counts taken at a hard-coded frequency, say --
    would pass the rest of the suite."""

    @staticmethod
    def three_months() -> xr.DataArray:
        """A 3-hourly ``par`` field spanning January to March."""
        return three_hourly_field("par", n_days=90, start="2013-01-01")

    def test_monthly_totals_match_an_independent_groupby(self):
        """Compared against a pandas groupby on the calendar month, which
        shares no code with the resample under test."""
        par = self.three_months()
        frame = pd.Series(par.values, index=pd.DatetimeIndex(par["time"].values))
        expected = frame.groupby(frame.index.month).sum().to_numpy()

        monthly = aggregate_time(par, "MS")
        np.testing.assert_allclose(monthly.values, expected)
        assert monthly.sizes["time"] == 3

    def test_the_monthly_periods_are_the_month_starts(self):
        """A frequency silently doubled to ``"2MS"`` conserves the total and
        would pass a sum-conservation check; the labels are what catch it."""
        monthly = aggregate_time(self.three_months(), "MS")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(monthly["time"].values),
            pd.date_range("2013-01-01", periods=3, freq="MS"),
            check_names=False,
        )

    def test_monthly_counts_are_taken_over_the_month(self):
        """``aggregation_counts`` at a frequency other than daily, against a
        numpy count, with a month made deliberately short so the three counts
        differ and a hard-coded period cannot pass."""
        par = self.three_months()
        values = par.values.copy()
        values[STEPS_PER_DAY * 31 : STEPS_PER_DAY * 33] = np.nan  # two days in February
        values[STEPS_PER_DAY * 60 : STEPS_PER_DAY * 61] = np.nan  # one day in March
        par = par.copy(data=values)

        counts = aggregation_counts(par, "MS")
        stamps = pd.DatetimeIndex(par["time"].values)
        expected = [
            int(np.isfinite(values[stamps.month == month]).sum()) for month in (1, 2, 3)
        ]
        np.testing.assert_array_equal(counts.values, expected)
        assert len(set(expected)) == 3

    def test_the_completeness_guard_is_taken_at_the_target_period(self):
        """A whole month missing must come back as one missing month. A guard
        that took its counts at a hard-coded ``"1D"`` would align them to the
        wrong axis and drop a month that was fully observed."""
        par = self.three_months()
        values = par.values.copy()
        values[: STEPS_PER_DAY * 31] = np.nan  # all of January
        par = par.copy(data=values)

        monthly = aggregate_time(par, "MS")
        np.testing.assert_array_equal(np.isnan(monthly.values), [True, False, False])
        stamps = pd.DatetimeIndex(par["time"].values)
        np.testing.assert_allclose(
            monthly.values[1], np.nansum(values[stamps.month == 2])
        )


class TestPartialPeriods:
    def test_a_partial_period_is_returned_not_scaled(self):
        """Three of a day's eight rows sum to those three values. Scaling
        assumes the absent rows resemble the present ones, which for a diurnal
        flux is false in the worst direction."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY + 3 :] = np.nan
        daily = aggregate_time(par, "1D")
        expected = np.nansum(daily_blocks(par)[1])
        assert daily.values[1] == pytest.approx(expected)
        # The whole-day total of the same three rows scaled up, which is what
        # a scaling implementation would return instead.
        assert not daily.values[1] == pytest.approx(expected * STEPS_PER_DAY / 3)

    def test_min_count_drops_a_period_with_too_few_values(self):
        """``min_count=8`` keeps only whole days, which is how a caller states
        a completeness rule."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY + 3 :] = np.nan
        daily = aggregate_time(par, "1D", min_count=STEPS_PER_DAY)
        np.testing.assert_array_equal(np.isnan(daily.values), [False, True, True])

    def test_min_count_counts_finite_values_not_rows(self):
        """A period holding eight rows of which three are finite fails
        ``min_count=8``, so the rule cannot be satisfied by missing data."""
        par = three_hourly_field("par")
        par[3:STEPS_PER_DAY] = np.nan
        assert aggregation_counts(par, "1D").values[0] == 3
        assert np.isnan(aggregate_time(par, "1D", min_count=4).values[0])
        assert np.isfinite(aggregate_time(par, "1D", min_count=3).values[0])

    def test_min_count_applies_to_the_mean_rule_too(self):
        """Not only to the sum: ``.mean()`` accepts no ``min_count`` of its
        own, so this is where a sum-only implementation shows."""
        tair = three_hourly_field("air_temperature")
        tair[3:STEPS_PER_DAY] = np.nan
        assert np.isnan(aggregate_time(tair, "1D", min_count=4).values[0])
        assert np.isfinite(aggregate_time(tair, "1D", min_count=3).values[0])

    @pytest.mark.parametrize("how", ["last", "first"])
    def test_min_count_applies_to_last_and_first_too(self, how):
        """The remaining two rules, for the same reason."""
        par = three_hourly_field("par")
        par[3:STEPS_PER_DAY] = np.nan
        assert np.isnan(aggregate_time(par, "1D", how=how, min_count=4).values[0])

    def test_min_count_below_one_raises(self):
        """Zero would mean a period formed from nothing still produces a
        value, which for a sum is the zero the default exists to prevent."""
        par = three_hourly_field("par")
        with pytest.raises(ValueError, match="min_count must be at least 1"):
            aggregate_time(par, "1D", min_count=0)
        with pytest.raises(ValueError, match="min_count must be at least 1"):
            aggregate_time(par, "1D", min_count=-3)

    @pytest.mark.parametrize("min_count", [1.5, "8", True, False, None])
    def test_min_count_must_be_an_integer(self, min_count):
        """``min_count=1.5`` would compare as a float and quietly work, and
        ``True`` is an ``int`` in Python, so without the boolean guard it
        would silently pass as 1."""
        with pytest.raises(ValueError, match="min_count must be an integer"):
            aggregate_time(three_hourly_field("par"), "1D", min_count=min_count)


class TestExplicitHow:
    def test_how_overrides_the_attribute(self):
        """``how="mean"`` on ``par``, whose attribute says sum."""
        par = three_hourly_field("par")
        daily = aggregate_time(par, "1D", how="mean")
        np.testing.assert_allclose(daily.values, daily_blocks(par).mean(axis=1))

    def test_how_admits_a_field_without_the_attribute(self):
        """An observation, or an observation error variance, carries no
        ``aggregation`` attribute; with ``how`` given none is looked for."""
        variance = three_hourly_field("total_soil_carbon_variance")
        daily = aggregate_time(variance, "1D", how="mean")
        np.testing.assert_allclose(daily.values, daily_blocks(variance).mean(axis=1))

    def test_last_and_first_take_the_period_s_end_and_start(self):
        """Distinguishable only because the fixture is not symmetric within a
        period."""
        par = three_hourly_field("par")
        blocks = daily_blocks(par)
        assert not np.allclose(blocks[:, 0], blocks[:, -1])
        np.testing.assert_allclose(
            aggregate_time(par, "1D", how="last").values, blocks[:, -1]
        )
        np.testing.assert_allclose(
            aggregate_time(par, "1D", how="first").values, blocks[:, 0]
        )

    def test_last_skips_a_missing_final_value(self):
        """The last *observed* value of the period, not the last row, which
        would be ``NaN`` whenever a record ends mid-period."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY - 2 :] = np.nan
        expected = daily_blocks(par)[0, STEPS_PER_DAY - 3]
        last = aggregate_time(par, "1D", how="last")
        assert last.values[0] == pytest.approx(expected)

    def test_first_skips_a_missing_opening_value(self):
        """The counterpart to ``last``: the first *observed* value of the
        period, not the first row, which is ``NaN`` wherever a record starts
        part-way through one."""
        par = three_hourly_field("par")
        par[:2] = np.nan
        expected = daily_blocks(par)[0, 2]
        first = aggregate_time(par, "1D", how="first")
        assert first.values[0] == pytest.approx(expected)

    def test_min_and_max_take_the_period_s_extremes(self):
        par = three_hourly_field("par")
        blocks = daily_blocks(par)
        np.testing.assert_allclose(aggregate_time(par, "1D", how="max").values, blocks.max(axis=1))
        np.testing.assert_allclose(aggregate_time(par, "1D", how="min").values, blocks.min(axis=1))

    def test_an_unknown_how_raises_and_lists_the_methods(self):
        """A typo such as ``how="total"`` would otherwise reach xarray."""
        with pytest.raises(ValueError, match=r"how must be one of \['sum'"):
            aggregate_time(three_hourly_field("par"), "1D", how="total")

    def test_every_reduction_is_accepted_as_how(self):
        for how in REDUCTIONS:
            assert aggregate_time(three_hourly_field("lai"), "1D", how=how).sizes["time"] == 3


def annual_field(name: str, years, *, month_day: str = "07-15") -> xr.DataArray:
    """An annual series, the shape the constraint product has."""
    return xr.DataArray(
        np.arange(float(len(years))),
        dims="time",
        coords={"time": pd.to_datetime([f"{y}-{month_day}" for y in years])},
        name=name,
        attrs={"units": "arbitrary", "long_name": f"Synthetic {name}"},
    )


class TestUpsamplingIsRefused:
    def test_a_finer_target_than_the_source_raises(self):
        """``"1h"`` on a 3-hourly field returns a field that is two-thirds
        ``NaN`` with no error, and the emptiness reads as missing data."""
        with pytest.raises(ValueError, match="would interpolate rather than"):
            aggregate_time(three_hourly_field("par"), "1h")

    def test_a_target_only_slightly_finer_raises(self):
        """``"2h"`` on a 3-hourly field is upsampling by a ratio of 1.5, so a
        check with slack in it would let this through."""
        with pytest.raises(ValueError, match="would interpolate rather than"):
            aggregate_time(three_hourly_field("par"), "2h")

    def test_an_annual_field_refuses_a_monthly_target(self):
        """The case that arises for real: the annual constraints in a report
        that aggregates every variable to one frequency."""
        annual = annual_field("lai", (2012, 2013, 2014, 2015))
        with pytest.raises(ValueError, match="would interpolate rather than"):
            aggregate_time(annual, "MS", how="last")

    @pytest.mark.parametrize("freq", ["YS", "YE"])
    @pytest.mark.parametrize("years", [(2011, 2012, 2013), (2015, 2016)])
    def test_an_annual_field_accepts_an_annual_target_across_a_leap_year(
        self, freq, years
    ):
        """The regression that a duration comparison gets wrong. A year is 365
        days or 366, so an annual series spanning a leap year has a spacing
        longer than a measured non-leap year, and a check that compares the
        two refuses a downsample that is plainly one. The constraint product
        is exactly this series, and ``how="last"`` is what its own docstring
        tells a caller to pass."""
        annual = annual_field("lai", years)
        result = aggregate_time(annual, freq, how="last")
        assert result.sizes["time"] == len(years)
        np.testing.assert_allclose(sorted(result.values), sorted(annual.values))

    @pytest.mark.parametrize(
        ("freq", "source_freq"), [("QS", "QS"), ("W", "7D"), ("ME", "ME")]
    )
    def test_a_source_at_its_own_frequency_is_accepted(self, freq, source_freq):
        """Quarters, weeks and month-ends all have lengths that vary, and a
        duration comparison measured from one probe date refuses every one of
        them."""
        times = pd.date_range("2020-01-05", periods=8, freq=source_freq)
        field = xr.DataArray(
            np.arange(8.0), dims="time", coords={"time": times}, name="par",
            attrs={AGGREGATION_ATTR: "sum"},
        )
        assert aggregate_time(field, freq).sizes["time"] == 8

    def test_a_target_equal_to_the_source_spacing_is_a_no_op_with_provenance(self):
        """Aggregating a daily field to ``"1D"`` is a no-op, not an error --
        but it is still an aggregation, so it records what it did. A
        short-circuit that returned the input unchanged would pass on the
        values alone."""
        values = np.arange(1.0, 4.0)
        daily = xr.DataArray(
            values,
            dims="time",
            coords={"time": pd.date_range("2013-01-01", periods=3, freq="1D")},
            name="par",
            attrs={"units": "arbitrary", "long_name": "Synthetic par", AGGREGATION_ATTR: "sum"},
        )
        result = aggregate_time(daily, "1D")
        np.testing.assert_allclose(result.values, values)
        assert result.name == "par"
        assert result.attrs["aggregation_applied"] == "sum"
        assert result.attrs["aggregation_freq"] == "1D"

    def test_a_single_timestamp_is_not_upsampling(self):
        """One row produces one period at any frequency, so there is nothing
        to refuse. A one-year slice of the annual constraints is this case."""
        one = annual_field("lai", (2015,))
        assert aggregate_time(one, "YS", how="last").sizes["time"] == 1

    def test_a_gapped_field_at_its_own_cadence_is_a_no_op(self):
        """A daily record missing a day, aggregated to ``"1D"``: nothing is
        grouped and one empty period appears, and that is not upsampling."""
        times = pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-04"])
        field = xr.DataArray(np.arange(3.0), dims="time", coords={"time": times}, name="x")
        result = aggregate_time(field, "1D", how="sum")
        np.testing.assert_allclose(result.values, [0.0, 1.0, np.nan, 2.0])
        np.testing.assert_array_equal(aggregation_counts(field, "1D").values, [1, 1, 0, 1])

    def test_a_sparse_annual_field_aggregates_to_years(self):
        """The ragged constraint product, one site with a missing year."""
        sparse = annual_field("lai", (2005, 2007, 2008))
        result = aggregate_time(sparse, "YS", how="last")
        assert result.sizes["time"] == 4
        np.testing.assert_allclose(result.values, [0.0, np.nan, 1.0, 2.0])

    def test_a_three_hourly_record_missing_one_row_is_not_upsampled_at_its_own_step(self):
        par = three_hourly_field("par", n_days=1)
        par = par.isel(time=[0, 1, 2, 4, 5, 6, 7])
        assert aggregate_time(par, "3h").sizes["time"] == STEPS_PER_DAY

    def test_a_long_gap_does_not_make_a_downsample_look_like_one(self):
        """A record with a month-long hole is still 3-hourly, and a daily
        aggregation of it is a downsample. A check that took the widest gap as
        the source's spacing would refuse this."""
        times = pd.DatetimeIndex(
            list(pd.date_range("2013-01-01", periods=8, freq="3h"))
            + list(pd.date_range("2013-02-01", periods=8, freq="3h"))
        )
        field = xr.DataArray(
            np.arange(16.0), dims="time", coords={"time": times}, name="par",
            attrs={AGGREGATION_ATTR: "sum"},
        )
        assert aggregate_time(field, "1D").sizes["time"] == 32

    def test_a_coarser_target_of_no_fixed_length_is_allowed(self):
        """``"MS"`` and ``"YS"`` have no fixed length, so the check cannot
        rest on converting them to a duration."""
        par = three_hourly_field("par", n_days=40)
        for freq in ("MS", "YS"):
            result = aggregate_time(par, freq)
            assert result.values.sum() == pytest.approx(par.values.sum())

    @pytest.mark.parametrize("freq", ["every other Tuesday", None, "0D", "-1D"])
    def test_a_frequency_that_is_not_a_positive_offset_alias_raises(self, freq):
        """``to_offset(None)`` returns ``None`` rather than raising, so
        ``freq=None`` reaches the resample as a ``TypeError`` unless it is
        caught here."""
        with pytest.raises(ValueError, match="freq"):
            aggregate_time(three_hourly_field("par"), freq)


class TestWhatSurvivesAggregation:
    def test_the_name_and_the_attributes_are_carried_through(self):
        """``units`` and ``long_name`` above all: the plotting layer builds
        its axis label from them, and ``keep_attrs`` defaults have changed
        between xarray versions."""
        par = three_hourly_field("par")
        daily = aggregate_time(par, "1D")
        assert daily.name == "par"
        assert daily.attrs["units"] == par.attrs["units"]
        assert daily.attrs["long_name"] == par.attrs["long_name"]

    def test_the_aggregation_is_recorded_in_the_attributes(self):
        """A canonical per-timestep total carries the same ``units`` at every
        resolution, so without these two a 3-hourly and a daily NEE field are
        indistinguishable while differing by a factor of eight."""
        nee = three_hourly_field("nee", n_days=40)
        daily = aggregate_time(nee, "1D")
        monthly = aggregate_time(nee, "MS")
        assert daily.attrs["aggregation_applied"] == "sum"
        assert daily.attrs["aggregation_freq"] == "1D"
        assert monthly.attrs["aggregation_freq"] == "MS"
        assert daily.attrs["units"] == monthly.attrs["units"]

    def test_the_recorded_method_is_the_one_used_not_the_one_asked_for(self):
        """With ``how`` given it is ``how``; without, it is the attribute's
        rule rather than the string ``None``."""
        par = three_hourly_field("par")
        assert aggregate_time(par, "1D").attrs["aggregation_applied"] == "sum"
        assert (
            aggregate_time(par, "1D", how="mean").attrs["aggregation_applied"] == "mean"
        )

    def test_aggregating_does_not_mutate_the_input(self):
        """The attributes are copied rather than added to in place, so the
        3-hourly field a caller keeps does not gain the daily field's
        provenance."""
        par = three_hourly_field("par")
        before = dict(par.attrs)
        aggregate_time(par, "1D")
        assert par.attrs == before

    def test_the_other_dimensions_and_their_coordinates_are_untouched(self):
        """``member``, ``site``, and ``lon``/``lat`` as non-dimension
        coordinates on ``site``: the canonical field convention has to survive
        an aggregation, or the result cannot be plotted or mapped."""
        field = three_hourly_field("par", dims=("member", "site", "time"))
        daily = aggregate_time(field, "1D")
        assert daily.dims == ("member", "site", "time")
        assert daily.sizes["member"] == field.sizes["member"]
        np.testing.assert_array_equal(daily["site"].values, field["site"].values)
        np.testing.assert_allclose(daily["lon"].values, field["lon"].values)
        np.testing.assert_allclose(daily["lat"].values, field["lat"].values)
        assert daily["lon"].dims == ("site",)

    def test_the_values_are_right_with_every_dimension_present(self):
        """The reduction over the right axis, checked cell by cell against
        ``numpy``, so a correct shape with scrambled contents fails."""
        field = three_hourly_field("par", dims=("member", "site", "time"), n_days=3)
        daily = aggregate_time(field, "1D")
        expected = field.values.reshape(2, 2, -1, STEPS_PER_DAY).sum(axis=-1)
        np.testing.assert_allclose(daily.values, expected)

    def test_the_result_is_independent_of_the_stored_dimension_order(self):
        """A ``(time, site)`` field and a ``(site, time)`` field holding the
        same values aggregate to the same numbers. An implementation that
        reshapes or transposes without naming dimensions passes every
        same-shaped fixture and scrambles this one."""
        canonical = three_hourly_field("par", dims=("site", "time"))
        transposed = canonical.transpose("time", "site")
        assert transposed.dims == ("time", "site")

        from_canonical = aggregate_time(canonical, "1D")
        from_transposed = aggregate_time(transposed, "1D")
        xr.testing.assert_allclose(
            from_canonical, from_transposed.transpose("site", "time")
        )
        expected = canonical.values.reshape(2, -1, STEPS_PER_DAY).sum(axis=-1)
        np.testing.assert_allclose(from_canonical.values, expected)

    def test_aggregating_over_a_single_period_reproduces_the_reduction(self):
        """The degenerate case, where the answer is one number computed by
        ``numpy`` over the whole array."""
        par = three_hourly_field("par", n_days=3)
        whole = aggregate_time(par, "YS")
        assert whole.sizes["time"] == 1
        assert whole.values[0] == pytest.approx(par.values.sum())


class TestTheTimeAxis:
    def test_the_aggregated_time_coordinate_says_what_its_labels_mark(self):
        """The source's label attributes describe the source's steps and do
        not survive; the clock attributes do."""
        par = three_hourly_field("par", n_days=40)
        par["time"].attrs = {
            TIME_LABEL_ATTR: TimeLabel.INTERVAL_END.value,
            "time_label_note": "covers (h - 3, h]",
            "time_zone": "UTC",
            "clock_status": "inferred",
        }
        daily = aggregate_time(par, "1D")
        assert daily["time"].attrs[TIME_LABEL_ATTR] == TimeLabel.INTERVAL_START.value
        assert "(h - 3, h]" not in daily["time"].attrs["time_label_note"]
        assert daily["time"].attrs["time_zone"] == "UTC"
        assert daily["time"].attrs["clock_status"] == "inferred"
        monthly = aggregate_time(par, "ME")
        assert monthly["time"].attrs[TIME_LABEL_ATTR] == TimeLabel.INTERVAL_END.value
        assert aggregate_time(par, "YS")["time"].attrs[TIME_LABEL_ATTR] == "interval_start"

    def test_freq_must_be_a_string(self):
        """A ``Timedelta`` is an offset pandas accepts, but it cannot be
        written as an attribute."""
        with pytest.raises(ValueError, match="freq must be a pandas offset alias"):
            aggregate_time(three_hourly_field("par"), pd.Timedelta("1D"))

    def test_a_field_with_no_time_dimension_raises(self):
        """An initial-condition field is ``(member, site)``."""
        static = three_hourly_field("par", dims=("member", "site"))
        with pytest.raises(ValueError, match="needs 'time'"):
            aggregate_time(static, "1D")

    def test_a_field_with_a_time_dimension_but_no_coordinate_raises(self):
        """``resample`` needs the coordinate; without this check the failure
        comes out of pandas naming neither the array nor the dimension."""
        bare = xr.DataArray(np.arange(8.0), dims="time", name="par")
        with pytest.raises(ValueError, match="no 'time' coordinate"):
            aggregate_time(bare, "1D")

    def test_an_integer_time_coordinate_raises(self):
        """Not every ``time`` axis is datetime; SIPNET's own output arrives as
        year, day and hour columns until ``sipnet_time_index`` has run."""
        counted = xr.DataArray(
            np.arange(8.0), dims="time", coords={"time": np.arange(8)}, name="par"
        )
        with pytest.raises(ValueError, match="needs datetimes"):
            aggregate_time(counted, "1D")

    def test_a_non_increasing_time_coordinate_raises(self):
        """Two sources concatenated in the wrong order group into periods that
        overlap, and the result is wrong rather than empty."""
        par = three_hourly_field("par")
        shuffled = par.copy()
        stamps = par["time"].values.copy()
        stamps[[3, 4]] = stamps[[4, 3]]
        shuffled = shuffled.assign_coords(time=stamps)
        with pytest.raises(ValueError, match="not strictly increasing"):
            aggregate_time(shuffled, "1D")

    def test_a_repeated_timestamp_raises(self):
        """Two rows for one instant is the concatenation of overlapping
        records, which is not an aggregation problem to solve here."""
        par = three_hourly_field("par")
        stamps = par["time"].values.copy()
        stamps[4] = stamps[3]
        with pytest.raises(ValueError, match="not strictly increasing"):
            aggregate_time(par.assign_coords(time=stamps), "1D")

    def test_a_timezone_aware_time_coordinate_is_accepted(self):
        """xarray groups a tz-aware axis correctly, and ``drivers`` records
        ``time_zone = "UTC"`` on the coordinate, so a caller localizing the
        index is doing the obvious thing. The dtype check has to recognize it
        rather than raise on the pandas extension dtype."""
        par = three_hourly_field("par")
        aware = par.assign_coords(
            time=pd.DatetimeIndex(par["time"].values).tz_localize("UTC")
        )
        daily = aggregate_time(aware, "1D")
        np.testing.assert_allclose(daily.values, daily_blocks(par).sum(axis=1))

    def test_an_empty_time_axis_raises_with_a_message_of_our_own(self):
        """A selection that matched no timestamps. Without the check, xarray
        reports ``__resample_dim__ must not be empty``, naming an internal of
        its own rather than the array or the argument."""
        empty = xr.DataArray(
            np.ones(0), dims="time", coords={"time": pd.DatetimeIndex([])}, name="par"
        )
        with pytest.raises(ValueError, match="axis is empty"):
            aggregate_time(empty, "1D")
        with pytest.raises(ValueError, match="axis is empty"):
            aggregation_counts(empty, "1D")

    def test_a_missing_timestamp_raises(self):
        """``NaT`` compares False against everything, so it slips past a
        monotonicity check and surfaces much later as a pandas message about
        a non-monotonic index."""
        par = three_hourly_field("par")
        stamps = par["time"].values.copy().astype("datetime64[ns]")
        stamps[4] = np.datetime64("NaT")
        with pytest.raises(ValueError, match="missing timestamp"):
            aggregate_time(par.assign_coords(time=stamps), "1D")

    def test_the_arguments_are_checked_before_the_array(self):
        """A mistake in the call is reported as itself. With the array
        checked first, passing a bad ``min_count`` alongside something that is
        not a field reports the field, and the caller fixes the wrong thing."""
        with pytest.raises(ValueError, match="min_count must be at least 1"):
            aggregate_time("not a field", "1D", min_count=0)

    def test_something_that_is_not_a_dataarray_raises(self):
        """A ``Dataset``, which is what an adapter that has not run yet would
        hand over."""
        par = three_hourly_field("par")
        with pytest.raises(ValueError, match="expected an xarray.DataArray"):
            aggregate_time(par.to_dataset(), "1D")

    def test_periods_are_labeled_with_their_own_start(self):
        """xarray's convention, and therefore the one the returned ``time``
        axis follows. The drivers label interval *ends*, so a day's rows
        labeled 00:00 to 21:00 group under that day: the same grouping
        SIPNET's own ``day`` column gives, which is the point."""
        par = three_hourly_field("par", n_days=3, start="2013-01-01")
        daily = aggregate_time(par, "1D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(daily["time"].values),
            pd.date_range("2013-01-01", periods=3, freq="1D"),
            check_names=False,
        )


class TestAggregationCounts:
    def test_counts_the_finite_values_in_each_period(self):
        """Against ``numpy``'s own count over the same rows."""
        par = three_hourly_field("par", n_days=3)
        par[STEPS_PER_DAY + 2 : STEPS_PER_DAY + 5] = np.nan
        counts = aggregation_counts(par, "1D")
        np.testing.assert_array_equal(
            counts.values, np.isfinite(daily_blocks(par)).sum(axis=1)
        )

    def test_a_period_with_no_observations_counts_zero(self):
        """Zero rather than missing, so a caller can test for it."""
        par = three_hourly_field("par")
        par[STEPS_PER_DAY : 2 * STEPS_PER_DAY] = np.nan
        assert aggregation_counts(par, "1D").values[1] == 0

    def test_it_aligns_with_what_aggregate_time_returns(self):
        """Same dimensions and same ``time`` axis, so ``.where(counts == 8)``
        needs no reindexing. This is the whole reason it is public."""
        field = three_hourly_field("par", dims=("member", "site", "time"))
        daily = aggregate_time(field, "1D")
        counts = aggregation_counts(field, "1D")
        assert counts.dims == daily.dims
        np.testing.assert_array_equal(counts["time"].values, daily["time"].values)
        whole = daily.where(counts == STEPS_PER_DAY)
        assert np.isfinite(whole.values).all()

    def test_it_does_not_need_the_aggregation_attribute(self):
        """A field with no rule is counted: a count is a property of the
        missingness, not of the variable."""
        field = three_hourly_field("lai")
        assert AGGREGATION_ATTR not in field.attrs
        counts = aggregation_counts(field, "1D")
        np.testing.assert_array_equal(counts.values, [STEPS_PER_DAY] * 3)

    def test_it_returns_integer_counts_and_is_not_a_canonical_field(self):
        """The documented contract. A float count breaks the ``counts == 8``
        idiom the module's own Usage section shows, and a count carrying the
        field's ``units`` would be plottable as if it were the field."""
        counts = aggregation_counts(three_hourly_field("par"), "1D")
        assert counts.dtype == np.int64
        assert counts.name is None
        assert counts.attrs == {}

    def test_a_period_holding_no_rows_at_all_counts_zero(self):
        """Distinct from a period whose rows are all missing: here the time
        axis itself has a gap, so the sum over the empty group is ``NaN``
        before the fill. Casting that to ``int64`` is undefined -- it
        saturates to 0 on arm64 and to INT64_MIN on x86-64, which is what the
        cluster runs -- so a machine-dependent count would read as fully
        observed there."""
        times = pd.DatetimeIndex(
            list(pd.date_range("2013-01-01", periods=8, freq="3h"))
            + list(pd.date_range("2013-01-05", periods=8, freq="3h"))
        )
        field = xr.DataArray(
            np.arange(16.0), dims="time", coords={"time": times}, name="par",
            attrs={AGGREGATION_ATTR: "sum"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            counts = aggregation_counts(field, "1D")
        np.testing.assert_array_equal(counts.values, [8, 0, 0, 0, 8])
        assert counts.dtype == np.int64
        assert np.isnan(aggregate_time(field, "1D").values[1:4]).all()

    def test_it_refuses_the_same_time_axes_aggregate_time_refuses(self):
        """The checks are shared, so a caller cannot get a count for a field
        that cannot be aggregated."""
        with pytest.raises(ValueError, match="needs 'time'"):
            aggregation_counts(three_hourly_field("par", dims=("site",)), "1D")
        with pytest.raises(ValueError, match="would interpolate rather than"):
            aggregation_counts(three_hourly_field("par"), "1h")


@pytest.mark.skipif(not REAL_FILE.exists(), reason="the raw driver file is not present")
class TestAgainstTheRealDrivers:
    """The seam. The synthetic cases exercise the branches; these confirm the
    rules hold on the file the model is actually run on."""

    @staticmethod
    def clim_frame() -> pd.DataFrame:
        """The real ``.clim`` file's year, day, time, PAR and air temperature.

        Read with ``pandas`` directly rather than through ``load_drivers``, so
        that the expected values and the values under test do not share a
        code path.
        """
        frame = pd.read_csv(
            REAL_FILE,
            sep=r"\s+",
            header=None,
            usecols=[1, 2, 3, 5, 7],
            names=["year", "day", "time", "tair", "par"],
        )
        frame["stamp"] = sipnet_time_index(frame["year"], frame["day"], frame["time"])
        return frame

    def real_field(self, name: str) -> xr.DataArray:
        frame = self.clim_frame()
        column = {"par": "par", "air_temperature": "tair"}[name]
        return xr.DataArray(
            frame[column].to_numpy(),
            dims="time",
            coords={"time": frame["stamp"].to_numpy()},
            name=name,
            attrs={
                "units": "arbitrary",
                "long_name": name,
                AGGREGATION_ATTR: AGGREGATION_FOR[name],
            },
        )

    def test_the_daily_par_total_equals_the_sum_over_the_whole_record(self):
        """A sum over days conserves the total, which a mean does not. An
        aggregation that dropped or double-counted a day boundary shows up
        here."""
        par = self.real_field("par")
        daily = aggregate_time(par, "1D")
        assert daily.values.sum() == pytest.approx(par.values.sum())
        assert daily.sizes["time"] == 4749

    def test_the_daily_par_totals_match_a_pandas_groupby(self):
        """Two independent paths to the same numbers."""
        frame = self.clim_frame()
        expected = frame.groupby(["year", "day"], sort=True)["par"].sum().to_numpy()
        daily = aggregate_time(self.real_field("par"), "1D")
        np.testing.assert_allclose(daily.values, expected)

    def test_the_daily_air_temperature_is_a_mean_of_the_real_rows(self):
        """The other rule, on the same file, against the same groupby."""
        frame = self.clim_frame()
        expected = frame.groupby(["year", "day"], sort=True)["tair"].mean().to_numpy()
        daily = aggregate_time(self.real_field("air_temperature"), "1D")
        np.testing.assert_allclose(daily.values, expected)

    def test_every_day_of_the_real_record_holds_eight_rows(self):
        """The driver files have no missing values and no partial days, which
        is what makes ``min_count=1`` safe on them. Asserted rather than
        assumed, so a regenerated file that broke it would be noticed."""
        counts = aggregation_counts(self.real_field("par"), "1D")
        np.testing.assert_array_equal(
            counts.values, np.full(counts.sizes["time"], STEPS_PER_DAY)
        )


# ── reduce_windows ────────────────────────────────────────────────────────────


def day_windows(start: str, n_days: int, *, closed: str = "left") -> pd.IntervalIndex:
    """*n_days* consecutive calendar-day windows from *start*."""
    edges = pd.date_range(start, periods=n_days + 1, freq="1D")
    return pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed=closed)


class TestReduceWindows:
    def test_daily_windows_reproduce_aggregate_time(self):
        """The regular case is a special case of the general one: left-closed
        calendar days labeled by their start are what ``resample("1D")`` does."""
        field = three_hourly_field("par", dims=("member", "site", "time"))
        windows = day_windows("2013-01-01", 3)
        by_window = reduce_windows(field, windows, "sum", labels=windows.left)
        xr.testing.assert_allclose(by_window, aggregate_time(field, "1D"))
        assert by_window.dims == field.dims

    @pytest.mark.parametrize("how", REDUCTIONS)
    def test_irregular_windows_against_a_pandas_groupby(self, how):
        """Windows of unequal length, not aligned to any frequency, checked
        against ``pandas.cut`` on the same edges."""
        field = three_hourly_field("par", n_days=4)
        edges = pd.to_datetime(
            ["2013-01-01 06:00", "2013-01-01 15:00", "2013-01-03 00:00", "2013-01-04 09:00"]
        )
        windows = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")
        result = reduce_windows(field, windows, how)

        series = pd.Series(field.values, index=field["time"].values)
        groups = series.groupby(pd.cut(series.index, windows), observed=False)
        expected = getattr(groups, how)().to_numpy()
        np.testing.assert_allclose(result.values, expected)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result["time"].values), pd.DatetimeIndex(windows.right), check_names=False
        )

    def test_rows_outside_every_window_are_ignored(self):
        field = three_hourly_field("par", n_days=3)
        windows = day_windows("2013-01-02", 1)
        result = reduce_windows(field, windows, "sum")
        assert result.sizes["time"] == 1
        assert result.values[0] == pytest.approx(daily_blocks(field)[1].sum())

    def test_a_window_holding_no_rows_is_missing_and_counts_zero(self):
        field = three_hourly_field("par", n_days=3)
        windows = day_windows("2013-01-02", 3)  # the third day is past the record
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a NaN-to-int cast is platform-dependent
            result = reduce_windows(field, windows, "sum")
            counts = window_counts(field, windows)
        assert np.isnan(result.values[2]) and result.values[2] != 0.0
        np.testing.assert_array_equal(counts.values, [STEPS_PER_DAY, STEPS_PER_DAY, 0])
        assert counts.dtype == np.int64
        assert counts.name is None and counts.attrs == {}

    def test_no_window_holds_any_row(self):
        """Every window past the record: all missing, all counts zero, and
        the other dimensions and their coordinates still there."""
        field = three_hourly_field("par", dims=("site", "time"))
        windows = day_windows("2014-01-01", 2)
        result = reduce_windows(field, windows, "mean")
        assert result.dims == ("site", "time") and np.isnan(result.values).all()
        assert result["lon"].dims == ("site",)
        np.testing.assert_array_equal(window_counts(field, windows).values, 0)

    def test_min_count_masks_a_partial_window(self):
        field = three_hourly_field("par", n_days=2)
        field[STEPS_PER_DAY + 3 :] = np.nan
        windows = day_windows("2013-01-01", 2)
        loose = reduce_windows(field, windows, "sum")
        strict = reduce_windows(field, windows, "sum", min_count=STEPS_PER_DAY)
        assert np.isfinite(loose.values[1]) and np.isnan(strict.values[1])
        assert loose.values[1] == pytest.approx(np.nansum(daily_blocks(field)[1]))

    def test_the_closed_side_decides_a_label_on_a_shared_edge(self):
        """The drivers label interval ends, so right-closed windows collect a
        day's rows labeled 03:00 to 24:00; left-closed ones collect 00:00 to
        21:00. Same edges, different rows."""
        field = three_hourly_field("par", n_days=3)
        edges = pd.date_range("2013-01-01", periods=3, freq="1D")
        right = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="right")
        left = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="left")
        values = field.values
        np.testing.assert_allclose(
            reduce_windows(field, right, "sum").values,
            [values[1:9].sum(), values[9:17].sum()],
        )
        np.testing.assert_allclose(
            reduce_windows(field, left, "sum").values,
            [values[0:8].sum(), values[8:16].sum()],
        )

    def test_labels_become_the_time_coordinate(self):
        """An observation operator labels the result with the observation's
        own keys, which need not be an edge of any window."""
        field = three_hourly_field("par", n_days=3)
        windows = day_windows("2013-01-01", 3)
        labels = pd.to_datetime(["2013-07-15", "2014-07-15", "2015-07-15"])
        result = reduce_windows(field, windows, "sum", labels=labels)
        pd.testing.assert_index_equal(pd.DatetimeIndex(result["time"].values), labels, check_names=False)

    def test_a_single_window_over_the_whole_record_is_the_reduction(self):
        """The static case: one window, one number, computed by ``numpy``."""
        field = three_hourly_field("par", dims=("member", "site", "time"), n_days=3)
        whole = pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2012-12-31")], [pd.Timestamp("2013-01-04")], closed="right"
        )
        result = reduce_windows(field, whole, "mean")
        assert result.sizes["time"] == 1
        np.testing.assert_allclose(result.values[..., 0], field.values.mean(axis=-1))

    def test_the_result_is_independent_of_the_stored_dimension_order(self):
        canonical = three_hourly_field("par", dims=("site", "time"))
        transposed = canonical.transpose("time", "site")
        windows = day_windows("2013-01-01", 3)
        xr.testing.assert_allclose(
            reduce_windows(canonical, windows, "sum"),
            reduce_windows(transposed, windows, "sum").transpose("site", "time"),
        )
        expected = canonical.values.reshape(2, -1, STEPS_PER_DAY).sum(axis=-1)
        np.testing.assert_allclose(reduce_windows(canonical, windows, "sum").values, expected)

    def test_a_window_with_exactly_min_count_values_is_kept(self):
        """The boundary: ``>=``, not ``>``. With the default of 1, a window
        holding one value is a value, not missing."""
        field = three_hourly_field("par", n_days=2)
        field[STEPS_PER_DAY + 1 :] = np.nan
        windows = day_windows("2013-01-01", 2)
        loose = reduce_windows(field, windows, "sum")
        assert loose.values[1] == pytest.approx(daily_blocks(field)[1, 0])
        field[STEPS_PER_DAY + 3 :] = field.values[STEPS_PER_DAY]
        exact = reduce_windows(field, windows, "sum", min_count=6)
        assert np.isfinite(exact.values[1])
        assert np.isnan(reduce_windows(field, windows, "sum", min_count=7).values[1])

    def test_decreasing_windows_raise(self):
        """Non-overlapping but out of order: pandas accepts them, and the
        values would come back paired with the wrong labels."""
        field = three_hourly_field("par", n_days=3)
        windows = day_windows("2013-01-01", 3)[::-1]
        with pytest.raises(ValueError, match="increasing order"):
            reduce_windows(field, windows, "sum")

    def test_duplicate_labels_raise(self):
        field = three_hourly_field("par", n_days=3)
        windows = day_windows("2013-01-01", 3)
        labels = pd.to_datetime(["2013-07-15", "2013-07-15", "2015-07-15"])
        with pytest.raises(ValueError, match="strictly increasing"):
            reduce_windows(field, windows, "sum", labels=labels)

    def test_degenerate_windows_with_repeated_default_labels_raise(self):
        field = three_hourly_field("par", n_days=3)
        edge = pd.Timestamp("2013-01-02")
        windows = pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2013-01-01"), edge], [edge, edge], closed="left"
        )
        with pytest.raises(ValueError, match="right edges must be strictly increasing"):
            reduce_windows(field, windows, "sum")

    def test_numeric_labels_raise_rather_than_becoming_1970(self):
        field = three_hourly_field("par", n_days=3)
        with pytest.raises(ValueError, match="labels must be timestamps"):
            reduce_windows(field, day_windows("2013-01-01", 3), "sum", labels=[1, 2, 3])

    def test_windows_holding_nat_are_named_as_such(self):
        field = three_hourly_field("par", n_days=3)
        windows = pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2013-01-01"), pd.NaT], [pd.Timestamp("2013-01-02"), pd.NaT]
        )
        with pytest.raises(ValueError, match="NaT"):
            reduce_windows(field, windows, "sum")

    def test_window_counts_carry_the_labels_and_int64_when_nothing_matches(self):
        """The counts align with the reduction by coordinate, not only by
        position, in the all-outside branch too."""
        field = three_hourly_field("par", dims=("site", "time"))
        windows = day_windows("2014-01-01", 2)
        labels = pd.to_datetime(["2014-07-15", "2015-07-15"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            counts = window_counts(field, windows, labels=labels)
            reduced = reduce_windows(field, windows, "sum", labels=labels)
        assert counts.dtype == np.int64
        pd.testing.assert_index_equal(pd.DatetimeIndex(counts["time"].values), labels, check_names=False)
        pd.testing.assert_index_equal(pd.DatetimeIndex(reduced["time"].values), labels, check_names=False)

    def test_windows_in_another_datetime_resolution_are_accepted(self):
        """``sipnet_time_index`` produces nanoseconds; ``pd.to_datetime`` and
        ``pd.date_range`` produce microseconds under pandas 3. The two must
        meet without the caller converting."""
        n = STEPS_PER_DAY * 3
        stamps = sipnet_time_index([2013] * n, np.repeat([1, 2, 3], STEPS_PER_DAY), np.tile(np.arange(0, 24, 3.0), 3))
        assert stamps.unit == "ns"
        field = three_hourly_field("par", n_days=3).assign_coords(time=stamps)
        windows = day_windows("2013-01-01", 3)
        assert windows.left.unit == "us"
        result = reduce_windows(field, windows, "sum", labels=windows.left)
        np.testing.assert_allclose(result.values, daily_blocks(field).sum(axis=1))
        np.testing.assert_array_equal(window_counts(field, windows).values, STEPS_PER_DAY)
        coarse = field.assign_coords(time=stamps.as_unit("s"))
        np.testing.assert_allclose(
            reduce_windows(coarse, windows, "sum").values, daily_blocks(field).sum(axis=1)
        )

    def test_timezone_aware_field_and_windows_group_in_that_zone(self):
        """The rows keep their zone through the grouping: a window given in
        the field's zone collects the rows it names, not their UTC wall clock."""
        field = three_hourly_field("par", n_days=2)
        aware = field.assign_coords(
            time=pd.DatetimeIndex(field["time"].values).tz_localize("America/New_York")
        )
        edges = pd.date_range("2013-01-01", periods=3, freq="1D", tz="America/New_York")
        windows = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="left")
        result = reduce_windows(aware, windows, "sum")
        np.testing.assert_allclose(result.values, daily_blocks(field).sum(axis=1))
        np.testing.assert_array_equal(window_counts(aware, windows).values, STEPS_PER_DAY)

    def test_a_time_zone_mismatch_raises_rather_than_matching_nothing(self):
        field = three_hourly_field("par", n_days=2)
        naive_windows = day_windows("2013-01-01", 2)
        aware = field.assign_coords(time=pd.DatetimeIndex(field["time"].values).tz_localize("UTC"))
        with pytest.raises(ValueError, match="time zone"):
            reduce_windows(aware, naive_windows, "sum")
        edges = pd.date_range("2013-01-01", periods=3, freq="1D", tz="UTC")
        aware_windows = pd.IntervalIndex.from_arrays(edges[:-1], edges[1:], closed="left")
        with pytest.raises(ValueError, match="time zone"):
            reduce_windows(field, aware_windows, "sum")

    def test_the_default_labels_say_they_are_interval_ends(self):
        field = three_hourly_field("par", n_days=3)
        result = reduce_windows(field, day_windows("2013-01-01", 3), "sum")
        assert result["time"].attrs[TIME_LABEL_ATTR] == TimeLabel.INTERVAL_END.value

    def test_labels_given_as_a_dataarray_keep_their_attributes(self):
        """An observation operator labels the result with the observation's
        own coordinate, which says what its labels mean."""
        field = three_hourly_field("par", n_days=3)
        labels = xr.DataArray(
            pd.to_datetime(["2013-07-15", "2014-07-15", "2015-07-15"]),
            dims="time",
            attrs={TIME_LABEL_ATTR: TimeLabel.NOMINAL.value, "time_zone": "none"},
        )
        result = reduce_windows(field, day_windows("2013-01-01", 3), "sum", labels=labels)
        assert result["time"].attrs[TIME_LABEL_ATTR] == "nominal"
        assert result["time"].attrs["time_zone"] == "none"
        plain = reduce_windows(field, day_windows("2013-01-01", 3), "sum", labels=labels.values)
        assert plain["time"].attrs == {}

    def test_how_given_as_a_string_enum_is_stored_as_a_plain_string(self):
        """h5netcdf cannot write a ``str`` subclass as an attribute."""

        class Rule(StrEnum):
            SUM = "sum"

        field = three_hourly_field("par", n_days=3)
        result = reduce_windows(field, day_windows("2013-01-01", 3), Rule.SUM)
        assert type(result.attrs["aggregation_applied"]) is str
        field.attrs[AGGREGATION_ATTR] = Rule.SUM
        assert type(aggregate_time(field, "1D").attrs["aggregation_applied"]) is str

    def test_an_array_of_reductions_is_refused(self):
        field = three_hourly_field("par", n_days=3)
        with pytest.raises(ValueError, match="how must be one of"):
            reduce_windows(field, day_windows("2013-01-01", 3), np.array(["sum"]))

    def test_the_name_and_attributes_survive(self):
        field = three_hourly_field("par")
        result = reduce_windows(field, day_windows("2013-01-01", 3), "mean")
        assert result.name == "par"
        assert result.attrs["units"] == "arbitrary"
        assert result.attrs["aggregation_applied"] == "mean"
        assert "aggregation_applied" not in field.attrs

    def test_how_is_required_and_the_attribute_is_not_consulted(self):
        """A window is an observation's shape, and the field's attribute is
        about the field's own steps; nothing is inferred."""
        field = three_hourly_field("par")
        with pytest.raises(TypeError):
            reduce_windows(field, day_windows("2013-01-01", 3))  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="how must be one of"):
            reduce_windows(field, day_windows("2013-01-01", 3), None)

    def test_overlapping_windows_raise(self):
        field = three_hourly_field("par")
        overlapping = pd.IntervalIndex.from_arrays(
            [pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-01 12:00")],
            [pd.Timestamp("2013-01-02"), pd.Timestamp("2013-01-02 12:00")],
        )
        with pytest.raises(ValueError, match="windows overlap"):
            reduce_windows(field, overlapping, "sum")

    def test_windows_that_are_not_datetimes_or_not_an_interval_index_raise(self):
        field = three_hourly_field("par")
        with pytest.raises(ValueError, match="must be a pandas.IntervalIndex"):
            reduce_windows(field, [("2013-01-01", "2013-01-02")], "sum")
        with pytest.raises(ValueError, match="intervals of datetimes"):
            reduce_windows(field, pd.interval_range(0, 3), "sum")
        with pytest.raises(ValueError, match="windows is empty"):
            reduce_windows(field, pd.IntervalIndex([], closed="right"), "sum")

    def test_labels_are_checked_against_the_windows(self):
        field = three_hourly_field("par")
        windows = day_windows("2013-01-01", 3)
        with pytest.raises(ValueError, match="one label per window"):
            reduce_windows(field, windows, "sum", labels=windows.left[:2])
        with pytest.raises(ValueError, match="one label per window"):
            window_counts(field, windows, labels=windows.left[:2])
        with pytest.raises(ValueError, match="strictly increasing"):
            reduce_windows(field, windows, "sum", labels=windows.left[::-1])

    def test_the_field_checks_are_shared_with_aggregate_time(self):
        with pytest.raises(ValueError, match="needs 'time'"):
            reduce_windows(three_hourly_field("par", dims=("site",)), day_windows("2013-01-01", 1), "sum")
        with pytest.raises(ValueError, match="min_count must be at least 1"):
            reduce_windows(three_hourly_field("par"), day_windows("2013-01-01", 1), "sum", min_count=0)


class TestAgainstTheRealConstraints:
    """The annual product is ragged in year; a site's observed years must
    aggregate to years without being mistaken for upsampling."""

    def test_a_ragged_site_aggregates_to_years(self, real_constraint_fields):
        means, _ = real_constraint_fields
        wood = means["aboveground_wood_carbon"]
        counts = wood.notnull().sum("time")
        partial = [int(s) for s, n in zip(wood["site"].values, counts.values) if 0 < n < wood.sizes["time"]]
        if not partial:
            pytest.skip("no site with a partial record")
        one = wood.sel(site=partial[0]).dropna("time")
        result = aggregate_time(one, "YS", how="last")
        assert result.notnull().sum().item() == one.sizes["time"]
        np.testing.assert_allclose(result.dropna("time").values, one.values)


# ── time conventions ──────────────────────────────────────────────────────────


class TestTimeLabel:
    def test_the_values_are_the_strings_the_products_write(self):
        """The enum fixes a vocabulary that already exists in written files;
        its values must be those strings exactly."""
        assert TimeLabel.INTERVAL_END == "interval_end"
        assert TimeLabel.INTERVAL_START == "interval_start"
        assert TimeLabel.INSTANT == "instant"
        assert TimeLabel.NOMINAL == "nominal"
        assert TimeLabel.STATIC == "static"
        assert TimeLabel("interval_end") is TimeLabel.INTERVAL_END
        assert TIME_LABEL_ATTR == "time_label"

    def test_the_drivers_use_a_member(self):
        from sipnet_calibration.drivers import TIME_LABEL

        assert TIME_LABEL is TimeLabel.INTERVAL_END
        assert TIME_LABEL.value == "interval_end"

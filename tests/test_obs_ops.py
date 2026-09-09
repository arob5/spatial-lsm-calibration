"""Tests for the observation-space operations.

Only ``sipnet_time_index`` exists so far; the rest of the module is issue #6.
The cases here are built by hand from SIPNET's ``year``/``day``/``time``
labeling, including the drifting ``time`` column the ``.clim`` drivers carry
(issue #9), so that a future change cannot start trusting that column's value.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="docstring review stage; implementation follows")


class TestSipnetTimeIndex:
    def test_builds_the_nominal_grid_from_year_day_and_slot(self):
        """Eight 3-hourly rows of one day give midnight through 21:00."""

    def test_ignores_the_drift_in_the_time_column(self):
        """A label of 23.00 in the last slot of 31 December still maps to 21:00."""

    def test_leap_day_is_placed_correctly(self):
        """Day 60 of 2012 is 29 February; day 60 of 2013 is 1 March."""

    def test_spans_a_year_boundary_in_row_order(self):
        pass

    def test_other_timesteps(self):
        """``timestep_hours=24.0`` labels a daily file; ``1.0`` an hourly one."""

    def test_rejects_a_timestep_that_does_not_divide_24(self):
        pass

    def test_rejects_unequal_lengths(self):
        pass

    def test_rejects_day_366_in_a_non_leap_year(self):
        pass

    def test_rejects_a_time_outside_the_day(self):
        pass

    def test_rejects_a_label_outside_its_slot(self):
        """A drift of one full step or more is an error, not a reassignment."""

    def test_reproduces_the_real_files_grid(self):
        """Against ``data/raw/drivers/``, skipped when absent: 37,992 nominal
        timestamps from 2012-01-01T00:00 to 2024-12-31T21:00, 3 h apart."""

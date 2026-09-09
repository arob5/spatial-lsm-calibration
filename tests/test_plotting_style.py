"""Tests for :mod:`sipnet_calibration.plotting.style`.

Skipped until the module is implemented; collected so that the intended
assertions are on the record and reviewable.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="style.py is a contract skeleton; not yet implemented")


# ── role_style ────────────────────────────────────────────────────────────────


def test_role_style_line_takes_only_line_keywords():
    """``kind="line"`` returns color, line style and width, and no marker."""
    raise NotImplementedError


def test_role_style_band_takes_only_the_color():
    """``kind="band"`` returns the color alone; a fan owns its own opacity."""
    raise NotImplementedError


def test_role_style_points_forces_no_line():
    """``kind="points"`` returns the marker and ``linestyle="none"``."""
    raise NotImplementedError


def test_role_style_overrides_win():
    """An explicit keyword beats the role's, and an unrelated one passes through."""
    raise NotImplementedError


def test_role_style_returns_a_fresh_dictionary():
    """Mutating the result does not change :data:`ROLES`."""
    raise NotImplementedError


def test_role_style_rejects_an_unknown_role():
    """An unknown role raises, and the message lists the valid roles."""
    raise NotImplementedError


def test_role_style_rejects_an_unknown_kind():
    """An unknown artist kind raises, and the message lists the valid kinds."""
    raise NotImplementedError


# ── the palette ───────────────────────────────────────────────────────────────


def test_every_role_names_a_color():
    """Each :data:`ROLES` entry has a color, so no role falls back to the cycle."""
    raise NotImplementedError


def test_roles_are_distinguishable_without_color():
    """The roles differ in line style or marker, not in color alone."""
    raise NotImplementedError


# ── rcParams ──────────────────────────────────────────────────────────────────


def test_importing_the_package_does_not_change_rcparams():
    """Importing :mod:`sipnet_calibration.plotting` leaves ``rcParams`` alone."""
    raise NotImplementedError


def test_use_project_style_applies_rc_params():
    """After the call, every key of :data:`RC_PARAMS` is in ``rcParams``."""
    raise NotImplementedError


# ── axis_label ────────────────────────────────────────────────────────────────


def test_axis_label_reads_the_attributes(field_time):
    """The label is ``"<long_name> (<units>)"`` from ``attrs``."""
    raise NotImplementedError


@pytest.mark.parametrize("missing", ["units", "long_name"])
def test_axis_label_rejects_a_field_missing_an_attribute(field_time, missing):
    """A field without the attribute raises, and the message names it."""
    raise NotImplementedError


def test_axis_label_on_a_real_driver_field(real_driver_field):
    """The real driver attributes give a usable label."""
    raise NotImplementedError

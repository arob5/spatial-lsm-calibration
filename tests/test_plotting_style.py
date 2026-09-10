"""Tests for :mod:`sipnet_calibration.plotting.style`."""

from __future__ import annotations

import subprocess
import sys

import matplotlib
import pytest

from sipnet_calibration.plotting.style import (
    BAND_ALPHAS,
    CURVE_COLORS,
    RC_PARAMS,
    ROLES,
    axis_label,
    role_style,
    use_project_style,
)


# ── role_style ────────────────────────────────────────────────────────────────


def test_role_style_line_takes_only_line_keywords():
    """``kind="line"`` returns color, line style and width, and no marker."""
    style = role_style("obs", "line")
    assert set(style) <= {"color", "linestyle", "linewidth"}
    assert style["color"] == ROLES["obs"]["color"]
    assert "marker" not in style


def test_role_style_band_takes_only_the_color():
    """``kind="band"`` returns the color alone; a fan owns its own opacity."""
    assert role_style("posterior", "band") == {"color": ROLES["posterior"]["color"]}


def test_role_style_points_forces_no_line():
    """``kind="points"`` returns the marker and ``linestyle="none"``."""
    style = role_style("obs", "points")
    assert style["linestyle"] == "none"
    assert style["marker"] == ROLES["obs"]["marker"]
    assert style["markersize"] == ROLES["obs"]["markersize"]


def test_role_style_points_forces_no_line_even_for_a_line_role():
    """A role with a real line style still draws points without a line."""
    assert role_style("posterior", "points")["linestyle"] == "none"


def test_role_style_overrides_win():
    """An explicit keyword beats the role's; an unrelated one passes through."""
    style = role_style("prior", "line", linewidth=4.0, zorder=5)
    assert style["linewidth"] == 4.0
    assert style["zorder"] == 5
    assert style["color"] == ROLES["prior"]["color"]


def test_role_style_returns_a_fresh_dictionary():
    """Mutating the result does not change :data:`ROLES`."""
    before = dict(ROLES["prior"])
    role_style("prior")["color"] = "#ffffff"
    assert ROLES["prior"] == before


def test_role_style_rejects_an_unknown_role():
    """An unknown role raises, and the message lists the valid roles."""
    with pytest.raises(ValueError, match="unknown role") as raised:
        role_style("bayesian")
    assert "posterior" in str(raised.value)


def test_role_style_rejects_an_unknown_kind():
    """An unknown element kind raises, and the message lists the valid kinds."""
    with pytest.raises(ValueError, match="unknown kind") as raised:
        role_style("prior", "surface")
    assert "band" in str(raised.value)


# ── the palette ───────────────────────────────────────────────────────────────


def test_every_role_names_a_color():
    """Every :data:`ROLES` entry names a color, so none falls back to the
    cycle.
    """
    assert all("color" in style for style in ROLES.values())


def test_roles_are_distinguishable_without_color():
    """The roles differ in line style or marker, not in color alone."""
    signatures = {
        (style.get("linestyle"), style.get("marker")) for style in ROLES.values()
    }
    assert len(signatures) == len(ROLES)


def test_the_curve_colors_do_not_repeat():
    """A cycle with a repeat would give two curves the same color needlessly."""
    assert len(set(CURVE_COLORS)) == len(CURVE_COLORS)


def test_band_alphas_run_from_faint_to_solid():
    """The widest band is faintest, so narrower intervals read as stronger."""
    assert 0.0 < BAND_ALPHAS[0] < BAND_ALPHAS[1] <= 1.0


# ── rcParams ──────────────────────────────────────────────────────────────────


def test_importing_the_package_does_not_change_rcparams():
    """Importing :mod:`sipnet_calibration.plotting` leaves ``rcParams`` alone.

    Run in a subprocess: this process may already have imported the package,
    or applied the project style, and either would hide the regression.
    """
    code = (
        "import matplotlib; matplotlib.use('Agg');"
        "import sipnet_calibration.plotting;"
        "print(matplotlib.rcParams['axes.spines.top'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"
    assert RC_PARAMS["axes.spines.top"] is False


def test_use_project_style_applies_rc_params():
    """The settings the project actually depends on are applied.

    Named individually rather than looped over :data:`RC_PARAMS`, which would
    only be testing ``dict.update`` and would pass if an entry were deleted.
    """
    original = matplotlib.rcParams.copy()
    try:
        use_project_style()
        assert matplotlib.rcParams["figure.constrained_layout.use"] is True
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["legend.frameon"] is False
        cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        assert cycle == list(CURVE_COLORS)
        for key, value in RC_PARAMS.items():
            assert matplotlib.rcParams[key] == value
    finally:
        matplotlib.rcParams.update(original)


# ── axis_label ────────────────────────────────────────────────────────────────


def test_axis_label_reads_the_attributes(field_time):
    """The label is ``"<long_name> (<units>)"`` from ``attrs``."""
    assert axis_label(field_time) == "Mean air temperature over the timestep (deg C)"


@pytest.mark.parametrize("missing", ["units", "long_name"])
def test_axis_label_rejects_a_field_missing_an_attribute(field_time, missing):
    """A field without the attribute raises, and the message names it."""
    stripped = field_time.copy()
    stripped.attrs = {k: v for k, v in field_time.attrs.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        axis_label(stripped)


def test_axis_label_on_a_real_driver_field(real_driver_field):
    """The real driver attributes give a usable label."""
    label = axis_label(real_driver_field)
    assert "air temperature" in label.lower()
    assert label.endswith("(deg C)")

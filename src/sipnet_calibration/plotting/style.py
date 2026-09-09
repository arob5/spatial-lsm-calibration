"""Roles, colors and rcParams -- the display vocabulary the panels draw on.

Overview
--------
Two small tables and three functions keep style keywords out of every call
site. :data:`ROLES` maps a *semantic* role -- ``prior``, ``posterior``,
``obs``, ``truth`` -- onto matplotlib keywords, so that every figure in the
project reads the same way and a panel is asked for what a series *means*
rather than for a color. :data:`CURVE_COLORS` is the categorical cycle used
when individual curves have to be told apart.

Nothing here is applied on import. :func:`use_project_style` mutates the
global ``rcParams`` and is called by an experiment report in
``experiments/<task>/plots.py``, never by a library plotter: importing a
plotting module must not reach into a notebook's or another caller's figures.

Style precedence, in every panel and primitive in this package::

    primitive default  <-  ROLES[role]  <-  explicit **style keyword

An unrecognized keyword is passed through to matplotlib and raises there,
rather than being filtered out silently. :func:`role_style` is what projects a
role onto the subset of keywords a given artist kind understands, so one role
entry can serve a line, a band and a set of points without any primitive
having to know about roles.

Functions
---------
:func:`role_style`
    A role's keywords, projected onto one artist kind, with overrides applied.
:func:`use_project_style`
    Apply :data:`RC_PARAMS` to the global ``rcParams``.
:func:`axis_label`
    The axis label for a canonical field, from its attributes.

Notes
-----
The palette is Okabe-Ito, which is distinguishable under the common forms of
color vision deficiency and in grayscale. Roles are also separated by line
style and marker, so a figure does not rely on color alone.

:func:`axis_label` reads ``attrs`` only. It is the single place the
``VARIABLES`` registry (issue #6) plugs into the series layer: when the
registry lands, this function prefers ``VARIABLES[field.name].label`` over
``attrs["long_name"]`` and raises if ``attrs["units"]`` disagrees with the
registry's canonical unit. Nothing else in the series layer needs the
registry -- ``agg`` is the caller's verb, and ``cmap``/``center`` are map
concerns -- so this module deliberately does not import it.

Usage
-----
::

    from sipnet_calibration.plotting.style import role_style, use_project_style

    use_project_style()                        # in an L4 report, once
    role_style("obs", "points")                # -> color, marker, linestyle
    role_style("prior", "line", linewidth=2)   # override the width
"""

from __future__ import annotations

from typing import Any

import matplotlib
import xarray as xr

__all__ = [
    "BAND_ALPHAS",
    "CURVE_COLORS",
    "RC_PARAMS",
    "ROLES",
    "axis_label",
    "role_style",
    "use_project_style",
]

#: Semantic role -> matplotlib keywords. The keys are the roles a panel may be
#: asked for; the values are the full style for that role, from which
#: :func:`role_style` selects the part a given artist kind understands. Roles
#: differ in line style and marker as well as color, so a figure remains
#: readable in grayscale.
ROLES: dict[str, dict[str, Any]] = {
    "prior": {"color": "#999999", "linestyle": "-", "linewidth": 1.0},
    "posterior": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.2},
    "obs": {
        "color": "#000000",
        "linestyle": "none",
        "marker": "o",
        "markersize": 3.5,
    },
    "truth": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.4},
}

#: Categorical colors for curves that have to be told apart -- the ``label_by``
#: case in :func:`sipnet_calibration.plotting.series.series_panel`, where each
#: line is a different site. Okabe-Ito without black, which :data:`ROLES`
#: reserves for observations. Cycled if there are more curves than colors.
CURVE_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
)

#: Opacity of the widest and the narrowest band of a fan. Intermediate bands
#: are spaced linearly between them, so that a narrower -- more probable --
#: interval reads as more solid. Used by
#: :func:`sipnet_calibration.plotting.primitives.fan`.
BAND_ALPHAS: tuple[float, float] = (0.12, 0.35)

#: The project's matplotlib settings, applied only by
#: :func:`use_project_style`.
RC_PARAMS: dict[str, Any] = {
    "figure.constrained_layout.use": True,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=CURVE_COLORS),
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def role_style(role: str, kind: str = "line", **overrides: Any) -> dict[str, Any]:
    """A role's keywords for one artist kind, with *overrides* applied.

    Parameters
    ----------
    role:
        A key of :data:`ROLES`.
    kind:
        The artist the keywords are for, which decides which of the role's
        keywords are meaningful:

        ==========  ==================================================
        ``kind``    Keywords taken from the role
        ==========  ==================================================
        ``line``    ``color``, ``linestyle``, ``linewidth``
        ``band``    ``color`` only; a fan owns its own opacity
        ``points``  ``color``, ``marker``, ``markersize``, and
                    ``linestyle`` forced to ``"none"``
        ==========  ==================================================

        A role that does not name one of them simply does not contribute it,
        and the primitive's own default stands.
    **overrides:
        Keywords that win over the role's, whatever the kind. They are not
        filtered: an override is what the caller explicitly asked for, so it
        reaches matplotlib and raises there if it is not a real keyword.

    Returns
    -------
    dict
        A fresh dictionary; mutating it does not touch :data:`ROLES`.

    Raises
    ------
    ValueError
        If *role* is not a key of :data:`ROLES`, or *kind* is not one of
        ``"line"``, ``"band"`` or ``"points"``. The message lists the valid
        values.
    """
    raise NotImplementedError


def use_project_style() -> None:
    """Apply :data:`RC_PARAMS` to matplotlib's global ``rcParams``.

    Call this once in an experiment report. No module in this package calls
    it, and importing this package does not change ``rcParams``: a library
    that restyles a caller's figures on import is a library that cannot be
    imported safely from a notebook.

    Returns
    -------
    None
    """
    raise NotImplementedError


def axis_label(field: xr.DataArray) -> str:
    """The axis label for a canonical *field*, as ``"long name (units)"``.

    Parameters
    ----------
    field:
        A canonical field, carrying ``long_name`` and ``units`` in ``attrs``
        as every adapter in this project produces.

    Returns
    -------
    str
        For example ``"Mean air temperature over the timestep (deg C)"``.

    Raises
    ------
    ValueError
        If ``long_name`` or ``units`` is missing from ``attrs``, naming which.
        A field that has lost its attributes has usually lost them to an
        xarray operation that does not propagate them, which is worth knowing
        about rather than papering over with a bare variable name.

    Notes
    -----
    This is the seam for the ``VARIABLES`` registry (issue #6); see the module
    docstring.
    """
    raise NotImplementedError

"""Roles, colors and matplotlib settings.

The appearance of a series in this project's figures follows from what the
series *is* -- a prior, a posterior, an observation, a truth -- rather than
from keywords at the call site. This module holds that mapping, together with
the palette and the matplotlib settings the figures are drawn under, so that
the same quantity looks the same wherever it appears.

The palette is Okabe-Ito, which stays distinguishable under the common forms
of color vision deficiency, and the roles differ in line style and marker as
well as in color, so a figure survives being printed in grayscale.

Style is resolved in three stages, each overriding the one before: the drawing
function's own default, then the role, then any keyword the caller passes
explicitly. A keyword this module does not recognize is passed on to
matplotlib unchanged.

Importing this module does not change matplotlib's global ``rcParams``;
:func:`use_project_style` is what applies :data:`RC_PARAMS`.

Functions
---------
:func:`role_style`
    A role's keywords for one kind of element, with overrides applied.
:func:`use_project_style`
    Apply :data:`RC_PARAMS` to matplotlib's global ``rcParams``.
:func:`axis_label`
    The axis label for a field, from its ``units`` and ``long_name``.

Usage
-----
::

    from sipnet_calibration.plotting import role_style, use_project_style

    use_project_style()                        # once, where figures are made
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

#: Role name to matplotlib keywords. The keys are the roles a panel may be
#: asked for; :func:`role_style` selects from a value the part that applies to
#: a given kind of element.
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

#: Colors for curves that have to be told apart from one another, such as one
#: curve per site in a single panel. Cycled if there are more curves than
#: colors. Black is not among them; :data:`ROLES` uses it for observations.
CURVE_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
)

#: Opacity of the widest and of the narrowest band of a fan. Intermediate
#: bands are spaced linearly between the two, so a narrower interval is drawn
#: more solidly.
BAND_ALPHAS: tuple[float, float] = (0.12, 0.35)

#: The project's matplotlib settings, applied by :func:`use_project_style`.
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
    """A role's keywords for one kind of element, with *overrides* applied.

    Parameters
    ----------
    role:
        A key of :data:`ROLES`.
    kind:
        The kind of element the keywords are for, which decides which of the
        role's keywords are returned:

        ==========  ==================================================
        ``kind``    Keywords taken from the role
        ==========  ==================================================
        ``line``    ``color``, ``linestyle``, ``linewidth``
        ``band``    ``color``
        ``points``  ``color``, ``marker``, ``markersize``, and
                    ``linestyle`` set to ``"none"``
        ==========  ==================================================

        A role that does not name one of them does not contribute it.
    **overrides:
        Keywords that override the role's, whatever the kind. They are
        returned unfiltered, so an override that matplotlib does not accept
        raises where it is used rather than being dropped here.

    Returns
    -------
    dict
        A new dictionary; changing it does not affect :data:`ROLES`.

    Raises
    ------
    ValueError
        If *role* is not a key of :data:`ROLES`, or *kind* is not ``"line"``,
        ``"band"`` or ``"points"``. The message lists the valid values.
    """
    raise NotImplementedError


def use_project_style() -> None:
    """Apply :data:`RC_PARAMS` to matplotlib's global ``rcParams``.

    Call this once where a set of figures is produced. Nothing in this package
    calls it, and importing the package leaves ``rcParams`` untouched.

    Returns
    -------
    None
    """
    raise NotImplementedError


def axis_label(field: xr.DataArray) -> str:
    """The axis label for *field*, as ``"<long_name> (<units>)"``.

    Parameters
    ----------
    field:
        A field carrying ``long_name`` and ``units`` in ``attrs``, as the
        readers in this project produce.

    Returns
    -------
    str
        For example ``"Mean air temperature over the timestep (deg C)"``.

    Raises
    ------
    ValueError
        If ``long_name`` or ``units`` is missing from ``attrs``, naming which
        one. An xarray operation that does not carry attributes forward is the
        usual cause.
    """
    # The VARIABLES registry (issue #6) plugs in here when it lands: prefer
    # VARIABLES[field.name].label over attrs["long_name"], and raise if
    # attrs["units"] disagrees with the registry's canonical unit. This is the
    # only place in the series layer that needs it.
    raise NotImplementedError

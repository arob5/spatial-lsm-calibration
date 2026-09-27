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
    A role's keywords for one element of a figure, with overrides applied.
:func:`use_project_style`
    Apply :data:`RC_PARAMS` to matplotlib's global ``rcParams``.
:func:`axis_label`
    The axis label for a field, from its ``units`` and ``long_name``.
:func:`category_colors`
    One color per class of a categorical map.

Usage
-----
::

    from sipnet_calibration.plotting import role_style, use_project_style

    use_project_style()                        # once, where figures are made
    role_style("observation", "points")        # -> color, marker, linestyle
    role_style("prior", "line", linewidth=2)   # override the width
"""

from __future__ import annotations

import numbers
from collections.abc import Collection, Mapping
from typing import Any

import matplotlib
import numpy as np
import xarray as xr
from frozendict import frozendict

from sipnet_calibration.fields import check_field_is_a_dataarray, message_name
from sipnet_calibration.validation import as_bounded_integer, truncated

__all__ = [
    "BAND_ALPHAS",
    "CATEGORY_COLORS",
    "CURVE_COLORS",
    "RC_PARAMS",
    "ROLES",
    "axis_label",
    "category_colors",
    "check_key_is_known",
    "check_keywords_are_not_retired",
    "check_number_is_finite",
    "check_number_is_positive",
    "check_option_is_known",
    "check_value_is_a_number",
    "role_style",
    "use_project_style",
]

#: Role name to matplotlib keywords. The keys are the roles a panel may be
#: asked for; :func:`role_style` selects from a value the part that applies to
#: a given element. No two roles share a line style and marker, so
#: they stay apart in grayscale as well as in color.
ROLES: frozendict = frozendict(
    {
        "prior": frozendict({"color": "#999999", "linestyle": "--", "linewidth": 1.0}),
        "posterior": frozendict({"color": "#0072B2", "linestyle": "-", "linewidth": 1.2}),
        "observation": frozendict(
            {
                "color": "#000000",
                "linestyle": "none",
                "marker": "o",
                "markersize": 3.5,
            }
        ),
        "truth": frozendict({"color": "#D55E00", "linestyle": "-.", "linewidth": 1.4}),
    }
)

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

#: Colors for the classes of a categorical map, by class position. Up to eight
#: classes take Okabe-Ito, as :data:`CURVE_COLORS` plus black; more take
#: matplotlib's ``tab20``, which is not safe under color vision deficiency but
#: has enough distinct entries for the site-labels data sources' classes.
CATEGORY_COLORS: tuple[str, ...] = CURVE_COLORS + ("#000000",)

#: Opacity of the widest and of the narrowest band of a fan. Intermediate
#: bands are spaced linearly between the two, so a narrower interval is drawn
#: more solidly.
BAND_ALPHAS: tuple[float, float] = (0.12, 0.35)

#: The project's matplotlib settings, applied by :func:`use_project_style`.
RC_PARAMS: frozendict = frozendict(
    {
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
)


def role_style(role: str, element: str = "line", **overrides: Any) -> dict[str, Any]:
    """A role's keywords for one element of a figure, with *overrides* applied.

    Parameters
    ----------
    role:
        A key of :data:`ROLES`.
    element:
        The element the keywords are for, which decides which of the role's
        keywords are returned:

        ===========  ==================================================
        ``element``  Keywords taken from the role
        ===========  ==================================================
        ``line``     ``color``, ``linestyle``, ``linewidth``
        ``band``     ``color``
        ``points``   ``color``, ``marker``, ``markersize``, and
                     ``linestyle`` set to ``"none"``
        ===========  ==================================================

        A role that does not name one of them does not contribute it.
    **overrides:
        Keywords that override the role's, whatever the element. They are
        returned unfiltered, so an override that matplotlib does not accept
        raises where it is used rather than being dropped here.

    Returns
    -------
    dict
        A new dictionary; changing it does not affect :data:`ROLES`.

    Raises
    ------
    KeyError
        If *role* is not a key of :data:`ROLES`.
    ValueError
        If *element* is not ``"line"``, ``"band"`` or ``"points"``.
    """
    check_key_is_known(role, ROLES, message_name="role")
    check_option_is_known(element, tuple(_ELEMENT_KEYWORDS), message_name="element")
    style = {
        key: value
        for key, value in ROLES[role].items()
        if key in _ELEMENT_KEYWORDS[element]
    }
    if element == "points":
        style["linestyle"] = "none"
    style.update(overrides)
    return style


def category_colors(n: int) -> list[str]:
    """One color per class, for *n* classes, keyed by class position.

    Parameters
    ----------
    n:
        The number of classes.

    Returns
    -------
    list of str
        :data:`CATEGORY_COLORS` for up to eight classes, ``tab20`` for up to
        twenty. Class *i* gets entry *i* whichever classes a figure shows, so a
        class keeps its color across figures of one data source.

    Raises
    ------
    TypeError
        If *n* is a boolean, a float or not an integer.
    ValueError
        If *n* is negative or exceeds twenty; pass explicit colors instead.
    """
    n = as_bounded_integer(n, minimum=0, message_name="n")
    check_classes_fit_a_palette(n)
    if n <= len(CATEGORY_COLORS):
        return list(CATEGORY_COLORS[:n])
    return [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["tab20"].colors[:n]]


def use_project_style() -> None:
    """Apply :data:`RC_PARAMS` to matplotlib's global ``rcParams``.

    Call this once where a set of figures is produced. Nothing in this package
    calls it, and importing the package leaves ``rcParams`` untouched.
    """
    matplotlib.rcParams.update(RC_PARAMS)


def axis_label(field: xr.DataArray) -> str:
    """The axis label for *field*, as ``"<long_name> (<units>)"``.

    Parameters
    ----------
    field:
        A field carrying ``long_name`` and ``units`` in ``attrs``.

    Returns
    -------
    str
        For example ``"Air temperature (degC)"``.

    Raises
    ------
    TypeError
        If *field* is not a ``DataArray``.
    ValueError
        If ``long_name`` or ``units`` is missing from ``attrs``, naming which.
    """
    check_field_is_a_dataarray(field)
    check_field_has_a_label(field, message_name=message_name(field))
    return f"{field.attrs['long_name']} ({field.attrs['units']})"


# ── private helpers ───────────────────────────────────────────────────────────

#: Which of a role's keywords apply to each element. ``points`` also has
#: ``linestyle`` forced to ``"none"``, which is not taken from the role.
_ELEMENT_KEYWORDS = frozendict(
    {
        "line": ("color", "linestyle", "linewidth"),
        "band": ("color",),
        "points": ("color", "marker", "markersize"),
    }
)

#: The most classes a palette of :func:`category_colors` tells apart.
_MOST_CLASSES = 20


# ── checks ────────────────────────────────────────────────────────────────────


def check_classes_fit_a_palette(n: int) -> None:
    """*n* classes are few enough for :func:`category_colors` to tell apart."""
    if n > _MOST_CLASSES:
        raise ValueError(
            f"{n} classes are more than the palettes distinguish ({_MOST_CLASSES}); pass "
            "colors explicitly, one per class."
        )


def check_field_has_a_label(field: xr.DataArray, *, message_name: str) -> None:
    """*field* carries the ``long_name`` and ``units`` an axis label is made of."""
    missing = [name for name in ("long_name", "units") if not field.attrs.get(name)]
    if missing:
        raise ValueError(
            f"{message_name}: an axis label is made of attrs['long_name'] and "
            f"attrs['units'], and the field has no {' and no '.join(repr(m) for m in missing)}; "
            "set them, or keep them through the xarray operation that dropped them "
            "(keep_attrs=True)."
        )


def check_option_is_known(value: Any, options: Collection[str], *, message_name: str) -> None:
    """*value* is one of *options*, the strings an argument may be."""
    if value not in options:
        raise ValueError(
            f"{message_name} must be one of {list(options)}, got {value!r}; pass one of them."
        )


def check_key_is_known(
    key: Any, registry: Collection[str], *, alternatives: str = "", message_name: str
) -> None:
    """*key* names an entry of *registry*; *message_name* says what kind of entry."""
    if key not in registry:
        raise KeyError(
            f"unknown {message_name} {key!r}; pass one of {truncated(list(registry))}"
            f"{alternatives}."
        )


def check_keywords_are_not_retired(
    keywords: Mapping[str, Any], retired: Mapping[str, str], *, message_name: str
) -> None:
    """No keyword is one *message_name* has retired, which ``**kwargs`` would swallow."""
    for old, replacement in retired.items():
        if old in keywords:
            raise TypeError(f"{message_name} no longer takes {old}=; use {replacement}.")


def check_value_is_a_number(value: Any, *, message_name: str) -> None:
    """*value* is a real number, not a boolean."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{message_name} must be a number, got {type(value).__name__} {value!r}; pass a "
            "real number."
        )


def check_number_is_positive(value: float, *, message_name: str) -> None:
    """A number is positive: infinity is, ``NaN`` is not."""
    if not value > 0:
        raise ValueError(
            f"{message_name} must be positive, got {value!r}; pass a positive number."
        )


def check_number_is_finite(value: float, *, message_name: str) -> None:
    """A number is finite."""
    if not np.isfinite(value):
        raise ValueError(f"{message_name} must be finite, got {value!r}; pass a finite number.")

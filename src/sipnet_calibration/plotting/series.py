"""L2 -- time-series panels.

Overview
--------
One public function, :func:`series_panel`, draws one canonical field onto one
``Axes``. It covers a prior or posterior predictive ensemble, a driver
ensemble, an observation ensemble, a single deterministic run, and scattered
observations with error bars, because those differ in the *shape* of the field
and in the *role* it plays, not in the kind of function needed to draw them.

The panel takes ``ax`` and returns ``Axes``, never creates a figure, never
saves and never shows. Building the figure is :mod:`.facet`'s job; saving is
an experiment report's.

The sample-dim rule
-------------------
**``time`` is the x-axis, and every other dim present is a sample dim.**
``show="auto"`` resolves to ``"fan"`` when at least one sample dim is present
and ``"line"`` when none is. So

==========================  ================================================
Dims of the field           ``show="auto"`` draws
==========================  ================================================
``(time,)``                 one line
``(member, time)``          a fan over members
``(site, time)``            a fan over sites
``(member, site, time)``    a fan over the stacked member-site curves
==========================  ================================================

This generalizes the design spec's rule, which branched on ``member`` alone
and so had nothing to say about a field whose ensemble-like dimension is
``site`` -- one panel with a curve per site, which is a case this project
meets constantly. It is still a branch on the presence of a dim, not a mode
keyword.

There is deliberately no ``over=`` keyword naming the sample dim. To draw a
fan over members for one site, select the site; to draw one panel per site,
facet. The one display this does not express is a nested one -- per-site fans
over members, several sites, one panel -- which is left unbuilt rather than
paid for with a keyword.

Overlays
--------
**There is no ``obs=`` parameter.** Observations are themselves often an
ensemble (the NEE product has 25 members), so an ``obs=`` keyword would need
a rule for rendering "obs" differently from "model" that no single default
gets right. Because every plotter takes ``ax`` and returns ``Axes``, an
overlay is a second call::

    ax = series_panel(predicted, role="posterior")
    series_panel(observed, ax=ax, role="obs")

The same pattern gives truth lines and several aggregations on one panel,
with no extra machinery. An observation ensemble goes down exactly this path;
what is drawn differs only in ``role``.

Observation error, where it is a variance rather than an ensemble, is a
*second field* on the same grid -- which is what
:func:`sipnet_calibration.constraints.constraint_fields` produces -- and is
passed as ``variance=`` or ``standard_deviation=``.

Aggregation
-----------
Aggregation is not a keyword here. Callers apply
``sipnet_calibration.obs_ops.aggregate_time`` first, and it defaults to the
variable's own rule from the registry (``nee`` sums, ``air_temperature``
means)::

    series_panel(aggregate_time(nee, "1D"))     # yes
    series_panel(nee, temporal_agg="1D")        # no

That keeps a real subtlety at the call site: quantile-of-daily-sum is not
daily-sum-of-quantile, and which one is wanted is a modeling choice.

Notes
-----
**Labels come from the field's own attributes**, through
:func:`.style.axis_label`, not from the ``VARIABLES`` registry. Every adapter
in this project already writes ``units`` and ``long_name``, so the series
layer needs nothing from the registry: ``agg`` is the caller's verb and
``cmap``/``center`` are map concerns. :func:`.style.axis_label` is the one
place the registry plugs in when issue #6 lands.

**The panel does not call** ``fields.validate_field``. That checks the
canonical convention, which is an adapter's responsibility to establish once;
re-checking it inside a function a facet grid calls once per panel is both
the wrong altitude and the wrong cost. What the panel does check is its own
precondition -- that the field is a ``DataArray`` whose dims are a subset of
``(member, site, time)`` and include ``time`` -- which is a different
question and remains necessary after ``validate_field`` exists.

**Missing values pass through.** The panel does not drop, fill or interpolate
them; the primitives' rule applies, so lines and bands gap and points drop.
See :mod:`.primitives`.

Usage
-----
::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.plotting import series_panel

    tair = driver_fields(load_drivers([1, 27]))["air_temperature"]

    series_panel(tair.sel(site=1))                        # a fan over members
    series_panel(tair.sel(site=1), show="spaghetti")      # the members as lines
    series_panel(tair.isel(member=0), show="spaghetti", label_by="site")

    # An observation with its error variance, over a model panel.
    ax = series_panel(predicted_wood.sel(site=s))
    series_panel(observed_wood.sel(site=s), ax=ax, role="obs", show="points",
                 variance=wood_variance.sel(site=s))
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from matplotlib.axes import Axes

__all__ = ["SHOW_KINDS", "series_panel"]

#: What ``show`` may be. ``"auto"`` resolves by the sample-dim rule in the
#: module docstring; the rest force a rendering.
SHOW_KINDS: tuple[str, ...] = ("auto", "line", "spaghetti", "fan", "points")


def series_panel(
    field: xr.DataArray,
    ax: Axes | None = None,
    *,
    show: str = "auto",
    role: str = "posterior",
    levels: tuple[float, ...] = (0.5, 0.9),
    n_max: int = 25,
    label: str | None = None,
    label_by: str | None = None,
    variance: xr.DataArray | None = None,
    standard_deviation: xr.DataArray | None = None,
    n_sigma: float = 1.0,
    **style: Any,
) -> Axes:
    """Draw one canonical *field* against time on one ``Axes``.

    Parameters
    ----------
    field:
        A canonical field whose dims are a subset of ``(member, site, time)``
        and include ``time``. Every dim other than ``time`` is a sample dim;
        see the module docstring.
    ax:
        The axes to draw on. If ``None`` a figure and axes are created with
        ``matplotlib.pyplot.subplots`` -- the one concession to convenience,
        for interactive use. Inside a facet grid ``ax`` is always given, and
        no figure is created.
    show:
        One of :data:`SHOW_KINDS`. ``"auto"`` is ``"fan"`` when a sample dim
        is present and ``"line"`` when none is. ``"line"`` on a field with a
        sample dim, and ``"fan"`` or ``"spaghetti"`` on one without, are
        errors rather than silent reductions.
    role:
        A key of :data:`.style.ROLES`, deciding color, line style and marker.
    levels:
        Central interval widths for ``show="fan"``; see
        :func:`.primitives.fan`.
    n_max:
        The most curves ``show="spaghetti"`` draws; see
        :func:`.primitives.spaghetti`.
    label:
        The legend entry. ``None`` means the value of *role*, so the ordinary
        overlay produces a usable legend without anything being spelled out
        and :func:`.facet.facet` has labels to deduplicate. Pass matplotlib's
        ``"_nolegend_"`` to suppress the entry.
    label_by:
        The name of a coordinate on the sample dim, valid only with
        ``show="spaghetti"``. Each curve is then labeled with that
        coordinate's value and colored from :data:`.style.CURVE_COLORS`
        instead of from the role, which is what makes one panel with a curve
        per site readable. ``label`` is ignored when this is given.
    variance, standard_deviation:
        The observation error, as a second canonical field aligned with
        *field*, valid only with ``show="points"``. At most one may be given.
        Two keywords rather than one because the call site then says which
        quantity it handed over, and the square root happens in one tested
        place rather than at every call site.
    n_sigma:
        Multiplies the standard deviation to give the error bar's half-length.
        The default draws one standard deviation either side.
    **style:
        Passed through to the primitive, overriding the role's keywords.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on -- *ax* itself when it was given. The y-axis label
        is set from :func:`.style.axis_label`; the x-axis is left to
        matplotlib's date handling and no title is set, because a title is the
        facet grid's to give.

    Raises
    ------
    ValueError
        If *field* is not a ``DataArray``; if its dims are not a subset of
        ``(member, site, time)``, or do not include ``time`` -- the message
        names the dims found and points at selecting, aggregating, or
        :mod:`.maps`; if *show* is not in :data:`SHOW_KINDS` or does not match
        the field's shape; if *label_by* is given without
        ``show="spaghetti"``, or names a coordinate that is not on a sample
        dim; if both *variance* and *standard_deviation* are given, or either
        is given without ``show="points"``, or does not align with *field*; if
        a variance holds a negative value; or if *n_sigma* is not finite and
        positive.

    Notes
    -----
    ``show="fan"`` also draws the median as a line, and the legend entry goes
    on that line rather than on a band. A fan without a central curve is hard
    to read, and the median is the part of the summary a reader looks for
    first.

    Quantiles are taken with ``NaN`` ignored and over all sample dims at once,
    so a field with both ``member`` and ``site`` is summarized over the
    stacked set of curves rather than in two stages. That matters: a quantile
    of quantiles is not a quantile.
    """
    raise NotImplementedError

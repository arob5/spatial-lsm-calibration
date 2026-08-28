"""Plotting suite: matplotlib, ensemble- and site-aware.

Five layers, dependencies strictly upward:

* **L1** :mod:`primitives` -- signature always ``(ax, plain numpy, **style) -> artist``.
  No pandas, no xarray, no figure creation.
* **L2** :mod:`series`, :mod:`maps` -- one variable, one Axes, canonical field in.
* **L3** :mod:`facet` -- the one generic facet function; owns figure and axes
  construction, shared limits, legend de-duplication.
* **L4** experiment reports -- in ``experiments/<task>/plots.py``, *never* here.
  Anything that knows a task name is not core.
* **L5** :mod:`diagnostics` -- inference diagnostics (EKI history, marginals,
  coverage, per-site parameter maps).

Invariants, for every plotter in this package:

* takes ``ax``, returns ``Axes``;
* never calls ``plt.show()`` or ``savefig``, and never creates a figure
  implicitly (that is :mod:`facet`'s job);
* never accepts a ``SIPNETResult``, a DataFrame, or a path (that is an adapter's
  job, in :mod:`sipnet_calibration.fields`);
* takes style from :mod:`registry` and :mod:`style`, not from a dozen keywords.

Interactive single-run inspection is out of scope: ``pysipnet.viz.dashboard``
already owns it.
"""

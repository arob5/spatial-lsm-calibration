"""L5 -- inference diagnostics.

Planned:

* EKI history -- the stacked ``HistoryRecord`` fields versus step: beta ladder,
  misfit mean/min/max, center misfit, spread, ESS, ``n_valid``.
* Prior-versus-posterior marginals, and pairs.
* Rank / coverage checks.
* **Per-site parameters as a map** -- :func:`sipnet_calibration.plotting.maps.map_panel`
  reused on parameter space instead of output space. This is the payoff of the
  hierarchical model and the reason the map layer must not assume model output.

EKI hands back flat blocks that know nothing about space or time: ``(J, P)``
parameter ensembles and ``(J, N)`` predictions. The ``(site, variable, time)``
MultiIndex from pyEKI's ``index`` layer is what unstacks them into canonical
fields; see :mod:`sipnet_calibration.fields`.
"""

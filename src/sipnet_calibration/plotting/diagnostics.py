"""L5 -- inference diagnostics.

Planned:

* EKI history -- the stacked ``HistoryRecord`` fields versus step: beta ladder,
  misfit mean/min/max, center misfit, spread, ESS, ``n_valid``.
* Prior-versus-posterior marginals, and pairs.
* Rank / coverage checks.
* **Per-site parameters as a map** -- :func:`sipnet_calibration.plotting.maps.map_panel`
  reused on parameter space instead of output space. This is the payoff of the
  hierarchical model and the reason the map layer must not assume model output.

EKI hands back flat blocks that know nothing about space or time: ``(J, D)``
parameter ensembles and ``(J, N)`` predictions. A ``(J, N)`` block is unstacked
by :meth:`sipnet_calibration.observation.ObservationVector.fields`, which owns
the ``(site, product, time)`` index it was flattened with; a ``(J, D)`` block
by :meth:`sipnet_calibration.parameter_vector.ParameterVector.fields`.
"""

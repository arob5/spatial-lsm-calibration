"""L5 -- inference diagnostics.

Planned:

* EKI history -- the stacked ``HistoryRecord`` fields versus step: beta ladder,
  misfit mean/min/max, center misfit, spread, ESS, ``n_valid``.
* Prior-versus-posterior marginals, and pairs.
* Rank / coverage checks.
* **Per-site parameters as a map** -- :func:`sipnet_calibration.plotting.maps.map_panel`
  reused on parameter space instead of output space. This is the payoff of the
  hierarchical model and the reason the map layer must not assume model output.

EKI hands back batched Flat that knows nothing about space or time: ``(J, D)``
parameter ensembles and ``(J, N)`` predictions. A ``(J, D)`` one is labeled by
:meth:`sipnet_calibration.probability.Posterior.to_labeled`, and read at the
sites by :meth:`sipnet_calibration.site_dims.SiteDims.site_fields`, which is
what a map reads; predictions on the observation dims become fields through
:meth:`sipnet_calibration.observation.ObservationVector.to_fields`.
"""

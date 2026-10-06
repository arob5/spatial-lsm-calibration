"""The experiment's entry points, each run as ``python -m`` from the repository root.

Every public module here has a ``main``; the underscore modules are the
machinery they share. In the order a calibration runs them:

- setup: ``prepare_drivers`` (the corrected driver file), ``check_inputs``
  (everything is found and builds);
- ``prior_predictive``: the prior's draws through the model, then their
  figures and diagnosis;
- ``eki``: inference, then the ladder's figures;
- ``posterior_predictive``: an inference run's final ensemble through the
  model, then its figures and the run's diagnosis;
- ``compare_setups``: the EKI setups' figures side by side.

And for a stored run: ``diagnose`` rediagnoses it under ``config``'s current
noise model, ``draw_figures`` redraws its figures, and
``fit_nee_discrepancy`` fits NEE's discrepancy to its residuals.
"""

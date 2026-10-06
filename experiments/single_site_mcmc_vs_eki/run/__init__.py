"""The experiment's entry points, each run as ``python -m`` from the repository root.

Every public module here has a ``main``; the underscore modules are the
machinery they share. In the order a calibration runs them:

- setup: ``prepare_drivers`` (the corrected driver file), ``check_inputs``
  (everything is found and builds);
- ``prior_predictive``: the prior's draws through the model, then their
  figures and diagnosis;
- ``calibrate``: one algorithm on one model, written as a run;
- ``predict``: a run's samples through the model, its posterior predictive;
- ``diagnose``: a run's predictive check and NEE residual diagnostics;
- ``compare``: the cross-run tables.

And for a diagnosed run, ``fit_nee_discrepancy`` fits NEE's discrepancy to
its residuals.
"""

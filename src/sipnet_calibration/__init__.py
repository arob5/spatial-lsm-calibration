"""Shared code for spatial SIPNET parameter calibration.

Layout follows the design in ``logs/2026-08-28_Plotting Design Spec.md``
(Obsidian vault) and, for the observation layer, in
``logs/2026-09-15_Observation Operators and Observed Variables Design Spec.md``.
The data layer -- :mod:`sites`, :mod:`projection`, :mod:`fields`,
:mod:`time_conventions`, :mod:`observation_operators` -- is independent of
:mod:`plotting`; plotting depends on it, never the reverse.
"""

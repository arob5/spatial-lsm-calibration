"""Shared code for spatial SIPNET parameter calibration.

Layout follows the design in ``logs/2026-08-28_Plotting Design Spec.md``
(Obsidian vault). Data layer (:mod:`sites`, :mod:`projection`, :mod:`fields`,
:mod:`obs_ops`) is
independent of :mod:`plotting`; plotting depends on it, never the reverse.
:mod:`parameter_vector` defines the calibration vector -- calibration
parameters, their priors and bijectors, and the SIPNET maps to pySIPNET
parameters -- and depends on neither.
"""

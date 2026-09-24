"""Shared code for spatial SIPNET parameter calibration.

Data layer (:mod:`sites`, :mod:`projection`, :mod:`fields`,
:mod:`obs_ops`) is
independent of :mod:`plotting`; plotting depends on it, never the reverse.
:mod:`parameter_vector` defines the calibration vector -- calibration
parameters, their priors and bijectors, and the SIPNET maps to pySIPNET
parameters -- and depends on neither.
"""

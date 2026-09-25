"""The observation side of the inverse problem.

Two abstractions and a handful of functions:

* :class:`ObservationOperator` (a protocol) and the shipped operators
  :class:`SelectTimestep`, :class:`ReduceOverTimeBounds`,
  :class:`ReduceOverRun`, :class:`ComputeLeafAreaIndex`, with
  :data:`DEFAULT_OBS_OPS` binding a product to its default operator where the
  construction is documented.
* :class:`Observation` and :class:`ObservationVector`: the observed cells of
  an experiment in a fixed order, with Fields and Flat representations,
  ``y``, ``index``, ``positions`` and ``predict``.

The verbs the operators are written with: temporal alignment in
:mod:`~sipnet_calibration.observation.alignment` (``aggregate_time``,
``reduce_windows``, ``select_timestep_at``, the window builders and the
counts) and attribute-carrying arithmetic in
:mod:`~sipnet_calibration.observation.units` (``multiply``, ``divide``,
``add``, ``subtract``, ``step_length``). Unit conversion is pySIPNET's
:func:`pysipnet.units.convert_dataarray_units`, which :meth:`ObservationVector.predict`
applies.

The error model and the likelihood are not here; they belong to the
inference layer, which reads ``y``, ``index`` and ``positions`` off the
vector.
"""

from sipnet_calibration.observation.alignment import (
    DEFAULT_METHOD_FOR_KIND,
    RESAMPLING_METHODS,
    WINDOW_REDUCTIONS,
    aggregate_time,
    aggregation_counts,
    reduce_windows,
    run_window,
    select_timestep_at,
    window_counts,
    windows_from_time_bounds,
)
from sipnet_calibration.observation.operators import (
    DEFAULT_OBS_OPS,
    ComputeLeafAreaIndex,
    ObservationOperator,
    ReduceOverRun,
    ReduceOverTimeBounds,
    SelectTimestep,
    check_operator,
    extract_sipnet_parameter_at_coords,
    select_observed_sites,
    sipnet_parameter_spec,
)
from sipnet_calibration.observation.units import add, divide, multiply, step_length, subtract
from sipnet_calibration.observation.vector import INDEX_LEVELS, Observation, ObservationVector

__all__ = [
    "ComputeLeafAreaIndex",
    "DEFAULT_METHOD_FOR_KIND",
    "DEFAULT_OBS_OPS",
    "INDEX_LEVELS",
    "Observation",
    "ObservationOperator",
    "ObservationVector",
    "RESAMPLING_METHODS",
    "ReduceOverRun",
    "ReduceOverTimeBounds",
    "SelectTimestep",
    "WINDOW_REDUCTIONS",
    "add",
    "aggregate_time",
    "aggregation_counts",
    "check_operator",
    "divide",
    "extract_sipnet_parameter_at_coords",
    "multiply",
    "reduce_windows",
    "run_window",
    "select_observed_sites",
    "select_timestep_at",
    "sipnet_parameter_spec",
    "step_length",
    "subtract",
    "window_counts",
    "windows_from_time_bounds",
]

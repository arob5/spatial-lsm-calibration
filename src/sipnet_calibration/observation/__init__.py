"""The observation side of the inverse problem.

Two abstractions and a handful of functions:

* :class:`ObservationOperator` (a protocol) and the shipped operators
  :class:`SelectTimestep`, :class:`ReduceOverTimeBounds`,
  :class:`ReduceOverRun`, :class:`ComputeLeafAreaIndex`, with
  :data:`DEFAULT_OBS_OPS` binding a product to its default operator where the
  construction is established from a primary source. An operator is written
  with :func:`select_observed_sites`, which restricts the model output to the
  observed sites, and :func:`extract_sipnet_parameter_at_coords`, which lines
  a SIPNET parameter's values up with it; :func:`check_operator` checks one
  against the contract, through the same checks the vector applies
  (:func:`check_operator_declares_names`,
  :func:`check_model_output_carries_what_is_read`,
  :func:`check_result_is_on_the_observation_grid`).
* :class:`Observation` and :class:`ObservationVector`: the observed cells of
  an experiment in a fixed order, with Fields and Flat representations,
  ``y``, ``index`` (levels :data:`INDEX_LEVELS`), ``positions`` and
  ``predict``.

The verbs the operators are written with: temporal alignment in
:mod:`~sipnet_calibration.observation.time_alignment` (``aggregate_time``,
``reduce_windows``, ``select_timestep_at``, the window builders and the
counts). Arithmetic that keeps ``units``, ``constituent`` and ``kind`` true
is pySIPNET's :mod:`pysipnet.arithmetic` (``divide_with_units``,
``step_length`` and the rest), and a SIPNET parameter's values are labeled by
pySIPNET's :func:`~pysipnet.parameters.model.parameter_dataarray`. Unit
conversion is pySIPNET's :func:`pysipnet.units.convert_dataarray_units`,
which :meth:`ObservationVector.predict` applies.

The error model and the likelihood are not here; they belong to the
inference layer, which reads ``y``, ``index`` and ``positions`` off the
vector.
"""

from sipnet_calibration.observation.operators import (
    DEFAULT_OBS_OPS,
    ComputeLeafAreaIndex,
    ObservationOperator,
    ReduceOverRun,
    ReduceOverTimeBounds,
    SelectTimestep,
    check_model_output_carries_what_is_read,
    check_operator,
    check_operator_declares_names,
    check_result_is_on_the_observation_grid,
    extract_sipnet_parameter_at_coords,
    select_observed_sites,
)
from sipnet_calibration.observation.time_alignment import (
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
from sipnet_calibration.observation.vector import (
    INDEX_LEVELS,
    Observation,
    ObservationVector,
)

__all__ = [
    "DEFAULT_METHOD_FOR_KIND",
    "DEFAULT_OBS_OPS",
    "INDEX_LEVELS",
    "RESAMPLING_METHODS",
    "WINDOW_REDUCTIONS",
    "ComputeLeafAreaIndex",
    "Observation",
    "ObservationOperator",
    "ObservationVector",
    "ReduceOverRun",
    "ReduceOverTimeBounds",
    "SelectTimestep",
    "aggregate_time",
    "aggregation_counts",
    "check_model_output_carries_what_is_read",
    "check_operator",
    "check_operator_declares_names",
    "check_result_is_on_the_observation_grid",
    "extract_sipnet_parameter_at_coords",
    "reduce_windows",
    "run_window",
    "select_observed_sites",
    "select_timestep_at",
    "window_counts",
    "windows_from_time_bounds",
]

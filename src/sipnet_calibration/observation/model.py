"""The observation vector as components of a probabilistic model.

Where this sits
---------------
::

    observation.ObservationVector      sources, observation dims, constants
      -> observation.model             this module: their components
      -> probability.joint(...)        a model over them, bound at
                                       ``observation_vector.coords``

It is the part of :mod:`sipnet_calibration.observation` that needs the
probability layer, and so TFP: ``observation/__init__`` does not import it,
so the data sources and the PyEns workers, which import the observation
package, load neither.

What it reads
-------------
An :class:`~sipnet_calibration.observation.vector.ObservationVector`.

Data model
----------
For each observation source ``k``, in the vector's order, two
:class:`~sipnet_calibration.probability.spec.ArraySpec`\\ s, each a scalar
on :data:`~sipnet_calibration.probability.support.REAL`, indexed by the
source's observation dim (``observation_vector.observation_dim_name(k)``)
and in the observed values' ``units``:

- its **observed component**, named for the source, whose value
  ``observation_vector.observed_values_by_component()`` holds;
- its **prediction**, ``observation_vector.prediction_name(k)``: the forward
  model's value of the observed quantity, which a simulator outputs.

The observed values' ``constituent`` is not carried, since an
``ArraySpec`` has none: a prediction is converted into the observed values'
units, constituent included, before it is one.

Functions
---------
:func:`observed_components`, :func:`prediction_components`
    The two, one per source.

Usage
-----
::

    from sipnet_calibration.observation.model import observed_components, prediction_components

    observed_components(observation_vector)    # (ArraySpec("modis_leaf_area_index", ...), ...)
    prediction_components(observation_vector)  # (ArraySpec("predicted_modis_leaf_area_index", ...), ...)
"""

from __future__ import annotations

from sipnet_calibration.observation.vector import ObservationVector
from sipnet_calibration.probability import REAL, ArraySpec

__all__ = [
    "observed_components",
    "prediction_components",
]


def observed_components(observation_vector: ObservationVector) -> tuple[ArraySpec, ...]:
    """The observed components, one per source, in the vector's order, as
    the module docstring's data model has them."""
    return tuple(
        _component(observation_vector, name, name) for name in observation_vector.observation_source_names
    )


def prediction_components(observation_vector: ObservationVector) -> tuple[ArraySpec, ...]:
    """The predictions, one per source, in the vector's order: a forward
    model's outputs, as the module docstring's data model has them."""
    return tuple(
        _component(observation_vector, name, observation_vector.prediction_name(name))
        for name in observation_vector.observation_source_names
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _component(observation_vector: ObservationVector, observation_source_name: str, name: str) -> ArraySpec:
    """A scalar component on the source's observation dim, in its units."""
    return ArraySpec(
        name,
        units=observation_vector[observation_source_name].observed_values.attrs["units"],
        support=REAL,
        indexed_by=(observation_vector.observation_dim_name(observation_source_name),),
    )

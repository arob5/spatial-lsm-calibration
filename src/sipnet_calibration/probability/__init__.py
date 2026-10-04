"""The probability layer: models declared in specs, bound to labels, and
conditioned on data, independent of the rest of the package.

It is being built beside :mod:`sipnet_calibration.parameters`, which it
replaces; until then the parameter layer reads its supports, coercion and
probe points from here.

Where this sits
---------------
::

    probability.support     Support, Interval, Simplex, PositiveDefinite; the default bijectors
    probability.names       the names the layer reserves
    probability.labels      coords (stacked dims too), constants, label maps
      -> probability.spec      ArraySpec: one component's declaration and T
      -> probability.layout    Layout: named arrays as one flat vector, and its forms
    ──────── seam: labeled values, a dict of DataArrays ────────
      -> the adapter layer

No module here imports from ``sipnet_calibration`` outside ``probability``,
which ``tests/test_package.py`` enforces. It computes in ``float64``, which
importing the package turns on.

Modules
-------
:mod:`~sipnet_calibration.probability.support`
    The sets values lie in, and their default bijections.
:mod:`~sipnet_calibration.probability.names`
    The names the layer gives things, which nothing else may take.
:mod:`~sipnet_calibration.probability.labels`
    Coords, and the constants and label maps functions read: what each is,
    and how it is read at the labels in use.
:mod:`~sipnet_calibration.probability.spec`
    One component: the shape and labels of one value, the dims it is
    indexed by, its support and transform.
:mod:`~sipnet_calibration.probability.layout`
    The layout: Flat, values by name and labeled values, the two spaces,
    selection, and labeled values as one netCDF-ready Dataset.
"""

from sipnet_calibration.probability.layout import (
    LabeledValues,
    Layout,
    ValuesByName,
    decode_labeled_values,
    encode_labeled_values,
    validate_labeled_values,
    validate_values_by_name,
)
from sipnet_calibration.probability.names import RESERVED_NAMES, SAMPLE
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import (
    DEFAULT_BIJECTORS,
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
    PositiveDefinite,
    Simplex,
    Support,
    bijector_for,
)

__all__ = [
    "DEFAULT_BIJECTORS",
    "NON_NEGATIVE",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "POSITIVE_DEFINITE",
    "REAL",
    "RESERVED_NAMES",
    "SAMPLE",
    "SIMPLEX",
    "UNIT_INTERVAL",
    "ArraySpec",
    "Interval",
    "LabeledValues",
    "Layout",
    "PositiveDefinite",
    "Simplex",
    "Support",
    "ValuesByName",
    "bijector_for",
    "decode_labeled_values",
    "encode_labeled_values",
    "validate_labeled_values",
    "validate_values_by_name",
]

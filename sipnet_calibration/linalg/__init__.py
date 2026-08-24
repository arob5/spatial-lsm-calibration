"""Structured linear operators for Gaussian conditioning.

Operators represent matrices implicitly, by how they act on vectors, so that
known structure can be exploited instead of storing or factorizing dense
arrays.

- :mod:`~sipnet_calibration.linalg.base` defines the class hierarchy, the
  array-shape convention, and how to add a new operator.
- :mod:`~sipnet_calibration.linalg.leaves` holds operators defined by their
  own arrays.
- :mod:`~sipnet_calibration.linalg.composite` holds operators built from
  other operators.
- :mod:`~sipnet_calibration.linalg.testing` holds conformance checks for new
  operator types.

Import :mod:`sipnet_calibration` before creating any array, so that float64 is
enabled first.
"""
from .base import (
    LinOp,
    PSDOperator,
    SquareLinOp,
    UnsupportedOp,
    dense_matvec,
    densify,
    operator,
    static_field,
    tri_solve,
)
from .composite import BlockDiag, BlockDiagGeneral, HStack, Product
from .leaves import Dense, DensePSD, Diagonal, Identity, ScaledIdentity, Triangular

__all__ = [
    # base
    "LinOp",
    "SquareLinOp",
    "PSDOperator",
    "UnsupportedOp",
    "densify",
    "operator",
    "static_field",
    "dense_matvec",
    "tri_solve",
    # leaves
    "Identity",
    "ScaledIdentity",
    "Diagonal",
    "Dense",
    "Triangular",
    "DensePSD",
    # composites
    "Product",
    "HStack",
    "BlockDiag",
    "BlockDiagGeneral",
]

"""Structured linear operators for Gaussian conditioning.

Import order matters only in that :mod:`sipnet_calibration` must be imported
first, so that float64 is enabled before any array is created.
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

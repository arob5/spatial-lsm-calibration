"""The linear algebra the Gaussian laws are built on: one shim over pyEKI.
Private to the probability layer.

The covariance specs evaluate to structured operators, and a Gaussian law
holds one. Every operator and the ``Gaussian`` the layer uses come from
here, so the move from today's ``pyeki.linalg`` and ``pyeki.gauss`` to
EnsKit's ``linalg`` and ``distribution`` changes this file alone. Nothing
else in the package imports pyEKI for the probability layer.

The operators are pyEKI's, whose contract is its "Linear operator contract":
they hold their arrays as pytree leaves, so one is built and used inside a
JAX trace; a value precondition (a positive diagonal, a positive-definite
matrix) is not checked outside pyEKI's debug mode, and a violated one gives
``NaN`` or ``inf`` downstream, which the layer reads as no density.
"""

from __future__ import annotations

from pyeki.gauss import Gaussian
from pyeki.linalg import (
    DensePSD,
    PSDBlockDiag,
    PSDDiagonal,
    PSDLinOp,
    PSDScaled,
    UnsupportedOpError,
    block_diag,
)

__all__ = [
    "DensePSD",
    "Gaussian",
    "PSDBlockDiag",
    "PSDDiagonal",
    "PSDLinOp",
    "PSDScaled",
    "UnsupportedOpError",
    "block_diag",
]

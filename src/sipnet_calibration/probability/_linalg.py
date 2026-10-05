"""The linear algebra the Gaussian laws are built on: one shim over EnsKit.
Private to the probability layer.

The covariance specs evaluate to structured operators, and a Gaussian law
holds one. Every operator and the ``Gaussian`` the layer uses come from
here, from EnsKit's ``linalg`` and ``distribution``. Nothing else in the
package imports EnsKit for the probability layer.

The operators are EnsKit's, whose contract is its "Linear operator
contract": they hold their arrays as pytree leaves, so one is built and used
inside a JAX trace; a value precondition (a positive diagonal, a
positive-definite matrix) is not checked outside EnsKit's debug mode, and a
violated one gives ``NaN`` or ``inf`` downstream, which the layer reads as no
density. The ``Gaussian`` is EnsKit's over named blocks; the layer's
Gaussian laws hold one of one block.
"""

from __future__ import annotations

from enskit.distribution import Gaussian
from enskit.linalg import (
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

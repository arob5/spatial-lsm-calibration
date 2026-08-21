"""Composite operators: the ones built out of other operators.

``Product`` and ``HStack`` exist because ``PSDOperator.factor()`` is a
homomorphism into the *general* operator algebra, not a leaf operation:

    factor(A + B)   =  [ L_A  L_B ]        -> HStack
    factor(D A D)   =  D L_A               -> Product
    factor(A (x) B) =  L_A (x) L_B         -> rectangular Kron  (not yet)

so making ``factor`` first-class commits to that codomain existing.

Note both are ``LinOp``, not ``PSDOperator``: a general product of three
operators is not PSD, and PSD-ness is not tracked through composition here.
That is why a symmetric congruence ``diag(d) A diag(d)`` will need its own type
rather than being sugar over ``Product`` -- it is the one that knows the result
is still PSD.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .base import LinOp, PSDOperator, operator

__all__ = ["Product", "HStack", "BlockDiag"]


@operator
class Product(LinOp):
    """``ops[0] @ ops[1] @ ... @ ops[-1]``, applied right to left."""

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        if not self.ops:
            raise ValueError("Product needs at least one operator")
        for a, b in zip(self.ops[:-1], self.ops[1:]):
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"shape mismatch in Product: {a.shape} @ {b.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], self.ops[-1].shape[1])

    def matvec(self, x: Array) -> Array:
        for op in reversed(self.ops):
            x = op.matvec(x)
        return x

    def to_dense(self) -> Array:
        out = self.ops[-1].to_dense()
        for op in reversed(self.ops[:-1]):
            out = op.to_dense() @ out
        return out


@operator
class HStack(LinOp):
    """``[A_1  A_2  ...  A_m]`` -- blocks side by side, sharing a row count.

    Note this is *not* "apply each block to the whole input and concatenate the
    outputs" (that is a block *column*). Here the input is split along its
    trailing axis and the blocks' outputs are summed:
    ``[A_1 A_2] @ [x_1; x_2] = A_1 x_1 + A_2 x_2``.
    """

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        if not self.ops:
            raise ValueError("HStack needs at least one operator")
        rows = {op.shape[0] for op in self.ops}
        if len(rows) != 1:
            raise ValueError(
                f"HStack blocks must share a row count, got {[o.shape for o in self.ops]}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], sum(op.shape[1] for op in self.ops))

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for op in self.ops[:-1]:
            acc += op.shape[1]
            out.append(acc)
        return out

    def matvec(self, x: Array) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        total = self.ops[0].matvec(chunks[0])
        for op, chunk in zip(self.ops[1:], chunks[1:]):
            total = total + op.matvec(chunk)
        return total

    def to_dense(self) -> Array:
        return jnp.concatenate([op.to_dense() for op in self.ops], axis=-1)


@operator
class BlockDiag(PSDOperator):
    """``blockdiag(blocks)`` of PSD blocks -- the shape observation noise takes.

    Capabilities are *conditional on the children*: this can only ``solve`` if
    every block can. That is why :meth:`supports` is a method rather than a
    class attribute.
    """

    blocks: tuple[PSDOperator, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("BlockDiag needs at least one block")
        for b in self.blocks:
            if not isinstance(b, PSDOperator):
                raise TypeError(
                    f"BlockDiag blocks must be PSDOperator, got {type(b).__name__}"
                )

    @property
    def shape(self) -> tuple[int, int]:
        n = sum(b.shape[0] for b in self.blocks)
        return (n, n)

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[0]
            out.append(acc)
        return out

    def supports(self, name: str) -> bool:
        return super().supports(name) and all(b.supports(name) for b in self.blocks)

    def _blockwise(self, x: Array, method: str) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        return jnp.concatenate(
            [getattr(b, method)(c) for b, c in zip(self.blocks, chunks)], axis=-1
        )

    def matvec(self, x: Array) -> Array:
        return self._blockwise(x, "matvec")

    def solve(self, b: Array) -> Array:
        return self._blockwise(b, "solve")

    def whiten(self, x: Array) -> Array:
        return self._blockwise(x, "whiten")

    def factor(self) -> LinOp:
        return BlockDiagGeneral(tuple(b.factor() for b in self.blocks))

    def cholesky(self) -> LinOp:
        return BlockDiagGeneral(tuple(b.cholesky() for b in self.blocks))

    def diag(self) -> Array:
        return jnp.concatenate([b.diag() for b in self.blocks], axis=-1)

    def logdet(self) -> Array:
        return sum(b.logdet() for b in self.blocks)

    def to_dense(self) -> Array:
        return _dense_block_diag([b.to_dense() for b in self.blocks])


@operator
class BlockDiagGeneral(LinOp):
    """Block diagonal of arbitrary (possibly rectangular) blocks.

    What ``BlockDiag.factor()`` returns: the blocks' factors need not be square,
    so the result is a general ``LinOp``.
    """

    blocks: tuple[LinOp, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            sum(b.shape[0] for b in self.blocks),
            sum(b.shape[1] for b in self.blocks),
        )

    def matvec(self, x: Array) -> Array:
        splits, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[1]
            splits.append(acc)
        chunks = jnp.split(x, splits, axis=-1)
        return jnp.concatenate(
            [b.matvec(c) for b, c in zip(self.blocks, chunks)], axis=-1
        )

    def solve(self, b: Array) -> Array:  # pragma: no cover - not a SquareLinOp
        raise AttributeError("BlockDiagGeneral is not square; use BlockDiag")

    def to_dense(self) -> Array:
        return _dense_block_diag([b.to_dense() for b in self.blocks])


def _dense_block_diag(mats: list[Array]) -> Array:
    """Dense block diagonal, built independently of any ``matvec``."""
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = jnp.zeros((rows, cols), dtype=mats[0].dtype)
    r = c = 0
    for m in mats:
        out = out.at[r : r + m.shape[0], c : c + m.shape[1]].set(m)
        r, c = r + m.shape[0], c + m.shape[1]
    return out

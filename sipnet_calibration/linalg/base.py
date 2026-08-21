"""Structured positive-definite operators: the class hierarchy and its contract.

Three levels, split by what is actually well defined at each:

    LinOp         matvec, matmat, to_dense           (may be rectangular)
      |
    SquareLinOp   + solve, solve_mat, logdet, diag
      |
    PSDOperator   + factor, cholesky, whiten

The split is load-bearing rather than cosmetic. ``PSDOperator.factor()`` returns
the square root *as an operator*, and square roots are rectangular and not
self-adjoint -- so they belong in ``LinOp``, where they cannot advertise a
meaningless ``solve``. See the design log
``2026-08-20_Core Inference Primitives Design Spec`` for the reasoning.

Conventions fixed here and relied on everywhere downstream:

**Batch axes lead; the core operand shape trails.** This is the NumPy
generalized-ufunc rule, and it is what ``vmap`` produces by default.

===============  ====================  ==================================
method           core operand          batched signature
===============  ====================  ==================================
``matvec``       vector ``(n_in,)``    ``(..., n_in) -> (..., n_out)``
``matmat``       matrix ``(n_in, k)``  ``(..., n_in, k) -> (..., n_out, k)``
``solve``        vector ``(n,)``       ``(..., n) -> (..., n)``
``solve_mat``    matrix ``(n, k)``     ``(..., n, k) -> (..., n, k)``
===============  ====================  ==================================

The ``k`` in ``matmat`` is *not* a batch axis -- it is the column count of one
matrix operand, part of the core shape. ``matvec`` and ``matmat`` are therefore
separate methods that never inspect ``ndim``; NumPy overloaded exactly this in
``linalg.solve`` and had to change the rule in 2.0.

**Children must contract their trailing axis.** Composite operators call child
``matvec`` on arrays with leading batch axes. The obvious dense implementation
``self.M @ x`` violates this -- it contracts the *second-to-last* axis when
``ndim >= 2`` -- which produces a wrong answer with no error whenever the
operator is square. Use ``dense_matvec`` below.

**Unsupported operations raise; they never silently densify.** A base method
that falls back to ``to_dense() @ x`` turns an O(n^3) mistake into an invisible
one. Densification is an explicit call, :func:`densify`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "LinOp",
    "SquareLinOp",
    "PSDOperator",
    "UnsupportedOp",
    "densify",
    "operator",
    "static_field",
    "dense_matvec",
    "tri_solve",
]


class UnsupportedOp(NotImplementedError):
    """Raised when an operator has no cheap implementation of an operation.

    Deliberately *not* a silent fall back to dense linear algebra. If a dense
    result is genuinely wanted, ask for it: ``densify(op).solve(b)``.
    """

    def __init__(self, name: str, op: "LinOp") -> None:
        cls = type(op).__name__
        have = ", ".join(sorted(op.capabilities())) or "none"
        super().__init__(
            f"{cls} has no cheap `{name}`. It supports: {have}. "
            f"Use densify(op) for an explicit dense fallback."
        )
        self.name, self.operator_type = name, cls


# ---------------------------------------------------------------------------
# dataclass / pytree plumbing
# ---------------------------------------------------------------------------

# Field types that are almost always meant to be static. Leaving one of these
# as a pytree child means it arrives as a tracer under `jit`, and any use of it
# in a shape expression dies with "Shapes must be 1D sequences of concrete
# values" -- far from the declaration that caused it.
_SCALARISH = {"int", "float", "bool", "str", "NoneType", "None"}
_CONTAINERS = {"tuple", "list", "dict", "set", "frozenset"}


def _is_likely_static(ann: Any) -> bool:
    """Heuristic: would this annotation be a mistake as a pytree child?

    Must handle **string** annotations, because any module with
    ``from __future__ import annotations`` -- which is to say, all of them --
    hands ``dataclasses.fields`` the annotation as text rather than a type. An
    earlier version only checked ``isinstance(ann, type)`` and therefore never
    fired at all; a test caught it.

    Containers are judged by their parameters, so ``tuple[int, ...]`` is static
    but ``tuple[LinOp, ...]`` is a child -- the latter holds arrays.
    """
    if isinstance(ann, type):
        return ann.__name__ in _SCALARISH or ann.__name__ in _CONTAINERS
    if not isinstance(ann, str):
        return False
    head, _, rest = ann.strip().partition("[")
    head = head.strip().split(".")[-1]
    if head in _SCALARISH:
        return True
    if head in _CONTAINERS:
        parts = [
            p.strip().split(".")[-1]
            for p in rest.rstrip("]").split(",")
            if p.strip() not in ("", "...")
        ]
        return all(p in _SCALARISH for p in parts) if parts else True
    return False


def static_field(**kwargs: Any):
    """Declare a dataclass field as pytree *metadata* rather than a child."""
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def operator(cls: type) -> type:
    """Make ``cls`` a frozen dataclass registered as a JAX pytree.

    Fields marked with :func:`static_field` become ``meta_fields``; everything
    else becomes a child. Two guards, both for hazards that otherwise surface
    far from their cause:

    * A field annotated with a scalar/container type that is *not* marked
      static raises at class-definition time. Such a field would silently
      become a tracer under ``jit``.
    * ``eq=False``, so operators use identity equality. Dataclass ``__eq__``
      would compare arrays elementwise and raise "truth value ... is
      ambiguous"; the generated ``__hash__`` would raise on ``ArrayImpl``.
      Operators are always *traced* arguments, never ``static_argnums``.
    """
    cls = dataclass(frozen=True, eq=False)(cls)

    data_fields, meta_fields = [], []
    for f in dataclasses.fields(cls):
        if f.metadata.get("static", False):
            meta_fields.append(f.name)
            continue
        if _is_likely_static(f.type):
            name = f.type if isinstance(f.type, str) else f.type.__name__
            raise TypeError(
                f"{cls.__name__}.{f.name}: `{name}` field is not marked static, "
                f"so it would become a pytree child and arrive as a tracer under "
                f"jit. Use static_field(), or store it as an array."
            )
        data_fields.append(f.name)

    jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )
    return cls


# ---------------------------------------------------------------------------
# array helpers honouring the batch contract
# ---------------------------------------------------------------------------

def dense_matvec(M: Array, x: Array) -> Array:
    """``M @ x`` contracting the **trailing** axis of ``x``.

    ``M @ x`` is wrong here: for ``x.ndim >= 2`` it contracts the second-to-last
    axis. When the operator is square that is a wrong answer with no error.
    """
    return jnp.einsum("ij,...j->...i", M, x)


def tri_solve(L: Array, x: Array, *, lower: bool, trans: int = 0) -> Array:
    """Triangular solve contracting the trailing axis, with leading batch axes."""
    flat = x.reshape(-1, x.shape[-1]).T                     # (n, m)
    out = jax.scipy.linalg.solve_triangular(L, flat, lower=lower, trans=trans)
    return out.T.reshape(x.shape)


# ---------------------------------------------------------------------------
# level 1 -- LinOp
# ---------------------------------------------------------------------------

_OPTIONAL = ("solve", "solve_mat", "logdet", "diag", "factor", "cholesky", "whiten")


class LinOp:
    """A linear operator, possibly rectangular. Contract in the module docstring."""

    #: Set by subclasses that want to *withdraw* a capability their base provides
    #: (rare). Normal types simply override the methods they support.
    _WITHDRAWN: ClassVar[frozenset[str]] = frozenset()

    # -- required -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """``(n_out, n_in)``. A property, not a field, so it never becomes a tracer."""
        raise NotImplementedError(f"{type(self).__name__} must define `shape`")

    def matvec(self, x: Array) -> Array:
        raise NotImplementedError(f"{type(self).__name__} must define `matvec`")

    def to_dense(self) -> Array:
        """Dense array, built **independently of matvec**.

        Independence matters: the conformance harness checks ``matvec`` against
        ``to_dense``, and a ``to_dense`` implemented as ``self @ eye`` would make
        that check compare ``matvec`` with itself.
        """
        raise NotImplementedError(f"{type(self).__name__} must define `to_dense`")

    # -- derived ------------------------------------------------------------
    def matmat(self, X: Array) -> Array:
        """``A X`` for ``X`` of core shape ``(n_in, k)``, leading axes batched."""
        return self.matvec(X.swapaxes(-1, -2)).swapaxes(-1, -2)

    # -- capability introspection -------------------------------------------
    def supports(self, name: str) -> bool:
        """Whether ``name`` has a cheap implementation on this instance.

        Composites override this to intersect over their children: a
        ``BlockDiag`` can only ``solve`` if every block can. A class-level
        declaration cannot express that, which is why this is a method.
        """
        if name in self._WITHDRAWN:
            return False
        impl = getattr(type(self), name, None)
        return impl is not None and impl is not _BASE_IMPL.get(name)

    def capabilities(self) -> frozenset[str]:
        return frozenset(n for n in _OPTIONAL if self.supports(n))

    def _unsupported(self, name: str):
        raise UnsupportedOp(name, self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.shape}"


# ---------------------------------------------------------------------------
# level 2 -- SquareLinOp
# ---------------------------------------------------------------------------

class SquareLinOp(LinOp):
    """A square operator: inverse and determinant are at least well posed."""

    @property
    def n(self) -> int:
        return self.shape[0]

    def solve(self, b: Array) -> Array:
        self._unsupported("solve")

    def solve_mat(self, B: Array) -> Array:
        return self.solve(B.swapaxes(-1, -2)).swapaxes(-1, -2)

    def logdet(self) -> Array:
        """Log-determinant, as a **real JAX scalar** -- never a Python float.

        ``float(tracer)`` fails under ``jit``, and a complex intermediate (easy
        to reach via an FFT-diagonalized operator) makes ``float()`` fail
        outright. Return an array and let the caller decide.
        """
        self._unsupported("logdet")

    def diag(self) -> Array:
        self._unsupported("diag")


# ---------------------------------------------------------------------------
# level 3 -- PSDOperator
# ---------------------------------------------------------------------------

class PSDOperator(SquareLinOp):
    """A symmetric positive semi-definite operator.

    ``rmatvec``/``rmatmat`` are deliberately absent: PSD implies self-adjoint,
    so they would be aliases of ``matvec``/``matmat`` -- two more methods to
    implement and test per type, for nothing.
    """

    def factor(self) -> LinOp:
        """``L`` with ``L @ L.T == self``, as an operator of shape ``(n, k)``.

        **No constraint is imposed between ``k`` and ``n``.** Low-rank-plus-
        diagonal gives ``k = n + r``; a reduced-rank coregionalization gives
        ``k < n``, i.e. a genuinely singular operator for which ``solve`` does
        not exist at all.
        """
        self._unsupported("factor")

    def cholesky(self) -> LinOp:
        """Square triangular ``L`` with ``L @ L.T == self``.

        Only defined when the factor is square. Prefer :meth:`whiten` at call
        sites that just need ``L^{-1} x`` -- whitening needs square-and-
        invertible, not triangularity.
        """
        self._unsupported("cholesky")

    def whiten(self, x: Array) -> Array:
        """``L^{-1} x`` where ``L @ L.T == self``. The observation-noise hot path."""
        return self.cholesky().solve(x)


# Snapshot of the base implementations, so `supports` can tell "inherited the
# raising default" from "overridden with something real".
_BASE_IMPL: dict[str, Callable] = {}
for _lvl in (LinOp, SquareLinOp, PSDOperator):
    for _name in _OPTIONAL:
        if _name in _lvl.__dict__:
            _BASE_IMPL[_name] = _lvl.__dict__[_name]
del _lvl, _name


# ---------------------------------------------------------------------------
# explicit densification
# ---------------------------------------------------------------------------

def densify(op: LinOp, *, max_n: int = 4096):
    """Materialize ``op`` as a dense operator, with a guard.

    The guard is on ``op.shape``, which is static even under ``jit``, so this
    raises at trace time rather than allocating.
    """
    from .leaves import Dense, DensePSD

    n_out, n_in = op.shape
    if max(n_out, n_in) > max_n:
        raise ValueError(
            f"densify({type(op).__name__}{op.shape}) exceeds max_n={max_n}. "
            f"Raise max_n deliberately if you really want an O(n^3) fallback."
        )
    A = op.to_dense()
    return DensePSD.from_matrix(A) if isinstance(op, PSDOperator) else Dense(A)

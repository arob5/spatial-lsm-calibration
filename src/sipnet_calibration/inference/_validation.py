"""The checks the inference adapters share, private to the package."""

from __future__ import annotations

from typing import Any

from sipnet_calibration.probability import Posterior

__all__ = [
    "check_posterior_is_a_posterior",
    "check_theta_has_rank",
]


# ── checks ────────────────────────────────────────────────────────────────────


def check_posterior_is_a_posterior(posterior: Any) -> None:
    """Check *posterior* is a ``Posterior``."""
    if not isinstance(posterior, Posterior):
        raise TypeError(
            f"posterior must be a Posterior, not {type(posterior).__name__}; "
            "make one with probability.condition_on(model, observed)."
        )


def check_theta_has_rank(shape: tuple[int, ...], rank: int, dimension: int, *, message_name: str) -> None:
    """Check a theta of *shape* is ``(J, D)`` (*rank* 2) or ``(D,)`` (*rank* 1)."""
    expected = "(J, D)" if rank == 2 else "(D,)"
    if len(shape) != rank:
        other = "log_density for one sample" if rank == 2 else "batched_log_density for a batch"
        raise ValueError(f"{message_name} takes theta of shape {expected}, not {shape}; use {other}.")
    if shape[-1] != dimension:
        raise ValueError(
            f"{message_name} takes theta with D = {dimension} entries, not {shape[-1]}; "
            "pass theta in the posterior's layout, posterior.parameters.unconstrained."
        )

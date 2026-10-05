"""The checks the inference adapters share, private to the package."""

from __future__ import annotations

from typing import Any

from sipnet_calibration.probability import Posterior

__all__ = [
    "check_posterior_is_a_posterior",
    "check_theta_has_rank",
    "check_theta_has_the_entries_of_theta",
    "check_theta_is_shaped",
]


# ── checks ────────────────────────────────────────────────────────────────────


def check_posterior_is_a_posterior(posterior: Any) -> None:
    """Check *posterior* is a ``Posterior``."""
    if not isinstance(posterior, Posterior):
        raise TypeError(
            f"posterior must be a Posterior, not {type(posterior).__name__}; "
            "make one with probability.condition_on(model, observed)."
        )


def check_theta_is_shaped(shape: tuple[int, ...], rank: int, dimension: int, *, message_name: str) -> None:
    """Check a theta of *shape* is ``(J, D)`` (*rank* 2) or ``(D,)`` (*rank* 1)."""
    check_theta_has_rank(shape, rank, message_name=message_name)
    check_theta_has_the_entries_of_theta(shape, dimension, message_name=message_name)


def check_theta_has_rank(shape: tuple[int, ...], rank: int, *, message_name: str) -> None:
    """Check a theta of *shape* has *rank* axes: a batch (2) or one sample (1)."""
    if len(shape) != rank:
        expected = "(J, D)" if rank == 2 else "(D,)"
        other = "log_density for one sample" if rank == 2 else "batched_log_density for a batch"
        raise ValueError(f"{message_name} takes theta of shape {expected}, not {shape}; use {other}.")


def check_theta_has_the_entries_of_theta(shape: tuple[int, ...], dimension: int, *, message_name: str) -> None:
    """Check a theta of *shape* has ``D`` entries on its last axis."""
    if shape[-1] != dimension:
        raise ValueError(
            f"{message_name} takes theta with D = {dimension} entries, not {shape[-1]}; "
            "pass theta in the posterior's layout, posterior.parameters.unconstrained."
        )

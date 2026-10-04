"""Supports, moved to :mod:`sipnet_calibration.probability.support`; this
module re-exports them for the parameter layer until it is removed."""

from sipnet_calibration.probability.support import (
    DEFAULT_BIJECTORS,
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
    Simplex,
    Support,
    bijector_for,
    check_interval_is_valid,
)

__all__ = [
    "DEFAULT_BIJECTORS",
    "NON_NEGATIVE",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "REAL",
    "SIMPLEX",
    "UNIT_INTERVAL",
    "Interval",
    "Simplex",
    "Support",
    "bijector_for",
    "check_interval_is_valid",
]

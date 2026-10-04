"""The parameter layer's private coercion, moved to
:mod:`sipnet_calibration.probability._validation`; re-exported here until
the parameter layer is removed."""

from sipnet_calibration.probability._validation import (
    as_count,
    as_names,
    as_sequence,
    check_names_are_unique,
    truncated,
)

__all__ = [
    "as_count",
    "as_names",
    "as_sequence",
    "check_names_are_unique",
    "truncated",
]

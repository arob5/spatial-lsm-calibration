"""Tests for the names the probability layer reserves."""

from __future__ import annotations

from sipnet_calibration import conventions
from sipnet_calibration.probability import names


def test_the_layers_sample_is_the_packages():
    """The probability layer cannot import conventions, which must not
    import it, so each holds the literal; they agree."""
    assert names.SAMPLE == conventions.SAMPLE


def test_the_layers_own_names_are_reserved():
    assert names.RESERVED_NAMES == {
        names.SAMPLE, names.THETA, names.THETA_ENTRY, names.COMPONENT_LEVEL, names.ELEMENT_LEVEL
    }

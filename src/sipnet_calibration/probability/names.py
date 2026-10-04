"""The names the probability layer gives things itself, and the names it
therefore refuses for a component, an input, an axis or a dim.

The layer imports nothing of the package outside itself, so it cannot read
:mod:`sipnet_calibration.conventions`, which in turn must not import the
layer (it would load TFP). The batch dim of a batch of draws is the literal
``"sample"`` in both; ``tests/test_probability_names.py`` checks that they
agree.
"""

from __future__ import annotations

__all__ = [
    "COMPONENT_LEVEL",
    "ELEMENT_LEVEL",
    "RESERVED_NAMES",
    "SAMPLE",
    "THETA",
    "THETA_ENTRY",
]

#: The batch dim of a batch of draws, as :data:`sipnet_calibration.conventions.SAMPLE`.
SAMPLE = "sample"

#: The level of a layout's index naming each entry's component, and the
#: selector that keeps some components.
COMPONENT_LEVEL = "component"

#: The level of a layout's index naming each entry's element.
ELEMENT_LEVEL = "element"

#: The labeled form's name for theta, beside the natural values.
THETA = "theta"

#: The dim of :data:`THETA`'s entries.
THETA_ENTRY = "theta_entry"

#: Names no component, input, element axis, dim or level of a stacked dim may
#: take, since the layer gives them meanings of its own.
RESERVED_NAMES = frozenset({THETA, THETA_ENTRY, COMPONENT_LEVEL, ELEMENT_LEVEL, SAMPLE})

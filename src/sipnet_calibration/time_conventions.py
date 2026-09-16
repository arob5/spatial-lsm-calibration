"""What a ``time`` label marks: the one vocabulary every product uses.

Every product that carries a ``time`` coordinate records, in that
coordinate's ``time_label`` attribute, what its labels mean: the end of the
interval a value covers, the start, an instant, or a bookkeeping key that is
not a time at all. The drivers and the annual constraints write one of these
values, the model-output adapter will, and the observation operators read it
rather than assuming a convention.

Usage
-----
::

    from sipnet_calibration.time_conventions import TIME_LABEL_ATTR, TimeLabel

    attrs[TIME_LABEL_ATTR] = TimeLabel.INTERVAL_END.value
    TimeLabel(field["time"].attrs[TIME_LABEL_ATTR]) is TimeLabel.INTERVAL_END
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["TIME_LABEL_ATTR", "TimeLabel"]

#: The attribute on a ``time`` coordinate that holds a :class:`TimeLabel` value.
TIME_LABEL_ATTR = "time_label"


class TimeLabel(StrEnum):
    """What the label on a ``time`` coordinate marks."""

    INTERVAL_END = "interval_end"
    """The value covers the interval **ending** at the label.

    A row labeled ``h`` with step ``d`` covers ``(h - d, h]``, and a state
    reported in that row is the value at ``h``. The meteorological drivers,
    and by inheritance SIPNET's output, are labeled this way; the reading is
    inferred from the data rather than documented (``data/README.md``
    note 16).
    """

    INTERVAL_START = "interval_start"
    """The value covers the interval **starting** at the label.

    A row labeled ``h`` with step ``d`` covers ``[h, h + d)``, and a state
    reported in that row is the value at ``h + d``. This is the convention
    pySIPNET documents for SIPNET's own labeling.
    """

    INSTANT = "instant"
    """The value is a measurement or a state at the labeled instant."""

    NOMINAL = "nominal"
    """The label is a bookkeeping key, not a time.

    The annual constraint product's July 15 keys are the source's annual
    snapshot convention; nothing should read one as the instant an
    observation was taken.
    """

    STATIC = "static"
    """The quantity has no time. A label, if one is kept, is a reference key
    for indexing and says nothing about when the value holds."""

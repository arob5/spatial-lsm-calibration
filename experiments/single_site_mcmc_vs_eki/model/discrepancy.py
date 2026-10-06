"""NEE's model discrepancy: a short, a long and a recurring term.

``MODEL.md``, "NEE error", motivates the form and says how its parameters
are estimated. For the windows of one NEE observation source, ending at
times :math:`t_W` in days, the discrepancy covariance is

.. math::

    \\Sigma^\\delta_{WW'} =
    \\sigma_s^2 e^{-|\\Delta| / \\tau_s}
    + \\sigma_\\ell^2 e^{-|\\Delta| / \\tau_\\ell}
    + \\sigma_p^2 \\exp\\!\\Big(-\\frac{2 \\sin^2(\\pi \\Delta / P)}{\\lambda^2}\\Big)
      \\, e^{-|\\Delta| / \\tau_p},
    \\qquad \\Delta = t_W - t_{W'},\\ P = 365.25 \\text{ days},

with the last factor 1 when :math:`\\tau_p` is not given (a bias that recurs
unchanged every year). A term whose standard deviation is 0 is left out, so
a :class:`NEEDiscrepancy` with only the short term is the single exponential
term of ``MODEL.md``'s "Noise model".

This module holds the form only; ``config.NEE_DISCREPANCY`` holds the values
and ``model/noise.py`` declares ``R`` with them.
"""

from dataclasses import dataclass, fields
from datetime import timedelta

import numpy as np

__all__ = ["NEEDiscrepancy", "YEAR", "check_discrepancy_is_valid"]

#: The recurring term's period, in days.
YEAR = 365.25


@dataclass(frozen=True, kw_only=True)
class NEEDiscrepancy:
    """The discrepancy of one NEE observation source.

    Parameters
    ----------
    short_standard_deviation, long_standard_deviation, recurring_standard_deviation
        The terms' standard deviations, :math:`\\sigma_s, \\sigma_\\ell,
        \\sigma_p`, in the observed values' units (umol m-2 s-1). 0 leaves a
        term out.
    short_timescale, long_timescale
        :math:`\\tau_s, \\tau_\\ell`, as ``timedelta`` values or numbers of days.
    recurring_width
        :math:`\\lambda`, dimensionless: the recurring term's correlation
        falls to :math:`e^{-1/2}` at a separation of about
        :math:`\\lambda P / (2\\pi)` days of the year.
    recurring_timescale
        :math:`\\tau_p`, the timescale over which the recurring bias drifts
        from year to year, as a ``timedelta`` or a number of days, or ``None`` for
        none.
    provenance
        Where the values come from.
    """

    short_standard_deviation: float
    short_timescale: timedelta
    long_standard_deviation: float = 0.0
    long_timescale: timedelta = timedelta(days=30)
    recurring_standard_deviation: float = 0.0
    recurring_width: float = 0.5
    recurring_timescale: timedelta | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        check_discrepancy_is_valid(self)

    def covariance(self, times, xp=np):
        """The covariance between every pair of *times*, in days, ``(n, n)``.

        *xp* is the array module, NumPy or ``jax.numpy``; the fit of the
        parameters differentiates through the ``jax.numpy`` form, with the
        values traced. A term is left out only when its standard deviation
        is the number 0.
        """
        times = xp.asarray(times)
        separation = times[:, None] - times[None, :]
        distance = xp.abs(separation)
        covariance = self.short_standard_deviation**2 * xp.exp(
            -distance / _days(self.short_timescale)
        )
        if not _is_zero(self.long_standard_deviation):
            covariance = covariance + self.long_standard_deviation**2 * xp.exp(
                -distance / _days(self.long_timescale)
            )
        if not _is_zero(self.recurring_standard_deviation):
            recurring = xp.exp(
                -2.0 * xp.sin(np.pi * separation / YEAR) ** 2 / self.recurring_width**2
            )
            if self.recurring_timescale is not None:
                recurring = recurring * xp.exp(
                    -distance / _days(self.recurring_timescale)
                )
            covariance = covariance + self.recurring_standard_deviation**2 * recurring
        return covariance

    def total_variance(self) -> float:
        """The discrepancy's variance at one window, the sum of its terms'."""
        return (
            self.short_standard_deviation**2
            + self.long_standard_deviation**2
            + self.recurring_standard_deviation**2
        )

    def describe(self) -> dict:
        """The values, timescales in days, as a flat mapping for a report."""
        return {
            field.name: (_days(value) if isinstance(value, timedelta) else value)
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


def _days(duration) -> float:
    """A timescale in days: a ``timedelta``, or a number already in days."""
    if isinstance(duration, timedelta):
        return duration / timedelta(days=1)
    return duration


def _is_zero(value) -> bool:
    """Whether *value* is the number 0, which a traced value never is."""
    return isinstance(value, int | float) and value == 0


def _is_number(value) -> bool:
    """Whether *value* is a plain number, not a traced one."""
    return isinstance(value, int | float)


# ── checks ──


def check_discrepancy_is_valid(discrepancy: NEEDiscrepancy) -> None:
    """Each standard deviation is non-negative and each timescale and width positive.

    Only numbers are checked: the fit builds discrepancies of traced values.
    """
    for name in (
        "short_standard_deviation",
        "long_standard_deviation",
        "recurring_standard_deviation",
    ):
        value = getattr(discrepancy, name)
        if _is_number(value) and not value >= 0:
            raise ValueError(f"{name} is {value}; a standard deviation is non-negative")
    for name in (
        "short_timescale",
        "long_timescale",
        "recurring_timescale",
        "recurring_width",
    ):
        value = getattr(discrepancy, name)
        if value is not None and _is_number(_days(value)) and not _days(value) > 0:
            raise ValueError(f"{name} is {value}; it must be positive")

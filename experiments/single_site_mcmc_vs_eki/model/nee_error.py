"""NEE error models: the memory of each NEE source's discrepancy.

``MODEL.md``, "The NEE error model", builds the model. For the windows of one
NEE source, ending at times :math:`t_W` in days, the discrepancy covariance is
:math:`v^\\delta \\rho(t_W - t_{W'})` with the correlation

.. math::

    \\rho(\\Delta) = \\sum_m w_m \\kappa_m(\\Delta), \\qquad
    w_m = \\frac{\\sigma_m^2}{\\sum_{m'} \\sigma_{m'}^2},

over the terms present: a short and a long exponential,
:math:`\\kappa_\\tau(\\Delta) = e^{-|\\Delta|/\\tau}`, and a term recurring
every year, :math:`\\exp(-2\\sin^2(\\pi\\Delta/P)/\\lambda^2)`, with
:math:`P = 365.25` days. A term is given as the standard deviation and the
timescale (or width) of the maximum-likelihood fit it comes from, so the
weights are the fit's shares of the variance; only the shape is used here,
and :math:`v^\\delta` is ``config.NEE_DISCREPANCY_STANDARD_DEVIATION``
squared, the same for every error model.

:data:`NEE_ERROR_MODELS` holds the three error models the experiment
compares.
"""

from dataclasses import dataclass

import numpy as np
from frozendict import frozendict

__all__ = ["NEE_ERROR_MODELS", "YEAR", "Memory", "NEEErrorModel"]

#: The recurring term's period, in days.
YEAR = 365.25


@dataclass(frozen=True, kw_only=True)
class Memory:
    """The correlation of one NEE source's discrepancy.

    Parameters
    ----------
    short, long : (float, float) or None
        A fitted exponential term's ``(standard deviation, timescale in days)``.
    recurring : (float, float) or None
        The fitted recurring term's ``(standard deviation, width lambda)``.
    """

    short: tuple[float, float]
    long: tuple[float, float] | None = None
    recurring: tuple[float, float] | None = None

    def weights(self) -> dict[str, float]:
        """Each present term's share of the variance, :math:`w_m`."""
        variances = {
            name: term[0] ** 2
            for name in ("short", "long", "recurring")
            if (term := getattr(self, name)) is not None
        }
        total = sum(variances.values())
        return {name: variance / total for name, variance in variances.items()}

    def correlation(self, times, xp=np):
        """:math:`\\rho(t_i - t_j)` for every pair of *times*, in days, ``(n, n)``."""
        times = xp.asarray(times)
        separation = times[:, None] - times[None, :]
        weights = self.weights()
        correlation = weights["short"] * xp.exp(-xp.abs(separation) / self.short[1])
        if self.long is not None:
            correlation = correlation + weights["long"] * xp.exp(
                -xp.abs(separation) / self.long[1]
            )
        if self.recurring is not None:
            width = self.recurring[1]
            correlation = correlation + weights["recurring"] * xp.exp(
                -2.0 * xp.sin(np.pi * separation / YEAR) ** 2 / width**2
            )
        return correlation


@dataclass(frozen=True, kw_only=True)
class NEEErrorModel:
    """One error model: the memory of each NEE source, and where it comes from."""

    night: Memory
    day: Memory
    provenance: str

    def memory(self, source_name: str) -> Memory:
        """The memory of NEE source *source_name*."""
        return {"nee_night_centered": self.night, "nee_day_centered": self.day}[
            source_name
        ]


_FIT = (
    "run/fit_nee_discrepancy.py's {variant} fit to the residuals of the first "
    "calibration (single-term discrepancy, observed data)"
)

#: The error models compared, by name. Each nests the one before it; the
#: values are the fits of ``nee_discrepancy_fit.csv``.
NEE_ERROR_MODELS = frozendict(
    {
        "short_memory": NEEErrorModel(
            night=Memory(short=(1.0, 0.7399)),
            day=Memory(short=(1.0, 1.7218)),
            provenance="the short term of the two-term fit: "
            + _FIT.format(variant="two-term"),
        ),
        "long_memory": NEEErrorModel(
            night=Memory(short=(0.6028, 0.7399), long=(1.2007, 57.17)),
            day=Memory(short=(1.9556, 1.7218), long=(2.2976, 36.50)),
            provenance=_FIT.format(variant="two-term"),
        ),
        "recurring_bias": NEEErrorModel(
            night=Memory(
                short=(0.6904, 1.1697), long=(0.7947, 64.21), recurring=(0.7242, 0.4365)
            ),
            day=Memory(
                short=(2.0237, 1.7809), long=(1.6781, 46.63), recurring=(1.3036, 0.2078)
            ),
            provenance=_FIT.format(variant="three-term"),
        ),
    }
)

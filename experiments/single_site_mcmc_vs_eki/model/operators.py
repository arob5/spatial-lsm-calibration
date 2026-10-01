"""The experiment's observation operators beyond the library's.

Each is an :class:`~sipnet_calibration.observation.ObservationOperator`: it
declares what it reads, takes one site's model output and one observation
source's observed values, and returns the prediction on the observed values'
grid, in units the observation vector converts into the observed values'
own. The contract is ``sipnet_calibration.observation.operators``' module
docstring; which operator reads which observation source is bound in
``config.OBSERVATION_OPERATORS``.

- :class:`AverageRateOverWindows` predicts a mean rate over each observation's
  window from a per-step total: observed NEE, a mean flux over twelve hours,
  from SIPNET's NEE per timestep.
- :class:`ComputeAbovegroundBiomass` predicts dry aboveground biomass over
  each observation's window from SIPNET's wood carbon and a carbon fraction:
  LandTrendr's biomass.
"""

from dataclasses import dataclass

import xarray as xr
from pysipnet.arithmetic import divide_with_units, step_length

from sipnet_calibration import fields
from sipnet_calibration.conventions import TIME
from sipnet_calibration.observation.operators import restrict_to_observed_sites
from sipnet_calibration.observation.time_alignment import (
    check_run_spans_the_windows,
    reduce_windows,
    windows_from_observed_values,
)

__all__ = ["AverageRateOverWindows", "ComputeAbovegroundBiomass"]


@dataclass(frozen=True)
class AverageRateOverWindows:
    """The mean rate of a per-step total over each observation's own window.

    The model variable, a total over each timestep, is divided by the step's
    length (``pysipnet.arithmetic.step_length``), which makes it a rate, and
    that rate is averaged over each window, weighted by step length
    (:func:`~sipnet_calibration.observation.time_alignment.reduce_windows`,
    ``how="mean"``). A step belongs to the window its end falls in, so a
    window whose edges are step edges averages exactly the steps inside it.

    Requires the observed values to carry ``window_start`` and
    ``window_end``.

    Parameters
    ----------
    output_variable_name:
        The pySIPNET output variable to read, a ``timestep_total`` such as
        ``"net_ecosystem_exchange"``.

    Raises
    ------
    ValueError
        On a call, if the observed values carry no windows; if a window
        reaches a step or more beyond the model record; or for any refusal of
        ``restrict_to_observed_sites``, ``divide_with_units`` or
        ``reduce_windows``.
    """

    output_variable_name: str

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return (self.output_variable_name,)

    @property
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ()

    def __call__(
        self, model_output, observed_values, *, sipnet_parameter_fields=None
    ) -> xr.DataArray:
        total = restrict_to_observed_sites(
            model_output[self.output_variable_name], observed_values
        )
        rate = divide_with_units(total, step_length(total))
        windows = windows_from_observed_values(observed_values)
        observed = fields.message_name(observed_values, "the observation source")
        check_run_spans_the_windows(
            rate, windows, f"{type(self).__name__} on {observed}"
        )
        return reduce_windows(rate, windows, "mean", labels=observed_values[TIME])


@dataclass(frozen=True)
class ComputeAbovegroundBiomass:
    """Dry aboveground biomass: wood carbon averaged over each window, over a carbon fraction.

    SIPNET's ``wood_carbon`` (``plantWoodC``) is its aboveground wood; coarse
    and fine roots are pools of their own, and PEcAn's ``model2netcdf.SIPNET``
    reports it as ``AbvGrndWood``. It is averaged over each observation's
    window, weighted by step length, and divided by the fraction of dry wood
    mass that is carbon, giving dry biomass in the model's mass per area with
    no constituent. Leaves are left out: the biomass products estimate the
    woody part of trees, and SIPNET's deciduous leaves come and go within a
    window.

    Requires the observed values to carry ``window_start`` and
    ``window_end``, and to be dry biomass (no ``constituent``).

    Parameters
    ----------
    carbon_fraction:
        The mass fraction of dry wood that is carbon, in (0, 1).

    Raises
    ------
    TypeError
        On construction, if *carbon_fraction* is not a number.
    ValueError
        On construction, if *carbon_fraction* is not in (0, 1). On a call, if
        the observed values carry no windows; if a window reaches a step or
        more beyond the model record; or for any refusal of
        ``restrict_to_observed_sites`` or ``reduce_windows``.
    """

    carbon_fraction: float

    def __post_init__(self) -> None:
        check_carbon_fraction_is_a_number(self.carbon_fraction)
        check_carbon_fraction_is_between_zero_and_one(self.carbon_fraction)

    @property
    def output_variable_names(self) -> tuple[str, ...]:
        return ("wood_carbon",)

    @property
    def sipnet_parameter_names_read(self) -> tuple[str, ...]:
        return ()

    def __call__(
        self, model_output, observed_values, *, sipnet_parameter_fields=None
    ) -> xr.DataArray:
        wood_carbon = restrict_to_observed_sites(
            model_output["wood_carbon"], observed_values
        )
        windows = windows_from_observed_values(observed_values)
        observed = fields.message_name(observed_values, "the observation source")
        check_run_spans_the_windows(
            wood_carbon, windows, f"{type(self).__name__} on {observed}"
        )
        mean_carbon = reduce_windows(
            wood_carbon, windows, "mean", labels=observed_values[TIME]
        )
        # Carbon to dry mass changes the substance, which no pySIPNET verb
        # does, so the attributes are set here: the units stay, the
        # constituent goes.
        biomass = mean_carbon / self.carbon_fraction
        biomass.attrs = {
            key: value
            for key, value in mean_carbon.attrs.items()
            if key not in ("constituent", "long_name", "description")
        }
        biomass.attrs["long_name"] = "Aboveground dry biomass"
        biomass.attrs["derivation"] = (
            f"wood_carbon averaged over each window, divided by the carbon "
            f"fraction {self.carbon_fraction:g}"
        )
        biomass.name = "aboveground_biomass"
        return biomass


# ── checks ──


def check_carbon_fraction_is_a_number(carbon_fraction: float) -> None:
    """A carbon fraction is a number, not a boolean."""
    if isinstance(carbon_fraction, bool) or not isinstance(
        carbon_fraction, (int, float)
    ):
        raise TypeError(
            f"the carbon fraction is {carbon_fraction!r}, not a number; pass a float"
        )


def check_carbon_fraction_is_between_zero_and_one(carbon_fraction: float) -> None:
    """A carbon fraction is strictly between 0 and 1."""
    if not 0 < carbon_fraction < 1:
        raise ValueError(
            f"the carbon fraction is {carbon_fraction}, not in (0, 1); pass the "
            "mass fraction of dry wood that is carbon"
        )

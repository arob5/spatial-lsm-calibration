"""The experiment's inputs at its site, each through the library's own reader.

One loader per data source the calibration reads, each restricted to
``config.SITE`` and to the choices ``config`` makes, so every later step reads
its inputs the same way. ``scripts/describe.py`` loads them all and prints
what it found, which is the check that the configuration points at data that
exists.
"""

import warnings

import pandas as pd
import xarray as xr

from sipnet_calibration import constraints, drivers, initial_conditions, sites
from sipnet_calibration import net_ecosystem_exchange as nee

from .. import config

__all__ = [
    "constraint_fields",
    "driver_dataset",
    "initial_condition_fields",
    "observed_nee_fields",
    "site_table",
]


def site_table() -> pd.DataFrame:
    """The site table's row for the site, as a one-row site table."""
    return sites.select_sites(sites.load_sites(), site_ids=[config.SITE])


def driver_dataset() -> xr.Dataset:
    """The prepared drivers of the site and member, on ``(driver_member, site, time)``.

    Run ``scripts/prepare_drivers.py`` first: the raw files are refused by
    pySIPNET.
    """
    with warnings.catch_warnings():
        # The files hold exact zeros of vpd where SIPNET clamps, which pySIPNET
        # warns about on read; a property of the files, not of this experiment.
        warnings.simplefilter("ignore", UserWarning)
        return drivers.load_drivers(
            [config.SITE],
            source_indices=[config.DRIVER_SOURCE_INDEX],
            root=config.PREPARED_DRIVERS_ROOT,
            site_table=site_table(),
            time_zone=config.DRIVER_TIME_ZONE,
        )


def observed_nee_fields() -> dict[str, xr.DataArray]:
    """The calibration and validation NEE series at the site, keyed by series name.

    Each is the full processed record; which steps and days are kept, and the
    validation series' held-out years, are the observation operators' choices.
    """
    return nee.net_ecosystem_exchange_fields(
        [config.CALIBRATION_NEE_SERIES, config.VALIDATION_NEE_SERIES],
        sites=[config.SITE],
    )


def constraint_fields() -> dict[str, xr.DataArray]:
    """The configured constraints at the site, keyed by constraint name."""
    return constraints.constraint_fields(config.CONSTRAINT_NAMES, sites=[config.SITE])


def initial_condition_fields() -> dict[str, xr.DataArray]:
    """Every initial condition at the site, on ``(initial_condition_member, site)``."""
    return initial_conditions.initial_condition_fields(sites=[config.SITE])

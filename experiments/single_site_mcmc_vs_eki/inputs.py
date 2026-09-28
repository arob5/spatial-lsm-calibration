"""The experiment's inputs at its site, each through the library's own reader.

One loader per data source the calibration reads, each restricted to
``config.SITE`` and to the choices ``config`` makes, so every later step reads
its inputs the same way. Run as a script, it loads them all and prints what it
found, which is the check that the configuration points at data that exists.

Usage
-----
    uv run python experiments/single_site_mcmc_vs_eki/inputs.py
"""

import sys
import warnings

import pandas as pd
import xarray as xr

import config
from sipnet_calibration import constraints, drivers, initial_conditions, sites
from sipnet_calibration import net_ecosystem_exchange as nee
from sipnet_calibration.conventions import TIME
from sipnet_calibration.validation import range_summary

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

    Run ``prepare_drivers.py`` first: the raw files are refused by pySIPNET.
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


# ── entry point ──


def main() -> int:
    """Load every input and print what was found."""
    try:
        table = site_table()
        driver_data = driver_dataset()
        observed_nee = observed_nee_fields()
        constraint_data = constraint_fields()
        initial_condition_data = initial_condition_fields()
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    row = table.iloc[0]
    print(
        f"site {config.SITE}: {row['site_name']} ({row['lon']:.4f}, {row['lat']:.4f})"
    )
    print(
        f"\ndrivers, member {config.DRIVER_SOURCE_INDEX}, clock {config.DRIVER_TIME_ZONE}: "
        f"{_time_extent(driver_data[TIME])}"
    )
    print("\nobserved NEE")
    for name, field in observed_nee.items():
        print(f"  {name}: {_coverage(field)}, {field.attrs.get('units')}")
    print("\nconstraints")
    for name, field in constraint_data.items():
        print(f"  {name}: {_coverage(field)}, {field.attrs.get('units')}")
    print("  excluded: " + ", ".join(config.EXCLUDED_CONSTRAINTS))
    print(
        "\ninitial conditions, over members: minimum, median, maximum, negative count"
    )
    for name, field in initial_condition_data.items():
        print(
            f"  {name}: {range_summary(field.values.ravel())}, {field.attrs.get('units')}"
        )
    return 0


# ── helpers ──


def _time_extent(time: xr.DataArray) -> str:
    """The first and last time labels and their count."""
    return f"{time.values[0]} to {time.values[-1]}, {time.size} steps"


def _coverage(field: xr.DataArray) -> str:
    """How many values a field at one site holds, and over which times."""
    values = field.squeeze(drop=True)
    observed = values.notnull()
    if TIME not in values.dims:
        return f"{int(observed.sum())} static value"
    times = values[TIME].values[observed.values]
    if not times.size:
        return "no values"
    return f"{times.size} values, {str(times[0])[:10]} to {str(times[-1])[:10]}"


if __name__ == "__main__":
    sys.exit(main())

"""Print what the experiment's model reads and builds: the inputs, the observations, the noise.

Overview
--------
Loads every input, builds the calibration and validation posteriors
(``model/calibration.py``), and prints what the inputs, the observation
vectors and the noise covariances hold. It runs no SIPNET and writes
nothing; it is the check that ``config`` points at data that exists and
that the model builds from it.

Input data
----------
The prepared driver file (``run/prepare_drivers.py``), the processed
files ``model/inputs.py`` reads, and ``config``.

Output data
-----------
None; three reports on standard output:

- **inputs**: the site, the drivers' record, each NEE series' and
  constraint's coverage, and the initial conditions' range over members;
- **observations**: both observation vectors, one row per observation
  source, and the fraction of each month's NEE windows each keeps;
- **noise**: what each observation source contributes to ``R``
  (``noise.noise_summary``).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.check_inputs
"""

import sys

import pandas as pd
import xarray as xr

from sipnet_calibration.conventions import TIME, WINDOW_START
from sipnet_calibration.observation import ObservationSource
from sipnet_calibration.validation import range_summary

from .. import config
from ..model import calibration, inputs, noise

__all__ = ["main"]


# ── entry point ──


def main() -> int:
    """Print the three reports, or the error that stopped one."""
    try:
        describe_inputs()
        posteriors = {
            "calibration": calibration.calibration_posterior(),
            "validation": calibration.validation_posterior(),
        }
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            describe_observations(
                *(calibration.observation_vector(p) for p in posteriors.values())
            )
            describe_noise(posteriors)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


# ── the steps ──


def describe_inputs() -> None:
    """The site, and what each input holds at it."""
    table = inputs.site_table()
    driver_data = inputs.driver_dataset()
    observed_nee = inputs.observed_nee_fields()
    constraint_data = inputs.constraint_fields()
    initial_condition_data = inputs.initial_condition_fields()

    row = table.iloc[0]
    print(
        f"site {config.SITE}: {row['site_name']} ({row['lon']:.4f}, {row['lat']:.4f})"
    )
    print(
        f"\ndrivers, member {config.DRIVER_SOURCE_INDEX}, clock "
        f"{config.DRIVER_TIME_ZONE}: {_time_extent(driver_data[TIME])}"
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
            f"  {name}: {range_summary(field.values.ravel())}, "
            f"{field.attrs.get('units')}"
        )


def describe_observations(calibration_vector, validation_vector) -> None:
    """Both observation vectors, and the NEE windows each keeps by month."""
    print(f"\ncalibration: {calibration_vector!r}")
    print(calibration_vector.describe().to_string())
    print(f"\nvalidation: {validation_vector!r}")
    print(validation_vector.describe().to_string())
    print("\nNEE windows kept, as a fraction of the period's windows, by month")
    for label, vector, period in (
        ("calibration", calibration_vector, config.CALIBRATION_NEE_PERIOD),
        ("validation", validation_vector, config.VALIDATION_NEE_PERIOD),
    ):
        for name in config.NEE_WINDOWS:
            print(f"  {label} {name}: {_kept_by_month(vector[name], period)}")


def describe_noise(posteriors: dict) -> None:
    """What each observation source contributes to each posterior's ``R``."""
    summaries = pd.concat(
        {
            name: noise.noise_summary(posterior, calibration.observation_vector(posterior))
            for name, posterior in posteriors.items()
        },
        names=["vector"],
    )
    print()
    print(summaries.to_string(float_format=lambda x: f"{x:.3g}"))
    print("\nthe columns are described by noise.noise_summary")


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


def _kept_by_month(source: ObservationSource, period: tuple[int, int]) -> list[float]:
    """The fraction of each month's windows in *period* the source keeps."""
    kept = pd.DatetimeIndex(source.observed_values[WINDOW_START].values)
    first, last = period
    days = pd.date_range(f"{first}-01-01", f"{last}-12-31", freq="D")
    kept_counts = pd.Series(kept.month).value_counts()
    all_counts = pd.Series(days.month).value_counts()
    return [
        round(float(kept_counts.get(m, 0) / all_counts[m]), 2) for m in range(1, 13)
    ]


if __name__ == "__main__":
    sys.exit(main())

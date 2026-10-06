"""Prepare the experiment's driver file: the raw ERA5 file, corrected to mean what SIPNET reads.

Overview
--------
Reads the configured site's raw driver file for the configured member, applies
four corrections for known defects of the ERA5 driver files, and writes the
result through pySIPNET, which validates it, to the experiment's output. The
raw file is never edited. After the corrections every row means what SIPNET's
climate format says a row means: the labels are the UTC start of a three-hour
step, and every forcing value is a total or a mean over that step. A SIPNET
run on the prepared file therefore has its output on true UTC, the observed
NEE's clock, and nothing downstream needs to know about the defects.

Input data
----------
``config.RAW_DRIVERS_ROOT / f"ERA5_{SITE}_{DRIVER_SOURCE_INDEX}" / "ERA5.*.clim"``:
exactly one SIPNET climate file in the legacy 14-column layout, one row per
three-hourly ERA5 validity time from 00:00 UTC on 1 January of its first year,
eight rows a day, with no gaps (``data/README.md``, Drivers).

Output data
-----------
One file under ``config.PREPARED_DRIVERS_ROOT``, in a directory named as the
raw one, in pySIPNET's standard 12-column layout (the legacy layout's site and
soil-wetness columns are ignored by SIPNET and dropped). It is named
``ERA5.<member>.<first day>.<last day>.clim`` for the days its first and last
steps start on, which the shift of correction 2 moves back by three hours, so
``drivers.load_drivers(root=PREPARED_DRIVERS_ROOT)`` reads it as it reads a
raw root. Its labels are declared UTC (``config.DRIVER_TIME_ZONE``). Beside
it, ``provenance.json`` records the code, packages, command and raw file it
was made from (``run/_provenance.py``).

Notes
-----
**What SIPNET assumes.** A row of a climate file is one timestep. Its
``year``, ``day_of_year`` and ``hour_of_day`` label the **start** of the step,
``timestep_length`` its length, and every forcing value describes the whole
step: radiation and precipitation as totals over it, temperature, humidity and
wind as means over it. pySIPNET builds the model's time axis from the labels
and lengths on the same reading.

**What the raw files hold.** Row ``k`` of a raw file is the ERA5 field at
validity time ``t_k``, ``3 k`` hours after 00:00 UTC on its first day
(``data/README.md`` Note 16, established from the generating code, the raw
ERA5 files and ECMWF's documentation). That departs from SIPNET's reading in
four ways, each corrected here, in this order:

1. **The hour labels drift** (Note 15, issue #9). The hour column steps by
   3.000685 h, not 3, so by the last row of a year it is two hours late, and
   pySIPNET refuses the file. The cause is PEcAn's ``met2CF.ERA5``, which spread
   each year's rows over the wrong span; the values themselves are on the exact
   three-hourly grid. *Correction:* the labels are rebuilt from position,
   ``t_k = t_0 + 3 k`` hours, after checking that each drifting hour is still
   in its three-hour slot and that the day labels, which do not drift, agree
   with position in every row.

2. **Radiation and precipitation cover the step ending at the label**
   (Note 16). They come from ERA5's accumulated fields, which run over the
   three hours ending at the validity time, so the row labeled ``t_k``
   describes ``(t_k - 3 h, t_k]``, where SIPNET reads it as
   ``[t_k, t_k + 3 h)``. Left alone, the model's time axis runs three hours
   late against the UTC observations: at Harvard Forest that moves afternoon
   uptake across the UTC day boundary, and it misplaces every sub-daily
   comparison. *Correction:* every label moves back three hours, so row ``k``
   is labeled ``t_k - 3 h``, the UTC start of the interval its accumulations
   cover. The record then starts at 21:00 UTC on the day before the raw
   file's first. Radiation and precipitation keep their values exactly.

3. **Temperature, humidity and wind are snapshots, not step means**
   (Note 16). ``air_temperature``, ``vapor_pressure_deficit``,
   ``vapor_pressure`` and ``wind_speed`` come from ERA5's instantaneous fields,
   at ``t_k``, which after correction 2 is the **end** of row ``k``'s step.
   SIPNET treats each as the mean over the step, so a row carries a value from
   one edge of its interval, a 1.5-hour phase error in the diurnal cycle.
   *Correction:* each is replaced by the mean of its two edge values, the
   snapshots at ``t_{k-1}`` and ``t_k``: the trapezoidal estimate of the step
   mean. The first row has no earlier snapshot and keeps its own value. The
   vapor-pressure deficit is averaged as given rather than recomputed from
   averaged temperature and humidity; at this spacing the difference is small
   beside the error corrected.

4. **The soil temperature anticipates the air temperature.** ERA5's ensemble
   product has no soil temperature, so PEcAn's ``met2model.SIPNET`` makes one
   (PEcAn ``models/sipnet/R/met2model.SIPNET.R``, the ``soil_temperature``
   branch): an exponential filter of air temperature with a timescale of 15
   days, applied with R's ``stats::convolve``. With that function's defaults
   the filter weights the **following** weeks, wrapping from each year's end to
   its start, so the soil warms before the air does. In Harvard Forest's
   file (site 4977, member 1) that filter reproduces the soil temperature to
   within a thousandth of a degree in every year, and on daily means the soil
   temperature leads the air temperature by about a week, where a real soil
   lags it. The script prints the lead before and after the correction. SIPNET reads soil temperature for
   heterotrophic and root respiration and for its frozen-soil threshold, so
   the defect shifts the seasonal timing of respiration, hence of NEE.
   *Correction:* soil temperature is recomputed as the same exponential filter
   run forward in time, a causal exponential moving average of the corrected
   air temperature, ``s_k = s_{k-1} + a (T_k - s_{k-1})`` with
   ``a = 1 - exp(-dt / tau)``, ``dt`` the step length and ``tau``
   ``config.SOIL_TEMPERATURE_TIMESCALE``, continuous across years. It starts
   from the mean air temperature over the record's first ``tau``, whose error
   decays with that timescale. This is the filter PEcAn's comment says it
   borrowed, not a soil model: the timescale is PEcAn's.

**Left as they are.** ``soil_vapor_pressure_deficit`` was computed by PEcAn
from the anticipating soil temperature, but SIPNET reads it and uses it
nowhere, so it is kept as written. The small negative excursions of radiation
and precipitation and the few zero vapor-pressure deficits (Note 17) are kept:
SIPNET clamps them, and ``drivers.load_drivers`` counts them.

**Building timestamps.** The project otherwise never builds a timestamp from a
``.clim`` label; this script exists to correct those labels, so it derives the
validity times from row position and then checks that the time axis pySIPNET
builds from the prepared file ends every step exactly at its validity time.

All four defects belong in the files at their source; issue #9 tracks the
first.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.prepare_drivers
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.climate import CLIMATE_COLUMNS, ClimateDrivers

from sipnet_calibration import drivers
from sipnet_calibration.conventions import TIME, TIMESTEP_START
from sipnet_calibration.io import write_checked

from .. import config
from . import _provenance

#: The raw files' 14 columns, in the legacy layout's order: pySIPNET's
#: climate columns between a site identifier and a soil-wetness value, both
#: of which SIPNET ignores.
RAW_COLUMN_NAMES = ("site_identifier", *CLIMATE_COLUMNS, "soil_wetness")

#: The time between consecutive ERA5 validity times.
STEP = pd.Timedelta(hours=3)

#: Rows in one day.
ROWS_PER_DAY = pd.Timedelta(days=1) // STEP

#: The raw files' step length, ``timestep_length``, in days.
STEP_LENGTH_DAYS = STEP / pd.Timedelta(days=1)

#: Columns from ERA5's accumulated fields: totals over the three hours ending
#: at the validity time (correction 2), kept exactly.
ACCUMULATED_COLUMN_NAMES = ("photosynthetically_active_radiation", "precipitation")

#: Columns from ERA5's instantaneous fields: snapshots at the validity time,
#: turned into step means (correction 3).
INSTANTANEOUS_COLUMN_NAMES = (
    "air_temperature",
    "vapor_pressure_deficit",
    "vapor_pressure",
    "wind_speed",
)

#: Columns kept exactly as written: the accumulations, the step length, and
#: the soil deficit SIPNET does not use.
UNCHANGED_COLUMN_NAMES = (
    *ACCUMULATED_COLUMN_NAMES,
    "timestep_length",
    "soil_vapor_pressure_deficit",
)


# ── entry point ──


def main() -> int:
    """Prepare the configured site's driver file for the configured member."""
    try:
        raw_path = find_raw_driver_file(
            config.RAW_DRIVERS_ROOT, config.SITE, config.DRIVER_SOURCE_INDEX
        )
        raw = read_raw_driver_file(raw_path)
        validity_times = validity_times_of(raw)
        prepared = corrected_drivers(raw, validity_times)
        prepared_path = prepared_driver_path(raw_path, prepared)
        write_prepared_driver_file(prepared_path, prepared, raw, validity_times)
        _provenance.write_provenance(
            prepared_path.parent / "provenance.json", input_files=[raw_path]
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {prepared_path}")
    print(summary(raw, prepared))
    return 0


# ── the steps ──


def find_raw_driver_file(root: Path, site: int, source_index: int) -> Path:
    """The one raw ``.clim`` file of *site* and *source_index* under *root*."""
    directory = root / drivers.DRIVER_DIRECTORY_TEMPLATE.format(
        site=site, member=source_index
    )
    paths = sorted(directory.glob(drivers.DRIVER_FILE_GLOB))
    check_directory_holds_one_driver_file(directory, paths)
    return paths[0]


def read_raw_driver_file(path: Path) -> pd.DataFrame:
    """The raw file as a table, one row per line, under :data:`RAW_COLUMN_NAMES`."""
    # Read as text, not through pySIPNET, which refuses the drifting labels.
    table = pd.read_csv(path, sep=r"\s+", header=None)
    check_raw_file_has_the_legacy_layout(table, path)
    table.columns = list(RAW_COLUMN_NAMES)
    return table


def validity_times_of(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Each row's ERA5 validity time, ``t_k = t_0 + 3 k`` hours (correction 1)."""
    check_raw_steps_are_three_hours(raw)
    check_file_holds_whole_days(raw)
    check_raw_hours_are_in_their_slots(raw)
    first = raw.iloc[0]
    start = pd.Timestamp(year=int(first["year"]), month=1, day=1) + pd.Timedelta(
        days=int(first["day_of_year"]) - 1
    )
    validity_times = pd.date_range(start, periods=len(raw), freq=STEP)
    check_day_labels_agree_with_positions(raw, validity_times)
    return validity_times


def corrected_drivers(
    raw: pd.DataFrame, validity_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """The drivers with corrections 1 to 4 applied, under pySIPNET's column names."""
    prepared = raw.loc[:, list(CLIMATE_COLUMNS)].copy()
    # Corrections 1 and 2: label each row with the UTC start of the interval
    # its accumulations cover, three hours before its validity time.
    starts = validity_times - STEP
    prepared["year"] = starts.year
    prepared["day_of_year"] = starts.dayofyear
    prepared["hour_of_day"] = starts.hour + starts.minute / 60
    # Correction 3: each snapshot column becomes the mean of the step's two
    # edge values; the first row has no earlier snapshot.
    for name in INSTANTANEOUS_COLUMN_NAMES:
        snapshots = raw[name].to_numpy()
        earlier = np.concatenate([snapshots[:1], snapshots[:-1]])
        prepared[name] = (earlier + snapshots) / 2
    # Correction 4: soil temperature as a causal filter of the corrected air
    # temperature.
    prepared["soil_temperature"] = causal_soil_temperature(
        prepared["air_temperature"].to_numpy()
    )
    return prepared


def causal_soil_temperature(air_temperature: np.ndarray) -> np.ndarray:
    """The exponential moving average of *air_temperature*, run forward in time.

    ``s_k = s_{k-1} + a (T_k - s_{k-1})``, ``a = 1 - exp(-dt / tau)``, from
    ``s_{-1}``, the mean of *air_temperature* over the first ``tau``; ``dt``
    is :data:`STEP` and ``tau`` is ``config.SOIL_TEMPERATURE_TIMESCALE``.
    """
    steps_per_timescale = pd.Timedelta(config.SOIL_TEMPERATURE_TIMESCALE) / STEP
    weight = 1 - np.exp(-1 / steps_per_timescale)
    soil_temperature = np.empty_like(air_temperature)
    previous = air_temperature[: int(round(steps_per_timescale))].mean()
    for k, temperature in enumerate(air_temperature):
        previous += weight * (temperature - previous)
        soil_temperature[k] = previous
    return soil_temperature


def prepared_driver_path(raw_path: Path, prepared: pd.DataFrame) -> Path:
    """Where the prepared file goes: named, as the reader requires, for its record."""
    source_index = drivers.DRIVER_FILE_PATTERN.match(raw_path.name).group(1)
    first, last = (
        pd.Timestamp(year=int(row["year"]), month=1, day=1)
        + pd.Timedelta(days=int(row["day_of_year"]) - 1)
        for row in (prepared.iloc[0], prepared.iloc[-1])
    )
    name = f"ERA5.{source_index}.{first.date()}.{last.date()}.clim"
    return config.PREPARED_DRIVERS_ROOT / raw_path.parent.name / name


def write_prepared_driver_file(
    path: Path,
    prepared: pd.DataFrame,
    raw: pd.DataFrame,
    validity_times: pd.DatetimeIndex,
) -> None:
    """Write *prepared* through pySIPNET to *path*, checked by reading it back."""
    # Constructing ClimateDrivers validates the labels against the step lengths.
    climate = ClimateDrivers.from_dataframe(
        prepared, n_columns=12, time_zone=config.DRIVER_TIME_ZONE
    )

    def write(partial: Path) -> None:
        climate.to_file(partial)

    def check(partial: Path) -> None:
        written = ClimateDrivers.from_file(partial, time_zone=config.DRIVER_TIME_ZONE)
        check_steps_end_at_the_validity_times(written.xarray, validity_times)
        check_unchanged_columns_are_as_written(written.pandas, raw)
        check_corrected_columns_are_finite(written.pandas)

    write_checked(path, write, check)


def summary(raw: pd.DataFrame, prepared: pd.DataFrame) -> str:
    """What the corrections changed, as the script prints it."""
    first = prepared.iloc[0]
    lines = [
        f"first step starts {int(first['year'])}, day {int(first['day_of_year'])}, "
        f"{first['hour_of_day']:g}:00 UTC",
        "soil temperature's lead over air temperature, daily means, in days "
        f"(positive = ahead): raw {_soil_lead_days(raw):+d}, "
        f"prepared {_soil_lead_days(prepared):+d}",
    ]
    for name in INSTANTANEOUS_COLUMN_NAMES:
        change = np.abs(prepared[name].to_numpy() - raw[name].to_numpy())
        lines.append(
            f"{name}: mean absolute change {change.mean():.3g}, largest {change.max():.3g}"
        )
    return "\n".join(lines)


# ── helpers ──


def _soil_lead_days(table: pd.DataFrame) -> int:
    """The shift, in days, that best matches daily soil to daily air temperature."""
    day = np.arange(len(table)) // ROWS_PER_DAY
    air = table.groupby(day)["air_temperature"].mean().to_numpy()
    soil = table.groupby(day)["soil_temperature"].mean().to_numpy()
    shifts = range(-40, 41)
    # A positive shift moves air temperature earlier, matching a soil ahead of it.
    correlations = [
        np.corrcoef(
            air[max(s, 0) : len(air) + min(s, 0)],
            soil[max(-s, 0) : len(soil) + min(-s, 0)],
        )[0, 1]
        for s in shifts
    ]
    return list(shifts)[int(np.argmax(correlations))]


# ── checks ──


def check_directory_holds_one_driver_file(directory: Path, paths: list[Path]) -> None:
    """A member's driver directory holds exactly one ``.clim`` file."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no driver directory {directory}; copy it from the SCC's "
            "ERA5_2012_2024 directory (data/README.md, Drivers)"
        )
    if len(paths) != 1:
        raise ValueError(
            f"{directory} holds {len(paths)} .clim files, not one; "
            "leave only the file the runs should read"
        )


def check_raw_file_has_the_legacy_layout(table: pd.DataFrame, path: Path) -> None:
    """The raw file has the 14 columns of the legacy layout, every field filled."""
    if table.shape[1] != len(RAW_COLUMN_NAMES) or table.isna().any().any():
        raise ValueError(
            f"{path} does not have {len(RAW_COLUMN_NAMES)} filled columns in every "
            "row; this is not an ERA5 driver file in the legacy layout"
        )


def check_raw_steps_are_three_hours(raw: pd.DataFrame) -> None:
    """Every raw row declares a three-hour step."""
    if not np.allclose(raw["timestep_length"], STEP_LENGTH_DAYS):
        raise ValueError(
            f"the raw file declares steps other than {STEP_LENGTH_DAYS} days; "
            "the corrections assume the three-hourly ERA5 grid"
        )


def check_file_holds_whole_days(raw: pd.DataFrame) -> None:
    """The file holds whole days of :data:`ROWS_PER_DAY` rows."""
    if len(raw) % ROWS_PER_DAY:
        raise ValueError(
            f"the file holds {len(raw)} rows, not whole days of {ROWS_PER_DAY}; "
            "relabeling by position is unsafe"
        )


def check_raw_hours_are_in_their_slots(raw: pd.DataFrame) -> None:
    """Every row's drifting hour still falls in its three-hour slot of the day."""
    slots = (raw["hour_of_day"] // (STEP / pd.Timedelta(hours=1))).astype(int)
    expected = np.arange(len(raw)) % ROWS_PER_DAY
    wrong = np.flatnonzero(slots.to_numpy() != expected)
    if wrong.size:
        raise ValueError(
            f"{wrong.size} rows, the first row {wrong[0]}, are not in their "
            "three-hour slot of the day; the drift is not the one Note 15 "
            "describes, so relabeling by position is unsafe"
        )


def check_day_labels_agree_with_positions(
    raw: pd.DataFrame, validity_times: pd.DatetimeIndex
) -> None:
    """Each row's year and day label is the day of its validity time."""
    agrees = (raw["year"].to_numpy() == validity_times.year) & (
        raw["day_of_year"].to_numpy() == validity_times.dayofyear
    )
    if not agrees.all():
        first = int(np.flatnonzero(~agrees)[0])
        raise ValueError(
            f"row {first} is labeled year {raw['year'].iloc[first]}, day "
            f"{raw['day_of_year'].iloc[first]}, but its position puts it on "
            f"{validity_times[first].date()}; the file has a gap or a repeated "
            "day, so relabeling by position is unsafe"
        )


def check_steps_end_at_the_validity_times(
    written: xr.Dataset, validity_times: pd.DatetimeIndex
) -> None:
    """pySIPNET's time axis ends every step at its row's ERA5 validity time."""
    ends = pd.DatetimeIndex(written[TIME].values)
    starts = pd.DatetimeIndex(written[TIMESTEP_START].values)
    if not (ends.equals(validity_times) and starts.equals(validity_times - STEP)):
        raise ValueError(
            "the prepared file's steps do not run from three hours before each "
            "validity time to it; the relabeling is wrong"
        )


def check_unchanged_columns_are_as_written(
    written: pd.DataFrame, raw: pd.DataFrame
) -> None:
    """The columns no correction touches hold the raw values, row for row."""
    for name in UNCHANGED_COLUMN_NAMES:
        if not np.array_equal(written[name].to_numpy(), raw[name].to_numpy()):
            raise ValueError(
                f"{name} differs from the raw file; the corrections must keep "
                "it exactly"
            )


def check_corrected_columns_are_finite(written: pd.DataFrame) -> None:
    """Every corrected column holds finite values."""
    for name in (*INSTANTANEOUS_COLUMN_NAMES, "soil_temperature"):
        if not np.isfinite(written[name].to_numpy()).all():
            raise ValueError(
                f"{name} holds a value that is not finite after correction"
            )


if __name__ == "__main__":
    sys.exit(main())

"""The raw files' dims, the two time axes, and where each file is.

The dims only the raw files use, the two regular time axes -- the raw files'
local-standard-time axis and the processed files' UTC axis -- and the path of
every file the package reads or writes. Nothing here reads or writes anything.
The dims and coordinates the processed files share with the rest of the
project are :mod:`sipnet_calibration.conventions`'.

Contents
--------
:data:`TOWER`, :data:`TIME_INDEX`
    The raw files' dims.
:data:`RAW_START`, :data:`RAW_END`, :data:`PROCESSED_START`, :data:`PROCESSED_END`
    The ends of the raw (local standard time) and processed (UTC) axes.
:class:`Resolution`, :data:`HALF_HOURLY`, :data:`HOURLY`, :data:`RESOLUTIONS`, :func:`resolve_resolution`
    The two source resolutions, with their step and their raw and processed
    time axes.
:data:`TOWER_TABLE_FILE`, :data:`AMERIFLUX_SITE_LIST_FILE`, :data:`POOL_INPUT_LIST_FILE`
    The file names of the tower table and its two inputs.
The path functions
    :func:`default_raw_directory` and :func:`default_source_root` for the
    storage-backed raw files, :func:`raw_path` and
    :func:`ameriflux_site_list_path` within it; :func:`tower_table_path` and
    :func:`pool_input_list_path` for the tracked ones. The processed files'
    paths are :mod:`~sipnet_calibration.net_ecosystem_exchange.processed`'s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from frozendict import frozendict

from sipnet_calibration.conventions import data_root, tracked_data_root
from sipnet_calibration.validation import truncated

__all__ = [
    "AMERIFLUX_SITE_LIST_FILE",
    "HALF_HOURLY",
    "HOURLY",
    "POOL_INPUT_LIST_FILE",
    "PROCESSED_END",
    "PROCESSED_START",
    "RAW_END",
    "RAW_START",
    "RESOLUTIONS",
    "Resolution",
    "TIME_INDEX",
    "TOWER",
    "TOWER_TABLE_FILE",
    "ameriflux_site_list_path",
    "default_raw_directory",
    "default_source_root",
    "pool_input_list_path",
    "raw_path",
    "resolve_resolution",
    "tower_table_path",
]


#: The tower dim of the raw files: the AmeriFlux site identifier.
TOWER = "tower"

#: The raw files' time dim: a step's position on the local-standard-time axis.
TIME_INDEX = "time_index"

#: The first step start and the end of the last step of the raw files' axis, in
#: local standard time. A day of margin on each side of 2012-2024 means that no
#: UTC step of that period is lost when a tower's stamps are shifted by its
#: offset, whatever the offset between -24 and +24 hours.
RAW_START = pd.Timestamp("2011-12-31T00:00")
RAW_END = pd.Timestamp("2025-01-02T00:00")

#: The first step start and the end of the last step of the processed files'
#: axis, in UTC.
PROCESSED_START = pd.Timestamp("2012-01-01T00:00")
PROCESSED_END = pd.Timestamp("2025-01-01T00:00")


@dataclass(frozen=True)
class Resolution:
    """One of the source's two temporal resolutions."""

    name: str
    """``"half_hourly"`` or ``"hourly"``; part of the raw and processed file names."""

    source_code: str
    """The code in the source file name: ``"HH"`` or ``"HR"``."""

    step: pd.Timedelta
    """The step length."""

    @property
    def raw_file(self) -> str:
        """The converted raw file for this resolution."""
        return f"ameriflux_nee_{self.name}.nc"

    def raw_step_starts(self) -> pd.DatetimeIndex:
        """Every step start of the raw axis, local standard time, naive."""
        return pd.date_range(RAW_START, RAW_END - self.step, freq=self.step)

    def processed_step_starts(self) -> pd.DatetimeIndex:
        """Every step start of the processed axis, UTC, naive."""
        return pd.date_range(PROCESSED_START, PROCESSED_END - self.step, freq=self.step)

    def raw_offset_of_processed_start(self, utc_offset: float) -> int:
        """The raw index of the processed axis's first step, for a tower on this offset.

        A tower's standard-time stamp is UTC plus its offset, so the first
        processed step, which starts at :data:`PROCESSED_START` UTC, starts at
        ``PROCESSED_START + offset`` on the raw axis.

        Parameters
        ----------
        utc_offset:
            The tower's local standard time minus UTC, in hours.

        Raises
        ------
        ValueError
            If the offset is not finite, not a whole number of steps, or puts
            the processed axis outside the raw axis.
        """
        check_offset_is_finite(utc_offset)
        steps = (PROCESSED_START - RAW_START + pd.Timedelta(hours=utc_offset)) / self.step
        check_offset_is_whole_steps(utc_offset, steps, self)
        start = int(steps)
        check_offset_keeps_the_processed_axis_inside_the_raw_axis(utc_offset, start, self)
        return start


HALF_HOURLY = Resolution(name="half_hourly", source_code="HH", step=pd.Timedelta(minutes=30))
HOURLY = Resolution(name="hourly", source_code="HR", step=pd.Timedelta(minutes=60))

#: The resolutions, by name.
RESOLUTIONS: frozendict[str, Resolution] = frozendict({r.name: r for r in (HALF_HOURLY, HOURLY)})

#: The tower table, under ``data/raw/net_ecosystem_exchange/``; tracked.
TOWER_TABLE_FILE = "ameriflux_towers.csv"

#: AmeriFlux's site listing as downloaded beside the FLUXNET files; not tracked,
#: because it carries principal investigators' contact details.
AMERIFLUX_SITE_LIST_FILE = "ameri_sites.tsv"

#: The reanalysis's list of AmeriFlux towers added to the site pool; tracked.
POOL_INPUT_LIST_FILE = "Unmatched_Sites.csv"


def resolve_resolution(name: str) -> Resolution:
    """The resolution named *name*, or a ``KeyError`` listing those that exist."""
    check_resolution_is_known(name)
    return RESOLUTIONS[name]


def default_raw_directory() -> Path:
    """Where the storage-backed raw files are: ``data/raw/net_ecosystem_exchange/``.

    The converted time series and AmeriFlux's site listing, neither tracked.
    ``$SIPNET_CALIBRATION_DATA`` replaces ``data/`` when set.
    """
    return data_root() / "raw" / "net_ecosystem_exchange"


def default_source_root() -> Path:
    """Where the AmeriFlux FLUXNET files are: ``<raw directory>/fluxnet``.

    Present only on the SCC, as a symlink to the download.
    """
    return default_raw_directory() / "fluxnet"


def raw_path(resolution: Resolution | str, directory: Path | str | None = None) -> Path:
    """The converted raw file of a resolution: ``<directory>/ameriflux_nee_<name>.nc``."""
    resolution = resolve_resolution(resolution) if isinstance(resolution, str) else resolution
    base = Path(directory) if directory is not None else default_raw_directory()
    return base / resolution.raw_file


def ameriflux_site_list_path(directory: Path | str | None = None) -> Path:
    """AmeriFlux's site listing: ``<directory>/ameri_sites.tsv``."""
    base = Path(directory) if directory is not None else default_raw_directory()
    return base / AMERIFLUX_SITE_LIST_FILE


def tower_table_path(directory: Path | str | None = None) -> Path:
    """The tower table: ``<directory>/ameriflux_towers.csv``.

    It is tracked, so by default it is found from
    :func:`~sipnet_calibration.conventions.tracked_data_root`: in the checkout
    whatever ``$SIPNET_CALIBRATION_DATA`` says.
    """
    base = Path(directory) if directory is not None else _tracked_raw_directory()
    return base / TOWER_TABLE_FILE


def pool_input_list_path(directory: Path | str | None = None) -> Path:
    """The pool's list of added AmeriFlux towers: ``<directory>/Unmatched_Sites.csv``.

    Tracked, and found as :func:`tower_table_path` finds the tower table.
    """
    base = Path(directory) if directory is not None else _tracked_raw_directory()
    return base / POOL_INPUT_LIST_FILE


# ── private helpers ───────────────────────────────────────────────────────────


def _tracked_raw_directory() -> Path:
    """The tracked part of ``data/raw/net_ecosystem_exchange/``, in the checkout."""
    return tracked_data_root() / "raw" / "net_ecosystem_exchange"


# ── checks ────────────────────────────────────────────────────────────────────


def check_resolution_is_known(name: str) -> None:
    """A resolution name is one of :data:`RESOLUTIONS`."""
    if name not in RESOLUTIONS:
        raise KeyError(f"no resolution named {name!r}; pass one of {truncated(RESOLUTIONS)}.")


def check_offset_is_finite(utc_offset: float) -> None:
    """A tower's UTC offset is a finite number of hours."""
    if not math.isfinite(utc_offset):
        raise ValueError(
            f"a UTC offset of {utc_offset} h is not a clock; a tower with no recovered offset "
            "is excluded in the tower table."
        )


def check_offset_is_whole_steps(utc_offset: float, steps: float, resolution: Resolution) -> None:
    """A tower's UTC offset moves its stamps by a whole number of steps."""
    if steps != int(steps):
        raise ValueError(
            f"a UTC offset of {utc_offset} h is not a whole number of {resolution.name} "
            "steps, so the tower's stamps cannot be moved onto the UTC axis without splitting "
            "a step; exclude the tower in the tower table."
        )


def check_offset_keeps_the_processed_axis_inside_the_raw_axis(
    utc_offset: float, start: int, resolution: Resolution
) -> None:
    """A tower's UTC offset keeps the whole processed axis within the raw axis."""
    n_processed = len(resolution.processed_step_starts())
    if start < 0 or start + n_processed > len(resolution.raw_step_starts()):
        raise ValueError(
            f"a UTC offset of {utc_offset} h puts the processed axis outside the "
            f"{resolution.name} raw axis; an offset beyond a day is not a clock, so correct "
            "the tower table."
        )

"""Dimension names, file locations, the two time axes, and shared helpers.

The names of the dimensions and coordinates the package's files use, where
each file is expected on disk, the two regular time axes -- the raw files'
local-standard-time axis and the products' UTC axis -- and the small pieces
more than one module needs. Nothing here reads or writes anything.

Contents
--------
:data:`TOWER`, :data:`TIME_INDEX`, :data:`SITE`, :data:`TIME`, :data:`BOUNDS`, :data:`MEMBER`
    The dimension names, spelled once, and the time coordinate names beside
    them.
:class:`Resolution`, :data:`RESOLUTIONS`
    The two source resolutions, half-hourly and hourly, with their step and
    their raw and product time axes.
The path functions
    :func:`default_raw_dir`, :func:`default_source_root`, :func:`raw_path`,
    :func:`tower_table_path`, :func:`ameriflux_site_list_path`,
    :func:`pool_input_list_path`, :func:`default_product_dir` and
    :func:`product_path` say where each file is expected, all honoring
    ``$SIPNET_CALIBRATION_DATA``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sipnet_calibration import conventions

__all__ = [
    "AMERIFLUX_SITE_LIST_FILE",
    "BOUNDS",
    "HALF_HOURLY",
    "HOURLY",
    "MEMBER",
    "POOL_INPUT_LIST_FILE",
    "PRODUCT_END",
    "PRODUCT_START",
    "RAW_END",
    "RAW_START",
    "RESOLUTIONS",
    "Resolution",
    "SITE",
    "TIME",
    "TIME_BOUNDS",
    "TIME_INDEX",
    "TIME_STEP_LENGTH",
    "TIME_STEP_START",
    "TOWER",
    "TOWER_TABLE_FILE",
    "ameriflux_site_list_path",
    "default_product_dir",
    "default_raw_dir",
    "default_source_root",
    "pool_input_list_path",
    "product_path",
    "raw_path",
    "resolve_resolution",
    "tower_table_path",
]


#: The tower dimension of the raw files: the AmeriFlux site identifier.
TOWER = "tower"

#: The raw files' time dimension: a step's position on the local-standard-time axis.
TIME_INDEX = "time_index"

#: The site dimension of the products, carrying the 1-8000 identifier.
SITE = "site"

#: The products' time dimension and coordinate: the UTC end of each step.
TIME = "time"

#: The second dimension of ``time_bounds``.
BOUNDS = "bounds"

#: The ensemble member dimension, present only in a product from an ensemble source.
MEMBER = "member"

#: The products' time coordinates besides ``time``, named as pySIPNET names them.
TIME_STEP_START = "time_step_start"
TIME_STEP_LENGTH = "time_step_length"
TIME_BOUNDS = "time_bounds"

#: The first step start and the end of the last step of the raw files' axis, in
#: local standard time. A day of margin on each side of 2012-2024 means that no
#: UTC step of that period is lost when a tower's stamps are shifted by its
#: offset, whatever the offset between -24 and +24 hours.
RAW_START = pd.Timestamp("2011-12-31T00:00")
RAW_END = pd.Timestamp("2025-01-02T00:00")

#: The first step start and the end of the last step of the products' axis, in UTC.
PRODUCT_START = pd.Timestamp("2012-01-01T00:00")
PRODUCT_END = pd.Timestamp("2025-01-01T00:00")


@dataclass(frozen=True)
class Resolution:
    """One of the source's two temporal resolutions."""

    name: str
    """``"half_hourly"`` or ``"hourly"``; part of the raw file's and the products' names."""

    source_code: str
    """The code in the source file name: ``"HH"`` or ``"HR"``."""

    minutes: int
    """The step length."""

    @property
    def step(self) -> pd.Timedelta:
        """The step length as a ``Timedelta``."""
        return pd.Timedelta(minutes=self.minutes)

    @property
    def raw_file(self) -> str:
        """The converted raw file for this resolution."""
        return f"ameriflux_nee_{self.name}.nc"

    def raw_step_starts(self) -> pd.DatetimeIndex:
        """Every step start of the raw axis, local standard time, naive."""
        return pd.date_range(RAW_START, RAW_END - self.step, freq=self.step)

    def product_step_starts(self) -> pd.DatetimeIndex:
        """Every step start of the product axis, UTC, naive."""
        return pd.date_range(PRODUCT_START, PRODUCT_END - self.step, freq=self.step)

    def raw_offset_of_product_start(self, utc_offset_hours: float) -> int:
        """The raw index of the product's first step for a tower on this offset.

        A tower's standard-time stamp is UTC plus its offset, so the product's
        first step, which starts at :data:`PRODUCT_START` UTC, starts at
        ``PRODUCT_START + offset`` on the raw axis.

        Raises
        ------
        ValueError
            If the offset is not a whole number of steps, or puts the product
            window outside the raw axis.
        """
        minutes = (PRODUCT_START - RAW_START) / pd.Timedelta(minutes=1) + utc_offset_hours * 60
        steps = minutes / self.minutes
        if steps != np.floor(steps):
            raise ValueError(
                f"a UTC offset of {utc_offset_hours} h is not a whole number of "
                f"{self.minutes}-minute steps, so the tower's stamps cannot be moved "
                "onto the UTC axis without splitting a step"
            )
        start = int(steps)
        n_product = len(self.product_step_starts())
        if start < 0 or start + n_product > len(self.raw_step_starts()):
            raise ValueError(
                f"a UTC offset of {utc_offset_hours} h puts the product window outside the "
                "raw axis"
            )
        return start


HALF_HOURLY = Resolution(name="half_hourly", source_code="HH", minutes=30)
HOURLY = Resolution(name="hourly", source_code="HR", minutes=60)

#: The resolutions, by name.
RESOLUTIONS: dict[str, Resolution] = {r.name: r for r in (HALF_HOURLY, HOURLY)}

#: The tower table, under ``data/raw/net_ecosystem_exchange/``; tracked.
TOWER_TABLE_FILE = "ameriflux_towers.csv"

#: AmeriFlux's site listing as downloaded beside the FLUXNET files; untracked,
#: because it carries principal investigators' contact details.
AMERIFLUX_SITE_LIST_FILE = "ameri_sites.tsv"

#: The reanalysis's list of AmeriFlux towers added to the site pool; tracked.
POOL_INPUT_LIST_FILE = "Unmatched_Sites.csv"


def resolve_resolution(name: str) -> Resolution:
    """The resolution named *name*, or a ``KeyError`` listing those that exist."""
    try:
        return RESOLUTIONS[name]
    except KeyError:
        raise KeyError(f"No resolution {name!r}. Known: {sorted(RESOLUTIONS)}") from None


def default_raw_dir() -> Path:
    """Where the raw files live: ``data/raw/net_ecosystem_exchange/``."""
    return conventions.data_root() / "raw" / "net_ecosystem_exchange"


def default_source_root() -> Path:
    """Where the AmeriFlux FLUXNET files are expected: ``<raw dir>/fluxnet``.

    Present only on the SCC, as a symlink to the download.
    """
    return default_raw_dir() / "fluxnet"


def raw_path(resolution: Resolution | str, directory: Path | str | None = None) -> Path:
    """The converted raw file of a resolution: ``<directory>/ameriflux_nee_<name>.nc``."""
    resolution = resolve_resolution(resolution) if isinstance(resolution, str) else resolution
    base = Path(directory) if directory else default_raw_dir()
    return base / resolution.raw_file


def tower_table_path(directory: Path | str | None = None) -> Path:
    """The tower table: ``<directory>/ameriflux_towers.csv``."""
    return (Path(directory) if directory else default_raw_dir()) / TOWER_TABLE_FILE


def ameriflux_site_list_path(directory: Path | str | None = None) -> Path:
    """AmeriFlux's site listing: ``<directory>/ameri_sites.tsv``."""
    return (Path(directory) if directory else default_raw_dir()) / AMERIFLUX_SITE_LIST_FILE


def pool_input_list_path(directory: Path | str | None = None) -> Path:
    """The pool's list of added AmeriFlux towers: ``<directory>/Unmatched_Sites.csv``."""
    return (Path(directory) if directory else default_raw_dir()) / POOL_INPUT_LIST_FILE


def default_product_dir() -> Path:
    """Where the processed products live: ``data/processed/net_ecosystem_exchange/``."""
    return conventions.data_root() / "processed" / "net_ecosystem_exchange"


def product_path(product_name: str, directory: Path | str | None = None) -> Path:
    """A product's file: ``<directory>/<product_name>.nc``."""
    return (Path(directory) if directory else default_product_dir()) / f"{product_name}.nc"


#: CF attributes for the ``site`` coordinate.
_SITE_ATTRS = {
    "long_name": "Model site identifier",
    "comment": "The handed-down 1-8000 identifier of the site table; never renumbered.",
}

_LON_ATTRS = {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"}

_LAT_ATTRS = {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"}


def _utc_timestamp() -> str:
    """Now, as the ISO 8601 string the file attributes carry."""
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

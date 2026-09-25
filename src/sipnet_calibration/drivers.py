"""The data model for the meteorological drivers, read from the raw files.

Overview
--------
This module defines how the ERA5 driver ensemble is represented -- its
dimensions, coordinates, variables and attributes -- and provides the functions
that read it into that form. Everything about a single ``.clim`` file belongs
to pySIPNET, which owns the SIPNET climate format:
:class:`pysipnet.climate.ClimateDrivers` parses it, validates it, names its
columns and builds its time axis. What this module adds is the ensemble: the
``ERA5_<site>_<member>`` directory layout, the stacking of many files into
``(member, site, time)``, the site coordinates, and a few checks on the source
that pySIPNET has no reason to make.

Unlike the site table and the constraints, the drivers have **no
processed file**. SIPNET reads the raw ``.clim`` text directly (pySIPNET
symlinks it into the run directory), and the full ensemble is 80,000 files, so
rewriting it into a store would create a large cache that the model never
reads. Instead :func:`load_drivers` reads the raw files for the sites a caller
names and returns the canonical form in memory. The dependency still runs one
way::

    raw/drivers/ERA5_<site>_<member>/ERA5.<member>.<start>.<end>.clim
      -> pysipnet.climate.ClimateDrivers      one file
      -> this module          load_drivers() -> xarray.Dataset

with ``processed/sites/sites.csv`` joined on for the site coordinates. Anyone
wanting a cached subset writes it themselves,
``load_drivers(...).to_zarr(path)``, and reads it back with ``xarray``; nothing
here depends on such a cache existing. ``data/README.md`` documents the source
data and the open questions about it.

Input data
----------
``data/raw/drivers/``
    One directory per site and ensemble member, ``ERA5_<site>_<member>``,
    holding one file ``ERA5.<member>.<start>.<end>.clim``. ``<site>`` is the
    1-8000 site identifier and ``<member>`` the source's 1-based member index.
    :func:`default_drivers_root` says where the directory is expected to be.
    Each file is a SIPNET climate file, read by pySIPNET in whichever layout it
    has, and refused by pySIPNET if it fails its validation -- including a
    ``time`` column that disagrees with the declared step lengths. The ERA5
    files as generated fail it for that reason (``data/README.md`` Note 15), so
    until they are corrected :func:`load_drivers` refuses them.

``data/processed/sites/sites.csv``
    The site table, for the ``lon``/``lat`` coordinates and to confirm that the
    requested sites exist. Read through
    :func:`sipnet_calibration.sites.load_sites`; only its ``site_id``, ``lon``
    and ``lat`` columns are used, and ``site_id`` must be unique.

Data model
----------
:func:`load_drivers` returns an ``xarray.Dataset`` shaped as follows.

**Dimensions**: ``member``, ``site``, ``time``, and ``bounds`` for
``time_bounds``.

**Data variables**, all ``float64`` on ``(member, site, time)``: the eight
value columns of the climate file, under pySIPNET's registry names, listed in
:data:`DRIVER_VARIABLES` -- ``air_temperature``, ``soil_temperature``,
``photosynthetically_active_radiation``, ``precipitation``,
``vapor_pressure_deficit``, ``soil_vapor_pressure_deficit``,
``vapor_pressure`` and ``wind_speed``. Each carries the attributes pySIPNET
gives it -- ``units``, ``long_name``, ``description``, ``kind``,
``time_reference``, ``cell_methods`` and the rest -- plus
``units_provenance``: the units are the ones the SIPNET format documents, not
units the producer has confirmed.

``photosynthetically_active_radiation`` and ``precipitation`` also carry
``n_values_below_zero``, and ``vapor_pressure_deficit``,
``soil_vapor_pressure_deficit`` and ``wind_speed`` carry
``n_values_not_positive``. The source files hold small negative excursions
around zero, and exact zeros, which SIPNET clamps for the vapor pressure
deficit and wind speed; they are read through unchanged and counted, so that
nobody has to rediscover that radiation above zero is not a daylight test.

With ``allow_missing=True`` there is one more variable, ``bool`` on
``(member, site)``::

    driver_present(member, site)    whether a file existed for the pair

and the eight drivers are ``NaN`` where it is ``False``. Without the flag every
requested pair must exist, so the variable is not written.

**Coordinates**

======================= ================== ====================================
Name                    Dims               Meaning
======================= ================== ====================================
``member``              ``member``         0-based ``int16``, in ascending order
``source_member_index`` ``member``         the 1-based index in the directory
                                           name
``site``                ``site``           handed-down ``int32`` site id,
                                           ascending
``lon``, ``lat``        ``site``           from the site table, ``float64``
``time``                ``time``           ``datetime64[ns]``, the end of each
                                           step
``time_step_start``     ``time``           the start of each step
``time_step_length``    ``time``           each step's declared length,
                                           ``timedelta64[ns]``
``time_bounds``         ``time, bounds``   ``[time_step_start, time]``
======================= ================== ====================================

**Time.** The time coordinates are pySIPNET's, taken unchanged from
:attr:`pysipnet.climate.ClimateDrivers.xarray`: the same axis, with the same
attributes, that a SIPNET run on the file has for its output. Its
``year``/``day_of_year``/``hour_of_day`` row labels are not kept, since
``time_step_start`` is the same instant. A row's labels are the start of its
step on whatever clock the drivers use, and ``time`` is the step's end. The
clock is ``time_zone``, on ``time`` and on the Dataset, which is
``"undeclared"`` unless the caller declares one.

These are the semantics of SIPNET's format, which the variable attributes
state. They are true of a file that follows the format. The ERA5 files do not
(``data/README.md`` Note 16): their radiation and precipitation cover the step
ending at the label rather than starting there, and their other forcing
columns are instantaneous at the label rather than means over the step.

**Attributes** on the dataset: pySIPNET's -- ``Conventions``,
``time_convention``, ``time_zone``, ``time_axis_source`` and
``time_step_length_source`` among them -- and ``title``, ``source_root``,
``source_layout``, ``member_source = "met"``, ``member_correspondence``,
``n_sites``, ``n_members`` and ``coverage`` (``"complete"`` or ``"gaps"``).

**Missing values.** There are none in the source. A ``NaN`` appears only under
``allow_missing=True``, for a whole ``(member, site)`` pair whose file is
absent, and ``driver_present`` says which.

Functions
---------
:func:`load_drivers`
    Read the drivers for the sites named, checking every file on the way, and
    return the Dataset above.

:func:`driver_fields`
    Split the Dataset into fields -- one ``DataArray`` per variable
    with dims ``(member, site, time)`` and its own units. This is the view the
    plotting layer wants.

:func:`read_driver_file`
    Read one ``.clim`` file through pySIPNET and apply this module's own check
    on its values. The building block :func:`load_drivers` is made of, public
    so that tests and one-off surveys read a file exactly as the loader does.

:func:`available_members`
    Which member indices have a directory for a given site.

:func:`driver_file`
    The path of the one ``.clim`` file for a site and member.

:func:`default_drivers_root`
    Where the raw directory is expected to be, honoring
    ``$SIPNET_CALIBRATION_DATA``.

Notes
-----
**Why pySIPNET reads the files.** The climate format, its validation and the
meaning of its time columns are pySIPNET's, and SIPNET runs on exactly what
pySIPNET reads. A second parser here would be a second definition of the same
format, free to disagree with the one the model is run through. So nothing here
parses ``.clim`` text or builds a time axis; a file pySIPNET refuses is refused
here, with pySIPNET's reason.

**Why a reader and not a store.** SIPNET consumes the raw text, so a store
would be a second copy that only the analysis side reads, and the full
ensemble is hundreds of gigabytes of text. Reading direct means what is
plotted is read from the exact file the model ran on. The cost is that reads
are site-major only: a site's whole record is one file, but one timestep across
the pool means reading every file. Calibration and the per-site figures need
the former.

**Member indices.** ``member`` is 0-based to match every other product;
``source_member_index`` keeps the 1-based file index beside it so the mapping
to a directory is never guesswork. Whether driver member *i* corresponds to
initial-condition member *i* is not established (open question 12 in
``data/README.md``), and ``member_correspondence`` says so.

Usage
-----
Name the sites, get the canonical form::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.sites import load_sites, select_sites

    sites = select_sites(load_sites(), bbox=(-125, 24, -66, 50), sample=20, seed=0)
    drivers = load_drivers(sites["site_id"])          # every member present

    drivers["air_temperature"].dims                   # ('member', 'site', 'time')
    drivers["precipitation"].attrs["kind"]            # 'timestep_total'
    drivers["time"].attrs["time_zone"]                # 'undeclared'

    # Two members only, and tolerate sites that lack a file for one of them.
    partial = load_drivers([1, 27], members=[1, 2], allow_missing=True)
    partial["driver_present"].values

For plotting, take the per-variable view::

    fields = driver_fields(drivers)
    fields["photosynthetically_active_radiation"].attrs["units"]   # 'mol m-2'

A cached subset, if a workflow wants one, is the caller's business::

    drivers.to_zarr(path)
    import xarray as xr
    xr.open_zarr(path)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.climate import ClimateDrivers, normalize_time_zone
from pysipnet.dataset import unfilled_coordinates
from pysipnet.variables import CLIMATE_VARIABLES

from sipnet_calibration import conventions
from sipnet_calibration.fields import TIME_COORDS
from sipnet_calibration.sites import load_sites

__all__ = [
    "DRIVER_DIRECTORY_TEMPLATE",
    "DRIVER_FILE_GLOB",
    "DRIVER_PRESENT",
    "DRIVER_VARIABLES",
    "MEMBER_SOURCE",
    "NEGATIVE_TOLERANCE",
    "UNITS_PROVENANCE",
    "available_members",
    "default_drivers_root",
    "driver_fields",
    "driver_file",
    "load_drivers",
    "read_driver_file",
]

#: The driver variables, under pySIPNET's names, in climate-file column order:
#: every column of pySIPNET's climate registry outside its ``time`` group, which
#: becomes the time axis.
DRIVER_VARIABLES: tuple[str, ...] = tuple(
    spec.name for spec in CLIMATE_VARIABLES if spec.group != "time"
)

#: Why the units are what they are. Recorded on every variable so that no
#: consumer can take the units as confirmed.
UNITS_PROVENANCE = (
    "The units the SIPNET climate-file format documents for this column "
    "(pySIPNET's climate registry, and the conversions in sipnet.c), which is "
    "what SIPNET assumes when it reads the file. Whether the producer wrote the "
    "values in these units has not been confirmed; the magnitudes are "
    "consistent with them, which is evidence and not confirmation."
)

#: Which ensemble the ``member`` coordinate indexes. Member indices are
#: meaningful only within one source.
MEMBER_SOURCE = "met"

#: Name of the presence variable written under ``allow_missing=True``.
DRIVER_PRESENT = "driver_present"

#: Per-site-and-member directory under the drivers root, and the file inside
#: it, ``ERA5.<member>.<start>.<end>.clim``. The glob accepts any member and
#: any dates so that a file whose name disagrees with its directory is reported
#: as the mismatch it is rather than as a missing file; the reader checks both
#: against the directory and the data.
DRIVER_DIRECTORY_TEMPLATE = "ERA5_{site}_{member}"
DRIVER_FILE_GLOB = "ERA5.*.clim"

#: How far below zero photosynthetically active radiation and precipitation may
#: go before a file is refused. The source holds excursions of order 1e-5 and
#: 1e-15 that read as generator noise around zero; anything larger is a
#: different problem.
NEGATIVE_TOLERANCE = 1e-4


def default_drivers_root() -> Path:
    """Where the raw driver directory is expected to be.

    ``$SIPNET_CALIBRATION_DATA/raw/drivers`` when that variable is set, and
    otherwise ``data/raw/drivers`` under this checkout. Experiments name their
    paths in ``config.py``.
    """
    return conventions.data_root() / "raw" / "drivers"


def driver_file(root: Path | str, site: int, member: int) -> Path:
    """The ``.clim`` file for one site and one source member index.

    Parameters
    ----------
    root:
        The drivers root, laid out as :data:`DRIVER_DIRECTORY_TEMPLATE`.
    site:
        Site identifier, 1-8000.
    member:
        The source's 1-based member index, as in the directory name.

    Returns
    -------
    pathlib.Path
        The single file matching :data:`DRIVER_FILE_GLOB` in the pair's
        directory.

    Raises
    ------
    FileNotFoundError
        If the directory, or a file matching the glob inside it, is absent.
    ValueError
        If more than one file matches, since the layout promises exactly one.
    """
    name = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member=int(member))
    directory = Path(root) / name
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no driver directory for site {site} member {member}: {directory}"
        )
    matches = sorted(directory.glob(DRIVER_FILE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"{directory} holds no file matching {DRIVER_FILE_GLOB!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{directory} holds {len(matches)} files matching {DRIVER_FILE_GLOB!r}; "
            f"the layout promises one: {[m.name for m in matches]}"
        )
    return matches[0]


def available_members(root: Path | str, site: int) -> tuple[int, ...]:
    """The source member indices that have a directory for *site*.

    Parameters
    ----------
    root:
        The drivers root.
    site:
        Site identifier.

    Returns
    -------
    tuple of int
        1-based member indices in ascending order, possibly empty. Only the
        directory's existence is consulted; whether the file inside it is
        present and well formed is :func:`driver_file` and
        :func:`read_driver_file`'s business. A directory whose name is not
        exactly the template for its numbers, ``ERA5_3_01`` say, is ignored,
        since :func:`driver_file` could not find it either.
    """
    root = Path(root)
    members = []
    pattern = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member="*")
    for directory in root.glob(pattern):
        parsed = _site_member_from_directory(directory.name)
        if parsed is None or not directory.is_dir():
            continue
        canonical = DRIVER_DIRECTORY_TEMPLATE.format(site=parsed[0], member=parsed[1])
        if parsed[0] == int(site) and directory.name == canonical:
            members.append(parsed[1])
    return tuple(sorted(members))


def read_driver_file(path: Path | str, *, time_zone: str | None = None) -> ClimateDrivers:
    """Read one ``.clim`` file through pySIPNET and check its values.

    Parameters
    ----------
    path:
        The file to read.
    time_zone:
        The clock the file's labels are on, ``"UTC"`` or a fixed offset such
        as ``"UTC-07:00"``, passed to pySIPNET. ``None`` leaves it undeclared.

    Returns
    -------
    pysipnet.climate.ClimateDrivers
        The file, read and validated by pySIPNET.

    Raises
    ------
    ValueError
        If pySIPNET refuses the file, with pySIPNET's reason; or if
        photosynthetically active radiation or precipitation falls further
        below zero than :data:`NEGATIVE_TOLERANCE`. The message names the file.
    OSError
        If the file cannot be opened, as pySIPNET raises it:
        ``FileNotFoundError`` for a missing file or a dangling link, among
        others.
    """
    path = Path(path)
    try:
        climate = ClimateDrivers.from_file(path, time_zone=time_zone)
    except ValueError as error:
        raise ValueError(f"{path}: pySIPNET refused the file: {error}") from error
    _check_negative_excursions_bounded(climate.pandas, path)
    return climate


def load_drivers(
    sites: Iterable[int],
    *,
    members: Iterable[int] | None = None,
    root: Path | str | None = None,
    sites_table: pd.DataFrame | None = None,
    allow_missing: bool = False,
    time_zone: str | None = None,
) -> xr.Dataset:
    """Read the drivers for the given sites into the canonical form.

    Parameters
    ----------
    sites:
        Site identifiers to read, any iterable of integers. Returned in
        ascending order whatever order they are given in, duplicates dropped.
        Every one must be in the site table.
    members:
        Source member indices (1-based, as in the directory names) to read,
        any iterable of integers, likewise sorted and de-duplicated. ``None``
        means every member that has a directory for any of the requested
        sites.
    root:
        The drivers root. Defaults to :func:`default_drivers_root`.
    sites_table:
        The site table, as :func:`sipnet_calibration.sites.load_sites` returns
        it. Loaded from its default location when ``None``. Only ``site_id``,
        ``lon`` and ``lat`` are read, and ``site_id`` must be unique.
    allow_missing:
        What to do about a ``(site, member)`` pair with no file. ``False``, the
        default, raises, because a missing driver member that became ``NaN``
        would propagate silently through any statistic over members. ``True``
        fills the pair with ``NaN`` and adds :data:`DRIVER_PRESENT`. At least
        one requested pair must have a file either way.
    time_zone:
        The clock the files' labels are on, declared to pySIPNET for every
        file; see :func:`read_driver_file`. ``None`` leaves it undeclared.

    Returns
    -------
    xarray.Dataset
        The `Data model`_ described in the module docstring: the eight
        :data:`DRIVER_VARIABLES` on ``(member, site, time)``, ``float64``, with
        pySIPNET's time coordinates, ``lon``/``lat`` on ``site`` and
        ``source_member_index`` on ``member``.

    Raises
    ------
    FileNotFoundError
        If the root does not exist; if *members* is ``None`` and no requested
        site has a driver directory; if no requested pair has a file at all;
        or if a requested pair has no file and *allow_missing* is ``False``.
    ValueError
        If *time_zone* is neither ``"UTC"`` nor a fixed UTC offset; if *sites*
        or *members* is empty, or holds anything but positive whole numbers,
        discovered members included; if the site table lacks ``site_id``, ``lon`` or ``lat`` or
        repeats a ``site_id``; if a site is not in the site table; if a pair's
        directory holds more than one ``.clim`` file; if a file fails
        :func:`read_driver_file`, its name does not follow the template, the
        directory and file-name members disagree, or the dates in the file name
        do not match its first and last day; or if two files are not on one
        time axis, since the time coordinates are shared by every file.
    """
    root = Path(root) if root is not None else default_drivers_root()
    if not root.is_dir():
        raise FileNotFoundError(f"drivers root {root} is not a directory")
    time_zone = normalize_time_zone(time_zone)

    site_ids = _site_ids(sites)
    table = sites_table if sites_table is not None else load_sites()
    _check_site_table_is_usable(table)
    _check_sites_are_in_the_site_table(site_ids, table)

    member_ids = _member_ids(members, root=root, sites=site_ids)

    paths, present = _locate_files(root, sites=site_ids, members=member_ids)
    if not present.any():
        raise FileNotFoundError(
            f"no driver files under {root} for sites {site_ids.tolist()} and "
            f"members {member_ids.tolist()}"
        )
    if not allow_missing:
        _check_members_complete(present, sites=site_ids, members=member_ids, root=root)

    arrays, reference = _read_all(paths, present, time_zone=time_zone)
    return _assemble(
        arrays,
        present=present,
        reference=reference,
        sites=site_ids,
        members=member_ids,
        table=table,
        root=root,
        allow_missing=allow_missing,
    )


def driver_fields(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """One ``DataArray`` per driver variable, in :data:`DRIVER_VARIABLES` order.

    Each field has dims ``(member, site, time)``, is named for its variable,
    keeps that variable's attributes, and carries pySIPNET's
    :data:`sipnet_calibration.fields.TIME_COORDS` with ``lon``/``lat`` and
    ``source_member_index`` as non-dimension coordinates -- the field
    shape, and the same time coordinates a model field has. This is the view
    facet-by-variable consumes, matching
    :func:`sipnet_calibration.constraints.constraint_fields`.

    Parameters
    ----------
    dataset:
        As returned by :func:`load_drivers`.

    Returns
    -------
    dict
        Keyed by variable name. :data:`DRIVER_PRESENT`, if present, is not a
        field and is left out.

    Raises
    ------
    ValueError
        If any of :data:`DRIVER_VARIABLES` is absent from *dataset*.

    Notes
    -----
    ``time_bounds`` does not ride on a field, its ``bounds`` dimension being no
    field dimension, so ``time``'s ``bounds`` attribute is dropped with it, as
    :func:`sipnet_calibration.fields.from_sipnet_output` does for a model
    field.
    """
    missing = [name for name in DRIVER_VARIABLES if name not in dataset.data_vars]
    if missing:
        raise ValueError(
            f"dataset is missing driver variables {missing}; found "
            f"{sorted(dataset.data_vars)}"
        )
    fields = {}
    for name in DRIVER_VARIABLES:
        field = dataset[name].copy(deep=False)
        field["time"].attrs = {
            key: value for key, value in field["time"].attrs.items() if key != "bounds"
        }
        fields[name] = field
    return fields


# ── supporting helpers ────────────────────────────────────────────────────────

_DIRECTORY_PATTERN = re.compile(r"^ERA5_(\d+)_(\d+)$")
_FILE_PATTERN = re.compile(r"^ERA5\.(\d+)\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.clim$")

#: Variables whose sub-zero or non-positive values are counted, and the
#: attribute each count is written to.
_COUNT_BELOW_ZERO = ("photosynthetically_active_radiation", "precipitation")
_COUNT_NOT_POSITIVE = ("vapor_pressure_deficit", "soil_vapor_pressure_deficit", "wind_speed")

#: The time coordinates the Dataset takes from pySIPNET: the ones a field
#: keeps, and the CF bounds pair that only a Dataset can carry.
_DATASET_TIME_COORDS = (*TIME_COORDS, "time_bounds")


def _site_member_from_directory(name: str) -> tuple[int, int] | None:
    """``(site, member)`` from an ``ERA5_<site>_<member>`` name, else ``None``."""
    match = _DIRECTORY_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _dates_from_file_name(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The ``<start>`` and ``<end>`` dates embedded in a ``.clim`` file name."""
    match = _FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: file name does not follow ERA5.<member>.<start>.<end>.clim"
        )
    try:
        return pd.Timestamp(match.group(2)), pd.Timestamp(match.group(3))
    except ValueError as error:
        raise ValueError(f"{path}: file name carries an invalid date: {error}") from error


def _site_ids(sites: Iterable[int]) -> np.ndarray:
    """Requested sites as a sorted, de-duplicated ``int32`` array."""
    return _positive_integers(sites, name="site identifiers", dtype=np.int32)


def _member_ids(
    members: Iterable[int] | None, *, root: Path, sites: np.ndarray
) -> np.ndarray:
    """Requested members as a sorted ``int16`` array, discovered when ``None``."""
    if members is None:
        found: set[int] = set()
        for site in sites:
            found.update(available_members(root, int(site)))
        if not found:
            raise FileNotFoundError(
                f"no driver directories under {root} for sites {sites.tolist()}"
            )
        members = sorted(found)
    return _positive_integers(
        members,
        name="member indices (the source's 1-based directory indices)",
        dtype=np.int16,
    )


def _positive_integers(values: Iterable[int], *, name: str, dtype) -> np.ndarray:
    """*values* as a sorted, de-duplicated array of *dtype*, or a clear error.

    Strings, booleans, non-whole floats, non-finite values, non-positive
    values and anything that would wrap when narrowed to *dtype* are refused,
    since each would otherwise resolve to a plausible-looking wrong directory.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(
            f"{name} must be an iterable of integers, got {type(values).__name__}"
        )
    array = np.asarray(list(values))
    if array.size == 0:
        raise ValueError(f"no {name} requested")
    if array.dtype.kind == "f":
        if np.any(~np.isfinite(array)) or np.any(array != np.floor(array)):
            raise ValueError(f"{name} must be whole numbers, found non-integer values")
    elif array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers, got {array.dtype}")
    array = np.unique(array.astype(np.int64))
    limit = np.iinfo(dtype).max
    if np.any(array < 1) or np.any(array > limit):
        bad = array[(array < 1) | (array > limit)]
        raise ValueError(f"{name} must lie within 1..{limit}, found {bad[:5].tolist()}")
    return array.astype(dtype)


def _locate_files(
    root: Path, *, sites: np.ndarray, members: np.ndarray
) -> tuple[dict[tuple[int, int], Path], np.ndarray]:
    """Paths for every ``(member, site)`` pair that has one, and a presence mask."""
    present = np.zeros((members.size, sites.size), dtype=bool)
    paths: dict[tuple[int, int], Path] = {}
    for j, site in enumerate(sites):
        for i, member in enumerate(members):
            try:
                paths[(i, j)] = driver_file(root, int(site), int(member))
            except FileNotFoundError:
                continue
            present[i, j] = True
    return paths, present


def _read_all(
    paths: dict[tuple[int, int], Path], present: np.ndarray, *, time_zone: str | None
) -> tuple[dict[str, np.ndarray], xr.Dataset]:
    """Read every located file into ``(member, site, time)`` arrays.

    The first file read supplies the time axis; every later file is checked to
    be on the same one before its values are copied in. Cells with no file stay
    ``NaN``. Returns the arrays and the first file's pySIPNET Dataset, whose
    time coordinates and attributes the result takes.
    """
    reference: xr.Dataset | None = None
    reference_path: Path | None = None
    arrays: dict[str, np.ndarray] = {}

    for (i, j), path in sorted(paths.items(), key=lambda item: (item[0][1], item[0][0])):
        dataset = read_driver_file(path, time_zone=time_zone).xarray
        site, member = _site_member_from_directory(path.parent.name)
        _check_file_name_matches_contents(path, dataset, member=member)
        if reference is None:
            reference, reference_path = dataset, path
            shape = present.shape + (dataset.sizes["time"],)
            arrays = {name: np.full(shape, np.nan) for name in DRIVER_VARIABLES}
        else:
            _check_time_axes_identical(reference, dataset, reference_path=reference_path, path=path)
        for name in DRIVER_VARIABLES:
            arrays[name][i, j, :] = dataset[name].to_numpy()
    assert reference is not None
    return arrays, reference


def _assemble(
    arrays: dict[str, np.ndarray],
    *,
    present: np.ndarray,
    reference: xr.Dataset,
    sites: np.ndarray,
    members: np.ndarray,
    table: pd.DataFrame,
    root: Path,
    allow_missing: bool,
) -> xr.Dataset:
    """Put the arrays into the Dataset the module docstring describes."""
    dims = ("member", "site", "time")
    coordinates = table.set_index("site_id").loc[sites]
    data_vars = {}
    for name in DRIVER_VARIABLES:
        values = arrays[name]
        attrs = {**reference[name].attrs, "units_provenance": UNITS_PROVENANCE}
        observed = values[present]
        if name in _COUNT_BELOW_ZERO:
            attrs["n_values_below_zero"] = int(np.count_nonzero(observed < 0))
        if name in _COUNT_NOT_POSITIVE:
            attrs["n_values_not_positive"] = int(np.count_nonzero(observed <= 0))
        data_vars[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    if allow_missing:
        data_vars[DRIVER_PRESENT] = xr.DataArray(
            present,
            dims=("member", "site"),
            attrs={
                "long_name": "Whether a driver file existed for the member and site",
                "comment": "The eight driver variables are NaN where this is False.",
            },
        )

    dataset = xr.Dataset(
        data_vars,
        coords={
            "member": np.arange(members.size, dtype=np.int16),
            "source_member_index": ("member", members.astype(np.int16)),
            "site": sites.astype(np.int32),
            "lon": ("site", coordinates["lon"].to_numpy(np.float64)),
            "lat": ("site", coordinates["lat"].to_numpy(np.float64)),
            **{name: reference[name].variable for name in _DATASET_TIME_COORDS},
        },
    )
    dataset["member"].attrs = {
        "long_name": "Ensemble member",
        "comment": (
            "0-based, meaningful only within this source; source_member_index "
            "is the 1-based index in the directory name."
        ),
    }
    dataset["source_member_index"].attrs = {
        "long_name": "Member index in the source directory name (1-based)"
    }
    dataset["site"].attrs = {
        "long_name": "Model site identifier",
        "comment": "The handed-down 1-8000 identifier; never renumbered.",
    }
    dataset.attrs = {
        **reference.attrs,
        "title": "ERA5 meteorological drivers in SIPNET climate-file form",
        "source_root": str(root),
        "source_layout": f"{DRIVER_DIRECTORY_TEMPLATE}/{DRIVER_FILE_GLOB}",
        "member_source": MEMBER_SOURCE,
        "member_correspondence": (
            "Not established. Whether driver member i corresponds to "
            "initial-condition or NEE member i is open question 12 in "
            "data/README.md; nothing here assumes it does."
        ),
        "n_sites": int(sites.size),
        "n_members": int(members.size),
        "coverage": "complete" if present.all() else "gaps",
    }
    return unfilled_coordinates(dataset)


# ── checks ────────────────────────────────────────────────────────────────────


def _check_negative_excursions_bounded(frame: pd.DataFrame, path: Path) -> None:
    """Radiation and precipitation never fall below ``-NEGATIVE_TOLERANCE``.

    Small negatives are known and read through; a large one would be a
    different kind of problem and is refused.
    """
    for name in _COUNT_BELOW_ZERO:
        values = frame[name].to_numpy()
        low = values < -NEGATIVE_TOLERANCE
        if low.any():
            raise ValueError(
                f"{path}: {int(low.sum())} {name} value(s) below "
                f"-{NEGATIVE_TOLERANCE:g}, the lowest {values.min():.4g}. Small "
                "negative excursions around zero are known; these are not small."
            )


def _check_file_name_matches_contents(path: Path, dataset: xr.Dataset, *, member: int) -> None:
    """The directory's member agrees with the file name, and the dates with the data.

    The member index appears in both the directory and the file name and the
    two must agree; the ``<start>`` and ``<end>`` dates in the file name must
    be the days the first and last steps start on, as the drivers label them.
    """
    match = _FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"{path}: file name does not follow ERA5.<member>.<start>.<end>.clim"
        )
    if int(match.group(1)) != member:
        raise ValueError(
            f"{path}: the file name says member {int(match.group(1))}, the "
            f"directory says member {member}"
        )
    start, end = _dates_from_file_name(path)
    starts = pd.DatetimeIndex(dataset["time_step_start"].values)
    first, last = starts[0].normalize(), starts[-1].normalize()
    if (start, end) != (first, last):
        raise ValueError(
            f"{path}: the file name covers {start.date()} to {end.date()} but the "
            f"data runs {first.date()} to {last.date()}"
        )


def _check_time_axes_identical(
    reference: xr.Dataset, dataset: xr.Dataset, *, reference_path: Path, path: Path
) -> None:
    """Two files are on one time axis: the same step starts and lengths.

    The time coordinates are taken from the first file read and applied to
    all of them, which is sound only if every file's axis is the same.
    """
    if dataset.sizes["time"] != reference.sizes["time"]:
        raise ValueError(
            f"{path} has {dataset.sizes['time']} steps where {reference_path} has "
            f"{reference.sizes['time']}; every file read together must share one "
            "time axis"
        )
    for name in ("time_step_start", "time_step_length"):
        a = reference[name].to_numpy()
        b = dataset[name].to_numpy()
        if not np.array_equal(a, b):
            first = int(np.flatnonzero(a != b)[0])
            raise ValueError(
                f"{path}: {name} differs from {reference_path} first at step "
                f"{first} ({b[first]!r} against {a[first]!r}); every file read "
                "together must share one time axis"
            )


def _check_site_table_is_usable(table: pd.DataFrame) -> None:
    """The site table has the columns read here, and one row per site."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"sites_table must be a DataFrame, got {type(table).__name__}")
    missing = [column for column in ("site_id", "lon", "lat") if column not in table.columns]
    if missing:
        raise ValueError(
            f"the site table lacks column(s) {missing}; pass it as load_sites() "
            "returns it, with site_id as a column rather than the index"
        )
    duplicated = table["site_id"][table["site_id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(
            f"the site table repeats site id(s) {sorted(set(duplicated.tolist()))[:5]}"
        )


def _check_sites_are_in_the_site_table(sites: np.ndarray, table: pd.DataFrame) -> None:
    """Every requested site exists in the site table, so it has coordinates."""
    unknown = sorted(set(sites.tolist()) - set(table["site_id"].tolist()))
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested site(s) are not in the site table, for "
            f"example {unknown[:10]}"
        )


def _check_members_complete(
    present: np.ndarray, *, sites: np.ndarray, members: np.ndarray, root: Path
) -> None:
    """Every requested ``(site, member)`` pair has a file, unless gaps are allowed.

    The message lists the missing pairs, and says that ``allow_missing=True``
    reads the rest with ``NaN`` in their place.
    """
    if present.all():
        return
    missing = [
        (int(sites[j]), int(members[i]))
        for i, j in zip(*np.nonzero(~present), strict=True)
    ]
    shown = ", ".join(f"site {s} member {m}" for s, m in missing[:10])
    more = f", and {len(missing) - 10} more" if len(missing) > 10 else ""
    raise FileNotFoundError(
        f"{len(missing)} requested (site, member) pair(s) have no driver file "
        f"under {root}: {shown}{more}. Pass allow_missing=True to read the rest "
        "with NaN in their place and a driver_present array saying which."
    )

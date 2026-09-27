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
``(driver_member, site, time)``, the site coordinates, and a few checks on the
source that pySIPNET has no reason to make.

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

**Dimensions**: ``driver_member``
(:data:`~sipnet_calibration.conventions.DRIVER_MEMBER`, the drivers' own
ensemble, a batch dim), ``site``, ``time``, and ``bounds`` for
``time_bounds``.

**Data variables**, all ``float64`` on ``(driver_member, site, time)``: the
eight value columns of the climate file, under pySIPNET's registry names,
listed in :data:`DRIVER_VARIABLE_NAMES`. Each carries the attributes pySIPNET
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
``(driver_member, site)``::

    driver_present(driver_member, site)    whether a file existed for the pair

and the eight drivers are ``NaN`` where it is ``False``. Without the flag every
requested pair must exist, so the variable is not written.

**Coordinates**

======================= ================== ====================================
Name                    Dims               Meaning
======================= ================== ====================================
``driver_member``       ``driver_member``  0-based ``int64``, the member's
                                           identity: ``source_index - 1``,
                                           whatever members are loaded
``source_index``        ``driver_member``  ``int64``, the 1-based index in the
                                           directory name, in the order the
                                           members were asked for (ascending
                                           when discovered)
``site``                ``site``           ``int32`` site id, in the order
                                           the sites were asked for
``lon``, ``lat``        ``site``           from the site table, ``float64``,
                                           with CF attributes
``time``                ``time``           ``datetime64[ns]``, the end of each
                                           step
``timestep_start``      ``time``           the start of each step
``timestep_length``     ``time``           each step's declared length,
                                           ``timedelta64[ns]``
``time_bounds``         ``time, bounds``   ``[timestep_start, time]``
======================= ================== ====================================

**Time.** The time coordinates are pySIPNET's, taken unchanged from
:attr:`pysipnet.climate.ClimateDrivers.xarray`: the same axis, with the same
attributes, that a SIPNET run on the file has for its output. Its
``year``/``day_of_year``/``hour_of_day`` row labels are not kept, since
``timestep_start`` is the same instant. A row's labels are the start of its
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
``timestep_length_source`` among them -- and ``title``, ``source_root``,
``source_layout``, ``n_sites``, ``n_driver_members`` and ``coverage``
(``"complete"`` or ``"gaps"``).

**Missing values.** There are none in the source. A ``NaN`` appears only under
``allow_missing=True``, for a whole ``(driver_member, site)`` pair whose file is
absent, and ``driver_present`` says which.

Functions
---------
:func:`load_drivers`
    Read the drivers for the sites named, checking every file on the way, and
    return the Dataset above.

:func:`driver_fields`
    Split the Dataset into fields -- one ``DataArray`` per variable with
    dims ``(driver_member, site, time)`` and its own units. This is the view
    the plotting layer wants.

:func:`read_driver_file`
    Read one ``.clim`` file through pySIPNET and apply this module's own check
    on its values. The building block :func:`load_drivers` is made of, public
    so that tests and one-off surveys read a file exactly as the loader does.

:func:`available_source_indices`
    Which source indices have a directory for a given site.

:func:`driver_file`
    The path of the one ``.clim`` file for a site and source index.

:func:`default_drivers_root`
    Where the raw directory is expected to be, honoring
    ``$SIPNET_CALIBRATION_DATA``.

The checks
    One invariant each: on the request, on the directory layout, and on
    each file's values, name and time axis.

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

**Member indices.** ``driver_member`` is 0-based, ``source_index - 1``, so a
member's label is its identity: two loads of different members align member
for member, as a batch dim's one name promises. ``source_index`` keeps the
1-based file index beside it so the mapping to a directory is never
guesswork. Whether driver member *i* corresponds to initial-condition member
*i* is not established (open question 12 in ``data/README.md``), which is why
the two ensembles carry different dim names: xarray and PyEns cross them
rather than pair them by label.

Usage
-----
Name the sites, get the canonical form::

    from sipnet_calibration.drivers import driver_fields, load_drivers
    from sipnet_calibration.sites import EXTENTS, load_sites, select_sites

    site_table = select_sites(load_sites(), bbox=EXTENTS["CONUS"], n_random=20, seed=0)
    drivers = load_drivers(site_table["site_id"])     # every member present

    drivers["air_temperature"].dims   # ('driver_member', 'site', 'time')
    drivers["precipitation"].attrs["kind"]            # 'timestep_total'
    drivers["time"].attrs["time_zone"]                # 'undeclared'

    # Two members only, and tolerate sites that lack a file for one of them.
    partial = load_drivers([1, 27], source_indices=[1, 2], allow_missing=True)
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

from sipnet_calibration.conventions import (
    BATCH_LABEL_DTYPE,
    DRIVER_MEMBER,
    SITE,
    SITE_DTYPE,
    SOURCE_INDEX,
    SOURCE_INDEX_ATTRIBUTES,
    TIME,
    TIME_BOUNDS,
    TIME_COORD_NAMES,
    TIMESTEP_LENGTH,
    TIMESTEP_START,
    data_root,
)
from sipnet_calibration.fields import batch_coordinate, without_stale_time_attributes
from sipnet_calibration.sites import load_sites, site_coordinates
from sipnet_calibration.validation import (
    as_bounded_integer,
    as_positive_integers,
    as_site_ids,
    check_names_are_unique,
    truncated,
)

__all__ = [
    "DRIVER_DIRECTORY_PATTERN",
    "DRIVER_DIRECTORY_TEMPLATE",
    "DRIVER_FILE_GLOB",
    "DRIVER_FILE_PATTERN",
    "DRIVER_PRESENT",
    "DRIVER_VARIABLE_NAMES",
    "NEGATIVE_TOLERANCE",
    "UNITS_PROVENANCE",
    "available_source_indices",
    "default_drivers_root",
    "driver_fields",
    "driver_file",
    "load_drivers",
    "read_driver_file",
]

#: The driver variables, under pySIPNET's names, in climate-file column order:
#: every column of pySIPNET's climate registry outside its ``time`` group, which
#: becomes the time axis.
DRIVER_VARIABLE_NAMES: tuple[str, ...] = tuple(
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

#: Name of the presence variable written under ``allow_missing=True``.
DRIVER_PRESENT = "driver_present"

#: Per-site-and-member directory under the drivers root, ``<member>`` being
#: the source index.
DRIVER_DIRECTORY_TEMPLATE = "ERA5_{site}_{member}"

#: The file inside a :data:`DRIVER_DIRECTORY_TEMPLATE` directory,
#: ``ERA5.<member>.<start>.<end>.clim``. The glob accepts any member and any
#: dates so that a file whose name disagrees with its directory is reported as
#: the mismatch it is rather than as a missing file; the reader checks both
#: against the directory and the data.
DRIVER_FILE_GLOB = "ERA5.*.clim"

#: A :data:`DRIVER_DIRECTORY_TEMPLATE` name, exactly: group 1 is the site id and
#: group 2 the source index.
DRIVER_DIRECTORY_PATTERN = re.compile(r"^ERA5_(\d+)_(\d+)$")

#: A driver file's name, exactly: group 1 is the source index, and groups 2 and
#: 3 the ``<start>`` and ``<end>`` dates, as ``YYYY-MM-DD``.
DRIVER_FILE_PATTERN = re.compile(
    r"^ERA5\.(\d+)\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.clim$"
)

#: How far below zero photosynthetically active radiation and precipitation may
#: go before a file is refused. The source holds excursions of order 1e-5 and
#: 1e-15 that read as generator noise around zero; anything larger is a
#: different problem.
NEGATIVE_TOLERANCE = 1e-4


def load_drivers(
    sites: Iterable[int],
    *,
    source_indices: Iterable[int] | None = None,
    root: Path | str | None = None,
    site_table: pd.DataFrame | None = None,
    allow_missing: bool = False,
    time_zone: str | None = None,
) -> xr.Dataset:
    """Read the drivers for the given sites into the canonical form.

    Parameters
    ----------
    sites:
        Site ids to read, a sequence of integers, each named once, returned
        in the order given. Every one must be in the site table.
    source_indices:
        The members to read, by their 1-based index in the directory names
        (the ``source_index`` coordinate of the result), a sequence of
        integers, each named once, returned in the order given. ``None``
        means every member that has a directory for any of the requested
        sites, in ascending order.
    root:
        The drivers root. Defaults to :func:`default_drivers_root`.
    site_table:
        The site table, as :func:`sipnet_calibration.sites.load_sites` returns
        it. Loaded from its default location when ``None``. Only ``site_id``,
        ``lon`` and ``lat`` are read, and ``site_id`` must be unique.
    allow_missing:
        What to do about a ``(site, source index)`` pair with no file.
        ``False``, the default, raises, because a missing driver member that
        became ``NaN`` would propagate silently through any statistic over
        members. ``True`` fills the pair with ``NaN`` and adds
        :data:`DRIVER_PRESENT`. At least one requested pair must have a file
        either way.
    time_zone:
        The clock the files' labels are on, declared to pySIPNET for every
        file; see :func:`read_driver_file`. ``None`` leaves it undeclared.

    Returns
    -------
    xarray.Dataset
        The Data model described in the module docstring: the eight
        :data:`DRIVER_VARIABLE_NAMES` on ``(driver_member, site, time)``,
        ``float64``, with pySIPNET's time coordinates, ``lon``/``lat`` on
        ``site`` and ``source_index`` on ``driver_member``.

    Raises
    ------
    FileNotFoundError
        If the root is absent; if no requested site has a driver directory
        or no requested pair has a file, whatever *allow_missing* says; or
        if a requested pair has no file and *allow_missing* is ``False``.
    TypeError
        If *sites*, *source_indices* or *site_table* has the wrong type.
    KeyError
        If a site is not in the site table.
    ValueError
        If an argument has a wrong value, or a directory or a file fails a
        check of this module.
    """
    root = Path(root) if root is not None else default_drivers_root()
    check_drivers_root_is_a_directory(root)
    time_zone = normalize_time_zone(time_zone)

    site_ids = _site_id_array(sites)
    table = site_table if site_table is not None else load_sites()
    # Located before any file is read, so a site the table lacks fails fast.
    coordinates = site_coordinates(site_ids.tolist(), table)

    indices = _source_index_array(source_indices, root=root, sites=site_ids)

    paths, present = _locate_files(root, sites=site_ids, source_indices=indices)
    check_some_pair_has_a_file(present, sites=site_ids, source_indices=indices, root=root)
    if not allow_missing:
        check_every_requested_pair_has_a_file(
            present, sites=site_ids, source_indices=indices, root=root
        )

    arrays, reference = _read_located_files(paths, present, time_zone=time_zone)
    return _drivers_dataset(
        arrays,
        present=present,
        reference=reference,
        coordinates=coordinates,
        source_indices=indices,
        root=root,
        allow_missing=allow_missing,
    )


def driver_fields(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """One ``DataArray`` per driver variable, in :data:`DRIVER_VARIABLE_NAMES` order.

    Each field has dims ``(driver_member, site, time)``, is named for its
    variable, keeps that variable's attributes, and carries pySIPNET's
    :data:`sipnet_calibration.conventions.TIME_COORD_NAMES` with
    ``lon``/``lat`` and ``source_index`` as non-dimension coordinates --
    the field shape, and the same time coordinates a model field has. This is
    the view facet-by-variable consumes, matching
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
        If any of :data:`DRIVER_VARIABLE_NAMES` is absent from *dataset*.

    Notes
    -----
    ``time_bounds`` does not ride on a field, its ``bounds`` dimension being no
    field dimension, so ``time``'s ``bounds`` attribute is dropped with it, as
    :func:`sipnet_calibration.fields.to_model_output` does for a model output.
    """
    check_drivers_have_the_variables(dataset)
    fields = {}
    for name in DRIVER_VARIABLE_NAMES:
        field = dataset[name].copy(deep=False)
        field[TIME].attrs = without_stale_time_attributes(field[TIME].attrs)
        fields[name] = field
    return fields


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
    climate = _climate_drivers_of_file(path, time_zone=time_zone)
    check_radiation_and_precipitation_are_not_below_zero(climate.pandas, message_name=str(path))
    return climate


def driver_file(root: Path | str, site: int, source_index: int) -> Path:
    """The ``.clim`` file for one site and one member's source index.

    Parameters
    ----------
    root:
        The drivers root, laid out as :data:`DRIVER_DIRECTORY_TEMPLATE`.
    site:
        The site id.
    source_index:
        The member's 1-based index in the directory name.

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
    name = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member=int(source_index))
    directory = Path(root) / name
    check_driver_directory_exists(directory, site=site, source_index=source_index)
    matches = sorted(directory.glob(DRIVER_FILE_GLOB))
    check_driver_directory_holds_a_file(directory, matches)
    check_driver_directory_holds_one_file(directory, matches)
    return matches[0]


def available_source_indices(root: Path | str, site: int) -> tuple[int, ...]:
    """The source indices of the members that have a directory for *site*.

    Parameters
    ----------
    root:
        The drivers root.
    site:
        The site id.

    Returns
    -------
    tuple of int
        1-based source indices in ascending order, possibly empty. Only the
        directory's existence is consulted; whether the file inside it is
        present and well formed is :func:`driver_file` and
        :func:`read_driver_file`'s business. A directory whose name is not
        exactly the template for its numbers, ``ERA5_3_01`` say, is ignored,
        since :func:`driver_file` could not find it either.
    """
    root = Path(root)
    source_indices = []
    pattern = DRIVER_DIRECTORY_TEMPLATE.format(site=int(site), member="*")
    for directory in root.glob(pattern):
        parsed = _site_and_source_index_of_directory(directory.name)
        if parsed is None or not directory.is_dir():
            continue
        canonical = DRIVER_DIRECTORY_TEMPLATE.format(site=parsed[0], member=parsed[1])
        if parsed[0] == int(site) and directory.name == canonical:
            source_indices.append(parsed[1])
    return tuple(sorted(source_indices))


def default_drivers_root() -> Path:
    """Where the raw driver directory is expected to be.

    ``$SIPNET_CALIBRATION_DATA/raw/drivers`` when that variable is set, and
    otherwise ``data/raw/drivers`` under this checkout. Experiments name their
    paths in ``config.py``.
    """
    return data_root() / "raw" / "drivers"


# ── private helpers ───────────────────────────────────────────────────────────

#: Variables whose values below zero are counted, into ``n_values_below_zero``.
_NAMES_COUNTED_BELOW_ZERO = ("photosynthetically_active_radiation", "precipitation")

#: Variables whose values not above zero are counted, into ``n_values_not_positive``.
_NAMES_COUNTED_NOT_POSITIVE = (
    "vapor_pressure_deficit",
    "soil_vapor_pressure_deficit",
    "wind_speed",
)

#: The time coordinates the Dataset takes from pySIPNET: the ones a field
#: keeps, and the CF bounds pair that only a Dataset can carry.
_DATASET_TIME_COORD_NAMES = (*TIME_COORD_NAMES, TIME_BOUNDS)


def _climate_drivers_of_file(path: Path, *, time_zone: str | None) -> ClimateDrivers:
    """The file read by pySIPNET; its refusal is a ``ValueError`` naming the file."""
    try:
        return ClimateDrivers.from_file(path, time_zone=time_zone)
    except ValueError as error:
        raise ValueError(
            f"{path}: pySIPNET refused the file: {error}; correct the file, since SIPNET "
            "would run on what pySIPNET reads."
        ) from error


def _site_and_source_index_of_directory(name: str) -> tuple[int, int] | None:
    """``(site, source index)`` from an ``ERA5_<site>_<member>`` name, else ``None``."""
    match = DRIVER_DIRECTORY_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _dates_of_file_name(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The ``<start>`` and ``<end>`` dates of a file name that follows the template."""
    match = DRIVER_FILE_PATTERN.match(path.name)
    return pd.Timestamp(match.group(2)), pd.Timestamp(match.group(3))


def _is_a_date(text: str) -> bool:
    """Whether *text*, ``YYYY-MM-DD``, is a real calendar date."""
    try:
        pd.Timestamp(text)
    except ValueError:
        return False
    return True


def _site_id_array(sites: Iterable[int]) -> np.ndarray:
    """Requested sites as an ``int32`` array, in the order given."""
    site_ids = as_site_ids(sites, message_name="sites")
    check_request_is_not_empty(site_ids, example="site id", message_name="sites")
    return np.asarray(site_ids, dtype=SITE_DTYPE)


def _source_index_array(
    source_indices: Iterable[int] | None, *, root: Path, sites: np.ndarray
) -> np.ndarray:
    """Requested source indices as ``int64``, in order, or the discovered ones."""
    if source_indices is None:
        found: set[int] = set()
        for site in sites:
            found.update(available_source_indices(root, int(site)))
        check_some_driver_directory_exists(found, root=root, sites=sites)
        source_indices = sorted(found)
    return _as_source_indices(source_indices)


def _as_source_indices(source_indices: Iterable[int]) -> np.ndarray:
    """*source_indices* as an ``int64`` array, in the order given, or a clear error.

    Strings, booleans, floats, non-positive values, values beyond ``int64``
    and repeats are refused, since each would otherwise resolve to a
    plausible-looking wrong directory or overflow.
    """
    positive = as_positive_integers(source_indices, message_name="source_indices")
    # Each index, and its driver_member label one below it, fits int64.
    indices = tuple(
        as_bounded_integer(
            index,
            minimum=1,
            maximum=int(np.iinfo(BATCH_LABEL_DTYPE).max),
            message_name=f"source_indices[{position}]",
        )
        for position, index in enumerate(positive)
    )
    check_request_is_not_empty(indices, example="source index", message_name="source indices")
    check_names_are_unique(indices, message_name="source_indices")
    return np.asarray(indices, dtype=BATCH_LABEL_DTYPE)


def _locate_files(
    root: Path, *, sites: np.ndarray, source_indices: np.ndarray
) -> tuple[dict[tuple[int, int], Path], np.ndarray]:
    """The path of each ``(driver_member, site)`` pair that has one, and the mask."""
    present = np.zeros((source_indices.size, sites.size), dtype=bool)
    paths: dict[tuple[int, int], Path] = {}
    for j, site in enumerate(sites):
        for i, source_index in enumerate(source_indices):
            try:
                paths[(i, j)] = driver_file(root, int(site), int(source_index))
            except FileNotFoundError:
                continue
            present[i, j] = True
    return paths, present


def _read_located_files(
    paths: dict[tuple[int, int], Path], present: np.ndarray, *, time_zone: str | None
) -> tuple[dict[str, np.ndarray], xr.Dataset]:
    """Read every located file into ``(driver_member, site, time)`` arrays.

    The first file read supplies the time axis; every later file is checked to
    be on the same one before its values are copied in. A ``(driver_member,
    site)`` pair with no file stays ``NaN``. Returns the arrays and the first
    file's pySIPNET Dataset, whose time coordinates and attributes the result
    takes.
    """
    reference: xr.Dataset | None = None
    reference_path: Path | None = None
    arrays: dict[str, np.ndarray] = {}

    for (i, j), path in sorted(paths.items(), key=lambda item: (item[0][1], item[0][0])):
        dataset = read_driver_file(path, time_zone=time_zone).xarray
        _, source_index = _site_and_source_index_of_directory(path.parent.name)
        check_driver_file_name_follows_the_template(path)
        check_driver_file_name_agrees_with_its_directory(path, source_index=source_index)
        check_driver_file_name_dates_are_dates(path)
        check_driver_file_name_dates_are_its_record(path, dataset)
        if reference is None:
            reference, reference_path = dataset, path
            shape = present.shape + (dataset.sizes[TIME],)
            arrays = {name: np.full(shape, np.nan) for name in DRIVER_VARIABLE_NAMES}
        else:
            check_driver_files_share_a_time_axis(
                reference, dataset, reference_path=reference_path, path=path
            )
        for name in DRIVER_VARIABLE_NAMES:
            arrays[name][i, j, :] = dataset[name].to_numpy()
    # load_drivers has checked that some pair has a file, so reference is set.
    return arrays, reference


def _drivers_dataset(
    arrays: dict[str, np.ndarray],
    *,
    present: np.ndarray,
    reference: xr.Dataset,
    coordinates: dict[str, xr.DataArray],
    source_indices: np.ndarray,
    root: Path,
    allow_missing: bool,
) -> xr.Dataset:
    """Put the arrays into the Dataset the module docstring describes."""
    dims = (DRIVER_MEMBER, SITE, TIME)
    data_vars = {}
    for name in DRIVER_VARIABLE_NAMES:
        values = arrays[name]
        attrs = {**reference[name].attrs, "units_provenance": UNITS_PROVENANCE}
        observed = values[present]
        if name in _NAMES_COUNTED_BELOW_ZERO:
            attrs["n_values_below_zero"] = int(np.count_nonzero(observed < 0))
        if name in _NAMES_COUNTED_NOT_POSITIVE:
            attrs["n_values_not_positive"] = int(np.count_nonzero(observed <= 0))
        data_vars[name] = xr.DataArray(values, dims=dims, attrs=attrs)
    if allow_missing:
        data_vars[DRIVER_PRESENT] = xr.DataArray(
            present,
            dims=(DRIVER_MEMBER, SITE),
            attrs={
                "long_name": "Whether a driver file existed for the driver member and site",
                "comment": "The eight driver variables are NaN where this is False.",
            },
        )

    dataset = xr.Dataset(
        data_vars,
        coords={
            # A member's label is its identity, the same in every load.
            DRIVER_MEMBER: batch_coordinate(DRIVER_MEMBER, source_indices - 1),
            SOURCE_INDEX: (
                DRIVER_MEMBER,
                source_indices.astype(BATCH_LABEL_DTYPE),
                dict(SOURCE_INDEX_ATTRIBUTES),
            ),
            **coordinates,
            **{name: reference[name].variable for name in _DATASET_TIME_COORD_NAMES},
        },
    )
    dataset.attrs = {
        **reference.attrs,
        "title": "ERA5 meteorological drivers in SIPNET climate-file form",
        "source_root": str(root),
        "source_layout": f"{DRIVER_DIRECTORY_TEMPLATE}/{DRIVER_FILE_GLOB}",
        "n_sites": int(dataset.sizes[SITE]),
        "n_driver_members": int(source_indices.size),
        "coverage": "complete" if present.all() else "gaps",
    }
    return unfilled_coordinates(dataset)


# ── checks ────────────────────────────────────────────────────────────────────


def check_drivers_root_is_a_directory(root: Path) -> None:
    """The drivers root is a directory."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"drivers root {root} is not a directory; pass root=, or link the drivers "
            "under data/raw/drivers."
        )


def check_request_is_not_empty(
    values: tuple[int, ...], *, example: str, message_name: str
) -> None:
    """A request names at least one site, or one source index."""
    if not values:
        raise ValueError(f"no {message_name} requested; pass at least one {example}.")


def check_some_driver_directory_exists(
    source_indices: set[int], *, root: Path, sites: np.ndarray
) -> None:
    """Some requested site has a driver directory, when the members are discovered."""
    if not source_indices:
        raise FileNotFoundError(
            f"no driver directories under {root} for sites {truncated(sites.tolist())}; check "
            "the root and the site ids."
        )


def check_some_pair_has_a_file(
    present: np.ndarray, *, sites: np.ndarray, source_indices: np.ndarray, root: Path
) -> None:
    """At least one requested ``(site, source index)`` pair has a driver file."""
    if not present.any():
        raise FileNotFoundError(
            f"no driver files under {root} for sites {truncated(sites.tolist())} and "
            f"source indices {truncated(source_indices.tolist())}; check the root and the "
            "site ids."
        )


def check_every_requested_pair_has_a_file(
    present: np.ndarray, *, sites: np.ndarray, source_indices: np.ndarray, root: Path
) -> None:
    """Every requested ``(site, source index)`` pair has a driver file."""
    missing = [
        f"site {int(sites[j])} source index {int(source_indices[i])}"
        for i, j in zip(*np.nonzero(~present), strict=True)
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} requested pair(s) have no driver file under {root}: "
            f"{truncated(missing)}; pass allow_missing=True to read the rest with NaN in "
            "their place and a driver_present array saying which."
        )


def check_driver_directory_exists(directory: Path, *, site: int, source_index: int) -> None:
    """The directory of one site and source index exists."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no driver directory for site {site} source index {source_index}: {directory}; "
            "check the drivers root, or link the directory from the SCC."
        )


def check_driver_directory_holds_a_file(directory: Path, matches: list[Path]) -> None:
    """A driver directory holds a file matching :data:`DRIVER_FILE_GLOB`."""
    if not matches:
        raise FileNotFoundError(
            f"{directory} holds no file matching {DRIVER_FILE_GLOB!r}; copy its .clim file "
            "from the SCC."
        )


def check_driver_directory_holds_one_file(directory: Path, matches: list[Path]) -> None:
    """A driver directory holds no more than one file matching :data:`DRIVER_FILE_GLOB`."""
    if len(matches) > 1:
        raise ValueError(
            f"{directory} holds {len(matches)} files matching {DRIVER_FILE_GLOB!r}, where the "
            f"layout promises one: {truncated([match.name for match in matches])}; remove the "
            "extras."
        )


def check_radiation_and_precipitation_are_not_below_zero(
    frame: pd.DataFrame, *, message_name: str
) -> None:
    """Radiation and precipitation stay above ``-NEGATIVE_TOLERANCE``."""
    # Small negatives are known and read through; a large one would be a
    # different kind of problem and is refused.
    for name in _NAMES_COUNTED_BELOW_ZERO:
        values = frame[name].to_numpy()
        low = values < -NEGATIVE_TOLERANCE
        if low.any():
            raise ValueError(
                f"{message_name}: {int(low.sum())} {name} value(s) below "
                f"-{NEGATIVE_TOLERANCE:g}, the lowest {values.min():.4g}; small negative "
                "excursions around zero are known, but these are not small, so correct the "
                "file."
            )


def check_driver_file_name_follows_the_template(path: Path) -> None:
    """A driver file is named ``ERA5.<member>.<start>.<end>.clim``."""
    if DRIVER_FILE_PATTERN.match(path.name) is None:
        raise ValueError(
            f"{path}: file name does not follow ERA5.<member>.<start>.<end>.clim; rename it."
        )


def check_driver_file_name_agrees_with_its_directory(path: Path, *, source_index: int) -> None:
    """A driver file's name and its directory's give the same source index."""
    named = int(DRIVER_FILE_PATTERN.match(path.name).group(1))
    if named != source_index:
        raise ValueError(
            f"{path}: the file name says member {named}, the directory says member "
            f"{source_index}; one of the two was misnamed, so correct it."
        )


def check_driver_file_name_dates_are_dates(path: Path) -> None:
    """A driver file's name carries two real calendar dates."""
    match = DRIVER_FILE_PATTERN.match(path.name)
    invalid = [text for text in match.group(2, 3) if not _is_a_date(text)]
    if invalid:
        raise ValueError(
            f"{path}: file name carries the invalid date(s) {truncated(invalid)}; rename the file "
            "for the days its first and last steps start on."
        )


def check_driver_file_name_dates_are_its_record(path: Path, dataset: xr.Dataset) -> None:
    """A driver file's name gives the days its first and last steps start on."""
    start, end = _dates_of_file_name(path)
    starts = pd.DatetimeIndex(dataset[TIMESTEP_START].values)
    first, last = starts[0].normalize(), starts[-1].normalize()
    if (start, end) != (first, last):
        raise ValueError(
            f"{path}: the file name covers {start.date()} to {end.date()} but the data runs "
            f"{first.date()} to {last.date()}; the file is not the record its name says."
        )


def check_driver_files_share_a_time_axis(
    reference: xr.Dataset, dataset: xr.Dataset, *, reference_path: Path, path: Path
) -> None:
    """Two driver files have the same step starts and lengths."""
    # The time coordinates are taken from the first file read and applied to
    # all of them, which is sound only if every file's axis is the same.
    if dataset.sizes[TIME] != reference.sizes[TIME]:
        raise ValueError(
            f"{path} has {dataset.sizes[TIME]} steps where {reference_path} has "
            f"{reference.sizes[TIME]}; every file read together must share one time axis, "
            "so load them separately."
        )
    for name in (TIMESTEP_START, TIMESTEP_LENGTH):
        expected = reference[name].to_numpy()
        found = dataset[name].to_numpy()
        if not np.array_equal(expected, found):
            first = int(np.flatnonzero(expected != found)[0])
            raise ValueError(
                f"{path}: {name} differs from {reference_path} first at step {first} "
                f"({found[first]!r} against {expected[first]!r}); every file read together "
                "must share one time axis, so load them separately."
            )


def check_drivers_have_the_variables(dataset: xr.Dataset) -> None:
    """A drivers Dataset holds every one of :data:`DRIVER_VARIABLE_NAMES`."""
    missing = [name for name in DRIVER_VARIABLE_NAMES if name not in dataset.data_vars]
    if missing:
        raise ValueError(
            f"dataset is missing driver variables {truncated(missing)}, holding "
            f"{truncated(sorted(dataset.data_vars))}; pass the Dataset load_drivers returns."
        )

"""PEcAn's source files: what one looks like, and how to parse it.

The ensemble arrives as 800,000 netCDF-3 files, one per ``(site, member)``,
under ``data/raw/initial_conditions/files/`` and present only on the SCC. This
module holds the contract those files satisfy and the parser that enforces it.
:mod:`sipnet_calibration.initial_conditions.raw` turns the parsed records into
the single tracked netCDF everything else reads.

The rationale for parsing and checking in one place is in
:func:`read_source_file`'s Notes.

Contents
--------
:data:`SOURCE` is the format, one :class:`SourceFormat`: the file layout, a
:class:`SourceVariable` per variable a file may carry, the fill value, and what
the degenerate ``time`` variable must say. It is what :func:`read_source_file`
holds a file to.

:data:`SOURCE_SCRIPT`, :data:`SOURCE_SCRIPT_NOTE` and :data:`NOMINAL_DATE` are
provenance rather than format: they say how PEcAn drew the ensemble, are copied
into the processed file's attributes, and are checked against nothing.

:func:`read_source_file` parses one file, :func:`read_source_directory` one
site's directory, and :func:`site_member_from_file_name` decodes a file name.
Each parsed file is a :class:`SourceFile`.

The checks, one invariant each, are what :func:`read_source_file` and
:func:`read_source_directory` hold a file and a directory to.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from frozendict import frozendict
from scipy.io import netcdf_file

from sipnet_calibration.validation import truncated


__all__ = [
    "NOMINAL_DATE",
    "SOURCE",
    "SOURCE_SCRIPT",
    "SOURCE_SCRIPT_NOTE",
    "SourceFile",
    "SourceFormat",
    "SourceVariable",
    "check_source_variable_is_known",
    "read_source_directory",
    "read_source_file",
    "site_member_from_file_name",
]


@dataclass(frozen=True)
class SourceVariable:
    """One variable a source file may carry, and the attributes it declares."""

    name: str
    """The variable's name in the source files and in the raw file."""

    units: str
    """Its ``units`` attribute, verbatim. PEcAn's ``standard_vars.csv`` string,
    recorded and not interpreted: soil moisture is a 0-100 percentage despite
    its ``(-)``."""

    long_name: str
    """Its ``long_name`` attribute, verbatim."""


@dataclass(frozen=True)
class SourceFormat:
    """The format every one of PEcAn's source files satisfies.

    :func:`read_source_file` checks a file against this and refuses anything
    that differs, so all 800,000 are held to one description rather than to
    whatever the three files in a local checkout happen to show.

    Notes
    -----
    The variables are one record each rather than parallel ``units`` and
    ``long_name`` mappings, so the two cannot come to disagree about which
    variables exist.
    """

    file_template: str
    """The layout under the source root: one directory per site holding one
    file per member, ``<member>`` being the 1-based member index."""

    variables: Mapping[str, SourceVariable]
    """The variables a file may carry, by name, in the order the specs and the
    raw file use. A file holding any other variable is refused."""

    fill_value: float
    """The ``_FillValue`` every variable declares. None is present in the
    ensemble, and a file that holds one is refused."""

    time_units: str
    """The degenerate ``time`` variable's ``units``: an unsubstituted template
    no calendar library can parse (issue #3). Asserted identical in every file."""

    time_long_name: str
    """That variable's ``long_name``, PEcAn's standard string for the dimension."""

    time_value: float
    """That variable's one value."""

    def __post_init__(self) -> None:
        # frozen=True freezes the field, not the dict behind it. Without this a
        # caller could add a variable, and the specs would follow: they read
        # this mapping at attribute-access time, so an already-built spec would
        # start reporting different source units.
        object.__setattr__(self, "variables", frozendict(self.variables))

    def __hash__(self) -> int:
        # dataclass(frozen=True) generates a __hash__ that hashes the fields,
        # and a mapping is not hashable.
        return hash((self.file_template, self.names, self.fill_value))

    @property
    def names(self) -> tuple[str, ...]:
        """The variable names, in order."""
        return tuple(self.variables)


#: The format of the files under ``data/raw/initial_conditions/files/``.
SOURCE = SourceFormat(
    file_template="{site}/IC_site_{site}_{member}.nc",
    variables={
        variable.name: variable
        for variable in (
            SourceVariable("AbvGrndWood", "kg C m-2", "Above ground woody biomass"),
            SourceVariable("wood_carbon_content", "kg C m-2", "Wood Carbon Content"),
            SourceVariable("leaf_carbon_content", "kg C m-2", "Leaf Carbon Content"),
            SourceVariable(
                "soil_organic_carbon_content",
                "kg C m-2",
                "Soil Organic Carbon Content by Layer",
            ),
            SourceVariable("SoilMoistFrac", "(-)", "Average Layer Fraction of Saturation"),
        )
    },
    fill_value=-999.0,
    time_units="days since [year]-01-01 00:00:00 UTC",
    time_long_name="Time middle averaging period",
    time_value=1.0,
)


#: The PEcAn script that draws the ensemble and writes the source files, and
#: the caveat every reader must see beside it.
SOURCE_SCRIPT = (
    "/projectnb/dietzelab/dongchen/anchorSites/IC_prep_anchorSites.R (Dongchen Zhang, "
    "2024-03-27); the same code is modules/assim.sequential/inst/anchor/"
    "IC_prep_anchorSites.Rmd on PEcAn develop"
)

SOURCE_SCRIPT_NOTE = (
    "That script targets the 343 anchor sites. The 8000-site files were written on "
    "2025-07-23 by a run whose script was not found; they match the script's "
    "construction exactly (five variables, wood = biomass - leaf bitwise, soil "
    "moisture in percent), so this is the template for that run, not a confirmed "
    "record. Open question 24 in data/README.md."
)

#: The date the PEcAn script sampled the upstream products at, from its own code
#: (``time_poimt <- as.Date("2011-07-15")``, the variable name spelled as the
#: script spells it). The files carry no date.
NOMINAL_DATE = "2011-07-15"


@dataclass(frozen=True)
class SourceFile:
    """One source file, parsed.

    Attributes
    ----------
    site, member:
        The numbers the file name encodes; ``member`` is the source's 1-based
        index.
    values:
        Source variable name to the file's one value, for the variables the
        file carries. Every value is finite and not the fill.
    """

    site: int
    member: int
    values: Mapping[str, float]


def site_member_from_file_name(name: str) -> tuple[int, int] | None:
    """The ``(site, member)`` a file name encodes, or ``None`` if it is not one.

    Only a name that is exactly ``IC_site_<site>_<member>.nc`` with plain
    positive integers counts; ``IC_site_01_5.nc`` does not, since it would not
    round-trip through ``SOURCE.file_template``.
    """
    match = _SOURCE_FILE_NAME.match(name)
    if match is None:
        return None
    return int(match.group("site")), int(match.group("member"))


def read_source_file(path: Path | str) -> SourceFile:
    """Parse one source file exactly and run the per-file checks.

    Parameters
    ----------
    path:
        The file, named as ``SOURCE.file_template`` within its site
        directory.

    Returns
    -------
    SourceFile
        The file's variables under their source names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* is not a regular file, or the file or its name departs from
        :data:`SOURCE` in any way this module checks.

    Notes
    -----
    The checks live here so that a file is checked wherever it is parsed and
    the conversion applies exactly one set of rules to all 800,000. Read with
    ``scipy.io.netcdf_file``, which reads netCDF-3 directly and needs no
    library beyond the project's; the files cannot be opened by ``h5netcdf``
    at all, and the ``time`` units are undecodable, so the generic xarray path
    would need two workarounds for nothing the parser wants.
    """
    path = Path(path)
    check_source_file_exists(path)
    check_source_file_is_a_regular_file(path)
    check_source_file_name_is_the_template(path)
    site, member = site_member_from_file_name(path.name)
    check_source_file_is_in_its_site_directory(path, site=site)
    with _opened_source_file(path) as handle:
        check_source_file_layout_is_the_template(handle, message_name=str(path))
        check_source_time_is_the_template(handle, message_name=str(path))
        values = _source_values(handle, message_name=str(path))
    check_source_file_carries_a_variable(values, message_name=str(path))
    return SourceFile(site=site, member=member, values=values)


def read_source_directory(root: Path | str, site: int) -> list[SourceFile]:
    """Parse every file of one site's directory, refusing anything else in it.

    Parameters
    ----------
    root:
        The source tree.
    site:
        The site whose directory ``<root>/<site>`` to read.

    Returns
    -------
    list of SourceFile
        One record per file, in file-name order.

    Raises
    ------
    FileNotFoundError
        If the site's directory is absent, or as :func:`read_source_file`.
    ValueError
        If the directory is empty or holds an entry that is not a source file,
        or as :func:`read_source_file`.

    Notes
    -----
    Hidden files such as ``.DS_Store`` are skipped as filesystem debris. This
    is the unit of parallel work in
    ``scripts/raw_sources/convert_initial_conditions.py``, which is why it
    lives here: a worker process has to be able to import it.
    """
    directory = Path(root) / str(int(site))
    check_source_site_directory_exists(directory)
    records = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        check_source_directory_entry_is_a_source_file(path)
        records.append(read_source_file(path))
    check_source_site_directory_holds_files(records, directory=directory)
    return records


# ── private helpers ───────────────────────────────────────────────────────────

#: A source file's name: ``IC_site_<site>_<member>.nc``, both plain positive integers.
_SOURCE_FILE_NAME = re.compile(r"^IC_site_(?P<site>[1-9]\d*)_(?P<member>[1-9]\d*)\.nc$")


def _opened_source_file(path: Path) -> Any:
    """*path* opened by ``scipy.io.netcdf_file``; unreadable, it is a ``ValueError``."""
    try:
        return netcdf_file(str(path), "r", mmap=False, maskandscale=False)
    except Exception as error:
        # Deliberately broad. scipy's reader raises whatever the malformation
        # happens to produce -- a file truncated inside the variable header
        # reaches `frombuffer(b"", ">i")[0]` and raises IndexError -- and a
        # traceback from one of 800,000 files does not say which file it was.
        raise ValueError(
            f"{path}: not readable as netCDF-3 classic ({error}); copy the file again from "
            "the source tree."
        ) from error


def _source_values(handle: Any, *, message_name: str) -> dict[str, float]:
    """Source variable name to its one value, for each data variable, each checked."""
    values: dict[str, float] = {}
    for name, variable in handle.variables.items():
        if name == "time":
            continue
        subject = f"{message_name}: {name}"
        check_source_variable_is_known(name, message_name=message_name)
        check_source_variable_is_a_scalar_on_time(variable, message_name=subject)
        check_source_variable_is_float64(variable, message_name=subject)
        check_source_variable_attributes_are_the_template(name, variable, message_name=subject)
        data = np.asarray(variable.data, dtype=np.float64).ravel()
        check_source_variable_holds_one_value(data, message_name=subject)
        value = float(data[0])
        check_source_value_is_not_the_fill(value, message_name=subject)
        check_source_value_is_finite(value, message_name=subject)
        values[name] = value
    return values


def _decode_attribute(value: Any) -> Any:
    """One netCDF-3 attribute as a Python string, float or list."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return float(value.ravel()[0]) if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _netcdf_attributes(obj: Any) -> dict[str, Any]:
    """The attributes of a ``scipy.io.netcdf_file`` or one of its variables."""
    return {str(key): _decode_attribute(value) for key, value in obj._attributes.items()}


# ── checks ────────────────────────────────────────────────────────────────────


def check_source_file_layout_is_the_template(handle: Any, *, message_name: str) -> None:
    """A source file's layout is the template: classic, bare, one record of ``time``."""
    check_source_file_is_netcdf3_classic(handle, message_name=message_name)
    check_source_file_has_no_global_attributes(handle, message_name=message_name)
    check_source_file_has_only_the_time_dimension(handle, message_name=message_name)
    check_source_time_is_the_record_dimension(handle, message_name=message_name)
    check_source_time_has_one_record(handle, message_name=message_name)


def check_source_time_is_the_template(handle: Any, *, message_name: str) -> None:
    """A source file's ``time`` variable is the degenerate template every file has."""
    check_source_file_has_a_time_variable(handle, message_name=message_name)
    check_source_time_attributes_are_the_template(handle, message_name=message_name)
    check_source_time_value_is_the_template(handle, message_name=message_name)


def check_source_file_exists(path: Path) -> None:
    """A source file exists, as a file or as a link."""
    if not (path.exists() or path.is_symlink()):
        raise FileNotFoundError(
            f"no such initial condition file: {path}; check the source root and the file name."
        )


def check_source_file_is_a_regular_file(path: Path) -> None:
    """A source file is a regular file, or a link to one."""
    if not path.is_file():
        raise ValueError(
            f"{path} is not a regular file; on the SCC the source tree is symlinked, so a "
            "broken link looks like this rather than like a missing file, and the link "
            "needs repairing."
        )


def check_source_file_name_is_the_template(path: Path) -> None:
    """A source file is named ``IC_site_<site>_<member>.nc``."""
    if site_member_from_file_name(path.name) is None:
        raise ValueError(
            f"{path}: name is not IC_site_<site>_<member>.nc; a source file is named as "
            f"{SOURCE.file_template} lays it out."
        )


def check_source_file_is_in_its_site_directory(path: Path, *, site: int) -> None:
    """A source file sits in the directory of the site its name encodes."""
    if path.parent.name != str(site):
        raise ValueError(
            f"{path}: the file name says site {site} but the directory is "
            f"{path.parent.name!r}; the layout is {SOURCE.file_template}, so move the file."
        )


def check_source_site_directory_exists(directory: Path) -> None:
    """A site's source directory exists."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory}: no such site directory; check the source root and the site id."
        )


def check_source_directory_entry_is_a_source_file(path: Path) -> None:
    """An entry of a site's source directory is an ``IC_site_<site>_<member>.nc``."""
    if site_member_from_file_name(path.name) is None:
        raise ValueError(
            f"{path}: not an IC_site_<site>_<member>.nc file; a source site directory holds "
            "nothing else, so move it out."
        )


def check_source_site_directory_holds_files(records: list[SourceFile], *, directory: Path) -> None:
    """A site's source directory holds at least one source file."""
    if not records:
        raise ValueError(
            f"{directory}: holds no files; copy the site's files from the source tree."
        )


def check_source_file_is_netcdf3_classic(handle: Any, *, message_name: str) -> None:
    """A source file is netCDF-3 classic, version byte 1."""
    if handle.version_byte != 1:
        raise ValueError(
            f"{message_name}: netCDF-3 version byte is {handle.version_byte}, expected 1 "
            "(classic); the file is not one PEcAn wrote."
        )


def check_source_file_has_no_global_attributes(handle: Any, *, message_name: str) -> None:
    """A source file carries no global attribute."""
    attrs = _netcdf_attributes(handle)
    if attrs:
        raise ValueError(
            f"{message_name}: carries global attributes {truncated(sorted(attrs))}; source "
            "files carry none, so the file is not one PEcAn wrote."
        )


def check_source_file_has_only_the_time_dimension(handle: Any, *, message_name: str) -> None:
    """A source file's only dimension is ``time``."""
    if set(handle.dimensions) != {"time"}:
        raise ValueError(
            f"{message_name}: dimensions are {truncated(sorted(handle.dimensions))}, expected "
            "['time']; a layer-resolved file would look like this, and needs a spec change."
        )


def check_source_time_is_the_record_dimension(handle: Any, *, message_name: str) -> None:
    """A source file's ``time`` is its unlimited record dimension."""
    if handle.dimensions["time"] is not None:
        raise ValueError(
            f"{message_name}: the time dimension is not the unlimited record dimension; the "
            "file is not one PEcAn wrote."
        )


def check_source_time_has_one_record(handle: Any, *, message_name: str) -> None:
    """A source file's ``time`` has one record."""
    if handle._recs != 1:
        raise ValueError(
            f"{message_name}: time has {handle._recs} records, expected 1; the file is not "
            "one PEcAn wrote."
        )


def check_source_file_has_a_time_variable(handle: Any, *, message_name: str) -> None:
    """A source file has a ``time`` variable."""
    if "time" not in handle.variables:
        raise ValueError(
            f"{message_name}: has no time variable; the file is not one PEcAn wrote."
        )


def check_source_time_attributes_are_the_template(handle: Any, *, message_name: str) -> None:
    """A source file's ``time`` attributes are the unsubstituted template."""
    attrs = _netcdf_attributes(handle.variables["time"])
    expected = {"units": SOURCE.time_units, "long_name": SOURCE.time_long_name}
    if attrs != expected:
        raise ValueError(
            f"{message_name}: time attributes are {attrs}, expected {expected}; a substituted "
            "year would mean issue #3 was fixed upstream, so notice it rather than average it "
            "away."
        )


def check_source_time_value_is_the_template(handle: Any, *, message_name: str) -> None:
    """A source file's ``time`` holds the one template value."""
    value = np.asarray(handle.variables["time"].data, dtype=np.float64).ravel()
    if value.size != 1 or value[0] != SOURCE.time_value:
        raise ValueError(
            f"{message_name}: time value is {value.tolist()}, expected [{SOURCE.time_value}]; "
            "the file is not one PEcAn wrote."
        )


def check_source_variable_is_known(name: str, *, message_name: str) -> None:
    """A source file's data variable is one of :data:`SOURCE`'s."""
    if name not in SOURCE.variables:
        raise ValueError(
            f"{message_name}: variable {name!r} is not one the source files carry "
            f"({truncated(sorted(SOURCE.variables))}); a new variable is a spec change, not a new "
            "column."
        )


def check_source_variable_is_a_scalar_on_time(variable: Any, *, message_name: str) -> None:
    """A source data variable is on ``("time",)`` alone."""
    if variable.dimensions != ("time",):
        raise ValueError(
            f"{message_name} has dims {variable.dimensions}, expected ('time',); a "
            "layer-resolved variable would look like this, and needs a spec change."
        )


def check_source_variable_is_float64(variable: Any, *, message_name: str) -> None:
    """A source data variable is ``float64``."""
    if variable.data.dtype.newbyteorder("=") != np.dtype(np.float64):
        raise ValueError(
            f"{message_name} is {variable.data.dtype}, expected float64; the file is not one "
            "PEcAn wrote."
        )


def check_source_variable_attributes_are_the_template(
    name: str, variable: Any, *, message_name: str
) -> None:
    """A source data variable's attributes are exactly the template's three."""
    attrs = _netcdf_attributes(variable)
    expected = {
        "_FillValue": SOURCE.fill_value,
        "long_name": SOURCE.variables[name].long_name,
        "units": SOURCE.variables[name].units,
    }
    if attrs != expected:
        raise ValueError(
            f"{message_name} attributes are {attrs}, expected exactly {expected}; an "
            "unexpected scale_factor or add_offset would silently rescale the value, so the "
            "format needs a spec change."
        )


def check_source_variable_holds_one_value(data: np.ndarray, *, message_name: str) -> None:
    """A source data variable holds one value."""
    if data.size != 1:
        raise ValueError(
            f"{message_name} holds {data.size} values, expected 1; the file is not one PEcAn "
            "wrote."
        )


def check_source_value_is_not_the_fill(value: float, *, message_name: str) -> None:
    """A source value is not the fill value."""
    if value == SOURCE.fill_value:
        raise ValueError(
            f"{message_name} holds the fill value {SOURCE.fill_value}; no file in the "
            "ensemble does, and the raw file has no representation for an explicit fill "
            "distinct from an absent variable, so the format needs a spec change."
        )


def check_source_value_is_finite(value: float, *, message_name: str) -> None:
    """A source value is finite."""
    if not np.isfinite(value):
        raise ValueError(
            f"{message_name} holds the non-finite value {value!r}; the file is not one PEcAn "
            "wrote."
        )


def check_source_file_carries_a_variable(values: dict[str, float], *, message_name: str) -> None:
    """A source file carries at least one data variable."""
    if not values:
        raise ValueError(
            f"{message_name}: carries no data variable; the file is not one PEcAn wrote."
        )

"""PEcAn's source files: what one looks like, and how to parse it.

The ensemble arrives as 800,000 netCDF-3 files, one per ``(site, member)``,
under ``data/raw/initial_conditions/files/`` and present only on the SCC. This
module holds the contract those files satisfy and the parser that enforces it,
so that a file is checked wherever it is read and all 800,000 are held to one
set of rules. :mod:`sipnet_calibration.initial_conditions.raw` turns the parsed
records into the single tracked netCDF everything else reads.

Contents
--------
:data:`SOURCE` is the format, one :class:`SourceFormat`: the file layout, a
:class:`SourceVariable` per variable a file may carry, the fill value, and what
the degenerate ``time`` variable must say. It is what :func:`read_source_file`
holds a file to.

:data:`SOURCE_SCRIPT`, :data:`SOURCE_SCRIPT_NOTE` and :data:`NOMINAL_DATE` are
provenance rather than format: they say how PEcAn drew the ensemble, are copied
into the processed product's attributes, and are checked against nothing.

:func:`read_source_file` parses one file, :func:`read_source_directory` one
site's directory, and :func:`site_member_from_file_name` decodes a file name.
Each parsed file is a :class:`SourceFile`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file

__all__ = [
    "NOMINAL_DATE",
    "SOURCE",
    "SOURCE_SCRIPT",
    "SOURCE_SCRIPT_NOTE",
    "SourceFile",
    "SourceFormat",
    "SourceVariable",
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

#: The date the PEcAn script sampled the source products at, from its own code
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
        If the name is not the template or disagrees with its directory; the
        file is not netCDF-3 classic; it has a global attribute, a dimension
        other than an unlimited length-1 ``time``, or a ``time`` variable
        whose attributes or value differ from the source template; a data
        variable is not a scalar ``float64`` on ``("time",)``, is not one of
        ``SOURCE.names``, or carries attributes other than exactly the
        expected ``_FillValue``, ``long_name`` and ``units``; a value is the
        fill or not finite; or the file carries no data variable.

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
    if not path.is_file():
        raise FileNotFoundError(f"no such initial condition file: {path}")
    parsed = site_member_from_file_name(path.name)
    if parsed is None:
        raise ValueError(f"{path}: name is not IC_site_<site>_<member>.nc")
    site, member = parsed
    if path.parent.name != str(site):
        raise ValueError(
            f"{path}: the file name says site {site} but the directory is "
            f"{path.parent.name!r}; the layout is {SOURCE.file_template}"
        )
    try:
        handle = netcdf_file(str(path), "r", mmap=False, maskandscale=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{path}: not readable as netCDF-3 classic ({error})") from error
    with handle:
        _check_source_file_is_classic_with_no_global_attributes(handle, path)
        _check_source_time_is_the_degenerate_template(handle, path)
        values = _check_and_read_source_variables(handle, path)
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
    ValueError
        If the directory is missing or empty, or holds an entry that is not
        an ``IC_site_<site>_<member>.nc`` file (hidden files such as
        ``.DS_Store`` are skipped as filesystem debris), plus whatever
        :func:`read_source_file` raises for a file.

    Notes
    -----
    The unit of parallel work in ``scripts/convert_initial_conditions.py``,
    which is why it lives here: a worker process has to be able to import it.
    """
    directory = Path(root) / str(int(site))
    if not directory.is_dir():
        raise ValueError(f"{directory}: no such site directory")
    records = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if site_member_from_file_name(path.name) is None:
            raise ValueError(
                f"{path}: not an IC_site_<site>_<member>.nc file; a source site "
                "directory holds nothing else"
            )
        records.append(read_source_file(path))
    if not records:
        raise ValueError(f"{directory}: holds no files")
    return records


_SOURCE_FILE_NAME = re.compile(r"^IC_site_(?P<site>[1-9]\d*)_(?P<member>[1-9]\d*)\.nc$")

#: Attribute names each source data variable must carry, exactly.
_SOURCE_VARIABLE_ATTRIBUTES = frozenset({"_FillValue", "long_name", "units"})


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


def _check_source_file_is_classic_with_no_global_attributes(handle: Any, path: Path) -> None:
    if handle.version_byte != 1:
        raise ValueError(
            f"{path}: netCDF-3 version byte is {handle.version_byte}, expected 1 (classic)"
        )
    attrs = _netcdf_attributes(handle)
    if attrs:
        raise ValueError(
            f"{path}: carries global attributes {sorted(attrs)}; source files carry none"
        )
    if set(handle.dimensions) != {"time"}:
        raise ValueError(
            f"{path}: dimensions are {sorted(handle.dimensions)}, expected exactly ['time']"
        )
    if handle.dimensions["time"] is not None:
        raise ValueError(f"{path}: the time dimension is not the unlimited record dimension")
    if handle._recs != 1:
        raise ValueError(f"{path}: time has {handle._recs} records, expected 1")


def _check_source_time_is_the_degenerate_template(handle: Any, path: Path) -> None:
    if "time" not in handle.variables:
        raise ValueError(f"{path}: has no time variable")
    time = handle.variables["time"]
    attrs = _netcdf_attributes(time)
    expected = {"units": SOURCE.time_units, "long_name": SOURCE.time_long_name}
    if attrs != expected:
        raise ValueError(
            f"{path}: time attributes are {attrs}, expected {expected}. A substituted year "
            "would mean issue #3 was fixed upstream; notice it rather than average it away."
        )
    value = np.asarray(time.data, dtype=np.float64).ravel()
    if value.size != 1 or value[0] != SOURCE.time_value:
        raise ValueError(f"{path}: time value is {value.tolist()}, expected [{SOURCE.time_value}]")


def _check_and_read_source_variables(handle: Any, path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, variable in handle.variables.items():
        if name == "time":
            continue
        if name not in SOURCE.variables:
            raise ValueError(
                f"{path}: variable {name!r} is not one the source files carry "
                f"({sorted(SOURCE.variables)}). A new variable is a spec change, not a new column."
            )
        if variable.dimensions != ("time",):
            raise ValueError(
                f"{path}: {name} has dims {variable.dimensions}, expected ('time',); a "
                "layer-resolved variable would look like this"
            )
        if variable.data.dtype.newbyteorder("=") != np.dtype(np.float64):
            raise ValueError(f"{path}: {name} is {variable.data.dtype}, expected float64")
        attrs = _netcdf_attributes(variable)
        expected = {
            "_FillValue": SOURCE.fill_value,
            "long_name": SOURCE.variables[name].long_name,
            "units": SOURCE.variables[name].units,
        }
        if attrs != expected:
            raise ValueError(
                f"{path}: {name} attributes are {attrs}, expected exactly {expected}. An "
                "unexpected scale_factor or add_offset would silently rescale the value."
            )
        data = np.asarray(variable.data, dtype=np.float64).ravel()
        if data.size != 1:
            raise ValueError(f"{path}: {name} holds {data.size} values, expected 1")
        value = float(data[0])
        if value == SOURCE.fill_value:
            raise ValueError(
                f"{path}: {name} holds the fill value {SOURCE.fill_value}. No file in the "
                "ensemble does, and the raw file has no representation for an explicit "
                "fill distinct from an absent variable."
            )
        if not np.isfinite(value):
            raise ValueError(f"{path}: {name} holds the non-finite value {value!r}")
        values[name] = value
    if not values:
        raise ValueError(f"{path}: carries no data variable")
    return values

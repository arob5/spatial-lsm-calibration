#!/usr/bin/env python
"""Build the initial-condition product.

Overview
--------
Walk the raw initial-condition tree, parse one small netCDF per
``(site, member)`` pair, and assemble them into the single netCDF the rest of
the project reads. The source is one small file per pair, hundreds of
thousands of them; the product is a single file of a few tens of megabytes,
which is the only form in which this ensemble exists off the SCC. The run
prints the sizes and counts it actually saw.

``sipnet_calibration.initial_conditions`` holds the schema, the per-file parser
and the reader. This script is the writer, and its own round-trip check reads
the file back with
:func:`sipnet_calibration.initial_conditions.load_initial_conditions` -- the
same function every consumer uses -- so the two cannot drift apart.

No parameter mapping and no unit conversion happen here. Initial conditions
reach SIPNET as parameters, and three of the four mappings depend on parameters
we calibrate, so the mapping is a modeling decision for the experiment layer;
see the Notes in the library module.

Input data
----------
``--root``, default ``data/raw/initial_conditions``
    One directory per site, named for the 1-8000 site identifier, holding one
    file per ensemble member named ``IC_site_<site>_<member>.nc`` with the
    source's 1-based member index. Each file is netCDF-3 classic with a
    length-1 unlimited ``time`` dimension and one ``float64`` scalar variable
    per initial-condition field, declaring ``_FillValue = -999.0``. Parsed by
    :func:`~sipnet_calibration.initial_conditions.read_ic_file`, which runs
    every per-file check.

``--sites``, default ``data/processed/sites/sites.csv``
    The site table, which supplies the site axis and the ``lon``/``lat``
    coordinates. Read through
    :func:`sipnet_calibration.sites.load_sites`.

Output data
-----------
``--out``, default ``data/processed/ic.nc``::

    initial_aboveground_wood_carbon(member, site)   float64, kg C m-2
    initial_wood_carbon(member, site)               float64, kg C m-2
    initial_soil_organic_carbon(member, site)       float64, kg C m-2
    ic_present(member, site)                        bool
    variable_present(member, site, variable)        bool

``site`` is the whole 1-8000 pool, whether or not a file exists for it, so that
the product's site axis is the pool rather than whichever sites happened to be
under the root. ``member`` is the discovered ensemble, 0-based, with
``source_member_index`` carrying the 1-based file index. There is no ``time``
dimension; what the source's degenerate one claimed is kept in the dataset
attributes. The run prints the coverage, the per-variable extremes and counts,
and the distinct variable-set signatures it saw.

Notes
-----
**A missing ``(site, member)`` file is fatal unless ``--allow-gaps``.** A
``NaN`` member would propagate silently through any statistic over members, so
an incomplete ensemble has to be opted into rather than discovered later. With
the flag the gaps are reported, filled with ``NaN``, recorded in
``ic_present``, and the dataset's ``coverage`` attribute becomes ``"gaps"``.
The flag is what makes the script runnable in a checkout holding only part of
the ensemble.

**The variable set is not settled, and an unregistered variable is fatal.**
Only the variables confirmed by inspecting files are registered, so a survey
of the full ensemble stops on a file carrying one of the others, with the
evidence needed to specify it. See open question 6 in ``data/README.md``.

**Three kinds of absence, two presence arrays.** A ``NaN`` in a data variable
means the pair had no file, or the file lacked the variable, or the file held an
explicit ``-999.0``. ``ic_present`` and ``variable_present`` tell them apart,
and ``variable_present & isnan(value)`` is the explicit-fill indicator. Both
arrays are written unconditionally so that consumers never branch on the
product's shape.

**Equality of the two wood variables is reported, not asserted.** The source's
``AbvGrndWood`` and ``wood_carbon_content`` are bitwise identical in every file
available, but de-duplicating them on that evidence would be a guess, and
asserting the equality would turn a legitimate file into a failure. Both are
kept and the count of disagreeing cells is printed.

**Non-physical values are counted, not clamped**, following the driver reader.
A negative carbon stock is meaningless, but clamping would hide an upstream
artifact, and no bound is established to assert against.

Output is written to a ``.partial`` path and renamed only once it reads back
through the library loader, so a failed check cannot leave a corrupt file where
the canonical one belongs.

Usage
-----
::

    python scripts/ingest_ic.py
    python scripts/ingest_ic.py --root /projectnb/dietzelab/.../IC --jobs 16

In a checkout holding only a few files::

    python scripts/ingest_ic.py --allow-gaps
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sipnet_calibration.initial_conditions import (
    IC_FILE_GLOB,
    IC_FILE_TEMPLATE,
    IC_PRESENT,
    IC_VARIABLE_ATTRS,
    IC_VARIABLES,
    MEMBER_SOURCE,
    SOURCE_FILL_VALUE,
    SOURCE_TIME_LONG_NAME,
    SOURCE_TIME_UNITS,
    SOURCE_TIME_VALUE,
    SOURCE_VARIABLE_NAMES,
    VARIABLE_PRESENT,
    IcFileContents,
    available_members,
    available_sites,
    default_ic_path,
    default_ic_root,
    load_initial_conditions,
    read_ic_file,
    site_member_from_file_name,
    time_attrs,
    variable_attrs,
)
from sipnet_calibration.sites import load_sites

#: Compression applied to every array, float and boolean alike.
COMPRESSION = {"zlib": True, "complevel": 4}

#: Default number of worker threads for the read. The read is dominated by
#: filesystem latency rather than by parsing, so concurrency is what makes a
#: full run tractable on a networked filesystem, and is worth raising there.
DEFAULT_JOBS = 8

#: Ceiling on the worker count. The read is I/O bound, so more threads than
#: this buys nothing, and an unbounded value would exhaust the process thread
#: limit at the full 800,000-file scale.
MAX_JOBS = 128


class IngestError(Exception):
    """A check failed, or an input is not what it claims to be."""


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the ingest, report, and return an exit status.

    Errors from the checks and from I/O are reported as messages rather than
    tracebacks, so a failed run says which invariant broke.
    """
    args = parse_args(argv)
    root = args.root if args.root is not None else default_ic_root()
    out = args.out if args.out is not None else default_ic_path()

    try:
        dataset, contents = ingest(
            root, args.sites, out, allow_gaps=args.allow_gaps, jobs=args.jobs
        )
    except (IngestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(describe_initial_conditions(dataset, contents))
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line: ``--root``, ``--sites``, ``--out``, ``--allow-gaps``,
    ``--jobs``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="The raw initial-condition root. Defaults to "
        "data/raw/initial_conditions.",
    )
    parser.add_argument(
        "--sites",
        type=Path,
        default=None,
        help="The site table. Defaults to data/processed/sites/sites.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write. Defaults to data/processed/ic.nc.",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Write a product with missing (site, member) pairs, filled with "
        "NaN and recorded in ic_present, instead of stopping. Needed in a "
        "checkout holding only part of the ensemble.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"Worker threads for the read (default {DEFAULT_JOBS}). The read "
        "is I/O bound, so this is worth raising on a networked filesystem.",
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────


def ingest(
    root: Path,
    sites_path: Path | None,
    out: Path,
    *,
    allow_gaps: bool,
    jobs: int,
) -> tuple[xr.Dataset, dict[tuple[int, int], IcFileContents]]:
    """Discover, read, check, build and write.

    Index the tree, load the site table, check the index against it, parse
    every file, check what spans files, build the grids and the Dataset,
    write it.

    Returns
    -------
    tuple
        What was written, and the parsed files, both for the report.
    """
    index = discover_files(root)
    sites = load_sites(sites_path)

    check_paths_follow_the_layout(index)
    check_no_duplicate_site_member_pairs(index)
    check_sites_are_in_the_site_table(index, sites)
    check_every_pool_site_has_a_directory(index, sites, allow_gaps=allow_gaps)
    check_members_are_the_same_at_every_site(index, allow_gaps=allow_gaps)

    contents = read_all_files(index, jobs=jobs)
    check_time_metadata_agrees_across_files(contents)
    check_source_long_names_agree_across_files(contents)

    grids = build_grids(contents, index, sites)
    check_every_variable_was_renamed(grids)

    dataset = build_dataset(grids, index, allow_gaps=allow_gaps)
    write_dataset(dataset, out)
    return dataset, contents


def discover_files(root: Path) -> FileIndex:
    """Index every initial-condition file under *root*.

    Walks the site directories, taking the site from the directory name and the
    member from the file name, and returns the paths keyed by
    ``(site, member)`` together with the discovered site and member sets.

    Filesystem debris is ignored rather than reported, but a file that *does*
    look like an initial-condition file and disagrees with its directory is
    kept, so that :func:`check_paths_follow_the_layout` can report it.

    Raises
    ------
    IngestError
        If *root* is not a directory, or holds no initial-condition file at
        all.
    """
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"initial-condition root {root} is not a directory")

    paths: dict[tuple[int, int], Path] = {}
    for site in available_sites(root):
        for member in available_members(root, site):
            paths[(site, member)] = root / str(site) / IC_FILE_TEMPLATE.format(
                site=site, member=member
            )

    # available_members reads the member out of the file name, so a file filed
    # under the wrong site directory resolves to a path that does not exist.
    # Recover the real path here so the layout check can name both numbers.
    for (site, member), path in list(paths.items()):
        if path.is_file():
            continue
        matches = [
            candidate
            for candidate in sorted((root / str(site)).glob("IC_site_*.nc"))
            if candidate.name.endswith(f"_{member}.nc")
        ]
        if matches:
            paths[(site, member)] = matches[0]

    if not paths:
        raise IngestError(
            f"no initial-condition files under {root}. Expected the layout "
            f"<site>/{IC_FILE_TEMPLATE.format(site='<site>', member='<member>')}."
        )

    sites = tuple(sorted({site for site, _ in paths}))
    members = tuple(sorted({member for _, member in paths}))
    return FileIndex(root=root, paths=paths, sites=sites, members=members)


def read_all_files(
    index: FileIndex, *, jobs: int
) -> dict[tuple[int, int], IcFileContents]:
    """Parse every indexed file, in parallel, keyed by ``(site, member)``.

    Each file goes through
    :func:`~sipnet_calibration.initial_conditions.read_ic_file`, so every
    per-file check runs here and a failure names the file.

    Threads rather than processes: the work is a stat, an open and about a
    kilobyte of read per file, so it is I/O bound and the GIL is not the
    constraint.

    Raises
    ------
    IngestError
        If any file fails its checks. The message names the file and the
        invariant, and the run stops rather than dropping the file, because a
        dropped file would become an indistinguishable ``NaN``.
    """
    keys = sorted(index.paths)
    contents: dict[tuple[int, int], IcFileContents] = {}
    workers = min(max(1, int(jobs)), MAX_JOBS)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(read_ic_file, index.paths[key]): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                contents[key] = future.result()
            except (ValueError, OSError) as error:
                raise IngestError(
                    f"site {key[0]} member {key[1]}: {error}"
                ) from error
    finally:
        # cancel_futures, because the default shutdown drains the whole
        # queue before the exception surfaces. At 800,000 files that means
        # a bad file found in the first second would keep the filesystem
        # busy for the rest of the run before reporting.
        pool.shutdown(wait=False, cancel_futures=True)
    return contents


def build_grids(
    contents: dict[tuple[int, int], IcFileContents],
    index: FileIndex,
    sites: pd.DataFrame,
) -> Grids:
    """Lay the parsed files out on the dense ``(member, site)`` grid.

    Every site of the pool gets a column whether or not a file exists for it,
    and every discovered member a row. Cells with no file stay ``NaN`` with
    both presence flags ``False``; a file that lacked a variable leaves that
    variable ``NaN`` with ``variable_present`` ``False``; a file that held the
    declared fill leaves ``NaN`` with ``variable_present`` ``True``.

    The source-to-processed rename is applied here, from the single mapping in
    the library. It is safe at this point because a parsed file names its
    variables, so a record carries its own identity and the rename cannot
    mis-pair a value with a variable.
    """
    site = sites["site_id"].to_numpy(np.int32)
    check_member_indices_are_representable(index)
    source_members = np.asarray(index.members, dtype=np.int16)
    shape = (source_members.size, site.size)

    values = {name: np.full(shape, np.nan) for name in IC_VARIABLES}
    ic_present = np.zeros(shape, dtype=bool)
    variable_present = np.zeros(shape + (len(IC_VARIABLES),), dtype=bool)
    explicit_fills = dict.fromkeys(IC_VARIABLES, 0)

    site_row = {int(value): row for row, value in enumerate(site)}
    member_row = {int(value): row for row, value in enumerate(source_members)}

    for (site_id, member), parsed in contents.items():
        i, j = member_row[member], site_row[site_id]
        ic_present[i, j] = True
        for source, value in parsed.values.items():
            name = SOURCE_VARIABLE_NAMES[source]
            values[name][i, j] = value
            variable_present[i, j, IC_VARIABLES.index(name)] = True
            if source in parsed.explicit_fills:
                explicit_fills[name] += 1

    return Grids(
        values=values,
        ic_present=ic_present,
        variable_present=variable_present,
        explicit_fills=explicit_fills,
        site=site,
        source_members=source_members,
        lon=sites["lon"].to_numpy(np.float64),
        lat=sites["lat"].to_numpy(np.float64),
    )


def build_dataset(grids: Grids, index: FileIndex, *, allow_gaps: bool) -> xr.Dataset:
    """Assemble the Dataset, with the attributes that travel with it."""
    dims = ("member", "site")
    data_vars: dict[str, xr.DataArray] = {
        name: xr.DataArray(grids.values[name], dims=dims) for name in IC_VARIABLES
    }
    data_vars[IC_PRESENT] = xr.DataArray(grids.ic_present, dims=dims)
    data_vars[VARIABLE_PRESENT] = xr.DataArray(
        grids.variable_present, dims=dims + ("variable",)
    )

    dataset = xr.Dataset(
        data_vars,
        coords={
            "member": np.arange(grids.source_members.size, dtype=np.int16),
            "source_member_index": ("member", grids.source_members),
            "site": grids.site,
            "lon": ("site", grids.lon),
            "lat": ("site", grids.lat),
            "variable": np.asarray(IC_VARIABLES, dtype=object),
        },
    )
    annotate_dataset(dataset, grids, index, allow_gaps=allow_gaps)
    return dataset


def write_dataset(dataset: xr.Dataset, out: Path) -> None:
    """Write to a ``.partial`` path, verify the round trip, then rename.

    A failed check leaves the partial file for inspection and nothing at the
    canonical path, so a later read cannot pick up a half-written product.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=netcdf_encoding())

    check_round_trip(dataset, partial)
    partial.replace(out)


def describe_initial_conditions(
    dataset: xr.Dataset,
    contents: dict[tuple[int, int], IcFileContents],
) -> str:
    """A report of what was written, for the run log.

    Reports the coverage, the number of sites and members and pairs found
    against those expected, and per variable ``carried`` (files that held
    it), ``n`` (of those, the ones with a value rather than a fill), its
    extremes, its explicit-fill count and its count of non-positive values.
    Also the distinct variable-set signatures with their file counts, and
    the number of cells where the two wood variables are both present and
    unequal -- the measurement that would settle whether they are
    duplicates.

    ``carried`` and ``n`` differ by exactly the explicit fills, which is
    why both are printed.
    """
    n_members, n_sites = dataset.sizes["member"], dataset.sizes["site"]
    present = int(dataset[IC_PRESENT].values.sum())
    expected = n_members * n_sites
    lines = [
        f"members {n_members}  sites {n_sites}  variables {len(IC_VARIABLES)}",
        f"source member indices {dataset['source_member_index'].values.tolist()}",
        f"coverage {dataset.attrs['coverage']}: {present} of {expected} "
        f"(site, member) pairs have a file ({present / expected:.2%})",
        f"sites with at least one file "
        f"{int((dataset[IC_PRESENT].values.any(axis=0)).sum())}",
    ]

    for index, name in enumerate(IC_VARIABLES):
        array = dataset[name]
        finite = np.isfinite(array.values)
        carried = int(
            dataset[VARIABLE_PRESENT]
            .transpose("member", "site", "variable")
            .values[:, :, index]
            .sum()
        )
        extent = (
            f"[{array.values[finite].min():.6g}, {array.values[finite].max():.6g}]"
            if finite.any()
            else "[none present]"
        )
        lines.append(
            f"  {name:<32s} carried={carried:>6d}  n={int(finite.sum()):>6d}  "
            f"range {extent}  fills {array.attrs['n_explicit_fills']}  "
            f"non-positive {array.attrs['n_values_not_positive']}"
        )

    lines.append("variable-set signatures across the files read:")
    for signature, count in sorted(
        _variable_set_signatures(contents).items(), key=lambda item: -item[1]
    ):
        lines.append(f"  {count:>7d} file(s): {list(signature)}")

    disagreeing = _wood_variables_disagreeing(dataset)
    lines.append(
        f"cells where {IC_VARIABLES[0]} and {IC_VARIABLES[1]} are both present "
        f"and unequal: {disagreeing}"
    )
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


@dataclass(frozen=True)
class FileIndex:
    """Every initial-condition file found, and the axes they imply.

    Attributes
    ----------
    root:
        The raw root the walk started from, recorded into the product.
    paths:
        ``(site, member)`` -> path, with *source* member indices.
    sites:
        Site identifiers found, ascending. Not the product's site axis, which
        is the whole pool; this is what the tree actually holds.
    members:
        Source member indices found, ascending, over all sites. This *is* the
        product's member axis.
    """

    root: Path
    paths: dict[tuple[int, int], Path]
    sites: tuple[int, ...]
    members: tuple[int, ...]


@dataclass(frozen=True)
class Grids:
    """The dense arrays and the axes they are indexed on.

    Attributes
    ----------
    values:
        Processed variable name -> ``(member, site)`` ``float64`` array.
    ic_present:
        ``(member, site)`` boolean: a file existed for the pair.
    variable_present:
        ``(member, site, variable)`` boolean: the file carried the variable.
    explicit_fills:
        Processed variable name -> how many cells held the declared fill,
        for the variable attributes.
    site:
        The full pool, ``int32``, ascending.
    source_members:
        The discovered 1-based member indices, ``int16``, ascending.
    lon, lat:
        Site coordinates, ``float64``, aligned to *site*.
    """

    values: dict[str, np.ndarray]
    ic_present: np.ndarray
    variable_present: np.ndarray
    explicit_fills: dict[str, int]
    site: np.ndarray
    source_members: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


def netcdf_encoding() -> dict:
    """Explicit on-disk encoding, so nothing is inherited from a default.

    Compression on all five arrays, and ``_FillValue = NaN`` on the three
    float variables only -- the presence companions have no missing state, and
    giving a boolean array a fill value would invent one.
    """
    encoding = {
        name: {**COMPRESSION, "_FillValue": np.nan} for name in IC_VARIABLES
    }
    for name in (IC_PRESENT, VARIABLE_PRESENT):
        encoding[name] = dict(COMPRESSION)
    return encoding


def annotate_dataset(
    dataset: xr.Dataset, grids: Grids, index: FileIndex, *, allow_gaps: bool
) -> None:
    """Attach the attributes that have to travel with the product.

    Per variable: the registered metadata, the units status and provenance, the
    counterpart constraint variable where there is one, and the two runtime
    counts. On the coordinates: what ``member``, ``source_member_index``,
    ``site`` and ``variable`` mean. On the dataset: the title, the source root
    and layout, the source fill value, the history, ``member_source``,
    ``member_correspondence``, the sizes, the coverage, and the five
    attributes recording the dropped ``time`` coordinate.

    ``member_correspondence`` says that no correspondence with the driver or
    net-ecosystem-exchange ensembles is established, because xarray aligns
    integer member labels silently and the wrong assumption would combine
    unrelated members without any error. Open question 12 in
    ``data/README.md``.
    """
    for name in IC_VARIABLES:
        values = dataset[name].values
        finite = np.isfinite(values)
        dataset[name].attrs = {
            **variable_attrs(name),
            "n_explicit_fills": int(grids.explicit_fills[name]),
            "n_values_not_positive": int((values[finite] <= 0).sum()),
        }

    dataset[IC_PRESENT].attrs = {
        "long_name": "Whether a file existed for the member and site",
        "comment": (
            "Every initial-condition variable is NaN where this is False. "
            f"{VARIABLE_PRESENT} is False there too."
        ),
    }
    dataset[VARIABLE_PRESENT].attrs = {
        "long_name": "Whether the file carried the variable",
        "comment": (
            "False both where no file existed and where the file did not carry "
            "the variable. True with a NaN value means the file carried an "
            f"explicit {SOURCE_FILL_VALUE} fill, which is the only way to tell "
            "the two apart."
        ),
    }

    dataset["member"].attrs = {
        "long_name": "Ensemble member",
        "comment": (
            "0-based, meaningful only within this source; source_member_index "
            "is the 1-based index in the file name."
        ),
    }
    dataset["source_member_index"].attrs = {
        "long_name": "Member index in the source file name (1-based)"
    }
    dataset["site"].attrs = {
        "long_name": "Model site identifier",
        "comment": "The handed-down 1-8000 identifier; never renumbered.",
    }
    dataset["variable"].attrs = {
        "long_name": "Initial-condition variable",
        "comment": (
            "Processed names, indexing variable_present only. Each variable's "
            "source name is the source_name attribute of the variable itself."
        ),
    }

    complete = bool(grids.ic_present.all())
    dataset.attrs = {
        "title": "SIPNET initial-condition ensemble in source units",
        "source_root": str(index.root),
        "source_layout": (
            f"<site>/{IC_FILE_TEMPLATE.format(site='<site>', member='<member>')}"
        ),
        "source_fill_value": SOURCE_FILL_VALUE,
        "history": "scripts/ingest_ic.py",
        "member_source": MEMBER_SOURCE,
        "member_correspondence": (
            "Not established. Whether initial-condition member i corresponds "
            "to driver or NEE member i is open question 12 in "
            "data/README.md; nothing here assumes it does. The ensembles are "
            "different sizes, which argues against a simple pairing."
        ),
        "n_sites": int(grids.site.size),
        "n_members": int(grids.source_members.size),
        "coverage": "complete" if complete else "gaps",
        **time_attrs(),
    }
    if not complete and allow_gaps:
        dataset.attrs["coverage_note"] = (
            "Written with --allow-gaps: some (site, member) pairs had no file. "
            f"{IC_PRESENT} says which."
        )


def _wood_variables_disagreeing(dataset: xr.Dataset) -> int:
    """Cells where both wood variables are present and unequal.

    Compared bitwise, since equality is what is in question and a tolerance
    would decide the answer in advance. Reported, never asserted.
    """
    first, second = dataset[IC_VARIABLES[0]].values, dataset[IC_VARIABLES[1]].values
    present = dataset[VARIABLE_PRESENT].transpose("member", "site", "variable").values
    # Presence, not finiteness: a cell where one variable is an explicit
    # fill and the other a real number is a disagreement, and finiteness
    # would drop it. A NaN-ness mismatch between two present cells is one
    # as well.
    both = present[:, :, 0] & present[:, :, 1]
    first, second = first[both], second[both]
    nan_first, nan_second = np.isnan(first), np.isnan(second)
    differing_nan = nan_first != nan_second
    differing_value = ~nan_first & ~nan_second & (first != second)
    return int(np.count_nonzero(differing_nan | differing_value))


def _variable_set_signatures(
    contents: dict[tuple[int, int], IcFileContents],
) -> dict[tuple[str, ...], int]:
    """Distinct sorted variable-name tuples, with how many files carried each.

    This is the measurement that answers the open variable-set question, which
    is why the run prints it whether or not it is uniform.
    """
    signatures: dict[tuple[str, ...], int] = {}
    for parsed in contents.values():
        signature = tuple(sorted(parsed.values))
        signatures[signature] = signatures.get(signature, 0) + 1
    return signatures


def _sample(values, limit: int = 8) -> str:
    """The first *limit* of *values*, with a count of what is not shown.

    Coverage failures can name thousands of sites, and a message that long
    buries the invariant it is reporting.
    """
    listed = list(values)
    head = listed[:limit]
    if len(listed) <= limit:
        return str(head)
    return f"{head} and {len(listed) - limit} more"


# ── checks ────────────────────────────────────────────────────────────────────


def check_paths_follow_the_layout(index: FileIndex) -> None:
    """Every path is ``<site>/IC_site_<site>_<member>.nc``, sites agreeing.

    The directory's site identifier must equal the one embedded in the file
    name. Once the arrays are built nothing records which path a cell came
    from, so a file filed under the wrong site would be unattributable
    afterwards. Raises naming the offending paths and which two numbers
    disagree.
    """
    wrong = []
    for (site, member), path in sorted(index.paths.items()):
        expected = IC_FILE_TEMPLATE.format(site=site, member=member)
        if path.name != expected:
            wrong.append(
                f"{path}: directory says site {site}, file name says "
                f"{path.name!r} (expected {expected!r})"
            )
    if wrong:
        raise IngestError(
            "initial-condition files disagree with their directories:\n  "
            + "\n  ".join(wrong)
            + "\nThe product records no path per cell, so a misfiled file "
            "cannot be attributed after the arrays are built."
        )


def check_no_duplicate_site_member_pairs(index: FileIndex) -> None:
    """No two paths resolve to one ``(site, member)`` cell.

    The layout promises one file per pair. Two would mean one silently
    overwrote the other in the grid, with no way to tell which was kept.
    """
    seen: dict[tuple[int, int], list[Path]] = {}
    for site in index.sites:
        for path in sorted((index.root / str(site)).glob(IC_FILE_GLOB)):
            # The same parse and the same is_file() test discovery uses, so
            # a directory, a dangling symlink or a name off the template is
            # debris to both rather than a phantom duplicate here.
            parsed = site_member_from_file_name(path.name)
            if parsed is None or not path.is_file():
                continue
            key = (site, parsed[1])
            seen.setdefault(key, []).append(path)
    duplicates = {key: paths for key, paths in seen.items() if len(paths) > 1}
    if duplicates:
        listed = "\n  ".join(
            f"site {site} member {member}: {[str(p) for p in paths]}"
            for (site, member), paths in sorted(duplicates.items())
        )
        raise IngestError(
            "more than one file resolves to the same (site, member) cell:\n  "
            f"{listed}\nThe layout promises one file per pair; one would "
            "silently overwrite the other."
        )


def check_sites_are_in_the_site_table(index: FileIndex, sites: pd.DataFrame) -> None:
    """Every discovered site identifier is in the site table.

    A site under the root that the pool does not know about means the two
    disagree about what the site identifiers are, and the identifiers are a
    shared key with collaborators. Raises listing the unknown identifiers.
    """
    known = set(sites["site_id"].to_numpy().tolist())
    unknown = sorted(site for site in index.sites if site not in known)
    if unknown:
        raise IngestError(
            f"{len(unknown)} site(s) under {index.root} are not in the site "
            f"table: {_sample(unknown)}. The 1-8000 identifiers are a shared "
            "key with collaborators, so a tree that disagrees with the pool "
            "is an error rather than an extension."
        )


def check_every_pool_site_has_a_directory(
    index: FileIndex, sites: pd.DataFrame, *, allow_gaps: bool
) -> None:
    """Every site in the pool has a directory under the root.

    Fatal unless *allow_gaps*, and reports how many are absent with a sample of
    the identifiers rather than all of them, since a full pool would bury the
    message. This is the check a partial checkout trips.
    """
    if allow_gaps:
        return
    pool = sites["site_id"].to_numpy().tolist()
    found = set(index.sites)
    absent = [site for site in pool if int(site) not in found]
    if absent:
        raise IngestError(
            f"{len(absent)} of {len(pool)} pool sites have no "
            f"initial-condition file under {index.root}: {_sample(absent)}. "
            "A site whose directory exists but holds nothing parseable "
            "counts here too. Pass --allow-gaps to write a product for the "
            "sites that are present, filling the rest with NaN and "
            "recording it in ic_present."
        )


def check_members_are_the_same_at_every_site(
    index: FileIndex, *, allow_gaps: bool
) -> None:
    """Every site that has any file has the whole discovered member set.

    A site missing one member of an otherwise complete ensemble is the case
    that would quietly skew a statistic over members.
    Fatal unless *allow_gaps*, and reports the sites and the members they lack.
    """
    if allow_gaps:
        return
    expected = set(index.members)
    ragged = []
    for site in index.sites:
        found = {member for (site_id, member) in index.paths if site_id == site}
        missing = sorted(expected - found)
        if missing:
            ragged.append(f"site {site} lacks member(s) {_sample(missing)}")
    if ragged:
        raise IngestError(
            "the ensemble is ragged over sites:\n  "
            + "\n  ".join(ragged)
            + f"\nThe {len(expected)} member indices found across the tree are "
            f"{_sample(sorted(expected))}. A site missing one member would "
            "skew any statistic taken over members. Pass --allow-gaps to write "
            "it anyway, recorded in ic_present."
        )


def check_member_indices_are_representable(index: FileIndex) -> None:
    """Every discovered member index fits the ``member`` axis's dtype.

    The axis is ``int16`` to match every other product. A file name
    carrying a larger number would otherwise abort the run with a bare
    ``OverflowError`` from the array cast, which is neither a reported
    error nor a named invariant.
    """
    limit = int(np.iinfo(np.int16).max)
    beyond = [member for member in index.members if member > limit]
    if beyond:
        raise IngestError(
            f"member indices {_sample(beyond)} exceed {limit}, the largest "
            "the int16 member axis holds. The source's member index comes "
            "from the file name, so this is a file named outside the "
            "layout's range rather than a limit worth raising."
        )


def check_time_metadata_agrees_across_files(
    contents: dict[tuple[int, int], IcFileContents],
) -> None:
    """Every file's ``time`` units, long name and value are the registered ones.

    The product keeps one copy of this metadata in its attributes, so a file
    that disagreed would be invisible afterwards -- and a file whose ``units``
    were *substituted* rather than the ``[year]`` template would mean issue #3
    had been fixed upstream, which is a thing to notice rather than to average
    away.
    """
    expected = (SOURCE_TIME_UNITS, SOURCE_TIME_LONG_NAME, SOURCE_TIME_VALUE)
    for (site, member), parsed in sorted(contents.items()):
        found = (parsed.time_units, parsed.time_long_name, parsed.time_value)
        if found != expected:
            raise IngestError(
                f"site {site} member {member}: time metadata is {found!r}, "
                f"expected {expected!r}. The product records one copy of this, "
                "so a file that disagrees would be invisible afterwards. A "
                "substituted year here would mean issue #3 was fixed upstream, "
                "which is a thing to act on rather than to average away."
            )


def check_source_long_names_agree_across_files(
    contents: dict[tuple[int, int], IcFileContents],
) -> None:
    """Every file's per-variable ``long_name`` is the registered one.

    Only one copy survives into the product. A file where the same variable
    name carried a different long name would mean the name does not have one
    meaning across the ensemble, which the schema assumes it does.
    """
    for (site, member), parsed in sorted(contents.items()):
        for source, found in parsed.long_names.items():
            processed = SOURCE_VARIABLE_NAMES[source]
            expected = IC_VARIABLE_ATTRS[processed]["source_long_name"]
            if found != expected:
                raise IngestError(
                    f"site {site} member {member}: variable {source!r} has "
                    f"long_name {found!r}, expected {expected!r}. The schema "
                    "assumes a source variable name means one thing across the "
                    "ensemble, and only one copy reaches the product."
                )


def check_every_variable_was_renamed(grids: Grids) -> None:
    """The source-to-processed mapping covered every variable read.

    A source variable with no processed name would otherwise be dropped between
    the parse and the grid, silently narrowing the product.
    """
    missing = [name for name in IC_VARIABLES if name not in grids.values]
    extra = [name for name in grids.values if name not in IC_VARIABLES]
    if missing or extra:
        raise IngestError(
            f"the grid holds {sorted(grids.values)}, expected exactly "
            f"{list(IC_VARIABLES)} (missing {missing}, unexpected {extra}). "
            "A source variable with no processed name would be dropped "
            "silently between the parse and the grid."
        )


def check_round_trip(dataset: xr.Dataset, path: Path) -> None:
    """The written file reads back through the library loader unchanged.

    Reads *path* with
    :func:`~sipnet_calibration.initial_conditions.load_initial_conditions` --
    the same function every consumer uses, never a parallel reader -- and
    compares values, dtypes, coordinates and attributes. This is what keeps the
    writer here and the schema in the library from drifting apart, and it runs
    before the ``.partial`` file is renamed.
    """
    with load_initial_conditions(path) as back:
        for name in list(IC_VARIABLES) + [IC_PRESENT, VARIABLE_PRESENT]:
            written, read = dataset[name], back[name]
            if written.dtype != read.dtype:
                raise IngestError(
                    f"{path}: {name!r} was written as {written.dtype} and read "
                    f"back as {read.dtype}."
                )
            if not np.array_equal(written.values, read.values, equal_nan=True):
                raise IngestError(
                    f"{path}: {name!r} does not read back as it was written."
                )
            if dict(written.attrs) != dict(read.attrs):
                raise IngestError(
                    f"{path}: {name!r} attributes changed on the round trip."
                )

        for coord in ("member", "source_member_index", "site", "lon", "lat"):
            if not np.array_equal(dataset[coord].values, back[coord].values):
                raise IngestError(f"{path}: coordinate {coord!r} does not round-trip.")
        written_vars = [str(v) for v in dataset["variable"].values]
        read_vars = [str(v) for v in back["variable"].values]
        if written_vars != read_vars:
            raise IngestError(f"{path}: coordinate 'variable' does not round-trip.")

        if dict(dataset.attrs) != dict(back.attrs):
            differing = sorted(
                key
                for key in set(dataset.attrs) | set(back.attrs)
                if dataset.attrs.get(key) != back.attrs.get(key)
            )
            raise IngestError(
                f"{path}: dataset attributes changed on the round trip: {differing}"
            )


if __name__ == "__main__":
    sys.exit(main())

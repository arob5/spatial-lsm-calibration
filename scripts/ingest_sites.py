#!/usr/bin/env python3
"""Build the site table.

Overview
--------
Read the point shapefile that defines the 8000-site pool and write it as a CSV,
carrying every field of the shapefile plus the grid indices and the Ameriflux
identifier. The shapefile is the only input under ``data/raw/`` that is tracked
in version control, so this is the one ingest script that runs end to end on a
laptop, and every other product joins against its output on ``site_id``.

Input data
----------
``--shapefile``
    ``data/raw/sites/pts.shp`` and companions: one single-point ``POINT``
    record per site, in descending latitude, with record *N* being site *N*.
    Geometry is float64 lon/lat on WGS 84. The ``.dbf`` attribute table holds
    ``site_id``, ``site_names``, ``site_order``, ``cluster`` and ``landcover``;
    the three numeric fields are declared with 15 decimals and so arrive as
    floats. ``pts.cpg`` declares the ``.dbf`` encoding, which is UTF-8.
    :data:`N_SITES` is the pool size the checks require.

``--site-id-map``
    ``data/site_id_map.csv``: rows of ``Site_ID, index``, mapping an Ameriflux
    identifier onto a site id by exact match. It covers a subset of the pool.

Output data
-----------
``--out``, default ``data/processed/sites/sites.csv``, one row per site::

    site_id, lon, lat, lon_index, lat_index, site_name, site_order,
    cluster, landcover, ameriflux_site_id

``site_name`` is the shapefile's ``site_names`` renamed to the singular, and
``ameriflux_site_id`` is that file's ``Site_ID`` renamed to say which identifier
it means; it is the empty string for sites with no counterpart. The
column set and dtypes come from ``SITE_COLUMNS`` and ``SITE_COLUMN_DTYPES`` in
:mod:`sipnet_calibration.sites`, so this writer and
:func:`sipnet_calibration.sites.load_sites` cannot drift apart. There is
deliberately no ``pft`` column: a labeling is an experimental choice, not site
metadata.

Notes
-----
Two things about this table are easy to get wrong and quiet when they go wrong,
and most of the checking exists to make them loud.

**The encoding.** A few site names carry non-ASCII bytes. Read as latin-1 they
do not raise; they decode to ``'RayÃ³n (MX-Ray)'``, which still looks like a
plausible site label. So the declared encoding is read from the ``.cpg`` and
asserted rather than left to a library default, and ``--encoding`` rejects
anything that disagrees rather than honoring it. The run reports how many such
names it saw.

**The coordinates.** The geometry is float64 and CSV formatting is where that
precision goes, which is why the output is read back and compared **bitwise**
before the script reports success. That check earns its place -- it caught
pandas' default CSV parser reading a fraction of the longitudes back inexactly.
The fix was the *reader*, ``float_precision="round_trip"`` in
:func:`sipnet_calibration.sites.load_sites`, not the write format, which is
inexact under the default parser either way. See :data:`FLOAT_FORMAT`.

A ``.dbf`` null means different things in the two kinds of column, so it is
handled two ways. In a **text** column it becomes the empty string, which is
what "missing" already means there, and is the only marker that survives a read
with ``keep_default_na=False``. In a **numeric** column it is an error: an
integer dtype cannot hold ``NaN``, ``cluster`` and ``landcover`` have no spare
value, and ``site_order``'s 0 already means "a sampled point", so there is
nowhere to put it and nothing to do but say so.

The checks are named individually in the ``checks`` section below. Before
writing: the declared encoding, the record and shape counts, ``site_id`` being
exactly ``1..N_SITES`` in record order, the numeric fields being integral,
``site_order``'s non-zero values being a permutation of 1..1093, every
coordinate resolving on ``SITE_GRID`` to a distinct index pair, and the
Ameriflux map naming each site at most once and only sites that exist. After
writing: the bitwise round trip, which also catches the sites named literally
``NA``.

Run it from the project environment: it imports ``sipnet_calibration``, numpy,
pandas and ``pyshp``, so a bare system ``python3`` fails on the first import.
Either activate the environment, as the README describes, or use ``uv run``.

Usage
-----
::

    python scripts/ingest_sites.py
    uv run scripts/ingest_sites.py
    python scripts/ingest_sites.py --out /tmp/sites.csv
    python scripts/ingest_sites.py --help
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

from sipnet_calibration.sites import (
    SITE_COLUMN_DTYPES,
    SITE_COLUMNS,
    SITE_GRID,
    default_sites_path,
    load_sites,
)

#: Repository root, as seen from ``scripts/``.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SHAPEFILE = REPO_ROOT / "data" / "raw" / "sites" / "pts.shp"
DEFAULT_SITE_ID_MAP = REPO_ROOT / "data" / "site_id_map.csv"
#: Where the table is written, which is exactly where
#: :func:`sipnet_calibration.sites.load_sites` will look for it -- including
#: when ``$SIPNET_CALIBRATION_DATA`` redirects both. Hard-coding the checkout
#: path here meant that, with that variable set, a default run wrote one place
#: and every consumer read another, and the run still reported success.
DEFAULT_OUT = default_sites_path()

#: The encoding ``pts.cpg`` is expected to declare, normalized by
#: :func:`normalize_encoding`.
EXPECTED_ENCODING = "utf-8"

#: Site count of the pool. The identifiers are handed down and shared with
#: collaborators' files, so this is a fixed property of the data, not a
#: configurable one.
N_SITES = 8000

#: Highest ``site_order`` value; the non-zero values are a permutation of
#: ``1..NAMED_SITE_COUNT`` and the remaining sites carry 0.
NAMED_SITE_COUNT = 1093

#: ``float_format`` for the coordinate columns: ``None``, meaning pandas' default
#: of ``repr``, the shortest decimal string that reads back as the same float64.
#:
#: **The write format is not what makes the round trip exact; the read setting
#: is.** The C parser ``pandas.read_csv`` uses by default is inexact for both
#: candidates, so writing shorter does not rescue it:
#:
#: =================  ====================  ==========================
#: written as         default parser        ``float_precision`` set
#: =================  ====================  ==========================
#: ``repr`` (this)    some rows inexact     exact
#: ``"%.17g"``        some rows inexact     exact
#: =================  ====================  ==========================
#:
#: ``test_the_reader_setting_is_what_makes_the_round_trip_exact`` measures both
#: columns, so the numbers live there rather than here.
#:
#: ``repr`` is kept because it is shortest and is exact by construction under
#: Python's own ``float()``, so the file is right for any reader that parses
#: correctly. What guarantees this project's round trip is
#: :func:`sipnet_calibration.sites.load_sites` reading with
#: ``float_precision="round_trip"``; ``check_csv_round_trip`` is what proves it.
FLOAT_FORMAT = None

class IngestError(Exception):
    """Raised when an invariant of the input or the output does not hold."""


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Read the shapefile, build the table, write it, and prove nothing was lost."""
    args = parse_args(argv)
    if not args.shapefile.is_file():
        print(f"error: {args.shapefile} is not a file", file=sys.stderr)
        return 2
    if not args.site_id_map.is_file():
        print(f"error: {args.site_id_map} is not a file", file=sys.stderr)
        return 2

    print(f"Reading {args.shapefile} as {args.encoding} ...", file=sys.stderr)
    try:
        contents = read_shapefile(args.shapefile, encoding=args.encoding)
        check_encoding_is_utf8(contents, encoding_used=args.encoding)
        ameriflux = read_ameriflux_map(args.site_id_map)
        table = build_site_table(contents, ameriflux)
        write_checked_site_table(table, args.out)
    except IngestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - reported, never a traceback
        # Reading a shapefile, a CSV and a .cpg raises plenty that is not an
        # IngestError: ValueError from an off-grid coordinate, EmptyDataError
        # from a truncated map, LookupError from a bad --encoding,
        # IsADirectoryError from --out. None of those is a bug in this script,
        # so none should reach the user as a traceback.
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"\nWrote {args.out}")
    print(describe_site_table(table))
    print("\nThe CSV round trip is bitwise exact for lon and lat.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help=f"Point shapefile to read. Default {DEFAULT_SHAPEFILE}.",
    )
    parser.add_argument(
        "--site-id-map",
        type=Path,
        default=DEFAULT_SITE_ID_MAP,
        help=f"Ameriflux identifier map. Default {DEFAULT_SITE_ID_MAP}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"CSV to write, creating its directory. Default {DEFAULT_OUT}.",
    )
    parser.add_argument(
        "--encoding",
        default=EXPECTED_ENCODING,
        help=(
            "Encoding of the .dbf attribute table. Default "
            f"{EXPECTED_ENCODING}, which is what the .cpg declares; a value "
            "disagreeing with the .cpg is rejected rather than honored."
        ),
    )
    return parser.parse_args(argv)


# ── the ingest steps, in the order main calls them ────────────────────────────
#
# Each step is named for what it produces. Filesystem access is confined to
# the two readers and the writer; everything between them takes plain Python
# data and returns plain Python data, so it is testable without fixtures on
# disk.


def read_shapefile(shp_path: Path, *, encoding: str) -> ShapefileContents:
    """Read a point shapefile's geometry and attribute table into memory.

    *encoding* is passed to the reader explicitly rather than being inferred, so
    that a mismatch with the ``.cpg`` is something a check reports rather than
    something the library resolves silently.
    """
    declared = read_declared_encoding(shp_path)
    with shapefile.Reader(str(shp_path), encoding=encoding) as reader:
        # fields[0] is the deletion flag, which is not a real attribute.
        field_names = tuple(field[0] for field in reader.fields[1:])
        records = tuple(
            {name: record[name] for name in field_names} for record in reader.records()
        )
        shapes = reader.shapes()
        shape_types = tuple(int(shape.shapeType) for shape in shapes)
        points = tuple(
            tuple((float(x), float(y)) for x, y in shape.points) for shape in shapes
        )
    return ShapefileContents(
        declared_encoding=declared,
        field_names=field_names,
        records=records,
        shape_types=shape_types,
        points=points,
    )


def read_ameriflux_map(path: Path) -> dict[int, str]:
    """Integer site id -> Ameriflux ``Site_ID``, from ``site_id_map.csv``.

    The file's columns are ``Site_ID`` and ``index``; ``index`` is the integer
    site id. The processed table calls the identifier ``ameriflux_site_id``,
    because ``Site_ID`` is opaque about which of the two identifiers it means.
    """
    frame = pd.read_csv(
        path,
        dtype={"Site_ID": str, "index": np.int64},
        keep_default_na=False,
        na_values=[],
    )
    missing = {"Site_ID", "index"} - set(frame.columns)
    if missing:
        raise IngestError(f"{path} is missing column(s) {sorted(missing)}")
    check_ameriflux_rows_are_one_per_site(frame, source=path)
    return dict(zip(frame["index"].tolist(), frame["Site_ID"].tolist(), strict=True))


def build_site_table(
    contents: ShapefileContents, ameriflux: dict[int, str]
) -> pd.DataFrame:
    """The processed site table, with the columns of :data:`SITE_COLUMNS`.

    Every invariant this script asserts about its input is checked here, so a
    caller cannot get an unvalidated table. Returns a frame of
    :data:`N_SITES` rows in ascending ``site_id`` order, which for this
    shapefile is also record order.
    """
    check_record_count(contents)
    check_shapes_are_single_points(contents)
    check_fields_are_present(
        contents,
        required=("site_id", "site_names", "site_order", "cluster", "landcover"),
    )

    site_ids = _as_integer(
        numeric_field(contents.records, "site_id"),
        name="site_id",
        dtype=np.int32,
    )
    check_site_ids_are_the_full_range(site_ids)

    lon = np.array([points[0][0] for points in contents.points], dtype=np.float64)
    lat = np.array([points[0][1] for points in contents.points], dtype=np.float64)
    check_coordinates_are_finite(lon, lat)
    # Raises if any coordinate is further than the default tolerance from a cell
    # center, which would mean the wrong grid or the wrong CRS.
    lon_index, lat_index = SITE_GRID.lonlat_to_index(lon, lat)
    check_index_pairs_are_distinct(lon_index, lat_index)

    integral = {
        field: _as_integer(
            numeric_field(contents.records, field),
            name=field,
            dtype=SITE_COLUMN_DTYPES[field],
        )
        for field in ("site_order", "cluster", "landcover")
    }
    check_site_order_is_a_permutation(integral["site_order"])

    names = text_field(contents.records, "site_names")

    check_ameriflux_map_is_usable(ameriflux, site_ids=site_ids)
    ameriflux_column = [ameriflux.get(int(site_id), "") for site_id in site_ids]

    table = pd.DataFrame(
        {
            "site_id": site_ids,
            "lon": lon,
            "lat": lat,
            "lon_index": _as_integer(lon_index, name="lon_index", dtype=np.int32),
            "lat_index": _as_integer(lat_index, name="lat_index", dtype=np.int32),
            "site_name": names,
            "site_order": integral["site_order"],
            "cluster": integral["cluster"],
            "landcover": integral["landcover"],
            "ameriflux_site_id": ameriflux_column,
        }
    )
    return table[list(SITE_COLUMNS)]


def write_site_table(table: pd.DataFrame, out_path: Path) -> None:
    """Write the table as CSV, creating its directory if it is absent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format=FLOAT_FORMAT)


def describe_site_table(table: pd.DataFrame) -> str:
    """A short summary of the written table, for the terminal."""
    named = table["site_order"] != 0
    non_ascii = [
        f"{row.site_id} {row.site_name!r}"
        for row in table.itertuples()
        if not row.site_name.isascii()
    ]
    mapped = table["ameriflux_site_id"] != ""
    na_hazards = na_hazard_site_ids(
        table["site_name"].tolist(), site_ids=table["site_id"].to_numpy()
    )
    lines = [
        f"  rows                  : {len(table)}",
        f"  site_id               : {table.site_id.min()}..{table.site_id.max()}",
        f"  longitude             : {table.lon.min():.4f} to {table.lon.max():.4f}",
        f"  latitude              : {table.lat.min():.4f} to {table.lat.max():.4f}",
        f"  named sites           : {int(named.sum())}"
        f" (site_order 1..{int(table.site_order.max())})",
        f"  sampled points        : {int((~named).sum())}",
        f"  cluster               : {sorted(table.cluster.unique().tolist())}",
        f"  landcover             : {sorted(table.landcover.unique().tolist())}",
        f"  Ameriflux identifiers : {int(mapped.sum())}",
        f"  non-ASCII site names  : {len(non_ascii)}",
    ]
    lines += [f"      {entry}" for entry in non_ascii]
    lines.append(
        f"  names a default read_csv would null: {len(na_hazards)}"
        + (f" (sites {na_hazards})" if na_hazards else "")
    )
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────

@dataclass(frozen=True)
class ShapefileContents:
    """The parts of a point shapefile this script uses.

    Attributes
    ----------
    declared_encoding:
        Whatever the ``.cpg`` file said, normalized, or ``None`` if there is no
        ``.cpg``. Kept rather than acted on so a check can report it.
    field_names:
        Attribute field names in ``.dbf`` order, with the deletion flag dropped.
    records:
        One dict per record, keyed by field name, in record order.
    shape_types:
        The ``shapeType`` of each shape, in record order.
    points:
        The points of each shape, in record order. A well-formed point
        shapefile has exactly one per shape; this keeps them all so that a check
        can say otherwise.
    """

    declared_encoding: str | None
    field_names: tuple[str, ...]
    records: tuple[dict[str, object], ...]
    shape_types: tuple[int, ...]
    points: tuple[tuple[tuple[float, float], ...], ...]


def normalize_encoding(name: str) -> str:
    """A codec name in a comparable form: ``'UTF-8 '`` -> ``'utf-8'``."""
    return name.strip().lower().replace("_", "-")


def read_declared_encoding(shp_path: Path) -> str | None:
    """The encoding declared by the ``.cpg`` beside *shp_path*, if there is one.

    Returns ``None`` when the file is absent, which is not itself an error: it is
    :func:`check_encoding_is_utf8` that decides what to do about it.
    """
    cpg_path = shp_path.with_suffix(".cpg")
    if not cpg_path.is_file():
        return None
    return normalize_encoding(cpg_path.read_text())


def text_field(records: tuple[dict[str, object], ...], field: str) -> list[str]:
    """A character field's values, with a ``.dbf`` null as the empty string.

    The empty string is what "missing" means in this table's text columns:
    ``ameriflux_site_id`` already uses it for every site with no Ameriflux
    counterpart, and it is the only missing marker that survives the round trip,
    since :func:`sipnet_calibration.sites.load_sites` reads with
    ``keep_default_na=False``. So a null site name reads back as absent rather
    than as the four-character name ``None``, which is what ``str(None)`` would
    have made of it.

    A ``bytes`` value raises instead. That means the attribute table did not
    decode, which is the failure this whole script is arranged to make loud, and
    ``str(b'...')`` would bury it in a name that looks almost plausible.
    """
    values = []
    for index, record in enumerate(records):
        value = record[field]
        if value is None:
            values.append("")
        elif isinstance(value, str):
            values.append(value)
        elif isinstance(value, bytes):
            raise IngestError(
                f"{field} at record index {index} is undecoded bytes "
                f"({value[:32]!r}); the attribute table was read with the wrong "
                "encoding"
            )
        else:
            raise IngestError(
                f"{field} at record index {index} is {type(value).__name__} "
                f"({value!r}), not text"
            )
    return values


def numeric_field(records: tuple[dict[str, object], ...], field: str) -> list[object]:
    """A numeric field's values, with a ``.dbf`` null reported rather than cast.

    Unlike the text columns, the integer columns have no way to say "missing":
    an integer dtype cannot hold ``NaN``, ``cluster`` and ``landcover`` have no
    spare value, and ``site_order``'s 0 already means "a sampled point". So a
    null here is an error, and the point of this function is that the message
    says so and names the record -- reaching :func:`check_values_are_integral`
    as a ``NaN`` would report "non-finite values" and leave the reader guessing
    whether the source held a null or a genuine ``NaN``.
    """
    null_at = [index for index, record in enumerate(records) if record[field] is None]
    if null_at:
        raise IngestError(
            f"{field} is null in {len(null_at)} record(s), first at record index "
            f"{null_at[0]}; the processed table has no way to record a missing "
            f"{field}, so this has to be resolved in the source"
        )
    return [record[field] for record in records]


def na_hazard_site_ids(names: list[str], *, site_ids: np.ndarray) -> list[int]:
    """Sites whose name a default ``read_csv`` would turn into a missing value.

    Some sites are named literally ``NA``, which pandas treats as null unless
    told otherwise. That is not an error in the data and this is not
    a check: :func:`sipnet_calibration.sites.load_sites` reads the name column as
    text with ``keep_default_na=False``, and
    :func:`check_csv_round_trip` compares ``site_name`` value by value, so a
    reader that lost them would fail there. The list is returned so the run
    reports which sites depend on that, rather than leaving it to be
    rediscovered.
    """
    hazards = {"", *pd._libs.parsers.STR_NA_VALUES}
    return [
        int(site_id)
        for site_id, name in zip(site_ids, names, strict=True)
        if name in hazards
    ]


# ── checks ────────────────────────────────────────────────────────────────────
#
# One invariant per function, each raising with the invariant named.

def check_encoding_is_utf8(contents: ShapefileContents, *, encoding_used: str) -> None:
    """Fail unless the ``.cpg`` declares UTF-8 and that is what was read."""
    if contents.declared_encoding is None:
        raise IngestError(
            "no .cpg beside the shapefile, so the attribute encoding is undeclared; "
            f"expected one declaring {EXPECTED_ENCODING}"
        )
    if contents.declared_encoding != EXPECTED_ENCODING:
        raise IngestError(
            f"the .cpg declares {contents.declared_encoding!r}, not "
            f"{EXPECTED_ENCODING!r}; two site names decode to plausible-looking "
            "garbage under a single-byte encoding rather than raising, so this "
            "is not safe to override"
        )
    if normalize_encoding(encoding_used) != EXPECTED_ENCODING:
        raise IngestError(
            f"the shapefile was read as {encoding_used!r} but its .cpg declares "
            f"{contents.declared_encoding!r}"
        )


def check_record_count(contents: ShapefileContents) -> None:
    """Fail unless the shapefile holds exactly :data:`N_SITES` records."""
    if len(contents.records) != N_SITES:
        raise IngestError(
            f"expected {N_SITES} records, found {len(contents.records)}"
        )
    if len(contents.points) != len(contents.records):
        raise IngestError(
            f"{len(contents.points)} shapes against {len(contents.records)} "
            "attribute records"
        )


def check_shapes_are_single_points(contents: ShapefileContents) -> None:
    """Fail unless every shape is a ``POINT`` carrying exactly one point."""
    point_type = int(shapefile.POINT)
    wrong_type = [
        index for index, kind in enumerate(contents.shape_types) if kind != point_type
    ]
    if wrong_type:
        raise IngestError(
            f"{len(wrong_type)} shape(s) are not POINT ({point_type}); "
            f"first at record index {wrong_type[0]}"
        )
    wrong_count = [
        index for index, points in enumerate(contents.points) if len(points) != 1
    ]
    if wrong_count:
        raise IngestError(
            f"{len(wrong_count)} POINT shape(s) do not carry exactly one point; "
            f"first at record index {wrong_count[0]}"
        )


def check_fields_are_present(contents: ShapefileContents, *, required: tuple[str, ...]) -> None:
    """Fail unless the ``.dbf`` carries every field this script reads."""
    missing = [name for name in required if name not in contents.field_names]
    if missing:
        raise IngestError(
            f"the .dbf is missing field(s) {missing}; it holds "
            f"{list(contents.field_names)}"
        )


def check_site_ids_are_the_full_range(site_ids: np.ndarray) -> None:
    """Fail unless ``site_id`` is exactly ``1..N_SITES`` in record order.

    Record *N* is site *N*: every other product joins on this, and the
    identifiers are not ours to renumber, so a permutation is as much a failure
    as a gap.
    """
    expected = np.arange(1, N_SITES + 1)
    if np.array_equal(site_ids, expected):
        return
    if site_ids.shape != expected.shape:
        raise IngestError(
            f"site_id is not 1..{N_SITES} in record order: {site_ids.size} "
            f"value(s), expected {expected.size}"
        )
    wrong = int(np.flatnonzero(site_ids != expected)[0])
    raise IngestError(
        f"site_id is not 1..{N_SITES} in record order: record index {wrong} "
        f"holds {site_ids[wrong]}, expected {expected[wrong]}"
    )


def check_values_are_integral(values: np.ndarray, *, name: str) -> None:
    """Fail unless every value of a float-typed ``.dbf`` field is a whole number.

    ``cluster``, ``landcover`` and ``site_order`` are declared in the ``.dbf`` as
    numerics with 15 decimals, so they arrive as floats. Casting them is only
    safe while this holds.
    """
    non_finite = np.flatnonzero(~np.isfinite(values))
    if non_finite.size:
        index = int(non_finite[0])
        raise IngestError(
            f"{name} holds {values[index]!r} at record index {index}; "
            f"{non_finite.size} value(s) are not finite"
        )
    fractional = np.flatnonzero(values != np.floor(values))
    if fractional.size:
        index = int(fractional[0])
        raise IngestError(
            f"{name} holds a non-integral value {values[index]!r} at record index "
            f"{index}, so it cannot be cast to an integer"
        )


def check_values_fit_dtype(values: np.ndarray, *, name: str, dtype) -> None:
    """Fail unless every value survives the cast to *dtype* unchanged.

    Without this, ``astype`` wraps silently: a ``cluster`` of 200 becomes -56 in
    ``int8``, and the round-trip check cannot see it because both the written
    and the read-back column are ``int8``. Integrality is not enough; range has
    to be checked too, and against the *target* type rather than ``int64``.
    """
    info = np.iinfo(dtype)
    outside = np.flatnonzero((values < info.min) | (values > info.max))
    if outside.size:
        index = int(outside[0])
        raise IngestError(
            f"{name} holds {values[index]!r} at record index {index}, outside the "
            f"range of {np.dtype(dtype).name} ({info.min}..{info.max}); casting "
            f"it would wrap silently"
        )


def _as_integer(values, *, name: str, dtype) -> np.ndarray:
    """Check integrality and range, then cast. The only integer cast used here.

    Everything the table stores as an integer arrives as a float (the ``.dbf``
    declares 15 decimals) or as ``int64`` (the grid indices), so every cast is a
    narrowing one and every one goes through both checks.
    """
    wide = np.asarray(values, dtype=np.float64)
    check_values_are_integral(wide, name=name)
    check_values_fit_dtype(wide, name=name, dtype=dtype)
    return np.asarray(values).astype(dtype)


def check_site_order_is_a_permutation(site_order: np.ndarray) -> None:
    """Fail unless the non-zero ``site_order`` values are ``1..NAMED_SITE_COUNT``.

    The zeros are the sampled points; the rest are the named sites, and the
    values are a ranking, so a repeat means two sites claim one rank.
    """
    named = np.sort(site_order[site_order != 0])
    expected = np.arange(1, NAMED_SITE_COUNT + 1)
    if named.shape != expected.shape or not np.array_equal(named, expected):
        raise IngestError(
            f"the non-zero site_order values are not a permutation of "
            f"1..{NAMED_SITE_COUNT}: {named.size} non-zero values, "
            f"{np.unique(named).size} distinct, max {named.max() if named.size else 0}"
        )


def check_coordinates_are_finite(lon: np.ndarray, lat: np.ndarray) -> None:
    """Fail on a NaN or infinite coordinate, naming the record it came from.

    ``SITE_GRID.lonlat_to_index`` rejects these too, but it works on the whole
    array and cannot say which site is at fault. It is also the wrong place to
    learn about it: a NaN here means the geometry is damaged, not that the grid
    or the CRS is wrong, and the two want different responses.
    """
    bad = np.flatnonzero(~(np.isfinite(lon) & np.isfinite(lat)))
    if bad.size:
        index = int(bad[0])
        raise IngestError(
            f"{bad.size} record(s) have a non-finite coordinate, first at record "
            f"index {index}: lon={lon[index]!r}, lat={lat[index]!r}"
        )


def check_index_pairs_are_distinct(
    lon_index: np.ndarray, lat_index: np.ndarray
) -> None:
    """Fail unless the grid index pairs identify the sites uniquely.

    Two sites on one cell would make the indices useless as a position and would
    mean the pool is not the subsample of the grid it is documented to be.
    """
    pairs = np.stack([lon_index, lat_index], axis=1)
    distinct = np.unique(pairs, axis=0)
    if distinct.shape[0] != pairs.shape[0]:
        raise IngestError(
            f"{pairs.shape[0] - distinct.shape[0]} site(s) share a grid cell with "
            "another; the index pair is meant to be a unique position"
        )


def check_ameriflux_rows_are_one_per_site(frame: pd.DataFrame, *, source: Path) -> None:
    """Fail unless each site id appears once, and no identifier is blank.

    This runs on the *frame*, before it becomes a dict: ``dict(zip(...))`` keeps
    the last row for a repeated site id and drops the rest without a word, so by
    the time :func:`check_ameriflux_map_is_usable` sees the mapping the evidence
    is gone. A repeated site id means two towers were matched to one model site,
    which is a decision to make rather than to discard -- see the co-located
    instruments section of ``data/README.md``.
    """
    if frame.empty:
        raise IngestError(
            f"{source} holds no rows; a header-only map is a truncated or wrongly "
            "pathed file, not a site pool with no Ameriflux counterparts"
        )
    repeated = frame["index"][frame["index"].duplicated()].unique().tolist()
    if repeated:
        raise IngestError(
            f"{source} maps {len(repeated)} site id(s) more than once: "
            f"{sorted(repeated)[:10]}"
        )
    blank = frame.index[frame["Site_ID"].str.strip() == ""].tolist()
    if blank:
        raise IngestError(
            f"{source} has a blank Site_ID on {len(blank)} row(s), first at row "
            f"{blank[0]}"
        )


def check_ameriflux_map_is_usable(mapping: dict[int, str], *, site_ids: np.ndarray) -> None:
    """Fail unless the Ameriflux identifiers are unique and map to real sites."""
    identifiers = list(mapping.values())
    if len(set(identifiers)) != len(identifiers):
        seen: set[str] = set()
        repeated = sorted({name for name in identifiers if name in seen or seen.add(name)})
        raise IngestError(f"duplicate Ameriflux identifier(s) {repeated}")
    known = set(site_ids.tolist())
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise IngestError(
            f"{len(unknown)} Ameriflux row(s) name a site id that is not in the "
            f"shapefile: {unknown[:10]}"
        )


def write_checked_site_table(table: pd.DataFrame, out_path: Path) -> None:
    """Write the table, verify the round trip, and only then publish it.

    The check has to run against a real file, so the file is written to a
    sibling temporary path and renamed over *out_path* once it passes. Writing
    to *out_path* directly would mean that a failed check leaves a corrupt table
    at the canonical path -- the one every other product joins against -- while
    the script exits non-zero. The rename is atomic on a POSIX filesystem, so
    *out_path* is either the previous table or a fully checked new one.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    try:
        write_site_table(table, partial)
        check_csv_round_trip(table, partial)
        partial.replace(out_path)
    finally:
        partial.unlink(missing_ok=True)


def check_csv_round_trip(written: pd.DataFrame, out_path: Path) -> None:
    """Fail unless reading the CSV back reproduces the table it was written from.

    ``lon`` and ``lat`` are compared **bitwise** rather than approximately: this
    is the assertion the script exists for, since CSV formatting is the one
    place this table can silently lose information.
    """
    read_back = load_sites(out_path)
    if list(read_back.columns) != list(written.columns):
        raise IngestError(
            f"the CSV round trip changed the columns: wrote "
            f"{list(written.columns)}, read {list(read_back.columns)}"
        )
    if len(read_back) != len(written):
        raise IngestError(
            f"the CSV round trip changed the row count: wrote {len(written)}, "
            f"read {len(read_back)}"
        )

    for column in ("lon", "lat"):
        original = written[column].to_numpy(dtype=np.float64)
        returned = read_back[column].to_numpy(dtype=np.float64)
        # Bitwise, not approximate: `==` on float64 is exact, and neither column
        # holds NaN, so this is the comparison it looks like.
        differing = np.flatnonzero(original != returned)
        if differing.size:
            index = int(differing[0])
            raise IngestError(
                f"{column} did not survive the CSV round trip: "
                f"{differing.size} of {original.size} values differ, first at row "
                f"{index}, {original[index]!r} written against {returned[index]!r} "
                f"read back (difference {returned[index] - original[index]:.3g})"
            )

    for column in written.columns:
        if column in ("lon", "lat"):
            continue
        original = written[column].to_numpy()
        returned = read_back[column].to_numpy()
        differing = np.flatnonzero(original != returned)
        if differing.size:
            index = int(differing[0])
            raise IngestError(
                f"{column} did not survive the CSV round trip: "
                f"{differing.size} value(s) differ, first at row {index}, "
                f"{original[index]!r} written against {returned[index]!r} read back"
            )


if __name__ == "__main__":
    raise SystemExit(main())

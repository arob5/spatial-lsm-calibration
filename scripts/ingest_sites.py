#!/usr/bin/env python
"""Build the site table.

Overview
--------
Read the point shapefile that defines the site pool and write it as the site
table, a CSV carrying every field of the shapefile plus the grid indices and
the Ameriflux identifier. Every other data source joins against it on
``site_id``.

Input data
----------
``--shapefile``, default the repository's ``data/raw/sites/pts.shp``
    One single-point ``POINT`` record per site, in descending latitude, with
    record *N* being site *N*, and :data:`sipnet_calibration.sites.N_SITES`
    records. Geometry is float64 lon/lat on WGS 84. The ``.dbf`` attribute
    table holds the fields of :data:`SHAPEFILE_FIELD_NAMES`; the three
    integer ones are declared with 15 decimals and so arrive as floats.
    ``pts.cpg`` declares the ``.dbf`` encoding, which is UTF-8.

``--site-id-map``, default the repository's ``data/site_id_map.csv``
    Rows of ``Site_ID, index``, mapping an Ameriflux identifier onto a site id
    by exact match. It covers a subset of the pool.

Output data
-----------
``--output``, default :func:`sipnet_calibration.sites.default_sites_path`
    One row per site, ascending by ``site_id``, with the columns
    ``SITE_COLUMNS`` and dtypes ``SITE_COLUMN_DTYPES`` of
    :mod:`sipnet_calibration.sites`, which documents them. ``site_name`` is the
    shapefile's ``site_names``, and ``ameriflux_site_id`` the map's
    ``Site_ID``, the empty string for a site with no counterpart.

Notes
-----
Two things about this table are easy to get wrong and quiet when they go wrong,
and most of the checking exists to make them loud.

**The encoding.** A few site names carry non-ASCII bytes. Read as latin-1 they
do not raise; they decode to ``'RayÃ³n (MX-Ray)'``, which still looks like a
plausible site label. So the declared encoding is read from the ``.cpg`` and
checked rather than left to a library default, and ``--encoding`` refuses
anything that disagrees rather than honoring it.

**The coordinates.** The geometry is float64 and CSV formatting is where that
precision goes, so the output is read back through
:func:`sipnet_calibration.sites.load_sites` and compared **bitwise** before the
file is moved into place. What makes the round trip exact is that reader's
``float_precision="round_trip"``, not the write format (:data:`FLOAT_FORMAT`).

A ``.dbf`` null is the empty string in a text column, which is what "missing"
already means there, and an error in an integer column, which has no value to
put in its place (``site_order``'s 0 already means "a sampled point").

The table is written through :func:`sipnet_calibration.io.write_checked`.

Usage
-----
::

    uv run python scripts/ingest_sites.py
    uv run python scripts/ingest_sites.py --output /tmp/sites.csv
    uv run python scripts/ingest_sites.py --help
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

from sipnet_calibration.conventions import LAT, LON, SITE_ID
from sipnet_calibration.io import write_checked
from sipnet_calibration.sites import (
    N_SITES,
    SITE_COLUMN_DTYPES,
    SITE_COLUMNS,
    SITE_GRID,
    default_sites_path,
    load_sites,
)
from sipnet_calibration.validation import truncated

#: The repository, where the tracked inputs are.
REPOSITORY = Path(__file__).resolve().parents[1]

#: The tracked shapefile, found from the checkout.
DEFAULT_SHAPEFILE = REPOSITORY / "data" / "raw" / "sites" / "pts.shp"

#: The tracked Ameriflux identifier map, found from the checkout.
DEFAULT_SITE_ID_MAP = REPOSITORY / "data" / "site_id_map.csv"

#: Where the table is written: where :func:`sipnet_calibration.sites.load_sites`
#: reads it, ``$SIPNET_CALIBRATION_DATA`` included.
DEFAULT_OUTPUT = default_sites_path()

#: The encoding ``pts.cpg`` is expected to declare, normalized by
#: :func:`normalize_encoding`.
EXPECTED_ENCODING = "utf-8"

#: The ``.dbf`` fields the table is built from.
SHAPEFILE_FIELD_NAMES = (SITE_ID, "site_names", "site_order", "cluster", "landcover")

#: The ``.dbf`` fields besides ``site_id`` that arrive as floats and are
#: stored as integers.
INTEGER_FIELD_NAMES = ("site_order", "cluster", "landcover")

#: Highest ``site_order`` value; the non-zero values are a permutation of
#: ``1..NAMED_SITE_COUNT`` and the remaining sites carry 0.
NAMED_SITE_COUNT = 1093

#: ``float_format`` for the coordinate columns: ``None``, pandas' default of
#: ``repr``, the shortest decimal string that reads back as the same float64.
#: The C parser ``pandas.read_csv`` uses by default reads some rows inexactly
#: whatever the write format (``test_the_reader_setting_is_what_makes_the_
#: round_trip_exact`` measures it), so the exactness comes from
#: :func:`sipnet_calibration.sites.load_sites` reading with
#: ``float_precision="round_trip"``; ``repr`` is kept as the shortest form and
#: exact for any reader that parses correctly.
FLOAT_FORMAT = None


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Read the shapefile, build the table, and write it once it reads back exactly."""
    args = parse_args(argv)
    print(f"Reading {args.shapefile} as {args.encoding} ...", file=sys.stderr)
    try:
        check_input_is_a_file(args.shapefile, message_name="--shapefile")
        check_input_is_a_file(args.site_id_map, message_name="--site-id-map")
        contents = read_shapefile(args.shapefile, encoding=args.encoding)
        check_encoding_is_utf8(contents, encoding_used=args.encoding)
        ameriflux = read_ameriflux_map(args.site_id_map)
        table = build_site_table(contents, ameriflux)
        write_checked_site_table(table, args.output)
    except (
        IngestError, OSError, ValueError, LookupError, TypeError, shapefile.ShapefileException
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"\nWrote {args.output}")
    print(describe_site_table(table))
    print("\nThe CSV round trip is bitwise exact for lon and lat.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV to write, creating its directory. Default {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--encoding",
        default=EXPECTED_ENCODING,
        help=(
            "Encoding of the .dbf attribute table. Default "
            f"{EXPECTED_ENCODING}, which is what the .cpg declares; a value "
            "disagreeing with the .cpg is refused rather than honored."
        ),
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────
#
# Filesystem access is confined to the two readers and the writer; everything
# between them takes plain Python data and returns plain Python data, so it is
# testable without fixtures on disk.


def read_shapefile(shp_path: Path, *, encoding: str) -> ShapefileContents:
    """Read a point shapefile's geometry and attribute table into memory.

    *encoding* is passed to the reader explicitly rather than inferred, so a
    mismatch with the ``.cpg`` is something :func:`check_encoding_is_utf8`
    reports rather than something the library resolves silently.
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

    The file's ``index`` column is the integer site id.
    """
    frame = pd.read_csv(
        path,
        dtype={"Site_ID": str, "index": np.int64},
        keep_default_na=False,
        na_values=[],
    )
    check_ameriflux_rows_are_valid(frame, source=path)
    return dict(zip(frame["index"].tolist(), frame["Site_ID"].tolist(), strict=True))


def build_site_table(contents: ShapefileContents, ameriflux: dict[int, str]) -> pd.DataFrame:
    """The site table: :data:`N_SITES` rows in ascending ``site_id`` order.

    Every invariant this script holds its input to is checked here, so a caller
    cannot get an unchecked table.
    """
    check_shapefile_is_the_site_pool(contents)
    site_ids = as_integer_array(
        numeric_field(contents.records, SITE_ID),
        dtype=SITE_COLUMN_DTYPES[SITE_ID],
        message_name=SITE_ID,
    )
    check_site_ids_are_the_full_range(site_ids)

    lon, lat = point_coordinates(contents)
    check_coordinates_are_finite(lon, lat)
    # Raises if a coordinate is further than the default tolerance from a cell
    # center, which would mean the wrong grid or the wrong CRS.
    lon_index, lat_index = SITE_GRID.lonlat_to_index(lon, lat)
    check_index_pairs_are_distinct(lon_index, lat_index)

    integers = {
        name: as_integer_array(
            numeric_field(contents.records, name),
            dtype=SITE_COLUMN_DTYPES[name],
            message_name=name,
        )
        for name in INTEGER_FIELD_NAMES
    }
    check_site_order_is_a_permutation(integers["site_order"])
    check_ameriflux_map_is_usable(ameriflux, site_ids=site_ids)

    table = pd.DataFrame(
        {
            SITE_ID: site_ids,
            LON: lon,
            LAT: lat,
            "lon_index": as_integer_array(lon_index, dtype=np.int32, message_name="lon_index"),
            "lat_index": as_integer_array(lat_index, dtype=np.int32, message_name="lat_index"),
            "site_name": text_field(contents.records, "site_names"),
            **integers,
            "ameriflux_site_id": [ameriflux.get(int(site_id), "") for site_id in site_ids],
        }
    )
    return table[list(SITE_COLUMNS)]


def write_checked_site_table(table: pd.DataFrame, out_path: Path) -> None:
    """Write the table through a ``.partial`` file, moved in once it reads back exactly."""
    write_checked(
        out_path,
        write=lambda partial: write_site_table(table, partial),
        check=lambda partial: check_csv_round_trip(table, partial),
    )


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
        table["site_name"].tolist(), site_ids=table[SITE_ID].to_numpy()
    )
    lines = [
        f"  rows                  : {len(table)}",
        f"  site_id               : {table[SITE_ID].min()}..{table[SITE_ID].max()}",
        f"  longitude             : {table[LON].min():.4f} to {table[LON].max():.4f}",
        f"  latitude              : {table[LAT].min():.4f} to {table[LAT].max():.4f}",
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


class IngestError(RuntimeError):
    """An input or the written table breaks an invariant the site table depends on."""


@dataclass(frozen=True)
class ShapefileContents:
    """The parts of a point shapefile this script uses.

    Attributes
    ----------
    declared_encoding:
        What the ``.cpg`` file says, normalized, or ``None`` if there is no
        ``.cpg``.
    field_names:
        Attribute field names in ``.dbf`` order, without the deletion flag.
    records:
        One dict per record, keyed by field name, in record order.
    shape_types:
        The ``shapeType`` of each shape, in record order.
    points:
        Every point of each shape, in record order; a well-formed point
        shapefile has exactly one per shape.
    """

    declared_encoding: str | None
    field_names: tuple[str, ...]
    records: tuple[dict[str, object], ...]
    shape_types: tuple[int, ...]
    points: tuple[tuple[tuple[float, float], ...], ...]


def write_site_table(table: pd.DataFrame, out_path: Path) -> None:
    """Write the table as CSV, creating its directory if it is absent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format=FLOAT_FORMAT)


def normalize_encoding(name: str) -> str:
    """A codec name in a comparable form: ``'UTF-8 '`` -> ``'utf-8'``."""
    return name.strip().lower().replace("_", "-")


def read_declared_encoding(shp_path: Path) -> str | None:
    """The encoding the ``.cpg`` beside *shp_path* declares, or ``None`` without one."""
    cpg_path = shp_path.with_suffix(".cpg")
    if not cpg_path.is_file():
        return None
    return normalize_encoding(cpg_path.read_text())


def point_coordinates(contents: ShapefileContents) -> tuple[np.ndarray, np.ndarray]:
    """The float64 ``(lon, lat)`` of each shape's first point, in record order."""
    lon = np.array([points[0][0] for points in contents.points], dtype=np.float64)
    lat = np.array([points[0][1] for points in contents.points], dtype=np.float64)
    return lon, lat


def text_field(records: tuple[dict[str, object], ...], field: str) -> list[str]:
    """A character field's values, with a ``.dbf`` null as the empty string.

    The empty string is what "missing" means in the table's text columns, and
    the only missing marker that survives the round trip, since
    :func:`sipnet_calibration.sites.load_sites` reads with
    ``keep_default_na=False``; ``str(None)`` would make the name ``None``.

    Raises
    ------
    IngestError
        If a value is not text, undecoded bytes included.
    """
    check_text_field_holds_text(records, field)
    return ["" if record[field] is None else record[field] for record in records]


def numeric_field(records: tuple[dict[str, object], ...], field: str) -> list[object]:
    """A numeric field's values.

    Raises
    ------
    IngestError
        If a value is a ``.dbf`` null, which an integer column cannot hold.
    """
    check_numeric_field_has_no_nulls(records, field)
    return [record[field] for record in records]


def as_integer_array(values, *, dtype, message_name: str) -> np.ndarray:
    """*values* cast to the integer *dtype*, once they are whole and in its range.

    Raises
    ------
    IngestError
        If a value is not finite, not whole, or outside the range of *dtype*.
    """
    wide = np.asarray(values, dtype=np.float64)
    check_values_are_integral(wide, message_name=message_name)
    check_values_fit_dtype(wide, dtype=dtype, message_name=message_name)
    return np.asarray(values).astype(dtype)


def na_hazard_site_ids(names: list[str], *, site_ids: np.ndarray) -> list[int]:
    """Sites whose name a default ``read_csv`` would turn into a missing value.

    Some sites are named literally ``NA``. That is not an error, and
    :func:`check_csv_round_trip` would catch a reader that lost them; the list
    is for the run report.
    """
    hazards = {"", *pd._libs.parsers.STR_NA_VALUES}
    return [
        int(site_id)
        for site_id, name in zip(site_ids, names, strict=True)
        if name in hazards
    ]


# ── checks ────────────────────────────────────────────────────────────────────


def check_input_is_a_file(path: Path, *, message_name: str) -> None:
    """An input the command line names is a file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{message_name} {path} is not a file; pass the path of an existing file."
        )


def check_encoding_is_utf8(contents: ShapefileContents, *, encoding_used: str) -> None:
    """The ``.cpg`` declares UTF-8, and the shapefile was read as UTF-8."""
    if contents.declared_encoding is None:
        raise IngestError(
            "no .cpg beside the shapefile, so the attribute encoding is undeclared; "
            f"restore the .cpg, which declares {EXPECTED_ENCODING}."
        )
    if contents.declared_encoding != EXPECTED_ENCODING:
        raise IngestError(
            f"the .cpg declares {contents.declared_encoding!r}, not "
            f"{EXPECTED_ENCODING!r}; two site names decode to plausible-looking "
            "garbage under a single-byte encoding rather than raising, so restore "
            "the shapefile rather than override its encoding."
        )
    if normalize_encoding(encoding_used) != EXPECTED_ENCODING:
        raise IngestError(
            f"the shapefile was read as {encoding_used!r} but its .cpg declares "
            f"{contents.declared_encoding!r}; drop --encoding."
        )


def check_ameriflux_rows_are_valid(frame: pd.DataFrame, *, source: Path) -> None:
    """The Ameriflux map's rows are usable, before they become a mapping."""
    check_ameriflux_map_has_its_columns(frame, source=source)
    check_ameriflux_map_has_rows(frame, source=source)
    check_ameriflux_rows_are_one_per_site(frame, source=source)
    check_ameriflux_identifiers_are_not_blank(frame, source=source)


def check_ameriflux_map_has_its_columns(frame: pd.DataFrame, *, source: Path) -> None:
    """The Ameriflux map has its ``Site_ID`` and ``index`` columns."""
    missing = sorted({"Site_ID", "index"} - set(frame.columns))
    if missing:
        raise IngestError(
            f"{source} is missing column(s) {missing}; the map's header is Site_ID,index."
        )


def check_ameriflux_map_has_rows(frame: pd.DataFrame, *, source: Path) -> None:
    """The Ameriflux map holds at least one row."""
    if frame.empty:
        raise IngestError(
            f"{source} holds no rows; a header-only map is a truncated or wrongly "
            "pathed file, not a site pool with no Ameriflux counterparts, so check "
            "the path."
        )


def check_ameriflux_rows_are_one_per_site(frame: pd.DataFrame, *, source: Path) -> None:
    """Each site id appears in the Ameriflux map at most once."""
    # dict(zip(...)) would keep the last row for a repeated site id and drop the
    # rest without a word. A repeated site id means two towers were matched to
    # one model site, which is a decision to make rather than to discard -- see
    # the co-located instruments section of data/README.md.
    site_ids = frame["index"]
    repeated = sorted(site_ids[site_ids.duplicated()].unique().tolist())
    if repeated:
        raise IngestError(
            f"{source} maps {len(repeated)} site id(s) more than once, "
            f"{truncated(repeated)}; decide which tower each site keeps."
        )


def check_ameriflux_identifiers_are_not_blank(frame: pd.DataFrame, *, source: Path) -> None:
    """No row of the Ameriflux map has a blank ``Site_ID``."""
    blank = frame.index[frame["Site_ID"].str.strip() == ""].tolist()
    if blank:
        raise IngestError(
            f"{source} has a blank Site_ID on {len(blank)} row(s), first at row "
            f"{blank[0]}; fill or drop those rows."
        )


def check_shapefile_is_the_site_pool(contents: ShapefileContents) -> None:
    """The shapefile holds one point per site of the pool, with the fields read."""
    check_record_count_is_the_pool_size(contents)
    check_every_record_has_one_shape(contents)
    check_shapes_are_single_points(contents)
    check_fields_are_present(contents, required=SHAPEFILE_FIELD_NAMES)


def check_record_count_is_the_pool_size(contents: ShapefileContents) -> None:
    """The shapefile holds exactly :data:`N_SITES` records."""
    if len(contents.records) != N_SITES:
        raise IngestError(
            f"expected {N_SITES} records, found {len(contents.records)}; the shapefile "
            "is not the site pool's, so check which file was copied."
        )


def check_every_record_has_one_shape(contents: ShapefileContents) -> None:
    """The shapefile holds as many shapes as attribute records."""
    if len(contents.points) != len(contents.records):
        raise IngestError(
            f"{len(contents.points)} shapes against {len(contents.records)} attribute "
            "records; the .shp and .dbf are from different files, so copy them together."
        )


def check_shapes_are_single_points(contents: ShapefileContents) -> None:
    """Every shape is a ``POINT`` carrying exactly one point."""
    point_type = int(shapefile.POINT)
    wrong_type = [
        index for index, kind in enumerate(contents.shape_types) if kind != point_type
    ]
    if wrong_type:
        raise IngestError(
            f"{len(wrong_type)} shape(s) are not POINT ({point_type}), first at record "
            f"index {wrong_type[0]}; the site pool is a point shapefile."
        )
    wrong_count = [
        index for index, points in enumerate(contents.points) if len(points) != 1
    ]
    if wrong_count:
        raise IngestError(
            f"{len(wrong_count)} POINT shape(s) do not carry exactly one point, first at "
            f"record index {wrong_count[0]}; the site pool is one point per site."
        )


def check_fields_are_present(contents: ShapefileContents, *, required: tuple[str, ...]) -> None:
    """The ``.dbf`` carries every field this script reads."""
    missing = [name for name in required if name not in contents.field_names]
    if missing:
        raise IngestError(
            f"the .dbf is missing field(s) {missing}, holding "
            f"{list(contents.field_names)}; check which shapefile was copied."
        )


def check_numeric_field_has_no_nulls(records: tuple[dict[str, object], ...], field: str) -> None:
    """No record of a numeric ``.dbf`` field is null."""
    # A null reaching check_values_are_integral as NaN would be reported as a
    # non-finite value, leaving the reader to guess whether the source held a
    # null or a genuine NaN.
    null_at = [index for index, record in enumerate(records) if record[field] is None]
    if null_at:
        raise IngestError(
            f"{field} is null in {len(null_at)} record(s), first at record index "
            f"{null_at[0]}; the site table has no way to record a missing {field}, so "
            "this has to be resolved in the source."
        )


def check_text_field_holds_text(records: tuple[dict[str, object], ...], field: str) -> None:
    """Every record of a character ``.dbf`` field is text or null."""
    for index, record in enumerate(records):
        value = record[field]
        if isinstance(value, bytes):
            raise IngestError(
                f"{field} at record index {index} is undecoded bytes ({value[:32]!r}); "
                "the attribute table was read with the wrong encoding, so drop --encoding."
            )
        if value is not None and not isinstance(value, str):
            raise IngestError(
                f"{field} at record index {index} is {type(value).__name__} ({value!r}), "
                "not text; check which shapefile was copied."
            )


def check_site_ids_are_the_full_range(site_ids: np.ndarray) -> None:
    """``site_id`` is exactly ``1..N_SITES`` in record order."""
    # Record N is site N: every other data source joins on this, and the ids are
    # not ours to renumber, so a permutation is as much a failure as a gap.
    expected = np.arange(1, N_SITES + 1)
    if np.array_equal(site_ids, expected):
        return
    if site_ids.shape != expected.shape:
        raise IngestError(
            f"site_id is not 1..{N_SITES} in record order: {site_ids.size} value(s), "
            f"expected {expected.size}; the shapefile is not the site pool's."
        )
    wrong = int(np.flatnonzero(site_ids != expected)[0])
    raise IngestError(
        f"site_id is not 1..{N_SITES} in record order: record index {wrong} holds "
        f"{site_ids[wrong]}, expected {expected[wrong]}; site ids are never renumbered, "
        "so restore the shapefile's record order."
    )


def check_values_are_integral(values: np.ndarray, *, message_name: str) -> None:
    """Every value of a float-typed ``.dbf`` field is a finite whole number."""
    non_finite = np.flatnonzero(~np.isfinite(values))
    if non_finite.size:
        index = int(non_finite[0])
        raise IngestError(
            f"{message_name} holds {values[index]!r} at record index {index}, and "
            f"{non_finite.size} value(s) are not finite; resolve them in the source."
        )
    fractional = np.flatnonzero(values != np.floor(values))
    if fractional.size:
        index = int(fractional[0])
        raise IngestError(
            f"{message_name} holds a non-integral value {values[index]!r} at record "
            f"index {index}, so it cannot be cast to an integer; resolve it in the source."
        )


def check_values_fit_dtype(values: np.ndarray, *, dtype, message_name: str) -> None:
    """Every value survives the cast to the integer *dtype* unchanged."""
    # astype wraps silently: a cluster of 200 becomes -56 in int8, and the round
    # trip cannot see it, since the written and the read-back column are both
    # int8. So range is checked against the target type, not int64.
    info = np.iinfo(dtype)
    outside = np.flatnonzero((values < info.min) | (values > info.max))
    if outside.size:
        index = int(outside[0])
        raise IngestError(
            f"{message_name} holds {values[index]!r} at record index {index}, outside the "
            f"range of {np.dtype(dtype).name} ({info.min}..{info.max}), so casting it "
            "would wrap silently; widen the column's dtype in SITE_COLUMN_DTYPES."
        )


def check_site_order_is_a_permutation(site_order: np.ndarray) -> None:
    """The non-zero ``site_order`` values are a permutation of ``1..NAMED_SITE_COUNT``."""
    named = np.sort(site_order[site_order != 0])
    expected = np.arange(1, NAMED_SITE_COUNT + 1)
    if named.shape != expected.shape or not np.array_equal(named, expected):
        raise IngestError(
            f"the non-zero site_order values are not a permutation of "
            f"1..{NAMED_SITE_COUNT}: {named.size} non-zero values, "
            f"{np.unique(named).size} distinct, max {named.max() if named.size else 0}; "
            "two sites claiming one rank means the shapefile changed."
        )


def check_coordinates_are_finite(lon: np.ndarray, lat: np.ndarray) -> None:
    """Every coordinate is finite."""
    # SITE_GRID.lonlat_to_index refuses these too, but it cannot say which site
    # is at fault, and a NaN here means damaged geometry rather than the wrong
    # grid or CRS.
    bad = np.flatnonzero(~(np.isfinite(lon) & np.isfinite(lat)))
    if bad.size:
        index = int(bad[0])
        raise IngestError(
            f"{bad.size} record(s) have a non-finite coordinate, first at record index "
            f"{index}: lon={lon[index]!r}, lat={lat[index]!r}; the geometry is damaged, "
            "so restore the shapefile."
        )


def check_index_pairs_are_distinct(lon_index: np.ndarray, lat_index: np.ndarray) -> None:
    """No two sites share a grid cell."""
    pairs = np.stack([lon_index, lat_index], axis=1)
    distinct = np.unique(pairs, axis=0)
    if distinct.shape[0] != pairs.shape[0]:
        raise IngestError(
            f"{pairs.shape[0] - distinct.shape[0]} site(s) share a grid cell with "
            "another; the index pair is a site's unique position on SITE_GRID, so "
            "check the grid and the CRS."
        )


def check_ameriflux_map_is_usable(mapping: dict[int, str], *, site_ids: np.ndarray) -> None:
    """The Ameriflux identifiers are unique and name sites of the shapefile."""
    check_ameriflux_identifiers_are_unique(mapping)
    check_ameriflux_sites_are_in_the_shapefile(mapping, site_ids=site_ids)


def check_ameriflux_identifiers_are_unique(mapping: dict[int, str]) -> None:
    """No Ameriflux identifier is mapped onto two sites."""
    seen: set[str] = set()
    repeated = sorted({name for name in mapping.values() if name in seen or seen.add(name)})
    if repeated:
        raise IngestError(
            f"duplicate Ameriflux identifier(s) {truncated(repeated)}; map each tower "
            "onto one site."
        )


def check_ameriflux_sites_are_in_the_shapefile(
    mapping: dict[int, str], *, site_ids: np.ndarray
) -> None:
    """Every site id the Ameriflux map names is a site of the shapefile."""
    unknown = sorted(set(mapping) - set(site_ids.tolist()))
    if unknown:
        raise IngestError(
            f"{len(unknown)} Ameriflux row(s) name a site id that is not in the "
            f"shapefile, {truncated(unknown)}; correct or drop those rows."
        )


def check_csv_round_trip(written: pd.DataFrame, out_path: Path) -> None:
    """The CSV reads back through the library loader as the table it was written from."""
    read_back = load_sites(out_path)
    check_round_trip_keeps_the_shape(written, read_back)
    check_round_trip_keeps_the_coordinates_bitwise(written, read_back)
    check_round_trip_keeps_the_other_columns(written, read_back)


def check_round_trip_keeps_the_shape(written: pd.DataFrame, read_back: pd.DataFrame) -> None:
    """The table read back has the columns and the rows it was written with."""
    if list(read_back.columns) != list(written.columns):
        raise IngestError(
            f"the CSV round trip changed the columns: wrote {list(written.columns)}, "
            f"read {list(read_back.columns)}; the writer and SITE_COLUMNS disagree."
        )
    if len(read_back) != len(written):
        raise IngestError(
            f"the CSV round trip changed the row count: wrote {len(written)}, read "
            f"{len(read_back)}; inspect the kept partial file."
        )


def check_round_trip_keeps_the_coordinates_bitwise(
    written: pd.DataFrame, read_back: pd.DataFrame
) -> None:
    """``lon`` and ``lat`` read back bit for bit."""
    for column in (LON, LAT):
        original = written[column].to_numpy(dtype=np.float64)
        returned = read_back[column].to_numpy(dtype=np.float64)
        # Bitwise, not approximate: == on float64 is exact, and neither column
        # holds NaN, so this is the comparison it looks like.
        differing = np.flatnonzero(original != returned)
        if differing.size:
            index = int(differing[0])
            raise IngestError(
                f"{column} did not survive the CSV round trip: {differing.size} of "
                f"{original.size} values differ, first at row {index}, "
                f"{original[index]!r} written against {returned[index]!r} read back "
                f"(difference {returned[index] - original[index]:.3g}); read with "
                "float_precision='round_trip'."
            )


def check_round_trip_keeps_the_other_columns(
    written: pd.DataFrame, read_back: pd.DataFrame
) -> None:
    """Every column besides ``lon`` and ``lat`` reads back value for value."""
    for column in written.columns:
        if column in (LON, LAT):
            continue
        original = written[column].to_numpy()
        returned = read_back[column].to_numpy()
        differing = np.flatnonzero(original != returned)
        if differing.size:
            index = int(differing[0])
            raise IngestError(
                f"{column} did not survive the CSV round trip: {differing.size} value(s) "
                f"differ, first at row {index}, {original[index]!r} written against "
                f"{returned[index]!r} read back; inspect the kept partial file."
            )


if __name__ == "__main__":
    raise SystemExit(main())

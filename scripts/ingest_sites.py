#!/usr/bin/env python3
"""Build the site table: ``raw/sites/pts.*`` -> ``processed/sites/sites.csv``.

The point shapefile in ``data/raw/sites/`` defines the 8000-site pool. It is the
only input under ``data/raw/`` that is tracked in version control, so this is the
one ingest script that runs end to end on a laptop. Every other product joins
against its output on ``site_id``.

The output carries every field of the shapefile, so nothing is lost in
translation, plus the grid indices from
``sipnet_calibration.sites.SITE_GRID`` and the Ameriflux identifier from
``data/site_id_map.csv``:

    site_id, lon, lat, lon_idx, lat_idx, site_name, site_order, cluster,
    landcover, ameriflux_site_id

Two things about this table are easy to get wrong and quiet when they go wrong,
and most of what follows exists to make them loud.

**The encoding.** ``pts.cpg`` declares UTF-8, and exactly two of the 8000 site
names carry non-ASCII bytes. Read as latin-1 they do not raise; they decode to
``'RayÃ³n (MX-Ray)'``, which still looks like a plausible site label. So the
declared encoding is read from the ``.cpg`` and asserted rather than left to a
library default.

**The coordinates.** The geometry is float64 and CSV formatting is where that
precision is lost: the retired ``data/site_ids.csv`` differed from the shapefile
by up to 4.1e-13 degrees for exactly this reason. So the output is read back and
compared **bitwise** against the values read from the shapefile before the
script reports success. That check earns its place: it caught the natural
choice of ``float_format="%.17g"`` moving 1632 of the 8000 longitudes, because
pandas' default CSV parser does not read 17-digit strings back exactly. See
:data:`FLOAT_FORMAT`.

Asserts before writing, each in a named check with its own message:

* ``pts.cpg`` declares UTF-8 and that is the encoding the reader was given;
* 8000 records, 8000 single-point ``POINT`` shapes;
* ``site_id`` is exactly 1..8000 in record order;
* ``cluster``, ``landcover`` and ``site_order`` hold integral values (the
  ``.dbf`` declares them as numerics with 15 decimals, so they arrive as floats
  and must be cast rather than written through);
* ``site_order``'s non-zero values are a permutation of 1..1093;
* every coordinate resolves on ``SITE_GRID``, to 8000 distinct index pairs;
* no duplicate Ameriflux identifier, and every mapped site id exists.

Asserts after writing:

* the CSV round trip is bitwise exact for ``lon`` and ``lat``, and equal for
  every other column. That last part is doing real work: eight sites are named
  literally ``NA``, so a reader that took pandas' default missing values would
  fail here rather than silently nulling them.

Examples
--------
The whole job, with the repository's own paths::

    python3 scripts/ingest_sites.py

Write somewhere else, leaving ``data/processed/`` alone::

    python3 scripts/ingest_sites.py --out /tmp/sites.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

from sipnet_calibration.sites import SITE_COLUMNS, SITE_GRID

#: Repository root, as seen from ``scripts/``.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SHAPEFILE = REPO_ROOT / "data" / "raw" / "sites" / "pts.shp"
DEFAULT_SITE_ID_MAP = REPO_ROOT / "data" / "site_id_map.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "sites" / "sites.csv"

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
#: of ``repr``, which is the shortest decimal string that reads back as the same
#: float64.
#:
#: ``"%.17g"`` is the obvious alternative and it is **wrong here**, which is
#: worth recording because it looks safer. 17 significant digits do round-trip
#: through Python's ``float()``, but not through the C parser
#: ``pandas.read_csv`` uses by default: writing -93.2875010172526 as
#: ``-93.287501017252595`` and reading it back with default settings yields
#: -93.28750101725261, off by 1.4e-14. 1632 of the 8000 longitudes moved that
#: way. So the coordinates are written short and exact, and
#: :func:`sipnet_calibration.sites.load_sites` additionally reads with
#: ``float_precision="round_trip"`` so that neither side relies on the other's
#: formatting.
FLOAT_FORMAT = None


class IngestError(Exception):
    """Raised when an invariant of the input or the output does not hold."""


# ── reading ──────────────────────────────────────────────────────────────────
#
# All filesystem access lives here. Everything below takes plain Python data and
# returns plain Python data, so it is testable without fixtures on disk.


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
    frame = pd.read_csv(path, dtype={"Site_ID": str, "index": np.int64})
    missing = {"Site_ID", "index"} - set(frame.columns)
    if missing:
        raise IngestError(f"{path} is missing column(s) {sorted(missing)}")
    return dict(zip(frame["index"].tolist(), frame["Site_ID"].tolist(), strict=True))


# ── checks ───────────────────────────────────────────────────────────────────
#
# One invariant per function, each raising with the invariant named, so a failure
# says which expectation broke rather than printing a traceback.


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
    if not np.all(np.isfinite(values)):
        raise IngestError(f"{name} holds non-finite values")
    fractional = np.flatnonzero(values != np.floor(values))
    if fractional.size:
        index = int(fractional[0])
        raise IngestError(
            f"{name} holds a non-integral value {values[index]!r} at record index "
            f"{index}, so it cannot be cast to an integer"
        )


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
    if np.any(site_order < 0):
        raise IngestError("site_order holds negative values")


def check_index_pairs_are_distinct(lon_idx: np.ndarray, lat_idx: np.ndarray) -> None:
    """Fail unless the grid index pairs identify the sites uniquely.

    Two sites on one cell would make the indices useless as a position and would
    mean the pool is not the subsample of the grid it is documented to be.
    """
    pairs = np.stack([lon_idx, lat_idx], axis=1)
    distinct = np.unique(pairs, axis=0)
    if distinct.shape[0] != pairs.shape[0]:
        raise IngestError(
            f"{pairs.shape[0] - distinct.shape[0]} site(s) share a grid cell with "
            "another; the index pair is meant to be a unique position"
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


def na_hazard_site_ids(names: list[str], *, site_ids: np.ndarray) -> list[int]:
    """Sites whose name a default ``read_csv`` would turn into a missing value.

    Eight of the 8000 sites are named literally ``NA``, which pandas treats as
    null unless told otherwise. That is not an error in the data and this is not
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


# ── building the table ───────────────────────────────────────────────────────


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

    site_ids = np.array([record["site_id"] for record in contents.records], dtype=np.int64)
    check_site_ids_are_the_full_range(site_ids)

    lon = np.array([points[0][0] for points in contents.points], dtype=np.float64)
    lat = np.array([points[0][1] for points in contents.points], dtype=np.float64)
    # Raises if any coordinate is further than the default tolerance from a cell
    # center, which would mean the wrong grid or the wrong CRS.
    lon_idx, lat_idx = SITE_GRID.lonlat_to_index(lon, lat)
    check_index_pairs_are_distinct(lon_idx, lat_idx)

    integral = {}
    for field in ("site_order", "cluster", "landcover"):
        values = np.array(
            [record[field] for record in contents.records], dtype=np.float64
        )
        check_values_are_integral(values, name=field)
        integral[field] = values.astype(np.int64)
    check_site_order_is_a_permutation(integral["site_order"])

    names = [str(record["site_names"]) for record in contents.records]

    check_ameriflux_map_is_usable(ameriflux, site_ids=site_ids)
    ameriflux_column = [ameriflux.get(int(site_id), "") for site_id in site_ids]

    table = pd.DataFrame(
        {
            "site_id": site_ids.astype(np.int32),
            "lon": lon,
            "lat": lat,
            "lon_idx": lon_idx.astype(np.int32),
            "lat_idx": lat_idx.astype(np.int32),
            "site_name": names,
            "site_order": integral["site_order"].astype(np.int32),
            "cluster": integral["cluster"].astype(np.int8),
            "landcover": integral["landcover"].astype(np.int8),
            "ameriflux_site_id": ameriflux_column,
        }
    )
    return table[list(SITE_COLUMNS)]


# ── writing, and proving the write lost nothing ──────────────────────────────


def write_site_table(table: pd.DataFrame, out_path: Path) -> None:
    """Write the table as CSV, creating its directory if it is absent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format=FLOAT_FORMAT)


def check_csv_round_trip(written: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Fail unless reading the CSV back reproduces the table it was written from.

    ``lon`` and ``lat`` are compared **bitwise** rather than approximately: this
    is the assertion the script exists for, since CSV formatting is the one
    place this table can silently lose information. Returns the frame that was
    read back, so a caller can report on the file rather than on memory.
    """
    from sipnet_calibration.sites import load_sites

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
    return read_back


# ── reporting ────────────────────────────────────────────────────────────────


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


# ── entry point ──────────────────────────────────────────────────────────────


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
    contents = read_shapefile(args.shapefile, encoding=args.encoding)
    try:
        check_encoding_is_utf8(contents, encoding_used=args.encoding)
        ameriflux = read_ameriflux_map(args.site_id_map)
        table = build_site_table(contents, ameriflux)
        write_site_table(table, args.out)
        check_csv_round_trip(table, args.out)
    except IngestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"\nWrote {args.out}")
    print(describe_site_table(table))
    print("\nThe CSV round trip is bitwise exact for lon and lat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

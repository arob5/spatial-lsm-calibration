#!/usr/bin/env python
"""Build the map basemap from the tracked Natural Earth archives.

Overview
--------
Read each Natural Earth layer named in
:data:`sipnet_calibration.plotting.basemap.BASEMAP_LAYERS` from
``data/raw/natural_earth/``, keep the parts near enough the projection center
to be drawn, and write them into the package as the file
:func:`~sipnet_calibration.plotting.basemap.load_basemap` reads.

Input data
----------
``--raw-directory``, default the repository's ``data/raw/natural_earth/``
    The zipped Natural Earth 1:50m shapefiles that
    ``scripts/raw_sources/download_natural_earth.py`` downloads, read in place
    with ``pyshp``. Polyline layers are read as lines and the lakes' polygons as
    their rings.

Output data
-----------
``--output``, default :func:`~sipnet_calibration.plotting.basemap.basemap_path`
    The basemap, in the layout of :mod:`sipnet_calibration.plotting.basemap`'s
    data model, tracked in git.

Notes
-----
The file is written through :func:`sipnet_calibration.io.write_checked`, and
its check reads it back with
:func:`~sipnet_calibration.plotting.basemap.load_basemap`.

**The output is not byte-reproducible**, because ``numpy.savez`` stamps each
member with the time it was written. Its *arrays* are: ``tests/test_basemap.py``
rebuilds from the tracked archives and compares them with the tracked file, so
a hand-edit or a stale build is a test failure.

Usage
-----
::

    uv run python scripts/build_basemap.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import shapefile

from sipnet_calibration.io import file_md5, write_checked
from sipnet_calibration.plotting.basemap import (
    BASEMAP_LAYERS,
    basemap_path,
    clip_to_drawable,
    load_basemap,
    write_basemap,
)

#: Where the tracked Natural Earth archives are: in this repository, whatever
#: ``$SIPNET_CALIBRATION_DATA`` says, since a tracked input is found from the
#: checkout rather than from the storage-backed data root.
DEFAULT_RAW_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "raw" / "natural_earth"


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Build the basemap from the archives and write it, or report why not."""
    args = parse_args(argv)
    try:
        parts, source_md5 = read_layers(args.raw_directory)
        write_basemap_file(parts, source_md5, args.output)
        print(describe_basemap(parts, args.output))
    except (
        IngestError, OSError, ValueError, LookupError, TypeError, shapefile.ShapefileException
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=DEFAULT_RAW_DIRECTORY,
        help=f"Where the Natural Earth archives are. Default: {DEFAULT_RAW_DIRECTORY}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=basemap_path(),
        help="Where to write the basemap. Default: inside the package.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def read_layers(raw_directory: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
    """Every layer's drawable parts, and the md5 of the archive each came from."""
    parts: dict[str, list[np.ndarray]] = {}
    source_md5: dict[str, str] = {}
    for name, layer in BASEMAP_LAYERS.items():
        archive = raw_directory / layer.source_file
        check_archive_exists(archive)
        parts[name] = read_layer(archive)
        source_md5[name] = file_md5(archive)
    return parts, source_md5


def read_layer(archive: Path) -> list[np.ndarray]:
    """Each line (or polygon ring) of *archive*, clipped to the drawable region."""
    kept: list[np.ndarray] = []
    with shapefile.Reader(str(archive)) as reader:
        for shape in reader.iterShapes():
            if not shape.points:
                continue
            points = np.asarray(shape.points, dtype=float)
            bounds = list(shape.parts) + [len(points)]
            for start, stop in zip(bounds[:-1], bounds[1:]):
                kept.extend(clip_to_drawable(points[start:stop, 0], points[start:stop, 1]))
    return kept


def write_basemap_file(
    parts: dict[str, list[np.ndarray]], source_md5: dict[str, str], path: Path
) -> None:
    """Write through a ``.partial`` file, moved in once it reads back as the parts."""
    write_checked(
        path,
        write=lambda partial: write_basemap(parts, source_md5, partial),
        check=lambda partial: check_round_trip(parts, partial),
    )


def describe_basemap(parts: dict[str, list[np.ndarray]], path: Path) -> str:
    """What was written, per layer, for the terminal."""
    lines = [f"wrote {path} ({path.stat().st_size:,} bytes)"]
    for name, layer_parts in parts.items():
        vertices = sum(len(part) for part in layer_parts)
        lines.append(f"  {name:10s} {len(layer_parts):6,d} parts {vertices:9,d} vertices")
    return "\n".join(lines)


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """The written basemap is not what was built."""


# ── checks ────────────────────────────────────────────────────────────────────


def check_archive_exists(archive: Path) -> None:
    """A layer's Natural Earth archive exists."""
    if not archive.is_file():
        raise FileNotFoundError(
            f"{archive} does not exist; the archives are tracked, so if one is missing, "
            "fetch it with scripts/raw_sources/download_natural_earth.py."
        )


def check_round_trip(parts: dict[str, list[np.ndarray]], path: Path) -> None:
    """The file reads back through the library loader as the parts written."""
    # Vertices are stored as float32, so the comparison allows that rounding.
    loaded = load_basemap(path)
    for name, layer_parts in parts.items():
        if len(loaded[name]) != len(layer_parts):
            raise IngestError(
                f"layer {name!r} read back with a different number of parts; inspect the "
                "kept partial file."
            )
        for written, read in zip(layer_parts, loaded[name]):
            if written.shape != read.shape or not np.allclose(written, read, atol=1e-5):
                raise IngestError(
                    f"layer {name!r} did not read back as written; inspect the kept "
                    "partial file."
                )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Download the Natural Earth layers the map basemap is drawn from.

Overview
--------
The spatial panels in :mod:`sipnet_calibration.plotting.maps` draw coastlines,
country and state/province boundaries and large lakes under the data. Those
come from Natural Earth's 1:50m vectors, which this script downloads from
Natural Earth's own CDN into ``data/raw/natural_earth/``, checking each archive
against the md5 recorded in :data:`SOURCES`.

Input data
----------
The four archives named in :data:`SOURCES`, over HTTPS from
``naciscdn.org``, the CDN ``naturalearthdata.com`` links its downloads to.

Output data
-----------
``--output-directory``, default the repository's ``data/raw/natural_earth/``
    One ``.zip`` per layer, byte for byte as served.

Notes
-----
**Not part of the ingest pipeline.** Like the other scripts in
``raw_sources/``, this *creates* a tracked raw input rather than processing
one, and a normal working copy never runs it: the archives are tracked, so a
checkout already has them. It exists so that where they came from is code
rather than prose, and so that fetching them again is one command.

**Nothing is unpacked, clipped or converted here.** The archives are the raw
input. ``scripts/build_basemap.py`` reads them and writes the clipped
polylines the plotting layer loads.

**A changed md5 is an error, not an update.** Natural Earth republishes
layers in place under the same URL when it issues a new version, so the md5
is what says the bytes are the ones the tracked basemap was built from. To
adopt a new release deliberately, run with ``--no-check``, record the new md5s
in :data:`SOURCES` and ``data/raw/natural_earth/provenance.md``, and rebuild
the basemap.

Each archive is written through :func:`sipnet_calibration.io.write_checked`,
and its check compares its md5 with the recorded one.

Natural Earth is in the public domain; see
https://www.naturalearthdata.com/about/terms-of-use/.

Usage
-----
::

    uv run python scripts/raw_sources/download_natural_earth.py
    uv run python scripts/raw_sources/download_natural_earth.py --no-check
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sipnet_calibration.io import file_md5, write_checked

#: Natural Earth's CDN, which the download links on naturalearthdata.com
#: resolve to.
BASE_URL = "https://naciscdn.org/naturalearth/50m"

#: Where the tracked archives go: this repository's ``data/raw/natural_earth/``,
#: whatever ``$SIPNET_CALIBRATION_DATA`` says, since a tracked input lives in
#: the checkout.
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "raw" / "natural_earth"


@dataclass(frozen=True)
class Source:
    """One Natural Earth layer: where it is served from and what it hashes to."""

    file_name: str
    theme: str
    md5: str

    @property
    def url(self) -> str:
        """Where Natural Earth serves the archive."""
        return f"{BASE_URL}/{self.theme}/{self.file_name}"


#: The layers the basemap draws, with the md5 of the archive each was built
#: from.
SOURCES: tuple[Source, ...] = (
    Source("ne_50m_coastline.zip", "physical", "7639330d2519efa1005eac0407172130"),
    Source("ne_50m_lakes.zip", "physical", "93de5a3d32451e7c18c75425b2f071dd"),
    Source("ne_50m_admin_0_boundary_lines_land.zip", "cultural", "2c6695791ef99755162094e7bfad97b2"),
    Source("ne_50m_admin_1_states_provinces_lines.zip", "cultural", "4f1373b05294a5848886e72f0f6a30a5"),
)


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Download every archive of :data:`SOURCES`, or report why one failed."""
    args = parse_args(argv)
    try:
        for source in SOURCES:
            digest = download(source, args.output_directory, check=not args.no_check)
            print(f"{source.file_name}  md5 {digest}  <- {source.url}")
    except (IngestError, OSError, ValueError, LookupError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line, as the module docstring's Usage describes it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Where the archives go. Default: {DEFAULT_OUTPUT_DIRECTORY}.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Keep an archive whose md5 differs from the recorded one, and print it.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def download(source: Source, output_directory: Path, *, check: bool) -> str:
    """Fetch one archive into *output_directory*; its md5.

    With *check* false, an archive whose md5 differs from the recorded one is
    kept rather than refused.
    """
    request = urllib.request.Request(source.url, headers={"User-Agent": "sipnet-calibration"})

    def fetch(partial: Path) -> None:
        """Write the archive, as served, to *partial*."""
        with urllib.request.urlopen(request, timeout=60) as response:
            partial.write_bytes(response.read())

    written = write_checked(
        output_directory / source.file_name,
        write=fetch,
        check=(
            (lambda partial: check_archive_md5_is_recorded(source, file_md5(partial)))
            if check
            else (lambda partial: None)
        ),
    )
    return file_md5(written)


# ── supporting types and helpers ──────────────────────────────────────────────


class IngestError(RuntimeError):
    """An archive was not what :data:`SOURCES` records."""


# ── checks ────────────────────────────────────────────────────────────────────


def check_archive_md5_is_recorded(source: Source, digest: str) -> None:
    """A downloaded archive's md5 is the one :data:`SOURCES` records for it."""
    if digest != source.md5:
        raise IngestError(
            f"{source.file_name} from {source.url} has md5 {digest}, but {source.md5} is "
            "recorded, so Natural Earth has probably issued a new release; to adopt it, "
            "re-run with --no-check, record the new md5 in SOURCES and in "
            "data/raw/natural_earth/provenance.md, and rebuild the basemap with "
            "scripts/build_basemap.py."
        )


if __name__ == "__main__":
    raise SystemExit(main())

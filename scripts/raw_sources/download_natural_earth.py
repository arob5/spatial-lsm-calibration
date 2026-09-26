#!/usr/bin/env python
"""Download the Natural Earth layers the map basemap is drawn from.

Overview
--------
The spatial panels in :mod:`sipnet_calibration.plotting.maps` draw coastlines,
country and state/province boundaries and large lakes under the data. Those
come from Natural Earth's 1:50m vectors, which this script downloads from
Natural Earth's own CDN into ``data/raw/natural_earth/``, checking each archive
against the md5 recorded in :data:`SOURCES`.

Like the other scripts in ``raw_sources/``, this **creates** a tracked raw input
rather than processing one, and a normal working copy never runs it: the
archives are tracked, so a checkout already has them. It exists so that where
they came from is code rather than prose, and so that fetching them again is
one command.

Input data
----------
The four archives named in :data:`SOURCES`, over HTTPS from
``naciscdn.org``, the CDN ``naturalearthdata.com`` links its downloads to.

Output data
-----------
``--out-dir``, default ``data/raw/natural_earth/``
    One ``.zip`` per layer, byte for byte as served. Each is written to a
    ``.partial`` path and renamed only once its md5 has been checked, so a
    failed or interrupted download cannot leave a corrupt archive where a
    tracked one belongs; the ``.partial`` file is kept for inspection and its
    path printed.

Notes
-----
**Nothing is unpacked, clipped or converted here.** The archives are the raw
input. ``scripts/build_basemap.py`` reads them and writes the clipped
polylines the plotting layer loads.

**A changed md5 is an error, not an update.** Natural Earth republishes
layers in place under the same URL when it issues a new version, so the md5
is what says the bytes are the ones the tracked basemap was built from. To
adopt a new release deliberately, run with ``--no-check``, record the new md5s
in :data:`SOURCES` and ``data/raw/natural_earth/provenance.md``, and rebuild
the basemap.

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

DEFAULT_OUT_DIR = Path("data/raw/natural_earth")


@dataclass(frozen=True)
class Source:
    """One Natural Earth layer: where it is served from and what it hashes to."""

    file_name: str
    theme: str
    md5: str

    @property
    def url(self) -> str:
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
    args = parse_args(argv)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for source in SOURCES:
            digest = download(source, args.out_dir, check=not args.no_check)
            print(f"{source.file_name}  md5 {digest}  <- {source.url}")
    except (DownloadError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where the archives go. Default: {DEFAULT_OUT_DIR}.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Keep an archive whose md5 differs from the recorded one, and print it.",
    )
    return parser.parse_args(argv)


# ── the steps, in the order main calls them ───────────────────────────────────


def download(source: Source, out_dir: Path, *, check: bool) -> str:
    """Fetch one archive into *out_dir*, returning its md5."""
    request = urllib.request.Request(source.url, headers={"User-Agent": "sipnet-calibration"})

    def fetch(partial: Path) -> None:
        with urllib.request.urlopen(request, timeout=60) as response:
            partial.write_bytes(response.read())

    def check_digest(partial: Path) -> None:
        if check:
            check_md5_matches(source, file_md5(partial))

    return file_md5(write_checked(out_dir / source.file_name, write=fetch, check=check_digest))


# ── supporting types and helpers ──────────────────────────────────────────────


class DownloadError(RuntimeError):
    """An archive was not what :data:`SOURCES` records."""


# ── checks ────────────────────────────────────────────────────────────────────


def check_md5_matches(source: Source, digest: str) -> None:
    if digest != source.md5:
        raise DownloadError(
            f"{source.file_name} from {source.url} has md5 {digest}, but {source.md5} "
            "is recorded. Natural Earth has probably issued a new release. To adopt "
            "it, re-run with --no-check, record the new md5 in SOURCES and in "
            "data/raw/natural_earth/provenance.md, and rebuild the basemap with "
            "scripts/build_basemap.py."
        )


if __name__ == "__main__":
    sys.exit(main())

"""Settings for the 2026-09-25 progress presentation.

The single source of truth for the choices the deck makes: which sites it
features, which site-labels product it groups by, and where its inputs and
outputs live. ``plots.py``, ``precompute_drivers.py`` and ``slides.qmd`` read
these rather than repeating them.
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent

#: Sites shown in every site-level figure, display name to site id. Harvard
#: Forest is ``US-Ha1``. Site 4705's name in the site table says ``US-Bar``,
#: the AmeriFlux Bartlett tower, while its ``ameriflux_site_id`` is ``US-xBR``,
#: the NEON tower at the same forest.
FEATURED_SITES = {
    "Harvard Forest": 4977,
    "Bartlett": 4705,
}

#: The site-labels product every by-class figure groups by.
SITE_LABELS = "pft_16class"

#: The three-class labels the reanalysis used, for comparison.
REANALYSIS_SITE_LABELS = "reanalysis_3pft"

#: The frame of every map covering the whole pool.
MAP_EXTENT = "NORTH_AMERICA"

#: Summaries computed on the SCC by ``precompute_drivers.py`` and copied back.
#: Untracked; see this directory's README.
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

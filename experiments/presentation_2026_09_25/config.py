"""Settings for the 2026-09-25 progress presentation.

The single source of truth for the choices the deck makes: which sites it
features, which sites and driver members it runs, which site-labels product it
groups by, and where its inputs and outputs live. ``plots.py``,
``relabel_drivers.py``, ``precompute_drivers.py`` and ``slides.qmd`` read these
rather than repeating them.
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

#: The New England research sites, display name to site id. Other entries in
#: the site table at Harvard Forest (4974, 4976, 4983) and Howland (4478, 4481)
#: are left out: each is within a few km of the site kept, so shares its drivers.
NEW_ENGLAND_SITES = {
    "Harvard Forest": 4977,
    "Bartlett": 4705,
    "Hubbard Brook": 4734,
    "Howland Forest": 4480,
    "Argyle": 4511,
    "Great Mountain Forest": 5089,
    "Plum Island": 4930,
}

#: Three sites from every class of ``SITE_LABELS``, drawn by
#: ``draw_pft_sites.py``, which reproduces this literal: among the sites the
#: producer assigned directly, up to two with an AmeriFlux id, and the rest
#: without one. Each comment lists the sites' AmeriFlux ids.
PFT_SITES = {
    "Evergreen_Needleleaf_Forest__P1": (1810, 2359, 2732),  # -, -, -
    "Evergreen_Needleleaf_Forest__P2": (4360, 6121, 4462),  # US-xWR, US-NC1, -
    "Evergreen_Broadleaf_Forest": (6504, 6900, 7828),  # US-HB2, -, -
    "Deciduous_Broadleaf_Forest__P1_P2_P3": (4336, 5301, 5304),  # US-PFr, US-SSH, -
    "Deciduous_Broadleaf_Forest__P4_P5": (5588, 6126, 6860),  # US-xUK, US-NC3, -
    "Mixed_Forest__P1": (1873, 2476, 2662),  # -, -, -
    "Mixed_Forest__P2": (4307, 4480, 6420),  # US-PFe, US-Ho1, -
    "Open_Shrublands__P1": (699, 1546, 1914),  # -, -, -
    "Open_Shrublands__P2": (6361, 6771, 7193),  # US-Seg, US-xSR, -
    "Open_Vegetation_Complex_P1": (4888, 5406, 1968),  # US-Rms, US-xNW, -
    "Open_Vegetation_Complex_P2": (4624, 5752, 4255),  # US-CS8, US-Myb, -
    "Open_Vegetation_Complex_P3": (5361, 7179, 7382),  # US-xRM, US-xSB, -
    "Open_Vegetation_Complex_P4": (5275, 6030, 6698),  # US-xCP, US-AR1, -
    "CroplandPool__Broad_Croplands": (5467, 5735, 7624),  # US-RGo, US-Tw3, -
    "CroplandPool__Cereal_Croplands": (4407, 5732, 5658),  # US-MN1, US-DS3, -
    "Permanent_Wetlands": (5162, 7196, 3536),  # US-WPT, US-LA3, -
}

#: The driver members every run uses: the source's 1-based member indices, as
#: in the ``ERA5_<site>_<member>`` directory names.
DRIVER_MEMBERS = tuple(range(1, 11))

#: Every site the deck runs SIPNET at, ascending.
DRIVER_SITES = tuple(
    sorted({*NEW_ENGLAND_SITES.values(), *(s for sites in PFT_SITES.values() for s in sites)})
)

#: The year the driver time series show, by the start of each daily cell.
DRIVER_SERIES_YEAR = 2023

#: The first and last day, inclusive, of the short window the drivers are also
#: shown over, at their own 3-hourly step.
DRIVER_SERIES_WINDOW = ("2023-07-01", "2023-07-14")

#: Where Dongchen's ERA5 driver ensemble lives on the SCC, as symlinked
#: into ``data/raw/drivers`` there.
DRIVERS_SOURCE = "/projectnb/dietzelab/dongchen/anchorSites/NA_runs/ERA5_2012_2024"

#: The site-labels product every by-class figure groups by.
SITE_LABELS = "pft_16class"

#: The three-class labels the reanalysis used, for comparison.
REANALYSIS_SITE_LABELS = "reanalysis_3pft"

#: The frame of every map covering the whole pool.
MAP_EXTENT = "NORTH_AMERICA"

#: Summaries computed on the SCC by ``precompute_drivers.py`` and copied back.
#: Untracked; see this directory's README.
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

#: The drivers of ``DRIVER_SITES`` x ``DRIVER_MEMBERS`` with their hour column
#: corrected, written by ``relabel_drivers.py`` in the ``data/raw/drivers/``
#: layout. A temporary stand-in for regenerated drivers; untracked.
RELABELED_DRIVERS_DIR = OUTPUT_DIR / "drivers_relabeled"

#: The annual values and monthly climatology of every site's drivers, written
#: by ``precompute_drivers.py`` on the SCC; untracked.
DRIVER_SUMMARY_DIR = OUTPUT_DIR / "driver_summaries"

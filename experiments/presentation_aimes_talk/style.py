"""The figures' style, from the slides' brand file, so plots match the slides.

The palette and fonts are read from ``theme/_brand.yml`` (``config.BRAND_FILE``),
the same file the slide theme reads. Beyond the palette, this module fixes
what each color *means* across every figure, so a method or a kind of site
looks the same everywhere:

- methods: ``METHOD_COLORS`` (site by site, spatial, IPL, prior);
- sites: ``SITE_MARKERS`` (tower, sparse, held-out), distinguished by shape
  as well as color;
- fields: ``FIELD_COLORMAP`` for parameter values, ``UNCERTAINTY_COLORMAP``
  for posterior SDs, and ``DEVIATION_COLORMAP`` for terms centered at 0.
"""

import matplotlib as mpl
import yaml
from cycler import cycler

from . import config

__all__ = [
    "DEVIATION_COLORMAP",
    "FIELD_COLORMAP",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "PALETTE",
    "SITE_MARKERS",
    "UNCERTAINTY_COLORMAP",
    "apply_style",
]

with open(config.BRAND_FILE) as brand_file:
    _BRAND = yaml.safe_load(brand_file)

#: The brand palette: name -> hex (paper, ink, stone, mist, forest, sky, soil,
#: sun, clay, leaf).
PALETTE: dict[str, str] = _BRAND["color"]["palette"]

#: What each method is drawn in.
METHOD_COLORS = {
    "prior": PALETTE["stone"],
    "site_by_site": PALETTE["clay"],
    "spatial_reference": PALETTE["forest"],
    "ipl": PALETTE["sky"],
}
#: What each method is called on a figure.
METHOD_LABELS = {
    "prior": "prior only",
    "site_by_site": "site by site",
    "spatial_reference": "spatial (exact)",
    "ipl": "spatial (scalable)",
}

#: How each kind of site is drawn: scatter keywords.
SITE_MARKERS = {
    "tower": dict(marker="^", s=90, color=PALETTE["sun"], edgecolor=PALETTE["ink"], linewidth=1.0, zorder=4),
    "sparse": dict(marker="o", s=22, color="white", edgecolor=PALETTE["ink"], linewidth=0.7, zorder=3),
    "held_out": dict(marker="X", s=55, color=PALETTE["clay"], edgecolor="white", linewidth=0.6, zorder=4),
}

FIELD_COLORMAP = "viridis"
UNCERTAINTY_COLORMAP = "YlOrBr"
DEVIATION_COLORMAP = "PuOr_r"


def apply_style(font_scale: float = 1.0) -> None:
    """Set matplotlib's defaults to the brand style, transparent on the slide's paper."""
    ink, stone = PALETTE["ink"], PALETTE["stone"]
    base = 14 * font_scale
    body_font = _BRAND["typography"]["base"]["family"]
    heading_font = _BRAND["typography"]["headings"]["family"]
    mpl.rcParams.update(
        {
            # Falls back to DejaVu Sans where the brand fonts are not installed.
            "font.family": "sans-serif",
            "font.sans-serif": [body_font, heading_font, "DejaVu Sans"],
            "font.size": base,
            "axes.titlesize": base * 1.1,
            "axes.titleweight": "bold",
            "axes.labelsize": base,
            "xtick.labelsize": base * 0.85,
            "ytick.labelsize": base * 0.85,
            "legend.fontsize": base * 0.9,
            "mathtext.fontset": "cm",
            "axes.prop_cycle": cycler(color=[PALETTE[name] for name in ("forest", "sky", "sun", "clay", "soil")]),
            "text.color": ink,
            "axes.labelcolor": ink,
            "axes.edgecolor": stone,
            "xtick.color": stone,
            "ytick.color": stone,
            "xtick.labelcolor": ink,
            "ytick.labelcolor": ink,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": stone,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )

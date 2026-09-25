"""Animate drivers through their monthly climatology, one GIF per variable.

Overview
--------
For each of ``config.DRIVER_ANIMATION_VARIABLES``, maps the ensemble mean of
its monthly climatology over North America, one frame per month on one color
scale, and writes the animation as a GIF that ``slides.qmd`` embeds. Made once,
since drawing twelve maps of 8000 sites on every render would be slow.

Input data
----------
``config.DRIVER_SUMMARY_DIR / "driver_monthly_climatology.nc"``
    Written by ``precompute_drivers.py`` on the SCC: each driver on
    ``(member, site, month)``.

Output data
-----------
``config.DRIVER_ANIMATION_DIR / "<variable>.gif"``, one per variable: twelve
frames, January to December, looping.

Usage
-----
From this directory, so that ``config`` imports::

    python make_driver_animations.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from sipnet_calibration.plotting import animate_map, use_project_style

import config
import plots

#: Seconds each month is shown for.
SECONDS_PER_FRAME = 0.7

#: The size and resolution of each GIF.
FIGURE_SIZE = (8.0, 4.9)
DPI = 110


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    use_project_style()
    # Each frame's title is set afresh, so its clearance of the longitude labels
    # along the map's top edge has to be the default.
    plt.rcParams.update({
        "font.size": 20, "axes.titlesize": 22, "axes.labelsize": 15, "ytick.labelsize": 14,
        "axes.titlepad": 26,
    })
    climatology = plots.load_driver_summary("driver_monthly_climatology")
    for name in config.DRIVER_ANIMATION_VARIABLES:
        path = config.DRIVER_ANIMATION_DIR / f"{name}.gif"
        write_animation(climatology[name], path)
        print(f"wrote {path}")
    return 0


# ── steps ────────────────────────────────────────────────────────────────────


def write_animation(field, path: Path) -> None:
    """Animate one variable's climatology and save it as a GIF at *path*."""
    frames = plots.driver_climatology_frames(field)
    dim = next(d for d in frames.dims if d != "site")
    figure, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    animation = animate_map(frames, dim, ax=ax, extent=config.MAP_EXTENT, robust=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow picks the format from the extension, so the partial keeps ".gif".
    partial = path.with_name(f"{path.stem}.partial.gif")
    animation.save(partial, writer=PillowWriter(fps=1 / SECONDS_PER_FRAME), dpi=DPI)
    plt.close(figure)
    os.replace(partial, path)


if __name__ == "__main__":
    sys.exit(main())

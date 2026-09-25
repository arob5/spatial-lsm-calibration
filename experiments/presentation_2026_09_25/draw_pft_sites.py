"""Draw the deck's example sites for each class of the 16-class PFT labels.

Overview
--------
Draws three sites from every class of ``config.SITE_LABELS``, favoring flux
tower sites but always including at least one site without a tower, and prints
the draw as the Python literal ``config.PFT_SITES`` holds. The draw is made once
and pasted into the config, so every script in this directory and every
machine see the same sites; this script is how that literal was made, and
rerunning it reproduces it.

Input data
----------
``data/processed/sites/sites.csv``
    The site table, for ``ameriflux_site_id``: a site is a tower site when it
    has one.
``data/processed/site_labels/pft_16class.csv``
    The class of every site.
``data/raw/site_labels/site_pft_16class.csv``
    The raw labels, for ``final_pft_direct``: only sites the producer assigned
    to a class directly are drawn, not the ones assigned by nearest ecological
    profile.

Output data
-----------
None written. The literal is printed to standard output.

Notes
-----
In each class, up to ``TOWERS_PER_CLASS`` tower sites are drawn, and the
remaining places are drawn from the sites without a tower. A class with fewer
tower sites than that gets more sites without one, and a class with none
gets only sites without one.

Usage
-----
From this directory, so that ``config`` imports::

    python draw_pft_sites.py
"""

import sys

import pandas as pd

from sipnet_calibration import site_labels
from sipnet_calibration.sites import load_sites, select_sites

import config

#: Sites drawn per class.
SITES_PER_CLASS = 3

#: At most this many of them are tower sites, so at least one has no tower.
TOWERS_PER_CLASS = 2

#: The seed of every draw.
SEED = 20260925


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    candidates = candidate_sites()
    draw = {label: draw_class(candidates, label) for label in class_order()}
    try:
        check_every_class_is_full(draw)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(format_literal(draw, candidates))
    return 0


# ── steps ────────────────────────────────────────────────────────────────────


def candidate_sites() -> pd.DataFrame:
    """The directly assigned sites, with their class and whether they have a tower."""
    spec = site_labels.resolve_site_labels(config.SITE_LABELS)
    raw = site_labels.read_raw(spec)
    direct = raw.loc[raw["final_pft_direct"] != "", spec.site_column]
    table = load_sites().merge(site_labels.load_site_labels(spec), on="site_id")
    table["tower"] = table["ameriflux_site_id"] != ""
    return table[table["site_id"].isin(direct)].reset_index(drop=True)


def class_order() -> tuple[str, ...]:
    """The classes, in the order the spec declares them."""
    return site_labels.resolve_site_labels(config.SITE_LABELS).labels


def draw_class(candidates: pd.DataFrame, label: str) -> tuple[int, ...]:
    """The tower sites, then the others, drawn for one class, each part in id order."""
    in_class = candidates["label"] == label
    towers = candidates[in_class & candidates["tower"]]
    n_towers = min(TOWERS_PER_CLASS, len(towers))
    others = candidates[in_class & ~candidates["tower"]]
    drawn = [
        select_sites(towers, sample=n_towers, seed=SEED),
        select_sites(others, sample=SITES_PER_CLASS - n_towers, seed=SEED),
    ]
    return tuple(int(site) for part in drawn for site in part["site_id"])


def format_literal(draw: dict[str, tuple[int, ...]], candidates: pd.DataFrame) -> str:
    """``PFT_SITES = {...}``, with each site's tower id, if any, as a comment."""
    towers = candidates.set_index("site_id")["ameriflux_site_id"]
    lines = ["PFT_SITES = {"]
    for label, sites in draw.items():
        ids = ", ".join(str(site) for site in sites)
        names = ", ".join(towers[site] or "-" for site in sites)
        lines.append(f'    "{label}": ({ids}),  # {names}')
    lines.append("}")
    return "\n".join(lines)


# ── checks ───────────────────────────────────────────────────────────────────


def check_every_class_is_full(draw: dict[str, tuple[int, ...]]) -> None:
    """Every class drew ``SITES_PER_CLASS`` distinct sites."""
    for label, sites in draw.items():
        if len(set(sites)) != SITES_PER_CLASS:
            raise ValueError(
                f"{label} drew {len(set(sites))} distinct site(s), not {SITES_PER_CLASS}"
            )


if __name__ == "__main__":
    sys.exit(main())

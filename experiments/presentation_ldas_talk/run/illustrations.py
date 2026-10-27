"""Compute the talk's illustrations and save them.

Overview
--------
Two pictures that are not experiments: the 1D transect (a parameter's prior
and its posterior after a few measurements) and draws from a spatial prior
over the real North American site pool. Every number comes from ``config``.

Input data
----------
For the North America draws: the site table (``scripts/ingest_sites.py``
writes it), the site covariate table and the NEE tower table, both tracked.

Output data
-----------
``config.TRANSECT_FILE`` (``results.Transect``) and
``config.NORTH_AMERICA_FILE`` (``results.NorthAmericaPrior``).

Notes
-----
The North America draws factor an 8000 x 8000 covariance: about a gigabyte
of memory and some seconds.

Usage
-----
::

    uv run python -m experiments.presentation_ldas_talk.run.illustrations
"""

import numpy as np

from .. import config, results
from ..illustrations.north_america import draw_north_america_prior
from ..illustrations.transect import compute_transect


def main() -> None:
    transect = compute_transect(
        np.random.default_rng(config.TRANSECT_SEED),
        length_scale=config.TRANSECT_LENGTH_SCALE,
        field_sd=config.TRANSECT_FIELD_SD,
        measurement_locations=config.TRANSECT_MEASUREMENT_LOCATIONS,
        measurement_sd=config.TRANSECT_MEASUREMENT_SD,
        grid_size=config.TRANSECT_GRID_SIZE,
        n_draws=config.TRANSECT_N_DRAWS,
    )
    results.save(transect, config.TRANSECT_FILE)
    print(f"wrote {config.TRANSECT_FILE}")

    north_america = draw_north_america_prior(
        np.random.default_rng(config.NORTH_AMERICA_SEED),
        n_draws=config.NORTH_AMERICA_N_DRAWS,
        length_scale_km=config.NORTH_AMERICA_LENGTH_SCALE_KM,
        field_sd=config.NORTH_AMERICA_FIELD_SD,
        trend_slope=config.NORTH_AMERICA_TREND_SLOPE,
        site_noise_sd=config.NORTH_AMERICA_SITE_NOISE_SD,
    )
    results.save(north_america, config.NORTH_AMERICA_FILE)
    print(f"wrote {config.NORTH_AMERICA_FILE} ({len(north_america.tower_site_ids)} tower sites)")


if __name__ == "__main__":
    main()

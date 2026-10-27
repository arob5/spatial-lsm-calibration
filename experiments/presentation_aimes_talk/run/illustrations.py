"""Compute the talk's illustrations and save them.

Overview
--------
Pictures that are not experiments: the 1D transect (a parameter's prior and
its posterior after a few measurements), draws from a spatial prior over
North America, and a parameter at every site under the non-spatial
calibration options. Every number comes from ``config``.

Input data
----------
For the North America draws: the site table (``scripts/ingest_sites.py``
writes it), the site covariate table and the NEE tower table, both tracked.
For the calibration options: the site table and the 16-class PFT site
labels, tracked.

Output data
-----------
``config.TRANSECT_FILE`` (``results.Transect``),
``config.NORTH_AMERICA_FILE`` (``results.NorthAmericaPrior``) and
``config.CALIBRATION_OPTIONS_FILE`` (``results.CalibrationOptions``).

Notes
-----
The North America draws factor the covariance of every field-grid point
near a site (about 10,000 at the default spacing): about a gigabyte of
memory and some seconds.

Usage
-----
::

    uv run python -m experiments.presentation_aimes_talk.run.illustrations
"""

import numpy as np

from .. import config, results
from ..illustrations.calibration_options import draw_calibration_options
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
        length_scale=config.NORTH_AMERICA_LENGTH_SCALE,
        field_sd=config.NORTH_AMERICA_FIELD_SD,
        trend_slope=config.NORTH_AMERICA_TREND_SLOPE,
        grid_spacing=config.NORTH_AMERICA_GRID_SPACING,
        display_spacing=config.NORTH_AMERICA_DISPLAY_SPACING,
        land_radius=config.NORTH_AMERICA_LAND_RADIUS,
        max_hole_area=config.NORTH_AMERICA_MAX_HOLE_AREA,
    )
    results.save(north_america, config.NORTH_AMERICA_FILE)
    print(f"wrote {config.NORTH_AMERICA_FILE} ({len(north_america.tower_site_ids)} tower sites)")

    calibration_options = draw_calibration_options(
        np.random.default_rng(config.CALIBRATION_OPTIONS_SEED),
        parameter_sd=config.CALIBRATION_OPTIONS_PARAMETER_SD,
        site_labels_name=config.CALIBRATION_OPTIONS_SITE_LABELS,
    )
    results.save(calibration_options, config.CALIBRATION_OPTIONS_FILE)
    print(f"wrote {config.CALIBRATION_OPTIONS_FILE} ({len(calibration_options.pft_names)} PFTs)")


if __name__ == "__main__":
    main()

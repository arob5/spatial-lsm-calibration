"""Compute the talk's illustrations and save them.

Overview
--------
Pictures that are not experiments: draws from a spatial prior over North
America and its prior and posterior uncertainty, and a parameter at every
site under the non-spatial calibration options. Every number comes
from ``config``.

Input data
----------
For the North America draws: the site table (``scripts/ingest_sites.py``
writes it), the site covariate table and the NEE tower table, both tracked.
For the calibration options: the site table and the 16-class PFT site
labels, tracked.

Output data
-----------
``config.NORTH_AMERICA_FILE`` (``results.NorthAmericaPrior``),
``config.UNCERTAINTY_FILE`` (``results.UncertaintyMaps``) and
``config.CALIBRATION_OPTIONS_FILE`` (``results.CalibrationOptions``).

Notes
-----
The North America draws factor the covariance of every field-grid point
near a site (about 10,000 at the default spacing), and the uncertainty maps
the covariance of the 8000 sites' data: about a gigabyte of memory each,
and some seconds.

Usage
-----
::

    uv run python -m experiments.presentation_aimes_talk.run.illustrations
"""

import numpy as np

from .. import config, results
from ..illustrations.calibration_options import draw_calibration_options
from ..illustrations.continent import build_continent_grids
from ..illustrations.north_america import draw_north_america_prior
from ..illustrations.uncertainty import compute_uncertainty_maps


def main() -> None:
    # The continental maps share one pair of grids.
    grids = build_continent_grids(
        grid_spacing=config.NORTH_AMERICA_GRID_SPACING,
        display_spacing=config.NORTH_AMERICA_DISPLAY_SPACING,
        land_radius=config.NORTH_AMERICA_LAND_RADIUS,
        max_hole_area=config.NORTH_AMERICA_MAX_HOLE_AREA,
    )
    north_america = draw_north_america_prior(
        np.random.default_rng(config.NORTH_AMERICA_SEED),
        grids,
        n_draws=config.NORTH_AMERICA_N_DRAWS,
        length_scale=config.NORTH_AMERICA_LENGTH_SCALE,
        field_sd=config.NORTH_AMERICA_FIELD_SD,
        trend_slope=config.NORTH_AMERICA_TREND_SLOPE,
    )
    results.save(north_america, config.NORTH_AMERICA_FILE)
    print(f"wrote {config.NORTH_AMERICA_FILE} ({len(north_america.tower_site_ids)} tower sites)")

    uncertainty = compute_uncertainty_maps(
        grids,
        length_scale=config.NORTH_AMERICA_LENGTH_SCALE,
        field_sd=config.NORTH_AMERICA_FIELD_SD,
        n_towers=config.UNCERTAINTY_N_TOWERS,
        tower_noise_sd=config.UNCERTAINTY_TOWER_NOISE_SD,
        site_noise_sd=config.UNCERTAINTY_SITE_NOISE_SD,
    )
    results.save(uncertainty, config.UNCERTAINTY_FILE)
    print(f"wrote {config.UNCERTAINTY_FILE}")

    calibration_options = draw_calibration_options(
        np.random.default_rng(config.CALIBRATION_OPTIONS_SEED),
        parameter_sd=config.CALIBRATION_OPTIONS_PARAMETER_SD,
        site_labels_name=config.CALIBRATION_OPTIONS_SITE_LABELS,
    )
    results.save(calibration_options, config.CALIBRATION_OPTIONS_FILE)
    print(f"wrote {config.CALIBRATION_OPTIONS_FILE} ({len(calibration_options.pft_names)} PFTs)")


if __name__ == "__main__":
    main()

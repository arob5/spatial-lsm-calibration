"""Two ways to calibrate across sites without spatial structure, as maps.

The slide comparing calibration options shows one generic parameter
``theta`` at every site of the real pool under each assumption. This module
draws the two that are not spatial; the spatial one is a draw from
``north_america``:

- **independent**: a separate value at each site, ``theta_s ~ N(0, sd^2)``
  independently, so neighboring sites share nothing;
- **by PFT**: one value per plant functional type, ``theta_k ~ N(0, sd^2)``
  independently, and every site takes its PFT's value,
  ``theta_s = theta_pft(s)``.

The PFTs are a site-labels data source of the repository, read through
``sipnet_calibration.site_labels``. The values are illustrations, not
fitted: only their pattern on the map matters.
"""

import numpy as np

from sipnet_calibration.site_labels import (
    LABEL_COLUMN,
    build_site_labels,
    read_raw,
    resolve_site_labels,
)
from sipnet_calibration.sites import load_sites

from ..results import CalibrationOptions

__all__ = ["draw_calibration_options"]


def draw_calibration_options(rng: np.random.Generator, *, parameter_sd: float, site_labels_name: str) -> CalibrationOptions:
    """Draw the independent and by-PFT parameter values at every site of the pool."""
    site_table = load_sites()
    site_ids = site_table["site_id"].to_numpy()

    spec = resolve_site_labels(site_labels_name)
    site_labels = build_site_labels(spec, read_raw(spec)).set_index("site_id")
    pft = site_labels.loc[site_ids, LABEL_COLUMN]
    pft_codes = pft.cat.codes.to_numpy()

    independent = parameter_sd * rng.standard_normal(len(site_ids))
    value_per_pft = parameter_sd * rng.standard_normal(len(spec.class_names))
    return CalibrationOptions(
        site_ids=site_ids,
        lon=site_table["lon"].to_numpy(),
        lat=site_table["lat"].to_numpy(),
        pft_names=np.asarray(spec.class_names),
        pft_codes=pft_codes,
        independent=independent,
        by_pft=value_per_pft[pft_codes],
    )

"""Two-dimensional posterior densities on regular grids, and their contours.

A site's posterior over its two parameters is shown as contours of
highest-density regions: the smallest region holding a given probability.
The site-by-site posterior is known exactly on a grid
(``inference.site_by_site``); the spatial posterior is known only through
MCMC draws, so its density is estimated by a Gaussian kernel density
estimate (KDE). The checks here measure whether that estimate can be
trusted for the contours drawn:

- **reproduction**: draw samples from an exactly known density, estimate it
  by the same KDE, and measure how much of the exact probability lies inside
  the KDE's regions. If the KDE added width or smoothed away a curve, the
  KDE's 90% region would hold more or less than 90% of the exact mass;
- **coverage**: the fraction of the MCMC draws themselves inside the KDE's
  regions, which should match each region's probability.

A density on a grid is an array ``(n_0, n_1)``, its first index along
``axis_0`` and its second along ``axis_1``, normalized so that its sum times
the cell area is 1.
"""

import numpy as np
from scipy.stats import gaussian_kde

__all__ = [
    "fraction_of_samples_inside",
    "gaussian_kde_on_grid",
    "highest_density_levels",
    "mass_inside",
    "sample_from_grid_density",
]


def highest_density_levels(density: np.ndarray, probabilities) -> list[float]:
    """Density thresholds whose superlevel sets hold each probability (one per probability)."""
    values = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(values) / values.sum()
    return [float(values[np.searchsorted(cumulative, probability)]) for probability in probabilities]


def gaussian_kde_on_grid(
    samples: np.ndarray, axis_0: np.ndarray, axis_1: np.ndarray, *, bandwidth_factor: float
) -> np.ndarray:
    """A Gaussian KDE of ``samples`` ``(n, 2)`` evaluated on the grid, normalized on it.

    The bandwidth is ``bandwidth_factor`` times Scott's rule, SciPy's
    default, applied to the samples' covariance. Scott's rule assumes a
    Gaussian and oversmooths a curved, ridge-like posterior, widening its
    regions; the reproduction check measures by how much.
    """
    estimate = gaussian_kde(samples.T)
    estimate.set_bandwidth(estimate.scotts_factor() * bandwidth_factor)
    mesh_0, mesh_1 = np.meshgrid(axis_0, axis_1, indexing="ij")
    density = estimate(np.vstack([mesh_0.ravel(), mesh_1.ravel()])).reshape(mesh_0.shape)
    return density / (density.sum() * _cell_area(axis_0, axis_1))


def sample_from_grid_density(
    density: np.ndarray, axis_0: np.ndarray, axis_1: np.ndarray, rng: np.random.Generator, n_samples: int
) -> np.ndarray:
    """``n_samples`` draws ``(n, 2)`` from a grid density: a cell by its probability, then uniform within it."""
    probabilities = (density / density.sum()).ravel()
    cells = rng.choice(probabilities.size, size=n_samples, p=probabilities)
    index_0, index_1 = np.unravel_index(cells, density.shape)
    width_0, width_1 = axis_0[1] - axis_0[0], axis_1[1] - axis_1[0]
    return np.column_stack([
        axis_0[index_0] + width_0 * (rng.uniform(size=n_samples) - 0.5),
        axis_1[index_1] + width_1 * (rng.uniform(size=n_samples) - 0.5),
    ])


def mass_inside(density: np.ndarray, region: np.ndarray) -> float:
    """The fraction of ``density``'s probability inside ``region``, a boolean grid of its shape."""
    return float(density[region].sum() / density.sum())


def fraction_of_samples_inside(
    samples: np.ndarray, density: np.ndarray, axis_0: np.ndarray, axis_1: np.ndarray, level: float
) -> float:
    """The fraction of ``samples`` in the region where ``density >= level``, read at each sample's nearest cell."""
    index_0 = _nearest_index(axis_0, samples[:, 0])
    index_1 = _nearest_index(axis_1, samples[:, 1])
    return float(np.mean(density[index_0, index_1] >= level))


# ── helpers ─────────────────────────────────────────────────────────────────


def _cell_area(axis_0: np.ndarray, axis_1: np.ndarray) -> float:
    return float((axis_0[1] - axis_0[0]) * (axis_1[1] - axis_1[0]))


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The index of each value's nearest point on a regular ``axis``, clipped to the axis."""
    return np.clip(np.rint((values - axis[0]) / (axis[1] - axis[0])).astype(int), 0, len(axis) - 1)

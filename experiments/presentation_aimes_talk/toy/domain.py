"""The toy region: where the sites are, their climate, and their drivers.

Space is the unit square. Three sets of locations are used:

- **training sites**, which carry observations;
- **held-out sites**, which carry none, used to score predictions at sites a
  method has not seen;
- **the map grid**, a regular grid on which the parameter fields are drawn.

The climate covariate ``z(x)`` stands in for an environmental covariate
such as mean annual temperature; it enters both the drivers and the prior's
climate trend (``spatial_prior``).
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Drivers",
    "Region",
    "build_region",
    "climate_covariate",
    "mean_annual_temperature",
    "simulate_drivers",
]


@dataclass(frozen=True)
class Region:
    """The locations of the toy problem, each an ``(n, 2)`` array in [0, 1]^2."""

    training_locations: np.ndarray
    held_out_locations: np.ndarray
    map_grid_locations: np.ndarray
    #: The map grid has ``map_grid_size`` points per side, ordered row by row
    #: (x1 fastest), so ``values.reshape(size, size)`` is an image with x2 on
    #: the first axis.
    map_grid_size: int


@dataclass(frozen=True)
class Drivers:
    """Monthly drivers at each training site, each ``(n_sites, n_months)``."""

    temperature: np.ndarray
    light: np.ndarray


def build_region(
    rng: np.random.Generator,
    *,
    n_training_sites: int,
    n_held_out_sites: int,
    map_grid_size: int,
) -> Region:
    """Scatter the training and held-out sites uniformly, and lay out the map grid."""
    training_locations = rng.uniform(size=(n_training_sites, 2))
    held_out_locations = rng.uniform(size=(n_held_out_sites, 2))
    # Cell centers of a map_grid_size x map_grid_size grid.
    axis = (np.arange(map_grid_size) + 0.5) / map_grid_size
    x1, x2 = np.meshgrid(axis, axis)
    map_grid_locations = np.column_stack([x1.ravel(), x2.ravel()])
    return Region(
        training_locations=training_locations,
        held_out_locations=held_out_locations,
        map_grid_locations=map_grid_locations,
        map_grid_size=map_grid_size,
    )


def climate_covariate(locations: np.ndarray) -> np.ndarray:
    """The covariate z(x) = 2 (0.5 - x2) + 0.35 sin(2 pi x1), shape ``(n,)``.

    Decreasing to the north (x2), with a gentle east-west wave; it lies
    within about [-1.35, 1.35].
    """
    x1, x2 = locations[:, 0], locations[:, 1]
    return 2.0 * (0.5 - x2) + 0.35 * np.sin(2.0 * np.pi * x1)


def mean_annual_temperature(locations: np.ndarray, *, center: float, scale: float) -> np.ndarray:
    """Mean annual temperature (deg C), ``center + scale * z(x)``."""
    return center + scale * climate_covariate(locations)


def simulate_drivers(
    locations: np.ndarray,
    rng: np.random.Generator,
    *,
    n_months: int,
    mat_center: float,
    mat_scale: float,
    seasonal_amplitude: float,
    weather_autocorrelation: float,
    weather_innovation_sd: float,
) -> Drivers:
    """Monthly temperature and light at each location (equations in ``config``)."""
    n_sites = len(locations)
    month = np.arange(n_months)
    mat = mean_annual_temperature(locations, center=mat_center, scale=mat_scale)
    seasonal_cycle = seasonal_amplitude * np.sin(2.0 * np.pi * (month - 3) / 12)
    weather = _ar1_series(
        rng,
        n_series=n_sites,
        length=n_months,
        autocorrelation=weather_autocorrelation,
        innovation_sd=weather_innovation_sd,
    )
    temperature = mat[:, None] + seasonal_cycle[None, :] + weather

    light_amplitude = 0.3 + 0.4 * locations[:, 1]
    light = 1.0 + light_amplitude[:, None] * np.sin(2.0 * np.pi * (month - 2.5) / 12)[None, :]
    return Drivers(temperature=temperature, light=light)


# ── helpers ─────────────────────────────────────────────────────────────────


def _ar1_series(
    rng: np.random.Generator,
    *,
    n_series: int,
    length: int,
    autocorrelation: float,
    innovation_sd: float,
) -> np.ndarray:
    """Independent stationary AR(1) series, shape ``(n_series, length)``."""
    series = np.empty((n_series, length))
    stationary_sd = innovation_sd / np.sqrt(1.0 - autocorrelation**2)
    series[:, 0] = stationary_sd * rng.standard_normal(n_series)
    for t in range(1, length):
        series[:, t] = autocorrelation * series[:, t - 1] + innovation_sd * rng.standard_normal(n_series)
    return series

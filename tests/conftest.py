"""Shared test fixtures.

The matplotlib backend is set to ``Agg`` here, before anything imports
``pyplot``, so that the plotting tests never need a display and never open a
window. The plotting tests assert on artist data and properties --
``line.get_xydata()``, ``collection.get_paths()``, colors, labels, axis
limits -- and never on rendered images, which are brittle across matplotlib
versions and say nothing about why a test failed.

The synthetic fixtures build canonical fields at each subset of the
``(member, site, time)`` dimensions, with a real ``DatetimeIndex`` on
``time``, ``lon``/``lat`` as non-dimension coordinates on ``site``, and
``units``/``long_name`` in ``attrs``.

The real-data fixtures read the driver files and the annual constraint product
present in this working copy, and skip when they are not there.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

#: The variable the synthetic fields stand in for, with the attributes a real
#: driver field carries.
SYNTHETIC_NAME = "air_temperature"
SYNTHETIC_ATTRS = {
    "units": "deg C",
    "long_name": "Mean air temperature over the timestep",
}

#: Site ids and their coordinates, matching the two sites whose driver files
#: are present locally, so that a synthetic field and a real one address the
#: same sites.
SYNTHETIC_SITES = (1, 27)
SYNTHETIC_LON = (-24.5625, -78.5625)
SYNTHETIC_LAT = (82.5458, 44.0654)


def make_canonical_field(
    dims: tuple[str, ...],
    *,
    n_time: int = 24,
    n_member: int = 3,
    seed: int = 0,
) -> xr.DataArray:
    """A synthetic canonical field with exactly *dims*, in canonical order.

    Parameters
    ----------
    dims:
        Any subset of ``("member", "site", "time")``, in any order; the result
        is transposed into canonical order.
    n_time:
        Length of the ``time`` dim, 3-hourly from 2012-01-01. Ignored when
        ``time`` is not in *dims*.
    n_member:
        Length of the ``member`` dim. Ignored when ``member`` is not in *dims*.
    seed:
        Seed for the values, so a test can compare two calls.

    Returns
    -------
    xarray.DataArray
        Named :data:`SYNTHETIC_NAME`, carrying :data:`SYNTHETIC_ATTRS`, with
        ``lon``/``lat`` on ``site`` whenever ``site`` is present.
    """
    unknown = set(dims) - {"member", "site", "time"}
    if unknown:
        raise ValueError(f"not canonical dims: {sorted(unknown)}")

    sizes = {"member": n_member, "site": len(SYNTHETIC_SITES), "time": n_time}
    order = tuple(d for d in ("member", "site", "time") if d in dims)
    shape = tuple(sizes[d] for d in order)

    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)

    coords: dict[str, object] = {}
    if "member" in order:
        coords["member"] = np.arange(n_member, dtype=np.int16)
    if "site" in order:
        coords["site"] = np.asarray(SYNTHETIC_SITES, dtype=np.int32)
        coords["lon"] = ("site", np.asarray(SYNTHETIC_LON))
        coords["lat"] = ("site", np.asarray(SYNTHETIC_LAT))
    if "time" in order:
        coords["time"] = pd.date_range("2012-01-01", periods=n_time, freq="3h")

    return xr.DataArray(
        values, dims=order, coords=coords, name=SYNTHETIC_NAME, attrs=dict(SYNTHETIC_ATTRS)
    )


@pytest.fixture
def ax():
    """A fresh ``Axes``, with its figure closed afterwards."""
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


@pytest.fixture
def field_time() -> xr.DataArray:
    """``(time,)`` -- one deterministic run."""
    return make_canonical_field(("time",))


@pytest.fixture
def field_member_time() -> xr.DataArray:
    """``(member, time)`` -- an ensemble at one site."""
    return make_canonical_field(("member", "time"))


@pytest.fixture
def field_site_time() -> xr.DataArray:
    """``(site, time)`` -- one curve per site, the sample dim being ``site``."""
    return make_canonical_field(("site", "time"))


@pytest.fixture
def field_member_site_time() -> xr.DataArray:
    """``(member, site, time)`` -- two sample dims at once."""
    return make_canonical_field(("member", "site", "time"))


@pytest.fixture
def field_member_site() -> xr.DataArray:
    """``(member, site)`` -- no ``time``, so no series panel can draw it."""
    return make_canonical_field(("member", "site"))


@pytest.fixture
def field_with_gaps() -> xr.DataArray:
    """``(member, time)`` with one timestep missing in every member.

    Timestep 5 is ``NaN`` for every member, so a fan's quantiles there are
    ``NaN`` and the band gaps; timestep 9 is ``NaN`` for the first member
    only, so the quantiles there are finite and taken over the rest.
    """
    field = make_canonical_field(("member", "time"))
    values = field.values.copy()
    values[:, 5] = np.nan
    values[0, 9] = np.nan
    return field.copy(data=values)


@pytest.fixture(scope="session")
def real_driver_field() -> xr.DataArray:
    """``air_temperature`` from the driver files present in this working copy.

    Sites 1 and 27 by members 1, 2 and 5. Only three of the six pairs have a
    file, so the field is half missing and ``driver_present`` says where.
    """
    drivers = pytest.importorskip("sipnet_calibration.drivers")
    try:
        dataset = drivers.load_drivers([1, 27], members=[1, 2, 5], allow_missing=True)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"driver files not available in this working copy: {error}")
    return drivers.driver_fields(dataset)["air_temperature"]


@pytest.fixture(scope="session")
def real_driver_presence() -> xr.DataArray:
    """``driver_present`` for the same request as :func:`real_driver_field`."""
    drivers = pytest.importorskip("sipnet_calibration.drivers")
    try:
        dataset = drivers.load_drivers([1, 27], members=[1, 2, 5], allow_missing=True)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"driver files not available in this working copy: {error}")
    return dataset[drivers.DRIVER_PRESENT]


@pytest.fixture(scope="session")
def real_constraint_fields() -> tuple[dict, dict]:
    """The annual constraint means and their error variances, as field dicts.

    Both are keyed on processed variable name, with dims ``(site, time)`` over
    the whole site pool and the annual snapshots, and are ragged: most cells
    are unobserved.
    """
    constraints = pytest.importorskip("sipnet_calibration.constraints")
    try:
        dataset = constraints.load_constraints()
    except FileNotFoundError as error:
        pytest.skip(f"constraint product not available in this working copy: {error}")
    return (
        constraints.constraint_fields(dataset),
        constraints.constraint_fields(dataset, statistic="variance"),
    )

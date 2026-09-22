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

The real-data fixtures read the driver files and the constraint products
present in this working copy, and skip when they are not there. The SIPNET
output fixtures read pySIPNET's own test fixtures out of the checkout beside
this one, since pySIPNET is installed from git and does not ship them; they
skip when there is no such checkout, and the one that runs the model skips
without a compiled binary too.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

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
    """The constraint observations and their error variances, as field dicts.

    Both are keyed on constraint name. The fields have dims ``(site, time)``
    over the whole site pool and each product's own time labels, or
    ``(site,)`` for the static soil carbon, and are ragged: most cells are
    unobserved. The variances are the squares of the reported standard
    deviations.
    """
    constraints = pytest.importorskip("sipnet_calibration.constraints")
    try:
        means = constraints.constraint_fields()
        sds = constraints.constraint_sds()
    except FileNotFoundError as error:
        pytest.skip(f"constraint products not available in this working copy: {error}")
    return means, {name: sd**2 for name, sd in sds.items()}


# ── real SIPNET output ────────────────────────────────────────────────────────


#: Inside a pySIPNET checkout: the independently-authored Niwot Ridge input set,
#: and real SIPNET output from running the standard model on its first rows.
NIWOT_REFERENCE = Path("tests/fixtures/niwot_reference")
NIWOT_GOLDEN = Path("tests/fixtures/golden/niwot_standard.out.csv")

#: The local driver file the 3-hourly tests run SIPNET on, if it is present.
SITE_1_DRIVERS = (
    Path(__file__).resolve().parents[1]
    / "data/raw/drivers/ERA5_1_1/ERA5.1.2012-01-01.2024-12-31.clim"
)

#: Whole days of it to run, at 8 steps per day.
SITE_1_DAYS = 8


def pysipnet_checkout() -> Path | None:
    """The pySIPNET source checkout, whose test fixtures are the only SIPNET inputs here.

    pySIPNET is installed from git, so its ``tests/fixtures`` are not on the
    Python path. ``$PYSIPNET_SOURCE`` names the checkout outright and is used
    alone when it is set, so a wrong value is reported rather than quietly
    replaced; otherwise the search walks up from here looking for a sibling
    clone. Candidates under ``.claude`` are skipped: a pySIPNET *worktree*
    parked beside this one is on whatever branch its session left it, where a
    sibling clone is the checkout the pin was taken from. Returns ``None`` when
    there is none, which is what the fixtures below skip on.
    """
    named = os.environ.get("PYSIPNET_SOURCE")
    if named:
        candidate = Path(named)
        return candidate if _has_niwot_reference(candidate) else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pySIPNET"
        if ".claude" in candidate.parts:
            continue
        if _has_niwot_reference(candidate):
            return candidate
    return None


def _has_niwot_reference(candidate: Path) -> bool:
    """Whether *candidate* is a pySIPNET checkout carrying the reference inputs."""
    return (candidate / NIWOT_REFERENCE / "sipnet.clim").is_file()


@pytest.fixture(scope="session")
def niwot_output():
    """Real SIPNET output for the Niwot Ridge fixture, as a ``SIPNETOutput``.

    pySIPNET's golden baseline: the standard model run on the first rows of the
    reference climate, committed in its repository, so this needs no binary.
    The climate's own step lengths come with it, which matters because Niwot's
    steps alternate between day and night and are not all the same length --
    the case a length-weighted mean exists for.
    """
    checkout = pysipnet_checkout()
    if checkout is None:
        pytest.skip("no pySIPNET checkout beside this one; set $PYSIPNET_SOURCE")
    golden = checkout / NIWOT_GOLDEN
    if not golden.is_file():
        pytest.skip(f"pySIPNET's golden output is not at {golden}")

    from pysipnet.io.clim_io import read_clim_file
    from pysipnet.output import SIPNETOutput

    frame = pd.read_csv(golden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate = read_clim_file(checkout / NIWOT_REFERENCE / "sipnet.clim")
    lengths = climate.pandas["time_step_length"].to_numpy()[: len(frame)]
    return SIPNETOutput.from_dataframe(frame, time_step_length=lengths, run_id="niwot-golden")


@pytest.fixture(scope="session")
def site_1_result(tmp_path_factory):
    """A real SIPNET run of the Niwot parameters on this copy's 3-hourly site-1 drivers.

    :data:`SITE_1_DAYS` whole days of ``ERA5_1_1``, which is the only 3-hourly
    input here and so the only one that can show a daily total being eight
    steps. Skipped where the driver file or the compiled binary is absent.
    """
    checkout = pysipnet_checkout()
    if checkout is None:
        pytest.skip("no pySIPNET checkout beside this one; set $PYSIPNET_SOURCE")
    if not SITE_1_DRIVERS.is_file():
        pytest.skip(f"site 1 drivers are not in this working copy ({SITE_1_DRIVERS})")

    from pysipnet.io.clim_io import read_clim_file
    from pysipnet.parameters.model import ModelFlags
    from pysipnet.runner import SIPNETRunner

    cache = _sipnet_cache_dir(checkout)
    if cache is None:
        pytest.skip("no compiled SIPNET binary; run 'make sipnet' in the pySIPNET checkout")

    # Session-scoped, because the SIPNETResult holds the climate it ran on and
    # so outlives the fixture; pytest removes the directory afterwards.
    climate_path = tmp_path_factory.mktemp("site-1-drivers") / "sipnet.clim"
    rows = SITE_1_DRIVERS.read_text().splitlines(keepends=True)[: 8 * SITE_1_DAYS]
    climate_path.write_text("".join(rows))
    with warnings.catch_warnings():
        # The site-1 record has exact zeros where SIPNET clamps, which pySIPNET
        # warns about on read; it is a property of the file, not of this run.
        # Only the read is silenced: a warning about the run itself is the sort
        # pySIPNET makes loud on purpose.
        warnings.simplefilter("ignore")
        climate = read_clim_file(climate_path)
    parameters = niwot_parameters(checkout)
    runner = SIPNETRunner(flags=ModelFlags.standard(), cache_dir=cache)
    return runner.run(parameters, climate, run_id="site-1")


def niwot_parameters(checkout: Path):
    """The reference ``sipnet.param`` as a ``SIPNETParameters``.

    A stand-in for the production reader pySIPNET has not written yet (its
    issue #19); built generically from the public name mapping so that a new
    parameter needs no change here. A parameter the file does not name keeps
    pySIPNET's own default, which is how the upstream fixture predating a
    submodel is read at all.
    """
    from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file
    from pysipnet.parameters.model import SIPNETParameters

    raw = read_param_file(checkout / NIWOT_REFERENCE / "sipnet.param")
    groups: dict[str, dict[str, float]] = {name: {} for name in SIPNETParameters.model_fields}
    for dotted, sipnet_name in PYTHON_TO_SIPNET.items():
        group, _, field = dotted.partition(".")
        if group in groups and sipnet_name in raw:
            groups[group][field] = raw[sipnet_name]
    return SIPNETParameters(
        **{
            name: SIPNETParameters.model_fields[name].annotation(**values)
            for name, values in groups.items()
        }
    )


def _sipnet_cache_dir(checkout: Path) -> Path | None:
    """Where a compiled SIPNET binary is, preferring the one this venv would use."""
    from pysipnet.runner import BINARY_NAME, SIPNETRunner

    for cache in (SIPNETRunner().cache_dir, checkout / ".sipnet_cache"):
        if (cache / BINARY_NAME).exists():
            return cache
    return None

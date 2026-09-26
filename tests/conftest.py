"""Shared test fixtures.

The matplotlib backend is set to ``Agg`` here, before anything imports
``pyplot``, so that the plotting tests never need a display and never open a
window. The plotting tests assert on artist data and properties --
``line.get_xydata()``, ``collection.get_paths()``, colors, labels, axis
limits -- and never on rendered images, which are brittle across matplotlib
versions and say nothing about why a test failed.

The synthetic fixtures build fields at each subset of the ``(sample, site,
time)`` dimensions (and of other batch dims), with a real ``DatetimeIndex`` on
``time``, ``lon``/``lat`` as non-dimension coordinates on ``site``, and
``units``/``long_name`` in ``attrs``.

The in-memory builders make what several test files need: a site table
(:func:`site_table_of`, and the :func:`site_table` fixture that hands it
out), a stack of Niwot runs (:func:`niwot_stack_of`), observed values that are
dated, static or attributed to windows (:func:`dated_observation`,
:func:`static_observation`, :func:`windowed_observation`), and a stand-in
SIPNET model (:class:`ScaledNiwot`). :func:`load_script` imports a script, and
every figure a test makes is closed after it (:func:`close_figures`).

The real-data fixtures read the driver files, the site table and the
constraint products present in this working copy, found through
:func:`sipnet_calibration.conventions.data_root`, and skip when they are not
there; a tracked input is found from :data:`REPOSITORY` instead, since it is
always in the checkout. The local driver files carry the drifting hour column
of ``data/README.md`` Note 15, which pySIPNET refuses, so the driver fixtures
read them with that one column rewritten to regular 3-hourly labels; every
value is the file's own. The SIPNET output fixtures read the Niwot reference
data pySIPNET ships inside the package, so they need neither a pySIPNET
checkout nor a binary; the one that runs the model skips without a binary,
which ``pysipnet install-sipnet`` provides.
"""

from __future__ import annotations

import functools
import importlib.util
import subprocess
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pysipnet.model import SIPNETModel  # noqa: E402
from pysipnet.runner import SIPNETRunError  # noqa: E402

from sipnet_calibration import conventions  # noqa: E402

#: The repository root, which the scripts are found under, since they are not
#: importable modules. Data is never found from here; it is under
#: :func:`sipnet_calibration.conventions.data_root`.
REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(path: str) -> ModuleType:
    """Import a script by its path from the repository root.

    Parameters
    ----------
    path:
        Such as ``"scripts/ingest_sites.py"``.

    Returns
    -------
    types.ModuleType
        The script as a module, registered in ``sys.modules`` under its file
        stem, so a dataclass or a pickle can find its classes.
    """
    location = REPOSITORY / path
    spec = importlib.util.spec_from_file_location(location.stem, location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[location.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test made, however it ends."""
    yield
    plt.close("all")


#: The variable the synthetic fields stand in for, with the units and long
#: name a real driver field carries from pySIPNET's climate registry.
SYNTHETIC_NAME = "air_temperature"
SYNTHETIC_ATTRS = {
    "units": "degC",
    "long_name": "Air temperature",
}

#: Site ids and their coordinates, matching the two sites whose driver files
#: are present locally, so that a synthetic field and a real one address the
#: same sites.
SYNTHETIC_SITES = (1, 27)
SYNTHETIC_LON = (-24.5625, -78.5625)
SYNTHETIC_LAT = (82.5458, 44.0654)


def make_field(
    dims: tuple[str, ...],
    *,
    n_time: int = 24,
    n_sample: int = 3,
    seed: int = 0,
) -> xr.DataArray:
    """A synthetic field with exactly *dims*, in the order fields are written.

    Parameters
    ----------
    dims:
        ``site`` and ``time``, and any batch dim names (``"sample"``,
        ``"initial_condition_member"``, ...), in any order; the result is
        transposed into ``(*batch, site, time)``, the batch dims in the order
        given.
    n_time:
        Length of the ``time`` dim, 3-hourly from 2012-01-01. Ignored when
        ``time`` is not in *dims*.
    n_sample:
        Length of every batch dim, labeled ``0`` to ``n_sample - 1``
        (``int64``).
    seed:
        Seed for the values, so a test can compare two calls.

    Returns
    -------
    xarray.DataArray
        Named :data:`SYNTHETIC_NAME`, carrying :data:`SYNTHETIC_ATTRS`, with
        ``lon``/``lat`` on ``site`` whenever ``site`` is present.
    """
    reserved = set(conventions.SPATIAL_DIM_NAMES) - {"site"}
    if reserved & set(dims):
        raise ValueError(f"not dims make_field builds: {sorted(reserved & set(dims))}")

    batch = tuple(d for d in dims if d not in ("site", "time"))
    sizes = {**dict.fromkeys(batch, n_sample), "site": len(SYNTHETIC_SITES), "time": n_time}
    order = (*batch, *(d for d in ("site", "time") if d in dims))
    shape = tuple(sizes[d] for d in order)

    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)

    coords: dict[str, object] = {}
    for dim in batch:
        coords[dim] = np.arange(n_sample, dtype=conventions.BATCH_LABEL_DTYPE)
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
    """A fresh ``Axes``; :func:`close_figures` closes its figure afterwards."""
    _, axes = plt.subplots()
    return axes


@pytest.fixture
def field_time() -> xr.DataArray:
    """``(time,)`` -- one deterministic run."""
    return make_field(("time",))


@pytest.fixture
def field_sample_time() -> xr.DataArray:
    """``(sample, time)`` -- a batch at one site."""
    return make_field(("sample", "time"))


@pytest.fixture
def field_site_time() -> xr.DataArray:
    """``(site, time)`` -- one curve per site, the sample dim being ``site``."""
    return make_field(("site", "time"))


@pytest.fixture
def field_sample_site_time() -> xr.DataArray:
    """``(sample, site, time)`` -- a batch over sites."""
    return make_field(("sample", "site", "time"))


@pytest.fixture
def field_sample_site() -> xr.DataArray:
    """``(sample, site)`` -- no ``time``, so no series panel can draw it."""
    return make_field(("sample", "site"))


@pytest.fixture
def field_with_gaps() -> xr.DataArray:
    """``(sample, time)`` with one timestep missing in every sample.

    Timestep 5 is ``NaN`` for every sample, so a fan's quantiles there are
    ``NaN`` and the band gaps; timestep 9 is ``NaN`` for the first sample
    only, so the quantiles there are finite and taken over the rest.
    """
    field = make_field(("sample", "time"))
    values = field.values.copy()
    values[:, 5] = np.nan
    values[0, 9] = np.nan
    return field.copy(data=values)


#: The local driver files, under the data root. Through
#: :func:`~sipnet_calibration.conventions.data_root`, so that a run pointed at
#: another tree with ``$SIPNET_CALIBRATION_DATA`` moves this with everything
#: else rather than half-relocating.
DRIVERS_ROOT = conventions.data_root() / "raw" / "drivers"

#: The ``(site, member)`` pairs whose driver files this working copy holds, and
#: the only ones the driver fixtures read. On the SCC the drivers root holds the
#: whole ensemble, which no test needs.
LOCAL_DRIVER_PAIRS = ((1, 1), (1, 2), (27, 5))


def with_regular_hour_column(text: str) -> str:
    """A ``.clim`` file's text with its hour column set to ``3 * slot``.

    For the local ERA5 files: 14 tab-separated fields a row, 3-hourly, eight
    rows to a day from hour 0 (``data/README.md`` open question 15), so the row
    at position ``k`` of a day is labeled ``3 * (k % 8)``. Only that column
    changes; every other field is kept as written. It fixes the drift of Note
    15 and nothing else: the accumulated columns still cover the step ending at
    the label (Note 16).
    """
    lines = []
    for k, line in enumerate(text.splitlines()):
        fields = line.split("\t")
        assert len(fields) == 14, f"row {k} has {len(fields)} tab-separated fields, not 14"
        assert int(float(fields[3]) // 3) == k % 8, f"row {k} is not in slot {k % 8}"
        fields[3] = f"{3 * (k % 8):9.6f}"
        lines.append("\t".join(fields))
    assert len(lines) % 8 == 0, "the file does not hold whole days of eight rows"
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def regular_drivers_root(tmp_path_factory) -> Path:
    """The local driver files, laid out as the originals, with regular hour labels.

    The file of each pair in :data:`LOCAL_DRIVER_PAIRS` under
    :data:`DRIVERS_ROOT` is copied through :func:`with_regular_hour_column`.
    Skipped when none of them is present.
    """
    files = []
    for site, member in LOCAL_DRIVER_PAIRS:
        files += sorted((DRIVERS_ROOT / f"ERA5_{site}_{member}").glob("ERA5.*.clim"))
    if not files:
        pytest.skip(f"no local driver files in this working copy under {DRIVERS_ROOT}")
    root = tmp_path_factory.mktemp("regular-drivers")
    for path in files:
        target = root / path.parent.name / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_text(with_regular_hour_column(path.read_text()))
    return root


@pytest.fixture(scope="session")
def real_drivers(regular_drivers_root, real_site_table) -> xr.Dataset:
    """The local drivers for sites 1 and 27 by source indices 1, 2 and 5.

    Only three of the six pairs have a file, so the Dataset is half missing
    and ``driver_present`` says where.
    """
    from sipnet_calibration import drivers

    with warnings.catch_warnings():
        # The files hold exact zeros of vpd where SIPNET clamps, which pySIPNET
        # warns about on read; a property of the files, not of any test.
        warnings.simplefilter("ignore")
        return drivers.load_drivers(
            [1, 27],
            source_indices=[1, 2, 5],
            root=regular_drivers_root,
            site_table=real_site_table,
            allow_missing=True,
        )


@pytest.fixture(scope="session")
def real_driver_field(real_drivers) -> xr.DataArray:
    """``air_temperature`` from :func:`real_drivers`."""
    from sipnet_calibration.drivers import driver_fields

    return driver_fields(real_drivers)["air_temperature"]


@pytest.fixture(scope="session")
def real_driver_presence(real_drivers) -> xr.DataArray:
    """``driver_present`` for the same request as :func:`real_driver_field`."""
    from sipnet_calibration.drivers import DRIVER_PRESENT

    return real_drivers[DRIVER_PRESENT]


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
        standard_deviations = constraints.constraint_standard_deviations()
    except FileNotFoundError as error:
        pytest.skip(f"constraint products not available in this working copy: {error}")
    return means, {
        name: standard_deviation**2 for name, standard_deviation in standard_deviations.items()
    }


# ── real SIPNET output ────────────────────────────────────────────────────────


#: The local driver file the 3-hourly tests run SIPNET on, relative to a
#: drivers root.
SITE_1_DRIVERS = "ERA5_1_1/ERA5.1.2012-01-01.2024-12-31.clim"

#: Whole days of it to run, at 8 steps per day.
SITE_1_DAYS = 8


@pytest.fixture(scope="session")
def real_site_table():
    """The real site table, or a skip when the ingest has not been run here.

    Anything that labels a field with a ``site`` reaches for this, directly or
    through :func:`~sipnet_calibration.fields.stack_sipnet_outputs`, so the
    guard belongs in one place rather than in each module that happens to.
    """
    from sipnet_calibration.sites import load_sites

    try:
        return load_sites()
    except FileNotFoundError as error:
        pytest.skip(f"site table not available in this working copy: {error}")


@pytest.fixture(scope="session")
def niwot_output():
    """Real SIPNET output for the Niwot Ridge reference inputs, as a ``SIPNETOutput``.

    pySIPNET's golden baseline, shipped inside the package since its PR #40:
    the standard model run on the first rows of the reference climate, paired
    with that climate's own step lengths. No binary and no pySIPNET checkout
    are needed. The step lengths matter because Niwot's steps alternate between
    day and night and are not all the same length -- the case a length-weighted
    mean exists for.

    It carries ``ModelFlags.standard()``, so selecting a variable SIPNET wrote
    as constant zero under those flags -- the nitrogen group, ``litter_carbon``,
    ``methane_production`` -- is refused rather than handed back as zeros. A
    test that wants one of those needs its own output.
    """
    from pysipnet import niwot_reference_output

    return niwot_reference_output()


@pytest.fixture(scope="session")
def site_1_result(regular_drivers_root):
    """A real SIPNET run of the Niwot parameters on this copy's 3-hourly site-1 drivers.

    :data:`SITE_1_DAYS` whole days of ``ERA5_1_1`` from
    :func:`regular_drivers_root`, which is the only 3-hourly input here and so
    the only one that can show a daily total being eight steps. Skipped where
    the driver file or a SIPNET binary is absent; the binary is whatever
    :func:`pysipnet.build.find_binary` resolves, so ``pysipnet install-sipnet``
    is what makes this run.
    """
    from pysipnet.build import find_binary, missing_binary_message
    from pysipnet.climate import ClimateDrivers
    from pysipnet.parameters.model import ModelFlags
    from pysipnet.runner import SIPNETRunner

    site_1_drivers = regular_drivers_root / SITE_1_DRIVERS
    if not site_1_drivers.is_file():
        pytest.skip(
            f"site 1 drivers are not in this working copy ({SITE_1_DRIVERS}); "
            "copy or link the ERA5_1_1 directory from the SCC"
        )
    if find_binary() is None:
        pytest.skip(missing_binary_message())

    with warnings.catch_warnings():
        # The site-1 record has exact zeros where SIPNET clamps, which pySIPNET
        # warns about on read; it is a property of the file, not of this run.
        # Only the read is silenced: a warning about the run itself is the sort
        # pySIPNET makes loud on purpose.
        warnings.simplefilter("ignore")
        climate = ClimateDrivers.from_file(site_1_drivers).head(8 * SITE_1_DAYS)
    return SIPNETRunner(flags=ModelFlags.standard()).run(
        niwot_parameters(), climate, run_id="site-1"
    )


def niwot_parameters():
    """The reference ``sipnet.param`` as a ``SIPNETParameters``.

    A stand-in for the production reader pySIPNET has not written yet (its
    issue #19); built generically from the public name mapping so that a new
    parameter needs no change here. A parameter the file does not name keeps
    pySIPNET's own default, which is how the upstream fixture predating a
    submodel is read at all.
    """
    from pysipnet import niwot_reference_files
    from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file
    from pysipnet.parameters.model import SIPNETParameters

    raw = read_param_file(niwot_reference_files().param)
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


# ── in-memory site tables ─────────────────────────────────────────────────────


def site_table_of(
    *site_ids: int,
    lon: float | Sequence[float] | None = None,
    lat: float | Sequence[float] | None = None,
    keyed: bool = False,
) -> pd.DataFrame:
    """A site table holding only what a lookup reads: ``site_id``, ``lon``, ``lat``.

    Parameters
    ----------
    site_ids:
        The sites, in the table's row order.
    lon, lat:
        One value for every site, or one per site. By default each site gets
        its own, ``-100 - site / 100`` and ``40 + site / 100``, so two sites
        can be told apart by their coordinates.
    keyed:
        Whether to key the table on ``site_id``, as
        :func:`sipnet_calibration.sites.site_lookup` does.
    """
    ids = np.asarray(site_ids, dtype=conventions.SITE_DTYPE)
    lon = -100.0 - ids / 100 if lon is None else np.broadcast_to(np.asarray(lon, float), ids.shape)
    lat = 40.0 + ids / 100 if lat is None else np.broadcast_to(np.asarray(lat, float), ids.shape)
    table = pd.DataFrame({conventions.SITE_ID: ids, conventions.LON: lon, conventions.LAT: lat})
    return table.set_index(conventions.SITE_ID, drop=False) if keyed else table


def write_site_table_csv(
    path: Path,
    site_ids: Sequence[int],
    *,
    lon: Sequence[float],
    lat: Sequence[float],
    landcover: Sequence[int] | None = None,
) -> Path:
    """Write a site table with every column of the schema, as the ingest does.

    Parameters
    ----------
    path:
        Where to write it; its directory is created.
    site_ids, lon, lat:
        The sites and their coordinates.
    landcover:
        The landcover class of each site; class 1 for every site by default.

    Returns
    -------
    pathlib.Path
        *path*, holding a table :func:`sipnet_calibration.sites.load_sites`
        accepts.
    """
    from sipnet_calibration.sites import SITE_COLUMNS

    n = len(site_ids)
    frame = pd.DataFrame(
        {
            conventions.SITE_ID: np.array(site_ids, dtype=conventions.SITE_DTYPE),
            conventions.LON: list(lon),
            conventions.LAT: list(lat),
            "lon_index": np.arange(n, dtype=np.int32) + 1000,
            "lat_index": np.arange(n, dtype=np.int32) + 2000,
            "site_name": [f"site {site}" for site in site_ids],
            "site_order": np.zeros(n, dtype=np.int32),
            "cluster": np.ones(n, dtype=np.int8),
            "landcover": np.ones(n, dtype=np.int8)
            if landcover is None
            else np.array(landcover, dtype=np.int8),
            "ameriflux_site_id": [""] * n,
        }
    )
    assert tuple(frame.columns) == SITE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# ── Niwot runs ────────────────────────────────────────────────────────────────


@functools.cache
def niwot_reference():
    """pySIPNET's Niwot reference output, read once."""
    from pysipnet import niwot_reference_output

    return niwot_reference_output()


def niwot_stack_of(
    output_variable_names: Sequence[str],
    *,
    sites: Sequence[int] = (1, 2),
    n_samples: int = 2,
    lengths: dict[int, int] | None = None,
) -> xr.Dataset:
    """A stack of Niwot runs on ``(sample, site, time)``, told apart by known factors.

    Parameters
    ----------
    output_variable_names:
        The variables each run carries.
    sites:
        The site of each run; the site at position ``i`` is the Niwot output
        times ``1 + i / 2``.
    n_samples:
        How many samples; sample ``j`` is the site's run times ``0.5 ** j``.
    lengths:
        The number of timesteps of a site's record, by site, where it is
        shorter than the Niwot record; the stack pads it with ``NaN``.

    Returns
    -------
    xarray.Dataset
        What :func:`sipnet_calibration.fields.stack_model_outputs` makes of
        the runs, with ``lon``/``lat`` from :func:`site_table_of`.
    """
    from sipnet_calibration.fields import stack_model_outputs

    base = niwot_reference().select(list(output_variable_names))
    lengths = lengths or {}
    runs = {}
    for position, site in enumerate(sites):
        record = base.isel(time=slice(0, lengths[site])) if site in lengths else base
        for sample in range(n_samples):
            factor = (1 + position / 2) * 0.5**sample
            runs[(sample, site)] = record.map(_scaled_keeping_attributes, factor=factor)
    return stack_model_outputs(runs, site_table=site_table_of(*sites))


def _scaled_keeping_attributes(variable: xr.DataArray, *, factor: float) -> xr.DataArray:
    scaled = variable * factor
    scaled.attrs = dict(variable.attrs)
    return scaled


# ── observed values ───────────────────────────────────────────────────────────


def dated_observation(
    sites: Sequence[int],
    times: Sequence,
    *,
    values: np.ndarray | None = None,
    units: str = "m2 m-2",
    constituent: str = "",
    name: str = "modis_leaf_area_index",
) -> xr.DataArray:
    """Observed values on ``(site, time)``: ones, unless *values* are given."""
    times = pd.DatetimeIndex(times)
    data = np.ones((len(sites), len(times))) if values is None else np.asarray(values, float)
    return xr.DataArray(
        data,
        dims=(conventions.SITE, conventions.TIME),
        coords={conventions.SITE: list(sites), conventions.TIME: times},
        attrs=_observation_attributes(units, constituent),
        name=name,
    )


def static_observation(
    sites: Sequence[int],
    *,
    values: np.ndarray | None = None,
    units: str = "Mg ha-1",
    constituent: str = "C",
    name: str = "soilgrids_soil_organic_carbon",
) -> xr.DataArray:
    """Observed values on ``(site,)``, with no time: ones, unless *values* are given."""
    data = np.ones(len(sites)) if values is None else np.asarray(values, float)
    return xr.DataArray(
        data,
        dims=conventions.SITE,
        coords={conventions.SITE: list(sites)},
        attrs=_observation_attributes(units, constituent),
        name=name,
    )


def windowed_observation(
    sites: Sequence[int],
    times: Sequence,
    *,
    window_length: str = "1D",
    **keywords,
) -> xr.DataArray:
    """:func:`dated_observation`, each value attributed to the window ending at its label.

    The window edges are the coordinates
    :data:`sipnet_calibration.conventions.WINDOW_START` and
    :data:`~sipnet_calibration.conventions.WINDOW_END` on ``time``, as the
    constraints' reader writes them; each window is *window_length* long.
    """
    observed = dated_observation(sites, times, **keywords)
    ends = pd.DatetimeIndex(observed[conventions.TIME].values)
    return observed.assign_coords(
        {
            conventions.WINDOW_START: (conventions.TIME, ends - pd.Timedelta(window_length)),
            conventions.WINDOW_END: (conventions.TIME, ends),
        }
    )


def _observation_attributes(units: str, constituent: str) -> dict[str, str]:
    attrs = {"units": units}
    if constituent:
        attrs["constituent"] = constituent
    return attrs


# ── a stand-in SIPNET model ───────────────────────────────────────────────────

#: The ``soil_carbon`` at which :class:`ScaledNiwot` leaves wood carbon as the
#: Niwot output has it.
SOIL_REFERENCE = 1.0e4

#: :class:`ScaledNiwot` "fails at its parameters" past this rate, writes NaN
#: in a band above it, times out in a band above that, has its parameters
#: refused by pydantic at or below :data:`INVALID`, and "fails in the
#: machinery" between :data:`INVALID` and zero.
BLOW_UP = 1e6
NAN_BAND = 2e6
TIMEOUT_BAND = 3e6
INVALID = -BLOW_UP


class _PositiveRate(BaseModel):
    """Stands in for pySIPNET's validation of a parameter's domain."""

    rate: float = Field(gt=0)


class ScaledNiwot(SIPNETModel):
    """A SIPNETModel whose run is the Niwot output scaled by two parameters.

    ``wood_carbon`` is multiplied by ``max_photosynthesis_rate / 10`` (which
    the example vector shares across sites) and by ``soil_carbon /
    SOIL_REFERENCE`` (which it varies by site), so which parameter values
    reached which run can be read off the result, site by site. The run is as
    long as its drivers, so which drivers reached which run shows too. The
    rate also selects a failure, by the bands of :data:`BLOW_UP`. Defined at
    module level so PyEns can pickle it.
    """

    def __call__(self, *, climate=None, events=None, **overrides):
        from pysipnet.output import SIPNETOutput

        rate = float(overrides["max_photosynthesis_rate"])
        if rate <= INVALID:
            _PositiveRate(rate=rate)
        if rate < 0:
            raise RuntimeError("the node died")
        if rate > TIMEOUT_BAND:
            raise subprocess.TimeoutExpired(cmd="sipnet", timeout=0.001)
        if rate > BLOW_UP and rate <= NAN_BAND:
            raise SIPNETRunError(
                "SIPNET blew up", returncode=1, stdout="", stderr="", workdir=Path("/tmp")
            )
        n = climate.n_timesteps
        frame = niwot_reference().pandas.iloc[:n].copy()
        frame["wood_carbon"] = (
            frame["wood_carbon"]
            * (rate / 10.0)
            * (float(overrides["soil_carbon"]) / SOIL_REFERENCE)
        )
        if rate > NAN_BAND:
            frame.loc[frame.index[-5:], "wood_carbon"] = np.nan
        return SimpleNamespace(outputs=SIPNETOutput.from_dataframe(frame, climate=climate))


def scaled_niwot_model(model_class: type[ScaledNiwot] = ScaledNiwot) -> ScaledNiwot:
    """*model_class* over the Niwot parameters, on a runner that needs no binary."""
    from pysipnet.parameters.model import ModelFlags
    from pysipnet.runner import SIPNETRunner

    return model_class(
        SIPNETRunner(flags=ModelFlags.standard(), verify_binary=False),
        base_params=niwot_parameters(),
    )

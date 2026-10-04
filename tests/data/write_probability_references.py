"""Write the references the probability layer is held to, from today's code.

Overview
--------
Five priors of today's parameter layer are sampled and evaluated at fixed
thetas, and today's ``ForwardModel`` is evaluated on ``test_forward``'s
stand-in SIPNET at fixed thetas; the results are stored, one file per case.
The probability layer that replaces ``sipnet_calibration.parameters`` must
reproduce them bit for bit: its prior draws and densities (refactor PR P3)
and its predictions, `ForwardModel.evaluate` (P5). ``tests/test_probability_references.py`` checks
that this script reproduces its files byte for byte.

Input data
----------
Tracked only, so the files can be made in any checkout:

- the initial-condition ensemble, ``data/raw/initial_conditions/``, read
  through ``initial_conditions.read_raw``, for the two priors PR #69 fits
  to site 4977's members;
- the Niwot reference output and parameters pySIPNET ships, which the
  stand-in SIPNET of ``tests/conftest.py`` scales.

Output data
-----------
``probability_references/<case>.nc`` beside this script. A prior case holds:

- ``draws`` on ``(draw, entry)``: ``prior.sample(jax.random.key(seed),
  n_draws)``, with ``seed`` and ``n_draws`` the dataset's attributes;
- ``theta`` on ``(sample, entry)``: the draws, then theta = 0, then
  ``n_standard`` draws of N(0, I) from ``numpy.random.default_rng(seed)``;
- ``log_prob`` on ``sample``: ``prior.log_prob(theta)``;
- ``natural:<parameter>`` and, where the prior has derived parameters,
  ``derived:<derived parameter>``: the labeled natural values at theta;
- ``constant:<name>``: what the case was built from that a test cannot
  rebuild without data, such as the members a prior is fitted to.

Theta's entries are labeled by ``entry``, today's unconstrained entry
names, with the levels of today's unconstrained index beside it as string
coordinates ``entry_<level>``, empty where a level does not apply.

A forward case holds ``theta`` on ``(sample, entry)`` (prior draws, then
theta = 0), ``predictions`` on ``(sample, observation)`` and ``valid`` on
``sample``, with the observation vector's index levels on ``observation`` as
the coordinates ``observation_site``, ``observation_source`` and
``observation_time``.

Notes
-----
The cases are those of the design's PR plan for P1:

- ``single_site_mcmc_vs_eki``: PR #69's prior, transcribed from
  ``experiments/single_site_mcmc_vs_eki/model/prior.py`` at 7949640 (its
  thirteen parameters, D = 15), its two initial-carbon priors fitted to site
  4977's members as the experiment fits them;
- ``example_calibration``: ``calibration.example_calibration``'s prior at
  sites 1, 27 and 4711, PFTs deciduous, conifer, deciduous;
- ``hierarchy``: a centered partial pooling of a site-level quantity within
  PFTs, given a derived parameter (each site's PFT mean, through a
  membership) and a shared spread, beside site offsets given the spread;
- ``copula``: a Gaussian copula over three scalar parameters;
- ``per_pft_simplex``: an allocation simplex on ``pft``, each PFT about its
  own center, a constant on ``(pft, allocation_part)``;
- ``forward_example``: ``test_forward``'s setup, the example calibration at
  sites 1 and 27 on the stand-in, site 27's drivers cut short, and two
  observation sources.

The files are byte-reproducible because h5netcdf writes no timestamps and
every value comes from code that is deterministic on one machine and one
``uv.lock``. A change of platform, of JAX or TFP, or of pySIPNET's Niwot
reference data can change the last bits; the byte test then fails, and the
files are rewritten only from the code as it was before the refactor
changed it.

Usage
-----
::

    uv run python tests/data/write_probability_references.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from pyens import SequentialBackend
from pysipnet import niwot_reference_output
from tensorflow_probability.substrates import jax as tfp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # conftest's stand-in SIPNET, when run as a script

from conftest import located, scaled_niwot_model, site_table_of  # noqa: E402

from sipnet_calibration import io  # noqa: E402
from sipnet_calibration.calibration import example_calibration  # noqa: E402
from sipnet_calibration.forward import ForwardModel  # noqa: E402
from sipnet_calibration.initial_conditions import read_raw, resolve_initial_condition  # noqa: E402
from sipnet_calibration.observation import (  # noqa: E402
    DEFAULT_OBS_OPS,
    ObservationSource,
    ObservationVector,
    SelectTimestep,
)
from sipnet_calibration.parameters import (  # noqa: E402
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    DerivedParameter,
    DerivedParameters,
    Parameter,
    ParameterVector,
    Prior,
    PriorTerm,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    softmax_normal,
)
from sipnet_calibration.site_dims import SiteDims  # noqa: E402

tfd, tfb = tfp.distributions, tfp.bijectors

#: Where the files are written.
DIRECTORY = HERE / "probability_references"

#: The seed of every draw: the prior's key, and NumPy's generator.
SEED = 20261004

#: Prior draws, and draws of N(0, I), per prior case.
N_DRAWS, N_STANDARD = 8, 2

#: PR #69's site, and the initial conditions its two fitted priors read.
SINGLE_SITE = 4977
FITTED_INITIAL_CONDITION_NAMES = ("initial_wood_carbon", "initial_soil_organic_carbon")

#: The allocation simplex's element axis and its labels.
ALLOCATION_PART = "allocation_part"
ALLOCATION_PARTS = ("leaf", "wood", "fine_root", "coarse_root")

#: The prior cases' sites, their PFTs, and the PFTs in label order.
SITES = (1, 27, 4711)
PFT_OF_SITE = ("deciduous", "conifer", "deciduous")
PFT = ("conifer", "deciduous")

#: The forward case's sites, PFTs and locations, as ``test_forward`` has them.
FORWARD_SITES = (1, 27)
FORWARD_PFT = ("temperate.deciduous", "boreal.coniferous")
FORWARD_LON, FORWARD_LAT = (-105.0, -70.0), (40.0, 45.0)

#: How many steps of the Niwot drivers site 27 runs on, as in ``test_forward``.
SHORT_STEPS = 40

#: The prior draws the forward case runs, before theta = 0.
N_FORWARD_DRAWS = 4

#: Every reference term's provenance.
PROVENANCE = "Probability-layer reference."


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    try:
        paths = write_references(DIRECTORY)
    except (KeyError, ValueError) as error:
        sys.exit(f"write_probability_references: {error}")
    for path in paths:
        print(f"wrote {path.relative_to(HERE)}")


def write_references(directory: Path) -> list[Path]:
    """Every case's file in *directory*, written and checked together."""
    datasets = {name: build() for name, build in CASES.items()}
    return io.write_checked_together(
        (directory / f"{name}.nc", _writer(dataset), _checker(dataset)) for name, dataset in datasets.items()
    )


# ── the cases ─────────────────────────────────────────────────────────────────


def single_site_mcmc_vs_eki() -> xr.Dataset:
    """PR #69's prior at 7949640, its two initial carbons fitted to site 4977's members."""
    members = initial_condition_members(SINGLE_SITE, FITTED_INITIAL_CONDITION_NAMES)
    vector = ParameterVector(parameters=[
        Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
        Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
        Parameter(name="optimum_photosynthesis_temperature", support=REAL, units="degC"),
        Parameter(name="half_saturation_light", support=POSITIVE, units="mol m-2 d-1"),
        Parameter(name="soil_water_holding_capacity", support=POSITIVE, units="cm"),
        Parameter(name="leaf_on_growth", support=POSITIVE, units="g m-2"),
        Parameter(name="leaf_on_growing_degree_days", support=POSITIVE, units="K d"),
        Parameter(name="allocation", support=SIMPLEX, units="1", shape=(len(ALLOCATION_PARTS),),
                  element_labels={ALLOCATION_PART: ALLOCATION_PARTS}),
        Parameter(name="wood_respiration_rate_at_10c", support=POSITIVE, units="yr-1"),
        Parameter(name="soil_respiration_flux_at_10c", support=POSITIVE, units="g m-2 yr-1"),
        Parameter(name="soil_respiration_q10", support=POSITIVE, units="1"),
        Parameter(name="initial_wood_carbon", support=POSITIVE, units="kg m-2"),
        Parameter(name="initial_soil_organic_carbon", support=POSITIVE, units="kg m-2"),
    ])
    prior = Prior(vector, [
        _term("photosynthetic_capacity", log_normal_from_interval(lower=140.0, upper=450.0)),
        _term("respiration_share", logit_normal_from_interval(lower=0.04, upper=0.20)),
        _term("optimum_photosynthesis_temperature", tfd.Normal(jnp.float64(22.0), jnp.float64(2.5))),
        _term("half_saturation_light", log_normal_from_interval(lower=4.6, upper=26.3)),
        _term("soil_water_holding_capacity", log_normal_from_interval(lower=15.0, upper=150.0)),
        _term("leaf_on_growth", log_normal_from_interval(lower=50.0, upper=180.0)),
        _term("leaf_on_growing_degree_days", log_normal_from_interval(lower=500.0, upper=1100.0)),
        _term("allocation", softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]),
                                           logit_sd=jnp.array([0.25, 0.30, 0.30]))),
        _term("wood_respiration_rate_at_10c", log_normal_from_interval(lower=0.006, upper=0.04)),
        _term("soil_respiration_flux_at_10c", log_normal_from_interval(lower=200.0, upper=900.0)),
        _term("soil_respiration_q10", log_normal_from_interval(lower=1.3, upper=3.2)),
        *(_term(name, log_normal_from_samples(members[name].values)) for name in FITTED_INITIAL_CONDITION_NAMES),
    ])
    return prior_reference(prior, constants={f"{name}_members": members[name] for name in members})


def example_calibration_prior() -> xr.Dataset:
    """``calibration.example_calibration``'s prior at :data:`SITES`."""
    site_dims = SiteDims(site_table=site_table_of(*SITES), site_labels={"pft": PFT_OF_SITE})
    _, prior, _ = example_calibration(site_dims)
    return prior_reference(prior)


def hierarchy() -> xr.Dataset:
    """A centered partial pooling within PFTs, given a derived location, beside offsets given a spread."""
    vector = ParameterVector(
        parameters=[
            Parameter(name="pft_log_mean", support=REAL, units="1", indexed_by=("pft",)),
            Parameter(name="spread", support=POSITIVE, units="1"),
            Parameter(name="soil_carbon", support=POSITIVE, units="g m-2", indexed_by=("site",)),
            Parameter(name="offsets", support=REAL, units="1", indexed_by=("site",)),
        ],
        coords={"pft": PFT, "site": np.asarray(SITES, dtype=np.int32)},
    )
    derived = DerivedParameters(parameter_vector=vector, derived_parameters=[DerivedParameter(
        name="site_log_mean", units="1", indexed_by=("site",), given=("pft_log_mean",),
        memberships={"pft_of_site": _pft_of_site()}, function=_site_log_mean,
    )])
    prior = Prior(vector, [
        _term("pft_log_mean", iid_over_dim(tfd.Normal(jnp.float64(np.log(1e4)), jnp.float64(1.0)))),
        _term("spread", log_normal(median=0.5, geometric_sd=2.0)),
        _term("soil_carbon", _soil_carbon_given_its_mean, given=("site_log_mean", "spread")),
        _term("offsets", iid_over_dim(_offset_given_the_spread), given=("spread",)),
    ], derived_parameters=derived)
    return prior_reference(prior)


def copula() -> xr.Dataset:
    """A Gaussian copula over a log-normal and two logit-normals."""
    vector = ParameterVector(parameters=[
        Parameter(name="photosynthetic_capacity", support=POSITIVE, units="nmol g-1 s-1"),
        Parameter(name="respiration_share", support=OPEN_UNIT_INTERVAL, units="1"),
        Parameter(name="leaf_fall_fraction", support=OPEN_UNIT_INTERVAL, units="1"),
    ])
    marginals = {
        "photosynthetic_capacity": log_normal(median=250.0, geometric_sd=1.6),
        "respiration_share": logit_normal(median=0.18, logit_sd=0.35),
        "leaf_fall_fraction": logit_normal(median=0.5, logit_sd=1.0),
    }
    correlation = [[1.0, -0.4, 0.2], [-0.4, 1.0, 0.3], [0.2, 0.3, 1.0]]
    prior = Prior(vector, [PriorTerm(
        parameter_names=tuple(marginals), distribution=gaussian_copula(marginals, correlation=correlation),
        provenance=PROVENANCE,
    )])
    return prior_reference(prior)


def per_pft_simplex() -> xr.Dataset:
    """An allocation simplex on ``pft``, each PFT about its own center."""
    vector = ParameterVector(
        parameters=[Parameter(name="allocation", support=SIMPLEX, units="1", shape=(len(ALLOCATION_PARTS),),
                              indexed_by=("pft",), element_labels={ALLOCATION_PART: ALLOCATION_PARTS})],
        coords={"pft": PFT},
    )
    center = xr.DataArray(
        [[0.25, 0.35, 0.10, 0.30], [0.18, 0.45, 0.065, 0.305]],
        dims=("pft", ALLOCATION_PART), coords={"pft": list(PFT), ALLOCATION_PART: list(ALLOCATION_PARTS)},
    )
    prior = Prior(vector, [_term(
        "allocation", independent_over_dim(softmax_normal, logit_sd=jnp.array([0.25, 0.30, 0.30])),
        constants={"center": center},
    )])
    return prior_reference(prior, constants={"center": center})


def forward_example() -> xr.Dataset:
    """``test_forward``'s forward model: the example calibration at two sites, on the stand-in."""
    site_table = site_table_of(*FORWARD_SITES, lon=list(FORWARD_LON), lat=list(FORWARD_LAT))
    site_dims = SiteDims(site_table=site_table, site_labels={"pft": FORWARD_PFT})
    vector, prior, sipnet_map = example_calibration(site_dims)
    reference = niwot_reference_output()
    forward = ForwardModel(
        scaled_niwot_model(), vector, sipnet_map, site_dims=site_dims,
        climate={1: reference.climate, 27: reference.climate.head(SHORT_STEPS)},
        backend=SequentialBackend(), observation_vector=forward_observation_vector(site_table),
    )
    theta = np.concatenate([
        np.asarray(prior.sample(jax.random.key(SEED), N_FORWARD_DRAWS)),
        np.zeros((1, vector.unconstrained.size)),
    ])
    evaluation = forward.evaluate(theta)
    index = forward.observation_vector.index
    dataset = xr.Dataset(
        {
            "theta": (("sample", "entry"), theta),
            "predictions": (("sample", "observation"), np.asarray(evaluation.predictions)),
            "valid": ("sample", np.asarray(evaluation.valid)),
        },
        coords={
            **_entry_coordinates(vector.unconstrained),
            "observation_site": ("observation", index.get_level_values("site").to_numpy(np.int32)),
            "observation_source": ("observation", index.get_level_values("observation_source").astype(str).to_numpy()),
            "observation_time": ("observation", index.get_level_values("time").to_numpy("datetime64[ns]")),
        },
    )
    dataset.attrs = {"seed": SEED, "n_draws": N_FORWARD_DRAWS}
    return dataset


#: The cases, by file stem. After the functions, which it names.
CASES: Mapping[str, Callable[[], xr.Dataset]] = {
    "single_site_mcmc_vs_eki": single_site_mcmc_vs_eki,
    "example_calibration": example_calibration_prior,
    "hierarchy": hierarchy,
    "copula": copula,
    "per_pft_simplex": per_pft_simplex,
    "forward_example": forward_example,
}


# ── what a case is built from and stores ──────────────────────────────────────


def prior_reference(prior: Prior, *, constants: Mapping[str, xr.DataArray] | None = None) -> xr.Dataset:
    """A prior's draws, and its log density and natural values at the reference thetas."""
    vector = prior.parameter_vector
    size = vector.unconstrained.size
    draws = np.asarray(prior.sample(jax.random.key(SEED), N_DRAWS))
    standard = np.random.default_rng(SEED).standard_normal((N_STANDARD, size))
    theta = np.concatenate([draws, np.zeros((1, size)), standard])
    values_by_parameter = vector.flat_to_values(vector.to_natural(theta))
    natural = vector.values_to_dataset(values_by_parameter, batch_dims=("sample",))
    variables = {
        "draws": (("draw", "entry"), draws),
        "theta": (("sample", "entry"), theta),
        "log_prob": ("sample", np.asarray(prior.log_prob(theta))),
        **{f"natural:{name}": natural[name] for name in natural.data_vars},
    }
    if prior.derived_parameters is not None:
        derived = prior.derived_parameters.values_to_dataset(
            prior.derived_parameters.values(values_by_parameter), batch_dims=("sample",)
        )
        variables |= {f"derived:{name}": derived[name] for name in derived.data_vars}
    variables |= {f"constant:{name}": value for name, value in (constants or {}).items()}
    dataset = xr.Dataset(variables, coords=_entry_coordinates(vector.unconstrained))
    for variable in dataset.variables.values():
        variable.attrs = {}
    dataset.attrs = {"seed": SEED, "n_draws": N_DRAWS, "n_standard": N_STANDARD}
    return dataset


def initial_condition_members(site: int, names: tuple[str, ...]) -> dict[str, xr.DataArray]:
    """Each initial condition's members at *site*, in member order, from the tracked raw file.

    The processed file's values are the raw file's, transposed, so these are
    the members ``initial_condition_fields(sites=[site])`` holds, which PR #69
    fits to.
    """
    raw = read_raw()
    return {
        name: xr.DataArray(
            raw[resolve_initial_condition(name).source_name].sel(site=site).values.astype(np.float64),
            dims="initial_condition_member",
        )
        for name in names
    }


def forward_observation_vector(site_table: pd.DataFrame) -> ObservationVector:
    """``test_forward``'s two observation sources, at three of the Niwot record's time labels."""
    wood = niwot_reference_output().select(["wood_carbon"])["wood_carbon"]
    labels = pd.DatetimeIndex(wood["time"].values[[5, 20, 30]])

    def observed(name: str, values: list[list[float]], attrs: dict[str, str]) -> xr.DataArray:
        array = xr.DataArray(values, dims=("site", "time"), coords={"site": list(FORWARD_SITES), "time": labels},
                             attrs=attrs, name=name)
        return located(array, site_table=site_table)

    biomass = observed("landtrendr_aboveground_biomass", [[100.0, np.nan, 120.0], [110.0, 115.0, np.nan]],
                       {"units": "Mg ha-1", "constituent": "C"})
    lai = observed("modis_leaf_area_index", [[3.0, 2.0, np.nan], [np.nan, 1.0, 1.5]], {"units": "m2 m-2"})
    return ObservationVector(observation_sources=[
        ObservationSource(observation_source_name="landtrendr_aboveground_biomass", observed_values=biomass,
                          operator=SelectTimestep("wood_carbon")),
        ObservationSource(observation_source_name="modis_leaf_area_index", observed_values=lai,
                          operator=DEFAULT_OBS_OPS["modis_leaf_area_index"]),
    ])


# ── helpers ───────────────────────────────────────────────────────────────────


def _term(name: str, distribution, **arguments) -> PriorTerm:
    """A term over the one parameter *name*."""
    return PriorTerm(parameter_names=(name,), distribution=distribution, provenance=PROVENANCE, **arguments)


def _pft_of_site() -> xr.DataArray:
    """Each site's PFT, the membership the hierarchy's location reads."""
    return xr.DataArray(list(PFT_OF_SITE), dims="site", coords={"site": np.asarray(SITES, dtype=np.int32)},
                        name="pft")


def _site_log_mean(pft_log_mean, pft_of_site):
    return pft_log_mean[..., pft_of_site]


def _soil_carbon_given_its_mean(site_log_mean, spread):
    return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(site_log_mean, spread), 1), tfb.Exp())


def _offset_given_the_spread(spread):
    return tfd.Normal(jnp.float64(0.0), spread)


def _entry_coordinates(unconstrained: ParameterVector) -> dict:
    """Theta's entry names, and its index's levels as strings, empty where a level does not apply."""
    index = unconstrained.index
    levels = {
        f"entry_{level}": ("entry", np.array(["" if pd.isna(v) else str(v) for v in index.get_level_values(level)]))
        for level in index.names
    }
    return {"entry": list(unconstrained.entry_names), **levels}


def _writer(dataset: xr.Dataset) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        dataset.to_netcdf(path, engine="h5netcdf")

    return write


def _checker(dataset: xr.Dataset) -> Callable[[Path], None]:
    def check(path: Path) -> None:
        with xr.open_dataset(path, engine="h5netcdf") as written:
            check_file_reads_back_as_the_reference(written.load(), dataset, path)

    return check


# ── checks ────────────────────────────────────────────────────────────────────


def check_file_reads_back_as_the_reference(written: xr.Dataset, dataset: xr.Dataset, path: Path) -> None:
    """The file read back is the reference it was written from, value for value and label for label."""
    if not written.identical(dataset):
        raise ValueError(f"{path.name} does not read back as the reference it was written from; inspect it")


if __name__ == "__main__":
    main()

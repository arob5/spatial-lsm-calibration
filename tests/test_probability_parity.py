"""The probability layer reproduces the parameter layer's priors, bit for bit.

``tests/data/probability_references/`` holds five priors' draws, densities
and natural values as the parameter layer computed them, written by a script
removed with that layer (``tests/data/write_probability_references.py``,
last run at commit d89a336). Each case is declared here again in the
probability layer's terms, bound at the same labels and conditioned on
nothing, and its posterior must give the same theta order, the same draws
from the same key, the same log density and the same natural values, to the
last bit.

The files depend on the JAX and TFP builds: if an upgrade moves their bits,
rerun the script from that commit's tree in the new environment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from conftest import REPOSITORY, site_table_of
from sipnet_calibration import calibration
from sipnet_calibration.site_dims import SiteDims
from sipnet_calibration.probability import (
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    SIMPLEX,
    ArraySpec,
    DeterministicSpec,
    FactorSpec,
    condition_on,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
    joint,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    softmax_normal,
)

tfd, tfb = tfp.distributions, tfp.bijectors

REFERENCES = REPOSITORY / "tests" / "data" / "probability_references"

ALLOCATION_PART = "allocation_part"
ALLOCATION_PARTS = ("leaf", "wood", "fine_root", "coarse_root")
SITES = (1, 27, 4711)
PFT_OF_SITE = ("deciduous", "conifer", "deciduous")
PFT = ("conifer", "deciduous")


def _factor(name, law, *, units="1", support=None, **arguments):
    spec_arguments = {"support": support} if support is not None else {}
    return FactorSpec(ArraySpec(name, units=units, **spec_arguments), law=law, provenance="Parity.", **arguments)


def single_site_mcmc_vs_eki(reference):
    members = {name: reference[f"constant:{name}_members"].values for name in ("initial_wood_carbon", "initial_soil_organic_carbon")}
    return joint(
        _factor("photosynthetic_capacity", log_normal_from_interval(lower=140.0, upper=450.0), support=POSITIVE),
        _factor("respiration_share", logit_normal_from_interval(lower=0.04, upper=0.20), support=OPEN_UNIT_INTERVAL),
        _factor("optimum_photosynthesis_temperature", tfd.Normal(jnp.float64(22.0), jnp.float64(2.5)), units="degC"),
        _factor("half_saturation_light", log_normal_from_interval(lower=4.6, upper=26.3), support=POSITIVE),
        _factor("soil_water_holding_capacity", log_normal_from_interval(lower=15.0, upper=150.0), support=POSITIVE),
        _factor("leaf_on_growth", log_normal_from_interval(lower=50.0, upper=180.0), support=POSITIVE),
        _factor("leaf_on_growing_degree_days", log_normal_from_interval(lower=500.0, upper=1100.0), support=POSITIVE),
        FactorSpec(
            ArraySpec("allocation", units="1", support=SIMPLEX, element_axes={ALLOCATION_PART: ALLOCATION_PARTS}),
            law=softmax_normal(center=jnp.array([0.18, 0.45, 0.065, 0.305]), logit_sd=jnp.array([0.25, 0.30, 0.30])),
        ),
        _factor("wood_respiration_rate_at_10c", log_normal_from_interval(lower=0.006, upper=0.04), support=POSITIVE),
        _factor("soil_respiration_flux_at_10c", log_normal_from_interval(lower=200.0, upper=900.0), support=POSITIVE),
        _factor("soil_respiration_q10", log_normal_from_interval(lower=1.3, upper=3.2), support=POSITIVE),
        *(_factor(name, log_normal_from_samples(members[name]), support=POSITIVE) for name in members),
    ).bind(coords={})


def example_calibration(reference):
    site_dims = SiteDims(site_table=site_table_of(*SITES), site_labels={"pft": PFT_OF_SITE})
    return joint(*calibration.example_calibration(site_dims)[0]).bind(coords={"pft": list(PFT), "site": list(SITES)})


def hierarchy(reference):
    sites = np.asarray(SITES, dtype=np.int32)
    pft_of_site = xr.DataArray(list(PFT_OF_SITE), dims="site", coords={"site": sites}, name="pft")

    def soil_carbon_given_its_mean(site_log_mean, spread):
        return tfd.TransformedDistribution(tfd.Independent(tfd.Normal(site_log_mean, spread), 1), tfb.Exp())

    return joint(
        FactorSpec(ArraySpec("pft_log_mean", units="1", indexed_by=("pft",)),
                   law=iid_over_dim(tfd.Normal(jnp.float64(np.log(1e4)), jnp.float64(1.0)))),
        _factor("spread", log_normal(median=0.5, geometric_sd=2.0), support=POSITIVE),
        DeterministicSpec(ArraySpec("site_log_mean", units="1", indexed_by=("site",)),
                          function=lambda pft_log_mean, pft_of_site: pft_log_mean[..., pft_of_site],
                          label_maps={"pft_of_site": pft_of_site}),
        FactorSpec(ArraySpec("soil_carbon", units="g m-2", support=POSITIVE, indexed_by=("site",)),
                   law=soil_carbon_given_its_mean),
        FactorSpec(ArraySpec("offsets", units="1", indexed_by=("site",)),
                   law=iid_over_dim(lambda spread: tfd.Normal(jnp.float64(0.0), spread))),
    ).bind(coords={"pft": list(PFT), "site": sites})


def copula(reference):
    marginals = {
        "photosynthetic_capacity": log_normal(median=250.0, geometric_sd=1.6),
        "respiration_share": logit_normal(median=0.18, logit_sd=0.35),
        "leaf_fall_fraction": logit_normal(median=0.5, logit_sd=1.0),
    }
    correlation = [[1.0, -0.4, 0.2], [-0.4, 1.0, 0.3], [0.2, 0.3, 1.0]]
    return joint(FactorSpec(
        [ArraySpec("photosynthetic_capacity", units="nmol g-1 s-1", support=POSITIVE),
         ArraySpec("respiration_share", units="1", support=OPEN_UNIT_INTERVAL),
         ArraySpec("leaf_fall_fraction", units="1", support=OPEN_UNIT_INTERVAL)],
        law=gaussian_copula(marginals, correlation=correlation),
    )).bind(coords={})


def per_pft_simplex(reference):
    center = reference["constant:center"].astype(np.float64)
    return joint(FactorSpec(
        ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=("pft",),
                  element_axes={ALLOCATION_PART: ALLOCATION_PARTS}),
        law=independent_over_dim(softmax_normal, logit_sd=jnp.array([0.25, 0.30, 0.30])),
        constants={"center": center},
    )).bind(coords={"pft": list(PFT)})


CASES = {
    "single_site_mcmc_vs_eki": single_site_mcmc_vs_eki,
    "example_calibration": example_calibration,
    "hierarchy": hierarchy,
    "copula": copula,
    "per_pft_simplex": per_pft_simplex,
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    with xr.open_dataset(REFERENCES / f"{request.param}.nc", engine="h5netcdf") as reference:
        reference = reference.load()
    return reference, condition_on(CASES[request.param](reference), {})


def test_theta_is_laid_out_as_today(case):
    reference, posterior = case
    index = posterior.parameters.unconstrained.index
    assert len(index) == reference.sizes["entry"]
    assert _strings(index.get_level_values("component")) == list(reference["entry_parameter"].values)
    assert _strings(index.get_level_values("element")) == list(reference["entry_element"].values)
    for level in ("pft", "site"):
        if f"entry_{level}" in reference.coords:
            assert _strings(index.get_level_values(level)) == list(reference[f"entry_{level}"].values)


def test_prior_draws_are_todays_bit_for_bit(case):
    reference, posterior = case
    draws = posterior.sample_prior(jax.random.key(int(reference.attrs["seed"])), int(reference.attrs["n_draws"]))
    np.testing.assert_array_equal(np.asarray(draws), reference["draws"].values)


def test_log_prior_is_todays_bit_for_bit(case):
    reference, posterior = case
    np.testing.assert_array_equal(np.asarray(posterior.log_prior(reference["theta"].values)), reference["log_prob"].values)


def test_natural_values_are_todays_bit_for_bit(case):
    reference, posterior = case
    values = posterior.natural_values(reference["theta"].values)
    stored = [name for name in reference.data_vars if name.startswith(("natural:", "derived:"))]
    assert sorted(name.split(":", 1)[1] for name in stored) == sorted(values)
    for name in stored:
        np.testing.assert_array_equal(np.asarray(values[name.split(":", 1)[1]]), reference[name].values, err_msg=name)


def test_labeled_values_carry_todays_labels(case):
    reference, posterior = case
    labeled = posterior.to_labeled(reference["theta"].values)
    for name in (n for n in reference.data_vars if n.startswith("natural:")):
        stored = reference[name]
        array = labeled[name.split(":", 1)[1]]
        for dim in stored.dims:
            if dim != "sample":
                assert list(array.indexes[dim]) == list(stored.indexes[dim]), (name, dim)
        np.testing.assert_array_equal(array.transpose(*stored.dims).values, stored.values)


def _strings(values) -> list[str]:
    return ["" if pd.isna(v) else str(v) for v in values]

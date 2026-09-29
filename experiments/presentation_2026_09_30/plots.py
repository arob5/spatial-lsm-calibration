"""One function per figure of the deck, over the experiment's readers and figures.

Each reads the runs ``config`` names through
``experiments.single_site_mcmc_vs_eki.model.outputs`` and draws with that
experiment's figure functions, so a slide shows what the experiment's own
figures show. Nothing here runs a model.
"""

import sys
from functools import cache

import pandas as pd

import config

# The experiment is a package under the repository root, which a document
# rendered from this directory does not have on its path.
if str(config.REPOSITORY) not in sys.path:
    sys.path.insert(0, str(config.REPOSITORY))

from experiments.single_site_mcmc_vs_eki.figures import (  # noqa: E402
    diagnostics as diagnostic_figures,
)
from experiments.single_site_mcmc_vs_eki.figures import eki as eki_figures  # noqa: E402
from experiments.single_site_mcmc_vs_eki.figures import (  # noqa: E402
    prior_predictive as predictive_figures,
)
from experiments.single_site_mcmc_vs_eki.model import prior  # noqa: E402
from experiments.single_site_mcmc_vs_eki.model.outputs import (  # noqa: E402
    load_diagnostics,
    load_eki_run,
    load_predictive,
)

__all__ = [
    "eki_ladder",
    "eki_marginals",
    "posterior_predictive_coverage",
    "posterior_predictive_nee_annual",
    "posterior_predictive_nee_seasonal",
    "predictive_check",
    "prior_marginals",
    "prior_predictive_coverage",
    "prior_predictive_nee",
    "prior_predictive_nee_annual",
    "prior_predictive_nee_seasonal",
    "prior_predictive_pools",
    "residual_autocorrelation",
    "synthetic_recovery",
    "towers",
    "weekly_residuals",
]


# ── the prior and its predictive ──


def prior_marginals():
    """Each calibrated parameter's prior, as the prior predictive's draws."""
    parameters = pd.read_csv(
        config.PRIOR_PREDICTIVE_DIRECTORY / "parameters.csv", index_col=0
    )
    return predictive_figures.plot_prior_marginals(parameters)


def prior_predictive_nee():
    """Both NEE sources against the prior predictive, whole record and one year."""
    return predictive_figures.plot_nee_windows(_prior_predictive())


def prior_predictive_pools():
    """LAI, biomass and soil carbon against the prior predictive."""
    return predictive_figures.plot_pool_observations(_prior_predictive())


def prior_predictive_nee_seasonal():
    """NEE's seasonal cycle, observed against the prior predictive."""
    return predictive_figures.plot_nee_seasonal_cycle(_prior_predictive())


def prior_predictive_nee_annual():
    """Annual NEE, the prior predictive against both towers."""
    return predictive_figures.plot_nee_annual(_prior_predictive())


def prior_predictive_coverage():
    """Per source, the observations inside the prior predictive's intervals."""
    return predictive_figures.plot_coverage(_prior_predictive())


# ── EKI ──


def synthetic_recovery():
    """The synthetic run's posterior against the truth, each entry standardized."""
    entry_names = list(
        prior.calibration()[0].index.get_level_values("unconstrained_name")
    )
    return eki_figures.plot_recovery(_eki_run("synthetic"), entry_names)


def eki_ladder():
    """The observed run's ladder: the level, the misfits, the ensemble's health."""
    return eki_figures.plot_ladder(_eki_run("observed"))


def eki_marginals():
    """The observed run's prior and posterior ensembles, per parameter."""
    return eki_figures.plot_marginals(_eki_run("observed"))


# ── the first calibration's posterior predictive ──


def posterior_predictive_nee_seasonal():
    """NEE's seasonal cycle, observed against the posterior predictive."""
    return predictive_figures.plot_nee_seasonal_cycle(
        _posterior_predictive(), kind="posterior"
    )


def posterior_predictive_nee_annual():
    """Annual NEE, the posterior predictive against both towers."""
    return predictive_figures.plot_nee_annual(_posterior_predictive(), kind="posterior")


def posterior_predictive_coverage(vector: str = "calibration"):
    """Per source, the observations inside the posterior predictive's intervals."""
    return predictive_figures.plot_coverage(
        _posterior_predictive(), kind="posterior", vector=vector
    )


# ── the first calibration's diagnostics ──


def predictive_check():
    """The posterior predictive check per source, calibration and held out."""
    return diagnostic_figures.plot_predictive_check(_diagnostics())


def weekly_residuals():
    """Each year's weekly NEE residual, and the part that recurs every year."""
    return diagnostic_figures.plot_weekly_residuals(_diagnostics())


def residual_autocorrelation():
    """The NEE residuals' autocorrelation against the one R implied."""
    return diagnostic_figures.plot_autocorrelation(_diagnostics())


def towers():
    """US-Ha1's NEE windows against US-xHA's, where both keep one."""
    return diagnostic_figures.plot_towers(_diagnostics())


# ── helpers ──


@cache
def _prior_predictive() -> dict:
    """The prior predictive's outputs, read once."""
    return load_predictive(config.PRIOR_PREDICTIVE_DIRECTORY)


@cache
def _posterior_predictive() -> dict:
    """The first calibration's posterior predictive, read once."""
    return load_predictive(config.EKI_OBSERVED_DIRECTORY / "posterior_predictive")


@cache
def _eki_run(data: str) -> dict:
    """One of the first calibration's EKI runs, read once."""
    directory = {
        "synthetic": config.EKI_SYNTHETIC_DIRECTORY,
        "observed": config.EKI_OBSERVED_DIRECTORY,
    }[data]
    return load_eki_run(directory)


@cache
def _diagnostics() -> dict:
    """The first calibration's diagnostics, under the R it ran with."""
    return load_diagnostics(config.EKI_OBSERVED_DIRECTORY)

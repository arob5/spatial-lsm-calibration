"""The deck's choices: which runs of the Harvard Forest experiment it shows.

The deck reads what ``experiments/single_site_mcmc_vs_eki`` wrote and runs
no model. Its runs are named here, not taken from that experiment's current
setup, so the deck shows the same runs whatever the experiment runs next.
"""

from pathlib import Path

__all__ = [
    "EKI_OBSERVED_DIRECTORY",
    "EKI_SYNTHETIC_DIRECTORY",
    "EXPERIMENT_OUTPUT_DIRECTORY",
    "FIRST_CALIBRATION_SETUP",
    "PRIOR_PREDICTIVE_DIRECTORY",
    "REPOSITORY",
]

#: The repository root, which holds the experiment package.
REPOSITORY = Path(__file__).resolve().parents[2]

#: What the Harvard Forest experiment wrote; untracked.
EXPERIMENT_OUTPUT_DIRECTORY = (
    REPOSITORY / "experiments" / "single_site_mcmc_vs_eki" / "output"
)

#: The 200-draw prior predictive.
PRIOR_PREDICTIVE_DIRECTORY = EXPERIMENT_OUTPUT_DIRECTORY / "prior_predictive"

#: The EKI setup of the first calibration: the single-term NEE discrepancy.
FIRST_CALIBRATION_SETUP = "single_term_discrepancy"

#: The first calibration's EKI runs, on synthetic and on observed data.
EKI_SYNTHETIC_DIRECTORY = (
    EXPERIMENT_OUTPUT_DIRECTORY / "eki" / FIRST_CALIBRATION_SETUP / "synthetic"
)
EKI_OBSERVED_DIRECTORY = (
    EXPERIMENT_OUTPUT_DIRECTORY / "eki" / FIRST_CALIBRATION_SETUP / "observed"
)

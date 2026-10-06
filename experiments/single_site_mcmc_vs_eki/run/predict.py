"""The posterior predictive of a run: calibration, held-out and daily output in one pass.

Overview
--------
Draws up to ``--samples`` of a run's samples (by weight, systematically, for
a weighted run; evenly spaced for MCMC; all of EKI's), runs SIPNET once at
each, and from that one pass writes the predictions of the calibration
vector and of the held-out tower's NEE, the daily model output, and the
held-out log score.

Input data
----------
The run's ``samples.nc`` (``algorithms/records.py``) and everything the
model reads.

Output data
-----------
Under ``<run>/predictive/``:

- ``samples.csv``: which of the run's samples were run, with their scales;
- ``predictions/ensemble/calibration/<source>.nc`` and
  ``predictions/ensemble/validation/<source>.nc`` (the held-out tower): each
  source's predictions on ``(sample, site[, time])``;
- ``observed/calibration/<source>.nc``, ``observed/validation/<source>.nc``:
  the observed values and :math:`\\sqrt{\\operatorname{diag} C_k}`;
- ``ensemble_daily.nc``: ``config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES``, daily;

the layout ``model/outputs.py``'s ``load_predictive`` reads, as the prior
predictive's;
- ``heldout_scores.csv``: per held-out source, the log predictive density
  :math:`\\log \\frac1K \\sum_m \\mathcal N(y^{\\rm val}_k; \\mathcal
  G^{\\rm val}_k(\\theta_m), s_{mk} C^{\\rm val}_k)`, ``s = 1`` for fixed
  noise, with its Monte Carlo standard error, and :math:`2\\Phi_k/n_k` at
  the median over samples;
- ``cost.json``.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.predict --model long_memory/fixed --run eki
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.special

from sipnet_calibration import smc
from sipnet_calibration.conventions import SAMPLE
from sipnet_calibration.observation.model import prediction_components
from sipnet_calibration.probability import Layout

from .. import config
from ..algorithms.records import Cost, load_samples
from ..model import noise
from ..model.likelihood import NoiseModel
from ..models import (
    MODEL_NAMES,
    SIMULATOR_NAME,
    Model,
    fixed_posterior,
    heldout_posterior,
    observation_vector,
)

__all__ = ["main"]

#: The directory each vector's files go in, as ``model/outputs.py`` reads them.
_DIRECTORY_NAMES = {"calibration": "calibration", "heldout": "validation"}


def main(argv: list[str] | None = None) -> int:
    """Run the predictive of one run and write it."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    model = Model.parse(arguments.model)
    run_directory = model.directory(arguments.run)
    directory = run_directory / "predictive"
    directory.mkdir(parents=True, exist_ok=True)
    samples = load_samples(run_directory)
    chosen = _chosen_samples(samples, arguments.samples)
    theta = samples["theta"].values[chosen]
    posteriors = {"calibration": fixed_posterior(model), "heldout": heldout_posterior(model)}
    vectors = {label: observation_vector(p) for label, p in posteriors.items()}
    runs = posteriors["calibration"].simulators[SIMULATOR_NAME].runs
    cost = Cost()
    with cost.phase("predictive") as phase:
        evaluation = runs.evaluate(
            posteriors["calibration"].simulator_inputs(theta, SIMULATOR_NAME),
            observation_vectors=list(vectors.values()),
            output_variable_names=config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES,
            freq="1D",
        )
        phase["sipnet_runs"], phase["forward_calls"] = len(theta), 1
    if len(evaluation.failures):
        print(f"{len(evaluation.failures)} runs failed", flush=True)
    scales = {
        name: samples[f"{name}_noise_scale"].values[chosen]
        for name in config.NOISE_SCALED_SOURCES
        if f"{name}_noise_scale" in samples
    }
    pd.DataFrame({"run_sample": chosen, **scales}).to_csv(directory / "samples.csv", index=False)
    for (label, vector), predictions in zip(vectors.items(), evaluation.predictions, strict=True):
        name = _DIRECTORY_NAMES[label]
        _write_fields(directory / "predictions" / "ensemble" / name, vector, predictions)
        _write_observed(directory / "observed" / name, posteriors[label], vector)
    evaluation.model_output.to_netcdf(directory / "ensemble_daily.nc")
    heldout = NoiseModel(posteriors["heldout"], inferred=model.noise == "inferred")
    _heldout_scores(heldout, evaluation.predictions[1], scales).to_csv(
        directory / "heldout_scores.csv"
    )
    cost.write(directory)
    print(f"wrote {directory}", flush=True)
    return 0


def _chosen_samples(samples, n: int) -> np.ndarray:
    """The samples to run: by weight for a weighted run, evenly spaced for a
    long unweighted one, else all."""
    count = samples.sizes["sample"]
    log_weights = samples["log_weight"].values
    valid = samples["valid"].values.astype(bool) if "valid" in samples else np.ones(count, bool)
    if not np.allclose(log_weights, log_weights[0]):
        weights = np.where(valid, log_weights, -np.inf)
        return np.sort(smc.systematic_resample(weights, n, np.random.default_rng(config.REWEIGHTING_SEED)))
    indices = np.flatnonzero(valid)
    if len(indices) <= n:
        return indices
    return indices[np.linspace(0, len(indices) - 1, n).round().astype(int)]


def _write_fields(directory, vector, predictions) -> None:
    """Each source's predictions as a field of *vector*."""
    directory.mkdir(parents=True, exist_ok=True)
    layout = Layout(prediction_components(vector), coords=vector.coords)
    labeled = layout.values_to_labeled(dict(predictions), batch_dims=(SAMPLE,))
    source_of = {vector.prediction_name(name): name for name in vector.observation_source_names}
    for name, field in vector.to_fields(labeled).items():
        field.to_netcdf(directory / f"{source_of.get(name, name)}.nc")


def _write_observed(directory, posterior, vector) -> None:
    """Each source's observed values, with :math:`\\sqrt{\\operatorname{diag} C_k}`."""
    directory.mkdir(parents=True, exist_ok=True)
    standard_deviations = vector.to_fields(noise.noise_standard_deviations(posterior))
    for name in vector.observation_source_names:
        dataset = vector[name].observed_values.to_dataset(name="value")
        dataset["noise_standard_deviation"] = standard_deviations[name]
        dataset.to_netcdf(directory / f"{name}.nc")


def _heldout_scores(noise_model: NoiseModel, predictions, scales) -> pd.DataFrame:
    """Per held-out source: the log predictive density, its standard error and 2 Phi / n."""
    rows = {}
    for source in noise_model.sources:
        g = np.full((len(next(iter(predictions.values()))), noise_model.y.size), np.nan)
        g[:, source.positions] = np.asarray(predictions[f"predicted_{source.name}"])
        failed = np.isnan(g[:, source.positions]).any(axis=1)
        q = noise_model.quadratic_forms(np.nan_to_num(g, nan=0.0))[source.name]
        q[failed] = np.nan
        scale = scales.get(source.name, np.ones(len(q)))
        log_density = -0.5 * (q / scale + source.size * np.log(2 * np.pi * scale) + source.log_determinant)
        finite = np.isfinite(log_density)
        log_density = log_density[finite]
        K = len(log_density)
        score = scipy.special.logsumexp(log_density) - np.log(K)
        relative = np.exp(log_density - log_density.max())
        standard_error = np.std(relative, ddof=1) / np.sqrt(K) / np.mean(relative)
        rows[source.name] = {
            "observations": source.size,
            "log_predictive_density": score,
            "monte_carlo_standard_error": standard_error,
            "misfit_ratio_median": float(np.median(q[finite] / scale[finite]) / source.size),
            "samples": K,
        }
    return pd.DataFrame(rows).T.rename_axis("observation_source")


def _parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--run", required=True, help="the run's directory name, e.g. eki or eki_gibbs_common_is")
    parser.add_argument("--samples", type=int, default=300, help="how many samples to run")
    return parser


if __name__ == "__main__":
    sys.exit(main())

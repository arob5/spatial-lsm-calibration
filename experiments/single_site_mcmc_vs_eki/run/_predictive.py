"""Predictive runs: an ensemble, and optionally one run by hand, against both observation vectors.

What the prior and the posterior predictive share: evaluating an ensemble of
theta under the calibration and the validation posteriors
(``model/calibration.py``), each sample's predictions and log likelihood,
running it once more for daily model output, and writing it all in one
layout, which ``model/outputs.py``'s ``load_predictive`` reads. A sample
that is invalid, its run failed or its SIPNET parameters outside pySIPNET's
domain, is written as NaN, with log likelihood ``-inf``, rather than
stopping the run, and the failures are printed.

Output data
-----------
Under the directory given:

- ``single_run_daily.nc`` (with a center), ``ensemble_daily.nc``: the model
  output variables of ``config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES``,
  aggregated to days by each variable's kind, on ``(site, time)`` and
  ``(sample, site, time)``;
- ``predictions/<run>/<vector>/<observation source>.nc``, ``<run>`` being
  ``single_run`` or ``ensemble`` and ``<vector>`` ``calibration`` or
  ``validation``: each observation source's predictions, in its observed
  values' units, on its observed values' grid;
- ``observed/<vector>/<observation source>.nc``: each source's observed
  values, ``value``, with ``noise_standard_deviation``, the square root of
  the diagonal of its noise covariance block;
- ``parameters.csv``: theta's natural values, one row per run
  (``single_run`` first, when there is one, then the samples), with each
  run's log likelihood under the calibration posterior.
"""

import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SAMPLE, SITE
from sipnet_calibration.fields import to_model_output
from sipnet_calibration.observation import aggregate_time
from sipnet_calibration.probability import Posterior

from .. import config
from ..model import calibration, inputs, noise, prior

__all__ = [
    "run_ensemble",
    "run_once_by_hand",
    "run_predictive",
    "write_observations",
    "write_outputs",
]


def run_predictive(
    directory, calibration_posterior, validation_posterior, theta, *, center=None
) -> None:
    """Run *theta*, and one run by hand at *center* if given, and write it all.

    *theta* is ``(J, D)`` and *center* ``(D,)``, a point of both posteriors.
    """
    posteriors = {
        "calibration": calibration_posterior,
        "validation": validation_posterior,
    }
    directory.mkdir(parents=True, exist_ok=True)
    single = None
    if center is not None:
        single = run_once_by_hand(posteriors, center)
        print("one run by hand: done")
    ensemble = run_ensemble(posteriors, theta)
    print(f"ensemble of {theta.shape[0]}: done")
    rows = np.asarray(theta) if center is None else np.vstack([center, theta])
    write_outputs(directory, calibration_posterior, rows, single, ensemble)
    write_observations(directory, posteriors)


def run_once_by_hand(posteriors: dict[str, Posterior], theta) -> dict:
    """One run at *theta*, through each layer the forward map composes."""
    posterior = posteriors["calibration"]
    runs = posterior.simulators[calibration.SIMULATOR_NAME].runs
    # The map: what the simulator reads at theta, its natural values and the
    # inputs, to SIPNET parameter fields; one run's keywords are their values.
    values = posterior.simulator_inputs(np.asarray(theta)[None], calibration.SIMULATOR_NAME)
    sipnet_parameter_fields = runs.sipnet_parameter_map.sipnet_parameter_fields(
        values, site_dims=runs.site_dims
    )
    at_the_run = sipnet_parameter_fields.isel({SAMPLE: 0}).sel({SITE: config.SITE})
    sipnet_overrides = {
        name: float(value) for name, value in at_the_run.data_vars.items()
    }
    # One SIPNET run, and its output as model output.
    sipnet_result = runs.sipnet_model(**sipnet_overrides)
    vectors = {
        label: calibration.observation_vector(posterior)
        for label, posterior in posteriors.items()
    }
    output_variable_names = sorted(
        set(config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES).union(
            *(vector.output_variable_names for vector in vectors.values())
        )
    )
    model_output = to_model_output(
        sipnet_result,
        output_variable_names=output_variable_names,
        site=config.SITE,
        site_table=inputs.site_table(),
    )
    # Each vector's operators, reading the run's own SIPNET parameters, as on
    # a worker.
    predicted = {}
    for label, vector in vectors.items():
        run_parameters = xr.Dataset(
            {
                name: sipnet_result.parameters.dataarray(name)
                for name in vector.sipnet_parameter_names_read
            },
            coords={name: model_output[name] for name in (SITE, "lon", "lat")},
        )
        predicted[label] = vector.predict(
            model_output, sipnet_parameter_fields=run_parameters
        )
    daily = xr.Dataset(
        {
            name: aggregate_time(model_output[name], "1D")
            for name in config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES
        }
    )
    return {
        "predicted": predicted,
        "daily": daily,
        "log_likelihood": np.asarray(posterior.log_likelihood(theta))[None],
    }


def run_ensemble(posteriors: dict[str, Posterior], theta) -> dict:
    """*theta* evaluated under each posterior, then run once more for daily output."""
    predicted = {}
    for label, posterior in posteriors.items():
        evaluation = posterior.evaluate(theta)
        _report_failures(
            label, evaluation.simulator_records[calibration.SIMULATOR_NAME]
        )
        predicted[label] = calibration.predicted_fields(posterior, evaluation)
        if label == "calibration":
            log_likelihood = np.asarray(evaluation.log_likelihood)
    posterior = posteriors["calibration"]
    daily = (
        posterior.simulators[calibration.SIMULATOR_NAME]
        .runs.evaluate(
            posterior.simulator_inputs(theta, calibration.SIMULATOR_NAME),
            output_variable_names=config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES,
            freq="1D",
        )
    )
    _report_failures("daily output", daily)
    return {
        "predicted": predicted,
        "daily": daily.model_output,
        "log_likelihood": log_likelihood,
    }


def write_outputs(directory, posterior: Posterior, theta, single, ensemble) -> None:
    """Every output file, as the module docstring lists them.

    *theta* is the one run's row, when there is one, then the ensemble's.
    """
    runs = {"ensemble": ensemble}
    if single is not None:
        single["daily"].to_netcdf(directory / "single_run_daily.nc")
        runs = {"single_run": single, **runs}
    ensemble["daily"].to_netcdf(directory / "ensemble_daily.nc")
    for run_name, run in runs.items():
        for vector_name, fields in run["predicted"].items():
            target = directory / "predictions" / run_name / vector_name
            target.mkdir(parents=True, exist_ok=True)
            for source_name, field in fields.items():
                field.to_netcdf(target / f"{source_name}.nc")
    natural = prior.natural_table(posterior, theta)
    n_samples = len(ensemble["log_likelihood"])
    natural.index = [
        *(["single_run"] if single is not None else []),
        *[f"sample_{i}" for i in range(n_samples)],
    ]
    natural["log_likelihood"] = np.concatenate(
        [run["log_likelihood"] for run in runs.values()]
    )
    natural.to_csv(directory / "parameters.csv")


def write_observations(directory, posteriors: dict[str, Posterior]) -> None:
    """Each source's observed values, with its total noise standard deviation."""
    for vector_name, posterior in posteriors.items():
        vector = calibration.observation_vector(posterior)
        noise_standard_deviations = vector.to_fields(
            noise.noise_standard_deviations(posterior)
        )
        target = directory / "observed" / vector_name
        target.mkdir(parents=True, exist_ok=True)
        for source_name in vector.observation_source_names:
            dataset = vector[source_name].observed_values.to_dataset(name="value")
            dataset["noise_standard_deviation"] = noise_standard_deviations[
                source_name
            ]
            dataset.to_netcdf(target / f"{source_name}.nc")


# ── helpers ──


def _report_failures(label: str, runs_evaluation) -> None:
    """Print how many runs failed, and why."""
    failures = runs_evaluation.failures
    if len(failures):
        print(f"{label}: {len(failures)} runs failed")
        print(failures[["error", "message"]].drop_duplicates().to_string())

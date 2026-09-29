"""Predictive runs: an ensemble, and optionally one run by hand, against both observation vectors.

What the prior and the posterior predictive share: running an ensemble of
theta through the forward model for the calibration and validation
predictions and for daily model output, scoring each row under the
calibration likelihood, and writing it all in one layout, which
``model/outputs.py``'s ``load_predictive`` reads. A member that fails, or
whose SIPNET parameters leave pySIPNET's domain (the forward model's
``out_of_domain="fail_row"``, as EKI runs it), is written as NaN rather than
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
  run's log likelihood.
"""

import numpy as np
import xarray as xr

from sipnet_calibration.conventions import SITE
from sipnet_calibration.fields import to_model_output
from sipnet_calibration.observation import aggregate_time

from .. import config
from ..model import inputs, noise, observations, sipnet

__all__ = [
    "run_ensemble",
    "run_once_by_hand",
    "run_predictive",
    "write_observations",
    "write_outputs",
]


def run_predictive(
    directory, vector, sipnet_map, external_inputs, samples, *, center=None
) -> None:
    """Run *samples*, and one run by hand at *center* if given, and write it all."""
    calibration = observations.calibration_observation_vector()
    validation = observations.validation_observation_vector()
    directory.mkdir(parents=True, exist_ok=True)
    single = None
    if center is not None:
        single = run_once_by_hand(
            vector, sipnet_map, center, calibration, validation, external_inputs
        )
        print("one run by hand: done")
    ensemble = run_ensemble(
        vector, sipnet_map, samples, calibration, validation, external_inputs
    )
    print(f"ensemble of {samples.shape[0]}: done")
    rows = [np.asarray(samples)] if center is None else [center, np.asarray(samples)]
    write_outputs(directory, vector, np.vstack(rows), single, ensemble)
    write_observations(
        directory, {"calibration": calibration, "validation": validation}
    )


def run_once_by_hand(
    vector, sipnet_map, theta, calibration, validation, external_inputs
) -> dict:
    """One run at *theta*, through each layer the forward model composes."""
    # The map: theta and the external inputs to SIPNET parameter fields.
    sipnet_parameter_fields = sipnet_map.sipnet_parameter_fields(
        vector, theta, external_inputs=external_inputs
    )
    # One run's keywords: each field's value at the site.
    sipnet_overrides = {
        name: float(sipnet_parameter_fields[name].sel({SITE: config.SITE}))
        for name in sipnet_parameter_fields.data_vars
    }
    # One SIPNET run, and its output as model output.
    sipnet_result = sipnet.sipnet_model()(**sipnet_overrides)
    output_variable_names = sorted(
        set(config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES)
        | set(calibration.output_variable_names)
    )
    model_output = to_model_output(
        sipnet_result,
        output_variable_names=output_variable_names,
        site=config.SITE,
        site_table=inputs.site_table(),
    )
    # The operators read the run's own SIPNET parameters, as on a worker.
    run_parameters = xr.Dataset(
        {
            name: sipnet_result.parameters.dataarray(name)
            for name in calibration.sipnet_parameter_names_read
        },
        coords={name: model_output[name] for name in (SITE, "lon", "lat")},
    )
    predicted = {
        "calibration": calibration.predict(
            model_output, sipnet_parameter_fields=run_parameters
        ),
        "validation": validation.predict(
            model_output, sipnet_parameter_fields=run_parameters
        ),
    }
    likelihood = noise.calibration_likelihood(calibration)
    log_likelihood = float(
        likelihood.log_density(calibration.flat(predicted["calibration"]))
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
        "log_likelihood": np.array([log_likelihood]),
    }


def run_ensemble(
    vector, sipnet_map, samples, calibration, validation, external_inputs
) -> dict:
    """The ensemble through the forward model: predictions of both vectors, then daily output."""
    predicted = {}
    for label, observation_vector in (
        ("calibration", calibration),
        ("validation", validation),
    ):
        evaluation = sipnet.forward_model(
            vector,
            sipnet_map,
            observation_vector=observation_vector,
            external_inputs=external_inputs,
            out_of_domain="fail_row",
        ).evaluate(samples)
        _report_failures(label, evaluation)
        predicted[label] = evaluation.predicted_fields()
        if label == "calibration":
            likelihood = noise.calibration_likelihood(calibration)
            log_likelihood = np.asarray(likelihood.log_density(evaluation.predictions))
    daily = sipnet.forward_model(
        vector,
        sipnet_map,
        output_variable_names=config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES,
        freq="1D",
        external_inputs=external_inputs,
        out_of_domain="fail_row",
    ).evaluate(samples)
    _report_failures("daily output", daily)
    return {
        "predicted": predicted,
        "daily": daily.model_output,
        "log_likelihood": log_likelihood,
    }


def write_outputs(directory, vector, theta, single, ensemble) -> None:
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
    natural = vector.dataset(theta).to_dataframe()
    n_samples = len(ensemble["log_likelihood"])
    natural.index = [
        *(["single_run"] if single is not None else []),
        *[f"sample_{i}" for i in range(n_samples)],
    ]
    natural["log_likelihood"] = np.concatenate(
        [run["log_likelihood"] for run in runs.values()]
    )
    natural.to_csv(directory / "parameters.csv")


def write_observations(directory, vectors) -> None:
    """Each source's observed values, with its total noise standard deviation."""
    series = {
        "calibration": config.CALIBRATION_NEE_SERIES,
        "validation": config.VALIDATION_NEE_SERIES,
    }
    for vector_name, observation_vector in vectors.items():
        blocks = noise.noise_covariance_blocks(observation_vector, series[vector_name])
        target = directory / "observed" / vector_name
        target.mkdir(parents=True, exist_ok=True)
        for source_name in observation_vector:
            observed = observation_vector[source_name].observed_values
            total = np.sqrt(np.diag(blocks[source_name])).reshape(observed.shape)
            dataset = observed.to_dataset(name="value")
            dataset["noise_standard_deviation"] = observed.copy(data=total)
            dataset.to_netcdf(target / f"{source_name}.nc")


# ── helpers ──


def _report_failures(label: str, evaluation) -> None:
    """Print how many runs failed, and why."""
    failures = evaluation.failures
    if len(failures):
        print(f"{label}: {len(failures)} runs failed")
        print(failures[["error", "message"]].drop_duplicates().to_string())

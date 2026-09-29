"""The prior predictive: SIPNET run once, and as an ensemble, against the observations.

Overview
--------
Runs the calibration's prior (``model/prior.py``) two ways and writes what
came back, for ``figures/prior_predictive.py`` to draw:

1. **One run, by hand**, at the prior mean, through pySIPNET directly. It
   shows the layers the forward model composes: theta to SIPNET parameter
   fields (the SIPNET parameter map), fields to a run's keywords (SIPNET
   overrides), one ``SIPNETModel`` call, the run's output as model output,
   and the observation vector's operators on it.
2. **An ensemble**, ``config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE`` draws of the
   prior, through :class:`~sipnet_calibration.forward.ForwardModel` and
   PyEns on local workers: once for the predictions of the calibration and
   validation observations, once for daily model output.

Every run is scored under the calibration's likelihood (``model/noise.py``).

Input data
----------
The prepared driver file (``scripts/prepare_drivers.py``), the processed
files ``model/inputs.py`` reads, and ``config``.

Output data
-----------
Under ``config.PRIOR_PREDICTIVE_DIRECTORY``:

- ``single_run_daily.nc``, ``ensemble_daily.nc``: the model output variables
  of ``config.PRIOR_PREDICTIVE_OUTPUT_VARIABLE_NAMES``, aggregated to days by
  each variable's kind, on ``(site, time)`` and ``(sample, site, time)``;
- ``predictions/<run>/<vector>/<observation source>.nc``, ``<run>`` being
  ``single_run`` or ``ensemble`` and ``<vector>`` ``calibration`` or
  ``validation``: each observation source's predictions, in its observed
  values' units, on its observed values' grid;
- ``observed/<vector>/<observation source>.nc``: each source's observed
  values, ``value``, with ``noise_standard_deviation``, the square root of
  the diagonal of its noise covariance block;
- ``parameters.csv``: theta's natural values, one row per run
  (``single_run`` first, then the samples), with each run's log likelihood;
- ``calibration.csv``: ``describe_calibration`` of the calibration, the
  record of what was run;
- ``provenance.json``: the code, packages, command and inputs of the run
  (``scripts/provenance.py``).

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prior_predictive
    uv run python -m experiments.single_site_mcmc_vs_eki.scripts.prior_predictive --ensemble-size 32
"""

import argparse
import sys
import warnings

import jax
import numpy as np
import xarray as xr

from sipnet_calibration.calibration import describe_calibration
from sipnet_calibration.conventions import SITE
from sipnet_calibration.fields import to_model_output
from sipnet_calibration.observation import aggregate_time

from .. import config
from ..model import inputs, noise, observations, prior, sipnet
from . import provenance

__all__ = ["main"]


# ── entry point ──


def main(argv: list[str] | None = None) -> int:
    """Run the prior predictive and write its outputs."""
    warnings.filterwarnings("ignore", message=".*vapor_pressure_deficit.*")
    arguments = _parser().parse_args(argv)
    vector, calibration_prior, sipnet_map = prior.calibration()
    external_inputs = prior.external_inputs()
    calibration = observations.calibration_observation_vector()
    validation = observations.validation_observation_vector()
    directory = config.PRIOR_PREDICTIVE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    describe_calibration(vector, calibration_prior, sipnet_map).to_csv(
        directory / "calibration.csv"
    )

    center = np.asarray(calibration_prior.gaussian().mean)
    single = run_once_by_hand(
        vector, sipnet_map, center, calibration, validation, external_inputs
    )
    print("one run by hand: done")
    samples = calibration_prior.sample(
        jax.random.key(config.PRIOR_PREDICTIVE_SEED), arguments.ensemble_size
    )
    ensemble = run_ensemble(
        vector, sipnet_map, samples, calibration, validation, external_inputs
    )
    print(f"ensemble of {samples.shape[0]}: done")

    write_outputs(
        directory, vector, np.vstack([center, np.asarray(samples)]), single, ensemble
    )
    write_observations(
        directory, {"calibration": calibration, "validation": validation}
    )
    provenance.write_provenance(
        directory / "provenance.json", input_files=provenance.model_input_files()
    )
    print(f"wrote {directory}")
    return 0


# ── the steps ──


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
    ).evaluate(samples)
    _report_failures("daily output", daily)
    return {
        "predicted": predicted,
        "daily": daily.model_output,
        "log_likelihood": log_likelihood,
    }


def write_outputs(directory, vector, theta, single, ensemble) -> None:
    """Every output file, as the module docstring lists them."""
    single["daily"].to_netcdf(directory / "single_run_daily.nc")
    ensemble["daily"].to_netcdf(directory / "ensemble_daily.nc")
    for run_name, run in (("single_run", single), ("ensemble", ensemble)):
        for vector_name, fields in run["predicted"].items():
            target = directory / "predictions" / run_name / vector_name
            target.mkdir(parents=True, exist_ok=True)
            for source_name, field in fields.items():
                field.to_netcdf(target / f"{source_name}.nc")
    natural = vector.dataset(theta).to_dataframe()
    natural.index = ["single_run", *[f"sample_{i}" for i in range(theta.shape[0] - 1)]]
    natural["log_likelihood"] = np.concatenate(
        [single["log_likelihood"], ensemble["log_likelihood"]]
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


def _parser() -> argparse.ArgumentParser:
    """The command line: the ensemble's size."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ensemble-size", type=int, default=config.PRIOR_PREDICTIVE_ENSEMBLE_SIZE
    )
    return parser


def _report_failures(label: str, evaluation) -> None:
    """Print how many runs failed, and why."""
    failures = evaluation.failures
    if len(failures):
        print(f"{label}: {len(failures)} runs failed")
        print(failures[["error", "message"]].drop_duplicates().to_string())


if __name__ == "__main__":
    sys.exit(main())

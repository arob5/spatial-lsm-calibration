"""What running SIPNET needs, built from the configuration.

Each function builds one piece from ``config``, so every script runs SIPNET
the same way: the base SIPNET parameters, the runner and the model, the
drivers, the site's initial state, and the forward model that runs an
ensemble through PyEns.

Functions
---------
:func:`base_sipnet_parameters`
    The base parameter set every SIPNET parameter the map leaves unset takes.
:func:`sipnet_runner`, :func:`sipnet_model`
    pySIPNET's runner with the configured flags and run settings, and the
    model over it with the base parameters and the drivers.
:func:`climate_drivers`
    The prepared driver file, file-backed, as the forward model needs it.
:func:`initial_state`
    The site's initial conditions as external inputs, on ``site`` alone.
:func:`forward_model`
    A :class:`~sipnet_calibration.forward.ForwardModel` over the site,
    predicting an observation vector or returning model output.
"""

from collections.abc import Sequence

import xarray as xr
from pyens import LocalBackend
from pysipnet.climate import ClimateDrivers
from pysipnet.io.param_io import PYTHON_TO_SIPNET, read_param_file
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import SIPNETParameters
from pysipnet.runner import SIPNETRunner

from sipnet_calibration.forward import ForwardModel
from sipnet_calibration.observation import ObservationVector
from sipnet_calibration.parameter_vector import ParameterVector
from sipnet_calibration.sipnet_parameter_map import (
    INITIAL_STATE_NAMES,
    SIPNETParameterMap,
)

from .. import config
from . import inputs

__all__ = [
    "base_sipnet_parameters",
    "climate_drivers",
    "forward_model",
    "initial_state",
    "sipnet_model",
    "sipnet_runner",
]


def base_sipnet_parameters() -> SIPNETParameters:
    """``config.BASE_SIPNET_PARAMETER_FILE`` as pySIPNET parameters.

    Every parameter the file names is read; one it does not keeps pySIPNET's
    default. pySIPNET has no public ``.param`` reader yet (its issue #19), so
    the file is read through its name mapping, as ``tests/conftest.py`` does.
    """
    raw = read_param_file(config.BASE_SIPNET_PARAMETER_FILE)
    groups: dict[str, dict[str, float]] = {
        name: {} for name in SIPNETParameters.model_fields
    }
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


def sipnet_runner() -> SIPNETRunner:
    """pySIPNET's runner, with the configured flags, timeout and climate staging."""
    return SIPNETRunner(
        flags=config.MODEL_FLAGS,
        climate_staging=config.CLIMATE_STAGING,
        timeout=config.SIPNET_TIMEOUT.total_seconds(),
    )


def sipnet_model() -> SIPNETModel:
    """The model a run is one call of: the runner, the base parameters and the drivers."""
    return SIPNETModel(
        sipnet_runner(),
        base_params=base_sipnet_parameters(),
        base_climate=climate_drivers(),
    )


def climate_drivers() -> ClimateDrivers:
    """The prepared driver file, opened without reading it.

    File-backed, as the forward model requires under a parallel backend; the
    labels are validated at the first read. Run ``scripts/prepare_drivers.py``
    first.
    """
    directory = (
        config.PREPARED_DRIVERS_ROOT
        / f"ERA5_{config.SITE}_{config.DRIVER_SOURCE_INDEX}"
    )
    (path,) = sorted(directory.glob("ERA5.*.clim"))
    return ClimateDrivers.from_path(path, time_zone=config.DRIVER_TIME_ZONE)


def initial_state() -> xr.Dataset:
    """The site's initial conditions, each state at its median over the members.

    External inputs on ``site`` alone, in the processed file's units, under
    the names ``ComputeInitialConditions`` reads them by. Each state's median
    is taken separately, so they need not be one member's. The calibration
    reads the states it does not calibrate from here (``prior.external_inputs``).
    """
    fields = inputs.initial_condition_fields()
    return xr.Dataset(
        {
            name: fields[name]
            .median("initial_condition_member", keep_attrs=True)
            .astype("float64")
            for name in INITIAL_STATE_NAMES
        }
    )


def forward_model(
    parameter_vector: ParameterVector,
    sipnet_parameter_map: SIPNETParameterMap,
    *,
    observation_vector: ObservationVector | None = None,
    output_variable_names: Sequence[str] | None = None,
    freq: str | None = None,
    external_inputs: xr.Dataset | None = None,
) -> ForwardModel:
    """The forward model over the site, on the configured number of local workers.

    Give *observation_vector* for predictions ``(J, N)``, or
    *output_variable_names* (and optionally *freq*) for model output.
    *external_inputs* default to the site's whole initial state
    (:func:`initial_state`); a calibration that calibrates some initial
    states passes the rest.
    """
    return ForwardModel(
        sipnet_model(),
        parameter_vector,
        sipnet_parameter_map,
        climate={config.SITE: climate_drivers()},
        backend=LocalBackend(n_workers=config.N_WORKERS),
        external_inputs=initial_state() if external_inputs is None else external_inputs,
        observation_vector=observation_vector,
        output_variable_names=output_variable_names,
        freq=freq,
    )

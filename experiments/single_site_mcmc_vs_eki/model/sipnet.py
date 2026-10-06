"""What running SIPNET needs, built from the configuration.

Each function builds one piece from ``config``, so every script runs SIPNET
the same way: the base SIPNET parameters, the runner and the model, the
drivers, the site's initial state, and the runs that execute a batch through
PyEns.

Functions
---------
:func:`base_sipnet_parameters`
    The base parameter set every SIPNET parameter the map leaves unset takes.
:func:`sipnet_runner`, :func:`sipnet_model`
    pySIPNET's runner with the configured flags and run settings, and the
    model over it with the base parameters and the drivers.
:func:`climate_drivers`
    The prepared driver file, file-backed, as the runs need it.
:func:`initial_state`
    The site's initial conditions as external inputs, on ``site`` alone.
:func:`sipnet_runs`
    :class:`~sipnet_calibration.forward.SIPNETRuns` at the site: SIPNET once
    per sample, reduced to an observation vector's predictions (the forward
    map, ``model/calibration.py``'s simulator) or to model output.
"""

from typing import Literal

import xarray as xr
from pyens import LocalBackend
from pysipnet.climate import ClimateDrivers
from pysipnet.model import SIPNETModel
from pysipnet.parameters.model import SIPNETParameters
from pysipnet.runner import SIPNETRunner

from sipnet_calibration.forward import SIPNETRuns
from sipnet_calibration.sipnet_parameter_map import (
    INITIAL_STATE_NAMES,
    SIPNETParameterMap,
)
from sipnet_calibration.site_dims import SiteDims

from .. import config
from . import inputs

__all__ = [
    "base_sipnet_parameters",
    "climate_drivers",
    "initial_state",
    "sipnet_model",
    "sipnet_runner",
    "sipnet_runs",
]


def base_sipnet_parameters() -> SIPNETParameters:
    """``config.BASE_SIPNET_PARAMETER_FILE`` as pySIPNET parameters, read by pySIPNET."""
    return SIPNETParameters.from_param_file(config.BASE_SIPNET_PARAMETER_FILE)


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

    File-backed, as the runs require under a parallel backend; the
    labels are validated at the first read. Run ``run/prepare_drivers.py``
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
    the names ``initial_condition_rules`` read them by. Each state's median
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


def sipnet_runs(
    sipnet_parameter_map: SIPNETParameterMap,
    site_dims: SiteDims,
    *,
    out_of_domain: Literal["raise", "fail_row"] = "fail_row",
) -> SIPNETRuns:
    """SIPNET at the site, on the configured number of local workers.

    *sipnet_parameter_map* and *site_dims* are the calibration's
    (``model/prior.py``). *out_of_domain* is the runs': ``"fail_row"``, the
    default, marks a sample whose SIPNET parameters leave pySIPNET's domain
    invalid, as a failed run is; ``"raise"`` refuses the batch.
    """
    return SIPNETRuns(
        sipnet_model(),
        sipnet_parameter_map=sipnet_parameter_map,
        site_dims=site_dims,
        climate={config.SITE: climate_drivers()},
        backend=LocalBackend(n_workers=config.N_WORKERS),
        out_of_domain=out_of_domain,
    )

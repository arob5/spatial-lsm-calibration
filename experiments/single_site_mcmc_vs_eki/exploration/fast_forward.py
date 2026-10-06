"""A fast forward model for exploration: SIPNET in a process pool, predictions by index arithmetic.

**An exploration tool, not the calibration's forward map.** The calibration
runs through :class:`~sipnet_calibration.forward.SIPNETSimulator`, whose
observation operators are general. Here every operator of this experiment is
specialized to what its observations are at this site: each NEE window
covers exactly four three-hourly steps, each LAI label falls in one known
step, each LandTrendr year and the soil carbon average whole steps. So each
prediction is a sum or a pick over a precomputed index, which is far cheaper
than the library's window reductions, and :func:`check_fast_predictions`
confirms it equals ``ObservationVector.predict`` on a real run.

Functions
---------
:class:`FastPredictor`
    Built once from an observation vector at the site and the model's time
    axis; turns one run's output arrays into predictions, ``(N,)``, source
    by source in the vector's order, each in time order.
:func:`run_batch`
    SIPNET for each of a batch of SIPNET overrides, in a process pool, each
    run's predictions and annual summaries returned.
:func:`check_fast_predictions`
    One run's fast predictions against the library's.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from pysipnet.units import conversion_factor

from sipnet_calibration.conventions import SITE, TIME, WINDOW_END, WINDOW_START
from sipnet_calibration.fields import to_model_output
from sipnet_calibration.observation import ObservationVector

from .. import config
from ..model import inputs, sipnet

__all__ = [
    "ANNUAL_SUMMARY_NAMES",
    "FastPredictor",
    "check_fast_predictions",
    "run_batch",
]

#: The output variables a fast run reads.
OUTPUT_VARIABLE_NAMES = (
    "net_ecosystem_exchange",
    "gross_primary_production",
    "ecosystem_respiration",
    "above_ground_respiration",
    "root_respiration",
    "heterotrophic_respiration",
    "leaf_carbon",
    "wood_carbon",
    "fine_root_carbon",
    "coarse_root_carbon",
    "soil_carbon",
)

#: The annual summaries each fast run returns, per calendar year of the record:
#: flux totals in g C m-2 yr-1 and pool maxima or year-end values in g C m-2.
ANNUAL_SUMMARY_NAMES = (
    "gpp",
    "respiration",
    "above_ground_respiration",
    "root_respiration",
    "heterotrophic_respiration",
    "nee",
    "leaf_carbon_max",
    "wood_carbon_end",
    "fine_root_carbon_end",
    "coarse_root_carbon_end",
    "soil_carbon_end",
)

#: g C m-2 d-1 to umol CO2 m-2 s-1.
_RATE_TO_OBSERVED = conversion_factor(
    units="g m-2 d-1", constituent="C", to_units="umol m-2 s-1", to_constituent="CO2"
)


@dataclass(frozen=True)
class FastPredictor:
    """Precomputed step indices for each observation of a vector.

    Build with :meth:`from_vector`; call :meth:`predict` on one run's arrays.
    """

    source_names: tuple[str, ...]
    kinds: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    step_lengths_days: np.ndarray

    @classmethod
    def from_vector(
        cls, vector: ObservationVector, step_ends: pd.DatetimeIndex
    ) -> "FastPredictor":
        """The indices of every source of *vector* on the model's step ends."""
        step_starts = step_ends - pd.Timedelta(hours=3)
        names, kinds, indices = [], [], []
        for name in vector:
            observed = vector[name].observed_values.squeeze()
            if name in config.NEE_WINDOWS or name == "landtrendr_aboveground_biomass":
                left = pd.DatetimeIndex(observed[WINDOW_START].values)
                right = pd.DatetimeIndex(observed[WINDOW_END].values)
                first = step_ends.searchsorted(left, side="right")
                last = step_ends.searchsorted(right, side="right")
                kind = "nee" if name in config.NEE_WINDOWS else "wood"
                indices.append(np.stack([first, last], axis=1))
            elif name == "modis_leaf_area_index":
                labels = pd.DatetimeIndex(observed[TIME].values)
                indices.append(step_starts.searchsorted(labels, side="left") - 1)
                kind = "lai"
            elif name == "soilgrids_soil_organic_carbon":
                indices.append(np.array([[0, len(step_ends)]]))
                kind = "soil"
            else:
                raise KeyError(f"no fast operator for {name!r}")
            names.append(name)
            kinds.append(kind)
        lengths = np.diff(np.concatenate([[step_starts[0].value], step_ends.asi8])) / (
            86400e9
        )
        return cls(tuple(names), tuple(kinds), tuple(indices), lengths)

    def predict(self, arrays: dict, leaf_carbon_per_area: float) -> np.ndarray:
        """One run's predictions, source by source, in the observed units."""
        pieces = []
        cumulative = {
            name: np.concatenate([[0.0], np.cumsum(arrays[name] * weight)])
            for name, weight in (
                ("net_ecosystem_exchange", 1.0),
                ("wood_carbon", self.step_lengths_days),
                ("soil_carbon", self.step_lengths_days),
            )
        }
        length_cumulative = np.concatenate([[0.0], np.cumsum(self.step_lengths_days)])
        for kind, index in zip(self.kinds, self.indices, strict=True):
            if kind == "lai":
                pieces.append(arrays["leaf_carbon"][index] / leaf_carbon_per_area)
                continue
            first, last = index[:, 0], index[:, 1]
            span = length_cumulative[last] - length_cumulative[first]
            if kind == "nee":
                total = cumulative["net_ecosystem_exchange"]
                pieces.append(_RATE_TO_OBSERVED * (total[last] - total[first]) / span)
            elif kind == "wood":
                total = cumulative["wood_carbon"]
                mean = (total[last] - total[first]) / span
                pieces.append(1e-2 * mean / config.WOOD_CARBON_FRACTION)
            else:
                total = cumulative["soil_carbon"]
                pieces.append(1e-2 * (total[last] - total[first]) / span)
        return np.concatenate(pieces)


def run_batch(
    overrides_batch: list[dict[str, float]],
    predictor: FastPredictor | None = None,
    *,
    n_workers: int | None = None,
) -> list[dict]:
    """SIPNET at each of *overrides_batch*, in a process pool.

    Each element of the result is ``{"predictions": (N,) or None, "annual":
    DataFrame of ANNUAL_SUMMARY_NAMES by year, "error": str or None}``; a run
    that fails carries its error and NaN predictions.
    """
    n_workers = n_workers or config.N_WORKERS
    with ProcessPoolExecutor(n_workers, initializer=_start_worker) as pool:
        return list(
            pool.map(_run_one, overrides_batch, [predictor] * len(overrides_batch))
        )


def check_fast_predictions(
    vector: ObservationVector, sipnet_result, overrides: dict, predictor: FastPredictor
) -> float:
    """The largest absolute difference between the fast and the library predictions."""
    model_output = to_model_output(
        sipnet_result,
        output_variable_names=list(vector.output_variable_names),
        site=config.SITE,
        site_table=inputs.site_table(),
    )
    parameters = xr.Dataset(
        {
            n: sipnet_result.parameters.dataarray(n)
            for n in vector.sipnet_parameter_names_read
        },
        coords={n: model_output[n] for n in (SITE, "lon", "lat")},
    )
    predicted = vector.predict(model_output, sipnet_parameter_fields=parameters)
    # At one site each source's field holds its observations alone, in time order.
    library = np.concatenate([np.ravel(predicted[name].values) for name in vector])
    fast = predictor.predict(
        _arrays_of(sipnet_result), overrides["leaf_carbon_per_area"]
    )
    return float(np.max(np.abs(library - fast)))


# ── the workers ──

_MODEL = None


def _start_worker() -> None:
    """Build the model once per worker process."""
    import warnings

    warnings.filterwarnings("ignore")
    global _MODEL
    _MODEL = sipnet.sipnet_model()


def _run_one(overrides: dict, predictor: FastPredictor | None) -> dict:
    """One run's predictions and annual summaries."""
    try:
        result = _MODEL(**overrides)
        arrays = _arrays_of(result)
    except Exception as error:  # a failed run is data here, not a crash
        return {
            "predictions": None,
            "annual": None,
            "error": f"{type(error).__name__}: {error}"[:300],
        }
    predictions = (
        None
        if predictor is None
        else predictor.predict(arrays, overrides["leaf_carbon_per_area"])
    )
    return {
        "predictions": predictions,
        "annual": _annual_summaries(arrays),
        "error": None,
    }


def _arrays_of(sipnet_result) -> dict:
    """The read output variables of one run as NumPy arrays, with the step ends."""
    dataset = sipnet_result.outputs.xarray
    arrays = {name: dataset[name].to_numpy() for name in OUTPUT_VARIABLE_NAMES}
    arrays["time"] = pd.DatetimeIndex(dataset[TIME].values)
    return arrays


def _annual_summaries(arrays: dict) -> pd.DataFrame:
    """Per calendar year of step ends: flux totals, pool maxima and year-end values."""
    year = arrays["time"].year
    frame = pd.DataFrame(
        {
            "gpp": arrays["gross_primary_production"],
            "respiration": arrays["ecosystem_respiration"],
            "above_ground_respiration": arrays["above_ground_respiration"],
            "root_respiration": arrays["root_respiration"],
            "heterotrophic_respiration": arrays["heterotrophic_respiration"],
            "nee": arrays["net_ecosystem_exchange"],
            "leaf_carbon_max": arrays["leaf_carbon"],
            "wood_carbon_end": arrays["wood_carbon"],
            "fine_root_carbon_end": arrays["fine_root_carbon"],
            "coarse_root_carbon_end": arrays["coarse_root_carbon"],
            "soil_carbon_end": arrays["soil_carbon"],
        },
        index=year,
    )
    grouped = frame.groupby(level=0)
    sums = grouped[
        [
            "gpp",
            "respiration",
            "above_ground_respiration",
            "root_respiration",
            "heterotrophic_respiration",
            "nee",
        ]
    ].sum()
    return sums.join(grouped[["leaf_carbon_max"]].max()).join(
        grouped[
            [
                "wood_carbon_end",
                "fine_root_carbon_end",
                "coarse_root_carbon_end",
                "soil_carbon_end",
            ]
        ].last()
    )

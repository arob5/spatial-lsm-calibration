"""The observation vector as components of a probabilistic model.

Where this sits
---------------
::

    observation.ObservationVector      sources, observation dims, constants
      -> observation.model             this module: their components
      -> probability.joint(...)        a model over them, bound at
                                       ``observation_vector.coords``

It is the part of :mod:`sipnet_calibration.observation` that needs the
probability layer, and so TFP: ``observation/__init__`` does not import it,
so the data sources and the PyEns workers, which import the observation
package, load neither.

What it reads
-------------
An :class:`~sipnet_calibration.observation.vector.ObservationVector`, and
for a noise factor a covariance spec
(:mod:`~sipnet_calibration.probability.covariance`).

Data model
----------
For each observation source ``k``, in the vector's order, two
:class:`~sipnet_calibration.probability.spec.ArraySpec`\\ s, each a scalar
on :data:`~sipnet_calibration.probability.support.REAL`, indexed by the
source's observation dim (``observation_vector.observation_dim_name(k)``)
and in the observed values' ``units``:

- its **observed component**, named for the source, whose value
  ``observation_vector.observed_values_by_component()`` holds;
- its **prediction**, ``observation_vector.prediction_name(k)``: the forward
  model's value of the observed quantity, which a simulator outputs.

The observed values' ``constituent`` is not carried, since an
``ArraySpec`` has none: a prediction is converted into the observed values'
units, constituent included, before it is one.

A source's **noise factor** is the law of its observed component given its
prediction,

.. math::

    y_k \\mid m_k, \\phi \\sim \\mathcal N\\big(m_k,\\ \\Sigma_k(\\phi)\\big),

a :class:`~sipnet_calibration.probability.parts.FactorSpec` whose law is a
:class:`~sipnet_calibration.probability.parts.GaussianSpec`, holding the
source's constants (``observation_vector.constants(k)``) that its
covariance reads.

Functions
---------
:func:`observed_components`, :func:`prediction_components`
    The two, one per source.
:func:`noise_factor`
    A source's noise factor.

Usage
-----
::

    from sipnet_calibration.observation.model import observed_components, prediction_components

    observed_components(observation_vector)    # (ArraySpec("modis_leaf_area_index", ...), ...)
    prediction_components(observation_vector)  # (ArraySpec("predicted_modis_leaf_area_index", ...), ...)

    def block(time_since_epoch, standard_deviation):
        lag = jnp.abs(time_since_epoch[:, None] - time_since_epoch[None, :]) / 86_400.0
        return jnp.diag(jnp.maximum(standard_deviation, 0.66) ** 2) + 0.5**2 * jnp.exp(-lag / 30.0)

    noise_factor(
        observation_vector, "modis_leaf_area_index",
        covariance=BlockDiagonalSpec(DenseSpec(block), by="site"),
        provenance="MCD15A3H LAI_StdDev floored at 0.66; discrepancy 0.5, 30 d (reasoned).",
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import xarray as xr

from sipnet_calibration.observation.vector import (
    CALENDAR_YEAR,
    OBSERVED,
    STANDARD_DEVIATION,
    TIME_SINCE_EPOCH,
    WINDOW_LENGTH,
    ObservationVector,
)
from sipnet_calibration.probability import REAL, ArraySpec, CovarianceSpec, FactorSpec, GaussianSpec
from sipnet_calibration.validation import as_names, truncated

__all__ = [
    "noise_factor",
    "observed_components",
    "prediction_components",
]

#: The names of the constants a source may have (``ObservationVector.constants``).
_SOURCE_CONSTANT_NAMES = frozenset({OBSERVED, STANDARD_DEVIATION, TIME_SINCE_EPOCH, CALENDAR_YEAR, WINDOW_LENGTH})


def observed_components(observation_vector: ObservationVector) -> tuple[ArraySpec, ...]:
    """The observed components, one per source, in the vector's order, as
    the module docstring's data model has them."""
    return tuple(
        _component(observation_vector, name, name) for name in observation_vector.observation_source_names
    )


def prediction_components(observation_vector: ObservationVector) -> tuple[ArraySpec, ...]:
    """The prediction components, one per source, in the vector's order: a
    forward model's outputs, as the module docstring's data model has them."""
    return tuple(
        _component(observation_vector, name, observation_vector.prediction_name(name))
        for name in observation_vector.observation_source_names
    )


def noise_factor(
    observation_vector: ObservationVector,
    observation_source_names: str | Sequence[str],
    /,
    *,
    covariance: CovarianceSpec,
    constants: Mapping[str, xr.DataArray] | None = None,
    label_maps: Mapping[str, xr.DataArray] | None = None,
    provenance: str | None = None,
) -> FactorSpec:
    """The noise factor of one source, its observed component centered on its
    prediction, as the module docstring's data model has it.

    Parameters
    ----------
    observation_vector:
        Positional-only.
    observation_source_names:
        Positional-only. One source, by name.
    covariance:
        Keyword-only. :math:`\\Sigma_k`; ``BlockDiagonalSpec(..., by="site")``
        gives one block per site. Of the source's constants
        (:meth:`ObservationVector.constants <sipnet_calibration.observation.vector.ObservationVector.constants>`),
        the factor holds those it reads; the components it names, such as a
        noise variance, are read.
    constants, label_maps:
        Keyword-only. More, beside the source's, each read; a label map such
        as ``observation_vector.year_label_map(k)``.
    provenance:
        Keyword-only. Where the noise model comes from.

    Returns
    -------
    FactorSpec
        Over the source's observed component, its law
        ``GaussianSpec(mean=observation_vector.prediction_name(k), covariance=covariance)``.

    Raises
    ------
    KeyError
        If the source is not in the vector.
    TypeError
        As :class:`~sipnet_calibration.probability.parts.FactorSpec` and
        :class:`~sipnet_calibration.probability.parts.GaussianSpec`, and if
        *covariance* is not a covariance spec or *constants* not a mapping.
    ValueError
        As those, and for more than one source; a constant named like one
        of the source's that the covariance reads; or a covariance reading a
        source constant the source does not have, such as a standard
        deviation it was not given.

    Notes
    -----
    The factor holds the observed values as the constant ``observed``, when
    the covariance reads it. Conditioning on other values (synthetic data)
    then needs a factor made from
    ``observation_vector.with_observed_values(...)``, or the covariance
    follows the real data; and a covariance built from the data makes the
    evidence an empirical-Bayes quantity.

    A noise factor over several sources, whose errors are correlated, is not
    built: its event would span several observation dims.
    """
    names = (observation_source_names,) if isinstance(observation_source_names, str) else as_names(
        observation_source_names, message_name="observation_source_names"
    )
    check_noise_factor_is_over_one_source(names)
    check_covariance_is_a_covariance_spec(covariance)
    check_constants_are_a_mapping(constants)
    (name,) = names
    event = _component(observation_vector, name, name)
    held = observation_vector.constants(name)
    extra = {} if constants is None else dict(constants)
    check_covariance_reads_constants_the_source_has(covariance, held, extra, message_name=name)
    source_constants = {constant_name: value for constant_name, value in held.items() if constant_name in covariance.reads}
    check_constants_are_named_apart_from_the_sources(source_constants, extra, message_name=name)
    return FactorSpec(
        event,
        law=GaussianSpec(mean=observation_vector.prediction_name(name), covariance=covariance),
        constants={**source_constants, **extra},
        label_maps=label_maps,
        provenance=provenance,
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _component(observation_vector: ObservationVector, observation_source_name: str, name: str) -> ArraySpec:
    """A scalar component on the source's observation dim, in its units."""
    return ArraySpec(
        name,
        units=observation_vector[observation_source_name].observed_values.attrs["units"],
        support=REAL,
        indexed_by=(observation_vector.observation_dim_name(observation_source_name),),
    )


# ── checks ────────────────────────────────────────────────────────────────────


def check_noise_factor_is_over_one_source(names: Sequence[str]) -> None:
    """A noise factor is one source's: one over several, whose event would
    span several observation dims, is not built."""
    if len(names) != 1:
        raise ValueError(
            f"a noise factor over {list(names)} is asked for; give one source. A noise factor whose errors "
            "are correlated across sources is not supported: give each source its own."
        )


def check_covariance_is_a_covariance_spec(covariance: object) -> None:
    """A noise factor's covariance is declared by a covariance spec."""
    if not isinstance(covariance, CovarianceSpec):
        raise TypeError(
            f"a noise factor's covariance is a {type(covariance).__name__}; give a covariance spec, such as "
            "DiagonalSpec(...) or BlockDiagonalSpec(DenseSpec(...), by='site')."
        )


def check_constants_are_a_mapping(constants: object) -> None:
    """A noise factor's further constants are ``{name: DataArray}``."""
    if constants is not None and not isinstance(constants, Mapping):
        raise TypeError(f"a noise factor's constants are a {type(constants).__name__}; give {{name: DataArray}}.")


def check_covariance_reads_constants_the_source_has(
    covariance: CovarianceSpec,
    held: Mapping[str, xr.DataArray],
    constants: Mapping[str, xr.DataArray],
    *,
    message_name: str,
) -> None:
    """A noise factor's covariance reads only the source constants its
    source has: a static source has no times, one without standard
    deviations none."""
    lacking = [n for n in covariance.reads if n in _SOURCE_CONSTANT_NAMES and n not in held and n not in constants]
    if lacking:
        raise ValueError(
            f"the covariance of {message_name!r}'s noise factor reads {truncated(lacking)}, which the source does "
            f"not have (it has {list(held)}); give the source a standard deviation, or read only what it has."
        )


def check_constants_are_named_apart_from_the_sources(
    source_constants: Mapping[str, xr.DataArray], constants: Mapping[str, xr.DataArray], *, message_name: str
) -> None:
    """A noise factor's own constants are not named like the source's that
    its covariance reads, which one name would then mean twice."""
    clashing = [name for name in constants if name in source_constants]
    if clashing:
        raise ValueError(
            f"the noise factor of {message_name!r} is given constants {truncated(clashing)}, named like the "
            "source's own constants its covariance reads; name them apart."
        )


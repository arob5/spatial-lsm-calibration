"""The probability layer: models declared in specs, bound to labels, and
conditioned on data, independent of the rest of the package.

It is being built beside :mod:`sipnet_calibration.parameters`, which it
replaces; until then the parameter layer reads its supports, coercion,
probe points, families and builders from here, through re-export shims.

Where this sits
---------------
::

    probability.support     Support, Interval, Simplex, PositiveDefinite; the default bijectors
    probability.names       the names the layer reserves
    probability.labels      coords (stacked dims too), constants, label maps
      -> probability.spec      ArraySpec: one component's declaration and T
      -> probability.layout    Layout: named arrays as one flat vector, and its forms
    probability.laws        Law, as_law, pushforward; GaussianLaw
      -> probability.families  one value's law: log_normal, ..., normal, inverse_gamma, inverse_wishart
      -> probability.builders  a law over a block: iid_over_dim, independent_over_dim, gaussian_copula
    probability.covariance  covariance specs: DiagonalSpec, DenseSpec, ..., BlockDiagonalSpec
      -> probability.parts     FactorSpec, GaussianSpec, DeterministicSpec, their decorators; Simulator
      -> probability.model     joint -> ModelSpec, bind -> FactoredDistribution
      -> probability.posterior condition_on -> Posterior: theta's density
    ──────── seam: labeled values, a dict of DataArrays ────────
      -> the adapter layer

No module here imports from ``sipnet_calibration`` outside ``probability``,
which ``tests/test_package.py`` enforces, and only the private
``probability._linalg`` imports pyEKI, whose operators and ``Gaussian`` the
Gaussian laws are built on. It computes in ``float64``, which importing the
package turns on.

Modules
-------
:mod:`~sipnet_calibration.probability.support`
    The sets values lie in, and their default bijections.
:mod:`~sipnet_calibration.probability.names`
    The names the layer gives things, which nothing else may take.
:mod:`~sipnet_calibration.probability.labels`
    Coords, and the constants and label maps functions read: what each is,
    and how it is read at the labels in use.
:mod:`~sipnet_calibration.probability.spec`
    One component: the shape and labels of one value, the dims it is
    indexed by, its support and transform.
:mod:`~sipnet_calibration.probability.layout`
    The layout: Flat, values by name and labeled values, the two spaces,
    selection, and labeled values as one netCDF-ready Dataset.
:mod:`~sipnet_calibration.probability.laws`
    What a factor evaluates to: the law protocol, the pushforward, and a
    Gaussian over a block with a structured covariance.
:mod:`~sipnet_calibration.probability.families`
    One value's law from a few interpretable numbers.
:mod:`~sipnet_calibration.probability.builders`
    A factor's law over a block, built for the labels in use.
:mod:`~sipnet_calibration.probability.covariance`
    How a Gaussian factor's covariance is built, as structure over labels:
    diagonals, dense blocks, sums, scalings, block-diagonal groupings, a
    matrix component's submatrices.
:mod:`~sipnet_calibration.probability.parts`
    Factors and deterministics: declarations of laws and computed
    components, and the keyword rule that says what they read; simulators,
    computed outside JAX for a batch of samples.
:mod:`~sipnet_calibration.probability.model`
    The declared model and the model bound to labels: sampling and the
    joint density.
:mod:`~sipnet_calibration.probability.posterior`
    Bayes' rule: the target an inference algorithm reads, and the
    likelihood as a Gaussian, when it is one.
"""

from sipnet_calibration.probability.builders import (
    Builder,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
)
from sipnet_calibration.probability.families import (
    InverseWishart,
    inverse_gamma,
    inverse_wishart,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    normal,
    softmax_normal,
)
from sipnet_calibration.probability.covariance import (
    BlockDiagonalSpec,
    CovarianceSpec,
    DenseSpec,
    DiagonalSpec,
    ScaledSpec,
    SubmatrixSpec,
    SumSpec,
)
from sipnet_calibration.probability.laws import GaussianLaw, Law, as_law, pushforward
from sipnet_calibration.probability.layout import (
    LabeledValues,
    Layout,
    ValuesByName,
    decode_labeled_values,
    encode_labeled_values,
    validate_labeled_values,
    validate_values_by_name,
)
from sipnet_calibration.probability.model import FactoredDistribution, ModelSpec, joint
from sipnet_calibration.probability.names import RESERVED_NAMES, SAMPLE
from sipnet_calibration.probability.parts import (
    DeterministicSpec,
    FactorSpec,
    GaussianSpec,
    Simulator,
    SimulatorOutput,
    deterministic,
    factor,
)
from sipnet_calibration.probability.posterior import (
    GaussianLikelihood,
    Posterior,
    PosteriorEvaluation,
    condition_on,
)
from sipnet_calibration.probability.spec import ArraySpec
from sipnet_calibration.probability.support import (
    DEFAULT_BIJECTORS,
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    POSITIVE_DEFINITE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
    PositiveDefinite,
    Simplex,
    Support,
    bijector_for,
)

__all__ = [
    "DEFAULT_BIJECTORS",
    "NON_NEGATIVE",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "POSITIVE_DEFINITE",
    "REAL",
    "RESERVED_NAMES",
    "SAMPLE",
    "SIMPLEX",
    "UNIT_INTERVAL",
    "ArraySpec",
    "BlockDiagonalSpec",
    "Builder",
    "CovarianceSpec",
    "DenseSpec",
    "DeterministicSpec",
    "DiagonalSpec",
    "FactorSpec",
    "FactoredDistribution",
    "GaussianLaw",
    "GaussianLikelihood",
    "GaussianSpec",
    "Interval",
    "InverseWishart",
    "LabeledValues",
    "Law",
    "Layout",
    "ModelSpec",
    "PositiveDefinite",
    "Posterior",
    "PosteriorEvaluation",
    "ScaledSpec",
    "Simplex",
    "Simulator",
    "SimulatorOutput",
    "SubmatrixSpec",
    "SumSpec",
    "Support",
    "ValuesByName",
    "as_law",
    "bijector_for",
    "condition_on",
    "decode_labeled_values",
    "deterministic",
    "encode_labeled_values",
    "factor",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
    "inverse_gamma",
    "inverse_wishart",
    "joint",
    "log_normal",
    "log_normal_from_interval",
    "log_normal_from_samples",
    "logit_normal",
    "logit_normal_from_interval",
    "logit_normal_from_samples",
    "normal",
    "pushforward",
    "softmax_normal",
    "validate_labeled_values",
    "validate_values_by_name",
]

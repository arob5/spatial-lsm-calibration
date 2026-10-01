"""The parameter layer: what is calibrated, what is computed from it, and
what is believed beforehand, independent of the rest of the package.

Where this sits
---------------
::

    parameters.support      Support, Interval, Simplex; the default bijectors
    parameters.labels       coords, constants, memberships: labeled values read as arrays
      -> parameters.parameter   Parameter: one value's description and T
      -> parameters.vector      ParameterVector: the layout, its three forms
      -> parameters.derived     DerivedParameters: y = f(x), over one vector
      -> parameters.prior       Prior: the density of theta
    ──────── seam: the labeled natural values, an xr.Dataset ────────
      -> site_dims, sipnet_parameter_map, forward   (the adapter layer)

No module here imports from ``sipnet_calibration`` outside ``parameters``,
which ``tests/test_package.py`` enforces: sites, site labels, SIPNET and the
project's reserved names are the adapter layer's. So the layer can be
replaced (by ProbPipe, say) by anything that produces the labeled natural
values. It computes in ``float64``, which importing the package turns on.

Modules
-------
:mod:`~sipnet_calibration.parameters.support`
    The sets values lie in, and their default bijections.
:mod:`~sipnet_calibration.parameters.labels`
    Coords, and the constants and memberships functions read: what each is,
    and how it is read at the labels in use.
:mod:`~sipnet_calibration.parameters.parameter`
    One array-valued unknown and its unconstrained counterpart: the shape
    of one value, and its block over the dims it is indexed by.
:mod:`~sipnet_calibration.parameters.vector`
    The layout: Flat, values by parameter and the labeled form, the two
    spaces, and selection.
:mod:`~sipnet_calibration.parameters.derived`
    Derived parameters, pure array functions of parameters, constants and
    memberships.
:mod:`~sipnet_calibration.parameters.prior`
    The prior over a vector, its terms and builders.
"""

from sipnet_calibration.parameters.derived import DerivedParameter, DerivedParameters
from sipnet_calibration.parameters.parameter import Parameter
from sipnet_calibration.parameters.prior import (
    GaussianMoments,
    Prior,
    PriorTerm,
    gaussian_copula,
    iid_over_dim,
    independent_over_dim,
    log_normal,
    log_normal_from_interval,
    log_normal_from_samples,
    logit_normal,
    logit_normal_from_interval,
    logit_normal_from_samples,
    softmax_normal,
)
from sipnet_calibration.parameters.support import (
    DEFAULT_BIJECTORS,
    NON_NEGATIVE,
    OPEN_UNIT_INTERVAL,
    POSITIVE,
    REAL,
    SIMPLEX,
    UNIT_INTERVAL,
    Interval,
    Simplex,
    Support,
    bijector_for,
)
from sipnet_calibration.parameters.vector import (
    ParameterDataset,
    ParameterVector,
    ValuesByParameter,
    check_parameter_vectors_share_a_layout,
    validate_parameter_dataset,
    validate_values_by_parameter,
)

__all__ = [
    "DEFAULT_BIJECTORS",
    "NON_NEGATIVE",
    "OPEN_UNIT_INTERVAL",
    "POSITIVE",
    "REAL",
    "SIMPLEX",
    "UNIT_INTERVAL",
    "DerivedParameter",
    "DerivedParameters",
    "GaussianMoments",
    "Interval",
    "Parameter",
    "ParameterDataset",
    "ParameterVector",
    "Prior",
    "PriorTerm",
    "Simplex",
    "Support",
    "ValuesByParameter",
    "bijector_for",
    "check_parameter_vectors_share_a_layout",
    "gaussian_copula",
    "iid_over_dim",
    "independent_over_dim",
    "log_normal",
    "log_normal_from_interval",
    "log_normal_from_samples",
    "logit_normal",
    "logit_normal_from_interval",
    "logit_normal_from_samples",
    "softmax_normal",
    "validate_parameter_dataset",
    "validate_values_by_parameter",
]

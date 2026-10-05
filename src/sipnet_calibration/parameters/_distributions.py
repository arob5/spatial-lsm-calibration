"""What the prior and the builders of prior functions both know of TFP's
distributions, moved to :mod:`sipnet_calibration.probability.laws` and
:mod:`sipnet_calibration.probability.builders`; re-exported here for the
parameter layer until it is removed. Private to the parameter layer.
"""

from sipnet_calibration.probability.builders import Builder as IndexShapedFunction
from sipnet_calibration.probability.laws import CARRIES_ITS_BIJECTOR, distribution_name

__all__ = [
    "CARRIES_ITS_BIJECTOR",
    "IndexShapedFunction",
    "distribution_name",
]

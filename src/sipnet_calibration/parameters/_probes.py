"""The probe points, moved to :mod:`sipnet_calibration.probability._probes`;
re-exported here until the parameter layer is removed."""

from sipnet_calibration.probability._probes import bijectors_agree, joint_probe_points, probe_points

__all__ = ["bijectors_agree", "joint_probe_points", "probe_points"]

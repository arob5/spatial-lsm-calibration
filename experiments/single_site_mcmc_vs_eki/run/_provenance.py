"""The record written beside a run's outputs: what code, packages and inputs made them.

A run's ``provenance.json`` names everything its outputs depend on beyond
their own directory, so a figure or a number can be traced back after the
code has moved on:

- ``git``: the repository's commit, and whether the working tree differed
  from it (``dirty``), with the paths that did;
- ``packages``: each companion package's installed version and, for a git
  install, the commit it was installed from (PEP 610's ``direct_url.json``);
- ``sipnet``: the SIPNET tag and commit pySIPNET pins, and the binary's path
  and MD5;
- ``command``: the command line and the Python that ran it;
- ``config``: ``repr`` of every public constant of ``config``;
- ``input_files``: the path, size and MD5 of each file the run read;
- ``written``: the UTC time the record was written.

A dirty tree is recorded, not refused: a record of what ran is worth more
than a refusal to run.

Beside it, :func:`write_calibration` writes the calibration's own record,
:data:`CALIBRATION_FILE_NAMES`: ``calibration_parts.csv``, one row per part
of the model (each prior factor, the forward map and each noise factor) with
its law, what it is given and its provenance; ``calibration_components.csv``,
one row per component and input with its role in the posterior; and
``calibration_sipnet_parameters.csv``, one row per SIPNET parameter written,
with its rule or fixed value.
"""

import json
import subprocess
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from pysipnet import build

from sipnet_calibration import (
    constraints,
    initial_conditions,
    sites,
    site_labels,
)
from sipnet_calibration import net_ecosystem_exchange as nee
from sipnet_calibration.io import file_md5, utc_timestamp
from sipnet_calibration.probability import Posterior

from .. import config
from ..model import prior
from ..models import SIMULATOR_NAME

__all__ = [
    "CALIBRATION_FILE_NAMES",
    "COMPANION_PACKAGE_NAMES",
    "model_input_files",
    "write_calibration",
    "write_provenance",
]

#: The packages whose versions a run records.
COMPANION_PACKAGE_NAMES = ("pysipnet", "pyens", "enskit")

#: The calibration's record: per part of the model, per component, and per
#: SIPNET parameter.
CALIBRATION_FILE_NAMES = (
    "calibration_parts.csv",
    "calibration_components.csv",
    "calibration_sipnet_parameters.csv",
)


def write_provenance(path: Path, *, input_files: Iterable[Path]) -> None:
    """Write the record of the current run to *path*, as JSON."""
    record = {
        "git": _git_state(),
        "packages": {name: _package_state(name) for name in COMPANION_PACKAGE_NAMES},
        "sipnet": _sipnet_state(),
        "command": {"argv": sys.argv, "python": sys.version},
        "config": {
            name: repr(getattr(config, name)) for name in sorted(config.__all__)
        },
        "input_files": [_file_state(Path(file)) for file in input_files],
        "written": utc_timestamp(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def write_calibration(directory: Path, posterior: Posterior) -> None:
    """Write the calibration's record, :data:`CALIBRATION_FILE_NAMES`, to *directory*."""
    simulator = posterior.simulators[SIMULATOR_NAME]
    tables = (
        posterior.model.describe(),
        posterior.describe(),
        simulator.runs.sipnet_parameter_map.describe(),
    )
    for name, table in zip(CALIBRATION_FILE_NAMES, tables, strict=True):
        table.to_csv(directory / name)


def model_input_files() -> list[Path]:
    """Every file a run of the calibration reads, in the order ``model`` reads them."""
    driver_directory = (
        config.PREPARED_DRIVERS_ROOT
        / f"ERA5_{config.SITE}_{config.DRIVER_SOURCE_INDEX}"
    )
    return [
        *sorted(driver_directory.glob("ERA5.*.clim")),
        sites.default_site_table_path(),
        *(
            nee.net_ecosystem_exchange_path(name)
            for name in (config.CALIBRATION_NEE_SERIES, config.VALIDATION_NEE_SERIES)
        ),
        *(constraints.constraint_path(name) for name in config.CONSTRAINT_NAMES),
        initial_conditions.default_processed_path(),
        site_labels.site_labels_path(prior.SITE_LABELS_NAME),
        prior.FIXED_SIPNET_PARAMETERS_FILE,
        Path(config.BASE_SIPNET_PARAMETER_FILE),
    ]


# ── helpers ──


def _git_state() -> dict:
    """The commit of the repository holding this file, and what differs from it."""
    directory = Path(__file__).resolve().parent
    commit = _git(directory, "rev-parse", "HEAD")
    changed = _git(directory, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": bool(changed),
        "changed_paths": [line[3:] for line in changed.splitlines()],
    }


def _git(directory: Path, *arguments: str) -> str:
    """The output of one git command run in *directory*."""
    return subprocess.run(
        ["git", *arguments],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_state(name: str) -> dict:
    """An installed package's version, and the git commit it came from if any."""
    distribution = metadata.distribution(name)
    direct_url = distribution.read_text("direct_url.json")
    state = {"version": distribution.version}
    if direct_url is not None:
        source = json.loads(direct_url)
        state["url"] = source.get("url")
        state["commit"] = source.get("vcs_info", {}).get("commit_id")
    return state


def _sipnet_state() -> dict:
    """The SIPNET pySIPNET pins, and the binary a run would use."""
    candidate = build.find_binary()
    binary = Path(candidate.path) if candidate is not None else None
    return {
        "pinned_tag": build.SIPNET_PINNED_TAG,
        "pinned_commit": build.SIPNET_PINNED_COMMIT,
        "binary": str(binary) if binary else None,
        "binary_md5": file_md5(binary) if binary else None,
    }


def _file_state(path: Path) -> dict:
    """A file's resolved path, size and MD5."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "md5": file_md5(resolved),
    }

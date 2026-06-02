"""Utilities for making experiment directories importable.

Every experiment follows the same layout: a top-level ``config.py`` that is
the single source of truth, plus ``notebooks/`` and ``run_mcmc.py`` that all
import from it. Since ``config.py`` is not installed as a package, scripts
and notebooks must add the experiment directory to ``sys.path``.

Use ``ensure_experiment_on_path(__file__)`` at the top of any script or
notebook that lives inside an experiment directory.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_experiment_on_path(anchor: str | Path) -> Path:
    """Add the experiment directory to ``sys.path`` if it is not already there.

    Parameters
    ----------
    anchor:
        Path to the calling file (pass ``__file__``). The experiment
        directory is derived as follows:

        * If ``anchor`` is inside a ``notebooks/`` subdirectory, the
          experiment directory is the parent of ``notebooks/``.
        * Otherwise the directory containing ``anchor`` is used directly.

    Returns
    -------
    Path
        The experiment directory that was added to ``sys.path``.

    Examples
    --------
    In ``run_mcmc.py`` (experiment root):

    .. code-block:: python

        from sipnet_calibration import ensure_experiment_on_path
        ensure_experiment_on_path(__file__)
        from config import build_prior, load_model, MCMC_KWARGS

    In a notebook (inside ``notebooks/``):

    .. code-block:: python

        from sipnet_calibration import ensure_experiment_on_path
        ensure_experiment_on_path(__file__)
        from config import build_prior, load_model, GROUND_TRUTH
    """
    anchor = Path(anchor).resolve()
    directory = anchor.parent

    # Step up one level when called from inside notebooks/
    if directory.name == "notebooks":
        directory = directory.parent

    path_str = str(directory)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    return directory

"""The EKI setups side by side: how the NEE error model moves the calibration.

Overview
--------
Draws ``figures/comparison.py``'s figures for the setups it compares,
``COMPARED_SETUPS``: each setup's posterior predictive seasonal cycle, the
parameters the error model moves, and the daytime residual's slow and fast
parts. ``MODEL.md``, "The error model and the fast-slow trade-off", reads
them.

Input data
----------
Each compared setup's observed-data run, ``output/eki/<setup>/observed``,
with its posterior predictive and diagnostics.

Output data
-----------
The figures, under ``config.FIGURE_DIRECTORY / "comparison"``.

Usage
-----
From the repository root::

    uv run python -m experiments.single_site_mcmc_vs_eki.run.compare_setups
"""

import sys

from ..figures.comparison import draw_comparison_figures

__all__ = ["main"]


# ── entry point ──


def main() -> int:
    """Draw the compared setups' figures."""
    try:
        draw_comparison_figures()
    except FileNotFoundError as error:
        print(
            f"error: {error}; run each setup's EKI and posterior predictive first",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

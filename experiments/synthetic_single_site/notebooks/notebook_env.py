"""Configure sys.path for notebooks in this experiment.

Import this module at the top of every notebook in this directory:

    import notebook_env  # noqa: F401  (side-effects only)
    from config import build_prior, load_model, GROUND_TRUTH, OUTPUT_DIR

This adds the experiment directory (parent of notebooks/) to sys.path so
that config.py and run_mcmc.py are importable without installing them.
"""
import sys
from pathlib import Path

_experiment_dir = Path(__file__).parent.parent.resolve()
if str(_experiment_dir) not in sys.path:
    sys.path.insert(0, str(_experiment_dir))

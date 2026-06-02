"""Configure sys.path for notebooks in this experiment.

Import this module at the top of every notebook in this directory:

    import notebook_env  # noqa: F401  (side-effects only)
    from config import build_prior, load_model, GROUND_TRUTH, OUTPUT_DIR

Delegates to sipnet_calibration.ensure_experiment_on_path so the logic for
finding the experiment root lives in one place.
"""
from sipnet_calibration import ensure_experiment_on_path

ensure_experiment_on_path(__file__)

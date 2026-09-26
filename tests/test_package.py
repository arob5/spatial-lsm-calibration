"""Tests for what importing ``sipnet_calibration`` itself does."""

from __future__ import annotations

import subprocess
import sys


def test_importing_any_module_turns_on_64_bit_jax():
    """The package's one import-time side effect, from its lightest module."""
    code = (
        "import jax; assert not jax.config.jax_enable_x64; "
        "import sipnet_calibration.conventions; print(jax.config.jax_enable_x64)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"


def test_the_package_re_exports_nothing():
    code = (
        "import sipnet_calibration as package; "
        "print(sorted(n for n in vars(package) if not n.startswith('_')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_the_package_declares_its_empty_public_api():
    import sipnet_calibration

    assert sipnet_calibration.__all__ == []


def test_the_data_sources_and_observation_do_not_import_the_parameter_vector():
    """initial_conditions and observation imported parameter_vector, and with it TFP and pyEKI."""
    code = (
        "import sys; import sipnet_calibration.initial_conditions, "
        "sipnet_calibration.observation; "
        "print(sorted(m for m in ('sipnet_calibration.parameter_vector', "
        "'tensorflow_probability', 'pyeki') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_the_operators_type_hints_resolve():
    """ObservedValues was imported for type checking only, so get_type_hints failed."""
    import typing

    from sipnet_calibration.observation import ObservationSource, check_operator
    from sipnet_calibration.observation.operators import ObservationOperator

    for annotated in (ObservationOperator.__call__, check_operator, ObservationSource):
        assert typing.get_type_hints(annotated)


def test_every_module_compiles_without_a_warning():
    """Four docstrings wrote an invalid escape, which Python 3.14 warns about."""
    import warnings
    from pathlib import Path

    import sipnet_calibration

    root = Path(sipnet_calibration.__file__).parent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for path in sorted(root.rglob("*.py")):
            compile(path.read_text(), str(path), "exec")

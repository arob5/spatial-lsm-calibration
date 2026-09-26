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
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"


def test_the_package_re_exports_nothing():
    code = (
        "import sipnet_calibration as package; "
        "print(sorted(n for n in vars(package) if not n.startswith('_')))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"

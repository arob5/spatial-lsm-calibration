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


def test_the_data_sources_and_observation_do_not_import_the_parameter_layer():
    """initial_conditions and observation import neither the parameter nor
    the probability layer, nor TFP and EnsKit."""
    code = (
        "import sys; import sipnet_calibration.initial_conditions, "
        "sipnet_calibration.observation; "
        "print(sorted(m for m in ('sipnet_calibration.parameters', "
        "'sipnet_calibration.probability', 'tensorflow_probability', 'enskit') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_every_public_type_hint_resolves():
    """Names imported for type checking only made get_type_hints raise NameError.

    ObservedValues in the operators, then SIPNETResult in fields. A name
    re-exported from another package, such as EnsKit's ``Gaussian`` in
    ``probability._linalg``, is that package's to resolve.
    """
    import importlib
    import inspect
    import pkgutil
    import typing

    import sipnet_calibration

    unresolved = []
    for module_info in pkgutil.walk_packages(
        sipnet_calibration.__path__, f"{sipnet_calibration.__name__}."
    ):
        module = importlib.import_module(module_info.name)
        for name in getattr(module, "__all__", ()):
            public = getattr(module, name)
            if not getattr(public, "__module__", "").startswith(f"{sipnet_calibration.__name__}."):
                continue
            annotated = [public] if inspect.isfunction(public) else []
            if inspect.isclass(public):
                annotated = [public, *filter(inspect.isfunction, vars(public).values())]
            for item in annotated:
                try:
                    typing.get_type_hints(item)
                except NameError as error:
                    unresolved.append(f"{module_info.name}.{name}: {error}")
    assert unresolved == []


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


def _imports_outside(package: str, allowed: tuple[str, ...]) -> list[str]:
    """The modules of the package that files under *package* import, by name
    or relatively, other than *package* and *allowed*."""
    import ast
    import importlib
    from pathlib import Path

    layer = importlib.import_module(package)
    own = (package, *allowed)
    outside = []
    for path in sorted(Path(layer.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.level == 0 else [f"relative level {node.level}"] * (node.level > 1)
            else:
                continue
            for name in names:
                inside = any(name == o or name.startswith(f"{o}.") for o in own)
                if name.startswith("relative") or (name.split(".")[0] == "sipnet_calibration" and not inside):
                    outside.append(f"{path.name}: {name}")
    return outside


def _loaded_outside(package: str, allowed: tuple[str, ...], companions: tuple[str, ...] = ("pysipnet", "pyens", "enskit")) -> str:
    """The modules of the package and of the *companions* that importing
    *package* loads, other than *package* and *allowed*, as printed."""
    own = (package, *allowed)
    code = (
        f"import sys; import {package}; own = {own!r}; "
        "print(sorted(m for m in sys.modules if (m.startswith('sipnet_calibration.') "
        "and not any(m == o or m.startswith(o + '.') for o in own)) or m.split('.')[0] in "
        f"{companions!r}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _files_importing(package: str, top_level: str) -> list[str]:
    """The files under *package* that import *top_level*, or a module of it."""
    import ast
    import importlib
    from pathlib import Path

    layer = importlib.import_module(package)
    found = []
    for path in sorted(Path(layer.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0
                else []
            )
            if any(name.split(".")[0] == top_level for name in names):
                found.append(path.name)
                break
    return found


def test_the_parameter_layer_imports_nothing_of_the_package_outside_itself():
    """The parameter layer is replaceable only while it is independent: no
    module under parameters/ imports another module of the package but the
    probability layer, whose supports, coercion and probe points it
    re-exports, and importing it loads none, nor pySIPNET or PyEns. EnsKit
    it loads only through the probability layer's shim."""
    allowed = ("sipnet_calibration.probability",)
    assert _imports_outside("sipnet_calibration.parameters", allowed) == []
    assert _loaded_outside("sipnet_calibration.parameters", allowed, ("pysipnet", "pyens")) == "[]"
    assert _files_importing("sipnet_calibration.parameters", "enskit") == []


def test_the_probability_layer_imports_nothing_of_the_package_outside_itself():
    """The probability layer imports no module of the package outside
    itself, by name or relatively, and importing it loads none, nor
    pySIPNET or PyEns."""
    assert _imports_outside("sipnet_calibration.probability", ()) == []
    assert _loaded_outside("sipnet_calibration.probability", (), ("pysipnet", "pyens")) == "[]"


def test_the_probability_layer_reaches_enskit_through_one_shim():
    """Only ``probability/_linalg.py`` imports EnsKit, so a change to
    EnsKit's linalg or Gaussian changes that file alone."""
    assert _files_importing("sipnet_calibration.probability", "enskit") == ["_linalg.py"]


def test_the_probability_layer_never_imports_numpyro_or_gpjax():
    """numpyro and GPJax are not dependencies: the layer recognizes their
    distributions by class name, so no file imports either and importing
    the package loads neither."""
    for top_level in ("numpyro", "gpjax"):
        assert _files_importing("sipnet_calibration", top_level) == []
    code = "import sys; import sipnet_calibration.probability; print(sorted(m for m in ('numpyro', 'gpjax') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_the_inference_adapters_read_only_a_posterior():
    """The inference package imports the probability layer, ``smc`` and
    ``validation`` and nothing else of the package, so it reads a posterior
    and knows nothing of SIPNET. Importing it loads no other module of the
    package but the ``io`` and ``conventions`` those two import, and no
    PyEns; pySIPNET it loads through ``conventions``, which reads dim names
    from it."""
    allowed = ("sipnet_calibration.probability", "sipnet_calibration.smc", "sipnet_calibration.validation")
    assert _imports_outside("sipnet_calibration.inference", allowed) == []
    loaded = (*allowed, "sipnet_calibration.io", "sipnet_calibration.conventions")
    assert _loaded_outside("sipnet_calibration.inference", loaded, ("pyens",)) == "[]"

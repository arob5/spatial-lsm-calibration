"""Tests for where ``sipnet_calibration.conventions`` finds the data.

Only :func:`~sipnet_calibration.conventions.tracked_data_root` is tested here;
every data source's own resolvers are tested in that data source's test file.
"""

from __future__ import annotations

from pathlib import Path

import sipnet_calibration
from sipnet_calibration import conventions

#: ``data/`` of this checkout, found from the package rather than from the
#: module under test.
CHECKOUT_DATA = Path(sipnet_calibration.__file__).resolve().parents[2] / "data"


def test_tracked_data_root_is_the_checkouts_whatever_the_variable_says(monkeypatch, tmp_path):
    monkeypatch.setenv(conventions.DATA_ROOT_ENV_VAR, str(tmp_path))
    assert conventions.data_root() == tmp_path
    assert conventions.tracked_data_root() == CHECKOUT_DATA
    monkeypatch.delenv(conventions.DATA_ROOT_ENV_VAR)
    assert conventions.tracked_data_root() == CHECKOUT_DATA == conventions.data_root()


def test_tracked_data_root_is_the_data_root_under_a_non_editable_install(non_editable_install):
    assert conventions.tracked_data_root() == non_editable_install == conventions.data_root()

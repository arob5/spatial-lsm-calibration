"""Tests for ``sipnet_calibration.io``: the ``.partial`` protocol and provenance."""

from __future__ import annotations

import hashlib
import re

import pytest

from sipnet_calibration.io import file_md5, partial_path, utc_timestamp, write_checked


def _write(text):
    return lambda partial: partial.write_text(text)


def _refuse(partial):
    raise ValueError("the check failed")


class TestWriteChecked:
    def test_a_passing_check_moves_the_file_into_place_and_leaves_no_partial(self, tmp_path):
        out = tmp_path / "nested" / "product.csv"
        assert write_checked(out, _write("new\n"), lambda partial: None) == out
        assert out.read_text() == "new\n"
        assert not partial_path(out).exists()

    def test_the_check_reads_the_partial_file_not_the_destination(self, tmp_path):
        out = tmp_path / "product.csv"
        seen = []
        write_checked(out, _write("new\n"), lambda partial: seen.append(partial.read_text()))
        assert seen == ["new\n"]

    def test_a_failed_check_keeps_the_partial_and_prints_its_path(self, tmp_path, capsys):
        out = tmp_path / "product.csv"
        out.write_text("previous\n")
        with pytest.raises(ValueError, match="the check failed"):
            write_checked(out, _write("new\n"), _refuse)
        assert out.read_text() == "previous\n"
        assert partial_path(out).read_text() == "new\n"
        assert str(partial_path(out)) in capsys.readouterr().err

    def test_a_failed_write_keeps_what_it_wrote(self, tmp_path, capsys):
        out = tmp_path / "product.csv"

        def half_write(partial):
            partial.write_text("half")
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            write_checked(out, half_write, lambda partial: None)
        assert not out.exists() and partial_path(out).read_text() == "half"
        assert str(partial_path(out)) in capsys.readouterr().err

    def test_a_write_that_wrote_nothing_prints_nothing(self, tmp_path, capsys):
        def no_write(partial):
            raise OSError("no space")

        with pytest.raises(OSError):
            write_checked(tmp_path / "product.csv", no_write, lambda partial: None)
        assert capsys.readouterr().err == ""

    def test_a_rerun_overwrites_a_kept_partial(self, tmp_path):
        out = tmp_path / "product.csv"
        with pytest.raises(ValueError):
            write_checked(out, _write("bad\n"), _refuse)
        write_checked(out, _write("good\n"), lambda partial: None)
        assert out.read_text() == "good\n" and not partial_path(out).exists()


def test_the_partial_path_sits_beside_its_destination(tmp_path):
    assert partial_path(tmp_path / "a.nc") == tmp_path / "a.nc.partial"
    assert partial_path(tmp_path / "noext") == tmp_path / "noext.partial"


def test_file_md5_is_the_digest_of_the_whole_file(tmp_path):
    path = tmp_path / "blob"
    content = bytes(range(256)) * 10_000
    path.write_bytes(content)
    assert file_md5(path) == hashlib.md5(content).hexdigest()


def test_utc_timestamp_is_iso_8601_in_utc():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_timestamp())

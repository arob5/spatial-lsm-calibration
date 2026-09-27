"""Tests for ``sipnet_calibration.io``: the ``.partial`` protocol and provenance."""

from __future__ import annotations

import datetime
import hashlib
import re
import time

import pytest

from sipnet_calibration.io import (
    file_md5,
    partial_path,
    utc_timestamp,
    write_checked,
    write_checked_together,
)


def _write(text):
    return lambda partial: partial.write_text(text)


def _refuse(partial):
    raise ValueError("the check failed")


class TestWriteChecked:
    def test_a_passing_check_moves_the_file_into_place_and_leaves_no_partial(self, tmp_path):
        out = tmp_path / "nested" / "processed.csv"
        assert write_checked(out, _write("new\n"), lambda partial: None) == out
        assert out.read_text() == "new\n"
        assert not partial_path(out).exists()

    def test_the_check_reads_the_partial_file_not_the_destination(self, tmp_path):
        out = tmp_path / "processed.csv"
        seen = []
        write_checked(out, _write("new\n"), lambda partial: seen.append(partial.read_text()))
        assert seen == ["new\n"]

    def test_a_failed_check_keeps_the_partial_and_prints_its_path(self, tmp_path, capsys):
        out = tmp_path / "processed.csv"
        out.write_text("previous\n")
        with pytest.raises(ValueError, match="the check failed"):
            write_checked(out, _write("new\n"), _refuse)
        assert out.read_text() == "previous\n"
        assert partial_path(out).read_text() == "new\n"
        assert str(partial_path(out)) in capsys.readouterr().err

    def test_a_failed_write_keeps_what_it_wrote(self, tmp_path, capsys):
        out = tmp_path / "processed.csv"

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
            write_checked(tmp_path / "processed.csv", no_write, lambda partial: None)
        assert capsys.readouterr().err == ""

    def test_a_rerun_overwrites_a_kept_partial(self, tmp_path):
        out = tmp_path / "processed.csv"
        with pytest.raises(ValueError):
            write_checked(out, _write("bad\n"), _refuse)
        write_checked(out, _write("good\n"), lambda partial: None)
        assert out.read_text() == "good\n" and not partial_path(out).exists()

    def test_a_stale_partial_is_removed_before_an_early_failure_and_not_reported(
        self, tmp_path, capsys
    ):
        """A rerun that fails before it writes must not report the last run's
        file as its own."""
        out = tmp_path / "processed.csv"
        partial_path(out).write_text("STALE FROM LAST WEEK")

        def fails_first(partial):
            raise OSError("encoding could not be built")

        with pytest.raises(OSError, match="encoding"):
            write_checked(out, fails_first, lambda partial: None)
        assert not partial_path(out).exists()
        assert capsys.readouterr().err == ""

    def test_a_check_never_sees_a_stale_partial(self, tmp_path):
        out = tmp_path / "processed.csv"
        partial_path(out).write_text("stale")
        seen = []
        with pytest.raises(FileNotFoundError, match="did not write"):
            write_checked(out, lambda partial: None, lambda partial: seen.append(partial.exists()))
        assert seen == [] and not out.exists()

    def test_a_write_that_writes_the_destination_is_caught_and_reported_truthfully(
        self, tmp_path, capsys
    ):
        out = tmp_path / "processed.csv"
        out.write_text("previous\n")
        seen = []
        with pytest.raises(ValueError, match="changed .*processed.csv; a write writes only"):
            write_checked(out, lambda partial: out.write_text("direct\n"), seen.append)
        assert seen == []
        err = capsys.readouterr().err
        assert "unchanged" not in err
        assert f"{out} no longer holds its previous content" in err

    def test_a_write_that_writes_both_is_not_reported_as_leaving_the_destination(
        self, tmp_path, capsys
    ):
        out = tmp_path / "processed.csv"
        out.write_text("previous\n")

        def both(partial):
            partial.write_text("new\n")
            out.write_text("direct\n")

        with pytest.raises(ValueError, match="a write writes only"):
            write_checked(out, both, lambda partial: None)
        err = capsys.readouterr().err
        assert str(partial_path(out)) in err and "unchanged" not in err

    def test_a_failed_run_leaves_no_directory_it_made_empty(self, tmp_path):
        out = tmp_path / "new" / "nested" / "processed.csv"

        def no_write(partial):
            raise OSError("no space")

        with pytest.raises(OSError):
            write_checked(out, no_write, lambda partial: None)
        assert not (tmp_path / "new").exists()

    def test_a_failed_check_keeps_the_directory_holding_the_partial(self, tmp_path):
        out = tmp_path / "new" / "processed.csv"
        with pytest.raises(ValueError):
            write_checked(out, _write("bad\n"), _refuse)
        assert partial_path(out).read_text() == "bad\n"

    def test_an_interrupt_keeps_the_partial_and_prints_its_path(self, tmp_path, capsys):
        out = tmp_path / "processed.csv"

        def interrupted(partial):
            partial.write_text("half")
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            write_checked(out, interrupted, lambda partial: None)
        assert partial_path(out).read_text() == "half"
        assert str(partial_path(out)) in capsys.readouterr().err


class TestWriteCheckedTogether:
    def _files(self, tmp_path, texts, check=lambda partial: None):
        return [
            (tmp_path / f"{name}.txt", _write(text), check) for name, text in texts.items()
        ]

    def test_every_file_is_moved_in_once_every_check_passes(self, tmp_path):
        written = write_checked_together(self._files(tmp_path, {"a": "1", "b": "2"}))
        assert [path.read_text() for path in written] == ["1", "2"]
        assert not list(tmp_path.glob("*.partial"))

    def test_a_failed_second_check_moves_no_file_and_keeps_both_partials(
        self, tmp_path, capsys
    ):
        write_checked_together(self._files(tmp_path, {"a": "old a", "b": "old b"}))

        def refuse_b(partial):
            if partial.name.startswith("b"):
                raise ValueError("b is wrong")

        with pytest.raises(ValueError, match="b is wrong"):
            write_checked_together(self._files(tmp_path, {"a": "new a", "b": "new b"}, refuse_b))
        assert (tmp_path / "a.txt").read_text() == "old a"
        assert (tmp_path / "b.txt").read_text() == "old b"
        err = capsys.readouterr().err
        for name in ("a", "b"):
            assert str(partial_path(tmp_path / f"{name}.txt")) in err

    def test_one_destination_given_twice_is_refused_before_anything_is_written(
        self, tmp_path
    ):
        out = tmp_path / "a.txt"
        out.write_text("previous")
        with pytest.raises(ValueError, match="a.txt more than once"):
            write_checked_together(
                [
                    (out, _write("first"), lambda partial: None),
                    (tmp_path / "." / "a.txt", _write("second"), lambda partial: None),
                ]
            )
        assert out.read_text() == "previous" and not partial_path(out).exists()

    def test_every_write_runs_before_any_check(self, tmp_path):
        calls = []

        def write(name):
            return lambda partial: calls.append(f"write {name}") or partial.write_text(name)

        def check(name):
            return lambda partial: calls.append(f"check {name}")

        write_checked_together(
            [(tmp_path / name, write(name), check(name)) for name in ("a", "b")]
        )
        assert calls == ["write a", "write b", "check a", "check b"]

    def test_a_failed_rename_is_reported_with_which_file_is_new(
        self, tmp_path, monkeypatch, capsys
    ):
        from pathlib import Path

        write_checked_together(self._files(tmp_path, {"a": "old a", "b": "old b"}))
        real_replace = Path.replace

        def fail_on_b(self, target):
            if Path(target).name == "b.txt":
                raise OSError("disk full")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", fail_on_b)
        with pytest.raises(OSError, match="disk full"):
            write_checked_together(self._files(tmp_path, {"a": "new a", "b": "new b"}))
        monkeypatch.undo()

        assert (tmp_path / "a.txt").read_text() == "new a"
        assert (tmp_path / "b.txt").read_text() == "old b"
        assert partial_path(tmp_path / "b.txt").read_text() == "new b"
        err = capsys.readouterr().err
        assert "not all moved into place" in err and str(tmp_path / "a.txt") in err
        assert str(partial_path(tmp_path / "b.txt")) in err


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


def test_utc_timestamp_is_utc_whatever_the_local_time_zone(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        stamp = datetime.datetime.strptime(utc_timestamp(), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        assert abs((now - stamp).total_seconds()) < 120
    finally:
        monkeypatch.undo()
        time.tzset()

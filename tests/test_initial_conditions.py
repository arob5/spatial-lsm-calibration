"""Tests for the initial-condition schema, its reader and the ingest script.

Most cases run against small synthetic netCDF-3 files written to ``tmp_path``
in the real layout, because the three real files exercise only the happy path.
``scipy.io.netcdf_file`` writes genuine netCDF-3 classic with an unlimited
record dimension, so the fixtures are the same kind of file the source is, with
no new dependency.

The traps worth a file each: a variable the file does not carry; an explicit
``-999.0``; a ``NaN`` that is *not* the declared fill; a ``time`` of length 2;
a units string that disagrees with the registered one; a variable carrying a
layer dimension; a variable outside the schema, including the two that are
reported but unspecified; a file whose embedded site disagrees with its
directory; and a site missing one member of an otherwise complete ensemble.

The cases at the end run against the real files under
``data/raw/initial_conditions/`` and are skipped when they are absent. Those
are the ones that pin the format facts -- the record dimension, the fill value,
the units strings, the undecodable ``time`` template, the bitwise equality of
the two wood variables -- to the actual data rather than to a fixture that was
written to match.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import netcdf_file

from sipnet_calibration.constraints import CONSTRAINT_VARIABLES
from sipnet_calibration.constraints import (
    SOURCE_VARIABLE_NAMES as CONSTRAINT_SOURCE_NAMES,
)
from sipnet_calibration.sites import SITE_COLUMNS
from sipnet_calibration.initial_conditions import (
    DATA_ROOT_ENV_VAR,
    IC_FILE_TEMPLATE,
    IC_PRESENT,
    IC_VARIABLE_ATTRS,
    IC_VARIABLES,
    MEMBER_SOURCE,
    RELATED_CONSTRAINT_VARIABLES,
    SOURCE_FILL_VALUE,
    SOURCE_TIME_LONG_NAME,
    SOURCE_TIME_UNITS,
    SOURCE_TIME_VALUE,
    SOURCE_VARIABLE_NAMES,
    UNITS_STATUS,
    UNSPECIFIED_VARIABLES,
    VARIABLE_PRESENT,
    available_members,
    available_sites,
    default_ic_path,
    default_ic_root,
    ic_file,
    initial_condition_fields,
    load_initial_conditions,
    read_ic_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = REPO_ROOT / "data" / "raw" / "initial_conditions"

#: The three source variable names, in source file order.
SOURCE_NAMES = tuple(SOURCE_VARIABLE_NAMES)


def _load_ingest_module():
    """Import ``scripts/ingest_ic.py``, which is a script."""
    path = REPO_ROOT / "scripts" / "ingest_ic.py"
    spec = importlib.util.spec_from_file_location("ingest_ic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_ic"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_module()


# ── synthetic files ───────────────────────────────────────────────────────────


def _write_ic_file(
    path,
    values=None,
    *,
    units=None,
    long_names=None,
    fill_value=SOURCE_FILL_VALUE,
    fill_dtype=np.float64,
    time_length=1,
    time_units=SOURCE_TIME_UNITS,
    time_long_name=SOURCE_TIME_LONG_NAME,
    time_value=SOURCE_TIME_VALUE,
    write_time=True,
    extra_dims=None,
):
    """Write one synthetic netCDF-3 initial-condition file.

    Defaults reproduce a real file: an unlimited ``time`` of length 1 holding
    ``1.0``, the template units string, and ``float64`` scalars on
    ``("time",)`` declaring ``_FillValue = -999.0``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = {name: 1.0 + index for index, name in enumerate(SOURCE_NAMES)}
    units = units or {}
    long_names = long_names or {}
    extra_dims = extra_dims or {}

    with netcdf_file(str(path), "w") as dataset:
        dataset.createDimension("time", None)
        for name, size in extra_dims.items():
            dataset.createDimension(name, size)
        if write_time:
            time = dataset.createVariable("time", "d", ("time",))
            time[:] = np.arange(1, time_length + 1, dtype=np.float64) * time_value
            time.units = time_units
            time.long_name = time_long_name

        for name, value in values.items():
            dims = ("time",) + tuple(extra_dims)
            shape = (time_length,) + tuple(extra_dims.values())
            variable = dataset.createVariable(name, "d", dims)
            variable[:] = np.full(shape, value, dtype=np.float64)
            variable.units = units.get(
                name, IC_VARIABLE_ATTRS.get(SOURCE_VARIABLE_NAMES.get(name, ""), {})
                .get("units", "kg C m-2")
            )
            if fill_value is not None:
                variable._FillValue = fill_dtype(fill_value)
            variable.long_name = long_names.get(
                name,
                IC_VARIABLE_ATTRS.get(SOURCE_VARIABLE_NAMES.get(name, ""), {}).get(
                    "source_long_name", "Synthetic variable"
                ),
            )
    return path


@pytest.fixture
def write_ic_file():
    """A factory writing one synthetic netCDF-3 initial-condition file.

    Takes the destination, a mapping of source variable name to value, and
    optional overrides for the units, long names, fill value, time length,
    time metadata and per-variable dimensions, so that each check below gets a
    file that trips exactly it.

    The fill value is written as ``numpy.float64``, matching the real files:
    handed a Python float, the writer emits ``float32``, which is why the
    check compares the value numerically rather than by dtype.
    """
    return _write_ic_file


def _build_tree(root, pairs, **kwargs):
    """Write one file per ``(site, member)`` pair, in the real layout."""
    for site, member in pairs:
        _write_ic_file(
            Path(root) / str(site) / IC_FILE_TEMPLATE.format(site=site, member=member),
            values={
                name: float(site) + index / 10 + member / 100
                for index, name in enumerate(SOURCE_NAMES)
            },
            **kwargs,
        )
    return Path(root)


@pytest.fixture
def ic_tree(tmp_path):
    """A complete two-site, two-member tree in the real layout.

    The baseline the coverage checks are varied against, and the input the
    grid and dataset cases build from.
    """
    return _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1), (27, 2)])


@pytest.fixture
def site_table():
    """A small site table with the columns the ingest reads.

    Covers the sites in ``ic_tree`` plus some the tree has no directory for,
    so that the pool axis is wider than the data.
    """
    return pd.DataFrame(
        {
            "site_id": np.array([1, 5, 27, 40], dtype=np.int32),
            "lon": np.array([-24.5625, -30.0, -78.595834, -100.5], dtype=np.float64),
            "lat": np.array([82.545834, 70.0, 80.612501, 45.25], dtype=np.float64),
        }
    )


@pytest.fixture
def sites_csv(tmp_path, site_table):
    """``site_table`` written as a schema-complete CSV.

    The in-process fixture carries only the three columns the grid needs, but
    :func:`~sipnet_calibration.sites.load_sites` validates the whole site-table
    schema, so anything going through the script's ``main`` needs every column.
    """
    table = site_table.copy()
    table["lon_index"] = np.arange(table.shape[0], dtype=np.int32)
    table["lat_index"] = np.arange(table.shape[0], dtype=np.int32)
    table["site_name"] = "weighted_sample"
    table["site_order"] = np.zeros(table.shape[0], dtype=np.int32)
    table["cluster"] = np.ones(table.shape[0], dtype=np.int8)
    table["landcover"] = np.ones(table.shape[0], dtype=np.int8)
    table["ameriflux_site_id"] = ""
    path = tmp_path / "sites.csv"
    table[list(SITE_COLUMNS)].to_csv(path, index=False)
    return path


@pytest.fixture
def built(ic_tree, site_table):
    """The Dataset the ingest builds from ``ic_tree``, plus its parsed files."""
    index = ingest.discover_files(ic_tree)
    contents = ingest.read_all_files(index, jobs=2)
    grids = ingest.build_grids(contents, index, site_table)
    dataset = ingest.build_dataset(grids, index, allow_gaps=True)
    return dataset, contents, grids, index


# ── schema constants ──────────────────────────────────────────────────────────


class TestSchemaConstants:
    def test_every_source_variable_maps_to_a_processed_name_in_source_order(self):
        """``IC_VARIABLES`` is the registered source variables, renamed, in order."""
        assert IC_VARIABLES == tuple(SOURCE_VARIABLE_NAMES.values())
        assert len(IC_VARIABLES) == len(set(IC_VARIABLES))

    def test_processed_names_follow_the_naming_convention(self):
        """Lower case with underscores, no abbreviation, all prefixed ``initial_``."""
        for name in IC_VARIABLES:
            assert name == name.lower()
            assert " " not in name and "-" not in name
            assert name.startswith("initial_")

    def test_processed_names_do_not_collide_with_the_constraint_names(self):
        """No processed name is shared with ``constraints.SOURCE_VARIABLE_NAMES``.

        The collision this avoids is the reason for the prefix: both sources
        carry a variable named ``AbvGrndWood``, in units that differ by a
        factor of ten, and the ``VARIABLES`` registry holds one unit per name.
        """
        assert set(IC_VARIABLES).isdisjoint(CONSTRAINT_VARIABLES)
        # And the collision is real: the same source name in both sources.
        shared = set(SOURCE_VARIABLE_NAMES) & set(CONSTRAINT_SOURCE_NAMES)
        assert "AbvGrndWood" in shared

    def test_every_variable_has_the_full_attribute_set(self):
        """``IC_VARIABLE_ATTRS`` is complete for every variable."""
        for name in IC_VARIABLES:
            attrs = IC_VARIABLE_ATTRS[name]
            for key in (
                "units",
                "long_name",
                "source_name",
                "source_long_name",
                "aggregation",
            ):
                assert attrs[key], f"{name} is missing {key}"

    def test_source_names_round_trip_through_the_attributes(self):
        """Each entry's ``source_name`` is its key in ``SOURCE_VARIABLE_NAMES``."""
        for source, processed in SOURCE_VARIABLE_NAMES.items():
            assert IC_VARIABLE_ATTRS[processed]["source_name"] == source

    def test_every_variable_is_a_stock(self):
        """``aggregation`` is instantaneous throughout; these are pools, not totals."""
        for name in IC_VARIABLES:
            assert IC_VARIABLE_ATTRS[name]["aggregation"] == "instantaneous"

    def test_unspecified_variables_are_not_registered(self):
        """The two reported variables are absent from the schema, by design.

        Registering them would mean inventing a unit for a file nobody has
        seen. This test is what fails, deliberately, if someone adds them
        without the evidence.
        """
        assert UNSPECIFIED_VARIABLES == ("leaf_carbon_content", "SoilMoistFrac")
        for name in UNSPECIFIED_VARIABLES:
            assert name not in SOURCE_VARIABLE_NAMES

    def test_related_constraint_variables_name_real_constraint_variables(self):
        """Each counterpart is in ``constraints.CONSTRAINT_VARIABLES``."""
        for name, related in RELATED_CONSTRAINT_VARIABLES.items():
            assert name in IC_VARIABLES
            assert related["variable"] in CONSTRAINT_VARIABLES
            assert related["status"] in ("unconfirmed", "contradicted")
            assert related["unit_factor"] > 0

    def test_the_related_soil_variable_is_marked_contradicted(self):
        """The soil counterpart's status records evidence against identity.

        A reader who converts on the strength of a matching unit string would
        be wrong, and the status field is what says so.
        """
        soil = RELATED_CONSTRAINT_VARIABLES["initial_soil_organic_carbon"]
        assert soil["variable"] == "total_soil_carbon"
        assert soil["status"] == "contradicted"

    def test_the_wood_counterpart_carries_the_factor_of_ten(self):
        """kg C m-2 to Mg C ha-1, the conversion a caller would otherwise guess."""
        wood = RELATED_CONSTRAINT_VARIABLES["initial_aboveground_wood_carbon"]
        assert wood["variable"] == "aboveground_wood_carbon"
        assert wood["unit_factor"] == 10.0

    def test_units_status_says_the_units_are_only_the_source_attribute(self):
        assert UNITS_STATUS == "source_attribute"


# ── paths ─────────────────────────────────────────────────────────────────────


class TestPaths:
    def test_default_root_honors_the_data_root_variable(self, monkeypatch, tmp_path):
        """``$SIPNET_CALIBRATION_DATA`` relocates the raw directory."""
        monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
        assert default_ic_root() == tmp_path / "raw" / "initial_conditions"

    def test_default_path_honors_the_data_root_variable(self, monkeypatch, tmp_path):
        """And the written product, so the two move together."""
        monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
        assert default_ic_path() == tmp_path / "processed" / "ic.nc"

    def test_ic_file_resolves_the_pair(self, ic_tree):
        assert ic_file(ic_tree, 1, 2).name == "IC_site_1_2.nc"
        assert ic_file(ic_tree, 27, 1).parent.name == "27"

    def test_ic_file_raises_when_the_site_directory_is_absent(self, ic_tree):
        with pytest.raises(FileNotFoundError, match="no initial-condition directory"):
            ic_file(ic_tree, 999, 1)

    def test_ic_file_raises_when_the_member_file_is_absent(self, ic_tree):
        with pytest.raises(FileNotFoundError, match="no initial-condition file"):
            ic_file(ic_tree, 1, 77)

    def test_available_sites_lists_the_directories_in_order(self, ic_tree):
        """Sites come back ascending, from the directory names only."""
        assert available_sites(ic_tree) == (1, 27)

    def test_available_sites_ignores_filesystem_debris(self, ic_tree):
        """A ``.DS_Store`` and a non-numeric directory are skipped, not reported.

        Both shapes are present in the development checkout, so failing on
        them would make the script unrunnable there.
        """
        (ic_tree / ".DS_Store").write_bytes(b"junk")
        (ic_tree / "notes").mkdir()
        (ic_tree / "007").mkdir()  # leading zero is not the template
        assert available_sites(ic_tree) == (1, 27)

    def test_available_sites_of_a_missing_root_is_empty(self, tmp_path):
        assert available_sites(tmp_path / "nope") == ()

    def test_available_members_lists_the_files_in_order(self, ic_tree):
        assert available_members(ic_tree, 1) == (1, 2)

    def test_available_members_keeps_a_file_whose_site_disagrees(
        self, ic_tree, write_ic_file
    ):
        """A mismatched file is returned, so the layout check can report it.

        Filtering it out here would turn a misfiled file into a missing one.
        """
        write_ic_file(ic_tree / "27" / "IC_site_1_3.nc")
        assert 3 in available_members(ic_tree, 27)

    def test_available_members_ignores_a_name_off_the_template(
        self, ic_tree, write_ic_file
    ):
        """``IC_site_1_01.nc`` is not the template and is skipped."""
        write_ic_file(ic_tree / "1" / "IC_site_1_01.nc")
        assert available_members(ic_tree, 1) == (1, 2)


# ── read_ic_file ──────────────────────────────────────────────────────────────


class TestReadIcFile:
    def test_parses_the_variables_under_their_source_names(
        self, tmp_path, write_ic_file
    ):
        """Values exactly as written, keyed by source name, plus the metadata."""
        path = write_ic_file(
            tmp_path / "f.nc", values={name: 2.5 for name in SOURCE_NAMES}
        )
        contents = read_ic_file(path)
        assert set(contents.values) == set(SOURCE_NAMES)
        assert all(value == 2.5 for value in contents.values.values())
        assert contents.time_units == SOURCE_TIME_UNITS
        assert contents.time_long_name == SOURCE_TIME_LONG_NAME
        assert contents.time_value == SOURCE_TIME_VALUE
        assert contents.explicit_fills == frozenset()

    def test_masks_the_declared_fill_and_records_it(self, tmp_path, write_ic_file):
        """A ``-999.0`` becomes ``NaN`` and its name appears in ``explicit_fills``."""
        values = {name: 1.0 for name in SOURCE_NAMES}
        values["AbvGrndWood"] = SOURCE_FILL_VALUE
        path = write_ic_file(tmp_path / "f.nc", values=values)
        contents = read_ic_file(path)
        assert np.isnan(contents.values["AbvGrndWood"])
        assert contents.explicit_fills == frozenset({"AbvGrndWood"})

    def test_reports_only_the_variables_the_file_carries(
        self, tmp_path, write_ic_file
    ):
        """A file with two of the three variables yields two keys, not three."""
        path = write_ic_file(
            tmp_path / "f.nc",
            values={"AbvGrndWood": 1.0, "soil_organic_carbon_content": 2.0},
        )
        contents = read_ic_file(path)
        assert set(contents.values) == {"AbvGrndWood", "soil_organic_carbon_content"}

    def test_rejects_a_time_dimension_longer_than_one(self, tmp_path, write_ic_file):
        """Length 2 would mean these are not static initial conditions."""
        path = write_ic_file(tmp_path / "f.nc", time_length=2)
        with pytest.raises(ValueError, match="'time' has length 2"):
            read_ic_file(path)

    def test_rejects_a_file_without_a_time_variable(self, tmp_path, write_ic_file):
        path = write_ic_file(tmp_path / "f.nc", write_time=False)
        with pytest.raises(ValueError, match="no 'time' variable"):
            read_ic_file(path)

    def test_rejects_a_variable_with_a_layer_dimension(self, tmp_path, write_ic_file):
        """The soil variable's long name says "by Layer", so this is live.

        A layer-resolved variable must not be flattened into one cell
        silently.
        """
        path = write_ic_file(
            tmp_path / "f.nc",
            values={"soil_organic_carbon_content": 3.0},
            extra_dims={"layer": 4},
        )
        with pytest.raises(ValueError, match="expected \\('time',\\)"):
            read_ic_file(path)

    def test_rejects_a_variable_outside_the_schema(self, tmp_path, write_ic_file):
        path = write_ic_file(tmp_path / "f.nc", values={"something_new": 1.0})
        with pytest.raises(ValueError, match="unregistered variable"):
            read_ic_file(path)

    def test_names_the_blocker_for_an_unspecified_variable(
        self, tmp_path, write_ic_file
    ):
        """A file carrying ``SoilMoistFrac`` fails saying what is needed.

        The message points at the two unspecified variables and open question
        6, so a survey of the full ensemble stops with the answer rather than
        recording a guess.
        """
        path = write_ic_file(
            tmp_path / "f.nc", values={"AbvGrndWood": 1.0, "SoilMoistFrac": 40.0}
        )
        with pytest.raises(ValueError) as error:
            read_ic_file(path)
        message = str(error.value)
        assert "SoilMoistFrac" in message
        assert "not specified" in message
        assert "open question 6" in message

    @pytest.mark.parametrize("name", UNSPECIFIED_VARIABLES)
    def test_both_unspecified_variables_name_the_blocker(
        self, tmp_path, write_ic_file, name
    ):
        path = write_ic_file(tmp_path / "f.nc", values={name: 1.0})
        with pytest.raises(ValueError, match="open question 6"):
            read_ic_file(path)

    def test_rejects_a_file_with_no_data_variable(self, tmp_path, write_ic_file):
        path = write_ic_file(tmp_path / "f.nc", values={})
        with pytest.raises(ValueError, match="holds no data variable"):
            read_ic_file(path)

    def test_rejects_a_wrong_fill_value(self, tmp_path, write_ic_file):
        path = write_ic_file(tmp_path / "f.nc", fill_value=-9999.0)
        with pytest.raises(ValueError, match="declares _FillValue"):
            read_ic_file(path)

    def test_rejects_a_variable_declaring_no_fill_value(
        self, tmp_path, write_ic_file
    ):
        path = write_ic_file(tmp_path / "f.nc", fill_value=None)
        with pytest.raises(ValueError, match="declares no _FillValue"):
            read_ic_file(path)

    def test_accepts_a_fill_value_written_as_float32(self, tmp_path, write_ic_file):
        """The value is compared numerically; the attribute's dtype is not the point."""
        path = write_ic_file(tmp_path / "f.nc", fill_dtype=np.float32)
        contents = read_ic_file(path)
        assert set(contents.values) == set(SOURCE_NAMES)

    def test_rejects_units_that_disagree_with_the_registered_units(
        self, tmp_path, write_ic_file
    ):
        """A file in ``Mg C ha-1`` is refused rather than silently mis-scaled."""
        path = write_ic_file(
            tmp_path / "f.nc",
            values={"AbvGrndWood": 46.0},
            units={"AbvGrndWood": "Mg C ha-1"},
        )
        with pytest.raises(ValueError, match="Mg C ha-1"):
            read_ic_file(path)

    def test_rejects_a_non_finite_value_that_is_not_the_fill(
        self, tmp_path, write_ic_file
    ):
        """A source ``NaN`` would be indistinguishable from a fill once masked.

        The product's whole account of missingness rests on that distinction,
        so the read is unmasked and this case is refused.
        """
        path = write_ic_file(tmp_path / "f.nc", values={"AbvGrndWood": np.nan})
        with pytest.raises(ValueError, match="non-finite value"):
            read_ic_file(path)

    def test_rejects_a_file_that_is_not_netcdf3(self, tmp_path):
        path = tmp_path / "f.nc"
        path.write_bytes(b"not a netcdf file at all")
        with pytest.raises(ValueError, match="could not be read"):
            read_ic_file(path)

    def test_reads_a_netcdf3_file_that_h5netcdf_cannot_open(
        self, tmp_path, write_ic_file
    ):
        """The engine is pinned to ``scipy`` deliberately; ``h5netcdf`` fails here."""
        path = write_ic_file(tmp_path / "f.nc")
        with pytest.raises(Exception):
            xr.open_dataset(path, decode_times=False, engine="h5netcdf")
        assert read_ic_file(path).values  # the pinned engine reads it


# ── discovery and layout ──────────────────────────────────────────────────────


class TestDiscovery:
    def test_indexes_every_pair_in_the_tree(self, ic_tree):
        """Paths keyed by ``(site, member)`` with source member indices."""
        index = ingest.discover_files(ic_tree)
        assert set(index.paths) == {(1, 1), (1, 2), (27, 1), (27, 2)}
        assert index.paths[(1, 2)].name == "IC_site_1_2.nc"

    def test_discovers_the_union_of_members_across_sites(self, tmp_path):
        """The member axis is the union, so a ragged tree still has one axis."""
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 94)])
        index = ingest.discover_files(root)
        assert index.sites == (1, 27)
        assert index.members == (1, 2, 94)

    def test_raises_when_the_root_is_not_a_directory(self, tmp_path):
        with pytest.raises(ingest.IngestError, match="is not a directory"):
            ingest.discover_files(tmp_path / "nope")

    def test_raises_when_the_tree_holds_no_file(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ingest.IngestError, match="no initial-condition files"):
            ingest.discover_files(tmp_path / "empty")

    def test_ignores_debris_at_both_levels(self, ic_tree):
        """A ``.DS_Store`` beside the site directories and inside one."""
        (ic_tree / ".DS_Store").write_bytes(b"junk")
        (ic_tree / "1" / ".DS_Store").write_bytes(b"junk")
        index = ingest.discover_files(ic_tree)
        assert set(index.paths) == {(1, 1), (1, 2), (27, 1), (27, 2)}


class TestLayoutChecks:
    def test_accepts_a_conforming_tree(self, ic_tree):
        ingest.check_paths_follow_the_layout(ingest.discover_files(ic_tree))

    def test_rejects_a_file_whose_site_disagrees_with_its_directory(
        self, ic_tree, write_ic_file
    ):
        """``27/IC_site_1_3.nc`` is unattributable once the arrays are built."""
        write_ic_file(ic_tree / "27" / "IC_site_1_3.nc")
        index = ingest.discover_files(ic_tree)
        with pytest.raises(ingest.IngestError, match="disagree with their directories"):
            ingest.check_paths_follow_the_layout(index)

    def test_the_message_names_both_numbers(self, ic_tree, write_ic_file):
        """So the fix is obvious from the failure alone."""
        write_ic_file(ic_tree / "27" / "IC_site_1_3.nc")
        index = ingest.discover_files(ic_tree)
        with pytest.raises(ingest.IngestError) as error:
            ingest.check_paths_follow_the_layout(index)
        message = str(error.value)
        assert "site 27" in message
        assert "IC_site_1_3.nc" in message
        assert "IC_site_27_3.nc" in message

    def test_rejects_two_paths_resolving_to_one_cell(self, ic_tree, write_ic_file):
        """One would silently overwrite the other in the grid."""
        write_ic_file(ic_tree / "1" / "IC_site_27_1.nc")
        index = ingest.discover_files(ic_tree)
        with pytest.raises(ingest.IngestError, match="same \\(site, member\\) cell"):
            ingest.check_no_duplicate_site_member_pairs(index)

    def test_rejects_a_site_absent_from_the_site_table(self, tmp_path, site_table):
        """The identifiers are a shared key; a tree that disagrees is an error."""
        root = _build_tree(tmp_path / "ic", [(1, 1), (9999, 1)])
        index = ingest.discover_files(root)
        with pytest.raises(ingest.IngestError, match="not in the site"):
            ingest.check_sites_are_in_the_site_table(index, site_table)


# ── coverage ──────────────────────────────────────────────────────────────────


class TestCoverageChecks:
    def test_a_complete_rectangle_passes_both_checks(self, tmp_path):
        table = pd.DataFrame(
            {
                "site_id": np.array([1, 27], dtype=np.int32),
                "lon": [0.0, 1.0],
                "lat": [2.0, 3.0],
            }
        )
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1), (27, 2)])
        index = ingest.discover_files(root)
        ingest.check_every_pool_site_has_a_directory(index, table, allow_gaps=False)
        ingest.check_members_are_the_same_at_every_site(index, allow_gaps=False)

    def test_a_pool_site_without_a_directory_is_fatal_by_default(
        self, ic_tree, site_table
    ):
        """The development checkout's case: two sites of the pool's many."""
        index = ingest.discover_files(ic_tree)
        with pytest.raises(ingest.IngestError, match="have no directory under"):
            ingest.check_every_pool_site_has_a_directory(
                index, site_table, allow_gaps=False
            )

    def test_a_pool_site_without_a_directory_is_allowed_with_the_flag(
        self, ic_tree, site_table
    ):
        index = ingest.discover_files(ic_tree)
        ingest.check_every_pool_site_has_a_directory(
            index, site_table, allow_gaps=True
        )

    def test_the_absent_site_message_is_a_sample_not_a_list(self, tmp_path):
        """Reporting every identifier would bury the message."""
        table = pd.DataFrame(
            {
                "site_id": np.arange(1, 200, dtype=np.int32),
                "lon": np.zeros(199),
                "lat": np.zeros(199),
            }
        )
        root = _build_tree(tmp_path / "ic", [(1, 1)])
        index = ingest.discover_files(root)
        with pytest.raises(ingest.IngestError) as error:
            ingest.check_every_pool_site_has_a_directory(
                index, table, allow_gaps=False
            )
        assert "more" in str(error.value)
        assert len(str(error.value)) < 500

    def test_a_site_missing_one_member_is_fatal_by_default(self, tmp_path):
        """The ragged-ensemble case, which would quietly skew a member statistic."""
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1)])
        index = ingest.discover_files(root)
        with pytest.raises(ingest.IngestError, match="ragged over sites"):
            ingest.check_members_are_the_same_at_every_site(index, allow_gaps=False)

    def test_a_site_missing_one_member_is_allowed_with_the_flag(self, tmp_path):
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1)])
        index = ingest.discover_files(root)
        ingest.check_members_are_the_same_at_every_site(index, allow_gaps=True)

    def test_the_ragged_message_names_the_sites_and_the_missing_members(
        self, tmp_path
    ):
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1)])
        index = ingest.discover_files(root)
        with pytest.raises(ingest.IngestError) as error:
            ingest.check_members_are_the_same_at_every_site(index, allow_gaps=False)
        message = str(error.value)
        assert "site 27" in message
        assert "[2]" in message


# ── cross-file metadata ───────────────────────────────────────────────────────


class TestCrossFileChecks:
    def test_agreeing_time_metadata_passes(self, built):
        _, contents, _, _ = built
        ingest.check_time_metadata_agrees_across_files(contents)

    def test_rejects_a_file_whose_time_units_differ(self, tmp_path):
        """Only one copy reaches the product, so a disagreement must be loud."""
        root = _build_tree(tmp_path / "ic", [(1, 1)])
        _build_tree(root, [(1, 2)], time_units="days since 2012-01-01")
        contents = ingest.read_all_files(ingest.discover_files(root), jobs=1)
        with pytest.raises(ingest.IngestError, match="time metadata is"):
            ingest.check_time_metadata_agrees_across_files(contents)

    def test_rejects_a_substituted_time_units_string(self, tmp_path):
        """A concrete year would mean issue #3 was fixed upstream.

        That is a thing to notice and act on, not to average away, so it fails
        rather than being accepted as an improvement.
        """
        root = _build_tree(
            tmp_path / "ic",
            [(1, 1)],
            time_units="days since 2012-01-01 00:00:00 UTC",
        )
        contents = ingest.read_all_files(ingest.discover_files(root), jobs=1)
        with pytest.raises(ingest.IngestError, match="issue #3"):
            ingest.check_time_metadata_agrees_across_files(contents)

    def test_rejects_a_file_whose_time_value_differs(self, tmp_path):
        root = _build_tree(tmp_path / "ic", [(1, 1)], time_value=5.0)
        contents = ingest.read_all_files(ingest.discover_files(root), jobs=1)
        with pytest.raises(ingest.IngestError, match="time metadata is"):
            ingest.check_time_metadata_agrees_across_files(contents)

    def test_rejects_a_file_whose_source_long_name_differs(self, tmp_path):
        """The schema assumes a source variable name has one meaning."""
        root = _build_tree(
            tmp_path / "ic", [(1, 1)], long_names={"AbvGrndWood": "Something else"}
        )
        contents = ingest.read_all_files(ingest.discover_files(root), jobs=1)
        with pytest.raises(ingest.IngestError, match="long_name"):
            ingest.check_source_long_names_agree_across_files(contents)


# ── build_grids ───────────────────────────────────────────────────────────────


class TestBuildGrids:
    def test_lays_values_out_on_member_by_site(self, built):
        """Members ascending on the first axis, the whole pool on the second."""
        _, _, grids, _ = built
        for name in IC_VARIABLES:
            assert grids.values[name].shape == (2, 4)
        assert grids.source_members.tolist() == [1, 2]

    def test_the_site_axis_is_the_whole_pool(self, built, site_table):
        """A site with no file still gets a column, all ``NaN``."""
        _, _, grids, _ = built
        assert grids.site.tolist() == site_table["site_id"].tolist()
        absent = grids.site.tolist().index(5)
        assert np.all(np.isnan(grids.values[IC_VARIABLES[0]][:, absent]))
        assert not grids.ic_present[:, absent].any()

    def test_applies_the_rename_from_the_library_mapping(self, built):
        _, _, grids, _ = built
        assert set(grids.values) == set(IC_VARIABLES)

    def test_a_pair_with_no_file_is_nan_with_both_flags_false(self, tmp_path, site_table):
        root = _build_tree(tmp_path / "ic", [(1, 1), (27, 1), (27, 2)])
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        member_row = grids.source_members.tolist().index(2)
        site_col = grids.site.tolist().index(1)
        assert not grids.ic_present[member_row, site_col]
        assert not grids.variable_present[member_row, site_col].any()
        assert np.isnan(grids.values[IC_VARIABLES[0]][member_row, site_col])

    def test_a_file_lacking_a_variable_is_nan_with_variable_present_false(
        self, tmp_path, site_table
    ):
        root = tmp_path / "ic"
        _write_ic_file(
            root / "1" / "IC_site_1_1.nc", values={"AbvGrndWood": 1.0}
        )
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        col = grids.site.tolist().index(1)
        soil = IC_VARIABLES.index("initial_soil_organic_carbon")
        assert grids.ic_present[0, col]
        assert not grids.variable_present[0, col, soil]
        assert np.isnan(grids.values["initial_soil_organic_carbon"][0, col])

    def test_an_explicit_fill_is_nan_with_variable_present_true(
        self, tmp_path, site_table
    ):
        """This is the discriminator the data model promises.

        ``variable_present & isnan(value)`` has to mean explicit fill and
        nothing else, and this is the case that distinguishes it from the two
        above.
        """
        root = tmp_path / "ic"
        values = {name: 1.0 for name in SOURCE_NAMES}
        values["soil_organic_carbon_content"] = SOURCE_FILL_VALUE
        _write_ic_file(root / "1" / "IC_site_1_1.nc", values=values)
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        col = grids.site.tolist().index(1)
        soil = IC_VARIABLES.index("initial_soil_organic_carbon")
        assert grids.ic_present[0, col]
        assert grids.variable_present[0, col, soil]
        assert np.isnan(grids.values["initial_soil_organic_carbon"][0, col])

    def test_counts_the_explicit_fills_per_variable(self, tmp_path, site_table):
        root = tmp_path / "ic"
        values = {name: 1.0 for name in SOURCE_NAMES}
        values["AbvGrndWood"] = SOURCE_FILL_VALUE
        _write_ic_file(root / "1" / "IC_site_1_1.nc", values=values)
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        assert grids.explicit_fills["initial_aboveground_wood_carbon"] == 1
        assert grids.explicit_fills["initial_wood_carbon"] == 0

    def test_source_member_indices_are_kept_beside_the_zero_based_axis(
        self, tmp_path, site_table
    ):
        """Members ``1, 2, 94`` become ``0, 1, 2`` with the source indices kept."""
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 94)])
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        dataset = ingest.build_dataset(grids, index, allow_gaps=True)
        assert dataset["member"].values.tolist() == [0, 1, 2]
        assert dataset["source_member_index"].values.tolist() == [1, 2, 94]


# ── build_dataset ─────────────────────────────────────────────────────────────


class TestBuildDataset:
    def test_has_exactly_the_declared_variables(self, built):
        """The three floats and the two presence companions, nothing else."""
        dataset, _, _, _ = built
        assert set(dataset.data_vars) == set(IC_VARIABLES) | {
            IC_PRESENT,
            VARIABLE_PRESENT,
        }

    def test_the_presence_companions_are_written_unconditionally(self, tmp_path, site_table):
        """Even for a complete tree where both are all ``True``.

        A schema whose shape depends on the data makes every consumer branch.
        """
        table = site_table[site_table["site_id"].isin([1, 27])].reset_index(drop=True)
        root = _build_tree(tmp_path / "ic", [(1, 1), (1, 2), (27, 1), (27, 2)])
        index = ingest.discover_files(root)
        grids = ingest.build_grids(ingest.read_all_files(index, jobs=1), index, table)
        dataset = ingest.build_dataset(grids, index, allow_gaps=False)
        assert dataset[IC_PRESENT].values.all()
        assert dataset[VARIABLE_PRESENT].values.all()
        assert dataset.attrs["coverage"] == "complete"

    def test_dtypes_are_float64_and_bool(self, built):
        dataset, _, _, _ = built
        for name in IC_VARIABLES:
            assert dataset[name].dtype == np.float64
        for name in (IC_PRESENT, VARIABLE_PRESENT):
            assert dataset[name].dtype == np.bool_

    def test_has_no_time_dimension(self, built):
        dataset, _, _, _ = built
        assert "time" not in dataset.dims
        assert "time" not in dataset.coords

    def test_records_what_the_dropped_time_coordinate_claimed(self, built):
        """The units template, the long name and the value, verbatim."""
        dataset, _, _, _ = built
        assert dataset.attrs["source_time_units"] == SOURCE_TIME_UNITS
        assert dataset.attrs["source_time_long_name"] == SOURCE_TIME_LONG_NAME
        assert dataset.attrs["source_time_value"] == SOURCE_TIME_VALUE
        assert dataset.attrs["time_status"] == "dropped"
        assert "issue #3" in dataset.attrs["time_note"]

    def test_coordinates_are_member_site_lon_lat_and_variable(self, built):
        dataset, _, _, _ = built
        for coord in ("member", "source_member_index", "site", "lon", "lat", "variable"):
            assert coord in dataset.coords
        assert dataset["lon"].dims == ("site",)
        assert dataset["lat"].dims == ("site",)

    def test_member_is_zero_based_int16_and_site_is_int32(self, built):
        dataset, _, _, _ = built
        assert dataset["member"].dtype == np.int16
        assert dataset["site"].dtype == np.int32
        assert dataset["member"].values.tolist() == [0, 1]

    def test_every_variable_carries_its_units_and_provenance(self, built):
        dataset, _, _, _ = built
        for name in IC_VARIABLES:
            attrs = dataset[name].attrs
            assert attrs["units"] == IC_VARIABLE_ATTRS[name]["units"]
            assert attrs["units_status"] == UNITS_STATUS
            assert "not confirmed by the producer" in attrs["units_provenance"].lower()

    def test_variables_with_a_counterpart_carry_the_conversion_factor(self, built):
        """So the factor of ten lives in the product, not in someone's head."""
        dataset, _, _, _ = built
        wood = dataset["initial_aboveground_wood_carbon"].attrs
        assert wood["related_constraint_variable"] == "aboveground_wood_carbon"
        assert wood["related_constraint_unit_factor"] == 10.0
        assert "related_constraint_variable" not in dataset["initial_wood_carbon"].attrs

    def test_records_member_source_and_that_correspondence_is_unestablished(self, built):
        """``member_source`` is ``"ic"`` and the correspondence attribute says no.

        xarray aligns integer member labels silently, so this attribute is the
        only thing standing between a caller and pairing initial-condition
        member 3 with driver member 3.
        """
        dataset, _, _, _ = built
        assert dataset.attrs["member_source"] == MEMBER_SOURCE == "ic"
        assert "Not established" in dataset.attrs["member_correspondence"]
        assert "open question 12" in dataset.attrs["member_correspondence"]

    def test_coverage_is_gaps_when_anything_is_missing(self, built):
        dataset, _, _, _ = built
        assert dataset.attrs["coverage"] == "gaps"

    def test_counts_non_positive_values_without_clamping_them(
        self, tmp_path, site_table
    ):
        """A negative carbon stock is passed through and counted, per the drivers."""
        root = tmp_path / "ic"
        _write_ic_file(
            root / "1" / "IC_site_1_1.nc",
            values={name: -2.0 for name in SOURCE_NAMES},
        )
        index = ingest.discover_files(root)
        grids = ingest.build_grids(
            ingest.read_all_files(index, jobs=1), index, site_table
        )
        dataset = ingest.build_dataset(grids, index, allow_gaps=True)
        for name in IC_VARIABLES:
            assert dataset[name].attrs["n_values_not_positive"] == 1
            assert dataset[name].sel(member=0, site=1).item() == -2.0


# ── writing and the round trip ────────────────────────────────────────────────


class TestWriteDataset:
    def test_writes_the_canonical_path_after_the_checks_pass(self, tmp_path, built):
        dataset, _, _, _ = built
        out = tmp_path / "out" / "ic.nc"
        ingest.write_dataset(dataset, out)
        assert out.is_file()
        assert not out.with_suffix(".nc.partial").exists()

    def test_a_failed_round_trip_leaves_nothing_at_the_canonical_path(
        self, tmp_path, built, monkeypatch
    ):
        """The partial file stays for inspection; the canonical name does not appear."""
        dataset, _, _, _ = built
        out = tmp_path / "out" / "ic.nc"

        def fail(dataset, path):
            raise ingest.IngestError("forced failure")

        monkeypatch.setattr(ingest, "check_round_trip", fail)
        with pytest.raises(ingest.IngestError, match="forced failure"):
            ingest.write_dataset(dataset, out)
        assert not out.exists()
        assert out.with_suffix(".nc.partial").is_file()

    def test_values_dtypes_coords_and_attributes_all_survive(self, tmp_path, built):
        """Including the booleans, which netCDF has no native type for."""
        dataset, _, _, _ = built
        out = tmp_path / "ic.nc"
        ingest.write_dataset(dataset, out)
        with load_initial_conditions(out) as back:
            for name in list(IC_VARIABLES) + [IC_PRESENT, VARIABLE_PRESENT]:
                assert back[name].dtype == dataset[name].dtype
                assert np.array_equal(
                    back[name].values, dataset[name].values, equal_nan=True
                )
                assert dict(back[name].attrs) == dict(dataset[name].attrs)
            assert dict(back.attrs) == dict(dataset.attrs)

    def test_creates_the_output_directory(self, tmp_path, built):
        dataset, _, _, _ = built
        out = tmp_path / "deep" / "nested" / "ic.nc"
        ingest.write_dataset(dataset, out)
        assert out.is_file()


# ── load_initial_conditions ───────────────────────────────────────────────────


@pytest.fixture
def product(tmp_path, built):
    """A written product, for the reader's validation cases to corrupt."""
    dataset, _, _, _ = built
    out = tmp_path / "ic.nc"
    ingest.write_dataset(dataset, out)
    return out


def _rewrite(path, mutate):
    """Read a product with decoding off, mutate it, write it back."""
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        modified = mutate(dataset.load().copy(deep=True))
    path.unlink()
    modified.to_netcdf(path, engine="h5netcdf")
    return path


class TestLoadInitialConditions:
    def test_reads_a_conforming_product(self, product):
        with load_initial_conditions(product) as dataset:
            assert set(dataset.data_vars) == set(IC_VARIABLES) | {
                IC_PRESENT,
                VARIABLE_PRESENT,
            }

    def test_raises_when_the_file_is_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no initial-condition product"):
            load_initial_conditions(tmp_path / "nope.nc")

    def test_rejects_a_missing_data_variable(self, product):
        _rewrite(product, lambda ds: ds.drop_vars(IC_VARIABLES[0]))
        with pytest.raises(ValueError, match="data variables are"):
            load_initial_conditions(product)

    def test_rejects_an_extra_data_variable(self, product):
        def add(ds):
            ds["surprise"] = ds[IC_VARIABLES[0]]
            return ds

        _rewrite(product, add)
        with pytest.raises(ValueError, match="data variables are"):
            load_initial_conditions(product)

    def test_rejects_a_wrong_dtype(self, product):
        def demote(ds):
            ds[IC_VARIABLES[0]] = ds[IC_VARIABLES[0]].astype(np.float32)
            return ds

        _rewrite(product, demote)
        with pytest.raises(ValueError, match="expected float64"):
            load_initial_conditions(product)

    def test_rejects_wrong_dims(self, product):
        def transpose(ds):
            values = ds[IC_VARIABLES[0]].transpose("site", "member")
            ds[IC_VARIABLES[0]] = xr.DataArray(
                values.values, dims=("site", "member"), attrs=ds[IC_VARIABLES[0]].attrs
            )
            return ds

        _rewrite(product, transpose)
        with pytest.raises(ValueError, match="expected \\('member', 'site'\\)"):
            load_initial_conditions(product)

    def test_rejects_a_member_axis_that_is_not_zero_based(self, product):
        def shift(ds):
            return ds.assign_coords(
                member=ds["member"].values.astype(np.int16) + np.int16(1)
            )

        _rewrite(product, shift)
        with pytest.raises(ValueError, match="is not 0"):
            load_initial_conditions(product)

    def test_rejects_an_unsorted_site_axis(self, product):
        def reverse(ds):
            return ds.isel(site=slice(None, None, -1))

        _rewrite(product, reverse)
        with pytest.raises(ValueError, match="not strictly ascending"):
            load_initial_conditions(product)

    def test_rejects_a_variable_coordinate_out_of_order(self, product):
        """The ``variable_present`` axis has to line up with ``IC_VARIABLES``."""
        def shuffle(ds):
            return ds.assign_coords(variable=list(reversed(IC_VARIABLES)))

        _rewrite(product, shuffle)
        with pytest.raises(ValueError, match="'variable' is"):
            load_initial_conditions(product)

    def test_rejects_a_missing_variable_attribute(self, product):
        def strip(ds):
            attrs = dict(ds[IC_VARIABLES[0]].attrs)
            attrs.pop("units_provenance")
            ds[IC_VARIABLES[0]].attrs = attrs
            return ds

        _rewrite(product, strip)
        with pytest.raises(ValueError, match="missing attributes"):
            load_initial_conditions(product)

    def test_rejects_a_missing_dataset_attribute(self, product):
        def strip(ds):
            attrs = dict(ds.attrs)
            attrs.pop("member_correspondence")
            ds.attrs = attrs
            return ds

        _rewrite(product, strip)
        with pytest.raises(ValueError, match="dataset attributes"):
            load_initial_conditions(product)

    def test_rejects_a_wrong_member_source(self, product):
        def relabel(ds):
            ds.attrs = {**ds.attrs, "member_source": "met"}
            return ds

        _rewrite(product, relabel)
        with pytest.raises(ValueError, match="member_source is"):
            load_initial_conditions(product)

    def test_rejects_a_variable_present_where_no_file_was(self, product):
        """``variable_present`` cannot be ``True`` where ``ic_present`` is ``False``."""
        def lie(ds):
            ds[VARIABLE_PRESENT].values[:] = True
            return ds

        _rewrite(product, lie)
        with pytest.raises(ValueError, match="is True where"):
            load_initial_conditions(product)

    def test_rejects_a_finite_value_where_variable_present_is_false(self, product):
        """Otherwise the presence arrays and the values would tell different stories."""
        def lie(ds):
            ds[IC_VARIABLES[0]].values[:] = 1.0
            return ds

        _rewrite(product, lie)
        with pytest.raises(ValueError, match="is finite where"):
            load_initial_conditions(product)


# ── initial_condition_fields ──────────────────────────────────────────────────


class TestInitialConditionFields:
    def test_returns_one_field_per_variable_in_order(self, built):
        dataset, _, _, _ = built
        fields = initial_condition_fields(dataset)
        assert tuple(fields) == IC_VARIABLES

    def test_each_field_is_a_canonical_field(self, built):
        """Dims a subset of ``(member, site, time)``, ``lon``/``lat`` on ``site``."""
        dataset, _, _, _ = built
        for field in initial_condition_fields(dataset).values():
            assert set(field.dims) <= {"member", "site", "time"}
            assert field.dims == ("member", "site")
            assert field["lon"].dims == ("site",)
            assert field["lat"].dims == ("site",)
            assert "variable" not in field.coords

    def test_each_field_carries_its_own_units_and_long_name(self, built):
        dataset, _, _, _ = built
        for name, field in initial_condition_fields(dataset).items():
            assert field.attrs["units"] == IC_VARIABLE_ATTRS[name]["units"]
            assert field.attrs["long_name"] == IC_VARIABLE_ATTRS[name]["long_name"]

    def test_the_presence_companions_are_not_fields(self, built):
        """They are neither canonical nor per-variable, so they are left out."""
        dataset, _, _, _ = built
        fields = initial_condition_fields(dataset)
        assert IC_PRESENT not in fields
        assert VARIABLE_PRESENT not in fields

    def test_raises_when_a_variable_is_absent(self, built):
        dataset, _, _, _ = built
        with pytest.raises(ValueError, match="missing initial-condition variables"):
            initial_condition_fields(dataset.drop_vars(IC_VARIABLES[0]))


# ── the report ────────────────────────────────────────────────────────────────


class TestReport:
    def test_reports_the_coverage_and_the_axis_sizes(self, built):
        dataset, contents, _, _ = built
        report = ingest.describe_initial_conditions(dataset, contents)
        assert "members 2" in report
        assert "sites 4" in report
        assert "coverage gaps" in report

    def test_reports_per_variable_counts_and_extremes(self, built):
        """The measurements that belong in a run log rather than in documentation."""
        dataset, contents, _, _ = built
        report = ingest.describe_initial_conditions(dataset, contents)
        for name in IC_VARIABLES:
            assert name in report
        assert "range" in report

    def test_reports_the_distinct_variable_set_signatures(self, built):
        """The measurement that answers the open variable-set question."""
        dataset, contents, _, _ = built
        report = ingest.describe_initial_conditions(dataset, contents)
        assert "variable-set signatures" in report
        assert "AbvGrndWood" in report

    def test_reports_two_signatures_when_the_files_differ(self, tmp_path, site_table):
        root = tmp_path / "ic"
        _write_ic_file(root / "1" / "IC_site_1_1.nc")
        _write_ic_file(root / "1" / "IC_site_1_2.nc", values={"AbvGrndWood": 1.0})
        index = ingest.discover_files(root)
        contents = ingest.read_all_files(index, jobs=1)
        grids = ingest.build_grids(contents, index, site_table)
        dataset = ingest.build_dataset(grids, index, allow_gaps=True)
        report = ingest.describe_initial_conditions(dataset, contents)
        assert len(ingest._variable_set_signatures(contents)) == 2
        assert "1 file(s)" in report

    def test_reports_disagreeing_wood_cells_without_asserting_anything(
        self, tmp_path, site_table
    ):
        """A tree where the two wood variables differ still ingests.

        Asserting the equality would turn a legitimate file into a failure,
        and de-duplicating the variable on the available evidence would be a
        guess, so the count is reported and both variables are kept.
        """
        root = tmp_path / "ic"
        _write_ic_file(
            root / "1" / "IC_site_1_1.nc",
            values={
                "AbvGrndWood": 1.0,
                "wood_carbon_content": 2.0,
                "soil_organic_carbon_content": 3.0,
            },
        )
        index = ingest.discover_files(root)
        contents = ingest.read_all_files(index, jobs=1)
        grids = ingest.build_grids(contents, index, site_table)
        dataset = ingest.build_dataset(grids, index, allow_gaps=True)
        assert ingest._wood_variables_disagreeing(dataset) == 1
        report = ingest.describe_initial_conditions(dataset, contents)
        assert "unequal: 1" in report
        # Both variables survive; nothing is de-duplicated.
        assert dataset["initial_aboveground_wood_carbon"].sel(
            member=0, site=1
        ).item() == 1.0
        assert dataset["initial_wood_carbon"].sel(member=0, site=1).item() == 2.0


# ── the script end to end ─────────────────────────────────────────────────────


class TestMain:
    def test_writes_the_product_and_returns_zero(self, tmp_path, ic_tree, sites_csv):
        sites = sites_csv
        out = tmp_path / "ic.nc"
        code = ingest.main(
            [
                "--root", str(ic_tree),
                "--sites", str(sites),
                "--out", str(out),
                "--allow-gaps",
            ]
        )
        assert code == 0
        assert out.is_file()

    def test_returns_one_and_writes_nothing_when_a_check_fails(
        self, tmp_path, ic_tree, sites_csv
    ):
        sites = sites_csv
        out = tmp_path / "ic.nc"
        code = ingest.main(
            ["--root", str(ic_tree), "--sites", str(sites), "--out", str(out)]
        )
        assert code == 1
        assert not out.exists()

    def test_reports_a_check_failure_as_a_message_not_a_traceback(
        self, tmp_path, ic_tree, sites_csv, capsys
    ):
        sites = sites_csv
        ingest.main(
            ["--root", str(ic_tree), "--sites", str(sites), "--out", str(tmp_path / "x.nc")]
        )
        captured = capsys.readouterr()
        assert captured.err.startswith("error: ")
        assert "Traceback" not in captured.err

    def test_leaves_every_input_byte_for_byte_unchanged(
        self, tmp_path, ic_tree, sites_csv
    ):
        """Hashes every file under the root before and after the run.

        ``raw/`` is read-only by contract, and this is the test that proves the
        script honors it.
        """
        def digest(root):
            return {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        sites = sites_csv
        before = digest(ic_tree)
        sites_before = hashlib.sha256(sites.read_bytes()).hexdigest()
        ingest.main(
            [
                "--root", str(ic_tree),
                "--sites", str(sites),
                "--out", str(tmp_path / "ic.nc"),
                "--allow-gaps",
            ]
        )
        assert digest(ic_tree) == before
        assert hashlib.sha256(sites.read_bytes()).hexdigest() == sites_before

    def test_help_works(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            ingest.parse_args(["--help"])
        assert exit_info.value.code == 0
        assert "initial-condition" in capsys.readouterr().out


# ── the real files ────────────────────────────────────────────────────────────


def real_pairs() -> list[tuple[int, int]]:
    if not REAL_ROOT.is_dir():
        return []
    pairs = []
    for site in available_sites(REAL_ROOT):
        for member in available_members(REAL_ROOT, site):
            pairs.append((site, member))
    return sorted(pairs)


needs_real_files = pytest.mark.skipif(
    not real_pairs(), reason="data/raw/initial_conditions/ is not present"
)


@needs_real_files
class TestRealFiles:
    """Against ``data/raw/initial_conditions/``; skipped when absent.

    These are the cases that pin the format to the data rather than to a
    fixture written to match it, parameterized over whatever pairs are present
    rather than a hard-coded list, so the class does not go stale when more
    files arrive.
    """

    @pytest.mark.parametrize("pair", real_pairs())
    def test_every_local_file_parses_and_passes_every_check(self, pair):
        contents = read_ic_file(ic_file(REAL_ROOT, *pair))
        assert set(contents.values) <= set(SOURCE_VARIABLE_NAMES)
        assert contents.values

    @pytest.mark.parametrize("pair", real_pairs())
    def test_every_local_file_is_netcdf3_classic(self, pair):
        """Magic ``CDF\\x01``, which is why the engine is ``scipy``."""
        with open(ic_file(REAL_ROOT, *pair), "rb") as handle:
            assert handle.read(4) == b"CDF\x01"

    @pytest.mark.parametrize("pair", real_pairs())
    def test_time_is_the_unlimited_record_dimension_of_length_one(self, pair):
        with netcdf_file(str(ic_file(REAL_ROOT, *pair)), "r", mmap=False) as raw:
            # scipy records the unlimited dimension's length as None.
            assert raw.dimensions["time"] is None
            assert raw.variables["time"].shape == (1,)

    @pytest.mark.parametrize("pair", real_pairs())
    def test_the_time_units_attribute_is_still_the_unsubstituted_template(self, pair):
        """Pins issue #3 to the data. If this fails, the defect was fixed upstream."""
        contents = read_ic_file(ic_file(REAL_ROOT, *pair))
        assert contents.time_units == SOURCE_TIME_UNITS
        assert "[year]" in contents.time_units

    def test_default_decoding_still_raises(self):
        """The reason ``decode_times=False`` is not optional."""
        with pytest.raises(ValueError, match="unable to decode time units"):
            xr.open_dataset(ic_file(REAL_ROOT, *real_pairs()[0]))

    def test_h5netcdf_cannot_open_them(self):
        """The reason the engine is pinned rather than left to the default."""
        with pytest.raises(Exception):
            xr.open_dataset(
                ic_file(REAL_ROOT, *real_pairs()[0]),
                decode_times=False,
                engine="h5netcdf",
            )

    @pytest.mark.parametrize("pair", real_pairs())
    def test_every_variable_declares_the_expected_fill_value(self, pair):
        with netcdf_file(str(ic_file(REAL_ROOT, *pair)), "r", mmap=False) as raw:
            for name, variable in raw.variables.items():
                if name == "time":
                    continue
                assert float(variable._attributes["_FillValue"]) == SOURCE_FILL_VALUE

    def test_no_local_file_holds_an_explicit_fill(self):
        """So the explicit-fill path is exercised only by synthetic files.

        Recorded as a test rather than as prose, since it is a property of the
        data that would change without notice.
        """
        for pair in real_pairs():
            assert read_ic_file(ic_file(REAL_ROOT, *pair)).explicit_fills == frozenset()

    @pytest.mark.parametrize("pair", real_pairs())
    def test_the_two_wood_variables_are_bitwise_equal(self, pair):
        """In every local file, across the sites and members present.

        The measurement behind the decision to keep both variables rather than
        de-duplicate: it is evidence of a duplicate, not proof of one, and
        three files cannot settle it.
        """
        with netcdf_file(str(ic_file(REAL_ROOT, *pair)), "r", mmap=False) as raw:
            first = raw.variables["AbvGrndWood"].data.tobytes()
            second = raw.variables["wood_carbon_content"].data.tobytes()
        assert first == second

    @pytest.mark.parametrize("pair", real_pairs())
    def test_no_local_file_carries_an_unspecified_variable(self, pair):
        """None of them has ``leaf_carbon_content`` or ``SoilMoistFrac``.

        Which is exactly why they are unregistered and why the branch waits on
        the survey.
        """
        contents = read_ic_file(ic_file(REAL_ROOT, *pair))
        assert set(contents.values).isdisjoint(UNSPECIFIED_VARIABLES)

    @pytest.mark.slow
    def test_the_local_tree_ingests_with_allow_gaps(self, tmp_path):
        """The real files on the full pool axis, with ``coverage`` reading gaps.

        The end-to-end case that can run here: a real, honest, gappy product.
        """
        out = tmp_path / "ic.nc"
        code = ingest.main(
            ["--root", str(REAL_ROOT), "--out", str(out), "--allow-gaps"]
        )
        assert code == 0
        with load_initial_conditions(out) as dataset:
            assert dataset.attrs["coverage"] == "gaps"
            assert dataset.sizes["site"] == 8000
            assert int(dataset[IC_PRESENT].values.sum()) == len(real_pairs())
            # Values arrive bit-exact from the source.
            for site, member in real_pairs():
                index = dataset["source_member_index"].values.tolist().index(member)
                expected = read_ic_file(ic_file(REAL_ROOT, site, member))
                for source, value in expected.values.items():
                    stored = dataset[SOURCE_VARIABLE_NAMES[source]].sel(
                        member=index, site=site
                    ).item()
                    assert stored == value

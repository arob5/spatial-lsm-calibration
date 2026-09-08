"""Tests for the annual constraint schema, its reader, and its ingest.

Most cases run against a small synthetic long table, built so that the traps are
present at a size where the expected answer can be written out by hand. The
trap that matters most is the variable pairing: ``obs.cov`` carries no dimension
names, so the only thing saying which variance belongs to which variable is the
column order of the paired ``obs.mean`` entry, and a site observing a *subset*
of the variables is where a positional mistake shows up.

The cases at the end run the real R export against the tracked ``.Rdata`` files
and compare against values pulled independently out of R. They are skipped when
``Rscript`` or the raw files are absent, and marked slow because the export
takes about half a minute.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sipnet_calibration.constraints as constraints_module
from sipnet_calibration.constraints import (
    CONSTRAINT_VARIABLE_ATTRS,
    CONSTRAINT_VARIABLES,
    LONG_COLUMNS,
    OBSERVATION_MEAN,
    OBSERVATION_VARIANCE,
    SOURCE_VARIABLE_NAMES,
    UNITS_STATUS,
    constraint_fields,
    default_constraints_path,
    load_constraints,
    read_long_table,
    snapshot_dates,
)
from sipnet_calibration.sites import SITE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_constraints.R"
RAW_RDATA = REPO_ROOT / "data" / "raw" / "constraints" / "sda_8k_site_rdata"
RAW_MEAN = RAW_RDATA / "obs.mean.Rdata"
RAW_COV = RAW_RDATA / "obs.cov.Rdata"

#: The six sites with no observations at all, in each of 2012, 2013 and 2014.
EMPTY_SITES = [9, 143, 483, 1487, 2686, 3012]
EMPTY_SNAPSHOTS = ["2012-07-15", "2013-07-15", "2014-07-15"]

#: Per-variable row counts of the real source, keyed by *source* name, from
#: data/README.md's coverage table. The export must reproduce these exactly.
REAL_COUNTS = {
    "AbvGrndWood": 39273,
    "LAI": 99632,
    "SoilMoistFrac": 79740,
    "TotSoilCarb": 103870,
}
REAL_N_ROWS = sum(REAL_COUNTS.values())

#: The same counts keyed by processed name, for checking the written product.
REAL_COUNTS_PROCESSED = {
    SOURCE_VARIABLE_NAMES[source]: count for source, count in REAL_COUNTS.items()
}


def _load_ingest_module():
    """Import ``scripts/ingest_constraints.py``, which is a script."""
    path = REPO_ROOT / "scripts" / "ingest_constraints.py"
    spec = importlib.util.spec_from_file_location("ingest_constraints", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_constraints"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_module()


# ── synthetic fixtures ────────────────────────────────────────────────────────

#: A long table small enough to reason about, exercising the cases that matter:
#: a site observing every variable, one observing a subset in the middle of the
#: alphabet, one observing a single variable, one observing nothing at all, and
#: a variable that appears in one snapshot but not the other.
#: Rows carry *source* variable names, because that is what the long table
#: holds; the ingest script renames them.
SYNTHETIC_ROWS = [
    # snapshot,      site, source variable, mean,   variance
    ("2012-07-15", 1, "AbvGrndWood", 10.0, 100.0),
    ("2012-07-15", 1, "LAI", 1.5, 0.4356),
    ("2012-07-15", 1, "TotSoilCarb", 20.0, 400.0),
    ("2012-07-15", 2, "LAI", 2.5, 0.49),
    ("2012-07-15", 2, "TotSoilCarb", 30.0, 900.0),
    ("2012-07-15", 4, "TotSoilCarb", 40.0, 0.0),
    ("2013-07-15", 1, "SoilMoistFrac", 55.0, 1.25),
    ("2013-07-15", 1, "TotSoilCarb", 21.0, 441.0),
    ("2013-07-15", 2, "AbvGrndWood", 0.0, 0.0),
    ("2013-07-15", 4, "SoilMoistFrac", 12.5, 2.5),
]

SYNTHETIC_SITES = [1, 2, 3, 4]


def _write_sites(path: Path, site_ids=SYNTHETIC_SITES) -> Path:
    """A minimal site table that ``load_sites`` accepts."""
    frame = pd.DataFrame(
        {
            "site_id": np.array(site_ids, dtype=np.int32),
            "lon": [-100.0 - index for index in range(len(site_ids))],
            "lat": [40.0 + index for index in range(len(site_ids))],
            "lon_index": np.arange(len(site_ids), dtype=np.int32) + 1000,
            "lat_index": np.arange(len(site_ids), dtype=np.int32) + 2000,
            "site_name": [f"site {index}" for index in site_ids],
            "site_order": np.zeros(len(site_ids), dtype=np.int32),
            "cluster": np.ones(len(site_ids), dtype=np.int8),
            "landcover": np.ones(len(site_ids), dtype=np.int8),
            "ameriflux_site_id": [""] * len(site_ids),
        }
    )
    assert tuple(frame.columns) == SITE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_long_table(path: Path, rows=SYNTHETIC_ROWS) -> Path:
    """Write rows as the R script would, with ``%.17g`` doubles."""
    lines = [",".join(LONG_COLUMNS)]
    for snapshot, site, variable, mean, variance in rows:
        lines.append(f"{snapshot},{site},{variable},{mean:.17g},{variance:.17g}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _manifest_for(rows=SYNTHETIC_ROWS) -> dict:
    """The manifest the R script would have written for these rows."""
    frame = pd.DataFrame(rows, columns=list(LONG_COLUMNS))
    counts = {}
    for snapshot in sorted(frame["snapshot_date"].unique()):
        subset = frame[frame["snapshot_date"] == snapshot]
        counts[snapshot] = {
            name: int((subset["variable"] == name).sum())
            for name in SOURCE_VARIABLE_NAMES
        }
    extremes = {}
    for name in SOURCE_VARIABLE_NAMES:
        subset = frame[frame["variable"] == name]
        if subset.empty:
            continue
        extremes[name] = {
            "n": len(subset),
            "mean_min": f"{subset['mean'].min():.17g}",
            "mean_max": f"{subset['mean'].max():.17g}",
            "variance_min": f"{subset['variance'].min():.17g}",
            "variance_max": f"{subset['variance'].max():.17g}",
            "n_nonpositive_variance": int((subset["variance"] <= 0).sum()),
        }
    return {
        "generated_by": "test",
        "generated_at": "2026-09-08T00:00:00-0400",
        "mean_file": "synthetic",
        "cov_file": "synthetic",
        "variables": list(SOURCE_VARIABLE_NAMES),
        "n_sites": len(SYNTHETIC_SITES),
        "snapshot_dates": sorted(frame["snapshot_date"].unique().tolist()),
        "n_snapshots": frame["snapshot_date"].nunique(),
        "n_rows": len(frame),
        "counts_by_snapshot_variable": counts,
        "extremes": extremes,
        "empty_site_snapshots": {},
        "n_empty_site_snapshots": 0,
        "max_abs_offdiagonal": 0,
        "covariances_all_diagonal": True,
    }


@pytest.fixture
def synthetic(tmp_path):
    """Paths for a synthetic run, with the inputs already written."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_for()))
    return {
        "long_table": _write_long_table(tmp_path / "long.csv"),
        "manifest": manifest_path,
        "sites": _write_sites(tmp_path / "processed" / "sites" / "sites.csv"),
        "out": tmp_path / "processed" / "constraints_annual.nc",
        "tmp_path": tmp_path,
    }


@pytest.fixture
def ingested(synthetic):
    """A synthetic run, ingested and loaded back."""
    status = ingest.main(
        [
            "--long-table", str(synthetic["long_table"]),
            "--manifest", str(synthetic["manifest"]),
            "--sites", str(synthetic["sites"]),
            "--out", str(synthetic["out"]),
        ]
    )
    assert status == 0
    return load_constraints(synthetic["out"])


# ── the schema ────────────────────────────────────────────────────────────────


class TestSchemaConstants:
    def test_variables_are_alphabetical(self):
        # obs.cov has no dimension names, so this order is the only thing
        # pairing a variance with its variable. See the module docstring.
        assert list(CONSTRAINT_VARIABLES) == sorted(CONSTRAINT_VARIABLES)

    def test_processed_names_follow_the_naming_convention(self):
        for name in CONSTRAINT_VARIABLES:
            assert name == name.lower()
            assert " " not in name and "-" not in name

    def test_every_source_variable_has_a_processed_name(self):
        assert set(SOURCE_VARIABLE_NAMES.values()) == set(CONSTRAINT_VARIABLES)
        assert len(SOURCE_VARIABLE_NAMES) == len(CONSTRAINT_VARIABLES)

    def test_every_processed_variable_has_attributes(self):
        assert set(CONSTRAINT_VARIABLE_ATTRS) == set(CONSTRAINT_VARIABLES)
        for name, attrs in CONSTRAINT_VARIABLE_ATTRS.items():
            assert SOURCE_VARIABLE_NAMES[attrs["source_name"]] == name

    def test_snapshot_dates_use_the_source_convention(self):
        dates = snapshot_dates([2012, 2024])
        assert list(dates.strftime("%Y-%m-%d")) == ["2012-07-15", "2024-07-15"]


# ── reading the long table ────────────────────────────────────────────────────


class TestReadLongTable:
    def test_reads_the_expected_columns(self, synthetic):
        table = read_long_table(synthetic["long_table"])
        assert tuple(table.columns) == LONG_COLUMNS
        assert len(table) == len(SYNTHETIC_ROWS)

    def test_a_seventeen_digit_double_round_trips_exactly(self, tmp_path):
        # The value the site table lost to %.17g plus a non-round-trip parser.
        awkward = float.fromhex("0x1.921fb54442d18p+1")
        path = _write_long_table(
            tmp_path / "long.csv",
            [("2012-07-15", 1, "LAI", awkward, awkward)],
        )
        table = read_long_table(path)
        assert table["mean"][0] == awkward
        assert table["variance"][0] == awkward

    def test_rejects_unexpected_columns(self, tmp_path):
        path = tmp_path / "long.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="expected columns"):
            read_long_table(path)

    def test_rejects_an_unknown_variable(self, tmp_path):
        path = _write_long_table(
            tmp_path / "long.csv", [("2012-07-15", 1, "Nitrogen", 1.0, 1.0)]
        )
        with pytest.raises(ValueError, match="not in SOURCE_VARIABLE_NAMES"):
            read_long_table(path)

    def test_rejects_an_empty_table(self, tmp_path):
        path = tmp_path / "long.csv"
        path.write_text(",".join(LONG_COLUMNS) + "\n")
        with pytest.raises(ValueError, match="no rows"):
            read_long_table(path)


# ── the pivot ─────────────────────────────────────────────────────────────────


class TestPivot:
    def test_dims_and_coordinates(self, ingested):
        assert ingested[OBSERVATION_MEAN].dims == ("site", "time", "variable")
        assert list(ingested["site"].values) == SYNTHETIC_SITES
        assert list(ingested["variable"].values) == list(CONSTRAINT_VARIABLES)
        assert ingested["lon"].dims == ("site",)
        assert ingested["lat"].dims == ("site",)

    def test_every_pool_site_gets_a_row_even_when_never_observed(self, ingested):
        # Site 3 appears in no row of the long table.
        assert 3 in ingested["site"].values
        assert bool(np.all(np.isnan(ingested[OBSERVATION_MEAN].sel(site=3).values)))

    @pytest.mark.parametrize(
        "snapshot,site,source_variable,mean,variance",
        [(row[0], row[1], row[2], row[3], row[4]) for row in SYNTHETIC_ROWS],
    )
    def test_each_observation_lands_in_its_own_cell(
        self, ingested, snapshot, site, source_variable, mean, variance
    ):
        variable = SOURCE_VARIABLE_NAMES[source_variable]
        assert float(
            ingested[OBSERVATION_MEAN].sel(
                site=site, time=snapshot, variable=variable
            )
        ) == mean
        assert float(
            ingested[OBSERVATION_VARIANCE].sel(
                site=site, time=snapshot, variable=variable
            )
        ) == variance

    def test_a_site_observing_a_subset_pairs_variances_correctly(self, ingested):
        # Site 2 in 2012 observes LAI and TotSoilCarb but not AbvGrndWood, so a
        # positional pairing would put LAI's variance on AbvGrndWood. This is
        # the case the whole variable-ordering convention exists for.
        cell = ingested.sel(site=2, time="2012-07-15")
        wood, lai = "aboveground_wood_carbon", "lai"
        soil = "total_soil_carbon"
        assert np.isnan(float(cell[OBSERVATION_MEAN].sel(variable=wood)))
        assert np.isnan(float(cell[OBSERVATION_VARIANCE].sel(variable=wood)))
        assert float(cell[OBSERVATION_MEAN].sel(variable=lai)) == 2.5
        assert float(cell[OBSERVATION_VARIANCE].sel(variable=lai)) == 0.49
        assert float(cell[OBSERVATION_MEAN].sel(variable=soil)) == 30.0
        assert float(cell[OBSERVATION_VARIANCE].sel(variable=soil)) == 900.0

    def test_source_names_are_renamed_to_processed_names(self, ingested):
        # The long table carried "TotSoilCarb"; the product carries
        # "total_soil_carbon", with the same value in the same cell.
        assert "TotSoilCarb" not in ingested["variable"].values
        assert float(
            ingested[OBSERVATION_MEAN].sel(
                site=1, time="2012-07-15", variable="total_soil_carbon"
            )
        ) == 20.0

    def test_an_unmapped_source_variable_is_refused(self, synthetic):
        # A new variable in the source is a schema change, not a new row.
        rows = SYNTHETIC_ROWS + [("2012-07-15", 1, "Nitrogen", 1.0, 1.0)]
        _write_long_table(synthetic["long_table"], rows)
        status = ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(synthetic["out"]),
            ]
        )
        assert status == 1
        assert not synthetic["out"].exists()

    def test_unobserved_cells_are_nan_not_zero(self, ingested):
        # A zero here would be an observation of no biomass, which is a real and
        # different statement from "not observed".
        assert np.isnan(float(
            ingested[OBSERVATION_MEAN].sel(
                site=1, time="2012-07-15", variable="soil_moisture_fraction"
            )
        ))

    def test_a_variable_absent_from_one_snapshot_is_nan_throughout_it(self, ingested):
        absent = ingested[OBSERVATION_MEAN].sel(
            time="2012-07-15", variable="soil_moisture_fraction"
        )
        assert bool(np.all(np.isnan(absent.values)))

    def test_zero_variances_are_preserved_not_floored(self, ingested):
        # Flooring is a modeling decision and must not happen at ingest.
        assert float(
            ingested[OBSERVATION_VARIANCE].sel(
                site=4, time="2012-07-15", variable="total_soil_carbon"
            )
        ) == 0.0
        assert float(
            ingested[OBSERVATION_VARIANCE].sel(
                site=2, time="2013-07-15", variable="aboveground_wood_carbon"
            )
        ) == 0.0

    def test_a_zero_observation_is_distinct_from_a_missing_one(self, ingested):
        observed_zero = ingested[OBSERVATION_MEAN].sel(
            site=2, time="2013-07-15", variable="aboveground_wood_carbon"
        )
        assert float(observed_zero) == 0.0
        assert not np.isnan(float(observed_zero))

    def test_mean_and_variance_are_missing_together(self, ingested):
        # The Usage section of the module docstring tells callers they can
        # select variances with the same expression as the means and get an
        # aligned vector. That holds only if the two arrays are NaN in exactly
        # the same cells.
        mean_missing = np.isnan(ingested[OBSERVATION_MEAN].values)
        variance_missing = np.isnan(ingested[OBSERVATION_VARIANCE].values)
        assert np.array_equal(mean_missing, variance_missing)

    def test_observed_cell_count_matches_the_long_table(self, ingested):
        expected = len(SYNTHETIC_ROWS)
        mean, variance = OBSERVATION_MEAN, OBSERVATION_VARIANCE
        assert int(np.isfinite(ingested[mean].values).sum()) == expected
        assert int(np.isfinite(ingested[variance].values).sum()) == expected


# ── attributes ────────────────────────────────────────────────────────────────


class TestAttributes:
    def test_units_are_carried_but_flagged_unconfirmed(self, ingested):
        assert ingested[OBSERVATION_MEAN].attrs["units_status"] == UNITS_STATUS
        assert "reanalysis" in ingested[OBSERVATION_MEAN].attrs["units_provenance"]
        assert ingested.attrs["variable_lai_units"] == "m2 m-2"
        assert ingested.attrs["variable_lai_source_name"] == "LAI"

    def test_the_snapshot_key_is_labeled_nominal(self, ingested):
        # Not an instant and not an interval: a bookkeeping key.
        assert ingested["time"].attrs["time_label"] == "nominal"
        assert "not observation dates" in ingested["time"].attrs["time_label_note"]

    def test_diagonality_is_recorded_in_the_file(self, ingested):
        assert ingested.attrs["covariances_all_diagonal"] == "true"

    def test_the_source_resolution_is_recorded(self, ingested):
        assert ingested.attrs["source_resolution"] == "annual"


# ── the manifest handshake ────────────────────────────────────────────────────


class TestManifestChecks:
    def _run(self, synthetic, manifest: dict) -> int:
        synthetic["manifest"].write_text(json.dumps(manifest))
        return ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(synthetic["out"]),
            ]
        )

    def test_a_non_diagonal_source_is_refused(self, synthetic):
        manifest = _manifest_for()
        manifest["covariances_all_diagonal"] = False
        manifest["max_abs_offdiagonal"] = 1e-9
        assert self._run(synthetic, manifest) == 1
        assert not synthetic["out"].exists()

    def test_an_inconsistent_diagonality_claim_is_refused(self, synthetic):
        manifest = _manifest_for()
        manifest["max_abs_offdiagonal"] = 3.0  # contradicts the True flag
        assert self._run(synthetic, manifest) == 1
        assert not synthetic["out"].exists()

    def test_a_row_count_disagreement_is_refused(self, synthetic):
        manifest = _manifest_for()
        manifest["n_rows"] = len(SYNTHETIC_ROWS) + 1
        assert self._run(synthetic, manifest) == 1
        assert not synthetic["out"].exists()

    def test_a_per_variable_count_disagreement_is_refused(self, synthetic):
        manifest = _manifest_for()
        manifest["counts_by_snapshot_variable"]["2012-07-15"]["LAI"] += 1
        manifest["n_rows"] += 1
        assert self._run(synthetic, manifest) == 1

    def test_a_truncated_extreme_is_refused(self, synthetic):
        # What a writer or parser losing precision would look like.
        manifest = _manifest_for()
        manifest["extremes"]["TotSoilCarb"]["mean_max"] = "40.000000000000001"
        assert self._run(synthetic, manifest) == 1
        assert not synthetic["out"].exists()

    def test_a_manifest_missing_a_key_is_refused(self, synthetic):
        manifest = _manifest_for()
        del manifest["extremes"]
        assert self._run(synthetic, manifest) == 1

    def test_a_manifest_naming_other_variables_is_refused(self, synthetic):
        manifest = _manifest_for()
        manifest["variables"] = ["LAI", "AbvGrndWood", "SoilMoistFrac", "TotSoilCarb"]
        assert self._run(synthetic, manifest) == 1

    def test_unparseable_json_is_reported_not_raised(self, synthetic):
        synthetic["manifest"].write_text("{not json")
        status = ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(synthetic["out"]),
            ]
        )
        assert status == 1


class TestOtherIngestChecks:
    def test_a_duplicated_triple_is_refused(self, synthetic, tmp_path):
        rows = SYNTHETIC_ROWS + [SYNTHETIC_ROWS[0]]
        _write_long_table(synthetic["long_table"], rows)
        synthetic["manifest"].write_text(json.dumps(_manifest_for(rows)))
        status = ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(synthetic["out"]),
            ]
        )
        assert status == 1
        assert not synthetic["out"].exists()

    def test_a_site_absent_from_the_site_table_is_refused(self, synthetic):
        rows = SYNTHETIC_ROWS + [("2012-07-15", 99, "LAI", 1.0, 1.0)]
        _write_long_table(synthetic["long_table"], rows)
        synthetic["manifest"].write_text(json.dumps(_manifest_for(rows)))
        status = ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(synthetic["out"]),
            ]
        )
        assert status == 1

    def test_help_works(self):
        with pytest.raises(SystemExit) as caught:
            ingest.parse_args(["--help"])
        assert caught.value.code == 0

    def test_creates_its_output_directory(self, synthetic):
        out = synthetic["tmp_path"] / "nested" / "deeper" / "constraints_annual.nc"
        status = ingest.main(
            [
                "--long-table", str(synthetic["long_table"]),
                "--manifest", str(synthetic["manifest"]),
                "--sites", str(synthetic["sites"]),
                "--out", str(out),
            ]
        )
        assert status == 0
        assert out.exists()

    def test_leaves_no_partial_file_behind_on_success(self, synthetic, ingested):
        partial = synthetic["out"].with_suffix(synthetic["out"].suffix + ".partial")
        assert not partial.exists()

    def test_a_failed_round_trip_leaves_nothing_at_the_canonical_path(
        self, synthetic, monkeypatch
    ):
        # The reason the write is staged: a corrupt product must not appear
        # where a later read will pick it up.
        def explode(dataset, path):
            raise ingest.IngestError("synthetic round-trip failure")

        monkeypatch.setattr(ingest, "check_round_trip", explode)
        with pytest.raises(ingest.IngestError):
            ingest.ingest(
                synthetic["long_table"],
                synthetic["manifest"],
                synthetic["sites"],
                synthetic["out"],
            )
        assert not synthetic["out"].exists()


# ── the reader's validation ───────────────────────────────────────────────────


class TestLoadConstraintsValidation:
    def test_a_missing_file_names_the_commands_that_build_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="export_constraints.R"):
            load_constraints(tmp_path / "absent.nc")

    def _write(self, dataset: xr.Dataset, path: Path) -> Path:
        dataset.to_netcdf(path, engine="h5netcdf")
        return path

    def test_missing_a_data_variable_is_rejected(self, ingested, tmp_path):
        dropped = ingested.drop_vars(OBSERVATION_VARIANCE)
        path = self._write(dropped, tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="missing data variables"):
            load_constraints(path)

    def test_a_permuted_variable_coordinate_is_rejected(self, ingested, tmp_path):
        permuted = ingested.isel(variable=[1, 0, 2, 3])
        path = self._write(permuted, tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="variable coordinate is"):
            load_constraints(path)

    def test_missing_lon_lat_is_rejected(self, ingested, tmp_path):
        path = self._write(ingested.drop_vars(["lon", "lat"]), tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="missing the 'lon' coordinate"):
            load_constraints(path)

    def test_transposed_dims_are_rejected(self, ingested, tmp_path):
        path = self._write(
            ingested.transpose("time", "site", "variable"), tmp_path / "bad.nc"
        )
        with pytest.raises(ValueError, match="has dims"):
            load_constraints(path)

    def test_a_descending_site_axis_is_rejected(self, ingested, tmp_path):
        reversed_sites = ingested.isel(site=slice(None, None, -1))
        path = self._write(reversed_sites, tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="strictly ascending"):
            load_constraints(path)


# ── the canonical view ────────────────────────────────────────────────────────


class TestConstraintFields:
    def test_returns_one_field_per_variable_in_registry_order(self, ingested):
        fields = constraint_fields(ingested)
        assert list(fields) == list(CONSTRAINT_VARIABLES)

    def test_fields_have_canonical_dims(self, ingested):
        # dims a subset of (member, site, time), with no 'variable' dim.
        for name, field in constraint_fields(ingested).items():
            assert field.dims == ("site", "time")
            assert set(field.dims) <= {"member", "site", "time"}
            assert field.name == name

    def test_fields_keep_lon_lat_on_site(self, ingested):
        for field in constraint_fields(ingested).values():
            assert field["lon"].dims == ("site",)
            assert field["lat"].dims == ("site",)

    def test_fields_carry_units_and_the_unconfirmed_flag(self, ingested):
        fields = constraint_fields(ingested)
        assert fields["lai"].attrs["units"] == "m2 m-2"
        assert fields["lai"].attrs["long_name"] == "Leaf area index"
        assert fields["lai"].attrs["units_status"] == UNITS_STATUS
        assert fields["lai"].attrs["source_name"] == "LAI"

    def test_variance_fields_carry_squared_units(self, ingested):
        fields = constraint_fields(ingested, statistic="variance")
        assert fields["total_soil_carbon"].attrs["units"] == "(kg C m-2)2"
        assert "variance" in fields["total_soil_carbon"].attrs["long_name"]

    def test_values_match_the_stored_form(self, ingested):
        fields = constraint_fields(ingested)
        expected = ingested[OBSERVATION_MEAN].sel(variable="lai").values
        assert np.array_equal(fields["lai"].values, expected, equal_nan=True)

    def test_an_unknown_statistic_is_rejected(self, ingested):
        with pytest.raises(ValueError, match="must be 'mean' or 'variance'"):
            constraint_fields(ingested, statistic="sd")


# ── against the real source ───────────────────────────────────────────────────

real_source = pytest.mark.skipif(
    shutil.which("Rscript") is None or not RAW_MEAN.exists() or not RAW_COV.exists(),
    reason="needs Rscript and the raw .Rdata files",
)

#: Values pulled independently out of R, not through the export path, so that
#: this compares the pipeline against the source rather than against itself.
#: Reproduce with the snippet in the pull request description.
GOLDEN = [
    ("2012-07-15", 1, "TotSoilCarb", 74.285900878906247, 2733.3597351932531),
    ("2015-07-15", 1, "SoilMoistFrac", 39.795073866844199, 0.11057357240832996),
    ("2020-07-15", 100, "LAI", 0.10000000000000001, 0.43560000000000004),
    ("2020-07-15", 100, "TotSoilCarb", 47.221975708007811, 1483.739946376169),
    ("2024-07-15", 4102, "SoilMoistFrac", 32.3921382427216, 1.5981839466459915),
    ("2018-07-15", 7999, "LAI", 4.2999999999999998, 0.48999999999999994),
    ("2016-07-15", 3281, "TotSoilCarb", 32.995159912109372, 364.64238131884485),
    ("2022-07-15", 500, "SoilMoistFrac", 24.294959008693699, 9.2264278039089529),
]


@pytest.fixture(scope="module")
def real_export(tmp_path_factory):
    """Run the R export against the tracked source files, once."""
    directory = tmp_path_factory.mktemp("real")
    long_table = directory / "long.csv"
    manifest = directory / "manifest.json"
    before = (RAW_MEAN.stat().st_mtime_ns, RAW_COV.stat().st_mtime_ns)
    completed = subprocess.run(
        [
            "Rscript", str(EXPORT_SCRIPT),
            "--mean", str(RAW_MEAN), "--cov", str(RAW_COV),
            "--out", str(long_table), "--manifest", str(manifest),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return {
        "long_table": long_table,
        "manifest": json.loads(manifest.read_text()),
        "manifest_path": manifest,
        "raw_mtimes_before": before,
    }


needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="needs Rscript"
)

#: An R snippet building a three-site, one-snapshot pair of objects in the
#: source's shape. ``OFFDIAG`` is substituted with the off-diagonal element to
#: plant, so the same fixture serves the control and the mutant.
SYNTHETIC_RDATA_SNIPPET = """
out <- commandArgs(trailingOnly = TRUE)[[1]]
frame_of <- function(...) data.frame(..., check.names = FALSE)
obs.mean <- list(`2012-07-15` = list(
  `1` = frame_of(LAI = 1.5, TotSoilCarb = 20),
  `2` = frame_of(TotSoilCarb = 30),
  `3` = frame_of(AbvGrndWood = 5, LAI = 2, TotSoilCarb = 40)
))
two <- matrix(c(0.4356, OFFDIAG, OFFDIAG, 400), nrow = 2)
obs.cov <- list(`2012-07-15` = list(
  `1` = two,
  `2` = 900,
  `3` = diag(c(100, 0.49, 1600))
))
save(obs.mean, file = file.path(out, "obs.mean.Rdata"))
save(obs.cov, file = file.path(out, "obs.cov.Rdata"))
"""


def _write_synthetic_rdata(directory: Path, off_diagonal: str) -> Path:
    """Build a tiny source pair in R, with *off_diagonal* planted."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "make.R"
    script.write_text(SYNTHETIC_RDATA_SNIPPET.replace("OFFDIAG", off_diagonal))
    completed = subprocess.run(
        ["Rscript", str(script), str(directory)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return directory


def _run_export(directory: Path, extra: list[str] | None = None):
    """Run the export over a synthetic source pair in *directory*."""
    return subprocess.run(
        [
            "Rscript", str(EXPORT_SCRIPT),
            "--mean", str(directory / "obs.mean.Rdata"),
            "--cov", str(directory / "obs.cov.Rdata"),
            "--out", str(directory / "long.csv"),
            "--manifest", str(directory / "manifest.json"),
            "--expect-sites", "3",
            *(extra or []),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


@needs_r
class TestExportDiagonalityAssertion:
    """The assertion the whole storage choice rests on, exercised both ways.

    Once the CSV exists the off-diagonal is gone, so this is the last place the
    claim can be checked -- which makes it worth proving the check fires rather
    than assuming it would.
    """

    def test_a_diagonal_source_is_accepted(self, tmp_path):
        directory = _write_synthetic_rdata(tmp_path / "ok", "0")
        completed = _run_export(directory)
        assert completed.returncode == 0, completed.stderr
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["covariances_all_diagonal"] is True
        assert manifest["max_abs_offdiagonal"] == 0
        assert manifest["n_rows"] == 6

    def test_a_planted_off_diagonal_element_is_rejected(self, tmp_path):
        directory = _write_synthetic_rdata(tmp_path / "bad", "1e-9")
        completed = _run_export(directory)
        assert completed.returncode == 1
        assert "off-diagonal" in completed.stderr
        assert not (directory / "long.csv").exists()
        assert not (directory / "manifest.json").exists()

    def test_variances_are_paired_with_the_right_variable(self, tmp_path):
        # Site 1 observes LAI and TotSoilCarb, whose variances are 0.4356 and
        # 400; site 3 observes three variables. A positional slip shows up here.
        directory = _write_synthetic_rdata(tmp_path / "pairing", "0")
        assert _run_export(directory).returncode == 0
        table = read_long_table(directory / "long.csv")
        rows = table.set_index(["site_id", "variable"])
        assert rows.loc[(1, "LAI"), "variance"] == 0.4356
        assert rows.loc[(1, "TotSoilCarb"), "variance"] == 400.0
        assert rows.loc[(2, "TotSoilCarb"), "variance"] == 900.0
        assert rows.loc[(3, "AbvGrndWood"), "variance"] == 100.0
        assert rows.loc[(3, "LAI"), "variance"] == 0.49
        assert rows.loc[(3, "TotSoilCarb"), "variance"] == 1600.0

    def test_a_site_count_disagreement_is_rejected(self, tmp_path):
        directory = _write_synthetic_rdata(tmp_path / "count", "0")
        completed = _run_export(directory, extra=["--expect-sites", "4"])
        assert completed.returncode == 1
        assert "site names are not" in completed.stderr

    def test_an_unknown_option_is_rejected(self, tmp_path):
        directory = _write_synthetic_rdata(tmp_path / "opt", "0")
        completed = _run_export(directory, extra=["--nonsense", "1"])
        assert completed.returncode == 1
        assert "unknown option" in completed.stderr


class TestDocstringExamples:
    """The Usage examples in the module docstring have to actually run.

    Extracted from the shipped docstring rather than copied here, so that the
    text and the tested code cannot diverge. They read the product at the
    default path, so they are skipped where the pipeline has not been run.

    This catches an example that no longer *works* -- a renamed function, a
    stale keyword, a variable that is gone -- which is how examples usually
    rot. It does not check that an example still says something sensible; that
    is what the tests of the functions themselves are for.
    """

    @staticmethod
    def _usage_code_blocks() -> list[str]:
        usage = constraints_module.__doc__.split("Usage\n-----", 1)[1]
        blocks = re.findall(r"::\n\n((?:(?: {4}.*)?\n)+)", usage)
        return [textwrap.dedent(block) for block in blocks]

    def test_the_docstring_has_usage_examples(self):
        assert len(self._usage_code_blocks()) >= 3

    @pytest.mark.skipif(
        not default_constraints_path().exists(),
        reason="needs the built product at the default path",
    )
    def test_every_usage_example_executes(self):
        namespace: dict = {}
        for index, code in enumerate(self._usage_code_blocks(), start=1):
            compiled = compile(code, f"<docstring block {index}>", "exec")
            exec(compiled, namespace)  # noqa: S102 - the docstring is the input


@real_source
@pytest.mark.slow
class TestRealExport:
    def test_leaves_the_source_files_untouched(self, real_export):
        after = (RAW_MEAN.stat().st_mtime_ns, RAW_COV.stat().st_mtime_ns)
        assert after == real_export["raw_mtimes_before"]

    def test_every_covariance_was_exactly_diagonal(self, real_export):
        # The claim the whole storage choice rests on.
        assert real_export["manifest"]["covariances_all_diagonal"] is True
        assert real_export["manifest"]["max_abs_offdiagonal"] == 0

    def test_row_count_matches_the_documented_coverage(self, real_export):
        assert real_export["manifest"]["n_rows"] == REAL_N_ROWS

    def test_per_variable_counts_match_the_documented_coverage(self, real_export):
        for name, expected in REAL_COUNTS.items():
            assert real_export["manifest"]["extremes"][name]["n"] == expected

    def test_per_variable_extremes_are_distinct(self, real_export):
        # A data.table scoping collision once made every variable report the
        # same global extremes; four identical entries is the signature.
        maxima = {
            name: entry["mean_max"]
            for name, entry in real_export["manifest"]["extremes"].items()
        }
        assert len(set(maxima.values())) == len(maxima), maxima

    def test_the_expected_zero_variances_are_present(self, real_export):
        extremes = real_export["manifest"]["extremes"]
        assert extremes["AbvGrndWood"]["n_nonpositive_variance"] == 929
        for name in ("LAI", "SoilMoistFrac", "TotSoilCarb"):
            assert extremes[name]["n_nonpositive_variance"] == 0

    def test_the_empty_site_snapshots_are_the_documented_ones(self, real_export):
        empty = real_export["manifest"]["empty_site_snapshots"]
        assert sorted(empty) == EMPTY_SNAPSHOTS
        for snapshot in EMPTY_SNAPSHOTS:
            assert sorted(empty[snapshot]) == EMPTY_SITES
        assert real_export["manifest"]["n_empty_site_snapshots"] == 18

    def test_thirteen_annual_snapshots(self, real_export):
        assert real_export["manifest"]["n_snapshots"] == 13
        assert real_export["manifest"]["snapshot_dates"][0] == "2012-07-15"
        assert real_export["manifest"]["snapshot_dates"][-1] == "2024-07-15"


@pytest.fixture(scope="module")
def real_dataset(real_export, tmp_path_factory):
    """The real source, exported and ingested, loaded back once."""
    sites = REPO_ROOT / "data" / "processed" / "sites" / "sites.csv"
    if not sites.exists():
        pytest.skip("needs data/processed/sites/sites.csv; run ingest_sites.py")
    out = tmp_path_factory.mktemp("real_out") / "constraints_annual.nc"
    status = ingest.main(
        [
            "--long-table", str(real_export["long_table"]),
            "--manifest", str(real_export["manifest_path"]),
            "--sites", str(sites),
            "--out", str(out),
        ]
    )
    assert status == 0
    return load_constraints(out)


@real_source
@pytest.mark.slow
class TestRealIngest:
    def test_covers_the_whole_site_pool(self, real_dataset):
        assert real_dataset.sizes["site"] == 8000
        assert real_dataset["site"].values[0] == 1
        assert real_dataset["site"].values[-1] == 8000

    @pytest.mark.parametrize("snapshot,site,source_variable,mean,variance", GOLDEN)
    def test_golden_values_are_bitwise_equal_to_the_source(
        self, real_dataset, snapshot, site, source_variable, mean, variance
    ):
        variable = SOURCE_VARIABLE_NAMES[source_variable]
        assert float(
            real_dataset[OBSERVATION_MEAN].sel(
                site=site, time=snapshot, variable=variable
            )
        ) == mean
        assert float(
            real_dataset[OBSERVATION_VARIANCE].sel(
                site=site, time=snapshot, variable=variable
            )
        ) == variance

    @pytest.mark.parametrize("site", EMPTY_SITES)
    @pytest.mark.parametrize("snapshot", EMPTY_SNAPSHOTS)
    def test_the_empty_site_snapshots_are_all_nan(self, real_dataset, site, snapshot):
        cell = real_dataset[OBSERVATION_MEAN].sel(site=site, time=snapshot)
        assert bool(np.all(np.isnan(cell.values)))

    def test_soil_moisture_is_absent_before_2015(self, real_dataset):
        early = real_dataset[OBSERVATION_MEAN].sel(
            variable="soil_moisture_fraction",
            time=slice("2012-01-01", "2014-12-31"),
        )
        assert bool(np.all(np.isnan(early.values)))

    def test_aboveground_wood_is_absent_in_2024(self, real_dataset):
        assert bool(np.all(np.isnan(
            real_dataset[OBSERVATION_MEAN]
            .sel(variable="aboveground_wood_carbon", time="2024-07-15")
            .values
        )))

    def test_observed_cell_counts_match_the_documented_coverage(self, real_dataset):
        for index, name in enumerate(CONSTRAINT_VARIABLES):
            values = real_dataset[OBSERVATION_MEAN].values[:, :, index]
            observed = int(np.isfinite(values).sum())
            assert observed == REAL_COUNTS_PROCESSED[name]

    def test_the_file_is_small_enough_to_load_whole(self, real_dataset):
        cells = int(np.prod(real_dataset[OBSERVATION_MEAN].shape))
        assert cells == 8000 * 13 * 4

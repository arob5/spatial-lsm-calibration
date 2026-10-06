"""Tests for the site dims: the sites, their labels and covariates, and
reading labeled values at the sites.

The coords are the site ids and the classes some site carries, in declared
order; a label map is each site's class, or each class's coarser class;
values are read pointwise at each site's labels, batch dims first and
element axes last; ``site_fields`` are fields; ``select`` is the selection
of sites by class. Every check is provoked once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import site_table_of
from sipnet_calibration.fields import validate_field
from sipnet_calibration.probability import POSITIVE, SIMPLEX, ArraySpec, Layout
from sipnet_calibration.site_dims import SiteDims

SITES = (1, 27, 4711)
PFT = ("deciduous", "conifer", "deciduous")
BIOME = ("temperate", "boreal", "temperate")


def site_table(**columns) -> pd.DataFrame:
    return site_table_of(*SITES).assign(**columns)


@pytest.fixture(scope="module")
def site_dims() -> SiteDims:
    return SiteDims(site_table=site_table(elevation=[100.0, 2500.0, 800.0]),
                    site_labels={"pft": PFT, "biome": BIOME}, covariate_names=["elevation"])


@pytest.fixture(scope="module")
def layout(site_dims) -> Layout:
    return Layout(
        [
            ArraySpec("share", units="1"),
            ArraySpec("allocation", units="1", support=SIMPLEX, indexed_by=("pft",),
                      element_axes={"part": ("leaf", "wood", "root")}),
            ArraySpec("soil", units="kg m-2", support=POSITIVE, indexed_by=("site",)),
            ArraySpec("rate", units="yr-1", support=POSITIVE, indexed_by=("site", "pft")),
        ],
        coords={"site": site_dims.coords["site"], "pft": site_dims.coords["pft"]},
    )


@pytest.fixture(scope="module")
def values(layout):
    theta = np.random.default_rng(0).standard_normal((2, layout.unconstrained.size))
    values_by_name = layout.flat_to_values(layout.to_natural(theta))
    return values_by_name, xr.Dataset(layout.values_to_labeled(values_by_name, batch_dims=("sample",)))


# ── the dims ──────────────────────────────────────────────────────────────────


def test_the_coords_are_the_site_ids_and_the_classes_present(site_dims):
    coords = site_dims.coords
    assert list(coords) == ["site", "pft", "biome"]
    assert coords["site"].tolist() == list(SITES) and coords["site"].dtype == np.int32
    assert coords["pft"].tolist() == ["conifer", "deciduous"]
    assert site_dims.sites == SITES and site_dims.n_sites == 3


def test_classes_keep_their_declared_order():
    table = pd.DataFrame({
        "site_id": [1, 27, 4711, 9000],
        "label": pd.Categorical(["temperate", "boreal", "temperate", "grass"], categories=["grass", "temperate", "boreal"]),
    })
    assert SiteDims(site_table=site_table(), site_labels={"pft": table}).coords["pft"].tolist() == ["temperate", "boreal"]


def test_site_labels_given_as_a_series_are_read_by_site_id():
    labels = pd.Series(["conifer", "deciduous", "deciduous"], index=[27, 1, 4711])
    assert SiteDims(site_table=site_table(), site_labels={"pft": labels}).site_labels["pft"] == PFT
    with pytest.raises(KeyError, match="no label for site"):
        SiteDims(site_table=site_table(), site_labels={"pft": labels.iloc[:2]})


def test_the_site_labels_it_holds_cannot_be_changed_through_the_callers_table():
    table = pd.DataFrame({"site_id": list(SITES), "label": list(PFT)})
    site_dims = SiteDims(site_table=site_table(), site_labels={"pft": table})
    table.loc[0, "label"] = "grass"
    assert site_dims.site_labels["pft"] == PFT
    held = site_dims.site_table
    held.loc[0, "lon"] = 0.0
    assert site_dims.site_table.loc[0, "lon"] != 0.0


def test_the_site_table_keeps_what_was_named():
    site_dims = SiteDims(site_table=site_table(temperature=[1.0, 2.0, 3.0], cluster=[4, 5, 6]),
                         site_labels={"pft": PFT}, covariate_names=["temperature"])
    assert list(site_dims.site_table.columns) == ["site_id", "lon", "lat", "temperature", "pft"]
    assert repr(site_dims) == "SiteDims(sites=3, site_labels=['pft'], covariate_names=['temperature'])"


# ── label maps and covariates ─────────────────────────────────────────────────


def test_labels_along_site_are_each_sites_class(site_dims):
    labels = site_dims.labels("pft")
    assert labels.name == "pft" and labels.dims == ("site",)
    assert labels.values.tolist() == list(PFT) and labels["site"].values.tolist() == list(SITES)


def test_labels_along_a_finer_dim_are_the_coarser_class(site_dims):
    biome_of_pft = site_dims.labels("biome", along="pft")
    assert biome_of_pft.dims == ("pft",) and biome_of_pft.name == "biome"
    assert dict(zip(biome_of_pft["pft"].values, biome_of_pft.values)) == {"conifer": "boreal", "deciduous": "temperate"}


def test_dims_that_do_not_nest_are_refused():
    site_dims = SiteDims(site_table=site_table(), site_labels={"pft": PFT, "biome": ("temperate", "boreal", "boreal")})
    with pytest.raises(ValueError, match="does not nest in 'biome'"):
        site_dims.labels("biome", along="pft")


def test_labels_name_two_dims(site_dims):
    with pytest.raises(ValueError, match="labels nothing but itself"):
        site_dims.labels("site")
    with pytest.raises(KeyError, match="no site labels 'grass'"):
        site_dims.labels("grass")
    with pytest.raises(KeyError, match="neither 'site' nor a site-labels name"):
        site_dims.labels("pft", along="region")
    with pytest.raises(ValueError, match="maps a dim to itself"):
        site_dims.labels("pft", along="pft")


def test_a_covariate_is_a_constant_on_site(site_dims):
    covariate = site_dims.covariate("elevation")
    assert covariate.dims == ("site",) and covariate.dtype == np.float64
    assert covariate.values.tolist() == [100.0, 2500.0, 800.0]
    with pytest.raises(KeyError, match="no site covariate"):
        site_dims.covariate("slope")


# ── values at the sites ───────────────────────────────────────────────────────


def test_values_are_read_pointwise_at_each_sites_labels(site_dims, values):
    values_by_name, dataset = values
    at_sites = site_dims.at_sites(dataset)
    pft = [site_dims.coords["pft"].get_loc(p) for p in PFT]
    np.testing.assert_allclose(at_sites["allocation"], np.asarray(values_by_name["allocation"])[:, pft])
    rate = np.asarray(values_by_name["rate"])
    np.testing.assert_allclose(at_sites["rate"], rate[:, np.arange(3), pft])
    np.testing.assert_allclose(at_sites["soil"], values_by_name["soil"])
    np.testing.assert_allclose(at_sites["share"], np.repeat(np.asarray(values_by_name["share"])[:, None], 3, axis=1))


def test_batch_dims_come_first_and_element_axes_last(site_dims, values):
    at_sites = site_dims.at_sites(values[1].transpose("part", ...))
    assert at_sites["allocation"].dims == ("sample", "site", "part")
    assert at_sites["share"].dims == ("sample", "site")
    assert at_sites["lon"].values.tolist() == site_table()["lon"].tolist()
    assert at_sites["soil"].attrs["units"] == "kg m-2"


def test_a_value_on_a_subset_of_the_labels_is_read_at_the_sites(site_dims, values):
    extra = values[1].reindex(site=[*SITES, 9999], fill_value=0.0)
    np.testing.assert_allclose(site_dims.at_sites(extra)["soil"], site_dims.at_sites(values[1])["soil"])


def test_a_value_lacking_a_sites_label_is_refused(site_dims, values):
    with pytest.raises(KeyError, match="no value at the pft label"):
        site_dims.at_sites(values[1].sel(pft=["conifer"]))
    with pytest.raises(KeyError, match="no value at the site label"):
        site_dims.at_sites(values[1].sel(site=[1, 27]))
    with pytest.raises(ValueError, match="has no 'site' coordinate"):
        site_dims.at_sites(values[1].drop_vars("site"))
    with pytest.raises(TypeError, match="are an xarray Dataset"):
        site_dims.at_sites(values[1]["soil"])


def test_site_fields_are_fields(site_dims, values):
    site_fields = site_dims.site_fields(values[1])
    assert set(site_fields.data_vars) == {"share", "soil", "rate", "allocation.leaf", "allocation.wood", "allocation.root"}
    for name, variable in site_fields.data_vars.items():
        validate_field(variable, message_name=str(name))
        assert variable.dims == ("sample", "site")
    np.testing.assert_allclose(site_fields["allocation.wood"], site_dims.at_sites(values[1])["allocation"][..., 1])
    assert site_fields["allocation.wood"].attrs["element"] == "wood"
    assert site_fields["share"].attrs["units"] == "1"


# ── selection ─────────────────────────────────────────────────────────────────


def test_select_keeps_the_sites_of_a_class(site_dims):
    deciduous = site_dims.select(pft=["deciduous"])
    assert deciduous.sites == (1, 4711) and deciduous.coords["pft"].tolist() == ["deciduous"]
    assert deciduous.site_table["elevation"].tolist() == [100.0, 800.0]
    assert site_dims.select([27, 4711], biome=["temperate"]).sites == (4711,)
    # The declared classes survive, so selecting back is the same order.
    assert site_dims.select(pft=["deciduous"]).site_table["pft"].cat.categories.tolist() == ["conifer", "deciduous"]


def test_selection_is_refused_by_the_projects_rules(site_dims):
    with pytest.raises(KeyError, match="no site"):
        site_dims.select([2])
    with pytest.raises(KeyError, match="no site labels 'region'"):
        site_dims.select(region=["x"])
    with pytest.raises(KeyError, match="no label \\['grass'\\]"):
        site_dims.select(pft=["grass"])
    with pytest.raises(TypeError, match="are strings"):
        site_dims.select(pft=[1])
    with pytest.raises(TypeError, match="must be a sequence"):
        site_dims.select(pft="conifer")
    with pytest.raises(ValueError, match="keeps no site"):
        site_dims.select([27], pft=["deciduous"])
    with pytest.raises(ValueError, match="keeps nothing"):
        site_dims.select(pft=[])
    with pytest.raises(ValueError, match="with sites="):
        site_dims.select(site=[1])


# ── construction ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"site_table": site_table_of()}, ValueError, "no site"),
        ({"site_table": site_table_of(27, 1)}, ValueError, "ascending"),
        ({"site_labels": {"pft": [1, 2, 1]}}, TypeError, "not strings"),
        ({"site_labels": {"pft": ["a"]}}, ValueError, "one per site"),
        ({"site_labels": {"pft": pd.DataFrame({"site_id": [1, 27], "label": ["a", "b"]})}}, KeyError, "no label for site"),
        ({"site_labels": {"pft": pd.DataFrame({"site_id": list(SITES)})}}, KeyError, "lack the column"),
        ({"covariate_names": ["temperature"]}, KeyError, "not a column"),
        ({"site_table": site_table(cluster=[4, 5, 6]), "covariate_names": ["cluster"]}, TypeError, "not float64"),
        ({"site_table": site_table(gap=[1.0, np.nan, 2.0]), "covariate_names": ["gap"]}, ValueError, "not finite"),
        ({"site_labels": {"time": PFT}}, ValueError, "reserved name"),
        ({"site_table": site_table(sample=[1.0, 2.0, 3.0]), "covariate_names": ["sample"]}, ValueError, "reserved name"),
        ({"site_table": site_table(pft=[1.0, 2.0, 3.0]), "site_labels": {"pft": PFT}, "covariate_names": ["pft"]},
         ValueError, "both a site-labels name and a site covariate"),
    ],
)
def test_malformed_site_dims_are_refused(arguments, error, match):
    with pytest.raises(error, match=match):
        SiteDims(**{"site_table": site_table(), **arguments})

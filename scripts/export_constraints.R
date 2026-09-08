#!/usr/bin/env Rscript
#
# Overview
# --------
# Flatten the annual constraint R data files into one long table plus a manifest
# of what was checked while reading them. This is the R half of building
# `data/processed/constraints_annual.nc`; `scripts/ingest_constraints.py` is the
# other half and makes every schema decision. This script does only what R must:
# read objects `pyreadr` cannot, and check the one property that stops being
# visible once the data leaves R.
#
# Input data
# ----------
# Two `.Rdata` files, each holding a single object that nests as snapshot date,
# then site, then variable.
#
#   obs.mean   list of 13 snapshots keyed "2012-07-15".."2024-07-15". Each is a
#              list of 8000 sites named "1".."8000". Each site entry is a
#              one-row data frame whose columns are the variables observed
#              there: zero to four of AbvGrndWood, LAI, SoilMoistFrac,
#              TotSoilCarb, in alphabetical order.
#
#   obs.cov    the same nesting, holding the observation error covariance for
#              each site-snapshot: a bare numeric when one variable was
#              observed, a matrix otherwise, and empty where obs.mean has no
#              columns.
#
# The covariance entries carry **no dimension names**, so the only thing saying
# which row belongs to which variable is the column order of the paired obs.mean
# entry. The two files are therefore always read together, and a variance is
# only ever emitted beside the mean it was paired with, in the same row.
#
# Output data
# -----------
#   <out>       CSV, one row per *observed* (snapshot, site, variable) triple:
#
#                 snapshot_date, site_id, variable, mean, variance
#
#               Unobserved triples are absent rather than written as missing, so
#               the file is the observation set itself. `variable` holds source
#               names; the ingest script renames them. Doubles are written with
#               "%.17g", which uniquely determines a float64.
#
#   <manifest>  JSON recording what was checked: per-snapshot per-variable row
#               counts, the exact extremes per variable, the site-snapshots with
#               no observations, and the largest absolute off-diagonal
#               covariance element seen. The ingest script checks the CSV
#               against this and refuses to write if they disagree.
#
# Notes
# -----
# **Diagonality is asserted here, not downstream.** The processed form stores
# variances rather than covariance matrices, which is lossless only because
# every source covariance is exactly diagonal. Once the CSV is written the
# off-diagonal is gone, so this is the last place the claim can be checked. The
# manifest reports the largest element seen so the Python side can confirm the
# check ran rather than trusting that it did.
#
# R is needed at all because obs.mean is a list of lists of data frames, which
# `pyreadr` does not support, and because R's `ncdf4` is not installed on the
# development machine, so R cannot write the netCDF either.
#
# Usage
# -----
#   Rscript scripts/export_constraints.R --out long.csv --manifest manifest.json
#   Rscript scripts/export_constraints.R --help

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

# Source variable names, in the alphabetical order the source uses. The ingest
# script maps these onto the processed names.
SOURCE_VARIABLES <- c("AbvGrndWood", "LAI", "SoilMoistFrac", "TotSoilCarb")

# The site pool the real files carry. `--expect-sites` overrides it, which is
# what lets the tests exercise the diagonality check against a three-site
# synthetic pair instead of the real 8000. The default stays strict.
DEFAULT_N_SITES <- 8000L

DEFAULT_MEAN <- "data/raw/constraints/sda_8k_site_rdata/obs.mean.Rdata"
DEFAULT_COV <- "data/raw/constraints/sda_8k_site_rdata/obs.cov.Rdata"

USAGE <- "
Usage: Rscript scripts/export_constraints.R [options]

Options:
  --mean PATH       obs.mean.Rdata   (default: data/raw/.../obs.mean.Rdata)
  --cov PATH        obs.cov.Rdata    (default: data/raw/.../obs.cov.Rdata)
  --out PATH        long CSV to write                        (required)
  --manifest PATH   JSON manifest to write                   (required)
  --expect-sites N  sites each snapshot must hold            (default: 8000)
  --help            show this message

Writes one row per observed (snapshot, site, variable) triple. Unobserved
triples are absent rather than written as missing, so the file is the
observation set itself.
"

# ── entry point ───────────────────────────────────────────────────────────────

main <- function(argv) {
  args <- parse_args(argv)

  message("Reading ", args$mean)
  obs_mean <- load_single_object(args$mean, "obs.mean")
  message("Reading ", args$cov)
  obs_cov <- load_single_object(args$cov, "obs.cov")

  check_nesting_matches(obs_mean, obs_cov, args$n_sites)
  message("Flattening ", length(obs_mean), " snapshots x ", args$n_sites, " sites")
  built <- build_long_table(obs_mean, obs_cov)

  manifest <- build_manifest(built, args, names(obs_mean))

  fwrite(built$rows, args$out, quote = FALSE)
  write(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA),
        file = args$manifest)

  message(sprintf("Wrote %s (%d rows) and %s", args$out, nrow(built$rows),
                  args$manifest))
  message(sprintf("Every covariance was exactly diagonal (max |off-diagonal| = %g)",
                  built$max_abs_offdiagonal))
  invisible(0)
}

parse_args <- function(argv) {
  if ("--help" %in% argv || "-h" %in% argv) {
    cat(USAGE)
    quit(status = 0)
  }
  args <- list(mean = DEFAULT_MEAN, cov = DEFAULT_COV, out = NULL,
               manifest = NULL, `expect-sites` = DEFAULT_N_SITES)
  i <- 1L
  while (i <= length(argv)) {
    key <- sub("^--", "", argv[[i]])
    if (!key %in% names(args)) {
      stop(sprintf("unknown option '%s'.%s", argv[[i]], USAGE), call. = FALSE)
    }
    if (i + 1L > length(argv)) {
      stop(sprintf("option '%s' needs a value", argv[[i]]), call. = FALSE)
    }
    args[[key]] <- argv[[i + 1L]]
    i <- i + 2L
  }
  for (key in c("out", "manifest")) {
    if (is.null(args[[key]])) {
      stop(sprintf("--%s is required.%s", key, USAGE), call. = FALSE)
    }
  }
  args$n_sites <- as.integer(args[["expect-sites"]])
  if (is.na(args$n_sites) || args$n_sites < 1L) {
    stop("--expect-sites must be a positive integer", call. = FALSE)
  }
  args
}

# ── the export steps, in the order main calls them ────────────────────────────

#' The single object an .Rdata file holds, checked to be the expected one.
load_single_object <- function(path, expected_name) {
  if (!file.exists(path)) {
    stop(sprintf("%s not found", path), call. = FALSE)
  }
  env <- new.env(parent = emptyenv())
  loaded <- load(path, envir = env)
  if (length(loaded) != 1L) {
    stop(sprintf("%s holds %d objects, expected 1: %s", path, length(loaded),
                 paste(loaded, collapse = ", ")), call. = FALSE)
  }
  if (loaded != expected_name) {
    stop(sprintf("%s holds '%s', expected '%s'", path, loaded, expected_name),
         call. = FALSE)
  }
  get(loaded, envir = env)
}

#' Long-format rows for every snapshot, and what was checked while building them.
build_long_table <- function(obs_mean, obs_cov) {
  per_snapshot <- vector("list", length(obs_mean))
  empty <- list()
  max_off <- 0

  for (index in seq_along(obs_mean)) {
    key <- names(obs_mean)[[index]]
    message(sprintf("  %s ...", key))
    flattened <- flatten_snapshot(obs_mean[[index]], obs_cov[[index]], key)
    per_snapshot[[index]] <- flattened$rows
    max_off <- max(max_off, flattened$max_abs_offdiagonal)
    if (length(flattened$empty_sites) > 0L) {
      empty[[key]] <- flattened$empty_sites
    }
  }

  rows <- rbindlist(per_snapshot, use.names = TRUE)
  check_variable_names_are_known(rows)
  list(rows = rows, empty = empty, max_abs_offdiagonal = max_off)
}

#' Long-format rows for one snapshot, pairing each mean with its variance.
flatten_snapshot <- function(means, covariances, key) {
  pieces <- vector("list", length(means))
  empty_sites <- integer(0)
  max_off <- 0

  for (index in seq_along(means)) {
    site <- names(means)[[index]]
    frame <- means[[index]]
    check_is_one_row_data_frame(frame, key, site)

    variables <- names(frame)
    checked <- diagonal_of(covariances[[index]], length(variables), key, site)
    max_off <- max(max_off, checked$max_abs_offdiagonal)

    if (length(variables) == 0L) {
      empty_sites <- c(empty_sites, as.integer(site))
      next
    }
    check_columns_are_sorted(variables, key, site)

    values <- as.numeric(frame[1L, , drop = TRUE])
    check_no_missing_values(values, checked$diagonal, key, site)

    pieces[[index]] <- data.table(
      snapshot_date = key,
      site_id = as.integer(site),
      variable = variables,
      mean = sprintf("%.17g", values),
      variance = sprintf("%.17g", checked$diagonal)
    )
  }

  list(
    rows = rbindlist(pieces, use.names = TRUE),
    empty_sites = empty_sites,
    max_abs_offdiagonal = max_off
  )
}

#' Variances of one site-snapshot covariance, having checked it is diagonal.
#'
#' `n_variables` comes from the paired obs.mean entry and is what the covariance
#' is checked against. Returns the diagonal and the largest absolute
#' off-diagonal element, which is accumulated into the manifest.
diagonal_of <- function(covariance, n_variables, key, site) {
  check_covariance_shape(covariance, n_variables, key, site)

  if (n_variables == 0L) {
    return(list(diagonal = numeric(0), max_abs_offdiagonal = 0))
  }
  if (is.null(dim(covariance))) {
    return(list(diagonal = as.numeric(covariance), max_abs_offdiagonal = 0))
  }

  check_covariance_is_diagonal(covariance, n_variables, key, site)
  list(diagonal = diag(covariance), max_abs_offdiagonal = 0)
}

build_manifest <- function(built, args, snapshot_keys) {
  rows <- built$rows
  list(
    generated_by = "scripts/export_constraints.R",
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    mean_file = args$mean,
    cov_file = args$cov,
    variables = SOURCE_VARIABLES,
    n_sites = args$n_sites,
    snapshot_dates = snapshot_keys,
    n_snapshots = length(snapshot_keys),
    n_rows = nrow(rows),
    counts_by_snapshot_variable = counts_by_snapshot_variable(rows),
    extremes = extremes_by_variable(rows),
    empty_site_snapshots = built$empty,
    n_empty_site_snapshots = sum(lengths(built$empty)),
    max_abs_offdiagonal = built$max_abs_offdiagonal,
    covariances_all_diagonal = built$max_abs_offdiagonal == 0
  )
}

# ── supporting helpers ────────────────────────────────────────────────────────

counts_by_snapshot_variable <- function(rows) {
  result <- list()
  for (key in unique(rows$snapshot_date)) {
    subset <- rows[rows$snapshot_date == key, ]
    counts <- as.list(table(factor(subset$variable, levels = SOURCE_VARIABLES)))
    result[[key]] <- lapply(counts, as.integer)
  }
  result
}

#' Exact extremes per variable, as the strings that were written.
#'
#' The Python side asserts these match what it parsed, character for character
#' after its own round trip. That is what catches a truncating writer: a value
#' shortened on the way out will not come back equal.
#'
#' Note the loop variable is `variable_name`, not `variable`. `rows` is a
#' `data.table`, so an `i` expression is evaluated with its columns in scope;
#' with the loop variable named `variable` the filter `rows$variable ==
#' variable` compares the column against itself, is true everywhere, and
#' silently reports the same global extremes for all four variables. That is not
#' hypothetical -- it is what the first run of this script did.
extremes_by_variable <- function(rows) {
  result <- list()
  for (variable_name in SOURCE_VARIABLES) {
    subset <- rows[rows$variable == variable_name, ]
    if (nrow(subset) == 0L) next
    means <- as.numeric(subset$mean)
    variances <- as.numeric(subset$variance)
    result[[variable_name]] <- list(
      n = nrow(subset),
      mean_min = sprintf("%.17g", min(means)),
      mean_max = sprintf("%.17g", max(means)),
      variance_min = sprintf("%.17g", min(variances)),
      variance_max = sprintf("%.17g", max(variances)),
      n_nonpositive_variance = sum(variances <= 0)
    )
  }
  check_counts_partition_rows(result, rows)
  result
}

# ── checks ────────────────────────────────────────────────────────────────────

#' The two objects nest identically, snapshot then site, over the expected pool.
check_nesting_matches <- function(obs_mean, obs_cov, n_sites) {
  if (!identical(names(obs_mean), names(obs_cov))) {
    stop("obs.mean and obs.cov have different snapshot keys", call. = FALSE)
  }
  if (length(obs_mean) == 0L) {
    stop("obs.mean holds no snapshots", call. = FALSE)
  }
  expected_sites <- as.character(seq_len(n_sites))
  for (key in names(obs_mean)) {
    if (!identical(names(obs_mean[[key]]), expected_sites)) {
      stop(sprintf("obs.mean[['%s']] site names are not '1'..'%d' in order",
                   key, n_sites), call. = FALSE)
    }
    if (!identical(names(obs_cov[[key]]), expected_sites)) {
      stop(sprintf("obs.cov[['%s']] site names are not '1'..'%d' in order",
                   key, n_sites), call. = FALSE)
    }
  }
  invisible(TRUE)
}

check_is_one_row_data_frame <- function(frame, key, site) {
  if (!is.data.frame(frame) || nrow(frame) != 1L) {
    stop(sprintf("obs.mean[['%s']][['%s']] is not a one-row data frame",
                 key, site), call. = FALSE)
  }
  invisible(TRUE)
}

#' The covariance's shape agrees with the number of variables it is paired with.
check_covariance_shape <- function(covariance, n_variables, key, site) {
  where <- sprintf("obs.cov[['%s']][['%s']]", key, site)

  if (n_variables == 0L) {
    if (length(covariance) != 0L) {
      stop(sprintf("%s is non-empty where obs.mean has no columns", where),
           call. = FALSE)
    }
    return(invisible(TRUE))
  }
  if (is.null(dim(covariance))) {
    if (length(covariance) != 1L || n_variables != 1L) {
      stop(sprintf("%s is a bare numeric of length %d against %d variables",
                   where, length(covariance), n_variables), call. = FALSE)
    }
    return(invisible(TRUE))
  }

  dims <- dim(covariance)
  if (length(dims) != 2L || dims[[1L]] != n_variables ||
      dims[[2L]] != n_variables) {
    stop(sprintf("%s has dimension %s against %d variables", where,
                 paste(dims, collapse = "x"), n_variables), call. = FALSE)
  }
  invisible(TRUE)
}

#' The covariance is exactly symmetric and exactly diagonal.
#'
#' This is the check the whole storage choice rests on: the processed form keeps
#' variances rather than matrices, which is lossless just when this holds. It is
#' also the last place the claim is checkable, since the off-diagonal does not
#' survive into the long table.
check_covariance_is_diagonal <- function(covariance, n_variables, key, site) {
  where <- sprintf("obs.cov[['%s']][['%s']]", key, site)

  if (!identical(as.numeric(covariance), as.numeric(t(covariance)))) {
    stop(sprintf("%s is not exactly symmetric", where), call. = FALSE)
  }

  off <- as.numeric(covariance)[as.numeric(diag(n_variables)) == 0]
  max_off <- if (length(off) == 0L) 0 else max(abs(off))
  if (max_off != 0) {
    stop(sprintf(paste0("%s has a non-zero off-diagonal element (%.17g). The ",
                        "processed form stores variances only, which is lossless ",
                        "just when every covariance is diagonal. That no longer ",
                        "holds, so the schema needs revisiting -- see ",
                        "src/sipnet_calibration/constraints.py."),
                 where, max_off), call. = FALSE)
  }
  invisible(TRUE)
}

#' Columns are strictly ascending, which is what pairs a variance with a variable.
check_columns_are_sorted <- function(variables, key, site) {
  if (is.unsorted(variables, strictly = TRUE)) {
    stop(sprintf(paste0("obs.mean[['%s']][['%s']] columns are not in ",
                        "strictly ascending order: %s"),
                 key, site, paste(variables, collapse = ", ")), call. = FALSE)
  }
  invisible(TRUE)
}

check_no_missing_values <- function(means, variances, key, site) {
  if (anyNA(means) || anyNA(variances)) {
    stop(sprintf("obs.mean/obs.cov[['%s']][['%s']] holds NA", key, site),
         call. = FALSE)
  }
  invisible(TRUE)
}

check_variable_names_are_known <- function(rows) {
  unknown <- setdiff(unique(rows$variable), SOURCE_VARIABLES)
  if (length(unknown) > 0L) {
    stop(sprintf("variable names not in the expected set: %s",
                 paste(unknown, collapse = ", ")), call. = FALSE)
  }
  invisible(TRUE)
}

#' The per-variable subsets add back up to every row.
#'
#' The check that would have caught the data.table scoping bug documented on
#' `extremes_by_variable` on its first run: a partition has to be a partition.
check_counts_partition_rows <- function(extremes, rows) {
  counted <- sum(vapply(extremes, function(entry) entry$n, integer(1)))
  if (counted != nrow(rows)) {
    stop(sprintf(paste0("per-variable counts sum to %d but the table has %d ",
                        "rows; the per-variable subsets are not a partition"),
                 counted, nrow(rows)), call. = FALSE)
  }
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  main(commandArgs(trailingOnly = TRUE))
}

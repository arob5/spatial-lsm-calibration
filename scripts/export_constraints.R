#!/usr/bin/env Rscript
#
# Flatten the annual constraint R data files to a long CSV plus a manifest.
#
#   data/raw/constraints/sda_8k_site_rdata/obs.{mean,cov}.Rdata
#     -> <out>           one row per observed (snapshot, site, variable) triple
#     -> <manifest>      what this script checked, as JSON
#
# Why this step exists in R at all: `obs.mean` is a list of lists of one-row
# data frames, which `pyreadr` does not support, and R's `ncdf4` is not
# installed on the development machine, so R cannot write the netCDF either.
# So R does the one thing only R can do -- read the objects and flatten them --
# and `scripts/ingest_constraints.py` makes every schema decision.
#
# Two things here are load-bearing.
#
# **`obs.cov` carries no dimension names.** The only thing that says which row
# of a covariance matrix belongs to which variable is the column order of the
# corresponding `obs.mean` entry. So the two files are always read together and
# a variance is only ever emitted alongside the mean it was paired with, in the
# same row.
#
# **Diagonality is asserted here, not downstream.** The processed form stores
# variances rather than matrices, which is only lossless because every source
# covariance is exactly diagonal. Once the CSV is written the off-diagonal is
# gone, so this is the last place the claim can be checked. The manifest records
# the largest off-diagonal element seen, so the Python side can confirm the check
# really ran rather than trusting that it did.
#
# Usage:
#   Rscript scripts/export_constraints.R --help

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

VARIABLES <- c("AbvGrndWood", "LAI", "SoilMoistFrac", "TotSoilCarb")

# The site pool the real files carry. `--expect-sites` overrides it, which is
# what lets the tests exercise the diagonality assertion against a three-site
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

# ── argument handling ─────────────────────────────────────────────────────────

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

# ── reading the source ────────────────────────────────────────────────────────

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

validate_nesting <- function(obs_mean, obs_cov, n_sites) {
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

# ── the covariance check ──────────────────────────────────────────────────────

#' Variances of one site-snapshot covariance, asserting it is diagonal.
#'
#' Returns a list with the diagonal and the largest absolute off-diagonal
#' element, which is accumulated into the manifest. `n_variables` comes from the
#' paired `obs.mean` entry and is what the covariance is checked against.
diagonal_of <- function(covariance, n_variables, key, site) {
  where <- sprintf("obs.cov[['%s']][['%s']]", key, site)

  if (n_variables == 0L) {
    if (length(covariance) != 0L) {
      stop(sprintf("%s is non-empty where obs.mean has no columns", where),
           call. = FALSE)
    }
    return(list(diagonal = numeric(0), max_abs_offdiagonal = 0))
  }

  if (is.null(dim(covariance))) {
    if (length(covariance) != 1L || n_variables != 1L) {
      stop(sprintf("%s is a bare numeric of length %d against %d variables",
                   where, length(covariance), n_variables), call. = FALSE)
    }
    return(list(diagonal = as.numeric(covariance), max_abs_offdiagonal = 0))
  }

  dims <- dim(covariance)
  if (length(dims) != 2L || dims[[1L]] != n_variables || dims[[2L]] != n_variables) {
    stop(sprintf("%s has dimension %s against %d variables", where,
                 paste(dims, collapse = "x"), n_variables), call. = FALSE)
  }
  if (!identical(as.numeric(covariance), as.numeric(t(covariance)))) {
    stop(sprintf("%s is not exactly symmetric", where), call. = FALSE)
  }

  matrix_values <- as.numeric(covariance)
  off <- matrix_values[as.numeric(diag(n_variables)) == 0]
  max_off <- if (length(off) == 0L) 0 else max(abs(off))
  if (max_off != 0) {
    stop(sprintf(paste0("%s has a non-zero off-diagonal element (%.17g). The ",
                        "processed form stores variances only, which is lossless ",
                        "just when every covariance is diagonal. That no longer ",
                        "holds, so the schema needs revisiting -- see ",
                        "src/sipnet_calibration/constraints.py."),
                 where, max_off), call. = FALSE)
  }
  list(diagonal = diag(covariance), max_abs_offdiagonal = 0)
}

# ── flattening ────────────────────────────────────────────────────────────────

#' Long-format rows for one snapshot, and what was checked while building them.
flatten_snapshot <- function(means, covariances, key) {
  pieces <- vector("list", length(means))
  empty_sites <- integer(0)
  max_off <- 0

  for (index in seq_along(means)) {
    site <- names(means)[[index]]
    frame <- means[[index]]

    if (!is.data.frame(frame) || nrow(frame) != 1L) {
      stop(sprintf("obs.mean[['%s']][['%s']] is not a one-row data frame",
                   key, site), call. = FALSE)
    }

    variables <- names(frame)
    checked <- diagonal_of(covariances[[index]], length(variables), key, site)
    max_off <- max(max_off, checked$max_abs_offdiagonal)

    if (length(variables) == 0L) {
      empty_sites <- c(empty_sites, as.integer(site))
      next
    }
    if (is.unsorted(variables, strictly = TRUE)) {
      stop(sprintf(paste0("obs.mean[['%s']][['%s']] columns are not in ",
                          "strictly ascending order: %s"),
                   key, site, paste(variables, collapse = ", ")), call. = FALSE)
    }

    values <- as.numeric(frame[1L, , drop = TRUE])
    if (anyNA(values) || anyNA(checked$diagonal)) {
      stop(sprintf("obs.mean/obs.cov[['%s']][['%s']] holds NA", key, site),
           call. = FALSE)
    }

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
  unknown <- setdiff(unique(rows$variable), VARIABLES)
  if (length(unknown) > 0L) {
    stop(sprintf("variable names not in the expected set: %s",
                 paste(unknown, collapse = ", ")), call. = FALSE)
  }
  list(rows = rows, empty = empty, max_abs_offdiagonal = max_off)
}

# ── the manifest ──────────────────────────────────────────────────────────────

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
  for (variable_name in VARIABLES) {
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

  # The check that would have caught the scoping bug on its first run: a
  # per-variable partition of the rows has to add back up to all of them.
  counted <- sum(vapply(result, function(entry) entry$n, integer(1)))
  if (counted != nrow(rows)) {
    stop(sprintf(paste0("per-variable counts sum to %d but the table has %d ",
                        "rows; the per-variable subsets are not a partition"),
                 counted, nrow(rows)), call. = FALSE)
  }
  result
}

counts_by_snapshot_variable <- function(rows) {
  result <- list()
  for (key in unique(rows$snapshot_date)) {
    subset <- rows[rows$snapshot_date == key, ]
    counts <- as.list(table(factor(subset$variable, levels = VARIABLES)))
    result[[key]] <- lapply(counts, as.integer)
  }
  result
}

build_manifest <- function(built, args, snapshot_keys) {
  rows <- built$rows
  list(
    generated_by = "scripts/export_constraints.R",
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    mean_file = args$mean,
    cov_file = args$cov,
    variables = VARIABLES,
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

# ── entry point ───────────────────────────────────────────────────────────────

main <- function(argv) {
  args <- parse_args(argv)

  message("Reading ", args$mean)
  obs_mean <- load_single_object(args$mean, "obs.mean")
  message("Reading ", args$cov)
  obs_cov <- load_single_object(args$cov, "obs.cov")

  validate_nesting(obs_mean, obs_cov, args$n_sites)
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

if (sys.nframe() == 0L) {
  main(commandArgs(trailingOnly = TRUE))
}

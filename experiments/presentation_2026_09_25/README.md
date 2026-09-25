# Progress presentation, 2026-09-25

A Quarto reveal.js deck that builds the calibration pipeline up one piece at a
time: sites and plant functional types, drivers, initial conditions,
constraints, the parameter vector, and the observation operator. Each piece
states what it is, what is assumed, and what is still open.

| File | Role |
|---|---|
| `config.py` | the deck's choices: featured sites, the sites and driver members it runs, site labels, map extent, paths |
| `draw_pft_sites.py` | draws `config.PFT_SITES`, three sites per 16-class PFT; rerun to reproduce it |
| `relabel_drivers.py` | temporary: copies the drivers of `config.DRIVER_SITES` with their drifting hour column corrected, on the SCC |
| `plots.py` | one function per figure, over the library's loaders and plotters |
| `slides.qmd` | the deck: prose, assumptions, open questions, and the code shown |
| `slides.css` | slide styling |
| `precompute_drivers.py` | annual values and monthly climatology of every site's drivers, on the SCC |
| `precompute_drivers.qsub` | runs it as an array job, then combines the parts |
| `make_driver_animations.py` | GIFs of the monthly driver climatology for `config.DRIVER_ANIMATION_VARIABLES`, from those summaries |

## Where it runs

Everything except the drivers reads tracked inputs, so the deck is developed
and rendered locally. The driver figures need all 8000 sites x 10 members,
which only the SCC has: `precompute_drivers.py` writes small summaries to
`outputs/` there, and they are copied back with `rsync`. `outputs/` is
untracked.

Until the ERA5 drivers are regenerated, pySIPNET refuses them because their
hour column drifts (`data/README.md` Note 15). `relabel_drivers.py` writes
corrected copies to `config.RELABELED_DRIVERS_DIR`, which is the drivers root
to pass wherever one is taken. It corrects that column only.

The processed products have to exist first. From the worktree root:

```bash
python scripts/ingest_sites.py
python scripts/ingest_site_labels.py
python scripts/ingest_initial_conditions.py
python scripts/ingest_constraints.py
```

## Rendering and editing

Point Quarto at the worktree's own interpreter, so the deck imports this
worktree's `sipnet_calibration`:

```bash
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
quarto preview experiments/presentation_2026_09_25/slides.qmd
```

`quarto preview` re-renders on every save. For interactive work, open
`slides.qmd` in VS Code or Positron with the Quarto extension, select the
worktree's `.venv` as the kernel, and run cells one at a time. A full render
re-runs every cell, and takes seconds, because the heavy work is done ahead of
time.

Every rendering option lives in `slides.qmd`'s front matter. There is no
`_quarto.yml`: Quarto 1.6 did not apply its `format` options to this document,
and its `freeze` applies only to whole-project renders.

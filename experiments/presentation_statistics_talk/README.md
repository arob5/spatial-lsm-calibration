# Statistics talk

A Quarto reveal.js deck on the Harvard Forest calibration for an audience of
statisticians: the carbon cycle and land surface modeling in brief, then the
models, the algorithms and the results. The group-meeting version, with the
data setup and the priors in full, is the experiment's report
(`experiments/single_site_mcmc_vs_eki/report/report.qmd`).

| File | Role |
|---|---|
| `slides.qmd` | the deck: prose, math and the figures |
| `plots.py` | one function per figure |
| `figures/` | the hand-drawn diagrams, as SVG: the pools and fluxes (with its observation layer, `id="observations"`) and the eddy-covariance tower |
| `slides.css` | slide styling, including the `.corner-note` aside |

The diagrams are plain SVG, editable in any SVG editor. `plots.py` inlines
them, so their text takes the slide's font; each one's classes and marker ids
carry its own prefix (`bm-`, `ec-`), since inline SVG styles apply to the
whole page.

## What it reads

The site table and the tower table, through the library. The "Model: one
run" slide runs SIPNET once, at the prior's center, through the experiment
`experiments/single_site_mcmc_vs_eki` (about a second, after the prior
binds): it needs the experiment's prepared driver file
(`run/prepare_drivers.py`) and a SIPNET binary (`pysipnet install-sipnet`).
The results slides, still to come, will read what the experiment writes under
its untracked `output/`, which the experiment's `README.md` says how to make.

## Rendering and editing

Point Quarto at the worktree's own interpreter:

```bash
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
quarto preview experiments/presentation_statistics_talk/slides.qmd
```

`quarto preview` re-renders on every save. Every rendering option lives in
`slides.qmd`'s front matter.

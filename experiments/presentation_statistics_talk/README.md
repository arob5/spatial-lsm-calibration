# Statistics talk

A Quarto reveal.js deck on the Harvard Forest calibration for an audience of
statisticians: the carbon cycle and land surface modeling in brief, then the
models, the algorithms and the results. The group-meeting version, with the
data setup and the priors in full, is the experiment's report
(`experiments/single_site_mcmc_vs_eki/report/report.qmd`).

| File | Role |
|---|---|
| `slides.qmd` | the deck: prose, math and the figures |
| `slides.css` | slide styling |

## What it reads

The deck runs no model. It reads what `experiments/single_site_mcmc_vs_eki`
wrote under its untracked `output/`, which the experiment's `README.md` says
how to make.

## Rendering and editing

Point Quarto at the worktree's own interpreter:

```bash
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
quarto preview experiments/presentation_statistics_talk/slides.qmd
```

`quarto preview` re-renders on every save. Every rendering option lives in
`slides.qmd`'s front matter.

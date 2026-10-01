# Presentation, 2026-09-30

A Quarto reveal.js deck on the Harvard Forest calibration: the problem at one
site, the prior and its prior predictive, EKI on synthetic and observed data,
the first calibration's posterior predictive check, what the NEE residuals
show, and the revised NEE error model. It follows the format of the
2026-09-25 deck (tag `presentation-2026-09-25`).

| File | Role |
|---|---|
| `config.py` | the deck's choices: which runs of the experiment it shows |
| `plots.py` | one function per figure, over the experiment's readers and figure functions |
| `slides.qmd` | the deck: prose, math and the figures |
| `slides.css` | slide styling |

## What it reads

The deck runs no model. It reads what `experiments/single_site_mcmc_vs_eki`
wrote under its untracked `output/`: the prior predictive, and the first
calibration's EKI runs (setup `single_term_discrepancy`), their posterior
predictive and their diagnostics. Those have to exist first; the experiment's
`README.md` says how to make them.

## Rendering and editing

Point Quarto at the worktree's own interpreter:

```bash
export QUARTO_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python
quarto preview experiments/presentation_2026_09_30/slides.qmd
```

`quarto preview` re-renders on every save. Every rendering option lives in
`slides.qmd`'s front matter, as in the previous deck.

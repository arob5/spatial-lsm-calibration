---
name: review-pr
description: Adversarially review a pull request by dispatching several independent review agents in parallel, one per concern (correctness and numerics, bugs and edge cases, documentation, tests, the PR description), then consolidate their findings, apply the small fixes, and stop for the author's input on anything that is a real judgment call. Use when asked to review a PR, review the current branch, or check a change before merging.
---

# Adversarially review a pull request

## What to review

`$ARGUMENTS` may name a PR number, a branch, or nothing. With nothing, review
the PR for the current branch (`gh pr view`), or if there is none, the diff
against `main`.

Establish the ground truth before dispatching anything:

```bash
gh pr view <n>                    # description
gh pr view <n> --comments         # the thread, which is where staleness hides
git diff main...HEAD --stat       # what actually changed
git log --oneline main..HEAD
uv run pytest -q                  # the real test count, not the claimed one
```

## Dispatch the agents

Send them **in one message so they run concurrently**, in the background. Give
each a **non-overlapping scope** and say so in its brief, so they do not
duplicate each other or each other's findings.

The standard scopes:

1. **Correctness and numerics** — the mathematics, floating point, precision and
   dtype choices, unit conversions, overflow, round-tripping, and any numerical
   claim the code or its comments make.
2. **Bugs and edge cases** — control flow, error paths, argument handling, empty
   and malformed inputs, resource leaks, partial failures, and language-specific
   traps (R's `$` partial matching, pandas dtype and NA surprises, path
   handling).
3. **Documentation** — docstrings lead with an overview and describe the public
   API rather than the design reasoning behind it; nothing stale or redundant;
   clear, precise, not wordy, not overly technical, and no "LLM jargon". Judge
   against `CLAUDE.md`'s conventions, including whether the PR's own new files
   follow the conventions they introduce.
4. **Tests** — quality and coverage, found by **mutation testing** rather than
   by reading: break the source deliberately, see whether anything fails, and
   treat a silent pass as the finding.
5. **The PR description** — is it accurate and current against the branch as it
   now stands, given that it was probably written before the last few commits.

Adjust the set to the change: drop numerics for a documentation-only PR, split a
large scope in two, add a scope for anything unusual in the diff. Prefer more
narrow agents over fewer broad ones.

Every brief must include:

- **"Your job is to find problems, not to praise."** Say it plainly; otherwise
  agents report that everything looks good.
- **Verify by running, not by reading.** Point them at
  `.venv/bin/python`, `uv run python`, and `Rscript`, and tell them where the
  real data is. Reading code finds typos; running it finds bugs.
- **What is out of their scope**, naming the other agents' territory.
- **Do not modify tracked files.** For the tests agent, which must mutate
  source to do its job: restore with `git checkout --` and confirm
  `git status` is clean before reporting. Never commit.
- **A findings format**: file:line, what is wrong, the reproduction and its
  output, and a severity. Ask them to state which checks they ran and found
  *sound*, so the coverage of the review itself is visible.
- **Lead with the most serious findings**, and no padding.

## Consolidate

When the agents report back:

- **Deduplicate.** Several agents often find one underlying problem from
  different directions. Merge those, and say so.
- **Verify before acting.** Agents are confidently wrong sometimes. Reproduce
  any finding you are going to act on. Discard the ones that do not hold, and
  say which and why rather than silently dropping them.
- **Sort into two piles:**
  - **Small and unambiguous** — a wrong number, a stale reference, a dangling
    cross-reference, a missing test for an existing check, a typo, wording that
    is plainly worse. Fix these now.
  - **A real judgment call** — anything changing the schema, an interface, a
    design decision, the scope of the PR, or a documented convention; anything
    where two defensible answers exist; anything you would have to guess the
    author's preference to decide. **Stop and ask.** Do not decide these.
- Re-run the suite after the fixes, and update the PR description or add a
  comment recording what changed.

## Report

Give the author:

- The findings that mattered, grouped by severity, with what you did about each.
- What you fixed, in one line each.
- **The judgment calls, stated as questions with a recommendation and the
  trade-off** — this is the part they act on, so put it last where it is easy to
  find, and do not bury it in prose.
- What the agents checked and found sound, briefly, so the review's coverage is
  visible.
- Anything an agent claimed that you could not reproduce.

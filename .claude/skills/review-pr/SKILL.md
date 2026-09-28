---
name: review-pr
description: Adversarially review a pull request, sized to the change - a solo pass for small or non-central PRs, up to four parallel agents (code, targeted mutation testing, docs and description) for code that computes results - then verify the findings, apply the small fixes, and stop for the author on judgment calls. Use when asked to review a PR, review the current branch, or check a change before merging.
---

# Adversarially review a pull request

One review round, sized to the change. The cost of a review is mostly agents
re-reading the repository and re-running the suite; this skill spends that
only where a missed bug would change a result.

## 1. Gather the ground truth once

`$ARGUMENTS` may name a PR number, a branch, or nothing (then the current
branch's PR, or failing that its diff against `origin/main`). Review in a
checkout of the PR's head that has its own `.venv` (see CLAUDE.md); create a
worktree if none exists.

Diff against the PR's **base**, not `main`: PRs here are often stacked.

```bash
gh pr view <n> --json number,title,body,baseRefName,headRefName,comments
git fetch -q origin
git diff --stat origin/<base>...HEAD
git log --oneline origin/<base>..HEAD
```

Write the context to one directory under the session scratchpad,
`$SCRATCH/review-<n>/`, so agents read it
instead of each re-fetching it: `pr.json` (the view above), `diff.patch`
(`git diff origin/<base>...HEAD`), `files.txt` (changed paths).

Start the full suite **in the background** now (`.venv/bin/python -m pytest -q`,
`run_in_background`) and read its result during consolidation. It is the only
full-suite run in the review, apart from the one after fixes.

## 2. Size the review

Pick the smallest tier that fits; say which one you picked and why.

| Tier | When | Who reviews |
|------|------|-------------|
| **Solo** | Docs, config, renames, convention cleanups, or a small change to code that does not compute results | You alone: read the diff, run the tests of the touched modules, check the description. No agents. |
| **Standard** | New or changed logic | Agents A, B and C below, in parallel |
| **Deep** | Code that computes results (numerics, units, time alignment, data placement, likelihood, inference) *and* a large or subtle diff | A split into A1 correctness/numerics and A2 bugs/edge cases, plus B and C: four agents, never more |

When unsure between two tiers, take the smaller and escalate only a specific
module whose diff turns out to warrant it. For a non-central PR, ask the author
before going above Solo.

The agents:

- **A: code** (default model) — the mathematics, floating point and dtypes,
  unit conversions, round-tripping, numerical claims in comments; control flow,
  error paths, argument handling, empty and malformed inputs, partial failures,
  pandas dtype/NA and R `$` partial-matching traps.
- **B: tests** (`model: "sonnet"`) — test adequacy by **targeted mutation**:
  break the changed logic, run the relevant test file, and report every
  mutation that survives. At most ~10 mutations, aimed at the diff's new
  checks, branches, boundary comparisons and arithmetic; not at unchanged code.
- **C: docs and description** (`model: "sonnet"`) — docstrings and comments
  against CLAUDE.md and the standing documentation rules (overview first,
  length justified, math written out, design reasoning in Notes, nothing stale,
  no vault references, American spelling); and whether the PR description is
  accurate for the branch as it now stands, including its test count against
  the suite result you pass it.

## 3. Brief the agents

Send all briefs **in one message**, in the background. Each brief is short and
carries:

- **"Your job is to find problems, not to praise."**
- **The scope, and what is out of it** by naming the other agents' scopes.
- **Where the context is**: `$SCRATCH/review-<n>/`. Read `diff.patch` and the
  changed files, and follow a call into unchanged code only when a finding
  depends on it. Do not survey the repository. CLAUDE.md is already loaded.
- **Verify by running, cheaply**: `.venv/bin/python -c` / a scratch script, or
  `Rscript`, against the real data where it exists (CLAUDE.md, Data section).
  Run single test files with `.venv/bin/python -m pytest <file> -x -q`, never the
  full suite.
- **Do not modify tracked files, and never commit.** Agent B instead works in
  its own detached worktree, so its mutations cannot corrupt what the other
  agents are running:

  ```bash
  git worktree add --detach $SCRATCH/review-<n>/mutation <head-sha>
  cd $SCRATCH/review-<n>/mutation && uv sync -q
  ```

  symlinking any untracked `data/raw/<dir>` its tests need from the root, and
  removing the worktree (`git worktree remove --force`) before reporting.
- **The report**: at most ~10 findings, most serious first, each as
  `file:line — what is wrong — a copy-pasteable reproduction and its output —
  severity (bug / gap / nit)`. Then one line listing what was checked and found
  sound. No preamble, no summary of the PR, under ~500 words. Report nits only
  where they break a CLAUDE.md convention.

## 4. Consolidate

- **Deduplicate** findings reached from different directions, and say so.
- **Verify what you will act on, by re-running the agent's reproduction**, not
  by re-deriving it. A finding without a reproduction is "plausible" at best;
  say so rather than acting on it. Discard what does not hold, and say which
  and why.
- **Sort into two piles:**
  - **Small and unambiguous** — a wrong number, a stale reference, a missing
    test for an existing check, a typo, plainly worse wording. Fix these now.
  - **A judgment call** — anything changing a schema, an interface, a design
    decision, the PR's scope or a documented convention; anything with two
    defensible answers. **Stop and ask.** Do not decide these.
- After the fixes, run the tests of the touched modules, then the full suite
  once, and update the PR description or add a comment recording what changed.

**Do not start a second review round.** If the fixes changed behavior in code
that computes results, say so and offer one, scoped to those fixes.

## 5. Report

- What tier you ran, and why.
- The findings that mattered, by severity, with what you did about each; what
  you fixed, one line each.
- What was checked and found sound, briefly.
- Anything an agent claimed that did not reproduce.
- **Last, the judgment calls as questions**, each with a recommendation and the
  trade-off. This is what the author acts on; do not bury it.

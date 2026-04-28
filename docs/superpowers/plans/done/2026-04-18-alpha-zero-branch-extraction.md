# Alpha-Zero Branch Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy `sa_filter` branch selector with a cheaper and more defensible `k`-th positive-root workflow driven by the 2D GPR surrogate, active learning, and bounded solve budgets.

**Architecture:** Fit one 2D GPR for `f(x_M/L, log10(EI))`, extract all positive surrogate roots per `EI` slice, define branch `k` as the `k`-th smallest positive root, and use continuity only as a guardrail. Retrain the surrogate in batches with all cached solved points, reuse cached rows only when they meet the final `alpha_accept_tol`, and cap new solves per `log10(EI)` bin.

**Tech Stack:** Julia, Surferbot, existing GP code in `Julia/experiments/analyze_single_alpha_zero_curve.jl`, CSV cache reuse, Plots for diagnostics.

---

## Chunk 1: Branch Selector Rewrite

### Task 1: Replace `sa_filter` as the primary selector

**Files:**
- Modify: `Julia/experiments/analyze_single_alpha_zero_curve.jl`
- Test: `Julia/test/test_analysis_helpers.jl` or new focused test file if cleaner

- [ ] Add a helper that, for each surrogate `log10(EI)` slice, returns all positive zero crossings sorted by `x_M/L`.
- [ ] Define branch `k` as the `k`-th smallest positive root with `x_M/L > x_eps`.
- [ ] Mark slices with `max_x |f(x, log10(EI))| < tau_flat` as degenerate and skip them.
- [ ] Keep continuity checking as a rejection/tie-break rule only.
- [ ] Preserve `sa_ratio` in the CSV as a diagnostic, not as the branch selector.

### Task 2: Update CLI and docs

**Files:**
- Modify: `Julia/experiments/analyze_single_alpha_zero_curve.jl`

- [ ] Replace the user-facing `sa_filter` argument with `branch_index` (default `1`).
- [ ] Keep backward compatibility only if trivial; otherwise remove the old path cleanly.
- [ ] Update the entrypoint docstring to explain `branch_index`, degenerate slices, and the leftmost-root logic.

## Chunk 2: Active Learning Loop

### Task 3: Add bounded iterative refinement

**Files:**
- Modify: `Julia/experiments/analyze_single_alpha_zero_curve.jl`

- [ ] Introduce `alpha_accept_tol` with default `0.005`.
- [ ] Use **all** cached solved rows for retraining, with no filtering.
- [ ] Reuse cached rows as final outputs only if they are near the requested sample and satisfy `|alpha| <= alpha_accept_tol`.
- [ ] Add `log10(EI)` binning and cap new solves per bin at `5`.
- [ ] Retrain the 2D GPR after a configurable batch of new solves, not after every solve.
- [ ] Stop refinement when all sampled points are accepted or bin budgets are exhausted.

### Task 4: Preserve cheap solver usage

**Files:**
- Modify: `Julia/experiments/analyze_single_alpha_zero_curve.jl`

- [ ] Keep all branch finding on the surrogate only.
- [ ] Only run expensive Julia solves on final sampled points that are not reusable from cache.
- [ ] Keep threaded execution for these expensive reruns.

## Chunk 3: Diagnostics

### Task 5: Improve output diagnostics

**Files:**
- Modify: `Julia/experiments/analyze_single_alpha_zero_curve.jl`

- [ ] Swap overlay axes so horizontal is `log10(EI)` and vertical is `x_M/L`.
- [ ] Show the surrogate field, zero contour, sampled branch, and bad rerun points.
- [ ] Log per-iteration counts: reused, newly solved, accepted, rejected, and per-bin budget use.

### Task 6: Add focused verification

**Files:**
- Modify: `Julia/test/...` as appropriate

- [ ] Add a focused test for branch indexing on synthetic multi-root slices.
- [ ] Add a focused test that cache rows are always used for retraining.
- [ ] Add a focused test that final reuse still respects `alpha_accept_tol`.
- [ ] Run the focused tests plus a parse/load smoke check for the experiment entrypoint.


# Julia Surferbot Run Video Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Julia-native, headless MP4 generator for Surferbot runs, plus same-stem provenance JSON metadata.

**Architecture:** Keep the solver core unchanged and add a small video-focused helper module that normalizes run inputs, renders a stationary-domain `eta(x,t)` animation with `Plots.jl`/GR, and writes a matching provenance JSON file. A thin script entry point should call that helper so the runtime path is reusable from tests, scripts, and later notebook workflows.

**Tech Stack:** Julia, `Plots.jl`, `GR`, `FFMPEG` through `Plots.mp4`, existing `Surferbot` run structures, and a minimal JSON writer or JSON dependency if needed.

---

## File Structure

- Create: `Julia/src/video.jl`
- Modify: `Julia/src/Surferbot.jl`
- Create: `Julia/scripts/plot_surferbot_run.jl`
- Create: `Julia/test/test_video.jl`
- Modify: `Julia/test/runtests.jl`
- Modify: `Julia/Project.toml` only if a JSON package is required

## Task 1: Define the video helper module

**Files:**
- Create: `Julia/src/video.jl`
- Modify: `Julia/src/Surferbot.jl`
- Test: `Julia/test/test_video.jl`

- [ ] **Step 1: Write the failing test**

Create a test that builds a tiny synthetic run record and checks that:
- the video helper normalizes the record
- the provenance writer returns a JSON string containing `script_name`, `output_basename`, and `U`
- the output contract uses a shared basename for `.mp4` and `.json`

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_video.jl`
Expected: fail because `video.jl` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement helper functions in `Julia/src/video.jl` for:
- normalizing a run input into a common record
- building provenance data
- writing the provenance JSON file
- rendering a stationary `eta(x,t)` animation with `Plots.jl` and `GR`

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_video.jl`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/video.jl Julia/src/Surferbot.jl Julia/test/test_video.jl
git commit -m "Add Julia run video helper module"
```

## Task 2: Add the script entry point

**Files:**
- Create: `Julia/scripts/plot_surferbot_run.jl`
- Modify: `Julia/test/runtests.jl`

- [ ] **Step 1: Write the failing test**

Add a smoke-style test or helper invocation that imports the script entry point and confirms it can render a tiny run to a temporary directory with:
- `waves.mp4`
- `waves.json`

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_video.jl`
Expected: fail until the script entry point exists.

- [ ] **Step 3: Write minimal implementation**

Create the thin script entry point so users can run:

```bash
julia --project=. scripts/plot_surferbot_run.jl <run-dir-or-input>
```

The script should call the helper module and default to headless MP4 output only.

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_video.jl`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Julia/scripts/plot_surferbot_run.jl Julia/test/runtests.jl
git commit -m "Add Julia run video script"
```

## Task 3: Verify headless video generation

**Files:**
- Modify: `Julia/test/test_video.jl`

- [ ] **Step 1: Write the failing test**

Add a smoke test that renders a very small, valid run record to a temp directory and asserts the two output files exist.

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_video.jl`
Expected: fail until actual MP4 generation is implemented.

- [ ] **Step 3: Write minimal implementation**

Keep the renderer headless and use a short video length and low frame count in the test to control runtime.

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_video.jl`
Expected: pass, with `waves.mp4` and `waves.json` written to the temp directory.

- [ ] **Step 5: Commit**

```bash
git add Julia/test/test_video.jl
git commit -m "Verify Julia run video output"
```


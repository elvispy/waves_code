# Julia Surferbot Run Video Design

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Julia-native, headless script that renders a saved Surferbot run to an MP4 video and writes same-stem provenance metadata alongside it.

**Architecture:** The video path should be a thin presentation layer over the existing Julia solver output shape. The script will normalize either a saved run directory or an in-memory run object into a common record, then render a stationary-domain `eta(x,t)` animation with the contact region and motor marker overlaid. Output artifacts share one basename: `waves.mp4` and `waves.json`, where the JSON records provenance such as source, parameters, git commit, and key outputs.

**Tech Stack:** Julia, `Plots.jl` with the `GR` backend, `FFMPEG` via `Plots.mp4`, existing `Surferbot` solver types, and a small local JSON writer if a new dependency is not needed.

---

## Scope

This feature is deliberately minimal:
- one headless MP4 generator
- one same-stem provenance JSON file
- stationary-domain animation only
- no static figure bundle
- no moving-camera mode

The output must be usable both for archived run directories and for a freshly solved in-memory result.

## Output Contract

For an output base name `waves`, the script must write:
- `waves.mp4`
- `waves.json`

The JSON file must include at least:
- input source type and path, if applicable
- `FlexibleParams` values or equivalent run parameters
- git commit hash or an explicit “unknown” placeholder
- script name
- output basename
- key scalar outputs if available, such as `U`, `thrust`, and `power`

## Provenance Rule

The MP4 and JSON must always share the same basename and live in the same output directory. No extra provenance folder should be required.

## Success Criteria

- A Julia command can render a headless MP4 from a run input without opening a GUI window.
- The same script can write a matching JSON provenance file.
- The implementation uses the existing Julia plotting stack already present in the project.
- The result is simple enough that it can become the default run-video path later.


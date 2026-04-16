# REQUIREMENTS

## As Is

The repository has a MATLAB analysis workflow for modal decomposition along the uncoupled beam-end white curve in `MATLAB/test/analyze_modal_decomposition_along_beam_curve_uncoupled.m`. That workflow loads `sweepMotorPositionEI2.mat`, extracts the lowest beam-end `alpha = 0` branch, reruns the flexible solver at sampled points, projects the raft displacement onto a free-free beam basis using `MATLAB/src/decompose_raft_freefree_modes.m`, and writes a four-panel PDF figure.

The Julia codebase currently has the matched flexible solver in `Julia/src/Surferbot.jl` and postprocessing in `Julia/src/postprocess.jl`, but it does not have any of the following:
- a Julia equivalent of `decompose_raft_freefree_modes`
- a Julia script for the uncoupled beam-end modal decomposition analysis
- a Julia-native sweep artifact for the uncoupled `x_M`-`EI` study
- plotting support for the analysis figure in the Julia project dependencies

The Julia script namespace currently contains only debug dump helpers and the optimization driver. There is no established Julia analysis pipeline for the second-family modal figures.

## To Be

The Julia project should contain an explicit Julia-native port of the uncoupled beam-end modal decomposition workflow. The Julia workflow should first generate its own uncoupled `x_M`-`EI` sweep artifact directly from the Julia solver, then extract the lowest uncoupled beam-end `alpha = 0` branch from that native artifact, rerun the Julia flexible solver at sampled points, decompose the raft response into free-free modes using a Julia port of the MATLAB modal helper, and save an analysis figure and machine-readable outputs from Julia only.

The modal decomposition logic should live in reusable Julia code rather than being embedded entirely inside the script, so later coupled/other analysis scripts can reuse it. The script should be usable from the Julia project root with a clear entry point and progress logging similar to the MATLAB script.

## Requirements

1. The Julia project shall provide a reusable modal decomposition implementation equivalent to `MATLAB/src/decompose_raft_freefree_modes.m` for raft responses returned by the Julia solver.
2. The Julia modal decomposition implementation shall compute the same core outputs as MATLAB: retained mode indices, mode types, `betaL`, `beta`, modal coefficients `q`, projected `Q` and `F`, balance residual, energy fractions, reconstructed raft displacement, reconstruction error, raft coordinates, basis matrix, and Gram condition number.
3. The Julia project shall provide a script named `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` that generates a Julia-native uncoupled beam-end sweep artifact directly from the Julia solver.
4. The Julia project shall provide a script named `Julia/scripts/analyze_modal_decomposition_along_beam_curve_uncoupled.jl` that performs the uncoupled beam-end workflow without calling MATLAB.
5. The Julia native sweep artifact shall store the beam-end fields and parameter grids required by the branch extraction workflow, using a Julia-native file format.
6. The Julia analysis script shall reproduce the MATLAB branch extraction logic for the lowest uncoupled beam-end `alpha = 0` / `S ~ 0` curve.
7. The Julia analysis script shall rerun the Julia flexible solver at sampled points along that curve and print progress for each sampled case.
8. The Julia analysis script shall save a figure and a machine-readable summary of the sampled modal quantities in the Julia-side data/output path.
9. The implementation shall keep the solver logic in Julia and shall not shell out to MATLAB, Python, or other external runtimes for the analysis itself.
10. The Julia project dependencies shall include whatever packages are needed for native sweep storage and figure output for this workflow.

## Acceptance Criteria

### Requirement 1
- A reusable Julia module or source file exists under `Julia/src/` for free-free modal decomposition.
- The new code accepts solver outputs and raft data without requiring MATLAB structs.

### Requirement 2
- Unit tests verify the Julia modal helper returns the expected field set.
- Unit tests verify rigid modes and at least one elastic mode are produced when requested.
- Unit tests verify energy fractions sum to one within tolerance when the modal coefficients are nonzero.

### Requirement 3
- `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` exists and can be run with `julia --project=Julia`.

### Requirement 4
- `Julia/scripts/analyze_modal_decomposition_along_beam_curve_uncoupled.jl` exists and can be run with `julia --project=Julia`.

### Requirement 5
- The sweep artifact stores parameter grids plus `eta_left_beam` and `eta_right_beam` on the full grid.
- The analysis script fails with a clear error if those required fields are missing from the native artifact.

### Requirement 6
- The script computes `alpha_beam`, `S_grid`, `A_grid`, `SA_ratio`, and branch crossings using the same formulas and selection logic as the MATLAB script.

### Requirement 7
- The script logs the total number of sampled points.
- The script logs each sampled case index and its `EI` and `x_M/L` values while solving.

### Requirement 8
- The script writes a PDF figure analogous to the MATLAB output.
- The script writes a delimited machine-readable table or similar artifact containing the sampled modal quantities.

### Requirement 9
- The script imports and uses Julia solver code only.
- No command execution or foreign-language bridge is required to finish the analysis.

### Requirement 10
- `Julia/Project.toml` declares the native storage and plotting dependencies needed by the scripts.
- The script runs using those Julia dependencies under the project environment.

## Testing Plan

1. Add focused unit tests for the Julia modal decomposition helper:
   - trapz weights / weighted inner product behavior on a simple grid
   - free-free root and mode bank generation sanity
   - modal decomposition output structure on a synthetic raft response
   - energy fraction normalization and reconstruction error finiteness
2. Add a focused integration test for branch extraction logic on a tiny synthetic dataset that exercises the `alpha = 0` crossing selection rules.
3. Add a smoke test for the native sweep writer helpers that verifies a small artifact can be written and reloaded.
4. Run existing Julia solver tests to ensure the new analysis code does not regress the solver package surface.
5. Perform smoke runs of the new sweep and analysis scripts, verifying they emit progress and write output files.

## Implementation Plan

1. Add the per-task Julia dependencies for native sweep storage and plotting.
   - Test by importing the new packages under `julia --project=Julia`.
2. Implement a reusable Julia modal decomposition module under `Julia/src/` ported from MATLAB `decompose_raft_freefree_modes`.
   - Test with new unit tests on synthetic inputs before using real solver outputs.
3. Expose the modal helper from `Surferbot.jl` if needed by scripts without forcing unrelated package API churn.
   - Test package load and helper import.
4. Implement native sweep IO and branch-extraction helpers for the uncoupled beam-end workflow.
   - Test artifact write/load and crossing selection logic on a small synthetic grid.
5. Implement `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` to generate the native uncoupled sweep artifact.
   - Test on a reduced smoke run that checks file creation and expected field layout.
6. Implement `Julia/scripts/analyze_modal_decomposition_along_beam_curve_uncoupled.jl` using the new helper, native sweep artifact, and plotting backend.
   - Test on a smoke run that checks file creation and progress output.
7. Run the focused Julia tests and smoke invocations of both scripts, then fix any integration issues.
   - Test by rerunning the focused suite and confirming the output artifacts exist.

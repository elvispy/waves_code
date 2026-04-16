# REQUIREMENTS

## As Is

The Julia solver core is implemented in `Julia/src/Surferbot.jl`, `fd.jl`, `postprocess.jl`, and `utils.jl`. There is no reusable universal sweep layer in the Julia package. The repository instead has task-specific sweep scripts, including `Julia/scripts/sweep_motor_position_EI_uncoupled.jl`, and a large number of MATLAB `sweep_*` scripts. This encourages one-off sweep implementations and makes Julia less suitable as the default workflow entry point.

The current Julia-native uncoupled sweep script already writes a `JLD2` artifact, but it is specialized to one parameter family and does not expose a general parameter-grid engine.

## To Be

The Julia project should provide a single reusable sweep engine that can run parameter-grid sweeps over any subset of `FlexibleParams` inputs, collect a standard set of solver summary outputs, and save a Julia-native artifact. Julia should then expose one generic runnable sweep script based on that engine, while the existing uncoupled sweep script should become a thin preset wrapper around the generic sweep engine rather than a separate implementation.

The sweep layer should be generic enough for future parameter studies, but small and opinionated enough to remain part of the minimal default Julia stack.

## Requirements

1. The Julia project shall provide a reusable sweep module under `Julia/src/` for parameter-grid sweeps of `FlexibleParams`.
2. The sweep module shall accept a base `FlexibleParams` instance plus a parameter grid definition over one or more fields and run the solver on the full Cartesian product.
3. The sweep module shall collect a standard summary output set for each grid point, including at least `U`, `power`, `thrust`, and beam/domain edge metrics.
4. The sweep module shall save and load sweep artifacts in a Julia-native format.
5. The Julia project shall provide a generic runnable script `Julia/scripts/run_sweep.jl` that uses the sweep module.
6. The specialized script `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` shall be rewritten as a thin preset wrapper over the universal sweep module.
7. The implementation shall keep the sweep logic in Julia only and shall not depend on MATLAB artifacts or MATLAB execution.

## Acceptance Criteria

### Requirement 1
- A new source file exists under `Julia/src/` for sweeping.
- The module is included and exported from `Surferbot.jl`.

### Requirement 2
- The sweep API accepts a parameter grid over arbitrary `FlexibleParams` field names.
- A test verifies the Cartesian grid size and parameter override behavior.

### Requirement 3
- Each sweep point stores or exposes at least `U`, `power`, `thrust`, `eta_left_beam`, `eta_right_beam`, `eta_left_domain`, and `eta_right_domain`.
- A test verifies these fields are present in sweep outputs.

### Requirement 4
- The sweep artifact can be written and reloaded via Julia.
- A test verifies a small artifact round-trip.

### Requirement 5
- `Julia/scripts/run_sweep.jl` exists and runs under `julia --project=Julia`.

### Requirement 6
- `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` calls the generic sweep module rather than reimplementing the sweep loop.

### Requirement 7
- The sweep workflow does not read `.mat` files and does not shell out to MATLAB.

## Testing Plan

1. Add focused tests for parameter-grid expansion and sweep summary collection.
2. Add a small sweep smoke test with tiny `n`/`M` settings and a `2x2` grid.
3. Add a persistence test for artifact save/load.
4. Run the existing Julia analysis/helper tests to ensure the new sweep layer integrates cleanly.

## Implementation Plan

1. Implement a reusable sweep module with grid expansion and summary extraction.
   - Test grid expansion and summary extraction on synthetic or tiny real cases.
2. Add native artifact save/load helpers using the existing `JLD2` dependency.
   - Test artifact round-trip.
3. Include/export the sweep module from `Surferbot.jl`.
   - Test package import and helper availability.
4. Add `Julia/scripts/run_sweep.jl` as the generic sweep entry point.
   - Test script load and a tiny smoke invocation.
5. Refactor `Julia/scripts/sweep_motor_position_EI_uncoupled.jl` into a preset wrapper over the generic sweep engine.
   - Test the wrapper on a tiny smoke sweep.
6. Add sweep tests to `Julia/test/runtests.jl` and run the focused suite.

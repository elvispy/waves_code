## As Is

The repository has two MATLAB sweep artifacts for the motor-position/EI plane:

- `MATLAB/test/data/sweepMotorPositionEI.mat`
- `MATLAB/test/data/sweepMotorPositionEI2.mat`

They are saved as opaque MATLAB table objects and cannot be consumed directly by the Julia sweep loader. The Julia analysis pipeline expects native `JLD2` sweep artifacts using `SweepArtifact` and `SweepSummary` from `Julia/src/sweep.jl`.

The current Julia uncoupled analysis script loads a native file named `sweep_motorPosition_EI_uncoupled.jld2`. There is no migration path from the existing MATLAB sweep files into that Julia-native format.

Inspection of the real MATLAB files shows that once unwrapped by MATLAB, each file contains a table `S` with 20 columns:

- `thrust_N`
- `N_x`
- `M_z`
- `tail_flat_ratio`
- `args`
- `EI`
- `motor_position`
- `Sxx`
- `eta_edge_ratio`
- `n_used`
- `M_used`
- `eta_1`
- `eta_end`
- `eta_left_domain`
- `eta_right_domain`
- `eta_left_beam`
- `eta_right_beam`
- `eta_beam_ratio`
- `success`
- `retries`

The embedded `args` struct stores `power` and `thrust`, but does not store `U`.

## To Be

The repository should provide a repeatable migration path that converts the two existing MATLAB motor-position/EI sweep files into Julia-native `JLD2` sweep artifacts that load through `load_sweep(...)` and behave like artifacts produced by the Julia sweep layer.

The final artifacts should:

- be written in Julia-native `JLD2` format
- use the existing `SweepArtifact` / `SweepSummary` schema
- preserve the motor-position and EI axes from the MATLAB sweep
- preserve the stored beam/domain edge amplitudes, thrust, power-derived quantities, and tail flatness metrics
- reconstruct the intended base sweep parameters using known sweep presets rather than row 1
- set `U = NaN` explicitly because the source files do not contain it

The translation workflow may use MATLAB as a one-time helper to unwrap the opaque tables, but the final consumable artifacts must be Julia-native and free of `.mat` dependencies.

## Requirements

1. Provide a translation workflow for `sweepMotorPositionEI.mat` and `sweepMotorPositionEI2.mat` that generates Julia-native `JLD2` sweep artifacts.
2. The translated artifacts must conform to the existing Julia sweep contract in `Julia/src/sweep.jl`.
3. The translation must preserve all recoverable sweep summary quantities stored in the MATLAB files.
4. The translator must reconstruct correct sweep axes and assign rows to the right grid coordinates independent of MATLAB row order.
5. The translator must use explicit baseline sweep presets for coupled and uncoupled cases instead of inferring `base_params` from the first table row.
6. The missing quantity `U` must be handled explicitly and non-deceptively.
7. The workflow must be runnable from the repository without manual editing of source files.

## Acceptance Criteria

1. Provide a translation workflow for `sweepMotorPositionEI.mat` and `sweepMotorPositionEI2.mat` that generates Julia-native `JLD2` sweep artifacts.
   - A Julia entrypoint exists that processes the two target MATLAB sweep files and writes two `JLD2` files.
   - The workflow can be run from the repo root or Julia directory with documented defaults.

2. The translated artifacts must conform to the existing Julia sweep contract in `Julia/src/sweep.jl`.
   - Each output file loads successfully with `load_sweep(...)`.
   - Each loaded artifact is a `SweepArtifact` containing `label`, `base_params`, `parameter_axes`, and `summaries`.

3. The translation must preserve all recoverable sweep summary quantities stored in the MATLAB files.
   - `thrust`, `power`, `power_input`, beam/domain edge amplitudes, beam/domain ratios, and `tail_flat_ratio` are transferred into `SweepSummary`.
   - `U` is set to `NaN` because it is not present in the source files.

4. The translator must reconstruct correct sweep axes and assign rows to the right grid coordinates independent of MATLAB row order.
   - The translated artifact has `25` motor-position points and `57` EI points for both files.
   - A spot check of known rows matches the source values at the correct grid locations.

5. The translator must use explicit baseline sweep presets for coupled and uncoupled cases instead of inferring `base_params` from the first table row.
   - The uncoupled artifact uses the same baseline semantics as `default_uncoupled_motor_position_EI_sweep()`.
   - The coupled artifact uses an explicit coupled preset with `d = 0.03` and the same shared nominal constants as the MATLAB sweep definition.

6. The missing quantity `U` must be handled explicitly and non-deceptively.
   - The implementation contains no guessed or fabricated `U` values.
   - `U` is stored as `NaN` in all translated summaries and this behavior is documented in code or docstrings.

7. The workflow must be runnable from the repository without manual editing of source files.
   - The translator creates any temporary export files programmatically.
   - The workflow does not require the user to hand-edit MATLAB or Julia scripts before running.

## Testing Plan

- Add a focused Julia test for the row-to-`SweepArtifact` reconstruction logic using a small synthetic exported dataset.
- Run the translator on the two real MATLAB sweep files.
- Load both generated `JLD2` artifacts with `load_sweep(...)`.
- Verify artifact sizes, axis lengths, labels, baseline parameters, and at least one transferred summary value per file.

## Implementation Plan

1. Add explicit coupled sweep preset support in Julia so the translator has a correct `base_params` source for the coupled artifact.
   - Test by checking the preset fields directly in a small Julia test.

2. Add a MATLAB helper that unwraps an opaque sweep table and exports a flat numeric CSV containing all recoverable row-level quantities needed by `SweepSummary`.
   - Test by running it on one target `.mat` file and verifying the CSV header and row count.

3. Add Julia helpers that read the exported CSV, reconstruct axes, map rows into `SweepSummary`, and build a `SweepArtifact`.
   - Test with a small synthetic CSV fixture in a Julia unit test.

4. Add a Julia translation entrypoint that orchestrates MATLAB export for the two target files and writes the final `JLD2` artifacts.
   - Test by running the entrypoint end-to-end on both real files.

5. Verify the generated artifacts load through `load_sweep(...)` and match expected dimensions and representative values.
   - Test with a Julia smoke check after translation.

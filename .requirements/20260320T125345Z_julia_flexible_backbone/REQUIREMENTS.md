# Julia Flexible Backbone Requirements

## As Is

The repository contains a validated MATLAB flexible solver centered on `MATLAB/src/flexible_surferbot_v2.m`, `MATLAB/src/build_system_v2.m`, and `MATLAB/src/calculate_surferbot_outputs.m`. The Julia directory contains partial numerical utilities and exploratory rigid/AD code, but it does not yet provide a coherent flexible solver package, a stable public API, or test coverage that certifies MATLAB parity.

The current Julia code is fragmented across `DtN.jl`, `integration.jl`, `utils.jl`, `rigid.jl`, and two exploratory tests. There is no package entry module, no `runtests.jl`, no typed parameter/result API, no flexible sparse system assembly, and no verified AD path for the flexible simulation.

## To Be

The Julia directory becomes the primary backbone for the flexible simulation. It exposes a typed solver API that reproduces the core MATLAB simulation behavior, plus a thin MATLAB-style compatibility wrapper. The implementation is test-driven, with primitive, assembly, solver-parity, and AD tests. Automatic differentiation is supported through an explicit differentiated solve boundary so gradients of scalar objectives like thrust and power can be computed and verified.

## Requirements

1. The Julia codebase must expose a coherent package entry point and test harness.
2. The Julia codebase must provide a typed flexible-solver API that separates parameter derivation, sparse assembly, solve, and post-processing.
3. The Julia implementation must reproduce the core MATLAB flexible simulation outputs for a documented set of parity cases.
4. The Julia implementation must include a MATLAB-style compatibility wrapper for users who want MATLAB-like call semantics.
5. The Julia implementation must support AD for scalar objectives derived from the flexible solver.
6. The Julia implementation must include automated test coverage for primitives, assembly behavior, forward-solver parity, and AD behavior.
7. Experimental Julia files that do not fit the new backbone may be refactored, replaced, or removed.

## Acceptance Criteria

### Requirement 1

- A top-level Julia module file exists and exports the supported public API.
- `Julia/test/runtests.jl` exists and runs the package tests.
- The Julia test command can execute without relying on ad hoc script files.

### Requirement 2

- A `FlexibleParams` type exists.
- A `FlexibleResult` type exists.
- `flexible_solver(params::FlexibleParams; return_system=false)` exists.
- Sparse assembly and post-processing can be tested independently of the full solver entry point.

### Requirement 3

- A documented set of MATLAB golden cases is created or imported.
- For each golden case, Julia matches core scalar outputs within justified tolerances.
- Field-level comparisons exist for at least one small representative case.

### Requirement 4

- A wrapper function exists that accepts keyword-style inputs and returns MATLAB-like outputs.
- The wrapper delegates to the typed API rather than duplicating implementation logic.

### Requirement 5

- At least one forward-mode or reverse-mode gradient path exists for scalar objectives of the flexible solver.
- Gradients for selected parameters such as `EI`, `motor_position`, or `omega` are validated against numerical references on small problems.

### Requirement 6

- Primitive tests cover DtN, Simpson weights, Gaussian load, dispersion, and finite-difference operators.
- Assembly tests cover matrix shape, masks, and bulk residual structure.
- Solver tests cover MATLAB parity for representative cases.
- AD tests cover at least two scalar objective gradients.

### Requirement 7

- Legacy Julia experiment files are either integrated into the new architecture or removed if superseded.
- No required production behavior depends on notebook-style exploratory scripts.

## Testing Plan

1. Establish a baseline by running the current Julia test command and recording failures.
2. Add package-structure tests first and watch them fail.
3. Port or repair primitive modules with narrow tests.
4. Add manufactured-solution tests for sparse assembly before implementing the full flexible solver.
5. Add MATLAB parity tests for one small stable case, then expand to a few representative cases.
6. Add AD tests comparing Julia gradients against finite-difference or complex-step references.
7. Re-run the full Julia test suite after each completed slice.

## Implementation Plan

1. Create the Julia package entry point and `runtests.jl`.
   Test: package-load and export tests.
2. Refactor or replace primitive Julia modules so their APIs are stable and testable.
   Test: primitive unit tests for DtN, integration, utilities, and FD operators.
3. Implement the flexible sparse assembly module modeled on MATLAB `build_system_v2.m`.
   Test: assembly shape and manufactured residual tests.
4. Implement the typed flexible solver and post-processing layers.
   Test: small MATLAB parity case.
5. Add the MATLAB-style compatibility wrapper.
   Test: wrapper smoke test against the typed API.
6. Implement differentiated solve support for scalar objectives.
   Test: AD gradient parity against numerical references.
7. Prune or replace superseded Julia experimental files.
   Test: full Julia suite still passes after cleanup.

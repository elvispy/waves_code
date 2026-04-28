# Julia Flexible Backbone Design

## Goal

Replace the current experimental Julia code with a production Julia backbone that reproduces the core MATLAB flexible simulation, supports automatic differentiation, and is validated by automated tests.

## Design Summary

The real Julia API will be typed and layered. MATLAB compatibility will be provided by a thin wrapper rather than shaping the internal architecture around positional outputs and ad hoc parameter parsing.

The primary public entry point will be a typed solver:

```julia
result = flexible_solver(params::FlexibleParams; return_system=false)
```

This solver will return a structured result object that includes scalar outputs, grids, solved fields, and metadata. Internally, the solver will be split into parameter derivation, sparse system assembly, linear solve, and post-processing. This separation is required for testability and for clean AD support.

## API Shape

### Public types

- `FlexibleParams`
  Stores physical constants, raft properties, domain settings, solver settings, and forcing/boundary-condition controls.
- `FlexibleResult`
  Stores `U`, `power`, `thrust`, `x`, `z`, `phi`, `phi_z`, `eta`, `pressure`, and metadata.

### Public functions

- `flexible_solver(params::FlexibleParams; return_system=false)`
  Main typed backbone entry point.
- `flexible_surferbot_v2_julia(; kwargs...)`
  MATLAB-style compatibility wrapper returning MATLAB-like outputs.
- `flexible_objective(params::FlexibleParams, objective::Symbol)`
  Scalar objective helper for AD and optimization workflows.

### Internal functions

- `derive_params`
- `assemble_flexible_system`
- `solve_flexible_system`
- `postprocess_flexible_solution`

## Architecture

### Layer 1: Parameter normalization

Convert user-facing dimensional parameters into a validated derived-parameter structure:

- nondimensional groups
- domain depth and wavenumber defaults
- grid counts and coordinates
- masks for raft-contact and free-surface regions
- distributed forcing weights

This layer must match MATLAB behavior as closely as practical because it controls the actual numerical problem being solved.

### Layer 2: Sparse assembly

Build the coupled fluid-structure sparse system corresponding to MATLAB `build_system_v2.m`. This layer is the numerical core and must be a pure function of derived parameters and forcing.

### Layer 3: Linear solve

Solve the sparse system without mixing in post-processing or plotting behavior. This boundary is also where custom AD behavior will be attached.

### Layer 4: Post-processing

Port MATLAB `calculate_surferbot_outputs.m` into a separate Julia module that computes thrust, power, speed, surface elevation, and raft pressure.

## AD Strategy

AD support will not rely on blindly differentiating through sparse solver internals. Instead:

- system assembly remains differentiable with respect to scalar parameters
- the linear solve boundary gets explicit tangent/adjoint support
- scalar objective gradients are validated against finite-difference references on small cases

This keeps the forward solver reliable and the differentiated path auditable.

## Testing Strategy

Tests will target the typed Julia API, not the compatibility wrapper.

Coverage will be layered:

- primitive tests for DtN, integration, Gaussian load, dispersion, and finite-difference matrices
- sparse assembly tests for dimensions, masks, and manufactured-solution residuals
- solver parity tests against a small MATLAB golden corpus
- AD tests comparing Julia gradients to finite-difference or complex-step references

## Migration Constraints

- Existing Julia files may be replaced, refactored, or pruned freely.
- MATLAB remains the behavioral oracle until Julia parity is established.
- Python is a secondary structural reference only.

## Recommended First Slice

Start by turning the Julia directory into a coherent package with a real `runtests.jl`, then port and test the numerical primitives before attempting the flexible backbone assembly.

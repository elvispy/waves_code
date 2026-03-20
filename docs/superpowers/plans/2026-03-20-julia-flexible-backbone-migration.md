# Julia Flexible Backbone Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current MATLAB flexible backbone with a Julia implementation that reproduces the validated solver behavior, is covered by automated tests, and is compatible with forward/reverse-mode AD.

**Architecture:** Keep MATLAB as the short-term numerical oracle, treat the Python package as a secondary structural reference, and evolve the Julia code into the primary package. Port the flexible solver in layers: numerical primitives, system assembly, post-processing, and AD rules. Keep sparse linear algebra and finite-difference construction factored so the forward solve and the differentiated solve can be tested independently.

**Tech Stack:** Julia, Test.jl, SparseArrays, LinearSolve.jl, ForwardDiff.jl, ChainRulesCore/ChainRules.jl, optional Zygote.jl for reverse-mode validation

---

## Repository Map

### Current backbone and references

- `MATLAB/src/flexible_surferbot_v2.m`
  Main flexible entry point. Parses parameters, computes nondimensional groups, chooses grid/domain defaults, calls assembly, and post-processes outputs.
- `MATLAB/src/build_system_v2.m`
  Core sparse coupled fluid-structure assembly. This is the actual numerical backbone that must be ported faithfully.
- `MATLAB/src/calculate_surferbot_outputs.m`
  Converts solved fields into `U`, `power`, `thrust`, `eta`, and pressure. This should become a separate Julia post-processing module.
- `MATLAB/src/getNonCompactFDmatrix.m`
- `MATLAB/src/getNonCompactFDmatrix2D.m`
- `MATLAB/src/getNonCompactFDMWeights.m`
- `MATLAB/src/buildMappedFD.m`
- `MATLAB/src/buildMappedFD_anyOrder.m`
- `MATLAB/src/dispersion_k.m`
- `MATLAB/src/gaussian_load.m`
- `MATLAB/src/solve_system.m`
  Numerical support functions needed by the backbone.

### Existing MATLAB validation surface

- `MATLAB/test/test_pde_residual_single.m`
  Best direct reference for PDE residual validation.
- `MATLAB/test/test_quick_checks.m`
- `MATLAB/test/test_sweep_frequency.m`
- `MATLAB/test/test_thrust_vs_*.m`
- `MATLAB/test/test_power_vs_EI.m`
  Behavioral and sweep-style validation suite. These tests define what “matching MATLAB” should mean at the system level.

### Python structural reference

- `python/src/surferbot/flexible_surferbot.py`
  Earlier flexible implementation using JAX/tensor operators. Useful for decomposition ideas, not the truth source.
- `python/src/surferbot/rigid.py`
  More mature rigid reference and likely the closest ancestor of the current Julia rigid solver.
- `python/src/surferbot/DtN.py`
- `python/src/surferbot/integration.py`
- `python/src/surferbot/utils.py`
- `python/tests/test_DtN.py`
- `python/tests/test_utils.py`
- `python/tests/test_solver.py`
  Existing automated tests that can be mirrored into Julia for the primitive pieces.

### Existing Julia starting point

- `Julia/src/DtN.jl`
  Ported DtN generator.
- `Julia/src/integration.jl`
  Ported Simpson weights.
- `Julia/src/utils.jl`
  Contains a direct solve wrapper, Gaussian load, and dispersion relation.
- `Julia/src/rigid.jl`
  Rigid solver prototype and the closest local style reference.
- `Julia/test/implicitAD.jl`
  Experimental implicit differentiation sketch.
- `Julia/test/zygotest.jl`
  Experimental reverse-mode sparse solve sketch.

### Immediate package gaps

- No top-level Julia module file such as `Julia/src/surferbot.jl`.
- No Julia `Test.jl` test harness mirroring the package API.
- No flexible Julia solver module.
- No AD-safe linear solve abstraction with explicit tangent/adjoint strategy.
- No MATLAB-to-Julia parity fixtures or golden-output corpus.

## Target File Structure

### Create

- `Julia/src/surferbot.jl`
  Package entry point and exports.
- `Julia/src/fd.jl`
  Finite-difference weights/matrices, including 1D and 2D operators that replace the MATLAB non-compact FD helpers.
- `Julia/src/flexible.jl`
  `flexible_solver` entry point matching the MATLAB backbone responsibilities.
- `Julia/src/flexible_system.jl`
  Sparse system assembly corresponding to `build_system_v2.m`.
- `Julia/src/postprocess.jl`
  Julia port of `calculate_surferbot_outputs.m`.
- `Julia/src/ad.jl`
  Implicit differentiation helpers and custom rules for the linear solve path.
- `Julia/test/runtests.jl`
  Test entry point.
- `Julia/test/test_dtn.jl`
- `Julia/test/test_integration.jl`
- `Julia/test/test_utils.jl`
- `Julia/test/test_fd.jl`
- `Julia/test/test_flexible_system.jl`
- `Julia/test/test_flexible_solver.jl`
- `Julia/test/test_ad.jl`
- `Julia/test/fixtures/`
  Golden fixtures derived from MATLAB runs or analytically constructed cases.

### Modify

- `Julia/Project.toml`
  Ensure the package name, test dependencies, and AD dependencies are coherent.
- `Julia/Manifest.toml`
  Regenerate after dependency cleanup.
- `Julia/src/DtN.jl`
  Align naming/types with package conventions and test coverage.
- `Julia/src/integration.jl`
  Align exported API and test coverage.
- `Julia/src/utils.jl`
  Split solver concerns from utilities and make types AD-friendly.
- `Julia/src/rigid.jl`
  Convert to package-internal style or keep as separate module with shared primitives.

## Migration Strategy

### Task 1: Stabilize the Julia package boundary

**Files:**
- Create: `Julia/src/surferbot.jl`
- Modify: `Julia/src/DtN.jl`
- Modify: `Julia/src/integration.jl`
- Modify: `Julia/src/utils.jl`
- Modify: `Julia/src/rigid.jl`
- Test: `Julia/test/runtests.jl`

- [ ] **Step 1: Write failing package-load tests**

```julia
using Test
using surferbot

@test isdefined(surferbot, :DtN_generator)
@test isdefined(surferbot, :simpson_weights)
@test isdefined(surferbot, :dispersion_k)
```

- [ ] **Step 2: Run the failing tests**

Run: `cd Julia && julia --project -e 'using Pkg; Pkg.test()'`
Expected: FAIL because the top-level module and exports are incomplete.

- [ ] **Step 3: Add the top-level package module and wire existing modules into it**

- [ ] **Step 4: Re-run package tests until they pass**

Run: `cd Julia && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS for module wiring tests.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/surferbot.jl Julia/src/DtN.jl Julia/src/integration.jl Julia/src/utils.jl Julia/src/rigid.jl Julia/test/runtests.jl
git commit -m "feat: establish julia surferbot package skeleton"
```

### Task 2: Port and verify numerical primitives before touching the flexible solver

**Files:**
- Create: `Julia/src/fd.jl`
- Modify: `Julia/src/utils.jl`
- Test: `Julia/test/test_dtn.jl`
- Test: `Julia/test/test_integration.jl`
- Test: `Julia/test/test_utils.jl`
- Test: `Julia/test/test_fd.jl`
- Reference: `MATLAB/src/getNonCompactFDmatrix.m`
- Reference: `MATLAB/src/getNonCompactFDmatrix2D.m`
- Reference: `MATLAB/src/getNonCompactFDMWeights.m`
- Reference: `MATLAB/src/dispersion_k.m`
- Reference: `MATLAB/src/gaussian_load.m`

- [ ] **Step 1: Write failing DtN, Simpson, Gaussian load, dispersion, and FD matrix tests**

Use analytical cases and small matrix shape/value tests. Include one manufactured harmonic field test for the 2D operators.

- [ ] **Step 2: Run the targeted tests to verify proper failure**

Run: `cd Julia && julia --project test/test_dtn.jl`
Expected: FAIL on missing/incorrect primitive interfaces.

- [ ] **Step 3: Implement or repair each primitive in the smallest possible increments**

- [ ] **Step 4: Re-run the narrow test files after each primitive**

Run: `cd Julia && julia --project -e 'using Pkg; Pkg.test(test_args=["test_dtn","test_integration","test_utils","test_fd"])'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/fd.jl Julia/src/DtN.jl Julia/src/integration.jl Julia/src/utils.jl Julia/test/test_dtn.jl Julia/test/test_integration.jl Julia/test/test_utils.jl Julia/test/test_fd.jl
git commit -m "feat: port core numerical primitives for julia backbone"
```

### Task 3: Build the flexible sparse assembly as a pure function

**Files:**
- Create: `Julia/src/flexible_system.jl`
- Test: `Julia/test/test_flexible_system.jl`
- Reference: `MATLAB/src/build_system_v2.m`
- Reference: `MATLAB/test/test_pde_residual_single.m`

- [ ] **Step 1: Write failing tests for system dimensions, block placement, masks, and manufactured residual behavior**

Focus on:
- matrix shape and sparsity pattern
- contact/free/bulk index partitioning
- boundary-condition rows
- zero residual for a manufactured harmonic field in the bulk rows

- [ ] **Step 2: Run the system tests and confirm they fail for the expected reason**

Run: `cd Julia && julia --project test/test_flexible_system.jl`
Expected: FAIL because `assemble_flexible_system` does not exist or does not match the MATLAB layout.

- [ ] **Step 3: Implement `assemble_flexible_system(args)` as a pure sparse-assembly function**

Return at minimum:
- assembled sparse matrix
- rhs vector
- structured indices/masks
- grid arrays

- [ ] **Step 4: Re-run the targeted system tests**

Run: `cd Julia && julia --project test/test_flexible_system.jl`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/flexible_system.jl Julia/test/test_flexible_system.jl
git commit -m "feat: port flexible sparse system assembly"
```

### Task 4: Port the flexible entry point and post-processing against MATLAB parity

**Files:**
- Create: `Julia/src/flexible.jl`
- Create: `Julia/src/postprocess.jl`
- Test: `Julia/test/test_flexible_solver.jl`
- Fixture: `Julia/test/fixtures/`
- Reference: `MATLAB/src/flexible_surferbot_v2.m`
- Reference: `MATLAB/src/calculate_surferbot_outputs.m`
- Reference: `MATLAB/test/test_quick_checks.m`
- Reference: `MATLAB/test/test_thrust_vs_*.m`

- [ ] **Step 1: Produce a small MATLAB golden corpus**

Generate a few stable parameter sets and save:
- scalar outputs (`U`, `power`, `thrust`)
- grid arrays
- surface fields
- pressure on raft

- [ ] **Step 2: Write failing Julia parity tests against the golden corpus**

Use relative tolerances per quantity, not bitwise equality.

- [ ] **Step 3: Implement `flexible_solver` and `calculate_outputs` with the minimal API needed for the parity tests**

- [ ] **Step 4: Re-run parity tests and tune tolerances only after investigating mismatches**

Run: `cd Julia && julia --project test/test_flexible_solver.jl`
Expected: PASS with documented tolerances.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/flexible.jl Julia/src/postprocess.jl Julia/test/test_flexible_solver.jl Julia/test/fixtures
git commit -m "feat: add julia flexible solver with matlab parity tests"
```

### Task 5: Make the solver AD-enabled through the linear solve boundary

**Files:**
- Create: `Julia/src/ad.jl`
- Modify: `Julia/src/flexible.jl`
- Modify: `Julia/src/flexible_system.jl`
- Test: `Julia/test/test_ad.jl`
- Reference: `Julia/test/implicitAD.jl`
- Reference: `Julia/test/zygotest.jl`

- [ ] **Step 1: Write failing AD tests for scalar objectives of the flexible solver**

Examples:
- derivative of thrust with respect to `EI`
- derivative of thrust with respect to `motor_position`
- derivative of power with respect to `omega`

Each test should compare AD output to finite-difference or complex-step reference on a tiny problem.

- [ ] **Step 2: Run the AD tests and confirm they fail in a controlled way**

Run: `cd Julia && julia --project test/test_ad.jl`
Expected: FAIL because sparse solve differentiation is not yet implemented or is unstable.

- [ ] **Step 3: Implement implicit differentiation or custom `rrule`/`frule` at the solve boundary**

Requirements:
- avoid differentiating through sparse factorization internals directly
- treat system assembly as differentiable
- solve tangent/adjoint linear systems explicitly

- [ ] **Step 4: Re-run AD tests and broaden coverage**

Run: `cd Julia && julia --project -e 'using Pkg; Pkg.test(test_args=["test_ad","test_flexible_solver"])'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/ad.jl Julia/src/flexible.jl Julia/src/flexible_system.jl Julia/test/test_ad.jl
git commit -m "feat: add ad-enabled flexible solver gradients"
```

### Task 6: Promote Julia to the new backbone and preserve MATLAB as validation-only

**Files:**
- Modify: `README.md`
- Optionally create: `docs/julia-migration.md`
- Optionally create: `MATLAB/test/export_golden_case.m`

- [ ] **Step 1: Write failing documentation or smoke tests that assert the Julia entry point exists and is the documented path**

- [ ] **Step 2: Update repository documentation and developer workflow**

Document:
- how to run Julia tests
- how to regenerate MATLAB parity fixtures
- what “TDD-certified” means in this repo
- how AD gradients are validated

- [ ] **Step 3: Run the full Julia test suite**

Run: `cd Julia && julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/julia-migration.md MATLAB/test/export_golden_case.m
git commit -m "docs: document julia flexible backbone workflow"
```

## TDD Rules For Execution

- Every production change starts with a failing Julia test.
- Keep MATLAB as the oracle, not the implementation target.
- Prefer tiny deterministic fixtures over large sweep tests during the red-green loop.
- Add sweep/regression tests only after the local primitive or module is stable.
- Do not introduce AD plumbing until the non-differentiated forward solver is numerically verified.

## Recommended First Execution Slice

If you want the fastest high-confidence start, execute only:

1. Task 1
2. Task 2
3. The matrix-shape and manufactured-solution parts of Task 3

That gets the Julia package into a state where the flexible port can proceed with real tests instead of notebook-style experimentation.

# Surferbot Thrust Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Julia optimization workflow that maximizes thrust over `xA` and `logEI` using implicit differentiation and `LBFGS`, with a softplus penalty enforcing a maximum actuator power input.

**Architecture:** Keep the primal `Surferbot` solver as the source of truth and add a separate optimization layer that maps `theta -> params -> solve -> objective/gradient`. Compute gradients at the solve boundary with implicit differentiation rather than by differentiating through sparse factorization internals.

**Tech Stack:** Julia, `Surferbot`, sparse linear solves, implicit differentiation, `Optim.jl`, `Test`

---

## File Map

- Create: `Julia/src/optimization.jl`
  - optimization parameter struct
  - objective config
  - power normalization helper
  - softplus penalty
  - objective evaluation
  - implicit-gradient interface
  - optimization result struct
- Modify: `Julia/src/Surferbot.jl`
  - include/export optimization module if appropriate
  - expose minimal solver internals needed by optimization
- Modify: `Julia/Project.toml`
  - add optimization dependency if missing
- Create: `Julia/scripts/optimization.jl`
  - runnable script for one optimization run
- Create: `Julia/test/test_optimization_objective.jl`
  - objective construction and power-penalty tests
- Create: `Julia/test/test_optimization_gradients.jl`
  - implicit-gradient vs finite-difference checks
- Modify: `Julia/test/runtests.jl`
  - register the new tests

## Chunk 1: Objective and Parameterization

### Task 1: Add the optimization module shell

**Files:**
- Create: `Julia/src/optimization.jl`
- Modify: `Julia/src/Surferbot.jl`

- [ ] **Step 1: Write the failing package-surface test**

Add a test in `Julia/test/test_optimization_objective.jl` that imports the planned symbols:

- `OptimizationParams`
- `OptimizationConfig`
- `thrust_objective`
- `softplus_penalty`

- [ ] **Step 2: Run the new test to verify failure**

Run:
```bash
cd /Users/eaguerov/Documents/Github/waves_code/Julia
JULIA_DEPOT_PATH=$PWD/.julia_depot:/Users/eaguerov/.julia /Users/eaguerov/.julia/juliaup/julia-1.12.1+0.x64.apple.darwin14/bin/julia --project -e 'using Test, Surferbot; include("test/test_optimization_objective.jl")'
```

Expected: import or symbol failure.

- [ ] **Step 3: Add the minimal optimization module**

Create `Julia/src/optimization.jl` with:

- `OptimizationParams`
- `OptimizationConfig`
- `softplus_penalty`
- placeholder `thrust_objective`

Modify `Julia/src/Surferbot.jl` to include and export the new module API.

- [ ] **Step 4: Run the test to verify pass**

Run the same command.

Expected: package-surface test passes.

- [ ] **Step 5: Commit**

```bash
git add Julia/src/optimization.jl Julia/src/Surferbot.jl Julia/test/test_optimization_objective.jl
git commit -m "feat: add optimization module surface"
```

### Task 2: Define theta-to-parameter mapping

**Files:**
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_objective.jl`

- [ ] **Step 1: Write the failing mapping test**

Add tests that:

- map `theta = [xA, logEI]` into `FlexibleParams`
- verify `EI = exp(logEI)`
- verify `motor_position = xA`

- [ ] **Step 2: Run the mapping test to verify failure**

Use the same direct Julia test command.

- [ ] **Step 3: Implement `theta_to_params` minimally**

Add a helper such as:

```julia
theta_to_params(theta, base_params)
```

that returns a new `FlexibleParams`.

- [ ] **Step 4: Run the test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/src/optimization.jl Julia/test/test_optimization_objective.jl
git commit -m "feat: add optimization parameter mapping"
```

### Task 3: Define positive input power and penalized objective

**Files:**
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_objective.jl`

- [ ] **Step 1: Write the failing objective tests**

Add tests for:

- positive power normalization from solver output
- zero or near-zero penalty when `power <= Pmax`
- increasing penalty when `power > Pmax`

- [ ] **Step 2: Run the tests to verify failure**

- [ ] **Step 3: Implement**

Add:

- `input_power(result_or_power)`
- `softplus_penalty(power, Pmax, mu, beta)`
- `thrust_objective(theta, base_params, config)`

Use the penalized objective:

```julia
J = -thrust + mu * softplus(power - Pmax)^2
```

- [ ] **Step 4: Run the tests to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/src/optimization.jl Julia/test/test_optimization_objective.jl
git commit -m "feat: add penalized thrust objective"
```

## Chunk 2: Implicit Gradient

### Task 4: Expose the primal state needed by optimization

**Files:**
- Modify: `Julia/src/Surferbot.jl`
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_gradients.jl`

- [ ] **Step 1: Write the failing state-access test**

Add a test that requests the primal state and confirms access to:

- assembled system
- solution vector or split fields
- thrust and power outputs

- [ ] **Step 2: Run the test to verify failure**

- [ ] **Step 3: Implement minimal state exposure**

Prefer a structured return path such as:

```julia
evaluate_state(theta, base_params, config)
```

that wraps `flexible_solver(...; return_system=true)` and returns the solved state plus outputs.

- [ ] **Step 4: Run the test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/src/Surferbot.jl Julia/src/optimization.jl Julia/test/test_optimization_gradients.jl
git commit -m "feat: expose solver state for optimization"
```

### Task 5: Implement derivative actions for `xA`

**Files:**
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_gradients.jl`

- [ ] **Step 1: Write the failing `xA` gradient test**

Add a finite-difference comparison for the objective gradient with respect to `xA` at one interior point.

- [ ] **Step 2: Run the test to verify failure**

- [ ] **Step 3: Implement `db/dxA` and any needed `dA/dxA * x` action**

For the first version, expect most `xA` dependence to enter through the Gaussian load in `b`.

Implement helpers like:

```julia
db_dxa(theta, ...)
dA_times_x_dxa(theta, x, ...)
```

- [ ] **Step 4: Implement the implicit objective gradient contribution for `xA`**

Add the scalar-gradient assembly path using the primal and adjoint solve.

- [ ] **Step 5: Run the test to verify pass**

- [ ] **Step 6: Commit**

```bash
git add Julia/src/optimization.jl Julia/test/test_optimization_gradients.jl
git commit -m "feat: add implicit gradient for motor position"
```

### Task 6: Implement derivative actions for `logEI`

**Files:**
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_gradients.jl`

- [ ] **Step 1: Write the failing `logEI` gradient test**

Add a finite-difference comparison for the objective gradient with respect to `logEI`.

- [ ] **Step 2: Run the test to verify failure**

- [ ] **Step 3: Implement `dA/dEI * x` and chain rule to `logEI`**

Implement helpers like:

```julia
dA_times_x_dEI(theta, x, ...)
dA_times_x_dlogEI(theta, x, ...) = EI * dA_times_x_dEI(...)
```

and `db/dlogEI` if needed.

- [ ] **Step 4: Implement the implicit objective gradient contribution for `logEI`**

- [ ] **Step 5: Run the test to verify pass**

- [ ] **Step 6: Commit**

```bash
git add Julia/src/optimization.jl Julia/test/test_optimization_gradients.jl
git commit -m "feat: add implicit gradient for logEI"
```

### Task 7: Validate the full 2D implicit gradient

**Files:**
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_gradients.jl`

- [ ] **Step 1: Write the failing vector-gradient test**

Add a test comparing the full implicit gradient `[dJ/dxA, dJ/dlogEI]` against central finite differences.

- [ ] **Step 2: Run the test to verify failure**

- [ ] **Step 3: Implement `objective_and_gradient`**

Add a function like:

```julia
objective_and_gradient(theta, base_params, config)
```

that returns both `J` and `grad`.

- [ ] **Step 4: Run the test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/src/optimization.jl Julia/test/test_optimization_gradients.jl
git commit -m "feat: validate full implicit optimization gradient"
```

## Chunk 3: LBFGS Driver

### Task 8: Add the optimizer dependency and wrapper

**Files:**
- Modify: `Julia/Project.toml`
- Modify: `Julia/src/optimization.jl`
- Test: `Julia/test/test_optimization_objective.jl`

- [ ] **Step 1: Write the failing optimizer-surface test**

Add a test that checks the existence of an optimization entry point such as:

```julia
run_thrust_optimization(...)
```

- [ ] **Step 2: Run the test to verify failure**

- [ ] **Step 3: Add `Optim.jl` and the minimal wrapper**

Implement a wrapper around `LBFGS` using the validated objective and gradient.

- [ ] **Step 4: Run the test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/Project.toml Julia/src/optimization.jl Julia/test/test_optimization_objective.jl
git commit -m "feat: add LBFGS optimization wrapper"
```

### Task 9: Add the runnable script

**Files:**
- Create: `Julia/scripts/optimization.jl`
- Modify: `Julia/src/optimization.jl`

- [ ] **Step 1: Write the failing script smoke test**

If a script test is too heavy, add a minimal callable path test from Julia instead.

- [ ] **Step 2: Run the smoke test to verify failure**

- [ ] **Step 3: Implement the script**

The script should:

- define baseline parameters
- define `Pmax`, `mu`, bounds, and initial guess
- call `run_thrust_optimization`
- print:
  - optimal `xA`
  - optimal `EI`
  - final thrust
  - final power
  - final penalty

- [ ] **Step 4: Run the smoke test to verify pass**

- [ ] **Step 5: Commit**

```bash
git add Julia/scripts/optimization.jl Julia/src/optimization.jl
git commit -m "feat: add thrust optimization script"
```

### Task 10: Register and run the new test suite

**Files:**
- Modify: `Julia/test/runtests.jl`

- [ ] **Step 1: Add the new optimization tests to `runtests.jl`**

- [ ] **Step 2: Run the focused optimization tests**

Run:
```bash
cd /Users/eaguerov/Documents/Github/waves_code/Julia
JULIA_DEPOT_PATH=$PWD/.julia_depot:/Users/eaguerov/.julia /Users/eaguerov/.julia/juliaup/julia-1.12.1+0.x64.apple.darwin14/bin/julia --project -e 'using Test, Surferbot; include("test/test_optimization_objective.jl"); include("test/test_optimization_gradients.jl")'
```

Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run:
```bash
cd /Users/eaguerov/Documents/Github/waves_code/Julia
JULIA_DEPOT_PATH=$PWD/.julia_depot:/Users/eaguerov/.julia /Users/eaguerov/.julia/juliaup/julia-1.12.1+0.x64.apple.darwin14/bin/julia --project test/runtests.jl
```

Expected: existing solver parity tests still pass.

- [ ] **Step 4: Commit**

```bash
git add Julia/test/runtests.jl Julia/test/test_optimization_objective.jl Julia/test/test_optimization_gradients.jl
git commit -m "test: add optimization test coverage"
```

# Refactor Surferbot for Automatic Differentiation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Surferbot solver to be type-generic, enabling robust and high-performance Automatic Differentiation (AD) for optimization.

**Architecture:**
- Make `FlexibleParams` and `FlexibleResult` parametric (`{T}`).
- Update core numerical functions (`derive_params`, `assemble_flexible_system`, `getNonCompactFDmatrix`, `dispersion_k`) to be type-agnostic.
- Replace manual finite-difference assembly derivatives in `SurferbotOptimization` with `ForwardDiff.jl`.
- Fix the fragile `Main.Surferbot` dependency in the optimization module.

**Tech Stack:**
- Julia
- ForwardDiff.jl
- SparseArrays
- LinearAlgebra

---

### Task 1: Parametric Structs in Surferbot.jl

**Files:**
- Modify: `Julia/src/Surferbot.jl`

- [ ] **Step 1: Update `FlexibleParams` to be parametric.**
  ```julia
  Base.@kwdef struct FlexibleParams{T<:Real}
      sigma::T = 72.2e-3
      # ... all other Real fields ...
      bc::Symbol = :radiative
  end
  ```
- [ ] **Step 2: Update `FlexibleResult` to be parametric.**
  ```julia
  struct FlexibleResult{T<:Real}
      U::T
      power::T
      thrust::T
      x::Vector{T}
      z::Vector{T}
      phi::Matrix{Complex{T}}
      phi_z::Matrix{Complex{T}}
      eta::Vector{Complex{T}}
      pressure::Vector{Complex{T}}
      metadata::NamedTuple
  end
  ```

### Task 2: Type-Generic FD and Utils

**Files:**
- Modify: `Julia/src/fd.jl`
- Modify: `Julia/src/utils.jl`

- [ ] **Step 1: Update `getNonCompactFDmatrix` to use generic element type.**
  Modify the signature to infer or accept `T`. Use `spzeros(T, npx, npx)`.
- [ ] **Step 2: Update `dispersion_k` in `utils.jl`.**
  Ensure internal calculations don't force `Float64` or `ComplexF64`.

### Task 3: Generic Preprocessing and Assembly

**Files:**
- Modify: `Julia/src/Surferbot.jl`

- [ ] **Step 1: Update `derive_params(params::FlexibleParams{T})`.**
  Ensure all derived scales and grids use type `T`.
- [ ] **Step 2: Update `assemble_flexible_system`.**
  Initialize `S11`, `S12`, etc., using `spzeros(Complex{T}, ...)`.

### Task 4: AD Integration in Optimization Module

**Files:**
- Modify: `Julia/src/optimization.jl`

- [ ] **Step 1: Replace `differentiate_assembly` with `ForwardDiff`.**
  ```julia
  function differentiate_assembly(theta, base_params, config)
      f = (th) -> begin
          p = theta_to_params(th, base_params)
          sys = Surferbot.assemble_flexible_system(p)
          return vcat(vec(sys.A), sys.b)
      end
      J = ForwardDiff.jacobian(f, theta)
      # ... unpack dA, db ...
  end
  ```
- [ ] **Step 2: Remove `Main.Surferbot` references.**
  Call `Surferbot.assemble_flexible_system` directly.

### Task 5: Verification

**Files:**
- Test: `Julia/test/test_optimization_gradients.jl`
- Run: `Julia/scripts/optimization.jl`

- [ ] **Step 1: Run existing gradient tests.**
  Verify that AD gradients match finite differences (as they do now) but without manual step tuning.
- [ ] **Step 2: Run the optimization demo.**
  Ensure the full loop still converges correctly.

# Surferbot Thrust Optimization Design

## Goal

Add a Julia optimization workflow that uses the existing `Surferbot` solver to maximize thrust with respect to motor position and flexural rigidity under a maximum power-input constraint, using implicit differentiation and `LBFGS`.

## Scope

The first optimization problem is intentionally narrow:

- Decision variables:
  - `xA`: motor position
  - `logEI`: logarithm of flexural rigidity
- Objective:
  - maximize thrust
- Constraint handling:
  - softplus penalty on excess input power
- Optimizer:
  - `LBFGS`

This first version does not optimize `omega`, raft length, or other parameters. Those are deferred until the optimization and gradient machinery are validated.

## Optimization Formulation

Let `theta = [xA, logEI]`. Define

- `EI = exp(logEI)`
- `T(theta)`: mean thrust returned by the solver
- `P(theta)`: mean actuator power input, using the paper definition
- `Pmax`: allowable power budget

The optimization minimizes

`J(theta) = -T(theta) + mu * softplus(P(theta) - Pmax)^2`

where `mu > 0` is a penalty weight.

This formulation is chosen because:

- `LBFGS` is unconstrained
- the penalty is smooth
- the softplus transition is differentiable and numerically stable
- the problem remains suitable for implicit/adjoint differentiation

## Differentiation Strategy

The solver is based on a sparse linear system

`A(theta) * x(theta) = b(theta)`

For the scalar objective `J(theta)`, gradients should be computed with implicit differentiation at the solve boundary rather than by differentiating through sparse factorization internals.

The implementation should compute:

- the primal state `x`
- the adjoint state associated with the scalar objective
- derivative actions, not full tensors:
  - `(dA/dxA) * x`
  - `(dA/dlogEI) * x`
  - `db/dxA`
  - `db/dlogEI`

Because `EI = exp(logEI)`, the implementation must apply

- `d/dlogEI = EI * d/dEI`

The optimization code should keep the primal sparse solve unchanged and attach derivative logic at the solver boundary.

## Power Definition

The power budget must use the paper’s mean input-power definition from `overleaf/wave-driven-propulsion/main.tex`:

`<P>_A = -(omega/2) * integral Im{ f_hat(x) * eta_hat(x)^* } dx`

The Julia workflow must normalize this to a positive input-power quantity suitable for optimization. The sign convention should be documented explicitly in code.

## Code Structure

The feature should introduce a focused optimization layer rather than embedding optimization logic into `Surferbot.jl`.

Planned components:

- `Julia/src/optimization.jl`
  - parameter mapping
  - penalized objective
  - implicit gradient helpers
  - result structs
- `Julia/scripts/optimization.jl`
  - runnable optimization entry point
- `Julia/test/test_optimization_objective.jl`
  - objective and power-penalty checks
- `Julia/test/test_optimization_gradients.jl`
  - implicit-gradient vs finite-difference validation

`Julia/src/Surferbot.jl` may require small extensions to expose solver internals needed by the optimization layer, but it should remain the main primal-solver API.

## Validation

The optimization layer is only trustworthy if gradients are validated before running `LBFGS`.

Required checks:

- finite-difference comparison for `dJ/dxA`
- finite-difference comparison for `dJ/dlogEI`
- check at an interior parameter point
- check that the penalty activates correctly above `Pmax`
- check that optimization moves toward higher thrust while respecting the power budget approximately

## Risks

- The current solver is complex-valued internally, while the optimization objective is real-valued.
- The postprocessing stage contributes to `T(theta)` and `P(theta)` and must be handled consistently in the gradient.
- If the solve fallback path differs from the main sparse path, gradient definitions may become inconsistent.

These risks are acceptable for the first version as long as:

- gradients are validated numerically
- the main sparse solve path is used consistently during optimization

## Recommendation

Implement the optimization workflow in two stages:

1. Build and validate the penalized objective and its implicit gradient for `xA` and `logEI`.
2. Add the runnable `LBFGS` driver only after the gradient validation passes.

This gives the cleanest path to a publishable optimization result while preserving the MATLAB-matching solver core.

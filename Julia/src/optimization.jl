module SurferbotOptimization

using LinearAlgebra
using Surferbot

export OptimizationParams,
       OptimizationConfig,
       OptimizationResult,
       softplus_penalty,
       theta_to_params,
       input_power,
       thrust_objective,
       objective_and_gradient,
       run_thrust_optimization

"""
    OptimizationParams

Container for the optimization variables and simple box constraints.

- `theta0[1] = xA`, the motor position along the raft.
- `theta0[2] = logEI`, the logarithm of the flexural rigidity.

The use of `logEI` keeps `EI = exp(logEI)` strictly positive and improves
scaling during optimization.
"""
Base.@kwdef struct OptimizationParams
    theta0::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

"""
    OptimizationConfig

Configuration for the penalized thrust optimization problem.

The objective minimized by this module is

`J(theta) = -T(theta) + mu * softplus(P_in(theta) - Pmax)^2`

where:

- `T(theta)` is the mean thrust
- `P_in(theta)` is the mean actuator power input
- `Pmax` is the allowed power budget
- `mu` is the penalty weight
- `beta` controls the sharpness of the softplus transition

`fd_step` is used for directional differentiation of the scalar postprocessed
objective, and `sens_step` is used for finite-difference approximation of the
assembly derivatives `dA/dtheta` and `db/dtheta`.
"""
Base.@kwdef struct OptimizationConfig
    Pmax::Float64
    mu::Float64 = 1.0
    beta::Float64 = 50.0
    fd_step::Float64 = 1e-6
    sens_step::Float64 = 1e-6
end

"""
    OptimizationResult

Summary of a completed optimization run.

- `theta` stores the final optimization variables `[xA, logEI]`
- `objective` is the final penalized objective value
- `thrust` is the final mean thrust
- `power_input` is the final mean input power used in the penalty
- `gradient` is the final objective gradient
- `iterations` is the number of iterations performed
- `converged` indicates whether the gradient tolerance was met
"""
struct OptimizationResult
    theta::Vector{Float64}
    objective::Float64
    thrust::Float64
    power_input::Float64
    gradient::Vector{Float64}
    iterations::Int
    converged::Bool
end

"""
    softplus(z, beta)

Smooth approximation of `max(z, 0)`.

This is used to impose the power constraint through a differentiable penalty.
For large positive `z`, `softplus(z, beta) ≈ z`; for large negative `z`, it is
close to zero.
"""
function softplus(z::Real, beta::Real)
    if beta * z > 40
        return float(z)
    elseif beta * z < -40
        return exp(beta * z) / beta
    else
        return log1p(exp(beta * z)) / beta
    end
end

"""
    softplus_penalty(power_input, Pmax, mu; beta=50.0)

Return the smooth penalty applied when the input power exceeds the allowed
budget `Pmax`.

The penalty is

`mu * softplus(power_input - Pmax)^2`

so that points below the power budget are essentially unpenalized, while
points above the budget are pushed back toward feasibility.
"""
function softplus_penalty(power_input::Real, Pmax::Real, mu::Real; beta::Real=50.0)
    excess = power_input - Pmax
    return mu * softplus(excess, beta)^2
end

"""
    theta_to_params(theta, base_params)

Map the optimization variables `theta = [xA, logEI]` into a concrete
`FlexibleParams` instance.

This is the bridge between the low-dimensional optimization problem and the
full Surferbot simulation.
"""
function theta_to_params(theta::AbstractVector{<:Real}, base_params)
    @assert length(theta) == 2
    return typeof(base_params)(;
        sigma = base_params.sigma,
        rho = base_params.rho,
        omega = base_params.omega,
        nu = base_params.nu,
        g = base_params.g,
        L_raft = base_params.L_raft,
        motor_position = float(theta[1]),
        d = base_params.d,
        EI = exp(float(theta[2])),
        rho_raft = base_params.rho_raft,
        L_domain = base_params.L_domain,
        domain_depth = base_params.domain_depth,
        n = base_params.n,
        M = base_params.M,
        ooa = base_params.ooa,
        motor_inertia = base_params.motor_inertia,
        motor_force = base_params.motor_force,
        forcing_width = base_params.forcing_width,
        bc = base_params.bc,
    )
end

"""
    input_power(power)
    input_power(result)

Convert the solver's power sign convention into a positive actuator input power.

The paper defines the mean input power through the cycle-averaged actuator work
rate. In the current solver implementation the reported `power` is negative
when the actuator injects power, so this helper returns `P_in = -power`.
"""
input_power(power::Real) = -float(power)
input_power(result) = input_power(result.power)

"""
    build_output_args(derived, params)

Assemble the dimensional and nondimensional quantities required by the
postprocessing formulas for thrust, speed, and input power.
"""
function build_output_args(derived, params)
    return (
        sigma = params.sigma,
        rho = params.rho,
        omega = params.omega,
        nu = params.nu,
        g = params.g,
        L_raft = params.L_raft,
        d = derived.d,
        nd_groups = derived.nd_groups,
        x_contact = derived.x_contact,
        x = derived.x .* derived.L_c,
        loads = derived.loads .* derived.F_c ./ derived.L_c,
        N = derived.N,
        M = derived.M,
        dx = derived.dx .* derived.L_c,
        dz = derived.dz .* derived.L_c,
        t_c = derived.t_c,
        L_c = derived.L_c,
        m_c = derived.m_c,
        k = derived.k,
        ooa = params.ooa,
    )
end

"""
    split_state(solution, derived)

Split the stacked harmonic state vector into the fields `phi` and `phi_z` on
the `(M, N)` grid.
"""
function split_state(solution::AbstractVector, derived)
    NP = derived.N * derived.M
    phi = reshape(solution[1:NP], derived.M, derived.N)
    phi_z = reshape(solution[(NP + 1):(2 * NP)], derived.M, derived.N)
    return phi, phi_z
end

"""
    evaluate_from_state(solution, derived, params, config)

Evaluate the scalar optimization objective and the physically relevant outputs
from a solved harmonic state.

This helper applies the postprocessing formulas from the solver to obtain
thrust, drift speed, and mean input power, then forms the penalized objective

`J = -T + penalty(P_in - Pmax)`.
"""
function evaluate_from_state(solution::AbstractVector, derived, params, config::OptimizationConfig)
    phi, phi_z = split_state(solution, derived)
    args = build_output_args(derived, params)
    U, power, thrust, eta, p = Surferbot.calculate_surferbot_outputs(args, phi, phi_z, Surferbot.getNonCompactFDmatrix, Surferbot.getNonCompactFDmatrix2D)
    Pin = input_power(power)
    penalty = softplus_penalty(Pin, config.Pmax, config.mu; beta=config.beta)
    objective = -thrust + penalty
    return (
        objective = objective,
        thrust = thrust,
        power = power,
        power_input = Pin,
        penalty = penalty,
        U = U,
        eta = eta,
        pressure = p,
        phi = phi,
        phi_z = phi_z,
        args = args,
    )
end

"""
    evaluate_primal(theta, base_params, config)

Solve the primal linear system for the parameter vector `theta` and return the
full state together with postprocessed outputs.

Mathematically, this computes the state `x(theta)` from

`A(theta) * x(theta) = b(theta)`.
"""
function evaluate_primal(theta::AbstractVector{<:Real}, base_params, config::OptimizationConfig)
    params = theta_to_params(theta, base_params)
    system = Surferbot.assemble_flexible_system(params)
    full_solution = Surferbot.solve_tensor_system(system.A, system.b)
    NP = system.derived.N * system.derived.M
    state = full_solution[1:(2 * NP)]
    outputs = evaluate_from_state(state, system.derived, params, config)
    return (
        params = params,
        system = system,
        full_solution = full_solution,
        state = state,
        outputs = outputs,
    )
end

"""
    thrust_objective(theta, base_params, config)

Return the scalar penalized objective for the current parameter vector.

The objective is minimized by the optimizer, so higher thrust lowers the
objective, while excess actuator power increases it.
"""
function thrust_objective(theta::AbstractVector{<:Real}, base_params, config::OptimizationConfig)
    eval = evaluate_primal(theta, base_params, config)
    return eval.outputs.objective
end

"""Clamp the optimization variables to their box constraints."""
function clamp_theta(theta, opt_params::OptimizationParams)
    return clamp.(theta, opt_params.lower, opt_params.upper)
end

"""Choose a scale-aware central-difference step for a scalar variable."""
function central_step(value::Real, base_step::Real)
    return base_step * max(1.0, abs(float(value)))
end

"""
    differentiate_assembly(theta, idx, base_params, config)

Approximate the derivatives of the assembled linear system with respect to one
optimization variable using centered finite differences.

For the selected parameter `theta[idx]`, this returns approximations to

- `dA/dtheta_idx`
- `db/dtheta_idx`

These are used in the forward implicit sensitivity equation

`A * s_i = db/dtheta_i - (dA/dtheta_i) * x`

where `s_i = dx/dtheta_i`.
"""
function differentiate_assembly(theta, idx::Int, base_params, config::OptimizationConfig)
    h = central_step(theta[idx], config.sens_step)
    theta_plus = collect(theta)
    theta_minus = collect(theta)
    theta_plus[idx] += h
    theta_minus[idx] -= h

    params_plus = theta_to_params(theta_plus, base_params)
    params_minus = theta_to_params(theta_minus, base_params)
    system_plus = Main.Surferbot.assemble_flexible_system(params_plus)
    system_minus = Main.Surferbot.assemble_flexible_system(params_minus)
    dA = (system_plus.A - system_minus.A) / (2h)
    db = (system_plus.b - system_minus.b) / (2h)
    return dA, db
end

"""
    directional_objective_derivative(state, state_direction, theta, theta_direction, derived, base_params, config)

Differentiate the scalar postprocessed objective along a coupled
state-parameter direction.

This computes the directional derivative of `J(x(theta), theta)` using a
centered difference in the combined direction

- `x -> x ± h * state_direction`
- `theta -> theta ± h * theta_direction`

so it captures both explicit parameter dependence in the postprocessing and the
implicit dependence through the state.
"""
function directional_objective_derivative(state, state_direction, theta, theta_direction, derived, base_params, config::OptimizationConfig)
    h = config.fd_step
    theta_plus = theta .+ h .* theta_direction
    theta_minus = theta .- h .* theta_direction
    params_plus = theta_to_params(theta_plus, base_params)
    params_minus = theta_to_params(theta_minus, base_params)
    derived_plus = Surferbot.derive_params(params_plus)
    derived_minus = Surferbot.derive_params(params_minus)
    out_plus = evaluate_from_state(state .+ h .* state_direction, derived_plus, params_plus, config)
    out_minus = evaluate_from_state(state .- h .* state_direction, derived_minus, params_minus, config)
    return (out_plus.objective - out_minus.objective) / (2h)
end

"""
    objective_and_gradient(theta, base_params, config)

Return the scalar objective, its gradient, and the primal evaluation at
`theta`.

The gradient is computed by forward implicit differentiation. For each
parameter `theta_i`, the state sensitivity `s_i = dx/dtheta_i` is obtained from

`A * s_i = db/dtheta_i - (dA/dtheta_i) * x`

and the scalar objective derivative is then evaluated along that state
direction.

This implementation uses forward sensitivities rather than an adjoint because
the current optimization problem has only two design variables.
"""
function objective_and_gradient(theta::AbstractVector{<:Real}, base_params, config::OptimizationConfig)
    primal = evaluate_primal(theta, base_params, config)
    A = primal.system.A
    z = primal.full_solution
    grad = zeros(Float64, length(theta))

    for idx in eachindex(theta)
        dA, db = differentiate_assembly(theta, idx, base_params, config)
        sensitivity_full = Surferbot.solve_tensor_system(A, db - dA * z)
        sensitivity_state = sensitivity_full[1:length(primal.state)]
        theta_dir = zeros(Float64, length(theta))
        theta_dir[idx] = 1.0
        grad[idx] = directional_objective_derivative(primal.state, sensitivity_state, theta, theta_dir, primal.system.derived, base_params, config)
    end

    return primal.outputs.objective, grad, primal
end

"""
    backtracking_line_search(theta, direction, objective_value, gradient, base_params, config, opt_params)

Simple Armijo backtracking line search used by the local LBFGS driver.
"""
function backtracking_line_search(theta, direction, objective_value, gradient, base_params, config, opt_params)
    step = 1.0
    c1 = 1e-4
    directional = dot(gradient, direction)
    while step > 1e-8
        trial = clamp_theta(theta .+ step .* direction, opt_params)
        trial_objective = thrust_objective(trial, base_params, config)
        if trial_objective <= objective_value + c1 * step * directional
            return step, trial_objective
        end
        step *= 0.5
    end
    return 0.0, objective_value
end

"""
    lbfgs_direction(gradient, s_history, y_history)

Compute the limited-memory BFGS search direction from the stored secant pairs.

This implements the standard two-loop recursion using the history

- `s_k = theta_{k+1} - theta_k`
- `y_k = grad_{k+1} - grad_k`
"""
function lbfgs_direction(gradient, s_history, y_history)
    q = copy(gradient)
    m = length(s_history)
    alpha = zeros(Float64, m)
    rho = zeros(Float64, m)

    for i in m:-1:1
        rho[i] = 1 / dot(y_history[i], s_history[i])
        alpha[i] = rho[i] * dot(s_history[i], q)
        q .-= alpha[i] .* y_history[i]
    end

    if m > 0
        gamma = dot(s_history[end], y_history[end]) / dot(y_history[end], y_history[end])
        r = gamma .* q
    else
        r = copy(q)
    end

    for i in 1:m
        beta = rho[i] * dot(y_history[i], r)
        r .+= s_history[i] .* (alpha[i] - beta)
    end

    return -r
end

"""
    run_thrust_optimization(base_params, opt_params, config; maxiter=20, gtol=1e-6, memory=5)

Run a standalone limited-memory BFGS optimization of the Surferbot design
variables `[xA, logEI]`.

The routine minimizes the penalized objective

`J(theta) = -T(theta) + mu * softplus(P_in(theta) - Pmax)^2`

subject to simple box constraints enforced by clamping after each step.

This driver is intentionally kept outside `Surferbot.jl` so that optimization
remains a separate engineering layer on top of the primal solver.
"""
function run_thrust_optimization(base_params, opt_params::OptimizationParams, config::OptimizationConfig; maxiter::Int=20, gtol::Float64=1e-6, memory::Int=5)
    theta = clamp_theta(copy(opt_params.theta0), opt_params)
    objective, gradient, primal = objective_and_gradient(theta, base_params, config)
    s_history = Vector{Vector{Float64}}()
    y_history = Vector{Vector{Float64}}()
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        if norm(gradient) <= gtol
            converged = true
            break
        end

        direction = lbfgs_direction(gradient, s_history, y_history)
        step, _ = backtracking_line_search(theta, direction, objective, gradient, base_params, config, opt_params)
        if step == 0.0
            break
        end

        theta_next = clamp_theta(theta .+ step .* direction, opt_params)
        objective_next, gradient_next, primal_next = objective_and_gradient(theta_next, base_params, config)

        s = theta_next - theta
        y = gradient_next - gradient
        ys = dot(y, s)
        if ys > 1e-12
            push!(s_history, collect(s))
            push!(y_history, collect(y))
            if length(s_history) > memory
                popfirst!(s_history)
                popfirst!(y_history)
            end
        end

        theta = theta_next
        objective = objective_next
        gradient = gradient_next
        primal = primal_next
    end

    if norm(gradient) <= gtol
        converged = true
    end

    return OptimizationResult(
        collect(theta),
        objective,
        primal.outputs.thrust,
        primal.outputs.power_input,
        collect(gradient),
        iterations,
        converged,
    )
end

end

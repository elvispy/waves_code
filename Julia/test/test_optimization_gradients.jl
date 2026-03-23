using Test
using Surferbot
include(joinpath(@__DIR__, "..", "src", "optimization.jl"))
const SBO = SurferbotOptimization

const GRAD_BASE_PARAMS = FlexibleParams(
    sigma = 0.0,
    rho = 1000.0,
    nu = 1e-6,
    g = 10 * 9.81,
    L_raft = 0.1,
    motor_position = 0.5 * 0.1 / 2,
    d = 0.1 / 2,
    EI = 100 * 3.0e9 * 3e-2 * (9.9e-4)^3 / 12,
    rho_raft = 0.018 * 3.0,
    domain_depth = 0.2,
    n = 41,
    M = 30,
    motor_inertia = 0.13e-3 * 2.5e-3,
    bc = :radiative,
    omega = 2 * pi * 10,
    L_domain = 0.3,
)

function total_fd_gradient(theta, base_params, config; h=1e-6)
    grad = zeros(Float64, length(theta))
    for i in eachindex(theta)
        step = h * max(1.0, abs(theta[i]))
        tp = copy(theta)
        tm = copy(theta)
        tp[i] += step
        tm[i] -= step
        grad[i] = (SBO.thrust_objective(tp, base_params, config) - SBO.thrust_objective(tm, base_params, config)) / (2step)
    end
    grad
end

@testset "optimization gradients" begin
    theta = [0.012, log(GRAD_BASE_PARAMS.EI)]
    config = SBO.OptimizationConfig(Pmax = 1e-5, mu = 10.0, beta = 20.0, fd_step = 1e-6, sens_step = 1e-6)
    objective, grad, primal = SBO.objective_and_gradient(theta, GRAD_BASE_PARAMS, config)
    fd_grad = total_fd_gradient(theta, GRAD_BASE_PARAMS, config)

    @test objective isa Float64
    @test length(grad) == 2
    @test primal.outputs.thrust isa Float64
    @test isapprox(grad[1], fd_grad[1]; atol=1e-5, rtol=5e-3)
    @test isapprox(grad[2], fd_grad[2]; atol=1e-5, rtol=5e-3)
end

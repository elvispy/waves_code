using Test
using Surferbot

@testset "modal decomposition" begin
    @test Surferbot.trapz_weights([0.0, 1.0, 2.0]) ≈ [0.5, 1.0, 0.5]

    roots = Surferbot.freefree_betaL_roots(3)
    @test length(roots) == 3
    @test roots[1] ≈ 4.730040744862704 atol = 1e-8
    @test roots[2] ≈ 7.853204624095838 atol = 1e-8

    xi = collect(range(0.0, 1.0; length=21))
    psi = Surferbot.freefree_mode_shape(xi, 1.0, roots[1])
    @test length(psi) == length(xi)
    @test maximum(abs.(psi)) ≈ 1.0 atol = 1e-12

    Phi = [ones(21) xi]
    Ψ, keep = Surferbot.weighted_mgs(Phi, Surferbot.trapz_weights(xi))
    @test size(Ψ, 2) == 2
    @test all(keep)

    x = collect(range(-0.025, 0.025; length=21))
    contact = trues(length(x))
    η = ComplexF64.(1 .+ 0.2 .* x .+ 0.05 .* sin.(2π .* (x .- minimum(x)) ./ (maximum(x) - minimum(x))))
    pressure = ComplexF64.(0.1 .+ 0.05 .* cos.(π .* x ./ maximum(abs.(x))))
    loads = ComplexF64.(0.02 .* exp.(-(x ./ 0.01).^2))
    args = (
        x_contact = contact,
        L_raft = 0.05,
        d = 0.03,
        EI = 1.0,
        rho_raft = 0.052,
        omega = 2π * 10,
        pressure = pressure,
        loads = loads,
    )

    modal = Surferbot.decompose_raft_freefree_modes(x, η, pressure, loads, args; num_modes=6, verbose=false)
    @test modal isa Surferbot.ModalDecomposition
    @test !isempty(modal.n)
    @test length(modal.q) == length(modal.n) == length(modal.mode_type)
    @test modal.mode_type[1] == "rigid"
    @test any(==("elastic"), modal.mode_type)
    @test isfinite(modal.recon_rel_err)
    @test size(modal.Psi, 1) == length(x)
    @test sum(modal.energy_frac) ≈ 1.0 atol = 1e-10
    @test all(isfinite, modal.energy_frac)
end

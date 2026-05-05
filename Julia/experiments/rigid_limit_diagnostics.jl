"""
rigid_limit_diagnostics.jl

Tests several hypotheses underlying the rigid-limit α prediction:

  H1. Z_ψ off-diagonal decay  — Z_{01} ≈ 0 (symmetric-raft decoupling)
  H2. Gram matrix G ≈ I       — W_0, W_1 orthonormal on the discrete raft grid
  H3. Rigid-body modes first  — β[1] = β[2] = 0
  H4. Phase condition         — (D_rig − Z_{11})/(D_rig − Z_{00}) real
  H5. α = +1 at predicted x_M — |η_L| ≈ 0 in solver
  H6. EI convergence          — |α| → 1 as EI → ∞
  H7. Z_ψ symmetry            — Z_{01} ≈ Z_{10}  (reciprocity)
"""

using Surferbot
using Printf
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

# ── helpers ───────────────────────────────────────────────────────────────────

function rigid_body_denominator(bp, g_eff)
    rho_R = bp.rho_raft isa AbstractVector ? minimum(bp.rho_raft) : float(bp.rho_raft)
    d     = isnothing(bp.d) ? 0.05 : float(bp.d)
    return -rho_R * bp.omega^2 + bp.rho * g_eff * d
end

function load_Z_psi(bp, Fr; output_dir, num_modes=8)
    g_eff     = bp.L_raft * bp.omega^2 / Fr^2
    params_Fr = Surferbot.Sweep.apply_parameter_overrides(bp, (g = g_eff,))
    payload   = ModalPressureMap.load_or_compute_modal_pressure_map(
                    params_Fr; output_dir, num_modes_basis = num_modes)
    return ComplexF64.(payload.Z_psi), collect(Float64.(payload.beta)), g_eff
end

function alpha_at(bp, Fr, EI, xM)
    g_eff = bp.L_raft * bp.omega^2 / Fr^2
    p     = Surferbot.Sweep.apply_parameter_overrides(bp,
                (g = g_eff, EI = EI, motor_position = xM))
    res   = Surferbot.flexible_solver(p)
    m     = Surferbot.Analysis.beam_edge_metrics(res)
    return Surferbot.Analysis.beam_asymmetry(m.eta_left_beam, m.eta_right_beam),
           m.eta_left_beam, m.eta_right_beam
end

function predict_xM(bp, Fr; output_dir, num_modes=8)
    L         = bp.L_raft
    Z, _, g_eff = load_Z_psi(bp, Fr; output_dir, num_modes)
    D         = rigid_body_denominator(bp, g_eff)
    return (L / 6) * (D - Z[2, 2]) / (D - Z[1, 1]), Z
end

# ── H1 / H7 — off-diagonal structure and reciprocity ─────────────────────────

function print_Z_structure(bp; output_dir, num_modes=8)
    println("\n═══ H1 / H7 — Z_ψ off-diagonal decay & reciprocity ═══")
    derived = Surferbot.derive_params(bp)
    Z, _, _ = load_Z_psi(bp, derived.nd_groups.Fr; output_dir, num_modes)
    N       = size(Z, 1)

    diag_rms = sqrt(mean(abs2.(diag(Z))))
    println("  |Z_{nm}| / rms(diag)  [modes 1..$(N)]:")
    for i in 1:N
        print("  ")
        for j in 1:N
            @printf("%6.3f ", abs(Z[i,j]) / (diag_rms + 1e-30))
        end
        println()
    end

    sym_err = norm(Z - Z', Inf) / (norm(Z, Inf) + 1e-30)
    @printf("\n  ‖Z − Zᵀ‖∞ / ‖Z‖∞  = %.2e   (H7: reciprocity)\n", sym_err)
    @printf("  |Z_{01}| / |Z_{00}| = %.2e\n", abs(Z[1,2]) / (abs(Z[1,1]) + 1e-30))
    @printf("  |Z_{01}| / |Z_{11}| = %.2e\n", abs(Z[1,2]) / (abs(Z[2,2]) + 1e-30))
end

# ── H2 — Gram matrix ──────────────────────────────────────────────────────────

function print_gram_error(bp; num_modes=8)
    println("\n═══ H2 — Gram matrix G ≈ I on discrete raft grid ═══")
    derived = Surferbot.derive_params(bp)
    x_raft  = collect(Float64.(derived.x[derived.x_contact] .* derived.L_c))
    basis   = Surferbot.build_raw_freefree_basis(x_raft, bp.L_raft; num_modes)
    G       = Matrix{Float64}(basis.Phi' * (basis.Phi .* basis.w))
    @printf("  N_contact = %d,  ‖G − I‖∞ = %.2e\n", length(x_raft), norm(G - I, Inf))
    println("  First 4×4 block of G:")
    for i in 1:min(4, size(G,1))
        print("  ")
        for j in 1:min(4, size(G,2))
            @printf("%8.4f ", G[i,j])
        end
        println()
    end
end

# ── H3 — rigid-body modes are first ──────────────────────────────────────────

function print_beta_modes(bp; output_dir, num_modes=8)
    println("\n═══ H3 — First two modes are rigid-body (β = 0) ═══")
    derived = Surferbot.derive_params(bp)
    _, beta, _ = load_Z_psi(bp, derived.nd_groups.Fr; output_dir, num_modes)
    for (i, b) in enumerate(beta)
        tag = i <= 2 ? " ← expected rigid-body" : ""
        @printf("  β[%d] = %.4e m⁻¹%s\n", i, b, tag)
    end
end

# ── H4 / H5 — phase condition and α prediction across Fr ─────────────────────

function print_alpha_prediction_table(bp; output_dir, num_modes=8, EI_rigid=1e4)
    println("\n═══ H4 / H5 — Phase condition and α at predicted x_M ═══")
    rho_R = bp.rho_raft isa AbstractVector ? minimum(bp.rho_raft) : float(bp.rho_raft)
    L     = bp.L_raft
    kappa = EI_rigid / (rho_R * L^4 * bp.omega^2)
    @printf("  EI = %.1e  →  κ = %.1e\n\n", EI_rigid, kappa)

    @printf("  %-6s  %-10s  %-10s  %-10s  %-10s  %-8s  %-6s\n",
            "Fr", "Re(xM/L)", "Im/Re(xM)", "|Z01/Z00|", "|η_L|/|η_R|", "α", "H4?")
    println("  " * repeat("-", 70))

    for Fr in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        xM_pred, Z = predict_xM(bp, Fr; output_dir, num_modes)
        rel_imag   = abs(imag(xM_pred)) / (abs(real(xM_pred)) + 1e-30)
        xM_use     = clamp(real(xM_pred), -L/2 + eps(L), L/2 - eps(L))

        alpha, eta_L, eta_R = alpha_at(bp, Fr, EI_rigid, xM_use)
        ratio_LR = abs(eta_L) / (abs(eta_R) + 1e-30)

        @printf("  %-6.2f  %-+10.5f  %-10.2e  %-10.2e  %-10.4f  %-+8.4f  %s\n",
                Fr, real(xM_pred)/L, rel_imag,
                abs(Z[1,2]) / (abs(Z[1,1]) + 1e-30),
                ratio_LR, alpha,
                rel_imag < 0.05 ? "✓" : "✗")
    end
end

# ── H6 — EI convergence ───────────────────────────────────────────────────────

function print_EI_convergence(bp; output_dir, num_modes=8)
    println("\n═══ H6 — |α| → 1 as EI → ∞ ═══")
    derived = Surferbot.derive_params(bp)
    Fr      = derived.nd_groups.Fr
    L       = bp.L_raft
    rho_R   = bp.rho_raft isa AbstractVector ? minimum(bp.rho_raft) : float(bp.rho_raft)

    xM_pred, _ = predict_xM(bp, Fr; output_dir, num_modes)
    xM_use = clamp(real(xM_pred), -L/2 + eps(L), L/2 - eps(L))
    @printf("  Fr = %.3f,  predicted x_M/L = %+.5f\n\n", Fr, xM_use/L)

    @printf("  %-12s  %-10s  %-+8s  %-10s\n", "EI (N·m²)", "κ", "α", "|η_L|/|η_R|")
    println("  " * repeat("-", 46))

    for EI in [1e-5, 1e-3, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e6]
        kappa = EI / (rho_R * L^4 * bp.omega^2)
        alpha, eta_L, eta_R = alpha_at(bp, Fr, EI, xM_use)
        @printf("  %-12.1e  %-10.2e  %-+8.4f  %-10.4f\n",
                EI, kappa, alpha, abs(eta_L) / (abs(eta_R) + 1e-30))
    end
end

# ── main ──────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    bp = Surferbot.Sweep.apply_parameter_overrides(
             Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params, (;))

    print_Z_structure(bp;        output_dir)
    print_gram_error(bp)
    print_beta_modes(bp;         output_dir)
    print_alpha_prediction_table(bp; output_dir)
    print_EI_convergence(bp;     output_dir)
end

main()

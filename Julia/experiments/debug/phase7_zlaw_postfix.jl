"""
phase7_zlaw_postfix.jl

Tests the a-priori law

    q_pred = -(D - Z_dyn) \\ F        with    D[m] = EI β_m^4 - ρ_R ω^2 + d ρ g

across multiple (EI, x_M) test points using ONE Z_dyn matrix.

Z_dyn is the radiation impedance for the *dynamic* pressure only,
projected as d · p_dyn onto the W basis (then transformed to Psi).

Inviscid run (ν=0):
   p_dyn  = -iωρ φ                    (just the velocity-potential pressure)
   p_total = p_dyn  -  ρg η           (also adds hydrostatic on the raft)

Equivalently  Z_total = Z_dyn − d·ρg·I, so the law (D_old−Z_total)q=−f and
(D_correct−Z_dyn)q=−f are algebraically identical.  Z_dyn is the form used
in the appendix.

Z_dyn is computed once (8 prescribed-displacement solves) and then reused.
"""

using Surferbot
using LinearAlgebra
using Statistics
using Printf

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const MPM = Main.PrescribedWnDiagonalImpedance

# ─── Canonical Surferbot params (match CSV grid) ──────────────────────────────
const OMEGA    = 2π * 80.0
const L_RAFT   = 0.05
const D_RAFT   = 0.03
const RHO_RAFT = 0.052
const RHO_W    = 1000.0
const G_GRAV   = 9.81
const SIGMA    = 72.2e-3
const NU       = 0.0
const N_MODES  = 8
const BASE_EI  = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12

# Hydrostatic stiffness in the modal-balance D
const C_HYDRO = D_RAFT * RHO_W * G_GRAV   # = d·ρ·g  [Pa]

# ─── Helper: dynamic pressure on the raft (paper Bernoulli, ν=0 ⇒ p_dyn = -iωρ φ)
function p_dyn_at_raft(params, derived, phi, phi_z)
    contact = collect(Bool.(derived.x_contact))
    Nr = count(contact)
    dx_adim = derived.dx / derived.L_c
    D2r = Matrix(Surferbot.getNonCompactFDmatrix(Nr, 1.0, 2, params.ooa)) / dx_adim^2

    phi_raft = ComplexF64.(vec(phi[end, :])[contact])

    Gamma = derived.nd_groups.Gamma
    Re_   = derived.nd_groups.Re

    # Corrected Bernoulli: -iΓ φ + 2Γ/Re φ_xx (Laplace), no hydrostatic term
    visc = (params.nu == 0.0) ? zeros(ComplexF64, Nr) :
           (2 * Gamma / Re_) .* (D2r * phi_raft)
    p_dyn_adim = -im * Gamma .* phi_raft .+ visc

    p_dyn_dim = p_dyn_adim .* (derived.m_c * derived.L_c / derived.t_c^2) ./ derived.L_c^2
    return p_dyn_dim
end

function radiation_params()
    return Surferbot.FlexibleParams(;
        sigma = SIGMA, rho = RHO_W, omega = OMEGA, nu = NU, g = G_GRAV,
        L_raft = L_RAFT, d = Float64(D_RAFT), EI = 1e-5,
        rho_raft = RHO_RAFT, motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative,
        ooa = 4,
    )
end

function test_params(EI, motor_position)
    return Surferbot.FlexibleParams(;
        sigma = SIGMA, rho = RHO_W, omega = OMEGA, nu = NU, g = G_GRAV,
        L_raft = L_RAFT, motor_position = motor_position, d = Float64(D_RAFT),
        EI = Float64(EI), rho_raft = RHO_RAFT,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, ooa = 4,
    )
end

# ─── Compute Z_dyn (once) via prescribed-displacement radiation solves ───────
function compute_Z_dyn(; verbose=true)
    params_rad = radiation_params()
    assembled  = MPM.assemble_flexible_system(params_rad)
    derived    = assembled.derived
    basis_ctx  = MPM.raw_basis_context(params_rad, derived; num_modes_basis = N_MODES)
    Phi_W      = basis_ctx.basis.Phi
    w          = basis_ctx.weights
    G          = basis_ctx.gram

    Nmod = length(basis_ctx.basis.beta)
    Z_raw = zeros(ComplexF64, Nmod, Nmod)

    for n in 1:Nmod
        target  = MPM.prescribed_target(basis_ctx, params_rad, derived, basis_ctx.basis.n[n])
        reduced = MPM.build_reduced_system(assembled, target.phi_z_target)
        phi, phi_z = MPM.solve_prescribed_mode(assembled, reduced, target.phi_z_target)
        p_dyn = p_dyn_at_raft(params_rad, derived, phi, phi_z)
        rhs = Phi_W' * ((derived.d .* p_dyn) .* w)
        p_modal = G \ rhs
        Z_raw[:, n] = p_modal
    end

    psi_ctx    = MPM.psi_basis_context(basis_ctx)
    transforms = MPM.basis_transforms(basis_ctx, psi_ctx)
    Z_psi = ComplexF64.(transforms.psi_from_raw * Z_raw * transforms.raw_from_psi)

    return (Z_psi = Z_psi, Z_raw = Z_raw, beta = basis_ctx.basis.beta,
            Psi = psi_ctx.Psi)
end

# ─── Predict q at a single (EI, xM) point and compare to coupled solve ───────
function test_point(Z_psi, beta_basis, EI, xM_norm)
    motor_pos = xM_norm * L_RAFT
    params_test = test_params(EI, motor_pos)
    result_c = Surferbot.flexible_solver(params_test)
    modal = Surferbot.decompose_raft_freefree_modes(result_c;
                                                     num_modes = N_MODES,
                                                     include_rigid = true,
                                                     verbose = false)
    q_act = modal.q
    F_psi = modal.F
    Q_psi = modal.Q                                # = projection of d·p_total
    beta  = modal.beta
    Psi_end = modal.Psi[end, :]

    # D = EI β^4 - ρ_R ω² + d ρ g
    D_diag = ComplexF64[EI * beta[m]^4 - RHO_RAFT * OMEGA^2 + C_HYDRO
                        for m in 1:length(beta)]
    A_sys  = Diagonal(D_diag) - Z_psi[1:length(beta), 1:length(beta)]
    q_pred = -A_sys \ F_psi[1:length(beta)]

    rel_err = norm(q_pred .- q_act[1:length(beta)]) / max(norm(q_act), eps())

    # Convert modal.Q (Q_total) to Q_dyn:  Q_dyn = Q_total + d·ρg·q_act
    Q_dyn_act = Q_psi[1:length(beta)] .+ C_HYDRO .* q_act[1:length(beta)]
    Q_dyn_pred = Z_psi[1:length(beta), 1:length(beta)] * q_act[1:length(beta)]
    Q_dyn_rel  = norm(Q_dyn_pred .- Q_dyn_act) / max(norm(Q_dyn_act), eps())

    even_idx = filter(i -> i <= length(q_act), [1, 3, 5, 7])
    odd_idx  = filter(i -> i <= length(q_act), [2, 4, 6, 8])
    S_pred = sum(q_pred[i] * Psi_end[i] for i in even_idx)
    A_pred = sum(q_pred[i] * Psi_end[i] for i in odd_idx)
    S_act  = sum(q_act[i]  * Psi_end[i] for i in even_idx)
    A_act  = sum(q_act[i]  * Psi_end[i] for i in odd_idx)

    return (rel_err = rel_err, Q_dyn_rel = Q_dyn_rel,
            S_pred = S_pred, S_act = S_act,
            A_pred = A_pred, A_act = A_act,
            q_pred = q_pred, q_act = q_act[1:length(beta)],
            beta = beta)
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    println("="^96)
    println("Phase 7 — a-priori law with Z_dyn (corrected Bernoulli, ν=0)")
    println("D[m] = EI β_m^4 − ρ_R ω² + d ρ g       (c_hydro = $(round(C_HYDRO, sigdigits=4)) Pa)")
    println("="^96)

    println("\nComputing Z_dyn from $N_MODES prescribed-displacement solves ...")
    flush(stdout)
    Z = compute_Z_dyn()
    println("Done.")

    println("\nZ_dyn (Psi basis, real part):")
    for row in eachrow(real(Z.Z_psi))
        @printf("  %s\n", join([@sprintf("%9.1f", v) for v in row], "  "))
    end

    # Test points covering the (log10 EI, xM/L) plane
    test_pts = [
        (BASE_EI,        0.12, "canonical (base EI, xM/L=0.12)"),
        (BASE_EI,        0.30, "base EI, mid xM"),
        (BASE_EI,        0.45, "base EI, far xM"),
        (BASE_EI * 10,   0.20, "10x stiffer"),
        (BASE_EI * 100,  0.20, "100x stiffer"),
        (BASE_EI * 0.1,  0.20, "10x more flexible"),
        (BASE_EI * 0.01, 0.20, "100x more flexible"),
        (BASE_EI * 0.1,  0.40, "flexible + far xM"),
    ]

    println("\n" * "─"^96)
    @printf("%-6s %-8s %-7s | %-9s %-10s | %-10s %-10s | %-10s %-10s\n",
            "EI×",  "log10EI", "xM/L",
            "rel_err", "Q_dyn err",
            "|S_pred|", "|S_act|",
            "|A_pred|", "|A_act|")
    println("─"^96)

    for (EI, xM, label) in test_pts
        r = test_point(Z.Z_psi, Z.beta, EI, xM)
        @printf("%-6.2f %-8.2f %-7.3f | %-9.2e %-10.2e | %-10.3e %-10.3e | %-10.3e %-10.3e   %s\n",
                EI / BASE_EI, log10(EI), xM,
                r.rel_err, r.Q_dyn_rel,
                abs(r.S_pred), abs(r.S_act),
                abs(r.A_pred), abs(r.A_act),
                label)
    end

    println("─"^96)
    println("\nLegend:")
    println("  rel_err   = ‖q_pred − q_act‖ / ‖q_act‖")
    println("  Q_dyn err = ‖Z_dyn · q_act − Q_dyn_act‖ / ‖Q_dyn_act‖   (linearity check)")
    println("  Same Z_dyn (computed once at radiation_params) used at every point.")
end

main()

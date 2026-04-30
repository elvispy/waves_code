"""
phase6_zrad_spot_check.jl

Direct test: does q_pred = -(D - Z_rad) \\ F reproduce the actual q from a
coupled fluid solve?

Z_rad is obtained from PrescribedWnDiagonalImpedance (8 radiation solves, EI-
and xM-independent). One full coupled solve is run for each test point.

Modal balance being tested (Psi basis, dynamic pressure only):
  (EI β_m^4 + d ρ g - ρ_raft ω²) q_m = Q_m^dyn - F_m
  ⟹  q = -(D - Z_psi) \\ F

where  D[m] = EI β_m^4 + c_hydro - ρ_raft ω²  (rigid modes: β=0)
       c_hydro = d · ρ_water · g   [Pa]
       Z_psi maps Psi-modal displacement → Psi-modal DYNAMIC pressure
"""

using Surferbot
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const MPM = Main.PrescribedWnDiagonalImpedance

# ─── Physical constants matching the CSV sweep ────────────────────────────────
const OMEGA    = 2π * 80.0
const L_RAFT   = 0.05
const D_RAFT   = 0.03
const RHO_RAFT = 0.052
const RHO_W    = 1000.0
const G_GRAV   = 9.81
const SIGMA    = 72.2e-3
const N_MODES  = 8

# Psi basis is L2-normalised with physical (metre) trapz weights, so
# d ρg ∫ η Psi_m dx = d ρg · q_m exactly — no 1/L_raft scaling.
const C_HYDRO = D_RAFT * RHO_W * G_GRAV

# ─── Test points (log10 EI,  xM/L) ───────────────────────────────────────────
# Three representative cases spanning the EI range of the CSV grid.
const TEST_POINTS = [
    (-5.14, 0.20),   # near base EI, off-centre
    (-4.14, 0.36),   # stiff end of grid
    (-7.14, 0.10),   # flexible end of grid
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

function base_params_for_Z()
    # EI and motor_position are irrelevant for Z_psi (fluid property only)
    return Surferbot.FlexibleParams(;
        sigma         = SIGMA,
        rho           = RHO_W,
        omega         = OMEGA,
        nu            = 0.0,
        g             = G_GRAV,
        L_raft        = L_RAFT,
        d             = Float64(D_RAFT),
        EI            = 1e-5,
        rho_raft      = RHO_RAFT,
        motor_inertia = 0.13e-3 * 2.5e-3,
        bc            = :radiative,
        ooa           = 4,
    )
end

function test_params(log10_EI, xM_norm)
    return Surferbot.FlexibleParams(;
        sigma          = SIGMA,
        rho            = RHO_W,
        omega          = OMEGA,
        nu             = 0.0,
        g              = G_GRAV,
        L_raft         = L_RAFT,
        d              = Float64(D_RAFT),
        EI             = Float64(10^log10_EI),
        rho_raft       = RHO_RAFT,
        motor_position = Float64(xM_norm * L_RAFT),   # xM_over_L * L_raft
        motor_inertia  = 0.13e-3 * 2.5e-3,
        bc             = :radiative,
        ooa            = 4,
    )
end

function build_D(beta, EI)
    # β=0 for rigid modes, β>0 for elastic modes — both get +c_hydro
    return ComplexF64[EI * beta[m]^4 + C_HYDRO - RHO_RAFT * OMEGA^2 for m in eachindex(beta)]
end

function compute_SA(q, Psi_end)
    # even-index modes (0-based n=0,2,4,6 → 1-based 1,3,5,7) → symmetric S
    # odd-index  modes (0-based n=1,3,5,7 → 1-based 2,4,6,8) → antisymmetric A
    S = sum(q[i] * Psi_end[i] for i in [1,3,5,7] if i <= length(q))
    A = sum(q[i] * Psi_end[i] for i in [2,4,6,8] if i <= length(q))
    return S, A
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 80)
    println("Phase 6 — Z_rad spot check")
    println("c_hydro = d·ρ·g = $C_HYDRO Pa  (EI β^4 at β₁ ≈ $(round(7.85/L_RAFT,digits=1)) m⁻¹, EI=1e-5: ",
            round(1e-5 * (7.85/L_RAFT)^4, sigdigits=3), " Pa)")
    println()

    # ── Step 1: load or compute Z_rad ─────────────────────────────────────────
    # load_or_compute_modal_pressure_map caches to JLD2 (with provenance key
    # built from omega, d, sigma, L_raft, num_modes_basis).  On first run this
    # does 8 radiation solves (~1 min); subsequent runs load from cache.
    println("Loading/computing Z_psi via load_or_compute_modal_pressure_map ...")
    flush(stdout)

    params_Z = base_params_for_Z()
    slim = MPM.load_or_compute_modal_pressure_map(
        params_Z;
        num_modes_basis = N_MODES,
    )
    Z_psi = slim.Z_psi                           # 8×8 ComplexF64
    beta_Z = slim.beta                            # wavenumbers [m⁻¹]

    status = slim.cache_status
    println("Cache status: loaded=$(status.loaded)  path=$(status.path)")
    println("Cache key:    $(status.key)")
    println("Off-diagonal ratio (raw W basis): ",
            round.(slim.offdiag_ratio_raw, sigdigits=3))
    println()
    println("Z_psi (real part):")
    for row in eachrow(real(Z_psi))
        @printf("  %s\n", join([@sprintf("%8.1f", v) for v in row], "  "))
    end
    println("Z_psi (imag part):")
    for row in eachrow(imag(Z_psi))
        @printf("  %s\n", join([@sprintf("%8.1f", v) for v in row], "  "))
    end
    println()

    # ── Step 2: loop over test points ─────────────────────────────────────────
    header = @sprintf("%-10s %-6s | %-10s %-10s %-12s | %-8s %-8s | %-8s %-8s",
                      "log10(EI)", "xM/L", "|q_rel_err|", "||q_pred||", "||q_actual||",
                      "|S_pred|", "|S_act|", "|A_pred|", "|A_act|")
    println("-" ^ length(header))
    println(header)
    println("-" ^ length(header))

    for (log10_EI, xM_norm) in TEST_POINTS
        EI = 10^log10_EI

        # ── coupled solve ──
        params = test_params(log10_EI, xM_norm)
        result = Surferbot.flexible_solver(params)

        # ── modal decomposition ──
        modal = Surferbot.decompose_raft_freefree_modes(
            result; num_modes=N_MODES, include_rigid=true, verbose=false,
        )

        q_act = modal.q          # Psi basis
        F_psi = modal.F          # Psi basis loads
        beta  = modal.beta       # wavenumbers
        Psi   = modal.Psi        # Nr × N_MODES
        Psi_end = Psi[end, :]    # mode values at right beam end

        # ── a priori prediction ──
        D_vec  = build_D(beta, EI)
        A_sys  = Diagonal(D_vec) - Z_psi[1:length(beta), 1:length(beta)]
        q_pred = -A_sys \ F_psi[1:length(beta)]

        # ── relative error ──
        rel_err = norm(q_pred .- q_act[1:length(beta)]) / max(norm(q_act), 1e-30)

        # ── S/A amplitudes ──
        S_pred, A_pred = compute_SA(q_pred, Psi_end)
        S_act,  A_act  = compute_SA(q_act,  Psi_end)

        @printf("%-10.2f %-6.2f | %-10.4f %-10.4e %-12.4e | %-8.4f %-8.4f | %-8.4f %-8.4f\n",
                log10_EI, xM_norm,
                rel_err, norm(q_pred), norm(q_act),
                abs(S_pred), abs(S_act),
                abs(A_pred), abs(A_act))

        # ── per-mode breakdown ──
        println("  Per-mode comparison (mode, β[m⁻¹], |q_pred|, |q_act|, angle diff [°]):")
        for m in eachindex(beta)
            ang_diff = rad2deg(angle(q_pred[m]) - angle(q_act[m]))
            ang_diff = mod(ang_diff + 180, 360) - 180
            @printf("    m=%d  β=%7.2f  |q_pred|=%8.4e  |q_act|=%8.4e  Δφ=%7.1f°\n",
                    m, beta[m], abs(q_pred[m]), abs(q_act[m]), ang_diff)
        end
        println()
    end

    println("-" ^ length(header))
    println()
    println("Interpretation:")
    println("  |q_rel_err| < 0.10  →  law approximates well (10% error)")
    println("  |S_pred| ≈ |S_act|  →  α=0 curve prediction is consistent")
    println("  Large rel_err       →  Z=const assumption breaks down or D is wrong")
end

main()

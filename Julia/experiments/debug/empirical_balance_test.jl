"""
empirical_balance_test.jl

ONE coupled fluid solve at canonical Surferbot parameters. Then assemble the
modal-balance ingredients directly from the solve output and check whether

    D q  =  Q_total − f         (no contact-line term)
    D q  =  Q_total − f − K^σ   (with capillary contact-line term)

where:
  q         W-basis displacement coefficients     (from η projected on Φ)
  D         EI β⁴ − ρ_raft ω²                     (per-mode, real diagonal)
  Q_total   W-basis projection of d · p_total
            with p_total = p_dyn − ρg η  (REAL hydrostatic, recomputed
            directly from φ — *not* the imaginary-hydrostatic stored field)
  f         W-basis projection of motor load
  K^σ       d σ ( W_n(L/2)·η_x(L/2⁺) − W_n(−L/2)·η_x(−L/2⁻) )
            evaluated on the FREE-SURFACE SIDE of each beam edge

Both LHS and RHS are reported per mode with a relative residual.
"""

using Surferbot
using LinearAlgebra
using Statistics
using Printf

# ─── Canonical Surferbot parameters (match the CSV grid: d = 0.03) ───────────
const OMEGA    = 2π * 80.0
const L_RAFT   = 0.05
const D_RAFT   = 0.03
const RHO_RAFT = 0.052
const RHO_W    = 1000.0
const G_GRAV   = 9.81
const SIGMA    = 72.2e-3
const NU       = 0.0   # set to 0 per user request — pure inviscid test
const NUM_MODES = 10
const BASE_EI   = 3.0e9 * 3e-2 * (9.9e-4)^3 / 12          # ≈ 7.27e-6
const MOTOR_POS = 0.24 * L_RAFT / 2                         # = 0.006 m

function main()
    println("="^96)
    println("Empirical modal balance test — ONE fluid solve")
    @printf("  ω=%.3f  L=%.3f  d=%.3f  ρ_R=%.4f  EI=%.3e  xM=%.4f m\n",
            OMEGA, L_RAFT, D_RAFT, RHO_RAFT, BASE_EI, MOTOR_POS)
    println("="^96)

    params = FlexibleParams(
        sigma          = SIGMA,
        rho            = RHO_W,
        omega          = OMEGA,
        nu             = NU,
        g              = G_GRAV,
        L_raft         = L_RAFT,
        motor_position = MOTOR_POS,
        d              = D_RAFT,
        EI             = BASE_EI,
        rho_raft       = RHO_RAFT,
    )

    result = flexible_solver(params)
    args   = result.metadata.args
    contact_mask = args.x_contact
    Nr = count(contact_mask)
    @printf("Solve done.  Nr=%d raft points, total grid Nx×Nz = %d×%d\n\n",
            Nr, args.N, args.M)

    # ── Two pressure conventions for comparison ───────────────────────────────
    # (A) "Correct"   : p = −iω ρ φ − ρg η         (e^{iωt}, real hydrostatic)
    # (B) postprocess : p = +iω ρ φ − iρg η         (as stored in args.pressure)
    D2r = Matrix(Surferbot.getNonCompactFDmatrix(Nr, 1.0, 2, params.ooa)) /
          (args.dx / args.L_c)^2
    phi_surf = result.phi[end, :]
    phi_raft = phi_surf[contact_mask] .* (args.t_c / args.L_c^2)
    eta_raft_adim = result.eta[contact_mask] ./ args.L_c

    Gamma = args.nd_groups.Gamma
    Re    = args.nd_groups.Re
    Fr    = args.nd_groups.Fr

    visc = (params.nu == 0.0) ? zeros(ComplexF64, Nr) :
           (2 * Gamma / Re) .* (D2r * phi_raft)

    # (A) physically-derived pressure with e^{iωt} convention
    p_adim_A = -im * Gamma .* phi_raft .- visc .- (Gamma / Fr^2) .* eta_raft_adim
    p_dim_A  = p_adim_A .* (args.m_c * args.L_c / args.t_c^2) ./ args.L_c^2

    # (B) postprocess.jl convention (this is what args.pressure stores)
    p_dim_B  = args.pressure

    # ── Build raw free-free basis (W) on the raft grid ────────────────────────
    x_raft = args.x[contact_mask]
    w      = Surferbot.Modal.trapz_weights(x_raft)
    raw    = Surferbot.Modal.build_raw_freefree_basis(x_raft, args.L_raft;
                                                      num_modes=NUM_MODES,
                                                      include_rigid=true)
    Phi  = raw.Phi                # Nr × NUM_MODES
    beta = raw.beta               # length NUM_MODES
    G    = Phi' * (Phi .* w)      # gram

    eta_raft = result.eta[contact_mask]
    Weta  = eta_raft .* w
    Wdp_A = (args.d .* p_dim_A) .* w
    Wdp_B = (args.d .* p_dim_B) .* w
    Wf    = args.loads .* w

    q_w   = G \ (Phi' * Weta)
    Q_w_A = G \ (Phi' * Wdp_A)    # using "correct" pressure
    Q_w_B = G \ (Phi' * Wdp_B)    # using args.pressure (postprocess)
    F_w   = G \ (Phi' * Wf)

    # ── Capillary contact-line term K^σ (slope from FREE-SURFACE side) ────────
    xL = findfirst(contact_mask)
    xR = findlast(contact_mask)
    D1 = Matrix(Surferbot.getNonCompactFDmatrix(10, 1.0, 1, params.ooa)) / args.dx
    eta_dim = result.eta
    eta_x_L = dot(D1[end, :], eta_dim[(xL - 9):xL])         # slope just LEFT of beam
    eta_x_R = dot(D1[1, :],   eta_dim[xR:(xR + 9)])         # slope just RIGHT of beam

    W_L = Phi[1,   :]              # beam left edge values of each mode
    W_R = Phi[end, :]              # beam right edge values

    K_sigma = params.d * params.sigma .* (W_R .* eta_x_R .- W_L .* eta_x_L)

    # ── Assemble D and check both balances ────────────────────────────────────
    D_diag = ComplexF64[params.EI * beta[m]^4 - params.rho_raft * params.omega^2
                        for m in 1:NUM_MODES]

    LHS = D_diag .* q_w

    RHS_A = Q_w_A .- F_w
    RHS_B = Q_w_B .- F_w
    res_A = LHS .- RHS_A
    res_B = LHS .- RHS_B

    denom_A = max.(abs.(RHS_A), 1e-14 .+ 1e-3 .* mean(abs.(RHS_A)))
    denom_B = max.(abs.(RHS_B), 1e-14 .+ 1e-3 .* mean(abs.(RHS_B)))
    relpc_A = 100 .* abs.(res_A) ./ denom_A
    relpc_B = 100 .* abs.(res_B) ./ denom_B

    println("Per-mode balance: D q  vs  Q_total − f   (two pressure conventions)\n")
    @printf("%-3s %-9s %-11s %-11s %-11s | %-9s | %-11s %-11s | %-9s\n",
            "n", "β[1/m]", "|D q|", "|Q_A−f|", "|Q_B−f|", "rel%(A)",
            "Re(D q)", "Re(Q_A−f)", "rel%(B)")
    println("-"^110)
    for m in 1:NUM_MODES
        @printf("%-3d %-9.3f %-11.3e %-11.3e %-11.3e | %-9.2f | %-11.3e %-11.3e | %-9.2f\n",
                m-1, beta[m],
                abs(LHS[m]), abs(RHS_A[m]), abs(RHS_B[m]),
                relpc_A[m],
                real(LHS[m]), real(RHS_A[m]),
                relpc_B[m])
    end
    println()

    println("Aggregate residuals (norm over all $NUM_MODES modes):")
    @printf("  Convention A (correct, p = −iωρφ − ρgη):\n")
    @printf("    ‖D q − (Q_A − f)‖           = %.3e   (relative: %.2f%%)\n",
            norm(res_A), 100*norm(res_A)/norm(RHS_A))
    @printf("    ‖D q + K^σ − (Q_A − f)‖     = %.3e   (relative: %.2f%%)\n",
            norm(LHS .+ K_sigma .- RHS_A),
            100*norm(LHS .+ K_sigma .- RHS_A)/norm(RHS_A))
    println()
    @printf("  Convention B (postprocess, args.pressure):\n")
    @printf("    ‖D q − (Q_B − f)‖           = %.3e   (relative: %.2f%%)\n",
            norm(res_B), 100*norm(res_B)/norm(RHS_B))
    @printf("    ‖D q + K^σ − (Q_B − f)‖     = %.3e   (relative: %.2f%%)\n",
            norm(LHS .+ K_sigma .- RHS_B),
            100*norm(LHS .+ K_sigma .- RHS_B)/norm(RHS_B))
    println()

    println("Comparing Q_A vs Q_B per mode (real and imag parts, modes 0..3):")
    @printf("%-3s | %-14s %-14s | %-14s %-14s\n",
            "n", "Re(Q_A)", "Re(Q_B)", "Im(Q_A)", "Im(Q_B)")
    for m in 1:4
        @printf("%-3d | %-14.4e %-14.4e | %-14.4e %-14.4e\n",
                m-1, real(Q_w_A[m]), real(Q_w_B[m]),
                imag(Q_w_A[m]), imag(Q_w_B[m]))
    end
end

main()

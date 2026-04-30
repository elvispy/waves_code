"""
coupled_apriori_test.jl

A-priori law (corrected Bernoulli convention):

    q  =  -(D - Z_dyn) \\ F           D[m] = EI β_m^4 - ρ_R ω² + d ρ g

Z_dyn is the radiation impedance for the dynamic pressure only, computed once
from prescribed-displacement solves and cached to JLD2 by
`load_or_compute_modal_pressure_map` (PrescribedWnDiagonalImpedance).
It is independent of EI and motor position.

Phases
------
0  Sanity:  D q ≈ Q_total - F using args.pressure (post-fix postprocess.jl).
1  Load Z_dyn (8 prescribed solves, JLD2-cached on first call).
2  Spot-check predictions against 5 random rows of the CSV.
3  α=0 curve reconstruction:
   - For every EI in the 30k CSV grid:
     · sweep xM/L ∈ [0, 0.5] on a fine grid;
     · find ALL local minima of |S_pred(xM)| with |S|/|A| < cutoff = 0.5;
     · in the CSV slice, find ALL sign changes of α(xM) (linear interp);
   - For each predicted xM, evaluate α at that xM (interp from the slice).
   - Aggregate RMS of |α(xM_pred)| across all predictions.
"""

using Surferbot
using CSV, DataFrames
using LinearAlgebra, Statistics
using Printf, Random

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const MPM = Main.PrescribedWnDiagonalImpedance

# ─── Constants ────────────────────────────────────────────────────────────────
const RATIO_CUTOFF = 0.5
const NUM_MODES    = 8
const N_PHASE2     = 5
const N_XM_SWEEP   = 1000

# Canonical Surferbot params (match CSV grid)
const OMEGA    = 2π * 80.0
const L_RAFT   = 0.05
const D_RAFT   = 0.03
const RHO_RAFT = 0.052
const RHO_W    = 1000.0
const G_GRAV   = 9.81
const SIGMA    = 72.2e-3
const NU       = 0.0          # inviscid for clean Z_dyn

# Hydrostatic stiffness coefficient (kept symbolic — never hardcoded)
const C_HYDRO = D_RAFT * RHO_W * G_GRAV

# Mode parity (1-based Psi indices: rigid-heave=1, rigid-pitch=2, then alternating)
const EVEN_IDX = [1, 3, 5, 7]    # symmetric modes ⇒ S
const ODD_IDX  = [2, 4, 6, 8]    # antisymmetric modes ⇒ A

# ─── Helpers ──────────────────────────────────────────────────────────────────

function base_params()
    return Surferbot.FlexibleParams(;
        sigma = SIGMA, rho = RHO_W, omega = OMEGA, nu = NU, g = G_GRAV,
        L_raft = L_RAFT, d = Float64(D_RAFT), EI = 1e-5,
        rho_raft = RHO_RAFT, motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative,
        ooa = 4,
    )
end

function point_params(EI, motor_position)
    return Surferbot.FlexibleParams(;
        sigma = SIGMA, rho = RHO_W, omega = OMEGA, nu = NU, g = G_GRAV,
        L_raft = L_RAFT, motor_position = motor_position, d = Float64(D_RAFT),
        EI = Float64(EI), rho_raft = RHO_RAFT,
        motor_inertia = 0.13e-3 * 2.5e-3, bc = :radiative, ooa = 4,
    )
end

# Read 8 modal coefficients from CSV row (Psi basis — see header note in old script)
function read_qF(row)
    q = ComplexF64[row[Symbol("q_w$(n)_re")] + im*row[Symbol("q_w$(n)_im")] for n in 0:NUM_MODES-1]
    F = ComplexF64[row[Symbol("F_w$(n)_re")] + im*row[Symbol("F_w$(n)_im")] for n in 0:NUM_MODES-1]
    return q, F
end

# Project the motor force at xM (physical, m) onto the Psi basis on the raft grid.
function F_at_xM(xM_dim, params, derived, x_raft_dim, w, Psi)
    motor_force = params.motor_inertia * params.omega^2
    F_c = derived.F_c
    L_c = derived.L_c
    x_raft_adim = x_raft_dim ./ L_c
    xM_adim = xM_dim / L_c
    loads_adim = motor_force / F_c .*
                 Surferbot.gaussian_load(xM_adim, params.forcing_width, x_raft_adim)
    loads_dim = loads_adim .* (F_c / L_c)        # N/m
    return Psi' * (loads_dim .* w)               # length NUM_MODES, complex (real-valued really)
end

# Local minima of |S| where |S|/|A| < cutoff
function find_S_minima(xM_grid, S_arr, A_arr; cutoff=RATIO_CUTOFF)
    abs_S = abs.(S_arr)
    abs_A = abs.(A_arr)
    minima = Float64[]
    n = length(xM_grid)
    for i in 2:(n-1)
        if abs_S[i] < abs_S[i-1] && abs_S[i] < abs_S[i+1]
            ratio = abs_S[i] / max(abs_A[i], eps())
            if ratio < cutoff
                # parabolic refinement
                y0, y1, y2 = abs_S[i-1], abs_S[i], abs_S[i+1]
                denom = y0 - 2*y1 + y2
                if abs(denom) > 1e-30
                    delta = 0.5 * (y0 - y2) / denom
                    delta = clamp(delta, -1.0, 1.0)
                    dx = xM_grid[i+1] - xM_grid[i]
                    push!(minima, xM_grid[i] + delta * dx)
                else
                    push!(minima, xM_grid[i])
                end
            end
        end
    end
    return minima
end

# Linear-interp sign changes of α on (xM_csv, α_csv); both sorted by xM_csv.
function find_alpha_zeros(xM_csv, alpha_csv)
    zeros_ = Float64[]
    for i in 1:(length(xM_csv)-1)
        a, b = alpha_csv[i], alpha_csv[i+1]
        if a == 0
            push!(zeros_, xM_csv[i])
        elseif a * b < 0
            t = a / (a - b)
            push!(zeros_, xM_csv[i] + t * (xM_csv[i+1] - xM_csv[i]))
        end
    end
    return zeros_
end

# Linear interpolation of α at a query xM (clamped to grid range)
function interp_alpha(xM_csv, alpha_csv, xM_q)
    n = length(xM_csv)
    xM_q <= xM_csv[1] && return alpha_csv[1]
    xM_q >= xM_csv[end] && return alpha_csv[end]
    j = searchsortedfirst(xM_csv, xM_q)
    j = clamp(j, 2, n)
    x0, x1 = xM_csv[j-1], xM_csv[j]
    a0, a1 = alpha_csv[j-1], alpha_csv[j]
    t = (xM_q - x0) / (x1 - x0)
    return a0 + t * (a1 - a0)
end

# Compute S, A endpoint sums for a given q (Psi basis), Psi_end
SA(q, Psi_end) = (
    sum(q[i] * Psi_end[i] for i in EVEN_IDX),
    sum(q[i] * Psi_end[i] for i in ODD_IDX),
)

# ─── Phase 0: balance sanity check ────────────────────────────────────────────
function phase0(df)
    println("\n", "─"^96)
    println("Phase 0 — modal balance D q ≈ Q_total − F  (corrected postprocess)")
    println("─"^96)

    Random.seed!(42)
    rows = df[randperm(nrow(df))[1:N_PHASE2], :]
    rms_rel = Float64[]
    for (k, row) in enumerate(eachrow(rows))
        EI = 10^row.log10_EI
        params = point_params(EI, row.xM_over_L * L_RAFT)
        result = Surferbot.flexible_solver(params)
        modal  = Surferbot.decompose_raft_freefree_modes(result;
                       num_modes=NUM_MODES, include_rigid=true, verbose=false)
        D = ComplexF64[EI*modal.beta[m]^4 - RHO_RAFT*OMEGA^2 + C_HYDRO
                       for m in 1:length(modal.beta)]
        Lhs = D .* modal.q
        Rhs = modal.Q .- modal.F
        rel = norm(Lhs - Rhs) / max(norm(Rhs), eps())
        push!(rms_rel, rel)
        @printf("  row %d:  log10EI=%+.2f  xM/L=%.3f   ‖res‖/‖Q−F‖ = %.4f%%\n",
                k, row.log10_EI, row.xM_over_L, 100*rel)
    end
    @printf("  Mean balance residual: %.4f%%\n", 100*mean(rms_rel))
end

# ─── Phase 1: load Z_dyn ──────────────────────────────────────────────────────
function phase1()
    println("\n", "─"^96)
    println("Phase 1 — load Z_dyn (radiation impedance, 8 prescribed solves; JLD2 cached)")
    println("─"^96)

    params_rad = base_params()
    slim = MPM.load_or_compute_modal_pressure_map(params_rad;
                                                   num_modes_basis = NUM_MODES)
    @printf("  Cache loaded? %s   path=%s\n",
            slim.cache_status.loaded, slim.cache_status.path)
    @printf("  Cache key:    %s\n", slim.cache_status.key)
    return slim
end

# ─── Phase 2: spot-check predictions at random rows ───────────────────────────
function phase2(slim, df)
    println("\n", "─"^96)
    println("Phase 2 — predictions at $N_PHASE2 random rows (Z_dyn + Bernoulli fix)")
    println("─"^96)

    Z_psi = slim.Z_psi
    Random.seed!(7)
    rows = df[randperm(nrow(df))[1:N_PHASE2], :]
    @printf("%-9s %-7s | %-9s %-9s %-9s | %-9s %-9s\n",
            "log10EI", "xM/L", "‖Δq‖/‖q‖", "|ΔS|", "|ΔA|", "|S_act|", "|A_act|")
    println("─"^96)
    rel_errs = Float64[]
    for row in eachrow(rows)
        EI = 10^row.log10_EI
        params = point_params(EI, row.xM_over_L * L_RAFT)
        result = Surferbot.flexible_solver(params)
        modal  = Surferbot.decompose_raft_freefree_modes(result;
                       num_modes=NUM_MODES, include_rigid=true, verbose=false)
        beta = modal.beta
        Psi_end = modal.Psi[end, :]
        D = ComplexF64[EI*beta[m]^4 - RHO_RAFT*OMEGA^2 + C_HYDRO for m in 1:length(beta)]
        Asys = Diagonal(D) - Z_psi[1:length(beta), 1:length(beta)]
        q_pred = -Asys \ modal.F[1:length(beta)]

        rel = norm(q_pred - modal.q[1:length(beta)]) / max(norm(modal.q), eps())
        S_pred, A_pred = SA(q_pred, Psi_end)
        S_act,  A_act  = SA(modal.q, Psi_end)
        push!(rel_errs, rel)
        @printf("%+8.2f  %5.3f  | %.3e  %.3e  %.3e | %.3e %.3e\n",
                row.log10_EI, row.xM_over_L, rel,
                abs(S_pred - S_act), abs(A_pred - A_act),
                abs(S_act), abs(A_act))
    end
    @printf("  Mean ‖Δq‖/‖q‖ = %.4f%%\n", 100*mean(rel_errs))
end

# ─── Phase 3: full xM(EI) reconstruction across the 30k grid ──────────────────
function phase3(slim, df)
    println("\n", "─"^96)
    println("Phase 3 — α=0 curve reconstruction across the 30k grid (no xM filter)")
    println("─"^96)

    Z_psi = slim.Z_psi[1:NUM_MODES, 1:NUM_MODES]

    # Build basis once (independent of EI/xM)
    params_t = base_params()
    derived  = Surferbot.derive_params(params_t)
    contact  = derived.x_contact
    x_raft_dim = derived.x[contact] .* derived.L_c
    w = Surferbot.Modal.trapz_weights(x_raft_dim)
    raw = Surferbot.Modal.build_raw_freefree_basis(x_raft_dim, derived.params.L_raft;
                                                    num_modes=NUM_MODES, include_rigid=true)
    Phi = raw.Phi
    beta = raw.beta
    Psi, _ = Surferbot.Modal.weighted_mgs(Phi, w)
    Psi_end = Psi[end, :]

    # Pre-compute F_psi(xM) on the sweep grid (EI-independent)
    xM_norm_sweep = collect(range(0.0, 0.5; length=N_XM_SWEEP))
    F_grid = zeros(ComplexF64, NUM_MODES, N_XM_SWEEP)
    for (i, xM_norm) in enumerate(xM_norm_sweep)
        F_grid[:, i] = F_at_xM(xM_norm * L_RAFT, params_t, derived, x_raft_dim, w, Psi)
    end

    # Group CSV by EI (fuzzy match on log10_EI float values)
    EI_vals = sort(unique(df.log10_EI))
    n_EI = length(EI_vals)
    println("  $n_EI unique EI slices in CSV.")

    all_alpha_at_pred = Float64[]
    all_dxM            = Float64[]      # |xM_pred − nearest xM_actual|/L  for every prediction
    n_pred_total = 0
    n_actual_total = 0
    n_matched_total = 0
    n_unmatched     = 0          # predicted xM with NO actual α=0 in the slice

    @printf("\n  %-8s | %-3s %-3s | %-10s %-12s %-12s\n",
            "log10EI", "Np", "Na", "RMS|Δx|/L", "med|Δx|/L", "max|α_pred|")
    println("  ", "─"^75)

    for log10_EI in EI_vals
        EI = 10^log10_EI
        D = ComplexF64[EI*beta[m]^4 - RHO_RAFT*OMEGA^2 + C_HYDRO for m in 1:NUM_MODES]
        Asys = Diagonal(D) - Z_psi
        Asys_inv = inv(Asys)

        # Sweep xM
        S_arr = zeros(ComplexF64, N_XM_SWEEP)
        A_arr = zeros(ComplexF64, N_XM_SWEEP)
        for i in 1:N_XM_SWEEP
            q_pred = -(Asys_inv * F_grid[:, i])
            s, a = SA(q_pred, Psi_end)
            S_arr[i] = s
            A_arr[i] = a
        end

        # Predicted xM solutions
        xM_pred = find_S_minima(xM_norm_sweep, S_arr, A_arr)

        # Actual α(xM) sign changes from CSV slice
        slice = sort(filter(:log10_EI => x -> x == log10_EI, df), :xM_over_L)
        xM_csv = Vector{Float64}(slice.xM_over_L)
        alpha_csv = Vector{Float64}(slice.alpha)
        xM_act  = find_alpha_zeros(xM_csv, alpha_csv)

        # Evaluate α(xM_pred) by interp on the CSV slice; also distance to nearest actual zero.
        alpha_at_pred = [interp_alpha(xM_csv, alpha_csv, xp) for xp in xM_pred]
        append!(all_alpha_at_pred, abs.(alpha_at_pred))

        per_pred_dx = Float64[]
        for xp in xM_pred
            if isempty(xM_act)
                n_unmatched += 1
            else
                push!(per_pred_dx, minimum(abs(xp - xa) for xa in xM_act))
            end
        end
        append!(all_dxM, per_pred_dx)

        n_pred_total   += length(xM_pred)
        n_actual_total += length(xM_act)
        n_matched_total += min(length(xM_pred), length(xM_act))

        # Print every 30th slice to keep output readable
        idx_in_list = findfirst(==(log10_EI), EI_vals) - 1
        if idx_in_list % 30 == 0
            rms_dx  = isempty(per_pred_dx) ? NaN : sqrt(mean(abs2.(per_pred_dx)))
            med_dx  = isempty(per_pred_dx) ? NaN : median(per_pred_dx)
            maxabs  = isempty(alpha_at_pred) ? NaN : maximum(abs.(alpha_at_pred))
            @printf("  %+8.3f | %3d %3d | %.4e  %.4e   %.4e\n",
                    log10_EI, length(xM_pred), length(xM_act),
                    rms_dx, med_dx, maxabs)
        end
    end

    println("  ", "─"^75)
    println()
    @printf("  Total predicted xM minima:       %d   (across %d EI slices)\n",
            n_pred_total, n_EI)
    @printf("  Total actual α=0 zero crossings: %d   (data has both S=0 and A=0 branches)\n",
            n_actual_total)
    @printf("  Predictions with no actual zero in their slice: %d\n", n_unmatched)
    println()
    if !isempty(all_dxM)
        @printf("  Distance from each prediction to nearest actual α=0 (Δx/L):\n")
        @printf("    RMS    = %.4e\n", sqrt(mean(abs2.(all_dxM))))
        @printf("    mean   = %.4e\n", mean(all_dxM))
        @printf("    median = %.4e\n", median(all_dxM))
        @printf("    max    = %.4e\n", maximum(all_dxM))
        nclose = count(<=(0.005), all_dxM)
        @printf("    fraction within 0.5%% of L:  %.1f%%   (%d/%d)\n",
                100*nclose/length(all_dxM), nclose, length(all_dxM))
    end
    println()
    if !isempty(all_alpha_at_pred)
        @printf("  |α_actual(xM_pred)|  (sensitive to local steepness of α; informational):\n")
        @printf("    RMS    = %.4e\n", sqrt(mean(abs2.(all_alpha_at_pred))))
        @printf("    median = %.4e\n", median(all_alpha_at_pred))
        @printf("    max    = %.4e\n", maximum(all_alpha_at_pred))
    end
end

# ─── Main ─────────────────────────────────────────────────────────────────────
function main()
    println("="^96)
    println("coupled_apriori_test.jl — a-priori law validation against 30k α dataset")
    println("D[m] = EI β_m^4 - ρ_R ω² + d ρ g    (c_hydro symbolic; cutoff = $RATIO_CUTOFF)")
    println("="^96)

    csv_path = "Julia/output/csv/sweeper_coupled_full_grid.csv"
    df = CSV.read(csv_path, DataFrame)
    @printf("Loaded %d rows from %s\n", nrow(df), csv_path)
    @printf("  unique log10_EI = %d   unique xM_over_L = %d\n",
            length(unique(df.log10_EI)), length(unique(df.xM_over_L)))

    phase0(df)
    slim = phase1()
    phase2(slim, df)
    phase3(slim, df)
end

main()

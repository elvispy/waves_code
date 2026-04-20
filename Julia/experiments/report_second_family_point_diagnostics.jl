using Surferbot
using JLD2
using Printf

# Purpose: produce a readable, cache-backed diagnostic report for one clean
# point on the uncoupled beam-end second-family branch.
#
# The old ad hoc scripts are left in place for provenance:
# - check_empirical_qn_formula.jl
# - check_high_EI_qratio.jl
# - check_empirical_second_family_reduced_model.jl
#
# This script keeps the same tests, but separates preprocessing from the
# human-facing report section and caches the expensive rerun in JLD2.

function nearest_row(path::AbstractString; target_log10_EI::Float64=-3.389, max_abs_alpha::Float64=0.01)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
    col(name) = findfirst(==(name), header)
    idxEI = col("EI")
    idxX = col("xM_over_L")
    idxA = col("alpha_beam")

    best = nothing
    best_score = Inf
    for r in rows
        logEI = log10(parse(Float64, r[idxEI]))
        alpha = abs(parse(Float64, r[idxA]))
        alpha <= max_abs_alpha || continue
        score = abs(logEI - target_log10_EI)
        if score < best_score
            best = r
            best_score = score
        end
    end
    best === nothing && error("No row found near log10(EI)=$target_log10_EI with |alpha_beam| <= $max_abs_alpha.")
    return header, best
end

function cache_key(EI::Float64, xM_over_L::Float64, num_modes::Int)
    return @sprintf("EI_%.12e__xM_%.12e__modes_%d", EI, xM_over_L, num_modes)
end

function load_cached_case(cache_path::AbstractString, key::AbstractString)
    isfile(cache_path) || return nothing
    return jldopen(cache_path, "r") do io
        haskey(io, key) ? read(io, key) : nothing
    end
end

function save_cached_case(cache_path::AbstractString, key::AbstractString, payload)
    jldopen(cache_path, "a+") do io
        io[key] = payload
    end
end

function materialize_flexible_params(base_params)
    names = propertynames(base_params)
    kwargs = NamedTuple{names}(Tuple(getproperty(base_params, name) for name in names))
    return FlexibleParams{Float64}(;
        sigma = Float64(kwargs.sigma),
        rho = Float64(kwargs.rho),
        omega = Float64(kwargs.omega),
        nu = Float64(kwargs.nu),
        g = Float64(kwargs.g),
        L_raft = Float64(kwargs.L_raft),
        motor_position = Float64(kwargs.motor_position),
        d = isnothing(kwargs.d) ? nothing : Float64(kwargs.d),
        EI = Float64(kwargs.EI),
        rho_raft = Float64(kwargs.rho_raft),
        L_domain = isnothing(kwargs.L_domain) ? nothing : Float64(kwargs.L_domain),
        domain_depth = isnothing(kwargs.domain_depth) ? nothing : Float64(kwargs.domain_depth),
        n = isnothing(kwargs.n) ? nothing : Int(kwargs.n),
        M = isnothing(kwargs.M) ? nothing : Int(kwargs.M),
        ooa = kwargs.ooa,
        motor_inertia = Float64(kwargs.motor_inertia),
        motor_force = isnothing(kwargs.motor_force) ? nothing : Float64(kwargs.motor_force),
        forcing_width = Float64(kwargs.forcing_width),
        bc = kwargs.bc,
    )
end

function build_cached_payload(result; num_modes::Int)
    modal = decompose_raft_freefree_modes(result; num_modes=num_modes, verbose=false)
    metrics = beam_edge_metrics(result)
    args = result.metadata.args
    return (
        q = modal.q,
        F = modal.F,
        Q = modal.Q,
        n = modal.n,
        beta = modal.beta,
        Psi = modal.Psi,
        x_raft = modal.x_raft,
        eta_left_beam = metrics.eta_left_beam,
        eta_right_beam = metrics.eta_right_beam,
        S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2,
        A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2,
        motor_position = args.motor_position,
        rho_raft = args.rho_raft,
        omega = args.omega,
        L_raft = args.L_raft,
        EI = args.EI,
    )
end

function load_or_compute_case(
    output_dir::AbstractString,
    sweep_file::AbstractString,
    cache_path::AbstractString,
    EI::Float64,
    xM_over_L::Float64;
    num_modes::Int,
)
    key = cache_key(EI, xM_over_L, num_modes)
    cached = load_cached_case(cache_path, key)
    cached !== nothing && return cached, true

    artifact = load_sweep(joinpath(output_dir, sweep_file))
    base_params = materialize_flexible_params(artifact.base_params)
    params = apply_parameter_overrides(
        base_params,
        (EI=EI, motor_position=xM_over_L * artifact.base_params.L_raft),
    )
    result = flexible_solver(params)
    payload = build_cached_payload(result; num_modes=num_modes)
    save_cached_case(cache_path, key, payload)
    return payload, false
end

function delta_force_projection(F::AbstractVector, Psi_motor::AbstractVector)
    scale_F0 = F[1] / Psi_motor[1]
    return scale_F0 .* Psi_motor, scale_F0
end

function print_force_check(n, F, F_delta)
    println("\nFORCE PROJECTION CHECK")
    println("Checks whether the actual projected load F_n matches the delta-load proxy F_n ∝ W_n(x_M).")
    println("mode  n   F_actual                          F_delta                           relerr")
    for j in eachindex(F)
        relerr = abs(F[j] - F_delta[j]) / max(abs(F[j]), 1e-12)
        @printf(
            "%4d %2d  % .6e%+.6ei   % .6e%+.6ei   %.3e\n",
            j,
            n[j],
            real(F[j]),
            imag(F[j]),
            real(F_delta[j]),
            imag(F_delta[j]),
            relerr,
        )
    end
end

function print_q_check(n, q, q_pred)
    println("\nCOEFFICIENT FORMULA CHECK")
    println("Checks q_n = (Q_n - F_n) / (EI*beta_n^4 - rho_R*omega^2) against the projected q_n.")
    println("mode  n   q_from_projection                  q_from_force_balance              relerr")
    for j in eachindex(q)
        relerr = abs(q[j] - q_pred[j]) / max(abs(q[j]), 1e-12)
        @printf(
            "%4d %2d  % .6e%+.6ei   % .6e%+.6ei   %.3e\n",
            j,
            n[j],
            real(q[j]),
            imag(q[j]),
            real(q_pred[j]),
            imag(q_pred[j]),
            relerr,
        )
    end
end

function print_ratio_check(lhs_ratio, rhs_ratio)
    println("\nTWO-MODE RATIO CHECK")
    println("Checks q2/q0 ≈ (W2(x_M)/W0(x_M)) * (D0/D2).")
    @printf("lhs q2/q0 = % .6e%+.6ei\n", real(lhs_ratio), imag(lhs_ratio))
    @printf("rhs       = % .6e%+.6ei\n", real(rhs_ratio), imag(rhs_ratio))
    @printf("relative error = %.3e\n", abs(lhs_ratio - rhs_ratio) / max(abs(lhs_ratio), 1e-12))
end

function print_branch_equation(branch_lhs)
    println("\nTWO-MODE IMPLICIT BRANCH EQUATION")
    println("Checks -F0*W0(x_M)*W0(end)/D0 - F0*W2(x_M)*W2(end)/D2 ≈ 0.")
    @printf("LHS = % .6e%+.6ei\n", real(branch_lhs), imag(branch_lhs))
end

function print_S_closure(S_full, S_02, S_rest, A_beam)
    println("\nS-CLOSURE CHECK")
    println("Primary KPI is not relative error to S_full, because S_full is supposed to be near zero.")
    println("Report the absolute closure and the residual size relative to |A| and |S_02|.")
    @printf("S_full = % .6e%+.6ei  |S_full|=%.3e\n", real(S_full), imag(S_full), abs(S_full))
    @printf("S_02   = % .6e%+.6ei  |S_02|  =%.3e\n", real(S_02), imag(S_02), abs(S_02))
    @printf("S_rest = % .6e%+.6ei  |S_rest|=%.3e\n", real(S_rest), imag(S_rest), abs(S_rest))
    @printf("|A_beam|                 = %.3e\n", abs(A_beam))
    @printf("|S_full| / |A_beam|      = %.3e\n", abs(S_full) / max(abs(A_beam), 1e-12))
    @printf("|S_02| / |A_beam|        = %.3e\n", abs(S_02) / max(abs(A_beam), 1e-12))
    @printf("|S_rest| / |A_beam|      = %.3e\n", abs(S_rest) / max(abs(A_beam), 1e-12))
    @printf("|S_rest| / |S_02|        = %.3e\n", abs(S_rest) / max(abs(S_02), 1e-12))
    @printf("|S_full - (S_02+S_rest)| = %.3e\n", abs(S_full - (S_02 + S_rest)))
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    cache_file::AbstractString="second_family_point_cache.jld2",
    target_log10_EI::Float64=-3.389,
    max_abs_alpha::Float64=0.01,
    num_modes::Int=8,
)
    header, row = nearest_row(joinpath(output_dir, branch_csv); target_log10_EI=target_log10_EI, max_abs_alpha=max_abs_alpha)
    col(name) = findfirst(==(name), header)
    EI = parse(Float64, row[col("EI")])
    xM_over_L = parse(Float64, row[col("xM_over_L")])
    alpha = parse(Float64, row[col("alpha_beam")])

    cache_path = joinpath(output_dir, cache_file)
    payload, from_cache = load_or_compute_case(output_dir, sweep_file, cache_path, EI, xM_over_L; num_modes=num_modes)

    q = payload.q
    F = payload.F
    Q = payload.Q
    n = payload.n
    beta = payload.beta
    D = payload.EI .* (beta .^ 4) .- payload.rho_raft .* payload.omega^2
    q_pred = (Q .- F) ./ D

    j_motor = findmin(abs.(payload.x_raft .- payload.motor_position))[2]
    Psi_motor = payload.Psi[j_motor, :]
    F_delta, scale_F0 = delta_force_projection(F, Psi_motor)

    i0 = findfirst(==(0), n)
    i2 = findfirst(==(2), n)
    lhs_ratio = q[i2] / q[i0]
    rhs_ratio = (Psi_motor[i2] / Psi_motor[i0]) * (D[i0] / D[i2])

    W0_end = payload.Psi[1, i0]
    W2_end = payload.Psi[1, i2]
    branch_lhs = -(scale_F0 * Psi_motor[i0] * W0_end / D[i0]) - (scale_F0 * Psi_motor[i2] * W2_end / D[i2])

    even_idx = findall(iseven, n)
    S_full = zero(ComplexF64)
    S_02 = zero(ComplexF64)
    for j in even_idx
        contrib = q[j] * payload.Psi[1, j]
        S_full += contrib
        if n[j] == 0 || n[j] == 2
            S_02 += contrib
        end
    end
    S_rest = S_full - S_02

    # =========================
    # Human-facing report block
    # =========================
    cache_label = from_cache ? "hit" : "miss"
    println("Case: EI=$(EI), log10(EI)=$(log10(EI)), x_M/L=$(xM_over_L), alpha_beam=$(alpha), cache=$(cache_label)")
    print_force_check(n, F, F_delta)
    print_q_check(n, q, q_pred)
    print_ratio_check(lhs_ratio, rhs_ratio)
    print_branch_equation(branch_lhs)
    print_S_closure(S_full, S_02, S_rest, payload.A_beam)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

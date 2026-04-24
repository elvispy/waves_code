"""
test_high_EI_8mode_branch_recovery.jl

Tests whether the high-EI alpha=0 branch can be accurately recovered using 
an 8-mode reduced implicit equation. Compares empirical q_n and F_n 
reconstructions against numerical branch data.
"""
using Surferbot
using JLD2
using Statistics
using Printf

# Purpose: test the high-EI second-family branch with the first 8 modal
# coefficients. The script does two things:
# 1. At one clean branch point, rerun once and check that the 8-mode implicit
#    branch equation is approximately zero using empirical q_n and F_n.
# 2. Without using the branch x_M values, solve the reduced implicit equation
#    for x_M(EI) across the high-EI window and compare against the extracted
#    branch from the CSV.
#
# Note: the first 8 modes include n = 0:7, but only the even subset
# {0,2,4,6} contributes to the symmetric endpoint sum S.

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

function selected_rows(path::AbstractString; logEI_min::Float64=-3.65, logEI_max::Float64=-3.35)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
    idxEI = findfirst(==("EI"), header)
    selected = [r for r in rows if logEI_min <= log10(parse(Float64, r[idxEI])) <= logEI_max]
    return header, sort(selected; by = r -> parse(Float64, r[idxEI]))
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

function build_cached_payload(result; num_modes::Int=8)
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

function load_or_compute_case(output_dir::AbstractString, sweep_file::AbstractString, cache_path::AbstractString, EI::Float64, xM_over_L::Float64; num_modes::Int=8)
    key = cache_key(EI, xM_over_L, num_modes)
    cached = load_cached_case(cache_path, key)
    cached !== nothing && return cached, true

    artifact = load_sweep(joinpath(output_dir, sweep_file))
    params = apply_parameter_overrides(
        artifact.base_params,
        (EI=EI, motor_position=xM_over_L * artifact.base_params.L_raft),
    )
    result = flexible_solver(params)
    payload = build_cached_payload(result; num_modes=num_modes)
    save_cached_case(cache_path, key, payload)
    return payload, false
end

function linear_interp(x::AbstractVector{<:Real}, y::AbstractVector, xq::Real)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, length(x) - 1)
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])
end

function first_positive_root(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real}; branch_index::Int=1)
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            xs[i] > 1e-6 && push!(roots, Float64(xs[i]))
        elseif a * b < 0
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            root > 1e-6 && push!(roots, Float64(root))
        end
    end
    unique!(roots)
    length(roots) >= branch_index || return NaN
    return sort(roots)[branch_index]
end

function print_empirical_report(payload)
    q = payload.q
    F = payload.F
    Q = payload.Q
    n = payload.n
    D = payload.EI .* (payload.beta .^ 4) .- payload.rho_raft .* payload.omega^2

    println("\nEMPIRICAL 8-MODE CHECK AT THE ACTUAL BRANCH POINT")
    println("This uses one rerun and checks the implicit branch equation at the known x_M.")

    q_pred = (Q .- F) ./ D
    println("\nModal coefficient formula")
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

    j_motor = findmin(abs.(payload.x_raft .- payload.motor_position))[2]
    Psi_motor = payload.Psi[j_motor, :]
    scale_F0 = F[1] / Psi_motor[1]
    F_delta = scale_F0 .* Psi_motor

    println("\nDelta-load force projection")
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

    even_idx = findall(iseven, n)
    S_q = sum(q[j] * payload.Psi[1, j] for j in even_idx)
    S_force = sum(((Q[j] - F[j]) / D[j]) * payload.Psi[1, j] for j in even_idx)
    S_delta = sum(((-scale_F0 * Psi_motor[j]) / D[j]) * payload.Psi[1, j] for j in even_idx)
    A_beam = payload.A_beam

    println("\nImplicit branch equation at the actual x_M")
    @printf("S_q     = % .6e%+.6ei   |S_q|/|A|     = %.3e\n", real(S_q), imag(S_q), abs(S_q) / max(abs(A_beam), 1e-12))
    @printf("S_force = % .6e%+.6ei   |S_force|/|A| = %.3e\n", real(S_force), imag(S_force), abs(S_force) / max(abs(A_beam), 1e-12))
    @printf("S_delta = % .6e%+.6ei   |S_delta|/|A| = %.3e\n", real(S_delta), imag(S_delta), abs(S_delta) / max(abs(A_beam), 1e-12))
end

function print_branch_recovery_report(payload, header, rows; branch_index::Int=1)
    col(name) = findfirst(==(name), header)
    idxEI = col("EI")
    idxX = col("xM_over_L")
    idxA = col("alpha_beam")

    n = payload.n
    even_idx = findall(iseven, n)
    Dfun(EI, β) = EI * β^4 - payload.rho_raft * payload.omega^2
    xgrid = payload.x_raft ./ payload.L_raft
    W_end = payload.Psi[1, :]

    preds = Float64[]
    truth = Float64[]
    alphas = Float64[]
    EIs = Float64[]

    for r in rows
        EI = parse(Float64, r[idxEI])
        x_true = parse(Float64, r[idxX])
        vals = Float64[]
        for xnorm in xgrid
            s = 0.0
            for j in even_idx
                ψx = linear_interp(xgrid, payload.Psi[:, j], xnorm)
                s += ψx * W_end[j] / Dfun(EI, payload.beta[j])
            end
            push!(vals, s)
        end
        x_pred = first_positive_root(xgrid, vals; branch_index=branch_index)
        if isfinite(x_pred)
            push!(preds, x_pred)
            push!(truth, x_true)
            push!(alphas, parse(Float64, r[idxA]))
            push!(EIs, EI)
        end
    end

    println("\nBRANCH RECOVERY FROM THE 8-MODE IMPLICIT EQUATION")
    println("This solves the reduced equation for x_M(EI) without using the branch x_M values.")
    rmse = sqrt(mean((preds .- truth) .^ 2))
    @printf("Recovered %d / %d points\n", length(preds), length(rows))
    @printf("RMSE in x_M/L = %.3e\n", rmse)
    for i in unique(round.(Int, range(1, length(preds); length=min(6, length(preds)))))
        @printf(
            "EI=%.6e log10EI=%.3f data=%.4f pred=%.4f alpha=%.3e\n",
            EIs[i],
            log10(EIs[i]),
            truth[i],
            preds[i],
            alphas[i],
        )
    end
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    cache_file::AbstractString="second_family_point_cache.jld2",
    target_log10_EI::Float64=-3.389,
    max_abs_alpha::Float64=0.01,
    logEI_min::Float64=-3.65,
    logEI_max::Float64=-3.35,
    num_modes::Int=8,
    branch_index::Int=1,
)
    header, row = nearest_row(joinpath(output_dir, branch_csv); target_log10_EI=target_log10_EI, max_abs_alpha=max_abs_alpha)
    col(name) = findfirst(==(name), header)
    EI = parse(Float64, row[col("EI")])
    xM_over_L = parse(Float64, row[col("xM_over_L")])
    alpha = parse(Float64, row[col("alpha_beam")])

    cache_path = joinpath(output_dir, cache_file)
    payload, from_cache = load_or_compute_case(output_dir, sweep_file, cache_path, EI, xM_over_L; num_modes=num_modes)

    println("Case: EI=$(EI), log10(EI)=$(log10(EI)), x_M/L=$(xM_over_L), alpha_beam=$(alpha), cache=$(from_cache ? "hit" : "miss")")
    print_empirical_report(payload)
    range_header, range_rows = selected_rows(joinpath(output_dir, branch_csv); logEI_min=logEI_min, logEI_max=logEI_max)
    print_branch_recovery_report(payload, range_header, range_rows; branch_index=branch_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

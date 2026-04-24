"""
table_second_family_errors.jl

Batch-calculates relative errors for the delta-load approximation and modal 
force balance equations across a specified number of branch points.
"""
using Surferbot
using JLD2
using DelimitedFiles
using Printf

function linear_interp(x::AbstractVector{<:Real}, y::AbstractVector, xq::Real)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, length(x) - 1)
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])
end

function load_cached_basis(cache_path::AbstractString)
    cache = load(cache_path)
    isempty(cache) && error("Cache file is empty: $cache_path")
    key = first(sort!(collect(keys(cache))))
    payload = cache[key]
    return (
        beta = Float64.(payload.beta),
        Psi = payload.Psi,
        x_raft = Float64.(payload.x_raft),
        L_raft = Float64(payload.L_raft),
        rho_raft = Float64(payload.rho_raft),
        omega = Float64(payload.omega),
    )
end

function parse_csv(path::AbstractString)
    data, header = readdlm(path, ',', header=true)
    names = string.(vec(header))
    rows = [Dict(names[j] => string(data[i, j]) for j in axes(data, 2)) for i in axes(data, 1)]
    return rows
end

function pick_rows(rows, k::Int)
    sorted = sort(rows; by = r -> parse(Float64, r["EI"]))
    idxs = round.(Int, range(1, length(sorted); length=k))
    return sorted[idxs]
end

complex_from_row(row, prefix::String) = parse(Float64, row["$(prefix)_re"]) + im * parse(Float64, row["$(prefix)_im"])

"""
    main(; output_dir=..., branch_csv=..., cache_file=..., k=5)

Report `F_n` delta-load relative errors and `q_n` force-balance relative errors
for `k` branch points spanning the uncoupled refined branch from low to high EI.

This is CSV-based: it uses the stored `q_n`, `Q_n`, `F_n` values and the cached
modal basis only to evaluate `Psi_n(x_M)`.
"""
function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    cache_file::AbstractString="second_family_point_cache.jld2",
    k::Int=5,
)
    rows = pick_rows(parse_csv(joinpath(output_dir, branch_csv)), k)
    basis = load_cached_basis(joinpath(output_dir, cache_file))
    x_norm_grid = basis.x_raft ./ basis.L_raft

    println("POINT SUMMARY")
    println("idx,log10EI,EI,xM_over_L,alpha_beam")
    for (i, row) in enumerate(rows)
        EI = parse(Float64, row["EI"])
        xM = parse(Float64, row["xM_over_L"])
        alpha = parse(Float64, row["alpha_beam"])
        @printf("%d,%.6f,%.6e,%.4f,%.6e\n", i, log10(EI), EI, xM, alpha)
    end

    println("\nFN RELATIVE ERROR TABLE (delta-load vs stored F_n)")
    println("mode,pt1,pt2,pt3,pt4,pt5")
    for target_n in 2:6
        @printf("%d", target_n)
        j = target_n + 1
        for row in rows
            xM = parse(Float64, row["xM_over_L"])
            F0 = complex_from_row(row, "F0")
            F_actual = complex_from_row(row, "F$(target_n)")
            psi_motor = linear_interp(x_norm_grid, basis.Psi[:, j], xM)
            psi0_motor = linear_interp(x_norm_grid, basis.Psi[:, 1], xM)
            F_delta = (F0 / psi0_motor) * psi_motor
            value = abs(F_actual - F_delta) / max(abs(F_actual), 1e-12)
            @printf(",%.6e", value)
        end
        println()
    end

    println("\nQN RELATIVE ERROR TABLE ((Q-F)/D vs stored q_n)")
    println("mode,pt1,pt2,pt3,pt4,pt5")
    for target_n in 2:6
        @printf("%d", target_n)
        j = target_n + 1
        β = basis.beta[j]
        for row in rows
            EI = parse(Float64, row["EI"])
            q_actual = complex_from_row(row, "q$(target_n)")
            Q_actual = complex_from_row(row, "Q$(target_n)")
            F_actual = complex_from_row(row, "F$(target_n)")
            D = EI * β^4 - basis.rho_raft * basis.omega^2
            q_pred = (Q_actual - F_actual) / D
            value = abs(q_actual - q_pred) / max(abs(q_actual), 1e-12)
            @printf(",%.6e", value)
        end
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

"""
compare_high_EI_branch_mode_sets.jl

Evaluates how different subsets of even modes (e.g., {0,2} vs. {0,2,4,6}) 
influence the accuracy of alpha=0 branch predictions in the high-stiffness 
uncoupled regime.
"""
using JLD2
using Printf

# Purpose: compare reduced branch predictions for different even-mode sets
# against the high-EI uncoupled branch data, reusing the cached rerun basis.

function selected_rows(path::AbstractString; logEI_min::Float64=-3.65, logEI_max::Float64=-3.35)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
    idxEI = findfirst(==("EI"), header)
    selected = [r for r in rows if logEI_min <= log10(parse(Float64, r[idxEI])) <= logEI_max]
    return header, sort(selected; by = r -> parse(Float64, r[idxEI]))
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

function linear_interp(x::AbstractVector{<:Real}, y::AbstractVector, xq::Real)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    i = clamp(i, 1, length(x) - 1)
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])
end

function predict_x(payload, EI::Float64, modes::Vector{Int})
    Dfun(EI, β) = EI * β^4 - payload.rho_raft * payload.omega^2
    xgrid = payload.x_raft ./ payload.L_raft
    vals = Float64[]
    for xnorm in xgrid
        s = 0.0
        for m in modes
            j = findfirst(==(m), payload.n)
            j === nothing && continue
            ψx = linear_interp(xgrid, payload.Psi[:, j], xnorm)
            s += ψx * payload.Psi[1, j] / Dfun(EI, payload.beta[j])
        end
        push!(vals, s)
    end
    return first_positive_root(xgrid, vals; branch_index=1)
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    cache_file::AbstractString="second_family_point_cache.jld2",
)
    header, rows = selected_rows(joinpath(output_dir, branch_csv))
    idxEI = findfirst(==("EI"), header)
    idxX = findfirst(==("xM_over_L"), header)

    cache_dict = load(joinpath(output_dir, cache_file))
    entries = collect(keys(cache_dict))
    isempty(entries) && error("No cached payloads found in $cache_file.")
    cache = cache_dict[first(entries)]

    sample_idx = unique(round.(Int, range(1, length(rows); length=min(6, length(rows)))))
    println("EI,data,mode2,mode24,mode246")
    for i in sample_idx
        EI = parse(Float64, rows[i][idxEI])
        x_data = parse(Float64, rows[i][idxX])
        x2 = predict_x(cache, EI, [0, 2])
        x24 = predict_x(cache, EI, [0, 2, 4])
        x246 = predict_x(cache, EI, [0, 2, 4, 6])
        @printf("%.6e,%.4f,%.4f,%.4f,%.4f\n", EI, x_data, x2, x24, x246)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

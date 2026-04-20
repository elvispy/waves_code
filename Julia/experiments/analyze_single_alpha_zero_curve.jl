using Surferbot
using Statistics
using LinearAlgebra
using Base.Threads
using DelimitedFiles
using Plots
using Printf

# Purpose: extract one `alpha = 0` curve from a saved Julia sweep artifact,
# resimulate sampled points on that curve, and dump a detailed CSV for
# mechanism-level analysis.

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function load_sweep_artifact(path::AbstractString)
    artifact = load_sweep(path)
    hasproperty(artifact.parameter_axes, :motor_position) || error("Missing `motor_position` axis in $path.")
    hasproperty(artifact.parameter_axes, :EI) || error("Missing `EI` axis in $path.")
    return artifact
end

function edge_fields(summary, edge_source::Symbol)
    if edge_source == :domain
        return summary.eta_left_domain, summary.eta_right_domain
    elseif edge_source == :beam
        return summary.eta_left_beam, summary.eta_right_beam
    end
    error("edge_source must be :domain or :beam")
end

function build_scalar_fields(left_grid::AbstractMatrix, right_grid::AbstractMatrix)
    alpha_grid = beam_asymmetry.(left_grid, right_grid)
    S_grid = (right_grid .+ left_grid) ./ 2
    A_grid = (right_grid .- left_grid) ./ 2
    sa_ratio_grid = log10.(abs.(S_grid) ./ (abs.(A_grid) .+ eps()))
    return alpha_grid, sa_ratio_grid
end

function collect_zero_crossings(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix)
    asymmetry = beam_asymmetry.(left_grid, right_grid)
    S_grid = (right_grid .+ left_grid) ./ 2
    A_grid = (right_grid .- left_grid) ./ 2
    SA_ratio = log10.(abs.(S_grid) ./ (abs.(A_grid) .+ eps()))

    crossings = NamedTuple[]
    for ie in eachindex(EI_list)
        col = asymmetry[:, ie]
        for im in 1:(length(mp_norm_list) - 1)
            if col[im] * col[im + 1] < 0
                t = col[im] / (col[im] - col[im + 1])
                mp_zero = mp_norm_list[im] + t * (mp_norm_list[im + 1] - mp_norm_list[im])
                sa_zero = SA_ratio[im, ie] + t * (SA_ratio[im + 1, ie] - SA_ratio[im, ie])
                push!(crossings, (EI = EI_list[ie], xM_over_L = mp_zero, sa_ratio = sa_zero))
            end
        end
    end
    return crossings
end

function gaussian_rbf_weights(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; epsilon::Real)
    n = length(x)
    Φ = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        r = abs(float(x[i]) - float(x[j]))
        Φ[i, j] = exp(-(epsilon * r)^2)
    end
    return Φ \ collect(float.(y))
end

function gaussian_rbf_eval(xnodes::AbstractVector{<:Real}, λ::AbstractVector{<:Real}, x::Real; epsilon::Real)
    acc = 0.0
    @inbounds for i in eachindex(xnodes)
        r = abs(float(x) - float(xnodes[i]))
        acc += λ[i] * exp(-(epsilon * r)^2)
    end
    return acc
end

function refine_crossing_rbf(mp_norm_list, values, im::Int; stencil_radius::Int=2)
    left = max(1, im - stencil_radius)
    right = min(length(mp_norm_list), im + 1 + stencil_radius)
    xnodes = collect(float.(mp_norm_list[left:right]))
    ynodes = collect(float.(values[left:right]))

    spacings = diff(xnodes)
    positive = spacings[spacings .> 0]
    h = isempty(positive) ? 1.0 : median(positive)
    epsilon = 1 / max(h, eps())
    λ = gaussian_rbf_weights(xnodes, ynodes; epsilon=epsilon)

    a = float(mp_norm_list[im])
    b = float(mp_norm_list[im + 1])
    fa = gaussian_rbf_eval(xnodes, λ, a; epsilon=epsilon)
    fb = gaussian_rbf_eval(xnodes, λ, b; epsilon=epsilon)

    if !(fa * fb < 0)
        t = values[im] / (values[im] - values[im + 1])
        return a + t * (b - a)
    end

    for _ in 1:80
        mid = (a + b) / 2
        fm = gaussian_rbf_eval(xnodes, λ, mid; epsilon=epsilon)
        if abs(fm) < 1e-10 || abs(b - a) < 1e-8
            return mid
        end
        if fa * fm <= 0
            b = mid
            fb = fm
        else
            a = mid
            fa = fm
        end
    end
    return (a + b) / 2
end

function collect_zero_crossings_refined(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix)
    asymmetry, SA_ratio = build_scalar_fields(left_grid, right_grid)

    crossings = NamedTuple[]
    for ie in eachindex(EI_list)
        col = asymmetry[:, ie]
        for im in 1:(length(mp_norm_list) - 1)
            if col[im] * col[im + 1] < 0
                mp_zero = refine_crossing_rbf(mp_norm_list, col, im)
                t = (mp_zero - mp_norm_list[im]) / (mp_norm_list[im + 1] - mp_norm_list[im])
                sa_zero = SA_ratio[im, ie] + t * (SA_ratio[im + 1, ie] - SA_ratio[im, ie])
                push!(crossings, (EI = EI_list[ie], xM_over_L = mp_zero, sa_ratio = sa_zero))
            end
        end
    end
    return crossings
end

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    lx = isempty(dx) ? 0.05 : max(0.02, 4 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 4 * median(dy))
    sigma_f = max(std(values), 1e-3)
    noise = max(1e-6, 1e-3 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in i:n
        r2 = ((Float64(x[i]) - Float64(x[j])) / lx)^2 + ((Float64(y[i]) - Float64(y[j])) / ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i, j] = kij
        K[j, i] = kij
    end
    @inbounds for i in 1:n
        K[i, i] += noise^2 + 1e-10
    end

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (x = Float64.(x), y = Float64.(y), weights = weights, mean = mean_value, lx = lx, ly = ly, sigma_f2 = sigma_f^2)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    @inbounds for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * (model.sigma_f2 * exp(-0.5 * r2))
    end
    return model.mean + acc
end

function gp_training_points(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix; extra_points=nothing)
    alpha_grid, sa_ratio_grid = build_scalar_fields(left_grid, right_grid)
    logEI_list = log10.(Float64.(EI_list))

    xtrain = Float64[]
    ytrain = Float64[]
    alpha_train = Float64[]
    sa_train = Float64[]
    for ie in eachindex(logEI_list), im in eachindex(mp_norm_list)
        push!(xtrain, Float64(mp_norm_list[im]))
        push!(ytrain, Float64(logEI_list[ie]))
        push!(alpha_train, Float64(alpha_grid[im, ie]))
        push!(sa_train, Float64(sa_ratio_grid[im, ie]))
    end
    if !isnothing(extra_points)
        xextra, yextra, alpha_extra, sa_extra = extra_points
        append!(xtrain, xextra)
        append!(ytrain, yextra)
        append!(alpha_train, alpha_extra)
        append!(sa_train, sa_extra)
    end
    return xtrain, ytrain, alpha_train, sa_train, logEI_list
end

function collect_zero_crossings_gp(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix;
                                   mp_count::Int=201, logEI_count::Int=201, extra_points=nothing)
    xtrain, ytrain, alpha_train, sa_train, logEI_list =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)

    alpha_gp = fit_gp2d(xtrain, ytrain, alpha_train)
    sa_gp = fit_gp2d(xtrain, ytrain, sa_train)

    mp_dense = collect(range(minimum(Float64.(mp_norm_list)), maximum(Float64.(mp_norm_list)); length=mp_count))
    logEI_dense = collect(range(minimum(logEI_list), maximum(logEI_list); length=logEI_count))

    crossings = NamedTuple[]
    alpha_row = zeros(Float64, length(mp_dense))
    sa_row = zeros(Float64, length(mp_dense))
    for logEI in logEI_dense
        @inbounds for i in eachindex(mp_dense)
            alpha_row[i] = predict_gp2d(alpha_gp, mp_dense[i], logEI)
            sa_row[i] = predict_gp2d(sa_gp, mp_dense[i], logEI)
        end
        for im in 1:(length(mp_dense) - 1)
            if alpha_row[im] == 0
                push!(crossings, (EI = 10.0^logEI, xM_over_L = mp_dense[im], sa_ratio = sa_row[im]))
            elseif alpha_row[im] * alpha_row[im + 1] < 0
                t = alpha_row[im] / (alpha_row[im] - alpha_row[im + 1])
                mp_zero = mp_dense[im] + t * (mp_dense[im + 1] - mp_dense[im])
                sa_zero = sa_row[im] + t * (sa_row[im + 1] - sa_row[im])
                push!(crossings, (EI = 10.0^logEI, xM_over_L = mp_zero, sa_ratio = sa_zero))
            end
        end
    end
    return crossings
end

function nontrivial_candidates(candidates; boundary_band::Real)
    isempty(candidates) && return candidates
    sorted = sort(candidates; by = c -> c.xM_over_L)
    return filter(c -> c.xM_over_L > boundary_band, sorted)
end

function target_logEI_values(EI_list; n_sample::Int)
    logEI_list = sort(log10.(Float64.(EI_list)))
    return collect(range(first(logEI_list), last(logEI_list); length=n_sample))
end

function branch_candidates_at_logEI(mp_norm_list, logEI::Real, alpha_gp, sa_gp; mp_count::Int=401, boundary_band::Real=0.0)
    mp_dense = collect(range(minimum(Float64.(mp_norm_list)), maximum(Float64.(mp_norm_list)); length=mp_count))
    alpha_row = [predict_gp2d(alpha_gp, mp, logEI) for mp in mp_dense]
    sa_row = [predict_gp2d(sa_gp, mp, logEI) for mp in mp_dense]

    candidates = NamedTuple[]
    for im in 1:(length(mp_dense) - 1)
        if alpha_row[im] == 0
            push!(candidates, (EI = 10.0^logEI, xM_over_L = mp_dense[im], sa_ratio = sa_row[im], target_log10_EI = logEI))
        elseif alpha_row[im] * alpha_row[im + 1] < 0
            t = alpha_row[im] / (alpha_row[im] - alpha_row[im + 1])
            mp_zero = mp_dense[im] + t * (mp_dense[im + 1] - mp_dense[im])
            sa_zero = sa_row[im] + t * (sa_row[im + 1] - sa_row[im])
            push!(candidates, (EI = 10.0^logEI, xM_over_L = mp_zero, sa_ratio = sa_zero, target_log10_EI = logEI))
        end
    end
    return nontrivial_candidates(candidates; boundary_band=boundary_band)
end

function surrogate_models(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix; extra_points=nothing)
    xtrain, ytrain, alpha_train, sa_train, _ =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
    return fit_gp2d(xtrain, ytrain, alpha_train), fit_gp2d(xtrain, ytrain, sa_train)
end

function logEI_bin(logEI::Real; width::Real=0.03)
    return floor(Int, logEI / width)
end

function solve_counts_by_bin(rows; width::Real=0.03, branch_index::Int, edge_source::Symbol, sweep_file::AbstractString)
    counts = Dict{Int, Int}()
    for row in rows
        if get(row, :branch_index, branch_index) == branch_index &&
           get(row, :edge_source, String(edge_source)) == String(edge_source) &&
           get(row, :sweep_file, sweep_file) == sweep_file
            bin = logEI_bin(get(row, :target_log10_EI, log10(row.EI)); width=width)
            counts[bin] = get(counts, bin, 0) + 1
        end
    end
    return counts
end

function rows_for_sample(rows, sample_index::Int; branch_index::Int, edge_source::Symbol, sweep_file::AbstractString)
    return filter(rows) do row
        row.sample_index == sample_index &&
        get(row, :branch_index, branch_index) == branch_index &&
        get(row, :edge_source, String(edge_source)) == String(edge_source) &&
        get(row, :sweep_file, sweep_file) == sweep_file
    end
end

function best_row_for_sample(rows, sample_index::Int; branch_index::Int, edge_source::Symbol, sweep_file::AbstractString)
    candidates = rows_for_sample(rows, sample_index; branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)
    isempty(candidates) && return nothing
    return argmin(r -> abs(r.alpha), candidates)
end

function argmin(f, xs)
    best = first(xs)
    bestv = f(best)
    for x in Iterators.drop(xs, 1)
        v = f(x)
        if v < bestv
            best = x
            bestv = v
        end
    end
    return best
end

function iterative_sample_candidates(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix;
                                     cached_rows, edge_source::Symbol, branch_index::Int, n_sample::Int,
                                     max_iterations::Int=5, alpha_accept_tol::Real=5e-3,
                                     logEI_bin_width::Real=0.03, max_solves_per_bin::Int=5)
    target_logs = target_logEI_values(EI_list; n_sample=n_sample)
    boundary_band = length(mp_norm_list) >= 2 ? (maximum(mp_norm_list) - minimum(mp_norm_list)) / (201 - 1) : 0.0
    proposed = Dict{Int, NamedTuple}()
    accepted = Dict{Int, NamedTuple}()
    counts = solve_counts_by_bin(cached_rows; width=logEI_bin_width, branch_index=branch_index, edge_source=edge_source, sweep_file="")

    working_rows = copy(cached_rows)
    for iteration in 1:max_iterations
        extra_points = isempty(working_rows) ? nothing : cached_training_points(working_rows, edge_source)
        alpha_gp, sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
        for (sample_index, logEI) in enumerate(target_logs)
            candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=boundary_band)
            if length(candidates) >= branch_index
                cand = candidates[branch_index]
                proposed[sample_index] = (; cand..., curve_point_index = sample_index, sample_index = sample_index, iteration = iteration)
            end
        end
    end

    for sample_index in 1:n_sample
        haskey(proposed, sample_index) || continue
        accepted[sample_index] = proposed[sample_index]
    end
    return [accepted[i] for i in sort(collect(keys(accepted)))]
end

function write_alpha_overlay_plot(path::AbstractString, mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix,
                                  sampled, rows; extra_points=nothing, edge_source::Symbol=:domain,
                                  mp_count::Int=241, logEI_count::Int=241, bad_alpha_tol::Real=5e-2)
    xtrain, ytrain, alpha_train, _, logEI_list =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
    alpha_gp = fit_gp2d(xtrain, ytrain, alpha_train)

    mp_dense = collect(range(minimum(Float64.(mp_norm_list)), maximum(Float64.(mp_norm_list)); length=mp_count))
    logEI_dense = collect(range(minimum(logEI_list), maximum(logEI_list); length=logEI_count))
    alpha_pred = Matrix{Float64}(undef, length(mp_dense), length(logEI_dense))
    @inbounds for j in eachindex(logEI_dense), i in eachindex(mp_dense)
        alpha_pred[i, j] = predict_gp2d(alpha_gp, mp_dense[i], logEI_dense[j])
    end

    plt = contourf(
        logEI_dense,
        mp_dense,
        alpha_pred;
        levels=31,
        c=:balance,
        xlabel="log10(EI)",
        ylabel="x_M / L",
        colorbar_title="alpha ($(edge_source))",
        title="Predicted alpha field with traced branch",
        size=(1000, 700),
    )
    contour!(
        plt,
        logEI_dense,
        mp_dense,
        alpha_pred;
        levels=[0.0],
        color=:white,
        linewidth=2,
        label="GP alpha=0",
    )

    sampled_x = [p.xM_over_L for p in sampled]
    sampled_y = log10.([p.EI for p in sampled])
    scatter!(
        plt,
        sampled_y,
        sampled_x;
        color=:black,
        markersize=3,
        markerstrokewidth=0,
        label="predicted samples",
    )

    bad_rows = filter(r -> abs(r.alpha) > bad_alpha_tol, rows)
    if !isempty(bad_rows)
        scatter!(
            plt,
            log10.([r.EI for r in bad_rows]),
            [r.xM_over_L for r in bad_rows];
            markershape=:xcross,
            markersize=6,
            markercolor=:yellow,
            markerstrokecolor=:black,
            label="|alpha| > $(bad_alpha_tol)",
        )
    end

    savefig(plt, path)
    return path
end

function group_crossings_by_EI(crossings)
    by_EI = Dict{Float64, Vector{NamedTuple}}()
    for cross in crossings
        push!(get!(by_EI, cross.EI, NamedTuple[]), cross)
    end
    for vals in values(by_EI)
        sort!(vals; by=c -> c.xM_over_L)
    end
    return by_EI
end

function predict_next_x(curve_points, next_logEI)
    if length(curve_points) == 1
        return curve_points[end].xM_over_L
    end
    p1 = curve_points[end - 1]
    p2 = curve_points[end]
    y1 = log10(p1.EI)
    y2 = log10(p2.EI)
    if isapprox(y2, y1; atol=1e-14)
        return p2.xM_over_L
    end
    slope = (p2.xM_over_L - p1.xM_over_L) / (y2 - y1)
    return p2.xM_over_L + slope * (next_logEI - y2)
end

function track_branch(crossings; branch_index::Int=1, jump_factor::Real=4.0, min_jump_tol::Real=0.05, boundary_band::Real=0.0)
    branch_index >= 1 || error("branch_index must be >= 1")
    by_EI = group_crossings_by_EI(crossings)
    EI_desc = sort(collect(keys(by_EI)); rev=true)
    isempty(EI_desc) && return NamedTuple[]

    curve_points = NamedTuple[]
    accepted_steps = Float64[]

    seed_candidates = NamedTuple[]
    seed_idx = nothing
    for (i, EI) in pairs(EI_desc)
        candidates = nontrivial_candidates(by_EI[EI]; boundary_band=boundary_band)
        if length(candidates) >= branch_index
            seed_candidates = candidates
            seed_idx = i
            break
        end
    end
    isempty(seed_candidates) && return NamedTuple[]
    seed = seed_candidates[branch_index]
    push!(curve_points, seed)

    for EI in EI_desc[(seed_idx + 1):end]
        candidates = nontrivial_candidates(by_EI[EI]; boundary_band=boundary_band)
        length(candidates) < branch_index && continue
        chosen = candidates[branch_index]

        y = log10(EI)
        x_pred = predict_next_x(curve_points, y)

        jump = abs(chosen.xM_over_L - curve_points[end].xM_over_L)
        local_scale = isempty(accepted_steps) ? min_jump_tol : max(min_jump_tol, jump_factor * median(accepted_steps))

        if jump <= local_scale || abs(chosen.xM_over_L - x_pred) <= local_scale
            push!(curve_points, chosen)
            push!(accepted_steps, jump)
        end
    end

    return sort(curve_points; by = p -> p.EI)
end

function sample_curve_points(curve_points; n_sample::Int)
    isempty(curve_points) && error("No curve points were available for sampling.")
    n_sample >= 1 || error("n_sample must be positive.")
    length(curve_points) == 1 && return [(; curve_points[1]..., curve_point_index = 1)]

    ys = log10.([p.EI for p in curve_points])
    xs = [p.xM_over_L for p in curve_points]
    seglen = [hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in 1:(length(curve_points) - 1)]
    s = vcat(0.0, cumsum(seglen))
    total = s[end]

    if total <= 0
        return [(; curve_points[1]..., curve_point_index = 1) for _ in 1:n_sample]
    end

    targets = collect(range(0.0, total; length=n_sample))
    sampled = NamedTuple[]
    j = 1
    for t in targets
        while j < length(s) - 1 && s[j + 1] < t
            j += 1
        end
        s0 = s[j]
        s1 = s[j + 1]
        p0 = curve_points[j]
        p1 = curve_points[j + 1]
        τ = isapprox(s1, s0; atol=1e-14) ? 0.0 : (t - s0) / (s1 - s0)
        EI = 10.0^((1 - τ) * log10(p0.EI) + τ * log10(p1.EI))
        xM_over_L = (1 - τ) * p0.xM_over_L + τ * p1.xM_over_L
        sa_ratio = (1 - τ) * p0.sa_ratio + τ * p1.sa_ratio
        push!(sampled, (EI = EI, xM_over_L = xM_over_L, sa_ratio = sa_ratio, curve_point_index = j))
    end
    return sampled
end

function tail_flat_ratio(result)
    left_count = max(1, ceil(Int, 0.05 * length(result.eta)))
    tail = abs.(result.eta[1:left_count])
    return std(tail) / max(eps(), mean(tail))
end

function format_complex(z)
    return string(real(z), ",", imag(z), ",", abs(z), ",", rad2deg(angle(z)))
end

function parse_complex_columns(row::AbstractDict{<:AbstractString, <:AbstractString}, prefix::AbstractString)
    return ComplexF64(
        parse(Float64, row["$(prefix)_re"]),
        parse(Float64, row["$(prefix)_im"]),
    )
end


function load_cached_curve_rows(path::AbstractString, n_modes::Int)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    header = split(lines[1], ",")
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ",")
        length(values) == length(header) || continue
        row = Dict(header[i] => values[i] for i in eachindex(header))
        modal = (
            q = ComplexF64[parse_complex_columns(row, "q$(j)") for j in 0:(n_modes - 1)],
            Q = ComplexF64[parse_complex_columns(row, "Q$(j)") for j in 0:(n_modes - 1)],
            F = ComplexF64[parse_complex_columns(row, "F$(j)") for j in 0:(n_modes - 1)],
            balance_residual = ComplexF64[parse_complex_columns(row, "residual$(j)") for j in 0:(n_modes - 1)],
            energy_frac = Float64[parse(Float64, row["energy_frac$(j)"]) for j in 0:(n_modes - 1)],
            n = Int[parse(Int, row["mode_index$(j)"]) for j in 0:(n_modes - 1)],
            mode_type = String[row["mode_type$(j)"] for j in 0:(n_modes - 1)],
        )
        push!(rows, (
            branch_index = haskey(row, "branch_index") ? parse(Int, row["branch_index"]) : 1,
            edge_source = get(row, "edge_source", ""),
            sweep_file = get(row, "sweep_file", ""),
            iteration = haskey(row, "iteration") ? parse(Int, row["iteration"]) : 1,
            target_log10_EI = haskey(row, "target_log10_EI") ? parse(Float64, row["target_log10_EI"]) : log10(parse(Float64, row["EI"])),
            sample_index = parse(Int, row["sample_index"]),
            curve_point_index = parse(Int, row["curve_point_index"]),
            EI = parse(Float64, row["EI"]),
            xM_over_L = parse(Float64, row["xM_over_L"]),
            motor_position = parse(Float64, row["motor_position"]),
            omega = parse(Float64, row["omega"]),
            U = parse(Float64, row["U"]),
            power = parse(Float64, row["power"]),
            power_input = parse(Float64, row["power_input"]),
            thrust = parse(Float64, row["thrust"]),
            tail_flat_ratio = parse(Float64, row["tail_flat_ratio"]),
            eta_left_domain = parse_complex_columns(row, "eta_left_domain"),
            eta_right_domain = parse_complex_columns(row, "eta_right_domain"),
            eta_left_beam = parse_complex_columns(row, "eta_left_beam"),
            eta_right_beam = parse_complex_columns(row, "eta_right_beam"),
            alpha = parse(Float64, row["alpha"]),
            alpha_domain = parse(Float64, row["alpha_domain"]),
            alpha_beam = parse(Float64, row["alpha_beam"]),
            sa_ratio_domain = parse(Float64, row["sa_ratio_domain"]),
            sa_ratio_beam = parse(Float64, row["sa_ratio_beam"]),
            modal = modal,
        ))
    end
    return rows
end

function cached_training_points(rows, edge_source::Symbol)
    x = Float64[]
    y = Float64[]
    alpha = Float64[]
    sa = Float64[]
    for row in rows
        push!(x, row.xM_over_L)
        push!(y, log10(row.EI))
        if edge_source == :domain
            push!(alpha, row.alpha_domain)
            push!(sa, row.sa_ratio_domain)
        else
            push!(alpha, row.alpha_beam)
            push!(sa, row.sa_ratio_beam)
        end
    end
    return x, y, alpha, sa
end

function compatible_row(row; edge_source::Symbol, sweep_file::AbstractString)
    row_edge = get(row, :edge_source, "")
    row_sweep = get(row, :sweep_file, "")
    return (isempty(row_edge) || row_edge == String(edge_source)) &&
           (isempty(row_sweep) || row_sweep == sweep_file)
end

function training_rows(rows; edge_source::Symbol, sweep_file::AbstractString)
    return filter(row -> compatible_row(row; edge_source=edge_source, sweep_file=sweep_file), rows)
end

function reusable_rows(rows; edge_source::Symbol, sweep_file::AbstractString, branch_index::Int)
    return filter(rows) do row
        compatible_row(row; edge_source=edge_source, sweep_file=sweep_file) &&
        get(row, :branch_index, branch_index) == branch_index
    end
end

function find_reusable_cached_row(rows, point; x_tol::Real=0.01, logEI_tol::Real=0.03, alpha_accept_tol::Real=5e-2)
    isempty(rows) && return nothing
    target_y = log10(point.EI)
    best = nothing
    best_score = Inf
    for row in rows
        dx = abs(row.xM_over_L - point.xM_over_L)
        dy = abs(log10(row.EI) - target_y)
        if dx <= x_tol && dy <= logEI_tol && abs(row.alpha) <= alpha_accept_tol
            score = dx / x_tol + dy / logEI_tol
            if score < best_score
                best = row
                best_score = score
            end
        end
    end
    return best
end

function unique_row_key(row)
    return (
        get(row, :branch_index, 1),
        get(row, :edge_source, ""),
        get(row, :sweep_file, ""),
        get(row, :iteration, 1),
        row.sample_index,
        round(get(row, :target_log10_EI, log10(row.EI)); digits=12),
        round(row.EI; digits=12),
        round(row.xM_over_L; digits=12),
    )
end

function merge_output_rows(existing_rows, new_rows)
    merged = copy(existing_rows)
    seen = Set(unique_row_key(row) for row in existing_rows)
    for row in new_rows
        key = unique_row_key(row)
        if !(key in seen)
            push!(merged, row)
            push!(seen, key)
        end
    end
    return merged
end

function write_curve_csv(path::AbstractString, rows, n_modes::Int)
    open(path, "w") do io
        header = [
            "branch_index",
            "edge_source",
            "sweep_file",
            "iteration",
            "target_log10_EI",
            "sample_index",
            "curve_point_index",
            "EI",
            "log10_EI",
            "xM_over_L",
            "motor_position",
            "omega",
            "U",
            "power",
            "power_input",
            "thrust",
            "tail_flat_ratio",
            "eta_left_domain_re", "eta_left_domain_im", "eta_left_domain_abs", "eta_left_domain_phase_deg",
            "eta_right_domain_re", "eta_right_domain_im", "eta_right_domain_abs", "eta_right_domain_phase_deg",
            "eta_left_beam_re", "eta_left_beam_im", "eta_left_beam_abs", "eta_left_beam_phase_deg",
            "eta_right_beam_re", "eta_right_beam_im", "eta_right_beam_abs", "eta_right_beam_phase_deg",
            "alpha",
            "alpha_domain",
            "alpha_beam",
            "sa_ratio_domain",
            "sa_ratio_beam",
        ]
        for j in 0:(n_modes - 1)
            append!(header, [
                "q$(j)_re", "q$(j)_im", "q$(j)_abs", "q$(j)_phase_deg",
                "Q$(j)_re", "Q$(j)_im", "Q$(j)_abs", "q$(j)_phase_deg",
                "F$(j)_re", "F$(j)_im", "F$(j)_abs", "F$(j)_phase_deg",
                "residual$(j)_re", "residual$(j)_im", "residual$(j)_abs", "residual$(j)_phase_deg",
                "energy_frac$(j)",
                "mode_index$(j)",
                "mode_type$(j)",
            ])
        end
        println(io, join(header, ","))

        for row in rows
            # Compute T transformation from Psi -> W for all terms to be consistent
            w_weights = Surferbot.trapz_weights(row.modal.x_raft)
            G_mat = row.modal.Phi' * (row.modal.Phi .* w_weights)
            B_mat = row.modal.Phi' * (row.modal.Psi .* w_weights)
            T_psi_to_w = G_mat \ B_mat
            
            # Map coefficients to W basis
            q_w = row.modal.q_w
            Q_w = T_psi_to_w * row.modal.Q
            F_w = T_psi_to_w * row.modal.F
            R_w = T_psi_to_w * row.modal.balance_residual

            fields = String[
                string(get(row, :branch_index, 1)),
                string(get(row, :edge_source, "")),
                string(get(row, :sweep_file, "")),
                string(get(row, :iteration, 1)),
                string(get(row, :target_log10_EI, log10(row.EI))),
                string(row.sample_index),
                string(row.curve_point_index),
                string(row.EI),
                string(log10(row.EI)),
                string(row.xM_over_L),
                string(row.motor_position),
                string(row.omega),
                string(row.U),
                string(row.power),
                string(row.power_input),
                string(row.thrust),
                string(row.tail_flat_ratio),
            ]
            append!(fields, split(format_complex(row.eta_left_domain), ","))
            append!(fields, split(format_complex(row.eta_right_domain), ","))
            append!(fields, split(format_complex(row.eta_left_beam), ","))
            append!(fields, split(format_complex(row.eta_right_beam), ","))
            append!(fields, [
                string(row.alpha),
                string(row.alpha_domain),
                string(row.alpha_beam),
                string(row.sa_ratio_domain),
                string(row.sa_ratio_beam),
            ])
            for j in 1:n_modes
                append!(fields, split(format_complex(q_w[j]), ","))
                append!(fields, split(format_complex(Q_w[j]), ","))
                append!(fields, split(format_complex(F_w[j]), ","))
                append!(fields, split(format_complex(R_w[j]), ","))
                append!(fields, [
                    string(row.modal.energy_frac[j]),
                    string(row.modal.n[j]),
                    row.modal.mode_type[j],
                ])
            end
            println(io, join(fields, ","))
        end
    end
end

function beam_csv_path(path::AbstractString)
    stem, ext = splitext(path)
    return stem * "_beam" * ext
end

function default_curve_basename(sweep_file::AbstractString)
    return occursin("uncoupled", lowercase(sweep_file)) ?
        "single_alpha_zero_curve_details_uncoupled" :
        "single_alpha_zero_curve_details"
end

default_output_file(sweep_file::AbstractString) = default_curve_basename(sweep_file) * "_refined.csv"
default_cache_file(sweep_file::AbstractString) = default_curve_basename(sweep_file) * ".csv"

function write_beam_curve_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "branch_index,edge_source,sweep_file,iteration,target_log10_EI,sample_index,curve_point_index,EI,log10_EI,xM_over_L,motor_position,omega,U,power,power_input,thrust,tail_flat_ratio,eta_left_beam_re,eta_left_beam_im,eta_left_beam_abs,eta_left_beam_phase_deg,eta_right_beam_re,eta_right_beam_im,eta_right_beam_abs,eta_right_beam_phase_deg,alpha_beam,sa_ratio_beam")
        for row in rows
            fields = String[
                string(get(row, :branch_index, 1)),
                string(get(row, :edge_source, "")),
                string(get(row, :sweep_file, "")),
                string(get(row, :iteration, 1)),
                string(get(row, :target_log10_EI, log10(row.EI))),
                string(row.sample_index),
                string(row.curve_point_index),
                string(row.EI),
                string(log10(row.EI)),
                string(row.xM_over_L),
                string(row.motor_position),
                string(row.omega),
                string(row.U),
                string(row.power),
                string(row.power_input),
                string(row.thrust),
                string(row.tail_flat_ratio),
            ]
            append!(fields, split(format_complex(row.eta_left_beam), ","))
            append!(fields, split(format_complex(row.eta_right_beam), ","))
            append!(fields, [
                string(row.alpha_beam),
                string(row.sa_ratio_beam),
            ])
            println(io, join(fields, ","))
        end
    end
end

function build_row(sample_index, point, artifact, edge_source::Symbol, n_modes::Int; branch_index::Int, sweep_file::AbstractString, iteration::Int)
    params = apply_parameter_overrides(
        artifact.base_params,
        (motor_position = point.xM_over_L * artifact.base_params.L_raft, EI = point.EI),
    )
    result = flexible_solver(params)
    modal = decompose_raft_freefree_modes(result; num_modes=n_modes, verbose=false)
    metrics = beam_edge_metrics(result)
    S_domain = (metrics.eta_right_domain + metrics.eta_left_domain) / 2
    A_domain = (metrics.eta_right_domain - metrics.eta_left_domain) / 2
    S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
    A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
    alpha_selected = edge_source == :domain ?
        beam_asymmetry(metrics.eta_left_domain, metrics.eta_right_domain) :
        beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)

    return (
        branch_index = branch_index,
        edge_source = String(edge_source),
        sweep_file = sweep_file,
        iteration = iteration,
        target_log10_EI = get(point, :target_log10_EI, log10(point.EI)),
        sample_index = sample_index,
        curve_point_index = get(point, :curve_point_index, 0),
        EI = point.EI,
        xM_over_L = point.xM_over_L,
        motor_position = params.motor_position,
        omega = params.omega,
        U = result.U,
        power = result.power,
        power_input = -result.power,
        thrust = result.thrust,
        tail_flat_ratio = tail_flat_ratio(result),
        eta_left_domain = metrics.eta_left_domain,
        eta_right_domain = metrics.eta_right_domain,
        eta_left_beam = metrics.eta_left_beam,
        eta_right_beam = metrics.eta_right_beam,
        alpha = alpha_selected,
        alpha_domain = beam_asymmetry(metrics.eta_left_domain, metrics.eta_right_domain),
        alpha_beam = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam),
        sa_ratio_domain = log10(abs(S_domain) / (abs(A_domain) + eps())),
        sa_ratio_beam = log10(abs(S_beam) / (abs(A_beam) + eps())),
        modal = modal,
    )
end

"""
    main(data_dir=joinpath(@__DIR__, "..", "output");
         sweep_file="sweep_motor_position_EI_coupled_from_matlab.jld2",
         edge_source=:beam,
         branch_index=1,
         n_sample=100,
         n_modes=8,
         parallel=(nthreads() > 1),
         output_file=default_output_file(sweep_file))

Extract one `alpha = 0` curve from a saved sweep artifact and write a detailed
per-sample CSV after rerunning the Julia solver and modal decomposition. The
curve is extracted from a 2D GP surrogate in `(x_M/L, log10(EI))`, then
tracked from high `EI` to low `EI`. By default, the script extracts the first
nontrivial branch. The boundary-connected root at `x_M = 0` is treated as
branch `0` and discarded from the physical branch numbering.

Inputs:
- `data_dir`: directory containing the input sweep artifact and receiving the CSV.
- `sweep_file`: native Julia sweep artifact used to extract the curve.
- `edge_source`: `:domain` or `:beam`, selecting which edge definition defines `alpha`.
- `branch_index`: branch selector. `1` is the first positive branch, `2` the second, etc.
- `n_sample`: number of curve points to rerun in detail.
- `n_modes`: number of modal coefficients retained in the CSV.
- `parallel`: whether to parallelize sampled cases across Julia threads. Defaults to `true` when Julia is started with more than one thread.
- `cache_file`: optional CSV of previously evaluated curve points used to augment
  the GP training set and to reuse acceptable rows without rerunning the solver.
  If omitted, the script chooses a dataset-matched cache file automatically.
- `alpha_accept_tol`: cached rows with `|alpha| <= alpha_accept_tol` may be reused.
- `output_file`: output CSV filename. If omitted, the script chooses a dataset-matched refined filename automatically.

Example with all arguments explicit:
`main("output"; sweep_file="sweep_motor_position_EI_uncoupled_from_matlab.jld2", edge_source=:beam, branch_index=2, n_sample=150, n_modes=8, parallel=true, cache_file="single_alpha_zero_curve_details_uncoupled.csv", alpha_accept_tol=0.005, output_file="single_alpha_zero_curve_details_uncoupled_refined.csv")`
"""
function main(
    data_dir::AbstractString=joinpath(@__DIR__, "..", "output");
    sweep_file::AbstractString="sweep_motor_position_EI_coupled_from_matlab.jld2",
    edge_source::Symbol=:beam,
    branch_index::Int=1,
    n_sample::Int=100,
    n_modes::Int=8,
    parallel::Bool=(nthreads() > 1),
    cache_file=nothing,
    alpha_accept_tol::Real=5e-3,
    output_file::AbstractString="",
)
    data_dir = ensure_dir(normpath(data_dir))
    artifact = load_sweep_artifact(joinpath(data_dir, sweep_file))
    output_name = isempty(output_file) ? default_output_file(sweep_file) : String(output_file)
    output_path = joinpath(data_dir, output_name)
    cache_path = if isnothing(cache_file)
        candidate = joinpath(data_dir, default_cache_file(sweep_file))
        isfile(candidate) ? candidate : nothing
    else
        joinpath(data_dir, String(cache_file))
    end
    cached_rows = isnothing(cache_path) ? NamedTuple[] : load_cached_curve_rows(cache_path, n_modes)
    existing_output_rows = isfile(output_path) ? load_cached_curve_rows(output_path, n_modes) : NamedTuple[]
    all_rows = merge_output_rows(cached_rows, existing_output_rows)

    summaries = artifact.summaries
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft

    left_grid = similar(summaries, ComplexF64)
    right_grid = similar(summaries, ComplexF64)
    for idx in eachindex(summaries)
        left_grid[idx], right_grid[idx] = edge_fields(summaries[idx], edge_source)
    end

    relevant_training = training_rows(all_rows; edge_source=edge_source, sweep_file=sweep_file)
    sampled = iterative_sample_candidates(
        mp_norm_list, EI_list, left_grid, right_grid;
        cached_rows=relevant_training,
        edge_source=edge_source,
        branch_index=branch_index,
        n_sample=n_sample,
        alpha_accept_tol=alpha_accept_tol,
    )
    isempty(sampled) && error("No branch-$(branch_index) candidates were found for sampling.")

    println("Selected $(length(sampled)) target points from $(basename(joinpath(data_dir, sweep_file))) using edge_source=$(edge_source), branch_index=$(branch_index), backend=gpr2d")
    !isempty(relevant_training) && println("Using $(length(relevant_training)) cached evaluated points compatible with this dataset")

    BLAS.set_num_threads(1)
    log_lock = ReentrantLock()
    max_iterations = 5
    max_solves_per_bin = 5
    logEI_bin_width = 0.03
    best_rows = Dict{Int, Any}()
    working_rows = copy(all_rows)

    # Main active learning loop
    for iteration in 1:max_iterations
        # 1. Propose candidates using CURRENT surrogate (refined by any working_rows)
        # We filter working_rows to only include ones that actually ran a solve (alpha not NaN)
        # and match our current sweep/branch context.
        relevant_training = training_rows(working_rows; edge_source=edge_source, sweep_file=sweep_file)
        extra_pts = isempty(relevant_training) ? nothing : cached_training_points(relevant_training, edge_source)
        
        target_logs = target_logEI_values(EI_list; n_sample=n_sample)
        boundary_band = length(mp_norm_list) >= 2 ? (maximum(mp_norm_list) - minimum(mp_norm_list)) / (201 - 1) : 0.0
        
        alpha_gp, sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_pts)
        
        # 2. Identify which samples need solving
        queue_indices = Int[]
        points_to_solve = Dict{Int, NamedTuple}()
        
        # Bin counts based on ALL solves done so far
        bin_counts = solve_counts_by_bin(working_rows; width=logEI_bin_width, branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)

        for (sample_index, logEI) in enumerate(target_logs)
            # Find root of current GP surrogate at this logEI
            candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=boundary_band)
            if length(candidates) < branch_index
                continue
            end
            point = candidates[branch_index]
            point = (; point..., sample_index=sample_index, curve_point_index=sample_index)
            
            # Check if we already have a "good enough" solve for this sample
            existing_best = best_row_for_sample(working_rows, sample_index; branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)
            if !isnothing(existing_best) && abs(existing_best.alpha) <= alpha_accept_tol
                best_rows[sample_index] = existing_best
                continue
            end
            
            # If not, and we haven't exceeded bin budget, add to queue
            bin = logEI_bin(logEI; width=logEI_bin_width)
            if get(bin_counts, bin, 0) < max_solves_per_bin
                push!(queue_indices, sample_index)
                points_to_solve[sample_index] = point
                bin_counts[bin] = get(bin_counts, bin, 0) + 1
            end
        end

        if isempty(queue_indices)
            println("No more points to solve (all accepted or budgets reached).")
            break
        end

        println("Iteration $(iteration): running $(length(queue_indices)) new sampled cases")
        # Use a thread-safe storage for batch results
        current_batch_results = Vector{Any}(undef, length(queue_indices))
        
        if parallel && nthreads() > 1
            println("Running in parallel across $(nthreads()) threads")
            @threads for qidx in eachindex(queue_indices)
                s_idx = queue_indices[qidx]
                point = points_to_solve[s_idx]
                row = build_row(s_idx, point, artifact, edge_source, n_modes; branch_index=branch_index, sweep_file=sweep_file, iteration=iteration)
                current_batch_results[qidx] = row
                lock(log_lock) do
                    @printf("  case %d / %d: EI=%.4e, x_M/L=%.4f, alpha=%.6f\n", s_idx, n_sample, point.EI, point.xM_over_L, row.alpha)
                end
            end
        else
            for (qidx, s_idx) in enumerate(queue_indices)
                point = points_to_solve[s_idx]
                row = build_row(s_idx, point, artifact, edge_source, n_modes; branch_index=branch_index, sweep_file=sweep_file, iteration=iteration)
                current_batch_results[qidx] = row
                @printf("  case %d / %d: EI=%.4e, x_M/L=%.4f, alpha=%.6f\n", s_idx, n_sample, point.EI, point.xM_over_L, row.alpha)
            end
        end

        # Update knowledge base with new results
        new_batch_rows = filter(!isnothing, current_batch_results)
        working_rows = merge_output_rows(working_rows, new_batch_rows)
        for row in new_batch_rows
            existing = get(best_rows, row.sample_index, nothing)
            if isnothing(existing) || abs(row.alpha) < abs(existing.alpha)
                best_rows[row.sample_index] = row
            end
        end
    end

    # Final trace of the branch
    relevant_training = training_rows(working_rows; edge_source=edge_source, sweep_file=sweep_file)
    extra_pts = isempty(relevant_training) ? nothing : cached_training_points(relevant_training, edge_source)
    alpha_gp, sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_pts)
    
    final_sampled = NamedTuple[]
    target_logs = target_logEI_values(EI_list; n_sample=n_sample)
    boundary_band = length(mp_norm_list) >= 2 ? (maximum(mp_norm_list) - minimum(mp_norm_list)) / (201 - 1) : 0.0
    for (sample_index, logEI) in enumerate(target_logs)
        candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=boundary_band)
        if length(candidates) >= branch_index
            push!(final_sampled, (; candidates[branch_index]..., sample_index=sample_index, curve_point_index=sample_index))
        end
    end

    final_rows = Any[]
    for i in 1:n_sample
        row = get(best_rows, i, nothing)
        !isnothing(row) && push!(final_rows, row)
    end

    combined_rows = merge_output_rows(existing_output_rows, final_rows)
    write_curve_csv(output_path, combined_rows, n_modes)
    beam_output_path = beam_csv_path(output_path)
    write_beam_curve_csv(beam_output_path, combined_rows)
    overlay_path = replace(output_path, r"\.csv$" => "_branch$(branch_index)_overlay.pdf")
    write_alpha_overlay_plot(
        overlay_path,
        mp_norm_list,
        EI_list,
        left_grid,
        right_grid,
        final_sampled,
        final_rows;
        extra_points=extra_pts,
        edge_source=edge_source,
    )

    combined_rows = merge_output_rows(existing_output_rows, final_rows)
    write_curve_csv(output_path, combined_rows, n_modes)
    beam_output_path = beam_csv_path(output_path)
    write_beam_curve_csv(beam_output_path, combined_rows)
    overlay_path = replace(output_path, r"\.csv$" => "_branch$(branch_index)_overlay.pdf")
    write_alpha_overlay_plot(
        overlay_path,
        mp_norm_list,
        EI_list,
        left_grid,
        right_grid,
        sampled,
        final_rows;
        extra_points=isempty(training_rows(working_rows; edge_source=edge_source, sweep_file=sweep_file)) ? nothing : cached_training_points(training_rows(working_rows; edge_source=edge_source, sweep_file=sweep_file), edge_source),
        edge_source=edge_source,
    )
    println("Saved $(output_path)")
    println("Saved $(beam_output_path)")
    println("Saved $(overlay_path)")
    return (curve_csv = output_path, beam_csv = beam_output_path, overlay = overlay_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
    sweep_file = length(ARGS) >= 2 ? ARGS[2] : "sweep_motor_position_EI_coupled_from_matlab.jld2"
    edge_source = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :beam
    branch_index = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    n_sample = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 100
    output_file = length(ARGS) >= 6 ? ARGS[6] : ""
    parallel = length(ARGS) >= 7 ? lowercase(ARGS[7]) in ("1", "true", "yes", "y", "parallel") : (nthreads() > 1)
    cache_file = length(ARGS) >= 8 ? ARGS[8] : nothing
    alpha_accept_tol = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 5e-3
    main(data_dir; sweep_file=sweep_file, edge_source=edge_source, branch_index=branch_index, n_sample=n_sample, output_file=output_file, parallel=parallel, cache_file=cache_file, alpha_accept_tol=alpha_accept_tol)
end

using Surferbot
using Statistics
using LinearAlgebra
using Base.Threads
using DelimitedFiles

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

function collect_zero_crossings_gp(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix;
                                   mp_count::Int=201, logEI_count::Int=201, extra_points=nothing)
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

function filter_candidates(candidates; sa_filter::Symbol=:negative)
    if sa_filter == :negative
        return filter(c -> c.sa_ratio < 0, candidates)
    elseif sa_filter == :positive
        return filter(c -> c.sa_ratio > 0, candidates)
    elseif sa_filter == :none
        return candidates
    end
    error("sa_filter must be :negative, :positive, or :none")
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

function track_branch(crossings; sa_filter::Symbol=:negative, jump_factor::Real=4.0, min_jump_tol::Real=0.05)
    by_EI = group_crossings_by_EI(crossings)
    EI_desc = sort(collect(keys(by_EI)); rev=true)
    isempty(EI_desc) && return NamedTuple[]

    curve_points = NamedTuple[]
    accepted_steps = Float64[]

    seed_candidates = NamedTuple[]
    seed_idx = nothing
    for (i, EI) in pairs(EI_desc)
        candidates = filter_candidates(by_EI[EI]; sa_filter=sa_filter)
        if !isempty(candidates)
            seed_candidates = candidates
            seed_idx = i
            break
        end
    end
    isempty(seed_candidates) && return NamedTuple[]
    seed = reduce((a, b) -> a.xM_over_L <= b.xM_over_L ? a : b, seed_candidates)
    push!(curve_points, seed)

    for EI in EI_desc[(seed_idx + 1):end]
        candidates = filter_candidates(by_EI[EI]; sa_filter=sa_filter)
        isempty(candidates) && continue

        y = log10(EI)
        x_pred = predict_next_x(curve_points, y)
        chosen = reduce((a, b) ->
            abs(a.xM_over_L - x_pred) <= abs(b.xM_over_L - x_pred) ? a : b,
            candidates,
        )

        jump = abs(chosen.xM_over_L - curve_points[end].xM_over_L)
        local_scale = isempty(accepted_steps) ? min_jump_tol : max(min_jump_tol, jump_factor * median(accepted_steps))

        if jump <= local_scale
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

function write_curve_csv(path::AbstractString, rows, n_modes::Int)
    open(path, "w") do io
        header = [
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
                "Q$(j)_re", "Q$(j)_im", "Q$(j)_abs", "Q$(j)_phase_deg",
                "F$(j)_re", "F$(j)_im", "F$(j)_abs", "F$(j)_phase_deg",
                "residual$(j)_re", "residual$(j)_im", "residual$(j)_abs", "residual$(j)_phase_deg",
                "energy_frac$(j)",
                "mode_index$(j)",
                "mode_type$(j)",
            ])
        end
        println(io, join(header, ","))

        for row in rows
            fields = String[
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
                append!(fields, split(format_complex(row.modal.q[j]), ","))
                append!(fields, split(format_complex(row.modal.Q[j]), ","))
                append!(fields, split(format_complex(row.modal.F[j]), ","))
                append!(fields, split(format_complex(row.modal.balance_residual[j]), ","))
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

function build_row(sample_index, curve_points, point, artifact, edge_source::Symbol, n_modes::Int)
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
         sweep_file="sweep_motorPosition_EI_coupled_from_matlab.jld2",
         edge_source=:domain,
         sa_filter=:negative,
         n_sample=12,
         n_modes=8,
         parallel=false,
         output_file="single_alpha_zero_curve_details.csv")

Extract one `alpha = 0` curve from a saved sweep artifact and write a detailed
per-sample CSV after rerunning the Julia solver and modal decomposition. The
curve is extracted from a 2D GP surrogate in `(x_M/L, log10(EI))`, then
tracked from high `EI` to low `EI`.

Inputs:
- `data_dir`: directory containing the input sweep artifact and receiving the CSV.
- `sweep_file`: native Julia sweep artifact used to extract the curve.
- `edge_source`: `:domain` or `:beam`, selecting which edge definition defines `alpha`.
- `sa_filter`: `:negative`, `:positive`, or `:none`, used to keep one family of crossings.
- `n_sample`: number of curve points to rerun in detail.
- `n_modes`: number of modal coefficients retained in the CSV.
- `parallel`: whether to parallelize sampled cases across Julia threads.
- `cache_file`: optional CSV of previously evaluated curve points used to augment
  the GP training set and to reuse acceptable rows without rerunning the solver.
- `alpha_accept_tol`: cached rows with `|alpha| <= alpha_accept_tol` may be reused.
- `output_file`: output CSV filename.
"""
function main(
    data_dir::AbstractString=joinpath(@__DIR__, "..", "output");
    sweep_file::AbstractString="sweep_motorPosition_EI_coupled_from_matlab.jld2",
    edge_source::Symbol=:domain,
    sa_filter::Symbol=:negative,
    n_sample::Int=12,
    n_modes::Int=8,
    parallel::Bool=false,
    cache_file=nothing,
    alpha_accept_tol::Real=5e-2,
    output_file::AbstractString="single_alpha_zero_curve_details.csv",
)
    data_dir = ensure_dir(normpath(data_dir))
    artifact = load_sweep_artifact(joinpath(data_dir, sweep_file))
    cache_path = if isnothing(cache_file)
        candidate = joinpath(data_dir, output_file)
        isfile(candidate) ? candidate : nothing
    else
        joinpath(data_dir, String(cache_file))
    end
    cached_rows = isnothing(cache_path) ? NamedTuple[] : load_cached_curve_rows(cache_path, n_modes)

    summaries = artifact.summaries
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft

    left_grid = similar(summaries, ComplexF64)
    right_grid = similar(summaries, ComplexF64)
    for idx in eachindex(summaries)
        left_grid[idx], right_grid[idx] = edge_fields(summaries[idx], edge_source)
    end

    extra_points = isempty(cached_rows) ? nothing : cached_training_points(cached_rows, edge_source)
    crossings = collect_zero_crossings_gp(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
    curve_points = track_branch(crossings; sa_filter=sa_filter)
    sampled = sample_curve_points(curve_points; n_sample=n_sample)

    println("Selected $(length(curve_points)) points on the extracted curve")
    println("Sampling $(length(sampled)) points from $(basename(joinpath(data_dir, sweep_file))) using edge_source=$(edge_source), sa_filter=$(sa_filter), backend=gpr2d")
    !isempty(cached_rows) && println("Using $(length(cached_rows)) cached evaluated points from $(basename(cache_path))")

    BLAS.set_num_threads(1)
    rows = Vector{Any}(undef, length(sampled))
    log_lock = ReentrantLock()

    if parallel && nthreads() > 1
        println("Running sampled cases in parallel across $(nthreads()) Julia threads")
        @threads for i in eachindex(sampled)
            point = sampled[i]
            cached = find_reusable_cached_row(cached_rows, point; alpha_accept_tol=alpha_accept_tol)
            row = isnothing(cached) ? build_row(i, curve_points, point, artifact, edge_source, n_modes) : merge(cached, (sample_index=i, curve_point_index=point.curve_point_index))
            rows[i] = row
            lock(log_lock) do
                println(
                    "  case $(i) / $(length(sampled)): " *
                    "EI=$(point.EI), x_M/L=$(round(point.xM_over_L; digits=4)), " *
                    "alpha=$(round(row.alpha; sigdigits=6))" *
                    (isnothing(cached) ? "" : " [cached]")
                )
            end
        end
    else
        for (sample_index, point) in enumerate(sampled)
            cached = find_reusable_cached_row(cached_rows, point; alpha_accept_tol=alpha_accept_tol)
            row = isnothing(cached) ? build_row(sample_index, curve_points, point, artifact, edge_source, n_modes) : merge(cached, (sample_index=sample_index, curve_point_index=point.curve_point_index))
            rows[sample_index] = row
            println(
                "  case $sample_index / $(length(sampled)): " *
                "EI=$(point.EI), x_M/L=$(round(point.xM_over_L; digits=4)), " *
                "alpha=$(round(row.alpha; sigdigits=6))" *
                (isnothing(cached) ? "" : " [cached]")
            )
        end
    end

    output_path = joinpath(data_dir, output_file)
    write_curve_csv(output_path, rows, n_modes)
    println("Saved $(output_path)")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
    sweep_file = length(ARGS) >= 2 ? ARGS[2] : "sweep_motorPosition_EI_coupled_from_matlab.jld2"
    edge_source = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :domain
    sa_filter = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :negative
    n_sample = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12
    output_file = length(ARGS) >= 6 ? ARGS[6] : "single_alpha_zero_curve_details.csv"
    parallel = length(ARGS) >= 7 ? lowercase(ARGS[7]) in ("1", "true", "yes", "y", "parallel") : false
    cache_file = length(ARGS) >= 8 ? ARGS[8] : nothing
    alpha_accept_tol = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 5e-2
    main(data_dir; sweep_file=sweep_file, edge_source=edge_source, sa_filter=sa_filter, n_sample=n_sample, output_file=output_file, parallel=parallel, cache_file=cache_file, alpha_accept_tol=alpha_accept_tol)
end

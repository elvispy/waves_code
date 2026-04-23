using Surferbot
using Statistics
using LinearAlgebra
using Base.Threads
using DelimitedFiles
using Plots
using Printf

# Purpose: extract one `sa_ratio = 0` curve from a saved Julia sweep artifact,
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
    S_grid = (right_grid .+ left_grid) ./ 2
    A_grid = (right_grid .- left_grid) ./ 2
    sa_ratio_grid = log10.(abs.(S_grid) ./ (abs.(A_grid) .+ 1e-12))
    return sa_ratio_grid
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

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    # Sharper GP: use 1.5x median spacing instead of 4x to capture sharp branch physics
    lx = isempty(dx) ? 0.05 : max(0.015, 1.5 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 4 * median(dy))
    sigma_f = max(std(values), 1e-3)
    # Use higher regularization to avoid high-frequency ripples (garbage properties)
    # when many exact zeros are added to the training set.
    noise = max(1e-4, 2e-2 * sigma_f)

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
    sa_ratio_grid = build_scalar_fields(left_grid, right_grid)
    logEI_list = log10.(Float64.(EI_list))

    xtrain = Float64[]
    ytrain = Float64[]
    sa_train = Float64[]

    # 1. Add base grid points
    for ie in eachindex(logEI_list), im in eachindex(mp_norm_list)
        push!(xtrain, Float64(mp_norm_list[im]))
        push!(ytrain, Float64(logEI_list[ie]))
        push!(sa_train, Float64(sa_ratio_grid[im, ie]))
    end

    if !isnothing(extra_points)
        append!(xtrain, extra_points.x)
        append!(ytrain, extra_points.y)
        append!(sa_train, extra_points.sa)
    end
    return xtrain, ytrain, sa_train, logEI_list
end

function surrogate_models(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix; extra_points=nothing)
    xtrain, ytrain, sa_train, _ =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
    return fit_gp2d(xtrain, ytrain, sa_train)
end

function nontrivial_candidates(candidates, branch_index::Int; boundary_band::Real=0.0)
    isempty(candidates) && return candidates
    # 1. Sort found candidates by position
    sorted = sort(candidates; by = c -> c.xM_over_L)
    # 2. Smart Skip: skip center symmetry root (Branch 0) if not explicitly tracking it.
    if branch_index != 0 && !isempty(sorted) && sorted[1].xM_over_L < 0.03
        return sorted[2:end]
    end
    return sorted
end

function branch_candidates_at_logEI(mp_norm_list, logEI::Real, sa_gp; mp_count::Int=401, boundary_band::Real=0.0, window=nothing, branch_index::Int=1)
    # Search Domain: Hard lock to [0.01, 0.49] to avoid symmetry/edge effects
    mp_min = 0.01
    mp_max = 0.49
    
    if !isnothing(window)
        mp_min = max(mp_min, window[1])
        mp_max = min(mp_max, window[2])
    end

    mp_dense = collect(range(mp_min, mp_max; length=mp_count))
    sa_row = [predict_gp2d(sa_gp, mp, logEI) for mp in mp_dense]

    candidates = NamedTuple[]
    for im in 2:(length(mp_dense) - 1)
        # Check for local extrema
        is_min = sa_row[im] < sa_row[im-1] && sa_row[im] < sa_row[im+1]
        is_max = sa_row[im] > sa_row[im-1] && sa_row[im] > sa_row[im+1]

        if (branch_index == 1 && is_min) || ((branch_index == 2 || branch_index == 0) && is_max)
            # 3-point parabolic refinement for the GP extremum to get sub-grid precision
            x1, x2, x3 = mp_dense[im-1], mp_dense[im], mp_dense[im+1]
            y1, y2, y3 = sa_row[im-1], sa_row[im], sa_row[im+1]

            denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
            if abs(denom) > 1e-12
                x_opt = x2 - 0.5 * ((x2 - x1)^2 * (y2 - y3) - (x2 - x3)^2 * (y2 - y1)) / denom
                x_opt = clamp(x_opt, x1, x3)
                sa_opt = predict_gp2d(sa_gp, x_opt, logEI)
            else
                x_opt = x2
                sa_opt = y2
            end
            push!(candidates, (EI = 10.0^logEI, xM_over_L = x_opt, sa_ratio = sa_opt, target_log10_EI = logEI))
        end
    end
    return nontrivial_candidates(candidates, branch_index; boundary_band=boundary_band)
end

function target_logEI_values(EI_list; n_sample::Int)
    logEI_list = sort(log10.(Float64.(EI_list)))
    if n_sample <= 1
        return [logEI_list[1]]
    end
    return collect(range(first(logEI_list), last(logEI_list); length=n_sample))
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

function best_row_for_sample(rows, sample_index::Int; branch_index::Int, edge_source::Symbol, sweep_file::AbstractString)
    candidates = filter(rows) do r
        r.sample_index == sample_index &&
        get(r, :branch_index, branch_index) == branch_index &&
        get(r, :edge_source, String(edge_source)) == String(edge_source) &&
        get(r, :sweep_file, sweep_file) == sweep_file
    end
    isempty(candidates) && return nothing
    # For Branch 1, we want the point with the most negative sa_ratio (deepest valley)
    # For others, we want the most positive sa_ratio (highest peak)
    return branch_index == 1 ? argmin(r -> r.sa_ratio, candidates) : argmax(r -> r.sa_ratio, candidates)
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

function argmax(f, xs)
    best = first(xs)
    bestv = f(best)
    for x in Iterators.drop(xs, 1)
        v = f(x)
        if v > bestv
            best = x
            bestv = v
        end
    end
    return best
end

function choose_branch_candidate(candidates, branch_index::Int; anchor_xM=nothing)
    isempty(candidates) && return nothing
    
    # We no longer hard-filter by side/slope to avoid being "locked" incorrectly.
    # Instead, we rely on the anchor and proximity to select the manifold.
    if isnothing(anchor_xM)
        return length(candidates) >= branch_index ? candidates[branch_index] : candidates[end]
    end
    return argmin(c -> abs(c.xM_over_L - anchor_xM), candidates)
end

function same_point(a, b; x_tol::Real=1e-4, rel_EI_tol::Real=1e-8)
    return abs(a.xM_over_L - b.xM_over_L) <= x_tol &&
           abs(a.EI - b.EI) <= rel_EI_tol * max(abs(a.EI), abs(b.EI), 1.0)
end

function clamp_to_domain(x::Real, mp_norm_list)
    return clamp(Float64(x), 0.01, 0.49)
end

function slice_rows(rows, sample_index::Int; branch_index::Int, edge_source::Symbol, sweep_file::AbstractString)
    return filter(rows) do r
        get(r, :sample_index, -1) == sample_index &&
        get(r, :branch_index, branch_index) == branch_index &&
        get(r, :edge_source, String(edge_source)) == String(edge_source) &&
        get(r, :sweep_file, sweep_file) == sweep_file
    end
end

function propose_local_extremum_point(local_slice_rows, gp_point, mp_norm_list; window=nothing, branch_index::Int=1)
    # Target point from GP
    x_target = gp_point.xM_over_L

    # Define search bounds: Search Domain lock [0.01, 0.49]
    xmin = 0.01
    xmax = 0.49
    if !isnothing(window)
        xmin = max(xmin, window[1])
        xmax = min(xmax, window[2])
    end

    if isempty(local_slice_rows)
        return (; gp_point..., xM_over_L = clamp(x_target, xmin, xmax))
    end

    rows_sorted = sort(local_slice_rows; by = r -> r.xM_over_L)

    # If we have at least 3 points, try parabolic fit around the best point
    if length(rows_sorted) >= 3
        best_row = branch_index == 1 ? argmin(r -> r.sa_ratio, rows_sorted) : argmax(r -> r.sa_ratio, rows_sorted)
        idx_best = findfirst(r -> r.xM_over_L == best_row.xM_over_L, rows_sorted)
        
        if !isnothing(idx_best) && idx_best > 1 && idx_best < length(rows_sorted)
            # We have a bracket! Use 3-point parabolic fit.
            r1, r2, r3 = rows_sorted[idx_best-1], rows_sorted[idx_best], rows_sorted[idx_best+1]
            x1, x2, x3 = r1.xM_over_L, r2.xM_over_L, r3.xM_over_L
            y1, y2, y3 = r1.sa_ratio, r2.sa_ratio, r3.sa_ratio
            
            denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
            if abs(denom) > 1e-12
                x_opt = x2 - 0.5 * ((x2 - x1)^2 * (y2 - y3) - (x2 - x3)^2 * (y2 - y1)) / denom
                return (; gp_point..., xM_over_L = clamp(x_opt, xmin, xmax))
            end
        end
    end

    # Fallback/Exploration: take a damped step from the current best towards the GP target
    best_row = branch_index == 1 ? argmin(r -> r.sa_ratio, local_slice_rows) : argmax(r -> r.sa_ratio, local_slice_rows)
    x_next = 0.7 * x_target + 0.3 * best_row.xM_over_L
    return (; gp_point..., xM_over_L = clamp(x_next, xmin, xmax))
end

function cached_training_points(rows, edge_source::Symbol)
    # Filter only non-NaN results
    valid = filter(r -> !isnan(r.sa_ratio), rows)
    return (
        x = [r.xM_over_L for r in valid],
        y = [log10(r.EI) for r in valid],
        sa = [r.sa_ratio for r in valid]
    )
end

function training_rows(rows; edge_source::Symbol, sweep_file::AbstractString)
    # We use ANY solved point from this dataset/edge as training for the surrogate
    return filter(rows) do r
        get(r, :edge_source, String(edge_source)) == String(edge_source) &&
        get(r, :sweep_file, sweep_file) == sweep_file
    end
end

function merge_output_rows(existing_rows, new_rows)
    # Filter out redundant points and keep best for each coordinate
    merged = copy(existing_rows)
    append!(merged, new_rows)
    return merged
end

function row_key(row)
    return (
        Int(get(row, :branch_index, 1)),
        round(Float64(get(row, :target_log10_EI, log10(row.EI))); digits=10),
        round(Float64(row.xM_over_L); digits=10),
    )
end

function existing_row_keys(rows)
    return Set(row_key(row) for row in rows)
end

function load_existing_beam_keys(path::AbstractString)
    !isfile(path) && return Set{Tuple{Int, Float64, Float64}}()
    data_all = try
        readdlm(path, ',', header=true, quotes=true)
    catch e
        @warn "Failed to parse beam CSV $path: $e. Returning empty key set."
        return Set{Tuple{Int, Float64, Float64}}()
    end
    data, header = data_all
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)
    branch_col = col("branch_index")
    target_col = col("target_log10_EI")
    xm_col = col("xM_over_L")
    if isnothing(branch_col) || isnothing(target_col) || isnothing(xm_col)
        return Set{Tuple{Int, Float64, Float64}}()
    end

    keys = Set{Tuple{Int, Float64, Float64}}()
    for i in axes(data, 1)
        try
            push!(keys, (
                Int(data[i, branch_col]),
                round(Float64(data[i, target_col]); digits=10),
                round(Float64(data[i, xm_col]); digits=10),
            ))
        catch
        end
    end
    return keys
end

function format_complex(z)
    return "$(real(z)),$(imag(z)),$(abs(z)),$(rad2deg(angle(z)))"
end

function curve_csv_header(n_modes::Int)
    header = [
        "branch_index", "edge_source", "sweep_file", "iteration", "target_log10_EI",
        "sample_index", "curve_point_index", "EI", "log10_EI", "xM_over_L",
        "motor_position", "omega", "U", "power", "power_input", "thrust", "tail_flat_ratio",
        "rho_raft", "L_raft",
        "eta_left_domain_re", "eta_left_domain_im", "eta_left_domain_abs", "eta_left_domain_phase_deg",
        "eta_right_domain_re", "eta_right_domain_im", "eta_right_domain_abs", "eta_right_domain_phase_deg",
        "eta_left_beam_re", "eta_left_beam_im", "eta_left_beam_abs", "eta_left_beam_phase_deg",
        "eta_right_beam_re", "eta_right_beam_im", "eta_right_beam_abs", "eta_right_beam_phase_deg",
        "sa_ratio", "sa_ratio_domain", "sa_ratio_beam",
    ]
    for j in 0:(n_modes - 1)
        append!(header, [
            "q_w$(j)_re", "q_w$(j)_im", "q_w$(j)_abs", "q_w$(j)_phase_deg",
            "Q_w$(j)_re", "Q_w$(j)_im", "Q_w$(j)_abs", "Q_w$(j)_phase_deg",
            "F_w$(j)_re", "F_w$(j)_im", "F_w$(j)_abs", "F_w$(j)_phase_deg",
            "residual$(j)_re", "residual$(j)_im", "residual$(j)_abs", "residual$(j)_phase_deg",
            "beta$(j)", "Psi_left$(j)", "Psi_right$(j)",
            "energy_frac$(j)",
            "mode_index$(j)",
            "mode_type$(j)",
        ])
    end
    return header
end

function curve_csv_fields(row, n_modes::Int)
    # Map coefficients from the cleaned ModalDecomposition
    q_w = row.modal.q_w
    Q_w = row.modal.Q_w
    F_w = row.modal.F_w
    R_w = row.modal.balance_residual

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
        string(row.rho_raft),
        string(row.L_raft),
    ]
    append!(fields, split(format_complex(row.eta_left_domain), ","))
    append!(fields, split(format_complex(row.eta_right_domain), ","))
    append!(fields, split(format_complex(row.eta_left_beam), ","))
    append!(fields, split(format_complex(row.eta_right_beam), ","))
    append!(fields, [
        string(row.sa_ratio),
        string(row.sa_ratio_domain),
        string(row.sa_ratio_beam),
    ])
    for j in 1:n_modes
        append!(fields, split(format_complex(q_w[j]), ","))
        append!(fields, split(format_complex(Q_w[j]), ","))
        append!(fields, split(format_complex(F_w[j]), ","))
        append!(fields, split(format_complex(R_w[j]), ","))
        append!(fields, [
            string(row.modal.beta[j]),
            string(row.modal.Psi[1, j]),
            string(row.modal.Psi[end, j]),
            string(row.modal.energy_frac[j]),
            string(row.modal.n[j]),
            row.modal.mode_type[j],
        ])
    end
    return fields
end


function append_curve_csv(path::AbstractString, rows, n_modes::Int, written_keys::Set)
    rows_to_write = [row for row in rows if !(row_key(row) in written_keys)]
    isempty(rows_to_write) && return written_keys

    file_exists = isfile(path)
    open(path, file_exists ? "a" : "w") do io
        if !file_exists
            println(io, join(curve_csv_header(n_modes), ","))
        end
        for row in rows_to_write
            println(io, join(curve_csv_fields(row, n_modes), ","))
            push!(written_keys, row_key(row))
        end
    end
    return written_keys
end

function append_beam_curve_csv(path::AbstractString, rows, written_keys::Set)
    rows_to_write = [row for row in rows if !(row_key(row) in written_keys)]
    isempty(rows_to_write) && return written_keys

    file_exists = isfile(path)
    open(path, file_exists ? "a" : "w") do io
        if !file_exists
            println(io, "branch_index,target_log10_EI,log10_EI,xM_over_L,sa_ratio_beam")
        end
        for row in rows_to_write
            @printf(io, "%d,%.10f,%.6f,%.6f,%.6f\n",
                    get(row, :branch_index, 1),
                    get(row, :target_log10_EI, log10(row.EI)),
                    log10(row.EI),
                    row.xM_over_L,
                    row.sa_ratio_beam)
            push!(written_keys, row_key(row))
        end
    end
    return written_keys
end

function load_cached_curve_rows(path::AbstractString, n_modes::Int)
    !isfile(path) && return NamedTuple[]
    # readdlm can be sensitive to quotes in filenames; 
    # we specify quotes and ensure we use the comma delimiter.
    data_all = try
        readdlm(path, ',', header=true, quotes=true)
    catch e
        @warn "Failed to parse CSV $path: $e. Returning empty cache."
        return NamedTuple[]
    end
    
    data, header = data_all
    names = string.(vec(header))
    rows = NamedTuple[]
    
    # Helper to find column indices
    col(n) = findfirst(==(n), names)
    
    for i in axes(data, 1)
        # Minimal fields needed for training/re-eval
        try
            r = (
                branch_index = isnothing(col("branch_index")) ? 1 : Int(data[i, col("branch_index")]),
                target_log10_EI = isnothing(col("target_log10_EI")) ? Float64(log10(Float64(data[i, col("EI")]))) : Float64(data[i, col("target_log10_EI")]),
                EI = Float64(data[i, col("EI")]),
                xM_over_L = Float64(data[i, col("xM_over_L")]),
                sa_ratio = isnothing(col("sa_ratio")) ? 0.0 : Float64(data[i, col("sa_ratio")]),
                sa_ratio_beam = isnothing(col("sa_ratio_beam")) ? 0.0 : Float64(data[i, col("sa_ratio_beam")]),
                sample_index = Int(data[i, col("sample_index")]),
                sweep_file = String(data[i, col("sweep_file")]),
                edge_source = String(data[i, col("edge_source")]),
            )
            push!(rows, r)
        catch e
            @warn "Skipping corrupted row $i in $path: $e"
        end
    end
    return rows
end

function build_row(sample_index, point, artifact, edge_source, n_modes; branch_index, sweep_file, iteration)
    params = apply_parameter_overrides(artifact.base_params, (EI=point.EI, motor_position=point.xM_over_L * artifact.base_params.L_raft))
    result = flexible_solver(params)
    modal = decompose_raft_freefree_modes(result; num_modes=n_modes, verbose=false)
    metrics = beam_edge_metrics(result)

    S_domain = (metrics.eta_right_domain + metrics.eta_left_domain) / 2
    A_domain = (metrics.eta_right_domain - metrics.eta_left_domain) / 2
    sa_ratio_domain = log10(abs(S_domain) / (abs(A_domain) + 1e-12))
    S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
    A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
    sa_ratio_beam = log10(abs(S_beam) / (abs(A_beam) + 1e-12))

    sa_ratio_selected = edge_source == :domain ? sa_ratio_domain : sa_ratio_beam

    return (
        branch_index = branch_index,
        edge_source = String(edge_source),
        sweep_file = sweep_file,
        iteration = iteration,
        target_log10_EI = get(point, :target_log10_EI, log10(point.EI)),
        sample_index = sample_index,
        curve_point_index = sample_index,
        EI = params.EI,
        xM_over_L = point.xM_over_L,
        motor_position = params.motor_position,
        omega = params.omega,
        rho_raft = params.rho_raft,
        L_raft = params.L_raft,
        U = result.U,
        power = result.power,
        power_input = result.metadata.args.power,
        thrust = result.thrust,
        tail_flat_ratio = NaN, # Not in current result struct
        eta_left_domain = metrics.eta_left_domain,
        eta_right_domain = metrics.eta_right_domain,
        eta_left_beam = metrics.eta_left_beam,
        eta_right_beam = metrics.eta_right_beam,
        sa_ratio = sa_ratio_selected,
        sa_ratio_domain = sa_ratio_domain,
        sa_ratio_beam = sa_ratio_beam,
        modal = modal
    )
end

function refine_single_slice!(
    s_idx::Int,
    logEI::Real,
    initial_anchor_xM,
    working_rows,
    artifact,
    edge_source,
    n_modes::Int,
    branch_index::Int,
    sweep_file::AbstractString,
    mp_norm_list,
    EI_list,
    left_grid,
    right_grid;
    sa_accept_tol::Real,
    local_max_iterations::Int,
    iteration_offset::Int=0,
    tunnel_width::Real=0.10,
)
    local_rows_state = copy(working_rows)
    local_best = best_row_for_sample(local_rows_state, s_idx; branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)

    if !isnothing(local_best) && abs(local_best.sa_ratio) <= sa_accept_tol
        return (best_row=local_best, new_rows=NamedTuple[], logs=[
            "Slice $(s_idx): reused cached point at EI=$(local_best.EI), x_M/L=$(local_best.xM_over_L), sa_ratio=$(local_best.sa_ratio)"
        ])
    end

    logs = String["Slice $(s_idx): refining target log10(EI)=$(round(logEI; digits=6))"]
    new_rows = NamedTuple[]
    local_anchor_xM = !isnothing(local_best) ? local_best.xM_over_L : initial_anchor_xM

    for local_iteration in 1:local_max_iterations
        relevant_training = training_rows(local_rows_state; edge_source=edge_source, sweep_file=sweep_file)
        extra_pts = isempty(relevant_training) ? nothing : cached_training_points(relevant_training, edge_source)
        sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_pts)
        
        # Identity Tunneling: only look for roots near the anchor
        window = isnothing(local_anchor_xM) ? nothing : (local_anchor_xM - tunnel_width, local_anchor_xM + tunnel_width)
        candidates = branch_candidates_at_logEI(mp_norm_list, logEI, sa_gp; boundary_band=0.0, window=window)

        if isempty(candidates)
            push!(logs, "  local $(local_iteration) / $(local_max_iterations): no candidates found in window $(window)")
            break
        end

        gp_point = choose_branch_candidate(candidates, branch_index; anchor_xM=local_anchor_xM)
        local_slice_rows = slice_rows(local_rows_state, s_idx; branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)
        point = propose_local_root_point(local_slice_rows, gp_point, mp_norm_list; window=window)
        point = (; point..., sample_index=s_idx, curve_point_index=s_idx)

        if !isnothing(local_best) && same_point(local_best, point)
            push!(logs, "  local $(local_iteration) / $(local_max_iterations): local root update repeated same point, stopping local refinement")
            break
        end

        row = build_row(
            s_idx,
            point,
            artifact,
            edge_source,
            n_modes;
            branch_index=branch_index,
            sweep_file=sweep_file,
            iteration=iteration_offset + local_iteration,
        )

        # Physical Identity Guard: Self-Healing Anchor
        # Reject jump > 0.15.
        jump = isnothing(local_anchor_xM) ? 0.0 : abs(row.xM_over_L - local_anchor_xM)

        if jump > 0.15
            push!(logs, @sprintf("  REJECTED local %d: jump=%.3f (>0.15)", local_iteration, jump))
            # We still add to working state for GP training but NOT to the accepted result set
            local_rows_state = merge_output_rows(local_rows_state, [row])
            continue
        end

        push!(new_rows, row)
        local_rows_state = merge_output_rows(local_rows_state, [row])
        if isnothing(local_best) || abs(row.sa_ratio) < abs(local_best.sa_ratio)
            local_best = row
        end
        local_anchor_xM = local_best.xM_over_L

        push!(logs, @sprintf("  local %d / %d: EI=%.4e, x_M/L=%.4f, sa_ratio=%.6f",
                             local_iteration, local_max_iterations, row.EI, row.xM_over_L, row.sa_ratio))

        if abs(local_best.sa_ratio) <= sa_accept_tol
            break
        end
    end

    return (best_row=local_best, new_rows=new_rows, logs=logs)
end

function write_sa_overlay_plot(path::AbstractString, mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix,
                                  rows; edge_source::Symbol=:domain,
                                  mp_count::Int=241, logEI_count::Int=241)
    # Background GPR: only train on the initial coarse grid to preserve global field shape
    xtrain, ytrain, sa_train, logEI_list =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=nothing)
    sa_gp = fit_gp2d(xtrain, ytrain, sa_train)

    mp_dense = collect(range(minimum(Float64.(mp_norm_list)), maximum(Float64.(mp_norm_list)); length=mp_count))
    logEI_dense = collect(range(minimum(logEI_list), maximum(logEI_list); length=logEI_count))
    sa_pred = Matrix{Float64}(undef, length(mp_dense), length(logEI_dense))
    @inbounds for j in eachindex(logEI_dense), i in eachindex(mp_dense)
        sa_pred[i, j] = predict_gp2d(sa_gp, mp_dense[i], logEI_dense[j])
    end

    plt = contourf(
        logEI_dense,
        mp_dense,
        sa_pred;
        levels=31,
        c=:balance,
        xlabel="log10(EI)",
        ylabel="x_M / L",
        colorbar_title="sa_ratio ($(edge_source))",
        title="Predicted SA-ratio field with traced branch",
        size=(1000, 1200),
        interpolate=true,
    )
    contour!(
        plt,
        logEI_dense,
        mp_dense,
        sa_pred;
        levels=[0.0],
        color=:white,
        linewidth=2,
        label="GP sa_ratio=0",
    )

    solved_rows = sort(rows; by = r -> get(r, :target_log10_EI, log10(r.EI)))
    if !isempty(solved_rows)
        solved_x = [r.xM_over_L for r in solved_rows]
        solved_y = [get(r, :target_log10_EI, log10(r.EI)) for r in solved_rows]
        
        # Primary visual: scatter of true zeros from the solver
        scatter!(
            plt,
            solved_y,
            solved_x;
            color=:black,
            markersize=5,
            markerstrokewidth=1,
            markerstrokecolor=:white,
            label="solved zeros (CSV)",
        )
        
        # Solid line connecting them for continuity
        plot!(
            plt,
            solved_y,
            solved_x;
            color=:black,
            linewidth=2,
            alpha=0.7,
            label="",
        )
    end
    savefig(plt, path)
end

function default_output_file(sweep_file)
    # Match script name: analyze_single_alpha_zero_curve
    # We add the sweep_file base name as a suffix to distinguish different runs if needed,
    # or just use the script name. The user prefers script name.
    return "analyze_single_alpha_zero_curve.csv"
end

function default_cache_file(sweep_file)
    return nothing
end

function beam_csv_path(csv_path)
    # Multiple outputs from the same script: add suffix
    return replace(csv_path, r"\.csv$" => "_beam.csv")
end

function main(data_dir::AbstractString, sweep_file::AbstractString, edge_source_in::AbstractString,
              branch_index::Int, n_sample::Int, output_file::AbstractString, parallel::Bool;
              cache_file=nothing, sa_accept_tol::Float64=1e-3, n_modes::Int=8,
              local_max_iterations::Int=5, tunnel_width::Real=0.10)
    
    output_dir = ensure_dir(normpath(data_dir))
    sweep_path = joinpath(output_dir, "jld2", sweep_file)
    artifact = load_sweep_artifact(sweep_path)
    
    output_name = isempty(output_file) ? default_output_file(sweep_file) : String(output_file)
    output_path = joinpath(output_dir, "csv", output_name)
    edge_source = Symbol(edge_source_in)

    cache_path = if isnothing(cache_file)
        nothing
    else
        joinpath(output_dir, "jld2", String(cache_file))
    end
    
    cached_rows = isnothing(cache_path) ? NamedTuple[] : load_cached_curve_rows(cache_path, n_modes)
    existing_output_rows = isfile(output_path) ? load_cached_curve_rows(output_path, n_modes) : NamedTuple[]
    all_rows = merge_output_rows(cached_rows, existing_output_rows)
    written_curve_keys = existing_row_keys(existing_output_rows)
    beam_output_path = beam_csv_path(output_path)
    written_beam_keys = load_existing_beam_keys(beam_output_path)

    summaries = artifact.summaries
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft

    left_grid = similar(summaries, ComplexF64)
    right_grid = similar(summaries, ComplexF64)
    for idx in eachindex(summaries)
        left_grid[idx], right_grid[idx] = edge_fields(summaries[idx], edge_source)
    end

    # LOCAL VALIDATION: Ensure build_row actually works before starting the loop.
    # This catches FieldErrors and MethodErrors early.
    println("Validating solver infrastructure...")
    test_point = (EI = EI_list[1], xM_over_L = mp_norm_list[1], target_log10_EI = log10(EI_list[1]))
    try
        build_row(0, test_point, artifact, edge_source, n_modes; branch_index=branch_index, sweep_file=sweep_file, iteration=0)
        println("Infrastructure validated.")
    catch e
        @error "Solver infrastructure validation failed: $e"
        rethrow(e)
    end

    println("Selected $n_sample target points using edge_source=$(edge_source), branch_index=$(branch_index), backend=gpr2d")

    BLAS.set_num_threads(1)
    best_rows = Dict{Int, Any}()
    working_rows = copy(all_rows)
    target_logs = target_logEI_values(EI_list; n_sample=n_sample)
    target_indices = sort(collect(1:n_sample), rev=true)
    last_anchor_xM = nothing
    solve_counter = 0
    chunk_size = parallel && nthreads() > 1 ? nthreads() : 1

    for chunk_start in 1:chunk_size:length(target_indices)
        chunk = target_indices[chunk_start:min(chunk_start + chunk_size - 1, length(target_indices))]
        chunk_anchors = Dict{Int, Union{Nothing, Float64}}()
        
        # Robust Anchor: Use a Median Slope-Corrected projection.
        # This accounts for the branch trend while being immune to single-point outliers.
        function get_robust_slope(history_rows)
            isempty(history_rows) && return 0.0
            rows_vec = sort(collect(history_rows); by=r->r.EI, rev=true)
            n_hist = length(rows_vec)
            if n_hist < 3 return 0.0 end

            slopes = Float64[]
            for i in 1:min(5, n_hist - 1)
                dx = rows_vec[i].xM_over_L - rows_vec[i+1].xM_over_L
                dy = log10(rows_vec[i].EI) - log10(rows_vec[i+1].EI)
                if abs(dy) > 1e-8 push!(slopes, dx / dy) end
            end
            return isempty(slopes) ? 0.0 : clamp(median(slopes), -0.25, 0.25)
        end

        m_slope = get_robust_slope(values(best_rows))
        
        # Determine the most recent solved point to use as a base for projection
        history_vec = sort(collect(values(best_rows)); by=r->r.EI, rev=true)
        base_xM = isempty(history_vec) ? last_anchor_xM : history_vec[1].xM_over_L
        base_logEI = isempty(history_vec) ? target_logs[chunk[1]] : log10(history_vec[1].EI)

        for s_idx in chunk
            # Project anchor using the robust slope to avoid "8-point lag"
            target_logEI = target_logs[s_idx]
            proj_anchor = isnothing(base_xM) ? nothing : base_xM + m_slope * (target_logEI - base_logEI)
            chunk_anchors[s_idx] = proj_anchor
            
            existing = best_row_for_sample(working_rows, s_idx; branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)
            if !isnothing(existing)
                # If we have a cached point, it is always the best anchor
                chunk_anchors[s_idx] = existing.xM_over_L
                base_xM = existing.xM_over_L
                base_logEI = target_logEI
            end
        end

        chunk_results = Vector{Any}(undef, length(chunk))
        if parallel && nthreads() > 1 && length(chunk) > 1
            # Pass a local snapshot of working_rows to the threaded loop to avoid MethodError issues
            # and ensure threads operate on a consistent state for this chunk.
            rows_snapshot = copy(working_rows)
            @threads for cidx in eachindex(chunk)
                s_idx = chunk[cidx]
                chunk_results[cidx] = refine_single_slice!(
                    s_idx,
                    target_logs[s_idx],
                    chunk_anchors[s_idx],
                    rows_snapshot,
                    artifact,
                    edge_source,
                    n_modes,
                    branch_index,
                    sweep_file,
                    mp_norm_list,
                    EI_list,
                    left_grid,
                    right_grid;
                    sa_accept_tol=sa_accept_tol,
                    local_max_iterations=local_max_iterations,
                    iteration_offset=solve_counter + (cidx - 1) * local_max_iterations,
                    tunnel_width=tunnel_width,
                )
            end
        else
            for cidx in eachindex(chunk)
                s_idx = chunk[cidx]
                chunk_results[cidx] = refine_single_slice!(
                    s_idx,
                    target_logs[s_idx],
                    chunk_anchors[s_idx],
                    working_rows,
                    artifact,
                    edge_source,
                    n_modes,
                    branch_index,
                    sweep_file,
                    mp_norm_list,
                    EI_list,
                    left_grid,
                    right_grid;
                    sa_accept_tol=sa_accept_tol,
                    local_max_iterations=local_max_iterations,
                    iteration_offset=solve_counter + (cidx - 1) * local_max_iterations,
                    tunnel_width=tunnel_width,
                )
            end
        end
        solve_counter += length(chunk) * local_max_iterations

        for (cidx, s_idx) in enumerate(chunk)
            result = chunk_results[cidx]
            for line in result.logs
                println(line)
            end
            if !isempty(result.new_rows)
                working_rows = merge_output_rows(working_rows, result.new_rows)
            end
            if !isnothing(result.best_row)
                best_rows[s_idx] = result.best_row
                last_anchor_xM = result.best_row.xM_over_L
            end
        end

        if !isempty(best_rows)
            println("Saving progress to $output_path...")
            current_final_rows = [best_rows[i] for i in sort(collect(keys(best_rows)))]
            written_curve_keys = append_curve_csv(output_path, current_final_rows, n_modes, written_curve_keys)
            written_beam_keys = append_beam_curve_csv(beam_output_path, current_final_rows, written_beam_keys)
        end
    end

    final_rows = [best_rows[i] for i in sort(collect(keys(best_rows)))]
    written_curve_keys = append_curve_csv(output_path, final_rows, n_modes, written_curve_keys)
    written_beam_keys = append_beam_curve_csv(beam_output_path, final_rows, written_beam_keys)
    
    overlay_name = "analyze_single_alpha_zero_curve_branch$(branch_index)_overlay.pdf"
    overlay_path = joinpath(output_dir, "figures", overlay_name)
    
    write_sa_overlay_plot(
        overlay_path,
        mp_norm_list,
        EI_list,
        left_grid,
        right_grid,
        final_rows;
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
    parallel = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : (nthreads() > 1)
    local_max_iterations = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 5
    tunnel_width = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 0.10
    sa_accept_tol = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 1e-3

    main(
        data_dir,
        sweep_file,
        String(edge_source),
        branch_index,
        n_sample,
        output_file,
        parallel;
        local_max_iterations=local_max_iterations,
        tunnel_width=tunnel_width,
        sa_accept_tol=sa_accept_tol,
    )
end

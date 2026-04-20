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

    # 1. Add base grid points
    for ie in eachindex(logEI_list), im in eachindex(mp_norm_list)
        push!(xtrain, Float64(mp_norm_list[im]))
        push!(ytrain, Float64(logEI_list[ie]))
        push!(alpha_train, Float64(alpha_grid[im, ie]))
        push!(sa_train, Float64(sa_ratio_grid[im, ie]))
    end

    # 2. Symmetry Pinning: add xM=0 points as hard zeros
    for ie in eachindex(logEI_list)
        push!(xtrain, 0.0)
        push!(ytrain, logEI_list[ie])
        push!(alpha_train, 0.0)
        push!(sa_train, -5.0)
    end

    if !isnothing(extra_points)
        append!(xtrain, extra_points.x)
        append!(ytrain, extra_points.y)
        append!(alpha_train, extra_points.alpha)
        append!(sa_train, extra_points.sa)
    end
    return xtrain, ytrain, alpha_train, sa_train, logEI_list
end

function surrogate_models(mp_norm_list, EI_list, left_grid::AbstractMatrix, right_grid::AbstractMatrix; extra_points=nothing)
    xtrain, ytrain, alpha_train, sa_train, _ =
        gp_training_points(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_points)
    return fit_gp2d(xtrain, ytrain, alpha_train), fit_gp2d(xtrain, ytrain, sa_train)
end

function nontrivial_candidates(candidates; boundary_band::Real=0.0)
    isempty(candidates) && return candidates
    # 1. Sort found roots by position
    sorted = sort(candidates; by = c -> c.xM_over_L)
    # 2. Smart Skip: skip center symmetry root (Branch 0)
    # We use 1.5% as a tight margin for the symmetry root.
    if !isempty(sorted) && sorted[1].xM_over_L < 0.015
        return sorted[2:end]
    end
    return sorted
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

function target_logEI_values(EI_list; n_sample::Int)
    logEI_list = sort(log10.(Float64.(EI_list)))
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

function cached_training_points(rows, edge_source::Symbol)
    # Filter only non-NaN results
    valid = filter(r -> !isnan(r.alpha), rows)
    return (
        x = [r.xM_over_L for r in valid],
        y = [log10(r.EI) for r in valid],
        alpha = [r.alpha for r in valid],
        sa = [get(r, :sa_ratio_beam, 0.0) for r in valid]
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

function format_complex(z)
    return "$(real(z)),$(imag(z)),$(abs(z)),$(rad2deg(angle(z)))"
end

function write_curve_csv(path::AbstractString, rows, n_modes::Int)
    open(path, "w") do io
        header = [
            "branch_index", "edge_source", "sweep_file", "iteration", "target_log10_EI",
            "sample_index", "curve_point_index", "EI", "log10_EI", "xM_over_L",
            "motor_position", "omega", "U", "power", "power_input", "thrust", "tail_flat_ratio",
            "eta_left_domain_re", "eta_left_domain_im", "eta_left_domain_abs", "eta_left_domain_phase_deg",
            "eta_right_domain_re", "eta_right_domain_im", "eta_right_domain_abs", "eta_right_domain_phase_deg",
            "eta_left_beam_re", "eta_left_beam_im", "eta_left_beam_abs", "eta_left_beam_phase_deg",
            "eta_right_beam_re", "eta_right_beam_im", "eta_right_beam_abs", "eta_right_beam_phase_deg",
            "alpha", "alpha_domain", "alpha_beam", "sa_ratio_domain", "sa_ratio_beam",
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

function write_beam_curve_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "log10_EI,xM_over_L,alpha_beam,sa_ratio_beam")
        for row in rows
            @printf(io, "%.6f,%.6f,%.6e,%.6f\n", log10(row.EI), row.xM_over_L, row.alpha_beam, row.sa_ratio_beam)
        end
    end
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
                EI = Float64(data[i, col("EI")]),
                xM_over_L = Float64(data[i, col("xM_over_L")]),
                alpha = Float64(data[i, col("alpha")]),
                alpha_beam = Float64(data[i, col("alpha_beam")]),
                sa_ratio_beam = Float64(data[i, col("sa_ratio_beam")]),
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

    alpha_domain = result.thrust # Standard solver thrust
    alpha_beam = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)
    S_domain = (result.metadata.args.eta_right_domain + result.metadata.args.eta_left_domain) / 2
    A_domain = (result.metadata.args.eta_right_domain - result.metadata.args.eta_left_domain) / 2
    sa_ratio_domain = log10(abs(S_domain) / (abs(A_domain) + eps()))
    S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
    A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
    sa_ratio_beam = log10(abs(S_beam) / (abs(A_beam) + eps()))

    alpha_selected = edge_source == :domain ? alpha_domain : alpha_beam

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
        U = result.U,
        power = result.power,
        power_input = result.metadata.args.power_input,
        thrust = result.thrust,
        tail_flat_ratio = result.tail_flat_ratio,
        eta_left_domain = result.metadata.args.eta_left_domain,
        eta_right_domain = result.metadata.args.eta_right_domain,
        eta_left_beam = metrics.eta_left_beam,
        eta_right_beam = metrics.eta_right_beam,
        alpha = alpha_selected,
        alpha_domain = alpha_domain,
        alpha_beam = alpha_beam,
        sa_ratio_domain = sa_ratio_domain,
        sa_ratio_beam = sa_ratio_beam,
        modal = modal
    )
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
    savefig(plt, path)
end

function default_output_file(sweep_file)
    return replace(basename(sweep_file), r"\.jld2$" => "_single_branch.csv")
end

function default_cache_file(sweep_file)
    return "second_family_point_cache.jld2"
end

function beam_csv_path(csv_path)
    return replace(csv_path, r"\.csv$" => "_beam.csv")
end

function main(data_dir::AbstractString, sweep_file::AbstractString, edge_source_in::AbstractString,
              branch_index::Int, n_sample::Int, output_file::AbstractString, parallel::Bool;
              cache_file=nothing, alpha_accept_tol::Float64=5e-3, n_modes::Int=8)
    
    data_dir = ensure_dir(normpath(data_dir))
    artifact = load_sweep_artifact(joinpath(data_dir, sweep_file))
    output_name = isempty(output_file) ? default_output_file(sweep_file) : String(output_file)
    output_path = joinpath(data_dir, output_name)
    edge_source = Symbol(edge_source_in)

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

    println("Selected $n_sample target points using edge_source=$(edge_source), branch_index=$(branch_index), backend=gpr2d")

    BLAS.set_num_threads(1)
    log_lock = ReentrantLock()
    max_iterations = 5
    max_solves_per_bin = 5
    logEI_bin_width = 0.03
    best_rows = Dict{Int, Any}()
    working_rows = copy(all_rows)
    target_logs = target_logEI_values(EI_list; n_sample=n_sample)

    # Main active learning loop
    for iteration in 1:max_iterations
        # 1. Propose candidates using refined surrogate
        # Within each global iteration, we do a "local" iteration on the GP (paper-only)
        # to ensure the candidates are as stable as possible before solving.
        local_working_rows = copy(working_rows)
        alpha_gp, sa_gp = nothing, nothing
        
        # Local "paper" refinement (hallucination prevention)
        for local_iter in 1:5
            relevant_training = training_rows(local_working_rows; edge_source=edge_source, sweep_file=sweep_file)
            extra_pts = isempty(relevant_training) ? nothing : cached_training_points(relevant_training, edge_source)
            alpha_gp, sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_pts)
            
            # Update local rows with the NEW GP guesses (fake solves with alpha=0)
            proposed_batch = NamedTuple[]
            for (s_idx, logEI) in enumerate(target_logs)
                candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=0.0)
                if length(candidates) >= branch_index
                    push!(proposed_batch, (; candidates[branch_index]..., sample_index=s_idx, curve_point_index=s_idx))
                end
            end
            fake_solves = [(; p..., alpha=0.0, sa_ratio_beam=p.sa_ratio, sweep_file=sweep_file, edge_source=String(edge_source)) for p in proposed_batch]
            local_working_rows = merge_output_rows(working_rows, fake_solves)
        end
        
        # 2. Identify which samples need REAL solving
        queue_indices = Int[]
        points_to_solve = Dict{Int, NamedTuple}()
        bin_counts = solve_counts_by_bin(working_rows; width=logEI_bin_width, branch_index=branch_index, edge_source=edge_source, sweep_file=sweep_file)

        for (sample_index, logEI) in enumerate(target_logs)
            candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=0.0)
            if length(candidates) < branch_index
                continue
            end
            point = candidates[branch_index]
            point = (; point..., sample_index=sample_index, curve_point_index=sample_index)
            
            # Check if we already have an acceptable result for this sample
            existing = get(best_rows, sample_index, nothing)
            if !isnothing(existing) && abs(existing.alpha) <= alpha_accept_tol
                continue
            end
            
            # Budget check
            bin = logEI_bin(logEI; width=logEI_bin_width)
            if get(bin_counts, bin, 0) < max_solves_per_bin
                push!(queue_indices, sample_index)
                points_to_solve[sample_index] = point
                bin_counts[bin] = get(bin_counts, bin, 0) + 1
            end
        end

        n_converged = n_sample - length(queue_indices)
        println("Iteration $(iteration): $(n_converged) / $(n_sample) points already converged.")

        if isempty(queue_indices)
            println("All points converged or budgets reached.")
            break
        end

        println("Running $(length(queue_indices)) cases...")
        batch_results = Vector{Any}(undef, length(queue_indices))
        
        if parallel && nthreads() > 1
            println("Running in parallel across $(nthreads()) threads")
            @threads for qidx in eachindex(queue_indices)
                s_idx = queue_indices[qidx]
                point = points_to_solve[s_idx]
                row = build_row(s_idx, point, artifact, edge_source, n_modes; branch_index=branch_index, sweep_file=sweep_file, iteration=iteration)
                batch_results[qidx] = row
                lock(log_lock) do
                    @printf("  case %d / %d: EI=%.4e, x_M/L=%.4f, alpha=%.6f\n", s_idx, n_sample, point.EI, point.xM_over_L, row.alpha)
                end
            end
        else
            for (qidx, s_idx) in enumerate(queue_indices)
                point = points_to_solve[s_idx]
                row = build_row(s_idx, point, artifact, edge_source, n_modes; branch_index=branch_index, sweep_file=sweep_file, iteration=iteration)
                batch_results[qidx] = row
                @printf("  case %d / %d: EI=%.4e, x_M/L=%.4f, alpha=%.6f\n", s_idx, n_sample, point.EI, point.xM_over_L, row.alpha)
            end
        end

        # 3. Update knowledge base
        new_batch_rows = filter(!isnothing, batch_results)
        working_rows = merge_output_rows(working_rows, new_batch_rows)
        for row in new_batch_rows
            existing = get(best_rows, row.sample_index, nothing)
            if isnothing(existing) || abs(row.alpha) < abs(existing.alpha)
                best_rows[row.sample_index] = row
            end
        end
        
        # Intermediate retraining of GP for the next GLOBAL iteration
        # relevant_training updated for next pass
    end

    # Final trace of the branch using refined surrogate
    relevant_training = training_rows(working_rows; edge_source=edge_source, sweep_file=sweep_file)
    extra_pts = cached_training_points(relevant_training, edge_source)
    alpha_gp, sa_gp = surrogate_models(mp_norm_list, EI_list, left_grid, right_grid; extra_points=extra_pts)
    
    final_sampled = NamedTuple[]
    for (sample_index, logEI) in enumerate(target_logs)
        candidates = branch_candidates_at_logEI(mp_norm_list, logEI, alpha_gp, sa_gp; boundary_band=0.0)
        if length(candidates) >= branch_index
            push!(final_sampled, (; candidates[branch_index]..., sample_index=sample_index, curve_point_index=sample_index))
        end
    end

    final_rows = [best_rows[i] for i in sort(collect(keys(best_rows)))]
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
    
    main(data_dir, sweep_file, String(edge_source), branch_index, n_sample, output_file, parallel)
end

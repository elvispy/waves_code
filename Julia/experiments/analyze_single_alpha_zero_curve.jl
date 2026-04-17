using Surferbot
using Statistics
using LinearAlgebra

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
    asymmetry = beam_asymmetry.(left_grid, right_grid)
    S_grid = (right_grid .+ left_grid) ./ 2
    A_grid = (right_grid .- left_grid) ./ 2
    SA_ratio = log10.(abs.(S_grid) ./ (abs.(A_grid) .+ eps()))

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

function select_lowest_curve(crossings; sa_filter::Symbol=:negative)
    by_EI = Dict{Float64, Vector{NamedTuple}}()
    for cross in crossings
        push!(get!(by_EI, cross.EI, NamedTuple[]), cross)
    end

    selected = NamedTuple[]
    for EI in sort(collect(keys(by_EI)))
        candidates = by_EI[EI]
        if sa_filter == :negative
            candidates = filter(c -> c.sa_ratio < 0, candidates)
        elseif sa_filter == :positive
            candidates = filter(c -> c.sa_ratio > 0, candidates)
        elseif sa_filter != :none
            error("sa_filter must be :negative, :positive, or :none")
        end
        isempty(candidates) && continue
        push!(selected, reduce((a, b) -> a.xM_over_L <= b.xM_over_L ? a : b, candidates))
    end
    return selected
end

function sample_curve_points(curve_points; n_sample::Int)
    isempty(curve_points) && error("No curve points were available for sampling.")
    count = min(n_sample, length(curve_points))
    idx = unique(round.(Int, range(1, length(curve_points); length=count)))
    return curve_points[idx]
end

function tail_flat_ratio(result)
    left_count = max(1, ceil(Int, 0.05 * length(result.eta)))
    tail = abs.(result.eta[1:left_count])
    return std(tail) / max(eps(), mean(tail))
end

function format_complex(z)
    return string(real(z), ",", imag(z), ",", abs(z), ",", rad2deg(angle(z)))
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

function main(
    data_dir::AbstractString=joinpath(@__DIR__, "..", "output");
    sweep_file::AbstractString="sweep_motorPosition_EI_coupled_from_matlab.jld2",
    edge_source::Symbol=:domain,
    sa_filter::Symbol=:negative,
    n_sample::Int=12,
    n_modes::Int=8,
    output_file::AbstractString="single_alpha_zero_curve_details.csv",
)
    data_dir = ensure_dir(normpath(data_dir))
    artifact = load_sweep_artifact(joinpath(data_dir, sweep_file))

    summaries = artifact.summaries
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft

    left_grid = similar(summaries, ComplexF64)
    right_grid = similar(summaries, ComplexF64)
    for idx in eachindex(summaries)
        left_grid[idx], right_grid[idx] = edge_fields(summaries[idx], edge_source)
    end

    crossings = collect_zero_crossings_refined(mp_norm_list, EI_list, left_grid, right_grid)
    curve_points = select_lowest_curve(crossings; sa_filter=sa_filter)
    sampled = sample_curve_points(curve_points; n_sample=n_sample)

    println("Selected $(length(curve_points)) points on the extracted curve")
    println("Sampling $(length(sampled)) points from $(basename(joinpath(data_dir, sweep_file))) using edge_source=$(edge_source), sa_filter=$(sa_filter)")

    rows = NamedTuple[]
    for (sample_index, point) in enumerate(sampled)
        params = apply_parameter_overrides(
            artifact.base_params,
            (motor_position = point.xM_over_L * artifact.base_params.L_raft, EI = point.EI),
        )
        println("  case $sample_index / $(length(sampled)): EI=$(point.EI), x_M/L=$(round(point.xM_over_L; digits=4))")
        result = flexible_solver(params)
        modal = decompose_raft_freefree_modes(result; num_modes=n_modes, verbose=false)
        metrics = beam_edge_metrics(result)
        S_domain = (metrics.eta_right_domain + metrics.eta_left_domain) / 2
        A_domain = (metrics.eta_right_domain - metrics.eta_left_domain) / 2
        S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
        A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
        push!(rows, (
            sample_index = sample_index,
            curve_point_index = findfirst(==(point), curve_points),
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
            alpha = edge_source == :domain ?
                beam_asymmetry(metrics.eta_left_domain, metrics.eta_right_domain) :
                beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam),
            alpha_domain = beam_asymmetry(metrics.eta_left_domain, metrics.eta_right_domain),
            alpha_beam = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam),
            sa_ratio_domain = log10(abs(S_domain) / (abs(A_domain) + eps())),
            sa_ratio_beam = log10(abs(S_beam) / (abs(A_beam) + eps())),
            modal = modal,
        ))
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
    main(data_dir; sweep_file=sweep_file, edge_source=edge_source, sa_filter=sa_filter, n_sample=n_sample, output_file=output_file)
end

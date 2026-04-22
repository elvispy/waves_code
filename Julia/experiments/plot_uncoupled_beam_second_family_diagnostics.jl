
using Surferbot
using JLD2
using Plots
using LinearAlgebra
using Printf

# Purpose: visualize uncoupled beam-end second-family diagnostics from the
# native Julia sweep artifact and compare the numerical alpha=0 contour against
# a first-order analytic S=0 predictor.

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function beam_fields(artifact)
    summaries = artifact.summaries
    eta_left = map(s -> s.eta_left_beam, summaries)
    eta_right = map(s -> s.eta_right_beam, summaries)
    S = (eta_right .+ eta_left) ./ 2
    A = (eta_right .- eta_left) ./ 2
    alpha = beam_asymmetry.(eta_left, eta_right)
    log_ratio = log10.(abs.(S) ./ (abs.(A) .+ eps()))
    cos_gap = real.(S .* conj.(A)) ./ (abs.(S) .* abs.(A) .+ eps())
    return (; eta_left, eta_right, S, A, alpha, log_ratio, cos_gap)
end

function labeled_ticks(values::AbstractVector{<:Real})
    return collect(zip(values, string.(values)))
end

function all_positive_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
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
    return unique!(roots)
end

function load_reference_basis(output_dir::AbstractString, cache_file::AbstractString)
    cache_path = joinpath(output_dir, cache_file)
    isfile(cache_path) || error("Missing cache file: $cache_path")
    cache = load(cache_path)
    isempty(cache) && error("Cache file is empty: $cache_path")
    key = first(sort!(collect(keys(cache))))
    payload = cache[key]
    return (; beta=Float64.(payload.beta), Psi=payload.Psi, x_raft=Float64.(payload.x_raft), rho_raft=Float64(payload.rho_raft), omega=Float64(payload.omega))
end

function endpoint_weights(basis, combination::Symbol)
    left = vec(basis.Psi[1, :])
    right = vec(basis.Psi[end, :])
    if combination == :S
        return (left .+ right) ./ 2
    elseif combination == :A
        return (right .- left) ./ 2
    elseif combination == :L
        return left
    elseif combination == :R
        return right
    end
    error("Unsupported endpoint combination: $combination")
end

function endpoint_weights_from_lr(left::AbstractVector, right::AbstractVector, combination::Symbol)
    if combination == :S
        return (left .+ right) ./ 2
    elseif combination == :A
        return (right .- left) ./ 2
    elseif combination == :L
        return left
    elseif combination == :R
        return right
    end
    error("Unsupported endpoint combination: $combination")
end

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    n_points = length(xi_motor)
    Phi = zeros(Float64, n_points, max_mode + 1)
    beta = zeros(Float64, max_mode + 1)

    xi_end = [0.0]
    Phi_end = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_end[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_end[1, 2] = -L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            beta[n + 1] = betaL_el[n - 1] / L
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_end[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_end, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, beta, Phi_end=vec(Phi_end))
end

function analytic_second_family_points(
    artifact,
    output_dir::AbstractString;
    cache_file::AbstractString="second_family_point_cache.jld2",
    mode_numbers::Tuple{Vararg{Int}}=(0, 2),
    combination::Symbol=:S,
)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI = log10.(EI_list)
    basis = load_reference_basis(output_dir, cache_file)
    n = collect(0:(size(basis.Psi, 2) - 1))
    mode_idx = findall(j -> n[j] in mode_numbers, eachindex(n))
    xM_norm = basis.x_raft ./ params.L_raft
    W_end = endpoint_weights(basis, combination)

    Dfun(EI, β) = EI * β^4 - basis.rho_raft * basis.omega^2
    
    pts_logEI = Float64[]
    pts_xM = Float64[]
    
    for i in eachindex(EI_list)
        EI = EI_list[i]
        vals = Float64[]
        for row in axes(basis.Psi, 1)
            s = 0.0
            for j in mode_idx
                s += basis.Psi[row, j] * W_end[j] / Dfun(EI, basis.beta[j])
            end
            push!(vals, s)
        end
        roots = all_positive_roots(xM_norm, vals)
        for r in roots
            push!(pts_logEI, logEI[i])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function analytic_second_family_points_pure(
    artifact;
    mode_numbers::Tuple{Vararg{Int}}=(0, 2),
    combination::Symbol=:S,
)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI = log10.(EI_list)
    # Use a finer motor position grid for root finding
    xM_norm_grid = collect(range(-0.5, 0.5, length=1000))
    raw = raw_mode_shapes(params, xM_norm_grid; max_mode=maximum(mode_numbers))
    mode_idx = [n + 1 for n in mode_numbers]
    W_end = raw.Phi_end[mode_idx]
    
    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2

    # Reconstruct the original raft grid to compute G
    k_res = Surferbot.dispersion_k(params.omega, params.g, 0.05, params.nu, params.sigma, params.rho)
    n_guess = max(80, ceil(Int, 80 / (2 * pi / real(k_res)) * params.L_raft))
    n_raft = n_guess + mod(n_guess, 2) + 1
    x_raft = collect(range(-params.L_raft/2, params.L_raft/2, length=n_raft))
    
    w = Surferbot.trapz_weights(x_raft)
    raw_grid = raw_mode_shapes(params, x_raft ./ params.L_raft; max_mode=maximum(mode_numbers))
    Phi_subset = raw_grid.Phi[:, mode_idx]
    G = Phi_subset' * (Phi_subset .* w)
    G_inv = inv(G)

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for i in eachindex(EI_list)
        EI = EI_list[i]
        D_inv = diagm(0 => [1.0 / Dfun(EI, raw.beta[j]) for j in mode_idx])
        transfer = W_end' * D_inv * G_inv
        
        vals = Float64[]
        for row in eachindex(xM_norm_grid)
            F_W_xM = raw.Phi[row, mode_idx]
            push!(vals, dot(transfer, F_W_xM))
        end
        roots = all_positive_roots(xM_norm_grid, vals)
        for r in roots
            push!(pts_logEI, logEI[i])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function save_heatmap(path::AbstractString, logEI, xM_norm, field; title::AbstractString, cbar_title::AbstractString, clim=nothing, color=:viridis, colorbar_ticks=nothing)
    plt = heatmap(
        logEI,
        xM_norm,
        field;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title=title,
        colorbar_title=cbar_title,
        color=color,
        size=(900, 700),
        dpi=200,
        clim=clim,
        colorbar_ticks=colorbar_ticks,
    )
    savefig(plt, path)
    return path
end

function save_labeled_ratio_contour(path::AbstractString, logEI, xM_norm, field)
    levels = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    ticks = labeled_ticks(levels)
    ywb = cgrad([RGB(0.95, 0.82, 0.15), RGB(1.0, 1.0, 1.0), RGB(0.2, 0.45, 0.9)])
    clamped = clamp.(field, first(levels), last(levels))
    plt = contourf(
        logEI,
        xM_norm,
        clamped;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title="Uncoupled beam-end log10(|S| / |A|)",
        colorbar_title="log10(|S|/|A|)",
        color=ywb,
        levels=levels,
        size=(900, 700),
        dpi=200,
        colorbar_ticks=ticks,
    )
    contour!(
        plt,
        logEI,
        xM_norm,
        clamped;
        levels=levels,
        color=:black,
        linewidth=0.6,
        contour_labels=true,
        label=nothing,
    )
    savefig(plt, path)
    return path
end

function save_alpha_overlay(
    path::AbstractString,
    logEI::AbstractVector{<:Real},
    xM_norm::AbstractVector{<:Real},
    alpha::AbstractMatrix{<:Real};
    analytic_02=nothing,
    analytic_0246=nothing,
    analytic_A13=nothing,
    analytic_A1357=nothing,
    analytic_L=nothing,
    analytic_R=nothing,
)
    plt = heatmap(
        logEI,
        xM_norm,
        alpha;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title="Numerical alpha with a priori S=0 and A=0 predictors",
        colorbar_title="alpha",
        color=:RdBu,
        size=(900, 700),
        dpi=200,
        legend=:topright,
    )
    contour!(
        plt,
        logEI,
        xM_norm,
        alpha;
        levels=[0.0],
        color=:black,
        linewidth=2,
        label="numerical alpha=0",
    )
    # Numerical alpha = +/- 1 contours
    contour!(
        plt,
        logEI,
        xM_norm,
        alpha;
        levels=[-0.99, 0.99],
        color=[:red, :blue],
        linewidth=2,
        label=["numerical alpha=-1" "numerical alpha=1"],
    )

    if !isnothing(analytic_02)
        scatter!(
            plt,
            analytic_02.logEI,
            analytic_02.xM_norm;
            color=:gold,
            markersize=2,
            markerstrokewidth=0,
            label="a priori 0+2 (S=0)",
        )
    end
    if !isnothing(analytic_0246)
        scatter!(
            plt,
            analytic_0246.logEI,
            analytic_0246.xM_norm;
            color=:dodgerblue,
            markersize=2,
            markerstrokewidth=0,
            label="a priori 0+2+4+6 (S=0)",
        )
    end
    if !isnothing(analytic_A13)
        scatter!(
            plt,
            analytic_A13.logEI,
            analytic_A13.xM_norm;
            color=:magenta,
            markersize=2,
            markerstrokewidth=0,
            label="a priori 1+3 (A=0)",
        )
    end
    if !isnothing(analytic_A1357)
        scatter!(
            plt,
            analytic_A1357.logEI,
            analytic_A1357.xM_norm;
            color=:limegreen,
            markersize=2,
            markerstrokewidth=0,
            label="a priori 1+3+5+7 (A=0)",
        )
    end
    if !isnothing(analytic_L)
        scatter!(
            plt,
            analytic_L.logEI,
            analytic_L.xM_norm;
            color=:blue,
            markersize=2,
            markerstrokewidth=0,
            label="a priori L=0",
        )
    end
    if !isnothing(analytic_R)
        scatter!(
            plt,
            analytic_R.logEI,
            analytic_R.xM_norm;
            color=:red,
            markersize=2,
            markerstrokewidth=0,
            label="a priori R=0",
        )
    end
    savefig(plt, path)
    return path
end

function save_alpha_overlay_pure(
    path::AbstractString,
    logEI::AbstractVector{<:Real},
    xM_norm::AbstractVector{<:Real},
    alpha::AbstractMatrix{<:Real};
    analytic_02=nothing,
    analytic_0246=nothing,
    analytic_A13=nothing,
    analytic_A1357=nothing,
)
    plt = heatmap(
        logEI,
        xM_norm,
        alpha;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title="Numerical alpha with pure analytic a priori S=0 and A=0",
        colorbar_title="alpha",
        color=:RdBu,
        size=(900, 700),
        dpi=200,
        legend=:topright,
    )
    contour!(
        plt,
        logEI,
        xM_norm,
        alpha;
        levels=[0.0],
        color=:black,
        linewidth=2,
        label="numerical alpha=0",
    )
    if !isnothing(analytic_02)
        scatter!(
            plt,
            analytic_02.logEI,
            analytic_02.xM_norm;
            color=:gold,
            markersize=2,
            markerstrokewidth=0,
            label="pure a priori 0+2 (S=0)",
        )
    end
    if !isnothing(analytic_0246)
        scatter!(
            plt,
            analytic_0246.logEI,
            analytic_0246.xM_norm;
            color=:dodgerblue,
            markersize=2,
            markerstrokewidth=0,
            label="pure a priori 0+2+4+6 (S=0)",
        )
    end
    if !isnothing(analytic_A13)
        scatter!(
            plt,
            analytic_A13.logEI,
            analytic_A13.xM_norm;
            color=:magenta,
            markersize=2,
            markerstrokewidth=0,
            label="pure a priori 1+3 (A=0)",
        )
    end
    if !isnothing(analytic_A1357)
        scatter!(
            plt,
            analytic_A1357.logEI,
            analytic_A1357.xM_norm;
            color=:limegreen,
            markersize=2,
            markerstrokewidth=0,
            label="pure a priori 1+3+5+7 (A=0)",
        )
    end
    savefig(plt, path)
    return path
end

function main(
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output");
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    cache_file::AbstractString="second_family_point_cache.jld2",
)
    output_dir = ensure_dir(normpath(output_dir))
    artifact = load_sweep(joinpath(output_dir, sweep_file))
    fields = beam_fields(artifact)

    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI = log10.(EI_list)
    xM_norm = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    # S=0 predictors
    analytic_02 = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 2), combination=:S)
    analytic_0246 = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 2, 4, 6), combination=:S)
    
    # A=0 predictors
    analytic_A13 = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(1, 3), combination=:A)
    analytic_A1357 = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(1, 3, 5, 7), combination=:A)
    
    # Pure analytic a priori
    analytic_pure_02 = analytic_second_family_points_pure(artifact; mode_numbers=(0, 2), combination=:S)
    analytic_pure_0246 = analytic_second_family_points_pure(artifact; mode_numbers=(0, 2, 4, 6), combination=:S)
    analytic_pure_A13 = analytic_second_family_points_pure(artifact; mode_numbers=(1, 3), combination=:A)
    analytic_pure_A1357 = analytic_second_family_points_pure(artifact; mode_numbers=(1, 3, 5, 7), combination=:A)
    
    # L=0 and R=0 predictors
    analytic_L = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 1, 2, 3), combination=:L)
    analytic_R = analytic_second_family_points(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 1, 2, 3), combination=:R)

    ratio_path = joinpath(output_dir, "uncoupled_beam_log10_S_over_A.pdf")
    cos_path = joinpath(output_dir, "uncoupled_beam_cos_phase_gap.pdf")
    overlay_path = joinpath(output_dir, "uncoupled_beam_alpha_with_analytic_overlay.pdf")
    pure_overlay_path = joinpath(output_dir, "uncoupled_beam_alpha_with_pure_analytic_overlay.pdf")

    save_labeled_ratio_contour(ratio_path, logEI, xM_norm, fields.log_ratio)
    save_heatmap(
        cos_path,
        logEI,
        xM_norm,
        fields.cos_gap;
        title="Uncoupled beam-end cos(arg(S) - arg(A))",
        cbar_title="cos phase gap",
        clim=(-1, 1),
        color=:RdBu,
        colorbar_ticks=labeled_ticks([-1.0, 0.0, 1.0]),
    )
    save_alpha_overlay(
        overlay_path,
        logEI,
        xM_norm,
        fields.alpha;
        analytic_02=analytic_02,
        analytic_0246=analytic_0246,
        analytic_A13=analytic_A13,
        analytic_A1357=analytic_A1357,
        analytic_L=analytic_L,
        analytic_R=analytic_R,
    )
    save_alpha_overlay_pure(
        pure_overlay_path,
        logEI,
        xM_norm,
        fields.alpha;
        analytic_02=analytic_pure_02,
        analytic_0246=analytic_pure_0246,
        analytic_A13=analytic_pure_A13,
        analytic_A1357=analytic_pure_A1357,
    )

    println("Saved $ratio_path")
    println("Saved $cos_path")
    println("Saved $overlay_path")
    println("Saved $pure_overlay_path")
    return (ratio=ratio_path, cos=cos_path, overlay=overlay_path, pure_overlay=pure_overlay_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
    main(output_dir)
end

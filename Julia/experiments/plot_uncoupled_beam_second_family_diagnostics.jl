using Surferbot
using JLD2
using Plots
using LinearAlgebra

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
    end
    error("Unsupported endpoint combination: $combination")
end

function endpoint_weights_from_lr(left::AbstractVector, right::AbstractVector, combination::Symbol)
    if combination == :S
        return (left .+ right) ./ 2
    elseif combination == :A
        return (right .- left) ./ 2
    end
    error("Unsupported endpoint combination: $combination")
end

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    # xM_norm is x_M / L, where x_M is distance from center.
    # So x_M / L in [-0.5, 0.5].
    # Local xi for mode shape formula should be in [0, L].
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    n_points = length(xi_motor)
    Phi = zeros(Float64, n_points, max_mode + 1)
    beta = zeros(Float64, max_mode + 1)

    # We also need values at the left end (xi=0) for S calculation
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
        betaL_el = freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            beta[n + 1] = betaL_el[n - 1] / L
            Phi[:, n + 1] .= freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_end[1, n + 1] = freefree_mode_shape(xi_end, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, beta, Phi_end=vec(Phi_end))
end

function analytic_second_family_branch(
    artifact,
    output_dir::AbstractString;
    cache_file::AbstractString="second_family_point_cache.jld2",
    branch_index::Int=1,
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
    roots = fill(NaN, length(EI_list))
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
        roots[i] = first_positive_root(xM_norm, vals; branch_index=branch_index)
    end
    return (; logEI, xM_norm=roots)
end

function analytic_second_family_branch_pure(
    artifact;
    branch_index::Int=1,
    mode_numbers::Tuple{Vararg{Int}}=(0, 2),
    combination::Symbol=:S,
)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI = log10.(EI_list)
    # Use a finer motor position grid for root finding
    xM_norm_grid = collect(range(-0.5, 0.5, length=500))
    raw = raw_mode_shapes(params, xM_norm_grid; max_mode=maximum(mode_numbers))
    mode_idx = [n + 1 for n in mode_numbers]
    
    # W_end: values at the raft end (xi=0)
    W_end = raw.Phi_end[mode_idx]
    
    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2

    # We must account for the non-orthogonality of W_n on the discrete grid.
    # The true balance is: G * D * q^W = -F^W
    # So q^W = - D^-1 * G^-1 * F^W
    # S = (W_end)^T * q^W = - (W_end)^T * D^-1 * G^-1 * F^W(x_M)
    # where F^W_n \propto W_n(x_M)
    
    # Reconstruct the original raft grid to compute G
    N = round(Int, (80 - 1) * 3 * params.L_raft / params.L_raft) + 1 # Approximate logic, better to load the actual grid
    # To be perfectly rigorous, we should compute G on the exact raft grid used in the solve.
    # We can reconstruct it simply:
    x_raft = collect(range(-params.L_raft/2, params.L_raft/2, length=ceil(Int, 80 / (2 * pi / real(Surferbot.dispersion_k(params.omega, params.g, 0.05, params.nu, params.sigma, params.rho))) * params.L_raft) + mod(ceil(Int, 80 / (2 * pi / real(Surferbot.dispersion_k(params.omega, params.g, 0.05, params.nu, params.sigma, params.rho))) * params.L_raft), 2) + 1))
    
    w = Surferbot.trapz_weights(x_raft)
    raw_grid = raw_mode_shapes(params, x_raft ./ params.L_raft; max_mode=maximum(mode_numbers))
    Phi_subset = raw_grid.Phi[:, mode_idx]
    G = Phi_subset' * (Phi_subset .* w)
    G_inv = inv(G)

    roots = fill(NaN, length(EI_list))
    for i in eachindex(EI_list)
        EI = EI_list[i]
        
        # Build the D^-1 matrix
        D_inv = diagm(0 => [1.0 / Dfun(EI, raw.beta[j]) for j in mode_idx])
        
        # The transfer matrix from F^W to S
        transfer = W_end' * D_inv * G_inv
        
        vals = Float64[]
        for row in eachindex(xM_norm_grid)
            F_W_xM = raw.Phi[row, mode_idx]
            s = dot(transfer, F_W_xM)
            push!(vals, s)
        end
        roots[i] = first_positive_root(xM_norm_grid, vals; branch_index=branch_index)
    end
    return (; logEI, xM_norm=roots)
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

function save_contourf(path::AbstractString, logEI, xM_norm, field; title::AbstractString, cbar_title::AbstractString, levels, color=:viridis, colorbar_ticks=nothing)
    plt = contourf(
        logEI,
        xM_norm,
        field;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title=title,
        colorbar_title=cbar_title,
        color=color,
        levels=levels,
        size=(900, 700),
        dpi=200,
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

function save_alpha_overlay(path::AbstractString, logEI, xM_norm, alpha; analytic_02, analytic_0246, analytic_A13, analytic_A1357)
    plt = heatmap(
        logEI,
        xM_norm,
        alpha;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title="Numerical alpha with a posteriori (Psi basis) S=0 and A=0 predictors",
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
    mask_02 = isfinite.(analytic_02.xM_norm)
    plot!(
        plt,
        analytic_02.logEI[mask_02],
        analytic_02.xM_norm[mask_02];
        color=:gold,
        linewidth=3,
        label="a posteriori 0+2 (Psi)",
    )
    mask_0246 = isfinite.(analytic_0246.xM_norm)
    plot!(
        plt,
        analytic_0246.logEI[mask_0246],
        analytic_0246.xM_norm[mask_0246];
        color=:dodgerblue,
        linewidth=3,
        label="a posteriori 0+2+4+6 (Psi)",
    )
    mask_A13 = isfinite.(analytic_A13.xM_norm)
    any(mask_A13) && plot!(
        plt,
        analytic_A13.logEI[mask_A13],
        analytic_A13.xM_norm[mask_A13];
        color=:magenta,
        linewidth=3,
        linestyle=:dash,
        label="a posteriori A=0 (1+3, Psi)",
    )
    mask_A1357 = isfinite.(analytic_A1357.xM_norm)
    any(mask_A1357) && plot!(
        plt,
        analytic_A1357.logEI[mask_A1357],
        analytic_A1357.xM_norm[mask_A1357];
        color=:limegreen,
        linewidth=3,
        linestyle=:dash,
        label="a posteriori A=0 (1+3+5+7, Psi)",
    )
    savefig(plt, path)
    return path
end

function save_alpha_overlay_pure(path::AbstractString, logEI, xM_norm, alpha; analytic_02, analytic_0246, analytic_A13, analytic_A1357)
    plt = heatmap(
        logEI,
        xM_norm,
        alpha;
        xlabel="log10(EI)",
        ylabel="x_M / L",
        title="Numerical alpha with a priori (W basis) S=0 predictor",
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
    mask_02 = isfinite.(analytic_02.xM_norm)
    any(mask_02) && plot!(
        plt,
        analytic_02.logEI[mask_02],
        analytic_02.xM_norm[mask_02];
        color=:gold,
        linewidth=3,
        label="a priori 0+2 (W)",
    )
    mask_0246 = isfinite.(analytic_0246.xM_norm)
    any(mask_0246) && plot!(
        plt,
        analytic_0246.logEI[mask_0246],
        analytic_0246.xM_norm[mask_0246];
        color=:dodgerblue,
        linewidth=3,
        label="a priori 0+2+4+6 (W)",
    )
    mask_A13 = isfinite.(analytic_A13.xM_norm)
    any(mask_A13) && plot!(
        plt,
        analytic_A13.logEI[mask_A13],
        analytic_A13.xM_norm[mask_A13];
        color=:magenta,
        linewidth=3,
        linestyle=:dash,
        label="a priori A=0 (1+3, W)",
    )
    mask_A1357 = isfinite.(analytic_A1357.xM_norm)
    any(mask_A1357) && plot!(
        plt,
        analytic_A1357.logEI[mask_A1357],
        analytic_A1357.xM_norm[mask_A1357];
        color=:limegreen,
        linewidth=3,
        linestyle=:dash,
        label="a priori A=0 (1+3+5+7, W)",
    )
    savefig(plt, path)
    return path
end

"""
    main(output_dir=joinpath(@__DIR__, "..", "output");
         sweep_file="sweep_motor_position_EI_uncoupled_from_matlab.jld2")

Write three diagnostic figures for the uncoupled beam-end second-family
analysis:
- `uncoupled_beam_log10_S_over_A.pdf`
- `uncoupled_beam_cos_phase_gap.pdf`
- `uncoupled_beam_alpha_with_analytic_overlay.pdf`

Inputs:
- `output_dir`: directory containing the sweep artifact and receiving figures.
- `sweep_file`: native uncoupled Julia sweep artifact.
"""
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
    analytic_02 = analytic_second_family_branch(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 2), combination=:S)
    analytic_0246 = analytic_second_family_branch(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 2, 4, 6), combination=:S)
    analytic_A13 = analytic_second_family_branch(artifact, output_dir; cache_file=cache_file, branch_index=1, mode_numbers=(1, 3), combination=:A)
    analytic_A1357 = analytic_second_family_branch(artifact, output_dir; cache_file=cache_file, branch_index=1, mode_numbers=(1, 3, 5, 7), combination=:A)
    analytic_pure_02 = analytic_second_family_branch_pure(artifact; branch_index=1, mode_numbers=(0, 2), combination=:S)
    analytic_pure_0246 = analytic_second_family_branch_pure(artifact; branch_index=1, mode_numbers=(0, 2, 4, 6), combination=:S)
    analytic_pure_A13 = analytic_second_family_branch_pure(artifact; branch_index=1, mode_numbers=(1, 3), combination=:A)
    analytic_pure_A1357 = analytic_second_family_branch_pure(artifact; branch_index=1, mode_numbers=(1, 3, 5, 7), combination=:A)
    cos_ticks = labeled_ticks([-1.0, 0.0, 1.0])

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
        colorbar_ticks=cos_ticks,
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

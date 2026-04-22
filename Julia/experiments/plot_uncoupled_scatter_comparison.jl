using Surferbot
using JLD2
using Plots
using DelimitedFiles
using LinearAlgebra
using Printf

# Purpose: Generate two separate scatter plots for the uncoupled case:
# 1. A Priori (Reduced Order Model roots)
# 2. A Posteriori (Numerical simulation roots)
# Aesthetics copied from plot_uncoupled_beam_second_family_diagnostics.jl

function all_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            push!(roots, Float64(xs[i]))
        elseif a * b < 0
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            push!(roots, Float64(root))
        end
    end
    return unique!(roots)
end

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    n_points = length(xi_motor)
    Phi = zeros(Float64, n_points, max_mode + 1)
    
    xi_end_left = [0.0]
    xi_end_right = [L]
    Phi_left = zeros(Float64, 1, max_mode + 1)
    Phi_right = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_left[1, 1] = 1.0
    Phi_right[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_left[1, 2] = -L / 2
        Phi_right[1, 2] = L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_left[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_end_left, L, betaL_el[n - 1])[1]
            Phi_right[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_end_right, L, betaL_el[n - 1])[1]
        end
    end
    return (; Phi, Phi_left=vec(Phi_left), Phi_right=vec(Phi_right), beta_roots = [0.0; 0.0; betaL_el ./ L])
end

function get_apriori_roots(artifact, combination::Symbol; mode_numbers=(0, 2), n_ei=400)
    params = artifact.base_params
    EI_list_raw = collect(Float64.(artifact.parameter_axes.EI))
    logEI_fine = collect(range(log10(minimum(EI_list_raw)), log10(maximum(EI_list_raw)), length=n_ei))
    EI_list = 10.0 .^ logEI_fine
    
    xM_norm_grid = collect(range(-0.5, 0.5, length=1000))
    raw = raw_mode_shapes(params, xM_norm_grid; max_mode=maximum(mode_numbers))
    
    mode_idx = [n + 1 for n in mode_numbers]
    
    # Compute mass matrix G
    k_res = Surferbot.dispersion_k(params.omega, params.g, 0.05, params.nu, params.sigma, params.rho)
    n_guess = max(80, ceil(Int, 80 / (2 * pi / real(k_res)) * params.L_raft))
    n_raft = n_guess + mod(n_guess, 2) + 1
    x_raft = collect(range(-params.L_raft/2, params.L_raft/2, length=n_raft))
    w = Surferbot.trapz_weights(x_raft)
    raw_grid = raw_mode_shapes(params, x_raft ./ params.L_raft; max_mode=maximum(mode_numbers))
    Phi_subset = raw_grid.Phi[:, mode_idx]
    G = Phi_subset' * (Phi_subset .* w)
    G_inv = inv(G)

    W_L = raw.Phi_left[mode_idx]
    W_R = raw.Phi_right[mode_idx]
    W_end = if combination == :S
        (W_L .+ W_R) ./ 2
    elseif combination == :A
        (W_R .- W_L) ./ 2
    elseif combination == :L
        W_L
    elseif combination == :R
        W_R
    else
        error("Unknown combination: $combination")
    end

    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for i in eachindex(EI_list)
        EI = EI_list[i]
        D_inv = diagm(0 => [1.0 / Dfun(EI, raw.beta_roots[j]) for j in mode_idx])
        transfer = W_end' * D_inv * G_inv
        
        vals = Float64[]
        for row in eachindex(xM_norm_grid)
            F_Phi_xM = raw.Phi[row, mode_idx]
            push!(vals, dot(transfer, F_Phi_xM))
        end
        roots = all_roots(xM_norm_grid, vals)
        for r in roots
            push!(pts_logEI, logEI_fine[i])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function get_aposteriori_contours(artifact, level::Real; n_ei=400)
    summaries = artifact.summaries
    eta_left = map(s -> s.eta_left_beam, summaries)
    eta_right = map(s -> s.eta_right_beam, summaries)
    
    alpha = zeros(size(summaries))
    for i in eachindex(summaries)
        al, ar = abs(eta_left[i]), abs(eta_right[i])
        alpha[i] = (ar^2 - al^2) / (ar^2 + al^2 + 1e-15)
    end
    
    EI_list_raw = collect(Float64.(artifact.parameter_axes.EI))
    logEI_raw = log10.(EI_list_raw)
    xM_norm = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    logEI_fine = collect(range(minimum(logEI_raw), maximum(logEI_raw), length=n_ei))
    
    pts_logEI = Float64[]
    pts_xM = Float64[]
    
    for j in eachindex(logEI_fine)
        target_logEI = logEI_fine[j]
        idx = searchsortedfirst(logEI_raw, target_logEI)
        alpha_interp = if idx == 1
            alpha[:, 1]
        elseif idx > length(logEI_raw)
            alpha[:, end]
        else
            t = (target_logEI - logEI_raw[idx-1]) / (logEI_raw[idx] - logEI_raw[idx-1])
            (1-t) .* alpha[:, idx-1] .+ t .* alpha[:, idx]
        end
        
        vals = alpha_interp .- level
        roots = all_roots(xM_norm, vals)
        for r in roots
            push!(pts_logEI, target_logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function main(; show_aposteriori=false)
    # Paths
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_path = joinpath(output_dir, "sweep_motor_position_EI_uncoupled_from_matlab.jld2")
    
    !isfile(sweep_path) && error("Missing sweep artifact: $sweep_path")
    artifact = load_sweep(sweep_path)
    
    # Get alpha matrix for background heatmap
    summaries = artifact.summaries
    eta_left = map(s -> s.eta_left_beam, summaries)
    eta_right = map(s -> s.eta_right_beam, summaries)
    alpha_mat = zeros(size(summaries))
    for i in eachindex(summaries)
        al, ar = abs(eta_left[i]), abs(eta_right[i])
        alpha_mat[i] = (ar^2 - al^2) / (ar^2 + al^2 + 1e-15)
    end
    
    EI_list_raw = collect(Float64.(artifact.parameter_axes.EI))
    logEI_raw = log10.(EI_list_raw)
    xM_norm_raw = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    # Determine limits
    ei_min, ei_max = minimum(logEI_raw), maximum(logEI_raw)
    xm_min, xm_max = minimum(xM_norm_raw), maximum(xM_norm_raw)

    println("Calculating A Priori roots...")
    # alpha=0 is satisfied if S=0 (even modes) OR A=0 (odd modes)
    apriori_S0 = get_apriori_roots(artifact, :S; mode_numbers=(0, 2, 4, 6), n_ei=400)
    apriori_A0 = get_apriori_roots(artifact, :A; mode_numbers=(1, 3, 5, 7), n_ei=400)
    
    apriori_L = get_apriori_roots(artifact, :L; mode_numbers=(0, 1, 2, 3, 4, 5, 6, 7), n_ei=400)
    apriori_R = get_apriori_roots(artifact, :R; mode_numbers=(0, 1, 2, 3, 4, 5, 6, 7), n_ei=400)
    
    ms = 5 # markersize
    
    # Common plot settings
    plt_opts = (
        xlabel="log10(EI)",
        ylabel="x_M / L",
        xlim=(ei_min, ei_max),
        ylim=(xm_min, xm_max),
        grid=true,
        legend=:topright,
        size=show_aposteriori ? (1000, 1400) : (1000, 700),
        color=:RdBu,
        colorbar_title="alpha"
    )

    # 1. A Priori Plot
    p1 = heatmap(
        logEI_raw, xM_norm_raw, alpha_mat;
        title="A Priori Estimates (Reduced Order Model)",
        plt_opts...
    )
    # Scatter both S=0 and A=0 for the alpha=0 state
    scatter!(p1, apriori_S0.logEI, apriori_S0.xM_norm, label="alpha = 0 (S=0 ROM)", color=:black, markersize=ms, markershape=:circle, markerstrokewidth=0)
    scatter!(p1, apriori_A0.logEI, apriori_A0.xM_norm, label=nothing, color=:black, markersize=ms, markershape=:circle, markerstrokewidth=0)
    
    scatter!(p1, apriori_L.logEI, apriori_L.xM_norm, label="alpha = 1 (L=0 ROM)", color=:blue, markersize=ms, markershape=:rect, markerstrokewidth=0)
    scatter!(p1, apriori_R.logEI, apriori_R.xM_norm, label="alpha = -1 (R=0 ROM)", color=:red, markersize=ms, markershape=:utriangle, markerstrokewidth=0)
    
    if show_aposteriori
        println("Extracting A Posteriori roots...")
        aposteriori_0 = get_aposteriori_contours(artifact, 0.0; n_ei=400)
        aposteriori_L = get_aposteriori_contours(artifact, 0.99; n_ei=400)
        aposteriori_R = get_aposteriori_contours(artifact, -0.99; n_ei=400)

        # 2. A Posteriori Plot
        p2 = heatmap(
            logEI_raw, xM_norm_raw, alpha_mat;
            title="A Posteriori Estimates (Numerical Crossings)",
            plt_opts...
        )
        scatter!(p2, aposteriori_0.logEI, aposteriori_0.xM_norm, label="alpha = 0 (Numerical)", color=:black, markersize=ms, markershape=:circle, markerstrokewidth=0)
        scatter!(p2, aposteriori_L.logEI, aposteriori_L.xM_norm, label="alpha = 1 (Numerical)", color=:blue, markersize=ms, markershape=:rect, markerstrokewidth=0)
        scatter!(p2, aposteriori_R.logEI, aposteriori_R.xM_norm, label="alpha = -1 (Numerical)", color=:red, markersize=ms, markershape=:utriangle, markerstrokewidth=0)
        
        combined = plot(p1, p2, layout=(2, 1))
    else
        combined = p1
    end
    
    save_path = joinpath(output_dir, "uncoupled_scatter_comparison.pdf")
    savefig(combined, save_path)
    println("Saved scatter comparison to $save_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

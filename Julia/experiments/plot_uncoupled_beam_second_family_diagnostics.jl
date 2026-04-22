using Surferbot
using JLD2
using Plots
using LinearAlgebra
using Printf
using DelimitedFiles

# Purpose: final corrected uncoupled diagnostics.
# Integral Law: Root finding using Gram-Schmidt orthonormal coefficients (Psi basis).
# Theoretical Law: Root finding using analytical ROM (W basis).

function all_positive_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            xs[i] > 1e-6 && push!(roots, Float64(xs[i]))
        elseif a * b < 0
            # Linear interpolation for high-precision crossing
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            root > 1e-6 && push!(roots, Float64(root))
        end
    end
    return unique!(roots)
end

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    Phi = zeros(Float64, length(xi_motor), max_mode + 1)
    
    xi_L = [0.0]
    xi_R = [L]
    Phi_L = zeros(Float64, 1, max_mode + 1)
    Phi_R = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_L[1, 1] = 1.0
    Phi_R[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_L[1, 2] = -L / 2
        Phi_R[1, 2] = L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_L[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_L, L, betaL_el[n - 1])[1]
            Phi_R[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_R, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, Phi_L=vec(Phi_L), Phi_R=vec(Phi_R))
end

# Find integral roots using the orthonormal Psi basis from the summaries
function get_all_roots_integral_psi(
    artifact;
    mode_numbers=(0, 2),
    combination::Symbol=:S,
)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    mp_norm_list = collect(Float64.(artifact.parameter_axes.motor_position)) ./ params.L_raft

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for ie in eachindex(EI_list)
        col_vals = Float64[]
        for im in eachindex(mp_norm_list)
            sum_row = artifact.summaries[im, ie]
            
            # The summaries don't store full Psi coefficients.
            # But they store eta_left_beam and eta_right_beam.
            # IN THE INTEGRAL SENSE: S = (L+R)/2, A = (R-L)/2
            # These ARE the projections onto the orthonormal basis for symmetric/antisymmetric modes.
            L, R = sum_row.eta_left_beam, sum_row.eta_right_beam
            
            val = if combination == :S
                (L + R) / 2
            elseif combination == :A
                (R - L) / 2
            elseif combination == :L
                L
            elseif combination == :R
                R
            end
            push!(col_vals, real(val))
        end

        # Use interpolation to find the "true" numerical prediction crossing
        roots = all_positive_roots(mp_norm_list, col_vals)
        for r in roots
            push!(pts_logEI, logEI_axis[ie])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function get_all_roots_theoretical(
    artifact;
    mode_numbers=(0, 2),
    combination::Symbol=:S,
)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid = collect(range(0.0, 0.48, length=1000))
    raw = raw_mode_shapes(params, xM_grid; max_mode=maximum(mode_numbers))
    mode_idx = [n + 1 for n in mode_numbers]
    
    # End-weights at L/2 (Theoretical ROM)
    weights = if combination == :S
        (raw.Phi_L[mode_idx] .+ raw.Phi_R[mode_idx]) ./ 2
    elseif combination == :A
        (raw.Phi_R[mode_idx] .- raw.Phi_L[mode_idx]) ./ 2
    elseif combination == :L
        raw.Phi_L[mode_idx]
    elseif combination == :R
        raw.Phi_R[mode_idx]
    end
    
    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta_roots = [0.0; 0.0; betaL ./ params.L_raft]

    # Reconstruct G for Theoretical projection
    x_raft = collect(range(-params.L_raft/2, params.L_raft/2, length=201))
    w = Surferbot.trapz_weights(x_raft)
    raw_grid = raw_mode_shapes(params, x_raft ./ params.L_raft; max_mode=maximum(mode_numbers))
    G = raw_grid.Phi[:, mode_idx]' * (raw_grid.Phi[:, mode_idx] .* w)
    G_inv = inv(G)

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for i in eachindex(EI_list)
        EI = EI_list[i]
        D_inv = diagm(0 => [1.0 / Dfun(EI, beta_roots[n+1]) for n in mode_numbers])
        transfer = weights' * D_inv * G_inv
        
        vals = Float64[]
        for row in eachindex(xM_grid)
            F_W_xM = raw.Phi[row, mode_idx]
            push!(vals, dot(transfer, F_W_xM))
        end
        roots = all_positive_roots(xM_grid, vals)
        for r in roots
            push!(pts_logEI, logEI_axis[i])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_file = "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
    artifact = load_sweep(joinpath(output_dir, sweep_file))
    
    logEI_axis = log10.(collect(Float64.(artifact.parameter_axes.EI)))
    xM_axis = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    summaries = artifact.summaries
    alpha = beam_asymmetry.(map(s->s.eta_left_beam, summaries), map(s->s.eta_right_beam, summaries))
    
    # 1. Integral search (Direct numerical crossings using interpolation)
    println("Searching for all Integral branches...")
    int_S = get_all_roots_integral_psi(artifact; combination=:S)
    int_A = get_all_roots_integral_psi(artifact; combination=:A)
    int_L = get_all_roots_integral_psi(artifact; combination=:L)
    int_R = get_all_roots_integral_psi(artifact; combination=:R)
    
    # 2. Theoretical search (Algebraic ROM)
    println("Searching for all Theoretical branches...")
    theo_S02 = get_all_roots_theoretical(artifact; mode_numbers=(0, 2), combination=:S)
    theo_S0246 = get_all_roots_theoretical(artifact; mode_numbers=(0, 2, 4, 6), combination=:S)
    theo_A13 = get_all_roots_theoretical(artifact; mode_numbers=(1, 3), combination=:A)
    theo_A1357 = get_all_roots_theoretical(artifact; mode_numbers=(1, 3, 5, 7), combination=:A)
    theo_L = get_all_roots_theoretical(artifact; mode_numbers=(0, 1, 2, 3, 4, 5, 6, 7), combination=:L)
    theo_R = get_all_roots_theoretical(artifact; mode_numbers=(0, 1, 2, 3, 4, 5, 6, 7), combination=:R)

    plt_opts = (
        xlabel="log10(EI)", ylabel="x_M / L",
        colorbar_title="alpha", color=:RdBu, size=(900, 700), dpi=200, legend=:topright,
        xlims=(minimum(logEI_axis), maximum(logEI_axis)), ylims=(0.0, 0.48), clim=(-1,1)
    )

    # 3. Integral Plot
    p3 = heatmap(logEI_axis, xM_axis, alpha; title="Numerical alpha with INTEGRAL modal predictors", plt_opts...)
    scatter!(p3, int_S.logEI, int_S.xM_norm; color=:black, markersize=5, markerstrokewidth=0, label="int alpha=0")
    scatter!(p3, int_A.logEI, int_A.xM_norm; color=:black, markersize=5, markerstrokewidth=0, label=nothing)
    scatter!(p3, int_L.logEI, int_L.xM_norm; color=:blue, markersize=5, markerstrokewidth=0, label="int L=0")
    scatter!(p3, int_R.logEI, int_R.xM_norm; color=:red, markersize=5, markerstrokewidth=0, label="int R=0")
    savefig(p3, joinpath(output_dir, "plot_uncoupled_beam_second_family_diagnostics_3.pdf"))

    # 4. Theoretical Plot
    p4 = heatmap(logEI_axis, xM_axis, alpha; title="Numerical alpha with THEORETICAL modal predictors", plt_opts...)
    scatter!(p4, theo_S02.logEI, theo_S02.xM_norm; color=:gold, markersize=5, markerstrokewidth=0, label="theo 0+2 (S=0)")
    scatter!(p4, theo_S0246.logEI, theo_S0246.xM_norm; color=:dodgerblue, markersize=5, markerstrokewidth=0, label="theo 0+2+4+6 (S=0)")
    scatter!(p4, theo_A13.logEI, theo_A13.xM_norm; color=:magenta, markersize=5, markerstrokewidth=0, label="theo 1+3 (A=0)")
    scatter!(p4, theo_A1357.logEI, theo_A1357.xM_norm; color=:limegreen, markersize=5, markerstrokewidth=0, label="theo 1+3+5+7 (A=0)")
    scatter!(p4, theo_L.logEI, theo_L.xM_norm; color=:blue, markersize=5, markerstrokewidth=0, label="theo L=0")
    scatter!(p4, theo_R.logEI, theo_R.xM_norm; color=:red, markersize=5, markerstrokewidth=0, label="theo R=0")
    savefig(p4, joinpath(output_dir, "plot_uncoupled_beam_second_family_diagnostics_4.pdf"))

    println("Saved _3.pdf and _4.pdf with full multi-branch scatter using Psi-orthonormal integral.")
end

main()

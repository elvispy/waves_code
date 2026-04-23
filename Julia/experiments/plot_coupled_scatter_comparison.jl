using Surferbot
using JLD2
using CSV
using DataFrames
using Plots
using DelimitedFiles
using LinearAlgebra
using Printf

"""
    plot_coupled_scatter_comparison.jl

This definitive script generates the final two-panel comparison for the
coupled case.
- The TOP panel shows the THEORETICAL prediction (uncoupled ROM baseline).
- The BOTTOM panel shows the INTEGRAL ground truth, using the full-grid
  modal coefficients from the master CSV.
"""

function all_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]; b = vals[i + 1]
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

function get_integral_roots_from_csv(df::DataFrame, combination::Symbol, mode_numbers)
    pts_logEI = Float64[]
    pts_xM = Float64[]
    
    # End-weights for the analytical W basis
    w_ends = [iseven(n) ? 1.0 : -1.0 for n in mode_numbers]
    weights = if combination == :S
        [iseven(n) ? 1.0 : 0.0 for n in mode_numbers]
    elseif combination == :A
        [isodd(n) ? 1.0 : 0.0 for n in mode_numbers]
    elseif combination == :L
        [(-1.0)^n for n in mode_numbers]
    elseif combination == :R
        ones(length(mode_numbers))
    end
    
    for group in groupby(df, :log10_EI)
        logEI = first(group.log10_EI)
        xM_slice = group.xM_over_L
        
        col_vals = Float64[]
        for row in eachrow(group)
            val = 0.0 + 0.0im
            for (i, n) in enumerate(mode_numbers)
                q_re = row[Symbol("q_w$(n)_re")]
                q_im = row[Symbol("q_w$(n)_im")]
                qn = complex(q_re, q_im)
                val += qn * weights[i]
            end
            push!(col_vals, real(val))
        end
        
        roots = all_roots(xM_slice, col_vals)
        for r in roots
            push!(pts_logEI, logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end


function get_theoretical_roots(artifact, combination::Symbol, mode_numbers)
    params = artifact.base_params
    logEI_fine = collect(range(log10(minimum(artifact.parameter_axes.EI)), log10(maximum(artifact.parameter_axes.EI)), length=400))
    EI_list = 10.0 .^ logEI_fine
    
    xM_grid = collect(range(0.0, 0.49, length=500))
    
    L = params.L_raft
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta_roots = [0.0; 0.0; betaL ./ L]
    
    Phi_xM = hcat([n == 0 ? ones(length(xM_grid)) : (n == 1 ? (xM_grid .* L) : Surferbot.Modal.freefree_mode_shape(xM_grid .* L .+ L/2, L, betaL[n-1])) for n in mode_numbers]...)
    
    w_ends = if combination == :S
        [iseven(n) ? 1.0 : 0.0 for n in mode_numbers]
    elseif combination == :A
        [isodd(n) ? 1.0 : 0.0 for n in mode_numbers]
    elseif combination == :L
        [(-1.0)^n for n in mode_numbers]
    elseif combination == :R
        ones(length(mode_numbers))
    end
    
    x_raft_G = collect(range(-L/2, L/2, length=201))
    w_trapz_G = Surferbot.trapz_weights(x_raft_G)
    Phi_G = hcat([n == 0 ? ones(201) : (n == 1 ? x_raft_G : Surferbot.Modal.freefree_mode_shape(x_raft_G .+ L/2, L, betaL[n-1])) for n in mode_numbers]...)
    G = Phi_G' * (Phi_G .* w_trapz_G)
    G_inv = inv(G)
    
    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2
    
    pts_logEI = Float64[]
    pts_xM = Float64[]
    
    for (i, EI) in enumerate(EI_list)
        D_inv = diagm(0 => [1.0 / Dfun(EI, beta_roots[n+1]) for n in mode_numbers])
        transfer = w_ends' * D_inv * G_inv
        
        vals = [dot(transfer, Phi_xM[row, :]) for row in 1:size(Phi_xM, 1)]
        roots = all_roots(xM_grid, vals)
        for r in roots
            push!(pts_logEI, logEI_fine[i])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end


function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_path = joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    csv_path = joinpath(output_dir, "csv", "sweeper_coupled_full_grid.csv")
    figure_path = joinpath(output_dir, "figures", "plot_coupled_scatter_comparison.pdf")

    !isfile(sweep_path) && error("Sweep artifact not found: $sweep_path")
    !isfile(csv_path) && error("Full-grid modal CSV not found: $csv_path. Run experiments/sweeper_modal_coefficients.jl first.")
    
    artifact = load_sweep(sweep_path)
    df = CSV.read(csv_path, DataFrame)
    
    alpha_mat = beam_asymmetry.(map(s->s.eta_left_beam, artifact.summaries), map(s->s.eta_right_beam, artifact.summaries))
    logEI_raw = log10.(collect(Float64.(artifact.parameter_axes.EI)))
    xM_norm_raw = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft

    println("Calculating Theoretical roots (Qn=0)...")
    theo_S = get_theoretical_roots(artifact, :S, (0, 2, 4, 6))
    theo_A = get_theoretical_roots(artifact, :A, (1, 3, 5, 7))

    println("Calculating Integral roots from CSV...")
    int_S = get_integral_roots_from_csv(df, :S, 0:7)
    int_A = get_integral_roots_from_csv(df, :A, 0:7)
    int_L = get_integral_roots_from_csv(df, :L, 0:7)
    int_R = get_integral_roots_from_csv(df, :R, 0:7)

    plt_opts = (xlabel="log10(EI)", ylabel="x_M / L", colorbar_title="alpha", c=:balance, size=(1000, 1400), dpi=300, legend=:topright,
                interpolate=true, levels=31, xlims=extrema(logEI_raw), ylims=extrema(xM_norm_raw), clims=(-1,1))

    p1 = heatmap(logEI_raw, xM_norm_raw, alpha_mat; title="Theoretical (Uncoupled ROM Baseline)", plt_opts...)
    contour!(p1, logEI_raw, xM_norm_raw, alpha_mat; levels=[0.0], color=:white, linewidth=2, label="Numerical alpha=0")
    scatter!(p1, theo_S.logEI, theo_S.xM_norm, label="theo S=0", color=:black, markersize=4, markerstrokewidth=0)
    scatter!(p1, theo_A.logEI, theo_A.xM_norm, label="theo A=0", color=:magenta, markersize=4, markerstrokewidth=0)
    
    p2 = heatmap(logEI_raw, xM_raw_raw, alpha_mat; title="Integral (Coupled Numerical Projections)", plt_opts...)
    contour!(p2, logEI_raw, xM_norm_raw, alpha_mat; levels=[0.0], color=:white, linewidth=2, label="Numerical alpha=0")
    scatter!(p2, int_S.logEI, int_S.xM_norm, label="int S=0", color=:black, markersize=4, markerstrokewidth=0)
    scatter!(p2, int_A.logEI, int_A.xM_norm, label="int A=0", color=:magenta, markersize=4, markerstrokewidth=0)
    scatter!(p2, int_L.logEI, int_L.xM_norm, label="int L=0", color=:blue, markersize=4, markerstrokewidth=0)
    scatter!(p2, int_R.logEI, int_R.xM_norm, label="int R=0", color=:red, markersize=4, markerstrokewidth=0)

    combined = plot(p1, p2, layout=(2,1))
    savefig(combined, figure_path)
    println("Saved final comparison plot to $figure_path")
end

main()

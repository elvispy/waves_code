using Surferbot
using JLD2
using Plots
using DelimitedFiles
using LinearAlgebra
using Printf

# Purpose: Generate two separate scatter plots for the COUPLED case:
# 1. Theoretical (Reduced Order Model with established law)
# 2. Integral (Numerical simulation crossings from refined CSV q_w, F_w)

function all_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
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

function get_integral_roots_coupled(output_dir::AbstractString; combination::Symbol=:S)
    csv_file = joinpath(output_dir, "single_alpha_zero_curve_details_coupled_refined.csv")
    if !isfile(csv_file)
        @warn "Coupled refined CSV not found. Skipping Integral points."
        return nothing
    end

    data, header = readdlm(csv_file, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)
    
    n_pts = size(data, 1)
    pts_logEI = Float64[]
    pts_xM = Float64[]
    
    for i in 1:n_pts
        # The ONLY mechanism: use q_w and F_w
        val = 0.0 + 0.0im
        for n in 0:7
            # In the coupled case, q_w is the true integrated numerical displacement
            qn = complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")])
            
            # End-weights (Analytical W_n at L/2)
            # S = sum(q_even), A = sum(q_odd)
            weight = if combination == :S
                iseven(n) ? 1.0 : 0.0
            elseif combination == :A
                isodd(n) ? 1.0 : 0.0
            elseif combination == :L
                (-1.0)^n
            elseif combination == :R
                1.0
            end
            val += qn * weight
        end
        
        # In the refined branch CSV, these points are already the roots (val ~ 0)
        # So we just return the (logEI, xM) trajectory
        push!(pts_logEI, data[i, col("log10_EI")])
        push!(pts_xM, data[i, col("xM_over_L")])
    end
    
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function get_theoretical_roots_coupled(artifact, combination::Symbol; mode_numbers=(0, 2), n_ei=400)
    params = artifact.base_params
    EI_list_raw = collect(Float64.(artifact.parameter_axes.EI))
    logEI_fine = collect(range(log10(minimum(EI_list_raw)), log10(maximum(EI_list_raw)), length=n_ei))
    EI_list = 10.0 .^ logEI_fine
    
    xM_grid = collect(range(0.0, 0.48, length=500))
    
    # 1. Theoretical components
    L = params.L_raft
    xi_motor = xM_grid .* L .+ L/2
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta_roots = [0.0; 0.0; betaL ./ L]
    
    # Phi matrix on motor grid
    Phi = zeros(Float64, length(xi_motor), 8)
    Phi[:, 1] .= 1.0
    Phi[:, 2] .= xi_motor .- L/2
    for n in 2:7; Phi[:, n+1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL[n-1]); end
    
    # End-weights
    w_ends = if combination == :S
        [iseven(n) ? 1.0 : 0.0 for n in 0:7]
    elseif combination == :A
        [isodd(n) ? 1.0 : 0.0 for n in 0:7]
    elseif combination == :L
        [(-1.0)^n for n in 0:7]
    elseif combination == :R
        ones(8)
    end

    # G reconstruction for non-orthogonality
    x_raft = collect(range(-L/2, L/2, length=201))
    w_trapz = Surferbot.trapz_weights(x_raft)
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=8)
    G = raw_basis.Phi' * (raw_basis.Phi .* w_trapz)
    G_inv = inv(G)

    # Law: |Qf| = C * L^-1.27 * d^-0.11 * omega^-0.28 * |qn|^-0.14
    # (Simplified coupling for this version: Q = -F tracking)
    # The true a-priori balance is (D_beam + Z_fluid) q = -F
    Dfun(EI, β) = EI * β^4 - params.rho_raft * params.omega^2

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for i in eachindex(EI_list)
        EI = EI_list[i]
        # Diagonal stiffness/mass matrix
        D_inv = diagm(0 => [1.0 / Dfun(EI, beta_roots[n+1]) for n in 0:7])
        
        # S = w_ends' * q = w_ends' * D_inv * G_inv * F(xM)
        transfer = w_ends' * D_inv * G_inv
        
        vals = Float64[]
        for row in eachindex(xM_grid)
            F_vec = Phi[row, :] # Modal force at xM
            push!(vals, dot(transfer, F_vec))
        end
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
    sweep_path = joinpath(output_dir, "sweep_motor_position_EI_coupled_from_matlab.jld2")
    artifact = load_sweep(sweep_path)
    
    summaries = artifact.summaries
    alpha_mat = beam_asymmetry.(map(s->s.eta_left_beam, summaries), map(s->s.eta_right_beam, summaries))
    logEI_raw = log10.(collect(Float64.(artifact.parameter_axes.EI)))
    xM_norm_raw = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    # 1. Theoretical (Qn=0 check)
    println("Calculating Theoretical roots...")
    theo_S = get_theoretical_roots_coupled(artifact, :S)
    theo_A = get_theoretical_roots_coupled(artifact, :A)
    
    # 2. Integral (Numerical crossings from CSV q_w)
    println("Extracting Integral roots...")
    int_S = get_integral_roots_coupled(output_dir; combination=:S)
    int_A = get_integral_roots_coupled(output_dir; combination=:A)

    plt_opts = (
        xlabel="log10(EI)", ylabel="x_M / L",
        colorbar_title="alpha", color=:RdBu, size=(1000, 1400), dpi=200, legend=:topright,
        xlims=(minimum(logEI_raw), maximum(logEI_raw)), ylims=(0.0, 0.48), clim=(-1,1)
    )

    # Top Plot: Theoretical (ROM)
    p1 = heatmap(logEI_raw, xM_norm_raw, alpha_mat; title="Theoretical Estimates (Coupled ROM)", plt_opts...)
    scatter!(p1, theo_S.logEI, theo_S.xM_norm, label="theo alpha=0 (S=0)", color=:black, markersize=5, markerstrokewidth=0)
    scatter!(p1, theo_A.logEI, theo_A.xM_norm, label="theo A=0", color=:magenta, markersize=5, markerstrokewidth=0)

    # Bottom Plot: Integral (Numerical q_w)
    p2 = heatmap(logEI_raw, xM_norm_raw, alpha_mat; title="Integral Estimates (Numerical Projections)", plt_opts...)
    if !isnothing(int_S)
        scatter!(p2, int_S.logEI, int_S.xM_norm, label="int S=0", color=:black, markersize=5, markerstrokewidth=0)
    end
    if !isnothing(int_A)
        scatter!(p2, int_A.logEI, int_A.xM_norm, label="int A=0", color=:magenta, markersize=5, markerstrokewidth=0)
    end

    combined = plot(p1, p2, layout=(2,1))
    save_path = joinpath(output_dir, "plot_coupled_scatter_comparison.pdf")
    savefig(combined, save_path)
    println("Saved combined coupled comparison to $save_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

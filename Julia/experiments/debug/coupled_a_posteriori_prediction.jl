using Surferbot
using JLD2
using Plots
using Statistics
using Printf
using LinearAlgebra

"""
coupled_a_posteriori_prediction.jl

Performs a full-grid analysis of the hydrodynamic pressure contribution (Q) 
to radiation cancellation in the coupled case. Decomposes the symmetric 
radiation into mechanical and hydrodynamic components.
"""

# Cluster Job: Analyze the modal contribution of pressure (Q) to the second family zeros.
# This script is designed to run the full grid and save the results for later analysis.

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_file = joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    cache_file = joinpath(output_dir, "jld2", "second_family_point_cache.jld2")
    results_cache = joinpath(output_dir, "jld2", "coupled_a_posteriori_prediction_results.jld2")
    
    if !isfile(sweep_file)
        println("Sweep file not found: $sweep_file")
        return
    end
    
    println("Loading coupled artifact...")
    artifact = load_sweep(sweep_file)
    
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI = log10.(EI_list)
    xM_list = collect(Float64.(artifact.parameter_axes.motor_position))
    xM_norm = xM_list ./ artifact.base_params.L_raft
    
    n_EI = length(EI_list)
    n_xM = length(xM_list)
    
    # Check if we already have results
    if isfile(results_cache)
        println("Loading existing results from $results_cache...")
        data = load(results_cache)
        S_mech = data["S_mech"]
        S_hydro = data["S_hydro"]
        alpha_vals = data["alpha_vals"]
    else
        S_mech = zeros(ComplexF64, n_xM, n_EI)
        S_hydro = zeros(ComplexF64, n_xM, n_EI)
        alpha_vals = zeros(Float64, n_xM, n_EI)
        
        # Load reference metadata for mode selection
        ref_cache = load(cache_file)
        payload = ref_cache[first(keys(ref_cache))]
        even_idx = findall(iseven, payload.n)
        
        println("Starting full grid solve ($(n_EI * n_xM) points)...")
        for i in 1:n_EI
            EI = EI_list[i]
            for j in 1:n_xM
                xM = xM_list[j]
                params = apply_parameter_overrides(artifact.base_params, (EI=EI, motor_position=xM))
                
                # Full solver and decomposition
                result = flexible_solver(params)
                modal = decompose_raft_freefree_modes(result; num_modes=length(payload.n), verbose=false)
                
                # Correct discrete D_n balance for the Psi basis
                # D_n q_n = Q_n - F_n  => q_n = (Q_n - F_n) / D_n
                # S = sum( q_n * W_end_n )
                W_end_psi = modal.Psi[1, :] 
                D = [EI * modal.beta[k]^4 - artifact.base_params.rho_raft * params.omega^2 for k in eachindex(modal.beta)]
                
                s_m = 0.0im
                s_h = 0.0im
                for k in even_idx
                    s_m += (-modal.F[k] / D[k]) * W_end_psi[k]
                    s_h += (modal.Q[k] / D[k]) * W_end_psi[k]
                end
                
                S_mech[j, i] = s_m
                S_hydro[j, i] = s_h
                metrics = beam_edge_metrics(result)
                alpha_vals[j, i] = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)
            end
            @printf("Progress: %d/%d EI slices done (%.1f%%)\n", i, n_EI, i/n_EI*100)
            
            # Intermediate save every 10 slices
            if i % 10 == 0
                save(results_cache, Dict("S_mech" => S_mech, "S_hydro" => S_hydro, "alpha_vals" => alpha_vals))
            end
        end
        save(results_cache, Dict("S_mech" => S_mech, "S_hydro" => S_hydro, "alpha_vals" => alpha_vals))
    end
    
    # 2. Plotting
    println("Generating plots...")
    S_total = S_mech .+ S_hydro
    
    p1 = heatmap(logEI, xM_norm, alpha_vals, title="Numerical Alpha", color=:RdBu, clim=(-0.5, 0.5))
    contour!(p1, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=2, label="alpha=0")
    
    p2 = heatmap(logEI, xM_norm, real.(S_total), title="Re(S_mech + S_hydro)", color=:RdBu)
    contour!(p2, logEI, xM_norm, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="S_total=0")
    contour!(p2, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=1, linestyle=:dash, label="alpha=0")
    
    p3 = heatmap(logEI, xM_norm, alpha_vals, title="Pressure Shift Analysis", color=:RdBu, clim=(-0.2, 0.2))
    contour!(p3, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=2, label="Numerical alpha=0")
    contour!(p3, logEI, xM_norm, real.(S_mech), levels=[0.0], color=:gold, linewidth=2, label="Mechanical only (S_F=0)")
    contour!(p3, logEI, xM_norm, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="Total a posteriori (S_F+S_Q=0)")

    combined = plot(p1, p2, p3, layout=(1,3), size=(1500, 500))
    path = joinpath(output_dir, "figures", "coupled_a_posteriori_prediction.pdf")
    savefig(combined, path)
    println("Saved $path")
    end

main()

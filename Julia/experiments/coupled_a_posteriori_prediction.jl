using Surferbot
using JLD2
using Plots
using Statistics
using Printf
using LinearAlgebra

# Analyze the modal contribution of pressure (Q) to the second family zeros.

function main()
    output_dir = "Julia/output"
    sweep_file = joinpath(output_dir, "sweep_motor_position_EI_coupled_from_matlab.jld2")
    cache_file = joinpath(output_dir, "second_family_point_cache.jld2")
    
    # Use a small grid for local verification, full grid for cluster
    smoke_test = false 
    
    if !isfile(sweep_file)
        println("Sweep file not found: $sweep_file")
        return
    end
    
    println("Loading coupled artifact...")
    artifact = load_sweep(sweep_file)
    
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    xM_list = collect(Float64.(artifact.parameter_axes.motor_position))
    
    if smoke_test
        println("--- SMOKE TEST MODE (5x5 grid) ---")
        EI_list = EI_list[1:12:end]
        xM_list = xM_list[1:5:end]
    end
    
    logEI = log10.(EI_list)
    xM_norm = xM_list ./ artifact.base_params.L_raft
    
    n_EI = length(EI_list)
    n_xM = length(xM_list)
    
    # Grid for plots
    S_mech = zeros(ComplexF64, n_xM, n_EI)
    S_hydro = zeros(ComplexF64, n_xM, n_EI)
    alpha_vals = zeros(Float64, n_xM, n_EI)
    
    # Basis constants
    cache = load(cache_file)
    payload = cache[first(keys(cache))]
    W_end = payload.Psi[1, :]
    even_idx = findall(iseven, payload.n)
    
    println("Sampling coupled points and decomposing...")
    # To keep it fast, we'll sample a subset of the grid or just use the whole grid if n_EI*n_xM is small.
    # 57 x 25 = 1425 points. A bit slow but manageable.
    
    for i in 1:n_EI
        EI = EI_list[i]
        for j in 1:n_xM
            xM = xM_list[j]
            params = apply_parameter_overrides(artifact.base_params, (EI=EI, motor_position=xM))
            result = flexible_solver(params)
            modal = decompose_raft_freefree_modes(result; num_modes=length(payload.n), verbose=false)
            
            D = [EI * modal.beta[k]^4 - payload.rho_raft * payload.omega^2 for k in eachindex(modal.beta)]
            
            # Mechanical: -F/D
            # Hydro: Q/D
            s_m = 0.0im
            s_h = 0.0im
            for k in even_idx
                s_m += (-modal.F[k] / D[k]) * W_end[k]
                s_h += (modal.Q[k] / D[k]) * W_end[k]
            end
            
            S_mech[j, i] = s_m
            S_hydro[j, i] = s_h
            alpha_vals[j, i] = result.thrust # alpha_beam is what we usually track
            # Actually alpha_beam from metrics
            metrics = beam_edge_metrics(result)
            alpha_vals[j, i] = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)
        end
        if i % 10 == 0
            @printf("Progress: %d/%d EI slices done\n", i, n_EI)
        end
    end
    
    S_total = S_mech .+ S_hydro
    
    # 2. Plotting
    println("Generating plots...")
    p1 = heatmap(logEI, xM_norm, alpha_vals, title="Numerical Coupled Alpha", color=:RdBu, clim=(-0.5, 0.5))
    contour!(p1, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=2, label="alpha=0")
    
    p2 = heatmap(logEI, xM_norm, real.(S_total), title="Re(S_mech + S_hydro)", color=:RdBu, clim=(-1e-6, 1e-6))
    contour!(p2, logEI, xM_norm, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="S_total=0")
    contour!(p2, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=1, linestyle=:dash, label="alpha=0")
    
    p3 = heatmap(logEI, xM_norm, alpha_vals, title="Pressure Shift (Coupled Case)", color=:RdBu, clim=(-0.2, 0.2))
    contour!(p3, logEI, xM_norm, alpha_vals, levels=[0.0], color=:black, linewidth=2, label="Numerical alpha=0")
    contour!(p3, logEI, xM_norm, real.(S_mech), levels=[0.0], color=:gold, linewidth=2, label="Mechanical only (S_F=0)")
    contour!(p3, logEI, xM_norm, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="Total a posteriori (S_F+S_Q=0)")

    combined = plot(p1, p2, p3, layout=(1,3), size=(1500, 500))
    path = joinpath(output_dir, "coupled_a_posteriori_prediction.pdf")
    savefig(combined, path)
    println("Saved $path")
end

main()

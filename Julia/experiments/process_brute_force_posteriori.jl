
using Surferbot
using JLD2
using Plots
using Statistics
using Printf
using LinearAlgebra
using Base.Threads

"""
    process_brute_force_posteriori.jl

Process the 30k brute-force grid to generate the a-posteriori overlay plot.
"""

function main()
    input_file = "Julia/output/brute_force_30k_full.jld2"
    output_processed = "Julia/output/processed_posteriori_30k.jld2"
    output_plot = "Julia/output/coupled_a_posteriori_overlay_30k.pdf"
    
    if !isfile(input_file)
        println("Input file not found: $input_file")
        return
    end
    
    println("Loading brute-force grid...")
    data = load(input_file)
    results = data["results"]
    mp_norm_list = data["mp_norm_list"]
    logEI_list = data["logEI_list"]
    
    nx, nei = size(results)
    println("Grid size: $nx x $nei")
    
    # Pre-allocate
    alpha_num = zeros(Float64, nx, nei)
    S_forcing = zeros(ComplexF64, nx, nei)
    S_pressure = zeros(ComplexF64, nx, nei)
    sa_ratio = zeros(Float64, nx, nei)
    cos_gap = zeros(Float64, nx, nei)
    
    # Constants for D_n
    first_res = nothing
    for r in results
        if !isnothing(r)
            first_res = r
            break
        end
    end
    
    if isnothing(first_res)
        error("Results matrix is empty.")
    end
    
    rho_raft = first_res.metadata.args.rho_raft
    omega = first_res.metadata.args.omega
    n_modes = 8 
    
    println("Processing modal decompositions on $(nthreads()) threads...")
    
    @threads for idx in 1:(nx * nei)
        im = ((idx - 1) % nx) + 1
        ie = floor(Int, (idx - 1) / nx) + 1
        
        res = results[im, ie]
        if isnothing(res)
            alpha_num[im, ie] = NaN
            S_forcing[im, ie] = NaN
            S_pressure[im, ie] = NaN
            sa_ratio[im, ie] = NaN
            cos_gap[im, ie] = NaN
            continue
        end
        
        # 1. Numerical metrics
        contact = collect(Bool.(res.metadata.args.x_contact))
        idx_c = findall(contact)
        L_beam = res.eta[first(idx_c)]
        R_beam = res.eta[last(idx_c)]
        alpha_num[im, ie] = -(abs2(L_beam) - abs2(R_beam)) / (abs2(L_beam) + abs2(R_beam) + 1e-12)
        
        S_raw = (R_beam + L_beam) / 2
        A_raw = (R_beam - L_beam) / 2
        sa_ratio[im, ie] = log10(abs(S_raw) / (abs(A_raw) + 1e-12))
        cos_gap[im, ie] = real(S_raw * conj(A_raw)) / (abs(S_raw) * abs(A_raw) + 1e-12)
        
        # 2. Modal Decomposition
        modal = Surferbot.Modal.decompose_raft_freefree_modes(
            res.metadata.args.x, 
            res.eta, 
            res.pressure, 
            res.metadata.args.loads, 
            res.metadata.args; 
            num_modes=n_modes, 
            verbose=false
        )
        
        # 3. Calculate a posteriori terms
        even_indices = findall(n -> iseven(n), modal.n)
        
        W_end_psi = modal.Psi[1, :]
        EI = 10.0^logEI_list[ie]
        D = [EI * modal.beta[k]^4 - rho_raft * omega^2 for k in eachindex(modal.beta)]
        
        s_f = 0.0im
        s_q = 0.0im
        for k in even_indices
            s_f += (-modal.F[k] / D[k]) * W_end_psi[k]
            s_q += (modal.Q[k] / D[k]) * W_end_psi[k]
        end
        
        S_forcing[im, ie] = s_f
        S_pressure[im, ie] = s_q
    end
    
    println("Saving processed data to $output_processed...")
    jldsave(output_processed; 
        alpha_num=alpha_num, 
        S_forcing=S_forcing, 
        S_pressure=S_pressure,
        sa_ratio=sa_ratio,
        cos_gap=cos_gap,
        mp_norm_list=mp_norm_list,
        logEI_list=logEI_list
    )
    
    # 3. Plotting
    println("Generating plots...")
    S_total = S_forcing .+ S_pressure
    
    p1 = heatmap(logEI_list, mp_norm_list, alpha_num, 
                title="Numerical Alpha", xlabel="log10(EI)", ylabel="xM / L", 
                color=:RdBu, clim=(-0.5, 0.5), interpolate=true)
    contour!(p1, logEI_list, mp_norm_list, alpha_num, levels=[0.0], color=:black, linewidth=2, label="alpha=0")
    
    p2 = heatmap(logEI_list, mp_norm_list, real.(S_total), 
                title="A Posteriori Prediction Re(S_f + S_q)", xlabel="log10(EI)", ylabel="xM / L",
                color=:RdBu, clim=(-0.1, 0.1), interpolate=true)
    contour!(p2, logEI_list, mp_norm_list, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="S_total=0")
    contour!(p2, logEI_list, mp_norm_list, alpha_num, levels=[0.0], color=:black, linewidth=1, linestyle=:dash, label="alpha=0")
    
    p3 = heatmap(logEI_list, mp_norm_list, alpha_num, 
                title="Pressure Shift Overlay", xlabel="log10(EI)", ylabel="xM / L",
                color=:RdBu, clim=(-0.2, 0.2), interpolate=true)
    contour!(p3, logEI_list, mp_norm_list, alpha_num, levels=[0.0], color=:black, linewidth=2, label="Numerical alpha=0")
    contour!(p3, logEI_list, mp_norm_list, real.(S_forcing), levels=[0.0], color=:gold, linewidth=2, label="Mechanical only (S_f=0)")
    contour!(p3, logEI_list, mp_norm_list, real.(S_total), levels=[0.0], color=:green, linewidth=2, label="Total prediction=0")
    
    combined = plot(p1, p2, p3, layout=(1,3), size=(1500, 500), margin=5Plots.mm)
    savefig(combined, output_plot)
    println("Saved plot to $output_plot")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


using JLD2
using Plots
using Statistics
using LinearAlgebra

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    path = joinpath(output_dir, "sweeps", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    if !isfile(path)
        println("Error: JLD2 file not found at $path")
        return
    end
    
    artifact = load(path, "artifact")
    
    motor_position_list = vec(Float64.(artifact.parameter_axes.motor_position))
    EI_list = vec(Float64.(artifact.parameter_axes.EI))
    mp_norm_list = motor_position_list ./ artifact.base_params.L_raft
    logEI_list = log10.(EI_list)
    
    summaries = artifact.summaries
    n_mp = length(mp_norm_list)
    n_ei = length(logEI_list)
    
    S_grid = zeros(n_mp, n_ei)
    A_grid = zeros(n_mp, n_ei)
    cos_grid = zeros(n_mp, n_ei)
    
    for ie in 1:n_ei, im in 1:n_mp
        s = summaries[im, ie]
        L = s.eta_left_beam
        R = s.eta_right_beam
        
        S = (R + L) / 2
        A = (R - L) / 2
        
        S_grid[im, ie] = abs(S)
        A_grid[im, ie] = abs(A)
        
        # cos(arg(A) - arg(S)) = Re(A * conj(S)) / (|A| * |S|)
        if abs(A) > 1e-12 && abs(S) > 1e-12
            cos_grid[im, ie] = real(A * conj(S)) / (abs(A) * abs(S))
        else
            cos_grid[im, ie] = 0.0
        end
    end

    # Use a fixed wide range to see deep structural valleys
    v_min = -10.0
    v_max = -1.0

    # Plotting: Vertical arrangement for better detail
    p1 = heatmap(logEI_list, mp_norm_list, log10.(S_grid .+ 1e-12);
                title="|S| (log10)", xlabel="log10(EI)", ylabel="xM / L", 
                c=:viridis, clims=(v_min, v_max), interpolate=true)
    
    p2 = heatmap(logEI_list, mp_norm_list, log10.(A_grid .+ 1e-12);
                title="|A| (log10)", xlabel="log10(EI)", ylabel="xM / L", 
                c=:viridis, clims=(v_min, v_max), interpolate=true)
                
    p3 = heatmap(logEI_list, mp_norm_list, cos_grid;
                title="cos(arg(A) - arg(S))", xlabel="log10(EI)", ylabel="xM / L", 
                c=:balance, clims=(-1, 1), interpolate=true)
                
    combined = plot(p1, p2, p3, layout=(3,1), size=(1000, 2400), margin=10Plots.mm)
    
    output_path = joinpath(output_dir, "figures", "plot_branch_signatures.pdf")
    savefig(combined, output_path)
    println("Saved branch physical signatures plot to $output_path")
end

main()

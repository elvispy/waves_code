"""
plot_coupled_posteriori.jl

Calculates and visualizes the a posteriori contributions of mechanical forcing 
(F) and hydrodynamic pressure (Q) to the symmetric radiation field in the 
coupled case, using CSV-extracted modal coefficients.
"""
using Surferbot
using JLD2
using Plots
using Statistics
using Printf
using DelimitedFiles
using LinearAlgebra

# Simplified Plotting script for the A Posteriori Coupled Prediction.
# Uses coefficients directly from the CSV to avoid parameter mismatch.

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    csv_file = joinpath(output_dir, "csv", "analyze_single_alpha_zero_curve.csv")
    
    if !isfile(csv_file)
        csv_file = joinpath(output_dir, "csv", "coupled_branch_smoke.csv")
        if !isfile(csv_file) return end
    end
    
    println("Loading coupled data from $csv_file...")
    data, header = readdlm(csv_file, ',', header=true)
    names = string.(vec(header))
    col(name) = findfirst(==(name), names)
    
    logEI = log10.(data[:, col("EI")])
    xM_norm = data[:, col("xM_over_L")]
    
    n_pts = size(data, 1)
    
    # We use the analytical W_n(L/2) values. 
    # Left end (xi=0) is used in the porting logic.
    w_ends = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0] 
    even_modes = [0, 2, 4, 6]
    
    S_total = zeros(ComplexF64, n_pts)
    S_mech_contribution = zeros(ComplexF64, n_pts)
    S_hydro_contribution = zeros(ComplexF64, n_pts)
    
    for i in 1:n_pts
        # Modal displacement q_n (total)
        # q_n = q_mech + q_hydro
        # where q_mech = -F_n / D_n and q_hydro = Q_n / D_n
        
        for (idx, n) in enumerate(even_modes)
            # Standard columns (which are now W-basis)
            qn = data[i, col("q$(n)_re")] + im*data[i, col("q$(n)_im")]
            Qn = data[i, col("Q$(n)_re")] + im*data[i, col("Q$(n)_im")]
            Fn = data[i, col("F$(n)_re")] + im*data[i, col("F$(n)_im")]
            
            # Since qn = (Qn - Fn) / Dn, we can infer the contribution:
            # S_total = sum( qn * w_end )
            # S_hydro_part = sum( (Qn / (Qn - Fn) * qn) * w_end )
            # S_mech_part  = sum( (-Fn / (Qn - Fn) * qn) * w_end )
            
            S_total[i] += qn * w_ends[2*idx-1]
            
            # Use ratio to decompose qn
            ratio_h = Qn / (Qn - Fn + eps()im)
            ratio_m = -Fn / (Qn - Fn + eps()im)
            
            S_hydro_contribution[i] += (ratio_h * qn) * w_ends[2*idx-1]
            S_mech_contribution[i] += (ratio_m * qn) * w_ends[2*idx-1]
        end
    end
    
    # Plotting
    plt = plot(logEI, abs.(S_mech_contribution), label="|S_mech| (contribution)", linewidth=2, color=:gold)
    plot!(plt, logEI, abs.(S_hydro_contribution), label="|S_hydro| (contribution)", linewidth=2, color=:dodgerblue)
    plot!(plt, logEI, abs.(S_total), label="|S_total|", linewidth=3, color=:green)
    title!(plt, "Coupled Modal Contributions to S")
    xlabel!(plt, "log10(EI)")
    ylabel!(plt, "Magnitude")
    
    fig_path = joinpath(output_dir, "figures", "plot_coupled_posteriori.pdf")
    savefig(plt, fig_path)
    println("Saved $fig_path")
    
    reduction = abs.(S_total) ./ (abs.(S_mech_contribution) .+ abs.(S_hydro_contribution))
    println("Residual S ratio: ", mean(reduction) * 100, "%")
end

main()

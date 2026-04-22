using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

# Phase 1: Reframe the Target Quantity (Senior Analyst Recommendation)
# Audit: 1. Cancellation Ratio ||Q+F|| / ||F||
#        2. Stability of Dynamic Stiffness Z = (Q-F) / q

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    Random.seed!(42)
    indices = sort(randperm(size(data, 1))[1:10])

    println("--- Senior Analyst Phase 1: Conditioning & Stability ---")
    @printf("%-10s | %-12s | %-12s | %-12s\n", "Sample", "Cancel Ratio", "Z_0 (Re)", "Z_0 (Im)")
    println("-"^60)
    
    cancel_ratios = Float64[]
    Z_re_vals = Float64[]
    
    for i in indices
        Q_vec = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        F_vec = [complex(data[i, col("F_w$(n)_re")], data[i, col("F_w$(n)_im")]) for n in 0:3]
        q_vec = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        
        # 1. Cancellation Ratio (Residual / Drive)
        res_vec = Q_vec + F_vec
        eps_n = norm(res_vec) / norm(F_vec)
        push!(cancel_ratios, eps_n)
        
        # 2. Dynamic Stiffness Z = (Q-F)/q? No, the balance is G*D*q = Q-F.
        # But for n=0 (Rigid), G is roughly diagonal. 
        # Z_eff = (Q_w0 - F_w0) / q_w0
        Z_eff = (Q_vec[1] - F_vec[1]) / (q_vec[1] + 1e-18)
        push!(Z_re_vals, real(Z_eff))
        
        @printf("%-10d | %-12.4e | %-12.3e | %-12.3e\n", i, eps_n, real(Z_eff), imag(Z_eff))
    end
    
    println("-"^60)
    @printf("Mean Cancellation Ratio: %.4e\n", mean(cancel_ratios))
    @printf("Z_0 Coefficient of Var:  %.4f\n", std(Z_re_vals) / abs(mean(Z_re_vals)))
    
    println("\nConclusion for phase 1:")
    if mean(cancel_ratios) < 0.1
        println("VERIFIED: The solution is a small residual (Q approx -F).")
    end
    if (std(Z_re_vals) / abs(mean(Z_re_vals))) < 0.05
        println("SUCCESS: Total Impedance Z is a stable observable! A <5% law is possible for Z.")
    else
        println("STILL UNSTABLE: Even the transformed target Z varies by more than 5%.")
    end
end

main()

using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    Random.seed!(42)
    indices = sort(randperm(size(data, 1))[1:10])

    println("--- Audit: Zero-Fluid Hypothesis (Qn = 0) ---")
    errors = Float64[]
    for i in indices
        Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        # Prediction: Q_ap = 0
        err = norm(Q_num) / norm(Q_num) # wait, if Q_ap=0, error is norm(Q_num)/norm(Q_num) = 1.0
        # Wait, if Q_ap=0 is the hypothesis, then the error is always 100% unless Q_num is small.
        # Let's calculate the relative magnitude of Q_num vs F_num
        F_num = [complex(data[i, col("F_w$(n)_re")], data[i, col("F_w$(n)_im")]) for n in 0:3]
        q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        
        rel_mag = norm(Q_num) / norm(F_num)
        push!(errors, rel_mag)
        @printf("Sample %d: |Q| / |F| = %.4f\n", i, rel_mag)
    end
    println("-"^40)
    @printf("Mean Relative Fluid Force: %.4f\n", mean(errors))
end

main()

using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Printf

# Purpose: Extract the pure Complex Hydrodynamic Admittance from the dataset.
# Formula: Z_dyn,n = (Q_w,n + d*rho*g*q_w,n) / q_w,n

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    # Physical constants
    d = 0.03
    rho = 1000.0
    g = 9.81
    omega = data[1, col("omega")]
    L = data[1, col("L_raft")]
    
    modes = 0:3
    
    println("--- Empirical Fact-Finding: Dynamic Admittance ---")
    @printf("Mode | Re(Z_dyn) [Stiffness] | Im(Z_dyn) [Damping] | |Z| / (rho*g*d)\n")
    println("-"^70)
    
    for n in modes
        q = complex.(data[:, col("q_w$(n)_re")], data[:, col("q_w$(n)_im")])
        Q = complex.(data[:, col("Q_w$(n)_re")], data[:, col("Q_w$(n)_im")])
        
        # Isolate Dynamic Part
        # Q_rest = -d*rho*g*q
        # Q_total = Q_rest + Q_dyn  => Q_dyn = Q_total + d*rho*g*q
        Q_dyn = Q .+ (d * rho * g) .* q
        
        # Impedance Z = Q_dyn / q
        # We calculate the mean across the sweep to see if it's constant
        Z_vec = [Q_dyn[i] / (q[i] + 1e-18) for i in 1:length(q)]
        
        # Filter outliers where q is very small
        mask = abs.(q) .> (0.01 * mean(abs.(q)))
        Z_filtered = Z_vec[mask]
        
        z_mean = mean(Z_filtered)
        z_std = std(Z_filtered)
        
        # Normalize by static hydrostatic scale
        norm_val = abs(z_mean) / (rho * g * d)
        
        @printf("n=%-2d | %-18.3e | %-18.3e | %-12.2f\n", n, real(z_mean), imag(z_mean), norm_val)
    end
    
    println("\nObservations:")
    println("1. Re(Z_dyn) > 0 means the fluid adds STIFFNESS (Heave/Restoring boost).")
    println("2. Re(Z_dyn) < 0 means the fluid adds MASS (Added Mass dominates).")
    println("3. Im(Z_dyn) corresponds to Radiation Damping.")
end

main()

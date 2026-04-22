using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Printf

# Purpose: Systematic Correlation Audit to find the "Winning" Power Law for Qn.
# We test 4 physical hypotheses against the coupled dataset.

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    # Variables for Correlation
    n_pts = size(data, 1)
    Q0_mag = [abs(complex(data[i, col("Q_w0_re")], data[i, col("Q_w0_im")])) for i in 1:n_pts]
    
    L = data[:, col("L_raft")]
    H = fill(0.05, n_pts) # constant in this dataset
    d = fill(0.03, n_pts) # constant
    omega = data[:, col("omega")]
    EI = data[:, col("EI")]
    rho = 1000.0
    g = 9.81
    sigma = 72.2e-3
    
    # Calculate k_wave for each point
    k_w = [real(Surferbot.dispersion_k(omega[i], g, 0.0, 0.0, 0.0, rho)) for i in 1:n_pts]
    
    # Hypothesis Vectors
    # H1: Confined Flow (rho * w^2 * d * L^2 / H)
    H1 = (rho .* omega.^2 .* d .* L.^2 ./ H)
    # H2: Radiation (rho * w^2 * d * L / k_w)
    H2 = (rho .* omega.^2 .* d .* L ./ k_w)
    # H3: Sectional (rho * w^2 * d^3) - wait, d is width. 
    # Let's use d * L (Area)
    H3 = (rho .* omega.^2 .* d .* L)
    # H4: Structural (EI * beta^4 * q) ... we use q_num to see if Q tracks q
    q0_mag = [abs(complex(data[i, col("q_w0_re")], data[i, col("q_w0_im")])) for i in 1:n_pts]
    H4 = (EI .* (data[1, col("beta0")]^4) .* q0_mag)

    # Correlation Matrix
    println("--- Power Law Correlation Audit (Target: |Q_0|) ---")
    @printf("%-20s | %-12s\n", "Hypothesis", "Correlation")
    println("-"^35)
    @printf("%-20s | %-12.4f\n", "H1: Confined (L^2/H)", cor(Q0_mag, H1))
    @printf("%-20s | %-12.4f\n", "H2: Radiation (L/k)",   cor(Q0_mag, H2))
    @printf("%-20s | %-12.4f\n", "H3: Area (L*d)",        cor(Q0_mag, H3))
    @printf("%-20s | %-12.4f\n", "H4: Structural (D*q)",  cor(Q0_mag, H4))

    # --- THE DISCOVERY ---
    # Look for the ratio Q / Law to see if it is constant
    println("\n--- Law Magnitude Analysis (Mean Ratio) ---")
    @printf("Mode | Q / H1 (Confined) | Q / H2 (Rad) | Q / H3 (Area)\n")
    println("-"^55)
    for n in 0:3
        Qn = [abs(complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")])) for i in 1:n_pts]
        r1 = mean(Qn ./ H1)
        r2 = mean(Qn ./ H2)
        r3 = mean(Qn ./ H3)
        @printf("n=%-2d | %-16.4e | %-12.4e | %-12.4e\n", n, r1, r2, r3)
    end
end

main()

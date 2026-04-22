using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Printf

# Purpose: Ultra-lean debug to find the magnitude discrepancy source.
# 1. Load one high-res numerical result.
# 2. Compare numerical p(x) on raft with Spectral A-Priori p(x).
# 3. Check for normalization factors (d, rho, omega).

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    # Sample 32
    i = 32
    L = data[i, col("L_raft")]
    H = 0.05
    L_dom = 1.5
    d = 0.03
    rho = 1000.0
    omega = data[i, col("omega")]
    
    # Extract numerical Q_w and q_w
    # q_w is the vector of coefficients in the analytical Phi basis
    q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
    Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
    
    # --- DECONSTRUCTION STEP 1: Basis Projection Check ---
    # Reconstruct numerical Q_w force magnitude from the CSV
    # Force = Q_w * G (roughly)
    x_raft = collect(range(-L/2, L/2, length=101))
    w_trapz = Surferbot.trapz_weights(x_raft)
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=4)
    Phi = raw_basis.Phi
    G = Phi' * (Phi .* w_trapz)
    
    # The actual numerical force coefficients in the Phi basis are:
    # F_coeffs = G * Q_num
    F_num_coeffs = G * Q_num
    
    # --- DECONSTRUCTION STEP 2: Radiation Wave Law ---
    # eta(x) = Phi(x) * q_num
    eta_x = Phi * q_num
    
    # The pressure on a finite line source L in 2D is:
    # p(x) = rho * omega^2 * (Integral of G(x, x"") * eta(x") dx"")
    # For high freq, G(x, x"") is roughly the infinite-depth Green function pole part.
    
    function get_radiation_p(x_nodes, eta_x, L, rho, omega)
        # Numerical integration of the 2D Green function kernel:
        # G(x, x"") = K0(k|x-x"") or log|x-x""|? 
        # In 2D potential flow, G = 1/pi * log|r|.
        # P(x) approx -rho * omega^2 * Integral( 1/pi * log|x-x"| * eta(x") ) dx"
        
        dx = x_nodes[2] - x_nodes[1]
        p = zeros(ComplexF64, length(x_nodes))
        for i in eachindex(x_nodes)
            val = 0.0 + 0.0im
            for j in eachindex(x_nodes)
                r = abs(x_nodes[i] - x_nodes[j])
                # Regularize the log singularity
                kernel = (r < 1e-6) ? log(dx/2) : log(r)
                val += kernel * eta_x[j] * dx
            end
            p[i] = - (rho * omega^2 / pi) * val
        end
        return p
    end
    
    p_x_rad = get_radiation_p(x_raft, eta_x, L, rho, omega)
    F_rad_coeffs = Phi' * (p_x_rad .* d .* w_trapz)
    
    # --- DECONSTRUCTION STEP 3: The New Ratio ---
    println("--- Radiation Wave Magnitude Deconstruction ---")
    @printf("Mode | Num Force Mag | Radiation Mag | Ratio (Rad/Num)\n")
    for n in 1:4
        n_mag = abs(F_num_coeffs[n])
        r_mag = abs(F_rad_coeffs[n])
        @printf("n=%-2d | %.4e   | %.4e   | %.1f\n", n-1, n_mag, r_mag, r_mag/n_mag)
    end
    
    println("\nConclusion:")
    println("If Ratio is approx 1.0, the physics is Near-Field Potential Radiation.")
    
    println("\nConclusion:")
    println("If Ratio is approx 1.0, the physics is purely Local Confined Flow with Edge Release.")
    
    println("\nPhysical Scales Analysis:")
    @printf("rho * omega^2 * d * L^2 / H: %.2e\n", rho * omega^2 * d * L^2 / H)
    @printf("Total Numerical Force Sum:  %.2e\n", sum(abs.(F_num_coeffs)))
end

main()

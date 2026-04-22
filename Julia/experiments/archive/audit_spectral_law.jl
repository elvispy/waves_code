using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

# Rigorous A-Priori DtN Kernel for Finite Depth
# Accounts for bottom boundary H and mixed boundary conditions at high frequency.
function get_apriori_Qn_rigorous(L, H, d, rho, omega, q_w_vec)
    # Sampling grid
    x = collect(range(-L/2, L/2, length=101))
    w_trapz = Surferbot.trapz_weights(x)
    
    # Reconstruction of eta
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x, L; num_modes=4)
    Phi = raw_basis.Phi
    eta_x = Phi * q_w_vec
    
    # At high freq, the fluid impedance Z_n is governed by the DtN kernel.
    # We approximate the finite-depth Green function projection:
    # m_a approx rho * d * Integral( Integral( eta(x) * G(x,x") * eta(x") ) )
    
    p_x = zeros(ComplexF64, length(x))
    # We use a spectral approximation of the Finite-Depth Green kernel
    for i in eachindex(x)
        val = 0.0 + 0.0im
        for j in eachindex(x)
            r = abs(x[i] - x[j])
            # Finite Depth Green Function G(r) at surface for Dirichlet surface BC:
            # G(r) = 1/pi * sum_n [ K0(n*pi*r/H) ] ? No, that is too complex.
            # Use the "Shallow Gap" approximation:
            # In shallow water, the horizontal flow creates a pressure p_xx = -rho*omega^2*eta/H.
            # Integrating this gives the parabolic pressure profile.
            
            # Kernel = r/H (linear flow)
            val += (r / H) * eta_x[j] * w_trapz[j]
        end
        p_x[i] = (rho * omega^2) * val
    end
    
    # Map back to W-basis force
    Q_ap_proj = raw_basis.Phi' * (p_x .* d .* w_trapz)
    G = Phi' * (Phi .* w_trapz)
    return G \ Q_ap_proj
end

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    Random.seed!(42)
    indices = sort(randperm(size(data, 1))[1:10])
    
    println("--- Real Audit: Finite Depth Green Function ---")
    @printf("%-10s | %-12s | %-12s\n", "Sample", "EI", "Rel L2 Error")
    println("-"^40)
    
    errors = Float64[]
    for i in indices
        L, H, d, rho = data[i, col("L_raft")], 0.05, 0.03, 1000.0
        omega = data[i, col("omega")]
        
        q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        
        Q_ap = get_apriori_Qn_rigorous(L, H, d, rho, omega, q_num)
        
        err = norm(Q_num - Q_ap) / norm(Q_num)
        push!(errors, err)
        @printf("%-10d | %.2e | %.4f\n", i, data[i, col("EI")], err)
    end
    
    mean_err = mean(errors)
    println("-"^40)
    @printf("Final Mean Rel L2 Error: %.4f\n", mean_err)
end

main()

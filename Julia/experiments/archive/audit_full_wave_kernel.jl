using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

# The Full Wave Kernel Law (Iteration 19)
# Z_n = modal projection of the DtN inverse with gravity-wave pole
function get_apriori_Qn_wave(L, H, L_dom, d, rho, omega, q_w_vec)
    g = 9.81
    x = collect(range(-L/2, L/2, length=101))
    w_trapz = Surferbot.trapz_weights(x)
    
    # Analytical basis
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x, L; num_modes=4)
    Phi = raw_basis.Phi
    eta_x = Phi * q_w_vec
    
    # We construct the DtN Inverse Matrix G_fluid on the raft grid
    # Using the Spectral Green Function for a rectangular tank
    N_raft = length(x)
    G_fluid = zeros(ComplexF64, N_raft, N_raft)
    
    m_max = 1000 # Converge the series
    for m in 0:m_max
        lambda_m = m * pi / L_dom
        
        # Denominator is the DtN eigenvalue minus the free-surface term
        # This shifts the poles to the wave wavenumbers
        # Adding a small imaginary part for the radiation condition / damping
        denom = lambda_m * tanh(lambda_m * H) - (omega^2 / g) + 1im*1e-1
        
        # Tank modes (symmetric about center)
        # Note: L_dom is the full width of the tank
        # Integral(cos^2) = L_dom/2
        factor = (m == 0) ? 1/L_dom : 2/L_dom
        
        for i in 1:N_raft
            mode_i = cos(lambda_m * (x[i] + L_dom/2))
            for j in 1:N_raft
                mode_j = cos(lambda_m * (x[j] + L_dom/2))
                G_fluid[i, j] += mode_i * mode_j * factor / denom
            end
        end
    end
    
    # Pressure p = rho * omega^2 * G_fluid * (eta * w_trapz)
    # Wait, the Green function integration: p(x) = rho * omega^2 * Integral( G(x,x') eta(x') ) dx'
    p_x = (rho * omega^2) .* (G_fluid * (eta_x .* w_trapz))
    
    # Project to W-basis
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
    
    println("--- 10-Solve Audit: Full Wave Kernel ---")
    @printf("%-10s | %-12s | %-12s\n", "Sample", "EI", "Rel L2 Error")
    println("-"^40)
    
    errors = Float64[]
    for i in indices
        L, H, L_dom, d, rho = data[i, col("L_raft")], 0.05, 1.5, 0.03, 1000.0
        omega = data[i, col("omega")]
        
        q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        
        Q_ap = get_apriori_Qn_wave(L, H, L_dom, d, rho, omega, q_num)
        
        err = norm(Q_num - Q_ap) / norm(Q_num)
        push!(errors, err)
        @printf("%-10d | %.2e | %.4f\n", i, data[i, col("EI")], err)
    end
    
    mean_err = mean(errors)
    println("-"^40)
    @printf("Final Mean Rel L2 Error: %.4f\n", mean_err)
end

main()

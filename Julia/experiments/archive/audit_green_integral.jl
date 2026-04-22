using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf
using QuadGK

# Purpose: Test the "Green Integral Law" (The true first-principles law)
# Z_n = rho * d * omega^2 * <psi_n, K^-1 psi_n> / <psi_n, psi_n>
# where K is the full Gravity-Capillary DtN operator.

function get_green_integral_Qn(params, q_w_vec)
    L, H, d, rho, omega = params.L_raft, params.domain_depth, params.d, params.rho, params.omega
    g, sigma, nu = params.g, params.sigma, params.nu
    
    # 1. Setup RAFT grid and modes
    x = collect(range(-L/2, L/2, length=101))
    w_trapz = Surferbot.trapz_weights(x)
    raw = Surferbot.Modal.build_raw_freefree_basis(x, L; num_modes=4)
    Phi = raw.Phi
    eta_x = Phi * q_w_vec
    
    # 2. Construct the Spectral DtN Inverse (Green Function)
    # Since we are in a finite domain Ldom, we use the discrete spectrum.
    L_dom = params.L_domain
    p_x = zeros(ComplexF64, length(x))
    
    # We sum over the tank modes m*pi/Ldom
    # This is equivalent to the integral form but respects mass conservation.
    m_max = 1000
    for m in 0:m_max
        k_m = m * pi / L_dom
        
        # Exact Gravity-Capillary DtN Eigenvalue
        # K_m = (g*k + sigma/rho*k^3) / (omega^2) ... no, the DtN is phi_z / phi.
        # From Bernoulli: p = -rho(i*omega*phi + g*eta)
        # The correct relation is the one used in the solver assembly.
        
        # DtN kernel: k * tanh(kH)
        dtn = k_m * tanh(k_m * H + 1e-12)
        
        # Free-surface term including gravity and capillarity
        # (omega^2 - 4i*nu*k^2) / (g + sigma/rho*k^2)
        fs_term = (omega^2) / (g + (sigma/rho)*k_m^2)
        
        # Response factor (Admittance)
        denom = dtn - fs_term + 1im*1e-6 # small damping for stability
        
        # Tank mode projection
        mode_m = cos.(k_m .* (x .+ L_dom/2))
        factor = (m == 0) ? 1/L_dom : 2/L_dom
        
        eta_m = dot(mode_m, eta_x .* w_trapz)
        p_m = (rho * omega^2 * eta_m * factor) / denom
        
        p_x .+= p_m .* mode_m
    end
    
    # 3. Project back to W-basis Q_w
    Q_ap_proj = raw.Phi' * (p_x .* d .* w_trapz)
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
    
    println("--- Gold Standard Audit: Green Integral Law ---")
    @printf("%-10s | %-12s | %-12s\n", "Sample", "EI", "Rel L2 Error")
    println("-"^40)
    
    errors = Float64[]
    for i in indices
        # Reconstruct params for this sample
        L, H, L_dom, d, rho = data[i, col("L_raft")], 0.05, 1.5, 0.03, 1000.0
        omega = data[i, col("omega")]
        sigma = 72.2e-3
        nu = 1e-6
        g = 9.81
        
        params = (L_raft=L, domain_depth=H, L_domain=L_dom, d=d, rho=rho, omega=omega, g=g, sigma=sigma, nu=nu)
        
        q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        
        Q_ap = get_green_integral_Qn(params, q_num)
        
        err = norm(Q_num - Q_ap) / norm(Q_num)
        push!(errors, err)
        @printf("%-10d | %.2e | %.4f\n", i, data[i, col("EI")], err)
    end
    
    println("-"^40)
    @printf("\nFinal Mean Rel L2 Error: %.4f\n", mean(errors))
end

main()

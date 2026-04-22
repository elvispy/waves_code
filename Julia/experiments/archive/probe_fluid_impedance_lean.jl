using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Lean Hydrodynamic Probe to find the A-Priori Added Mass Law.
# Low resolution to avoid OOM.

function get_pure_fluid_impedance(params, mode_idx)
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    
    # Project pressure directly
    p_raft = result.p 
    w_trapz = Surferbot.trapz_weights(modal.x_raft)
    Q_n = dot(modal.Phi[:, mode_idx], p_raft .* w_trapz .* params.d)
    q_n = modal.q_w[mode_idx]
    
    return Q_n / q_n
end

function main()
    println("--- Lean Hydrodynamic Probe ---")
    
    L = 0.05
    H = 0.05
    L_dom = 1.0 # shorter tank for speed
    d = 0.03
    rho = 1000.0
    
    f_list = [10.0, 30.0, 50.0, 80.0]
    Z_freq = ComplexF64[]
    
    println("\nRunning Frequency Sweep (n=80, M=20)...")
    for f in f_list
        omega = 2pi * f
        params = Surferbot.FlexibleParams(
            L_raft = L, domain_depth = H, L_domain = L_dom,
            omega = omega, d = d,
            EI = 1e6, # Stiff
            rho_raft = 0.0,
            n = 80, M = 20 # Lean resolution
        )
        Z = get_pure_fluid_impedance(params, 1) # Mode 0
        push!(Z_freq, Z)
        @printf("   f = %2.0f Hz | Z = %.3e + %.3ei\n", f, real(Z), imag(Z))
    end
    
    # Regression: Z = K - m_a * omega^2
    # Since Q is the force resisting motion, Re(Z) should be positive for mass.
    # Note: Bernoulli P = -rho(i*w*phi + g*eta). Q_total includes both.
    
    omegas = 2pi .* f_list
    X = hcat(ones(length(omegas)), -omegas.^2)
    p = X \ real.(Z_freq)
    
    K_static = p[1]
    ma_derived = p[2]
    
    println("\n--- Discovery ---")
    @printf("Derived Static Stiffness K: %.3e N/m^2\n", K_static)
    @printf("Derived Added Mass ma:      %.4f kg/m\n", ma_derived)
    @printf("Theoretical rho*g*d:        %.3e N/m^2\n", rho * 9.81 * d)
    
    @printf("\nK / (rho*g*d) ratio: %.2f\n", K_static / (rho * 9.81 * d))
end

main()

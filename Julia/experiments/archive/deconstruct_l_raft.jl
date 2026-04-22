using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Single-variable sweep to find the L (Raft Length) dependency.
# Contract Step 2: Isolate the local geometric scaling.

function get_q0_impedance(params)
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    Qf = modal.Q_w[1] - modal.F_w[1]
    Gq = (G * modal.q_w)[1]
    return - (Qf / Gq)
end

function main()
    println("--- Deconstruction Step 2: L (Raft Length) Scaling ---")
    
    H = 0.05
    d = 0.03
    f = 80.0
    Ldom = 1.5
    
    L_list = [0.03, 0.06, 0.09, 0.12]
    Z_list = ComplexF64[]
    
    for L in L_list
        params = Surferbot.FlexibleParams(
            L_raft = L, domain_depth = H, L_domain = Ldom,
            omega = 2pi * f, d = d, EI = 1e6, rho_raft = 0.05,
            n = 200, M = 30, motor_position = 0.0, motor_force = 1.0
        )
        Z = get_q0_impedance(params)
        push!(Z_list, Z)
        @printf("   L = %.2f m | Z = %.3e + %.3ei\n", L, real(Z), imag(Z))
    end
    
    log_L = log.(L_list)
    log_Z = log.(abs.(Z_list))
    gamma = (log_Z[end] - log_Z[1]) / (log_L[end] - log_L[1])
    
    println("\n--- Findings ---")
    @printf("L (Raft Length) Scaling Exponent: %.4f\n", gamma)
    
    if abs(gamma - 2.0) < 0.2
        println("Hypothesis H_L2: Impedance scales with L^2 (2D Piston).")
    elseif abs(gamma - 3.0) < 0.2
        println("Hypothesis H_L3: Impedance scales with L^3 (Confined Gap Flow).")
    else
        @printf("Trend is L^{%.1f}. Building hybrid law.\n", gamma)
    end
end

main()

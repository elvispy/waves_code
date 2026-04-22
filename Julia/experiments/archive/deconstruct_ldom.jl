using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Single-variable sweep to find the Ldom dependency.
# Contract Step 1: Start small, find trends, build knowledgebase.

function get_q0_impedance(params)
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    
    Qf = modal.Q_w[1] - modal.F_w[1]
    Gq = (G * modal.q_w)[1]
    
    # Complex Impedance Z = -Qf / Gq
    return - (Qf / Gq)
end

function main()
    println("--- Deconstruction Step 1: L_dom Scaling ---")
    
    # Constant parameters
    L = 0.05
    H = 0.05
    d = 0.03
    f = 80.0
    omega = 2pi * f
    
    Ldom_list = [0.5, 1.0, 1.5, 2.0]
    Z_list = ComplexF64[]
    
    for Ldom in Ldom_list
        params = Surferbot.FlexibleParams(
            L_raft = L, domain_depth = H, L_domain = Ldom,
            omega = omega, d = d, EI = 1e4, rho_raft = 0.05,
            n = 200, M = 30, motor_position = 0.0, motor_force = 1.0
        )
        Z = get_q0_impedance(params)
        push!(Z_list, Z)
        @printf("   Ldom = %.2f m | Z = %.3e + %.3ei\n", Ldom, real(Z), imag(Z))
    end
    
    # Regression: log|Z| vs log(Ldom)
    log_L = log.(Ldom_list)
    log_Z = log.(abs.(Z_list))
    gamma = (log_Z[end] - log_Z[1]) / (log_L[end] - log_L[1])
    
    println("\n--- Findings ---")
    @printf("Ldom Scaling Exponent: %.4f\n", gamma)
    
    if abs(gamma - 1.0) < 0.1
        println("Hypothesis H_L1: Impedance scales LINEARLY with tank length.")
    elseif abs(gamma - 0.0) < 0.1
        println("Hypothesis H_L0: Impedance is INDEPENDENT of tank length (Local physics).")
    else
        println("Trend is complex. Re-evaluating BVP.")
    end
end

main()

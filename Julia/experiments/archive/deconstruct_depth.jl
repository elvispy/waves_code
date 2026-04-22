using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Single-variable sweep to find the H (Depth) dependency.
# Contract Step 3: Final geometric variable deconstruction.

function get_q0_impedance(params)
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    Qf = modal.Q_w[1] - modal.F_w[1]
    Gq = (G * modal.q_w)[1]
    return - (Qf / Gq)
end

function main()
    println("--- Deconstruction Step 3: H (Depth) Scaling ---")
    
    L = 0.05
    d = 0.03
    f = 80.0
    Ldom = 1.0
    
    H_list = [0.02, 0.04, 0.06, 0.08]
    Z_list = ComplexF64[]
    
    for H in H_list
        params = Surferbot.FlexibleParams(
            L_raft = L, domain_depth = H, L_domain = Ldom,
            omega = 2pi * f, d = d, EI = 1e6, rho_raft = 0.05,
            n = 200, M = 40, motor_position = 0.0, motor_force = 1.0
        )
        Z = get_q0_impedance(params)
        push!(Z_list, Z)
        @printf("   H = %.3f m | Z = %.3e + %.3ei\n", H, real(Z), imag(Z))
    end
    
    log_H = log.(H_list)
    log_Z = log.(abs.(Z_list))
    gamma = (log_Z[end] - log_Z[1]) / (log_H[end] - log_H[1])
    
    println("\n--- Findings ---")
    @printf("H (Depth) Scaling Exponent: %.4f\n", gamma)
    
    if abs(gamma + 1.0) < 0.2
        println("Hypothesis H_H_inv: Impedance scales with 1/H.")
    elseif abs(gamma - 0.0) < 0.1
        println("Hypothesis H_H0: Impedance is INDEPENDENT of depth.")
    else
        @printf("Trend is H^{%.2f}.\n", gamma)
    end
end

main()

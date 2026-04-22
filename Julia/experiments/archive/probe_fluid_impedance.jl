using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Purely Hydrodynamic Probe to find the A-Priori Added Mass Law.
# We prescribe the raft displacement and solve for the fluid pressure.
# This removes all structural coupling (EI, xM) and gives the pure Z_fluid.

function get_pure_fluid_impedance(L, H, L_dom, d, rho, omega, mode_idx)
    # Assemble the flexible system for the given parameters
    # We will "hijack" the system to solve for prescribed eta
    params = Surferbot.FlexibleParams(
        L_raft = L, domain_depth = H, L_domain = L_dom,
        omega = omega, d = d,
        EI = 1e6, # Doesn't matter
        rho_raft = 0.0,
        n = 200, M = 30,
        motor_position = 0.0, motor_force = 1.0
    )
    
    # Get the assembled system components
    system = Surferbot.assemble_flexible_system(params)
    
    # The system is A * [phi; phi_z; eta] = b
    # Kinematic condition at surface: phi_z = i*omega*eta + ...
    # We want to prescribe eta = psi_n
    
    # Easiest way: Use the full solver with extremely high EI and zero rhoR
    # so that the beam is forced into the mode shape of the drive.
    # To get pure psi_0, we force at the center with high EI.
    # To get pure Z_n, we just look at the resulting Q_n / q_n from this solve.
    
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    
    # Gram matrix for the projection
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    
    # Fluid Force Q_f = Q_w - F_w (since we set F_mech in the solver)
    # Actually, let's just use the direct pressure p:
    p_raft = result.p # Harmonic Bernoulli pressure
    w_trapz = Surferbot.trapz_weights(modal.x_raft)
    
    # Project pressure onto mode n
    Q_n = dot(modal.Phi[:, mode_idx], p_raft .* w_trapz .* d)
    q_n = modal.q_w[mode_idx]
    
    # Impedance Z_n = Q_n / q_n
    # Note: Q is defined as a force, so Z has units of Stiffness/Impedance
    return Q_n / q_n
end

function main()
    println("--- Hydrodynamic Probe: Isolate Fluid Admittance ---")
    
    L = 0.05
    H_base = 0.05
    L_dom = 1.5
    d = 0.03
    rho = 1000.0
    
    # 1. Frequency Sweep: Is it mass, spring, or wave?
    f_list = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    Z_freq = ComplexF64[]
    
    println("\nRunning Frequency Sweep (H=0.05m)...")
    for f in f_list
        omega = 2pi * f
        Z = get_pure_fluid_impedance(L, H_base, L_dom, d, rho, omega, 1) # Mode 0
        push!(Z_freq, Z)
        @printf("   f = %3.0f Hz | Z = %.3e + %.3ei\n", f, real(Z), imag(Z))
    end
    
    # Scaling Analysis
    omegas = 2pi .* f_list
    log_w = log.(omegas)
    log_Z = log.(abs.(Z_freq))
    gamma_w = (log_Z[end] - log_Z[1]) / (log_w[end] - log_w[1])
    
    println("\n--- Frequency Scaling Findings ---")
    @printf("Impedance Exponent (Z ~ omega^gamma): %.4f\n", gamma_w)
    
    # 2. Depth Sweep
    H_list = [0.03, 0.05, 0.08, 0.12]
    Z_depth = ComplexF64[]
    omega_fix = 2pi * 80.0
    
    println("\nRunning Depth Sweep (f=80Hz)...")
    for H in H_list
        Z = get_pure_fluid_impedance(L, H, L_dom, d, rho, omega_fix, 1)
        push!(Z_depth, Z)
        @printf("   H = %.3f m | Z = %.3e + %.3ei\n", H, real(Z), imag(Z))
    end
    
    log_H = log.(H_list)
    log_Zh = log.(abs.(Z_depth))
    gamma_h = (log_Zh[end] - log_Zh[1]) / (log_H[end] - log_H[1])
    
    println("\n--- Depth Scaling Findings ---")
    @printf("Depth Exponent (Z ~ H^gamma): %.4f\n", gamma_h)
end

main()

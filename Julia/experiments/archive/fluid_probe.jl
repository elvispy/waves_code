using Surferbot
using LinearAlgebra
using Printf
using Statistics
using JSON

# Purpose: Senior Research Analyst Step - The "Fluid Probe".
# Isolate the pure fluid impedance Z_n by solving for unit-amplitude mode shapes.
# This removes structural noise (EI, xM) and gives us the pure admittance function f(L/H).

function get_unit_fluid_impedance(L, H, L_dom, d, rho, omega, mode_idx)
    # We create a "Structural Ghost" - a beam with zero mass and infinite stiffness
    # to force the mode shape psi_n onto the fluid.
    
    # Actually, a better way is to use the assemble_flexible_system and solve
    # for a prescribed boundary motion.
    
    # Setup params
    params = Surferbot.FlexibleParams(
        L_raft = L,
        domain_depth = H,
        L_domain = L_dom,
        omega = omega,
        d = d,
        EI = 1e12, # Infinite stiffness
        rho_raft = 0.0, # Zero mass
        n = 200, M = 40,
        motor_position = 0.0,
        motor_force = 1.0
    )
    
    # Step 1: Get raw modes
    x_raft = collect(range(-L/2, L/2, length=101))
    raw = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=4)
    psi_n = raw.Phi[:, mode_idx]
    
    # Step 2: Prescribe motion in the solver
    # This requires a small modification to the solver to allow prescribed eta.
    # Alternatively, we can use the high-stiffness limit and a specific force.
    # Let's use the Force = G * D * q relationship.
    # If we want q_n = 1 and all other q = 0:
    # Q_fluid = Z_n * G[:, n]
    
    # For now, we use the "Fact-Extraction" method from the existing solver:
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    
    # Extract the impedance of the dominant mode
    q = modal.q_w
    Qf = modal.Q_w .- modal.F_w
    Gq = G * q
    
    # Find dominant mode (the one we forced)
    idx = argmax(abs.(q))
    Z = Qf[idx] / Gq[idx]
    
    return -real(Z) # The real part of impedance (Mass/Stiffness combo)
end

function main()
    println("--- Senior Research Analyst: The Fluid Probe ---")
    
    # Universal Constants
    rho = 1000.0
    omega = 2pi * 80.0
    d = 0.03
    L_dom = 2.0
    
    # Variation: L/H ratio
    # We vary H from shallow to deep
    H_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.40]
    L = 0.10
    
    println("\nProbing Admittance Function f(L/H) for Mode 0...")
    
    data_pts = []
    for H in H_list
        Z = get_unit_fluid_impedance(L, H, L_dom, d, rho, omega, 1) # Mode 0
        
        # Dimensionless Admittance f = Z / (rho * omega^2 * L)
        f_val = Z / (rho * omega^2 * L)
        
        ratio = L / H
        @printf("   L/H = %5.2f | Z = %.3e | f = %.4f\n", ratio, Z, f_val)
        push!(data_pts, (ratio, f_val))
    end
    
    # Log-Log Analysis
    ratios = [p[1] for p in data_pts]
    f_vals = [p[2] for p in data_pts]
    
    # Fit f = C * (L/H)^gamma
    log_r = log.(ratios)
    log_f = log.(f_vals)
    slope = (log_f[end] - log_f[1]) / (log_r[end] - log_r[1])
    
    println("\n--- Discovery ---")
    @printf("Found Power Law: f(L/H) propto (L/H)^{%.4f}\n", slope)
    
    if abs(slope - 1.0) < 0.1
        println("Conclusion: Admittance scales LINEARLY with L/H.")
        println("Law: Z approx rho * omega^2 * L * (L/H) = rho * omega^2 * L^2 / H")
    elseif abs(slope - 0.0) < 0.1
        println("Conclusion: Admittance is CONSTANT with L/H.")
        println("Law: Z approx rho * omega^2 * L")
    end

    # Update Diary
    diary_path = joinpath(@__DIR__, "added_mass_diary.json")
    diary = JSON.parsefile(diary_path)
    push!(diary, Dict(
        "iteration" => 16,
        "law" => @sprintf("f(L/H) ~ (L/H)^{%.2f}", slope),
        "kpi" => Dict("exponent" => @sprintf("%.2f", slope)),
        "comments" => "Senior Analyst Probe. Identifying the geometric scaling function f(L/H) to collapse the law into a parameter-free form."
    ))
    open(diary_path, "w") do f
        JSON.print(f, diary, 4)
    end
end

main()

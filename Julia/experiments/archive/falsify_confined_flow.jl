using Surferbot
using LinearAlgebra
using Printf
using Statistics

# Purpose: Falsify the "Confined Poisson Pressure" Law.
# Law: p_xx = -(rho * omega^2 / H) * eta(x)
# This accounts for global mass conservation in a 2D channel.

function solve_poisson_pressure(x_raft, eta_raft, L_dom, H, rho, omega)
    # Solve p_xx = - (rho * omega^2 / H) * eta(x)
    # with p = 0 at the tank ends x = +/- L_dom/2
    # Analytical solution for eta=1: p(x) = (rho * omega^2 / 2H) * ((L_dom/2)^2 - x^2)
    
    # We'll use a simple FD solver to handle any eta(x) mode shape
    Nx = length(x_raft)
    dx = x_raft[2] - x_raft[1]
    
    # But wait, the beam is only in part of the domain. 
    # The pressure p(x) exists in the WHOLE tank [-L_dom/2, L_dom/2].
    # In the free surface region (|x| > L/2), the BC is p = rho * g * eta + ...
    # At high freq, the free surface is effectively a pressure release (p=0).
    
    x_full = collect(range(-L_dom/2, L_dom/2, length=1001))
    dx_f = x_full[2] - x_full[1]
    rhs = zeros(length(x_full))
    
    # Map eta_raft to the full grid
    L_raft = x_raft[end] - x_raft[1]
    for i in eachindex(x_full)
        if abs(x_full[i]) <= L_raft/2
            # Interpolate eta (rough for heave n=0)
            rhs[i] = - (rho * omega^2 / H) * 1.0 # testing mode 0
        end
    end
    
    # 1D Poisson: p_xx = rhs, p(ends) = 0
    # A * p = rhs * dx^2
    Nf = length(x_full)
    A = Tridiagonal(ones(Nf-1), -2*ones(Nf), ones(Nf-1))
    p = A \ (rhs .* dx_f^2)
    
    # Extract p on the raft region
    p_raft = zeros(Nx)
    for i in eachindex(x_raft)
        # find closest in x_full
        idx = argmin(abs.(x_full .- x_raft[i]))
        p_raft[i] = p[idx]
    end
    
    return p_raft
end

function main()
    println("--- Drawing Board: Confined Flow Law ---")
    
    # 1. New Parameters
    L = 0.08
    L_dom = 0.80 # 10x raft length
    H = 0.04
    f = 50.0
    omega = 2pi * f
    d = 0.04
    rho = 1000.0
    g = 9.81
    
    params = Surferbot.FlexibleParams(
        L_raft = L,
        domain_depth = H,
        L_domain = L_dom,
        omega = omega,
        d = d,
        EI = 1e4,
        rho_raft = 0.05,
        n = 200, 
        M = 40,
        motor_position = 0.0, # Pure heave focus
        motor_force = 1.0
    )
    
    # 2. A Priori added mass from Poisson Law (Mode 0)
    # ma = Integral(p * psi) * d / (omega^2 * Integral(psi^2))
    x_raft = collect(range(-L/2, L/2, length=101))
    p_pred = solve_poisson_pressure(x_raft, ones(101), L_dom, H, rho, omega)
    
    w = Surferbot.trapz_weights(x_raft)
    ma_pred = (sum(p_pred .* w) * d) / (omega^2 * sum(ones(101) .* w))
    
    @printf("A Priori ma,0 (Poisson Law): %8.4f kg/m\n", ma_pred)
    
    # 3. Solve
    println("Running Full Solver...")
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    
    # 4. Compare
    # True ma = real( (Q_w - F_w) / (G * q_w * omega^2) )
    # But wait, Q_w = G * (fluid_impedance) * q_w
    # Let's just compare the Q_fluid_w directly
    Q_fluid_num = modal.Q_w[1] - modal.F_w[1]
    
    # Predict Q_fluid_apriori = G[1,1] * (ma_pred * omega^2 - d*rho*g) * q_w[1]
    # (Simplified diagonal coupling)
    hydro = d * rho * g
    G11 = modal.gram_cond # approximation or calculate
    # Better: use the actual G from Modal
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    
    Q_fluid_ap = G[1,1] * (ma_pred * omega^2 - hydro) * modal.q_w[1]
    
    @printf("\n--- Result for Heave (n=0) ---\n")
    @printf("Numerical Qf:  %.4e\n", real(Q_fluid_num))
    @printf("A Priori Qf:   %.4e\n", real(Q_fluid_ap))
    @printf("Error:         %.2f%%\n", abs(real(Q_fluid_ap) - real(Q_fluid_num))/abs(real(Q_fluid_num)) * 100)
    
    println("\nDissection:")
    # Check scaling
    # If Error is constant across configurations, Law is correct but missing a factor (e.g. 3D effect).
    # If Error varies, Law is missing physics (e.g. wave radiation).
end

main()

using Surferbot
using LinearAlgebra
using Printf
using Statistics
using Random

# Purpose: Generate the "Evidence Dataset" - 10 randomized high-fidelity solves
# where L, H, d, and EI are ALL varied. This is the only way to find a geometric law.

function main()
    Random.seed!(42)
    n_pts = 10
    
    # Store results for regression
    data_results = []
    
    println("--- Generating Evidence Dataset (10 Randomized Solves) ---")
    
    for i in 1:n_pts
        # Random parameters in physically relevant ranges
        L = 0.04 + rand() * 0.10      # [0.04, 0.14]
        H = 0.03 + rand() * 0.07      # [0.03, 0.10]
        d = 0.02 + rand() * 0.06      # [0.02, 0.08]
        EI = 10.0^(3.0 + rand()*3.0)  # [1e3, 1e6]
        f = 40.0 + rand() * 60.0      # [40, 100] Hz
        
        @printf("Solve %d/%d: L=%.3f, H=%.3f, d=%.3f, EI=%.1e, f=%.1f\n", i, n_pts, L, H, d, EI, f)
        
        params = Surferbot.FlexibleParams(
            L_raft = L,
            domain_depth = H,
            L_domain = max(2.0, 10*L),
            omega = 2pi * f,
            d = d,
            EI = EI,
            rho_raft = 0.05,
            n = 120, M = 30, # Safe resolution
            motor_position = 0.0,
            motor_force = 1.0
        )
        
        result = Surferbot.flexible_solver(params)
        modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
        
        # Extract q and Q in the W-basis
        # G is needed for the pure fluid force Q_f = Q_w - F_w
        G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
        
        push!(data_results, (
            L=L, H=H, d=d, EI=EI, f=f, omega=2pi*f,
            q = modal.q_w,
            Q = modal.Q_w,
            F = modal.F_w,
            G = G
        ))
    end
    
    # --- LOG-LOG REGRESSION TO FIND THE LAW ---
    # Target: |Q_f| = |Q_w - F_w|
    # Hypothesis: |Q_f| = C * L^a * H^b * d^c * w^e * |q|^k
    
    # We use mode 0 (Heave) for the first-order law derivation
    X = zeros(n_pts, 5) # log(L), log(H), log(d), log(w), log(|q0|)
    y = zeros(n_pts)    # log(|Qf0|)
    
    for i in 1:n_pts
        res = data_results[i]
        Qf = abs(res.Q[1] - res.F[1])
        X[i, 1] = log(res.L)
        X[i, 2] = log(res.H)
        X[i, 3] = log(res.d)
        X[i, 4] = log(res.omega)
        X[i, 5] = log(abs(res.q[1]))
        y[i] = log(Qf)
    end
    
    # Solve y = X * p  => p = (X'X) \ X'y
    p = (X' * X) \ (X' * y)
    
    println("\n--- The Empirical Law Found (Mode 0) ---")
    @printf("|Q_f,0| propto L^{%.2f} * H^{%.2f} * d^{%.2f} * omega^{%.2f} * |q_0|^{%.2f}\n", 
            p[1], p[2], p[3], p[4], p[5])
    
    # Check Stability
    residuals = y - X * p
    mean_err = mean(abs.(exp.(residuals) .- 1.0))
    @printf("Mean Rel Error of the Law: %.2f%%\n", mean_err * 100)
    
    if mean_err <= 0.05
        println("\nConclusion: Universal Power Law Established.")
    else
        println("\nConclusion: Complexity remains. Refining model with non-dimensional groups (kH).")
    end
end

main()

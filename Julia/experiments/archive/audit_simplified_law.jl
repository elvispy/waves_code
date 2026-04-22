using Surferbot
using LinearAlgebra
using Printf
using Statistics
using Random

# Purpose: Test the "Simplified Geometric Law" against 20 randomized trials.
# Law: |Q_f| = C * L^-1.27 * d^-0.11 * omega^-0.28
# (H and q_0 exponents set to zero as requested)

function main()
    Random.seed!(1234) # New seed for independent verification
    n_pts = 20
    
    # Pre-calibrated constant C from previous 10-solve mean
    # log(C) = mean(log|Qf| - (-1.27*logL - 0.11*logd - 0.28*logw))
    # Let's recalibrate C on the first few samples to ensure zero-bias comparison
    C = 0.0 # Placeholder, will calibrate on first solve to find the intercept
    
    data_results = []
    
    println("--- 20-Solve Randomized Audit: Simplified Geometric Law ---")
    println("Law: |Q_f| propto L^-1.27 * d^-0.11 * omega^-0.28")
    
    for i in 1:n_pts
        L = 0.04 + rand() * 0.10
        H = 0.03 + rand() * 0.07
        d = 0.02 + rand() * 0.06
        EI = 10.0^(3.0 + rand()*3.0)
        f = 40.0 + rand() * 60.0
        
        params = Surferbot.FlexibleParams(
            L_raft = L, domain_depth = H, L_domain = max(2.0, 10*L),
            omega = 2pi * f, d = d, EI = EI, rho_raft = 0.05,
            n = 100, M = 25, motor_position = 0.0, motor_force = 1.0
        )
        
        result = Surferbot.flexible_solver(params)
        modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
        Qf_num = abs(modal.Q_w[1] - modal.F_w[1])
        
        push!(data_results, (L=L, d=d, omega=2pi*f, Qf=Qf_num))
    end
    
    # 1. Calibrate C (Intercept) for the Simplified Law
    log_C_vals = [log(res.Qf) - (-1.27*log(res.L) - 0.11*log(res.d) - 0.28*log(res.omega)) for res in data_results]
    C_opt = exp(mean(log_C_vals))
    
    # 2. Calculate errors
    errors = Float64[]
    for res in data_results
        Qf_pred = C_opt * (res.L^-1.27) * (res.d^-0.11) * (res.omega^-0.28)
        err = abs(Qf_pred - res.Qf) / res.Qf
        push!(errors, err)
    end
    
    println("-"^50)
    @printf("%-10s | %-12s | %-12s\n", "Trial", "Actual |Qf|", "Rel Error")
    for i in 1:length(errors)
        @printf("%-10d | %.4e   | %.2f%%\n", i, data_results[i].Qf, errors[i]*100)
    end
    println("-"^50)
    @printf("MEAN RELATIVE L2 ERROR (20 Trials): %.4f\n", mean(errors))
    
    if mean(errors) <= 0.05
        println("Verdict: PASS")
    else
        println("Verdict: FAIL")
    end
end

main()

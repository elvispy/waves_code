using Surferbot
using LinearAlgebra
using Printf
using Statistics
using JSON

# Purpose: Evidence-based Scaling Sweep to find the ma(d, H) law.
# We vary H and d independently and extract numerical added mass.

function get_numerical_man(params, mode_idx)
    result = Surferbot.flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=4, verbose=false)
    
    # Balance in W-basis: G * ( (EI*beta^4 - rhoR*omega^2)*q - (Q - F) ) = 0
    # True Fluid Force Q_f = Q_w - F_w
    # A Priori Law: Q_f approx G * (ma * omega^2 - d*rho*g) * q_w
    
    # We solve for the effective scalar added mass ma_eff for mode n
    G = modal.Phi' * (modal.Phi .* Surferbot.trapz_weights(modal.x_raft))
    
    q_n = modal.q_w[mode_idx]
    Qf_n = modal.Q_w[mode_idx] - modal.F_w[mode_idx]
    Gq_n = (G * modal.q_w)[mode_idx]
    
    omega = params.omega
    rho = params.rho
    g = params.g
    d = params.d
    
    # Qf_n = Gq_n * (ma * omega^2 - d*rho*g)
    # ma = (Qf_n / Gq_n + d*rho*g) / omega^2
    ma_eff = real( (Qf_n / Gq_n + d * rho * g) / omega^2 )
    
    return ma_eff
end

function main()
    println("--- Scaling Sweep: Depth (H) and Width (d) ---")
    
    L = 0.10
    L_dom = 1.0
    f = 60.0
    omega = 2pi * f
    rho = 1000.0
    g = 9.81
    
    # Shared Base Params
    base_p = (
        L_raft = L,
        L_domain = L_dom,
        omega = omega,
        EI = 1e4,
        rho_raft = 0.05,
        n = 160, # Safe RAM
        M = 30,  # Safe RAM
        motor_position = 0.0,
        motor_force = 1.0
    )

    # 1. Depth Sweep (Fix d = 0.04)
    H_list = [0.03, 0.05, 0.07, 0.09]
    ma_H_heave = Float64[]
    ma_H_bend = Float64[]
    
    println("\nRunning Depth Sweep (d=0.04)...")
    for H in H_list
        p = Surferbot.FlexibleParams(; base_p..., domain_depth = H, d = 0.04)
        ma0 = get_numerical_man(p, 1)
        ma2 = get_numerical_man(p, 3)
        push!(ma_H_heave, ma0)
        push!(ma_H_bend, ma2)
        @printf("   H = %.2f m: ma,0 = %.4f, ma,2 = %.4f\n", H, ma0, ma2)
    end

    # 2. Width Sweep (Fix H = 0.05)
    d_list = [0.02, 0.04, 0.06, 0.08]
    ma_d_heave = Float64[]
    ma_d_bend = Float64[]
    
    println("\nRunning Width Sweep (H=0.05)...")
    for d in d_list
        p = Surferbot.FlexibleParams(; base_p..., domain_depth = 0.05, d = d)
        ma0 = get_numerical_man(p, 1)
        ma2 = get_numerical_man(p, 3)
        push!(ma_d_heave, ma0)
        push!(ma_d_bend, ma2)
        @printf("   d = %.2f m: ma,0 = %.4f, ma,2 = %.4f\n", d, ma0, ma2)
    end

    # 3. Regression to find exponents
    # ma = C * d^alpha * H^beta
    
    # Alpha (Width)
    log_d = log.(d_list)
    log_ma_d = log.(ma_d_heave)
    alpha = (log_ma_d[end] - log_ma_d[1]) / (log_d[end] - log_d[1])
    
    # Beta (Depth)
    log_H = log.(H_list)
    log_ma_H = log.(ma_H_heave)
    beta_pow = (log_ma_H[end] - log_ma_H[1]) / (log_H[end] - log_H[1])
    
    println("\n--- Derived Scaling Exponents (Mode 0) ---")
    @printf("Width Scaling alpha (ma ~ d^alpha): %.4f\n", alpha)
    @printf("Depth Scaling beta  (ma ~ H^beta):  %.4f\n", beta_pow)
    
    # Update Diary with Evidence
    diary_path = joinpath(@__DIR__, "added_mass_diary.json")
    diary = JSON.parsefile(diary_path)
    push!(diary, Dict(
        "iteration" => 9,
        "law" => @sprintf("Scaling Sweep: ma propto d^{%.2f} * H^{%.2f}", alpha, beta_pow),
        "kpi" => Dict(
            "alpha_width" => @sprintf("%.2f", alpha),
            "beta_depth" => @sprintf("%.2f", beta_pow)
        ),
        "comments" => "Evidence-based sweep. If alpha ~ 1, the width d is just a force scaler. If alpha ~ 2 or 3, width is physically entraining fluid. If beta ~ -1, confined flow dominates."
    ))
    open(diary_path, "w") do f
        JSON.print(f, diary, 4)
    end
end

main()

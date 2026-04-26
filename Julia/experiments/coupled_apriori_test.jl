using Surferbot
using JLD2
using Printf
using LinearAlgebra
using Statistics

"""
coupled_apriori_test.jl

Tests the diagonal hydrodynamic coupling hypothesis (Q_n ≈ H_nn * q_n) 
for the coupled case. Extracts the effective impedance shift from numerical 
solves to inform a priori prediction laws.
"""

# Establish facts for the coupled analytical a priori law.

function main()
    # 1. Load a coupled sweep point
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_file = joinpath(output_dir, "jld2", "sweep_motor_position_EI_coupled_from_matlab.jld2")
    artifact = load_sweep(sweep_file)
    
    # We pick a point where alpha is small
    target_log10_EI = -3.38
    EI_idx = argmin(abs.(log10.(artifact.parameter_axes.EI) .- target_log10_EI))
    EI = artifact.parameter_axes.EI[EI_idx]
    
    # In the coupled sweep, we find the xM where alpha=0
    # For now, let's just pick a point from the metadata if available, 
    # or just run a single solve.
    xM = 0.38 * artifact.base_params.L_raft
    
    params = apply_parameter_overrides(artifact.base_params, (EI=EI, motor_position=xM))
    result = flexible_solver(params)
    modal = decompose_raft_freefree_modes(result; num_modes=8, verbose=false)
    
    println("--- COUPLED MODAL BALANCE ANALYSIS ---")
    @printf("Case: log10EI=%.3f, xM/L=%.4f, alpha=%.4e\n", log10(EI), xM/params.L_raft, result.thrust)
    
    # 2. Extract H matrix a posteriori
    # Since Q_n = sum_m H_nm q_m, we have H_nm ≈ <W_n, p_m>
    # To get the full H matrix, we would need to run N solves with q_n=delta_nm.
    # But we can check the DIAGONAL H hypothesis: Q_n ≈ H_nn * q_n
    
    n = modal.n
    q = modal.q_w
    Q = modal.Q # Need Q in W-basis
    
    # Transform Q to W-basis
    w_w = Surferbot.trapz_weights(modal.x_raft)
    G = modal.Phi' * (modal.Phi .* w_w)
    B = modal.Phi' * (modal.Psi .* w_w)
    T = G \ B
    Q_w = T * modal.Q
    
    println("\nmode n   |q_n|       |Q_n|       |Q_n / q_n|   phase(Q/q) [deg]")
    for j in eachindex(n)
        ratio = Q_w[j] / (q[j] + eps()im)
        @printf("%2d    %d   %.3e   %.3e   %.3e   %7.1f\n", j, n[j], abs(q[j]), abs(Q_w[j]), abs(ratio), rad2deg(angle(ratio)))
    end
    
    # 3. Test the A Priori coupled law:
    # S \propto W_end^T (G D - H)^-1 F^W = 0
    # where H is the diagonal added mass matrix
    H_diag = diagm(0 => Q_w ./ q)
    D = diagm(0 => [EI * modal.beta[k]^4 - params.rho_raft * params.omega^2 for k in eachindex(modal.beta)])
    
    F_w = T * modal.F
    W_end = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0][1:length(n)]
    
    # Predict q_apriori
    q_pred = (G * D - H_diag) \ (-F_w)
    
    println("\n--- A PRIORI COEFFICIENT PREDICTION (with diagonal H) ---")
    println("mode n   |q_actual|   |q_pred|    rel_err")
    for j in eachindex(n)
        err = abs(q[j] - q_pred[j]) / abs(q[j])
        @printf("%2d    %d   %.3e   %.3e   %.2f%%\n", j, n[j], abs(q[j]), abs(q_pred[j]), err*100)
    end
    
    S_actual = dot(W_end, q)
    S_pred = dot(W_end, q_pred)
    println("\nS_actual = ", S_actual)
    println("S_pred   = ", S_pred)
end

main()

using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Plots
using Printf

# Purpose: Test the Obsidian relation phi_n = (i*omega / (k*tanh(kH))) * q_n
# using direct numerical evidence from the potential field.

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    # Sample 32
    i = 32
    L, H, L_dom, d, rho, g = data[i, col("L_raft")], 0.05, 1.5, 0.03, 1000.0, 9.81
    omega = data[i, col("omega")]
    EI = data[i, col("EI")]
    
    # 1. Run Solver to get full phi field
    params = Surferbot.FlexibleParams(
        L_raft = L, domain_depth = H, L_domain = L_dom,
        omega = omega, d = d, EI = EI,
        motor_position = data[i, col("motor_position")],
        motor_force = 1.0,
        n = 200, M = 40 # High res
    )
    result = Surferbot.flexible_solver(params)
    
    # 2. Extract Surface Potential (phi) and Elevation (eta) on the Raft
    contact = result.metadata.args.x_contact
    x_raft = result.x[contact]
    phi_raft = result.phi[end, contact] # phi at z=0
    eta_raft = result.eta[contact]
    w_trapz = Surferbot.trapz_weights(x_raft)
    
    # 3. Modal Projections (W-basis / Phi)
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x_raft, L; num_modes=4)
    Phi = raw_basis.Phi
    G = Phi' * (Phi .* w_trapz)
    
    # Modal coefficients q_n and phi_n
    # G * q_w = Phi' * W * eta
    q_w = G \ (Phi' * (eta_raft .* w_trapz))
    phi_w = G \ (Phi' * (phi_raft .* w_trapz))
    
    # 4. Compare with Obsidian Theory
    # R_theory = i*omega / (k * tanh(k*H))
    # We test with k = k_wave and k = beta_n
    k_wave = real(Surferbot.dispersion_k(omega, g, 0.0, 0.0, 0.0, rho))
    R_theory_wave = (im * omega) / (k_wave * tanh(k_wave * H))
    
    println("--- Obsidian Relation Test (Sample 32) ---")
    @printf("Mode | q_w magnitude | phi_w/q_w (Numerical) | R_theory (Wave-k) | Error Ratio\n")
    println("-"^80)
    
    for n in 1:4
        R_num = phi_w[n] / q_w[n]
        err_ratio = abs(R_num) / abs(R_theory_wave)
        @printf("n=%-2d | %.3e     | %.3e + %.3ei | %.3e + %.3ei | %.1f\n", 
                n-1, abs(q_w[n]), real(R_num), imag(R_num), real(R_theory_wave), imag(R_theory_wave), err_ratio)
    end
    
    # 5. Visualization
    p1 = plot(x_raft, abs.(phi_raft), label="|phi_num|", color=:blue, lw=2)
    # Theory spatial: phi_ap(x) = R_theory * eta(x)
    plot!(p1, x_raft, abs.(R_theory_wave .* eta_raft), label="|phi_theory|", color=:red, linestyle=:dash)
    title!(p1, "Surface Potential Profile (Numerical vs Theory)")
    
    save_path = joinpath(@__DIR__, "..", "output", "obsidian_phi_test.png")
    savefig(p1, save_path)
    println("\nDeconstruction plot saved to $save_path")
end

main()

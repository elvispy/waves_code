using Surferbot
using LinearAlgebra
using Printf

function main()
    params = FlexibleParams(
        sigma = 0.0,
        rho = 1000.0,
        omega = 2 * pi * 80.0,
        nu = 1e-6,
        g = 9.81,
        L_raft = 0.05,
        motor_position = 0.012,
        d = 0.05,
        EI = 1.64e-1,
        rho_raft = 0.052, 
        n = 120, M = 40
    )
    result = flexible_solver(params)
    
    args = result.metadata.args
    contact_mask = args.x_contact
    Nr = count(contact_mask)
    
    D2r_adim = Matrix(Surferbot.getNonCompactFDmatrix(Nr, 1.0, 2, params.ooa)) / (args.dx / args.L_c)^2
    D4r_adim = Matrix(Surferbot.getNonCompactFDmatrix(Nr, 1.0, 4, params.ooa)) / (args.dx / args.L_c)^4
    # Wait, the solver doesn't use D4 directly, it uses D2 * D2? 
    # Let's use D2r_adim^2 as the 4th derivative to perfectly match the solver's assembly (S32 -> D2r, S13 -> D2r).
    D4r = (D2r_adim^2) ./ args.L_c^4
    
    phi_surf = result.phi[end, :]
    phi_raft = phi_surf[contact_mask] .* (args.t_c / args.L_c^2)
    eta_raft = result.eta[contact_mask] ./ args.L_c 
    
    Gamma = args.nd_groups.Gamma
    Re = args.nd_groups.Re
    Fr = args.nd_groups.Fr
    
    p_adim_correct = -im * Gamma .* phi_raft .- (2 * Gamma / Re) .* (D2r_adim * phi_raft) .- (Gamma / Fr^2) .* eta_raft
    p_dim_correct = p_adim_correct .* (args.m_c * args.L_c / args.t_c^2) ./ args.L_c^2
    
    eta_dim = result.eta[contact_mask]
    
    # Calculate continuous PDE LHS discretely:
    # EI * eta_xxxx - rho_R * omega^2 * eta
    LHS_discrete = params.EI .* (D4r * eta_dim) .- params.rho_raft * params.omega^2 .* eta_dim
    RHS_discrete = args.d .* p_dim_correct .- args.loads
    
    w = Surferbot.Modal.trapz_weights(args.x[contact_mask])
    raw = Surferbot.Modal.build_raw_freefree_basis(args.x[contact_mask], args.L_raft; num_modes=10, include_rigid=true)
    Phi = raw.Phi
    G = Phi' * (Phi .* w)
    
    Weta = eta_dim .* w
    q_w = G \ (Phi' * Weta)
    
    # Project the discrete LHS and RHS
    WLHS = LHS_discrete .* w
    Proj_LHS = G \ (Phi' * WLHS)
    
    WRHS = RHS_discrete .* w
    Proj_RHS = G \ (Phi' * WRHS)
    
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta = [0.0; 0.0; betaL ./ args.L_raft]
    
    println("-"^100)
    @printf("%-5s | %-15s %-15s %-15s %-15s\n", "Mode", "Dn*q_n", "Proj_LHS", "Proj_RHS", "Q_n - F_n")
    println("-"^100)

    for i in 1:6
        Dn = params.EI * beta[i]^4 - params.rho_raft * params.omega^2
        lhs_ana = Dn * q_w[i]
        
        # Note: Proj_RHS is exactly Q_n - F_n
        @printf("%-5d | %-15.4e %-15.4e %-15.4e %-15.4e\n", 
                i-1, real(lhs_ana), real(Proj_LHS[i]), real(Proj_RHS[i]), real(Proj_RHS[i]))
    end
end
main()

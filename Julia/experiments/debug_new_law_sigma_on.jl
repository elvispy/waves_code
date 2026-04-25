using Surferbot
using LinearAlgebra
using Printf
using CSV
using DataFrames
using Random

"""
Julia/experiments/debug_new_law_sigma_on.jl

Validates the new force balance law including the capillary boundary coupling:
D_n q_n + F_sigma_n = Q_n - F_n
where F_sigma_n = d * sigma * (W_n(L/2) * eta_x(L/2^+) - W_n(-L/2) * eta_x(-L/2^-))
"""

function main()
    csv_path = "Julia/output/csv/sweeper_coupled_full_grid.csv"
    if !isfile(csv_path)
        error("CSV not found: $csv_path")
    end
    df = CSV.read(csv_path, DataFrame)
    
    Random.seed!(42)
    sample_indices = randperm(nrow(df))[1:7]
    df_sample = df[sample_indices, :]
    
    L_raft = 0.05
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta = [0.0; 0.0; betaL ./ L_raft]
    
    println("="^105)
    @printf("%-4s | %-4s | %-8s %-8s | %-12s %-12s %-12s | %-12s %-10s\n", 
            "Iter", "Mode", "logEI", "xM/L", "Dn*q_n", "F_sigma", "LHS_total", "Q_n - F_n", "Rel_Res %")
    println("="^105)

    for (iter, row) in enumerate(eachrow(df_sample))
        EI = 10^row.log10_EI
        motor_pos = row.xM_over_L * L_raft
        
        params = FlexibleParams(
            sigma = 72.2e-3, # Physical water value
            rho = 1000.0,
            omega = 2 * pi * 80.0,
            nu = 1e-6,
            g = 9.81,
            L_raft = L_raft,
            motor_position = motor_pos,
            d = 0.05,
            EI = EI,
            rho_raft = 0.052
        )
        
        result = flexible_solver(params)
        args = result.metadata.args
        contact_mask = args.x_contact
        Nr = count(contact_mask)
        
        D2r = Matrix(Surferbot.getNonCompactFDmatrix(Nr, 1.0, 2, params.ooa)) / (args.dx / args.L_c)^2
        
        phi_surf = result.phi[end, :]
        phi_raft = phi_surf[contact_mask] .* (args.t_c / args.L_c^2)
        eta_raft = result.eta[contact_mask] ./ args.L_c 
        
        Gamma = args.nd_groups.Gamma
        Re = args.nd_groups.Re
        Fr = args.nd_groups.Fr
        
        # Correct pressure calculation
        p_adim_correct = -im * Gamma .* phi_raft .- (2 * Gamma / Re) .* (D2r * phi_raft) .- (Gamma / Fr^2) .* eta_raft
        p_dim_correct = p_adim_correct .* (args.m_c * args.L_c / args.t_c^2) ./ args.L_c^2
        
        w = Surferbot.Modal.trapz_weights(args.x[contact_mask])
        raw = Surferbot.Modal.build_raw_freefree_basis(args.x[contact_mask], args.L_raft; num_modes=10, include_rigid=true)
        Phi = raw.Phi
        G = Phi' * (Phi .* w)
        
        Weta = result.eta[contact_mask] .* w
        q_w = G \ (Phi' * Weta)
        
        Wdp = (args.d .* p_dim_correct) .* w
        Q_w = G \ (Phi' * Wdp)
        
        Wf = args.loads .* w # Loads are sized to contact region
        F_w = G \ (Phi' * Wf)
        
        # Calculate surface slopes from the fluid side (one-sided FD)
        xL = findfirst(contact_mask)
        xR = findlast(contact_mask)
        
        D1_fluid = Matrix(Surferbot.getNonCompactFDmatrix(10, 1.0, 1, params.ooa)) / args.dx
        eta_dim = result.eta
        
        eta_x_surf_L = dot(D1_fluid[end, :], eta_dim[(xL - 9):xL])
        eta_x_surf_R = dot(D1_fluid[1, :], eta_dim[xR:(xR + 9)])
        
        # Edges of the basis modes
        W_L = Phi[1, :]
        W_R = Phi[end, :]
        
        Dn_all = [params.EI * beta[i]^4 - params.rho_raft * params.omega^2 for i in 1:10]
        force_damping = 1e-3 * sum(abs.(Dn_all .* q_w[1:10]))
        
        for i in 1:10
            n = i - 1
            Dn = Dn_all[i]
            
            F_sigma = params.d * params.sigma * (W_R[i] * eta_x_surf_R - W_L[i] * eta_x_surf_L)
            
            lhs = Dn * q_w[i]
            lhs_total = lhs + F_sigma
            rhs = Q_w[i] - F_w[i]
            
            residual = abs(lhs_total - rhs)
            denom = abs(rhs) + force_damping
            rel_res = (residual / denom) * 100
            
            @printf("%-4d | %-4d | %-8.2f %-8.3f | %-12.3e %-12.3e %-12.3e | %-12.3e %-10.2f%%\n", 
                    iter, n, row.log10_EI, row.xM_over_L, real(lhs), real(F_sigma), real(lhs_total), real(rhs), rel_res)
        end
        println("-"^105)
    end
end
main()

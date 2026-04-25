using Surferbot
using CSV
using DataFrames
using Random
using Printf
using LinearAlgebra

"""
Julia/experiments/debug_uncoupled_force_balance.jl

Checks the modal force balance equation Dn * qn = -Fn for 10 random points 
in the uncoupled sweep data. Reports the residual as a percentage of the forcing magnitude.
"""

function main()
    csv_path = "Julia/output/csv/sweeper_uncoupled_full_grid.csv"
    if !isfile(csv_path)
        error("CSV not found: $csv_path")
    end
    df = CSV.read(csv_path, DataFrame)
    
    Random.seed!(42)
    sample_indices = randperm(nrow(df))[1:10]
    df_sample = df[sample_indices, :]
    
    L_raft = 0.05
    rho_raft = 0.052
    omega = 2 * π * 80
    
    # We need beta roots for the analytical D_n
    betaL = Surferbot.Modal.freefree_betaL_roots(10)
    beta_roots = [0.0; 0.0; betaL ./ L_raft]
    Dfun(EI, b) = EI * b^4 - rho_raft * omega^2

    println("-"^110)
    @printf("%-5s %-5s %-8s | %-15s %-15s %-15s | %-12s\n", "Row", "Mode", "logEI", "Dn * qn", "-Fn", "Residual", "Rel_Res %")
    println("-"^110)

    for (idx, row) in enumerate(eachrow(df_sample))
        EI = 10^row.log10_EI
        logEI = row.log10_EI
        for n in 0:3
            # Extract coefficients
            qn = complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")])
            Fn = complex(row[Symbol("F_w$(n)_re")], row[Symbol("F_w$(n)_im")])
            
            Dn = Dfun(EI, beta_roots[n+1])
            
            # Modal Balance: Dn * qn = -Fn
            lhs = Dn * qn
            rhs = -Fn
            
            residual = abs(lhs - rhs)
            denominator = max(abs(rhs), 1e-12)
            rel_res = residual / denominator * 100
            
            @printf("%-5d %-5d %-8.2f | %-15.4e %-15.4e %-15.4e | %-12.2f%%\n", 
                    idx, n, logEI, real(lhs), real(rhs), residual, rel_res)
        end
        println("-"^110)
    end
end

main()

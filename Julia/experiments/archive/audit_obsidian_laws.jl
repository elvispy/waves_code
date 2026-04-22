using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Random
using Printf

# Test the three options from Obsidian notes: "Resonance distilled 2.md"
# Law O1: k = k_wave
# Law O2: k = k_schulkes (implicit solve)
# Law O3: k = beta_n

function solve_schulkes_k(omega, EI, rho, rho_R, g, H)
    # omega^2 = (EI*k^5/rho + g*k) / (k*rho_R/rho + coth(kH))
    # We find root of f(k) = (EI*k^5/rho + g*k) - omega^2 * (k*rho_R/rho + coth(kH))
    f(k) = (EI * k^5 / rho + g * k) - omega^2 * (k * rho_R / rho + 1.0 / tanh(k * H))
    
    # Simple bisection for the positive real root
    a, b = 1e-3, 1e5
    if f(a) * f(b) > 0
        return 1.0 # fallback
    end
    for _ in 1:100
        mid = (a + b) / 2
        if f(a) * f(mid) <= 0
            b = mid
        else
            a = mid
        end
    end
    return (a + b) / 2
end

function audit_obsidian_laws(L, H, L_dom, d, rho, g, EI, rho_R, omega, q_num, Q_num)
    # Construct Gram Matrix
    x = collect(range(-L/2, L/2, length=101))
    w = Surferbot.trapz_weights(x)
    raw = Surferbot.Modal.build_raw_freefree_basis(x, L; num_modes=4)
    G = raw.Phi' * (raw.Phi .* w)
    
    # 1. Option 1: k_wave
    k1 = real(Surferbot.dispersion_k(omega, g, 0.0, 0.0, 0.0, rho))
    ma1 = (d * rho) / (k1 * tanh(k1 * H))
    
    # 2. Option 2: k_schulkes
    k2 = solve_schulkes_k(omega, EI, rho, rho_R, g, H)
    ma2 = (d * rho) / (k2 * tanh(k2 * H))
    
    # 3. Option 3: k = beta_n
    ma3_vec = [ (d * rho) / (max(raw.beta[n+1], 1e-3) * tanh(max(raw.beta[n+1], 1e-3) * H)) for n in 0:3 ]
    
    hydro = d * rho * g
    
    # Predict Q_ap = G * diag(omega^2 * ma - hydro) * q
    Z1 = omega^2 * ma1 - hydro
    Z2 = omega^2 * ma2 - hydro
    Z3 = diagm(0 => [omega^2 * m - hydro for m in ma3_vec])
    
    Q_ap1 = G * (Z1 .* q_num)
    Q_ap2 = G * (Z2 .* q_num)
    Q_ap3 = G * (Z3 * q_num)
    
    err1 = norm(Q_num - Q_ap1) / norm(Q_num)
    err2 = norm(Q_num - Q_ap2) / norm(Q_num)
    err3 = norm(Q_num - Q_ap3) / norm(Q_num)
    
    return err1, err2, err3
end

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    Random.seed!(42)
    indices = sort(randperm(size(data, 1))[1:10])
    
    results = []
    
    println("--- Obsidian Law Audit: 3 Options ---")
    @printf("%-10s | %-12s | %-12s | %-12s\n", "Sample", "O1 (Wave)", "O2 (Schulkes)", "O3 (Beta_n)")
    println("-"^60)
    
    for i in indices
        L, H, L_dom, d, rho, g = data[i, col("L_raft")], 0.05, 1.5, 0.03, 1000.0, 9.81
        EI = data[i, col("EI")]
        rho_R = data[i, col("rho_raft")]
        omega = data[i, col("omega")]
        
        q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
        Q_num = [complex(data[i, col("Q_w$(n)_re")], data[i, col("Q_w$(n)_im")]) for n in 0:3]
        
        e1, e2, e3 = audit_obsidian_laws(L, H, L_dom, d, rho, g, EI, rho_R, omega, q_num, Q_num)
        push!(results, (e1, e2, e3))
        @printf("%-10d | %-12.4f | %-12.4f | %-12.4f\n", i, e1, e2, e3)
    end
    
    means = [mean([r[j] for r in results]) for j in 1:3]
    println("-"^60)
    @printf("Mean Errors: | %-12.4f | %-12.4f | %-12.4f\n", means[1], means[2], means[3])
end

main()

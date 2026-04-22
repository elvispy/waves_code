using Surferbot
using DelimitedFiles
using LinearAlgebra
using Statistics
using Plots
using Printf

# Purpose: Visual deconstruction of the magnitude failure.
# Plot p(x) numerical vs p(x) analytical (spectral sum).

function get_apriori_pressure(L, H, L_dom, d, rho, omega, q_w_vec)
    # Sampling grid - raft only
    x = collect(range(-L/2, L/2, length=101))
    w_trapz = Surferbot.trapz_weights(x)
    
    # Reconstruction of eta
    raw_basis = Surferbot.Modal.build_raw_freefree_basis(x, L; num_modes=4)
    Phi = raw_basis.Phi
    eta_x = Phi * q_w_vec
    
    p_x = zeros(ComplexF64, length(x))
    m_max = 200 # Reduced for speed
    for m in 0:m_max
        lambda_m = m * pi / L_dom
        mode_m = cos.(lambda_m .* (x .+ L_dom/2))
        eta_m = dot(mode_m, eta_x .* w_trapz) / (L_dom / 2)
        if m == 0; eta_m /= 2; end
        
        denom = lambda_m * tanh(lambda_m * H + 1e-12)
        # Hypothesis: DtN of empty tank
        p_m = (rho * omega^2 * eta_m) / max(denom, 1e-8)
        p_x .+= p_m .* mode_m
    end
    return x, p_x
end

function main()
    csv_path = joinpath(@__DIR__, "..", "output", "single_alpha_zero_curve_details_coupled_refined.csv")
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    i = 32
    L, H, L_dom, d, rho = data[i, col("L_raft")], 0.05, 1.5, 0.03, 1000.0
    omega = data[i, col("omega")]
    q_num = [complex(data[i, col("q_w$(n)_re")], data[i, col("q_w$(n)_im")]) for n in 0:3]
    
    x_ap, p_ap = get_apriori_pressure(L, H, L_dom, d, rho, omega, q_num)
    
    params = Surferbot.FlexibleParams(
        L_raft = L, domain_depth = H, L_domain = L_dom,
        omega = omega, d = d, EI = data[i, col("EI")], 
        motor_position = data[i, col("motor_position")],
        motor_force = 1.0
    )
    result = Surferbot.flexible_solver(params)
    
    x_num = result.x[result.metadata.args.x_contact]
    p_num = result.p 
    
    # 3. Plotting
    p1 = plot(x_num, real.(p_num), label="Numerical Re(p)", color=:blue, lw=2)
    plot!(p1, x_ap, real.(p_ap), label="A-Priori Re(p)", color=:red, linestyle=:dash)
    title!(p1, "Pressure Profile Comparison (Real)")
    
    # New: Full domain elevation to see wave radiation
    p3 = plot(result.x, abs.(result.eta), label="|eta| (Numerical)", color=:black, lw=1)
    vline!(p3, [-L/2, L/2], label="Raft Edges", linestyle=:dot, color=:gray)
    title!(p3, "Full Domain Elevation (Numerical)")

    combined = plot(p1, p3, layout=(2, 1), size=(800, 800))
    save_path = joinpath(@__DIR__, "..", "output", "pressure_deconstruction.png")
    savefig(combined, save_path)
    println("Deconstruction plot saved to $save_path")
    
    @printf("Peak Numerical Pressure: %.3e\n", maximum(abs.(p_num)))
    @printf("Peak A-Priori Pressure:  %.3e\n", maximum(abs.(p_ap)))
    @printf("Ratio (AP / Num):       %.1f\n", maximum(abs.(p_ap)) / maximum(abs.(p_num)))
end


main()

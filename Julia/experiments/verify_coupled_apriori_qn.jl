using Surferbot
using JLD2
using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics

# Purpose: Test the "A Priori Qn Law" against direct numerical pressure data.
# 1. Hypothesize Law: Qn = G * (omega^2 * ma,n - d*rho*g) * qn
# 2. Extract Numerical Qn and qn from high-res refined CSV.
# 3. Calculate metrics of success.

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    n_points = length(xi_motor)
    Phi = zeros(Float64, n_points, max_mode + 1)
    
    Phi[:, 1] .= 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
        end
    end
    return (; Phi)
end

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    csv_path = joinpath(output_dir, "csv", "analyze_single_alpha_zero_curve.csv")
    
    if !isfile(csv_path)
        error("Missing dataset: $csv_path. Please run the tracker first.")
    end

    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(n) = findfirst(==(n), names)

    # 1. Physical Context from the first row
    params_row = 1
    L = data[params_row, col("L_raft")]
    rho = 1000.0
    omega = data[params_row, col("omega")]
    g = 9.81
    H = 0.05
    d = 0.03

    # Solve for open-water wavenumber k_wave
    k_res = real(Surferbot.dispersion_k(omega, g, 0.05, 0.0, 0.0, 1000.0))
    
    println("--- Verification Context ---")
    @printf("Frequency: %.2f Hz, Raft Length: %.3f m\n", omega/(2pi), L)
    @printf("Open-water kw: %.3f (wavelength: %.3f m)\n", k_res, 2pi/k_res)
    println("----------------------------\n")

    modes_to_test = 0:3
    n_modes = length(modes_to_test)
    
    # Construct Gram matrix G (since Qw = projection of force onto non-orthonormal Phi)
    x_raft = collect(range(-L/2, L/2, length=201))
    w_grid = Surferbot.trapz_weights(x_raft)
    raw_grid = raw_mode_shapes((;L_raft=L), x_raft ./ L; max_mode=maximum(modes_to_test))
    Phi = raw_grid.Phi
    G = Phi' * (Phi .* w_grid)

    q_all = zeros(ComplexF64, n_modes, size(data, 1))
    Q_all = zeros(ComplexF64, n_modes, size(data, 1))
    for n in modes_to_test
        q_all[n+1, :] .= complex.(data[:, col("q_w$(n)_re")], data[:, col("q_w$(n)_im")])
        Q_all[n+1, :] .= complex.(data[:, col("Q_w$(n)_re")], data[:, col("Q_w$(n)_im")])
    end

    # Hypotheses
    # ma_A: Wave-based added mass
    ma_A = (d * rho) / (k_res * tanh(k_res * H))
    # ma_C: Geometric added mass (piston)
    ma_C = rho * d * d 
    hydrostatic = d * rho * g
    
    println("Testing Law D (Gram-aware Geometric Coupling)...")
    
    for n in modes_to_test
        # We test both Ma hypotheses with the Gram matrix
        Q_pred_D_all = zeros(ComplexF64, n_modes, size(data, 1))
        coeff = (omega^2 * ma_C - hydrostatic)
        
        for i in 1:size(data, 1)
            # We use the full q_w vector to account for cross-coupling via G
            # Note: Using negative sign as Law C indicated negative phase correlation
            Q_pred_D_all[:, i] = G * (-coeff .* q_all[:, i]) 
        end
        
        Q_num = Q_all[n+1, :]
        Q_pred_D = Q_pred_D_all[n+1, :]
        
        err_D = [abs(Q_pred_D[i] - Q_num[i]) / max(abs(Q_num[i]), 1e-15) for i in 1:length(Q_num)]
        corr_D = real.(Q_pred_D .* conj.(Q_num)) ./ (abs.(Q_pred_D) .* abs.(Q_num) .+ 1e-15)
        
        @printf("n=%-4d | Mean Err: %-12.2f%% | Phase Corr: %-12.3f\n", 
                n, mean(err_D)*100, mean(corr_D))
    end
    
    println("\nVerification complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

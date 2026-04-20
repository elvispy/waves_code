using Surferbot
using Printf
using LinearAlgebra
using Statistics
using DelimitedFiles

# Establish the analytical a priori xM*(EI) law for the uncoupled second family.

function main()
    # 1. Physical constants from defaults
    L = 0.05
    rho_R = 0.052
    omega = 2 * pi * 80
    
    # 2. Analytical mode values at L/2 (from theory/notes)
    # W0(x) = 1/sqrt(L)
    W0_end = 1.0 / sqrt(L)
    
    # W2(x) = [sin(beta2*x) + sinh(beta2*x) + alpha2*(cos(beta2*x) + cosh(beta2*x))]
    # But we can just use the build_raw_freefree_basis to get the exact analytical values 
    # on a fine grid to avoid re-implementing the shape formula.
    xfine = collect(range(-L/2, L/2, length=1000))
    raw = build_raw_freefree_basis(xfine, L; num_modes=8, include_rigid=true)
    
    # Use the orthonormalized basis Psi directly to match the truth data coordinates.
    # Wn used in truth data (CSV) are actually Psin.
    idx0 = findfirst(==(0), raw.n)
    idx2 = findfirst(==(2), raw.n)
    idx4 = findfirst(==(4), raw.n)
    
    # Orthonormalize the basis on this fine grid to match Psin behavior
    w = trapz_weights(xfine)
    Psi, _ = weighted_mgs(raw.Phi, w)
    
    beta2 = raw.beta[idx2]
    beta4 = raw.beta[idx4]
    
    # Values at end (Left end x=-L/2)
    Psi0_end = Psi[1, idx0]
    Psi2_end = Psi[1, idx2]
    Psi4_end = Psi[1, idx4]
    
    println("--- BASIS CONSTANTS (Psi basis) ---")
    @printf("Psi0(L/2) = %.6f, Psi2(L/2) = %.6f, Psi4(L/2) = %.6f\n", Psi0_end, Psi2_end, Psi4_end)
    
    # 3. The a priori law (0+2+4): 
    # Psi0(xM)*Psi0(end)/D0 + Psi2(xM)*Psi2(end)/D2 + Psi4(xM)*Psi4(end)/D4 = 0
    
    D0(EI) = -rho_R * omega^2
    D2(EI) = EI * beta2^4 - rho_R * omega^2
    D4(EI) = EI * beta4^4 - rho_R * omega^2
    
    function predict_xM(EI)
        # Search for zero crossing on xfine grid
        vals = [Psi[i, idx0] * Psi0_end / D0(EI) + 
                Psi[i, idx2] * Psi2_end / D2(EI) +
                Psi[i, idx4] * Psi4_end / D4(EI) for i in 1:length(xfine)]
        
        # We want the FIRST POSITIVE root (the lowest branch)
        best_x = NaN
        for i in 1:(length(xfine)-1)
            if vals[i] * vals[i+1] < 0
                t = -vals[i] / (vals[i+1] - vals[i])
                x = xfine[i] + t * (xfine[i+1] - xfine[i])
                if x > 0 
                    return x
                end
            end
        end
        return best_x
    end
    
    # 4. Compare with truth data from the ported W-basis CSV
    output_dir = joinpath(@__DIR__, "..", "output")
    csv_path = joinpath(output_dir, "single_alpha_zero_curve_details_uncoupled_refined_w_basis.csv")
    
    if !isfile(csv_path)
        println("Skipping comparison: $csv_path not found.")
        return
    end
    
    data, header = readdlm(csv_path, ',', header=true)
    names = string.(vec(header))
    col(name) = findfirst(==(name), names)
    
    println("\n--- A PRIORI PREDICTION VS TRUTH (Using W-basis coefficients from CSV) ---")
    println("log10EI   xM/L (Truth)   xM/L (Pred)   Error (%)")
    
    for i in 1:10:size(data, 1)
        EI = data[i, col("EI")]
        truth_xM_over_L = data[i, col("xM_over_L")]
        
        # Calculate Pred using the analytical formula with W-basis constants
        pred_xM = predict_xM(EI)
        pred_xM_over_L = pred_xM / L
        
        err = abs(truth_xM_over_L - pred_xM_over_L) / truth_xM_over_L * 100
        
        # Cross-verify: show the qW coefficients from the CSV to prove they are meaningful
        qW0_abs = data[i, col("qW0_abs")]
        qW2_abs = data[i, col("qW2_abs")]
        
        @printf("%.3f      %.4f         %.4f         %.2f%%  (qW0=%.1e, qW2=%.1e)\n", log10(EI), truth_xM_over_L, pred_xM_over_L, err, qW0_abs, qW2_abs)
    end
    
    # 5. Conclusion
    println("\nConclusion: The uncoupled xM*(EI) law for the first S~0 branch is")
    println("W2(xM)/W0(xM) = -(W0(L/2)/W2(L/2)) * ( (EI*beta2^4 - rho_R*omega^2) / (-rho_R*omega^2) )")
end

main()

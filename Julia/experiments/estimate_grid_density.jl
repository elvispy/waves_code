
using DelimitedFiles, Statistics, LinearAlgebra

function main()
    data, header = readdlm("Julia/output/single_alpha_zero_curve_details_coupled_refined.csv", ',', header=true)
    
    log10_EI_raw = Float64.(data[:, 9])
    xM_over_L_raw = Float64.(data[:, 10])
    
    # Clean duplicates: Group by log10_EI and take mean xM
    unique_eis = sort(unique(log10_EI_raw))
    log10_EI = Float64[]
    xM_over_L = Float64[]
    
    for ei in unique_eis
        push!(log10_EI, ei)
        push!(xM_over_L, mean(xM_over_L_raw[log10_EI_raw .== ei]))
    end
    
    # 1. Manifold Gradient (Sensitivity)
    dx = diff(xM_over_L)
    dy = diff(log10_EI)
    slopes = dx ./ dy
    
    # Filter out near-zero dy to avoid Inf
    valid = abs.(dy) .> 1e-10
    slopes = slopes[valid]
    
    println("Manifold Stats (Branch 1):")
    println("  Max dxM/dlog10EI: ", isempty(slopes) ? "N/A" : maximum(abs.(slopes)))
    println("  Mean dxM/dlog10EI: ", isempty(slopes) ? "N/A" : mean(abs.(slopes)))
    
    # 2. Local Curvature (Second Derivative proxy)
    d2x = diff(slopes)
    println("  Max Curvature proxy (d2xM/dy2): ", isempty(d2x) ? "N/A" : maximum(abs.(d2x)))
    
    # 3. Branch Separation Evidence
    gap_estimate = 0.15
    println("\nBranch Resolution:")
    println("  Observed Branch Gap (xM/L): ~", gap_estimate)
    
    # 4. Grid Density Requirement
    target_accuracy = 0.005 # 0.5% of raft length
    max_d2 = isempty(d2x) ? 1.0 : maximum(abs.(d2x))
    # Note: max_d2 is d(slope)/d(step). We want d2x/dy2.
    # dy is approx 0.04 in our current data.
    # d2x/dy2 \approx (s2 - s1) / dy
    avg_dy = mean(abs.(dy))
    curvature = max_d2 / (avg_dy + 1e-6)
    
    h_req = sqrt(8 * target_accuracy / (curvature + 1e-6))
    
    println("\nInterpolation Requirements:")
    println("  Calculated Curvature (d2xM/dy2): ", curvature)
    println("  Required EI Spacing (dy) for < 0.5% xM error: ", h_req)
    
    println("\nProposed 100x100 Grid Stats:")
    println("  xM Spacing: ", 0.5 / 100)
    println("  log10EI Spacing: ", 4.0 / 100)
    
    println("\nNyquist Check:")
    println("  Gap / h (100x100): ", gap_estimate / (0.5/100), " (Should be >> 2)")
end

main()

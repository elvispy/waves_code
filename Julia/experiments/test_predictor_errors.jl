using Surferbot
using JLD2
using Printf
using LinearAlgebra

# Test script to calculate L2 relative norms of the predictors

include(joinpath(@__DIR__, "plot_uncoupled_beam_second_family_diagnostics.jl"))

function calculate_l2_relative_error(truth_xs, pred_xs)
    valid_idx = isfinite.(truth_xs) .& isfinite.(pred_xs)
    if sum(valid_idx) == 0
        return NaN
    end
    t = truth_xs[valid_idx]
    p = pred_xs[valid_idx]
    
    # L2 relative norm: ||truth - pred||_2 / ||truth||_2
    return norm(t .- p) / norm(t)
end

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    sweep_file = "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
    cache_file = "second_family_point_cache.jld2"
    
    println("Loading artifact...")
    artifact = load_sweep(joinpath(output_dir, sweep_file))
    
    # Get Truth (numerical alpha=0)
    # We can get this by running the same logic plot_uncoupled_beam_second_family_diagnostics uses to plot the contour.
    # Actually, the easiest way to get the exact truth roots is to run the branch extraction on the alpha field.
    fields = beam_fields(artifact)
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    xM_norm = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
    
    truth_roots = fill(NaN, length(EI_list))
    for i in eachindex(EI_list)
        vals = fields.alpha[:, i] # alpha is likely (motor_position x EI) based on the lengths (25 x 57)
        truth_roots[i] = first_positive_root(xM_norm, vals; branch_index=1)
    end
    
    # Get Predictions
    println("Running 'analytic' (numerical Psi-based) predictor...")
    analytic_0246 = analytic_second_family_branch(artifact, output_dir; cache_file=cache_file, mode_numbers=(0, 2, 4, 6), combination=:S)
    
    println("Running 'pure_analytic' (a priori W-based) predictor...")
    analytic_pure_0246 = analytic_second_family_branch_pure(artifact; branch_index=1, mode_numbers=(0, 2, 4, 6), combination=:S)
    
    # Calculate Errors
    err_analytic = calculate_l2_relative_error(truth_roots, analytic_0246.xM_norm)
    err_pure = calculate_l2_relative_error(truth_roots, analytic_pure_0246.xM_norm)
    
    println("\n--- ERROR METRICS ---")
    @printf("Analytic (Psi-based) L2 Rel Error:      %.2f%%\n", err_analytic * 100)
    @printf("Pure Analytic (W-based) L2 Rel Error:   %.2f%%\n", err_pure * 100)
    
    println("\nDetailed Comparison:")
    println("log10EI    Truth      Analytic   Pure")
    for i in eachindex(EI_list)
        if isfinite(truth_roots[i]) || isfinite(analytic_0246.xM_norm[i]) || isfinite(analytic_pure_0246.xM_norm[i])
            @printf("%8.3f   %8.4f   %8.4f   %8.4f\n", 
                log10(EI_list[i]), 
                truth_roots[i], 
                analytic_0246.xM_norm[i], 
                analytic_pure_0246.xM_norm[i])
        end
    end
end

main()


using Surferbot
using JLD2
using Statistics
using LinearAlgebra
using Printf
using Base.Threads

"""
    brute_force_sweep.jl

Perform a high-resolution brute-force sweep over xM/L and log10(EI) space.
Stores full FlexibleResult objects (minus phi) for every grid point to enable all 
post-processing (modal decomposition, a-posteriori analysis, etc.).
"""

function main()
    # 1. Setup Grid
    # 100 in xM, 50 per decade for 6 decades (-2 to -8) -> 300 in EI
    nx = 100
    nei = 300
    
    output_dir = joinpath(@__DIR__, "..", "output")
    mkpath(output_dir)
    output_path = joinpath(output_dir, "brute_force_30k_full.jld2")
    
    # Define ranges
    L_raft = 0.05
    mp_norm_range = range(0.0, 0.49; length=nx)
    motor_position_list = collect(mp_norm_range .* L_raft)
    
    # log10(EI) range from -2 (stiff) to -8 (very soft)
    logEI_range = range(-2.0, -8.0; length=nei)
    EI_list = 10.0 .^ logEI_range
    
    println("Initializing $nx x $nei brute-force sweep...")
    println("  Output: $output_path")
    
    # 2. Initialize or Resume Cache
    results = Matrix{Any}(nothing, nx, nei)
    
    if isfile(output_path)
        println("  Found existing file, loading cache...")
        try
            loaded = load(output_path)
            if haskey(loaded, "results")
                results_loaded = loaded["results"]
                if size(results_loaded) == (nx, nei)
                    results .= results_loaded
                    println("  Successfully loaded $(count(!isnothing, results)) / $(nx*nei) points.")
                else
                    println("  Cache size mismatch, starting fresh.")
                end
            end
        catch e
            println("  Error loading cache: $e. Starting fresh.")
        end
    end
    
    # 3. Base Parameters
    # Using the standard coupled defaults
    base_params = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    
    # 4. Threaded Loop
    println("  Running simulations on $(nthreads()) threads...")
    
    total_sims = nx * nei
    done_count = Atomic{Int}(count(!isnothing, results))
    
    @threads for idx in 1:total_sims
        im = ((idx - 1) % nx) + 1
        ie = floor(Int, (idx - 1) / nx) + 1
        
        # Skip if already in cache
        if !isnothing(results[im, ie])
            continue
        end
        
        # Setup specific case
        params = Surferbot.Sweep.apply_parameter_overrides(base_params, (
            motor_position = motor_position_list[im],
            EI = EI_list[ie]
        ))
        
        # Solve and strip potential fields to save space
        try
            res = flexible_solver(params)
            # Create a lightweight version for storage
            results[im, ie] = (
                U = res.U,
                power = res.power,
                thrust = res.thrust,
                eta = res.eta,
                pressure = res.pressure,
                max_curvature = res.max_curvature,
                wave_steepness = res.wave_steepness,
                metadata = (args = res.metadata.args, params = res.metadata.params)
            )
        catch e
            @warn "Failed at im=$im, ie=$ie: $e"
        end
        
        # Periodically report progress and save
        new_count = atomic_add!(done_count, 1) + 1
        if new_count % 500 == 0 || new_count == total_sims
            @printf("  Progress: %d / %d (%.1f%%)\n", new_count, total_sims, new_count/total_sims*100)
            # Safe intermediate save
            if threadid() == 1
                tmp_path = output_path * ".tmp"
                jldsave(tmp_path; 
                    results=results, 
                    mp_norm_list=collect(mp_norm_range), 
                    logEI_list=collect(logEI_range),
                    motor_position_list=motor_position_list,
                    EI_list=EI_list
                )
                mv(tmp_path, output_path, force=true)
            end
        end
    end
    
    # 5. Final Save
    jldsave(output_path; 
        results=results, 
        mp_norm_list=collect(mp_norm_range), 
        logEI_list=collect(logEI_range),
        motor_position_list=motor_position_list,
        EI_list=EI_list
    )
    
    println("\nBrute-force sweep complete.")
    println("Saved to $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

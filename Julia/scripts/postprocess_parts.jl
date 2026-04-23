using Surferbot
using JLD2
using CSV
using DataFrames
using Printf
using Base.Threads

"""
    postprocess_parts.jl

Fixes the FieldError by reconstructing the necessary metadata on-the-fly.
Processes a directory of partial `.jld2` sweep artifacts, performs modal
decomposition, and collates results into a single, self-contained master CSV.
"""

function reconstruct_params(base_params, part_data, motor_pos)
    # The part file contains the EI for the entire slice
    ei = part_data["EI"]
    
    overrides = (
        motor_position = motor_pos,
        EI = ei,
    )
    
    # Use the same function as the sweep scripts to ensure consistency
    return Surferbot.Sweep.apply_parameter_overrides(base_params, overrides)
end

function process_part(part_path::AbstractString, base_params, motor_position_list, num_modes::Int)
    try
        part_data = load(part_path)
        if !haskey(part_data, "results")
            @warn "Part file $part_path is missing 'results' key."
            return []
        end
        
        results_chunk = part_data["results"]
        rows = []

        for (im, res) in enumerate(results_chunk)
            if isnothing(res) || !haskey(res, :metadata)
                continue
            end
            
            # 1. Reconstruct the full parameters for this grid point
            motor_pos = motor_position_list[im]
            params = reconstruct_params(base_params, part_data, motor_pos)
            
            # Reconstruct other needed inputs for decomposition
            x_raft = Surferbot.raft_nodes(params)
            loads = Surferbot.build_loads(x_raft, params)

            # 2. Perform the modal decomposition
            modal = Surferbot.Modal.decompose_raft_freefree_modes(
                x_raft, 
                res.eta, 
                res.pressure, 
                loads, 
                params; 
                num_modes=num_modes, 
                verbose=false
            )
            
            # 3. Build a row for the DataFrame, including key physical params
            row_data = (
                log10_EI = log10(params.EI),
                xM_over_L = params.motor_position / params.L_raft,
                L_raft = params.L_raft,
                omega = params.omega,
                d = params.d,
                rho_raft = params.rho_raft,
            )
            
            for n in 0:(num_modes-1)
                row_data = merge(row_data, NamedTuple{(
                    Symbol("q_w$(n)_re"), Symbol("q_w$(n)_im"),
                    Symbol("Q_w$(n)_re"), Symbol("Q_w$(n)_im"),
                    Symbol("F_w$(n)_re"), Symbol("F_w$(n)_im"),
                )}((
                    modal.q_w[n+1].re, modal.q_w[n+1].im,
                    modal.Q_w[n+1].re, modal.Q_w[n+1].im,
                    modal.F_w[n+1].re, modal.F_w[n+1].im
                )))
            end
            push!(rows, row_data)
        end
        return rows
    catch e
        @error "Failed to process part $part_path: $e"
        return []
    end
end

function main(parts_dir::AbstractString, output_csv::AbstractString; num_modes::Int=8)
    if !isdir(parts_dir)
        error("Parts directory not found: $parts_dir")
    end
    
    part_files = filter(f -> endswith(f, ".jld2"), readdir(parts_dir, join=true))
    num_parts = length(part_files)
    println("Found $num_parts parts in $parts_dir")
    
    if isfile(output_csv)
        @warn "Output file $output_csv already exists. It will be overwritten."
    end
    
    # Load base parameters to be used for reconstruction
    base_params = Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params
    L_raft = base_params.L_raft
    nx = 100 # From the brute_force_sweep script
    motor_position_list = collect(range(0.0, 0.49; length=nx) .* L_raft)

    # Process all parts in parallel
    all_rows = Vector{Any}(undef, num_parts)
    @threads for i in 1:num_parts
        @printf("Processing part %d/%d: %s
", i, num_parts, basename(part_files[i]))
        all_rows[i] = process_part(part_files[i], base_params, motor_position_list, num_modes)
    end
    
    # Flatten the array of arrays and write to CSV
    println("Collating results...")
    flat_rows = vcat(filter(!isempty, all_rows)...)
    
    if isempty(flat_rows)
        @warn "No data was processed. Output CSV will be empty."
        return
    end
    
    df = DataFrame(flat_rows)
    
    println("Writing to $output_csv...")
    CSV.write(output_csv, df)
    
    println("Done. Wrote $(size(df, 1)) rows to the master CSV.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    parts_dir = length(ARGS) >= 1 ? ARGS[1] : "Julia/output/parts"
    output_csv = length(ARGS) >= 2 ? ARGS[2] : "Julia/output/brute_force_modal_integrals_full_grid.csv"
    main(parts_dir, output_csv)
end

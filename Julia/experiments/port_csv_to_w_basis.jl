using Surferbot
using JLD2
using DelimitedFiles
using Printf
using LinearAlgebra

# Offline port of CSV databases from Psi_n basis to analytical W_n basis.
# This version adds qW, QW, and FW columns for each mode.

function format_complex(z)
    return "$(real(z)),$(imag(z)),$(abs(z)),$(rad2deg(angle(z)))"
end

function transform_row(row, names, T, n_modes)
    new_row = copy(row)
    
    # Extract coefficients
    q_psi = [parse(Float64, row["q$(n)_re"]) + im * parse(Float64, row["q$(n)_im"]) for n in 0:(n_modes-1)]
    Q_psi = [parse(Float64, row["Q$(n)_re"]) + im * parse(Float64, row["Q$(n)_im"]) for n in 0:(n_modes-1)]
    F_psi = [parse(Float64, row["F$(n)_re"]) + im * parse(Float64, row["F$(n)_im"]) for n in 0:(n_modes-1)]
    
    # Apply transformation T (Psi -> W)
    q_w = T * q_psi
    Q_w = T * Q_psi
    F_w = T * F_psi
    
    # We will build a new dict to control order if needed, but for now we just add fields
    for n in 0:(n_modes-1)
        new_row["qW$(n)_re"] = string(real(q_w[n+1]))
        new_row["qW$(n)_im"] = string(imag(q_w[n+1]))
        new_row["qW$(n)_abs"] = string(abs(q_w[n+1]))
        new_row["qW$(n)_phase_deg"] = string(rad2deg(angle(q_w[n+1])))
        
        new_row["QW$(n)_re"] = string(real(Q_w[n+1]))
        new_row["QW$(n)_im"] = string(imag(Q_w[n+1]))
        new_row["QW$(n)_abs"] = string(abs(Q_w[n+1]))
        new_row["QW$(n)_phase_deg"] = string(rad2deg(angle(Q_w[n+1])))
        
        new_row["FW$(n)_re"] = string(real(F_w[n+1]))
        new_row["FW$(n)_im"] = string(imag(F_w[n+1]))
        new_row["FW$(n)_abs"] = string(abs(F_w[n+1]))
        new_row["FW$(n)_phase_deg"] = string(rad2deg(angle(F_w[n+1])))
    end
    
    return new_row
end

function port_file(input_path, T, n_modes)
    println("Processing $input_path...")
    data, header = readdlm(input_path, ',', header=true)
    orig_names = string.(vec(header))
    
    # Build new header
    new_names = String[]
    # Find the insertion point: after sa_ratio_beam
    idx_sa = findfirst(==("sa_ratio_beam"), orig_names)
    append!(new_names, orig_names[1:idx_sa])
    
    for j in 0:(n_modes-1)
        # We'll follow the pattern of the new write_curve_csv
        append!(new_names, ["q$(j)_re", "q$(j)_im", "q$(j)_abs", "q$(j)_phase_deg"])
        append!(new_names, ["qW$(j)_re", "qW$(j)_im", "qW$(j)_abs", "qW$(j)_phase_deg"])
        append!(new_names, ["Q$(j)_re", "Q$(j)_im", "Q$(j)_abs", "Q$(j)_phase_deg"])
        append!(new_names, ["QW$(j)_re", "QW$(j)_im", "QW$(j)_abs", "QW$(j)_phase_deg"])
        append!(new_names, ["F$(j)_re", "F$(j)_im", "F$(j)_abs", "F$(j)_phase_deg"])
        append!(new_names, ["FW$(j)_re", "FW$(j)_im", "FW$(j)_abs", "FW$(j)_phase_deg"])
        append!(new_names, ["residual$(j)_re", "residual$(j)_im", "residual$(j)_abs", "residual$(j)_phase_deg"])
        append!(new_names, ["energy_frac$(j)", "mode_index$(j)", "mode_type$(j)"])
    end
    
    output_path = replace(input_path, ".csv" => "_w_basis.csv")
    
    open(output_path, "w") do io
        println(io, join(new_names, ","))
        for i in axes(data, 1)
            row_dict = Dict(orig_names[j] => string(data[i, j]) for j in axes(data, 2))
            transformed = transform_row(row_dict, orig_names, T, n_modes)
            println(io, join([transformed[n] for n in new_names], ","))
        end
    end
    println("Saved $output_path")
end

function main()
    output_dir = "Julia/output"
    cache_file = joinpath(output_dir, "second_family_point_cache.jld2")
    
    if !isfile(cache_file)
        println("Cache file not found.")
        return
    end
    
    cache = load(cache_file)
    payload = cache[first(keys(cache))]
    w = trapz_weights(payload.x_raft)
    raw = build_raw_freefree_basis(payload.x_raft, payload.L_raft; num_modes=length(payload.n), include_rigid=true)
    G = raw.Phi' * (raw.Phi .* w)
    B = raw.Phi' * (payload.Psi .* w)
    T = G \ B
    
    files_to_port = [
        "single_alpha_zero_curve_details_refined.csv",
        "single_alpha_zero_curve_details_uncoupled_refined.csv",
        "single_alpha_zero_curve_details.csv"
    ]
    
    for f in files_to_port
        input_path = joinpath(output_dir, f)
        if isfile(input_path)
            port_file(input_path, T, length(payload.n))
        end
    end
end

main()

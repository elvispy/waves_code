using Surferbot
using JLD2
using DelimitedFiles
using Printf
using LinearAlgebra

# Final Port: Move to W-basis coefficients EXCLUSIVELY and clean up Psi columns.

function transform_row(row, orig_names, T, n_modes)
    # Extract Psi coefficients
    q_psi = [parse(Float64, row["q$(n)_re"]) + im * parse(Float64, row["q$(n)_im"]) for n in 0:(n_modes-1)]
    Q_psi = [parse(Float64, row["Q$(n)_re"]) + im * parse(Float64, row["Q$(n)_im"]) for n in 0:(n_modes-1)]
    F_psi = [parse(Float64, row["F$(n)_re"]) + im * parse(Float64, row["F$(n)_im"]) for n in 0:(n_modes-1)]
    
    # Map to W
    q_w = T * q_psi
    Q_w = T * Q_psi
    F_w = T * F_psi
    
    new_row = copy(row)
    for n in 0:(n_modes-1)
        # Store W coefficients back into the standard q, Q, F columns
        new_row["q$(n)_re"] = string(real(q_w[n+1]))
        new_row["q$(n)_im"] = string(imag(q_w[n+1]))
        new_row["q$(n)_abs"] = string(abs(q_w[n+1]))
        new_row["q$(n)_phase_deg"] = string(rad2deg(angle(q_w[n+1])))
        
        new_row["Q$(n)_re"] = string(real(Q_w[n+1]))
        new_row["Q$(n)_im"] = string(imag(Q_w[n+1]))
        new_row["Q$(n)_abs"] = string(abs(Q_w[n+1]))
        new_row["Q$(n)_phase_deg"] = string(rad2deg(angle(Q_w[n+1])))
        
        new_row["F$(n)_re"] = string(real(F_w[n+1]))
        new_row["F$(n)_im"] = string(imag(F_w[n+1]))
        new_row["F$(n)_abs"] = string(abs(F_w[n+1]))
        new_row["F$(n)_phase_deg"] = string(rad2deg(angle(F_w[n+1])))
    end
    
    # Better: transform whole residual vector once
    R_psi_vec = [haskey(row, "residual$(n)_re") ? parse(Float64, row["residual$(n)_re"]) + im * parse(Float64, row["residual$(n)_im"]) : 0.0im for n in 0:(n_modes-1)]
    R_w_vec = T * R_psi_vec
    for n in 0:(n_modes-1)
        if haskey(row, "residual$(n)_re")
            new_row["residual$(n)_re"] = string(real(R_w_vec[n+1]))
            new_row["residual$(n)_im"] = string(imag(R_w_vec[n+1]))
            new_row["residual$(n)_abs"] = string(abs(R_w_vec[n+1]))
            new_row["residual$(n)_phase_deg"] = string(rad2deg(angle(R_w_vec[n+1])))
        end
    end
    
    return new_row
end

function port_file_in_place(path, T, n_modes)
    println("Porting $path...")
    data, header = readdlm(path, ',', header=true)
    names = string.(vec(header))
    
    open(path, "w") do io
        println(io, join(names, ","))
        for i in axes(data, 1)
            row_dict = Dict(names[j] => string(data[i, j]) for j in axes(data, 2))
            transformed = transform_row(row_dict, names, T, n_modes)
            println(io, join([transformed[n] for n in names], ","))
        end
    end
    println("Done.")
end

function main()
    output_dir = "Julia/output"
    cache_file = joinpath(output_dir, "second_family_point_cache.jld2")
    cache = load(cache_file)
    payload = cache[first(keys(cache))]
    w = trapz_weights(payload.x_raft)
    raw = build_raw_freefree_basis(payload.x_raft, payload.L_raft; num_modes=length(payload.n), include_rigid=true)
    T = (raw.Phi' * (raw.Phi .* w)) \ (raw.Phi' * (payload.Psi .* w))
    
    # All CSVs with modal data
    files = [
        "single_alpha_zero_curve_details_refined.csv",
        "single_alpha_zero_curve_details_smoke.csv",
        "single_alpha_zero_curve_details_tracking_smoke.csv",
        "single_alpha_zero_curve_details_uncoupled_refined.csv",
        "single_alpha_zero_curve_details.csv",
        "single_alpha_zero_curve_test.csv"
    ]
    
    for f in files
        path = joinpath(output_dir, f)
        if isfile(path)
            port_file_in_place(path, T, length(payload.n))
        end
    end
end
main()

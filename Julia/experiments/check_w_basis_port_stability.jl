using Surferbot
using JLD2
using Statistics
using Printf
using LinearAlgebra

include("test_high_EI_8mode_branch_recovery.jl")

function build_port_payload(cache_path::AbstractString)
    cache = load(cache_path)
    isempty(cache) && error("Cache file is empty: $cache_path")
    key = first(sort!(collect(keys(cache))))
    payload = cache[key]
    raw = build_raw_freefree_basis(payload.x_raft, payload.L_raft; num_modes=length(payload.n), include_rigid=true)
    T = psi_to_w_transform(raw.Phi, payload.Psi, raw.w)
    return (
        n = payload.n,
        beta = payload.beta,
        x_raft = raw.x_raft,
        x_norm = raw.x_raft ./ payload.L_raft,
        rho_raft = payload.rho_raft,
        omega = payload.omega,
        Psi = payload.Psi,
        Phi = raw.Phi,
        w = raw.w,
        T = T,
        phi_cond = raw.gram_cond,
    )
end

function branch_predict(rows, header, basis, basis_matrix; branch_index::Int=1, mode_numbers=(0, 2, 4, 6))
    col(name) = findfirst(==(name), header)
    idxEI = col("EI")
    idxX = col("xM_over_L")

    n = basis.n
    even_idx = [findfirst(==(m), n) for m in mode_numbers if findfirst(==(m), n) !== nothing]
    W_end = basis_matrix[1, :]
    Dfun(EI, β) = EI * β^4 - basis.rho_raft * basis.omega^2

    preds = Float64[]
    truth = Float64[]
    EIs = Float64[]
    for r in rows
        EI = parse(Float64, r[idxEI])
        x_true = parse(Float64, r[idxX])
        vals = Float64[]
        for row in axes(basis_matrix, 1)
            s = 0.0
            for j in even_idx
                s += basis_matrix[row, j] * W_end[j] / Dfun(EI, basis.beta[j])
            end
            push!(vals, s)
        end
        x_pred = first_positive_root(basis.x_norm, vals; branch_index=branch_index)
        if isfinite(x_pred)
            push!(preds, x_pred)
            push!(truth, x_true)
            push!(EIs, EI)
        end
    end
    return (; preds, truth, EIs)
end

function relative_l2(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    return norm(a .- b) / max(norm(a), 1e-12)
end

function align_predictions(a, b)
    amap = Dict(EI => (pred=a.preds[i], truth=a.truth[i]) for (i, EI) in enumerate(a.EIs))
    bmap = Dict(EI => b.preds[i] for (i, EI) in enumerate(b.EIs))
    common = sort!(collect(intersect(keys(amap), keys(bmap))))
    ap = [amap[EI].pred for EI in common]
    bp = [bmap[EI] for EI in common]
    tr = [amap[EI].truth for EI in common]
    return common, ap, bp, tr
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    cache_file::AbstractString="second_family_point_cache.jld2",
    logEI_min::Float64=-3.65,
    logEI_max::Float64=-3.35,
    branch_index::Int=1,
    threshold::Float64=1e-1,
)
    header, rows = selected_rows(joinpath(output_dir, branch_csv); logEI_min=logEI_min, logEI_max=logEI_max)
    basis = build_port_payload(joinpath(output_dir, cache_file))
    psi_pred = branch_predict(rows, header, basis, basis.Psi; branch_index=branch_index)
    w_pred = branch_predict(rows, header, basis, basis.Phi; branch_index=branch_index)
    common_EIs, psi_vals, w_vals, truth = align_predictions(psi_pred, w_pred)
    l2rel = relative_l2(psi_vals, w_vals)

    println("W-BASIS PORT STABILITY CHECK")
    @printf("phi_gram_cond=%.6e\n", basis.phi_cond)
    @printf("psi_points=%d\n", length(psi_pred.preds))
    @printf("w_points=%d\n", length(w_pred.preds))
    @printf("common_points=%d\n", length(common_EIs))
    @printf("relative_l2_distance=%.6e\n", l2rel)
    @printf("threshold=%.6e\n", threshold)
    println(l2rel <= threshold ? "status=PASS" : "status=FAIL")
    for i in eachindex(common_EIs)
        @printf(
            "EI=%.6e data=%.4f psi=%.4f w=%.4f\n",
            common_EIs[i],
            truth[i],
            psi_vals[i],
            w_vals[i],
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

using Surferbot
using Printf

# Purpose: rerun one branch point and compare modal coefficients obtained from
# the projected displacement against the direct force-balance formula from the
# uncoupled beam equation.

function nearest_row(path::AbstractString; target_log10_EI::Float64=-3.389, max_abs_alpha::Float64=0.01)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
    col(name) = findfirst(==(name), header)
    idxEI = col("EI")
    idxX = col("xM_over_L")
    idxA = col("alpha_beam")

    best = nothing
    best_score = Inf
    for r in rows
        logEI = log10(parse(Float64, r[idxEI]))
        alpha = abs(parse(Float64, r[idxA]))
        alpha <= max_abs_alpha || continue
        score = abs(logEI - target_log10_EI)
        if score < best_score
            best = r
            best_score = score
        end
    end
    best === nothing && error("No row found near log10(EI)=$target_log10_EI with |alpha_beam| <= $max_abs_alpha.")
    return header, best
end

function modal_force_balance(result; num_modes::Int=8)
    args = result.metadata.args
    modal = decompose_raft_freefree_modes(result; num_modes=num_modes, verbose=false)
    q = modal.q
    F = modal.F
    Q = modal.Q
    n = modal.n
    beta = modal.beta
    D = args.EI .* (beta .^ 4) .- args.rho_raft .* args.omega^2
    q_pred = (Q .- F) ./ D
    return (; modal, q_pred, D)
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString=joinpath("csv", "analyze_single_alpha_zero_curve.csv"),
    target_log10_EI::Float64=-3.389,
    max_abs_alpha::Float64=0.01,
    num_modes::Int=8,
)
    header, row = nearest_row(joinpath(output_dir, branch_csv); target_log10_EI=target_log10_EI, max_abs_alpha=max_abs_alpha)
    col(name) = findfirst(==(name), header)
    EI = parse(Float64, row[col("EI")])
    xM_over_L = parse(Float64, row[col("xM_over_L")])
    alpha = parse(Float64, row[col("alpha_beam")])

    artifact = load_sweep(joinpath(output_dir, "jld2", "sweep_motor_position_EI_uncoupled_from_matlab.jld2"))
    params = apply_parameter_overrides(
        artifact.base_params,
        (EI=EI, motor_position=xM_over_L * artifact.base_params.L_raft),
    )
    result = flexible_solver(params)
    balance = modal_force_balance(result; num_modes=num_modes)

    println("Case: EI=$(EI), log10(EI)=$(log10(EI)), x_M/L=$(xM_over_L), alpha_beam=$(alpha)")
    println("mode  n   q_from_projection                  q_from_force_balance              relerr")
    for j in eachindex(balance.modal.q)
        q = balance.modal.q[j]
        q_pred = balance.q_pred[j]
        relerr = abs(q - q_pred) / max(abs(q), 1e-12)
        @printf(
            "%4d %2d  % .6e%+.6ei   % .6e%+.6ei   %.3e\n",
            j,
            balance.modal.n[j],
            real(q),
            imag(q),
            real(q_pred),
            imag(q_pred),
            relerr,
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

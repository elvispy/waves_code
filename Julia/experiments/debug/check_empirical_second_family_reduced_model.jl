using Surferbot
using Printf

"""
check_empirical_second_family_reduced_model.jl

Validates the delta-load force projection and the two-term reduced branch 
equation against a full numerical solve at a clean alpha=0 branch point.
"""

# Purpose: on one clean branch point, check the delta-load force projection,
# the two-term reduced branch equation, and the S_02 closure against the full
# numerical solution.

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

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    target_log10_EI::Float64=-3.389,
    max_abs_alpha::Float64=0.01,
    num_modes::Int=8,
)
    header, row = nearest_row(joinpath(output_dir, branch_csv); target_log10_EI=target_log10_EI, max_abs_alpha=max_abs_alpha)
    col(name) = findfirst(==(name), header)
    EI = parse(Float64, row[col("EI")])
    xM_over_L = parse(Float64, row[col("xM_over_L")])
    alpha = parse(Float64, row[col("alpha_beam")])

    artifact = load_sweep(joinpath(output_dir, sweep_file))
    params = apply_parameter_overrides(
        artifact.base_params,
        (EI=EI, motor_position=xM_over_L * artifact.base_params.L_raft),
    )
    result = flexible_solver(params)
    modal = decompose_raft_freefree_modes(result; num_modes=num_modes, verbose=false)
    args = result.metadata.args

    println("Case: EI=$(EI), log10(EI)=$(log10(EI)), x_M/L=$(xM_over_L), alpha_beam=$(alpha)")

    # Actual modal force balance pieces.
    n = modal.n
    q = modal.q
    F = modal.F
    Q = modal.Q
    beta = modal.beta
    D = args.EI .* (beta .^ 4) .- args.rho_raft .* args.omega^2

    # Delta-load prediction for F_n: proportional to mode value at motor point.
    x_raft = modal.x_raft
    j_motor = findmin(abs.(x_raft .- args.motor_position))[2]
    Psi_motor = modal.Psi[j_motor, :]
    scale_F0 = F[1] / Psi_motor[1]
    F_delta = scale_F0 .* Psi_motor

    println("\nFn check against delta-load projection")
    println("mode  n   F_actual                          F_delta                           relerr")
    for j in eachindex(F)
        relerr = abs(F[j] - F_delta[j]) / max(abs(F[j]), 1e-12)
        @printf(
            "%4d %2d  % .6e%+.6ei   % .6e%+.6ei   %.3e\n",
            j,
            n[j],
            real(F[j]),
            imag(F[j]),
            real(F_delta[j]),
            imag(F_delta[j]),
            relerr,
        )
    end

    # Two-term q-ratio law.
    i0 = findfirst(==(0), n)
    i2 = findfirst(==(2), n)
    lhs_ratio = q[i2] / q[i0]
    rhs_ratio = (Psi_motor[i2] / Psi_motor[i0]) * (D[i0] / D[i2])

    println("\nq2/q0 ratio law")
    @printf("lhs q2/q0 = % .6e%+.6ei\n", real(lhs_ratio), imag(lhs_ratio))
    @printf("rhs       = % .6e%+.6ei\n", real(rhs_ratio), imag(rhs_ratio))
    @printf("relative error = %.3e\n", abs(lhs_ratio - rhs_ratio) / max(abs(lhs_ratio), 1e-12))

    println("\nModal balance components for mode 0 and 2")
    @printf("Mode 0: q=% .6e%+.6ei, F=% .6e%+.6ei, D=% .6e, F/D=% .6e\n", real(q[i0]), imag(q[i0]), real(F[i0]), imag(F[i0]), D[i0], real(-F[i0]/D[i0]))
    @printf("Mode 2: q=% .6e%+.6ei, F=% .6e%+.6ei, D=% .6e, F/D=% .6e\n", real(q[i2]), imag(q[i2]), real(F[i2]), imag(F[i2]), D[i2], real(-F[i2]/D[i2]))
    @printf("Mode 0 mismatch: q - (-F/D) = % .6e%+.6ei\n", real(q[i0] + F[i0]/D[i0]), imag(q[i0] + F[i0]/D[i0]))
    @printf("Mode 2 mismatch: q - (-F/D) = % .6e%+.6ei\n", real(q[i2] + F[i2]/D[i2]), imag(q[i2] + F[i2]/D[i2]))
    @printf("Mode 0 Q (hydro): % .6e%+.6ei\n", real(Q[i0]), imag(Q[i0]))
    @printf("Mode 2 Q (hydro): % .6e%+.6ei\n", real(Q[i2]), imag(Q[i2]))

    # Two-term implicit branch equation using actual motor-point basis values.
    W0_end = modal.Psi[1, i0]
    W2_end = modal.Psi[1, i2]
    branch_lhs = -(scale_F0 * Psi_motor[i0] * W0_end / D[i0]) - (scale_F0 * Psi_motor[i2] * W2_end / D[i2])
    println("\nTwo-term implicit branch equation")
    @printf("LHS = -F0*W0(xM)*W0(end)/D0 - F0*W2(xM)*W2(end)/D2 = % .6e%+.6ei\n", real(branch_lhs), imag(branch_lhs))

    # S_02 versus full S from modal reconstruction at beam ends.
    even_idx = findall(iseven, n)
    S_full = zero(ComplexF64)
    S_02 = zero(ComplexF64)
    for j in even_idx
        contrib = q[j] * modal.Psi[1, j]
        S_full += contrib
        if n[j] == 0 || n[j] == 2
            S_02 += contrib
        end
    end
    S_rest = S_full - S_02

    println("\nS_02 closure")
    @printf("S_full = % .6e%+.6ei  |S_full|=%.3e\n", real(S_full), imag(S_full), abs(S_full))
    @printf("S_02   = % .6e%+.6ei  |S_02|  =%.3e\n", real(S_02), imag(S_02), abs(S_02))
    @printf("S_rest = % .6e%+.6ei  |S_rest|=%.3e\n", real(S_rest), imag(S_rest), abs(S_rest))
    @printf("|S_02-S_full|/|S_full| = %.3e\n", abs(S_02 - S_full) / max(abs(S_full), 1e-12))
    @printf("|S_rest|/|S_02|        = %.3e\n", abs(S_rest) / max(abs(S_02), 1e-12))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

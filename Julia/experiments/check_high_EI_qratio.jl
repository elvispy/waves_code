using Surferbot
using Statistics
using Printf

"""
check_high_EI_qratio.jl

Verifies the high-stiffness asymptotic relation q2/q0 ≈ (W2/W0) * (D0/D2) 
using uncoupled sweep data. Reports the empirical ratio vs. theoretical 
prediction.
"""

# Purpose: check the high-EI ratio relation q2/q0 ≈ (W2/W0) * (D0/D2)
# on the existing uncoupled first nontrivial branch CSV, with no new solves.

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    branch_csv::AbstractString="single_alpha_zero_curve_details_uncoupled_refined.csv",
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    logEI_min::Float64=-3.65,
    logEI_max::Float64=-3.35,
)
    artifact = load_sweep(joinpath(output_dir, sweep_file))
    params = artifact.base_params
    args = derive_params(params)

    contact = collect(Bool.(args.x_contact))
    x_all = args.x
    x_raft = x_all[contact]
    w = trapz_weights(x_raft)
    xi = x_raft .+ params.L_raft / 2

    phi_raw = zeros(Float64, length(x_raft), 4)
    phi_raw[:, 1] .= 1.0
    phi_raw[:, 2] .= xi .- params.L_raft / 2
    betaL_el = freefree_betaL_roots(2)
    phi_raw[:, 3] .= freefree_mode_shape(xi, params.L_raft, betaL_el[1])
    phi_raw[:, 4] .= freefree_mode_shape(xi, params.L_raft, betaL_el[2])
    Psi, _ = weighted_mgs(phi_raw, w)

    W0 = Psi[:, 1]
    W2 = Psi[:, 3]
    beta2 = betaL_el[1] / params.L_raft
    D0(EI) = -params.rho_raft * params.omega^2
    D2(EI) = EI * beta2^4 - params.rho_raft * params.omega^2

    lines = readlines(joinpath(output_dir, branch_csv))
    header = split(lines[1], ",")
    rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
    col(name) = findfirst(==(name), header)
    idxEI = col("EI")
    idxX = col("xM_over_L")
    idxq0 = col("q0_re")
    idxq2 = col("q2_re")
    idxa = col("alpha_beam")

    selected = [r for r in rows if logEI_min <= log10(parse(Float64, r[idxEI])) <= logEI_max]
    println("selected_n=$(length(selected))")

    nearest_idx(xMdim) = findmin(abs.(x_raft .- xMdim))[2]

    relerr = Float64[]
    mag_ratio = Float64[]
    phase_diff = Float64[]

    for r in selected
        EI = parse(Float64, r[idxEI])
        xM_over_L = parse(Float64, r[idxX])
        xMdim = xM_over_L * params.L_raft
        j = nearest_idx(xMdim)

        q0 = parse(Float64, r[idxq0]) + im * parse(Float64, r[idxq0 + 1])
        q2 = parse(Float64, r[idxq2]) + im * parse(Float64, r[idxq2 + 1])
        lhs = q2 / q0
        rhs = (W2[j] / W0[j]) * (D0(EI) / D2(EI))

        push!(relerr, abs(lhs - rhs) / max(abs(lhs), 1e-12))
        push!(mag_ratio, abs(lhs) / max(abs(rhs), 1e-12))

        Δ = angle(lhs) - angle(rhs)
        while Δ > pi
            Δ -= 2pi
        end
        while Δ < -pi
            Δ += 2pi
        end
        push!(phase_diff, Δ)

        @printf(
            "EI=%.6e log10EI=%.3f xM/L=%.4f alpha=%.3e lhs=%.5f%+.5fi rhs=%.5f%+.5fi relerr=%.3e\n",
            EI,
            log10(EI),
            xM_over_L,
            parse(Float64, r[idxa]),
            real(lhs),
            imag(lhs),
            real(rhs),
            imag(rhs),
            relerr[end],
        )
    end

    println("summary median_relerr=$(median(relerr)) max_relerr=$(maximum(relerr))")
    println("summary median_mag_ratio=$(median(mag_ratio)) min_mag_ratio=$(minimum(mag_ratio)) max_mag_ratio=$(maximum(mag_ratio))")
    println("summary median_phase_diff_deg=$(rad2deg(median(phase_diff))) max_abs_phase_diff_deg=$(rad2deg(maximum(abs.(phase_diff))))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

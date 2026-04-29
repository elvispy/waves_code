using Surferbot
using Printf

"""
debug_uncoupled_custom_alpha_zero_modal_approx.jl

For one custom uncoupled `EI`, find a nontrivial empirical `alpha_beam = 0`
motor position by:
1. using the sweep artifact only to locate a nearby sign-change bracket, and
2. refining that bracket with direct reruns at the custom `EI`.

Then compare beam-end `S, A` against the modal proxies `S0246, A1357`.
"""

function normalize_flexible_params(params)
    return if params isa Surferbot.FlexibleParams
        params
    else
        Surferbot.FlexibleParams(;
            [k => getfield(params, k) for k in fieldnames(typeof(params)) if k in fieldnames(Surferbot.FlexibleParams)]...
        )
    end
end

function custom_EI_from_log10(log10_EI::Float64)
    return 10.0^log10_EI
end

function alpha_at(params::Surferbot.FlexibleParams, EI::Float64, xM_over_L::Float64)
    run_params = apply_parameter_overrides(params, (
        EI = EI,
        motor_position = xM_over_L * params.L_raft,
    ))
    result = flexible_solver(run_params)
    metrics = beam_edge_metrics(result)
    alpha = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)
    return alpha, result, metrics
end

function nearest_artifact_bracket(artifact, target_log10_EI::Float64; min_xM_over_L::Float64=0.05)
    params = normalize_flexible_params(artifact.base_params)
    mp_axis = collect(Float64.(artifact.parameter_axes.motor_position)) ./ params.L_raft
    EI_axis = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_axis)
    ie = argmin(abs.(logEI_axis .- target_log10_EI))

    alpha_col = [beam_asymmetry(artifact.summaries[imp, ie].eta_left_beam, artifact.summaries[imp, ie].eta_right_beam) for imp in eachindex(mp_axis)]
    S_col = [(artifact.summaries[imp, ie].eta_right_beam + artifact.summaries[imp, ie].eta_left_beam) / 2 for imp in eachindex(mp_axis)]
    A_col = [(artifact.summaries[imp, ie].eta_right_beam - artifact.summaries[imp, ie].eta_left_beam) / 2 for imp in eachindex(mp_axis)]
    sa_ratio = log10.(abs.(S_col) ./ (abs.(A_col) .+ eps()))

    brackets = NamedTuple[]
    for i in 1:(length(mp_axis) - 1)
        mp_axis[i] > min_xM_over_L || continue
        a = alpha_col[i]
        b = alpha_col[i + 1]
        if a * b < 0 && sa_ratio[i] < 0 && sa_ratio[i + 1] < 0
            push!(brackets, (
                x_lo = mp_axis[i],
                x_hi = mp_axis[i + 1],
                alpha_lo = a,
                alpha_hi = b,
                log10_EI_nearest = logEI_axis[ie],
            ))
        end
    end
    isempty(brackets) && error("No suitable alpha sign-change bracket found near log10(EI)=$target_log10_EI")
    return first(brackets)
end

function bisect_alpha_zero(params::Surferbot.FlexibleParams, EI::Float64, x_lo::Float64, x_hi::Float64; maxiter::Int=8)
    α_lo, _, _ = alpha_at(params, EI, x_lo)
    α_hi, _, _ = alpha_at(params, EI, x_hi)
    α_lo * α_hi < 0 || error("Alpha root not bracketed")

    best = nothing
    for _ in 1:maxiter
        x_mid = (x_lo + x_hi) / 2
        α_mid, result_mid, metrics_mid = alpha_at(params, EI, x_mid)
        best = (xM_over_L = x_mid, alpha = α_mid, result = result_mid, metrics = metrics_mid)
        if abs(α_mid) < 1e-3
            return best
        end
        if α_lo * α_mid < 0
            x_hi = x_mid
            α_hi = α_mid
        else
            x_lo = x_mid
            α_lo = α_mid
        end
    end
    return best
end

function modal_split(result; num_modes::Int=8)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=num_modes, verbose=false)
    left_weights = modal.Psi[1, :]
    S0246 = zero(ComplexF64)
    A1357 = zero(ComplexF64)
    for j in eachindex(modal.n)
        contrib = modal.q[j] * left_weights[j]
        if iseven(modal.n[j])
            S0246 += contrib
        else
            A1357 += contrib
        end
    end
    return (; modal, S0246, A1357)
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    target_log10_EI::Float64=-3.33,
    min_xM_over_L::Float64=0.05,
    maxiter::Int=8,
)
    artifact = load_sweep(joinpath(output_dir, "jld2", sweep_file))
    params = normalize_flexible_params(artifact.base_params)
    EI = custom_EI_from_log10(target_log10_EI)
    bracket = nearest_artifact_bracket(artifact, target_log10_EI; min_xM_over_L=min_xM_over_L)
    root = bisect_alpha_zero(params, EI, bracket.x_lo, bracket.x_hi; maxiter=maxiter)
    split = modal_split(root.result)

    S_beam = (root.metrics.eta_right_beam + root.metrics.eta_left_beam) / 2
    A_beam = (root.metrics.eta_right_beam - root.metrics.eta_left_beam) / 2

    println("Custom uncoupled alpha-zero rerun")
    @printf("  target log10(EI) = %.6f\n", target_log10_EI)
    @printf("  nearest artifact log10(EI) bracket source = %.6f\n", bracket.log10_EI_nearest)
    @printf("  bracket xM/L = [%.8f, %.8f]\n", bracket.x_lo, bracket.x_hi)
    @printf("  converged xM/L = %.8f\n", root.xM_over_L)
    @printf("  alpha_beam = % .6e\n", root.alpha)

    println("\nBeam-end amplitudes")
    @printf("  |S_beam|   = %.6e\n", abs(S_beam))
    @printf("  |A_beam|   = %.6e\n", abs(A_beam))

    println("\nModal approximations")
    @printf("  |S0246|    = %.6e\n", abs(split.S0246))
    @printf("  |A1357|    = %.6e\n", abs(split.A1357))

    println("\nRelative comparisons")
    @printf("  |S0246-S_beam| / |S_beam|   = %.6e\n", abs(split.S0246 - S_beam) / max(abs(S_beam), 1e-12))
    @printf("  |A1357-A_beam| / |A_beam|   = %.6e\n", abs(split.A1357 - A_beam) / max(abs(A_beam), 1e-12))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

using Surferbot
using Printf
using Random

"""
debug_uncoupled_random_slice_minima_check.jl

Pick one reproducible random uncoupled EI slice from the artifact with
log10(EI) >= min_log10_EI. Build the cheap 4-mode uncoupled predictor

    q_n(x_M) = -F_n(x_M) / D_n

then find:
1. local minima of |S_th| with |S_th|/|A_th| <= ratio_cutoff
2. local minima of |A_th| with |A_th|/|S_th| <= ratio_cutoff

Finally, run fluid solves only at those surviving x_M values and report the
empirical beam asymmetry alpha.
"""

function gaussian_load_nd(x0, sigma, x, L_raft)
    phi = exp.(-0.5 .* ((x .- x0) ./ sigma).^2)
    Z = sum(phi)
    return phi ./ Z ./ L_raft
end

function choose_random_EI(EI_list; min_log10_EI::Float64=-3.5, seed::Int=1234)
    logEI = log10.(collect(Float64.(EI_list)))
    mask = logEI .>= min_log10_EI
    any(mask) || error("No EI slices satisfy log10(EI) >= $min_log10_EI")
    candidates = collect(Float64.(EI_list[mask]))
    rng = MersenneTwister(seed)
    return rand(rng, candidates)
end

function cheap_theory_context(base_params, EI; num_modes::Int=4)
    params = apply_parameter_overrides(base_params, (
        EI = EI,
        motor_position = 0.2 * base_params.L_raft,
    ))
    result = flexible_solver(params)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(result; num_modes=num_modes, verbose=false)
    weights = Surferbot.Modal.trapz_weights(modal.x_raft)
    gram = modal.Psi' * (modal.Psi .* weights)
    D = ComplexF64.(EI .* modal.beta.^4 .- params.rho_raft .* params.omega^2)
    F0 = params.motor_inertia * params.omega^2
    sigma_f = 0.05 * params.L_raft
    return (; params, modal, weights, gram, D, F0, sigma_f)
end

function cheap_modal_response(ctx, xM_over_L::Float64)
    L = ctx.params.L_raft
    xM = xM_over_L * L
    load_dist = ctx.F0 .* gaussian_load_nd(xM, ctx.sigma_f, ctx.modal.x_raft, L)
    forcing = ctx.gram \ (ctx.modal.Psi' * (load_dist .* ctx.weights))
    return -ComplexF64.(forcing) ./ ctx.D
end

function split_SA(ctx, q)
    left_weights = ctx.modal.Psi[1, :]
    right_weights = ctx.modal.Psi[end, :]
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(ctx.modal.n)
        if iseven(ctx.modal.n[j])
            S += q[j] * left_weights[j]
        else
            A += q[j] * right_weights[j]
        end
    end
    return S, A
end

function find_filtered_minima(xgrid, values, ratio; ratio_cutoff::Float64)
    hits = NamedTuple[]
    for i in 2:length(xgrid)-1
        if values[i] <= values[i-1] && values[i] <= values[i+1] && ratio[i] <= ratio_cutoff
            push!(hits, (idx=i, xM_over_L=xgrid[i], value=values[i], ratio=ratio[i]))
        end
    end
    return hits
end

function run_case(base_params, EI, xM_over_L)
    params = apply_parameter_overrides(base_params, (
        EI = EI,
        motor_position = xM_over_L * base_params.L_raft,
    ))
    result = flexible_solver(params)
    metrics = beam_edge_metrics(result)
    alpha = beam_asymmetry(metrics.eta_left_beam, metrics.eta_right_beam)
    S_beam = (metrics.eta_right_beam + metrics.eta_left_beam) / 2
    A_beam = (metrics.eta_right_beam - metrics.eta_left_beam) / 2
    return (; alpha, S_beam, A_beam)
end

function main(;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "output"),
    sweep_file::AbstractString="sweep_motor_position_EI_uncoupled_from_matlab.jld2",
    num_modes::Int=4,
    seed::Int=1234,
    min_log10_EI::Float64=-3.5,
    exact_log10_EI::Union{Nothing, Float64}=nothing,
    xgrid=collect(range(0.0, 0.49, length=401)),
    ratio_cutoff::Float64=0.05,
)
    artifact = load_sweep(joinpath(output_dir, "jld2", sweep_file))
    EI = isnothing(exact_log10_EI) ?
        choose_random_EI(artifact.parameter_axes.EI; min_log10_EI=min_log10_EI, seed=seed) :
        10.0^exact_log10_EI
    ctx = cheap_theory_context(artifact.base_params, EI; num_modes=num_modes)

    absS = Float64[]
    absA = Float64[]
    ratio_SA = Float64[]
    ratio_AS = Float64[]
    for xM_over_L in xgrid
        q = cheap_modal_response(ctx, xM_over_L)
        S, A = split_SA(ctx, q)
        sabs = abs(S)
        aabs = abs(A)
        push!(absS, sabs)
        push!(absA, aabs)
        push!(ratio_SA, sabs / max(aabs, 1e-30))
        push!(ratio_AS, aabs / max(sabs, 1e-30))
    end

    hitsS = find_filtered_minima(xgrid, absS, ratio_SA; ratio_cutoff=ratio_cutoff)
    hitsA = find_filtered_minima(xgrid, absA, ratio_AS; ratio_cutoff=ratio_cutoff)

    println("Random uncoupled 4-mode branch check")
    @printf("  seed            = %d\n", seed)
    @printf("  log10(EI)       = %.6f\n", log10(EI))
    @printf("  EI              = %.12e\n", EI)
    @printf("  ratio cutoff    = %.3f\n", ratio_cutoff)

    println("\nS-family minima and empirical alpha")
    if isempty(hitsS)
        println("  none")
    else
        for hit in hitsS
            out = run_case(artifact.base_params, EI, hit.xM_over_L)
            @printf("  xM/L %.8f  |S_th| %.6e  |S|/|A| %.6e  |alpha| %.6e  |S_beam| %.6e  |A_beam| %.6e\n",
                hit.xM_over_L, hit.value, hit.ratio, abs(out.alpha), abs(out.S_beam), abs(out.A_beam))
        end
    end

    println("\nA-family minima and empirical alpha")
    if isempty(hitsA)
        println("  none")
    else
        for hit in hitsA
            out = run_case(artifact.base_params, EI, hit.xM_over_L)
            @printf("  xM/L %.8f  |A_th| %.6e  |A|/|S| %.6e  |alpha| %.6e  |S_beam| %.6e  |A_beam| %.6e\n",
                hit.xM_over_L, hit.value, hit.ratio, abs(out.alpha), abs(out.S_beam), abs(out.A_beam))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

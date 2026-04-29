using Surferbot
using JLD2
using Plots
using LinearAlgebra
using Printf
using DelimitedFiles
using CSV
using DataFrames
using Statistics
import SpecialFunctions: erf

"""
Julia/experiments/plot_dimensionless_diagnostics.jl

Dimensionless diagnostic plots for coupled and uncoupled motor-position/EI sweeps.

Design note:
- the empirical modal pressure map is identified in the raw analytical `W` basis
  by `prescribed_wn_diagonal_impedance.jl`
- this plotting script evaluates its theoretical branch in the orthonormal `Psi`
  basis used by the historical modal overlays and endpoint diagnostics
- we therefore cache `Z_raw` and also cache the deliberate `W -> Psi`
  operator conversion `Z_psi = T_{psi<-W} * Z_raw * T_{W<-psi}`

All fields manipulated here (`eta`, `pressure`, modal coefficients) are complex
Fourier amplitudes, not physical time-domain signals.
"""

include(joinpath(@__DIR__, "prescribed_wn_diagonal_impedance.jl"))
const ModalPressureMap = Main.PrescribedWnDiagonalImpedance

# --- Root-Finding Utils (Verbatim from plot_four_panel_diagnostics.jl) ---

function all_positive_roots(xs::AbstractVector{<:Real}, vals::AbstractVector{<:Real})
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = vals[i]
        b = vals[i + 1]
        if a == 0
            xs[i] > 1e-6 && push!(roots, Float64(xs[i]))
        elseif a * b < 0
            # Linear interpolation for high-precision crossing
            t = a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            root > 1e-6 && push!(roots, Float64(root))
        end
    end
    return unique!(roots)
end

function complex_roots(xs::AbstractVector{<:Real}, re_vals::AbstractVector{<:Real}, im_vals::AbstractVector{<:Real}; threshold=1e-3)
    # Find crossings of Re(eta) and filter by Im(eta) near 0
    roots = Float64[]
    for i in 1:(length(xs) - 1)
        a = re_vals[i]
        b = re_vals[i + 1]
        if a * b <= 0
            t = a == b ? 0.0 : a / (a - b)
            root = xs[i] + t * (xs[i + 1] - xs[i])
            
            # Interpolate Imaginary part at the root
            im_root = im_vals[i] + t * (im_vals[i+1] - im_vals[i])
            if abs(im_root) < threshold
                push!(roots, Float64(root))
            end
        end
    end
    return unique!(roots)
end

# --- Modal Framework Helpers (Verbatim from plot_four_panel_diagnostics.jl) ---

function coerce_flexible_params(params)
    params isa Surferbot.FlexibleParams && return params
    pairs = Pair{Symbol, Any}[]
    for k in fieldnames(Surferbot.FlexibleParams)
        if hasproperty(params, k)
            push!(pairs, k => getproperty(params, k))
        end
    end
    return Surferbot.FlexibleParams(; pairs...)
end

function raw_mode_shapes(params, xM_norm::AbstractVector{<:Real}; max_mode::Int=7)
    L = params.L_raft
    xi_motor = collect(float.(xM_norm)) .* L .+ L / 2
    Phi = zeros(Float64, length(xi_motor), max_mode + 1)
    
    xi_L = [0.0]
    xi_R = [L]
    Phi_L = zeros(Float64, 1, max_mode + 1)
    Phi_R = zeros(Float64, 1, max_mode + 1)

    Phi[:, 1] .= 1.0
    Phi_L[1, 1] = 1.0
    Phi_R[1, 1] = 1.0
    if max_mode >= 1
        Phi[:, 2] .= xi_motor .- L / 2
        Phi_L[1, 2] = -L / 2
        Phi_R[1, 2] = L / 2
    end
    n_elastic = max(0, max_mode - 1)
    if n_elastic > 0
        betaL_el = Surferbot.Modal.freefree_betaL_roots(n_elastic)
        for n in 2:max_mode
            Phi[:, n + 1] .= Surferbot.Modal.freefree_mode_shape(xi_motor, L, betaL_el[n - 1])
            Phi_L[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_L, L, betaL_el[n - 1])[1]
            Phi_R[1, n + 1] = Surferbot.Modal.freefree_mode_shape(xi_R, L, betaL_el[n - 1])[1]
        end
    end
    return (; xM_norm=collect(Float64.(xM_norm)), Phi, Phi_L=vec(Phi_L), Phi_R=vec(Phi_R))
end

# --- Root Extraction (Verbatim from plot_four_panel_diagnostics.jl) ---

function get_basis_for_plotting(params)
    fparams = coerce_flexible_params(params)
    res = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(res; num_modes=8, verbose=false)
    # Phi is raw analytical W, Psi is orthonormalized
    return (Phi=modal.Phi, Psi=modal.Psi, x=modal.x_raft) 
end

function get_roots_integral(csv_path, condition_name; modes=0:7)
    df = CSV.read(csv_path, DataFrame)
    L_raft = first(df.L_raft)
    basis = get_basis_for_plotting((L_raft=L_raft,))
    
    # Historical note: the CSV column names are `q_w*`, `Q_w*`, `F_w*`, but
    # `sweeper_modal_coefficients.jl` currently writes `modal.q`, `modal.Q`,
    # `modal.F`, i.e. Psi-basis coefficients. Therefore the integral overlays
    # must continue to use Psi endpoint weights here.
    w_end = basis.Psi[end, :]
    w_start = basis.Psi[1, :]

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for group in groupby(df, :log10_EI)
        logEI = first(group.log10_EI)
        xM_slice = group.xM_over_L
        
        re_vals = Float64[]
        im_vals = Float64[]
        
        for row in eachrow(group)
            val = 0.0 + 0.0im
            if condition_name == "S02"
                for n in (0, 2)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "S0246"
                for n in (0, 2, 4, 6)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "A13"
                for n in (1, 3)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "A1357"
                for n in (1, 3, 5, 7)
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            elseif condition_name == "eta_1"
                for n in 0:7
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_start[n+1]
                end
            elseif condition_name == "eta_end"
                for n in 0:7
                    val += complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")]) * w_end[n+1]
                end
            end
            push!(re_vals, real(val))
            push!(im_vals, imag(val))
        end
        
        roots = if startswith(condition_name, "eta")
            complex_roots(xM_slice, re_vals, im_vals)
        else
            all_positive_roots(xM_slice, re_vals)
        end
        
        for r in roots
            push!(pts_logEI, logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

function gaussian_load_nd(x0, sigma, x, L_raft)
    phi = exp.(-0.5 .* ((x .- x0) ./ sigma).^2)
    Z = sum(phi)
    return phi ./ Z ./ L_raft
end

function theoretical_modal_context(params; output_dir::AbstractString)
    fparams = coerce_flexible_params(params)
    num_modes = 4
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams;
        output_dir=output_dir,
        num_modes_basis=8,
    )
    basis_result = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(basis_result; num_modes=num_modes, verbose=false)
    Psi = modal.Psi
    weights = Surferbot.Modal.trapz_weights(modal.x_raft)
    psi_gram = Psi' * (Psi .* weights)
    Phi_raw = payload.raw_basis.Phi[:, 1:num_modes]
    raw_gram = payload.raw_basis.gram[1:num_modes, 1:num_modes]
    raw_from_psi = raw_gram \ (Phi_raw' * (Psi .* weights))
    psi_from_raw = psi_gram \ (Psi' * (Phi_raw .* weights))
    return (
        params = fparams,
        payload = payload,
        mode_numbers = collect(Int.(modal.n)),
        Psi = Matrix{Float64}(Psi),
        psi_gram = psi_gram,
        x_raft = collect(Float64.(modal.x_raft)),
        weights = collect(Float64.(weights)),
        w_start = Psi[1, :],
        w_end = Psi[end, :],
        beta = collect(Float64.(modal.beta)),
        Z_psi = ComplexF64.(psi_from_raw * payload.Z_raw[1:num_modes, 1:num_modes] * raw_from_psi),
        F0 = fparams.motor_inertia * fparams.omega^2,
        sigma_f = 0.05 * fparams.L_raft,
    )
end

function solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
    xM = xM_norm * theory_ctx.params.L_raft
    load_dist = theory_ctx.F0 .* gaussian_load_nd(xM, theory_ctx.sigma_f, theory_ctx.x_raft, theory_ctx.params.L_raft)
    forcing_rhs = theory_ctx.psi_gram \ (theory_ctx.Psi' * (load_dist .* theory_ctx.weights))
    structural = Diagonal(ComplexF64.(EI .* theory_ctx.beta .^ 4 .- theory_ctx.params.rho_raft .* theory_ctx.params.omega^2))
    return (structural - theory_ctx.Z_psi) \ (-ComplexF64.(forcing_rhs))
end

function theoretical_endpoint_diagnostics(q, theory_ctx)
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(theory_ctx.mode_numbers)
        if iseven(theory_ctx.mode_numbers[j])
            S += q[j] * theory_ctx.w_start[j]
        else
            A += q[j] * theory_ctx.w_end[j]
        end
    end
    eta_1 = sum(q .* theory_ctx.w_start)
    eta_end = sum(q .* theory_ctx.w_end)
    return (; S, A, eta_1, eta_end)
end

function find_filtered_minima(xgrid, values, ratio; ratio_cutoff::Float64)
    roots = Float64[]
    for i in 2:(length(xgrid) - 1)
        if values[i] <= values[i - 1] && values[i] <= values[i + 1] && ratio[i] < ratio_cutoff
            push!(roots, Float64(xgrid[i]))
        end
    end
    return roots
end

function compact_mode_range(mode_numbers::AbstractVector{<:Integer}, parity::Function)
    selected = [n for n in mode_numbers if parity(n)]
    isempty(selected) && return "?"
    return "$(first(selected))-$(last(selected))"
end

function theoretical_curve_labels(theory_ctx)
    s_range = compact_mode_range(theory_ctx.mode_numbers, iseven)
    a_range = compact_mode_range(theory_ctx.mode_numbers, isodd)
    return [
        "S_{$s_range}≈0",
        "A_{$a_range}≈0",
        "|eta_1|≈0",
        "|eta_end|≈0",
    ]
end

function get_roots_theoretical(artifact, condition_name; output_dir::AbstractString)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid = collect(range(0.0, 0.49, length=401))
    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)
    
    pts_logEI = Float64[]
    pts_xM = Float64[]

    for (iei, EI) in enumerate(EI_list)
        absS = Float64[]
        absA = Float64[]
        abs_eta_1 = Float64[]
        abs_eta_end = Float64[]

        for xM_norm in xM_grid
            q = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            diag = theoretical_endpoint_diagnostics(q, theory_ctx)
            push!(absS, abs(diag.S))
            push!(absA, abs(diag.A))
            push!(abs_eta_1, abs(diag.eta_1))
            push!(abs_eta_end, abs(diag.eta_end))
        end

        roots = if condition_name in ("S", "S02", "S0246")
            ratio = absS ./ max.(absA, eps())
            find_filtered_minima(xM_grid, absS, ratio; ratio_cutoff=0.05)
        elseif condition_name in ("A", "A13", "A1357")
            ratio = absA ./ max.(absS, eps())
            find_filtered_minima(xM_grid, absA, ratio; ratio_cutoff=0.05)
        elseif condition_name == "eta_1"
            denom = abs_eta_1 .+ abs_eta_end .+ eps()
            ratio = abs_eta_1 ./ denom
            find_filtered_minima(xM_grid, abs_eta_1, ratio; ratio_cutoff=0.05)
        elseif condition_name == "eta_end"
            denom = abs_eta_1 .+ abs_eta_end .+ eps()
            ratio = abs_eta_end ./ denom
            find_filtered_minima(xM_grid, abs_eta_end, ratio; ratio_cutoff=0.05)
        else
            Float64[]
        end

        for r in roots
            push!(pts_logEI, logEI_axis[iei])
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# --- GPR Smoothing Logic (Verbatim from plot_four_panel_diagnostics.jl) ---

function fit_gp2d(x::AbstractVector, y::AbstractVector, values::AbstractVector)
    n = length(values)
    mean_value = mean(values)
    centered = collect(Float64.(values .- mean_value))

    dx = diff(sort(unique(Float64.(x))))
    dy = diff(sort(unique(Float64.(y))))
    dx = dx[dx .> 0]
    dy = dy[dy .> 0]
    # Length scales: heuristic based on grid spacing
    lx = isempty(dx) ? 0.05 : max(0.02, 3 * median(dx))
    ly = isempty(dy) ? 0.15 : max(0.05, 3 * median(dy))
    sigma_f = max(std(values), 1e-3)
    noise = max(1e-4, 2e-2 * sigma_f)

    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        r2 = ((x[i] - x[j]) / lx)^2 + ((y[i] - y[j]) / ly)^2
        kij = sigma_f^2 * exp(-0.5 * r2)
        K[i, j] = kij
        K[j, i] = kij
    end
    for i in 1:n
        K[i, i] += noise^2 + 1e-10
    end

    F = cholesky(Symmetric(K))
    weights = F \ centered
    return (x = Float64.(x), y = Float64.(y), weights = weights, mean = mean_value, lx = lx, ly = ly, sigma_f2 = sigma_f^2, noise=noise)
end

function predict_gp2d(model, xq::Real, yq::Real)
    acc = 0.0
    for i in eachindex(model.weights)
        r2 = ((xq - model.x[i]) / model.lx)^2 + ((yq - model.y[i]) / model.ly)^2
        acc += model.weights[i] * (model.sigma_f2 * exp(-0.5 * r2))
    end
    return model.mean + acc
end

# --- Main Logic (Verbatim except for axis shift) ---

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    configs = [
        (name="unc_theo", coupled=false, source=:theoretical),
        (name="unc_int",  coupled=false, source=:integral),
        (name="cpl_theo", coupled=true,  source=:theoretical),
        (name="cpl_int",  coupled=true,  source=:integral)
    ]

    # Modular Toggle for GPR Smoothing
    USE_GPR = false 

    for cfg in configs
        println("Processing $(cfg.name)...")
        
        # Load Heatmap Data (JLD2 for Alpha background)
        jld2_file = cfg.coupled ? 
            "sweep_motor_position_EI_coupled_from_matlab.jld2" : 
            "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
        artifact = load_sweep(joinpath(output_dir, "jld2", jld2_file))
        
        # --- CALC SHIFT ---
        params = artifact.base_params
        shift = log10(params.rho_raft * params.L_raft^4 * params.omega^2)

        logEI_axis = log10.(collect(Float64.(artifact.parameter_axes.EI)))
        xM_axis = collect(Float64.(artifact.parameter_axes.motor_position)) ./ artifact.base_params.L_raft
        alpha = beam_asymmetry.(map(s->s.eta_left_beam, artifact.summaries), map(s->s.eta_right_beam, artifact.summaries))
        
        # --- Background GPR Smoothing ---
        local p_heatmap
        if USE_GPR
            println("  Fitting GPR background...")
            xtrain, ytrain, vtrain = Float64[], Float64[], Float64[]
            # Note: training feature space (x=logEI, y=xM)
            for (ie, lei) in enumerate(logEI_axis), (im, xm) in enumerate(xM_axis)
                push!(xtrain, lei)
                push!(ytrain, xm)
                push!(vtrain, alpha[im, ie])
            end
            model = fit_gp2d(xtrain, ytrain, vtrain)
            
            # Predict on a very dense grid for smoothness
            logEI_dense = collect(range(minimum(logEI_axis), maximum(logEI_axis), length=200))
            xM_dense = collect(range(minimum(xM_axis), maximum(xM_axis), length=200))
            alpha_dense = [predict_gp2d(model, lei, xm) for xm in xM_dense, lei in logEI_dense]
            
            # APPLY SHIFT TO HEATMAP AXIS
            p_heatmap = (logEI_dense .- shift, xM_dense, alpha_dense)
        else
            # APPLY SHIFT TO HEATMAP AXIS
            p_heatmap = (logEI_axis .- shift, xM_axis, alpha)
        end

        curve_names = cfg.source == :theoretical ?
            ["S", "A", "eta_1", "eta_end"] :
            ["S02", "S0246", "A13", "A1357", "eta_1", "eta_end"]
        results = Dict()
        
        if cfg.source == :integral
            csv_file = cfg.coupled ? "sweeper_coupled_full_grid.csv" : "sweeper_uncoupled_full_grid.csv"
            csv_path = joinpath(output_dir, "csv", csv_file)
            for cname in curve_names
                res = get_roots_integral(csv_path, cname)
                # APPLY SHIFT TO SCATTER RESULTS
                results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
            end
        else
            for cname in curve_names
                res = get_roots_theoretical(artifact, cname; output_dir=output_dir)
                # APPLY SHIFT TO SCATTER RESULTS
                results[cname] = (logK = res.logEI .- shift, xM_norm = res.xM_norm)
            end
        end
        
        # Plotting with expert scientific visualization
        # Okabe-Ito Colorblind-Safe Palette
        okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        
        plt_opts = (
            xlabel="log10(kappa)", ylabel="x_M / L", 
            colormap=:balance, clims=(-1, 1), 
            levels=51, # High granularity for smooth gradients
            interpolate=true,
            xlims=(minimum(logEI_axis) - shift, maximum(logEI_axis) - shift),
            ylims=(minimum(xM_axis), maximum(xM_axis)),
            legend=:topright, size=(1000, 750), margin=10Plots.mm,
            # Font sizes optimized for readability
            titlefontsize=14, guidefontsize=11, tickfontsize=9, legendfontsize=9,
            fontfamily="sans-serif"
        )
        
        # Determine Title
        fig_title = if cfg.name == "unc_theo"
            "Uncoupled Theoretical (Cached Z)"
        elseif cfg.name == "unc_int"
            "Uncoupled Integral"
        elseif cfg.name == "cpl_theo"
            "Coupled Theoretical (Cached Z)"
        elseif cfg.name == "cpl_int"
            "Coupled Integral"
        end

        p = heatmap(p_heatmap[1], p_heatmap[2], p_heatmap[3], title=fig_title; plt_opts...)
        
        curve_colors = cfg.source == :theoretical ?
            [okabe_ito[2], okabe_ito[4], okabe_ito[5], okabe_ito[6]] :
            [okabe_ito[1], okabe_ito[2], okabe_ito[3], okabe_ito[4], okabe_ito[5], okabe_ito[6]]
        markers = cfg.source == :theoretical ?
            [:rect, :utriangle, :star5, :star8] :
            [:circle, :rect, :diamond, :utriangle, :star5, :star8]
        labels = cfg.source == :theoretical ?
            theoretical_curve_labels(theoretical_modal_context(params; output_dir=output_dir)) :
            ["S02=0", "S0246=0", "A13=0", "A1357=0", "|eta_1|=0", "|eta_end|=0"]
        
        for (i, cname) in enumerate(curve_names)
            res = results[cname]
            scatter!(p, res.logK, res.xM_norm, 
                label=labels[i], color=curve_colors[i], marker=markers[i], 
                markersize=8, markerstrokewidth=0)
        end
        
        out_path = joinpath(output_dir, "figures", "plot_dimensionless_diagnostics_$(cfg.name).pdf")
        savefig(p, out_path)
        println("Saved $out_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

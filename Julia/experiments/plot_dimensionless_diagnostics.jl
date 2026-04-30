using Surferbot
using JLD2
using Plots
using LaTeXStrings
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

const NUM_MODES = 8
const RATIO_CUTOFF = 0.5     # consistent with coupled_apriori_test.jl

function get_basis_for_plotting(params)
    fparams = coerce_flexible_params(params)
    res = Surferbot.flexible_solver(fparams)
    modal = Surferbot.Modal.decompose_raft_freefree_modes(res; num_modes=NUM_MODES, verbose=false)
    # Phi is raw analytical W, Psi is orthonormalized
    return (Phi=modal.Phi, Psi=modal.Psi, x=modal.x_raft)
end

function get_roots_integral(csv_path, condition_name; modes=0:(NUM_MODES-1))
    df = CSV.read(csv_path, DataFrame)
    L_raft = first(df.L_raft)
    basis = get_basis_for_plotting((L_raft=L_raft,))
    # Endpoint weights — w_end is used for S, A, eta_end; eta_1 uses w_start.
    # CSV columns are named `q_w*` historically but actually store Psi-basis
    # coefficients (see sweeper_modal_coefficients.jl).
    w_end = basis.Psi[end, :]
    w_start = basis.Psi[1, :]

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for group in groupby(df, :log10_EI)
        logEI = first(group.log10_EI)
        sorted = sort(group, :xM_over_L)
        xM_slice = collect(Float64.(sorted.xM_over_L))

        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]
        for row in eachrow(sorted)
            S = 0.0 + 0.0im; A = 0.0 + 0.0im
            for n in 0:(NUM_MODES - 1)
                qn = complex(row[Symbol("q_w$(n)_re")], row[Symbol("q_w$(n)_im")])
                if iseven(n)
                    S += qn * w_end[n + 1]
                else
                    A += qn * w_end[n + 1]
                end
            end
            eta_end = S + A                        # right beam end
            eta_1   = S - A                        # left beam end (odd modes flip sign)
            push!(absS, abs(S))
            push!(absA, abs(A))
            push!(abs_eta_1, abs(eta_1))
            push!(abs_eta_end, abs(eta_end))
        end

        roots = roots_for_condition(condition_name, xM_slice,
                                    absS, absA, abs_eta_1, abs_eta_end)

        for r in roots
            push!(pts_logEI, logEI)
            push!(pts_xM, r)
        end
    end
    return (; logEI=pts_logEI, xM_norm=pts_xM)
end

# Unified condition→roots dispatcher (used by both integral and theoretical paths).
function roots_for_condition(condition_name, xgrid, absS, absA, abs_eta_1, abs_eta_end)
    if condition_name == "S"
        ratio = absS ./ max.(absA, eps())
        return find_filtered_minima(xgrid, absS, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "A"
        ratio = absA ./ max.(absS, eps())
        return find_filtered_minima(xgrid, absA, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_1"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        ratio = abs_eta_1 ./ denom
        return find_filtered_minima(xgrid, abs_eta_1, ratio; ratio_cutoff=RATIO_CUTOFF)
    elseif condition_name == "eta_end"
        denom = abs_eta_1 .+ abs_eta_end .+ eps()
        ratio = abs_eta_end ./ denom
        return find_filtered_minima(xgrid, abs_eta_end, ratio; ratio_cutoff=RATIO_CUTOFF)
    end
    return Float64[]
end

function theoretical_modal_context(params; output_dir::AbstractString)
    fparams = coerce_flexible_params(params)
    payload = ModalPressureMap.load_or_compute_modal_pressure_map(
        fparams;
        output_dir=output_dir,
        num_modes_basis=NUM_MODES,
    )
    derived = Surferbot.derive_params(fparams)
    Psi = payload.psi_basis.Psi
    return (
        params  = fparams,
        derived = derived,
        payload = payload,
        mode_numbers = collect(Int.(payload.mode_labels)),
        Psi     = Matrix{Float64}(Psi),
        x_raft  = collect(Float64.(payload.x_raft)),
        weights = collect(Float64.(payload.weights)),
        w_start = Psi[1, :],
        w_end   = Psi[end, :],
        beta    = collect(Float64.(payload.beta)),
        Z_psi   = ComplexF64.(payload.Z_psi),         # already in Psi basis (post-fix)
        c_hydro = derived.d * fparams.rho * fparams.g,  # d ρ g
        F0      = fparams.motor_inertia * fparams.omega^2,
        forcing_width = fparams.forcing_width,
    )
end

function solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
    p = theory_ctx.params
    F_c = theory_ctx.derived.F_c
    L_c = theory_ctx.derived.L_c

    # Loads via the same Surferbot.gaussian_load used by flexible_solver
    x_raft_adim = theory_ctx.x_raft ./ L_c
    loads_adim  = (theory_ctx.F0 / F_c) .*
                  Surferbot.gaussian_load(Float64(xM_norm), p.forcing_width, x_raft_adim)
    loads_dim   = loads_adim .* (F_c / L_c)               # N/m
    F_psi       = theory_ctx.Psi' * (loads_dim .* theory_ctx.weights)

    # D = EI β⁴ − ρ_R ω² + d ρ g  (hydrostatic restoring stiffness on diagonal)
    D = ComplexF64.(EI .* theory_ctx.beta .^ 4
                    .- p.rho_raft * p.omega^2
                    .+ theory_ctx.c_hydro)
    A_sys = Diagonal(D) - theory_ctx.Z_psi
    return -(A_sys \ ComplexF64.(F_psi))
end

# S/A/eta endpoints from a Psi-basis q-vector.  Uses w_end for both S and A so
# the integral and theoretical paths share the same definition.
function theoretical_endpoint_diagnostics(q, theory_ctx)
    S = zero(ComplexF64)
    A = zero(ComplexF64)
    for j in eachindex(theory_ctx.mode_numbers)
        if iseven(theory_ctx.mode_numbers[j])
            S += q[j] * theory_ctx.w_end[j]
        else
            A += q[j] * theory_ctx.w_end[j]
        end
    end
    eta_end = S + A                           # right beam end
    eta_1   = S - A                           # left beam end (odd modes flip)
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

const CURVE_NAMES  = ["S", "A", "eta_1", "eta_end"]
const CURVE_LABELS = [L"|S| = 0", L"|A| = 0", L"|\eta_1| = 0", L"|\eta_{\mathrm{end}}| = 0"]

function get_roots_theoretical(artifact, condition_name; output_dir::AbstractString)
    params = artifact.base_params
    EI_list = collect(Float64.(artifact.parameter_axes.EI))
    logEI_axis = log10.(EI_list)
    xM_grid = collect(range(0.0, 0.49, length=401))
    theory_ctx = theoretical_modal_context(params; output_dir=output_dir)

    pts_logEI = Float64[]
    pts_xM = Float64[]

    for (iei, EI) in enumerate(EI_list)
        absS = Float64[]; absA = Float64[]
        abs_eta_1 = Float64[]; abs_eta_end = Float64[]

        for xM_norm in xM_grid
            q = solve_theoretical_modal_response(EI, xM_norm, theory_ctx)
            diag = theoretical_endpoint_diagnostics(q, theory_ctx)
            push!(absS, abs(diag.S))
            push!(absA, abs(diag.A))
            push!(abs_eta_1, abs(diag.eta_1))
            push!(abs_eta_end, abs(diag.eta_end))
        end

        roots = roots_for_condition(condition_name, xM_grid,
                                    absS, absA, abs_eta_1, abs_eta_end)

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

        # Artifact still supplies base_params (for the κ shift) and the EI grid
        # over which the theoretical curve is computed. The α heatmap and the
        # integral overlay both come from the higher-resolution CSV so that
        # the heatmap field is self-consistent with the integral scatter.
        jld2_file = cfg.coupled ?
            "sweep_motor_position_EI_coupled_from_matlab.jld2" :
            "sweep_motor_position_EI_uncoupled_from_matlab.jld2"
        artifact = load_sweep(joinpath(output_dir, "jld2", jld2_file))

        params = artifact.base_params
        shift = log10(params.rho_raft * params.L_raft^4 * params.omega^2)

        # α heatmap from CSV (300 × 100 grid; matches the integral scatter exactly)
        csv_file = cfg.coupled ?
            "sweeper_coupled_full_grid.csv" :
            "sweeper_uncoupled_full_grid.csv"
        csv_path = joinpath(output_dir, "csv", csv_file)
        df_heat = CSV.read(csv_path, DataFrame)
        logEI_axis = sort(unique(df_heat.log10_EI))
        xM_axis    = sort(unique(df_heat.xM_over_L))
        alpha = zeros(Float64, length(xM_axis), length(logEI_axis))
        let row_lookup = Dict{Tuple{Float64,Float64}, Float64}(
                (row.log10_EI, row.xM_over_L) => row.alpha for row in eachrow(df_heat))
            for (j, le) in enumerate(logEI_axis), (i, xm) in enumerate(xM_axis)
                alpha[i, j] = row_lookup[(le, xm)]
            end
        end
        
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

        # Same 4 conditions in both paths so theoretical and integral plots line up directly.
        curve_names = CURVE_NAMES
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
        
        # ── Publication styling ───────────────────────────────────────────
        # Okabe–Ito colorblind-safe palette; pick four mutually distinct hues
        # that sit far from the red/blue heatmap extremes for visibility.
        okabe_ito  = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                      "#0072B2", "#D55E00", "#CC79A7", "#000000"]
        curve_colors = [okabe_ito[8], okabe_ito[1], okabe_ito[3], okabe_ito[7]]
                        # black,         orange,       green,        pink
        markers      = [:circle, :rect, :diamond, :utriangle]
        labels       = CURVE_LABELS

        # κ-window of physical interest; clip to CSV data extent so the panel
        # has no empty strip past the largest sampled κ.
        max_logK_data = maximum(logEI_axis) - shift
        XLIMS = (-4.0, min(0.0, max_logK_data))
        YLIMS = (0.0, 0.5)

        # Plot options shared by both heatmap and scatter calls.
        plt_opts = (
            xlabel = L"\log_{10}\,\kappa",
            ylabel = L"x_M / L",
            colormap = :balance,
            clims = (-1, 1),
            levels = 51,
            interpolate = true,
            xlims = XLIMS,
            ylims = YLIMS,
            legend = :topright,
            background_color_legend = RGBA(1, 1, 1, 0.85),
            foreground_color_legend = :black,
            legend_font_halign = :left,
            size = (820, 640),
            margin = 6Plots.mm,
            dpi = 220,
            titlefontsize = 14,
            guidefontsize = 14,
            tickfontsize = 12,
            legendfontsize = 11,
            fontfamily = "Computer Modern",
            framestyle = :box,
            grid = false,
            colorbar_title = L"\alpha",
            colorbar_titlefontsize = 14,
            colorbar_tickfontsize = 11,
        )

        # Title: keep prose outside math mode (GR's math renderer eats spaces
        # inside \textrm{...}); only Λ goes through math mode.
        adj         = cfg.coupled ? "Coupled" : "Uncoupled"
        method_str  = cfg.source == :theoretical ? "a-priori prediction" : "direct integration"
        Lambda_val  = cfg.coupled ? @sprintf("%.2f", params.d / params.L_raft) : "0"
        fig_title   = LaTeXString(
            "$adj raft, \$\\Lambda = $Lambda_val\$ — $method_str")

        p = heatmap(p_heatmap[1], p_heatmap[2], p_heatmap[3];
                    title = fig_title, plt_opts...)

        for (i, cname) in enumerate(curve_names)
            res = results[cname]
            # Restrict scatter to the publication κ-window.
            mask = (XLIMS[1] .<= res.logK .<= XLIMS[2]) .&
                   (YLIMS[1] .<= res.xM_norm .<= YLIMS[2])
            isempty(res.logK[mask]) && continue
            scatter!(p, res.logK[mask], res.xM_norm[mask];
                     label = labels[i],
                     color = curve_colors[i],
                     marker = markers[i],
                     markersize = 5,
                     markerstrokewidth = 0.6,
                     markerstrokecolor = :white,
                     markeralpha = 0.95)
        end
        
        fig_dir = joinpath(output_dir, "figures")
        out_pdf = joinpath(fig_dir, "plot_dimensionless_diagnostics_$(cfg.name).pdf")
        out_png = joinpath(fig_dir, "plot_dimensionless_diagnostics_$(cfg.name).png")
        savefig(p, out_pdf)
        savefig(p, out_png)
        println("Saved $out_pdf")
        println("Saved $out_png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

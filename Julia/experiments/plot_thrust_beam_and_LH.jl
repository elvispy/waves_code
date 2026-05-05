"""
plot_thrust_beam_and_LH.jl

Two heatmaps on the (log₁₀κ, xM/L) plane per coupling case:
  1. Beam  — Δ|η|²/L²  at the raft endpoints      (eta_{1,end}_beam columns)
  2. LH    — Δ|η|²/L²  at the computational-domain ends  (Longuet-Higgins proxy)
             (eta_{1,end}_domain columns)

Δ|η|² = |η_end|² − |η_1|²  (right minus left; same sign convention as α).
Color: signed log₁₀ scale  c = sign(Δ)·log₁₀(|Δ|/L² + ε),  diverging :balance cmap.
Source CSVs: output/csv/sweeper_{coupled,uncoupled}_full_grid.csv
"""

using CSV
using DataFrames
using Plots
using LaTeXStrings
using Printf
using Statistics
using Surferbot

# ─── Load and reshape ────────────────────────────────────────────────────────

function load_delta_grids(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)

    log10_EI_vals = sort(unique(Float64.(df.log10_EI)))
    xM_vals       = sort(unique(Float64.(df.xM_over_L)))
    nEI = length(log10_EI_vals)
    nxM = length(xM_vals)

    beam_grid   = fill(NaN, nxM, nEI)
    domain_grid = fill(NaN, nxM, nEI)

    L = Float64(first(df.L_raft))

    idx = Dict{Tuple{Float64,Float64}, Tuple{Int,Int}}(
        (log10_EI_vals[j], xM_vals[i]) => (i, j)
        for j in 1:nEI for i in 1:nxM)

    for row in eachrow(df)
        k = (Float64(row.log10_EI), Float64(row.xM_over_L))
        haskey(idx, k) || continue
        i, j = idx[k]
        η1b = complex(row.eta_1_beam_re,    row.eta_1_beam_im)
        ηEb = complex(row.eta_end_beam_re,  row.eta_end_beam_im)
        η1d = complex(row.eta_1_domain_re,  row.eta_1_domain_im)
        ηEd = complex(row.eta_end_domain_re, row.eta_end_domain_im)
        beam_grid[i, j]   = (abs2(ηEb) - abs2(η1b)) / L^2
        domain_grid[i, j] = (abs2(ηEd) - abs2(η1d)) / L^2
    end

    return (; log10_EI = log10_EI_vals, xM = xM_vals, beam_grid, domain_grid,
              L,
              rho_raft = Float64(first(df.rho_raft)),
              omega    = Float64(first(df.omega)))
end

# ─── Signed-log colour transform ─────────────────────────────────────────────
# c = sign(Δ) · log₁₀(|Δ| + ε),  ε = max|Δ| × eps_frac so the origin is
# suppressed rather than sent to -∞.

function signed_log10_grid(mat::AbstractMatrix{Float64}; eps_frac=1e-6)
    finite_vals = filter(isfinite, vec(mat))
    isempty(finite_vals) && return fill(NaN, size(mat))
    maxabs = maximum(abs, finite_vals)
    ε = maxabs * eps_frac + 1e-30
    return @. sign(mat) * log10(abs(mat) + ε)
end

# ─── Operating-point marker ───────────────────────────────────────────────────

function operating_point(bp, shift)
    lk  = log10(Float64(bp.EI)) - shift
    xm  = abs(Float64(bp.motor_position)) / Float64(bp.L_raft)
    return lk, xm
end

# ─── Render one panel ─────────────────────────────────────────────────────────

function render_panel(log10_kappa, xM_axis, delta_grid, fig_title, out_base, bp, shift)
    c = signed_log10_grid(delta_grid)

    finite_c = filter(isfinite, vec(c))
    clim_val = isempty(finite_c) ? 1.0 : quantile(abs.(finite_c), 0.99)
    clim_val = max(clim_val, 1e-3)

    max_logK = maximum(log10_kappa)
    XLIMS = (-4.0, max_logK)
    YLIMS = (0.0, 0.5)

    plt_opts = (
        xlabel             = L"\log_{10}\,\kappa",
        ylabel             = L"x_M / L",
        colormap           = :balance,
        clims              = (-clim_val, clim_val),
        interpolate        = true,
        xlims              = XLIMS,
        ylims              = YLIMS,
        legend             = false,
        colorbar           = true,
        colorbar_title     = L"\mathrm{sgn}(\Delta)\cdot\log_{10}|\Delta\eta^2|/L^2",
        colorbar_titlefontsize = 11,
        colorbar_tickfontsize  = 11,
        size               = (820, 640),
        margin             = 6Plots.mm,
        dpi                = 220,
        titlefontsize      = 14,
        guidefontsize      = 14,
        tickfontsize       = 12,
        fontfamily         = "Computer Modern",
        framestyle         = :box,
        grid               = false,
    )

    p = heatmap(log10_kappa, xM_axis, c; title = fig_title, plt_opts...)

    lk_star, xm_star = operating_point(bp, shift)
    scatter!(p, [lk_star], [xm_star];
             marker            = :star5,
             markersize        = 12,
             color             = :white,
             markerstrokecolor = :black,
             markerstrokewidth = 1,
             label             = false)

    savefig(p, out_base * ".pdf")
    savefig(p, out_base * ".png")
    println("Saved $(out_base).{pdf,png}")
end

# ─── Main ─────────────────────────────────────────────────────────────────────

function main()
    output_dir = joinpath(@__DIR__, "..", "output")
    fig_dir    = joinpath(output_dir, "figures")
    mkpath(fig_dir)

    cases = [
        (label = "coupled", csv = "sweeper_coupled_full_grid.csv", coupled = true),
    ]

    for case in cases
        csv_path = joinpath(output_dir, "csv", case.csv)
        if !isfile(csv_path)
            @warn "$(case.csv) not found in $(output_dir)/csv — skipping."
            continue
        end
        println("Processing $(case.label) ...")

        grids = load_delta_grids(csv_path)
        shift = log10(grids.rho_raft * grids.L^4 * grids.omega^2)
        log10_kappa = grids.log10_EI .- shift

        bp = case.coupled ?
            Surferbot.Analysis.default_coupled_motor_position_EI_sweep().base_params :
            Surferbot.Analysis.default_uncoupled_motor_position_EI_sweep().base_params

        adj        = case.coupled ? "Coupled" : "Uncoupled"
        Lambda_val = case.coupled ? @sprintf("%.2f", Float64(bp.d) / Float64(bp.L_raft)) : "0"

        render_panel(
            log10_kappa, grids.xM, grids.beam_grid,
            LaTeXString("$adj, \$\\Lambda=$Lambda_val\$ — beam \$\\Delta|\\eta|^2/L^2\$"),
            joinpath(fig_dir, "plot_thrust_beam_$(case.label)"),
            bp, shift)

        render_panel(
            log10_kappa, grids.xM, grids.domain_grid,
            LaTeXString("$adj, \$\\Lambda=$Lambda_val\$ — LH \$\\Delta|\\eta|^2/L^2\$"),
            joinpath(fig_dir, "plot_thrust_LH_$(case.label)"),
            bp, shift)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
